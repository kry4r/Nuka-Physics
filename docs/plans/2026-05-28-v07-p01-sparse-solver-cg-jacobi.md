# Nuka Physics v0.7 – Phase 1: Self-Written Sparse Linear Solver — CG + Jacobi / Block-Jacobi Preconditioners

> **Master plan reference:** §2 decision #12 (cuDSS phase 1 → self-written phase 2) + §8 risk register
> **Prerequisites:** v0.5 closed (cuDSS in place; we now start to replace it)
> **Blocks:** v0.7 Phase 2 (MINRES + ILU builds on this CG foundation)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Begin the multi-phase replacement of cuDSS with a self-written deterministic CUDA sparse linear solver suite. This phase ships **Conjugate Gradient (CG)** plus **Jacobi** and **Block-Jacobi** preconditioners — the simplest, well-understood, deterministic baseline.

By the end of this phase, the `SparseLinearSolver` abstraction (introduced v0.5 Phase 3) gains a `SelfWrittenCgBackend` option. cuDSS remains the default; self-written CG is opt-in via configuration. The two backends produce numerically agreeing results within 1e-6 relative on standard test matrices.

This is **not** the full replacement — MINRES, ILU(0), GMRES, AMG land in Phases 2 and 3. Full cuDSS retirement happens in v2.0 Phase 1.

## Tech Stack

- CUDA 12+
- CUB / Thrust (for warp-level primitives; deterministic reduction tree variants)
- Existing `SparseLinearSolver` abstraction (v0.5 Phase 3)

## Files to Create

- `src/diffsim/solver/cg_backend.hpp` — CG implementation
- `src/diffsim/solver/cg_backend.cu`
- `src/diffsim/solver/jacobi_preconditioner.cuh` — diagonal preconditioner
- `src/diffsim/solver/block_jacobi_preconditioner.cuh` — block-diagonal preconditioner (per-island)
- `src/diffsim/solver/deterministic_dot.cuh` — fixed-order tree reduction dot product
- `src/diffsim/solver/deterministic_axpy.cuh` — element-wise FMA (trivially deterministic)
- `src/diffsim/solver/sparse_matrix_csr_ops.cuh` — SpMV with deterministic ordering
- `tests/diffsim/solver/test_deterministic_dot.cu` — same inputs twice → bit-exact
- `tests/diffsim/solver/test_cg_vs_eigen.cpp` — known SPD test matrices; agreement < 1e-6
- `tests/diffsim/solver/test_cg_vs_cudss.cpp` — same inputs to both backends; agreement < 1e-5
- `tests/diffsim/solver/test_cg_convergence_rate.cpp` — verifies κ-conditioning matches theory

## Files to Modify

- `src/diffsim/sparse_solver_backend.hpp` — register `SelfWrittenCgBackend` factory key
- `src/include/nuka/nuka_diffsim.h` — add `NUKA_SOLVER_BACKEND_SELF_CG` enum value
- `src/c_abi/diffsim.cpp` — wire backend selection
- `cmake/Findcudss.cmake` — cuDSS now truly optional (graceful absence if user picks self-written)

## Tasks

### Task 7.1.1 — Deterministic dot product

`src/diffsim/solver/deterministic_dot.cuh`:

```cuda
// Tree reduction with fixed warp order. Same inputs → same output bit-exactly.
// Reference: PhysX TGS deterministic reduce; pre-CUDA-9 patterns.
__device__ float deterministic_dot(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    uint32_t n);

__global__ void deterministic_dot_kernel(const float* a, const float* b,
                                          uint32_t n, float* out);
```

Implementation: pad to power of two; reduce in fixed tree order; no `__ldcg` / `__ldcs` (their behavior may differ across architectures); use `__fadd_rn` for round-to-nearest-even consistency.

V3 test (later): run same dot twice on same device + driver → bit-exact equal.

### Task 7.1.2 — SpMV (sparse matrix-vector)

CSR format SpMV with deterministic per-row order:

```cuda
__global__ void spmv_csr_kernel(uint32_t rows, const uint32_t* row_ptr,
                                 const uint32_t* col_idx, const float* values,
                                 const float* x, float* y);
```

Each row is processed by one thread (or a warp for long rows). Within a row, accumulation is sequential along sorted column indices → deterministic regardless of how warps execute.

### Task 7.1.3 — CG main loop

```cpp
class SelfWrittenCgBackend : public SparseLinearSolver {
public:
    SelfWrittenCgBackend(const phi::DeviceContext& ctx, uint32_t max_iter = 200,
                         float tol = 1e-6f);

    void Analyze(const SparseMatrixCsr& A) override {
        n_ = A.rows;
        // Detect symmetry; CG only valid for SPD.
        // Allocate working vectors (r, p, Ap, z).
        // Build preconditioner.
    }
    void Factorize(const SparseMatrixCsr& A) override {
        A_ = &A;
        // Rebuild Jacobi diagonal cache.
    }
    void Solve(const float* b, float* x) override {
        // Standard CG with M^-1 preconditioning
        // r = b - A x
        // z = M^-1 r
        // p = z
        // rho = <r, z>
        // for k in [0, max_iter):
        //   Ap = A p
        //   alpha = rho / <p, Ap>
        //   x += alpha p
        //   r -= alpha Ap
        //   if <r, r> < tol^2: break
        //   z = M^-1 r
        //   rho_new = <r, z>
        //   beta = rho_new / rho
        //   p = z + beta p
        //   rho = rho_new
    }
    DeterminismLevel Determinism() const override { return DeterminismLevel::Strong; }

private:
    uint32_t n_, max_iter_;
    float tol_;
    const SparseMatrixCsr* A_;
    DeviceVector<float> r_, p_, Ap_, z_;
    std::unique_ptr<class Preconditioner> precond_;
};
```

### Task 7.1.4 — Jacobi preconditioner

`src/diffsim/solver/jacobi_preconditioner.cuh`:

```cuda
__global__ void jacobi_extract_diag_kernel(uint32_t rows, const uint32_t* row_ptr,
                                            const uint32_t* col_idx, const float* values,
                                            float* diag);

__global__ void jacobi_apply_kernel(uint32_t n, const float* diag_inv,
                                     const float* r, float* z);
```

Used as the default preconditioner. Diagonal-only, trivial to compute, trivially deterministic.

### Task 7.1.5 — Block-Jacobi preconditioner

`src/diffsim/solver/block_jacobi_preconditioner.cuh`:

For KKT systems from row scheduler, blocks correspond to constraint islands (graph-coloring output). Each island has a dense local block; invert per-block; apply per-block.

```cuda
// One block-jacobi for each island; block sizes vary.
// Use a Cholesky decomposition per block (since A is SPD per block under PGS).
// Determinism: block boundaries fixed by island partition, which is deterministic
// (from row scheduler).
```

Higher quality preconditioner than diagonal; significantly improves convergence rate for tightly coupled rigid systems.

### Task 7.1.6 — Backend selection plumbing

`src/diffsim/sparse_solver_backend.cpp`:

```cpp
std::unique_ptr<SparseLinearSolver> MakeSparseSolverBackend(std::string_view name,
                                                            const phi::DeviceContext& ctx) {
    if (name == "cudss")       return std::make_unique<CudssBackend>(ctx);
    if (name == "self_cg")     return std::make_unique<SelfWrittenCgBackend>(ctx);
    // Phases 2-3 add: self_minres, self_gmres, self_amg
    throw std::runtime_error("Unknown sparse solver backend");
}
```

C ABI:

```c
typedef enum {
    NUKA_SOLVER_BACKEND_CUDSS = 0,
    NUKA_SOLVER_BACKEND_SELF_CG = 1,
    NUKA_SOLVER_BACKEND_SELF_MINRES = 2,   /* v0.7 phase 2 */
    NUKA_SOLVER_BACKEND_SELF_GMRES = 3,    /* v0.7 phase 3 */
} nuka_sparse_solver_backend_t;

nuka_result_t nuka_world_set_sparse_solver_backend(nuka_world_handle w,
                                                    nuka_sparse_solver_backend_t backend);
```

### Task 7.1.7 — Tests

`tests/diffsim/solver/test_deterministic_dot.cu`:

```cpp
TEST(DeterministicDot, BitExactSameInputs) {
    auto ctx = MakeCtx();
    auto a = GenerateRandomDeviceVec(1000000, /*seed=*/42);
    auto b = GenerateRandomDeviceVec(1000000, /*seed=*/43);

    float r1 = deterministic_dot_host_caller(a, b);
    float r2 = deterministic_dot_host_caller(a, b);
    EXPECT_EQ(*(uint32_t*)&r1, *(uint32_t*)&r2);   // bit-exact
}
```

`tests/diffsim/solver/test_cg_vs_eigen.cpp`:

```cpp
TEST(CgVsEigen, AgreesOnRandomSpd) {
    // Generate SPD matrix A (e.g., A = M^T M for random M)
    // Solve A x = b with Eigen::ConjugateGradient (reference)
    // Solve with self-written CG
    // EXPECT_NEAR per-component < 1e-6
}
```

`tests/diffsim/solver/test_cg_convergence_rate.cpp`:

```cpp
TEST(CgConvergence, MatchesTheoryForKnownKappa) {
    // Build matrix with known condition number κ
    // CG should converge in O(sqrt(κ) * log(1/eps)) iterations
    // Assert iteration count within 1.5× theoretical bound
}
```

## Validation

- Deterministic dot bit-exact across runs.
- CG agrees with Eigen on SPD test matrices to 1e-6.
- CG agrees with cuDSS to 1e-5 on identical inputs.
- Block-Jacobi convergence rate ~2-5× better than plain Jacobi on rigid KKT systems.
- D1 strict: same input + same GPU + same driver → bit-exact result.
- No float atomics; lint passes.

## Exit Criteria for v0.7 Phase 1

1. `SelfWrittenCgBackend` registered and selectable via C ABI / Python.
2. Jacobi + Block-Jacobi preconditioners operational.
3. Determinism test passes.
4. cuDSS agreement test passes within 1e-5.
5. Convergence rate test confirms theoretical bound.
6. Lint clean; no float atomics.

## What This Phase Does Not Do

- No MINRES / GMRES (Phases 2 + 3).
- No ILU / AMG (Phases 2 + 3).
- Does not retire cuDSS — both backends coexist; cuDSS still default until v2.0 Phase 1.
- No new physics features — purely solver infrastructure work.
