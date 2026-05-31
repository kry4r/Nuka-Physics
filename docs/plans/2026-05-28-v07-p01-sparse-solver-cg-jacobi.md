# Nuka Physics v0.7 – Phase 1: Harden the Self-Written CG + Jacobi / Block-Jacobi Core into the General Solver Path

> **Master plan reference:** §2 decision #12 (self-written from v0.5) + §3 Round 3/13 amendment + §8 risk register
> **Prerequisites:** v0.5 closed (self-written CG + Jacobi/Block-Jacobi IFT core in place)
> **Blocks:** v0.7 Phase 2 (MINRES + ILU builds on this CG foundation)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

v0.5 Phase 3 already shipped a self-written deterministic **Conjugate Gradient (CG)** backend with **Jacobi** and **Block-Jacobi** preconditioners as the IFT solver, scoped to the SPD KKT / Schur systems of the rigid + Featherstone path. This phase **hardens and generalizes that core into the engine's general (non-IFT) sparse solver path**: larger and worse-conditioned systems, robust convergence diagnostics, and Block-Jacobi tuning. It is not "introducing CG to replace a third-party solver" — the project ships no closed-source SDK; the self-written CG already exists from v0.5.

By the end of this phase, the `SparseLinearSolver` abstraction (introduced v0.5 Phase 3) exposes a production-hardened `SelfWrittenCgBackend` usable across the engine, not just the IFT path. Results agree with a dense reference (Eigen LDLT) within 1e-6 relative on standard test matrices, and cross-method agreement is checked once MINRES/GMRES land.

This is **not** the full suite — MINRES, ILU(0), GMRES, AMG land in Phases 2 and 3 as further extensions of this same self-written core.

## Tech Stack

- CUDA 12+
- CUB / Thrust (for warp-level primitives; deterministic reduction tree variants)
- Existing `SparseLinearSolver` abstraction (v0.5 Phase 3)

## Files to Create

- `src/diffsim/solver/cg_diagnostics.{hpp,cuh}` — convergence diagnostics (residual history, stall / divergence detection, iteration-budget reporting) for the general path
- `tests/diffsim/solver/test_cg_vs_dense.cpp` — self-written CG vs a dense reference (Eigen LDLT) on general / worse-conditioned systems; agreement < 1e-6
- `tests/diffsim/solver/test_cg_convergence_rate.cpp` — verifies κ-conditioning matches theory on the harder general-path matrices
- `tests/diffsim/solver/test_block_jacobi_tuning.cpp` — Block-Jacobi block-size / island-grouping tuning sweep

## Files to Modify (the v0.5 self-written core, extended here)

- `src/diffsim/sparse_solver_cg.{hpp,cu}` — the CG + Jacobi/Block-Jacobi backend shipped in v0.5 Phase 3 for IFT; harden for the general (non-IFT) path: larger / worse-conditioned systems, convergence diagnostics, Block-Jacobi tuning
- `src/diffsim/sparse_solver_backend.hpp` — `SelfWrittenCgBackend` factory key (already registered in v0.5; extend for general-path config)
- `src/include/nuka/nuka_diffsim.h` — `NUKA_SOLVER_BACKEND_SELF_CG` enum value (already present from v0.5)
- `src/c_abi/diffsim.cpp` — extend backend selection / solver config plumbing
- `tests/diffsim/solver/test_cg_vs_eigen.cpp`, `tests/diffsim/solver/test_deterministic_dot.cu` — v0.5 correctness/determinism tests; extend coverage to the general-path matrices

## Tasks

> **Note — these recap the v0.5 self-written core that this phase hardens.** The deterministic dot/axpy/SpMV primitives, the CG main loop, and the Jacobi / Block-Jacobi preconditioners were all built in v0.5 Phase 3 (`src/diffsim/sparse_solver_cg.{hpp,cu}`) for the SPD IFT path. Tasks 7.1.1–7.1.5 below restate that core for context; the genuinely-new work in this phase is **convergence diagnostics, Block-Jacobi tuning, and hardening for larger / worse-conditioned general-path systems** (Task 7.1.6 onward).

### Task 7.1.1 — Deterministic dot product (recap of v0.5 core)

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

### Task 7.1.2 — SpMV (sparse matrix-vector) (recap of v0.5 core)

CSR format SpMV with deterministic per-row order:

```cuda
__global__ void spmv_csr_kernel(uint32_t rows, const uint32_t* row_ptr,
                                 const uint32_t* col_idx, const float* values,
                                 const float* x, float* y);
```

Each row is processed by one thread (or a warp for long rows). Within a row, accumulation is sequential along sorted column indices → deterministic regardless of how warps execute.

### Task 7.1.3 — CG main loop (recap of v0.5 core)

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

### Task 7.1.4 — Jacobi preconditioner (recap of v0.5 core)

`src/diffsim/solver/jacobi_preconditioner.cuh`:

```cuda
__global__ void jacobi_extract_diag_kernel(uint32_t rows, const uint32_t* row_ptr,
                                            const uint32_t* col_idx, const float* values,
                                            float* diag);

__global__ void jacobi_apply_kernel(uint32_t n, const float* diag_inv,
                                     const float* r, float* z);
```

Used as the default preconditioner. Diagonal-only, trivial to compute, trivially deterministic.

### Task 7.1.5 — Block-Jacobi preconditioner (recap of v0.5 core; tuned here for the general path)

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
    if (name == "self_cg")     return std::make_unique<SelfWrittenCgBackend>(ctx);
    // Phases 2-3 add: self_minres, self_gmres, self_amg
    throw std::runtime_error("Unknown sparse solver backend");
}
```

C ABI:

```c
typedef enum {
    NUKA_SOLVER_BACKEND_SELF_CG = 0,       /* self-written; default (shipped v0.5) */
    NUKA_SOLVER_BACKEND_SELF_MINRES = 1,   /* v0.7 phase 2 */
    NUKA_SOLVER_BACKEND_SELF_GMRES = 2,    /* v0.7 phase 3 */
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
- CG agrees with a dense reference (Eigen LDLT) to 1e-6 on identical inputs; cross-method agreement checked once MINRES/GMRES land (Phases 2-3).
- Block-Jacobi convergence rate ~2-5× better than plain Jacobi on rigid KKT systems.
- D1 strict: same input + same GPU + same driver → bit-exact result.
- No float atomics; lint passes.

## Exit Criteria for v0.7 Phase 1

1. `SelfWrittenCgBackend` hardened for the general (non-IFT) path and selectable via C ABI / Python.
2. Jacobi + Block-Jacobi preconditioners operational, with Block-Jacobi tuning for the general path.
3. Determinism test passes (D1 bit-exact).
4. Dense-reference (Eigen LDLT) agreement test passes within 1e-6.
5. Convergence rate / diagnostics test confirms theoretical bound.
6. Lint clean; no float atomics.

## What This Phase Does Not Do

- No MINRES / GMRES (Phases 2 + 3).
- No ILU / AMG (Phases 2 + 3).
- No closed-source SDK anywhere — the CG core was self-written from v0.5; this phase only hardens and generalizes it.
- No new physics features — purely solver infrastructure work.
