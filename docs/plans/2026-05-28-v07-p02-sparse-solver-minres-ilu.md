# Nuka Physics v0.7 – Phase 2: Self-Written Sparse Solver — MINRES + ILU(0) Preconditioner

> **Master plan reference:** §2 decision #12 + §8 risk register
> **Prerequisites:** v0.7 Phase 1 (CG + Jacobi backend in place)
> **Blocks:** v0.7 Phase 3 (GMRES + AMG round out the suite)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Extend the self-written sparse solver suite with **MINRES** (Minimum Residual) and **ILU(0)** (Incomplete LU with zero fill-in) preconditioner. MINRES handles **symmetric indefinite** systems — exactly what KKT systems from constraint solvers look like once contact / friction add unilateral constraints. CG works only for SPD, so MINRES extends the operating range significantly.

ILU(0) is the workhorse preconditioner for indefinite systems. It runs incomplete factorization on the matrix's existing sparsity pattern (no fill-in beyond what's there), producing an L * U ≈ A that's cheap to apply as preconditioner.

## Tech Stack

- CUDA 12+
- Existing CG + Jacobi infrastructure from v0.7 Phase 1
- Lanczos iteration math + Givens rotations (MINRES internals)

## Files to Create

- `src/diffsim/solver/minres_backend.hpp`
- `src/diffsim/solver/minres_backend.cu`
- `src/diffsim/solver/ilu0_preconditioner.hpp`
- `src/diffsim/solver/ilu0_preconditioner.cu` — symbolic + numeric factorization
- `src/diffsim/solver/triangular_solve.cu` — sparse triangular solve (forward / backward sub)
- `tests/diffsim/solver/test_minres_indefinite.cpp` — solve known indefinite system
- `tests/diffsim/solver/test_ilu0_factorization.cpp` — known reference factorization
- `tests/diffsim/solver/test_minres_vs_cudss_indefinite.cpp` — agreement test
- `tests/diffsim/solver/test_ilu0_vs_jacobi_convergence.cpp` — preconditioner quality comparison

## Files to Modify

- `src/diffsim/sparse_solver_backend.cpp` — register `self_minres` factory key
- `src/include/nuka/nuka_diffsim.h` — `NUKA_SOLVER_BACKEND_SELF_MINRES` enum
- `src/diffsim/ift_runner.cu` — choose MINRES automatically when KKT detected indefinite

## Tasks

### Task 7.2.1 — MINRES algorithm

`src/diffsim/solver/minres_backend.cu`:

MINRES uses Lanczos iteration to build an orthonormal basis of the Krylov subspace; at each step, applies Givens rotations to maintain a QR factorization of the tridiagonal subproblem. Reference: Paige & Saunders 1975.

Skeleton:

```cpp
class SelfWrittenMinresBackend : public SparseLinearSolver {
public:
    void Solve(const float* b, float* x) override {
        // Lanczos basis: v_old, v_cur, v_new
        // Tridiagonal entries: alpha_k, beta_k
        // Givens rotations: c1, s1, c2, s2
        // Residual norm bound: tracked via ||r||
        //
        // Each iteration:
        //   v_new = A v_cur - alpha v_cur - beta v_old   (Lanczos)
        //   beta_new = ||v_new||
        //   v_new /= beta_new
        //   Apply rotations to tridiagonal
        //   Update x = x + tau * w
        //   Check ||r|| < tol
    }
    DeterminismLevel Determinism() const override { return DeterminismLevel::Strong; }
};
```

All vector operations (dot, axpy) use the deterministic primitives from Phase 1. Beta values come from a deterministic L2 norm (norm = sqrt(dot(v, v))).

### Task 7.2.2 — ILU(0) preconditioner

`src/diffsim/solver/ilu0_preconditioner.cu`:

Symbolic phase: identify the same sparsity pattern as A — no fill-in.

Numeric phase: standard ILU(0) loop:

```
for i = 0 to n-1:
    for each k in row i where col(k) < i:        # lower triangle
        L[i,k] = A[i,k] / U[k,k]
        for each j in row i where col(j) > k AND (i,j) in pattern:
            U[i,j] -= L[i,k] * U[k,j]
```

GPU parallelism: ILU has serial structure along the natural row ordering. Use **level scheduling** — partition rows into "levels" where each level can be processed in parallel (no dependence on rows in same level). This is the standard CUDA ILU(0) approach (referenced from cuSPARSE's design).

Determinism: level partition is deterministic (fixed by sparsity pattern). Within a level, operations are parallel but independent. Final result is bit-exact across runs.

### Task 7.2.3 — Sparse triangular solve

ILU preconditioner application requires solving L y = r then U z = y. Each is a sparse triangular solve. Use level scheduling again.

```cuda
__global__ void triangular_solve_lower_level_kernel(
    uint32_t level_start, uint32_t level_end,
    const uint32_t* row_perm,
    const uint32_t* row_ptr, const uint32_t* col_idx, const float* values,
    const float* rhs, float* sol);
```

One kernel launch per level; within a level, all rows are independent.

### Task 7.2.4 — Backend integration

`SelfWrittenMinresBackend` now supports either Jacobi or ILU(0) as preconditioner, selectable via config.

ILU(0) is more expensive to factorize (per-update cost) but reduces iteration count significantly for indefinite KKT systems. Heuristic: use Jacobi for small systems / when topology changes frequently; use ILU(0) for large stable topologies.

### Task 7.2.5 — Auto-routing in IFT runner

`src/diffsim/ift_runner.cu`:

```cpp
void IftRunner::ChooseBackend(const SparseMatrixCsr& A) {
    bool indefinite = DetectIndefinite(A);   // any negative diagonal pivot?
    if (indefinite) {
        solver_ = MakeSparseSolverBackend("self_minres", ctx_);
    } else {
        solver_ = MakeSparseSolverBackend("self_cg", ctx_);
    }
}
```

The C ABI also lets users override.

### Task 7.2.6 — Tests

`tests/diffsim/solver/test_minres_indefinite.cpp`:

```cpp
TEST(MinresIndefinite, KnownTestProblem) {
    // Build A with both positive and negative eigenvalues
    // Solve with self-written MINRES
    // Verify ||A x - b|| < tol
}
```

`tests/diffsim/solver/test_ilu0_factorization.cpp`:

```cpp
TEST(Ilu0, MatchesReferenceImplementation) {
    // Small matrix; reference ILU(0) computed by hand
    // Self-written matches per-element to 1e-7
}
```

`tests/diffsim/solver/test_ilu0_vs_jacobi_convergence.cpp`:

```cpp
TEST(Ilu0VsJacobi, FewerIterations) {
    // Stiff rigid KKT system
    // ILU(0)-preconditioned MINRES converges in ≤ 0.5× iterations vs Jacobi
}
```

## Validation

- MINRES solves indefinite systems where CG cannot.
- ILU(0) factorization matches reference to 1e-7 element-wise.
- ILU(0)-preconditioned MINRES converges in ≤ 0.5× iterations vs Jacobi-preconditioned.
- Determinism: bit-exact across runs.
- D1 contract preserved (no float atomics).
- Per-step time penalty vs Phase 1 CG: ≤ 20% on suitable problems (MINRES has more memory traffic).

## Exit Criteria for v0.7 Phase 2

1. `SelfWrittenMinresBackend` operational with both Jacobi and ILU(0) preconditioners.
2. Indefinite KKT systems solved correctly.
3. Auto-routing chooses MINRES when indefinite detected.
4. cuDSS agreement test on indefinite systems within 1e-5.
5. ILU(0) outperforms Jacobi on test KKT problems.
6. Lint + determinism tests pass.

## What This Phase Does Not Do

- No GMRES (Phase 3) — that handles non-symmetric.
- No AMG (Phase 3) — that's the heavy-weight preconditioner.
- Does not yet retire cuDSS.
- No new physics.
