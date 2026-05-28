# Nuka Physics v0.7 – Phase 3: Self-Written Sparse Solver — GMRES + AMG Preconditioner

> **Master plan reference:** §2 decision #12 + §8 risk register
> **Prerequisites:** v0.7 Phase 2 (MINRES + ILU operational)
> **Blocks:** v2.0 Phase 1 (full cuDSS retirement)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Complete the self-written sparse solver suite with **GMRES** (Generalized Minimum Residual) for non-symmetric systems and **AMG** (Algebraic Multigrid) as the heavy-weight preconditioner.

After this phase, the self-written solver suite is **feature-complete** for all expected linear systems in the engine:

- SPD → CG + Jacobi/Block-Jacobi (Phase 1)
- Symmetric indefinite → MINRES + ILU(0) (Phase 2)
- Non-symmetric → GMRES + ILU(0)/AMG (Phase 3)
- Large stiff systems → AMG-preconditioned Krylov (Phase 3)

cuDSS is still the default in v0.7 (avoiding rocking the boat during S2 development); v2.0 Phase 1 makes self-written the default.

## Tech Stack

- CUDA 12+
- Existing Phases 1-2 infrastructure
- Algebraic multigrid theory (smoothed aggregation; Vaněk et al.)

## Files to Create

- `src/diffsim/solver/gmres_backend.hpp`
- `src/diffsim/solver/gmres_backend.cu`
- `src/diffsim/solver/amg_preconditioner.hpp`
- `src/diffsim/solver/amg_setup.cu` — coarsening + interpolation
- `src/diffsim/solver/amg_apply.cu` — V-cycle / W-cycle
- `src/diffsim/solver/amg_smoother.cu` — Gauss-Seidel / Chebyshev smoother
- `src/diffsim/solver/sparse_matrix_multiply.cu` — RAP coarsening triple product
- `tests/diffsim/solver/test_gmres_nonsymmetric.cpp`
- `tests/diffsim/solver/test_amg_setup.cpp`
- `tests/diffsim/solver/test_amg_vs_ilu0_stiff.cpp`
- `tests/diffsim/solver/test_full_suite_vs_cudss.cpp` — comprehensive comparison
- `docs/architecture/sparse-solver-suite.md` — final design + perf characteristics

## Files to Modify

- `src/diffsim/sparse_solver_backend.cpp` — register `self_gmres` + AMG options
- `src/include/nuka/nuka_diffsim.h` — `NUKA_SOLVER_BACKEND_SELF_GMRES` enum
- `src/diffsim/ift_runner.cu` — auto-route to GMRES for non-symmetric

## Tasks

### Task 7.3.1 — GMRES algorithm

Restarted GMRES(m): m = restart length, typically 30-50.

Internals: builds Krylov basis via Arnoldi iteration; solves least-squares problem via Givens rotations. Reference: Saad & Schultz 1986.

```cpp
class SelfWrittenGmresBackend : public SparseLinearSolver {
public:
    SelfWrittenGmresBackend(const phi::DeviceContext& ctx, uint32_t restart = 30,
                             uint32_t max_iter = 200, float tol = 1e-6f);

    void Solve(const float* b, float* x) override {
        // Outer loop: restart cycles
        // Inner loop: Arnoldi build Krylov basis
        // At end of each inner cycle: solve least squares for y; x += V y
        // No restart needed if converged inside one cycle
    }
};
```

GMRES memory grows as O(restart × n) — manageable at restart=30.

Determinism: Arnoldi orthogonalization via modified Gram-Schmidt (deterministic order) using Phase 1's deterministic dot.

### Task 7.3.2 — AMG setup (coarsening)

`src/diffsim/solver/amg_setup.cu`:

Algorithm: smoothed aggregation AMG (Vaněk et al. 1996).

Setup steps:
1. Compute strength-of-connection matrix from A.
2. Aggregate fine nodes into coarse aggregates.
3. Build prolongation operator P (interpolation; tentative + smoothed).
4. Coarsen A: A_c = P^T A P (sparse matrix triple product).
5. Recurse: build AMG on A_c until size threshold.

```cuda
// Strength-of-connection: |a_ij| > theta * sqrt(|a_ii| * |a_jj|)
__global__ void compute_strength_matrix_kernel(...);

// Aggregation: greedy maximal independent set
__global__ void aggregate_kernel(...);

// Build tentative prolongation: 1 at (node, its-aggregate); normalize
__global__ void build_tentative_prolongation_kernel(...);

// Jacobi-smooth the tentative prolongation
__global__ void smooth_prolongation_kernel(...);
```

Determinism: aggregation uses a deterministic tiebreak (node index) to avoid race-dependent variation.

### Task 7.3.3 — AMG application (V-cycle)

`src/diffsim/solver/amg_apply.cu`:

V-cycle:
```
function v_cycle(level, r, e):
    if level == coarsest:
        solve directly (small enough; use Cholesky)
        return
    pre-smooth: e = smooth(A_level, e, r) for n_pre iterations
    compute residual: r_new = r - A_level e
    restrict to coarse: r_coarse = P^T r_new
    v_cycle(level+1, r_coarse, e_coarse)
    interpolate: e += P e_coarse
    post-smooth: e = smooth(A_level, e, r) for n_post iterations
```

Smoother choices: Gauss-Seidel (sequential, harder to parallelize) or Chebyshev (parallel-friendly; recommended for GPU).

Determinism: Chebyshev smoother is naturally deterministic (only matrix-vector and AXPY ops).

### Task 7.3.4 — RAP coarsening triple product

Sparse matrix multiplication: `A_coarse = P^T * A * P`.

Two sparse-sparse matrix multiplies + transpose. Output sparsity pattern unknown a priori — use symbolic phase to predict pattern, then numeric phase.

Reference: NVIDIA's `cuSPARSE` SpGEMM design (we implement a deterministic variant; cuSPARSE makes no determinism guarantees).

### Task 7.3.5 — Integration

The C ABI exposes a `nuka_solver_config_t` struct allowing fine control:

```c
typedef struct {
    nuka_sparse_solver_backend_t backend;   /* CG / MINRES / GMRES / AUTO */
    uint8_t preconditioner;                 /* JACOBI / BLOCK_JACOBI / ILU0 / AMG */
    uint32_t max_iterations;
    float    tolerance;
    uint32_t gmres_restart;                 /* GMRES only */
    uint32_t amg_max_levels;                /* AMG only */
} nuka_solver_config_t;
```

Sensible defaults: AUTO routes by matrix symmetry / definiteness detection.

### Task 7.3.6 — Tests

`tests/diffsim/solver/test_full_suite_vs_cudss.cpp`:

```cpp
// Sweep matrix types: SPD, symmetric indefinite, non-symmetric
// Compare self-written suite vs cuDSS on each
// Assert agreement < 1e-5 on the entire suite
TEST(FullSuiteVsCudss, AllMatrixTypes) {
    for (auto& mat : test_matrix_zoo()) {
        for (auto backend : {"self_cg", "self_minres", "self_gmres"}) {
            auto x_self = SolveWith(mat, backend);
            auto x_cudss = SolveWith(mat, "cudss");
            EXPECT_LT(RelativeError(x_self, x_cudss), 1e-5);
        }
    }
}
```

`tests/diffsim/solver/test_amg_vs_ilu0_stiff.cpp`:

```cpp
// Generate stiff matrix (condition number 10^6+)
// AMG converges in ≤ 10 iterations
// ILU(0) requires > 50 iterations
// AMG total time should be less even with higher setup cost
```

## Validation

- GMRES converges on non-symmetric test matrices.
- AMG setup produces correct coarsening hierarchy.
- AMG V-cycle reduces residual by expected factor per cycle.
- Full suite agreement with cuDSS < 1e-5 on the matrix zoo.
- AMG outperforms ILU(0) on stiff matrices by ≥ 5× total time.
- Deterministic: bit-exact across runs.
- D1 contract preserved.

## Exit Criteria for v0.7 Phase 3

1. GMRES backend operational.
2. AMG preconditioner operational (setup + apply).
3. Full self-written suite (CG, MINRES, GMRES) agrees with cuDSS within 1e-5.
4. AUTO mode correctly routes by matrix structure.
5. AMG performance benefits demonstrated on stiff systems.
6. Documentation written (`docs/architecture/sparse-solver-suite.md`).
7. Determinism tests pass.
8. Lint clean.

## What This Phase Does Not Do

- Does not retire cuDSS (that's v2.0 Phase 1). Both coexist; cuDSS still default.
- Does not implement BiCGStab / IDR (not needed for our problems).
- Does not implement geometric multigrid (would require mesh hierarchy from physics; AMG suffices).
- Does not handle complex-valued systems.
