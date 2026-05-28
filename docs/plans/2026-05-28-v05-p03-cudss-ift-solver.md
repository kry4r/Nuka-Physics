# Nuka Physics v0.5 – Phase 3: cuDSS Integration + IFT-at-Convergence Solver

> **Master plan reference:** §3 Round 5 (path 3 IFT) + §2 decision 12 (cuDSS phase 1) + §8 risk register
> **Prerequisites:** v0.5 Phase 1 (adjoint codegen), Phase 2 (tape)
> **Blocks:** v0.5 Phase 4 (PyTorch backward uses IFT for converged subsystems)
> **Exit criteria gate:** v0.5
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Add the **Implicit Function Theorem (IFT) at convergence** path to the diff-sim infrastructure. For subsystems whose solver demonstrably converges (e.g., Featherstone ABA, PBF density iteration when it lands), use IFT to backprop through the converged KKT system in **one sparse linear solve**, instead of recording full tape over the inner iterations.

This phase ships **cuDSS as the initial sparse linear solver backend** (master plan decision 12 phase 1). Self-written deterministic CG / MINRES / GMRES / AMG replaces cuDSS post-v0.5 (separate v0.7+ phase).

Deliverables:

1. **cuDSS integration** as a runtime-loaded dependency (does not become a hard build requirement).
2. **Sparse KKT system builder** — assembles the linearized constraint system from row data.
3. **IFT adjoint solver** — given downstream gradient, runs a sparse linear solve to produce upstream gradient.
4. **Subsystem opt-in** — per-row IR `gradient_mode = ift_at_convergence` triggers IFT path; other rows continue using tape (Phase 2).
5. **Featherstone IFT path** — Featherstone ABA convergence is guaranteed (forward dynamics is direct, not iterative). IFT for Featherstone-internal constraints would be via analytical inverse dynamics — but actually Featherstone forward has a closed-form adjoint already (Phase 1). IFT here mainly serves contact PGS when configured to iterate until residual converges, and PBF density (v0.7).
6. **Determinism** — cuDSS reduction order is not bit-exact deterministic across runs. Document this; flag IFT path as D2 mode by default with note that self-written replacement (v0.7+) restores D1.

## Tech Stack

- cuDSS 0.3+ (NVIDIA sparse direct solver)
- CUDA 12+
- Sparse matrix formats (CSR + COO)
- Existing diffsim Phase 2 infrastructure

## Files to Create

- `src/diffsim/sparse_solver_backend.hpp` — abstract sparse linear solver interface
- `src/diffsim/sparse_solver_backend_cudss.cpp` — cuDSS implementation
- `src/diffsim/sparse_solver_backend_cudss.cu`
- `src/diffsim/kkt_builder.hpp` — build linearized KKT from row data
- `src/diffsim/kkt_builder.cu`
- `src/diffsim/ift_runner.hpp` — IFT-mode adjoint runner
- `src/diffsim/ift_runner.cu`
- `tests/diffsim/test_kkt_build.cpp`
- `tests/diffsim/test_ift_vs_tape_backward.cpp`
- `cmake/Findcudss.cmake` — locate cuDSS library
- `docs/architecture/diffsim-ift-design.md` — design note

## Tasks

### Task 5.3.1 — Sparse solver abstraction

`src/diffsim/sparse_solver_backend.hpp`:

```cpp
namespace nuka::diffsim {

class SparseLinearSolver {
public:
    virtual ~SparseLinearSolver() = default;

    // Set up symbolic factorization (called once per topology)
    virtual void Analyze(const SparseMatrixCsr& A) = 0;

    // Numerical factorization (called when matrix values change)
    virtual void Factorize(const SparseMatrixCsr& A) = 0;

    // Solve A·x = b
    virtual void Solve(const float* b, float* x) = 0;

    // Determinism level this backend provides
    virtual DeterminismLevel Determinism() const = 0;
};

// Factory; runtime selects backend
std::unique_ptr<SparseLinearSolver> MakeSparseSolverBackend(const std::string_view backend_name,
                                                            const phi::DeviceContext& ctx);

} // namespace
```

This abstraction lets us swap cuDSS for self-written later without changing call sites.

### Task 5.3.2 — cuDSS implementation

`src/diffsim/sparse_solver_backend_cudss.cpp`:

```cpp
#include <cuDSS.h>

class CudssBackend : public SparseLinearSolver {
public:
    CudssBackend(const phi::DeviceContext& ctx);
    ~CudssBackend() override;

    void Analyze(const SparseMatrixCsr& A) override {
        // cuDSS analyze phase
        cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_, data_, matrix_A_, x_, b_);
    }
    void Factorize(const SparseMatrixCsr& A) override {
        // Update values in matrix_A_, then run factorization
        cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, config_, data_, matrix_A_, x_, b_);
    }
    void Solve(const float* b, float* x) override {
        // Run solve phase
        cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, data_, matrix_A_, x_, b_);
    }
    DeterminismLevel Determinism() const override { return DeterminismLevel::Weak; }

private:
    cudssHandle_t handle_;
    cudssConfig_t config_;
    cudssData_t   data_;
    cudssMatrix_t matrix_A_, x_, b_;
};
```

CMake `Findcudss.cmake` locates the cuDSS library; if missing, the build proceeds without IFT support and tape-only backward is used.

### Task 5.3.3 — KKT system builder

For a converged PGS iteration with active constraints, the KKT matrix has structure:

```
[ M  J^T ] [ Δv ]   [ -r_dyn ]
[ J  0   ] [ Δλ ] = [ -r_con ]
```

where `M` is mass matrix, `J` is constraint Jacobian, `r_dyn` is dynamics residual, `r_con` is constraint residual.

For IFT adjoint, we solve the transposed system:

```
[ M  J^T ]^T [ adj_v ]   [ grad_v_in ]
[ J  0   ]   [ adj_λ ] = [ grad_λ_in ]
```

KKT builder assembles this sparse matrix in CSR from row Jacobian data already on device.

`src/diffsim/kkt_builder.cu`:

```cuda
void BuildKktCsr(const RowBuffers& rows, const BodyState& bodies,
                 SparseMatrixCsr& out_A)
{
    // Pass 1: count nonzeros per row of A
    // Pass 2: prefix-sum to row pointers
    // Pass 3: write column indices + values
    // No atomics; deterministic ordering
}
```

### Task 5.3.4 — IFT runner

`src/diffsim/ift_runner.cu`:

```cpp
void IftRunner::RunIftBackward(const RowSubset& rows_with_ift_mode,
                                const float* grad_outputs,
                                float* grad_inputs)
{
    // 1. Build KKT matrix for the subset
    SparseMatrixCsr A;
    BuildKktCsr(rows_with_ift_mode, ..., A);

    // 2. Analyze + factorize (cached if topology unchanged)
    if (topology_changed_) solver_->Analyze(A);
    solver_->Factorize(A);

    // 3. Solve A·x = grad_outputs
    solver_->Solve(grad_outputs, x);

    // 4. Distribute x into grad_inputs via row Jacobian transpose
    DistributeAdjointToInputs(rows_with_ift_mode, x, grad_inputs);
}
```

### Task 5.3.5 — Backward runner integration

`src/diffsim/backward_runner.cu` extended to route rows by `gradient_mode`:

```cpp
void BackwardRunner::RunStep(uint32_t step, const float* grad_obs) {
    // Group rows by gradient_mode
    auto tape_rows = FilterByMode(step, GradientMode::DenseAdjointStopGrad);
    auto ift_rows  = FilterByMode(step, GradientMode::IftAtConvergence);

    // Tape-based rows: codegen adjoint per Phase 1
    DispatchAdjointKernels(tape_rows, grad_obs, accumulator_);

    // IFT-based rows: sparse solve
    ift_runner_.RunIftBackward(ift_rows, grad_obs, accumulator_);
}
```

### Task 5.3.6 — Tests

`tests/diffsim/test_ift_vs_tape_backward.cpp`:

```cpp
TEST(IftVsTape, BackwardAgreement) {
    // Same scene; same forward; same downstream gradient
    // Run backward two ways: full-tape Phase 2 vs IFT Phase 3
    // For row subset with both gradient modes supported, gradients should agree to 1e-4 relative
    // (some difference allowed because IFT computes through the converged KKT,
    //  while tape steps through iterations).
}
```

`tests/diffsim/test_kkt_build.cpp`:

```cpp
TEST(KktBuilder, MatchesAnalyticalForSmallScene) {
    // 4-body chain; build KKT; compare to hand-computed reference
}
```

## Validation

- cuDSS solver factorizes + solves; numerical error vs dense `Eigen::LDLT` < 1e-6.
- KKT builder produces correct CSR matrix for known test cases.
- IFT backward matches tape backward within 1e-4 relative for rows where both modes apply.
- Cached symbolic factorization: when topology unchanged, only Factorize runs; perf benchmark confirms.
- Determinism: IFT mode is documented as D2; D1 strict mode uses tape only (and emits a warning if IFT is requested).

## Exit Criteria for v0.5 Phase 3

1. cuDSS integrated as optional backend (`-DNK_WITH_CUDSS=ON` flag).
2. `SparseLinearSolver` abstraction in place; future self-written backend (v0.7+) plugs in.
3. KKT builder produces correct sparse matrix for v0.1 row classes.
4. IFT runner solves and distributes gradients correctly.
5. Backward runner routes rows by `gradient_mode`.
6. IFT vs tape agreement test passes.
7. Performance: IFT path is ≥ 1.5× faster than tape recompute for Featherstone subsystem on 4096 envs.

## What This Phase Does Not Do

- Does **not** self-write CG / MINRES / GMRES / AMG. That is a post-v0.5 phase (`v07-pXX-self-written-sparse-solver.md`, written when v0.7 is being planned).
- Does not switch the default sparse backend. Tape (Phase 2) remains default; IFT is opt-in per row class.
- Does not add IFT for soft / fluid (v0.7 PBF density may opt in later).
- Does not implement preconditioner choices (self-written backend will).
