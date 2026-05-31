# Nuka Physics v0.5 – Phase 3: Self-Written Deterministic Sparse Solver + IFT-at-Convergence Solver

> **Master plan reference:** §3 Round 5 (path 3 IFT) + §2 decision 12 (self-written from v0.5) + §3 Round 3/13 amendment + §8 risk register
> **Prerequisites:** v0.5 Phase 1 (adjoint codegen), Phase 2 (tape)
> **Blocks:** v0.5 Phase 4 (PyTorch backward uses IFT for converged subsystems)
> **Exit criteria gate:** v0.5
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Add the **Implicit Function Theorem (IFT) at convergence** path to the diff-sim infrastructure. For subsystems whose solver demonstrably converges (e.g., Featherstone ABA, PBF density iteration when it lands), use IFT to backprop through the converged KKT system in **one sparse linear solve**, instead of recording full tape over the inner iterations.

This phase ships a **self-written deterministic sparse linear solver as the IFT backend — no closed-source SDK** (master plan §2 decision 12, §3 Round 3/13 amendment). The v0.5 scope (rigid + Featherstone) produces SPD KKT / Schur systems, so the minimal sufficient solver is **Conjugate Gradient (CG)** with **Jacobi / Block-Jacobi** preconditioning and **fixed-order tree reductions**, giving a **D1 bit-exact** IFT path from the start. The advanced methods — MINRES / ILU(0) / GMRES / AMG — extend this same self-written core in v0.7+; they are not in scope here.

Deliverables:

1. **Self-written CG + Jacobi/Block-Jacobi backend** as the IFT sparse solver. Always built — it is ours, not an optional third-party dependency.
2. **Sparse KKT system builder** — assembles the linearized constraint system from row data.
3. **IFT adjoint solver** — given downstream gradient, runs a sparse linear solve to produce upstream gradient.
4. **Subsystem opt-in** — per-row IR `gradient_mode = ift_at_convergence` triggers IFT path; other rows continue using tape (Phase 2).
5. **Featherstone IFT path** — Featherstone ABA convergence is guaranteed (forward dynamics is direct, not iterative). IFT for Featherstone-internal constraints would be via analytical inverse dynamics — but actually Featherstone forward has a closed-form adjoint already (Phase 1). IFT here mainly serves contact PGS when configured to iterate until residual converges, and PBF density (v0.7).
6. **Determinism** — the self-written CG backend uses fixed-order tree reductions and a fixed iteration cap, so the **IFT path is D1 bit-exact** (same input + same GPU + same driver → bit-exact result). No D2 caveat: IFT is usable directly under the D1 contract.

## Tech Stack

- Self-written CUDA CG (SPD Krylov solver) with Jacobi / Block-Jacobi preconditioners
- Deterministic fixed-order tree reductions (no float atomics)
- CUDA 12+
- Sparse matrix formats (CSR + COO)
- Existing diffsim Phase 2 infrastructure
- Eigen (host-side, for the dense reference solve in tests only)

## Files to Create

- `src/diffsim/sparse_solver_backend.hpp` — abstract sparse linear solver interface
- `src/diffsim/sparse_solver_cg.hpp` — self-written CG + Jacobi/Block-Jacobi backend
- `src/diffsim/sparse_solver_cg.cu`
- `src/diffsim/kkt_builder.hpp` — build linearized KKT from row data
- `src/diffsim/kkt_builder.cu`
- `src/diffsim/ift_runner.hpp` — IFT-mode adjoint runner
- `src/diffsim/ift_runner.cu`
- `tests/diffsim/test_kkt_build.cpp`
- `tests/diffsim/test_ift_vs_tape_backward.cpp`
- `tests/diffsim/test_cg_vs_dense.cpp` — self-written CG vs dense `Eigen::LDLT` reference
- `docs/architecture/diffsim-ift-design.md` — design note

## Tasks

### Task 5.3.1 — Sparse solver abstraction

`src/diffsim/sparse_solver_backend.hpp`:

```cpp
namespace nuka::diffsim {

class SparseLinearSolver {
public:
    virtual ~SparseLinearSolver() = default;

    // Set up symbolic structure (called once per topology)
    virtual void Analyze(const SparseMatrixCsr& A) = 0;

    // Numerical preparation (called when matrix values change; e.g. rebuild preconditioner)
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

This abstraction is the seam that v0.7+ extends with MINRES / GMRES / AMG backends — without changing call sites. In v0.5 the only registered backend is the self-written CG solver below.

### Task 5.3.2 — Self-written CG + Jacobi backend

`src/diffsim/sparse_solver_cg.hpp` / `.cu`:

A deterministic Conjugate Gradient solver for the SPD KKT / Schur systems that the v0.5 scope (rigid + Featherstone) produces. Determinism comes from (a) a **fixed iteration cap**, (b) **fixed-order tree reductions** for every dot product / norm, and (c) no float atomics anywhere in the loop.

```cpp
class SelfWrittenCgBackend : public SparseLinearSolver {
public:
    SelfWrittenCgBackend(const phi::DeviceContext& ctx, uint32_t max_iter = 200,
                         float tol = 1e-6f);
    ~SelfWrittenCgBackend() override;

    void Analyze(const SparseMatrixCsr& A) override {
        n_ = A.rows;
        // Allocate working vectors (r, p, Ap, z); CG is valid for SPD only.
        // Select preconditioner (Jacobi default; Block-Jacobi per island).
    }
    void Factorize(const SparseMatrixCsr& A) override {
        A_ = &A;
        // Rebuild the preconditioner cache (Jacobi diagonal / per-island blocks).
    }
    void Solve(const float* b, float* x) override {
        // Standard preconditioned CG with M^-1 (Jacobi / Block-Jacobi):
        //   r = b - A x;  z = M^-1 r;  p = z;  rho = <r, z>
        //   for k in [0, max_iter):
        //     Ap = A p
        //     alpha = rho / <p, Ap>            // deterministic dot
        //     x += alpha p;  r -= alpha Ap
        //     if <r, r> < tol^2: break          // deterministic dot
        //     z = M^-1 r;  rho_new = <r, z>;  beta = rho_new / rho
        //     p = z + beta p;  rho = rho_new
        // All reductions are fixed-order tree reductions → bit-exact.
    }
    DeterminismLevel Determinism() const override { return DeterminismLevel::Strong; }

private:
    uint32_t n_, max_iter_;
    float tol_;
    const SparseMatrixCsr* A_;
    DeviceVector<float> r_, p_, Ap_, z_;
    std::unique_ptr<class Preconditioner> precond_;   // Jacobi or Block-Jacobi
};
```

**Preconditioners:**
- **Jacobi** (diagonal): trivial to compute, trivially deterministic; the default.
- **Block-Jacobi** (per-island): blocks correspond to constraint islands from the graph-coloring output (deterministic partition). Each island has a dense local SPD block; invert per-block (Cholesky), apply per-block. Higher quality than diagonal; ~2–5× faster convergence on tightly coupled rigid KKT systems.

The CG solver is **always built** — there is no third-party `Find*.cmake` probe, no optional `-DNK_WITH_*` SDK flag, and no "build proceeds without IFT" fallback. The IFT path is unconditionally available because the solver is ours.

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

KKT builder assembles this sparse matrix in CSR from row Jacobian data already on device. (For the v0.5 SPD scope, the assembled system reduces to an SPD form — the Schur complement / regularized KKT — that CG solves directly; the indefinite-KKT general case is what MINRES handles in v0.7+.)

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

    // 2. Analyze (if topology changed) + (re)build preconditioner
    if (topology_changed_) solver_->Analyze(A);
    solver_->Factorize(A);

    // 3. Solve A·x = grad_outputs  (self-written deterministic CG)
    solver_->Solve(grad_outputs, x);

    // 4. Distribute x into grad_inputs via row Jacobian transpose
    DistributeAdjointToInputs(rows_with_ift_mode, x, grad_inputs);
}
```

The IFT math (transposed KKT solve, `BuildKktCsr`, `DistributeAdjointToInputs`) is **solver-agnostic** — it simply calls the self-written CG backend through the `SparseLinearSolver` interface.

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

`tests/diffsim/test_cg_vs_dense.cpp`:

```cpp
TEST(CgVsDense, AgreesWithEigenLdlt) {
    // Build a small SPD KKT/Schur system A, b
    // Reference: dense solve x_ref = Eigen::LDLT(A).solve(b)
    // Self-written CG: x_cg
    // EXPECT per-component relative error < 1e-6
}

TEST(CgDeterminism, BitExactSameInputs) {
    // Same A, b, same GPU + driver, run Solve twice → bit-exact result
}
```

## Validation

- Self-written CG solves the SPD KKT/Schur system; numerical error vs a **dense reference** (`Eigen::LDLT`, host) < 1e-6.
- KKT builder produces correct CSR matrix for known test cases.
- IFT backward matches tape backward within 1e-4 relative for rows where both modes apply.
- IFT gradients pass the **V3 finite-difference check** (< 1e-3 relative).
- Cached symbolic structure: when topology unchanged, only `Factorize` (preconditioner rebuild) runs; perf benchmark confirms.
- Determinism: the IFT path is **D1 bit-exact** — same input + same GPU + same driver → bit-exact result. No D2 mode, no warning when IFT is requested under the D1 contract.

## Exit Criteria for v0.5 Phase 3

1. Self-written deterministic CG + Jacobi/Block-Jacobi backend implemented and **always built** (no optional-dependency flag, no closed-source SDK).
2. `SparseLinearSolver` abstraction in place; the v0.7+ MINRES/GMRES/AMG backends plug into the same interface.
3. KKT builder produces correct sparse matrix for v0.1 row classes.
4. IFT runner solves and distributes gradients correctly.
5. Backward runner routes rows by `gradient_mode`.
6. CG vs dense-reference (`Eigen::LDLT`) agreement < 1e-6; IFT vs tape agreement test passes; V3 FD check passes.
7. IFT path is **D1 bit-exact** (determinism test passes).
8. Performance: IFT path is ≥ 1.5× faster than tape recompute for Featherstone subsystem on 4096 envs.

## What This Phase Does Not Do

- Does **not** ship MINRES / GMRES / AMG or ILU(0). Those extend the self-written core in v0.7+ (`v07-pXX-sparse-solver-*.md`). v0.5 ships only the minimal CG + Jacobi/Block-Jacobi needed for the SPD IFT path.
- Does not switch the default sparse backend. Tape (Phase 2) remains default; IFT is opt-in per row class.
- Does not add IFT for soft / fluid (v0.7 PBF density may opt in later).
- Does not depend on any closed-source SDK — the solver is self-written and always built.
