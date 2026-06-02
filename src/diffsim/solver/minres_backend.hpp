#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- self-written deterministic MINRES backend (v0.7 p02)
// ---------------------------------------------------------------------------
//
// SelfWrittenMinresBackend: a SECOND backend behind the SAME SparseLinearSolver
// seam as the v0.5/p01 CG backend. MINRES (Paige & Saunders 1975) is the minimum-
// residual Krylov method for SYMMETRIC INDEFINITE systems -- exactly what the KKT
// systems become once unilateral contact / friction constraints make A have BOTH
// positive and negative eigenvalues. CG only handles SPD (its p^T A p / r^T M^-1 r
// ratios lose positive-definiteness on an indefinite A and the iteration breaks
// down); MINRES uses the Lanczos tridiagonalization + Givens rotations to minimize
// ||A x - b||_2 over the Krylov subspace and is well-defined for any SYMMETRIC A.
//
// LAYOUT (identical to the CG backend -- this is a drop-in on the batched-dense
// seam, NOT a new CSR path): ONE WARP (32 lanes) per articulation / SPD-or-
// indefinite block, n <= kMaxBlockDim(12). Lane L owns matrix row L (streamed once
// into registers, fp64) and component L of every Lanczos / solution vector. The
// SpMV  A v  is the SAME fixed-order warp-shuffle gather the CG backend uses; every
// inner product / 2-norm is the SAME fixed-order WarpButterflySum butterfly. There
// is NO cross-warp communication, NO atomics, NO cuBLAS, NO order-dependent
// accumulation -- so the solve is D1 byte-exact two-run (the Lanczos/Givens
// recurrence is a fixed-order sequential scalar update per block).
//
// Preconditioners (config via SolveParams::preconditioner, reusing the CG enum):
//   Jacobi      -- M^-1 = 1/|diag(A)|. For an indefinite A the diagonal can be
//                  negative; MINRES's preconditioner MUST be SPD (it defines the
//                  inner product the residual is minimized in), so we use the
//                  ABSOLUTE value of the diagonal. This is the standard "absolute-
//                  Jacobi" choice for indefinite MINRES.
//   BlockJacobi -- reinterpreted here as ILU(0): an incomplete LU factorization of
//                  the block on its existing sparsity pattern, applied as the
//                  preconditioner via a sparse forward/backward triangular solve.
//                  See ilu0_preconditioner.{hpp,cu} + triangular_solve.cu. ILU(0)
//                  is only well-defined / SPD-helpful for a (quasi-)definite A; the
//                  caller selects it for ill-conditioned DEFINITE systems, Jacobi
//                  (absolute) for genuinely indefinite ones (the preconditioner-
//                  quality vs convergence-correctness split, documented in tests).
//
// Determinism (D1, Strong): every dot / norm is the fixed-order butterfly; the
// Lanczos beta = sqrt(dot(v,v)) and the Givens rotations are fp64 scalar ops in a
// fixed loop order; the solution update x += tau * w is per-lane. Same input + same
// GPU => byte-identical x. fp64 internal arithmetic (loads fp32->fp64, stores
// fp64->fp32) matches the CG backend so the agreement gate holds.
// ---------------------------------------------------------------------------

#include "diffsim/sparse_solver_backend.hpp"
#include "phi/device_context.hpp"

#include <string_view>

namespace nuka::diffsim {

class SelfWrittenMinresBackend final : public SparseLinearSolver {
public:
    explicit SelfWrittenMinresBackend(
        const phi::DeviceContext& context,
        DeterminismLevel determinism = DeterminismLevel::Strong);

    void Solve(const BatchedDenseSpdSystem& system, const float* b, float* x,
               const SolveParams& params) override;

    std::string_view Name() const override { return "minres"; }

private:
    const phi::DeviceContext& context_;
    DeterminismLevel determinism_;
};

}  // namespace nuka::diffsim
