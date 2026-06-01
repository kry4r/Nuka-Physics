#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- self-written deterministic CG backend (v0.5 p03 R1)
// ---------------------------------------------------------------------------
//
// SelfWrittenCgBackend: the ONLY v0.5 backend behind SparseLinearSolver. A
// preconditioned Conjugate Gradient solve of the batched-dense SPD Delassus
// system, ONE WARP (32 lanes) per articulation/block. The <= 12x12 block and the
// CG working vectors (x, r, z, p, Ap) are warp-resident: lane L owns row L of the
// block (its row of A in registers) and component L of every vector. There is NO
// cross-warp communication and NO atomics; each warp runs the whole CG iteration
// for its block independently. Grid = block_count -> one block per articulation
// across all envs, massively parallel.
//
// Determinism (D1, R6): every CG inner product / norm is a FIXED-ORDER warp-shuffle
// butterfly reduction (__shfl_down_sync, full 0xffffffff mask, 5 steps 16..1) over
// the warp; inactive lanes (>= n) carry the additive identity 0. Same input + same
// GPU => byte-identical x. Internal arithmetic is fp64 (loads fp32->fp64, stores
// fp64->fp32): fp64 is fully deterministic and keeps the COMPONENTWISE relative
// error under the 1e-6 gate (fp32 accumulation noise on a small solution component
// would otherwise exceed it).
//
// Preconditioners (both D1):
//   Jacobi      -- M^-1 = 1/diag(A). Genuine multi-iteration CG on a coupled block.
//   BlockJacobi -- per-block dense Cholesky (the exact island solve). The
//                  preconditioned residual z = A^-1 r is the exact correction, so
//                  CG converges in <= 1 iteration; remaining fixed iterations are
//                  no-ops guarded against 0/0 NaN by a deterministic residual floor.
//
// NaN guard: with a fixed iteration count, once the residual collapses (esp.
// BlockJacobi after iter 1) the alpha = rz/(p.Ap) and beta = rz_new/rz_old ratios
// would hit 0/0. A deterministic floor (rz_old <= kRzFloor) freezes alpha=beta=0,
// making x/r/p a stable fixed point -- no NaN, still bit-exact, no data-dependent
// control divergence between runs.
// ---------------------------------------------------------------------------

#include "diffsim/sparse_solver_backend.hpp"
#include "phi/device_context.hpp"

#include <string_view>

namespace nuka::diffsim {

class SelfWrittenCgBackend final : public SparseLinearSolver {
public:
    explicit SelfWrittenCgBackend(const phi::DeviceContext& context,
                                  DeterminismLevel determinism = DeterminismLevel::Strong);

    void Solve(const BatchedDenseSpdSystem& system, const float* b, float* x,
               const SolveParams& params) override;

    std::string_view Name() const override { return "cg"; }

private:
    const phi::DeviceContext& context_;
    DeterminismLevel determinism_;
};

}  // namespace nuka::diffsim
