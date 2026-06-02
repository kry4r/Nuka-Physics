#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- self-written deterministic GMRES(m) backend (v0.7 p03-A)
// ---------------------------------------------------------------------------
//
// SelfWrittenGmresBackend: a THIRD backend behind the SAME SparseLinearSolver
// seam as the v0.5/p01 CG (SPD) and v0.7/p02 MINRES (symmetric indefinite)
// backends. GMRES (Generalized Minimum RESidual, Saad & Schultz 1986) is the
// minimum-residual Krylov method for GENERAL NON-SYMMETRIC systems -- the case
// neither CG (needs SPD) nor MINRES (needs A == A^T, it relies on the symmetric
// Lanczos 3-term recurrence) can handle. GMRES instead runs the full Arnoldi
// iteration (a long recurrence: the new basis vector is orthogonalized against
// ALL prior ones), builds an upper-Hessenberg H, and minimizes ||A x - b||_2 over
// the Krylov subspace via a least-squares solve. To bound the O(restart x n) memory
// the iteration is RESTARTED every m steps (GMRES(m)).
//
// LAYOUT (identical batched-dense seam as CG/MINRES -- a drop-in, NOT a new CSR
// path): ONE WARP (32 lanes) per articulation / block, n <= kMaxBlockDim(12). Lane
// L owns matrix row L (streamed once into registers, fp64) and component L of every
// vector. The SpMV  A v  is the SAME fixed-order warp-shuffle gather; every inner
// product / 2-norm is the SAME fixed-order WarpButterflySum butterfly. Unlike the
// CG/MINRES short recurrences, GMRES must RETAIN the whole Krylov basis V (m+1
// vectors) and the Hessenberg/Givens scalars: those live in SHARED memory (the
// basis is warp-distributed -- column k, lane L holds V[k][L] -- and lane 0 runs
// the sequential Givens rotations + least-squares back-substitution on the warp-
// uniform Hessenberg). There is NO cross-warp communication, NO atomics, NO cuBLAS,
// NO order-dependent accumulation -> the solve is D1 byte-exact two-run.
//
// DETERMINISM (D1, Strong): Arnoldi orthogonalization is MODIFIED Gram-Schmidt
// (each projection uses the fixed-order butterfly dot, subtracted in fixed j order
// -- MGS, not CGS, both for numerical stability AND to fix the accumulation order);
// the Givens c/s and the least-squares back-substitution are fp64 scalar ops in a
// fixed loop order on lane 0; the per-lane x += V y update is independent. Same
// input + same GPU => byte-identical x. fp64 internal arithmetic (loads fp32->fp64,
// stores fp64->fp32) matches CG/MINRES so the < 1e-5 agreement gate holds.
//
// PRECONDITIONER (config via SolveParams::preconditioner, reusing the CG enum):
//   Jacobi      -- M^-1 = 1/diag(A). Right/left choice: we apply LEFT Jacobi
//                  (z = M^-1 r), the simplest D1 option, shipped now.
//   BlockJacobi -- reserved hook for ILU(0) (the p02 incomplete-LU factor) and,
//                  later, the AMG preconditioner. The apply seam is a single lambda
//                  `z = M^-1 r`; an ILU0/AMG apply drops in there without touching the
//                  Arnoldi/Givens machinery. See the .cu apply_precond DEFERRED-AMG
//                  SLOT note. Not wired to ILU0/AMG yet (Jacobi is the shipped GMRES
//                  preconditioner this phase). AMG is a v0.7+ suite CAPABILITY
//                  SEQUENCED to its named consumer -- the p09/p10/p11 soft/fluid +
//                  coupling adjoint, the first phase that assembles a large sparse
//                  STIFF operator through this seam (a <= 12-dim dense Delassus block
//                  gets zero benefit and can't exercise AMG's acceptance criterion).
//
// HONEST SCOPE: GMRES is for NON-SYMMETRIC systems. The engine's KKT/Delassus
// systems are symmetric (SPD or indefinite) and STAY on CG/MINRES -- ift_runner's
// auto-router routes ONLY genuinely non-symmetric batches here (symmetric ones are
// byte-identical on their existing path). This backend exists for the general
// non-symmetric linear systems the broader v0.7 solver suite must cover.
// ---------------------------------------------------------------------------

#include "diffsim/sparse_solver_backend.hpp"
#include "phi/device_context.hpp"

#include <cstdint>
#include <string_view>

namespace nuka::diffsim {

class SelfWrittenGmresBackend final : public SparseLinearSolver {
public:
    // restart = m (the Krylov-subspace cap before a restart cycle; Saad recommends
    // 30-50). For the n <= 12 batched-dense blocks the default m=30 > n always
    // converges within one cycle (GMRES is exact in <= n Arnoldi steps in fp64); a
    // SMALLER m (< n) exercises the genuine restart path and is covered by the test.
    explicit SelfWrittenGmresBackend(
        const phi::DeviceContext& context, uint32_t restart = 30u,
        DeterminismLevel determinism = DeterminismLevel::Strong);

    void Solve(const BatchedDenseSpdSystem& system, const float* b, float* x,
               const SolveParams& params) override;

    std::string_view Name() const override { return "gmres"; }

private:
    const phi::DeviceContext& context_;
    uint32_t restart_;
    DeterminismLevel determinism_;
};

// ---------------------------------------------------------------------------
// DetectBatchedNonSymmetric -- v0.7 p03-A auto-routing detector. READ-ONLY over the
// batched-dense `system`; writes ONLY the single uint32 device flag
// `out_nonsymmetric_flag` (the caller pre-zeros it). Sets the flag to 1 iff ANY
// block has an off-diagonal pair that differs by more than a RELATIVE margin:
//
//     |A[i][j] - A[j][i]| > relEps * max|entry|
//
// BYTE-SAFETY (the symmetric-path guarantee, the load-bearing property): the engine's
// Delassus A = J M^-1 J^T is stored BIT-SYMMETRIC (mirror-stored, A[i][j] and A[j][i]
// are the SAME float -- see test_kkt_build's memcmp(&aij,&aji)==0 assertion), so the
// off-diagonal difference is EXACTLY 0.0 and this detector NEVER flags it. Symmetric
// SPD / symmetric indefinite batches therefore stay on the EXISTING CG / MINRES path
// -> byte-identical output. NO FALSE POSITIVES on symmetric matrices. Only a
// genuinely non-symmetric batch flags -> routes to GMRES. The detector touches NO
// solver state. One warp per block (lane 0 does the fixed-order scan), the flag is a
// uint32 atomicOr (uint atomics are D1-safe -- order-independent OR). Deterministic.
//
// THRESHOLD: relEps sits ABOVE the fp32-roundoff floor of a "morally symmetric" but
// fp32-quantized matrix (~1e-7 relative) and BELOW the phase's 1e-5 agreement gate,
// so a near-symmetric fp32 matrix is NOT mis-routed to GMRES while a genuinely
// asymmetric one (O(1) relative difference) always is. See test_gmres_nonsymmetric
// NonSymmetricDetectorBytesafe.
//
// Enqueues on the context stream; the caller synchronizes + reads the flag back.
void DetectBatchedNonSymmetric(const phi::DeviceContext& context,
                               const BatchedDenseSpdSystem& system,
                               uint32_t* out_nonsymmetric_flag);

}  // namespace nuka::diffsim
