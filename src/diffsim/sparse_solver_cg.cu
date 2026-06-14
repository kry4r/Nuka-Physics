// ---------------------------------------------------------------------------
// nuka::diffsim -- self-written deterministic CG backend kernel (v0.5 p03 R1)
// ---------------------------------------------------------------------------
//
// Warp-per-articulation preconditioned CG over batched dense SPD blocks. See
// sparse_solver_cg.hpp for the full design contract (D1, fp64 internal arithmetic,
// fixed-order butterfly reductions, the two preconditioners, the NaN guard).
//
// Perf-conscious choices (D1-preserving throughout):
//   * ONE warp per block, grid = block_count -> independent warps, no global
//     sync, no atomics. Occupancy is bounded only by registers (block row + CG
//     vectors are per-lane fp64 scalars; the Cholesky factor lives in shared mem).
//   * Each lane streams its OWN matrix row (kMaxBlockDim contiguous floats) once
//     into registers; across the warp the 32 row-reads tile the block's contiguous
//     [stride] span -> coalesced, single fetch, never re-read inside the CG loop
//     (the SpMV Ap = A p reads the cached row, not global memory).
//   * `__restrict__` on every pointer; `__device__ __forceinline__` helpers; FMA
//     via plain `a*b+c` on fp64 (fma deterministic, single rounding).
//   * Butterfly reduction is the warp's only "communication" and is BOTH the fast
//     primitive AND bit-exact -- no perf/D1 tension.
// ---------------------------------------------------------------------------

#include "diffsim/solver/cg_diagnostics.cuh"
#include "diffsim/solver/gmres_backend.hpp"
#include "diffsim/solver/minres_backend.hpp"
#include "diffsim/sparse_solver_backend.hpp"
#include "diffsim/sparse_solver_cg.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::diffsim {

namespace {

constexpr uint32_t kWarpSize = 32u;
constexpr uint32_t kBlockStride = kMaxBlockDim * kMaxBlockDim;  // 144 floats/block

// Deterministic residual floor: once rz <= this, alpha/beta are frozen to 0 so a
// collapsed residual (esp. BlockJacobi after iter 1) never produces 0/0 -> NaN.
// In fp64 the squared residual of a well-formed solve floors well below this only
// at genuine convergence. Fixed constant => same input => same break => D1.
__device__ constexpr double kRzFloor = 1.0e-30;
// RELATIVE null-direction threshold for the per-block modified Cholesky
// (BlockJacobi). The Delassus blocks are SPD for full-rank contact configurations,
// but a rank-deficient PSD block (a dead/structurally-zero row OR an OBLIQUE null
// direction where the post-elimination Schur pivot is tiny-but-nonzero) must not be
// divided through: the old absolute floor 1e-300 -> sqrt -> 1e-150 -> divide
// OVERFLOWS to NaN on the oblique case (the nonzero Schur residual blows up). We
// instead test each post-elimination pivot against eps*max_diagonal: pivots at or
// below it are treated as NULL DIRECTIONS (factored diagonal set to 1, the column
// below zeroed, the matching solution component forced to 0 -> least-norm-style),
// keeping the result FINITE and exact on the range space, zero on the null space.
// eps = 1e-12 is well above the ~1e-16 relative pivot a near-singular block yields
// yet far below any full-rank Delassus pivot (O(effective mass)), so full-rank
// blocks NEVER trip the branch -> their factor (and every downstream bit) is
// IDENTICAL to before. The compare is a fixed-order data-deterministic threshold
// test (same input -> same branch) -> D1 byte-exact preserved.
__device__ constexpr double kNullPivotRelEps = 1.0e-12;

// Fixed-order warp-shuffle butterfly sum over all 32 lanes (inactive lanes carry
// 0). Steps 16,8,4,2,1 -> lane 0 holds the total; we broadcast it back to every
// lane with a final shfl so all lanes share the identical reduced value (bit-exact
// across lanes AND across runs). fp64, full mask -> no partial-participation UB.
__device__ __forceinline__ double WarpButterflySum(double v) {
    constexpr unsigned kFull = 0xffffffffu;
#pragma unroll
    for (uint32_t offset = kWarpSize / 2u; offset > 0u; offset >>= 1u) {
        v += __shfl_down_sync(kFull, v, offset);
    }
    // Lane 0 now has the sum; broadcast to all lanes (fixed source lane 0).
    return __shfl_sync(kFull, v, 0);
}

// One warp solves block `blk`. Lane `lane` (0..31) owns matrix row `lane` and the
// `lane`-th component of every vector. Only lanes [0, n) are "active"; lanes >= n
// hold zeros and contribute the additive identity to every reduction.
__device__ __forceinline__ void SolveOneBlock(
    const float* __restrict__ values, uint32_t n, const float* __restrict__ b_blk,
    float* __restrict__ x_blk, Preconditioner precond, uint32_t max_iter,
    double tol, bool run_to_fixed_iters, double* __restrict__ chol_shared,
    bool* __restrict__ null_shared, uint32_t lane, const CgConvergenceReport& diag,
    uint32_t blk) {
    const bool active = (lane < n);

    // Opt-in diagnostics: enabled iff ANY report channel is non-null. When off,
    // every diag branch below is not-taken on EVERY lane -> the math + stores are
    // exactly the v0.5 path (byte-identical x; the D1 / no-IFT-regression gate).
    const bool diag_on = (diag.iters_run != nullptr ||
                          diag.final_rel_resid != nullptr ||
                          diag.status != nullptr ||
                          diag.residual_history != nullptr);
    // Per-block history row (lane-0 local; null if not requested / off).
    float* __restrict__ hist =
        (diag_on && diag.residual_history != nullptr)
            ? diag.residual_history + static_cast<size_t>(blk) * diag.history_cap
            : nullptr;
    CgDiagState diag_state;  // lane-0 trajectory accumulator (unused unless diag_on)

    // --- Load this lane's matrix row into registers (fp64). Contiguous per-lane
    //     row read; across the warp the rows tile the block's [stride] span. ----
    double row[kMaxBlockDim];
#pragma unroll
    for (uint32_t c = 0u; c < kMaxBlockDim; ++c) {
        // Inactive lanes / out-of-range columns load 0 so SpMV stays a clean
        // n-truncated dot without per-element branching in the hot loop.
        double aij = 0.0;
        if (active && c < n) {
            aij = static_cast<double>(values[lane * kMaxBlockDim + c]);
        }
        row[c] = aij;
    }

    // diag(A) for the Jacobi preconditioner (this lane's diagonal entry).
    const double a_diag = active ? row[lane] : 0.0;

    // RHS component for this lane.
    const double b_i = active ? static_cast<double>(b_blk[lane]) : 0.0;

    // --- Optional per-block dense Cholesky (BlockJacobi). Factor A = L L^T in
    //     shared memory, computed by lane 0 (n <= 12 -> trivial), then every
    //     preconditioner apply is two triangular solves. Lane 0 writes L row-major
    //     into chol_shared[kMaxBlockDim^2]; we keep the lower triangle. ----------
    if (precond == Preconditioner::BlockJacobi) {
        if (lane == 0u) {
            // Block scale for the relative null-direction threshold: the largest
            // ORIGINAL diagonal (fixed order). For a PSD Delassus this bounds the
            // spectrum, so eps*max_diag is a scale-correct null cutoff. (max_diag is
            // > 0 for any non-empty block since diag(A) >= 0 and a structurally dead
            // row's diagonal is 0 while a live row's is > 0; if EVERY diagonal were 0
            // the whole block is null and every component is forced to 0 below.)
            double max_diag = 0.0;
            for (uint32_t d = 0u; d < n; ++d) {
                const double add = static_cast<double>(values[d * kMaxBlockDim + d]);
                if (add > max_diag) max_diag = add;
            }
            const double null_thresh = kNullPivotRelEps * max_diag;

            // Modified (pseudo) column Cholesky for PSD, fixed loop order (D1).
            // chol[i*md+j] = L_ij. A post-elimination pivot djj <= null_thresh marks
            // a NULL DIRECTION: we set the factored diagonal to 1 (a safe nonzero so
            // the triangular solves never divide by ~0), ZERO the column below it
            // (so the oblique Schur residual never propagates -> the NaN source), and
            // record null[j]; the solve forces z_j = 0 (least-norm on the range
            // space). Full-rank blocks never hit this branch, so for them djj, ljj =
            // sqrt(djj) and every s/ljj are the IDENTICAL ops on the IDENTICAL data
            // as before -> byte-identical factor.
            for (uint32_t j = 0u; j < n; ++j) {
                double djj = static_cast<double>(values[j * kMaxBlockDim + j]);
                for (uint32_t k = 0u; k < j; ++k) {
                    const double ljk = chol_shared[j * kMaxBlockDim + k];
                    djj -= ljk * ljk;
                }
                if (djj <= null_thresh) {
                    // Null direction: do NOT sqrt/divide the tiny (or oblique-residual)
                    // pivot. Safe unit diagonal + zeroed column => finite L; z_j = 0.
                    null_shared[j] = true;
                    chol_shared[j * kMaxBlockDim + j] = 1.0;
                    for (uint32_t i = j + 1u; i < n; ++i) {
                        chol_shared[i * kMaxBlockDim + j] = 0.0;
                    }
                    continue;
                }
                null_shared[j] = false;
                const double ljj = sqrt(djj);
                chol_shared[j * kMaxBlockDim + j] = ljj;
                for (uint32_t i = j + 1u; i < n; ++i) {
                    double s = static_cast<double>(values[i * kMaxBlockDim + j]);
                    for (uint32_t k = 0u; k < j; ++k) {
                        s -= chol_shared[i * kMaxBlockDim + k] *
                             chol_shared[j * kMaxBlockDim + k];
                    }
                    chol_shared[i * kMaxBlockDim + j] = s / ljj;
                }
            }
        }
        __syncwarp(0xffffffffu);  // publish L + null mask to the whole warp
    }

    // Applies the preconditioner: z = M^-1 r. For Jacobi, z_i = r_i / a_ii. For
    // BlockJacobi, z = A^-1 r via forward/back substitution on L (done by lane 0;
    // the result is scattered back through shared). Returns this lane's z_i.
    auto apply_precond = [&](double r_i) -> double {
        if (precond == Preconditioner::Jacobi) {
            if (!active) return 0.0;
            const double d = (a_diag > kRzFloor) ? a_diag : 1.0;
            return r_i / d;
        }
        // BlockJacobi: z = A^-1 r via two triangular solves on the cached factor
        // L. Each lane stages its r_i into the dedicated shared rhs row (the
        // [kBlockStride, kBlockStride+kMaxBlockDim) slot the kernel allocates past
        // L's [0,kBlockStride) region -- disjoint, so the in-place solve never
        // touches L). Lane 0 does the forward+back substitution in fixed order
        // (D1), then every lane reads back its component.
        double* rhs = chol_shared + kBlockStride;  // scratch [kMaxBlockDim]
        if (active) rhs[lane] = r_i;
        __syncwarp(0xffffffffu);
        if (lane == 0u) {
            // Forward solve L y = r  (overwrite rhs with y). A null row contributes
            // y_i = 0 (its column was zeroed in the factor and its diagonal is 1, so
            // the division is finite; forcing 0 is the least-norm choice). Full-rank
            // rows are never null -> rhs[i] = s / chol[i][i] EXACTLY as before.
            for (uint32_t i = 0u; i < n; ++i) {
                double s = rhs[i];
                for (uint32_t k = 0u; k < i; ++k) {
                    s -= chol_shared[i * kMaxBlockDim + k] * rhs[k];
                }
                rhs[i] = null_shared[i] ? 0.0 : (s / chol_shared[i * kMaxBlockDim + i]);
            }
            // Back solve L^T z = y (overwrite rhs with z). z_i = 0 on null rows; the
            // zeroed factor column means a null row also contributes nothing to the
            // range-space rows' back-substitution -> finite, range-space-exact.
            for (uint32_t ii = 0u; ii < n; ++ii) {
                const uint32_t i = n - 1u - ii;
                double s = rhs[i];
                for (uint32_t k = i + 1u; k < n; ++k) {
                    s -= chol_shared[k * kMaxBlockDim + i] * rhs[k];
                }
                rhs[i] = null_shared[i] ? 0.0 : (s / chol_shared[i * kMaxBlockDim + i]);
            }
        }
        __syncwarp(0xffffffffu);
        const double z_i = active ? rhs[lane] : 0.0;
        __syncwarp(0xffffffffu);
        return z_i;
    };

    // SpMV: returns (A p)_i for this lane = dot(row_i, p) over the warp. p is held
    // distributed (lane k has p_k); we gather it via shuffle into a fixed order.
    auto spmv = [&](const double p_self) -> double {
        // Pull every p_k (k=0..n-1) and accumulate row_i[k]*p_k in fixed k order.
        double acc = 0.0;
#pragma unroll
        for (uint32_t k = 0u; k < kMaxBlockDim; ++k) {
            const double p_k = __shfl_sync(0xffffffffu, p_self, k);
            acc += row[k] * p_k;  // row[k]==0 for k>=n or inactive lane
        }
        return active ? acc : 0.0;
    };

    // --- CG state (per-lane scalars). x0 = 0 => r0 = b. -----------------------
    double x_i = 0.0;
    double r_i = b_i;
    double z_i = apply_precond(r_i);
    double p_i = z_i;
    double rz = WarpButterflySum(r_i * z_i);  // r.z (preconditioned)

    // Relative-residual early-exit reference: ||b||^2 (fixed-order). Exit when
    // ||r||^2 <= tol^2 ||b||^2. Deterministic compare => D1.
    const double bb = WarpButterflySum(b_i * b_i);
    const double stop = tol * tol * bb;

    // Diagnostics trajectory (lane 0 only; the loop's MATH is identical to v0.5 --
    // every diag-only computation is gated on diag_on so the off path is byte-exact).
    // r0_sq == ||b||^2 (x0 = 0 => r0 = b); rz0 is the initial preconditioned residual.
    if (diag_on && lane == 0u) CgDiagInit(diag_state, bb, rz);
    uint32_t iters_run = 0u;
    bool converged = false;
    double rr_final = bb;  // last observed ||r||^2 (init: pre-loop residual)

    for (uint32_t iter = 0u; iter < max_iter; ++iter) {
        const double ap_i = spmv(p_i);
        const double pAp = WarpButterflySum(p_i * ap_i);

        // SPD guard: frozen ratios when the residual collapsed (rz->0) or the
        // direction degenerates (pAp->0). alpha=0 => x/r unchanged (fixed point).
        // The guard tripping pre-convergence is the runtime SPD-failure signal.
        const bool broke_down = !(rz > kRzFloor && pAp > kRzFloor);
        const double alpha = broke_down ? 0.0 : (rz / pAp);

        x_i += alpha * p_i;
        r_i -= alpha * ap_i;

        const double rr = WarpButterflySum(r_i * r_i);
        rr_final = rr;
        iters_run = iter + 1u;
        const bool now_converged = (!run_to_fixed_iters && rr <= stop);

        // Diagnostics: the preconditioned residual rz_obs (the quantity CG drives to
        // zero -- the stall test's input) is computed here ONLY when diag_on. ALL
        // lanes execute apply_precond + the butterfly UNIFORMLY (apply_precond's
        // BlockJacobi path uses __syncwarp/__shfl, so every lane must reach them in
        // the same order -- a lane-split would deadlock/UB). Only lane 0 records.
        // This is a diag-only cost; with diag off the loop math is byte-identical
        // to v0.5 (this whole block is skipped).
        if (diag_on) {
            const double z_obs = apply_precond(r_i);
            const double rz_obs = WarpButterflySum(r_i * z_obs);
            if (lane == 0u)
                CgDiagOnIter(diag_state, rr, rz_obs, broke_down, now_converged, iter,
                             hist, diag.history_cap);
        }

        if (now_converged) {  // deterministic early exit
            converged = true;
            break;
        }

        z_i = apply_precond(r_i);
        const double rz_new = WarpButterflySum(r_i * z_i);
        const double beta = (rz > kRzFloor) ? (rz_new / rz) : 0.0;
        p_i = z_i + beta * p_i;
        rz = rz_new;
    }

    // Write the opt-in per-block report (lane 0, fixed block slot). No-op channels
    // (null pointers) are skipped; an all-null report leaves x byte-identical.
    if (diag_on && lane == 0u)
        CgDiagFinalize(diag_state, diag, blk, rr_final, iters_run, converged);

    // --- Write back: x_blk has exactly kMaxBlockDim slots per block, so the store
    //     MUST be capped at kMaxBlockDim -- lanes >= kMaxBlockDim would clobber the
    //     next block's slots (cross-block race). Lanes [0,n) write x_i; padding
    //     lanes [n,kMaxBlockDim) write 0.0f (the deterministic padding the memcmp
    //     D1 gate compares). Lanes [kMaxBlockDim,32) write NOTHING but still took
    //     part in every butterfly reduction above (full-mask participation). ------
    if (lane < kMaxBlockDim) {
        x_blk[lane] = active ? static_cast<float>(x_i) : 0.0f;
    }
}

__global__ void SolveCgKernel(const float* __restrict__ values,
                              const uint32_t* __restrict__ block_dim,
                              const float* __restrict__ b, float* __restrict__ x,
                              uint32_t block_count, Preconditioner precond,
                              uint32_t max_iter, double tol,
                              bool run_to_fixed_iters, CgConvergenceReport diag) {
    const uint32_t blk = blockIdx.x;  // one warp/block per articulation
    if (blk >= block_count) return;
    const uint32_t lane = threadIdx.x;  // 0..31

    // Shared scratch: [0, kBlockStride) holds the Cholesky factor L (row-major
    // lower triangle); [kBlockStride, kBlockStride+kMaxBlockDim) is the triangular-
    // solve rhs/z staging. Only BlockJacobi touches it, but it is always allocated
    // (its presence does not affect Jacobi's bit pattern).
    __shared__ double chol_shared[kBlockStride + kMaxBlockDim];
    // Null-direction mask (BlockJacobi modified Cholesky): null_shared[j] => row j
    // is a PSD null direction (zeroed column + forced-zero solution component). Only
    // BlockJacobi writes/reads it; its presence never touches the Jacobi bit pattern.
    __shared__ bool null_shared[kMaxBlockDim];

    const uint32_t n = block_dim[blk];
    const float* __restrict__ values_blk = values + static_cast<size_t>(blk) * kBlockStride;
    const float* __restrict__ b_blk = b + static_cast<size_t>(blk) * kMaxBlockDim;
    float* __restrict__ x_blk = x + static_cast<size_t>(blk) * kMaxBlockDim;

    if (n == 0u) {
        // Empty articulation: deterministic zero padding, no solve. Capped at
        // kMaxBlockDim (this block owns exactly that many x slots).
        if (lane < kMaxBlockDim) x_blk[lane] = 0.0f;
        // Deterministic empty-block diagnostics: 0 iters, 0 residual, converged.
        if (lane == 0u) {
            if (diag.iters_run != nullptr) diag.iters_run[blk] = 0u;
            if (diag.final_rel_resid != nullptr) diag.final_rel_resid[blk] = 0.0f;
            if (diag.status != nullptr) diag.status[blk] = kCgStatusConverged;
        }
        return;
    }

    SolveOneBlock(values_blk, n, b_blk, x_blk, precond, max_iter, tol,
                  run_to_fixed_iters, chol_shared, null_shared, lane, diag, blk);
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("nuka::diffsim::SelfWrittenCgBackend ") +
                                 operation + ": " + cudaGetErrorString(result));
    }
}

// v0.7 p02 indefinite detector (read-only): one warp per block, lane 0 runs a
// fixed-order symmetric LDLT elimination computing the POST-ELIMINATION PIVOTS
// (Sylvester's law of inertia: their signs = the inertia of A). If any pivot is
// strictly negative beyond -kEps*max|diag| the block is INDEFINITE -> OR the uint
// flag (uint atomicOr is order-independent => D1; NOT a float atomic). No solver
// state is touched; A is read-only. SPD/PSD => all pivots >= 0 => flag never set.
// INDEFINITE-FLAG threshold: a pivot is "genuinely negative" only if it falls below
// -kIndefRelEps * max|diag|. This must sit ABOVE the fp32-quantization/roundoff
// floor of a true-ZERO eigenvalue (a PSD null direction's no-pivoting pivot is a
// tiny SIGNED roundoff, measured down to ~-1.6e-6 * max|diag| on fp32-quantized PSD
// blocks) and BELOW a genuinely indefinite pivot (an eigenvalue ~ -O(1)*scale gives
// a pivot whose LEAST-negative observed value over the indefinite test battery is
// ~-9.8e-2 * max|diag|). kIndefRelEps = 1e-3 is the clean separation: PSD/PSD-null
// NEVER trips (no SPD-path false positive -> byte-identical guarantee), every tested
// indefinite block DOES. See test_minres_vs_dense_indefinite IndefiniteDetectorBytesafe.
__device__ constexpr double kIndefRelEps = 1.0e-3;
// NULL-SKIP band: |pivot| <= kNullRelEps * max|diag| is treated as a (PSD) null
// direction -- skip its column so the elimination does not divide by ~0 / amplify
// roundoff. Far below kIndefRelEps so it never masks a genuine negative pivot.
__device__ constexpr double kNullRelEps = 1.0e-7;

__global__ void DetectIndefiniteKernel(const float* __restrict__ values,
                                       const uint32_t* __restrict__ block_dim,
                                       uint32_t block_count,
                                       uint32_t* __restrict__ flag) {
    const uint32_t blk = blockIdx.x;
    if (blk >= block_count) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t n = block_dim[blk];
    if (n == 0u) return;
    if (lane != 0u) return;  // lane 0 does the sequential elimination

    __shared__ double a_sh[kMaxBlockDim * kMaxBlockDim];
    const float* __restrict__ vb = values + static_cast<size_t>(blk) * kBlockStride;
    double max_diag = 0.0;
    for (uint32_t i = 0u; i < n; ++i) {
        for (uint32_t j = 0u; j < n; ++j)
            a_sh[i * kMaxBlockDim + j] =
                static_cast<double>(vb[i * kMaxBlockDim + j]);
        const double d = a_sh[i * kMaxBlockDim + i];
        const double ad = (d < 0.0) ? -d : d;
        if (ad > max_diag) max_diag = ad;
    }
    const double scale = (max_diag > 0.0 ? max_diag : 1.0);
    const double neg_thresh = -kIndefRelEps * scale;  // genuine-indefinite cutoff

    // Symmetric LDLT (no pivoting): pivot d_j = a_jj - sum_{k<j} L_jk^2 d_k, with
    // L_ij = (a_ij - sum_{k<j} L_ik L_jk d_k) / d_j. The strict-lower of a_sh holds
    // L_ij after column j is processed. A genuinely NEGATIVE pivot (dj < neg_thresh)
    // => the block has a negative inertia => INDEFINITE. A near-ZERO pivot
    // (|dj| <= kNullRelEps*max|diag|, a PSD null direction -- a true-zero eigenvalue
    // whose fp32-quantized no-pivoting pivot is a tiny SIGNED roundoff) is NOT
    // indefinite: it sits ABOVE neg_thresh so it never flags, and we skip its column
    // (treat L_ij in it as 0) so the elimination stays finite. (Detection only; A is
    // never modified.)
    const double pos_floor = kNullRelEps * scale;
    double piv[kMaxBlockDim];
    bool indefinite = false;
    for (uint32_t j = 0u; j < n; ++j) {
        double dj = a_sh[j * kMaxBlockDim + j];
        for (uint32_t k = 0u; k < j; ++k) {
            const double ljk = a_sh[j * kMaxBlockDim + k];
            dj -= ljk * ljk * piv[k];
        }
        piv[j] = dj;
        if (dj < neg_thresh) { indefinite = true; break; }
        // Near-zero pivot (PSD null direction): leave column's L_ij at 0 (skip).
        if (dj <= pos_floor) {
            for (uint32_t i = j + 1u; i < n; ++i)
                a_sh[i * kMaxBlockDim + j] = 0.0;
            continue;
        }
        for (uint32_t i = j + 1u; i < n; ++i) {
            double s = a_sh[i * kMaxBlockDim + j];
            for (uint32_t k = 0u; k < j; ++k)
                s -= a_sh[i * kMaxBlockDim + k] * a_sh[j * kMaxBlockDim + k] * piv[k];
            a_sh[i * kMaxBlockDim + j] = s / dj;
        }
    }
    if (indefinite) atomicOr(flag, 1u);
}

}  // namespace

SelfWrittenCgBackend::SelfWrittenCgBackend(cudaStream_t stream, int device_id,
                                           DeterminismLevel determinism)
    : stream_(stream), device_id_(device_id), determinism_(determinism) {}

void SelfWrittenCgBackend::Solve(const BatchedDenseSpdSystem& system,
                                 const float* b, float* x,
                                 const SolveParams& params) {
    if (system.block_count == 0u) return;
    if (system.values == nullptr || system.block_dim == nullptr || b == nullptr ||
        x == nullptr) {
        throw std::invalid_argument(
            "nuka::diffsim::SelfWrittenCgBackend::Solve: null system/vector pointer");
    }
    phi::ScopedDeviceGuard guard(device_id_);
    const cudaStream_t stream = stream_;

    // ONE warp per block; grid over blocks. Mirrors the engine's canonical
    // <<<count, 32>>> warp-per-articulation idiom. params.diagnostics is passed by
    // value; an all-null report (the default) makes the kernel take the exact v0.5
    // path -> byte-identical x (the D1 + no-IFT-regression guarantee).
    SolveCgKernel<<<system.block_count, kWarpSize, 0u, stream>>>(
        system.values, system.block_dim, b, x, system.block_count,
        params.preconditioner, params.max_iter, static_cast<double>(params.tol),
        params.run_to_fixed_iters, params.diagnostics);
    CheckCuda(cudaGetLastError(), "SolveCgKernel launch");
}

std::unique_ptr<SparseLinearSolver> MakeSparseSolverBackend(
    std::string_view name, cudaStream_t stream, int device_id,
    DeterminismLevel determinism) {
    // v0.7 p01: the self-written CG is the only SHIPPING backend. Accept its v0.5
    // aliases ("cg"/""/"default") AND the v0.7 canonical key "self_cg" (the C ABI
    // NUKA_SOLVER_BACKEND_SELF_CG round-trips through this name).
    if (name.empty() || name == "cg" || name == "default" || name == "self_cg") {
        return std::make_unique<SelfWrittenCgBackend>(stream, device_id, determinism);
    }
    // v0.7 p02: the self-written MINRES backend (symmetric INDEFINITE Krylov +
    // ILU(0) preconditioner) shares this same batched-dense seam. The C ABI
    // NUKA_SOLVER_BACKEND_SELF_MINRES round-trips through this "self_minres" name.
    if (name == "self_minres" || name == "minres") {
        return std::make_unique<SelfWrittenMinresBackend>(stream, device_id, determinism);
    }
    // v0.7 p03-A: the self-written GMRES backend (GENERAL non-symmetric Krylov,
    // restarted GMRES(m) + Jacobi preconditioner; ILU0/AMG hook reserved) shares the
    // same batched-dense seam. The C ABI NUKA_SOLVER_BACKEND_SELF_GMRES round-trips
    // through this "self_gmres" name. Default restart m=30 (Saad's 30-50 band).
    if (name == "self_gmres" || name == "gmres") {
        return std::make_unique<SelfWrittenGmresBackend>(stream, device_id, 30u, determinism);
    }
    throw std::invalid_argument(
        "nuka::diffsim::MakeSparseSolverBackend: unknown backend '" +
        std::string(name) +
        "' (v0.7 p03 ships 'self_cg'/'cg', 'self_minres'/'minres', "
        "'self_gmres'/'gmres')");
}

void DetectBatchedIndefinite(cudaStream_t stream, int device_id,
                             const BatchedDenseSpdSystem& system,
                             uint32_t* out_indefinite_flag) {
    if (system.block_count == 0u || out_indefinite_flag == nullptr) return;
    if (system.values == nullptr || system.block_dim == nullptr) {
        throw std::invalid_argument(
            "nuka::diffsim::DetectBatchedIndefinite: null system pointer");
    }
    phi::ScopedDeviceGuard guard(device_id);
    // One warp per block (lane 0 does the elimination). READ-ONLY over `system`;
    // writes only the caller-pre-zeroed uint flag. The caller synchronizes + reads.
    DetectIndefiniteKernel<<<system.block_count, kWarpSize, 0u, stream>>>(
        system.values, system.block_dim, system.block_count, out_indefinite_flag);
    CheckCuda(cudaGetLastError(), "DetectIndefiniteKernel launch");
}

}  // namespace nuka::diffsim
