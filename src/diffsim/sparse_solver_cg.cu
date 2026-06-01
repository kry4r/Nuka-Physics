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
    bool* __restrict__ null_shared, uint32_t lane) {
    const bool active = (lane < n);

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

    for (uint32_t iter = 0u; iter < max_iter; ++iter) {
        const double ap_i = spmv(p_i);
        const double pAp = WarpButterflySum(p_i * ap_i);

        // NaN guard: frozen ratios when the residual collapsed (rz->0) or the
        // direction degenerates (pAp->0). alpha=0 => x/r unchanged (fixed point).
        const double alpha = (rz > kRzFloor && pAp > kRzFloor) ? (rz / pAp) : 0.0;

        x_i += alpha * p_i;
        r_i -= alpha * ap_i;

        const double rr = WarpButterflySum(r_i * r_i);
        if (!run_to_fixed_iters && rr <= stop) break;  // deterministic early exit

        z_i = apply_precond(r_i);
        const double rz_new = WarpButterflySum(r_i * z_i);
        const double beta = (rz > kRzFloor) ? (rz_new / rz) : 0.0;
        p_i = z_i + beta * p_i;
        rz = rz_new;
    }

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
                              bool run_to_fixed_iters) {
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
        return;
    }

    SolveOneBlock(values_blk, n, b_blk, x_blk, precond, max_iter, tol,
                  run_to_fixed_iters, chol_shared, null_shared, lane);
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("nuka::diffsim::SelfWrittenCgBackend ") +
                                 operation + ": " + cudaGetErrorString(result));
    }
}

}  // namespace

SelfWrittenCgBackend::SelfWrittenCgBackend(const phi::DeviceContext& context,
                                           DeterminismLevel determinism)
    : context_(context), determinism_(determinism) {}

void SelfWrittenCgBackend::Solve(const BatchedDenseSpdSystem& system,
                                 const float* b, float* x,
                                 const SolveParams& params) {
    if (system.block_count == 0u) return;
    if (system.values == nullptr || system.block_dim == nullptr || b == nullptr ||
        x == nullptr) {
        throw std::invalid_argument(
            "nuka::diffsim::SelfWrittenCgBackend::Solve: null system/vector pointer");
    }
    phi::ScopedDeviceGuard guard(context_.device_id);
    const cudaStream_t stream = context_.stream.Native();

    // ONE warp per block; grid over blocks. Mirrors the engine's canonical
    // <<<count, 32>>> warp-per-articulation idiom.
    SolveCgKernel<<<system.block_count, kWarpSize, 0u, stream>>>(
        system.values, system.block_dim, b, x, system.block_count,
        params.preconditioner, params.max_iter, static_cast<double>(params.tol),
        params.run_to_fixed_iters);
    CheckCuda(cudaGetLastError(), "SolveCgKernel launch");
}

std::unique_ptr<SparseLinearSolver> MakeSparseSolverBackend(
    std::string_view name, const phi::DeviceContext& context,
    DeterminismLevel determinism) {
    if (name.empty() || name == "cg" || name == "default") {
        return std::make_unique<SelfWrittenCgBackend>(context, determinism);
    }
    throw std::invalid_argument(
        "nuka::diffsim::MakeSparseSolverBackend: unknown backend '" +
        std::string(name) + "' (v0.5 only ships 'cg')");
}

}  // namespace nuka::diffsim
