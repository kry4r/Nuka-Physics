// ---------------------------------------------------------------------------
// nuka::diffsim -- sparse triangular solve (host references + device test kernel)
//                  v0.7 p02
// ---------------------------------------------------------------------------
//
// The PRODUCTION triangular solve lives inline in triangular_solve.cuh (called by
// lane 0 of the MINRES warp kernel during the ILU(0) preconditioner apply). This
// translation unit provides:
//   (a) HOST reference forward/backward substitution + ILU(0) apply (the known-good
//       fp64 oracle the unit tests compare against), and
//   (b) a small batched DEVICE test kernel (one warp per block, lane 0 does the
//       substitution) so the triangular solve can be exercised + two-run D1 byte-
//       exact checked in isolation, without routing through the full MINRES backend.
//
// D1: a single fixed-order sequential pass (host) / lane-0 sequential pass on the
// device. No atomics, no parallel reduction, fp64 accumulation. Byte-exact two-run.
// ---------------------------------------------------------------------------

#include "diffsim/solver/ilu0_preconditioner.hpp"
#include "diffsim/solver/triangular_solve.cuh"
#include "diffsim/sparse_solver_backend.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace nuka::diffsim {

// --- HOST references (TEST-ONLY oracles) -----------------------------------

void TriSolveLowerUnitHost(const double* lu, uint32_t n, uint32_t stride,
                           const double* rhs, double* y_out) {
    for (uint32_t i = 0u; i < n; ++i) {
        double s = rhs[i];
        for (uint32_t k = 0u; k < i; ++k) {
            const double lik = lu[i * stride + k];
            if (lik == 0.0) continue;
            s -= lik * y_out[k];
        }
        y_out[i] = s;  // unit diagonal
    }
}

void TriSolveUpperHost(const double* lu, uint32_t n, uint32_t stride,
                       const double* rhs, double* z_out) {
    for (uint32_t ii = 0u; ii < n; ++ii) {
        const uint32_t i = n - 1u - ii;
        double s = rhs[i];
        for (uint32_t k = i + 1u; k < n; ++k) {
            const double uik = lu[i * stride + k];
            if (uik == 0.0) continue;
            s -= uik * z_out[k];
        }
        z_out[i] = s / lu[i * stride + i];
    }
}

// --- DEVICE batched triangular-solve test kernel ---------------------------
// One warp per block; lane 0 runs forward L y = r then backward U z = y in place on
// a shared scratch row, then every lane reads back its component. Mirrors the
// canonical <<<count, 32>>> warp-per-block idiom (so its D1 contract matches the
// MINRES/CG backends). `lu` is block-major (stride kMaxBlockDim^2, row-major n x n),
// `r`/`z` are block-major (stride kMaxBlockDim).
namespace {
constexpr uint32_t kWarpSize = 32u;
constexpr uint32_t kStride = kMaxBlockDim * kMaxBlockDim;

__global__ void Ilu0ApplyTestKernel(const float* __restrict__ lu,
                                     const uint32_t* __restrict__ block_dim,
                                     const float* __restrict__ r,
                                     float* __restrict__ z, uint32_t block_count) {
    const uint32_t blk = blockIdx.x;
    if (blk >= block_count) return;
    const uint32_t lane = threadIdx.x;

    __shared__ double lu_sh[kStride];
    __shared__ double scratch[kMaxBlockDim];

    const uint32_t n = block_dim[blk];
    const float* __restrict__ lu_blk = lu + static_cast<size_t>(blk) * kStride;
    const float* __restrict__ r_blk = r + static_cast<size_t>(blk) * kMaxBlockDim;
    float* __restrict__ z_blk = z + static_cast<size_t>(blk) * kMaxBlockDim;

    // Zero-pad output deterministically; empty block exits.
    if (lane < kMaxBlockDim) z_blk[lane] = 0.0f;
    if (n == 0u) return;

    // Stream LU tile + rhs into shared (coalesced).
    for (uint32_t idx = lane; idx < kStride; idx += kWarpSize)
        lu_sh[idx] = static_cast<double>(lu_blk[idx]);
    if (lane < n) scratch[lane] = static_cast<double>(r_blk[lane]);
    __syncwarp(0xffffffffu);

    if (lane == 0u) Ilu0ApplyDevice(lu_sh, n, kMaxBlockDim, scratch);
    __syncwarp(0xffffffffu);

    if (lane < n) z_blk[lane] = static_cast<float>(scratch[lane]);
}
}  // namespace

// Host launcher for the device triangular-solve test kernel. Device pointers; the
// caller owns + synchronizes (mirrors the rest of diffsim).
void LaunchIlu0ApplyTest(cudaStream_t stream, int device_id, const float* lu,
                         const uint32_t* block_dim, const float* r, float* z,
                         uint32_t block_count) {
    if (block_count == 0u) return;
    phi::ScopedDeviceGuard guard(device_id);
    Ilu0ApplyTestKernel<<<block_count, kWarpSize, 0u, stream>>>(
        lu, block_dim, r, z, block_count);
}

}  // namespace nuka::diffsim
