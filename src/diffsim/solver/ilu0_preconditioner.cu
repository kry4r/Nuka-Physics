// ---------------------------------------------------------------------------
// nuka::diffsim -- ILU(0) preconditioner (host reference + device test launcher)
//                  v0.7 p02
// ---------------------------------------------------------------------------
//
// The PRODUCTION ILU(0) factorization lives inline in ilu0_preconditioner.hpp
// (Ilu0FactorizeDevice, called by lane 0 of the MINRES warp kernel) so the MINRES
// backend can factor a block in shared memory without a global round-trip. This
// translation unit provides:
//   (a) the HOST reference factorization + apply (Ilu0FactorizeHost / Ilu0ApplyHost,
//       declared in the header) -- the known-good fp64 oracle the unit tests use,
//       computed by the SAME fixed-order IKJ recurrence as the device path, and
//   (b) a batched DEVICE factorization test kernel + launcher so the device factor
//       can be exercised + two-run D1 byte-exact checked in isolation.
//
// D1: a single fixed-order sequential IKJ pass (host) / lane-0 sequential pass on
// the device. No atomics, no parallel reduction, fp64. Byte-exact two-run.
// ---------------------------------------------------------------------------

#include "diffsim/solver/ilu0_preconditioner.hpp"
#include "diffsim/solver/triangular_solve.cuh"
#include "diffsim/sparse_solver_backend.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace nuka::diffsim {

// --- HOST reference ILU(0) factorization (TEST-ONLY oracle) ----------------
// Identical fixed-order IKJ recurrence to Ilu0FactorizeDevice; the device factor
// matches this byte-for-byte (both fp64). Structural zeros of `a` are never filled.
void Ilu0FactorizeHost(const double* a, uint32_t n, uint32_t stride,
                       double* lu_out) {
    double max_diag = 0.0;
    for (uint32_t d = 0u; d < n; ++d) {
        const double ad = a[d * stride + d];
        const double abs_ad = (ad < 0.0) ? -ad : ad;
        if (abs_ad > max_diag) max_diag = abs_ad;
    }
    const double pivot_floor = kIluPivotRelEps * (max_diag > 0.0 ? max_diag : 1.0);

    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j) lu_out[i * stride + j] = a[i * stride + j];

    for (uint32_t i = 1u; i < n; ++i) {
        for (uint32_t k = 0u; k < i; ++k) {
            if (a[i * stride + k] == 0.0) continue;
            double ukk = lu_out[k * stride + k];
            if (ukk <= pivot_floor && ukk >= -pivot_floor) {
                ukk = (ukk < 0.0) ? -1.0 : 1.0;
                lu_out[k * stride + k] = ukk;
            }
            const double lik = lu_out[i * stride + k] / ukk;
            lu_out[i * stride + k] = lik;
            for (uint32_t j = k + 1u; j < n; ++j) {
                if (a[i * stride + j] == 0.0) continue;
                lu_out[i * stride + j] -= lik * lu_out[k * stride + j];
            }
        }
        double uii = lu_out[i * stride + i];
        if (uii <= pivot_floor && uii >= -pivot_floor) {
            lu_out[i * stride + i] = (uii < 0.0) ? -1.0 : 1.0;
        }
    }
    {
        double u00 = lu_out[0];
        if (u00 <= pivot_floor && u00 >= -pivot_floor) {
            lu_out[0] = (u00 < 0.0) ? -1.0 : 1.0;
        }
    }
}

// --- HOST reference ILU(0) apply: z = (L U)^-1 r (forward then backward) -----
void Ilu0ApplyHost(const double* lu, uint32_t n, uint32_t stride, const double* r,
                   double* z_out) {
    // Forward L y = r (unit lower diagonal); write y into z_out in place.
    for (uint32_t i = 0u; i < n; ++i) {
        double s = r[i];
        for (uint32_t k = 0u; k < i; ++k) {
            const double lik = lu[i * stride + k];
            if (lik == 0.0) continue;
            s -= lik * z_out[k];
        }
        z_out[i] = s;
    }
    // Backward U z = y; overwrite z_out with z.
    for (uint32_t ii = 0u; ii < n; ++ii) {
        const uint32_t i = n - 1u - ii;
        double s = z_out[i];
        for (uint32_t k = i + 1u; k < n; ++k) {
            const double uik = lu[i * stride + k];
            if (uik == 0.0) continue;
            s -= uik * z_out[k];
        }
        z_out[i] = s / lu[i * stride + i];
    }
}

// --- DEVICE batched ILU(0) factorization test kernel -----------------------
namespace {
constexpr uint32_t kWarpSize = 32u;
constexpr uint32_t kStride = kMaxBlockDim * kMaxBlockDim;

__global__ void Ilu0FactorizeTestKernel(const float* __restrict__ a,
                                         const uint32_t* __restrict__ block_dim,
                                         float* __restrict__ lu,
                                         uint32_t block_count) {
    const uint32_t blk = blockIdx.x;
    if (blk >= block_count) return;
    const uint32_t lane = threadIdx.x;

    __shared__ double a_sh[kStride];
    __shared__ double lu_sh[kStride];

    const uint32_t n = block_dim[blk];
    const float* __restrict__ a_blk = a + static_cast<size_t>(blk) * kStride;
    float* __restrict__ lu_blk = lu + static_cast<size_t>(blk) * kStride;

    // Stream A into shared (coalesced).
    for (uint32_t idx = lane; idx < kStride; idx += kWarpSize)
        a_sh[idx] = static_cast<double>(a_blk[idx]);
    __syncwarp(0xffffffffu);

    if (n > 0u && lane == 0u) Ilu0FactorizeDevice(a_sh, n, kMaxBlockDim, lu_sh);
    __syncwarp(0xffffffffu);

    // Store the factor (and deterministic zero padding outside the n x n).
    for (uint32_t idx = lane; idx < kStride; idx += kWarpSize) {
        const uint32_t row = idx / kMaxBlockDim;
        const uint32_t col = idx % kMaxBlockDim;
        const bool in_block = (row < n && col < n);
        lu_blk[idx] = in_block ? static_cast<float>(lu_sh[idx]) : 0.0f;
    }
}
}  // namespace

// Host launcher for the device ILU(0) factorization test kernel.
void LaunchIlu0FactorizeTest(const phi::DeviceContext& context, const float* a,
                             const uint32_t* block_dim, float* lu,
                             uint32_t block_count) {
    if (block_count == 0u) return;
    phi::ScopedDeviceGuard guard(context.device_id);
    Ilu0FactorizeTestKernel<<<block_count, kWarpSize, 0u, context.stream.Native()>>>(
        a, block_dim, lu, block_count);
}

}  // namespace nuka::diffsim
