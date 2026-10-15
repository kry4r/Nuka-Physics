#pragma once

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace nuka::phi::nkops {

__device__ inline uint32_t RowGroupWidth(uint32_t dimension) {
    uint32_t width = 1u;
    while (width < dimension && width < static_cast<uint32_t>(warpSize)) width <<= 1u;
    return width;
}

__device__ inline double SumDenseRowGroup(double value, uint32_t width) {
    for (uint32_t offset = width / 2u; offset > 0u; offset /= 2u)
        value += __shfl_down_sync(0xffffffffu, value, offset, width);
    return __shfl_sync(0xffffffffu, value, 0u, width);
}

__device__ inline double SumDenseBlockWarp(double value) {
    for (uint32_t offset = warpSize / 2u; offset > 0u; offset /= 2u)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return __shfl_sync(0xffffffffu, value, 0u);
}

__device__ inline double SumDenseBlock(double value) {
    __shared__ double partial[32];
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    const uint32_t warps = blockDim.x / warpSize;
    value = SumDenseBlockWarp(value);
    if (lane == 0u) partial[warp] = value;
    __syncthreads();
    if (warp == 0u) {
        value = SumDenseBlockWarp(lane < warps ? partial[lane] : 0.0);
        if (lane == 0u) partial[0] = value;
    }
    __syncthreads();
    const double result = partial[0];
    __syncthreads();
    return result;
}

__device__ inline bool InFloatRange(double value) {
    return value >= -static_cast<double>(FLT_MAX) && value <= static_cast<double>(FLT_MAX);
}

// Solves H d = g from the lower triangle of H by LDL^T in shared memory, each sum taken in double over
// float factors in increasing column order; all CTA threads call with the same arguments.
template <uint32_t kMaxDof>
__device__ inline bool SolveDensePositiveBlock(const float* matrix, const float* rhs, float* direction,
                                               uint32_t dimension, uint32_t stride) {
    // Packed lower rows, (r, c) at r(r+1)/2 + c: the trailing block's sums, then L and the pivots.
    __shared__ double lower[kMaxDof * (kMaxDof + 1u) / 2u];
    __shared__ double sums[kMaxDof];
    __shared__ double solution[kMaxDof];
    __shared__ double column[2];
    if (dimension == 0u) return true;
    if (dimension > stride) return false;
    const uint32_t n = dimension;
    const uint32_t lane = threadIdx.x % warpSize;
    for (uint32_t r = threadIdx.x / warpSize; r < n; r += blockDim.x / warpSize)
        for (uint32_t c = lane; c <= r; c += warpSize)
            lower[r * (r + 1u) / 2u + c] = static_cast<double>(matrix[size_t{r} * stride + c]);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) sums[i] = static_cast<double>(rhs[i]);
    __syncthreads();
    for (uint32_t k = 0u; k < n; ++k) {
        // One warp takes the pivot, column k of L and y_k, keeping the slow double pipe free of repeats.
        bool valid = true;
        if (threadIdx.x < warpSize) {
            const double pivot = lower[k * (k + 1u) / 2u + k];
            const double value = sums[k];
            valid = pivot > 0.0 && pivot <= static_cast<double>(FLT_MAX) && static_cast<float>(pivot) > 0.0f &&
                    InFloatRange(value);
            const double diagonal = static_cast<float>(pivot);
            for (uint32_t r = k + 1u + lane; r < n; r += warpSize) {
                const double coefficient = lower[r * (r + 1u) / 2u + k] / diagonal;
                valid &= InFloatRange(coefficient);
                lower[r * (r + 1u) / 2u + k] = static_cast<float>(coefficient);
            }
            if (lane == 0u) {
                column[0] = diagonal;
                column[1] = static_cast<float>(value);
                solution[k] = column[1];
            }
        }
        if (__syncthreads_or(!valid)) return false;
        const double diagonal = column[0], y = column[1];
        // Column k leaves the trailing block and the forward substitution's remaining sums.
        const uint32_t m = n - k - 1u, entries = m * (m + 1u) / 2u;
        for (uint32_t e = threadIdx.x; e < entries + m; e += blockDim.x) {
            if (e >= entries) {
                const uint32_t r = k + 1u + e - entries;
                sums[r] -= lower[r * (r + 1u) / 2u + k] * y;
                continue;
            }
            uint32_t i = static_cast<uint32_t>((sqrtf(8.0f * static_cast<float>(e) + 1.0f) - 1.0f) * 0.5f);
            while (i * (i + 1u) / 2u > e) --i;
            while ((i + 1u) * (i + 2u) / 2u <= e) ++i;
            const uint32_t r = k + 1u + i, c = k + 1u + e - i * (i + 1u) / 2u;
            lower[r * (r + 1u) / 2u + c] -= lower[r * (r + 1u) / 2u + k] * lower[c * (c + 1u) / 2u + k] * diagonal;
        }
        __syncthreads();
    }
    bool valid = true;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const double value = solution[i] / static_cast<double>(static_cast<float>(lower[i * (i + 1u) / 2u + i]));
        valid &= InFloatRange(value);
        solution[i] = static_cast<float>(value);
    }
    if (__syncthreads_or(!valid)) return false;
    if (threadIdx.x == 0u) {
        for (uint32_t ii = n; ii > 0u && valid; --ii) {
            const uint32_t i = ii - 1u;
            double value = solution[i];
            for (uint32_t j = i + 1u; j < n; ++j) value -= lower[j * (j + 1u) / 2u + i] * solution[j];
            valid = InFloatRange(value);
            solution[i] = static_cast<float>(value);
        }
    }
    if (__syncthreads_or(!valid)) return false;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) direction[i] = static_cast<float>(solution[i]);
    __syncthreads();
    return true;
}

}  // namespace nuka::phi::nkops
