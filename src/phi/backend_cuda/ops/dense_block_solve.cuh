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

// All CTA threads call with the same arguments; failed solves may overwrite workspace and direction.
__device__ inline bool SolveDensePositiveBlock(
    float* matrix, const float* rhs, float* direction, float* diagonal,
    uint32_t dimension, uint32_t stride) {
    if (dimension == 0u) return true;
    if (dimension > stride) return false;
    __shared__ uint32_t valid;
    if (threadIdx.x == 0u) valid = 1u;
    __syncthreads();
    for (uint32_t k = 0u; k < dimension; ++k) {
        if (threadIdx.x == 0u) {
            double pivot = static_cast<double>(matrix[size_t{k} * stride + k]);
            for (uint32_t j = 0u; j < k; ++j) {
                const double coefficient = static_cast<double>(matrix[size_t{k} * stride + j]);
                pivot -= coefficient * coefficient * static_cast<double>(diagonal[j]);
            }
            if (!(pivot > 0.0 && pivot <= static_cast<double>(FLT_MAX))) {
                valid = 0u;
            } else {
                diagonal[k] = static_cast<float>(pivot);
                if (!(diagonal[k] > 0.0f)) valid = 0u;
            }
        }
        if (__syncthreads_or(threadIdx.x == 0u && valid == 0u)) return false;
        bool rows_valid = true;
        for (size_t r = size_t{k} + 1u + threadIdx.x; r < dimension; r += blockDim.x) {
            double coefficient = static_cast<double>(matrix[r * stride + k]);
            for (uint32_t j = 0u; j < k; ++j) {
                coefficient -= static_cast<double>(matrix[r * stride + j]) *
                    static_cast<double>(matrix[size_t{k} * stride + j]) *
                    static_cast<double>(diagonal[j]);
            }
            coefficient /= static_cast<double>(diagonal[k]);
            if (!(coefficient >= -static_cast<double>(FLT_MAX) &&
                  coefficient <= static_cast<double>(FLT_MAX))) {
                atomicExch(&valid, 0u);
                rows_valid = false;
            } else {
                matrix[r * stride + k] = static_cast<float>(coefficient);
            }
        }
        if (__syncthreads_or(!rows_valid)) return false;
    }
    for (size_t i = threadIdx.x; i < dimension; i += blockDim.x) direction[i] = rhs[i];
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (uint32_t i = 0u; i < dimension && valid != 0u; ++i) {
            double value = static_cast<double>(direction[i]);
            for (uint32_t j = 0u; j < i; ++j)
                value -= static_cast<double>(matrix[size_t{i} * stride + j]) *
                    static_cast<double>(direction[j]);
            if (!(value >= -static_cast<double>(FLT_MAX) && value <= static_cast<double>(FLT_MAX)))
                valid = 0u;
            else direction[i] = static_cast<float>(value);
        }
        for (uint32_t i = 0u; i < dimension && valid != 0u; ++i) {
            const double value = static_cast<double>(direction[i]) / static_cast<double>(diagonal[i]);
            if (!(value >= -static_cast<double>(FLT_MAX) && value <= static_cast<double>(FLT_MAX)))
                valid = 0u;
            else direction[i] = static_cast<float>(value);
        }
        for (uint32_t ii = dimension; ii > 0u && valid != 0u; --ii) {
            const uint32_t i = ii - 1u;
            double value = static_cast<double>(direction[i]);
            for (uint32_t j = i + 1u; j < dimension; ++j)
                value -= static_cast<double>(matrix[size_t{j} * stride + i]) *
                    static_cast<double>(direction[j]);
            if (!(value >= -static_cast<double>(FLT_MAX) && value <= static_cast<double>(FLT_MAX)))
                valid = 0u;
            else direction[i] = static_cast<float>(value);
        }
    }
    return __syncthreads_or(threadIdx.x == 0u && valid == 0u) == 0;
}

}  // namespace nuka::phi::nkops
