// ---------------------------------------------------------------------------
// n1_gaussian.cu -- in-place Philox-Gaussian sensor noise kernel (v0.5 p04 N1)
// ---------------------------------------------------------------------------
//
// data[idx] += Gaussian(mean, stddev) where the sample is a pure function of
// (seed, idx, seq). Counter-based => D1 two-run bit-exact and replay-stable; no
// atomics, no thread-order reductions, one element per thread. See philox.cuh.
// ---------------------------------------------------------------------------

#include "sensor/noise/n1_gaussian.hpp"
#include "sensor/noise/philox.cuh"

#include <cuda_runtime.h>

namespace nuka::sensor::noise {

namespace {

__global__ void GaussianNoiseKernel(float* __restrict__ data, uint32_t count,
                                     float mean, float stddev, uint64_t seed,
                                     uint64_t seq) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    data[idx] += GaussianSample(seed, idx, seq, mean, stddev);
}

}  // namespace

void LaunchGaussianNoise(cudaStream_t stream, int device_id, float* data,
                         uint32_t count, float mean, float stddev,
                         uint64_t seed, uint64_t seq) {
    if (data == nullptr || count == 0u) {
        return;
    }
    constexpr uint32_t kBlock = 256u;
    const uint32_t grid = (count + kBlock - 1u) / kBlock;
    GaussianNoiseKernel<<<grid, kBlock, 0, stream>>>(
        data, count, mean, stddev, seed, seq);
}

}  // namespace nuka::sensor::noise
