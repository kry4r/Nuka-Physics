// ---------------------------------------------------------------------------
// n1_poisson.cu -- in-place Philox-Poisson sensor noise kernel (v0.5 p04 N1)
// ---------------------------------------------------------------------------
//
// data[idx] += (float)Poisson(lambda) where the draw is a pure function of
// (seed, idx, seq) via Knuth's algorithm (philox.cuh). Counter-based => D1 two-
// run bit-exact and replay-stable; no atomics, one element per thread.
//
// Method choice (documented per spec): we ADD the raw Poisson count to the
// reading (shot noise superimposed on the value). On a zero buffer this makes
// each element Poisson(lambda) with mean == variance == lambda; on a real
// depth/lidar reading it adds the shot count. We deliberately do NOT use the
// zero-mean Poisson(lambda)-lambda form so the statistical test can assert BOTH
// mean ~= lambda and variance ~= lambda (the Poisson signature).
// ---------------------------------------------------------------------------

#include "sensor/noise/n1_poisson.hpp"
#include "sensor/noise/philox.cuh"

#include <cuda_runtime.h>

namespace nuka::sensor::noise {

namespace {

__global__ void PoissonNoiseKernel(float* __restrict__ data, uint32_t count,
                                   float lambda, uint64_t seed, uint64_t seq) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    data[idx] += static_cast<float>(PoissonSample(seed, idx, seq, lambda));
}

}  // namespace

void LaunchPoissonNoise(const phi::DeviceContext& ctx, float* data,
                        uint32_t count, float lambda, uint64_t seed,
                        uint64_t seq) {
    if (data == nullptr || count == 0u) {
        return;
    }
    constexpr uint32_t kBlock = 256u;
    const uint32_t grid = (count + kBlock - 1u) / kBlock;
    PoissonNoiseKernel<<<grid, kBlock, 0, ctx.stream.Native()>>>(
        data, count, lambda, seed, seq);
}

}  // namespace nuka::sensor::noise
