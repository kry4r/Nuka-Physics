#pragma once
// ---------------------------------------------------------------------------
// n1_poisson.hpp -- launch decl for in-place Poisson sensor noise (v0.5 p04 N1)
// ---------------------------------------------------------------------------

#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace nuka::sensor::noise {

// Adds a Philox-Poisson(lambda) sample IN PLACE to `data[0..count)`:
//     data[idx] += (float)Poisson(lambda)
// This is the "shot count added to the reading" form (the engine-test-friendly
// choice documented in the spec): added to a ZERO buffer the per-element result
// is Poisson(lambda), whose mean AND variance both equal lambda -- the variance
// equality is the signature that distinguishes it from Gaussian. For a non-zero
// depth/lidar reading it superimposes a Poisson count on the existing value.
//
// Pure function of (seed, element_idx, seq) => D1 two-run bit-exact and replay-
// stable. No atomics, one element per thread, __restrict__. Launches on
// ctx.stream (not synced here). See philox.cuh PoissonSample (Knuth's method).
void LaunchPoissonNoise(cudaStream_t stream, int device_id, float* data,
                        uint32_t count, float lambda, uint64_t seed,
                        uint64_t seq);

}  // namespace nuka::sensor::noise
