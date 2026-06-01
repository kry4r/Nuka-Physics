#pragma once
// ---------------------------------------------------------------------------
// n1_gaussian.hpp -- launch decl for in-place Gaussian sensor noise (v0.5 p04 N1)
// ---------------------------------------------------------------------------

#include "phi/device_context.hpp"

#include <cstdint>

namespace nuka::sensor::noise {

// Adds Philox-Gaussian(mean, stddev) noise IN PLACE to `data[0..count)`. Each
// element gets a sample that is a pure function of (seed, element_idx, seq), so
// the result is D1 two-run bit-exact AND replay-stable. `data` must be a device
// pointer to `count` contiguous floats. Launches on ctx.stream (NOT synced here;
// the caller synchronizes). One element per thread, coalesced, __restrict__.
void LaunchGaussianNoise(const phi::DeviceContext& ctx, float* data,
                         uint32_t count, float mean, float stddev,
                         uint64_t seed, uint64_t seq);

}  // namespace nuka::sensor::noise
