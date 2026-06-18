#pragma once
// ---------------------------------------------------------------------------
// heightfield_sample.hpp -- the ONE bilinear sampler over a HeightField grid.
//
// THE single height query for obs/spawn/render AND any host read of the cooked
// grid. Columns outside the grid clamp to the edge. The physics narrowphase reads
// the SAME grid via exact prism corners, so the bilinear sample and the prism tops
// agree at corners; bilinear is the correct smooth interpolant between them. The
// raw overload lets a kernel sample without the std::vector POD. Host/device
// portable (the qualifier macro gates on __CUDACC__, the philox.cuh precedent).
// ---------------------------------------------------------------------------

#include <cstdint>

#include "math/vec3.hpp"
#include "scene/terrain/heightfield.hpp"

#if defined(__CUDACC__)
#define NK_TERRAIN_SAMPLE_HD __host__ __device__
#else
#define NK_TERRAIN_SAMPLE_HD
#endif

namespace nuka::terrain {

// Bilinear absolute world z at world column (x,y) over the raw grid. Columns
// outside [origin, origin+extent] clamp to the edge row/col.
NK_TERRAIN_SAMPLE_HD inline float SampleHeightFieldZRaw(
    const float* values, uint32_t nrow, uint32_t ncol, math::Vec3 origin,
    float base_z, float scale_z, float cell_x, float cell_y, float x, float y) {
    if (values == nullptr || nrow < 2u || ncol < 2u || !(cell_x > 0.0f) ||
        !(cell_y > 0.0f)) {
        return base_z;
    }
    // Continuous grid coordinate, clamped to the interior so the +1 corner exists.
    float fc = (x - origin.x) / cell_x;
    float fr = (y - origin.y) / cell_y;
    if (fc < 0.0f) fc = 0.0f;
    if (fr < 0.0f) fr = 0.0f;
    const float max_c = static_cast<float>(ncol - 1u);
    const float max_r = static_cast<float>(nrow - 1u);
    if (fc > max_c) fc = max_c;
    if (fr > max_r) fr = max_r;
    uint32_t c0 = static_cast<uint32_t>(fc);
    uint32_t r0 = static_cast<uint32_t>(fr);
    if (c0 >= ncol - 1u) c0 = ncol - 2u;
    if (r0 >= nrow - 1u) r0 = nrow - 2u;
    const uint32_t c1 = c0 + 1u;
    const uint32_t r1 = r0 + 1u;
    const float tx = fc - static_cast<float>(c0);
    const float ty = fr - static_cast<float>(r0);
    const float v00 = values[static_cast<size_t>(r0) * ncol + c0];
    const float v10 = values[static_cast<size_t>(r0) * ncol + c1];
    const float v01 = values[static_cast<size_t>(r1) * ncol + c0];
    const float v11 = values[static_cast<size_t>(r1) * ncol + c1];
    const float top = v00 + (v10 - v00) * tx;
    const float bot = v01 + (v11 - v01) * tx;
    const float v = top + (bot - top) * ty;
    return base_z + v * scale_z;
}

// Bilinear absolute world z at world column (x,y) over a HeightField POD.
inline float SampleHeightFieldZ(const HeightField& hf, float x, float y) {
    return SampleHeightFieldZRaw(hf.values.data(), hf.nrow, hf.ncol, hf.origin,
                                 hf.base_z, hf.scale_z, hf.cell_x, hf.cell_y, x,
                                 y);
}

}  // namespace nuka::terrain
