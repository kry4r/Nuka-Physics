#pragma once
// ---------------------------------------------------------------------------
// heightfield.hpp -- the ONE elevation-grid representation (MuJoCo hfield model).
//
// A regular nrow x ncol grid of NORMALIZED [0,1] heights placed in the world by
// radius/elevation/base. Filled by either loader (image or parametric generator)
// and read by the ONE bilinear sampler (heightfield_sample.hpp) -- so obs/spawn/
// render and the physics cook share a single height definition by construction.
// World z at a corner = base_z + value*scale_z. Trivially copyable (the heights
// ride a std::vector; the placement is plain scalars), so the cook stages it into
// the device HeightfieldData locator with no analytic formula in any kernel.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

#include "math/vec3.hpp"

namespace nuka::terrain {

// A regular elevation grid. Heights NORMALIZED [0,1]; world z = base_z+v*scale_z.
struct HeightField {
    std::vector<float> values;     // row-major, length nrow*ncol; values[r*ncol+c].
    uint32_t nrow = 0u;            // rows along +Y.
    uint32_t ncol = 0u;            // cols along +X.
    float    cell_x = 0.0f;        // X spacing between columns.
    float    cell_y = 0.0f;        // Y spacing between rows.
    math::Vec3 origin{};           // world pos of grid corner cell (0,0) at value 0.
    float    base_z = 0.0f;        // absolute world z of value 0.
    float    scale_z = 0.0f;       // world rise for value 1 (== elevation extent).
};

// nrow,ncol>=2; both cell spacings > 0; values fully populated.
inline bool IsValid(const HeightField& hf) {
    return hf.nrow >= 2u && hf.ncol >= 2u && hf.cell_x > 0.0f &&
           hf.cell_y > 0.0f &&
           hf.values.size() ==
               static_cast<size_t>(hf.nrow) * static_cast<size_t>(hf.ncol);
}

// Absolute world z of grid corner (r,c).
inline float WorldZAt(const HeightField& hf, uint32_t r, uint32_t c) {
    const size_t i = static_cast<size_t>(r) * hf.ncol + c;
    return hf.base_z + hf.values[i] * hf.scale_z;
}

// (width_x, width_y, z-extent) of the grid footprint.
inline math::Vec3 Extent(const HeightField& hf) {
    return math::Vec3{static_cast<float>(hf.ncol - 1u) * hf.cell_x,
                      static_cast<float>(hf.nrow - 1u) * hf.cell_y, hf.scale_z};
}

}  // namespace nuka::terrain
