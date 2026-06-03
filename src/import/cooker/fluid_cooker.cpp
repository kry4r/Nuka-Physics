// ---------------------------------------------------------------------------
// nuka::import::cooker -- PBF fluid cooker (v0.7 p10-B). See fluid_cooker.hpp for
// the sampling convention + scope. THIN: no simulation math here, only a
// deterministic uniform-lattice fill + the uniform mass mapping.
// ---------------------------------------------------------------------------

#include "import/cooker/fluid_cooker.hpp"

#include <cmath>
#include <cstdint>

namespace nuka::import::cooker {

namespace {
using math::Vec3;
using runtime::fluid::PbfParticleSet;

// floor(L/s) clamped to >= 0 as a uint32_t (an empty/inverted extent -> 0).
uint32_t AxisCount(float length, float spacing) {
    if (length <= 0.0f || spacing <= 0.0f) {
        return 0u;
    }
    const float n = std::floor(length / spacing);
    if (n <= 0.0f) {
        return 0u;
    }
    return static_cast<uint32_t>(n);
}

bool SpecValid(const FluidBoxSpec& spec) {
    return spec.spacing > 0.0f && spec.rest_density > 0.0f;
}
}  // namespace

FluidLatticeCounts FluidBoxLatticeCounts(const FluidBoxSpec& spec) {
    FluidLatticeCounts c;
    if (!SpecValid(spec)) {
        return c;  // all-zero.
    }
    c.nx = AxisCount(spec.max_corner.x - spec.min_corner.x, spec.spacing);
    c.ny = AxisCount(spec.max_corner.y - spec.min_corner.y, spec.spacing);
    c.nz = AxisCount(spec.max_corner.z - spec.min_corner.z, spec.spacing);
    c.total = c.nx * c.ny * c.nz;
    return c;
}

float FluidParticleMass(const FluidBoxSpec& spec) {
    // mass of fluid filling one cubic cell at the rest density.
    return spec.rest_density * spec.spacing * spec.spacing * spec.spacing;
}

PbfParticleSet CookFluidBox(const FluidBoxSpec& spec) {
    PbfParticleSet out;
    if (!SpecValid(spec)) {
        return out;  // empty set on a bad spec (no throw).
    }
    const FluidLatticeCounts counts = FluidBoxLatticeCounts(spec);
    if (counts.total == 0u) {
        return out;
    }

    out.particle_mass = FluidParticleMass(spec);
    out.positions.reserve(counts.total);
    out.velocities.assign(counts.total, Vec3{0.0f, 0.0f, 0.0f});

    const float s = spec.spacing;
    const float half = 0.5f * s;
    // Cell-centered fill, x fastest -> deterministic byte-identical order.
    for (uint32_t iz = 0u; iz < counts.nz; ++iz) {
        const float z = spec.min_corner.z + (static_cast<float>(iz) * s) + half;
        for (uint32_t iy = 0u; iy < counts.ny; ++iy) {
            const float y = spec.min_corner.y + (static_cast<float>(iy) * s) + half;
            for (uint32_t ix = 0u; ix < counts.nx; ++ix) {
                const float x = spec.min_corner.x + (static_cast<float>(ix) * s) + half;
                out.positions.push_back(Vec3{x, y, z});
            }
        }
    }
    return out;
}

PbfParticleSet CookFluidSphere(const FluidBoxSpec& spec) {
    const PbfParticleSet box = CookFluidBox(spec);
    PbfParticleSet out;
    out.particle_mass = box.particle_mass;
    if (box.positions.empty()) {
        return out;
    }

    // Sphere inscribed in the box: center = box center, radius = half the SMALLEST
    // box extent (so the sphere fits inside the box on every axis).
    const Vec3 center{
        0.5f * (spec.min_corner.x + spec.max_corner.x),
        0.5f * (spec.min_corner.y + spec.max_corner.y),
        0.5f * (spec.min_corner.z + spec.max_corner.z)};
    const float ex = spec.max_corner.x - spec.min_corner.x;
    const float ey = spec.max_corner.y - spec.min_corner.y;
    const float ez = spec.max_corner.z - spec.min_corner.z;
    float min_extent = ex < ey ? ex : ey;
    min_extent = min_extent < ez ? min_extent : ez;
    const float radius = 0.5f * min_extent;
    const float r2 = radius * radius;

    out.positions.reserve(box.positions.size());
    for (const Vec3& p : box.positions) {
        const float dx = p.x - center.x;
        const float dy = p.y - center.y;
        const float dz = p.z - center.z;
        if (dx * dx + dy * dy + dz * dz <= r2) {
            out.positions.push_back(p);  // keep (surviving order preserved).
        }
    }
    out.velocities.assign(out.positions.size(), Vec3{0.0f, 0.0f, 0.0f});
    return out;
}

} // namespace nuka::import::cooker
