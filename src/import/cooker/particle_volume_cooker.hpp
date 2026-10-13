#pragma once
// Cell-centered particle sampling for authored material volumes.

#include "import/cooker/particle_lattice.hpp"
#include "math/vec3.hpp"
#include <cstdint>

namespace nuka::import::cooker {

struct ParticleVolumeSpec {
    math::Vec3 min_corner = {0.0f, 0.0f, 0.0f};
    math::Vec3 max_corner = {1.0f, 1.0f, 1.0f};
    float spacing = 0.1f;
    float rest_density = 1000.0f;
    // Per-cell displacement in [-jitter * spacing / 2, jitter * spacing / 2].
    float position_jitter = 0.0f;
};

struct ParticleLatticeCounts {
    uint32_t nx = 0u;
    uint32_t ny = 0u;
    uint32_t nz = 0u;
    uint32_t total = 0u;
};
ParticleLatticeCounts ParticleBoxLatticeCounts(const ParticleVolumeSpec& spec);
float LatticeParticleMass(const ParticleVolumeSpec& spec);

// Samples x fastest with uniform mass rho * spacing^3 and zero initial velocity.
runtime::fluid::ParticleLattice CookParticleBox(const ParticleVolumeSpec& spec);
// Retains box samples inside its inscribed sphere.
runtime::fluid::ParticleLattice CookParticleSphere(const ParticleVolumeSpec& spec);

} // namespace nuka::import::cooker
