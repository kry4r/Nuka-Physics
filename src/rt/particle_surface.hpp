#pragma once

#include "math/vec3.hpp"
#include "runtime/fluid/surface_mesher.hpp"
#include "runtime/particle_skin.hpp"

#include <cstdint>
#include <vector>

namespace nuka::rt {

// A view of environment-major particle positions owned by the simulation backend.
struct ParticlePositionSource {
    const math::Vec3* positions = nullptr;
    uint32_t particles_per_env = 0u;
    uint32_t env_count = 0u;
};

// All indices and ranges are local to one environment; mesh slots remain stable across reconstruction.
struct ParticleSurfaceBinding {
    enum class Kind { Triangles, Density, Grains };
    Kind kind = Kind::Triangles;
    uint32_t mesh_id = 0u;
    std::vector<uint32_t> triangle_particles;
    float normal_offset = 0.0f;
    uint32_t smooth_iters = 0u;
    float smooth_lambda = 0.5f;
    uint32_t particle_first = 0u;
    uint32_t particle_count = 0u;
    runtime::fluid::FluidSurfaceParams density;
    runtime::ParticleGrainParams grains;
};

}  // namespace nuka::rt
