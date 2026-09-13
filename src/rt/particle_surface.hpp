#pragma once

#include "math/vec3.hpp"

#include <cstdint>
#include <vector>

namespace nuka::rt {

// A view of environment-major particle positions owned by the simulation backend.
struct ParticlePositionSource {
    const math::Vec3* positions = nullptr;
    uint32_t particles_per_env = 0u;
    uint32_t env_count = 0u;
};

// Triangle corners index particles within one environment; the mesh slot is shared by its instances.
struct ParticleSurfaceBinding {
    uint32_t mesh_id = 0u;
    std::vector<uint32_t> triangle_particles;
    float normal_offset = 0.0f;
    uint32_t smooth_iters = 0u;
    float smooth_lambda = 0.5f;
};

}  // namespace nuka::rt
