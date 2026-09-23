#pragma once

#include "math/vec3.hpp"
#include "render/mesh_geometry.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime {

struct ParticleGrainParams {
    float radius = 0.0f;
    bool round = false;
    float radius_jitter = 0.0f;
    float tint_jitter = 0.0f;
};

// A zero count consumes the remainder; index_base preserves per-grain appearance in sliced inputs.
render::MeshGeometry BakeParticleSpheres(const std::vector<math::Vec3>& positions,
    uint32_t first, uint32_t count, float radius, bool round, float radius_jitter,
    float tint_jitter, uint32_t index_base = 0u);

}  // namespace nuka::runtime
