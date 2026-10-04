#pragma once

#include <cstdint>
#include <vector>

namespace nuka::import::cooker {

struct SimplifiedMesh {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
};

// A positive max_error admits only collapses that keep both sampled surface distances, source from
// result and result from source, within it.
SimplifiedMesh SimplifyMeshQem(const float* vertices, uint32_t vertex_count,
                               const uint32_t* indices, uint32_t triangle_count,
                               uint32_t triangle_limit, float max_error = 0.0f);

}  // namespace nuka::import::cooker
