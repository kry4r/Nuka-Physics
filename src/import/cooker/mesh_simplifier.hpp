#pragma once

#include <cstdint>
#include <vector>

namespace nuka::import::cooker {

struct SimplifiedMesh {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
};

SimplifiedMesh SimplifyMeshQem(const float* vertices, uint32_t vertex_count,
                               const uint32_t* indices, uint32_t triangle_count,
                               uint32_t triangle_limit);

}  // namespace nuka::import::cooker
