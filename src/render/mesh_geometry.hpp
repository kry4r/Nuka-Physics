#pragma once

#include <cstdint>
#include <vector>

namespace nuka::render {

// Backend-independent indexed triangles and analytic spheres in one local frame.
struct MeshGeometry {
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<uint32_t> indices;
    std::vector<float> sphere_centers;
    std::vector<float> sphere_radii;
    std::vector<float> sphere_colors;

    uint32_t VertexCount() const { return static_cast<uint32_t>(positions.size() / 3u); }
    uint32_t TriangleCount() const { return static_cast<uint32_t>(indices.size() / 3u); }
    uint32_t SphereCount() const { return static_cast<uint32_t>(sphere_radii.size()); }
    bool Empty() const { return positions.empty() || indices.empty(); }
};

}  // namespace nuka::render
