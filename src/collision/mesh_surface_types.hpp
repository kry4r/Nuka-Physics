#pragma once

#include <cstdint>

#include "math/vec3.hpp"

namespace nuka::collision {

inline constexpr uint32_t kMeshSurfaceClosed = 1u;
inline constexpr uint32_t kMeshSurfaceConvex = 2u;

struct MeshGeometryCounts {
    uint32_t vertices = 0u;
    uint32_t triangles = 0u;
    uint32_t nodes = 0u;
};

// Triangle vertex indices and BVH escape/triangle indices are mesh-local.
struct MeshSurfaceInfo {
    uint32_t vertex_offset = 0u;
    uint32_t vertex_count = 0u;
    uint32_t triangle_offset = 0u;
    uint32_t triangle_count = 0u;
    uint32_t node_offset = 0u;
    uint32_t node_count = 0u;
    uint32_t flags = 0u;
    uint32_t reserved = 0u;
};

// Preorder traversal advances to the next node or skips the complete subtree.
struct MeshBvhNode {
    math::Vec3 lower{};
    uint32_t escape = 0u;
    math::Vec3 upper{};
    uint32_t triangle = ~0u;
};

static_assert(sizeof(MeshSurfaceInfo) == 32u, "MeshSurfaceInfo storage layout");
static_assert(sizeof(MeshBvhNode) == 32u, "MeshBvhNode storage layout");

struct MeshSurfaceView {
    const float* vertices = nullptr;
    const uint32_t* triangles = nullptr;
    const MeshBvhNode* nodes = nullptr;
    MeshGeometryCounts counts{};
};

struct MeshSurfacePoint {
    float distance = 0.0f;
    math::Vec3 normal{1.0f, 0.0f, 0.0f};
    math::Vec3 point{};
    math::Vec3 barycentric{};
    uint32_t triangle = ~0u;
    uint32_t feature = ~0u;
    bool valid = false;
};

}  // namespace nuka::collision
