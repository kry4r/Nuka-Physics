#pragma once

#include <vector>

#include "collision/mesh_surface_types.hpp"

namespace nuka::import::cooker {

struct CookedMeshSurface {
    collision::MeshSurfaceInfo info{};
    std::vector<collision::MeshBvhNode> nodes;
};

void ValidateMeshSurfaceInput(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count);

CookedMeshSurface CookMeshSurface(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool require_convex = false);

}  // namespace nuka::import::cooker
