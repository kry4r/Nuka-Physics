#pragma once

#include <vector>

#include "collision/mesh_surface_types.hpp"
#include "import/cooker/convex_cover.hpp"

namespace nuka::import::cooker {

struct CookedMeshSurface {
    collision::MeshSurfaceInfo info{};
    std::vector<collision::MeshBvhNode> nodes;
    ConvexCoverResult cover;
    std::string cache_key;
    bool cache_hit = false;
    bool cover_hierarchy = false;
};

struct MeshSurfaceCookOptions {
    ConvexCoverParams cover;
    bool decompose = true;
    bool oriented_surface = false;
    bool allow_device = true;
    std::string cache_directory = ".nuka_cache";
};

void ValidateMeshSurfaceInput(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count);

CookedMeshSurface CookMeshSurface(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool require_convex = false);

CookedMeshSurface CookMeshSurfaceCached(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool require_convex,
    const MeshSurfaceCookOptions& options = {});

// Every source triangle occurs once; each node contains its complete subtree.
bool MeshSurfaceTreeValid(collision::MeshSurfaceView source, collision::MeshSurfaceInfo info);

void GroupMeshSurfaceByCover(CookedMeshSurface& surface, const float* vertices,
    const uint32_t* indices);

}  // namespace nuka::import::cooker
