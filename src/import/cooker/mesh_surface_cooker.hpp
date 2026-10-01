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

struct CookedMeshEdges {
    collision::MeshEdgeInfo info{};
    uint32_t topology_edge_count = 0u;
    std::vector<collision::MeshEdge> edges;
    std::vector<collision::MeshBvhNode> nodes;
    std::vector<uint32_t> triangle_edges;
    std::vector<uint32_t> triangle_vertex_owner;
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

std::vector<uint32_t> OrientMeshWinding(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool weld_vertices = true);

CookedMeshSurface CookMeshSurface(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool require_convex = false);

CookedMeshEdges CookMeshEdges(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool weld_vertices = true);

CookedMeshEdges CookWireEdges(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t edge_count);

CookedMeshSurface CookMeshSurfaceCached(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool require_convex,
    const MeshSurfaceCookOptions& options = {});

// Every source triangle occurs once; each node contains its complete subtree.
bool MeshSurfaceTreeValid(collision::MeshSurfaceView source, collision::MeshSurfaceInfo info);

void GroupMeshSurfaceByCover(CookedMeshSurface& surface, const float* vertices,
    const uint32_t* indices);

}  // namespace nuka::import::cooker
