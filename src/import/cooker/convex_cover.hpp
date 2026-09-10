#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "collision/mesh_surface_types.hpp"
#include "import/cooker/mesh_query_backend.hpp"

namespace nuka::import::cooker {

struct CoverPoint {
    double x = 0.0, y = 0.0, z = 0.0;
};

struct CoverPlane {
    CoverPoint normal;
    double offset = 0.0;
};

struct ConvexCoverPart {
    std::vector<CoverPlane> planes;
    std::vector<CoverPoint> vertices;
};

struct ConvexCoverParams {
    double relative_error = 0.03;
    uint32_t max_parts = 128u;
    uint32_t max_planes = 64u;
    uint32_t max_cells = 250000u;
    uint32_t max_operations = 1024u;
};

enum class ConvexCoverStatus : uint32_t {
    ExactSurface,
    Complete,
    BudgetExceeded,
    NumericalFailure,
    BackendFailure,
};

struct ConvexCoverResult {
    std::vector<ConvexCoverPart> parts;
    ConvexCoverStatus status = ConvexCoverStatus::ExactSurface;
    std::string backend;
    std::string reason;
    uint32_t operations = 0u;
    uint32_t distance_cells = 0u;
    uint64_t query_points = 0u;
};

// Approximate solid covers guide grouping; source triangles remain the collision oracle.
ConvexCoverResult BuildConvexCover(collision::MeshSurfaceView source,
    collision::MeshSurfaceInfo info, const ConvexCoverParams& params,
    MeshQueryBackend& queries);

}  // namespace nuka::import::cooker
