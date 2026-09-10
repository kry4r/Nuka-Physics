#pragma once
// Deterministic CPU V-HACD decomposition with canonical output and content caching.
// This optional approximation is separate from source triangle-surface cooking.

#include "import/cooker/convex_piece.hpp"

#include <cstdint>

namespace nuka::import::cooker {

/// Decomposition mode authored on a mesh (USD nuka:decompose / URDF / MJCF).
enum class DecomposeMode : uint8_t {
    Auto,   // prepare acceleration while preserving the authored surface
    Force,  // always run V-HACD
    Skip,   // preserve source vertices, triangles, and shape type
};

/// Parameters controlling the convex decomposition. Defaults match the spec.
struct ConvexDecompositionParams {
    uint32_t max_pieces             = 32;      // upper bound on output hulls
    uint32_t voxel_resolution       = 100000;  // V-HACD voxelization grid
    float    concavity_threshold    = 0.001f;  // min volume-error % (V-HACD)
    uint32_t max_vertices_per_piece = 64;      // max verts per output hull
    bool     project_hull_vertices  = true;    // shrink-wrap hull verts to source
};

/// Decompose a triangle mesh into convex pieces. PURE: always invokes V-HACD,
/// never consults a cache. Output is deterministic (canonicalized) for a given
/// (mesh, params). Returns succeeded=false with an error_message on bad input
/// or V-HACD failure (never throws).
///
/// @param vertices       x,y,z triples (length == 3 * vertex_count)
/// @param vertex_count   number of source vertices
/// @param indices        triangle indices (length == 3 * triangle_count)
/// @param triangle_count number of source triangles
ConvexDecompositionResult DecomposeMesh(const float* vertices, uint32_t vertex_count,
                                        const uint32_t* indices, uint32_t triangle_count,
                                        const ConvexDecompositionParams& params);

/// Content-hash key for (mesh bytes + serialized params): SHA-256 hex string.
/// Stable across runs/processes for identical inputs.
std::string ComputeCacheKey(const float* vertices, uint32_t vertex_count,
                            const uint32_t* indices, uint32_t triangle_count,
                            const ConvexDecompositionParams& params);

/// Cached decomposition. Same (mesh, params) => served from an in-process memo
/// (and an optional on-disk cache dir) instead of re-running the 0.5-5s V-HACD.
/// This is a thin memoizing wrapper over DecomposeMesh(); the underlying
/// decomposition is unchanged.
///
/// @param cache_dir  optional on-disk cache directory; empty => in-process memo
///                   only. The dir is created on first store if it does not
///                   exist. Files are named "<sha256>.nukacvx".
/// @param was_hit    out-param set to true if served from cache (memo or disk).
ConvexDecompositionResult DecomposeMeshCached(const float* vertices, uint32_t vertex_count,
                                              const uint32_t* indices, uint32_t triangle_count,
                                              const ConvexDecompositionParams& params,
                                              const std::string& cache_dir,
                                              bool* was_hit);

/// Clear the in-process decomposition memo (test hook).
void ClearDecompositionMemo();

} // namespace nuka::import::cooker
