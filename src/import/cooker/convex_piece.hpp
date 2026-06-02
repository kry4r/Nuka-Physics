#pragma once
// ---------------------------------------------------------------------------
// nuka::import::cooker – Convex decomposition result structures
// ---------------------------------------------------------------------------
// Plain data structures returned by DecomposeMesh(). These carry no scene /
// runtime types so they can live in a leaf library (nuka_cooker) that the
// scene cooker links without creating a dependency cycle.
//
// COOK-TIME / HOST-CPU ONLY (master plan §5.6 sanctioned exception). No CUDA,
// no per-step concerns.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::import::cooker {

/// A single convex piece: a closed triangle mesh that bounds a convex region.
struct ConvexPiece {
    std::vector<float>    vertices;       // x,y,z triples (length == 3 * vertex_count)
    std::vector<uint32_t> indices;        // triangle indices (length == 3 * triangle_count)
    float                 volume = 0.0f;  // approximate hull volume
    float                 aabb_min[3] = {0.0f, 0.0f, 0.0f};
    float                 aabb_max[3] = {0.0f, 0.0f, 0.0f};
};

/// Result of decomposing one mesh into a union of convex pieces.
struct ConvexDecompositionResult {
    std::vector<ConvexPiece> pieces;
    bool                     succeeded = false;
    std::string              error_message;
};

} // namespace nuka::import::cooker
