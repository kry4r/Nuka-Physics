#pragma once
// ---------------------------------------------------------------------------
// nuka::import - Self-written, SCOPED USDC (crate) binary mesh reader
// ---------------------------------------------------------------------------
//
// Reads the ONE Mesh prim out of a USDC ("crate") binary .usd file and returns
// its geometry (points + faceVertexIndices + faceVertexCounts). This is a
// dependency-free, from-scratch parser of the public OpenUSD crate format
// (bootstrap -> TOC -> TOKENS/STRINGS/FIELDS/FIELDSETS/PATHS/SPECS). It uses NO
// OpenUSD / pxr / usd-core code -- the no-closed-SDK pillar. The format itself
// (LZ4-block compression, the integer-coding scheme, the structural sections)
// is documented in the OpenUSD source (pxr/usd/sdf/crateFile.cpp,
// integerCoding.cpp, fastCompression.cpp) and re-implemented here.
//
// SCOPE (v0.8 C7a): this targets exactly the newton-assets cup mesh.usd, which
// is a single `def Mesh "/Model"` with point3f[] points (raw float32),
// int[] faceVertexIndices and int[] faceVertexCounts (USD integer-compressed),
// crate version 0.8.0. General crate features (variants / payloads / sublayers /
// nested references / multiple ref targets / TimeSamples / dictionaries) are
// OUT of scope and are reported via an exception rather than silently ignored.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::import {

// Flat geometry extracted from a crate Mesh prim. Element types match
// nuka::scene::CollisionShapeRecord / the USDA importer's UsdPrim so the result
// can be grafted straight in. File order preserved exactly (determinism / D1):
// no dedup, no reordering.
struct CrateMesh {
    std::string        prim_path;       // e.g. "/Model" (the resolved Mesh path)
    std::vector<float> points;          // x,y,z triples (point3f[] points), file order
    std::vector<int>   face_vertex_indices;  // int[] faceVertexIndices
    std::vector<int>   face_vertex_counts;   // int[] faceVertexCounts (per-face vert count)

    std::size_t PointCount() const { return points.size() / 3; }
};

// Read the Mesh prim at `target_prim_path` (e.g. "/Model") from the USDC crate
// file at `path`. If `target_prim_path` is empty, the first Mesh prim is used.
//
// Throws std::runtime_error on: a non-crate / unsupported-version file, a
// missing target Mesh prim, an out-of-scope crate feature, or any structural
// corruption. NEVER returns partial/garbage geometry silently.
CrateMesh ReadUsdcMesh(const std::string& path, const std::string& target_prim_path);

} // namespace nuka::import
