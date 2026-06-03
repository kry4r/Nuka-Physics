#pragma once
// ---------------------------------------------------------------------------
// nuka::import - Self-contained, dependency-free mesh-file loader
// ---------------------------------------------------------------------------
//
// Loads triangle geometry from on-disk mesh files into a flat, deterministic
// representation that can be assigned directly into a
// nuka::scene::CollisionShapeRecord (matching mesh_vertices / mesh_indices).
//
// Determinism (D1): the loaders preserve file order EXACTLY. No vertex dedup,
// no reordering, no float reassociation. A fixed input file therefore yields a
// byte-identical vertex/index array on every load.
//
// Self-written only: plain C++17/C++20 std + std::filesystem. No third-party
// mesh libraries (no assimp / tinyobj / etc.).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::import {

/// Flat triangle geometry. The element types match
/// nuka::scene::CollisionShapeRecord::mesh_vertices / mesh_indices exactly so
/// the result can be std::move-assigned straight into a shape record.
struct MeshGeometry {
    std::vector<float>    vertices;  // x,y,z triples (file order, no dedup)
    std::vector<uint32_t> indices;   // triangle indices (3 per triangle)

    /// Number of vertices (vertices.size() / 3).
    std::size_t VertexCount() const { return vertices.size() / 3; }
    /// Number of triangles (indices.size() / 3).
    std::size_t TriangleCount() const { return indices.size() / 3; }
};

/// Load an STL file, auto-detecting binary vs ASCII.
///
/// Binary detection rule: a file is binary iff its byte size equals
/// 84 + 50 * count, where count is the little-endian uint32 at offset 80
/// (header 80B + count 4B + 50B per triangle). Otherwise it is parsed as
/// ASCII. The "solid" prefix is NOT used as the discriminator because many
/// binary STLs also begin with the bytes "solid".
///
/// Throws std::runtime_error on a missing or malformed/truncated file.
MeshGeometry LoadStl(const std::string& path);

/// Load a Wavefront OBJ file (v / f lines). Faces may be `f a b c`,
/// `f a/t b/t c/t`, `f a//n b//n c//n`, and polygons with >3 vertices are fan
/// triangulated (v0, v_i, v_{i+1}). OBJ indices are 1-based; negative indices
/// are relative to the current vertex count. vt/vn/mtllib/usemtl/g/o/s are
/// ignored for geometry purposes.
///
/// Throws std::runtime_error on a missing or malformed file.
MeshGeometry LoadObj(const std::string& path);

/// Dispatch by lowercased file extension (".stl", ".obj").
/// Throws std::runtime_error on an unsupported extension.
MeshGeometry LoadMeshFile(const std::string& path);

} // namespace nuka::import
