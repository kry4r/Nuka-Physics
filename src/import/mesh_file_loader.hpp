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
// STL / OBJ / COLLADA readers are self-written (COLLADA on tinyxml2); glTF goes
// through the vendored cgltf reader (external/cgltf, MIT). No assimp / tinyobj.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::import {

/// A run of triangles drawn with one material: a glTF primitive or a COLLADA
/// <triangles material=...> block. Map paths are absolute; an image inside a file
/// (a .glb buffer) is named "file#offset+length".
struct MeshPart {
    std::uint32_t first_index = 0;  // offset into MeshGeometry::indices
    std::uint32_t index_count = 0;
    std::string material;           // the file's material name ("" when it names none)
    bool has_color = false;
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};  // base color RGBA when has_color
    float metallic = 0.0f;
    float roughness = 0.5f;
    float emissive[3] = {0.0f, 0.0f, 0.0f};
    std::string albedo_map;
    std::string roughness_map;
    std::string metallic_map;
    std::string normal_map;
    std::string emissive_map;
    std::uint32_t roughness_channel = 0;  // 0 R, 1 G, 2 B
    std::uint32_t metallic_channel = 0;
};

/// Flat triangle geometry. The element types match
/// nuka::scene::CollisionShapeRecord::mesh_vertices / mesh_indices exactly so
/// the result can be std::move-assigned straight into a shape record.
struct MeshGeometry {
    std::vector<float>    vertices;  // x,y,z triples (file order, no dedup)
    std::vector<float>    normals;   // x,y,z per-vertex normals, 1:1 with vertices (may be empty)
    std::vector<float>    uvs;       // u,v per-vertex coordinates, 1:1 with vertices (may be empty)
    std::vector<uint32_t> indices;   // triangle indices (3 per triangle)
    std::vector<MeshPart> parts;     // material runs over indices; empty for STL / OBJ

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

/// Load a Wavefront OBJ file (v / vt / vn / f). Faces may be `f a b c`,
/// `f a/t b/t c/t`, `f a//n b//n c//n`, and `f a/t/n b/t/n c/t/n`; polygons
/// with >3 vertices are fan triangulated (v0, v_i, v_{i+1}). OBJ indices are
/// 1-based; negative indices are relative to the current declaration count.
/// Position/UV/normal seams are expanded into deterministic per-corner vertices.
/// mtllib/usemtl/g/o/s are ignored.
///
/// Throws std::runtime_error on a missing or malformed file.
MeshGeometry LoadObj(const std::string& path);

/// Load a glTF 2.0 file (.gltf with external or embedded buffers, or .glb).
/// Every mesh the default scene instances is flattened in node order with its
/// world transform baked in, and +Y up becomes +Z up. Each triangle, strip or
/// fan primitive becomes one part carrying its material's PBR factors and maps;
/// points and lines are skipped. UVs are stored v up (v' = 1 - v). Throws
/// std::runtime_error on an unreadable or Draco-compressed file.
MeshGeometry LoadGltf(const std::string& path);

/// Load a COLLADA file: <triangles>, <polylist>, <polygons>, <tristrips> and
/// <trifans> under the visual scene's nodes (matrix / translate / rotate /
/// scale, instance_node), scaled to meters by <unit meter>. <up_axis> is ignored
/// as ROS tools do. Each primitive becomes one part colored by its bound
/// effect's diffuse color or texture. Throws std::runtime_error on a malformed file.
MeshGeometry LoadDae(const std::string& path);

/// The triangles of `mesh.parts[part]` as their own mesh, vertices renumbered
/// in first-use order; the result has no parts.
MeshGeometry ExtractPart(const MeshGeometry& mesh, std::size_t part);

/// Dispatch by lowercased file extension (".stl", ".obj", ".gltf", ".glb", ".dae").
/// Throws std::runtime_error on an unsupported extension.
MeshGeometry LoadMeshFile(const std::string& path);

} // namespace nuka::import
