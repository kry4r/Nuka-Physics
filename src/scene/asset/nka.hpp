#pragma once
// ---------------------------------------------------------------------------
// nuka::scene - .nka binary asset container (M2c).
//
// A .nka is a flat, deterministic, content-addressed chunk container holding
// the COOKED / heavy-byte assets a .nks scene references: visual meshes,
// collision meshes, V-HACD hull sets, sparse SDFs, sampling point sets and raw
// texture bytes. The .nks (json) sidecar carries the structural scene and
// AssetRef strings ("cup.nka#MESH/0") that index into a .nka.
//
// Layout (little-endian, version 1):
//   header  { char magic[4]="NKA1"; u32 version=1; u32 chunk_count; }
//   toc     chunk_count * { u32 fourcc; u64 offset; u64 size; u64 content_hash; }
//   payload chunk payloads, concatenated in INSERTION ORDER (offset/size point
//           into the file from byte 0).
//
//   content_hash = low 8 bytes of SHA-256(payload), little-endian.
//
// Determinism: chunk order == insertion order; the TOC and payload offsets are
// a pure function of the inserted bytes, so identical inputs => byte-identical
// files. AddChunk returns the PER-FOURCC index (the N-th MESH chunk has index N)
// so it lines up with AssetRef::index. NkaWriter is append-only; NkaFile is a
// read-only view that LoadChunk's the N-th payload of a given fourcc.
//
// The chunk payload byte layouts (MESH / CMSH / HULL / SDF0 / SAMP / TEXB) are
// the SELF-WRITTEN, fixed POD encodings defined in nka.cpp's free helpers
// (Encode*/Decode*); HULL reuses the existing .nukacvx SerializeResult layout.
//
// HOST-ONLY file format code: no CUDA, no third-party deps.
// ---------------------------------------------------------------------------

#include "import/cooker/convex_piece.hpp"
#include "import/cooker/sparse_sdf_storage.hpp"
#include "scene/asset/asset_ref.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nuka::scene {

// FourCC chunk tags (packed big-endian via MakeFourCC so the string round-trips
// through AssetRef). These are the chunk FAMILIES; AssetRef::index selects which
// chunk of a family.
inline uint32_t NkaTagMesh() { return MakeFourCC('M', 'E', 'S', 'H'); }  // visual mesh
inline uint32_t NkaTagCmsh() { return MakeFourCC('C', 'M', 'S', 'H'); }  // collision mesh source
inline uint32_t NkaTagHull() { return MakeFourCC('H', 'U', 'L', 'L'); }  // V-HACD piece set
inline uint32_t NkaTagSdf0() { return MakeFourCC('S', 'D', 'F', '0'); }  // sparse SDF
inline uint32_t NkaTagSamp() { return MakeFourCC('S', 'A', 'M', 'P'); }  // sampling point set
inline uint32_t NkaTagTexb() { return MakeFourCC('T', 'E', 'X', 'B'); }  // raw texture bytes

// One table-of-contents entry: fourcc family tag + file offset/size of the
// payload + low-64 of the SHA-256 content hash.
struct NkaTocEntry {
    uint32_t fourcc       = 0;
    uint64_t offset       = 0;
    uint64_t size         = 0;
    uint64_t content_hash = 0;
};

// -- visual / collision mesh payload (decoded) ------------------------------
// MESH carries positions + (optional) normals + (optional) uvs; CMSH carries
// only positions. Both share this decoded struct; the encoder omits the absent
// streams for CMSH and writes zero-filled normal/uv streams for MESH when those
// streams are absent on a visual mesh (see EncodeMesh / EncodeCollisionMesh).
struct NkaMesh {
    std::vector<float>    positions;  // x,y,z per vertex
    std::vector<float>    normals;    // x,y,z per vertex (empty => none)
    std::vector<float>    uvs;        // u,v per vertex (empty => none)
    std::vector<uint32_t> indices;    // triangle indices
};

// -- payload encode/decode (free functions; pure) ---------------------------
// MESH: u32 vert_count, u32 index_count, then pos(3f)*v, normal(3f)*v, uv(2f)*v,
//       then u32 index*index_count. Absent normals/uvs are written as zeros.
std::vector<uint8_t> EncodeMesh(const NkaMesh& mesh);
NkaMesh              DecodeMesh(const std::vector<uint8_t>& bytes);

// CMSH: u32 vert_count, u32 index_count, then pos(3f)*v, then u32 index*count.
std::vector<uint8_t> EncodeCollisionMesh(const std::vector<float>& positions,
                                         const std::vector<uint32_t>& indices);
void                 DecodeCollisionMesh(const std::vector<uint8_t>& bytes,
                                         std::vector<float>& positions,
                                         std::vector<uint32_t>& indices);

// HULL: reuse the .nukacvx SerializeResult layout (piece set).
std::vector<uint8_t>                       EncodeHull(const import::cooker::ConvexDecompositionResult& hull);
import::cooker::ConvexDecompositionResult  DecodeHull(const std::vector<uint8_t>& bytes);

// SDF0: { f32 origin[3]; f32 voxel_size; u32 dims[3]; u32 cell_count; }
//       then u64 cell_keys[cell_count], f32 cell_values[cell_count],
//       f32 cell_gradients[3*cell_count].
std::vector<uint8_t>             EncodeSdf(const import::cooker::SparseSdfData& sdf);
import::cooker::SparseSdfData    DecodeSdf(const std::vector<uint8_t>& bytes);

// SAMP: u32 count, then f32 xyz * count.
std::vector<uint8_t> EncodeSamples(const std::vector<float>& xyz);
std::vector<float>   DecodeSamples(const std::vector<uint8_t>& bytes);

// TEXB: raw bytes verbatim (the chunk payload IS the texture bytes).

// Low 8 bytes of SHA-256(payload), as a little-endian uint64.
uint64_t NkaContentHash(const std::vector<uint8_t>& payload);

// ---------------------------------------------------------------------------
// NkaWriter - append-only container builder.
// ---------------------------------------------------------------------------
class NkaWriter {
public:
    // Append a chunk; returns the per-fourcc INDEX (the N-th MESH chunk gets N),
    // which is what AssetRef::index addresses. Order is insertion order.
    uint32_t AddChunk(uint32_t fourcc, std::vector<uint8_t> payload);

    // Convenience overload from a byte range.
    uint32_t AddChunk(uint32_t fourcc, const uint8_t* data, size_t size);

    // Serialize the full container into a byte buffer (header + TOC + payloads).
    std::vector<uint8_t> ToBytes() const;

    // Write the container to `path` (creates/overwrites). Throws on I/O error.
    void Write(const std::string& path) const;

    size_t ChunkCount() const { return chunks_.size(); }

private:
    struct Chunk {
        uint32_t             fourcc = 0;
        std::vector<uint8_t> payload;
        uint64_t             content_hash = 0;
    };
    std::vector<Chunk> chunks_;
    // Per-fourcc running count so AddChunk's per-family index is O(1).
    std::unordered_map<uint32_t, uint32_t> fourcc_counts_;
};

// ---------------------------------------------------------------------------
// NkaFile - read-only container view.
// ---------------------------------------------------------------------------
class NkaFile {
public:
    using Toc = std::vector<NkaTocEntry>;

    // Open + parse a .nka file. Throws std::runtime_error on a missing file or a
    // malformed/short container.
    static NkaFile Open(const std::string& path);

    // Parse from an in-memory buffer (used by Open and by tests).
    static NkaFile FromBytes(std::vector<uint8_t> bytes);

    const Toc& toc() const { return toc_; }
    uint32_t   version() const { return version_; }

    // Load the payload of the `index`-th chunk of `fourcc`. Throws if absent.
    std::vector<uint8_t> LoadChunk(uint32_t fourcc, uint32_t index) const;

    // Count chunks of a given fourcc family.
    uint32_t CountChunks(uint32_t fourcc) const;

private:
    std::vector<uint8_t> bytes_;
    Toc                  toc_;
    uint32_t             version_ = 0;
};

} // namespace nuka::scene
