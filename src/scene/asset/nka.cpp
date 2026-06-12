// ---------------------------------------------------------------------------
// nuka::scene - .nka binary asset container implementation (M2c).
// ---------------------------------------------------------------------------
// HOST-ONLY file-format code. Pure, deterministic, dependency-free (only the
// vendored SHA-256 and the cooker POD structs). All multi-byte values are
// written in the machine's native byte order; the format is documented as
// little-endian and the targets are little-endian, matching every other cooked
// blob in this repo (.nukacvx etc.).
// ---------------------------------------------------------------------------

#include "scene/asset/nka.hpp"

#include "sha256.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace nuka::scene {

namespace {

// -- little-endian POD append/read over a byte vector -----------------------
template <typename T>
void AppendPod(std::vector<uint8_t>& out, const T& value) {
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}

void AppendBytes(std::vector<uint8_t>& out, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    out.insert(out.end(), p, p + n);
}

template <typename T>
T ReadPod(const std::vector<uint8_t>& in, size_t& cursor) {
    if (cursor + sizeof(T) > in.size()) {
        throw std::runtime_error("nka: truncated payload (POD read out of range)");
    }
    T value;
    std::memcpy(&value, in.data() + cursor, sizeof(T));
    cursor += sizeof(T);
    return value;
}

void ReadArray(const std::vector<uint8_t>& in, size_t& cursor, void* dst, size_t n) {
    if (cursor + n > in.size()) {
        throw std::runtime_error("nka: truncated payload (array read out of range)");
    }
    if (n) {
        std::memcpy(dst, in.data() + cursor, n);
    }
    cursor += n;
}

}  // namespace

// ---------------------------------------------------------------------------
// content hash
// ---------------------------------------------------------------------------
uint64_t NkaContentHash(const std::vector<uint8_t>& payload) {
    nuka::sha256::Hasher hasher;
    hasher.Update(payload.data(), payload.size());
    const std::array<uint8_t, 32> digest = hasher.Final();
    // Low 8 bytes as a little-endian uint64 (digest[0] is the least-significant
    // byte of the result), so the value is reproducible regardless of host
    // endianness.
    uint64_t h = 0;
    for (int i = 0; i < 8; ++i) {
        h |= static_cast<uint64_t>(digest[i]) << (8 * i);
    }
    return h;
}

// ---------------------------------------------------------------------------
// MESH
// ---------------------------------------------------------------------------
std::vector<uint8_t> EncodeMesh(const NkaMesh& mesh) {
    const auto vert_count = static_cast<uint32_t>(mesh.positions.size() / 3);
    const auto index_count = static_cast<uint32_t>(mesh.indices.size());

    std::vector<uint8_t> out;
    AppendPod(out, vert_count);
    AppendPod(out, index_count);

    // positions (3f / vert)
    AppendBytes(out, mesh.positions.data(), mesh.positions.size() * sizeof(float));

    // normals (3f / vert) -- write zeros when absent so the stream count is fixed
    if (!mesh.normals.empty()) {
        AppendBytes(out, mesh.normals.data(), mesh.normals.size() * sizeof(float));
    } else {
        const std::vector<float> zeros(static_cast<size_t>(vert_count) * 3u, 0.0f);
        AppendBytes(out, zeros.data(), zeros.size() * sizeof(float));
    }

    // uvs (2f / vert) -- write zeros when absent
    if (!mesh.uvs.empty()) {
        AppendBytes(out, mesh.uvs.data(), mesh.uvs.size() * sizeof(float));
    } else {
        const std::vector<float> zeros(static_cast<size_t>(vert_count) * 2u, 0.0f);
        AppendBytes(out, zeros.data(), zeros.size() * sizeof(float));
    }

    AppendBytes(out, mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t));
    return out;
}

NkaMesh DecodeMesh(const std::vector<uint8_t>& bytes) {
    size_t cursor = 0;
    const auto vert_count = ReadPod<uint32_t>(bytes, cursor);
    const auto index_count = ReadPod<uint32_t>(bytes, cursor);

    NkaMesh mesh;
    mesh.positions.resize(static_cast<size_t>(vert_count) * 3u);
    ReadArray(bytes, cursor, mesh.positions.data(), mesh.positions.size() * sizeof(float));

    mesh.normals.resize(static_cast<size_t>(vert_count) * 3u);
    ReadArray(bytes, cursor, mesh.normals.data(), mesh.normals.size() * sizeof(float));

    mesh.uvs.resize(static_cast<size_t>(vert_count) * 2u);
    ReadArray(bytes, cursor, mesh.uvs.data(), mesh.uvs.size() * sizeof(float));

    mesh.indices.resize(index_count);
    ReadArray(bytes, cursor, mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t));
    return mesh;
}

// ---------------------------------------------------------------------------
// CMSH
// ---------------------------------------------------------------------------
std::vector<uint8_t> EncodeCollisionMesh(const std::vector<float>& positions,
                                         const std::vector<uint32_t>& indices) {
    const auto vert_count = static_cast<uint32_t>(positions.size() / 3);
    const auto index_count = static_cast<uint32_t>(indices.size());

    std::vector<uint8_t> out;
    AppendPod(out, vert_count);
    AppendPod(out, index_count);
    AppendBytes(out, positions.data(), positions.size() * sizeof(float));
    AppendBytes(out, indices.data(), indices.size() * sizeof(uint32_t));
    return out;
}

void DecodeCollisionMesh(const std::vector<uint8_t>& bytes,
                         std::vector<float>& positions,
                         std::vector<uint32_t>& indices) {
    size_t cursor = 0;
    const auto vert_count = ReadPod<uint32_t>(bytes, cursor);
    const auto index_count = ReadPod<uint32_t>(bytes, cursor);

    positions.resize(static_cast<size_t>(vert_count) * 3u);
    ReadArray(bytes, cursor, positions.data(), positions.size() * sizeof(float));
    indices.resize(index_count);
    ReadArray(bytes, cursor, indices.data(), indices.size() * sizeof(uint32_t));
}

// ---------------------------------------------------------------------------
// HULL (the .nukacvx SerializeResult layout)
// ---------------------------------------------------------------------------
//   u32 piece_count;
//   per piece: u32 vbytes (== vertices.size()), u32 ibytes (== indices.size()),
//              f32 volume, f32 aabb_min[3], f32 aabb_max[3],
//              f32 vertices[vbytes], u32 indices[ibytes].
// (vbytes/ibytes count ELEMENTS, not bytes, matching the original layout.)
std::vector<uint8_t> EncodeHull(const import::cooker::ConvexDecompositionResult& hull) {
    std::vector<uint8_t> out;
    const auto piece_count = static_cast<uint32_t>(hull.pieces.size());
    AppendPod(out, piece_count);
    for (const auto& piece : hull.pieces) {
        const auto vbytes = static_cast<uint32_t>(piece.vertices.size());
        const auto ibytes = static_cast<uint32_t>(piece.indices.size());
        AppendPod(out, vbytes);
        AppendPod(out, ibytes);
        AppendPod(out, piece.volume);
        for (int i = 0; i < 3; ++i) AppendPod(out, piece.aabb_min[i]);
        for (int i = 0; i < 3; ++i) AppendPod(out, piece.aabb_max[i]);
        AppendBytes(out, piece.vertices.data(), vbytes * sizeof(float));
        AppendBytes(out, piece.indices.data(), ibytes * sizeof(uint32_t));
    }
    return out;
}

import::cooker::ConvexDecompositionResult DecodeHull(const std::vector<uint8_t>& bytes) {
    import::cooker::ConvexDecompositionResult result;
    size_t cursor = 0;
    const auto piece_count = ReadPod<uint32_t>(bytes, cursor);
    result.pieces.reserve(piece_count);
    for (uint32_t p = 0; p < piece_count; ++p) {
        import::cooker::ConvexPiece piece;
        const auto vbytes = ReadPod<uint32_t>(bytes, cursor);
        const auto ibytes = ReadPod<uint32_t>(bytes, cursor);
        piece.volume = ReadPod<float>(bytes, cursor);
        for (int i = 0; i < 3; ++i) piece.aabb_min[i] = ReadPod<float>(bytes, cursor);
        for (int i = 0; i < 3; ++i) piece.aabb_max[i] = ReadPod<float>(bytes, cursor);
        piece.vertices.resize(vbytes);
        ReadArray(bytes, cursor, piece.vertices.data(), vbytes * sizeof(float));
        piece.indices.resize(ibytes);
        ReadArray(bytes, cursor, piece.indices.data(), ibytes * sizeof(uint32_t));
        result.pieces.push_back(std::move(piece));
    }
    result.succeeded = true;
    return result;
}

// ---------------------------------------------------------------------------
// SDF0
// ---------------------------------------------------------------------------
//   f32 origin[3]; f32 voxel_size; u32 dims[3]; u32 cell_count;
//   u64 cell_keys[cell_count];
//   f32 cell_values[cell_count];
//   f32 cell_gradients[3*cell_count];
std::vector<uint8_t> EncodeSdf(const import::cooker::SparseSdfData& sdf) {
    std::vector<uint8_t> out;
    for (int i = 0; i < 3; ++i) AppendPod(out, sdf.origin[i]);
    AppendPod(out, sdf.voxel_size);
    for (int i = 0; i < 3; ++i) AppendPod(out, sdf.dims[i]);
    const auto cell_count = sdf.CellCount();
    AppendPod(out, cell_count);

    AppendBytes(out, sdf.cell_keys.data(), sdf.cell_keys.size() * sizeof(uint64_t));
    AppendBytes(out, sdf.cell_values.data(), sdf.cell_values.size() * sizeof(float));
    AppendBytes(out, sdf.cell_gradients.data(), sdf.cell_gradients.size() * sizeof(float));
    return out;
}

import::cooker::SparseSdfData DecodeSdf(const std::vector<uint8_t>& bytes) {
    import::cooker::SparseSdfData sdf;
    size_t cursor = 0;
    for (int i = 0; i < 3; ++i) sdf.origin[i] = ReadPod<float>(bytes, cursor);
    sdf.voxel_size = ReadPod<float>(bytes, cursor);
    for (int i = 0; i < 3; ++i) sdf.dims[i] = ReadPod<uint32_t>(bytes, cursor);
    const auto cell_count = ReadPod<uint32_t>(bytes, cursor);

    sdf.cell_keys.resize(cell_count);
    ReadArray(bytes, cursor, sdf.cell_keys.data(), sdf.cell_keys.size() * sizeof(uint64_t));
    sdf.cell_values.resize(cell_count);
    ReadArray(bytes, cursor, sdf.cell_values.data(), sdf.cell_values.size() * sizeof(float));
    sdf.cell_gradients.resize(static_cast<size_t>(cell_count) * 3u);
    ReadArray(bytes, cursor, sdf.cell_gradients.data(),
              sdf.cell_gradients.size() * sizeof(float));
    return sdf;
}

// ---------------------------------------------------------------------------
// SAMP
// ---------------------------------------------------------------------------
std::vector<uint8_t> EncodeSamples(const std::vector<float>& xyz) {
    std::vector<uint8_t> out;
    const auto count = static_cast<uint32_t>(xyz.size() / 3);
    AppendPod(out, count);
    AppendBytes(out, xyz.data(), xyz.size() * sizeof(float));
    return out;
}

std::vector<float> DecodeSamples(const std::vector<uint8_t>& bytes) {
    size_t cursor = 0;
    const auto count = ReadPod<uint32_t>(bytes, cursor);
    std::vector<float> xyz(static_cast<size_t>(count) * 3u);
    ReadArray(bytes, cursor, xyz.data(), xyz.size() * sizeof(float));
    return xyz;
}

// ---------------------------------------------------------------------------
// NkaWriter
// ---------------------------------------------------------------------------
uint32_t NkaWriter::AddChunk(uint32_t fourcc, std::vector<uint8_t> payload) {
    // Per-fourcc index == how many of this family precede us.
    uint32_t index = 0;
    for (const auto& c : chunks_) {
        if (c.fourcc == fourcc) ++index;
    }
    Chunk chunk;
    chunk.fourcc = fourcc;
    chunk.content_hash = NkaContentHash(payload);
    chunk.payload = std::move(payload);
    chunks_.push_back(std::move(chunk));
    return index;
}

uint32_t NkaWriter::AddChunk(uint32_t fourcc, const uint8_t* data, size_t size) {
    return AddChunk(fourcc, std::vector<uint8_t>(data, data + size));
}

std::vector<uint8_t> NkaWriter::ToBytes() const {
    constexpr size_t kHeaderSize = 4 /*magic*/ + 4 /*version*/ + 4 /*chunk_count*/;
    constexpr size_t kTocEntrySize = 4 /*fourcc*/ + 8 /*offset*/ + 8 /*size*/ + 8 /*hash*/;

    const auto chunk_count = static_cast<uint32_t>(chunks_.size());
    const size_t toc_size = kTocEntrySize * chunks_.size();
    const uint64_t payload_base = kHeaderSize + toc_size;

    std::vector<uint8_t> out;
    // header
    const char magic[4] = {'N', 'K', 'A', '1'};
    AppendBytes(out, magic, 4);
    const uint32_t version = 1;
    AppendPod(out, version);
    AppendPod(out, chunk_count);

    // TOC (payload offsets accumulate in insertion order)
    uint64_t offset = payload_base;
    for (const auto& c : chunks_) {
        AppendPod(out, c.fourcc);
        AppendPod(out, offset);
        const auto size = static_cast<uint64_t>(c.payload.size());
        AppendPod(out, size);
        AppendPod(out, c.content_hash);
        offset += c.payload.size();
    }

    // payloads
    for (const auto& c : chunks_) {
        AppendBytes(out, c.payload.data(), c.payload.size());
    }
    return out;
}

void NkaWriter::Write(const std::string& path) const {
    const std::vector<uint8_t> bytes = ToBytes();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("nka: cannot open for write: " + path);
    }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error("nka: write failed: " + path);
    }
}

// ---------------------------------------------------------------------------
// NkaFile
// ---------------------------------------------------------------------------
NkaFile NkaFile::Open(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("nka: cannot open for read: " + path);
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
    return FromBytes(std::move(bytes));
}

NkaFile NkaFile::FromBytes(std::vector<uint8_t> bytes) {
    NkaFile file;
    file.bytes_ = std::move(bytes);

    size_t cursor = 0;
    char magic[4];
    ReadArray(file.bytes_, cursor, magic, 4);
    if (magic[0] != 'N' || magic[1] != 'K' || magic[2] != 'A' || magic[3] != '1') {
        throw std::runtime_error("nka: bad magic (not a NKA1 container)");
    }
    file.version_ = ReadPod<uint32_t>(file.bytes_, cursor);
    if (file.version_ != 1) {
        throw std::runtime_error("nka: unsupported version");
    }
    const auto chunk_count = ReadPod<uint32_t>(file.bytes_, cursor);
    file.toc_.reserve(chunk_count);
    for (uint32_t i = 0; i < chunk_count; ++i) {
        NkaTocEntry e;
        e.fourcc = ReadPod<uint32_t>(file.bytes_, cursor);
        e.offset = ReadPod<uint64_t>(file.bytes_, cursor);
        e.size = ReadPod<uint64_t>(file.bytes_, cursor);
        e.content_hash = ReadPod<uint64_t>(file.bytes_, cursor);
        file.toc_.push_back(e);
    }
    return file;
}

uint32_t NkaFile::CountChunks(uint32_t fourcc) const {
    uint32_t n = 0;
    for (const auto& e : toc_) {
        if (e.fourcc == fourcc) ++n;
    }
    return n;
}

std::vector<uint8_t> NkaFile::LoadChunk(uint32_t fourcc, uint32_t index) const {
    uint32_t seen = 0;
    for (const auto& e : toc_) {
        if (e.fourcc != fourcc) continue;
        if (seen == index) {
            if (e.offset + e.size > bytes_.size()) {
                throw std::runtime_error("nka: chunk payload out of range");
            }
            return std::vector<uint8_t>(bytes_.begin() + static_cast<ptrdiff_t>(e.offset),
                                        bytes_.begin() +
                                            static_cast<ptrdiff_t>(e.offset + e.size));
        }
        ++seen;
    }
    throw std::runtime_error("nka: chunk not found (fourcc/index)");
}

} // namespace nuka::scene
