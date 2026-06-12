// ---------------------------------------------------------------------------
// nuka::scene::AssetCache implementation (M2c).
// ---------------------------------------------------------------------------
// HOST-ONLY file-format code. The hull payload reuses the .nukacvx layout
// (EncodeHull/DecodeHull in nka.cpp, byte-identical to the cooker's original
// SerializeResult) and the SDF payload reuses the .nka SDF0 layout, so a cache
// hit is bit-identical to the cold cook.
// ---------------------------------------------------------------------------

#include "scene/asset/asset_cache.hpp"

#include "scene/asset/nka.hpp"
#include "sha256.h"

#include <filesystem>
#include <fstream>
#include <iterator>

namespace nuka::scene {

AssetCache::AssetCache(std::string root) : root_(std::move(root)) {}

std::string AssetCache::HashBytes(const void* data, size_t size) {
    return nuka::sha256::HashHex(data, size);
}

std::string AssetCache::PathFor(const std::string& key, const char* kind) const {
    return (std::filesystem::path(root_) / (key + "." + kind)).string();
}

bool AssetCache::ReadFile(const std::string& path, std::vector<uint8_t>& out) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    out.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return true;
}

void AssetCache::WriteFile(const std::string& path,
                           const std::vector<uint8_t>& bytes) const {
    std::error_code ec;
    std::filesystem::create_directories(root_, ec);
    // Atomic-ish write: write to a temp sibling then rename, so a concurrent
    // reader never sees a half-written file.
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return;  // best-effort cache; a write failure just disables caching
        }
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        if (!out) {
            return;
        }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
    }
}

std::vector<uint8_t> AssetCache::GetOrPut(const std::string& key,
                                          const ByteProducer& producer) {
    const std::string path = PathFor(key, "bin");
    std::vector<uint8_t> bytes;
    if (ReadFile(path, bytes)) {
        return bytes;
    }
    bytes = producer();
    WriteFile(path, bytes);
    return bytes;
}

import::cooker::ConvexDecompositionResult AssetCache::GetOrCookHull(
    const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count,
    const import::cooker::ConvexDecompositionParams& params,
    const HullCookFn& cook_fn) {
    const std::string key = import::cooker::ComputeCacheKey(
        vertices, vertex_count, indices, triangle_count, params);
    const std::string path = PathFor(key, "nukacvx");

    std::vector<uint8_t> bytes;
    if (ReadFile(path, bytes)) {
        return DecodeHull(bytes);
    }
    import::cooker::ConvexDecompositionResult result = cook_fn();
    if (result.succeeded && !result.pieces.empty()) {
        WriteFile(path, EncodeHull(result));
    }
    return result;
}

import::cooker::SparseSdfData AssetCache::GetOrCookSdf(
    const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count,
    const import::cooker::SparseSdfParams& bake_params,
    const SdfCookFn& cook_fn) {
    const std::string key = import::cooker::ComputeSdfCacheKey(
        vertices, vertex_count, indices, triangle_count, bake_params);
    const std::string path = PathFor(key, "nukasdf");

    std::vector<uint8_t> bytes;
    if (ReadFile(path, bytes)) {
        return DecodeSdf(bytes);
    }
    import::cooker::SparseSdfData sdf = cook_fn();
    if (sdf.CellCount() > 0) {
        WriteFile(path, EncodeSdf(sdf));
    }
    return sdf;
}

} // namespace nuka::scene
