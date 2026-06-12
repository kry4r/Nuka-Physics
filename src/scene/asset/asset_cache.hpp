#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::AssetCache - content-addressed cook-artifact cache (M2c).
//
// A thin on-disk content-addressed store for the cook-time artifacts that are
// expensive to (re)produce: V-HACD hull sets and sparse SDFs (and a generic
// byte producer for meshes/textures). Keys reuse the EXISTING content-hash key
// functions (ComputeCacheKey for hulls, ComputeSdfCacheKey for SDFs); the
// payloads reuse the EXISTING binary serializations (the .nukacvx hull layout
// and the .nka SDF0 layout). The cache is purely a memoizer: a cache HIT must
// be bit-identical to the cold-cooked result.
//
// On-disk layout: <root>/<key>.<kind>, e.g.
//   .nuka_cache/<sha256>.nukacvx   (hull set)
//   .nuka_cache/<sha256>.nukasdf   (sparse SDF, SDF0 layout)
//   .nuka_cache/<sha256>.bin       (generic GetOrPut bytes)
//
// HOST-ONLY: no CUDA, no third-party deps.
// ---------------------------------------------------------------------------

#include "import/cooker/convex_decomposition.hpp"
#include "import/cooker/convex_piece.hpp"
#include "import/cooker/sparse_sdf_cooker.hpp"
#include "import/cooker/sparse_sdf_storage.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace nuka::scene {

class AssetCache {
public:
    // Default root ".nuka_cache/" (created on first store). Tests pass a temp
    // dir to stay hermetic.
    explicit AssetCache(std::string root = ".nuka_cache");

    const std::string& Root() const { return root_; }

    // Generic content-addressed byte store. `key` is the (already-computed)
    // content hash; `producer` runs only on a miss. Returns the cached bytes.
    using ByteProducer = std::function<std::vector<uint8_t>()>;
    std::vector<uint8_t> GetOrPut(const std::string& key, const ByteProducer& producer);

    // SHA-256 hex of raw bytes (the mesh/texture key helper).
    static std::string HashBytes(const void* data, size_t size);

    // V-HACD hull set, keyed by ComputeCacheKey(mesh, params). `cook_fn` runs on
    // a miss; the result is serialized with the .nukacvx layout. A HIT returns a
    // result bit-identical to the cold cook.
    using HullCookFn = std::function<import::cooker::ConvexDecompositionResult()>;
    import::cooker::ConvexDecompositionResult GetOrCookHull(
        const float* vertices, uint32_t vertex_count,
        const uint32_t* indices, uint32_t triangle_count,
        const import::cooker::ConvexDecompositionParams& params,
        const HullCookFn& cook_fn);

    // Sparse SDF, keyed by ComputeSdfCacheKey(mesh, bake_params). `cook_fn` runs
    // on a miss; the result is serialized with the .nka SDF0 layout.
    using SdfCookFn = std::function<import::cooker::SparseSdfData()>;
    import::cooker::SparseSdfData GetOrCookSdf(
        const float* vertices, uint32_t vertex_count,
        const uint32_t* indices, uint32_t triangle_count,
        const import::cooker::SparseSdfParams& bake_params,
        const SdfCookFn& cook_fn);

    // M5: the SDF sampling-point set (hull verts + edge midpoints, the .nka SAMP
    // chunk). Keyed by `key` (the caller passes the hull's content hash); the
    // payload is the .nka SAMP layout (EncodeSamples). `produce` runs on a miss
    // and returns the flat xyz pool. A HIT is bit-identical to the cold cook —
    // this is the AssetCache seam that lands SAMP in the .nka when a scene saves.
    using SampleProducer = std::function<std::vector<float>()>;
    std::vector<float> GetOrCookSamples(const std::string& key,
                                        const SampleProducer& produce);

private:
    std::string PathFor(const std::string& key, const char* kind) const;
    static bool ReadFile(const std::string& path, std::vector<uint8_t>& out);
    void WriteFile(const std::string& path, const std::vector<uint8_t>& bytes) const;

    std::string root_;
};

} // namespace nuka::scene
