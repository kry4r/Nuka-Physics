// ---------------------------------------------------------------------------
// Tests for V-HACD determinism + the SHA-256 content-hash cache
// ---------------------------------------------------------------------------
// Determinism: N runs of the PURE DecomposeMesh() on the same mesh + params
// must be byte-identical (V-HACD single-threaded / fixed params + canonicalized
// output). This drives DecomposeMesh() directly so the cache cannot mask it.
//
// Cache: a separate memoizing wrapper (DecomposeMeshCached) returns the same
// pieces on the 2nd call and reports was_hit. We exercise both the in-process
// memo and the on-disk cache dir.
// ---------------------------------------------------------------------------

#include "import/cooker/convex_decomposition.hpp"
#include "tests/import/vhacd_test_meshes.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using nuka::import::cooker::ClearDecompositionMemo;
using nuka::import::cooker::ComputeCacheKey;
using nuka::import::cooker::ConvexDecompositionParams;
using nuka::import::cooker::ConvexDecompositionResult;
using nuka::import::cooker::DecomposeMesh;
using nuka::import::cooker::DecomposeMeshCached;
using nuka::test::LShapeMesh;

// Serialize a result to a byte string for exact comparison.
std::string ResultBytes(const ConvexDecompositionResult& r) {
    std::string out;
    auto append = [&out](const void* p, size_t n) {
        out.append(static_cast<const char*>(p), n);
    };
    const uint32_t count = static_cast<uint32_t>(r.pieces.size());
    append(&count, sizeof(count));
    for (const auto& piece : r.pieces) {
        const uint32_t vc = static_cast<uint32_t>(piece.vertices.size());
        const uint32_t ic = static_cast<uint32_t>(piece.indices.size());
        append(&vc, sizeof(vc));
        append(&ic, sizeof(ic));
        append(&piece.volume, sizeof(piece.volume));
        append(piece.aabb_min, sizeof(piece.aabb_min));
        append(piece.aabb_max, sizeof(piece.aabb_max));
        if (vc) append(piece.vertices.data(), vc * sizeof(float));
        if (ic) append(piece.indices.data(), ic * sizeof(uint32_t));
    }
    return out;
}

TEST(VhacdDeterminism, RepeatedRunsAreByteIdentical) {
    const auto mesh = LShapeMesh();
    ConvexDecompositionParams params;
    params.max_pieces = 16;
    const uint32_t vcount = static_cast<uint32_t>(mesh.vertices.size() / 3);
    const uint32_t tcount = static_cast<uint32_t>(mesh.indices.size() / 3);

    const auto first = DecomposeMesh(mesh.vertices.data(), vcount,
                                     mesh.indices.data(), tcount, params);
    ASSERT_TRUE(first.succeeded) << first.error_message;
    const std::string golden = ResultBytes(first);

    constexpr int kRuns = 10;
    for (int i = 0; i < kRuns; ++i) {
        const auto r = DecomposeMesh(mesh.vertices.data(), vcount,
                                     mesh.indices.data(), tcount, params);
        ASSERT_TRUE(r.succeeded) << "run " << i << ": " << r.error_message;
        EXPECT_EQ(r.pieces.size(), first.pieces.size()) << "run " << i;
        EXPECT_EQ(ResultBytes(r), golden) << "run " << i << " differs byte-wise";
    }
}

TEST(VhacdDeterminism, CacheKeyIsStableAndParamSensitive) {
    const auto mesh = LShapeMesh();
    const uint32_t vcount = static_cast<uint32_t>(mesh.vertices.size() / 3);
    const uint32_t tcount = static_cast<uint32_t>(mesh.indices.size() / 3);

    ConvexDecompositionParams a;
    ConvexDecompositionParams b = a;
    b.max_pieces = a.max_pieces + 1;

    const std::string ka1 =
        ComputeCacheKey(mesh.vertices.data(), vcount, mesh.indices.data(), tcount, a);
    const std::string ka2 =
        ComputeCacheKey(mesh.vertices.data(), vcount, mesh.indices.data(), tcount, a);
    const std::string kb =
        ComputeCacheKey(mesh.vertices.data(), vcount, mesh.indices.data(), tcount, b);

    EXPECT_EQ(ka1.size(), 64u);          // SHA-256 hex
    EXPECT_EQ(ka1, ka2);                 // stable for identical input
    EXPECT_NE(ka1, kb);                  // sensitive to params
}

TEST(VhacdDeterminism, SecondCachedCallIsServedFromCache) {
    ClearDecompositionMemo();
    const auto mesh = LShapeMesh();
    ConvexDecompositionParams params;
    params.max_pieces = 16;
    const uint32_t vcount = static_cast<uint32_t>(mesh.vertices.size() / 3);
    const uint32_t tcount = static_cast<uint32_t>(mesh.indices.size() / 3);

    // On-disk cache dir, fresh.
    const auto dir = std::filesystem::temp_directory_path() /
                     ("nuka_vhacd_cache_" +
                      std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    bool hit1 = true;
    const auto r1 = DecomposeMeshCached(mesh.vertices.data(), vcount,
                                        mesh.indices.data(), tcount, params,
                                        dir.string(), &hit1);
    ASSERT_TRUE(r1.succeeded) << r1.error_message;
    EXPECT_FALSE(hit1);  // cold: not a hit

    bool hit2 = false;
    const auto r2 = DecomposeMeshCached(mesh.vertices.data(), vcount,
                                        mesh.indices.data(), tcount, params,
                                        dir.string(), &hit2);
    ASSERT_TRUE(r2.succeeded) << r2.error_message;
    EXPECT_TRUE(hit2);  // warm: served from the in-process memo
    EXPECT_EQ(ResultBytes(r1), ResultBytes(r2));

    // Drop the in-process memo so the next call must hit the ON-DISK cache.
    ClearDecompositionMemo();
    bool hit3 = false;
    const auto r3 = DecomposeMeshCached(mesh.vertices.data(), vcount,
                                        mesh.indices.data(), tcount, params,
                                        dir.string(), &hit3);
    ASSERT_TRUE(r3.succeeded) << r3.error_message;
    EXPECT_TRUE(hit3);  // served from disk
    EXPECT_EQ(ResultBytes(r1), ResultBytes(r3));

    std::filesystem::remove_all(dir, ec);
}

} // namespace
