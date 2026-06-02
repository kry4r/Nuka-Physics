// ---------------------------------------------------------------------------
// Tests for nuka::import::cooker::DecomposeMesh – concave mesh -> N pieces
// ---------------------------------------------------------------------------
// A concave L-shaped prism (stand-in for a robot finger / bracket collision
// mesh) cannot be represented by a single convex hull without filling its
// re-entrant notch, so V-HACD must produce more than one piece.
// ---------------------------------------------------------------------------

#include "import/cooker/convex_decomposition.hpp"
#include "tests/import/vhacd_test_meshes.hpp"

#include <gtest/gtest.h>

namespace {

using nuka::import::cooker::ConvexDecompositionParams;
using nuka::import::cooker::DecomposeMesh;
using nuka::test::LShapeMesh;

TEST(VhacdRobotFinger, ConcaveLShapeDecomposesToMultiplePieces) {
    const auto mesh = LShapeMesh();
    ConvexDecompositionParams params;
    params.max_pieces = 16;
    const auto result = DecomposeMesh(
        mesh.vertices.data(), static_cast<uint32_t>(mesh.vertices.size() / 3),
        mesh.indices.data(), static_cast<uint32_t>(mesh.indices.size() / 3),
        params);

    ASSERT_TRUE(result.succeeded) << result.error_message;
    // A concave L decomposes into a small number of convex pieces. Expect a
    // reasonable range (the spec's "1-3 / 5-15"-style sanity band): at least 2
    // (single hull would fill the notch) and not pathologically many.
    EXPECT_GE(result.pieces.size(), 2u);
    EXPECT_LE(result.pieces.size(), 16u);

    // Every piece must carry real geometry.
    for (const auto& piece : result.pieces) {
        EXPECT_GE(piece.vertices.size(), 4u * 3u);  // a hull needs >= 4 verts
        EXPECT_EQ(piece.vertices.size() % 3u, 0u);
        EXPECT_GT(piece.indices.size(), 0u);
        EXPECT_EQ(piece.indices.size() % 3u, 0u);
        EXPECT_GT(piece.volume, 0.0f);
    }
}

TEST(VhacdRobotFinger, MaxPiecesBoundIsRespected) {
    const auto mesh = LShapeMesh();
    ConvexDecompositionParams params;
    params.max_pieces = 2;
    const auto result = DecomposeMesh(
        mesh.vertices.data(), static_cast<uint32_t>(mesh.vertices.size() / 3),
        mesh.indices.data(), static_cast<uint32_t>(mesh.indices.size() / 3),
        params);

    ASSERT_TRUE(result.succeeded) << result.error_message;
    EXPECT_LE(result.pieces.size(), 2u);
}

} // namespace
