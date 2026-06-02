// ---------------------------------------------------------------------------
// Tests for nuka::import::cooker::DecomposeMesh – simple convex cube
// ---------------------------------------------------------------------------
// A solid cube is already convex, so V-HACD must return exactly one piece
// whose vertices/AABB bound the unit cube.
// ---------------------------------------------------------------------------

#include "import/cooker/convex_decomposition.hpp"
#include "tests/import/vhacd_test_meshes.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

using nuka::import::cooker::ConvexDecompositionParams;
using nuka::import::cooker::DecomposeMesh;
using nuka::test::UnitCubeMesh;

TEST(VhacdSimpleCube, CubeDecomposesToExactlyOnePiece) {
    const auto cube = UnitCubeMesh();
    ConvexDecompositionParams params;
    const auto result = DecomposeMesh(
        cube.vertices.data(), static_cast<uint32_t>(cube.vertices.size() / 3),
        cube.indices.data(), static_cast<uint32_t>(cube.indices.size() / 3),
        params);

    ASSERT_TRUE(result.succeeded) << result.error_message;
    EXPECT_EQ(result.pieces.size(), 1u);
}

TEST(VhacdSimpleCube, CubePieceHasSaneVerticesAndAabb) {
    const auto cube = UnitCubeMesh();  // spans [-0.5, 0.5]^3
    ConvexDecompositionParams params;
    const auto result = DecomposeMesh(
        cube.vertices.data(), static_cast<uint32_t>(cube.vertices.size() / 3),
        cube.indices.data(), static_cast<uint32_t>(cube.indices.size() / 3),
        params);

    ASSERT_TRUE(result.succeeded) << result.error_message;
    ASSERT_EQ(result.pieces.size(), 1u);

    const auto& piece = result.pieces.front();
    // A convex hull of a cube needs at least its 8 corners.
    EXPECT_GE(piece.vertices.size(), 8u * 3u);
    EXPECT_EQ(piece.vertices.size() % 3u, 0u);
    EXPECT_EQ(piece.indices.size() % 3u, 0u);
    EXPECT_GT(piece.indices.size(), 0u);

    // The hull AABB should closely match the source cube (within voxelization
    // tolerance); volume should be a non-trivial fraction of the 1.0 m^3 cube.
    const float tol = 0.1f;
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(piece.aabb_min[i], -0.5f, tol);
        EXPECT_NEAR(piece.aabb_max[i], 0.5f, tol);
    }
    EXPECT_GT(piece.volume, 0.5f);
    EXPECT_LT(piece.volume, 1.5f);

    // Every vertex lies inside the (slightly inflated) source bounds.
    for (size_t v = 0; v + 2 < piece.vertices.size(); v += 3) {
        EXPECT_GE(piece.vertices[v + 0], -0.5f - tol);
        EXPECT_LE(piece.vertices[v + 0], 0.5f + tol);
        EXPECT_GE(piece.vertices[v + 1], -0.5f - tol);
        EXPECT_LE(piece.vertices[v + 1], 0.5f + tol);
        EXPECT_GE(piece.vertices[v + 2], -0.5f - tol);
        EXPECT_LE(piece.vertices[v + 2], 0.5f + tol);
    }
}

TEST(VhacdSimpleCube, RejectsEmptyMesh) {
    ConvexDecompositionParams params;
    const auto result = DecomposeMesh(nullptr, 0, nullptr, 0, params);
    EXPECT_FALSE(result.succeeded);
    EXPECT_FALSE(result.error_message.empty());
}

} // namespace
