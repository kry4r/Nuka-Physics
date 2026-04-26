// ---------------------------------------------------------------------------
// Tests for nuka::collision ray query utilities
// ---------------------------------------------------------------------------

#include "collision/query.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace nuka;
using namespace nuka::collision;

// ---------------------------------------------------------------------------
// RaycastPlane -- ray hits ground plane
// ---------------------------------------------------------------------------

TEST(Raycast, HitsGroundPlane) {
    // Ray from y=10 pointing straight down.
    auto hit = RaycastPlane({0.0f, 10.0f, 0.0f}, {0.0f, -1.0f, 0.0f});
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(hit->distance, 10.0f, 1e-5f);
    EXPECT_NEAR(hit->normal.y, 1.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// RaycastPlane -- ray parallel to ground plane misses
// ---------------------------------------------------------------------------

TEST(Raycast, MissesGroundPlaneParallel) {
    // Horizontal ray at y=5 -- parallel to the plane.
    auto hit = RaycastPlane({0.0f, 5.0f, 0.0f}, {1.0f, 0.0f, 0.0f});
    EXPECT_FALSE(hit.has_value());
}

// ---------------------------------------------------------------------------
// RaycastSphere -- ray hits sphere
// ---------------------------------------------------------------------------

TEST(Raycast, HitsSphere) {
    // Ray from origin along +X, sphere at (5,0,0) radius 1.
    auto hit = RaycastSphere({0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
                              {5.0f, 0.0f, 0.0f}, 1.0f);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(hit->distance, 4.0f, 1e-5f);
    EXPECT_NEAR(hit->normal.x, -1.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// RaycastSphere -- ray misses sphere
// ---------------------------------------------------------------------------

TEST(Raycast, MissesSphere) {
    // Ray from origin along +X, sphere at (5,5,0) radius 1 -- too far off axis.
    auto hit = RaycastSphere({0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
                              {5.0f, 5.0f, 0.0f}, 1.0f);
    EXPECT_FALSE(hit.has_value());
}

// ---------------------------------------------------------------------------
// RaycastAABB -- ray hits box
// ---------------------------------------------------------------------------

TEST(Raycast, HitsAABB) {
    AABB box;
    box.min = {2.0f, -1.0f, -1.0f};
    box.max = {4.0f,  1.0f,  1.0f};

    auto hit = RaycastAABB({0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, box);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(hit->distance, 2.0f, 1e-5f);
    // Entry normal should point in -X direction (face facing the ray).
    EXPECT_NEAR(hit->normal.x, -1.0f, 1e-5f);
}
