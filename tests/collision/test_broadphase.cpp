// ---------------------------------------------------------------------------
// Tests for nuka::collision::DynamicBroadphase
// ---------------------------------------------------------------------------

#include "collision/dynamic_broadphase.hpp"

#include <gtest/gtest.h>

using namespace nuka::collision;

// ---------------------------------------------------------------------------
// Two overlapping boxes produce a pair
// ---------------------------------------------------------------------------

TEST(Broadphase, OverlappingBoxesProducePair) {
    AABB a;
    a.min = {0.0f, 0.0f, 0.0f};
    a.max = {2.0f, 2.0f, 2.0f};

    AABB b;
    b.min = {1.0f, 1.0f, 1.0f};
    b.max = {3.0f, 3.0f, 3.0f};

    DynamicBroadphase bp;
    bp.Update({a, b});

    const auto& pairs = bp.GetPairs();
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].body_a, 0u);
    EXPECT_EQ(pairs[0].body_b, 1u);
}

// ---------------------------------------------------------------------------
// Non-overlapping boxes produce no pair
// ---------------------------------------------------------------------------

TEST(Broadphase, NonOverlappingBoxesNoPair) {
    AABB a;
    a.min = {0.0f, 0.0f, 0.0f};
    a.max = {1.0f, 1.0f, 1.0f};

    AABB b;
    b.min = {5.0f, 5.0f, 5.0f};
    b.max = {6.0f, 6.0f, 6.0f};

    DynamicBroadphase bp;
    bp.Update({a, b});

    EXPECT_TRUE(bp.GetPairs().empty());
}

// ---------------------------------------------------------------------------
// Three-body scenario
// ---------------------------------------------------------------------------

TEST(Broadphase, ThreeBodyScenario) {
    // A and B overlap, B and C overlap, A and C do not.
    AABB a;
    a.min = {0.0f, 0.0f, 0.0f};
    a.max = {2.0f, 2.0f, 2.0f};

    AABB b;
    b.min = {1.0f, 1.0f, 1.0f};
    b.max = {4.0f, 4.0f, 4.0f};

    AABB c;
    c.min = {3.0f, 3.0f, 3.0f};
    c.max = {5.0f, 5.0f, 5.0f};

    DynamicBroadphase bp;
    bp.Update({a, b, c});

    const auto& pairs = bp.GetPairs();
    // A-B and B-C overlap; A-C does not.
    EXPECT_EQ(pairs.size(), 2u);
}
