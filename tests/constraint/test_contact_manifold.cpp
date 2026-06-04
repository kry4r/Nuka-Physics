// ---------------------------------------------------------------------------
// Tests for nuka::constraint::ContactManifold
// ---------------------------------------------------------------------------

#include "constraint/contact_manifold.hpp"

#include <gtest/gtest.h>

using namespace nuka::constraint;

// ---------------------------------------------------------------------------
// Adding points to manifold
// ---------------------------------------------------------------------------

TEST(ContactManifold, AddPoints) {
    ContactManifold m;
    m.a.handle = 0;
    m.b.handle = 1;

    ContactPoint pt;
    pt.position    = {1.0f, 0.0f, 0.0f};
    pt.normal      = {0.0f, 1.0f, 0.0f};
    pt.penetration = 0.01f;

    m.AddPoint(pt);
    EXPECT_EQ(m.point_count, 1u);

    m.AddPoint(pt);
    EXPECT_EQ(m.point_count, 2u);
}

// ---------------------------------------------------------------------------
// Max point clamping
// ---------------------------------------------------------------------------

TEST(ContactManifold, MaxPointClamping) {
    ContactManifold m;
    ContactPoint pt;

    for (int i = 0; i < 10; ++i) {
        m.AddPoint(pt);
    }

    EXPECT_EQ(m.point_count, ContactManifold::kMaxPoints);
}

// ---------------------------------------------------------------------------
// Clear resets count
// ---------------------------------------------------------------------------

TEST(ContactManifold, ClearResetsCount) {
    ContactManifold m;
    ContactPoint pt;
    m.AddPoint(pt);
    m.AddPoint(pt);
    EXPECT_EQ(m.point_count, 2u);

    m.Clear();
    EXPECT_EQ(m.point_count, 0u);
}

TEST(ContactManifold, StoresMaterialResponseParameters) {
    ContactManifold m;
    m.friction = 0.7f;
    m.restitution = 0.35f;

    EXPECT_FLOAT_EQ(m.friction, 0.7f);
    EXPECT_FLOAT_EQ(m.restitution, 0.35f);
}
