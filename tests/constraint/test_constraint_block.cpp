// ---------------------------------------------------------------------------
// Tests for nuka::constraint::ConstraintBlock and contact builder
// ---------------------------------------------------------------------------

#include "constraint/contact_builder.hpp"

#include <gtest/gtest.h>

using namespace nuka::constraint;

// ---------------------------------------------------------------------------
// BuildContactBlock(2) creates block with row_count 3
// ---------------------------------------------------------------------------

TEST(ConstraintBlock, BuildContactBlockRowCount) {
    auto block = BuildContactBlock(2);
    EXPECT_EQ(block.row_count, 4u);  // 2 normal + 2 friction tangent rows
}

// ---------------------------------------------------------------------------
// Block type is Contact
// ---------------------------------------------------------------------------

TEST(ConstraintBlock, BlockTypeIsContact) {
    auto block = BuildContactBlock(1);
    EXPECT_EQ(block.type, ConstraintType::Contact);
}

// ---------------------------------------------------------------------------
// BuildContactConstraints from manifolds
// ---------------------------------------------------------------------------

TEST(ConstraintBlock, BuildContactConstraintsFromManifolds) {
    ContactManifold m;
    m.body_a = 0;
    m.body_b = 1;

    ContactPoint pt;
    pt.position    = {0.0f, 0.0f, 0.0f};
    pt.normal      = {0.0f, 1.0f, 0.0f};
    pt.penetration = 0.02f;
    m.AddPoint(pt);
    m.AddPoint(pt);

    std::vector<ContactManifold> manifolds = {m};
    auto blocks = BuildContactConstraints(manifolds);

    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_EQ(blocks[0].body_a, 0u);
    EXPECT_EQ(blocks[0].body_b, 1u);
    EXPECT_EQ(blocks[0].row_count, 4u);  // 2 contacts + 2 friction tangent rows
    EXPECT_EQ(blocks[0].type, ConstraintType::Contact);
}

TEST(ConstraintBlock, ContactMaterialParametersPropagateFromManifold) {
    ContactManifold m;
    m.body_a = 0;
    m.body_b = 1;
    m.friction = 0.65f;
    m.restitution = 0.25f;

    ContactPoint pt;
    pt.position = {0.0f, 0.0f, 0.0f};
    pt.normal = {0.0f, 1.0f, 0.0f};
    pt.penetration = 0.01f;
    m.AddPoint(pt);

    auto blocks = BuildContactConstraints({m});

    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_FLOAT_EQ(blocks[0].friction, 0.65f);
    EXPECT_FLOAT_EQ(blocks[0].restitution, 0.25f);
    EXPECT_EQ(blocks[0].normal_row_count, 1u);
    EXPECT_EQ(blocks[0].friction_row_count, 2u);
    EXPECT_EQ(blocks[0].first_friction_row, 1u);
}
