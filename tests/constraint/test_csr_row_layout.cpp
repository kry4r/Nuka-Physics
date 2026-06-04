// ---------------------------------------------------------------------------
// Tests for nuka::constraint::RowBuffers and contact row building
// ---------------------------------------------------------------------------

#include "constraint/contact_builder.hpp"

#include <gtest/gtest.h>

using namespace nuka::constraint;

TEST(CsrRowLayout, BuildContactRowsForPointCount) {
    RowBuffers rows;
    const uint32_t appended = BuildContactRowsForPointCount(2u, &rows);

    EXPECT_EQ(appended, 4u);
    ASSERT_EQ(rows.RowCount(), 4u);
    EXPECT_EQ(rows.BodyIndexCount(), 8u);
    EXPECT_EQ(rows.JacobianDataCount(), 8u);
    EXPECT_EQ(rows.rows[0].row_class_id, kMaximalContactRowClassId);
    EXPECT_EQ(rows.materials[0].kind, RowKind::Contact);
}

TEST(CsrRowLayout, BuildContactRowsFromManifolds) {
    ContactManifold manifold;
    manifold.a.handle = 0u;
    manifold.b.handle = 1u;

    ContactPoint point;
    point.position = {0.0f, 0.0f, 0.0f};
    point.normal = {0.0f, 1.0f, 0.0f};
    point.penetration = 0.02f;
    manifold.AddPoint(point);
    manifold.AddPoint(point);

    RowBuffers rows;
    BuildContactRows(std::vector<ContactManifold>{manifold}, &rows);

    ASSERT_EQ(rows.RowCount(), 4u);
    EXPECT_EQ(rows.BodiesForRow(0u).body_a, 0u);
    EXPECT_EQ(rows.BodiesForRow(0u).body_b, 1u);
    EXPECT_EQ(rows.materials[0].normal_row_count, 2u);
    EXPECT_EQ(rows.materials[0].first_friction_row, 2u);
    EXPECT_EQ(rows.materials[0].friction_row_count, 2u);
    EXPECT_TRUE(IsContactNormalRow(rows, 0u));
    EXPECT_TRUE(IsFrictionRow(rows, 2u));
}

TEST(CsrRowLayout, ContactMaterialParametersPropagateFromManifold) {
    ContactManifold manifold;
    manifold.a.handle = 0u;
    manifold.b.handle = 1u;
    manifold.friction = 0.65f;
    manifold.restitution = 0.25f;

    ContactPoint point;
    point.position = {0.0f, 0.0f, 0.0f};
    point.normal = {0.0f, 1.0f, 0.0f};
    point.penetration = 0.075f;
    manifold.AddPoint(point);

    RowBuffers rows;
    BuildContactRows(std::vector<ContactManifold>{manifold}, &rows);

    ASSERT_EQ(rows.RowCount(), 3u);
    EXPECT_FLOAT_EQ(rows.materials[0].friction, 0.65f);
    EXPECT_FLOAT_EQ(rows.materials[0].restitution, 0.25f);
    EXPECT_FLOAT_EQ(rows.materials[0].position_error, 0.075f);
    EXPECT_FLOAT_EQ(rows.rows[0].rhs, 0.0f);
}
