// ---------------------------------------------------------------------------
// Tests: Revolute joint row building
// ---------------------------------------------------------------------------

#include "runtime/articulation/joint_constraints.hpp"

#include <gtest/gtest.h>

namespace {

nuka::constraint::RowBuffers BuildRows(nuka::math::Vec3 axis) {
    nuka::constraint::RowBuffers rows;
    nuka::runtime::articulation::AppendRevoluteRows(
        &rows,
        0u,
        1u,
        axis,
        nuka::math::Transform::Identity(),
        nuka::math::Transform::Identity(),
        -3.14f,
        3.14f);
    return rows;
}

} // namespace

TEST(RevoluteJoint, RowBuilderHasCorrectRowCount) {
    const auto rows = BuildRows({0.0f, 0.0f, 1.0f});
    EXPECT_EQ(rows.RowCount(), 5u);
}

TEST(RevoluteJoint, RowKindIsJoint) {
    const auto rows = BuildRows({0.0f, 0.0f, 1.0f});
    ASSERT_FALSE(rows.materials.empty());
    EXPECT_EQ(rows.materials[0].kind, nuka::constraint::RowKind::Joint);
}

TEST(RevoluteJoint, AxisDirectionReflectedInRotationalRows) {
    const auto rows = BuildRows({0.0f, 0.0f, 1.0f});
    for (uint32_t row = 3u; row < 5u; ++row) {
        const auto jacobian = rows.JacobianForRowBody(row, 0u);
        const float dot_with_axis =
            jacobian.angular.Dot(nuka::math::Vec3{0.0f, 0.0f, 1.0f});
        EXPECT_NEAR(dot_with_axis, 0.0f, 1.0e-5f);
    }
}

TEST(RevoluteJoint, XAxisRevoluteConstrainsYZRotation) {
    const auto rows = BuildRows({1.0f, 0.0f, 0.0f});

    ASSERT_EQ(rows.RowCount(), 5u);
    EXPECT_EQ(rows.BodiesForRow(0u).body_a, 0u);
    EXPECT_EQ(rows.BodiesForRow(0u).body_b, 1u);

    for (uint32_t row = 3u; row < 5u; ++row) {
        const auto jacobian = rows.JacobianForRowBody(row, 0u);
        const float dot = jacobian.angular.Dot(nuka::math::Vec3{1.0f, 0.0f, 0.0f});
        EXPECT_NEAR(dot, 0.0f, 1.0e-5f);
    }
}

TEST(RevoluteJoint, RowsHaveFiniteBounds) {
    const auto rows = BuildRows({0.0f, 0.0f, 1.0f});
    for (const auto& row : rows.rows) {
        EXPECT_LT(row.lower, row.upper);
    }
}
