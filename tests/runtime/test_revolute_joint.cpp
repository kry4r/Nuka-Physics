// ---------------------------------------------------------------------------
// Tests: Revolute joint constraint building
// ---------------------------------------------------------------------------

#include "runtime/articulation/joint_constraints.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <gtest/gtest.h>
#include <cmath>

TEST(RevoluteJoint, ConstraintBlockHasCorrectRowCount) {
    auto block = nuka::runtime::articulation::BuildRevoluteConstraint(
        0, 1, {0, 0, 1},
        nuka::math::Transform::Identity(),
        nuka::math::Transform::Identity(),
        -3.14f, 3.14f
    );
    EXPECT_EQ(block.row_count, 5u); // 3 translational + 2 rotational constraints
}

TEST(RevoluteJoint, ConstraintTypeIsJoint) {
    auto block = nuka::runtime::articulation::BuildRevoluteConstraint(
        0, 1, {0, 0, 1},
        nuka::math::Transform::Identity(),
        nuka::math::Transform::Identity(),
        -1.57f, 1.57f
    );
    EXPECT_EQ(block.type, nuka::constraint::ConstraintType::Joint);
}

TEST(RevoluteJoint, AxisDirectionReflectedInRotationalRows) {
    // Z-axis revolute
    auto block_z = nuka::runtime::articulation::BuildRevoluteConstraint(
        0, 1, {0, 0, 1},
        nuka::math::Transform::Identity(),
        nuka::math::Transform::Identity(),
        -3.14f, 3.14f
    );

    // Rotational rows (3 and 4) should be perpendicular to Z axis
    // Their angular Jacobians should have zero Z component (they constrain perp directions)
    for (uint32_t r = 3; r < 5; ++r) {
        const auto& ja = block_z.jacobian_angular_a[r];
        // The constrained directions are perpendicular to Z,
        // so their dot product with Z should be ~0
        float dot_with_axis = ja.Dot(nuka::math::Vec3{0, 0, 1});
        EXPECT_NEAR(dot_with_axis, 0.0f, 1e-5f);
    }
}

TEST(RevoluteJoint, XAxisRevoluteConstrainsYZRotation) {
    auto block = nuka::runtime::articulation::BuildRevoluteConstraint(
        2, 3, {1, 0, 0},
        nuka::math::Transform::Identity(),
        nuka::math::Transform::Identity(),
        -1.0f, 1.0f
    );

    EXPECT_EQ(block.row_count, 5u);
    EXPECT_EQ(block.body_a, 2u);
    EXPECT_EQ(block.body_b, 3u);

    // Rotational constraint rows should be perpendicular to X axis
    for (uint32_t r = 3; r < 5; ++r) {
        float dot = block.jacobian_angular_a[r].Dot(nuka::math::Vec3{1, 0, 0});
        EXPECT_NEAR(dot, 0.0f, 1e-5f);
    }
}

TEST(RevoluteJoint, LimitRowsHaveFiniteBounds) {
    auto block = nuka::runtime::articulation::BuildRevoluteConstraint(
        0, 1, {0, 0, 1},
        nuka::math::Transform::Identity(),
        nuka::math::Transform::Identity(),
        -1.57f, 1.57f
    );

    for (uint32_t r = 0; r < block.row_count; ++r) {
        EXPECT_LT(block.lower_limit[r], block.upper_limit[r]);
    }
}
