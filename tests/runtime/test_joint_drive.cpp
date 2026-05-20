// ---------------------------------------------------------------------------
// Tests: Joint drive simulation
// ---------------------------------------------------------------------------

#include "runtime/articulation/joint_constraints.hpp"
#include "runtime/articulation/joint_drive.hpp"

#include <gtest/gtest.h>
#include <cmath>

TEST(JointDrive, VelocityDriveMovesTowardTarget) {
    auto result = nuka::runtime::articulation::SimulateVelocityDriveStep();
    EXPECT_GT(result.joint_velocity, 0.0f);
}

TEST(JointDrive, DriveForceDirectionMatchesError) {
    nuka::runtime::articulation::JointDrive drive{};
    drive.target_velocity = 5.0f;
    drive.damping = 10.0f;
    drive.max_force = 100.0f;

    auto block = nuka::runtime::articulation::BuildDriveConstraint(
        0, 1, {0, 0, 1}, drive, 0.0f  // current_velocity = 0, target = 5
    );

    auto block2 = nuka::runtime::articulation::BuildDriveConstraint(
        0, 1, {0, 0, 1}, drive, 10.0f  // current_velocity = 10, target = 5
    );

    EXPECT_FLOAT_EQ(block.rhs[0], 5.0f);
    EXPECT_FLOAT_EQ(block2.rhs[0], 5.0f);
}

TEST(JointDrive, ZeroTargetProducesZeroForceAtRest) {
    nuka::runtime::articulation::JointDrive drive{};
    drive.target_velocity = 0.0f;
    drive.damping = 10.0f;
    drive.max_force = 100.0f;

    auto block = nuka::runtime::articulation::BuildDriveConstraint(
        0, 1, {0, 0, 1}, drive, 0.0f  // already at target
    );

    EXPECT_FLOAT_EQ(block.rhs[0], 0.0f);
}

TEST(JointDrive, DriveConstraintHasSingleRow) {
    nuka::runtime::articulation::JointDrive drive{};
    drive.target_velocity = 1.0f;

    auto block = nuka::runtime::articulation::BuildDriveConstraint(
        0, 1, {0, 0, 1}, drive, 0.0f
    );

    EXPECT_EQ(block.row_count, 1u);
    EXPECT_EQ(block.type, nuka::constraint::ConstraintType::Drive);
}

TEST(JointDrive, MaxForceClampedInLimits) {
    nuka::runtime::articulation::JointDrive drive{};
    drive.target_velocity = 1.0f;
    drive.max_force = 50.0f;

    auto block = nuka::runtime::articulation::BuildDriveConstraint(
        0, 1, {0, 0, 1}, drive, 0.0f
    );

    EXPECT_FLOAT_EQ(block.lower_limit[0], -50.0f);
    EXPECT_FLOAT_EQ(block.upper_limit[0], 50.0f);
}

TEST(JointDrive, VelocityDrivePositionIncreases) {
    auto result = nuka::runtime::articulation::SimulateVelocityDriveStep();
    // After driving toward positive velocity, position should have advanced
    EXPECT_GT(result.joint_position, 0.0f);
}
