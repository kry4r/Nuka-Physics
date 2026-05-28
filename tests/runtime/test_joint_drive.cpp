// ---------------------------------------------------------------------------
// Tests: Joint drive rows
// ---------------------------------------------------------------------------

#include "runtime/articulation/joint_constraints.hpp"
#include "runtime/articulation/joint_drive.hpp"

#include <gtest/gtest.h>

namespace {

nuka::constraint::RowBuffers BuildDriveRows(float target_velocity,
                                            float max_force,
                                            float current_velocity = 0.0f) {
    nuka::runtime::articulation::JointDrive drive{};
    drive.target_velocity = target_velocity;
    drive.damping = 10.0f;
    drive.max_force = max_force;

    nuka::constraint::RowBuffers rows;
    nuka::runtime::articulation::AppendDriveRow(
        &rows,
        0u,
        1u,
        {0.0f, 0.0f, 1.0f},
        drive,
        current_velocity);
    return rows;
}

} // namespace

TEST(JointDrive, VelocityDriveMovesTowardTarget) {
    auto result = nuka::runtime::articulation::SimulateVelocityDriveStep();
    EXPECT_GT(result.joint_velocity, 0.0f);
}

TEST(JointDrive, DriveForceDirectionMatchesError) {
    const auto rows = BuildDriveRows(5.0f, 100.0f, 0.0f);
    const auto rows2 = BuildDriveRows(5.0f, 100.0f, 10.0f);

    ASSERT_EQ(rows.RowCount(), 1u);
    ASSERT_EQ(rows2.RowCount(), 1u);
    EXPECT_FLOAT_EQ(rows.rows[0].rhs, 5.0f);
    EXPECT_FLOAT_EQ(rows2.rows[0].rhs, 5.0f);
}

TEST(JointDrive, ZeroTargetProducesZeroForceAtRest) {
    const auto rows = BuildDriveRows(0.0f, 100.0f);

    ASSERT_EQ(rows.RowCount(), 1u);
    EXPECT_FLOAT_EQ(rows.rows[0].rhs, 0.0f);
}

TEST(JointDrive, DriveConstraintHasSingleRow) {
    const auto rows = BuildDriveRows(1.0f, 100.0f);

    EXPECT_EQ(rows.RowCount(), 1u);
    ASSERT_EQ(rows.materials.size(), 1u);
    EXPECT_EQ(rows.materials[0].kind, nuka::constraint::RowKind::Drive);
}

TEST(JointDrive, MaxForceClampedInLimits) {
    const auto rows = BuildDriveRows(1.0f, 50.0f);

    ASSERT_EQ(rows.RowCount(), 1u);
    EXPECT_FLOAT_EQ(rows.rows[0].lower, -50.0f);
    EXPECT_FLOAT_EQ(rows.rows[0].upper, 50.0f);
}

TEST(JointDrive, VelocityDrivePositionIncreases) {
    auto result = nuka::runtime::articulation::SimulateVelocityDriveStep();
    EXPECT_GT(result.joint_position, 0.0f);
}
