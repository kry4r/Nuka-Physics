// ---------------------------------------------------------------------------
// CSR row scheduler and validation bridge coverage.
// ---------------------------------------------------------------------------

#include "constraint/row_builder.hpp"
#include "runtime/rigid/body_state.hpp"
#include "solver/diff_test_bridge.hpp"
#include "solver/gpu/row_scheduler.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace nuka;

namespace {

runtime::rigid::BodyState DynamicBody(math::Vec3 position,
                                      math::Vec3 velocity = math::Vec3::Zero()) {
    runtime::rigid::BodyState body;
    body.inv_mass = 1.0f;
    body.inv_inertia = {1.0f, 1.0f, 1.0f};
    body.position = position;
    body.linear_velocity = velocity;
    return body;
}

constraint::RowBuffers BuildGroundContact(uint32_t body_index) {
    constraint::RowBuffers rows;
    constraint::RowMaterial material;
    material.kind = constraint::RowKind::Contact;
    material.group_id = 0u;
    material.normal_row_count = 1u;
    material.first_friction_row = 1u;
    material.friction_row_count = 2u;
    material.friction = 0.5f;

    constraint::Row row;
    row.row_class_id = constraint::kMaximalContactRowClassId;
    row.lower = 0.0f;
    row.upper = constraint::kRowHugeLimit;
    row.flags = constraint::row_flags::Unilateral;
    rows.AddRow(row,
                {body_index, constraint::kInvalidBodyIndex},
                {{0.0f, 1.0f, 0.0f}, math::Vec3::Zero()},
                {{0.0f, -1.0f, 0.0f}, math::Vec3::Zero()},
                material);

    constraint::Row friction_row;
    friction_row.row_class_id = constraint::kMaximalContactRowClassId;
    friction_row.flags = constraint::row_flags::Friction;
    rows.AddRow(friction_row,
                {body_index, constraint::kInvalidBodyIndex},
                {{1.0f, 0.0f, 0.0f}, math::Vec3::Zero()},
                {{-1.0f, 0.0f, 0.0f}, math::Vec3::Zero()},
                material);
    rows.AddRow(friction_row,
                {body_index, constraint::kInvalidBodyIndex},
                {{0.0f, 0.0f, 1.0f}, math::Vec3::Zero()},
                {{0.0f, 0.0f, -1.0f}, math::Vec3::Zero()},
                material);
    return rows;
}

constraint::RowBuffers BuildConflictingRows() {
    constraint::RowBuffers rows;
    for (uint32_t row_index = 0u; row_index < 6u; ++row_index) {
        constraint::Row row;
        row.row_class_id = constraint::kMaximalJointRowClassId;
        row.lower = -constraint::kRowHugeLimit;
        row.upper = constraint::kRowHugeLimit;
        row.flags = constraint::row_flags::Equality;
        const uint32_t body_a = row_index % 3u;
        const uint32_t body_b = (row_index + 1u) % 3u;
        rows.AddRow(row,
                    {body_a, body_b},
                    {{1.0f, 0.0f, 0.0f}, math::Vec3::Zero()},
                    {{-1.0f, 0.0f, 0.0f}, math::Vec3::Zero()});
    }
    return rows;
}

std::vector<runtime::rigid::BodyState> BuildTwoBodyScene() {
    std::vector<runtime::rigid::BodyState> bodies;
    bodies.push_back(DynamicBody({0.0f, 0.0f, 0.0f},
                                 {3.0f, -2.0f, 0.0f}));
    bodies.push_back(DynamicBody({0.0f, 0.0f, 0.0f},
                                 {-1.0f, -3.0f, 0.5f}));
    return bodies;
}

} // namespace

TEST(RowScheduler, ColoringIsDeterministicAndRaceFree) {
    const auto rows = BuildConflictingRows();
    const auto first = solver::gpu::BuildRowColorPartitions(rows);
    const auto second = solver::gpu::BuildRowColorPartitions(rows);

    EXPECT_EQ(first.row_indices, second.row_indices);
    ASSERT_EQ(first.color_ranges.size(), second.color_ranges.size());
    for (size_t index = 0; index < first.color_ranges.size(); ++index) {
        EXPECT_EQ(first.color_ranges[index].row_offset,
                  second.color_ranges[index].row_offset);
        EXPECT_EQ(first.color_ranges[index].row_count,
                  second.color_ranges[index].row_count);
    }
    EXPECT_TRUE(solver::gpu::ValidateNoSharedBodiesPerColor(rows, first));
    EXPECT_GT(first.ColorCount(), 1u);
}

TEST(RowSolverDiffBridge, ContactSmokeAgreesWithReferenceReplay) {
    auto rows = BuildGroundContact(0u);
    std::vector<runtime::rigid::BodyState> bodies{
        DynamicBody({0.0f, 0.0f, 0.0f}, {4.0f, -2.0f, 0.0f})};

    solver::gpu::RowSolveConfig config;
    config.velocity_iterations = 16u;
    config.position_iterations = 0u;
    solver::DiffTestBridge bridge(phi::MakeDefaultDeviceContext());
    const auto report = bridge.SolveAndCompare(rows, bodies, config);

    EXPECT_EQ(report.row_count, 3u);
    EXPECT_TRUE(bridge.LastDivergence().empty());
    EXPECT_GE(bodies[0].linear_velocity.y, -1.0e-5f);
}

TEST(RowSolverDiffBridge, JointAndDriveSmokeAgreesWithReferenceReplay) {
    std::vector<runtime::rigid::BodyState> bodies = BuildTwoBodyScene();
    bodies[1].inv_mass = 0.0f;
    bodies[1].inv_inertia = math::Vec3::Zero();

    constraint::RowBuffers rows;
    constraint::AppendDriveRow(&rows,
                               0u,
                               1u,
                               math::Vec3::UnitZ(),
                               2.0f,
                               10.0f);

    solver::gpu::RowSolveConfig config;
    config.velocity_iterations = 8u;
    config.position_iterations = 0u;
    solver::DiffTestBridge bridge(phi::MakeDefaultDeviceContext());
    const auto report = bridge.SolveAndCompare(rows, bodies, config);

    EXPECT_EQ(report.row_count, 1u);
    EXPECT_TRUE(bridge.LastDivergence().empty());
    EXPECT_NEAR(bodies[0].angular_velocity.z, 2.0f, 1.0e-4f);
}
