// ---------------------------------------------------------------------------
// CUDA Universal Row solver tests.
// ---------------------------------------------------------------------------

#include "constraint/row_builder.hpp"
#include "runtime/rigid/body_state.hpp"
#include "solver/gpu/row_solver.cuh"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace nuka;

namespace {

runtime::rigid::BodyState DynamicBody(math::Vec3 velocity) {
    runtime::rigid::BodyState body{};
    body.inv_mass = 1.0f;
    body.inv_inertia = {1.0f, 1.0f, 1.0f};
    body.linear_velocity = velocity;
    return body;
}

constraint::RowBuffers BuildGroundContact(float friction, float restitution) {
    constraint::RowBuffers rows;

    constraint::RowMaterial material;
    material.kind = constraint::RowKind::Contact;
    material.group_id = 0u;
    material.normal_row_count = 1u;
    material.first_friction_row = 1u;
    material.friction_row_count = 2u;
    material.friction = friction;
    material.restitution = restitution;

    constraint::Row normal;
    normal.row_class_id = constraint::kMaximalContactRowClassId;
    normal.lower = 0.0f;
    normal.upper = constraint::kRowHugeLimit;
    normal.flags = constraint::row_flags::Unilateral;
    rows.AddRow(normal,
                {0u, constraint::kInvalidBodyIndex},
                {{0.0f, 1.0f, 0.0f}, math::Vec3::Zero()},
                {{0.0f, -1.0f, 0.0f}, math::Vec3::Zero()},
                material);

    constraint::Row friction_row;
    friction_row.row_class_id = constraint::kMaximalContactRowClassId;
    friction_row.flags = constraint::row_flags::Friction;
    rows.AddRow(friction_row,
                {0u, constraint::kInvalidBodyIndex},
                {{1.0f, 0.0f, 0.0f}, math::Vec3::Zero()},
                {{-1.0f, 0.0f, 0.0f}, math::Vec3::Zero()},
                material);
    rows.AddRow(friction_row,
                {0u, constraint::kInvalidBodyIndex},
                {{0.0f, 0.0f, 1.0f}, math::Vec3::Zero()},
                {{0.0f, 0.0f, -1.0f}, math::Vec3::Zero()},
                material);
    return rows;
}

} // namespace

TEST(CudaRowSolver, FrictionImpulseIsClampedByNormalImpulseOnDevice) {
    std::vector<runtime::rigid::BodyState> bodies{
        DynamicBody({10.0f, -2.0f, 0.0f})};
    auto rows = BuildGroundContact(0.5f, 0.0f);

    solver::gpu::RowSolveConfig config;
    config.velocity_iterations = 20u;
    config.position_iterations = 0u;
    const auto report =
        solver::gpu::SolveRows(phi::MakeDefaultDeviceContext(), rows, bodies, config);

    EXPECT_EQ(report.row_count, 3u);
    EXPECT_EQ(report.row_scheduler_report.row_kind,
              runtime::gpu::CudaConstraintRowBufferKind::UniversalRowCsr);
    EXPECT_EQ(report.row_scheduler_report.row_layout,
              runtime::gpu::CudaConstraintRowLayout::UniversalRowCsr);
    EXPECT_EQ(report.row_scheduler_report.schedule_mode,
              runtime::gpu::CudaConstraintRowScheduleMode::IslandColoredSweep);
    EXPECT_GT(rows.rows[0].lambda, 0.0f);
    EXPECT_LE(std::abs(rows.rows[1].lambda),
              0.5f * rows.rows[0].lambda + 1.0e-5f);
    EXPECT_NEAR(bodies[0].linear_velocity.x, 9.0f, 1.0e-5f);
}

TEST(CudaRowSolver, RestitutionBouncesAlongContactNormalOnDevice) {
    std::vector<runtime::rigid::BodyState> bodies{
        DynamicBody({0.0f, -3.0f, 0.0f})};
    auto rows = BuildGroundContact(0.0f, 0.5f);

    solver::gpu::RowSolveConfig config;
    config.velocity_iterations = 12u;
    config.position_iterations = 0u;
    solver::gpu::SolveRows(phi::MakeDefaultDeviceContext(), rows, bodies, config);

    EXPECT_NEAR(bodies[0].linear_velocity.y, 1.5f, 1.0e-5f);
    EXPECT_GT(rows.rows[0].lambda, 3.0f);
}

TEST(CudaRowSolver, DriveRowAppliesAngularVelocityOnDevice) {
    std::vector<runtime::rigid::BodyState> bodies(2);
    bodies[0] = DynamicBody(math::Vec3::Zero());
    bodies[1].inv_mass = 0.0f;
    bodies[0].inv_inertia = {1.0f, 1.0f, 1.0f};

    constraint::RowBuffers rows;
    constraint::AppendDriveRow(&rows,
                               0u,
                               1u,
                               math::Vec3::UnitZ(),
                               4.0f,
                               20.0f);

    solver::gpu::RowSolveConfig config;
    config.velocity_iterations = 12u;
    config.position_iterations = 0u;
    solver::gpu::SolveRows(phi::MakeDefaultDeviceContext(), rows, bodies, config);

    EXPECT_NEAR(bodies[0].angular_velocity.z, 4.0f, 1.0e-4f);
}
