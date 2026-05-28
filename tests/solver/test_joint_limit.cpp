// ---------------------------------------------------------------------------
// Tests: bounded joint row behavior in the GPU row solver
// ---------------------------------------------------------------------------

#include "runtime/rigid/body_state.hpp"
#include "solver/rigid_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

nuka::constraint::RowBuffers BuildAngularLimitRow(float lower, float upper) {
    nuka::constraint::RowBuffers rows;
    nuka::constraint::Row row;
    row.row_class_id = nuka::constraint::kMaximalJointRowClassId;
    row.lower = lower;
    row.upper = upper;
    row.flags = nuka::constraint::row_flags::Equality;
    nuka::constraint::RowMaterial material;
    material.kind = nuka::constraint::RowKind::Joint;
    rows.AddRow(row,
                {0u, 1u},
                {nuka::math::Vec3::Zero(), {0.0f, 0.0f, 1.0f}},
                {nuka::math::Vec3::Zero(), {0.0f, 0.0f, -1.0f}},
                material);
    return rows;
}

std::vector<nuka::runtime::rigid::BodyState> BuildBodies(float angular_z) {
    std::vector<nuka::runtime::rigid::BodyState> bodies(2);
    bodies[0].inv_mass = 1.0f;
    bodies[0].inv_inertia = {1.0f, 1.0f, 1.0f};
    bodies[0].angular_velocity = {0.0f, 0.0f, angular_z};
    return bodies;
}

} // namespace

TEST(JointLimit, VelocityClampedByLimit) {
    auto bodies = BuildBodies(10.0f);
    auto rows = BuildAngularLimitRow(-5.0f, 5.0f);

    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 20u;
    config.position_iterations = 0u;

    nuka::solver::SolveConstraints(rows, bodies, config);

    EXPECT_LT(std::abs(bodies[0].angular_velocity.z), 10.0f);
}

TEST(JointLimit, ImpulseStaysWithinLimits) {
    auto bodies = BuildBodies(100.0f);
    auto rows = BuildAngularLimitRow(-2.0f, 2.0f);

    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 10u;
    config.position_iterations = 0u;

    nuka::solver::SolveConstraints(rows, bodies, config);

    EXPECT_GE(rows.rows[0].lambda, -2.0f);
    EXPECT_LE(rows.rows[0].lambda, 2.0f);
}

TEST(JointLimit, ZeroVelocityProducesNoImpulse) {
    auto bodies = BuildBodies(0.0f);
    auto rows = BuildAngularLimitRow(-10.0f, 10.0f);

    nuka::solver::SolveConstraints(rows, bodies);

    EXPECT_NEAR(rows.rows[0].lambda, 0.0f, 1.0e-6f);
}
