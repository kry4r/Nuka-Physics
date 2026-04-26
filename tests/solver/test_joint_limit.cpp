// ---------------------------------------------------------------------------
// Tests: Joint limit enforcement in solver
// ---------------------------------------------------------------------------

#include "solver/rigid_solver.hpp"
#include "constraint/constraint_block.hpp"
#include "runtime/rigid/body_state.hpp"

#include <gtest/gtest.h>
#include <cmath>

TEST(JointLimit, VelocityClampedByLimit) {
    // Two bodies connected by a joint constraint with velocity limits.
    // Body A is spinning; the joint limit should reduce its velocity.
    std::vector<nuka::runtime::rigid::BodyState> bodies(2);

    // Body A: dynamic, spinning fast around Z
    bodies[0].inv_mass = 1.0f;
    bodies[0].inv_inertia = {1.0f, 1.0f, 1.0f};
    bodies[0].angular_velocity = {0.0f, 0.0f, 10.0f}; // fast spin

    // Body B: static anchor
    bodies[1].inv_mass = 0.0f; // infinite mass
    bodies[1].inv_inertia = {0.0f, 0.0f, 0.0f};

    // Build a joint constraint that limits relative angular velocity around Z
    nuka::constraint::ConstraintBlock joint{};
    joint.type = nuka::constraint::ConstraintType::Joint;
    joint.body_a = 0;
    joint.body_b = 1;
    joint.row_count = 1;

    // Angular constraint around Z axis
    joint.jacobian_linear_a[0]  = nuka::math::Vec3::Zero();
    joint.jacobian_angular_a[0] = {0.0f, 0.0f, 1.0f};
    joint.jacobian_linear_b[0]  = nuka::math::Vec3::Zero();
    joint.jacobian_angular_b[0] = {0.0f, 0.0f, -1.0f};
    joint.rhs[0] = 0.0f;         // target relative angular velocity = 0
    joint.lower_limit[0] = -5.0f; // impulse limits
    joint.upper_limit[0] = 5.0f;

    std::vector<nuka::constraint::ConstraintBlock> blocks;
    blocks.push_back(joint);

    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 20;
    config.position_iterations = 0;

    nuka::solver::SolveConstraints(blocks, bodies, config);

    // The solver should have reduced body A's angular velocity around Z
    EXPECT_LT(std::abs(bodies[0].angular_velocity.z), 10.0f);
}

TEST(JointLimit, ImpulseStaysWithinLimits) {
    std::vector<nuka::runtime::rigid::BodyState> bodies(2);
    bodies[0].inv_mass = 1.0f;
    bodies[0].inv_inertia = {1.0f, 1.0f, 1.0f};
    bodies[0].angular_velocity = {0.0f, 0.0f, 100.0f}; // very fast

    bodies[1].inv_mass = 0.0f;
    bodies[1].inv_inertia = {0.0f, 0.0f, 0.0f};

    nuka::constraint::ConstraintBlock joint{};
    joint.type = nuka::constraint::ConstraintType::Joint;
    joint.body_a = 0;
    joint.body_b = 1;
    joint.row_count = 1;

    joint.jacobian_linear_a[0]  = nuka::math::Vec3::Zero();
    joint.jacobian_angular_a[0] = {0.0f, 0.0f, 1.0f};
    joint.jacobian_linear_b[0]  = nuka::math::Vec3::Zero();
    joint.jacobian_angular_b[0] = {0.0f, 0.0f, -1.0f};
    joint.rhs[0] = 0.0f;
    joint.lower_limit[0] = -2.0f;
    joint.upper_limit[0] = 2.0f;

    std::vector<nuka::constraint::ConstraintBlock> blocks;
    blocks.push_back(joint);

    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 10;
    config.position_iterations = 0;

    nuka::solver::SolveConstraints(blocks, bodies, config);

    // Accumulated impulse should be clamped within limits
    EXPECT_GE(blocks[0].impulse[0], -2.0f);
    EXPECT_LE(blocks[0].impulse[0], 2.0f);
}

TEST(JointLimit, ZeroVelocityProducesNoImpulse) {
    std::vector<nuka::runtime::rigid::BodyState> bodies(2);
    bodies[0].inv_mass = 1.0f;
    bodies[0].inv_inertia = {1.0f, 1.0f, 1.0f};
    // Zero velocity -- already satisfied

    bodies[1].inv_mass = 0.0f;
    bodies[1].inv_inertia = {0.0f, 0.0f, 0.0f};

    nuka::constraint::ConstraintBlock joint{};
    joint.type = nuka::constraint::ConstraintType::Joint;
    joint.body_a = 0;
    joint.body_b = 1;
    joint.row_count = 1;
    joint.jacobian_linear_a[0]  = nuka::math::Vec3::Zero();
    joint.jacobian_angular_a[0] = {0.0f, 0.0f, 1.0f};
    joint.jacobian_linear_b[0]  = nuka::math::Vec3::Zero();
    joint.jacobian_angular_b[0] = {0.0f, 0.0f, -1.0f};
    joint.rhs[0] = 0.0f;
    joint.lower_limit[0] = -10.0f;
    joint.upper_limit[0] = 10.0f;

    std::vector<nuka::constraint::ConstraintBlock> blocks;
    blocks.push_back(joint);

    nuka::solver::SolveConstraints(blocks, bodies);

    EXPECT_NEAR(blocks[0].impulse[0], 0.0f, 1e-6f);
}
