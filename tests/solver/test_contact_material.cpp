// ---------------------------------------------------------------------------
// Tests: contact friction and restitution solver behavior
// ---------------------------------------------------------------------------

#include "constraint/constraint_block.hpp"
#include "runtime/rigid/body_state.hpp"
#include "solver/rigid_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using nuka::constraint::ConstraintBlock;
using nuka::constraint::ConstraintType;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;

BodyState DynamicBody(Vec3 velocity) {
    BodyState body{};
    body.inv_mass = 1.0f;
    body.inv_inertia = {1.0f, 1.0f, 1.0f};
    body.linear_velocity = velocity;
    return body;
}

ConstraintBlock BuildGroundContact(float friction, float restitution) {
    ConstraintBlock contact{};
    contact.type = ConstraintType::Contact;
    contact.body_a = 0;
    contact.body_b = ~0u;
    contact.row_count = 3;

    contact.jacobian_linear_a[0] = {0.0f, 1.0f, 0.0f};
    contact.jacobian_linear_b[0] = {0.0f, -1.0f, 0.0f};
    contact.lower_limit[0] = 0.0f;
    contact.upper_limit[0] = 1e6f;

    contact.jacobian_linear_a[1] = {1.0f, 0.0f, 0.0f};
    contact.jacobian_linear_b[1] = {-1.0f, 0.0f, 0.0f};
    contact.lower_limit[1] = 0.0f;
    contact.upper_limit[1] = 0.0f;

    contact.jacobian_linear_a[2] = {0.0f, 0.0f, 1.0f};
    contact.jacobian_linear_b[2] = {0.0f, 0.0f, -1.0f};
    contact.lower_limit[2] = 0.0f;
    contact.upper_limit[2] = 0.0f;

    contact.friction = friction;
    contact.restitution = restitution;
    contact.normal_row_count = 1;
    contact.first_friction_row = 1;
    contact.friction_row_count = 2;
    return contact;
}

} // namespace

TEST(ContactMaterial, FrictionImpulseIsClampedByNormalImpulse) {
    std::vector<BodyState> bodies{DynamicBody({10.0f, -2.0f, 0.0f})};
    std::vector<ConstraintBlock> blocks{BuildGroundContact(0.5f, 0.0f)};

    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 20;
    config.position_iterations = 0;

    nuka::solver::SolveConstraints(blocks, bodies, config);

    const float normal_impulse = blocks[0].impulse[0];
    const float tangent_impulse = std::abs(blocks[0].impulse[1]);
    EXPECT_GT(normal_impulse, 0.0f);
    EXPECT_LE(tangent_impulse, 0.5f * normal_impulse + 1e-5f);
    EXPECT_NEAR(bodies[0].linear_velocity.x, 9.0f, 1e-5f);
}

TEST(ContactMaterial, NoNormalImpulseProducesNoFrictionImpulse) {
    std::vector<BodyState> bodies{DynamicBody({10.0f, 1.0f, 0.0f})};
    std::vector<ConstraintBlock> blocks{BuildGroundContact(1.0f, 0.0f)};

    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 20;
    config.position_iterations = 0;

    nuka::solver::SolveConstraints(blocks, bodies, config);

    EXPECT_FLOAT_EQ(blocks[0].impulse[0], 0.0f);
    EXPECT_FLOAT_EQ(blocks[0].impulse[1], 0.0f);
    EXPECT_FLOAT_EQ(bodies[0].linear_velocity.x, 10.0f);
}

TEST(ContactMaterial, RestitutionBouncesAlongContactNormal) {
    std::vector<BodyState> bodies{DynamicBody({0.0f, -3.0f, 0.0f})};
    std::vector<ConstraintBlock> blocks{BuildGroundContact(0.0f, 0.5f)};

    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 12;
    config.position_iterations = 0;

    nuka::solver::SolveConstraints(blocks, bodies, config);

    EXPECT_NEAR(bodies[0].linear_velocity.y, 1.5f, 1e-5f);
    EXPECT_GT(blocks[0].impulse[0], 3.0f);
}
