// ---------------------------------------------------------------------------
// Tests: contact friction and restitution row behavior
// ---------------------------------------------------------------------------

#include "runtime/rigid/body_state.hpp"
#include "solver/rigid_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;

BodyState DynamicBody(Vec3 velocity) {
    BodyState body{};
    body.inv_mass = 1.0f;
    body.inv_inertia = {1.0f, 1.0f, 1.0f};
    body.linear_velocity = velocity;
    return body;
}

nuka::constraint::RowBuffers BuildGroundContact(float friction, float restitution) {
    nuka::constraint::RowBuffers rows;

    nuka::constraint::RowMaterial material;
    material.kind = nuka::constraint::RowKind::Contact;
    material.group_id = 0u;
    material.normal_row_count = 1u;
    material.first_friction_row = 1u;
    material.friction_row_count = 2u;
    material.friction = friction;
    material.restitution = restitution;

    nuka::constraint::Row normal;
    normal.row_class_id = nuka::constraint::kMaximalContactRowClassId;
    normal.lower = 0.0f;
    normal.upper = nuka::constraint::kRowHugeLimit;
    normal.flags = nuka::constraint::row_flags::Unilateral;
    rows.AddRow(normal,
                {0u, nuka::constraint::kInvalidBodyIndex},
                {{0.0f, 1.0f, 0.0f}, Vec3::Zero()},
                {{0.0f, -1.0f, 0.0f}, Vec3::Zero()},
                material);

    nuka::constraint::Row friction_x;
    friction_x.row_class_id = nuka::constraint::kMaximalContactRowClassId;
    friction_x.flags = nuka::constraint::row_flags::Friction;
    rows.AddRow(friction_x,
                {0u, nuka::constraint::kInvalidBodyIndex},
                {{1.0f, 0.0f, 0.0f}, Vec3::Zero()},
                {{-1.0f, 0.0f, 0.0f}, Vec3::Zero()},
                material);

    nuka::constraint::Row friction_z;
    friction_z.row_class_id = nuka::constraint::kMaximalContactRowClassId;
    friction_z.flags = nuka::constraint::row_flags::Friction;
    rows.AddRow(friction_z,
                {0u, nuka::constraint::kInvalidBodyIndex},
                {{0.0f, 0.0f, 1.0f}, Vec3::Zero()},
                {{0.0f, 0.0f, -1.0f}, Vec3::Zero()},
                material);
    return rows;
}

} // namespace

TEST(ContactMaterial, FrictionImpulseIsClampedByNormalImpulse) {
    std::vector<BodyState> bodies{DynamicBody({10.0f, -2.0f, 0.0f})};
    auto rows = BuildGroundContact(0.5f, 0.0f);

    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 20u;
    config.position_iterations = 0u;

    nuka::solver::SolveConstraints(rows, bodies, config);

    const float normal_impulse = rows.rows[0].lambda;
    const float tangent_impulse = std::abs(rows.rows[1].lambda);
    EXPECT_GT(normal_impulse, 0.0f);
    EXPECT_LE(tangent_impulse, 0.5f * normal_impulse + 1.0e-5f);
    EXPECT_NEAR(bodies[0].linear_velocity.x, 9.0f, 1.0e-5f);
}

TEST(ContactMaterial, NoNormalImpulseProducesNoFrictionImpulse) {
    std::vector<BodyState> bodies{DynamicBody({10.0f, 1.0f, 0.0f})};
    auto rows = BuildGroundContact(1.0f, 0.0f);

    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 20u;
    config.position_iterations = 0u;

    nuka::solver::SolveConstraints(rows, bodies, config);

    EXPECT_FLOAT_EQ(rows.rows[0].lambda, 0.0f);
    EXPECT_FLOAT_EQ(rows.rows[1].lambda, 0.0f);
    EXPECT_FLOAT_EQ(bodies[0].linear_velocity.x, 10.0f);
}

TEST(ContactMaterial, RestitutionBouncesAlongContactNormal) {
    std::vector<BodyState> bodies{DynamicBody({0.0f, -3.0f, 0.0f})};
    auto rows = BuildGroundContact(0.0f, 0.5f);

    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 12u;
    config.position_iterations = 0u;

    nuka::solver::SolveConstraints(rows, bodies, config);

    EXPECT_NEAR(bodies[0].linear_velocity.y, 1.5f, 1.0e-5f);
    EXPECT_GT(rows.rows[0].lambda, 3.0f);
}
