// ---------------------------------------------------------------------------
// Tests: Resting contact solver
// ---------------------------------------------------------------------------

#include "solver/rigid_solver.hpp"

#include <gtest/gtest.h>
#include <cmath>

TEST(RigidSolver, RestingContactPreventsFurtherPenetration) {
    auto result = nuka::solver::SolveUnitGroundContact();
    EXPECT_LE(result.max_penetration, 1e-3f);
}

TEST(RigidSolver, IterationsReported) {
    auto result = nuka::solver::SolveUnitGroundContact();
    EXPECT_GE(result.iterations_used, 1u);
    EXPECT_LE(result.iterations_used, 100u);
}

TEST(RigidSolver, SlopToleranceRespected) {
    // Build a scenario with small penetration within slop
    std::vector<nuka::runtime::rigid::BodyState> bodies(1);
    bodies[0].inv_mass = 1.0f;
    bodies[0].inv_inertia = {6.0f, 6.0f, 6.0f};
    bodies[0].position = {0.0f, -0.003f, 0.0f}; // 3mm penetration, within default slop of 5mm
    bodies[0].linear_velocity = {0.0f, 0.0f, 0.0f};

    nuka::constraint::ConstraintBlock contact{};
    contact.type = nuka::constraint::ConstraintType::Contact;
    contact.body_a = 0;
    contact.body_b = ~0u;
    contact.row_count = 1;
    contact.jacobian_linear_a[0]  = {0.0f, -1.0f, 0.0f};
    contact.jacobian_angular_a[0] = nuka::math::Vec3::Zero();
    contact.jacobian_linear_b[0]  = {0.0f, 1.0f, 0.0f};
    contact.jacobian_angular_b[0] = nuka::math::Vec3::Zero();
    contact.rhs[0] = 0.003f;  // 3mm penetration encoded as positive rhs (negated in solver)
    contact.lower_limit[0] = 0.0f;
    contact.upper_limit[0] = 1e6f;

    std::vector<nuka::constraint::ConstraintBlock> blocks;
    blocks.push_back(contact);

    nuka::solver::SolverConfig config{};
    config.slop = 0.005f;

    auto result = nuka::solver::SolveConstraints(blocks, bodies, config);

    // Penetration within slop should not trigger strong correction
    // The body should not have moved significantly upward
    EXPECT_LE(std::abs(bodies[0].position.y), 0.01f);
}

TEST(RigidSolver, EmptyInputReturnsZero) {
    std::vector<nuka::constraint::ConstraintBlock> blocks;
    std::vector<nuka::runtime::rigid::BodyState> bodies;
    auto result = nuka::solver::SolveConstraints(blocks, bodies);
    EXPECT_FLOAT_EQ(result.max_penetration, 0.0f);
    EXPECT_EQ(result.iterations_used, 0u);
}
