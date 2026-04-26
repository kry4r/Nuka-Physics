// ---------------------------------------------------------------------------
// Regression test: Two boxes stacked on a ground plane
// ---------------------------------------------------------------------------

#include "solver/rigid_solver.hpp"
#include "runtime/rigid/integrator.hpp"
#include "constraint/constraint_block.hpp"
#include "math/vec3.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace nuka;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Build a minimal two-box stack scenario:
//   body 0: bottom box sitting on ground plane (y = 0.5)
//   body 1: top box sitting on bottom box   (y = 1.5)
// Each box is 1m tall.  Ground is at y = 0.

struct StackResult {
    float max_penetration;
    float top_box_y;
    float bottom_box_y;
};

StackResult RunRestStackRegression() {
    std::vector<runtime::rigid::BodyState> bodies(2);

    // Bottom box: mass = 2 kg, resting on ground at y = 0.5
    bodies[0].inv_mass         = 0.5f;
    bodies[0].inv_inertia      = {6.0f, 6.0f, 6.0f};
    bodies[0].position         = {0.0f, 0.5f, 0.0f};
    bodies[0].linear_velocity  = {0.0f, 0.0f, 0.0f};
    bodies[0].angular_velocity = {0.0f, 0.0f, 0.0f};
    bodies[0].force            = {0.0f, 0.0f, 0.0f};
    bodies[0].torque           = {0.0f, 0.0f, 0.0f};

    // Top box: mass = 1 kg, sitting on top of bottom box at y = 1.5
    bodies[1].inv_mass         = 1.0f;
    bodies[1].inv_inertia      = {6.0f, 6.0f, 6.0f};
    bodies[1].position         = {0.0f, 1.5f, 0.0f};
    bodies[1].linear_velocity  = {0.0f, 0.0f, 0.0f};
    bodies[1].angular_velocity = {0.0f, 0.0f, 0.0f};
    bodies[1].force            = {0.0f, 0.0f, 0.0f};
    bodies[1].torque           = {0.0f, 0.0f, 0.0f};

    // Contact 1: bottom box vs ground (body_b = ~0u means static world)
    constraint::ConstraintBlock ground_contact{};
    ground_contact.type       = constraint::ConstraintType::Contact;
    ground_contact.body_a     = 0;
    ground_contact.body_b     = ~0u;
    ground_contact.row_count  = 1;
    ground_contact.jacobian_linear_a[0]  = {0.0f, -1.0f, 0.0f};
    ground_contact.jacobian_angular_a[0] = math::Vec3::Zero();
    ground_contact.jacobian_linear_b[0]  = {0.0f,  1.0f, 0.0f};
    ground_contact.jacobian_angular_b[0] = math::Vec3::Zero();
    ground_contact.rhs[0]          = 0.0f;
    ground_contact.lower_limit[0]  = 0.0f;
    ground_contact.upper_limit[0]  = 1e6f;

    // Contact 2: top box vs bottom box
    constraint::ConstraintBlock stack_contact{};
    stack_contact.type       = constraint::ConstraintType::Contact;
    stack_contact.body_a     = 1;
    stack_contact.body_b     = 0;
    stack_contact.row_count  = 1;
    stack_contact.jacobian_linear_a[0]  = {0.0f, -1.0f, 0.0f};
    stack_contact.jacobian_angular_a[0] = math::Vec3::Zero();
    stack_contact.jacobian_linear_b[0]  = {0.0f,  1.0f, 0.0f};
    stack_contact.jacobian_angular_b[0] = math::Vec3::Zero();
    stack_contact.rhs[0]          = 0.0f;
    stack_contact.lower_limit[0]  = 0.0f;
    stack_contact.upper_limit[0]  = 1e6f;

    std::vector<constraint::ConstraintBlock> blocks;
    blocks.push_back(ground_contact);
    blocks.push_back(stack_contact);

    auto result = solver::SolveConstraints(blocks, bodies);

    return {result.max_penetration, bodies[1].position.y, bodies[0].position.y};
}

}  // namespace

// ---------------------------------------------------------------------------
// Penetration is bounded after solving
// ---------------------------------------------------------------------------

TEST(RestStackRegression, PenetrationIsBounded) {
    const auto result = RunRestStackRegression();
    EXPECT_LT(result.max_penetration, 1e-2f);
}

// ---------------------------------------------------------------------------
// Top box stays above bottom box
// ---------------------------------------------------------------------------

TEST(RestStackRegression, TopBoxStaysAboveBottomBox) {
    const auto result = RunRestStackRegression();
    EXPECT_GT(result.top_box_y, result.bottom_box_y);
}

// ---------------------------------------------------------------------------
// Bottom box stays above ground
// ---------------------------------------------------------------------------

TEST(RestStackRegression, BottomBoxStaysAboveGround) {
    const auto result = RunRestStackRegression();
    EXPECT_GE(result.bottom_box_y, 0.0f);
}
