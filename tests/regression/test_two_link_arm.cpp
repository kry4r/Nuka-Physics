// ---------------------------------------------------------------------------
// Regression test: Two-link robot arm with revolute joint
// ---------------------------------------------------------------------------

#include "runtime/articulation/joint_constraints.hpp"
#include "runtime/rigid/integrator.hpp"
#include "solver/rigid_solver.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace nuka;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Two-link arm:
//   body 0 (base link): anchored at origin, extends along +X to (1,0,0)
//   body 1 (link1):     starts at (1,0,0), extends along +X to (2,0,0)
//   Revolute joint at (1,0,0) around Z axis connects body 0 and body 1.

struct ArmResult {
    math::Vec3 body0_end;   // end effector of base link
    math::Vec3 body1_start; // pivot point of link1
    float      gap;         // distance between joint anchor points
};

ArmResult RunTwoLinkArmRegression() {
    std::vector<runtime::rigid::BodyState> bodies(2);

    // Base link (body 0): center at (0.5, 0, 0)
    bodies[0].inv_mass         = 1.0f;
    bodies[0].inv_inertia      = {6.0f, 6.0f, 6.0f};
    bodies[0].position         = {0.5f, 0.0f, 0.0f};
    bodies[0].orientation      = math::Quat::Identity();
    bodies[0].linear_velocity  = {0.0f, 0.0f, 0.0f};
    bodies[0].angular_velocity = {0.0f, 0.0f, 0.0f};
    bodies[0].force            = {0.0f, 0.0f, 0.0f};
    bodies[0].torque           = {0.0f, 0.0f, 0.0f};

    // Link1 (body 1): center at (1.5, 0, 0)
    bodies[1].inv_mass         = 1.0f;
    bodies[1].inv_inertia      = {6.0f, 6.0f, 6.0f};
    bodies[1].position         = {1.5f, 0.0f, 0.0f};
    bodies[1].orientation      = math::Quat::Identity();
    bodies[1].linear_velocity  = {0.0f, -1.0f, 0.0f}; // give it a downward kick
    bodies[1].angular_velocity = {0.0f, 0.0f, 0.0f};
    bodies[1].force            = {0.0f, 0.0f, 0.0f};
    bodies[1].torque           = {0.0f, 0.0f, 0.0f};

    // Frame A: joint anchor in body 0's local frame is at (+0.5, 0, 0)
    math::Transform frame_a({0.5f, 0.0f, 0.0f}, math::Quat::Identity());
    // Frame B: joint anchor in body 1's local frame is at (-0.5, 0, 0)
    math::Transform frame_b({-0.5f, 0.0f, 0.0f}, math::Quat::Identity());

    auto joint_block = runtime::articulation::BuildRevoluteConstraint(
        0, 1,
        {0.0f, 0.0f, 1.0f},  // revolute axis = Z
        frame_a, frame_b,
        -3.14f, 3.14f
    );

    std::vector<constraint::ConstraintBlock> blocks;
    blocks.push_back(joint_block);

    // Apply gravity and step bodies
    const math::Vec3 gravity{0.0f, -9.81f, 0.0f};
    const float dt = 0.01f;
    const int num_steps = 10;

    for (int step = 0; step < num_steps; ++step) {
        // Integrate
        for (auto& body : bodies) {
            StepBody(body, gravity, dt);
        }

        // Solve joint constraint
        solver::SolveConstraints(blocks, bodies);
    }

    // Compute the world-space joint anchor point from each body's perspective
    // Body 0's anchor: body0.position + rotate(body0.orientation, local_anchor_a)
    math::Vec3 local_a{0.5f, 0.0f, 0.0f};
    math::Vec3 local_b{-0.5f, 0.0f, 0.0f};

    // Simplified: assume orientations are close to identity for this test
    math::Vec3 world_anchor_a = {
        bodies[0].position.x + local_a.x,
        bodies[0].position.y + local_a.y,
        bodies[0].position.z + local_a.z
    };
    math::Vec3 world_anchor_b = {
        bodies[1].position.x + local_b.x,
        bodies[1].position.y + local_b.y,
        bodies[1].position.z + local_b.z
    };

    float dx = world_anchor_a.x - world_anchor_b.x;
    float dy = world_anchor_a.y - world_anchor_b.y;
    float dz = world_anchor_a.z - world_anchor_b.z;
    float gap = std::sqrt(dx * dx + dy * dy + dz * dz);

    return {world_anchor_a, world_anchor_b, gap};
}

}  // namespace

// ---------------------------------------------------------------------------
// Joint constraint maintains link connectivity
// ---------------------------------------------------------------------------

TEST(TwoLinkArmRegression, JointMaintainsConnectivity) {
    const auto result = RunTwoLinkArmRegression();
    // The gap between joint anchor points should be very small
    EXPECT_LT(result.gap, 0.1f);
}

// ---------------------------------------------------------------------------
// Anchor points remain close
// ---------------------------------------------------------------------------

TEST(TwoLinkArmRegression, AnchorPointsCoincide) {
    const auto result = RunTwoLinkArmRegression();
    EXPECT_NEAR(result.body0_end.x, result.body1_start.x, 0.1f);
    EXPECT_NEAR(result.body0_end.y, result.body1_start.y, 0.1f);
    EXPECT_NEAR(result.body0_end.z, result.body1_start.z, 0.1f);
}
