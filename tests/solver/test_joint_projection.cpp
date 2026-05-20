// ---------------------------------------------------------------------------
// Tests: joint position projection and mass-weighted correction
// ---------------------------------------------------------------------------

#include "runtime/articulation/joint_constraints.hpp"
#include "runtime/rigid/body_state.hpp"
#include "solver/rigid_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;

Vec3 WorldAnchor(const BodyState& body, const Vec3& local_anchor) {
    return body.position + body.orientation.Rotate(local_anchor);
}

float Distance(const Vec3& a, const Vec3& b) {
    return (a - b).Length();
}

BodyState MakeBody(float inv_mass, Vec3 inv_inertia, Vec3 position) {
    BodyState body{};
    body.inv_mass = inv_mass;
    body.inv_inertia = inv_inertia;
    body.position = position;
    body.orientation = Quat::Identity();
    return body;
}

} // namespace

TEST(JointProjection, StaticAnchorPullsDynamicAnchorOntoPivot) {
    std::vector<BodyState> bodies;
    bodies.push_back(MakeBody(0.0f, Vec3::Zero(), {0.0f, 0.0f, 0.0f}));
    bodies.push_back(MakeBody(1.0f, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}));

    auto joint = nuka::runtime::articulation::BuildRevoluteConstraint(
        0, 1,
        Vec3::UnitZ(),
        Transform(Vec3::Zero(), Quat::Identity()),
        Transform(Vec3::Zero(), Quat::Identity()),
        -3.14f, 3.14f);

    std::vector<nuka::constraint::ConstraintBlock> blocks{joint};
    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 0;
    config.position_iterations = 12;
    config.baumgarte = 0.5f;
    config.slop = 0.0f;

    nuka::solver::SolveConstraints(blocks, bodies, config);

    EXPECT_NEAR(bodies[0].position.y, 0.0f, 1e-6f);
    EXPECT_LT(Distance(WorldAnchor(bodies[0], Vec3::Zero()),
                       WorldAnchor(bodies[1], Vec3::Zero())),
              1e-3f);
}

TEST(JointProjection, MassWeightedCorrectionMovesLightBodyMore) {
    std::vector<BodyState> bodies;
    bodies.push_back(MakeBody(0.1f, Vec3::Zero(), {0.0f, 0.0f, 0.0f}));
    bodies.push_back(MakeBody(1.0f, Vec3::Zero(), {0.0f, 1.0f, 0.0f}));

    auto joint = nuka::runtime::articulation::BuildRevoluteConstraint(
        0, 1,
        Vec3::UnitZ(),
        Transform(Vec3::Zero(), Quat::Identity()),
        Transform(Vec3::Zero(), Quat::Identity()),
        -3.14f, 3.14f);

    std::vector<nuka::constraint::ConstraintBlock> blocks{joint};
    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 0;
    config.position_iterations = 1;
    config.baumgarte = 1.0f;
    config.slop = 0.0f;

    nuka::solver::SolveConstraints(blocks, bodies, config);

    const float heavy_body_motion = std::abs(bodies[0].position.y);
    const float light_body_motion = std::abs(1.0f - bodies[1].position.y);
    EXPECT_GT(light_body_motion, heavy_body_motion * 5.0f);
    EXPECT_LT(Distance(WorldAnchor(bodies[0], Vec3::Zero()),
                       WorldAnchor(bodies[1], Vec3::Zero())),
              1e-4f);
}

TEST(JointProjection, EccentricAnchorProjectionUsesAngularInertia) {
    const Vec3 parent_anchor = Vec3::Zero();
    const Vec3 child_anchor{-1.0f, 0.0f, 0.0f};

    std::vector<BodyState> bodies;
    bodies.push_back(MakeBody(0.0f, Vec3::Zero(), {0.0f, 0.0f, 0.0f}));
    bodies.push_back(MakeBody(1.0f, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 0.0f}));

    const float initial_gap = Distance(WorldAnchor(bodies[0], parent_anchor),
                                       WorldAnchor(bodies[1], child_anchor));

    auto joint = nuka::runtime::articulation::BuildRevoluteConstraint(
        0, 1,
        Vec3::UnitZ(),
        Transform(parent_anchor, Quat::Identity()),
        Transform(child_anchor, Quat::Identity()),
        -3.14f, 3.14f);

    std::vector<nuka::constraint::ConstraintBlock> blocks{joint};
    nuka::solver::SolverConfig config{};
    config.velocity_iterations = 0;
    config.position_iterations = 1;
    config.baumgarte = 1.0f;
    config.slop = 0.0f;

    nuka::solver::SolveConstraints(blocks, bodies, config);

    const float final_gap = Distance(WorldAnchor(bodies[0], parent_anchor),
                                     WorldAnchor(bodies[1], child_anchor));
    EXPECT_LT(final_gap, initial_gap * 0.25f);
    EXPECT_GT(std::abs(bodies[1].orientation.z), 1e-3f);
}
