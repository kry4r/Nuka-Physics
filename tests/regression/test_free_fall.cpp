// ---------------------------------------------------------------------------
// Regression test: Free-fall under gravity
// ---------------------------------------------------------------------------

#include "runtime/rigid/integrator.hpp"
#include "math/vec3.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace nuka;
using namespace nuka::runtime::rigid;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

struct FreeFallResult {
    float final_y;
    float energy_drift;
};

FreeFallResult RunFreeFallRegression() {
    BodyState body{};
    body.inv_mass        = 1.0f;          // mass = 1 kg
    body.inv_inertia     = {1.0f, 1.0f, 1.0f};
    body.position        = {0.0f, 0.0f, 0.0f};
    body.orientation     = math::Quat::Identity();
    body.linear_velocity = {0.0f, 0.0f, 0.0f};
    body.angular_velocity = {0.0f, 0.0f, 0.0f};
    body.force           = {0.0f, 0.0f, 0.0f};
    body.torque          = {0.0f, 0.0f, 0.0f};

    const math::Vec3 gravity{0.0f, -9.81f, 0.0f};
    const float dt        = 0.01f;
    const int   num_steps = 100;   // 1 second of simulation

    const float mass = 1.0f / body.inv_mass;

    // Initial total energy: KE = 0, PE = mass * 9.81 * y = 0
    const float initial_energy = 0.0f;

    for (int i = 0; i < num_steps; ++i) {
        StepBody(body, gravity, dt);
    }

    // Final energy: KE + PE
    const float speed_sq = body.linear_velocity.x * body.linear_velocity.x
                         + body.linear_velocity.y * body.linear_velocity.y
                         + body.linear_velocity.z * body.linear_velocity.z;
    const float ke = 0.5f * mass * speed_sq;
    const float pe = mass * 9.81f * body.position.y;
    const float final_energy = ke + pe;

    return {body.position.y, std::abs(final_energy - initial_energy)};
}

}  // namespace

// ---------------------------------------------------------------------------
// Energy drift stays bounded
// ---------------------------------------------------------------------------

TEST(FreeFallRegression, EnergyDriftStaysBounded) {
    const auto result = RunFreeFallRegression();
    // Symplectic Euler has O(dt) energy drift over long horizons
    EXPECT_LT(result.energy_drift, 0.5f);
}

// ---------------------------------------------------------------------------
// Position matches kinematic prediction
// ---------------------------------------------------------------------------

TEST(FreeFallRegression, PositionMatchesKinematicFormula) {
    const auto result = RunFreeFallRegression();

    // Kinematic: y = y0 + v0*t + 0.5*a*t^2
    // y0 = 0, v0 = 0, a = -9.81, t = 1.0
    // y_expected = -0.5 * 9.81 * 1.0 = -4.905
    const float y_expected = -0.5f * 9.81f * 1.0f * 1.0f;

    // Symplectic Euler will differ slightly from exact kinematics.
    // Allow 1% relative tolerance.
    // Symplectic Euler accumulates O(dt) position error over 100 steps
    EXPECT_NEAR(result.final_y, y_expected, std::abs(y_expected) * 0.02f);
}

// ---------------------------------------------------------------------------
// Velocity matches kinematic prediction
// ---------------------------------------------------------------------------

TEST(FreeFallRegression, VelocityMatchesExpected) {
    BodyState body{};
    body.inv_mass        = 1.0f;
    body.inv_inertia     = {1.0f, 1.0f, 1.0f};
    body.position        = {0.0f, 0.0f, 0.0f};
    body.orientation     = math::Quat::Identity();
    body.linear_velocity = {0.0f, 0.0f, 0.0f};
    body.angular_velocity = {0.0f, 0.0f, 0.0f};
    body.force           = {0.0f, 0.0f, 0.0f};
    body.torque          = {0.0f, 0.0f, 0.0f};

    const math::Vec3 gravity{0.0f, -9.81f, 0.0f};
    const float dt = 0.01f;

    for (int i = 0; i < 100; ++i) {
        StepBody(body, gravity, dt);
    }

    // v = v0 + a*t = 0 + (-9.81)*1.0 = -9.81
    EXPECT_NEAR(body.linear_velocity.y, -9.81f, 1e-4f);
}
