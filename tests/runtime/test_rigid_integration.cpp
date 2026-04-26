// ---------------------------------------------------------------------------
// Tests for nuka::runtime::rigid integrator
// ---------------------------------------------------------------------------

#include "runtime/rigid/integrator.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace nuka;
using namespace nuka::runtime::rigid;

// ---------------------------------------------------------------------------
// Gravity advances velocity
// ---------------------------------------------------------------------------

TEST(RigidIntegration, GravityAdvancesVelocity) {
    BodyState body{};
    body.inv_mass        = 1.0f;
    body.linear_velocity = {0.0f, 0.0f, 0.0f};

    IntegrateGravity(body, {0.0f, -9.81f, 0.0f}, 0.1f);
    EXPECT_NEAR(body.linear_velocity.y, -0.981f, 1e-4f);
}

// ---------------------------------------------------------------------------
// Static body not affected by gravity
// ---------------------------------------------------------------------------

TEST(RigidIntegration, StaticBodyNotAffectedByGravity) {
    BodyState body{};
    body.inv_mass        = 0.0f;  // static
    body.linear_velocity = {0.0f, 0.0f, 0.0f};

    IntegrateGravity(body, {0.0f, -9.81f, 0.0f}, 0.1f);
    EXPECT_FLOAT_EQ(body.linear_velocity.y, 0.0f);
}

// ---------------------------------------------------------------------------
// Force integration – linear
// ---------------------------------------------------------------------------

TEST(RigidIntegration, ForceIntegrationLinear) {
    BodyState body{};
    body.inv_mass        = 0.5f;  // mass = 2
    body.linear_velocity = {0.0f, 0.0f, 0.0f};
    body.force           = {10.0f, 0.0f, 0.0f};

    IntegrateForces(body, 0.1f);
    // v += F * inv_mass * dt = 10 * 0.5 * 0.1 = 0.5
    EXPECT_NEAR(body.linear_velocity.x, 0.5f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Force integration – angular
// ---------------------------------------------------------------------------

TEST(RigidIntegration, ForceIntegrationAngular) {
    BodyState body{};
    body.inv_mass          = 1.0f;
    body.angular_velocity  = {0.0f, 0.0f, 0.0f};
    body.torque            = {0.0f, 0.0f, 6.0f};
    body.inv_inertia       = {1.0f, 1.0f, 0.5f};

    IntegrateForces(body, 0.2f);
    // omega_z += torque_z * inv_inertia_z * dt = 6 * 0.5 * 0.2 = 0.6
    EXPECT_NEAR(body.angular_velocity.z, 0.6f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Static body not affected by forces
// ---------------------------------------------------------------------------

TEST(RigidIntegration, StaticBodyNotAffectedByForces) {
    BodyState body{};
    body.inv_mass    = 0.0f;  // static
    body.force       = {100.0f, 0.0f, 0.0f};
    body.torque      = {0.0f, 0.0f, 50.0f};
    body.inv_inertia = {1.0f, 1.0f, 1.0f};

    IntegrateForces(body, 0.1f);
    EXPECT_FLOAT_EQ(body.linear_velocity.x, 0.0f);
    EXPECT_FLOAT_EQ(body.angular_velocity.z, 0.0f);
}

// ---------------------------------------------------------------------------
// Position integration
// ---------------------------------------------------------------------------

TEST(RigidIntegration, PositionIntegration) {
    BodyState body{};
    body.inv_mass        = 1.0f;
    body.position        = {1.0f, 2.0f, 3.0f};
    body.linear_velocity = {10.0f, 0.0f, -5.0f};

    IntegrateVelocity(body, 0.1f);
    // pos += v * dt
    EXPECT_NEAR(body.position.x, 2.0f, 1e-5f);
    EXPECT_NEAR(body.position.y, 2.0f, 1e-5f);
    EXPECT_NEAR(body.position.z, 2.5f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Static body position unchanged
// ---------------------------------------------------------------------------

TEST(RigidIntegration, StaticBodyPositionUnchanged) {
    BodyState body{};
    body.inv_mass        = 0.0f;
    body.position        = {5.0f, 5.0f, 5.0f};
    body.linear_velocity = {10.0f, 10.0f, 10.0f};

    IntegrateVelocity(body, 1.0f);
    EXPECT_FLOAT_EQ(body.position.x, 5.0f);
    EXPECT_FLOAT_EQ(body.position.y, 5.0f);
    EXPECT_FLOAT_EQ(body.position.z, 5.0f);
}

// ---------------------------------------------------------------------------
// Angular velocity integration (orientation)
// ---------------------------------------------------------------------------

TEST(RigidIntegration, AngularVelocityIntegration) {
    BodyState body{};
    body.inv_mass         = 1.0f;
    body.orientation      = math::Quat::Identity();
    body.angular_velocity = {0.0f, 0.0f, 1.0f};  // spinning around Z

    IntegrateVelocity(body, 0.01f);

    // After a small dt the quaternion should still be roughly unit length
    const float norm = body.orientation.Norm();
    EXPECT_NEAR(norm, 1.0f, 1e-4f);

    // Orientation should have changed (no longer identity)
    const bool changed = (body.orientation.x != 0.0f ||
                          body.orientation.y != 0.0f ||
                          body.orientation.z != 0.0f);
    EXPECT_TRUE(changed);
}

// ---------------------------------------------------------------------------
// Full step combines gravity + forces + velocity integration
// ---------------------------------------------------------------------------

TEST(RigidIntegration, FullStepCombinesAll) {
    BodyState body{};
    body.inv_mass        = 1.0f;
    body.position        = {0.0f, 10.0f, 0.0f};
    body.linear_velocity = {0.0f, 0.0f, 0.0f};
    body.force           = {0.0f, 0.0f, 0.0f};
    body.torque          = {0.0f, 0.0f, 0.0f};
    body.inv_inertia     = {1.0f, 1.0f, 1.0f};

    const math::Vec3 gravity{0.0f, -9.81f, 0.0f};
    const float dt = 0.01f;

    StepBody(body, gravity, dt);

    // Velocity should be gravity * dt
    EXPECT_NEAR(body.linear_velocity.y, -9.81f * dt, 1e-5f);

    // Position should have moved by v * dt (symplectic Euler: use updated v)
    EXPECT_NEAR(body.position.y, 10.0f + body.linear_velocity.y * dt, 1e-5f);
}

// ---------------------------------------------------------------------------
// Full step on static body does nothing
// ---------------------------------------------------------------------------

TEST(RigidIntegration, FullStepStaticBodyUnchanged) {
    BodyState body{};
    body.inv_mass        = 0.0f;
    body.position        = {1.0f, 2.0f, 3.0f};
    body.linear_velocity = {0.0f, 0.0f, 0.0f};
    body.force           = {100.0f, 0.0f, 0.0f};

    StepBody(body, {0.0f, -9.81f, 0.0f}, 0.1f);

    EXPECT_FLOAT_EQ(body.position.x, 1.0f);
    EXPECT_FLOAT_EQ(body.position.y, 2.0f);
    EXPECT_FLOAT_EQ(body.position.z, 3.0f);
    EXPECT_FLOAT_EQ(body.linear_velocity.x, 0.0f);
}
