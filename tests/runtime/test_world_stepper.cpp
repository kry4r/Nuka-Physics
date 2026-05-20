// ---------------------------------------------------------------------------
// Tests for nuka::runtime::StepWorldInstance
// ---------------------------------------------------------------------------

#include "runtime/world_stepper.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <utility>

namespace {

nuka::scene::SceneIR BuildStaticAndDynamicScene() {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    ground.local_transform.position = {0.0f, 3.0f, 0.0f};
    scene.AddRigidBody(std::move(ground));

    nuka::scene::RigidBodyRecord falling;
    falling.name = "falling";
    falling.mass = 2.0f;
    falling.inertia = {4.0f, 5.0f, 6.0f};
    falling.local_transform.position = {0.0f, 10.0f, 0.0f};
    scene.AddRigidBody(std::move(falling));

    return scene;
}

} // namespace

TEST(WorldStepper, AdvancesDynamicBodiesAndKeepsStaticBodiesFixed) {
    const auto blob = nuka::scene::CookScene(BuildStaticAndDynamicScene());
    auto world = nuka::runtime::BuildWorld(blob);

    nuka::runtime::WorldStepOptions options;
    options.gravity = {0.0f, -9.81f, 0.0f};
    options.dt = 0.1f;
    options.step_count = 2;

    nuka::runtime::StepWorldInstance(world.template_view, world.instance, options);

    EXPECT_FLOAT_EQ(world.instance.poses[0].position.y, 3.0f);
    EXPECT_FLOAT_EQ(world.instance.linear_velocities[0].y, 0.0f);

    EXPECT_NEAR(world.instance.linear_velocities[1].y, -1.962f, 1e-5f);
    EXPECT_NEAR(world.instance.poses[1].position.y, 9.7057f, 1e-4f);
}

TEST(WorldStepper, AppliesForcesTorquesAndClearsAccumulatorsAfterEachStep) {
    const auto blob = nuka::scene::CookScene(BuildStaticAndDynamicScene());
    auto world = nuka::runtime::BuildWorld(blob);

    world.instance.forces[1] = {4.0f, 0.0f, 0.0f};
    world.instance.torques[1] = {0.0f, 10.0f, 0.0f};

    nuka::runtime::WorldStepOptions options;
    options.gravity = {0.0f, 0.0f, 0.0f};
    options.dt = 0.5f;
    options.step_count = 1;

    nuka::runtime::StepWorldInstance(world.template_view, world.instance, options);

    EXPECT_NEAR(world.instance.linear_velocities[1].x, 1.0f, 1e-5f);
    EXPECT_NEAR(world.instance.poses[1].position.x, 0.5f, 1e-5f);
    EXPECT_NEAR(world.instance.angular_velocities[1].y, 1.0f, 1e-5f);
    EXPECT_EQ(world.instance.forces[1], nuka::math::Vec3::Zero());
    EXPECT_EQ(world.instance.torques[1], nuka::math::Vec3::Zero());
}
