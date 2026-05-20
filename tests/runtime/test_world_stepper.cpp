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

TEST(WorldStepper, GeneratesCookedShapeContactsAndSolvesAgainstStaticPlane) {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    const auto ground_id = scene.AddRigidBody(std::move(ground));

    nuka::scene::CollisionShapeRecord plane;
    plane.body_id = ground_id;
    plane.type = nuka::scene::ShapeType::Plane;
    scene.AddCollisionShape(std::move(plane));

    nuka::scene::RigidBodyRecord box;
    box.name = "box";
    box.mass = 1.0f;
    box.inertia = {1.0f, 1.0f, 1.0f};
    box.local_transform.position = {0.0f, 0.45f, 0.0f};
    const auto box_id = scene.AddRigidBody(std::move(box));

    nuka::scene::CollisionShapeRecord box_shape;
    box_shape.body_id = box_id;
    box_shape.type = nuka::scene::ShapeType::Box;
    box_shape.half_extents = {0.5f, 0.5f, 0.5f};
    scene.AddCollisionShape(std::move(box_shape));

    auto world = nuka::runtime::BuildWorld(nuka::scene::CookScene(scene));
    world.instance.linear_velocities[box_id] = {0.0f, -1.0f, 0.0f};

    nuka::runtime::WorldStepOptions options;
    options.gravity = {0.0f, 0.0f, 0.0f};
    options.dt = 1.0f / 60.0f;

    const auto report = nuka::runtime::StepWorldInstance(world.template_view,
                                                         world.instance,
                                                         options);

    EXPECT_EQ(report.broadphase_pair_count, 1u);
    EXPECT_EQ(report.contact_manifold_count, 1u);
    EXPECT_GE(report.contact_point_count, 1u);
    EXPECT_GE(report.constraint_row_count, 3u);
    EXPECT_GE(world.instance.linear_velocities[box_id].y, -1e-5f);
    EXPECT_GT(world.instance.poses[box_id].position.y, 0.48f);
}

TEST(WorldStepper, UsesPlaneShapeTransformWhenGeneratingContacts) {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord ground;
    ground.name = "raised_ground";
    ground.is_static = true;
    const auto ground_id = scene.AddRigidBody(std::move(ground));

    nuka::scene::CollisionShapeRecord plane;
    plane.body_id = ground_id;
    plane.type = nuka::scene::ShapeType::Plane;
    plane.local_transform.position = {0.0f, 1.0f, 0.0f};
    scene.AddCollisionShape(std::move(plane));

    nuka::scene::RigidBodyRecord box;
    box.name = "box";
    box.mass = 1.0f;
    box.inertia = {1.0f, 1.0f, 1.0f};
    box.local_transform.position = {0.0f, 1.45f, 0.0f};
    const auto box_id = scene.AddRigidBody(std::move(box));

    nuka::scene::CollisionShapeRecord box_shape;
    box_shape.body_id = box_id;
    box_shape.type = nuka::scene::ShapeType::Box;
    box_shape.half_extents = {0.5f, 0.5f, 0.5f};
    scene.AddCollisionShape(std::move(box_shape));

    auto world = nuka::runtime::BuildWorld(nuka::scene::CookScene(scene));
    world.instance.linear_velocities[box_id] = {0.0f, -1.0f, 0.0f};

    nuka::runtime::WorldStepOptions options;
    options.gravity = {0.0f, 0.0f, 0.0f};
    options.dt = 1.0f / 60.0f;

    const auto report = nuka::runtime::StepWorldInstance(world.template_view,
                                                         world.instance,
                                                         options);

    EXPECT_EQ(report.contact_manifold_count, 1u);
    EXPECT_GT(world.instance.poses[box_id].position.y, 1.48f);
}
