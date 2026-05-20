// ---------------------------------------------------------------------------
// Tests for nuka::runtime::BuildWorld
// ---------------------------------------------------------------------------

#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

using namespace nuka;

// ---------------------------------------------------------------------------
// Single body
// ---------------------------------------------------------------------------

TEST(WorldBuild, BuildsTemplateAndSingleInstanceFromBlob) {
    scene::SceneIR scene;
    scene.AddRigidBody("root");
    const auto blob  = scene::CookScene(scene);
    auto       world = runtime::BuildWorld(blob);

    EXPECT_EQ(world.template_view.body_count, 1u);
    EXPECT_EQ(world.batch.instance_count, 1u);
    EXPECT_EQ(world.batch.template_count, 1u);
}

// ---------------------------------------------------------------------------
// Multiple bodies
// ---------------------------------------------------------------------------

TEST(WorldBuild, MultipleBodiesInTemplate) {
    scene::SceneIR scene;
    scene.AddRigidBody("a");
    scene.AddRigidBody("b");
    scene.AddRigidBody("c");
    const auto blob  = scene::CookScene(scene);
    auto       world = runtime::BuildWorld(blob);

    EXPECT_EQ(world.template_view.body_count, 3u);
    EXPECT_EQ(world.instance.body_count, 3u);
    EXPECT_EQ(world.instance.poses.size(), 3u);
    EXPECT_EQ(world.instance.linear_velocities.size(), 3u);
    EXPECT_EQ(world.instance.angular_velocities.size(), 3u);
    EXPECT_EQ(world.instance.forces.size(), 3u);
    EXPECT_EQ(world.instance.torques.size(), 3u);
}

// ---------------------------------------------------------------------------
// Velocities initialised to zero
// ---------------------------------------------------------------------------

TEST(WorldBuild, VelocitiesInitialisedToZero) {
    scene::SceneIR scene;
    scene.AddRigidBody("body");
    const auto blob  = scene::CookScene(scene);
    auto       world = runtime::BuildWorld(blob);

    EXPECT_EQ(world.instance.linear_velocities[0], math::Vec3::Zero());
    EXPECT_EQ(world.instance.angular_velocities[0], math::Vec3::Zero());
}

// ---------------------------------------------------------------------------
// Poses copied from template
// ---------------------------------------------------------------------------

TEST(WorldBuild, PosesCopiedFromTemplate) {
    scene::SceneIR scene;
    scene::RigidBodyRecord rec;
    rec.name            = "offset";
    rec.local_transform = math::Transform({1.0f, 2.0f, 3.0f}, math::Quat::Identity());
    scene.AddRigidBody(std::move(rec));

    const auto blob  = scene::CookScene(scene);
    auto       world = runtime::BuildWorld(blob);

    EXPECT_FLOAT_EQ(world.instance.poses[0].position.x, 1.0f);
    EXPECT_FLOAT_EQ(world.instance.poses[0].position.y, 2.0f);
    EXPECT_FLOAT_EQ(world.instance.poses[0].position.z, 3.0f);

    // Template should have the same pose
    EXPECT_FLOAT_EQ(world.template_view.body_table.poses[0].position.x, 1.0f);
}

TEST(WorldBuild, CopiesJointFramesAndActuatorsIntoTemplate) {
    scene::SceneIR scene;
    const auto parent = scene.AddRigidBody("parent");
    const auto child = scene.AddRigidBody("child");

    scene::JointRecord joint;
    joint.name = "hinge";
    joint.parent_body = parent;
    joint.child_body = child;
    joint.parent_frame.position = {0.25f, 0.0f, 0.0f};
    joint.child_frame.position = {-0.25f, 0.0f, 0.0f};
    const auto joint_id = scene.AddJoint(std::move(joint));

    scene::ActuatorRecord actuator;
    actuator.name = "motor";
    actuator.joint_id = joint_id;
    actuator.type = scene::ActuatorType::Velocity;
    actuator.gain = 3.0f;
    actuator.force_limit = 9.0f;
    scene.AddActuator(std::move(actuator));

    const auto blob = scene::CookScene(scene);
    auto world = runtime::BuildWorld(blob);

    ASSERT_EQ(world.template_view.joint_table.parent_frames.size(), 1u);
    ASSERT_EQ(world.template_view.actuator_table.joint_ids.size(), 1u);
    EXPECT_FLOAT_EQ(world.template_view.joint_table.parent_frames[0].position.x, 0.25f);
    EXPECT_FLOAT_EQ(world.template_view.joint_table.child_frames[0].position.x, -0.25f);
    EXPECT_EQ(world.template_view.actuator_count, 1u);
    EXPECT_EQ(world.template_view.actuator_table.joint_ids[0], joint_id);
    EXPECT_FLOAT_EQ(world.template_view.actuator_table.gains[0], 3.0f);
    EXPECT_FLOAT_EQ(world.template_view.actuator_table.force_limits[0], 9.0f);
}

// ---------------------------------------------------------------------------
// Forces and torques initialised to zero
// ---------------------------------------------------------------------------

TEST(WorldBuild, ForcesAndTorquesInitialisedToZero) {
    scene::SceneIR scene;
    scene.AddRigidBody("body");
    const auto blob  = scene::CookScene(scene);
    auto       world = runtime::BuildWorld(blob);

    EXPECT_EQ(world.instance.forces[0], math::Vec3::Zero());
    EXPECT_EQ(world.instance.torques[0], math::Vec3::Zero());
}

// ---------------------------------------------------------------------------
// Empty scene produces empty world
// ---------------------------------------------------------------------------

TEST(WorldBuild, EmptySceneProducesEmptyWorld) {
    scene::SceneIR scene;
    const auto blob  = scene::CookScene(scene);
    auto       world = runtime::BuildWorld(blob);

    EXPECT_EQ(world.template_view.body_count, 0u);
    EXPECT_EQ(world.instance.body_count, 0u);
    EXPECT_EQ(world.batch.template_count, 1u);
    EXPECT_EQ(world.batch.instance_count, 1u);
}
