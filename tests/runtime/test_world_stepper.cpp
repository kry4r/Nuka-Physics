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

TEST(WorldStepper, SolvesCookedRevoluteJointAnchors) {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord base;
    base.name = "base";
    base.is_static = true;
    base.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto base_id = scene.AddRigidBody(std::move(base));

    nuka::scene::RigidBodyRecord link;
    link.name = "link";
    link.mass = 1.0f;
    link.inertia = {1.0f, 1.0f, 1.0f};
    link.local_transform.position = {0.0f, 1.0f, 0.0f};
    const auto link_id = scene.AddRigidBody(std::move(link));

    nuka::scene::JointRecord joint;
    joint.name = "hinge";
    joint.type = nuka::scene::JointType::Revolute;
    joint.parent_body = base_id;
    joint.child_body = link_id;
    joint.axis = nuka::math::Vec3::UnitZ();
    joint.parent_frame.position = nuka::math::Vec3::Zero();
    joint.child_frame.position = nuka::math::Vec3::Zero();
    scene.AddJoint(std::move(joint));

    auto world = nuka::runtime::BuildWorld(nuka::scene::CookScene(scene));
    world.instance.linear_velocities[link_id] = {0.0f, -1.0f, 0.0f};

    nuka::runtime::WorldStepOptions options;
    options.gravity = {0.0f, 0.0f, 0.0f};
    options.dt = 1.0f / 60.0f;
    options.enable_contacts = false;
    options.solver_position_iterations = 12;
    options.solver_baumgarte = 0.5f;
    options.solver_slop = 0.0f;

    const auto report = nuka::runtime::StepWorldInstance(world.template_view,
                                                         world.instance,
                                                         options);

    EXPECT_EQ(report.joint_constraint_count, 1u);
    EXPECT_EQ(report.constraint_block_count, 1u);
    EXPECT_GE(report.constraint_row_count, 5u);
    EXPECT_NEAR(world.instance.poses[link_id].position.y, 0.0f, 1e-3f);
}

TEST(WorldStepper, AppliesCookedVelocityActuatorAsDriveConstraint) {
    nuka::scene::SceneIR scene;

    const auto parent_id = scene.AddRigidBody("parent");

    nuka::scene::RigidBodyRecord child;
    child.name = "child";
    child.mass = 1.0f;
    child.inertia = {1.0f, 1.0f, 1.0f};
    const auto child_id = scene.AddRigidBody(std::move(child));

    nuka::scene::JointRecord joint;
    joint.name = "hinge";
    joint.type = nuka::scene::JointType::Revolute;
    joint.parent_body = parent_id;
    joint.child_body = child_id;
    joint.axis = nuka::math::Vec3::UnitZ();
    const auto joint_id = scene.AddJoint(std::move(joint));

    nuka::scene::ActuatorRecord actuator;
    actuator.name = "velocity_motor";
    actuator.type = nuka::scene::ActuatorType::Velocity;
    actuator.joint_id = joint_id;
    actuator.gain = 4.0f;
    actuator.force_limit = 20.0f;
    scene.AddActuator(std::move(actuator));

    auto world = nuka::runtime::BuildWorld(nuka::scene::CookScene(scene));

    nuka::runtime::WorldStepOptions options;
    options.gravity = {0.0f, 0.0f, 0.0f};
    options.dt = 1.0f / 60.0f;
    options.enable_contacts = false;
    options.solver_position_iterations = 0;

    const auto report = nuka::runtime::StepWorldInstance(world.template_view,
                                                         world.instance,
                                                         options);

    EXPECT_EQ(report.drive_constraint_count, 1u);
    EXPECT_GT(world.instance.angular_velocities[child_id].z, 0.0f);
}

TEST(WorldStepper, MaintainsCookedVelocityActuatorTargetAcrossSteps) {
    nuka::scene::SceneIR scene;

    const auto parent_id = scene.AddRigidBody("parent");

    nuka::scene::RigidBodyRecord child;
    child.name = "child";
    child.mass = 1.0f;
    child.inertia = {1.0f, 1.0f, 1.0f};
    const auto child_id = scene.AddRigidBody(std::move(child));

    nuka::scene::JointRecord joint;
    joint.name = "hinge";
    joint.type = nuka::scene::JointType::Revolute;
    joint.parent_body = parent_id;
    joint.child_body = child_id;
    joint.axis = nuka::math::Vec3::UnitZ();
    const auto joint_id = scene.AddJoint(std::move(joint));

    nuka::scene::ActuatorRecord actuator;
    actuator.name = "velocity_motor";
    actuator.type = nuka::scene::ActuatorType::Velocity;
    actuator.joint_id = joint_id;
    actuator.gain = 2.0f;
    actuator.force_limit = 20.0f;
    scene.AddActuator(std::move(actuator));

    auto world = nuka::runtime::BuildWorld(nuka::scene::CookScene(scene));

    nuka::runtime::WorldStepOptions options;
    options.gravity = {0.0f, 0.0f, 0.0f};
    options.dt = 1.0f / 60.0f;
    options.enable_contacts = false;
    options.solver_position_iterations = 0;

    nuka::runtime::WorldStepReport report;
    for (int step = 0; step < 8; ++step) {
        report = nuka::runtime::StepWorldInstance(world.template_view,
                                                  world.instance,
                                                  options);
    }

    EXPECT_EQ(report.drive_constraint_count, 1u);
    const float relative_velocity =
        world.instance.angular_velocities[child_id].z
        - world.instance.angular_velocities[parent_id].z;
    EXPECT_NEAR(relative_velocity, 2.0f, 1e-4f);
}
