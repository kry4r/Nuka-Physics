// ---------------------------------------------------------------------------
// CUDA constraint assembly and solver tests.
// ---------------------------------------------------------------------------

#include "collision/gpu/broadphase.cuh"
#include "constraint/gpu/contact_generation.cuh"
#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"
#include "solver/gpu/cuda_constraint_solver.cuh"

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

using namespace nuka;

namespace {

struct PlaneBoxScene {
    scene::SceneIR scene;
    scene::BodyId box_body = scene::kInvalidBody;
};

PlaneBoxScene BuildPlaneBoxScene() {
    PlaneBoxScene result;

    scene::RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    const auto ground_id = result.scene.AddRigidBody(std::move(ground));

    scene::CollisionShapeRecord plane;
    plane.body_id = ground_id;
    plane.type = scene::ShapeType::Plane;
    result.scene.AddCollisionShape(std::move(plane));

    scene::RigidBodyRecord box;
    box.name = "box";
    box.mass = 1.0f;
    box.inertia = {1.0f, 1.0f, 1.0f};
    box.local_transform.position = {0.0f, 0.45f, 0.0f};
    result.box_body = result.scene.AddRigidBody(std::move(box));

    scene::CollisionShapeRecord box_shape;
    box_shape.body_id = result.box_body;
    box_shape.type = scene::ShapeType::Box;
    box_shape.half_extents = {0.5f, 0.5f, 0.5f};
    result.scene.AddCollisionShape(std::move(box_shape));

    return result;
}

scene::SceneIR BuildSingleBodyScene() {
    scene::SceneIR scene;
    scene::RigidBodyRecord body;
    body.name = "body";
    body.mass = 1.0f;
    body.inertia = {1.0f, 1.0f, 1.0f};
    scene.AddRigidBody(std::move(body));
    return scene;
}

struct JointDriveScene {
    scene::SceneIR scene;
    scene::BodyId child_body = scene::kInvalidBody;
};

JointDriveScene BuildJointDriveScene() {
    JointDriveScene result;

    scene::RigidBodyRecord parent;
    parent.name = "parent";
    parent.is_static = true;
    const auto parent_id = result.scene.AddRigidBody(std::move(parent));

    scene::RigidBodyRecord child;
    child.name = "child";
    child.mass = 1.0f;
    child.inertia = {1.0f, 1.0f, 1.0f};
    result.child_body = result.scene.AddRigidBody(std::move(child));

    scene::JointRecord joint;
    joint.name = "hinge";
    joint.type = scene::JointType::Revolute;
    joint.parent_body = parent_id;
    joint.child_body = result.child_body;
    joint.axis = math::Vec3::UnitZ();
    const auto joint_id = result.scene.AddJoint(std::move(joint));

    scene::ActuatorRecord actuator;
    actuator.name = "velocity_motor";
    actuator.type = scene::ActuatorType::Velocity;
    actuator.joint_id = joint_id;
    actuator.gain = 4.0f;
    actuator.force_limit = 20.0f;
    result.scene.AddActuator(std::move(actuator));

    return result;
}

struct JointProjectionScene {
    scene::SceneIR scene;
    scene::BodyId child_body = scene::kInvalidBody;
};

JointProjectionScene BuildJointProjectionScene() {
    JointProjectionScene result;

    scene::RigidBodyRecord parent;
    parent.name = "base";
    parent.is_static = true;
    const auto parent_id = result.scene.AddRigidBody(std::move(parent));

    scene::RigidBodyRecord child;
    child.name = "link";
    child.mass = 1.0f;
    child.inertia = {1.0f, 1.0f, 1.0f};
    child.local_transform.position = {0.0f, 1.0f, 0.0f};
    result.child_body = result.scene.AddRigidBody(std::move(child));

    scene::JointRecord joint;
    joint.name = "hinge";
    joint.type = scene::JointType::Revolute;
    joint.parent_body = parent_id;
    joint.child_body = result.child_body;
    joint.axis = math::Vec3::UnitZ();
    result.scene.AddJoint(std::move(joint));

    return result;
}

runtime::gpu::DeviceWorld UploadSceneToDevice(const scene::SceneIR& scene,
                                              runtime::BuiltWorld& world) {
    world = runtime::BuildWorld(scene::CookScene(scene));
    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);
    return device_world;
}

constraint::ConstraintBlock BuildGroundContact(float friction, float restitution) {
    constraint::ConstraintBlock contact;
    contact.type = constraint::ConstraintType::Contact;
    contact.body_a = 0u;
    contact.body_b = ~0u;
    contact.row_count = 3u;
    contact.normal_row_count = 1u;
    contact.first_friction_row = 1u;
    contact.friction_row_count = 2u;
    contact.friction = friction;
    contact.restitution = restitution;

    contact.jacobian_linear_a[0] = {0.0f, 1.0f, 0.0f};
    contact.jacobian_linear_b[0] = {0.0f, -1.0f, 0.0f};
    contact.lower_limit[0] = 0.0f;
    contact.upper_limit[0] = 1.0e6f;

    contact.jacobian_linear_a[1] = {1.0f, 0.0f, 0.0f};
    contact.jacobian_linear_b[1] = {-1.0f, 0.0f, 0.0f};

    contact.jacobian_linear_a[2] = {0.0f, 0.0f, 1.0f};
    contact.jacobian_linear_b[2] = {0.0f, 0.0f, -1.0f};
    return contact;
}

} // namespace

TEST(CudaConstraintSolver, SolvesPlaneBoxContactOnDevice) {
    const auto scene = BuildPlaneBoxScene();
    runtime::BuiltWorld world;
    auto device_world = UploadSceneToDevice(scene.scene, world);
    world.instance.linear_velocities[scene.box_body] = {0.0f, -1.0f, 0.0f};
    runtime::gpu::UploadDeviceState(device_world, world.instance);

    auto broadphase = collision::gpu::BuildCudaBroadphase(device_world);
    auto contacts = constraint::gpu::GenerateCudaContacts(device_world, broadphase);

    solver::gpu::CudaConstraintSolverConfig config;
    config.velocity_iterations = 10u;
    config.position_iterations = 4u;
    auto result = solver::gpu::SolveCudaConstraints(device_world, &contacts, config);
    const auto report = result.DownloadReport();
    const auto state = device_world.DownloadState();

    EXPECT_EQ(report.contact_constraint_count, 1u);
    EXPECT_GE(report.constraint_row_count, 3u);
    EXPECT_GE(state.linear_velocities[scene.box_body].y, -1.0e-5f);
    EXPECT_GT(state.poses[scene.box_body].position.y, 0.48f);
}

TEST(CudaConstraintSolver, FrictionImpulseIsClampedByNormalImpulseOnDevice) {
    runtime::BuiltWorld world;
    auto device_world = UploadSceneToDevice(BuildSingleBodyScene(), world);
    world.instance.linear_velocities[0] = {10.0f, -2.0f, 0.0f};
    runtime::gpu::UploadDeviceState(device_world, world.instance);

    solver::gpu::CudaConstraintSolverConfig config;
    config.velocity_iterations = 20u;
    config.position_iterations = 0u;
    auto result = solver::gpu::SolveCudaConstraintBlocks(
        device_world,
        {BuildGroundContact(0.5f, 0.0f)},
        config);

    const auto blocks = result.DownloadBlocks();
    const auto state = device_world.DownloadState();
    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_GT(blocks[0].impulse[0], 0.0f);
    EXPECT_LE(std::abs(blocks[0].impulse[1]), 0.5f * blocks[0].impulse[0] + 1.0e-5f);
    EXPECT_NEAR(state.linear_velocities[0].x, 9.0f, 1.0e-5f);
}

TEST(CudaConstraintSolver, RestitutionBouncesAlongContactNormalOnDevice) {
    runtime::BuiltWorld world;
    auto device_world = UploadSceneToDevice(BuildSingleBodyScene(), world);
    world.instance.linear_velocities[0] = {0.0f, -3.0f, 0.0f};
    runtime::gpu::UploadDeviceState(device_world, world.instance);

    solver::gpu::CudaConstraintSolverConfig config;
    config.velocity_iterations = 12u;
    config.position_iterations = 0u;
    auto result = solver::gpu::SolveCudaConstraintBlocks(
        device_world,
        {BuildGroundContact(0.0f, 0.5f)},
        config);

    const auto blocks = result.DownloadBlocks();
    const auto state = device_world.DownloadState();
    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_NEAR(state.linear_velocities[0].y, 1.5f, 1.0e-5f);
    EXPECT_GT(blocks[0].impulse[0], 3.0f);
}

TEST(CudaConstraintSolver, AppliesCookedVelocityActuatorOnDevice) {
    const auto scene = BuildJointDriveScene();
    runtime::BuiltWorld world;
    auto device_world = UploadSceneToDevice(scene.scene, world);

    solver::gpu::CudaConstraintSolverConfig config;
    config.velocity_iterations = 12u;
    config.position_iterations = 0u;
    auto result = solver::gpu::SolveCudaConstraints(device_world, nullptr, config);
    const auto report = result.DownloadReport();
    const auto state = device_world.DownloadState();

    EXPECT_EQ(report.drive_constraint_count, 1u);
    EXPECT_NEAR(state.angular_velocities[scene.child_body].z, 4.0f, 1.0e-4f);
}

TEST(CudaConstraintSolver, ProjectsCookedJointAnchorsOnDevice) {
    const auto scene = BuildJointProjectionScene();
    runtime::BuiltWorld world;
    auto device_world = UploadSceneToDevice(scene.scene, world);

    solver::gpu::CudaConstraintSolverConfig config;
    config.velocity_iterations = 0u;
    config.position_iterations = 12u;
    config.baumgarte = 0.5f;
    config.slop = 0.0f;
    auto result = solver::gpu::SolveCudaConstraints(device_world, nullptr, config);
    const auto report = result.DownloadReport();
    const auto state = device_world.DownloadState();

    EXPECT_EQ(report.joint_constraint_count, 1u);
    EXPECT_GE(report.constraint_row_count, 5u);
    EXPECT_NEAR(state.poses[scene.child_body].position.y, 0.0f, 1.0e-3f);
}
