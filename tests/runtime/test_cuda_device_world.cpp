// ---------------------------------------------------------------------------
// CUDA DeviceWorld upload tests.
// ---------------------------------------------------------------------------

#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

using namespace nuka;

TEST(CudaDeviceWorld, UploadsCookedRuntimeTablesAndDownloadsSummary) {
    scene::SceneIR scene;

    scene::RigidBodyRecord parent;
    parent.name = "parent";
    parent.mass = 0.0f;
    parent.local_transform.position = {1.0f, 2.0f, 3.0f};
    const auto parent_id = scene.AddRigidBody(std::move(parent));

    scene::RigidBodyRecord child;
    child.name = "child";
    child.mass = 2.0f;
    child.local_transform.position = {4.0f, 5.0f, 6.0f};
    const auto child_id = scene.AddRigidBody(std::move(child));

    scene::CollisionShapeRecord shape;
    shape.body_id = child_id;
    shape.type = scene::ShapeType::Capsule;
    shape.radius = 0.125f;
    shape.half_height = 0.75f;
    scene.AddCollisionShape(std::move(shape));

    scene::JointRecord joint;
    joint.parent_body = parent_id;
    joint.child_body = child_id;
    joint.axis = math::Vec3::UnitZ();
    const auto joint_id = scene.AddJoint(std::move(joint));

    scene::ActuatorRecord actuator;
    actuator.joint_id = joint_id;
    actuator.type = scene::ActuatorType::Velocity;
    actuator.gain = 7.0f;
    actuator.force_limit = 11.0f;
    scene.AddActuator(std::move(actuator));

    const auto blob = scene::CookScene(scene);
    const auto world = runtime::BuildWorld(blob);

    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    const auto summary = device_world.DownloadSummary();

    EXPECT_EQ(summary.body_count, 2u);
    EXPECT_EQ(summary.shape_count, 1u);
    EXPECT_EQ(summary.joint_count, 1u);
    EXPECT_EQ(summary.actuator_count, 1u);

    EXPECT_FLOAT_EQ(summary.first_body_position.x, 1.0f);
    EXPECT_FLOAT_EQ(summary.first_body_position.y, 2.0f);
    EXPECT_FLOAT_EQ(summary.first_body_position.z, 3.0f);
    EXPECT_FLOAT_EQ(summary.second_body_inv_mass, 0.5f);
    EXPECT_EQ(summary.first_shape_body_id, child_id);
    EXPECT_FLOAT_EQ(summary.first_shape_radius, 0.125f);
    EXPECT_FLOAT_EQ(summary.first_shape_half_height, 0.75f);
    EXPECT_EQ(summary.first_joint_child_body, child_id);
    EXPECT_FLOAT_EQ(summary.first_actuator_gain, 7.0f);
    EXPECT_FLOAT_EQ(summary.first_actuator_force_limit, 11.0f);

    EXPECT_GE(device_world.DeviceBytes(), sizeof(math::Transform) * 2u);
}
