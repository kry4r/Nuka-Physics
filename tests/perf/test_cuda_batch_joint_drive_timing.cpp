// ---------------------------------------------------------------------------
// Performance test: CUDA batched joint and drive solve timing
// ---------------------------------------------------------------------------

#include "runtime/gpu/batched_device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <utility>
#include <vector>

using namespace nuka;

namespace {

struct JointDriveScene {
    scene::SceneIR scene;
    scene::BodyId child_body = scene::kInvalidBody;
};

JointDriveScene BuildJointDriveScene() {
    JointDriveScene result;

    scene::RigidBodyRecord parent;
    parent.name = "parent";
    parent.is_static = true;
    const auto parent_body = result.scene.AddRigidBody(std::move(parent));

    scene::RigidBodyRecord child;
    child.name = "child";
    child.mass = 1.0f;
    child.inertia = {1.0f, 1.0f, 1.0f};
    child.local_transform.position = {0.0f, 1.0f, 0.0f};
    result.child_body = result.scene.AddRigidBody(std::move(child));

    scene::JointRecord joint;
    joint.name = "hinge";
    joint.type = scene::JointType::Revolute;
    joint.parent_body = parent_body;
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

} // namespace

TEST(CudaBatchJointDriveTiming, BatchedJointDriveSolveUnderOneSecond) {
    constexpr uint32_t kInstanceCount = 128;
    constexpr uint32_t kStepCount = 60;

    const auto fixture = BuildJointDriveScene();
    auto world = runtime::BuildWorld(scene::CookScene(fixture.scene));
    std::vector<runtime::WorldInstance> instances(kInstanceCount, world.instance);
    for (uint32_t instance_index = 0; instance_index < kInstanceCount; ++instance_index) {
        auto& instance = instances[instance_index];
        instance.poses[fixture.child_body].position.x =
            static_cast<float>(instance_index % 16) * 0.01f;
        instance.poses[fixture.child_body].position.y =
            1.0f + static_cast<float>(instance_index % 4) * 0.1f;
        instance.angular_velocities[fixture.child_body].z =
            static_cast<float>(instance_index % 3);
    }

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    runtime::gpu::CudaBatchedWorldStepOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 240.0f;
    options.step_count = kStepCount;
    options.enable_joints = true;
    options.enable_drives = true;
    options.solver_velocity_iterations = 12u;
    options.solver_position_iterations = 12u;
    options.solver_baumgarte = 0.5f;
    options.solver_slop = 0.0f;

    const auto start = std::chrono::high_resolution_clock::now();
    const auto report = runtime::gpu::StepBatchedCudaWorld(batch, options);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(report.instance_count, kInstanceCount);
    EXPECT_EQ(report.body_count_per_instance, 2u);
    EXPECT_EQ(report.simulated_step_count, kStepCount);
    EXPECT_EQ(report.joint_constraint_count, kInstanceCount * kStepCount);
    EXPECT_EQ(report.drive_constraint_count, kInstanceCount * kStepCount);
    EXPECT_GE(report.constraint_row_count, kInstanceCount * kStepCount * 6u);
    EXPECT_LT(ms, 1000) << "CUDA batched joint/drive solve took " << ms << " ms";
}
