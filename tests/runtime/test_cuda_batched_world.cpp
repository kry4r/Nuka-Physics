// ---------------------------------------------------------------------------
// CUDA batched world tests.
// ---------------------------------------------------------------------------

#include "runtime/gpu/batched_device_world.hpp"
#include "runtime/world_builder.hpp"
#include "runtime/world_stepper.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

using namespace nuka;

namespace {

void ExpectVecNear(math::Vec3 actual, math::Vec3 expected, float tolerance) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

scene::SceneIR BuildTwoBodyScene() {
    scene::SceneIR scene;

    scene::RigidBodyRecord first;
    first.name = "first";
    first.mass = 2.0f;
    first.inertia = {1.0f, 2.0f, 4.0f};
    first.local_transform.position = {1.0f, 2.0f, 3.0f};
    scene.AddRigidBody(std::move(first));

    scene::RigidBodyRecord second;
    second.name = "second";
    second.mass = 4.0f;
    second.inertia = {4.0f, 2.0f, 1.0f};
    second.local_transform.position = {-2.0f, 5.0f, 0.5f};
    scene.AddRigidBody(std::move(second));

    return scene;
}

struct PlaneBoxScene {
    scene::SceneIR scene;
    scene::BodyId ground_body = scene::kInvalidBody;
    scene::BodyId box_body = scene::kInvalidBody;
};

PlaneBoxScene BuildPlaneBoxScene() {
    PlaneBoxScene result;

    scene::RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    result.ground_body = result.scene.AddRigidBody(std::move(ground));

    scene::CollisionShapeRecord plane;
    plane.body_id = result.ground_body;
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

std::vector<runtime::WorldInstance> BuildInstances(const runtime::BuiltWorld& world) {
    std::vector<runtime::WorldInstance> instances(3, world.instance);
    instances[0].forces[0] = {2.0f, 0.0f, 0.0f};
    instances[1].poses[0].position.x += 10.0f;
    instances[1].forces[1] = {0.0f, 8.0f, 0.0f};
    instances[2].linear_velocities[1] = {0.0f, 2.0f, 0.0f};
    instances[2].torques[0] = {0.0f, 0.0f, 1.0f};
    return instances;
}

} // namespace

TEST(CudaBatchedWorld, UploadsMultipleInstancesOfSharedTemplate) {
    const auto blob = scene::CookScene(BuildTwoBodyScene());
    auto world = runtime::BuildWorld(blob);
    const auto instances = BuildInstances(world);

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);
    const auto state = batch.DownloadState();

    EXPECT_EQ(batch.InstanceCount(), 3u);
    EXPECT_EQ(batch.BodyCountPerInstance(), 2u);
    EXPECT_EQ(batch.TotalBodyCount(), 6u);
    EXPECT_EQ(state.instance_count, 3u);
    EXPECT_EQ(state.body_count_per_instance, 2u);
    ASSERT_EQ(state.poses.size(), 6u);
    ASSERT_EQ(state.linear_velocities.size(), 6u);
    EXPECT_NEAR(state.poses[2].position.x, 11.0f, 1.0e-5f);
    EXPECT_NEAR(state.linear_velocities[5].y, 2.0f, 1.0e-5f);
}

TEST(CudaBatchedWorld, StepsInstancesIndependentlyOnDevice) {
    const auto blob = scene::CookScene(BuildTwoBodyScene());
    auto world = runtime::BuildWorld(blob);
    auto instances = BuildInstances(world);
    auto references = instances;

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    runtime::gpu::CudaBatchedWorldStepOptions cuda_options;
    cuda_options.gravity = {0.0f, -10.0f, 0.0f};
    cuda_options.dt = 0.25f;
    cuda_options.step_count = 2u;
    cuda_options.clear_forces_after_step = true;

    runtime::WorldStepOptions cpu_options;
    cpu_options.gravity = cuda_options.gravity;
    cpu_options.dt = cuda_options.dt;
    cpu_options.step_count = cuda_options.step_count;
    cpu_options.clear_forces_after_step = cuda_options.clear_forces_after_step;
    cpu_options.enable_contacts = false;

    const auto report = runtime::gpu::StepBatchedCudaWorld(batch, cuda_options);
    for (auto& reference : references) {
        runtime::StepWorldInstance(world.template_view, reference, cpu_options);
    }

    const auto state = batch.DownloadState();
    EXPECT_EQ(report.instance_count, 3u);
    EXPECT_EQ(report.body_count_per_instance, 2u);
    EXPECT_EQ(report.total_body_count, 6u);
    EXPECT_EQ(report.simulated_step_count, 2u);
    EXPECT_EQ(report.kernel_launch_count, 2u);

    for (uint32_t instance = 0; instance < state.instance_count; ++instance) {
        for (uint32_t body = 0; body < state.body_count_per_instance; ++body) {
            const uint32_t flat = instance * state.body_count_per_instance + body;
            ExpectVecNear(state.poses[flat].position,
                          references[instance].poses[body].position,
                          1.0e-5f);
            ExpectVecNear(state.linear_velocities[flat],
                          references[instance].linear_velocities[body],
                          1.0e-5f);
            ExpectVecNear(state.angular_velocities[flat],
                          references[instance].angular_velocities[body],
                          1.0e-5f);
            EXPECT_EQ(state.forces[flat], math::Vec3::Zero());
            EXPECT_EQ(state.torques[flat], math::Vec3::Zero());
        }
    }
}

TEST(CudaBatchedWorld, UploadsSharedShapeTablesForBatchedContacts) {
    const auto fixture = BuildPlaneBoxScene();
    auto world = runtime::BuildWorld(scene::CookScene(fixture.scene));
    std::vector<runtime::WorldInstance> instances(2, world.instance);

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    EXPECT_EQ(batch.ShapeCountPerInstance(), 2u);
    EXPECT_EQ(batch.TotalShapeCount(), 4u);
    EXPECT_TRUE(batch.HasUploadedShapeTables());
}

TEST(CudaBatchedWorld, UploadsSharedCapsuleHalfHeightsForBatchedShapeTables) {
    scene::SceneIR scene;

    scene::RigidBodyRecord link;
    link.name = "capsule_link";
    link.mass = 1.0f;
    const auto link_body = scene.AddRigidBody(std::move(link));

    scene::CollisionShapeRecord capsule;
    capsule.body_id = link_body;
    capsule.type = scene::ShapeType::Capsule;
    capsule.radius = 0.18f;
    capsule.half_height = 0.42f;
    scene.AddCollisionShape(std::move(capsule));

    auto world = runtime::BuildWorld(scene::CookScene(scene));
    std::vector<runtime::WorldInstance> instances(2, world.instance);

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);
    const auto shape_tables = batch.DownloadShapeTables();

    EXPECT_TRUE(batch.HasUploadedShapeTables());
    ASSERT_EQ(shape_tables.types.size(), 1u);
    ASSERT_EQ(shape_tables.radii.size(), 1u);
    ASSERT_EQ(shape_tables.half_heights.size(), 1u);
    EXPECT_EQ(shape_tables.types[0], scene::ShapeType::Capsule);
    EXPECT_FLOAT_EQ(shape_tables.radii[0], 0.18f);
    EXPECT_FLOAT_EQ(shape_tables.half_heights[0], 0.42f);
}

TEST(CudaBatchedWorld, GeneratesContactsPerInstanceWithoutCrossPairs) {
    const auto fixture = BuildPlaneBoxScene();
    auto world = runtime::BuildWorld(scene::CookScene(fixture.scene));
    std::vector<runtime::WorldInstance> instances(2, world.instance);
    instances[1].poses[fixture.box_body].position.y = 2.0f;

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);
    auto broadphase = runtime::gpu::BuildBatchedCudaBroadphase(batch);
    auto contacts = runtime::gpu::GenerateBatchedCudaContacts(batch, broadphase);

    const auto report = contacts.DownloadReport();
    const auto manifolds = contacts.DownloadManifolds();

    EXPECT_EQ(report.instance_count, 2u);
    EXPECT_EQ(report.pair_count, 1u);
    EXPECT_EQ(report.contact_manifold_count, 1u);
    EXPECT_EQ(report.contact_point_count, 1u);
    ASSERT_EQ(manifolds.size(), 1u);
    EXPECT_EQ(manifolds[0].instance_index, 0u);
    EXPECT_EQ(manifolds[0].body_a, fixture.box_body);
    EXPECT_EQ(manifolds[0].body_b, fixture.ground_body);
    ASSERT_EQ(manifolds[0].point_count, 1u);
    EXPECT_NEAR(manifolds[0].points[0].penetration, 0.05f, 1.0e-6f);
}

TEST(CudaBatchedWorld, SolvesContactsIndependentlyOnDevice) {
    const auto fixture = BuildPlaneBoxScene();
    auto world = runtime::BuildWorld(scene::CookScene(fixture.scene));
    std::vector<runtime::WorldInstance> instances(2, world.instance);
    instances[0].linear_velocities[fixture.box_body] = {0.0f, -1.0f, 0.0f};
    instances[1].poses[fixture.box_body].position.y = 2.0f;
    instances[1].linear_velocities[fixture.box_body] = {0.0f, -3.0f, 0.0f};

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);
    auto broadphase = runtime::gpu::BuildBatchedCudaBroadphase(batch);
    auto contacts = runtime::gpu::GenerateBatchedCudaContacts(batch, broadphase);

    runtime::gpu::CudaBatchedConstraintSolverConfig config;
    config.velocity_iterations = 10u;
    config.position_iterations = 4u;
    auto result = runtime::gpu::SolveBatchedCudaContactConstraints(batch, contacts, config);

    const auto report = result.DownloadReport();
    const auto row_buffer = result.ConstraintRowBuffer();
    const auto state = batch.DownloadState();
    const uint32_t instance0_box = fixture.box_body;
    const uint32_t instance1_box = state.body_count_per_instance + fixture.box_body;

    EXPECT_EQ(report.contact_constraint_count, 1u);
    EXPECT_GE(report.constraint_row_count, 3u);
    EXPECT_EQ(row_buffer.kind,
              runtime::gpu::CudaConstraintRowBufferKind::UniversalRowCsr);
    EXPECT_EQ(row_buffer.layout,
              runtime::gpu::CudaConstraintRowLayout::UniversalRowCsr);
    EXPECT_EQ(row_buffer.schedule_mode,
              runtime::gpu::CudaConstraintRowScheduleMode::IslandColoredSweep);
    EXPECT_NE(row_buffer.device_rows, nullptr);
    EXPECT_EQ(row_buffer.owner_count, contacts.TotalPairSlotCount());
    EXPECT_EQ(row_buffer.rows_per_owner, 6u);
    EXPECT_GT(row_buffer.row_stride_bytes, 0u);
    EXPECT_EQ(report.row_scheduler_report.row_kind,
              runtime::gpu::CudaConstraintRowBufferKind::UniversalRowCsr);
    EXPECT_EQ(report.row_scheduler_report.row_layout,
              runtime::gpu::CudaConstraintRowLayout::UniversalRowCsr);
    EXPECT_EQ(report.row_scheduler_report.schedule_mode,
              runtime::gpu::CudaConstraintRowScheduleMode::IslandColoredSweep);
    EXPECT_EQ(report.row_scheduler_report.owner_count, row_buffer.owner_count);
    EXPECT_EQ(report.row_scheduler_report.row_count, row_buffer.row_count);
    EXPECT_EQ(report.row_scheduler_report.configured_iterations,
              config.velocity_iterations);
    EXPECT_EQ(report.row_scheduler_report.executed_iterations,
              config.velocity_iterations);
    EXPECT_EQ(report.row_scheduler_report.solver_launch_count,
              config.velocity_iterations);
    EXPECT_GE(report.row_scheduler_report.active_row_count,
              report.constraint_row_count);
    EXPECT_GT(report.row_scheduler_report.normal_impulse_count, 0u);
    EXPECT_GT(report.row_scheduler_report.normal_delta_impulse_magnitude, 0.0f);
    EXPECT_TRUE(std::isfinite(report.row_scheduler_report.max_residual));
    EXPECT_GE(state.linear_velocities[instance0_box].y, -1.0e-5f);
    EXPECT_GT(state.poses[instance0_box].position.y, 0.48f);
    EXPECT_NEAR(state.poses[instance1_box].position.y, 2.0f, 1.0e-5f);
    EXPECT_NEAR(state.linear_velocities[instance1_box].y, -3.0f, 1.0e-5f);
}

TEST(CudaBatchedWorld, StepPipelineCanSolveBatchedContacts) {
    const auto fixture = BuildPlaneBoxScene();
    auto world = runtime::BuildWorld(scene::CookScene(fixture.scene));
    std::vector<runtime::WorldInstance> instances(2, world.instance);
    instances[0].linear_velocities[fixture.box_body] = {0.0f, -1.0f, 0.0f};
    instances[1].poses[fixture.box_body].position.y = 2.0f;

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    runtime::gpu::CudaBatchedWorldStepOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 60.0f;
    options.step_count = 1u;
    options.enable_contacts = true;
    options.solver_velocity_iterations = 10u;
    options.solver_position_iterations = 4u;

    const auto report = runtime::gpu::StepBatchedCudaWorld(batch, options);
    const auto state = batch.DownloadState();

    EXPECT_EQ(report.simulated_step_count, 1u);
    EXPECT_EQ(report.contact_constraint_count, 1u);
    EXPECT_EQ(report.contact_manifold_count, 1u);
    EXPECT_GE(report.constraint_row_count, 3u);
    EXPECT_EQ(report.row_scheduler_report.row_kind,
              runtime::gpu::CudaConstraintRowBufferKind::UniversalRowCsr);
    EXPECT_EQ(report.row_scheduler_report.row_layout,
              runtime::gpu::CudaConstraintRowLayout::UniversalRowCsr);
    EXPECT_EQ(report.row_scheduler_report.schedule_mode,
              runtime::gpu::CudaConstraintRowScheduleMode::IslandColoredSweep);
    EXPECT_EQ(report.row_scheduler_report.configured_iterations,
              options.solver_velocity_iterations);
    EXPECT_EQ(report.row_scheduler_report.executed_iterations,
              options.solver_velocity_iterations);
    EXPECT_GE(report.row_scheduler_report.active_row_count,
              report.constraint_row_count);
    EXPECT_GT(report.row_scheduler_report.normal_impulse_count, 0u);
    EXPECT_TRUE(std::isfinite(report.row_scheduler_report.max_residual));
    EXPECT_GE(state.linear_velocities[fixture.box_body].y, -1.0e-5f);
}

TEST(CudaBatchedWorld, UploadsSharedJointAndActuatorTables) {
    const auto fixture = BuildJointDriveScene();
    auto world = runtime::BuildWorld(scene::CookScene(fixture.scene));
    std::vector<runtime::WorldInstance> instances(2, world.instance);

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    EXPECT_EQ(batch.JointCountPerInstance(), 1u);
    EXPECT_EQ(batch.ActuatorCountPerInstance(), 1u);
    EXPECT_TRUE(batch.HasUploadedJointTables());
    EXPECT_TRUE(batch.HasUploadedActuatorTables());
}

TEST(CudaBatchedWorld, ProjectsJointAnchorsIndependentlyOnDevice) {
    const auto fixture = BuildJointDriveScene();
    auto world = runtime::BuildWorld(scene::CookScene(fixture.scene));
    std::vector<runtime::WorldInstance> instances(2, world.instance);
    instances[0].poses[fixture.child_body].position.y = 1.0f;
    instances[1].poses[fixture.child_body].position.y = 2.0f;

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    runtime::gpu::CudaBatchedConstraintSolverConfig config;
    config.velocity_iterations = 0u;
    config.position_iterations = 12u;
    config.baumgarte = 0.5f;
    config.slop = 0.0f;

    auto result = runtime::gpu::SolveBatchedCudaConstraints(batch, nullptr, config);
    const auto report = result.DownloadReport();
    const auto state = batch.DownloadState();
    const uint32_t instance0_child = fixture.child_body;
    const uint32_t instance1_child = state.body_count_per_instance + fixture.child_body;

    EXPECT_EQ(report.joint_constraint_count, 2u);
    EXPECT_EQ(report.drive_constraint_count, 0u);
    EXPECT_GE(report.constraint_row_count, 10u);
    EXPECT_NEAR(state.poses[instance0_child].position.y, 0.0f, 1.0e-3f);
    EXPECT_NEAR(state.poses[instance1_child].position.y, 0.0f, 1.0e-3f);
}

TEST(CudaBatchedWorld, AppliesVelocityDrivesIndependentlyOnDevice) {
    const auto fixture = BuildJointDriveScene();
    auto world = runtime::BuildWorld(scene::CookScene(fixture.scene));
    std::vector<runtime::WorldInstance> instances(2, world.instance);
    instances[1].angular_velocities[fixture.child_body].z = 1.0f;

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    runtime::gpu::CudaBatchedConstraintSolverConfig config;
    config.velocity_iterations = 12u;
    config.position_iterations = 0u;

    auto result = runtime::gpu::SolveBatchedCudaConstraints(batch, nullptr, config);
    const auto report = result.DownloadReport();
    const auto row_buffer = result.ConstraintRowBuffer();
    const auto state = batch.DownloadState();
    const uint32_t instance0_child = fixture.child_body;
    const uint32_t instance1_child = state.body_count_per_instance + fixture.child_body;

    EXPECT_EQ(report.joint_constraint_count, 2u);
    EXPECT_EQ(report.drive_constraint_count, 2u);
    EXPECT_GE(report.constraint_row_count, 12u);
    EXPECT_EQ(row_buffer.kind,
              runtime::gpu::CudaConstraintRowBufferKind::UniversalRowCsr);
    EXPECT_EQ(row_buffer.layout,
              runtime::gpu::CudaConstraintRowLayout::UniversalRowCsr);
    EXPECT_EQ(row_buffer.schedule_mode,
              runtime::gpu::CudaConstraintRowScheduleMode::IslandColoredSweep);
    EXPECT_NE(row_buffer.device_rows, nullptr);
    EXPECT_EQ(row_buffer.owner_count,
              report.joint_constraint_count + report.drive_constraint_count);
    EXPECT_EQ(report.row_scheduler_report.owner_count, row_buffer.owner_count);
    EXPECT_EQ(report.row_scheduler_report.configured_iterations,
              config.velocity_iterations);
    EXPECT_EQ(report.row_scheduler_report.executed_iterations,
              config.velocity_iterations);
    EXPECT_GE(report.row_scheduler_report.active_row_count,
              report.constraint_row_count);
    EXPECT_GT(report.row_scheduler_report.normal_impulse_count, 0u);
    EXPECT_GT(report.row_scheduler_report.normal_delta_impulse_magnitude, 0.0f);
    EXPECT_TRUE(std::isfinite(report.row_scheduler_report.max_residual));
    EXPECT_NEAR(state.angular_velocities[instance0_child].z, 4.0f, 1.0e-4f);
    EXPECT_NEAR(state.angular_velocities[instance1_child].z, 4.0f, 1.0e-4f);
}

TEST(CudaBatchedWorld, StepPipelineCanSolveBatchedJointsAndDrives) {
    const auto fixture = BuildJointDriveScene();
    auto world = runtime::BuildWorld(scene::CookScene(fixture.scene));
    std::vector<runtime::WorldInstance> instances(2, world.instance);
    instances[0].poses[fixture.child_body].position.y = 1.0f;
    instances[1].angular_velocities[fixture.child_body].z = 1.0f;

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    runtime::gpu::CudaBatchedWorldStepOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 60.0f;
    options.step_count = 1u;
    options.enable_joints = true;
    options.enable_drives = true;
    options.solver_velocity_iterations = 12u;
    options.solver_position_iterations = 12u;
    options.solver_baumgarte = 0.5f;
    options.solver_slop = 0.0f;

    const auto report = runtime::gpu::StepBatchedCudaWorld(batch, options);
    const auto state = batch.DownloadState();

    EXPECT_EQ(report.simulated_step_count, 1u);
    EXPECT_EQ(report.joint_constraint_count, 2u);
    EXPECT_EQ(report.drive_constraint_count, 2u);
    EXPECT_GE(report.constraint_row_count, 12u);
    EXPECT_EQ(report.row_scheduler_report.row_kind,
              runtime::gpu::CudaConstraintRowBufferKind::UniversalRowCsr);
    EXPECT_EQ(report.row_scheduler_report.row_layout,
              runtime::gpu::CudaConstraintRowLayout::UniversalRowCsr);
    EXPECT_EQ(report.row_scheduler_report.schedule_mode,
              runtime::gpu::CudaConstraintRowScheduleMode::IslandColoredSweep);
    EXPECT_EQ(report.row_scheduler_report.configured_iterations,
              options.solver_velocity_iterations);
    EXPECT_EQ(report.row_scheduler_report.executed_iterations,
              options.solver_velocity_iterations);
    EXPECT_EQ(report.row_scheduler_report.solver_launch_count,
              options.solver_velocity_iterations);
    EXPECT_EQ(report.row_scheduler_report.diagnostic_launch_count, 1u);
    EXPECT_GE(report.row_scheduler_report.active_row_count,
              report.constraint_row_count);
    EXPECT_GT(report.row_scheduler_report.normal_impulse_count, 0u);
    EXPECT_GT(report.row_scheduler_report.normal_delta_impulse_magnitude, 0.0f);
    EXPECT_TRUE(std::isfinite(report.row_scheduler_report.max_residual));
    EXPECT_NEAR(state.angular_velocities[fixture.child_body].z, 4.0f, 1.0e-4f);
}
