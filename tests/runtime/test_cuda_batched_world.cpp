// ---------------------------------------------------------------------------
// CUDA batched world tests.
// ---------------------------------------------------------------------------

#include "runtime/gpu/batched_device_world.hpp"
#include "runtime/world_builder.hpp"
#include "runtime/world_stepper.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

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
    const auto state = batch.DownloadState();
    const uint32_t instance0_box = fixture.box_body;
    const uint32_t instance1_box = state.body_count_per_instance + fixture.box_body;

    EXPECT_EQ(report.contact_constraint_count, 1u);
    EXPECT_GE(report.constraint_row_count, 3u);
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
    EXPECT_GE(state.linear_velocities[fixture.box_body].y, -1.0e-5f);
}
