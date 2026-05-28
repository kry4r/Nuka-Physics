// ---------------------------------------------------------------------------
// CUDA world stepper tests.
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_world_stepper.hpp"
#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "runtime/world_stepper.hpp"
#include "scene/cooker.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <string>

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
    first.inertia = {1.0f, 2.0f, 3.0f};
    first.local_transform.position = {1.0f, 2.0f, 3.0f};
    scene.AddRigidBody(std::move(first));

    scene::RigidBodyRecord second;
    second.name = "second";
    second.mass = 4.0f;
    second.inertia = {3.0f, 2.0f, 1.0f};
    second.local_transform.position = {-2.0f, 5.0f, 0.5f};
    scene.AddRigidBody(std::move(second));

    return scene;
}

} // namespace

TEST(CudaWorldStepper, IntegratesGravityForceAndVelocityOnDevice) {
    scene::SceneIR scene;

    scene::RigidBodyRecord body;
    body.name = "body";
    body.mass = 2.0f;
    body.local_transform.position = {1.0f, 2.0f, 3.0f};
    scene.AddRigidBody(std::move(body));

    const auto blob = scene::CookScene(scene);
    auto world = runtime::BuildWorld(blob);
    world.instance.forces[0] = {4.0f, 0.0f, 0.0f};

    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);

    runtime::gpu::CudaWorldStepOptions options;
    options.gravity = {0.0f, -10.0f, 0.0f};
    options.dt = 0.5f;
    options.step_count = 1;
    options.clear_forces_after_step = true;

    const auto report = runtime::gpu::StepCudaWorld(device_world, options);
    const auto snapshot = device_world.DownloadStateSnapshot();

    EXPECT_EQ(report.simulated_step_count, 1u);
    EXPECT_EQ(report.body_count, 1u);
    EXPECT_EQ(report.kernel_launch_count, 1u);
    EXPECT_EQ(snapshot.body_count, 1u);

    EXPECT_FLOAT_EQ(snapshot.first_linear_velocity.x, 1.0f);
    EXPECT_FLOAT_EQ(snapshot.first_linear_velocity.y, -5.0f);
    EXPECT_FLOAT_EQ(snapshot.first_linear_velocity.z, 0.0f);
    EXPECT_FLOAT_EQ(snapshot.first_body_position.x, 1.5f);
    EXPECT_FLOAT_EQ(snapshot.first_body_position.y, -0.5f);
    EXPECT_FLOAT_EQ(snapshot.first_body_position.z, 3.0f);
    EXPECT_EQ(snapshot.first_force, math::Vec3::Zero());
}

TEST(CudaWorldStepper, UsesCallerOwnedStreamFromDeviceContext) {
    scene::SceneIR scene;

    scene::RigidBodyRecord body;
    body.name = "body";
    body.mass = 1.0f;
    body.local_transform.position = {0.0f, 4.0f, 0.0f};
    scene.AddRigidBody(std::move(body));

    const auto blob = scene::CookScene(scene);
    auto world = runtime::BuildWorld(blob);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        const auto context = phi::MakeDeviceContext(0, stream);
        auto device_world = runtime::gpu::UploadDeviceWorld(context, world.template_view);
        runtime::gpu::UploadDeviceState(context, device_world, world.instance);

        runtime::gpu::CudaWorldStepOptions options;
        options.gravity = {0.0f, -10.0f, 0.0f};
        options.dt = 0.25f;
        options.step_count = 2;
        options.clear_forces_after_step = true;

        const auto report = runtime::gpu::StepCudaWorld(context, device_world, options);
        const auto snapshot = device_world.DownloadStateSnapshot();

        EXPECT_EQ(report.simulated_step_count, 2u);
        EXPECT_EQ(report.kernel_launch_count, 2u);
        EXPECT_FLOAT_EQ(snapshot.first_linear_velocity.y, -5.0f);
        EXPECT_FLOAT_EQ(snapshot.first_body_position.y, 2.125f);
        EXPECT_EQ(cudaStreamQuery(stream), cudaSuccess);
    }
    EXPECT_EQ(cudaStreamQuery(stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(CudaWorldStepper, MatchesCpuReferenceForFreeFallAndForces) {
    const auto blob = scene::CookScene(BuildTwoBodyScene());
    auto cuda_world_host = runtime::BuildWorld(blob);
    auto cpu_world = runtime::BuildWorld(blob);

    cuda_world_host.instance.forces[0] = {4.0f, 0.0f, 0.0f};
    cuda_world_host.instance.forces[1] = {0.0f, 8.0f, -4.0f};
    cuda_world_host.instance.torques[0] = {0.0f, 2.0f, 0.0f};
    cuda_world_host.instance.torques[1] = {1.0f, 0.0f, 3.0f};
    cpu_world.instance = cuda_world_host.instance;

    auto device_world = runtime::gpu::UploadDeviceWorld(cuda_world_host.template_view);
    runtime::gpu::UploadDeviceState(device_world, cuda_world_host.instance);

    runtime::gpu::CudaWorldStepOptions cuda_options;
    cuda_options.gravity = {0.0f, -9.81f, 0.0f};
    cuda_options.dt = 1.0f / 120.0f;
    cuda_options.step_count = 12;
    cuda_options.clear_forces_after_step = true;

    runtime::WorldStepOptions cpu_options;
    cpu_options.gravity = cuda_options.gravity;
    cpu_options.dt = cuda_options.dt;
    cpu_options.step_count = cuda_options.step_count;
    cpu_options.clear_forces_after_step = cuda_options.clear_forces_after_step;
    cpu_options.enable_contacts = false;

    const auto cuda_report = runtime::gpu::StepCudaWorld(device_world, cuda_options);
    const auto cpu_report = runtime::StepWorldInstance(cpu_world.template_view,
                                                       cpu_world.instance,
                                                       cpu_options);
    const auto gpu_state = device_world.DownloadState();

    ASSERT_EQ(gpu_state.body_count, cpu_world.instance.body_count);
    ASSERT_EQ(gpu_state.poses.size(), cpu_world.instance.poses.size());
    ASSERT_EQ(gpu_state.linear_velocities.size(), cpu_world.instance.linear_velocities.size());
    ASSERT_EQ(gpu_state.angular_velocities.size(), cpu_world.instance.angular_velocities.size());
    ASSERT_EQ(gpu_state.forces.size(), cpu_world.instance.forces.size());
    ASSERT_EQ(gpu_state.torques.size(), cpu_world.instance.torques.size());

    EXPECT_EQ(cuda_report.simulated_step_count, cpu_report.simulated_step_count);
    EXPECT_EQ(cuda_report.body_count, cpu_world.instance.body_count);
    EXPECT_EQ(cuda_report.kernel_launch_count, cuda_options.step_count);

    for (uint32_t body_index = 0; body_index < gpu_state.body_count; ++body_index) {
        ExpectVecNear(gpu_state.poses[body_index].position,
                      cpu_world.instance.poses[body_index].position,
                      1.0e-5f);
        ExpectVecNear(gpu_state.linear_velocities[body_index],
                      cpu_world.instance.linear_velocities[body_index],
                      1.0e-5f);
        ExpectVecNear(gpu_state.angular_velocities[body_index],
                      cpu_world.instance.angular_velocities[body_index],
                      1.0e-5f);
        EXPECT_EQ(gpu_state.forces[body_index], math::Vec3::Zero());
        EXPECT_EQ(gpu_state.torques[body_index], math::Vec3::Zero());
    }
}

TEST(CudaWorldStepper, RejectsStepBeforeDeviceStateUpload) {
    const auto blob = scene::CookScene(BuildTwoBodyScene());
    auto world = runtime::BuildWorld(blob);
    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);

    try {
        runtime::gpu::StepCudaWorld(device_world);
        FAIL() << "StepCudaWorld should reject a DeviceWorld without uploaded state";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("UploadDeviceState"), std::string::npos) << message;
    }
}
