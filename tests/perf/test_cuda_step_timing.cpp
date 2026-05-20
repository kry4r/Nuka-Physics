// ---------------------------------------------------------------------------
// Performance test: CUDA rigid integration timing
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_world_stepper.hpp"
#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>
#include <chrono>

using namespace nuka;

TEST(CudaStepTiming, RigidIntegrationKernelUnderOneSecond) {
    scene::SceneIR scene;
    constexpr int body_count = 4096;

    for (int i = 0; i < body_count; ++i) {
        scene::RigidBodyRecord body;
        body.name = "body";
        body.mass = 1.0f + static_cast<float>(i % 7);
        body.inertia = {1.0f, 1.0f, 1.0f};
        body.local_transform.position = {
            static_cast<float>(i % 64),
            10.0f + static_cast<float>(i / 64),
            static_cast<float>((i / 64) % 64)
        };
        scene.AddRigidBody(std::move(body));
    }

    auto world = runtime::BuildWorld(scene::CookScene(scene));
    for (uint32_t i = 0; i < world.instance.body_count; ++i) {
        world.instance.forces[i] = {1.0f, 0.5f, -0.25f};
        world.instance.torques[i] = {0.1f, 0.2f, 0.3f};
    }

    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);

    runtime::gpu::CudaWorldStepOptions options;
    options.gravity = {0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 240;
    options.clear_forces_after_step = true;

    const auto start = std::chrono::high_resolution_clock::now();
    const auto report = runtime::gpu::StepCudaWorld(device_world, options);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(report.body_count, static_cast<uint32_t>(body_count));
    EXPECT_EQ(report.simulated_step_count, options.step_count);
    EXPECT_EQ(report.kernel_launch_count, options.step_count);
    EXPECT_LT(ms, 1000) << "CUDA rigid integration took " << ms << " ms";
}
