// ---------------------------------------------------------------------------
// Performance test: CUDA batched rigid integration timing
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

scene::SceneIR BuildBodyGrid(uint32_t body_count) {
    scene::SceneIR scene;
    for (uint32_t body_index = 0; body_index < body_count; ++body_index) {
        scene::RigidBodyRecord body;
        body.name = "batched_body";
        body.mass = 1.0f + static_cast<float>(body_index % 5);
        body.inertia = {1.0f, 1.0f, 1.0f};
        body.local_transform.position = {
            static_cast<float>(body_index % 8),
            2.0f + static_cast<float>(body_index / 8),
            static_cast<float>((body_index / 8) % 8)
        };
        scene.AddRigidBody(std::move(body));
    }
    return scene;
}

} // namespace

TEST(CudaBatchTiming, BatchedRigidIntegrationUnderOneSecond) {
    constexpr uint32_t kInstanceCount = 256;
    constexpr uint32_t kBodyCount = 32;

    auto world = runtime::BuildWorld(scene::CookScene(BuildBodyGrid(kBodyCount)));
    std::vector<runtime::WorldInstance> instances(kInstanceCount, world.instance);
    for (uint32_t instance_index = 0; instance_index < kInstanceCount; ++instance_index) {
        for (uint32_t body_index = 0; body_index < kBodyCount; ++body_index) {
            auto& instance = instances[instance_index];
            instance.poses[body_index].position.x += static_cast<float>(instance_index) * 0.01f;
            instance.forces[body_index] = {1.0f, 0.5f, -0.25f};
            instance.torques[body_index] = {0.1f, 0.2f, 0.3f};
        }
    }

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    runtime::gpu::CudaBatchedWorldStepOptions options;
    options.gravity = {0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 120u;
    options.clear_forces_after_step = true;

    const auto start = std::chrono::high_resolution_clock::now();
    const auto report = runtime::gpu::StepBatchedCudaWorld(batch, options);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(report.instance_count, kInstanceCount);
    EXPECT_EQ(report.body_count_per_instance, kBodyCount);
    EXPECT_EQ(report.total_body_count, kInstanceCount * kBodyCount);
    EXPECT_EQ(report.simulated_step_count, options.step_count);
    EXPECT_EQ(report.kernel_launch_count, options.step_count);
    EXPECT_LT(ms, 1000) << "CUDA batched rigid integration took " << ms << " ms";
}
