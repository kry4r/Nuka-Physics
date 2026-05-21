// ---------------------------------------------------------------------------
// Batched CUDA + Vulkan imported-scene demo timing coverage
// ---------------------------------------------------------------------------

#include "apps/debug_shell/scene_demo.hpp"
#include "phi/platform_contract.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>

TEST(BatchedVulkanSceneDemoTiming,
     ImportedUsdBatchedCudaSimulationVulkanRenderUnderOneSecond) {
    const auto output_path =
        std::filesystem::temp_directory_path() / "nuka_batched_vulkan_scene_demo_perf.ppm";
    std::filesystem::remove(output_path);

    nuka::app::BatchedSceneDemoOptions options;
    options.input_path = "examples/scenes/complete_robot.usda";
    options.output_path = output_path.string();
    options.width = 320;
    options.height = 180;
    options.instance_count = 8u;
    options.instance_spacing = {1.5f, 0.0f, 0.0f};
    options.simulation_steps = 60u;
    options.dt = 1.0f / 120.0f;
    options.render_backend = nuka::app::SceneDemoRenderBackend::Vulkan;

    const auto start = std::chrono::steady_clock::now();
    const auto result = nuka::app::ExportBatchedImportedSceneDebugView(options);
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(result.physics_backend, nuka::phi::PhysicsBackend::Cuda);
    EXPECT_TRUE(result.production_physics_backend);
    EXPECT_EQ(result.render_backend, nuka::app::SceneDemoRenderBackend::Vulkan);
    EXPECT_TRUE(result.production_render_backend);
    EXPECT_EQ(result.instance_count, options.instance_count);
    EXPECT_EQ(result.total_body_count,
              result.instance_count * result.body_count_per_instance);
    EXPECT_EQ(result.vulkan_render_width, options.width);
    EXPECT_EQ(result.vulkan_render_height, options.height);
    EXPECT_GT(result.debug_command_count, 0u);
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_GE(result.cuda_batched_constraint_row_count, options.instance_count);
    EXPECT_EQ(result.cuda_batched_imu_sample_count, options.instance_count);
    EXPECT_LT(elapsed_ms, 1000);
    EXPECT_TRUE(std::filesystem::exists(output_path));

    std::filesystem::remove(output_path);
}
