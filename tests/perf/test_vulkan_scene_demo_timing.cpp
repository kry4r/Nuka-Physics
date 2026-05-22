// ---------------------------------------------------------------------------
// Vulkan imported-scene demo timing coverage
// ---------------------------------------------------------------------------

#include "apps/debug_shell/scene_demo.hpp"
#include "phi/platform_contract.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>

namespace {

void RunVulkanSceneDemoTiming(const char* input_path, const char* artifact_name) {
    const auto output_path = std::filesystem::temp_directory_path() / artifact_name;
    std::filesystem::remove(output_path);

    nuka::app::SceneDemoOptions options;
    options.input_path = input_path;
    options.output_path = output_path.string();
    options.width = 320;
    options.height = 180;
    options.simulation_steps = 60u;
    options.dt = 1.0f / 120.0f;
    options.render_backend = nuka::app::SceneDemoRenderBackend::Vulkan;

    const auto start = std::chrono::steady_clock::now();
    const auto result = nuka::app::ExportImportedSceneDebugView(options);
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(result.physics_backend, nuka::phi::PhysicsBackend::Cuda);
    EXPECT_TRUE(result.production_physics_backend);
    EXPECT_EQ(result.render_backend, nuka::app::SceneDemoRenderBackend::Vulkan);
    EXPECT_TRUE(result.production_render_backend);
    EXPECT_EQ(result.vulkan_render_width, options.width);
    EXPECT_EQ(result.vulkan_render_height, options.height);
    EXPECT_GE(result.vulkan_physical_device_count, 1u);
    EXPECT_FALSE(result.vulkan_selected_device_name.empty());
    EXPECT_GT(result.debug_command_count, 0u);
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_GE(result.cuda_constraint_row_count, 1u);
    EXPECT_LT(elapsed_ms, 1000);
    EXPECT_TRUE(std::filesystem::exists(output_path));

    std::filesystem::remove(output_path);
}

void RunVulkanRenderSceneDemoTiming(const char* input_path, const char* artifact_name) {
    const auto output_path = std::filesystem::temp_directory_path() / artifact_name;
    std::filesystem::remove(output_path);

    nuka::app::SceneDemoOptions options;
    options.input_path = input_path;
    options.output_path = output_path.string();
    options.width = 320;
    options.height = 180;
    options.simulation_steps = 60u;
    options.dt = 1.0f / 120.0f;
    options.render_backend = nuka::app::SceneDemoRenderBackend::Vulkan;
    options.output_mode = nuka::app::SceneDemoOutputMode::RenderSceneMaterial;

    const auto start = std::chrono::steady_clock::now();
    const auto result = nuka::app::ExportImportedSceneDebugView(options);
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(result.physics_backend, nuka::phi::PhysicsBackend::Cuda);
    EXPECT_TRUE(result.production_physics_backend);
    EXPECT_EQ(result.render_backend, nuka::app::SceneDemoRenderBackend::Vulkan);
    EXPECT_TRUE(result.production_render_backend);
    EXPECT_EQ(result.output_mode, nuka::app::SceneDemoOutputMode::RenderSceneMaterial);
    EXPECT_EQ(result.vulkan_render_width, options.width);
    EXPECT_EQ(result.vulkan_render_height, options.height);
    EXPECT_EQ(result.render_scene_command_count, result.mesh_instance_count);
    EXPECT_EQ(result.debug_command_count, 0u);
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_GE(result.cuda_constraint_row_count, 1u);
    EXPECT_LT(elapsed_ms, 1000);
    EXPECT_TRUE(std::filesystem::exists(output_path));

    std::filesystem::remove(output_path);
}

} // namespace

TEST(VulkanSceneDemoTiming, ImportedMjcfSceneCudaSimulationVulkanRenderUnderOneSecond) {
    RunVulkanSceneDemoTiming("examples/scenes/complete_robot.xml",
                             "nuka_vulkan_scene_demo_mjcf_perf.ppm");
}

TEST(VulkanSceneDemoTiming, ImportedUsdSceneCudaSimulationVulkanRenderUnderOneSecond) {
    RunVulkanSceneDemoTiming("examples/scenes/complete_robot.usda",
                             "nuka_vulkan_scene_demo_usd_perf.ppm");
}

TEST(VulkanSceneDemoTiming, ImportedUsdSceneCudaSimulationRenderSceneMaterialUnderOneSecond) {
    RunVulkanRenderSceneDemoTiming(
        "examples/scenes/complete_robot.usda",
        "nuka_vulkan_scene_demo_renderscene_perf.ppm");
}
