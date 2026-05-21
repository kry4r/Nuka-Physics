// ---------------------------------------------------------------------------
// Tests for imported-scene debug render demo
// ---------------------------------------------------------------------------

#include "apps/debug_shell/headless_debug_renderer.hpp"
#include "apps/debug_shell/scene_demo.hpp"
#include "apps/debug_shell/debug_visualization.hpp"
#include "import/mjcf_importer.hpp"
#include "phi/platform_contract.hpp"
#include "scene/scene_pipeline.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

std::filesystem::path TempPpmPath(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

std::string ReadBinaryFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

} // namespace

TEST(HeadlessDebugRenderer, RasterizesImportedSceneDebugCommands) {
    const auto scene = nuka::import::LoadMjcf("examples/scenes/complete_robot.xml");
    const auto compiled = nuka::scene::BuildCompiledScene(scene);

    nuka::app::DebugVisualizationInput input;
    input.render_scene = &compiled.render;
    input.scene_graph = &compiled.graph;
    input.physics_world = &compiled.physics;

    const auto commands = nuka::app::BuildDebugVisualization(input);

    nuka::app::DebugRasterOptions options;
    options.width = 160;
    options.height = 120;

    const auto image = nuka::app::RasterizeDebugDrawList(commands, options);

    EXPECT_EQ(image.width, 160u);
    EXPECT_EQ(image.height, 120u);
    EXPECT_GT(image.NonBackgroundPixelCount(), 0u);
}

TEST(SceneDemo, ExportsMjcfSceneDebugViewToPpm) {
    const auto output_path = TempPpmPath("nuka_scene_demo_mjcf.ppm");
    std::filesystem::remove(output_path);

    nuka::app::SceneDemoOptions options;
    options.input_path = "examples/scenes/complete_robot.xml";
    options.output_path = output_path.string();
    options.width = 160;
    options.height = 120;

    const auto result = nuka::app::ExportImportedSceneDebugView(options);

    EXPECT_EQ(result.body_count, 2u);
    EXPECT_EQ(result.mesh_instance_count, 2u);
    EXPECT_EQ(result.camera_count, 1u);
    EXPECT_EQ(result.light_count, 1u);
    EXPECT_EQ(result.simulation_steps, 60u);
    EXPECT_NEAR(result.simulated_time_seconds, 1.0f, 1e-6f);
    EXPECT_GT(result.debug_command_count, 0u);
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_TRUE(std::filesystem::exists(output_path));

    {
        std::ifstream file(output_path, std::ios::binary);
        std::string magic;
        file >> magic;
        EXPECT_EQ(magic, "P6");
    }

    std::filesystem::remove(output_path);
}

TEST(SceneDemo, ExportsUsdSceneThroughSamePipeline) {
    const auto output_path = TempPpmPath("nuka_scene_demo_usd.ppm");
    std::filesystem::remove(output_path);

    nuka::app::SceneDemoOptions options;
    options.input_path = "tests/data/minimal_scene.usda";
    options.output_path = output_path.string();
    options.width = 160;
    options.height = 120;

    const auto result = nuka::app::ExportImportedSceneDebugView(options);

    EXPECT_EQ(result.body_count, 2u);
    EXPECT_EQ(result.mesh_instance_count, 2u);
    EXPECT_GT(result.debug_command_count, 0u);
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_TRUE(std::filesystem::exists(output_path));

    std::filesystem::remove(output_path);
}

TEST(SceneDemo, ExportsUsdExampleSceneThroughSamePipeline) {
    const auto output_path = TempPpmPath("nuka_scene_demo_usd_example.ppm");
    std::filesystem::remove(output_path);

    nuka::app::SceneDemoOptions options;
    options.input_path = "examples/scenes/complete_robot.usda";
    options.output_path = output_path.string();
    options.width = 160;
    options.height = 120;

    const auto result = nuka::app::ExportImportedSceneDebugView(options);

    EXPECT_EQ(result.body_count, 2u);
    EXPECT_EQ(result.mesh_instance_count, 2u);
    EXPECT_EQ(result.camera_count, 1u);
    EXPECT_EQ(result.light_count, 1u);
    EXPECT_GT(result.debug_command_count, 0u);
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_TRUE(std::filesystem::exists(output_path));

    std::filesystem::remove(output_path);
}

TEST(SceneDemo, SimulatesImportedSceneBeforeRendering) {
    const auto output_path = TempPpmPath("nuka_scene_demo_simulated.ppm");
    std::filesystem::remove(output_path);

    nuka::app::SceneDemoOptions options;
    options.input_path = "tests/data/minimal_scene.usda";
    options.output_path = output_path.string();
    options.width = 160;
    options.height = 120;
    options.simulation_steps = 10;
    options.dt = 0.01f;
    options.gravity = {0.0f, -9.81f, 0.0f};

    const auto result = nuka::app::ExportImportedSceneDebugView(options);

    EXPECT_EQ(result.simulation_steps, 10u);
    EXPECT_NEAR(result.simulated_time_seconds, 0.1f, 1e-6f);
    ASSERT_EQ(result.body_world_poses.size(), 2u);
    EXPECT_FLOAT_EQ(result.body_world_poses[0].position.y, 0.0f);
    EXPECT_NE(result.body_world_poses[1].position.y, 0.5f);
    EXPECT_NE(result.body_world_poses[1].position.z, 0.5f);
    EXPECT_GT(result.body_world_poses[1].position.y, -0.053955f);
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_TRUE(std::filesystem::exists(output_path));

    std::filesystem::remove(output_path);
}

TEST(SceneDemo, DefaultsImportedSceneSimulationToCudaBackendWhenAvailable) {
    const auto output_path = TempPpmPath("nuka_scene_demo_cuda_default.ppm");
    std::filesystem::remove(output_path);

    nuka::app::SceneDemoOptions options;
    options.input_path = "examples/scenes/complete_robot.usda";
    options.output_path = output_path.string();
    options.width = 160;
    options.height = 120;
    options.simulation_steps = 8;
    options.dt = 1.0f / 120.0f;

    const auto result = nuka::app::ExportImportedSceneDebugView(options);

    EXPECT_EQ(result.physics_backend, nuka::phi::PhysicsBackend::Cuda);
    EXPECT_TRUE(result.production_physics_backend);
    EXPECT_EQ(result.cuda_broadphase_pair_count, 1u);
    EXPECT_GE(result.cuda_constraint_block_count, 1u);
    EXPECT_GE(result.cuda_constraint_row_count, 5u);
    EXPECT_EQ(result.cuda_joint_constraint_count, 1u);
    EXPECT_EQ(result.cuda_drive_constraint_count, 1u);
    EXPECT_EQ(result.cuda_solver_velocity_iterations, 10u);
    EXPECT_EQ(result.cuda_solver_position_iterations, 4u);
    EXPECT_GE(result.cuda_max_position_error, 0.0f);
    EXPECT_GT(result.non_background_pixel_count, 0u);
    ASSERT_EQ(result.body_world_poses.size(), 2u);
    EXPECT_NE(result.body_world_poses[1].position.z, 0.5f);
    EXPECT_TRUE(std::filesystem::exists(output_path));

    std::filesystem::remove(output_path);
}

TEST(SceneDemo, RenderedImageChangesWhenSimulationChangesRuntimePose) {
    const auto initial_path = TempPpmPath("nuka_scene_demo_initial.ppm");
    const auto simulated_path = TempPpmPath("nuka_scene_demo_simulated_pixels.ppm");
    std::filesystem::remove(initial_path);
    std::filesystem::remove(simulated_path);

    nuka::app::SceneDemoOptions initial_options;
    initial_options.input_path = "tests/data/minimal_scene.usda";
    initial_options.output_path = initial_path.string();
    initial_options.width = 160;
    initial_options.height = 120;
    initial_options.simulation_steps = 0;
    initial_options.auto_fit_view = false;
    initial_options.view_center = {0.0f, -0.025f, 0.0f};
    initial_options.view_scale = 900.0f;

    nuka::app::SceneDemoOptions simulated_options = initial_options;
    simulated_options.output_path = simulated_path.string();
    simulated_options.simulation_steps = 10;
    simulated_options.dt = 0.01f;
    simulated_options.gravity = {0.0f, -9.81f, 0.0f};

    const auto initial_result = nuka::app::ExportImportedSceneDebugView(initial_options);
    const auto simulated_result = nuka::app::ExportImportedSceneDebugView(simulated_options);

    ASSERT_EQ(initial_result.body_world_poses.size(), simulated_result.body_world_poses.size());
    EXPECT_NE(initial_result.body_world_poses[1].position.y,
              simulated_result.body_world_poses[1].position.y);
    EXPECT_NE(ReadBinaryFile(initial_path), ReadBinaryFile(simulated_path));

    std::filesystem::remove(initial_path);
    std::filesystem::remove(simulated_path);
}
