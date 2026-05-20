// ---------------------------------------------------------------------------
// Tests for imported-scene debug render demo
// ---------------------------------------------------------------------------

#include "apps/debug_shell/headless_debug_renderer.hpp"
#include "apps/debug_shell/scene_demo.hpp"
#include "apps/debug_shell/debug_visualization.hpp"
#include "import/mjcf_importer.hpp"
#include "scene/scene_pipeline.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace {

std::filesystem::path TempPpmPath(const char* name) {
    return std::filesystem::temp_directory_path() / name;
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
