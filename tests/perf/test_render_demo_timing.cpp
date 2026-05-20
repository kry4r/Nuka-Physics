// ---------------------------------------------------------------------------
// Render demo timing coverage
// ---------------------------------------------------------------------------

#include "apps/debug_shell/scene_demo.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>

TEST(RenderDemoTiming, ImportedSceneDebugViewUnderOneSecond) {
    const auto output_path =
        std::filesystem::temp_directory_path() / "nuka_scene_demo_perf.ppm";
    std::filesystem::remove(output_path);

    nuka::app::SceneDemoOptions options;
    options.input_path = "examples/scenes/complete_robot.xml";
    options.output_path = output_path.string();
    options.width = 320;
    options.height = 180;

    const auto start = std::chrono::steady_clock::now();
    const auto result = nuka::app::ExportImportedSceneDebugView(options);
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(result.simulation_steps, 60u);
    EXPECT_NEAR(result.simulated_time_seconds, 1.0f, 1e-6f);
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_LT(elapsed_ms, 1000);
    EXPECT_TRUE(std::filesystem::exists(output_path));

    std::filesystem::remove(output_path);
}
