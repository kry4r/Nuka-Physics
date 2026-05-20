#pragma once
// ---------------------------------------------------------------------------
// nuka::app::scene_demo -- imported scene debug render demo
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <string>

namespace nuka::app {

struct SceneDemoOptions {
    std::string input_path;
    std::string output_path = "nuka_scene_demo.ppm";
    uint32_t width = 640;
    uint32_t height = 360;
};

struct SceneDemoResult {
    uint32_t body_count = 0;
    uint32_t mesh_instance_count = 0;
    uint32_t camera_count = 0;
    uint32_t light_count = 0;
    uint32_t debug_command_count = 0;
    size_t non_background_pixel_count = 0;
};

SceneDemoResult ExportImportedSceneDebugView(const SceneDemoOptions& options);

} // namespace nuka::app
