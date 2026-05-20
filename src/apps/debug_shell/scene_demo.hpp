#pragma once
// ---------------------------------------------------------------------------
// nuka::app::scene_demo -- imported scene debug render demo
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"

namespace nuka::app {

struct SceneDemoOptions {
    std::string input_path;
    std::string output_path = "nuka_scene_demo.ppm";
    uint32_t width = 640;
    uint32_t height = 360;
    uint32_t simulation_steps = 60;
    float dt = 1.0f / 60.0f;
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    bool auto_fit_view = true;
    float view_scale = 180.0f;
    math::Vec3 view_center = {0.0f, 0.0f, 0.0f};
};

struct SceneDemoResult {
    uint32_t body_count = 0;
    uint32_t mesh_instance_count = 0;
    uint32_t camera_count = 0;
    uint32_t light_count = 0;
    uint32_t debug_command_count = 0;
    size_t non_background_pixel_count = 0;
    uint32_t simulation_steps = 0;
    float simulated_time_seconds = 0.0f;
    std::vector<math::Transform> body_world_poses;
};

SceneDemoResult ExportImportedSceneDebugView(const SceneDemoOptions& options);

} // namespace nuka::app
