// ---------------------------------------------------------------------------
// nuka::app::scene_demo implementation
// ---------------------------------------------------------------------------

#include "apps/debug_shell/scene_demo.hpp"

#include "apps/debug_shell/debug_visualization.hpp"
#include "apps/debug_shell/headless_debug_renderer.hpp"
#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "runtime/world_stepper.hpp"
#include "scene/scene_pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace nuka::app {

namespace {

std::string LowercaseExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension;
}

scene::SceneIR LoadSceneForDemo(const std::string& path) {
    const std::string extension = LowercaseExtension(path);
    if (extension == ".xml") {
        return import::LoadMjcf(path);
    }
    if (extension == ".urdf") {
        return import::LoadUrdf(path);
    }
    if (extension == ".usd" || extension == ".usda" ||
        extension == ".usdc" || extension == ".usdz") {
        return import::LoadUsd(path);
    }
    throw std::runtime_error("Unsupported scene demo input extension: " + extension);
}

struct DebugDrawBoundsXY {
    math::Vec3 min = {0.0f, 0.0f, 0.0f};
    math::Vec3 max = {0.0f, 0.0f, 0.0f};
    bool valid = false;
};

DebugDrawBoundsXY ComputeDebugDrawBoundsXY(const DebugDrawList& commands) {
    DebugDrawBoundsXY bounds;
    if (commands.Commands().empty()) {
        return bounds;
    }

    math::Vec3 min_point{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        0.0f
    };
    math::Vec3 max_point{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        0.0f
    };

    const auto include_point = [&](math::Vec3 point) {
        min_point.x = std::min(min_point.x, point.x);
        min_point.y = std::min(min_point.y, point.y);
        max_point.x = std::max(max_point.x, point.x);
        max_point.y = std::max(max_point.y, point.y);
    };

    for (const auto& command : commands.Commands()) {
        include_point(command.position);
        switch (command.type) {
        case DrawCommandType::Line:
        case DrawCommandType::AABB:
            include_point(command.end);
            break;
        case DrawCommandType::Box:
            include_point(command.position - command.size);
            include_point(command.position + command.size);
            break;
        case DrawCommandType::Sphere: {
            const math::Vec3 radius{command.radius, command.radius, 0.0f};
            include_point(command.position - radius);
            include_point(command.position + radius);
            break;
        }
        case DrawCommandType::Capsule: {
            const math::Vec3 axis = command.end.Normalized();
            const math::Vec3 a = command.position - axis * command.half_height;
            const math::Vec3 b = command.position + axis * command.half_height;
            const math::Vec3 radius{command.radius, command.radius, 0.0f};
            include_point(a - radius);
            include_point(a + radius);
            include_point(b - radius);
            include_point(b + radius);
            break;
        }
        case DrawCommandType::Frame:
        case DrawCommandType::ContactPoint:
            break;
        }
    }

    bounds.min = min_point;
    bounds.max = max_point;
    bounds.valid = true;
    return bounds;
}

void FitRasterViewXY(const DebugDrawList& commands, DebugRasterOptions& options) {
    const DebugDrawBoundsXY bounds = ComputeDebugDrawBoundsXY(commands);
    if (!bounds.valid || options.width == 0 || options.height == 0) {
        return;
    }

    const float width_world = std::max(bounds.max.x - bounds.min.x, 1e-3f);
    const float height_world = std::max(bounds.max.y - bounds.min.y, 1e-3f);
    const float scale_x = (static_cast<float>(options.width) * 0.82f) / width_world;
    const float scale_y = (static_cast<float>(options.height) * 0.82f) / height_world;

    options.view_center = {
        (bounds.min.x + bounds.max.x) * 0.5f,
        (bounds.min.y + bounds.max.y) * 0.5f,
        0.0f
    };
    options.view_scale = std::min(options.view_scale, std::min(scale_x, scale_y));
}

} // namespace

SceneDemoResult ExportImportedSceneDebugView(const SceneDemoOptions& options) {
    if (options.input_path.empty()) {
        throw std::invalid_argument("SceneDemoOptions::input_path must not be empty");
    }
    if (options.output_path.empty()) {
        throw std::invalid_argument("SceneDemoOptions::output_path must not be empty");
    }

    const auto scene = LoadSceneForDemo(options.input_path);
    auto compiled = scene::BuildCompiledScene(scene);

    if (options.simulation_steps > 0) {
        runtime::WorldStepOptions step_options;
        step_options.gravity = options.gravity;
        step_options.dt = options.dt;
        step_options.step_count = options.simulation_steps;
        runtime::StepWorldInstance(compiled.physics.runtime_world.template_view,
                                   compiled.physics.runtime_world.instance,
                                   step_options);
        scene::ApplyRuntimeStateToCompiledScene(compiled.physics.runtime_world.instance,
                                                compiled);
    }

    DebugVisualizationInput input;
    input.render_scene = &compiled.render;
    input.scene_graph = &compiled.graph;
    input.physics_world = &compiled.physics;

    const DebugDrawList commands = BuildDebugVisualization(input);

    DebugRasterOptions raster_options;
    raster_options.width = options.width;
    raster_options.height = options.height;
    raster_options.view_scale = options.view_scale;
    raster_options.view_center = options.view_center;
    if (options.auto_fit_view) {
        FitRasterViewXY(commands, raster_options);
    }
    const DebugRasterImage image = RasterizeDebugDrawList(commands, raster_options);

    std::filesystem::path output_path(options.output_path);
    if (const auto parent = output_path.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    if (!image.WritePpm(options.output_path)) {
        throw std::runtime_error("Failed to write scene demo image: " + options.output_path);
    }

    SceneDemoResult result;
    result.body_count = compiled.physics.body_count;
    result.mesh_instance_count = compiled.render.mesh_instance_count;
    result.camera_count = compiled.render.camera_count;
    result.light_count = compiled.render.light_count;
    result.debug_command_count = static_cast<uint32_t>(commands.CommandCount());
    result.non_background_pixel_count = image.NonBackgroundPixelCount();
    result.simulation_steps = options.simulation_steps;
    result.simulated_time_seconds = options.dt * static_cast<float>(options.simulation_steps);
    result.body_world_poses.reserve(compiled.graph.NodeCount());
    for (const auto& node : compiled.graph.Nodes()) {
        result.body_world_poses.push_back(node.world_transform);
    }
    return result;
}

} // namespace nuka::app
