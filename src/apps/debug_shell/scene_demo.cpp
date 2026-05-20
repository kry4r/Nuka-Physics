// ---------------------------------------------------------------------------
// nuka::app::scene_demo implementation
// ---------------------------------------------------------------------------

#include "apps/debug_shell/scene_demo.hpp"

#include "apps/debug_shell/debug_visualization.hpp"
#include "apps/debug_shell/headless_debug_renderer.hpp"
#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "scene/scene_pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
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

} // namespace

SceneDemoResult ExportImportedSceneDebugView(const SceneDemoOptions& options) {
    if (options.input_path.empty()) {
        throw std::invalid_argument("SceneDemoOptions::input_path must not be empty");
    }
    if (options.output_path.empty()) {
        throw std::invalid_argument("SceneDemoOptions::output_path must not be empty");
    }

    const auto scene = LoadSceneForDemo(options.input_path);
    const auto compiled = scene::BuildCompiledScene(scene);

    DebugVisualizationInput input;
    input.render_scene = &compiled.render;
    input.scene_graph = &compiled.graph;
    input.physics_world = &compiled.physics;

    const DebugDrawList commands = BuildDebugVisualization(input);

    DebugRasterOptions raster_options;
    raster_options.width = options.width;
    raster_options.height = options.height;
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
    return result;
}

} // namespace nuka::app
