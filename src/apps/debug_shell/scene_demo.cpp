// ---------------------------------------------------------------------------
// nuka::app::scene_demo implementation
// ---------------------------------------------------------------------------

#include "apps/debug_shell/scene_demo.hpp"

#include "apps/debug_shell/debug_visualization.hpp"
#include "apps/debug_shell/headless_debug_renderer.hpp"
#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "phi/platform_contract.hpp"
#include "render/vulkan_renderer.hpp"
#include "runtime/world_stepper.hpp"
#include "scene/scene_pipeline.hpp"

#if defined(NUKA_HAS_CUDA_RUNTIME)
#include "collision/gpu/broadphase.cuh"
#include "constraint/gpu/contact_generation.cuh"
#include "runtime/gpu/cuda_world_stepper.hpp"
#include "runtime/gpu/device_world.hpp"
#include "solver/gpu/cuda_constraint_solver.cuh"
#endif

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
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

render::VulkanRgba8 ToVulkanColor(Rgba8 color) {
    return {color.r, color.g, color.b, color.a};
}

render::VulkanDebugDrawCommandType ToVulkanCommandType(DrawCommandType type) {
    switch (type) {
    case DrawCommandType::Line:
        return render::VulkanDebugDrawCommandType::Line;
    case DrawCommandType::Sphere:
        return render::VulkanDebugDrawCommandType::Sphere;
    case DrawCommandType::Capsule:
        return render::VulkanDebugDrawCommandType::Capsule;
    case DrawCommandType::Box:
        return render::VulkanDebugDrawCommandType::Box;
    case DrawCommandType::AABB:
        return render::VulkanDebugDrawCommandType::AABB;
    case DrawCommandType::Frame:
        return render::VulkanDebugDrawCommandType::Frame;
    case DrawCommandType::ContactPoint:
        return render::VulkanDebugDrawCommandType::ContactPoint;
    }
    return render::VulkanDebugDrawCommandType::Line;
}

std::vector<render::VulkanDebugDrawCommand> ToVulkanDebugCommands(
    const DebugDrawList& commands) {
    std::vector<render::VulkanDebugDrawCommand> out;
    out.reserve(commands.CommandCount());
    for (const auto& command : commands.Commands()) {
        render::VulkanDebugDrawCommand vulkan_command;
        vulkan_command.type = ToVulkanCommandType(command.type);
        vulkan_command.position = command.position;
        vulkan_command.end = command.end;
        vulkan_command.size = command.size;
        vulkan_command.radius = command.radius;
        vulkan_command.half_height = command.half_height;
        vulkan_command.color = command.color;
        out.push_back(vulkan_command);
    }
    return out;
}

bool WriteVulkanPpm(const std::string& path,
                    uint32_t width,
                    uint32_t height,
                    const std::vector<render::VulkanRgba8>& pixels) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    out << "P6\n" << width << " " << height << "\n255\n";
    for (const auto& pixel : pixels) {
        out.put(static_cast<char>(pixel.r));
        out.put(static_cast<char>(pixel.g));
        out.put(static_cast<char>(pixel.b));
    }
    return out.good();
}

#if defined(NUKA_HAS_CUDA_RUNTIME)
void StepCompiledSceneCuda(scene::CompiledScene& compiled,
                           const SceneDemoOptions& options,
                           SceneDemoResult& result) {
    auto& runtime_world = compiled.physics.runtime_world;
    auto device_world = runtime::gpu::UploadDeviceWorld(runtime_world.template_view);
    runtime::gpu::UploadDeviceState(device_world, runtime_world.instance);

    runtime::gpu::CudaWorldStepOptions integration_options;
    integration_options.gravity = options.gravity;
    integration_options.dt = options.dt;
    integration_options.step_count = 1u;
    integration_options.clear_forces_after_step = true;

    solver::gpu::CudaConstraintSolverConfig solver_config;
    solver_config.velocity_iterations = 10u;
    solver_config.position_iterations = 4u;
    solver_config.slop = 0.005f;
    solver_config.baumgarte = 0.2f;

    for (uint32_t step = 0; step < options.simulation_steps; ++step) {
        runtime::gpu::StepCudaWorld(device_world, integration_options);
        auto broadphase = collision::gpu::BuildCudaBroadphase(device_world);
        auto contacts = constraint::gpu::GenerateCudaContacts(device_world, broadphase);
        const auto contact_report = contacts.DownloadReport();
        auto solver_result =
            solver::gpu::SolveCudaConstraints(device_world, &contacts, solver_config);
        const auto solver_report = solver_result.DownloadReport();

        result.cuda_broadphase_pair_count = contact_report.pair_count;
        result.cuda_contact_manifold_count = contact_report.contact_manifold_count;
        result.cuda_contact_point_count = contact_report.contact_point_count;
        result.cuda_constraint_block_count = solver_report.constraint_block_count;
        result.cuda_constraint_row_count = solver_report.constraint_row_count;
        result.cuda_contact_constraint_count = solver_report.contact_constraint_count;
        result.cuda_joint_constraint_count = solver_report.joint_constraint_count;
        result.cuda_drive_constraint_count = solver_report.drive_constraint_count;
        result.cuda_solver_velocity_iterations = solver_report.velocity_iterations;
        result.cuda_solver_position_iterations = solver_report.position_iterations;
        result.cuda_max_position_error = solver_report.max_position_error;
    }

    const auto state = device_world.DownloadState();
    runtime_world.instance.poses = state.poses;
    runtime_world.instance.linear_velocities = state.linear_velocities;
    runtime_world.instance.angular_velocities = state.angular_velocities;
    runtime_world.instance.forces = state.forces;
    runtime_world.instance.torques = state.torques;
    scene::ApplyRuntimeStateToCompiledScene(runtime_world.instance, compiled);
}
#endif

void StepCompiledSceneCpuReference(scene::CompiledScene& compiled,
                                   const SceneDemoOptions& options) {
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
    SceneDemoResult result;

#if defined(NUKA_HAS_CUDA_RUNTIME)
    phi::BackendSelectionRequest backend_request;
    backend_request.policy = options.physics_backend_policy;
    const auto backend_selection = phi::ResolvePhysicsBackend(backend_request);
    result.physics_backend = backend_selection.selected_backend;
    result.production_physics_backend = backend_selection.production_backend;
#else
    if (options.physics_backend_policy == phi::BackendSelectionPolicy::ForceCuda) {
        throw std::runtime_error("CUDA physics backend selected, but nuka_runtime_gpu is not built");
    }
    result.physics_backend = phi::PhysicsBackend::CpuReference;
    result.production_physics_backend = false;
#endif

    if (options.simulation_steps > 0) {
        if (result.physics_backend == phi::PhysicsBackend::Cuda) {
#if defined(NUKA_HAS_CUDA_RUNTIME)
            StepCompiledSceneCuda(compiled, options, result);
#else
            throw std::runtime_error("CUDA physics backend selected, but nuka_runtime_gpu is not built");
#endif
        } else {
            StepCompiledSceneCpuReference(compiled, options);
        }
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

    std::filesystem::path output_path(options.output_path);
    if (const auto parent = output_path.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    size_t non_background_pixel_count = 0;
    if (options.render_backend == SceneDemoRenderBackend::Vulkan) {
        render::VulkanOffscreenOptions vulkan_options;
        vulkan_options.width = raster_options.width;
        vulkan_options.height = raster_options.height;
        vulkan_options.view_scale = raster_options.view_scale;
        vulkan_options.view_center = raster_options.view_center;
        vulkan_options.background = ToVulkanColor(raster_options.background);
        const auto vulkan_result =
            render::RenderDebugDrawListVulkan(ToVulkanDebugCommands(commands),
                                              vulkan_options);
        if (!WriteVulkanPpm(options.output_path,
                            vulkan_result.width,
                            vulkan_result.height,
                            vulkan_result.pixels)) {
            throw std::runtime_error("Failed to write scene demo image: " + options.output_path);
        }
        result.render_backend = SceneDemoRenderBackend::Vulkan;
        result.production_render_backend = vulkan_result.production_backend;
        result.vulkan_render_width = vulkan_result.width;
        result.vulkan_render_height = vulkan_result.height;
        result.vulkan_physical_device_count = vulkan_result.physical_device_count;
        result.vulkan_selected_device_name = vulkan_result.selected_device_name;
        non_background_pixel_count = vulkan_result.non_background_pixel_count;
    } else {
        const DebugRasterImage image = RasterizeDebugDrawList(commands, raster_options);
        if (!image.WritePpm(options.output_path)) {
            throw std::runtime_error("Failed to write scene demo image: " + options.output_path);
        }
        result.render_backend = SceneDemoRenderBackend::HeadlessReference;
        result.production_render_backend = false;
        non_background_pixel_count = image.NonBackgroundPixelCount();
    }

    result.body_count = compiled.physics.body_count;
    result.mesh_instance_count = compiled.render.mesh_instance_count;
    result.camera_count = compiled.render.camera_count;
    result.light_count = compiled.render.light_count;
    result.debug_command_count = static_cast<uint32_t>(commands.CommandCount());
    result.non_background_pixel_count = non_background_pixel_count;
    result.simulation_steps = options.simulation_steps;
    result.simulated_time_seconds = options.dt * static_cast<float>(options.simulation_steps);
    result.body_world_poses.reserve(compiled.graph.NodeCount());
    for (const auto& node : compiled.graph.Nodes()) {
        result.body_world_poses.push_back(node.world_transform);
    }
    return result;
}

} // namespace nuka::app
