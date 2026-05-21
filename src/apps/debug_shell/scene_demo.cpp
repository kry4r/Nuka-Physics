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
#include "runtime/gpu/batched_device_world.hpp"
#include "runtime/gpu/cuda_world_stepper.hpp"
#include "runtime/gpu/device_world.hpp"
#include "sensor/gpu/cuda_sensors.cuh"
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

void AppendTranslatedDebugCommands(const DebugDrawList& source,
                                   math::Vec3 translation,
                                   DebugDrawList& destination) {
    for (const auto& command : source.Commands()) {
        switch (command.type) {
        case DrawCommandType::Line:
            destination.AddLine(command.position + translation,
                                command.end + translation,
                                command.color);
            break;
        case DrawCommandType::Sphere:
            destination.AddSphere(command.position + translation,
                                  command.radius,
                                  command.color);
            break;
        case DrawCommandType::Capsule:
            destination.AddCapsule(command.position + translation,
                                   command.end,
                                   command.radius,
                                   command.half_height,
                                   command.color);
            break;
        case DrawCommandType::Box:
            destination.AddBox(command.position + translation,
                               command.size,
                               command.color);
            break;
        case DrawCommandType::AABB: {
            collision::AABB translated;
            translated.min = command.position + translation;
            translated.max = command.end + translation;
            destination.AddAABB(translated, command.color);
            break;
        }
        case DrawCommandType::Frame: {
            math::Transform transform;
            transform.position = command.position + translation;
            destination.AddFrame(transform);
            break;
        }
        case DrawCommandType::ContactPoint:
            destination.AddContactPoint(command.position + translation,
                                        command.end,
                                        command.radius);
            break;
        }
    }
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
void OffsetWorldInstance(runtime::WorldInstance& instance, math::Vec3 offset) {
    for (auto& pose : instance.poses) {
        pose.position += offset;
    }
}

std::vector<runtime::WorldInstance> BuildOffsetInstances(
    const runtime::WorldInstance& base_instance,
    uint32_t instance_count,
    math::Vec3 instance_spacing) {
    std::vector<runtime::WorldInstance> instances(instance_count, base_instance);
    for (uint32_t instance_index = 0; instance_index < instance_count; ++instance_index) {
        OffsetWorldInstance(instances[instance_index],
                            instance_spacing * static_cast<float>(instance_index));
    }
    return instances;
}

std::vector<sensor::gpu::BatchedCudaBodyRequest> BuildBatchedImuRequests(
    const scene::CookedSensorTable& sensor_table,
    uint32_t instance_count) {
    std::vector<sensor::gpu::BatchedCudaBodyRequest> requests;
    requests.reserve(sensor_table.types.size() * instance_count);
    for (uint32_t instance_index = 0; instance_index < instance_count; ++instance_index) {
        for (uint32_t sensor_index = 0;
             sensor_index < static_cast<uint32_t>(sensor_table.types.size());
             ++sensor_index) {
            const auto type = sensor_table.types[sensor_index];
            const scene::BodyId body_id = sensor_table.attached_bodies[sensor_index];
            if ((type == scene::SensorType::Imu || type == scene::SensorType::FramePose) &&
                body_id != scene::kInvalidBody) {
                requests.push_back({instance_index, body_id});
            }
        }
    }
    return requests;
}

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

    std::vector<scene::BodyId> imu_body_ids;
    const auto& sensor_table = compiled.physics.sensor_table;
    imu_body_ids.reserve(sensor_table.types.size());
    for (uint32_t sensor_index = 0;
         sensor_index < static_cast<uint32_t>(sensor_table.types.size());
         ++sensor_index) {
        const auto type = sensor_table.types[sensor_index];
        const scene::BodyId body_id = sensor_table.attached_bodies[sensor_index];
        if ((type == scene::SensorType::Imu || type == scene::SensorType::FramePose) &&
            body_id != scene::kInvalidBody) {
            imu_body_ids.push_back(body_id);
        }
    }
    if (!imu_body_ids.empty()) {
        const auto imu_result = sensor::gpu::QueryCudaImuSensor(device_world, imu_body_ids);
        const auto imu_samples = imu_result.DownloadSamples();
        result.cuda_imu_sample_count = imu_result.SampleCount();
        if (!imu_samples.empty()) {
            result.cuda_first_imu_position = imu_samples.front().position;
            result.cuda_first_imu_angular_velocity = imu_samples.front().angular_velocity;
            result.cuda_first_imu_linear_acceleration =
                imu_samples.front().linear_acceleration;
        }
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

void RequireExplicitCpuReferenceValidation(const SceneDemoOptions& options) {
    if (!options.allow_cpu_reference_validation) {
        throw std::runtime_error(
            "CPU physics is reference-only; set allow_cpu_reference_validation "
            "when intentionally running validation instead of CUDA production");
    }
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
    if (backend_selection.selected_backend == phi::PhysicsBackend::CpuReference) {
        RequireExplicitCpuReferenceValidation(options);
    }
    result.physics_backend = backend_selection.selected_backend;
    result.production_physics_backend = backend_selection.production_backend;
#else
    if (options.physics_backend_policy == phi::BackendSelectionPolicy::ForceCuda ||
        options.physics_backend_policy == phi::BackendSelectionPolicy::PreferCuda) {
        throw std::runtime_error("CUDA physics backend selected, but nuka_runtime_gpu is not built");
    }
    RequireExplicitCpuReferenceValidation(options);
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

BatchedSceneDemoResult ExportBatchedImportedSceneDebugView(
    const BatchedSceneDemoOptions& options) {
    if (options.input_path.empty()) {
        throw std::invalid_argument("BatchedSceneDemoOptions::input_path must not be empty");
    }
    if (options.output_path.empty()) {
        throw std::invalid_argument("BatchedSceneDemoOptions::output_path must not be empty");
    }
    if (options.instance_count == 0u) {
        throw std::invalid_argument("BatchedSceneDemoOptions::instance_count must be positive");
    }

#if !defined(NUKA_HAS_CUDA_RUNTIME)
    if (options.physics_backend_policy == phi::BackendSelectionPolicy::ForceCuda ||
        options.physics_backend_policy == phi::BackendSelectionPolicy::PreferCuda) {
        throw std::runtime_error(
            "CUDA batched scene demo selected, but nuka_runtime_gpu is not built");
    }
    throw std::runtime_error(
        "Batched scene demo requires CUDA production physics; CPU is reference-only");
#else
    const auto scene = LoadSceneForDemo(options.input_path);
    auto compiled = scene::BuildCompiledScene(scene);
    auto& runtime_world = compiled.physics.runtime_world;

    phi::BackendSelectionRequest backend_request;
    backend_request.policy = options.physics_backend_policy;
    const auto backend_selection = phi::ResolvePhysicsBackend(backend_request);
    if (backend_selection.selected_backend != phi::PhysicsBackend::Cuda ||
        !backend_selection.production_backend) {
        throw std::runtime_error(
            "Batched scene demo requires CUDA production physics; CPU is reference-only");
    }

    auto instances = BuildOffsetInstances(runtime_world.instance,
                                          options.instance_count,
                                          options.instance_spacing);
    auto batch = runtime::gpu::UploadBatchedDeviceWorld(runtime_world.template_view,
                                                        instances);

    runtime::gpu::CudaBatchedWorldStepOptions step_options;
    step_options.gravity = options.gravity;
    step_options.dt = options.dt;
    step_options.step_count = options.simulation_steps;
    step_options.clear_forces_after_step = true;
    step_options.enable_contacts = true;
    step_options.enable_joints = true;
    step_options.enable_drives = true;
    step_options.solver_velocity_iterations = 10u;
    step_options.solver_position_iterations = 4u;
    step_options.solver_slop = 0.005f;
    step_options.solver_baumgarte = 0.2f;

    const auto step_report = runtime::gpu::StepBatchedCudaWorld(batch, step_options);

    const auto imu_requests = BuildBatchedImuRequests(compiled.physics.sensor_table,
                                                      options.instance_count);
    sensor::gpu::CudaImuResult imu_result;
    std::vector<sensor::gpu::CudaImuSample> imu_samples;
    if (!imu_requests.empty()) {
        imu_result = sensor::gpu::QueryBatchedCudaImuSensor(batch, imu_requests);
        imu_samples = imu_result.DownloadSamples();
    }

    const auto state = batch.DownloadState();
    DebugDrawList combined_commands;
    BatchedSceneDemoResult result;
    result.body_world_poses_by_instance.resize(options.instance_count);

    for (uint32_t instance_index = 0; instance_index < options.instance_count; ++instance_index) {
        runtime::WorldInstance instance;
        instance.body_count = state.body_count_per_instance;
        instance.poses.resize(state.body_count_per_instance);
        instance.linear_velocities.resize(state.body_count_per_instance);
        instance.angular_velocities.resize(state.body_count_per_instance);
        instance.forces.resize(state.body_count_per_instance);
        instance.torques.resize(state.body_count_per_instance);

        for (uint32_t body_index = 0; body_index < state.body_count_per_instance; ++body_index) {
            const uint32_t flat_body =
                instance_index * state.body_count_per_instance + body_index;
            instance.poses[body_index] = state.poses[flat_body];
            instance.linear_velocities[body_index] = state.linear_velocities[flat_body];
            instance.angular_velocities[body_index] = state.angular_velocities[flat_body];
            instance.forces[body_index] = state.forces[flat_body];
            instance.torques[body_index] = state.torques[flat_body];
        }

        auto instance_compiled = scene::BuildCompiledScene(scene);
        scene::ApplyRuntimeStateToCompiledScene(instance, instance_compiled);

        DebugVisualizationInput input;
        input.render_scene = &instance_compiled.render;
        input.scene_graph = &instance_compiled.graph;
        input.physics_world = &instance_compiled.physics;
        const auto commands = BuildDebugVisualization(input);
        AppendTranslatedDebugCommands(commands, math::Vec3::Zero(), combined_commands);

        result.body_world_poses_by_instance[instance_index].reserve(
            instance_compiled.graph.NodeCount());
        for (const auto& node : instance_compiled.graph.Nodes()) {
            result.body_world_poses_by_instance[instance_index].push_back(
                node.world_transform);
        }
    }

    DebugRasterOptions raster_options;
    raster_options.width = options.width;
    raster_options.height = options.height;
    raster_options.view_scale = options.view_scale;
    raster_options.view_center = options.view_center;
    if (options.auto_fit_view) {
        FitRasterViewXY(combined_commands, raster_options);
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
            render::RenderDebugDrawListVulkan(ToVulkanDebugCommands(combined_commands),
                                              vulkan_options);
        if (!WriteVulkanPpm(options.output_path,
                            vulkan_result.width,
                            vulkan_result.height,
                            vulkan_result.pixels)) {
            throw std::runtime_error("Failed to write batched scene demo image: " +
                                     options.output_path);
        }
        result.render_backend = SceneDemoRenderBackend::Vulkan;
        result.production_render_backend = vulkan_result.production_backend;
        result.vulkan_render_width = vulkan_result.width;
        result.vulkan_render_height = vulkan_result.height;
        result.vulkan_physical_device_count = vulkan_result.physical_device_count;
        result.vulkan_selected_device_name = vulkan_result.selected_device_name;
        non_background_pixel_count = vulkan_result.non_background_pixel_count;
    } else {
        const DebugRasterImage image =
            RasterizeDebugDrawList(combined_commands, raster_options);
        if (!image.WritePpm(options.output_path)) {
            throw std::runtime_error("Failed to write batched scene demo image: " +
                                     options.output_path);
        }
        result.render_backend = SceneDemoRenderBackend::HeadlessReference;
        result.production_render_backend = false;
        non_background_pixel_count = image.NonBackgroundPixelCount();
    }

    result.instance_count = options.instance_count;
    result.body_count_per_instance = state.body_count_per_instance;
    result.total_body_count = state.instance_count * state.body_count_per_instance;
    result.mesh_instance_count_per_instance = compiled.render.mesh_instance_count;
    result.camera_count_per_instance = compiled.render.camera_count;
    result.light_count_per_instance = compiled.render.light_count;
    result.debug_command_count = static_cast<uint32_t>(combined_commands.CommandCount());
    result.non_background_pixel_count = non_background_pixel_count;
    result.simulation_steps = options.simulation_steps;
    result.simulated_time_seconds = options.dt * static_cast<float>(options.simulation_steps);
    result.physics_backend = backend_selection.selected_backend;
    result.production_physics_backend = backend_selection.production_backend;
    result.cuda_batched_simulated_step_count = step_report.simulated_step_count;
    result.cuda_batched_kernel_launch_count = step_report.kernel_launch_count;
    result.cuda_batched_broadphase_pair_count = step_report.broadphase_pair_count;
    result.cuda_batched_contact_manifold_count = step_report.contact_manifold_count;
    result.cuda_batched_contact_point_count = step_report.contact_point_count;
    result.cuda_batched_constraint_block_count = step_report.constraint_block_count;
    result.cuda_batched_constraint_row_count = step_report.constraint_row_count;
    result.cuda_batched_contact_constraint_count = step_report.contact_constraint_count;
    result.cuda_batched_joint_constraint_count = step_report.joint_constraint_count;
    result.cuda_batched_drive_constraint_count = step_report.drive_constraint_count;
    result.cuda_batched_solver_iterations = step_report.solver_iterations_used;
    result.cuda_batched_max_position_error = step_report.max_constraint_error;
    result.cuda_batched_imu_sample_count = imu_result.SampleCount();
    if (!imu_samples.empty()) {
        result.cuda_batched_first_imu_position = imu_samples.front().position;
        result.cuda_batched_first_imu_angular_velocity =
            imu_samples.front().angular_velocity;
        result.cuda_batched_first_imu_linear_acceleration =
            imu_samples.front().linear_acceleration;
    }
    return result;
#endif
}

} // namespace nuka::app
