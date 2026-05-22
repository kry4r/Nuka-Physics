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
#include "phi/platform_contract.hpp"
#include "render/vulkan_renderer.hpp"

namespace nuka::app {

enum class SceneDemoRenderBackend {
    Vulkan,
    HeadlessReference
};

enum class SceneDemoOutputMode {
    DebugOverlay,
    RenderSceneMaterial
};

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
    phi::BackendSelectionPolicy physics_backend_policy = phi::BackendSelectionPolicy::PreferCuda;
    bool allow_cpu_reference_validation = false;
    SceneDemoRenderBackend render_backend = SceneDemoRenderBackend::Vulkan;
    SceneDemoOutputMode output_mode = SceneDemoOutputMode::DebugOverlay;
};

struct SceneDemoResult {
    uint32_t body_count = 0;
    uint32_t mesh_instance_count = 0;
    uint32_t camera_count = 0;
    uint32_t light_count = 0;
    uint32_t debug_command_count = 0;
    uint32_t render_scene_command_count = 0;
    size_t non_background_pixel_count = 0;
    uint32_t simulation_steps = 0;
    float simulated_time_seconds = 0.0f;
    phi::PhysicsBackend physics_backend = phi::PhysicsBackend::CpuReference;
    bool production_physics_backend = false;
    SceneDemoRenderBackend render_backend = SceneDemoRenderBackend::HeadlessReference;
    SceneDemoOutputMode output_mode = SceneDemoOutputMode::DebugOverlay;
    bool production_render_backend = false;
    uint32_t vulkan_render_width = 0;
    uint32_t vulkan_render_height = 0;
    uint32_t vulkan_physical_device_count = 0;
    std::string vulkan_selected_device_name;
    uint32_t cuda_broadphase_pair_count = 0;
    uint32_t cuda_contact_manifold_count = 0;
    uint32_t cuda_contact_point_count = 0;
    uint32_t cuda_constraint_block_count = 0;
    uint32_t cuda_constraint_row_count = 0;
    uint32_t cuda_contact_constraint_count = 0;
    uint32_t cuda_joint_constraint_count = 0;
    uint32_t cuda_drive_constraint_count = 0;
    uint32_t cuda_solver_velocity_iterations = 0;
    uint32_t cuda_solver_position_iterations = 0;
    float cuda_max_position_error = 0.0f;
    uint32_t cuda_imu_sample_count = 0;
    math::Vec3 cuda_first_imu_position = math::Vec3::Zero();
    math::Vec3 cuda_first_imu_angular_velocity = math::Vec3::Zero();
    math::Vec3 cuda_first_imu_linear_acceleration = math::Vec3::Zero();
    std::vector<math::Transform> body_world_poses;
};

struct BatchedSceneDemoOptions {
    std::string input_path;
    std::string output_path = "nuka_batched_scene_demo.ppm";
    uint32_t width = 640;
    uint32_t height = 360;
    uint32_t simulation_steps = 60;
    float dt = 1.0f / 60.0f;
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    bool auto_fit_view = true;
    float view_scale = 180.0f;
    math::Vec3 view_center = {0.0f, 0.0f, 0.0f};
    uint32_t instance_count = 1;
    math::Vec3 instance_spacing = {2.0f, 0.0f, 0.0f};
    phi::BackendSelectionPolicy physics_backend_policy = phi::BackendSelectionPolicy::PreferCuda;
    SceneDemoRenderBackend render_backend = SceneDemoRenderBackend::Vulkan;
};

struct BatchedSceneDemoResult {
    uint32_t instance_count = 0;
    uint32_t body_count_per_instance = 0;
    uint32_t total_body_count = 0;
    uint32_t mesh_instance_count_per_instance = 0;
    uint32_t camera_count_per_instance = 0;
    uint32_t light_count_per_instance = 0;
    uint32_t debug_command_count = 0;
    size_t non_background_pixel_count = 0;
    uint32_t simulation_steps = 0;
    float simulated_time_seconds = 0.0f;
    phi::PhysicsBackend physics_backend = phi::PhysicsBackend::CpuReference;
    bool production_physics_backend = false;
    SceneDemoRenderBackend render_backend = SceneDemoRenderBackend::HeadlessReference;
    bool production_render_backend = false;
    uint32_t vulkan_render_width = 0;
    uint32_t vulkan_render_height = 0;
    uint32_t vulkan_physical_device_count = 0;
    std::string vulkan_selected_device_name;
    uint32_t cuda_batched_simulated_step_count = 0;
    uint32_t cuda_batched_kernel_launch_count = 0;
    uint32_t cuda_batched_broadphase_pair_count = 0;
    uint32_t cuda_batched_contact_manifold_count = 0;
    uint32_t cuda_batched_contact_point_count = 0;
    uint32_t cuda_batched_constraint_block_count = 0;
    uint32_t cuda_batched_constraint_row_count = 0;
    uint32_t cuda_batched_contact_constraint_count = 0;
    uint32_t cuda_batched_joint_constraint_count = 0;
    uint32_t cuda_batched_drive_constraint_count = 0;
    uint32_t cuda_batched_solver_iterations = 0;
    float cuda_batched_max_position_error = 0.0f;
    uint32_t cuda_batched_imu_sample_count = 0;
    math::Vec3 cuda_batched_first_imu_position = math::Vec3::Zero();
    math::Vec3 cuda_batched_first_imu_angular_velocity = math::Vec3::Zero();
    math::Vec3 cuda_batched_first_imu_linear_acceleration = math::Vec3::Zero();
    std::vector<std::vector<math::Transform>> body_world_poses_by_instance;
};

SceneDemoResult ExportImportedSceneDebugView(const SceneDemoOptions& options);
BatchedSceneDemoResult ExportBatchedImportedSceneDebugView(
    const BatchedSceneDemoOptions& options);

} // namespace nuka::app
