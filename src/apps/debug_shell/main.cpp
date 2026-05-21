// ---------------------------------------------------------------------------
// Nuka imported scene debug render demo CLI
// ---------------------------------------------------------------------------

#include "apps/debug_shell/scene_demo.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: nuka_scene_demo <scene.xml|scene.usda|scene.usd|scene.urdf> <output.ppm> [width height] [simulation_steps dt] [instance_count]\n";
        return 2;
    }

    nuka::app::SceneDemoOptions options;
    options.input_path = argv[1];
    options.output_path = argv[2];
    if (argc >= 5) {
        options.width = static_cast<uint32_t>(std::stoul(argv[3]));
        options.height = static_cast<uint32_t>(std::stoul(argv[4]));
    }
    if (argc >= 6) {
        options.simulation_steps = static_cast<uint32_t>(std::stoul(argv[5]));
    }
    if (argc >= 7) {
        options.dt = std::stof(argv[6]);
    }
    uint32_t instance_count = 1u;
    if (argc >= 8) {
        instance_count = static_cast<uint32_t>(std::stoul(argv[7]));
    }

    try {
        if (instance_count > 1u) {
            nuka::app::BatchedSceneDemoOptions batched_options;
            batched_options.input_path = options.input_path;
            batched_options.output_path = options.output_path;
            batched_options.width = options.width;
            batched_options.height = options.height;
            batched_options.simulation_steps = options.simulation_steps;
            batched_options.dt = options.dt;
            batched_options.gravity = options.gravity;
            batched_options.auto_fit_view = options.auto_fit_view;
            batched_options.view_scale = options.view_scale;
            batched_options.view_center = options.view_center;
            batched_options.physics_backend_policy = options.physics_backend_policy;
            batched_options.render_backend = options.render_backend;
            batched_options.instance_count = instance_count;

            const auto result =
                nuka::app::ExportBatchedImportedSceneDebugView(batched_options);
            std::cout << "Nuka batched scene demo exported " << options.output_path
                      << " instances=" << result.instance_count
                      << " bodies_per_instance=" << result.body_count_per_instance
                      << " total_bodies=" << result.total_body_count
                      << " sim_steps=" << result.simulation_steps
                      << " sim_time=" << result.simulated_time_seconds
                      << " backend=" << (result.physics_backend == nuka::phi::PhysicsBackend::Cuda
                          ? "cuda"
                          : "cpu-reference")
                      << " production_backend=" << (result.production_physics_backend ? "true" : "false")
                      << " render_backend=" << (result.render_backend == nuka::app::SceneDemoRenderBackend::Vulkan
                          ? "vulkan"
                          : "headless-reference")
                      << " production_render=" << (result.production_render_backend ? "true" : "false")
                      << " batched_rows=" << result.cuda_batched_constraint_row_count
                      << " batched_joint_blocks=" << result.cuda_batched_joint_constraint_count
                      << " batched_drive_blocks=" << result.cuda_batched_drive_constraint_count
                      << " batched_contact_blocks=" << result.cuda_batched_contact_constraint_count
                      << " batched_max_error=" << result.cuda_batched_max_position_error
                      << " batched_imu_samples=" << result.cuda_batched_imu_sample_count
                      << " debug_commands=" << result.debug_command_count
                      << " lit_pixels=" << result.non_background_pixel_count << '\n';
            return 0;
        }

        const auto result = nuka::app::ExportImportedSceneDebugView(options);
        std::cout << "Nuka scene demo exported " << options.output_path
                  << " bodies=" << result.body_count
                  << " meshes=" << result.mesh_instance_count
                  << " cameras=" << result.camera_count
                  << " lights=" << result.light_count
                  << " sim_steps=" << result.simulation_steps
                  << " sim_time=" << result.simulated_time_seconds
                  << " backend=" << (result.physics_backend == nuka::phi::PhysicsBackend::Cuda
                      ? "cuda"
                      : "cpu-reference")
                  << " production_backend=" << (result.production_physics_backend ? "true" : "false")
                  << " render_backend=" << (result.render_backend == nuka::app::SceneDemoRenderBackend::Vulkan
                      ? "vulkan"
                      : "headless-reference")
                  << " production_render=" << (result.production_render_backend ? "true" : "false")
                  << " cuda_rows=" << result.cuda_constraint_row_count
                  << " cuda_joint_blocks=" << result.cuda_joint_constraint_count
                  << " cuda_drive_blocks=" << result.cuda_drive_constraint_count
                  << " cuda_contact_blocks=" << result.cuda_contact_constraint_count
                  << " cuda_max_error=" << result.cuda_max_position_error
                  << " cuda_imu_samples=" << result.cuda_imu_sample_count
                  << " debug_commands=" << result.debug_command_count
                  << " lit_pixels=" << result.non_background_pixel_count << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "nuka_scene_demo: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
