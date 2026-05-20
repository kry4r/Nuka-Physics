// ---------------------------------------------------------------------------
// Nuka imported scene debug render demo CLI
// ---------------------------------------------------------------------------

#include "apps/debug_shell/scene_demo.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: nuka_scene_demo <scene.xml|scene.usda|scene.usd|scene.urdf> <output.ppm> [width height] [simulation_steps dt]\n";
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

    try {
        const auto result = nuka::app::ExportImportedSceneDebugView(options);
        std::cout << "Nuka scene demo exported " << options.output_path
                  << " bodies=" << result.body_count
                  << " meshes=" << result.mesh_instance_count
                  << " cameras=" << result.camera_count
                  << " lights=" << result.light_count
                  << " sim_steps=" << result.simulation_steps
                  << " sim_time=" << result.simulated_time_seconds
                  << " debug_commands=" << result.debug_command_count
                  << " lit_pixels=" << result.non_background_pixel_count << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "nuka_scene_demo: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
