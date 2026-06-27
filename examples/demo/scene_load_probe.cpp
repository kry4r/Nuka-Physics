// ---------------------------------------------------------------------------
// scene_load_probe -- headless verification of the editor's runtime load path.
//
// Runs the SAME viewer::LoadEditorScene chain the windowed editor uses (LoadLight
// -> CookToModel -> BuildRenderWorld -> nk::World -> Simulation) on every .nks the
// editor's Load panel discovers, with NO window/swapchain. Each scene must either
// LOAD (a built world) or fail GRACEFULLY (LoadEditorScene returns null + logs a
// reason) -- never abort/segfault. Prints "PROBE <scene> ..." flushed BEFORE each
// load so a hard crash names the offending scene.
// ---------------------------------------------------------------------------

#include "phi/backend.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/editor_scene.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace nphi = nuka::phi;
namespace viewer = nuka::runtime::app::viewer;

namespace {

// Mirror viewer_main::DiscoverScenes: examples/scenes + out (recursive) for .nks,
// sorted + deduped, relative to the cwd; unreadable roots are skipped.
std::vector<std::string> DiscoverScenes() {
    std::vector<std::string> out;
    const char* roots[] = {"examples/scenes", "out"};
    for (const char* root : roots) {
        std::error_code ec;
        if (!std::filesystem::is_directory(root, ec)) continue;
        try {
            for (const auto& e :
                 std::filesystem::recursive_directory_iterator(root)) {
                if (e.is_regular_file() && e.path().extension() == ".nks")
                    out.push_back(e.path().generic_string());
            }
        } catch (const std::exception&) { /* skip an unreadable subtree */ }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    nphi::Device* dev = nphi::InitBestDevice();
    if (!dev) { std::fprintf(stderr, "scene_load_probe: no physics device\n"); return 3; }
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    if (!backend) { std::fprintf(stderr, "scene_load_probe: no physics backend\n"); return 3; }

    std::vector<std::string> scenes;
    for (int i = 1; i < argc; ++i) scenes.push_back(argv[i]);
    if (scenes.empty()) scenes = DiscoverScenes();

    int loaded = 0, failed = 0;
    for (const std::string& s : scenes) {
        std::printf("PROBE %-32s ... ", s.c_str());
        std::fflush(stdout);  // flush BEFORE the load so a hard crash names the scene
        std::unique_ptr<viewer::EditorScene> es =
            viewer::LoadEditorScene(s, dev, backend, 1.0f / 240.0f);
        if (es && es->sim) {
            std::printf("LOADED  instances=%u dof=%u\n",
                        es->sim->GetRenderWorld().InstanceCount(), es->caps.dofs_per_env);
            ++loaded;
        } else {
            std::printf("GRACEFUL-FAIL (reason on stderr above)\n");
            ++failed;
        }
    }
    std::printf("SUMMARY loaded=%d graceful_fail=%d total=%zu\n",
                loaded, failed, scenes.size());
    return 0;
}
