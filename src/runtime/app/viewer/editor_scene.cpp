// ---------------------------------------------------------------------------
// LoadEditorScene -- the editor's single runtime load path.
//
// Cooks a .nks through the SAME scene::cook::CookToModel / render::BuildRenderWorld
// API the offscreen + smoke paths use, stands up an nk::World over the live phi
// backend, and wraps it in the frame-loop Simulation. The --scene flag and the
// in-UI Load button both call this; there is no startup-only cook.
//
// HOST-ONLY / zero-CUDA-token.
// ---------------------------------------------------------------------------

#include "runtime/app/viewer/editor_scene.hpp"

#include "render/render_world.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <utility>

namespace nuka::runtime::app::viewer {

namespace {

namespace nk = nuka::nk;
namespace cook = nuka::scene::cook;
namespace render = nuka::render;

// Strip collision mesh geometry so CookToModel skips the heavy V-HACD pass (the
// editor validates the live loop, not hull fidelity). Visual meshes are untouched.
nuka::scene::SceneIR LoadLightScene(const std::string& path) {
    nuka::scene::SceneIR scene = nuka::scene::nks::Load(path);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& shape = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        shape.mesh_vertices.clear();
        shape.mesh_indices.clear();
    }
    return scene;
}

nk::Pipeline::SolverConfig DefaultCfg(float dt) {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = -9.81f;
    return cfg;
}

}  // namespace

std::unique_ptr<EditorScene> LoadEditorScene(const std::string& path,
                                             nuka::phi::Device* device,
                                             nuka::phi::Backend* backend, float dt) {
    if (path.empty()) {
        std::fprintf(stderr, "[nuka_editor] load: empty path\n");
        return nullptr;
    }
    if (!std::filesystem::exists(path)) {
        std::fprintf(stderr, "[nuka_editor] load: scene not found: %s\n", path.c_str());
        return nullptr;
    }
    if (!(dt > 1e-6f)) dt = 1.0f / 240.0f;

    // Catch any throw from the cook/build/World chain (parse error, unrepresentable
    // cook, device-alloc) so a bad Load fails gracefully instead of killing the window.
    try {
        nuka::scene::SceneIR scene = LoadLightScene(path);
        cook::CookToModelResult cooked = cook::CookToModel(scene, 1);
        render::RenderWorld render_world =
            render::BuildRenderWorld(scene.Ecs(), cooked.scene_map);

        auto es = std::make_unique<EditorScene>();
        es->path = path;
        es->caps = cooked.model.capacities;  // copy BEFORE the model is moved
        es->terrain = std::move(cooked.terrain);  // retained for height-scan obs
        es->scene = std::move(scene);
        es->world = std::make_unique<nk::World>(std::move(cooked.model), 1u, device,
                                                backend, DefaultCfg(dt));
        if (!es->world || !es->world->Ready()) {
            std::fprintf(stderr, "[nuka_editor] load failed: %s: world not ready\n",
                         path.c_str());
            return nullptr;
        }
        es->sim = std::make_unique<nuka::runtime::app::Simulation>(
            *es->world, es->publisher, std::move(render_world));

        std::printf("[nuka_editor] loaded %s  dof=%u  instances=%u\n", path.c_str(),
                    es->caps.dofs_per_env, es->sim->GetRenderWorld().InstanceCount());
        return es;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[nuka_editor] load failed: %s: %s\n", path.c_str(), e.what());
        return nullptr;
    } catch (...) {
        std::fprintf(stderr, "[nuka_editor] load failed: %s: unknown error\n", path.c_str());
        return nullptr;
    }
}

}  // namespace nuka::runtime::app::viewer
