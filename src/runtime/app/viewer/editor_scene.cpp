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

// Strip collision mesh geometry from a COPY so CookToModel skips the heavy V-HACD
// pass (the editor cooks for the live loop, not hull fidelity). The original keeps
// its geometry so Save round-trips it -- a stripped Save would lose collision data.
nuka::scene::SceneIR LightCookCopy(const nuka::scene::SceneIR& full) {
    nuka::scene::SceneIR light = full;
    for (size_t i = 0; i < light.ShapeCount(); ++i) {
        auto& shape = light.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        shape.mesh_vertices.clear();
        shape.mesh_indices.clear();
    }
    return light;
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

// Cook es.scene (the editable authority) into es.world / sim / caps / terrain. Used
// by the initial load and by a re-cook after a structural tree edit. Tears the old
// sim down before the world it references; the caller WaitIdle's first.
bool CookSceneInto(EditorScene& es, nuka::phi::Device* device,
                   nuka::phi::Backend* backend, float dt) {
    if (!(dt > 1e-6f)) dt = 1.0f / 240.0f;
    nuka::scene::SceneIR light = LightCookCopy(es.scene);
    cook::CookToModelResult cooked = cook::CookToModel(light, 1);
    render::RenderWorld render_world =
        render::BuildRenderWorld(light.Ecs(), cooked.scene_map);

    es.sim.reset();    // references world + publisher; drop before the world
    es.world.reset();
    es.caps = cooked.model.capacities;        // copy BEFORE the model is moved
    es.terrain = std::move(cooked.terrain);   // retained for height-scan obs
    es.world = std::make_unique<nk::World>(std::move(cooked.model), 1u, device,
                                           backend, DefaultCfg(dt));
    if (!es.world || !es.world->Ready()) return false;
    es.sim = std::make_unique<nuka::runtime::app::Simulation>(
        *es.world, es.publisher, std::move(render_world));
    return true;
}

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

    // Catch any throw from the cook/build/World chain (parse error, unrepresentable
    // cook, device-alloc) so a bad Load fails gracefully instead of killing the window.
    try {
        // The FULL scene (collision meshes intact) is the editable authority; the
        // cook runs off a light copy so the live loop / behavior is unchanged.
        auto es = std::make_unique<EditorScene>();
        es->path = path;
        es->scene = nuka::scene::nks::Load(path);
        if (!CookSceneInto(*es, device, backend, dt)) {
            std::fprintf(stderr, "[nuka_editor] load failed: %s: world not ready\n",
                         path.c_str());
            return nullptr;
        }
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

bool RecookEditorScene(EditorScene& es, nuka::phi::Device* device,
                       nuka::phi::Backend* backend, float dt) {
    // Re-cook from the (mutated) authority. On failure es.world/sim are left reset;
    // the caller surfaces it and leaves the editor without a steppable world.
    try {
        return CookSceneInto(es, device, backend, dt);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[nuka_editor] re-cook failed: %s\n", e.what());
        return false;
    } catch (...) {
        std::fprintf(stderr, "[nuka_editor] re-cook failed: unknown error\n");
        return false;
    }
}

}  // namespace nuka::runtime::app::viewer
