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
#include <vector>

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

// Cook es.scene (the editable authority) into es.world / sim / caps / terrain.
// TRANSACTIONAL: the new World is built + Ready-checked FIRST; only then are the
// old sim/world swapped out, so a failed cook leaves the editor steppable.
bool CookSceneInto(EditorScene& es, nuka::phi::Device* device,
                   nuka::phi::Backend* backend, float dt) {
    if (!(dt > 1e-6f)) dt = 1.0f / 240.0f;
    nuka::scene::SceneIR light = LightCookCopy(es.scene);
    // The full-scene orchestrator (rigid/articulation + media cook) -- the SAME
    // path the C-ABI builder uses; media-free scenes cook byte-identically.
    cook::CookToModelResult cooked = cook::CookSceneToModel(light, 1, {});
    render::RenderWorld render_world =
        render::BuildRenderWorld(light.Ecs(), cooked.scene_map);

    const nk::ModelCapacities caps = cooked.model.capacities;  // pre-move copy
    nuka::terrain::HeightField terrain = std::move(cooked.terrain);
    auto world = std::make_unique<nk::World>(std::move(cooked.model), 1u, device,
                                             backend, DefaultCfg(dt));
    if (!world || !world->Ready()) return false;  // old world/sim untouched

    es.sim.reset();    // references the old world + publisher; drop before it
    es.world.reset();
    es.caps = caps;
    es.terrain = std::move(terrain);   // retained for height-scan obs
    es.world = std::move(world);
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
    // Re-cook from the (mutated) authority. On failure (throw or not-Ready) the
    // OLD world/sim stay live; the caller surfaces the error and keeps stepping.
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

bool ResetEditorScenePhysics(EditorScene& es) {
    if (!es.world || !es.world->Ready()) return false;
    // Restore the construction-time snapshot for every env: authored qpos / root pose
    // / particle state, with qdot / link & body / particle velocities returned to the
    // seeded (zero) initial and qddot / tau / lambda cleared -- the canonical device
    // reset shared with the RL path.
    if (es.world->Reset() != nuka::phi::Status::Ok) return false;
    // The snapshot does not carry drive targets, so re-seed them (env-major) to the
    // authored hold-drive values a live Drive edit may have overwritten. A particle-
    // only world has no links and skips this.
    const nk::Model& model = es.world->GetModel();
    const uint32_t L = model.capacities.links_per_env;
    const uint32_t E = (model.capacities.env_count > 0u) ? model.capacities.env_count : 1u;
    if (L > 0u && !model.hold_drives.targets.empty()) {
        std::vector<float> host(static_cast<size_t>(L) * E, 0.0f);
        for (uint32_t e = 0; e < E; ++e)
            for (uint32_t l = 0; l < L && l < model.hold_drives.targets.size(); ++l)
                host[static_cast<size_t>(e) * L + l] = model.hold_drives.targets[l];
        es.world->GetData().UploadField(nk::FieldId::DriveTarget, host.data(),
                                        host.size() * sizeof(float));
    }
    return true;
}

}  // namespace nuka::runtime::app::viewer
