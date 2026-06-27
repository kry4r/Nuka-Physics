#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer::EditorScene -- the scene-bound state the editor
// cooks at runtime from one .nks: the physics World, the frame-loop Simulation
// (owns the RenderWorld), and the SceneIR retained for entity->node lookups.
//
// ONE LOAD PATH: LoadEditorScene is the SINGLE cook->World->Simulation builder.
// Both the Load UI and the --scene flag call it -- there is no separate startup
// cook. The editor starts EMPTY (no EditorScene) and swaps one in on load.
//
// HOST-ONLY / zero-CUDA-token: the only device touch is nk::World (built here)
// stepping behind the phi vtable.
// ---------------------------------------------------------------------------

#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "runtime/app/pose_publisher.hpp"
#include "runtime/app/simulation.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_ir.hpp"
#include "scene/terrain/heightfield.hpp"

#include <memory>
#include <string>

namespace nuka::runtime::app::viewer {

// The cooked product of one .nks. It lives ONLY inside a unique_ptr: the
// Simulation holds a pointer into `publisher` and a reference into `*world`, so
// the struct's address must never move after construction (copy + move deleted).
// Member ORDER matters for teardown: `sim` (declared last of the three) is
// destroyed before `publisher` and `world` that it references.
struct EditorScene {
    nuka::scene::SceneIR                            scene;      // retained for Ecs()
    std::unique_ptr<nuka::nk::World>                world;
    nuka::runtime::app::HostDownloadPublisher       publisher;  // outlives `sim`
    std::unique_ptr<nuka::runtime::app::Simulation> sim;
    nuka::nk::ModelCapacities                        caps{};
    nuka::scene::SceneGraph                          editor_graph;
    std::string                                      path;
    // The baked terrain grid (valid when the .nks carried a TerrainRecord); a scene
    // controller samples it for the height-scan obs. Invalid for terrain-free scenes.
    nuka::terrain::HeightField                       terrain;

    EditorScene() = default;
    EditorScene(const EditorScene&) = delete;
    EditorScene& operator=(const EditorScene&) = delete;
    EditorScene(EditorScene&&) = delete;
    EditorScene& operator=(EditorScene&&) = delete;
};

// Cook `path` -> nk::Model -> World + RenderWorld + Simulation. Returns null (and
// logs to stderr) on a missing file, a failed cook, or a not-ready World. `dt`
// seeds the solver timestep. The SINGLE runtime load path for the editor.
std::unique_ptr<EditorScene> LoadEditorScene(const std::string& path,
                                             nuka::phi::Device* device,
                                             nuka::phi::Backend* backend, float dt);

}  // namespace nuka::runtime::app::viewer
