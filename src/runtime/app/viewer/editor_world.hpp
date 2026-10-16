#pragma once
// The editor's view of a physics world through the C ABI: the per-environment layout,
// the scene binding the renderer reads, live pose edits, and the step + publish frame.

#include "nuka/nuka.h"
#include "render/render_world.hpp"
#include "runtime/app/command_queue.hpp"
#include "runtime/app/systems.hpp"
#include "runtime/inference/world_controller.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"
#include "scene/terrain/heightfield.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::app::viewer {

struct WorldLayout {
    uint32_t env_count = 1u;
    uint32_t links_per_env = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t articulations_per_env = 0u;
    uint32_t dofs_per_env = 0u;
    uint32_t contacts_per_env = 0u;
    uint32_t contact_slots_per_env = 0u;
    std::vector<nuka_link_binding_t> links;  // per-environment link order
};

// Throws std::runtime_error when a query fails; `what` names the call.
void CheckResult(nuka_result_t result, const char* what);

WorldLayout QueryLayout(nuka_world_handle world);

// The cooked terrain grid; invalid when the world has none.
terrain::HeightField QueryTerrain(nuka_world_handle world);

// Body records map to their cooked rows (record order) and link bodies to their links,
// the rows BuildRenderWorld resolves instance poses from.
scene::SceneMap BindScene(const scene::SceneIR& scene, const WorldLayout& layout);

// A free body or a floating articulation root: the pose sources a live move reaches.
bool IsMovableSource(const render::PoseSource& source, const WorldLayout& layout);

// Teleports a movable instance to a visual-frame world pose in one environment and
// zeroes its velocity. False for an unknown or non-movable entity.
bool MoveEntityLive(nuka_world_handle world, const WorldLayout& layout,
                    const render::RenderWorld& render_world, uint32_t env_index,
                    const MoveEntity& cmd);

// Uniform PD gains on every non-root link of every environment, so drive targets act
// without a policy.
void ApplyHoldGains(nuka_world_handle world, const WorldLayout& layout, float kp, float kd);

class EditorSim {
public:
    EditorSim(nuka_world_handle world, WorldLayout layout, render::RenderWorld render_world);

    // Drains the queue, applies moves, runs `steps` controlled steps and publishes the
    // selected environment's poses. False when a step failed.
    bool FramePublish(InputIntents* intents, uint32_t steps);

    render::RenderWorld& GetRenderWorld() { return render_world_; }
    const render::RenderWorld& GetRenderWorld() const { return render_world_; }
    CommandQueue& Commands() { return commands_; }
    const WorldLayout& Layout() const { return layout_; }
    void SetEnvIndex(uint32_t env_index) { env_index_ = env_index % layout_.env_count; }
    uint32_t EnvIndex() const { return env_index_; }
    void SetController(inference::WorldController* controller) { controller_ = controller; }
    inference::WorldController* Controller() const { return controller_; }

private:
    void Publish();

    nuka_world_handle world_;
    WorldLayout layout_;
    render::RenderWorld render_world_;
    CommandQueue commands_;
    InputSystem input_;
    inference::WorldController* controller_ = nullptr;
    uint32_t env_index_ = 0u;
    std::vector<math::Transform> link_poses_;
    std::vector<math::Transform> body_poses_;
};

}  // namespace nuka::runtime::app::viewer
