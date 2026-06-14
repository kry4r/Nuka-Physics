#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app -- the four frame-loop systems (M8 manifest #8/#9).
//
// The host frame loop (Simulation::Frame) runs these in order:
//   InputSystem         -- drain the CommandQueue (M8 stub: drain + no-op; the
//                          M11 viewer's MoveEntity handler slots in here).
//   SimSystem           -- advance the physics one step (nk::World::Step, or
//                          StepPlanned for the deterministic graph-replay path).
//   TransformSyncSystem -- refresh the RenderWorld's live transforms from the
//                          physics state via a PosePublisher. THIS is the G1
//                          contract: world_xform == downloaded_pose ∘ visual_local.
//   RenderSystem        -- draw the RenderWorld to an offscreen target and keep
//                          the latest VulkanOffscreenReport for the recorder/gate.
//
// The render-driving system depends on nuka_render (Vulkan). Per Decision D5 the
// renderer is INJECTED (a pointer the Simulation owns only when rendering is on)
// and the actual draw call is gated, so a Vulkan-less build of this library still
// links: RenderSystem::Run becomes a no-op when NK_BUILD_VULKAN_VALIDATION is off
// and no renderer is ever attached. TransformSync + Sim + Input carry no Vulkan
// dependency at all.
//
// HOST-ONLY / zero-CUDA-token (src/runtime/app/** lint red-line).
// ---------------------------------------------------------------------------

#include "render/render_world.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"  // RasterOptions + (fwd) VulkanRasterRenderer
#include "render/vulkan_offscreen_types.hpp"          // VulkanOffscreenReport (pure POD)
#include "runtime/app/command_queue.hpp"
#include "runtime/app/pose_publisher.hpp"

#include <cstdint>
#include <vector>

namespace nuka::nk {
class World;
}  // namespace nuka::nk

namespace nuka::runtime::app {

// ---------------------------------------------------------------------------
// InputIntents - the resolved control intents the InputSystem extracted from one
// drained command batch (M8.5 T3). The Simulation/viewer reads these to apply the
// transport (play/pause/step/reset/env) + camera-reset. Edge flags: each is true
// at most once per Run(). The MoveEntity payloads (the M11 viewer's drag-to-move)
// are collected verbatim into `move_entities` so the caller can apply them through
// the GENERAL command path (ApplyMoveEntity -> nk::Data::UploadField); the count
// is kept for diagnostics.
// ---------------------------------------------------------------------------
struct InputIntents {
    bool     toggle_play   = false;
    bool     set_play      = false;   // explicit Play/Pause requested
    bool     play_value    = true;    // the value for set_play
    bool     step_once     = false;
    bool     reset         = false;
    bool     camera_reset  = false;
    bool     set_env       = false;
    uint32_t env_value     = 0u;
    uint32_t move_entity_count = 0u;  // how many MoveEntity were seen this batch.
    std::vector<MoveEntity> move_entities;  // M11: the drag-to-move payloads (FIFO).
};

// ---------------------------------------------------------------------------
// ApplyMoveEntity (VIEW-4) -- the GENERAL drag-to-move handler.
//
// Resolves `cmd.entity` to its cooked Data row + kind using the SAME pose_source
// the publisher reads (RenderInstance::pose_source, baked once at BuildRenderWorld
// from the cook's SceneMap CookedRef), then writes the requested world transform
// LIVE into the running nk::Data for the selected env:
//   * Body  -> UploadField(BodyPose, env*bodies_per_env + row); zero
//              BodyLinearVelocity/BodyAngularVelocity for that body (OD-6).
//   * Base  -> the articulation FLOATING root (the instance resolves to the root
//              link). UploadField(BasePose, env) so FK re-propagates from the new
//              base; zero the root link's LinkVelocity (6 floats) (OD-6).
//   * an interior articulation link -> NO-OP (its pose is an FK output, not an
//     independent Data row -- a live teleport would be overwritten next step).
//
// HARD R13: this writes RUNTIME Data ONLY (Data::UploadField). It NEVER touches
// the SceneIR / nuka_scene_set_local / Save -- the frozen .nks is never mutated.
// HARD highest-directive: this is the GENERAL path (entity -> Data field), NOT a
// per-demo "grab the cup" shim; it works for ANY Body / floating base.
//
// Returns true if a pose was applied (a resolvable Body/Base entity was found),
// false if the entity is unknown / Static / an interior link (no-op).
bool ApplyMoveEntity(nk::World& world, const render::RenderWorld& render_world,
                     uint32_t env_index, const MoveEntity& cmd);

// ---------------------------------------------------------------------------
// InputSystem - drains the CommandQueue once per frame and resolves intents.
//
// M8.5: a REAL minimal dispatch of the ViewerControl commands (play/pause/step/
// reset/camera/env) into an InputIntents the caller applies. MoveEntity is still
// counted-but-not-applied (full drag-to-move is M11); the returned batch is kept
// available for a future handler without re-touching the queue.
// ---------------------------------------------------------------------------
class InputSystem {
public:
    // Drain the queue, fold ViewerControl commands into `out_intents` (last write
    // wins for play state; edge flags latch), and return the raw batch (FIFO).
    std::vector<Command> Run(CommandQueue& queue, InputIntents* out_intents = nullptr) {
        std::vector<Command> batch = queue.Drain();
        if (out_intents == nullptr) return batch;
        for (const Command& c : batch) {
            switch (c.kind) {
                case Command::Kind::ViewerControl: {
                    using A = ViewerControl::Action;
                    switch (c.viewer_control.action) {
                        case A::TogglePlay:  out_intents->toggle_play = true; break;
                        case A::Play:        out_intents->set_play = true; out_intents->play_value = true; break;
                        case A::Pause:       out_intents->set_play = true; out_intents->play_value = false; break;
                        case A::StepOnce:    out_intents->step_once = true; break;
                        case A::Reset:       out_intents->reset = true; break;
                        case A::CameraReset: out_intents->camera_reset = true; break;
                        case A::SetEnv:      out_intents->set_env = true;
                                             out_intents->env_value = c.viewer_control.value; break;
                        case A::None:        break;
                    }
                    break;
                }
                case Command::Kind::MoveEntity:
                    ++out_intents->move_entity_count;
                    // VIEW-4: collect the payload; the caller (Simulation) applies
                    // it through ApplyMoveEntity (entity -> nk::Data, the general
                    // path) BEFORE the step so the drag perturbs THIS frame.
                    out_intents->move_entities.push_back(c.move_entity);
                    break;
                case Command::Kind::NoOp:
                    break;
            }
        }
        return batch;
    }
};

// ---------------------------------------------------------------------------
// SimSystem - advance the physics World one step.
//
// `planned` selects the deterministic CUDA-graph replay path (World::StepPlanned)
// over the plain linear dispatch (World::Step). Both are all-env; env selection
// happens only at publish/render time (Decision D4). Returns true when the step
// reported healthy (all ops Ok / plan execute Ok).
// ---------------------------------------------------------------------------
class SimSystem {
public:
    bool Run(nk::World& world, bool planned = false);
};

// ---------------------------------------------------------------------------
// TransformSyncSystem - refresh the RenderWorld live transforms (the G1 seam).
//
// Delegates to a PosePublisher (host-download in M8, device-scatter in M11),
// which writes `downloaded_pose * cached_visual_local` into every instance's
// world_xform for the selected env. The system owns NO pose logic itself -- it is
// the stable call site so swapping the publisher needs no edit here (Risk R5).
// ---------------------------------------------------------------------------
class TransformSyncSystem {
public:
    void Run(PosePublisher& publisher, const nk::World& world,
             uint32_t env_index, render::RenderWorld& render_world) {
        publisher.Publish(world, env_index, render_world);
    }
};

// ---------------------------------------------------------------------------
// RenderSystem - draw the RenderWorld and keep the latest report.
//
// Holds a non-owning VulkanRasterRenderer*. When the pointer is null (or this
// TU was built without Vulkan validation) Run is a no-op. The Simulation only
// attaches a renderer when rendering is enabled, so the G5 render-off path never
// reaches a draw call.
// ---------------------------------------------------------------------------
class RenderSystem {
public:
    // Render `world` with `options` into `out_report`. Returns true if a frame was
    // actually drawn (a renderer was attached and the draw succeeded).
    bool Run(render::VulkanRasterRenderer* renderer,
             const render::RenderWorld& world,
             const render::RasterOptions& options,
             render::VulkanOffscreenReport* out_report);
};

}  // namespace nuka::runtime::app
