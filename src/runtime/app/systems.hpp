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
// InputSystem - drains the CommandQueue once per frame (M8 stub).
//
// M8 just consumes the batch so producers never back up; handlers are no-ops
// with a clean seam for the M11 viewer's MoveEntity (the consumed batch is
// returned so a future handler can act on it without re-touching the queue).
// ---------------------------------------------------------------------------
class InputSystem {
public:
    // Drain + return the pending commands (FIFO). M8: no side effects beyond the
    // drain. M11: dispatch MoveEntity et al. here.
    std::vector<Command> Run(CommandQueue& queue) {
        std::vector<Command> batch = queue.Drain();
        // M8 stub: no command applied. (Seam: iterate `batch`, switch on Kind.)
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
