#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::Simulation - the host frame loop over an nk::World (M8 #10).
//
// Simulation is the FIRST frame-loop abstraction over nk::World (which itself has
// no loop -- Step() is a single linear pass). It owns the RenderWorld + a
// CommandQueue + the four systems and references (does not own) the physics
// World and the PosePublisher. A renderer is OPTIONAL: when present + enabled,
// Frame() syncs the RenderWorld from physics and draws an offscreen frame.
//
// Frame() order (recon §3.2):
//   1. InputSystem  -- drain the CommandQueue (M8 stub).
//   2. SimSystem    -- advance the physics one step (Step / StepPlanned).
//   3. IF rendering enabled:
//        TransformSyncSystem -- publish env's poses into the RenderWorld
//                               (downloaded_pose ∘ cached_visual_local, G1).
//        RenderSystem        -- raster the RenderWorld; keep the latest report.
//
// ★ G5 (throughput-neutral) HARD REQUIREMENT. When rendering is DISABLED the
// ENTIRE publish+render branch (step 3) is bypassed by a single `render_enabled_`
// gate -- the publisher's Data::DownloadField (the D2H copy) is NEVER issued and
// no draw happens. A render-off Frame() loop is therefore byte-for-byte the same
// device work as a bare `world.Step()` loop: no extra copy, no stream sync, no
// graph-timing perturbation. The bypass is one branch, deliberately not buried.
//
// HOST-ONLY / zero-CUDA-token (src/runtime/app/** lint red-line).
// ---------------------------------------------------------------------------

#include "render/render_world.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/vulkan_offscreen_types.hpp"
#include "runtime/app/command_queue.hpp"
#include "runtime/app/pose_publisher.hpp"
#include "runtime/app/scene_controller.hpp"
#include "runtime/app/systems.hpp"

#include <cstdint>
#include <utility>

namespace nuka::nk {
class World;
}  // namespace nuka::nk

namespace nuka::runtime::app {

class Simulation {
public:
    // Construct over a live physics World + a PosePublisher + a freshly built
    // RenderWorld (built once via render::BuildRenderWorld). The World and the
    // publisher are NOT owned (they outlive the Simulation); the RenderWorld is
    // MOVED in and owned. Rendering starts DISABLED -- EnableRendering attaches a
    // renderer + turns step 3 on.
    Simulation(nk::World& world, PosePublisher& publisher,
               render::RenderWorld render_world)
        : world_(world),
          publisher_(&publisher),
          render_world_(std::move(render_world)) {}

    // -- rendering control ---------------------------------------------------
    // Attach a renderer + enable the publish+render branch. `renderer` is NOT
    // owned (the caller keeps it alive for the Simulation's lifetime). Passing a
    // null renderer (or never calling this) keeps rendering OFF (G5 bypass).
    void EnableRendering(render::VulkanRasterRenderer* renderer,
                         const render::RasterOptions& options = {}) {
        renderer_       = renderer;
        raster_options_ = options;
        render_enabled_ = (renderer != nullptr);
    }
    void DisableRendering() { render_enabled_ = false; }
    bool RenderEnabled() const { return render_enabled_; }

    // -- publisher swap (M11 INT-8) -----------------------------------------
    // Re-point the PosePublisher behind the abstraction. The viewer uses this to
    // swap from the bootstrap HostDownloadPublisher to the zero-copy
    // CudaVulkanInteropPublisher ONCE the present renderer's exportable SSBO is
    // imported -- or to keep HostDownloadPublisher when interop is unavailable
    // (the graceful-fallback path on this lavapipe box). The new publisher is NOT
    // owned (it must outlive the Simulation, exactly like the ctor publisher).
    void SetPublisher(PosePublisher& publisher) { publisher_ = &publisher; }

    // -- controller (per-step control hook) ---------------------------------
    // Attach/clear the SceneController called before each governed step. NOT owned
    // (must outlive the Simulation). Null => no control hook (the default).
    void SetController(SceneController* controller) { controller_ = controller; }
    SceneController* Controller() const { return controller_; }

    // -- selection / step mode ----------------------------------------------
    void     SetEnvIndex(uint32_t env_index) { env_index_ = env_index; }  // D4
    uint32_t EnvIndex() const { return env_index_; }
    // Use the deterministic CUDA-graph replay path (StepPlanned) instead of Step.
    void SetPlannedStep(bool planned) { planned_ = planned; }
    bool PlannedStep() const { return planned_; }

    // -- the queue (producers push commands; InputSystem drains each frame) ---
    CommandQueue& Commands() { return command_queue_; }

    // -- advance ONE frame ---------------------------------------------------
    // Returns true when the physics step reported healthy. When rendering is
    // enabled the latest frame is stored (LatestReport()).
    bool Frame();

    // -- the PRESENT seam (M8.5 T3) -----------------------------------------
    // Drive ONE frame for the realtime windowed viewer: Input -> Sim -> publish
    // the selected env's FK poses into the RenderWorld (TransformSync) -- but do
    // NOT run the offscreen RenderSystem (the viewer draws the RenderWorld itself
    // via the PresentRenderer's swapchain path). This is the clean seam that lets
    // the viewer reuse the exact M8 step+publish without duplicating the publish
    // and without engaging the offscreen (D1) renderer. Unlike Frame(), publish
    // here is unconditional (the viewer always renders); the G5 offscreen bypass
    // does not apply because no offscreen draw is issued. `out_intents` (optional)
    // receives the resolved ViewerControl dispatch from the command queue.
    // `do_step` false publishes without advancing physics (a paused viewport that
    // still re-frames on env change); `step_count` runs that many fixed-dt steps
    // when stepping (the speed transport). Returns the step health (true if idle).
    bool FramePublish(InputIntents* out_intents = nullptr, bool do_step = true,
                      uint32_t step_count = 1u) {
        last_frame_rendered_ = false;
        // Drain the queue. When the caller does not want the intents back we still
        // need the MoveEntity payloads to apply the drag, so drain into a local.
        InputIntents local;
        InputIntents* intents = out_intents ? out_intents : &local;
        input_system_.Run(command_queue_, intents);
        // Apply drag-to-move LIVE through the general path (entity ->
        // nk::Data::UploadField) BEFORE the step; never touches the SceneIR/.nks.
        for (const MoveEntity& m : intents->move_entities) {
            ApplyMoveEntity(world_, render_world_, env_index_, m);
        }
        // A QUEUED StepOnce is honored in the SAME frame it is pushed (the queue
        // drained above), so an out-of-band step never lags the UI button.
        uint32_t steps = do_step ? step_count : 0u;
        if (intents->step_once && steps == 0u) steps = 1u;
        bool step_ok = true;
        for (uint32_t s = 0; s < steps; ++s) {
            // The general per-step control hook runs BEFORE the step it governs
            // (a controller owns its own decimation, counted in steps).
            if (controller_ != nullptr) controller_->OnStep(world_, env_index_);
            step_ok = sim_system_.Run(world_, planned_) && step_ok;
        }
        transform_sync_system_.Run(*publisher_, world_, env_index_, render_world_);
        return step_ok;
    }

    // -- introspection -------------------------------------------------------
    render::RenderWorld&                 GetRenderWorld() { return render_world_; }
    const render::RenderWorld&           GetRenderWorld() const { return render_world_; }
    const render::VulkanOffscreenReport& LatestReport() const { return latest_report_; }
    bool LastFrameRendered() const { return last_frame_rendered_; }
    nk::World&                           GetWorld() { return world_; }

private:
    nk::World&        world_;
    PosePublisher*    publisher_;  // not owned; rebindable via SetPublisher (INT-8)
    render::RenderWorld render_world_;
    CommandQueue      command_queue_;

    InputSystem         input_system_;
    SimSystem           sim_system_;
    TransformSyncSystem transform_sync_system_;
    RenderSystem        render_system_;

    SceneController*  controller_ = nullptr;  // not owned; per-step control hook
    render::VulkanRasterRenderer* renderer_ = nullptr;   // not owned
    render::RasterOptions         raster_options_{};
    render::VulkanOffscreenReport latest_report_{};

    uint32_t env_index_           = 0u;       // D4: env 0 by default
    bool     render_enabled_      = false;    // G5: render-off bypass gate
    bool     planned_             = false;    // StepPlanned vs Step
    bool     last_frame_rendered_ = false;
};

}  // namespace nuka::runtime::app
