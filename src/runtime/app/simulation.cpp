// ---------------------------------------------------------------------------
// nuka::runtime::app::Simulation::Frame (M8 manifest #11).
//
// The host frame loop body. Input -> Sim -> (gated) TransformSync + Render.
//
// ★ G5: the publish+render branch is bypassed WHOLESALE by `render_enabled_`.
// When rendering is off, Frame() issues ONLY the physics step -- no
// Data::DownloadField (no D2H copy), no draw, no stream sync. The render-off
// loop is therefore identical device work to a bare `world.Step()` loop, so the
// StepPlanned graph timing / perf red-lines are byte-for-byte unperturbed.
//
// HOST-ONLY / zero-CUDA-token (src/runtime/app/** lint red-line).
// ---------------------------------------------------------------------------

#include "runtime/app/simulation.hpp"

#include "nk/pipeline/world.hpp"

namespace nuka::runtime::app {

bool Simulation::Frame() {
    last_frame_rendered_ = false;

    // 1. Input: drain the command queue (M8 stub -- no handlers yet).
    input_system_.Run(command_queue_);

    // 2. Sim: advance the physics one step (all-env). This is the ONLY work the
    //    render-off path performs.
    const bool step_ok = sim_system_.Run(world_, planned_);

    // 3. Render branch -- FULLY GATED by render_enabled_ (G5 bypass). When off,
    //    NOTHING below runs: no pose download, no draw, no sync. The single
    //    branch is the structural guarantee that a render-off loop perturbs the
    //    step path by zero device work.
    if (render_enabled_) {
        // 3a. TransformSync: publish the selected env's FK poses into the
        //     RenderWorld as downloaded_pose ∘ cached_visual_local (G1). This is
        //     where the (host-download) D2H copy lives -- reached ONLY here.
        transform_sync_system_.Run(*publisher_, world_, env_index_, render_world_);

        // 3b. Render: raster the RenderWorld; keep the latest report for the
        //     recorder / gate to grab. No-op (returns false) if no renderer is
        //     attached or this TU was built without Vulkan validation.
        last_frame_rendered_ = render_system_.Run(renderer_, render_world_,
                                                   raster_options_, &latest_report_);
    }

    return step_ok;
}

}  // namespace nuka::runtime::app
