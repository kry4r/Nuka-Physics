#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer::ImGuiLayer -- the M8.5 viewer's UI (T3).
//
// Builds the custom-beautified Dear ImGui UI each frame into ImDrawData. It is a
// THIN showcase shell over the existing Simulation + RenderWorld + present path
// (recon §4: physics-PRIMARY, NOT a game engine / editor). The panels:
//
//   * Transport toolbar -- play / pause / step / reset / speed (x0.25..x4),
//     teal-accented (accent-filled active Play), heading font for the title.
//   * Stats panel       -- step time (ms) / fps / sub-steps / DOF / contact cap /
//     env index as tidy labeled rows + colored badges (accent=live, muted=idle).
//   * Scene tree        -- the RenderWorld instances (name, mesh source real vs
//     primitive, material) in a styled tree.
//   * Camera panel      -- orbit/pan/zoom readout + reset + fov slider (wires the
//     CameraController).
//
// BEYOND STOCK IMGUI (the OWNER custom-UI requirement): a DOCKED layout (a default
// dock split built once via the docking branch, ImGuiConfigFlags_DockingEnable),
// the central node left empty so the rendered viewport shows THROUGH, panels
// docked around it; section headers drawn in the heading font; the teal accent
// used purposefully (active transport button, live badge, accent separators);
// consistent custom padding. No default-gray floating windows.
//
// The UI is recorded by RecordUi() which is meant to run BETWEEN
// NukaImGuiContext::NewFrame() and ImGui::Render(). The actual draw-data submission
// (ImGui_ImplVulkan_RenderDrawData) is the present/offscreen overlay callback,
// NOT here -- so this header has NO Vulkan dependency.
//
// DETERMINISM (GATE-B): RecordUi() takes NO time/animation input -- every widget
// is driven from ViewerUiState + the snapshot. With a FIXED ViewerUiState the
// produced ImDrawData is deterministic (two records memcmp-identical once
// composited), which the GATE-B smoke relies on.
//
// HOST-ONLY / zero-CUDA-token. This TU includes the vendored ImGui (SYSTEM
// include) so it is gated behind NK_BUILD_VULKAN_VALIDATION with the rest of the
// viewer; it pulls no Vulkan headers.
// ---------------------------------------------------------------------------

#include "render/render_world.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

class CameraController;

// ---------------------------------------------------------------------------
// ViewerUiState -- the mutable transport / selection state the toolbar drives and
// the viewer reads to pace the simulation. RecordUi() mutates this in response to
// button clicks (play_toggled / step_requested / reset_requested are one-shot
// edge flags the viewer consumes + clears each frame). Holding it as plain data
// (no time input) is what makes the GATE-B composite deterministic.
// ---------------------------------------------------------------------------
struct ViewerUiState {
    bool     playing       = true;    // sim advancing vs paused
    float    speed         = 1.0f;    // x0.25 .. x4 multiplier
    uint32_t env_index     = 0u;      // selected env (D4)
    uint32_t selected_inst = ~0u;     // scene-tree selection (highlight only)

    // One-shot edge flags set by RecordUi(), consumed + cleared by the viewer.
    bool play_toggled    = false;
    bool step_requested  = false;
    bool reset_requested = false;
    bool camera_reset    = false;
};

// ---------------------------------------------------------------------------
// ViewerStats -- the per-frame physics/render facts the stats panel renders. The
// viewer fills these from its frame timing + nk::World capacities + the latest
// render report. Pure data; RecordUi() only reads it.
// ---------------------------------------------------------------------------
struct ViewerStats {
    float    step_time_ms   = 0.0f;
    float    fps            = 0.0f;
    uint32_t sub_steps      = 1u;
    uint32_t dof            = 0u;
    uint32_t links          = 0u;
    uint32_t bodies         = 0u;
    uint32_t contact_cap    = 0u;
    uint32_t draw_calls     = 0u;
    uint64_t non_bg_pixels  = 0u;
    uint64_t frame_index    = 0u;
    bool     step_healthy   = true;
    std::string device_name;
};

// ---------------------------------------------------------------------------
// ImGuiLayer -- records the docked panel UI each frame.
//
// Construct once (sets up the dock-layout-builds-once latch). Call RecordUi every
// frame between the context NewFrame and ImGui::Render. It reads the RenderWorld
// (scene tree) + stats and reads/writes ui_state + camera.
// ---------------------------------------------------------------------------
class ImGuiLayer {
public:
    ImGuiLayer() = default;

    // Enable the docking config flag on the CURRENT ImGui context. MUST be called
    // ONCE after NukaImGuiContext::Init() and BEFORE the first NewFrame() (ImGui
    // asserts ConfigFlags_DockingEnable is set pre-first-frame so .ini docking
    // settings are not lost). Standalone (no Vulkan); idempotent.
    void EnableDocking();

    // Record all panels into the current ImGui frame. `world` feeds the scene
    // tree, `stats` the stats panel, `camera` the camera panel readout + fov
    // slider, and `ui_state` the transport (mutated in place). NO Vulkan, NO time
    // input -- deterministic given identical inputs (GATE-B D1).
    void RecordUi(const render::RenderWorld& world, const ViewerStats& stats,
                  CameraController& camera, ViewerUiState& ui_state);

private:
    // The docking layout is built ONCE (the first frame the dockspace exists). A
    // latch so we do not re-split every frame (which would fight a user re-dock).
    bool dock_built_ = false;
};

}  // namespace nuka::runtime::app::viewer
