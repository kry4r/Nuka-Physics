#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer::ImGuiLayer -- the viewer's UI.
//
// Builds the custom-beautified Dear ImGui UI each frame into ImDrawData. It is a
// THIN showcase shell over the existing Simulation + RenderWorld + present path
// (physics-PRIMARY, not a game engine). The panels:
//
//   * Transport toolbar -- play / pause / step / reset / speed (x0.25..x4),
//     teal-accented (accent-filled active Play), heading font for the title.
//   * Stats panel       -- step time (ms) / fps / sub-steps / DOF / contact cap /
//     env index as tidy labeled rows + colored badges (accent=live, muted=idle).
//   * Scene tree        -- the SceneGraph hierarchy (robots as collapsible
//     path-prefixed subtrees, terrain/media at root) with per-node kind badges.
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
// DETERMINISM: RecordUi() takes NO time/animation input -- every widget
// is driven from ViewerUiState + the snapshot. With a FIXED ViewerUiState the
// produced ImDrawData is deterministic (two records memcmp-identical once
// composited), which the determinism smoke relies on.
//
// HOST-ONLY / zero-CUDA-token. This TU includes the vendored ImGui (SYSTEM
// include) so it is gated behind NK_BUILD_EDITOR / NK_BUILD_VULKAN_VALIDATION with
// the rest of the viewer; it pulls no Vulkan headers.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "render/render_world.hpp"
#include "runtime/app/viewer/editor_actions.hpp"
#include "runtime/app/viewer/scene_viewport.hpp"
#include "scene/ecs/entity.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::scene {
class SceneIR;
}

namespace nuka::runtime::app::viewer {

class CameraController;

// ---------------------------------------------------------------------------
// InspectorState -- the editable property values for the selected entity. The
// viewer SEEDS these from the authoritative scene when the selection changes (and
// re-reads a movable body's live pose each frame); the inspector widgets WRITE
// them and latch `*_changed` (a value moved -> live preview update) / `*_commit`
// (the edit finished -> persist the record). This is the SAME seam the Drive panel
// uses: the UI mutates plain state + flags, the viewer applies via the general
// edit path. NO nk / RenderWorld dependency lands in the UI layer.
// ---------------------------------------------------------------------------
struct InspectorState {
    bool  valid   = false;    // a selectable, editable entity is bound
    bool  movable = false;    // physics-driven (teleports) vs static placement

    // transform (WORLD pose of the instance; rotation as intrinsic-XYZ degrees)
    float pos[3]     = {0.0f, 0.0f, 0.0f};
    float rot_deg[3] = {0.0f, 0.0f, 0.0f};
    bool  transform_changed = false;  // a transform widget moved this frame (live)
    bool  transform_commit  = false;  // edit finished -> persist the record

    // material (present only when the selection carries an editable material)
    bool  has_material = false;
    float base_color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float roughness     = 0.5f;
    float metallic      = 0.0f;
    float emissive[3]   = {0.0f, 0.0f, 0.0f};
    float sheen         = 0.0f;   // RT-only (offline beauty)
    float transmission  = 0.0f;   // RT-only
    float ior           = 1.0f;   // RT-only
    float absorption[3] = {0.0f, 0.0f, 0.0f};  // RT-only
    bool  material_changed = false;
    bool  material_commit  = false;
};

// ---------------------------------------------------------------------------
// GizmoState -- the in-viewport transform manipulator settings for the selected
// entity. It edits the SAME world transform the inspector does: Translate and
// Rotate map 1:1 onto the rigid math::Transform. There is deliberately NO Scale
// op -- the engine transform carries no scale, so a scale handle could not round-
// trip through the records (it would be a non-persisting hack). The viewer reads
// op/space + enabled, and writes `active` (the gizmo is hovered/dragged) so camera
// orbit and entity picking stand down while the user manipulates it.
// ---------------------------------------------------------------------------
struct GizmoState {
    enum class Op : uint8_t { Translate, Rotate };
    bool enabled = true;             // master toggle (Entity panel)
    Op   op      = Op::Translate;    // active manipulation
    bool local   = false;            // local frame vs world frame
    // Grid snapping: translate step in world units, rotate step in degrees. ImGuizmo
    // rounds the drag to the step when snap_on; the steps are user-editable defaults.
    bool  snap_on        = false;
    float snap_translate = 0.25f;
    float snap_rotate    = 15.0f;
    bool active  = false;            // viewer-set: hovered/used this frame
    bool using_now = false;          // viewer-set: dragging this frame (drag-end = the
                                     // undo coalescing boundary; one op per drag)
};

// ---------------------------------------------------------------------------
// ViewerUiState -- the mutable transport / selection state the toolbar drives and
// the viewer reads to pace the simulation. RecordUi() queues actions for clicks and
// edits widget state in place; plain data with no time input keeps it deterministic.
// ---------------------------------------------------------------------------
struct ViewerUiState {
    bool     window_focused = true;
    bool     layout_modifier_down = false;
    bool     playing   = false;   // open PAUSED on the authored pose; Play/Step advances
    // Simulated-seconds per wall-second target: the physics thread accumulates
    // speed * wall time and consumes it in fixed-dt steps.
    float    speed     = 1.0f;
    uint32_t env_index = 0u;      // selected env

    // Read-only physics debug overlays (Show panel). Transient toggles read ONLY by
    // the between-frames overlay rebuild -- never serialized into the deterministic
    // record, so the default-off path leaves the rendered frame byte-identical.
    bool show_colliders = false;
    bool show_contacts  = false;

    // The single selection key: the picked/clicked ENTITY (viewport picker + tree
    // click write it; tree highlight + Entity panel read it). kInvalidEntity = none.
    nuka::scene::EntityId selected_entity = nuka::scene::kInvalidEntity;

    // The GENERIC per-DOF drive-target editor state. `drive_targets[d]` is
    // the slider value for DOF d of the SELECTED env; `drive_dirty[d]` latches when
    // a slider moved this frame so the viewer uploads ONLY changed DOFs (per-DOF
    // FieldId::DriveTarget UploadField, env-major offset env*per_env + d). NEVER a
    // hardcoded grasp/choreography table (a flat per-DOF editor). `dof_labels`
    // is the OPTIONAL cooked name->dof labelling (cosmetic); empty -> "dof N".
    // For the determinism smoke these are pre-seeded to FIXED values so RecordUi is
    // deterministic (no time/animation input).
    std::vector<float>       drive_targets;
    std::vector<uint8_t>     drive_dirty;   // per-DOF "changed this frame" latch
    std::vector<std::string> dof_labels;    // optional, size 0 or == drive_targets
    bool                     show_drive_panel = true;

    // Interactive keyboard teleop: a THIRD drive producer beside the Drive sliders and
    // the --policy controller, applied in the input loop (never RecordUi) and gated on
    // ImGui not owning the keyboard. Per-DOF nudges ([ / ]) write the SAME drive_targets
    // / drive_dirty seam; WASD/QE feed a [vx,vy,wyaw] command into a command-consuming
    // controller. The steps are editable defaults (no magic numbers); teleop_cmd is
    // viewer-set for the panel display.
    bool  teleop_enabled = false;   // master toggle (Teleop section)
    int   teleop_dof     = 0;       // the DOF [ / ] nudge moves (clamped to range)
    float teleop_step    = 0.05f;   // per-key-press drive-target nudge (rad / units)
    float teleop_lin     = 1.0f;    // WASD planar-velocity command magnitude
    float teleop_yaw     = 1.0f;    // QE yaw-rate command magnitude
    float teleop_cmd[3]  = {0.0f, 0.0f, 0.0f};  // viewer-set live (vx,vy,wyaw)
    // Persistent command baseline (--teleop-cmd / a script); keys add on top and
    // releasing every key returns to it. Zero for pure keyboard driving.
    float teleop_cmd_base[3] = {0.0f, 0.0f, 0.0f};
    // PD-hold makes drive targets effective for policy-free direct joint control:
    // the viewer uploads uniform gains to the actuated joints on the rising edge. The
    // gains are editable defaults (no magic numbers). Ignored when a command-consuming
    // controller (e.g. a policy) is attached -- it owns the gains.
    bool  teleop_hold = false;
    float teleop_kp   = 20.0f;
    float teleop_kd   = 0.5f;

    // Buttons, menus and drops queue actions here; the viewer runs them at the next frame
    // top, between frames, so RecordUi never changes the world.
    std::vector<ActionCall> actions;

    // Viewer-set hints from the document history for the Undo/Redo buttons.
    bool can_undo = false;
    bool can_redo = false;

    // ---- editor Load state (empty-start + runtime load) --------------------
    bool        has_scene      = false;   // viewer-set: a world is cooked
    std::string loaded_path;              // viewer-set: shown in the panel
    char        load_path[512] = {0};     // editable path field (dialog fallback)
    std::string load_error;               // viewer-set: why the last load failed

    // ---- editing -----------------------------------------------------------
    // The inspector's editable property state for selected_entity (seeded by the
    // viewer, written by the inspector widgets, applied via the general edit seam).
    InspectorState inspector;
    // The in-viewport transform gizmo settings (shares the inspector's edit seam).
    GizmoState     gizmo;
    char           save_path[512] = {0};
    bool           save_dirty = false;    // viewer-set: the scene has unsaved edits
    char           rename_buf[256] = {0}; // rename field for the selection

    // ---- live-world scripting console --------------------------------------
    // The Script panel edits `script_buf`; the viewer writes captured output to
    // `console_log` and the scene's /script node count.
    char        script_buf[16384] = {0};
    std::string console_log;
    int         script_node_count = 0;
};

// ---------------------------------------------------------------------------
// ViewerStats -- the per-frame physics/render facts the stats panel renders. The
// viewer fills these from its frame timing + nk::World capacities + the latest
// render report. Pure data; RecordUi() only reads it.
// ---------------------------------------------------------------------------
struct ViewerStats {
    float    step_time_ms   = 0.0f;
    float    fps            = 0.0f;
    float    frame_time_ms  = 0.0f;
    uint32_t sub_steps      = 1u;
    uint32_t dof            = 0u;
    uint32_t links          = 0u;
    uint32_t bodies         = 0u;
    uint32_t contact_cap    = 0u;
    uint32_t draw_calls     = 0u;
    uint64_t non_bg_pixels  = 0u;
    uint64_t frame_index    = 0u;
    bool     step_healthy   = true;
    bool     cpu_timing_available = false;
    bool     frame_timing_available = false;
    bool     draw_calls_available = false;
    bool     pixel_count_available = false;
    uint32_t debug_colliders = 0u;
    uint32_t debug_contacts = 0u;
    uint32_t debug_skipped_shapes = 0u;
    uint32_t debug_capacity = 0u;
    uint32_t debug_invalid_colliders = 0u;
    uint32_t debug_invalid_contacts = 0u;
    uint64_t debug_omitted_instances = 0u;
    bool     debug_contacts_budget_skipped = false;
    bool     debug_colliders_available = false;
    bool     debug_contacts_available = false;
    float    snapshot_age_ms = 0.0f;      // wall time since the shown snapshot was published
    uint64_t snapshot_drops = 0u;         // snapshots published but never shown
    uint64_t sim_ticks = 0u;
    uint64_t capped_batches = 0u;
    bool     zero_copy = false;           // poses reach the renderer without a host copy
    std::string publish_reason;           // why the host fallback publishes
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

    // Record all panels into the current ImGui frame. `scene` (nullable: empty
    // editor) drives the scene tree; deterministic given identical inputs.
    void RecordUi(const render::RenderWorld& world, const ViewerStats& stats,
                  CameraController& camera, ViewerUiState& ui_state,
                  const nuka::scene::SceneIR* scene = nullptr,
                  uint32_t framebuffer_width = ~uint32_t(0), uint32_t framebuffer_height = ~uint32_t(0));

    // Draw + run the transform gizmo for the selected entity over the viewport.
    // `world` is the entity's current WORLD pose; on a manipulation it is rewritten
    // with the new pose and `changed` latches true, so the viewer routes the result
    // through the SAME edit seam the inspector uses. `ui_state.gizmo.active` is set
    // when the gizmo is hovered/dragged (the viewer suppresses orbit + picking).
    // Call AFTER RecordUi each frame; a no-op when the gizmo is disabled. Uses the
    // CameraController's resolved basis/fov so it aligns with the rendered image.
    void DrawGizmo(CameraController& camera, uint32_t vp_w, uint32_t vp_h,
                   ViewerUiState& ui_state, math::Transform& world, bool& changed);

    // The central dock node in logical and framebuffer pixels; the scene renders only there.
    const SceneViewportRect& Viewport() const { return viewport_; }
    bool OwnsScenePointer() const { return viewport_hovered_; }

private:
    SceneViewportRect viewport_;
    // The docking layout is built ONCE (the first frame the dockspace exists). A
    // latch so we do not re-split every frame (which would fight a user re-dock).
    bool dock_built_ = false;
    bool viewport_hovered_ = false;
};

}  // namespace nuka::runtime::app::viewer
