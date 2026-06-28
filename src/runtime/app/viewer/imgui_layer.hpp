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
// DETERMINISM (GATE-B): RecordUi() takes NO time/animation input -- every widget
// is driven from ViewerUiState + the snapshot. With a FIXED ViewerUiState the
// produced ImDrawData is deterministic (two records memcmp-identical once
// composited), which the GATE-B smoke relies on.
//
// HOST-ONLY / zero-CUDA-token. This TU includes the vendored ImGui (SYSTEM
// include) so it is gated behind NK_BUILD_VULKAN_VALIDATION with the rest of the
// viewer; it pulls no Vulkan headers.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "render/render_world.hpp"
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
    bool active  = false;            // viewer-set: hovered/used this frame
};

// ---------------------------------------------------------------------------
// ViewerUiState -- the mutable transport / selection state the toolbar drives and
// the viewer reads to pace the simulation. RecordUi() mutates this in response to
// button clicks (play_toggled / step_requested / reset_requested are one-shot
// edge flags the viewer consumes + clears each frame). Holding it as plain data
// (no time input) is what makes the GATE-B composite deterministic.
// ---------------------------------------------------------------------------
struct ViewerUiState {
    bool     playing   = false;   // open PAUSED on the authored pose; Play/Step advances
    float    speed     = 1.0f;    // x0.25 .. x4 multiplier
    uint32_t env_index = 0u;      // selected env (D4)

    // The single selection key: the picked/clicked ENTITY (viewport picker + tree
    // click write it; tree highlight + Entity panel read it). kInvalidEntity = none.
    nuka::scene::EntityId selected_entity = nuka::scene::kInvalidEntity;

    // VIEW-2: the GENERIC per-DOF drive-target editor state. `drive_targets[d]` is
    // the slider value for DOF d of the SELECTED env; `drive_dirty[d]` latches when
    // a slider moved this frame so the viewer uploads ONLY changed DOFs (per-DOF
    // FieldId::DriveTarget UploadField, env-major offset env*per_env + d). NEVER a
    // hardcoded grasp/choreography table (OD-7: a flat per-DOF editor). `dof_labels`
    // is the OPTIONAL cooked name->dof labelling (cosmetic); empty -> "dof N".
    // For GATE-B (R12) these are pre-seeded to FIXED values so RecordUi is
    // deterministic (no time/animation input).
    std::vector<float>       drive_targets;
    std::vector<uint8_t>     drive_dirty;   // per-DOF "changed this frame" latch
    std::vector<std::string> dof_labels;    // optional, size 0 or == drive_targets
    bool                     show_drive_panel = true;

    // One-shot edge flags set by RecordUi(), consumed + cleared by the viewer.
    bool play_toggled    = false;
    bool step_requested  = false;
    bool reset_requested = false;
    bool camera_reset    = false;

    // ---- editor Load state (empty-start + runtime load) --------------------
    // Viewer-set has_scene/loaded_path; the panel's "Open Scene" / load_path set
    // load_request (consumed via LoadEditorScene); unload_request drops to empty.
    bool        has_scene      = false;   // viewer-set: a world is cooked
    std::string loaded_path;              // viewer-set: shown in the panel
    char        load_path[512] = {0};     // editable path field (dialog fallback)
    std::string load_request;             // one-shot: path to load
    bool        unload_request = false;   // one-shot: drop to empty

    // ---- editing -----------------------------------------------------------
    // The inspector's editable property state for selected_entity (seeded by the
    // viewer, written by the inspector widgets, applied via the general edit seam).
    InspectorState inspector;
    // The in-viewport transform gizmo settings (shares the inspector's edit seam).
    GizmoState     gizmo;
    // One-shot: a path to Save the edited scene to (Save / Save As). save_dirty is
    // a viewer-set hint that the in-memory scene has unsaved edits.
    std::string    save_request;
    char           save_path[512] = {0};
    bool           save_dirty = false;

    // ---- tree edit ops -----------------------------------------------------
    // One-shot structural-edit requests keyed on selected_entity (the chosen
    // parent / target). The viewer applies them through the general SceneIR seam,
    // re-cooking the world for the structural ones (add / delete).
    bool        add_box_request = false;   // add a box body under the selection
    bool        delete_request  = false;   // delete the selection + its subtree
    char        rename_buf[256] = {0};     // rename field for the selection
    bool        rename_request  = false;   // commit rename_buf onto the selection
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

    // Record all panels into the current ImGui frame. `scene` (nullable: empty
    // editor) drives the scene tree; deterministic given identical inputs.
    void RecordUi(const render::RenderWorld& world, const ViewerStats& stats,
                  CameraController& camera, ViewerUiState& ui_state,
                  const nuka::scene::SceneIR* scene = nullptr);

    // Draw + run the transform gizmo for the selected entity over the viewport.
    // `world` is the entity's current WORLD pose; on a manipulation it is rewritten
    // with the new pose and `changed` latches true, so the viewer routes the result
    // through the SAME edit seam the inspector uses. `ui_state.gizmo.active` is set
    // when the gizmo is hovered/dragged (the viewer suppresses orbit + picking).
    // Call AFTER RecordUi each frame; a no-op when the gizmo is disabled. Uses the
    // CameraController's resolved basis/fov so it aligns with the rendered image.
    void DrawGizmo(CameraController& camera, uint32_t vp_w, uint32_t vp_h,
                   ViewerUiState& ui_state, math::Transform& world, bool& changed);

private:
    // The docking layout is built ONCE (the first frame the dockspace exists). A
    // latch so we do not re-split every frame (which would fight a user re-dock).
    bool dock_built_ = false;
};

}  // namespace nuka::runtime::app::viewer
