#pragma once
// The editor shell: menu bar, toolbar, status bar, dockspace layouts, the built-in nui panels,
// the command palette and toasts. It reads and edits plain UI state and queues actions.

#include "math/transform.hpp"
#include "render/render_world.hpp"
#include "runtime/app/viewer/editor_actions.hpp"
#include "runtime/app/viewer/nui/nui.hpp"
#include "runtime/app/viewer/scene_viewport.hpp"
#include "scene/ecs/entity.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace nuka::scene {
class SceneIR;
}

namespace nuka::runtime::app::viewer {

class CameraController;

// Editable values of the selected entity. The host seeds them; widgets write them and latch
// `*_changed` (live preview) or `*_commit` (the gesture ended, record the edit).
struct InspectorState {
    bool valid = false;    // a selectable, editable entity is bound
    bool movable = false;  // physics-driven (teleports) vs static placement

    float pos[3] = {0.0f, 0.0f, 0.0f};      // world position
    float rot_deg[3] = {0.0f, 0.0f, 0.0f};  // intrinsic XYZ degrees
    bool transform_changed = false;
    bool transform_commit = false;

    bool has_material = false;
    float base_color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float roughness = 0.5f;
    float metallic = 0.0f;
    float emissive[3] = {0.0f, 0.0f, 0.0f};
    float sheen = 0.0f;         // offline beauty only
    float transmission = 0.0f;  // offline beauty only
    float ior = 1.0f;           // offline beauty only
    float absorption[3] = {0.0f, 0.0f, 0.0f};
    bool material_changed = false;
    bool material_commit = false;
};

// The viewport gizmo. Translate and rotate map onto the rigid transform; there is no scale.
struct GizmoState {
    enum class Op : uint8_t { Translate, Rotate };
    bool enabled = true;
    Op op = Op::Translate;
    bool local = false;
    bool snap_on = false;
    float snap_translate = 0.25f;  // world units
    float snap_rotate = 15.0f;     // degrees
    bool active = false;     // host-set: hovered or dragged this frame
    bool using_now = false;  // host-set: dragged this frame; the release ends one edit
};

// Transport, selection and panel state. The UI queues actions for anything that changes the
// world and edits only this plain state in place.
struct ViewerUiState {
    bool window_focused = true;
    bool layout_modifier_down = false;
    bool playing = false;
    float speed = 1.0f;  // simulated seconds per wall second
    uint32_t env_index = 0u;

    bool show_colliders = false;
    bool show_contacts = false;

    nuka::scene::EntityId selected_entity = nuka::scene::kInvalidEntity;
    // Bumped by the host whenever the scene tree may have changed.
    uint64_t scene_version = 0u;

    // Per-DOF drive targets of the selected env; a moved slider latches drive_dirty[d].
    std::vector<float> drive_targets;
    std::vector<uint8_t> drive_dirty;
    std::vector<std::string> dof_labels;  // empty or one per DOF

    // Keyboard teleop: [ / ] nudge a DOF, WASD/QE feed a command-consuming controller.
    bool teleop_enabled = false;
    int teleop_dof = 0;
    float teleop_step = 0.05f;
    float teleop_lin = 1.0f;
    float teleop_yaw = 1.0f;
    float teleop_cmd[3] = {0.0f, 0.0f, 0.0f};       // host-set live command
    float teleop_cmd_base[3] = {0.0f, 0.0f, 0.0f};  // keys add on top of this baseline
    bool teleop_hold = false;  // PD-hold makes drive targets effective without a policy
    float teleop_kp = 20.0f;
    float teleop_kd = 0.5f;

    // Queued for the next frame top, so recording the UI never changes the world.
    std::vector<ActionCall> actions;

    bool can_undo = false;
    bool can_redo = false;

    bool has_scene = false;
    std::string loaded_path;
    char load_path[512] = {0};
    std::string load_error;

    InspectorState inspector;
    GizmoState gizmo;
    char save_path[512] = {0};
    bool save_dirty = false;
    char rename_buf[256] = {0};

    char script_buf[16384] = {0};
    std::string console_log;
    int script_node_count = 0;
};

// Per-frame physics and render facts the panels show.
struct ViewerStats {
    float step_time_ms = 0.0f;
    float fps = 0.0f;
    float frame_time_ms = 0.0f;
    uint32_t sub_steps = 1u;
    uint32_t dof = 0u;
    uint32_t links = 0u;
    uint32_t bodies = 0u;
    uint32_t contact_cap = 0u;
    uint32_t draw_calls = 0u;
    uint64_t non_bg_pixels = 0u;
    uint64_t frame_index = 0u;
    bool step_healthy = true;
    bool cpu_timing_available = false;
    bool frame_timing_available = false;
    bool draw_calls_available = false;
    bool pixel_count_available = false;
    uint32_t debug_colliders = 0u;
    uint32_t debug_contacts = 0u;
    uint32_t debug_skipped_shapes = 0u;
    uint32_t debug_capacity = 0u;
    uint32_t debug_invalid_colliders = 0u;
    uint32_t debug_invalid_contacts = 0u;
    uint64_t debug_omitted_instances = 0u;
    bool debug_contacts_budget_skipped = false;
    bool debug_colliders_available = false;
    bool debug_contacts_available = false;
    float snapshot_age_ms = 0.0f;  // wall time since the shown snapshot was published
    uint64_t snapshot_drops = 0u;  // snapshots published but never shown
    uint64_t sim_ticks = 0u;
    uint64_t capped_batches = 0u;
    bool zero_copy = false;  // poses reach the renderer without a host copy
    std::string publish_reason;
    std::string device_name;
};

class SceneItems;

class ImGuiLayer {
public:
    ImGuiLayer();
    ~ImGuiLayer();
    ImGuiLayer(const ImGuiLayer&) = delete;
    ImGuiLayer& operator=(const ImGuiLayer&) = delete;

    // Sets the docking flag; call once before the first NewFrame.
    void EnableDocking();
    // Adds the shell's actions and gives every registered action its icon and menu visibility.
    void RegisterActions(ActionRegistry& registry);
    // Re-themes when the display or UI scale changed; call before NewFrame.
    void PrepareFrame(float display_scale);
    float UiScale() const { return ui_scale_; }
    void SetUiScale(float scale);
    // Applies a named layout preset at the next frame; false for an unknown name.
    bool SetLayout(std::string_view name);
    const std::string& Layout() const { return layout_; }
    nui::Host& Nui() { return host_; }

    // Records the frame's UI between NewFrame and Render. `scene` is null without a scene.
    void RecordUi(const render::RenderWorld& world, const ViewerStats& stats, CameraController& camera,
                  ViewerUiState& ui_state, const nuka::scene::SceneIR* scene = nullptr,
                  uint32_t framebuffer_width = ~uint32_t(0), uint32_t framebuffer_height = ~uint32_t(0));

    // Runs the gizmo on the selection's world pose; a drag rewrites `world` and sets `changed`.
    void DrawGizmo(CameraController& camera, uint32_t vp_w, uint32_t vp_h, ViewerUiState& ui_state,
                   math::Transform& world, bool& changed);

    // The central dock node in logical and framebuffer pixels; the scene renders only there.
    const SceneViewportRect& Viewport() const { return viewport_; }
    bool OwnsScenePointer() const { return viewport_hovered_; }

    // The data the panels read this frame.
    struct Frame {
        const render::RenderWorld* world = nullptr;
        const ViewerStats* stats = nullptr;
        CameraController* camera = nullptr;
        ViewerUiState* ui = nullptr;
        const nuka::scene::SceneIR* scene = nullptr;
    };

private:
    void Queue(const std::string& id, nui::json::Value args = nui::json::Value::Object());
    void AddPanels();
    void BuildStage(nui::VStack& root);
    void BuildProperties(nui::VStack& root);
    void BuildFile(nui::VStack& root);
    void BuildStats(nui::VStack& root);
    void BuildDebug(nui::VStack& root);
    void BuildCamera(nui::VStack& root);
    void BuildDrive(nui::VStack& root);
    void BuildScript(nui::VStack& root);
    void BuildConsole(nui::VStack& root);
    void BuildInspector(nui::VStack& root);
    void DrawMenuBar();
    void DrawToolbar();
    void DrawStatusBar();
    void DrawPalette();
    void DrawAbout();
    void DockLayout(unsigned int dockspace);
    bool MenuAction(const std::string& id);

    nui::Host host_;
    ActionRegistry* registry_ = nullptr;
    Frame frame_;
    SceneViewportRect viewport_;
    bool viewport_hovered_ = false;

    std::string layout_ = "authoring";
    bool layout_pending_ = true;
    unsigned int dock_left_ = 0u, dock_right_ = 0u, dock_bottom_ = 0u;
    float ui_scale_ = 1.0f;
    float applied_scale_ = 0.0f;

    std::shared_ptr<SceneItems> stage_items_;
    nui::TreeView* stage_tree_ = nullptr;
    nuka::scene::EntityId revealed_ = nuka::scene::kInvalidEntity;
    const nuka::scene::SceneIR* stage_scene_ = nullptr;
    // Pending Expand/Collapse All (+1/-1), applied after any scene-change reset of the tree.
    int stage_fold_ = 0;
    char stage_filter_[128] = {0};
    char console_filter_[128] = {0};
    nui::Plot* frame_plot_ = nullptr;
    nui::Plot* step_plot_ = nullptr;

    bool palette_open_ = false;
    bool palette_focus_ = false;
    char palette_query_[256] = {0};
    int palette_index_ = 0;
    std::vector<std::string> palette_recent_;
    bool about_open_ = false;
};

}  // namespace nuka::runtime::app::viewer
