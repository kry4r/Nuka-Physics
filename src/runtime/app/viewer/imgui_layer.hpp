#pragma once
// The editor shell: menu bar, toolbar, status bar, dockspace layouts, the built-in nui panels,
// the command palette and toasts. It reads and edits plain UI state and queues actions.

#include "math/transform.hpp"
#include "nuka/nuka.h"
#include "render/render_world.hpp"
#include "runtime/app/viewer/editor_actions.hpp"
#include "runtime/app/viewer/nui/nui.hpp"
#include "runtime/app/viewer/property_schema.hpp"
#include "runtime/app/viewer/robot_view.hpp"
#include "runtime/app/viewer/scene_viewport.hpp"
#include "runtime/app/viewer/script_tabs.hpp"
#include "runtime/app/viewer/timeline_view.hpp"
#include "scene/ecs/entity.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nuka::scene {
class SceneIR;
class Registry;
}

namespace nuka::runtime::app::viewer {

class CameraController;

// What the Stage panel shows a node as: "artic", "body", "joint", "script", ... or "group".
std::string NodeKind(const nuka::scene::Registry& reg, nuka::scene::EntityId e);

// Editable values of the selected entity. The host seeds them; widgets write them and latch
// `*_changed` (live preview) or `*_commit` (the gesture ended, record the edit).
struct InspectorState {
    bool valid = false;    // a selectable, editable entity is bound
    bool movable = false;  // physics-driven (teleports) vs static placement
    bool deformable = false;  // a medium the solver deforms

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

// One card of the start page.
struct StartEntry {
    std::string title;
    std::string detail;     // a path, a time or a check's finding
    std::string arg;        // the template or file the card's action takes
    std::string thumbnail;  // a PPM of the scene's viewport at its last save, or empty
    bool available = true;  // false: the file is gone or the check failed
};

// What the start page lists; the host refreshes it while the page shows.
struct StartPage {
    bool shown = false;
    std::vector<StartEntry> templates, recent, recovered, examples, checks;
};

// One importable file of the Assets panel.
struct AssetCard {
    std::string path;
    std::string name;
    std::string root;
    std::string kind;       // "scene", "urdf", "mjcf", "usd" or "mesh"
    std::string thumbnail;  // a PPM, or empty while it is drawn
};

// The Assets panel: its folders (examples, the document's own, then the user's) and their files.
struct AssetBrowser {
    std::vector<std::string> roots;
    std::vector<std::string> user_roots;  // the ones assets.remove_folder can drop
    std::vector<AssetCard> cards;
    uint64_t version = 0u;  // bumped when the cards change
    std::string root;       // the folder shown; empty shows every folder
    std::string kind;       // the kind shown; empty shows every kind
    char filter[128] = {0};
    bool shown = false;     // UI-set: the panel drew this frame, so pictures are wanted
};

// The import wizard over one file's analysis; an empty `path` keeps it closed.
struct ImportWizard {
    std::string path;
    nui::json::Value analysis;  // AnalysisToJson
    nui::json::Value pose;      // where a drop placed it; null keeps the origin
    char name[128] = {0};
    bool ground = true;
    bool open_as_new = false;  // otherwise the asset joins the open document
    bool add_allowed = false;  // a document is open to add to
    bool appear = false;       // host-set: open the modal at the next frame
};

// The collision dialog: what to build for the selection, and the last preview of it.
struct CollisionDialog {
    bool appear = false;
    bool dismiss = false;  // a generate or a load closes the open dialog
    std::string method = "hull";  // hull, vhacd, sdf or mesh
    int max_pieces = 16;
    bool replace = true;
    bool only_missing = false;
    bool subtree = true;
    nui::json::Value preview;  // PreviewToJson
};

// The dialog's options as scene.collision_preview and scene.generate_collision args.
inline nui::json::Value CollisionArgs(const CollisionDialog& d) {
    nui::json::Value args = nui::json::Value::Object();
    args.Set("method", nui::json::Value::Str(d.method));
    args.Set("max_pieces", nui::json::Value::Int(d.max_pieces));
    args.Set("replace", nui::json::Value::Bool(d.replace));
    args.Set("only_missing", nui::json::Value::Bool(d.only_missing));
    args.Set("subtree", nui::json::Value::Bool(d.subtree));
    return args;
}

// A field edit that rebuilds the world, held until Apply & Rebuild.
struct StagedEdit {
    std::string target;
    std::string field;
    std::string label;
    float after = 0.0f;
};

// Solver telemetry as the Solver panel, the heatmap and the toolbar show it; the host fills it.
struct SolverView {
    uint32_t mask = 0u;     // nuka_telemetry_bits_t the world records; 0 is off
    float alert = 1.0e-3f;  // [m/s]
    uint32_t iterations = 0u;
    bool heatmap = false;
    uint64_t session = 0u;
    uint64_t records = 0u, caps = 0u, dropped = 0u;
    uint32_t recent_caps = 0u;
    uint32_t recent_records = 0u;  // the newest records recent_caps counts over
    std::vector<nuka_telemetry_record_t> history;  // the newest records, oldest first
    uint64_t first = 0u;                           // the number of the first of them
    bool marked = false;                           // the last locate's point, drawn in the viewport
    float mark[3] = {0.0f, 0.0f, 0.0f};
    const char* mark_kind = "";  // the located element's label format, index and contact row (or -1)
    uint32_t mark_index = 0u;
    int64_t mark_row = -1;
    float mark_residual = 0.0f;  // [m/s]
};

inline constexpr uint32_t kSolverTelemetryAll =
    NUKA_TELEMETRY_SOLVE | NUKA_TELEMETRY_SWEEPS | NUKA_TELEMETRY_ENERGY | NUKA_TELEMETRY_VERTICES;

// The heatmap colour of a residual [m/s]: four decades up to ten times the alert, the alert at amber.
void HeatColor(float residual, float alert, float rgb[3]);
// The marked element, such as "contact row 12 / particle 147", in the UI language or in English.
std::string MarkLabel(const SolverView& v, bool translate);

// The drawn title bar: the host reports the window state, the UI where the bar may drag the
// window, in window coordinates, for the host's hit test.
struct TitleBar {
    bool maximized = false;
    float height = 0.0f;
    float edge = 0.0f;  // the resize band along a restored window's edges
    std::vector<std::pair<float, float>> drag;  // x spans of the bar that move the window
};

// The viewport pointer: select and move, pull on bodies and cloth, pin them in place, or measure
// between robot links.
enum class ViewportTool : uint8_t { Select, Perturb, Pin, Measure };

// World points to window pixels over the active pane; false behind the camera.
struct ViewProjector {
    float view[16];
    float proj[16];
    ImVec2 vmin, vmax;
    bool operator()(const math::Vec3& p, ImVec2* out) const;
};

// The viewport's shadings in render::ViewMode order, as view.mode and the mode menu name them.
struct ViewModeInfo {
    nuka::render::ViewMode mode;
    const char* name;
    const char* label;
};
inline constexpr ViewModeInfo kViewModes[] = {
    {nuka::render::ViewMode::Lit, "lit", "Lit"},
    {nuka::render::ViewMode::Unlit, "unlit", "Unlit"},
    {nuka::render::ViewMode::Normals, "normals", "Normals"},
    {nuka::render::ViewMode::Material, "material", "Material ID"},
    {nuka::render::ViewMode::Residual, "residual", "Residual"},
    {nuka::render::ViewMode::Contact, "contact", "Contact"},
    {nuka::render::ViewMode::Wireframe, "wireframe", "Wireframe"},
};

// A held load as the viewport draws it, in world space; the host fills it every frame.
struct HeldLoad {
    float at[3] = {0.0f, 0.0f, 0.0f};      // the grab point
    float target[3] = {0.0f, 0.0f, 0.0f};  // where it is pulled
    float force = 0.0f;                    // [N]
    bool body = false;                     // a link or body load, labelled with its force
};

// The RT preview as the viewport shows it; the host bumps `version` with each new image.
struct RtView {
    bool on = false;
    bool shown = false;  // the image shows the current view
    bool converged = false;
    uint32_t spp = 0u;
    uint32_t max_spp = 64u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint64_t version = 0u;
    std::vector<uint8_t> rgb;
};

// The scene viewport's panes: one view, two side by side, or four in a grid.
enum class ViewLayout : uint8_t { Single, Split, Quad };
inline constexpr uint32_t kViewPanes = 4u;

// A pane the layout shows, as its header names it; the host lists them in layout order.
struct ViewPane {
    uint32_t index = 0u;  // which of the kViewPanes views it draws
    nuka::render::ViewMode mode = nuka::render::ViewMode::Lit;
    std::string camera;  // the scene camera node it looks through; empty is the free camera
};

// Transport, selection and panel state. The UI queues actions for anything that changes the
// world and edits only this plain state in place.
struct ViewerUiState {
    bool window_focused = true;
    TitleBar title_bar;
    bool layout_modifier_down = false;
    bool playing = false;
    float speed = 1.0f;  // simulated seconds per wall second
    uint32_t env_index = 0u;

    bool show_colliders = false;
    bool show_contacts = false;
    nuka::render::ViewMode view_mode = nuka::render::ViewMode::Lit;  // the active pane's
    ViewLayout view_layout = ViewLayout::Single;
    uint32_t active_view = 0u;  // the pane input, the gizmo and the HUD go to
    std::vector<ViewPane> panes{ViewPane{}};
    std::vector<std::string> scene_cameras;  // the camera nodes a pane can look through
    RtView rt;
    std::vector<std::string> bookmarks;  // camera bookmark names, in bar order
    bool rendering = false;              // an offline render is running

    nuka::scene::EntityId selected_entity = nuka::scene::kInvalidEntity;
    // Every selected node, the primary `selected_entity` among them.
    std::vector<nuka::scene::EntityId> selection;
    std::vector<std::string> hidden;  // node paths drawn without their visuals
    std::vector<std::string> locked;  // node paths viewport picking skips
    // Bumped by the host whenever the scene tree may have changed.
    uint64_t scene_version = 0u;

    // Per-DOF drive targets of the selected env; a moved slider latches drive_dirty[d].
    std::vector<float> drive_targets;
    std::vector<uint8_t> drive_dirty;
    RobotView robot;

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
    ViewportTool tool = ViewportTool::Select;
    float perturb_gain = 1.0f;   // the grab spring's stiffness over its default
    std::vector<HeldLoad> held;
    // An Alt-drag box selection between two window points while the button is down.
    bool select_box = false;
    float select_box_from[2] = {0.0f, 0.0f};
    float select_box_to[2] = {0.0f, 0.0f};
    char save_path[512] = {0};
    bool save_dirty = false;
    char rename_buf[256] = {0};

    ScriptTabs scripts;
    // Completions from the host's script runner, asked and taken without waiting; unset when
    // the editor has no scripting.
    std::function<uint64_t(std::string code, int line, int column)> complete;
    std::function<bool(uint64_t ticket, std::string* json)> take_completion;
    double script_busy_s = 0.0;  // how long the running script has run; 0 when idle
    bool script_rerun = false;   // saving a tab runs it again
    std::string console_log;
    int console_level = 0;  // the console's level filter, see nui::LogView::level
    int script_node_count = 0;

    // A script owns the world: New and Open start another editor, closing keeps the world.
    bool script_session = false;
    bool script_driven = false;  // the script steps the world, not the editor
    std::string script_name;
    std::string doc_title = "untitled";
    std::string prompt;  // the action waiting on save, don't save or cancel
    StartPage start;

    AssetBrowser assets;
    ImportWizard import;
    CollisionDialog collision;
    // The selection's fields, host-built whenever the selection or the document changes.
    std::vector<PropertyGroup> properties;
    uint64_t properties_version = 0u;  // bumped with every rebuild of `properties`
    std::vector<StagedEdit> staged;
    SolverView solver;
    TimelineInfo timeline;  // the physics thread's history, refreshed every frame
    bool exporting = false;    // an mp4 export runs
    float export_done = 0.0f;  // its captured share, negative while ffmpeg encodes
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

    // The built-in panels, grouped as the editor's own extensions.
    struct BuiltinExtension {
        std::string id;
        std::string title;
        std::string description;
        std::vector<std::string> panels;
    };
    static const std::vector<BuiltinExtension>& Builtins();
    // A built-in panel comes back open in the column the current layout gives it.
    void AddBuiltinPanel(const std::string& panel_id);
    void RemoveBuiltinPanel(const std::string& panel_id);
    // The actions that come and go with a built-in extension's panels.
    std::vector<Action> BuiltinActions(const BuiltinExtension& ext);
    // Column-major view and GL-style projection of the scene viewport.
    void CameraMatrices(const CameraController& camera, float* view, float* proj) const;

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
    // A pane's rect this frame, invalid while the layout hides it; the active one is Viewport().
    const SceneViewportRect& PaneRect(uint32_t index) const { return pane_rects_[index]; }
    // The pane under a free pointer, or -1.
    int HoveredPane() const { return hovered_pane_; }
    // Points the viewport, its pointer and the gizmo at pane `index` for the rest of the frame.
    void SetActiveView(uint32_t index);

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
    void BuildStage(nui::VStack& root);
    void BuildProperties(nui::VStack& root);
    void BuildFile(nui::VStack& root);
    void BuildStats(nui::VStack& root);
    void BuildDebug(nui::VStack& root);
    void BuildSolver(nui::VStack& root);
    void BuildTimeline(nui::VStack& root);
    void BuildCamera(nui::VStack& root);
    void BuildJoints(nui::VStack& root);
    void BuildPoses(nui::VStack& root);
    void BuildScript(nui::VStack& root);
    void BuildConsole(nui::VStack& root);
    // The script tab bar with its keys, completions, disk-change bar and close prompt.
    void DrawScriptTabs();
    void DrawScriptStatus();
    // The console's Python line: Enter runs it, Tab completes, Up/Down walk the history.
    void DrawRepl();
    void BuildInspector(nui::VStack& root);
    void BuildAssets(nui::VStack& root);
    // The schema groups of the selection's fields, after the transform and material.
    void BuildFieldGroups(nui::VStack& root);
    // Keeps a Properties group open or closed as the user left it across the panel's rebuilds.
    nui::Collapsible& Fold(nui::Collapsible& group, const std::string& key);
    // A field edit reaches the world now when live, else it waits in the staged edits.
    void CommitField(const PropertyGroup& group, const PropertyField& field, float value);
    void DrawMenuBar();
    void DrawToolbar();
    void DrawStatusBar();
    void DrawPalette();
    void DrawAbout();
    void DrawStartPage();
    void DrawPrompt();
    void DrawImportWizard();
    void DrawCollisionDialog();
    // The floating Gain Tuner while it is open.
    void DrawGainTuner();
    // The floating Robot Assembler while it is open.
    void DrawAssembler();
    // The heatmap's scale and the located element's marker over the scene viewport.
    void DrawSolverHud(const ImVec2& vmin, const ImVec2& vmax);
    // Each held load's pull from its grab point toward its target, and an IK drag's miss as a dashed
    // line from the link to its goal.
    void DrawPerturbHud(const ImVec2& vmin, const ImVec2& vmax);
    // The robot overlays, the Measure tool's line and the hovered link's tooltip.
    void DrawRobotHud(const ImVec2& vmin, const ImVec2& vmax);
    ViewProjector Projector(const ImVec2& vmin, const ImVec2& vmax) const;
    // The viewport's top-right tools (RT preview, Render, layout, shading) and the box of a box selection.
    void DrawViewTools(const ImVec2& vmin, const ImVec2& vmax);
    // The RT preview, Render and layout buttons on one line.
    void DrawViewButtons();
    // The RT preview image over the viewport, under its HUD.
    void DrawRtImage(const ImVec2& vmin, const ImVec2& vmax);
    // Camera bookmark chips along the viewport's bottom edge.
    void DrawBookmarkBar(const ImVec2& vmin, const ImVec2& vmax);
    // The panes' dividers, the active one's border and each pane's camera, mode and maximize header.
    void DrawPaneHeaders(const ImVec2& cmin, const ImVec2& cmax, const ImVec2& pane, float gap);
    // While an asset card is dragged, the viewport takes the drop at the ground under the pointer.
    void DrawAssetDrop();
    // The texture of a PPM file, reloaded when the file changes; null without the file.
    ImTextureData* Thumbnail(const std::string& file);
    void DockLayout(unsigned int dockspace);
    bool MenuAction(const std::string& id);

    nui::Host host_;
    ActionRegistry* registry_ = nullptr;
    Frame frame_;
    SceneViewportRect viewport_;
    bool viewport_hovered_ = false;
    std::array<SceneViewportRect, kViewPanes> pane_rects_{};
    int hovered_pane_ = -1;
    bool pointer_free_ = false;  // focused, no layout modifier and no ImGui window under the pointer

    std::string layout_ = "authoring";
    bool layout_pending_ = true;
    int layout_settle_ = 0;  // frames until the new layout's tab bars scroll to their selected tabs
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
    char repl_[2048] = {0};
    std::vector<std::string> repl_history_;
    size_t repl_back_ = 0u;           // history steps back from the newest; 0 is the line being typed
    uint64_t repl_ticket_ = 0u;       // the console completion in flight
    std::vector<std::string> repl_options_;
    std::string repl_insert_;         // completion text the input takes on its next callback
    size_t repl_word_ = 0u;           // bytes of the word the insert replaces
    bool repl_focus_ = false;
    uint64_t shown_tab_ = 0u;         // the script tab the tab bar shows as selected
    uint64_t close_ask_ = 0u;         // the script tab waiting on save, discard or cancel
    nui::Plot* frame_plot_ = nullptr;
    nui::Plot* step_plot_ = nullptr;
    uint64_t staged_version_ = 0u;               // bumped whenever the UI changes the staged edits
    std::unordered_map<std::string, bool> folds_;  // Properties groups the user opened or closed
    uint64_t scrub_sent_ = ~0ull;  // the tick the timeline drag last asked for
    char branch_name_[64] = {0};   // the branch rename field
    std::unique_ptr<ImTextureData> rt_texture_;
    uint64_t rt_version_ = 0u;
    std::vector<std::unique_ptr<ImTextureData>> rt_retired_;  // until the backend destroys them
    char render_path_[512] = {0};  // the Render dialog's fields
    int render_size_[2] = {1920, 1080};
    int render_spp_ = 64;
    char bookmark_name_[128] = {0};  // the bookmark rename field
    char pose_name_[128] = {0};      // the pose rename field
    float tuner_gains_[3] = {};      // the tuner's kp, damping and force limit fields
    int tuner_active_ = -1;          // the field being dragged, which keeps its own value

    bool palette_open_ = false;
    bool palette_focus_ = false;
    char palette_query_[256] = {0};
    int palette_index_ = 0;
    std::vector<std::string> palette_recent_;
    bool about_open_ = false;
    struct Thumb {
        std::string file;
        std::filesystem::file_time_type stamp;
        std::unique_ptr<ImTextureData> texture;
    };
    std::vector<Thumb> thumbs_;  // kept for the editor's lifetime; ImGui's shutdown frees them
};

}  // namespace nuka::runtime::app::viewer
