// The editor shell. Panels are nui trees the host rebuilds only when their key changes; the
// menu bar, toolbar, status bar, palette and viewport HUD draw immediately each frame.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "imgui_internal.h"
#include "ImGuizmo.h"
#include "TextEditor.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/camera_controller.hpp"
#include "runtime/app/viewer/editor_profile.hpp"
#include "runtime/app/viewer/shell_util.hpp"
#include "scene/ecs/components.hpp"
#include "scene/ecs/registry.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_ir.hpp"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <unordered_map>
#include <utility>

namespace nuka::runtime::app::viewer {

namespace fs = std::filesystem;
namespace json = nuka::scene::json;
namespace theme = nuka::render::imgui::theme;
namespace icon = nuka::render::imgui::icon;
namespace draw = nuka::nui::draw;
using nuka::nui::TextStyle;
using nuka::nui::Tone;
using nuka::nui::Tr;
using nuka::render::imgui::FontRole;
using nuka::render::imgui::GetNukaFont;
using namespace shell;

struct StageKind {
    const char* glyph;
    ImVec4 color;
    const char* badge;
};

namespace {

constexpr std::pair<std::string_view, std::string_view> kKeyNames[] = {{"Equal", "="}, {"Minus", "-"}};

// The first alternative of a shortcut, with key names shown as the keys print.
std::string ShortcutText(const std::string& shortcut) {
    std::string text = shortcut.substr(0u, shortcut.find(','));
    for (const auto& [name, key] : kKeyNames) {
        const size_t at = text.find(name);
        if (at != std::string::npos) text.replace(at, name.size(), key);
    }
    return text;
}

constexpr float kSpeeds[] = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
constexpr const char* kSpeedLabels[] = {"0.25\xC3\x97", "0.5\xC3\x97", "1\xC3\x97", "2\xC3\x97", "4\xC3\x97"};

struct SpawnChoice {
    const char* label;
    const char* kind;
    const char* glyph;
};
const SpawnChoice kSpawns[] = {{"Box", "box", icon::kBox},
                               {"Sphere", "sphere", icon::kSphere},
                               {"Capsule", "capsule", icon::kCapsule},
                               {"Plane", "plane", icon::kPlane}};

struct PanelSpec {
    const char* id;
    const char* title;
    const char* glyph;
};
const PanelSpec kPanels[] = {{"stage", "Stage", icon::kStage},       {"property", "Properties", icon::kProperty},
                             {"file", "File", icon::kFile},          {"assets", "Assets", icon::kFolder},
                             {"stats", "Stats", icon::kStats},       {"debug", "Debug", icon::kDebug},
                             {"solver", "Solver", icon::kPlot},      {"timeline", "Timeline", icon::kTimeline},
                             {"camera", "Camera", icon::kCamera},
                             {"joints", "Joints", icon::kJoint},
                             {"poses", "Poses", icon::kRobot},
                             {"script", "Script", icon::kScript},    {"console", "Console", icon::kConsole},
                             {"nui", "UI Inspector", icon::kInspector}};

// Panels per dock column; the first of each column is its selected tab.
struct LayoutPreset {
    const char* name;
    const char* label;
    std::vector<const char*> left, right, bottom;
};
const LayoutPreset kLayouts[] = {
    {"authoring", "Authoring", {"stage", "file"}, {"property", "camera"},
     {"assets", "console", "script", "stats", "timeline"}},
    {"debug", "Debug", {"stage"}, {"solver", "debug", "stats", "property"}, {"console", "timeline", "script", "assets"}},
    {"robot", "Robot", {"stage"}, {"joints", "poses", "property"}, {"console", "timeline", "stats", "assets"}},
    {"data", "Data", {"file", "stage"}, {"stats", "camera", "property"}, {"assets", "console", "script", "timeline"}},
    {"training", "Training", {"stats", "stage"}, {"joints", "camera"}, {"console", "script", "assets", "timeline"}},
};

const LayoutPreset* FindLayout(std::string_view name) {
    for (const LayoutPreset& preset : kLayouts)
        if (name == preset.name) return &preset;
    return nullptr;
}

bool Builtin(std::string_view id) {
    return std::any_of(std::begin(kPanels), std::end(kPanels), [&](const PanelSpec& p) { return id == p.id; });
}

// The dock column a layout gives a panel: "left", "right", "bottom", or "" when it has none.
const char* LayoutColumn(const LayoutPreset& preset, std::string_view id) {
    const auto has = [&](const std::vector<const char*>& ids) {
        return std::any_of(ids.begin(), ids.end(), [&](const char* p) { return id == p; });
    };
    return has(preset.left) ? "left" : has(preset.right) ? "right" : has(preset.bottom) ? "bottom" : "";
}

// A menu lists its actions in this order; actions of its categories it does not name follow.
struct MenuSpec {
    std::string title;
    std::vector<std::string> categories;
    std::vector<std::string> items;
};

const std::vector<MenuSpec>& Menus() {
    static const std::vector<MenuSpec> menus = [] {
        std::vector<std::string> window;
        for (const LayoutPreset& preset : kLayouts) window.push_back(std::string("window.layout_") + preset.name);
        window.push_back("-");
        for (const PanelSpec& p : kPanels) window.push_back(std::string("window.toggle_") + p.id);
        window.push_back("-");
        window.push_back("window.reset_layout");
        return std::vector<MenuSpec>{
            {"File",
             {"file"},
             {"file.new", "file.open", "file.open_recent", "file.open_window", "-", "file.import", "-", "file.save",
              "file.save_as", "file.revert", "-", "file.close", "app.exit"}},
            {"Edit", {"edit"}, {"edit.undo", "edit.redo", "-", "edit.delete", "-", "scene.set_visible", "scene.set_locked"}},
            {"Create",
             {"scene"},
             {"scene.add_box", "scene.add_sphere", "scene.add_capsule", "scene.add_plane", "-", "file.import",
              "scene.collision_dialog"}},
            {"View",
             {"view"},
             {"app.command_palette", "-", "view.frame_all", "view.frame_selected", "view.reset_camera", "-",
              "view.layout_single", "view.layout_split", "view.layout_quad", "view.maximize", "-",
              "view.colliders", "view.contacts", "view.hide_overlays", "-", "scene.expand_all", "scene.collapse_all", "-",
              "view.ui_scale_up", "view.ui_scale_down", "view.ui_scale_reset", "view.toggle_language", "-", "view.screenshot"}},
            {"Simulation",
             {"sim", "solver", "timeline"},
             {"sim.toggle_play", "sim.step", "sim.reset", "-", "timeline.fork", "timeline.ghost", "timeline.clear", "-",
              "tool.perturb", "tool.pin", "-", "solver.telemetry", "solver.heatmap", "solver.locate"}},
            {"Robot",
             {"robot"},
             {"robot.fix_base", "robot.drive", "robot.watch", "robot.gain_tuner", "robot.assembler", "-", "robot.pose_capture",
              "robot.poses_save", "robot.poses_load"}},
            {"Tools",
             {"script", "tools"},
             {"script.new", "script.open", "script.save", "-", "script.run", "script.stop", "script.toggle_rerun", "-",
              "script.attach", "script.load", "script.clear_console", "-", "tools.nui_inspector"}},
            {"Window", {"window"}, window},
            {"Help", {"help"}, {"help.about"}},
            {"Extensions", {"extensions"}, {"extensions.manager", "extensions.rescan", "-"}},
        };
    }();
    return menus;
}

const char* CategoryTitle(const std::string& category) {
    for (const MenuSpec& menu : Menus())
        for (const std::string& c : menu.categories)
            if (c == category) return menu.title.c_str();
    return category.c_str();
}

// A subsequence match of lower-case text; consecutive letters and word starts score higher.
int Fuzzy(std::string_view needle, std::string_view hay) {
    int score = 0;
    size_t at = 0u, last = std::string_view::npos;
    for (const char c : needle) {
        while (at < hay.size() && hay[at] != c) ++at;
        if (at == hay.size()) return -1;
        score += 1;
        if (last != std::string_view::npos && at == last + 1u) score += 5;
        if (at == 0u || hay[at - 1u] == ' ' || hay[at - 1u] == '.' || hay[at - 1u] == '_') score += 8;
        last = at++;
    }
    return score;
}

StageKind ClassifyNode(const nuka::scene::Registry& reg, nuka::scene::EntityId e) {
    using namespace nuka::scene;
    if (e == kInvalidEntity || !reg.Alive(e)) return {icon::kScene, theme::kFgMuted, ""};
    if (const auto* sk = reg.Get<SystemKindComponent>(e)) {
        switch (sk->kind) {
            case SystemKindComponent::Articulated: return {icon::kRobot, theme::kAccent, "artic"};
            case SystemKindComponent::Soft: return {icon::kSoft, theme::kInfo, "soft"};
            case SystemKindComponent::Cloth: return {icon::kCloth, theme::kInfo, "cloth"};
            case SystemKindComponent::Fluid: return {icon::kFluid, theme::kInfo, "fluid"};
            case SystemKindComponent::Rigid: break;
        }
    }
    if (const auto* body = reg.Get<RigidBodyComponent>(e))
        return body->kinematic ? StageKind{icon::kStatic, theme::kFgDim, "static"} : StageKind{icon::kBody, theme::kFg, "body"};
    if (reg.Has<JointComponent>(e)) return {icon::kJoint, theme::kAccentDim, "joint"};
    if (reg.Has<CollisionShapeComponent>(e)) return {icon::kCollider, theme::kFgMuted, "geom"};
    if (reg.Has<VisualMeshComponent>(e)) return {icon::kVisual, theme::kFgMuted, "visual"};
    if (reg.Has<CameraComponent>(e)) return {icon::kCameraNode, theme::kFgMuted, "camera"};
    if (reg.Has<LightComponent>(e)) return {icon::kLight, theme::kWarn, "light"};
    if (reg.Has<ScriptComponent>(e)) return {icon::kScriptNode, theme::kInfo, "script"};
    return {icon::kGroup, theme::kFgDim, ""};
}

std::shared_ptr<nuka::scene::SceneNode> SelectedNode(const ImGuiLayer::Frame& f) {
    if (f.scene == nullptr || f.ui->selected_entity == nuka::scene::kInvalidEntity) return nullptr;
    return f.scene->Ecs().NodeOf(f.ui->selected_entity);
}

std::string SelectionPath(const ImGuiLayer::Frame& f) {
    const auto node = SelectedNode(f);
    if (!node) return std::string();
    const std::string path = f.scene->Tree().PathOf(node);
    return path.empty() ? node->name : path;
}

nui::ModelPtr<nui::Vec3> Floats3(float* v) {
    return nui::Bind<nui::Vec3>([v] { return nui::Vec3{v[0], v[1], v[2]}; },
                                [v](const nui::Vec3& x) { std::copy(x.begin(), x.end(), v); });
}

nui::ModelPtr<nui::Color> Floats4(float* v) {
    return nui::Bind<nui::Color>([v] { return nui::Color{v[0], v[1], v[2], v[3]}; },
                                 [v](const nui::Color& x) { std::copy(x.begin(), x.end(), v); });
}

nui::ModelPtr<nui::Color> Rgb(float* v) {
    return nui::Bind<nui::Color>([v] { return nui::Color{v[0], v[1], v[2], 1.0f}; },
                                 [v](const nui::Color& x) { std::copy(x.begin(), x.begin() + 3, v); });
}

void DrawNodeTree(const nui::Node& node) {
    const std::string text = Fmt("%s  #%u  %s  draws %llu", node.Kind(), node.handle, node.id.c_str(),
                                 static_cast<unsigned long long>(node.draws));
    if (node.Children().empty()) {
        ImGui::BulletText("%s", text.c_str());
        return;
    }
    if (ImGui::TreeNode(&node, "%s", text.c_str())) {
        for (const auto& child : node.Children()) DrawNodeTree(*child);
        ImGui::TreePop();
    }
}

// ---- gizmo matrices: column-major, matching the renderer's camera; GL-style projection
// because ImGuizmo applies its own screen flip.

void ModelMatrix(const math::Transform& t, float* m) {
    const math::Quat q = t.rotation.Normalized();
    const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    m[0] = 1.0f - 2.0f * (yy + zz); m[1] = 2.0f * (xy + wz);         m[2]  = 2.0f * (xz - wy); m[3] = 0.0f;
    m[4] = 2.0f * (xy - wz);        m[5] = 1.0f - 2.0f * (xx + zz);  m[6]  = 2.0f * (yz + wx); m[7] = 0.0f;
    m[8] = 2.0f * (xz + wy);        m[9] = 2.0f * (yz - wx);         m[10] = 1.0f - 2.0f * (xx + yy); m[11] = 0.0f;
    m[12] = t.position.x; m[13] = t.position.y; m[14] = t.position.z; m[15] = 1.0f;
}

// The 3x3 stays a pure rotation under translate and rotate, so scale never appears.
math::Transform TransformFromModel(const float* m) {
    math::Transform t;
    t.position = math::Vec3{m[12], m[13], m[14]};
    const float r00 = m[0], r10 = m[1], r20 = m[2];
    const float r01 = m[4], r11 = m[5], r21 = m[6];
    const float r02 = m[8], r12 = m[9], r22 = m[10];
    const float trace = r00 + r11 + r22;
    math::Quat q;
    if (trace > 0.0f) {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.w = 0.25f * s; q.x = (r21 - r12) / s; q.y = (r02 - r20) / s; q.z = (r10 - r01) / s;
    } else if (r00 > r11 && r00 > r22) {
        const float s = std::sqrt(1.0f + r00 - r11 - r22) * 2.0f;
        q.w = (r21 - r12) / s; q.x = 0.25f * s; q.y = (r01 + r10) / s; q.z = (r02 + r20) / s;
    } else if (r11 > r22) {
        const float s = std::sqrt(1.0f + r11 - r00 - r22) * 2.0f;
        q.w = (r02 - r20) / s; q.x = (r01 + r10) / s; q.y = 0.25f * s; q.z = (r12 + r21) / s;
    } else {
        const float s = std::sqrt(1.0f + r22 - r00 - r11) * 2.0f;
        q.w = (r10 - r01) / s; q.x = (r02 + r20) / s; q.y = (r12 + r21) / s; q.z = 0.25f * s;
    }
    t.rotation = q.Normalized();
    return t;
}

void ViewMatrix(const math::Vec3& eye, const math::Vec3& target, const math::Vec3& up, float* m) {
    const math::Vec3 f = (target - eye).Normalized();
    math::Vec3 s = f.Cross(up);
    if (s.LengthSq() < 1e-12f) s = math::Vec3{1.0f, 0.0f, 0.0f};
    s = s.Normalized();
    const math::Vec3 u = s.Cross(f);
    m[0] = s.x; m[4] = s.y; m[8]  = s.z;  m[12] = -s.Dot(eye);
    m[1] = u.x; m[5] = u.y; m[9]  = u.z;  m[13] = -u.Dot(eye);
    m[2] = -f.x; m[6] = -f.y; m[10] = -f.z; m[14] = f.Dot(eye);
    m[3] = 0.0f; m[7] = 0.0f; m[11] = 0.0f; m[15] = 1.0f;
}

void ProjMatrixGL(float fov_y_rad, float aspect, float znear, float zfar, float* m) {
    const float t = std::tan(fov_y_rad * 0.5f);
    for (int i = 0; i < 16; ++i) m[i] = 0.0f;
    m[0] = 1.0f / (aspect * t);
    m[5] = 1.0f / t;
    m[10] = (zfar + znear) / (znear - zfar);
    m[11] = -1.0f;
    m[14] = (2.0f * zfar * znear) / (znear - zfar);
}

}  // namespace

// ---- stage model --------------------------------------------------------------------------

std::string NodeKind(const nuka::scene::Registry& reg, nuka::scene::EntityId e) {
    const char* badge = ClassifyNode(reg, e).badge;
    return *badge != '\0' ? badge : "group";
}

// The scene graph as tree items: node id + 1, so item 0 stays the hidden root. Rebuilt when
// the host bumps the scene version.
class SceneItems final : public nui::ItemModel {
public:
    explicit SceneItems(const ImGuiLayer::Frame& frame) : frame_(frame) {}

    uint64_t Version() override {
        const uint64_t version = frame_.scene == nullptr ? 0u : frame_.ui->scene_version * 2u + 1u;
        if (version != built_) Rebuild(version);
        return version;
    }
    void Children(uint64_t item, std::vector<uint64_t>& out) const override {
        out.clear();
        if (item == 0u) {
            if (!entries_.empty()) out.push_back(1u);
            return;
        }
        out = entries_.at(item).children;
    }
    std::string_view Label(uint64_t item) const override {
        const Entry& e = entries_.at(item);
        return e.named ? std::string_view(e.label) : std::string_view(Tr(e.label));
    }
    const char* Icon(uint64_t item) const override { return entries_.at(item).kind.glyph; }
    ImVec4 IconColor(uint64_t item) const override { return entries_.at(item).kind.color; }
    std::string_view Badge(uint64_t item) const override { return entries_.at(item).kind.badge; }
    const std::string& Path(uint64_t item) const { return entries_.at(item).path; }
    nuka::scene::EntityId Entity(uint64_t item) const { return entries_.at(item).entity; }

private:
    struct Entry {
        std::string label;
        bool named = false;
        std::string path;
        nuka::scene::EntityId entity;
        StageKind kind;
        std::vector<uint64_t> children;
    };

    void Rebuild(uint64_t version) {
        built_ = version;
        entries_.clear();
        if (frame_.scene == nullptr) return;
        const nuka::scene::Registry& reg = frame_.scene->Ecs();
        std::vector<std::pair<std::shared_ptr<nuka::scene::SceneNode>, std::string>> stack;
        stack.emplace_back(frame_.scene->Tree().Root(), std::string());
        while (!stack.empty()) {
            auto [node, path] = std::move(stack.back());
            stack.pop_back();
            Entry& entry = entries_[node->id + 1u];
            entry.label = !node->name.empty() ? node->name : path.empty() ? "Scene" : "(unnamed)";
            entry.named = !node->name.empty();
            entry.entity = node->entity;
            entry.kind = ClassifyNode(reg, node->entity);
            for (auto child = node->first_child; child; child = child->next_sibling) {
                entry.children.push_back(child->id + 1u);
                stack.emplace_back(child, path.empty() ? child->name : path + "/" + child->name);
            }
            entry.path = std::move(path);
        }
    }

    const ImGuiLayer::Frame& frame_;
    std::unordered_map<uint64_t, Entry> entries_;
    uint64_t built_ = ~0ull;
};

// ---- setup --------------------------------------------------------------------------------

ImGuiLayer::ImGuiLayer() : stage_items_(std::make_shared<SceneItems>(frame_)) {
    nui::LoadTranslationDir(nui::BuiltinTranslationDir(), "nuka");
}

ImGuiLayer::~ImGuiLayer() = default;

void ImGuiLayer::EnableDocking() {
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // Layouts are built in code; an imgui.ini in the working directory would fight them.
    io.IniFilename = nullptr;
}

const std::vector<ImGuiLayer::BuiltinExtension>& ImGuiLayer::Builtins() {
    static const std::vector<BuiltinExtension> builtins = {
        {"nuka.stage", "Stage", "The scene tree: select, rename, reparent and spawn.", {"stage"}},
        {"nuka.properties", "Properties", "Transform, material and physics fields of the selection.", {"property"}},
        {"nuka.file", "File", "Open, save and load scenes.", {"file"}},
        {"nuka.assets", "Assets", "Importable files to drop into the scene or open.", {"assets"}},
        {"nuka.stats", "Stats", "Frame and physics timings.", {"stats"}},
        {"nuka.debug", "Debug Draw", "Collider and contact overlays.", {"debug"}},
        {"nuka.solver", "Solver", "Solver residuals, sweep budget alerts and the residual heatmap.", {"solver"}},
        {"nuka.timeline", "Timeline", "Seek, replay and fork the run, and compare branches with a ghost.", {"timeline"}},
        {"nuka.camera", "Camera", "Camera pose, framing and controls.", {"camera"}},
        {"nuka.joints", "Joints", "Joint states, drive targets and gains, the fixed base and keyboard teleop.", {"joints"}},
        {"nuka.poses", "Poses", "Named joint poses: blend to one, start from it, import and export.", {"poses"}},
        {"nuka.script", "Script", "The Python script editor and the console.", {"script", "console"}},
        {"nuka.ui_inspector", "UI Inspector", "The nui component tree, for extension authors.", {"nui"}},
    };
    return builtins;
}

void ImGuiLayer::AddBuiltinPanel(const std::string& panel_id) {
    using Builder = void (ImGuiLayer::*)(nui::VStack&);
    const Builder builders[] = {&ImGuiLayer::BuildStage,  &ImGuiLayer::BuildProperties, &ImGuiLayer::BuildFile,
                                &ImGuiLayer::BuildAssets, &ImGuiLayer::BuildStats,      &ImGuiLayer::BuildDebug,
                                &ImGuiLayer::BuildSolver, &ImGuiLayer::BuildTimeline,   &ImGuiLayer::BuildCamera,
                                &ImGuiLayer::BuildJoints, &ImGuiLayer::BuildPoses,      &ImGuiLayer::BuildScript,
                                &ImGuiLayer::BuildConsole, &ImGuiLayer::BuildInspector};
    const size_t i = static_cast<size_t>(
        std::find_if(std::begin(kPanels), std::end(kPanels), [&](const PanelSpec& p) { return panel_id == p.id; }) -
        std::begin(kPanels));
    std::lock_guard<std::recursive_mutex> lock(host_.Mutex());
    nui::Panel& panel = host_.AddPanel(kPanels[i].id, kPanels[i].title, kPanels[i].glyph);
    const Builder build = builders[i];
    panel.build = [this, build](nui::VStack& root) { (this->*build)(root); };
    panel.dock = LayoutColumn(*FindLayout(layout_), panel_id);
    panel.open = !panel.dock.empty();
    if (panel_id == "joints")
        panel.key = [this] {
            return static_cast<uint64_t>(frame_.ui->drive_targets.size()) * 7919u + frame_.ui->robot.joints.size();
        };
    if (panel_id == "property") panel.key = [this] { return frame_.ui->properties_version * 1000003u + staged_version_; };
}

void ImGuiLayer::RemoveBuiltinPanel(const std::string& panel_id) {
    if (panel_id == "stage") stage_tree_ = nullptr;
    if (panel_id == "stats") frame_plot_ = step_plot_ = nullptr;
    host_.RemovePanel(panel_id);
}

std::vector<Action> ImGuiLayer::BuiltinActions(const BuiltinExtension& ext) {
    std::vector<Action> actions;
    for (const std::string& pid : ext.panels) {
        const PanelSpec& p = *std::find_if(std::begin(kPanels), std::end(kPanels),
                                           [&](const PanelSpec& spec) { return pid == spec.id; });
        actions.push_back({"window.toggle_" + pid, p.title, "window", "", ShortcutScope::Global, {},
                           [this, pid] {
                               std::lock_guard<std::recursive_mutex> lock(host_.Mutex());
                               return host_.FindPanel(pid)->open;
                           },
                           [this, pid](const json::Value&) {
                               std::lock_guard<std::recursive_mutex> lock(host_.Mutex());
                               nui::Panel* panel = host_.FindPanel(pid);
                               panel->open = !panel->open;
                               panel->focus_request = panel->open;
                               return ActionResult{};
                           },
                           p.glyph});
    }
    if (ext.id == "nuka.ui_inspector") {
        Action inspector = actions.front();
        inspector.id = "tools.nui_inspector";
        inspector.category = "tools";
        actions.push_back(std::move(inspector));
    }
    return actions;
}

void ImGuiLayer::CameraMatrices(const CameraController& camera, float* view, float* proj) const {
    ViewMatrix(camera.ResolvedEye(), camera.ResolvedTarget(), math::Vec3{0.0f, 0.0f, 1.0f}, view);
    ProjMatrixGL(camera.fov_degrees * 3.14159265358979323846f / 180.0f, viewport_.Aspect(), 0.05f, 500.0f, proj);
}

void ImGuiLayer::Queue(const std::string& id, json::Value args) {
    frame_.ui->actions.push_back(ActionCall{id, std::move(args), "ui"});
}

void ImGuiLayer::SetUiScale(float scale) { ui_scale_ = std::clamp(scale, 0.5f, 3.0f); }

void ImGuiLayer::PrepareFrame(float display_scale) {
    const float scale = display_scale * ui_scale_;
    if (scale == applied_scale_) return;
    render::imgui::ApplyNukaTheme(scale);
    applied_scale_ = scale;
}

bool ImGuiLayer::SetLayout(std::string_view name) {
    if (FindLayout(name) == nullptr) return false;
    layout_ = std::string(name);
    layout_pending_ = true;
    return true;
}

void ImGuiLayer::RegisterActions(ActionRegistry& registry) {
    registry_ = &registry;
    host_.actions.run = [this](const std::string& id, const json::Value& args) { Queue(id, args); };
    host_.actions.enabled = [this](const std::string& id) {
        const Action* a = registry_->Find(id);
        return a != nullptr && (!a->enabled || a->enabled());
    };
    host_.actions.checked = [this](const std::string& id) {
        const Action* a = registry_->Find(id);
        return a == nullptr || !a->checked ? -1 : a->checked() ? 1 : 0;
    };
    host_.actions.label = [this](const std::string& id) {
        const Action* a = registry_->Find(id);
        return a != nullptr ? a->label : id;
    };
    host_.actions.shortcut = [this](const std::string& id) {
        const Action* a = registry_->Find(id);
        return a != nullptr ? ShortcutText(a->shortcut) : std::string();
    };

    const std::pair<const char*, const char*> icons[] = {
        {"file.new", icon::kNew},           {"file.open", icon::kOpen},               {"file.save", icon::kSave},
        {"file.open_window", icon::kNewWindow}, {"file.revert", icon::kReset},        {"file.close", icon::kClose},
        {"app.exit", icon::kExit},
        {"file.save_as", icon::kSaveAs},    {"edit.undo", icon::kUndo},               {"edit.redo", icon::kRedo},
        {"edit.delete", icon::kTrash},      {"script.run", icon::kRun},               {"script.clear_console", icon::kClear},
        {"script.attach", icon::kScriptNode}, {"script.load", icon::kFile},           {"sim.toggle_play", icon::kPlay},
        {"sim.step", icon::kStep},          {"sim.reset", icon::kReset},              {"view.frame_all", icon::kFrame},
        {"view.frame_selected", icon::kTarget}, {"view.reset_camera", icon::kHome},   {"view.colliders", icon::kCollider},
        {"view.contacts", icon::kAtom},     {"view.hide_overlays", icon::kVisual},    {"view.screenshot", icon::kCamera},
        {"file.import", icon::kImport},     {"scene.collision_dialog", icon::kHull},  {"scene.set_visible", icon::kEyeSlash},
        {"scene.set_locked", icon::kLock},  {"assets.add_folder", icon::kFolderPlus}, {"assets.rescan", icon::kReset}};
    for (const auto& [id, glyph] : icons) registry.Decorate(id, glyph, false);
    for (const auto& [id, glyph] : {std::pair{"view.layout_single", icon::kViewSingle}, std::pair{"view.layout_split", icon::kViewSplit},
                                    std::pair{"view.layout_quad", icon::kViewQuad}, std::pair{"view.maximize", icon::kMaximize}})
        registry.Decorate(id, glyph, false);
    for (const auto& [id, glyph] : {std::pair{"solver.telemetry", icon::kPlot}, std::pair{"solver.heatmap", icon::kPalette},
                                    std::pair{"solver.locate", icon::kTarget}, std::pair{"timeline.fork", icon::kFork},
                                    std::pair{"timeline.ghost", icon::kGhost}, std::pair{"timeline.clear", icon::kTrash}})
        registry.Decorate(id, glyph, false);
    registry.Decorate("tool.perturb", icon::kGrab, false);
    registry.Decorate("tool.pin", icon::kPin, false);
    for (const auto& [id, glyph] : {std::pair{"robot.fix_base", icon::kStatic}, std::pair{"robot.drive", icon::kDrive},
                                    std::pair{"robot.watch", icon::kPlot}, std::pair{"robot.pose_capture", icon::kAdd},
                                    std::pair{"robot.poses_save", icon::kSave}, std::pair{"robot.poses_load", icon::kOpen},
                                    std::pair{"robot.gain_tuner", icon::kWaves}, std::pair{"robot.gain_test", icon::kRun},
                                    std::pair{"robot.gain_clear", icon::kTrash}, std::pair{"robot.assembler", icon::kPlug}})
        registry.Decorate(id, glyph, false);
    const std::pair<const char*, const char*> script_icons[] = {
        {"script.new", icon::kNew},   {"script.open", icon::kOpen},     {"script.save", icon::kSave},
        {"script.stop", icon::kStop}, {"script.toggle_rerun", icon::kRepeat}, {"script.eval", icon::kTerminal}};
    for (const auto& [id, glyph] : script_icons) registry.Decorate(id, glyph, false);
    for (const char* id : {"script.close", "script.complete", "script.tabs", "script.goto", "script.console",
                           "script.result", "script.reload", "script.set_budget", "scene.nodes"})
        registry.Decorate(id, "", true);
    for (const char* id : {"edit.apply", "edit.rename", "edit.reparent", "scene.spawn", "scene.select", "sim.play",
                           "sim.pause", "sim.speed", "view.stats", "file.unload", "file.open_recent", "file.forget_recent",
                           "file.recover", "file.autosave", "app.answer_prompt", "app.check", "solver.status",
                           "solver.records", "solver.set_iterations", "window.restore", "window.state", "timeline.status",
                           "timeline.seek", "timeline.branch", "timeline.rename", "timeline.configure", "timeline.hash",
                           "sim.perturb", "sim.grab_particle", "sim.release", "sim.loads", "view.layout", "view.view",
                           "view.views", "robot.joints", "robot.set_targets", "robot.samples", "robot.poses",
                           "robot.pose_apply", "robot.pose_initial", "robot.pose_rename", "robot.pose_remove",
                           "robot.gain_status", "robot.gain_runs", "robot.links", "robot.asset_links", "robot.attach",
                           "robot.attach_asset"})
        registry.Decorate(id, "", true);
    registry.Rebind("file.open", "Ctrl+O");
    registry.Rebind("file.save", "Ctrl+S");
    registry.Rebind("file.save_as", "Ctrl+Shift+S");
    registry.Rebind("app.exit", "Ctrl+Q");
    registry.Rebind("edit.delete", "Delete");
    registry.Rebind("sim.toggle_play", "Space");
    registry.Rebind("script.run", "F5");
    registry.Rebind("script.stop", "Shift+F5");

    const Action* spawn = registry.Find("scene.spawn");
    const auto spawn_enabled = spawn->enabled;
    const auto spawn_run = spawn->run;
    for (const SpawnChoice& c : kSpawns) {
        const std::string kind = c.kind;
        registry.Add({std::string("scene.add_") + c.kind, c.label, "scene", "", ShortcutScope::Global, spawn_enabled, {},
                      [spawn_run, kind](const json::Value&) { return spawn_run(Args({{"kind", json::Value::Str(kind)}})); },
                      c.glyph});
    }

    for (const LayoutPreset& preset : kLayouts) {
        const std::string name = preset.name;
        registry.Add({"window.layout_" + name, std::string(preset.label) + " Layout", "window", "", ShortcutScope::Global, {},
                      [this, name] { return layout_ == name; },
                      [this, name](const json::Value&) {
                          SetLayout(name);
                          return ActionResult{};
                      },
                      icon::kLayout});
    }
    registry.Add({"window.reset_layout", "Reset Layout", "window", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      layout_pending_ = true;
                      return ActionResult{};
                  },
                  icon::kLayout});
    registry.Add({"window.focus", "Focus Panel", "window", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const std::string pid = args.At("panel").AsString();
                      std::lock_guard<std::recursive_mutex> lock(host_.Mutex());
                      nui::Panel* panel = host_.FindPanel(pid);
                      if (panel == nullptr) return ActionResult::Fail("no panel '" + pid + "'");
                      panel->open = true;
                      panel->focus_request = true;
                      return ActionResult{};
                  },
                  "", true});

    registry.Add({"app.command_palette", "Command Palette", "view", "Ctrl+P, Ctrl+Shift+P", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      palette_open_ = !palette_open_;
                      palette_focus_ = palette_open_;
                      palette_query_[0] = '\0';
                      palette_index_ = 0;
                      return ActionResult{};
                  },
                  icon::kCommand});
    const auto scaled = [this](float scale) {
        SetUiScale(scale);
        return ActionResult{true, json::Value::Float(ui_scale_), std::string()};
    };
    registry.Add({"view.ui_scale", "UI Scale", "view", "", ShortcutScope::Global, {}, {},
                  [scaled](const json::Value& args) { return scaled(static_cast<float>(args.At("value").AsDouble())); }, "",
                  true});
    registry.Add({"view.ui_scale_up", "Larger UI", "view", "Ctrl+Equal", ShortcutScope::Global, {}, {},
                  [this, scaled](const json::Value&) { return scaled(ui_scale_ + 0.1f); }, ""});
    registry.Add({"view.ui_scale_down", "Smaller UI", "view", "Ctrl+Minus", ShortcutScope::Global, {}, {},
                  [this, scaled](const json::Value&) { return scaled(ui_scale_ - 0.1f); }, ""});
    registry.Add({"view.ui_scale_reset", "Reset UI Scale", "view", "Ctrl+0", ShortcutScope::Global, {}, {},
                  [scaled](const json::Value&) { return scaled(1.0f); }, ""});
    registry.Add({"scene.expand_all", "Expand Stage", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      stage_fold_ = 1;
                      return ActionResult{};
                  },
                  icon::kChevronDown});
    registry.Add({"scene.collapse_all", "Collapse Stage", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      stage_fold_ = -1;
                      return ActionResult{};
                  },
                  icon::kChevronRight});
    registry.Add({"help.about", "About Nuka Editor", "help", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      about_open_ = true;
                      return ActionResult{};
                  },
                  icon::kInfo});
    registry.Add({"ui.toast", "Toast", "help", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const json::Value* level = args.Find("level");
                      const std::string name = level != nullptr ? level->AsString() : std::string("info");
                      const nui::Level tone = name == "ok"      ? nui::Level::Ok
                                              : name == "warn"  ? nui::Level::Warn
                                              : name == "error" ? nui::Level::Err
                                                                : nui::Level::Info;
                      host_.Toast(tone, args.At("text").AsString());
                      return ActionResult{};
                  },
                  "", true});
    // The switch steps through every locale with a table; English is always first.
    registry.Add({"view.toggle_language", "Switch Language", "view", "Ctrl+Shift+L", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      const std::vector<nui::LocaleInfo> locales = nui::Locales();
                      const std::string current = nui::Locale();
                      const auto it = std::find_if(locales.begin(), locales.end(), [&](const auto& l) { return l.id == current; });
                      const nui::LocaleInfo& next = it == locales.end() || it + 1 == locales.end() ? locales.front() : *(it + 1);
                      nui::SetLocale(next.id);
                      host_.Toast(nui::Level::Info, next.name);
                      return ActionResult{true, json::Value::Str(next.id), std::string()};
                  },
                  icon::kTranslate});
    registry.Add({"view.language", "Language", "view", "", ShortcutScope::Global, {}, {},
                  [](const json::Value& args) {
                      const std::string locale = args.At("locale").AsString();
                      const std::vector<nui::LocaleInfo> locales = nui::Locales();
                      if (std::none_of(locales.begin(), locales.end(), [&](const auto& l) { return l.id == locale; }))
                          return ActionResult::Fail("no string table for locale '" + locale + "'");
                      nui::SetLocale(locale);
                      return ActionResult{true, json::Value::Str(locale), std::string()};
                  },
                  "", true});
    registry.Add({"view.missing_strings", "Missing Strings", "view", "", ShortcutScope::Global, {}, {},
                  [](const json::Value&) {
                      json::Value out = json::Value::Array();
                      for (const std::string& text : nui::MissingStrings()) out.PushBack(json::Value::Str(text));
                      return ActionResult{true, std::move(out), std::string()};
                  },
                  "", true});
}

// ---- panels -------------------------------------------------------------------------------

void ImGuiLayer::BuildStage(nui::VStack& root) {
    root.Add<nui::Custom>([this] {
        const float s = nui::Scale();
        const float buttons = 3.0f * (ImGui::GetFrameHeight() + 2.0f * s + ImGui::GetStyle().ItemSpacing.x);
        ImGui::SetNextItemWidth(std::max(ImGui::GetContentRegionAvail().x - buttons, 60.0f * s));
        const std::string hint = Iconed(icon::kFilter, Tr("Filter"));
        ImGui::InputTextWithHint("##filter", hint.c_str(), stage_filter_, sizeof(stage_filter_));
        ImGui::SameLine();
        ImGui::BeginDisabled(!frame_.ui->has_scene);
        if (draw::IconButton("##add", icon::kAdd, false, Tr("Add primitive"))) ImGui::OpenPopup("##add_menu");
        ImGui::EndDisabled();
        if (ImGui::BeginPopup("##add_menu")) {
            for (const SpawnChoice& c : kSpawns)
                if (ImGui::MenuItemEx(Tr(c.label), c.glyph)) Queue(std::string("scene.add_") + c.kind);
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        if (draw::IconButton("##expand", icon::kChevronDown, false, Tr("Expand all"))) stage_tree_->ExpandAll();
        ImGui::SameLine();
        if (draw::IconButton("##collapse", icon::kChevronRight, false, Tr("Collapse all"))) stage_tree_->CollapseAll();
    });

    auto& tree = root.Add<nui::TreeView>(stage_items_);
    stage_tree_ = &tree;
    tree.filter = nui::Bind<std::string>([this] { return std::string(stage_filter_); });
    tree.is_selected = [this](uint64_t item) {
        const std::vector<nuka::scene::EntityId>& sel = frame_.ui->selection;
        const nuka::scene::EntityId e = stage_items_->Entity(item);
        return e != nuka::scene::kInvalidEntity && std::find(sel.begin(), sel.end(), e) != sel.end();
    };
    // A row clicked here is already in view, so the selection it makes needs no reveal.
    const auto select = [this](uint64_t item, const char* mode) {
        revealed_ = stage_items_->Entity(item);
        Queue("scene.select", Args({{"path", json::Value::Str(stage_items_->Path(item))}, {"mode", json::Value::Str(mode)}}));
    };
    tree.on_select = [select](uint64_t item) { select(item, "replace"); };
    tree.on_toggle = [select](uint64_t item) { select(item, "toggle"); };
    tree.on_range = [this](const std::vector<uint64_t>& items) {
        json::Value paths = json::Value::Array();
        for (const uint64_t item : items) paths.PushBack(json::Value::Str(stage_items_->Path(item)));
        revealed_ = stage_items_->Entity(items.back());
        Queue("scene.select", Args({{"path", json::Value::Str(stage_items_->Path(items.back()))},
                                    {"mode", json::Value::Str("range")},
                                    {"paths", std::move(paths)}}));
    };
    tree.on_rename = [this](uint64_t item, const std::string& name) {
        Queue("edit.rename", Args({{"path", json::Value::Str(stage_items_->Path(item))}, {"name", json::Value::Str(name)}}));
    };
    const auto listed = [this](const std::vector<std::string>& list, uint64_t item) {
        return std::find(list.begin(), list.end(), stage_items_->Path(item)) != list.end();
    };
    tree.is_dimmed = [this, listed](uint64_t item) { return listed(frame_.ui->hidden, item); };
    tree.row_tools = {
        {icon::kEyeSlash, icon::kVisual, "Show/Hide", [this, listed](uint64_t item) { return listed(frame_.ui->hidden, item); },
         [this, listed](uint64_t item) {
             Queue("scene.set_visible", Args({{"path", json::Value::Str(stage_items_->Path(item))},
                                              {"value", json::Value::Bool(listed(frame_.ui->hidden, item))}}));
         }},
        {icon::kLock, icon::kLockOpen, "Lock/Unlock", [this, listed](uint64_t item) { return listed(frame_.ui->locked, item); },
         [this, listed](uint64_t item) {
             Queue("scene.set_locked", Args({{"path", json::Value::Str(stage_items_->Path(item))},
                                             {"value", json::Value::Bool(!listed(frame_.ui->locked, item))}}));
         }}};
    tree.on_activate = [this](uint64_t) { Queue("view.frame_selected"); };
    tree.on_drop = [this](uint64_t dragged, uint64_t onto) {
        Queue("edit.reparent", Args({{"path", json::Value::Str(stage_items_->Path(dragged))},
                                     {"parent", json::Value::Str(stage_items_->Path(onto))}}));
    };
    tree.context_menu = [this](uint64_t item) {
        ViewerUiState& ui = *frame_.ui;
        const std::string path = stage_items_->Path(item);
        if (ImGui::IsWindowAppearing()) {
            const std::string_view name = stage_items_->Label(item);
            std::snprintf(ui.rename_buf, sizeof(ui.rename_buf), "%.*s", static_cast<int>(name.size()), name.data());
        }
        nui::PushTextStyle(TextStyle::Label);
        ImGui::TextColored(theme::kFgDim, "%s", Upper(path.empty() ? std::string(Tr("scene root")) : path).c_str());
        nui::PopTextStyle();
        ImGui::Separator();
        ImGui::SetNextItemWidth(170.0f * nui::Scale());
        const bool enter = ImGui::InputTextWithHint("##rename", Tr("New name"), ui.rename_buf, sizeof(ui.rename_buf),
                                                    ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        if (ImGui::SmallButton(Tr("Rename")) || enter) {
            Queue("edit.rename", Args({{"path", json::Value::Str(path)}, {"name", json::Value::Str(ui.rename_buf)}}));
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::BeginMenuEx(Tr("Add primitive"), icon::kAdd)) {
            for (const SpawnChoice& c : kSpawns)
                if (ImGui::MenuItemEx(Tr(c.label), c.glyph)) Queue("scene.spawn", Args({{"kind", json::Value::Str(c.kind)}}));
            ImGui::EndMenu();
        }
        if (ImGui::MenuItemEx(Tr("Frame"), icon::kTarget, "F")) Queue("view.frame_selected");
        if (ImGui::MenuItemEx(Tr("Generate Collision..."), icon::kHull)) {
            const std::vector<nuka::scene::EntityId>& sel = ui.selection;
            if (std::find(sel.begin(), sel.end(), stage_items_->Entity(item)) == sel.end())
                Queue("scene.select", Args({{"path", json::Value::Str(path)}}));
            Queue("scene.collision_dialog");
        }
        if (ImGui::MenuItemEx(Tr("Import Asset..."), icon::kImport)) Queue("file.import");
        if (ImGui::MenuItemEx(Tr("Delete subtree"), icon::kTrash, "Delete"))
            Queue("edit.delete", Args({{"path", json::Value::Str(path)}}));
    };
}

void ImGuiLayer::BuildProperties(nui::VStack& root) {
    const auto selected = [this] { return frame_.ui->selected_entity != nuka::scene::kInvalidEntity; };
    const auto editable = [this, selected] { return selected() && frame_.ui->inspector.valid; };
    InspectorState& ins = frame_.ui->inspector;

    auto& head = root.Add<nui::HStack>();
    head.visible_if = selected;
    auto& name = head.Add<nui::Label>(std::string(), Tone::Normal, TextStyle::Strong);
    name.text_fn = [this] {
        const auto node = SelectedNode(frame_);
        return !node ? std::string() : node->name.empty() ? std::string(Tr("(unnamed)")) : node->name;
    };
    auto& mobility = head.Add<nui::Badge>();
    mobility.text_fn = [this] {
        const InspectorState& ins = frame_.ui->inspector;
        return std::string(Tr(ins.deformable ? "deformable" : ins.movable ? "movable" : "fixed"));
    };
    mobility.tone_fn = [this] {
        return frame_.ui->inspector.movable || frame_.ui->inspector.deformable ? Tone::Accent : Tone::Dim;
    };
    mobility.visible_if = editable;

    auto& transform = Fold(root.Add<nui::Collapsible>("Transform", true, icon::kMove), "transform");
    transform.visible_if = editable;
    auto& tgrid = transform.Add<nui::PropertyGrid>();
    auto& position = tgrid.Add<nui::Vec3Field>("Position", Floats3(ins.pos), 0.01f);
    position.unit = "m";
    auto& rotation = tgrid.Add<nui::Vec3Field>("Rotation", Floats3(ins.rot_deg), 0.5f);
    rotation.format = "%.1f";
    rotation.unit = "\xC2\xB0";
    for (nui::Vec3Field* f : {&position, &rotation}) {
        f->on_change = [this](const nui::Vec3&) { frame_.ui->inspector.transform_changed = true; };
        f->on_commit = [this](const nui::Vec3&, const nui::Vec3&) { frame_.ui->inspector.transform_commit = true; };
    }

    auto& material = Fold(root.Add<nui::Collapsible>("Material", true, icon::kPalette), "material");
    material.visible_if = [this, editable] { return editable() && frame_.ui->inspector.has_material; };
    auto& mgrid = material.Add<nui::PropertyGrid>();
    auto& base = mgrid.Add<nui::ColorField>("Base color", Floats4(ins.base_color), true, false);
    auto& rough = mgrid.Add<nui::FloatField>("Roughness", nui::Ref(&ins.roughness), 0.01f, 0.0f, 1.0f);
    auto& metal = mgrid.Add<nui::FloatField>("Metallic", nui::Ref(&ins.metallic), 0.01f, 0.0f, 1.0f);
    auto& emissive = mgrid.Add<nui::ColorField>("Emissive", Rgb(ins.emissive), false, true);
    material.Add<nui::Separator>("Offline beauty");
    auto& rgrid = material.Add<nui::PropertyGrid>();
    auto& transmission = rgrid.Add<nui::FloatField>("Transmission", nui::Ref(&ins.transmission), 0.01f, 0.0f, 1.0f);
    auto& ior = rgrid.Add<nui::FloatField>("IOR", nui::Ref(&ins.ior), 0.01f, 1.0f, 2.5f);
    auto& sheen = rgrid.Add<nui::FloatField>("Sheen", nui::Ref(&ins.sheen), 0.01f, 0.0f, 1.0f);
    for (nui::FloatField* f : {&rough, &metal, &transmission, &ior, &sheen}) {
        f->slider = true;
        f->on_change = [this](const float&) { frame_.ui->inspector.material_changed = true; };
        f->on_commit = [this](const float&, const float&) { frame_.ui->inspector.material_commit = true; };
    }
    for (nui::ColorField* f : {&base, &emissive}) {
        f->on_change = [this](const nui::Color&) { frame_.ui->inspector.material_changed = true; };
        f->on_commit = [this](const nui::Color&, const nui::Color&) { frame_.ui->inspector.material_commit = true; };
    }
    BuildFieldGroups(root);
}

void ImGuiLayer::BuildFile(nui::VStack& root) {
    ViewerUiState& ui = *frame_.ui;
    auto& scene_box = root.Add<nui::Collapsible>("Scene", true, icon::kScene);
    auto& current = scene_box.Add<nui::Label>();
    current.wrap = true;
    current.text_fn = [this] { return frame_.ui->has_scene ? frame_.ui->loaded_path : std::string(Tr("No scene loaded")); };
    current.tone_fn = [this] { return frame_.ui->has_scene ? Tone::Accent : Tone::Dim; };
    auto& error = scene_box.Add<nui::Label>(std::string(), Tone::Err);
    error.wrap = true;
    error.text_fn = [this] { return frame_.ui->load_error; };
    error.visible_if = [this] { return !frame_.ui->load_error.empty(); };
    auto& buttons = scene_box.Add<nui::HStack>();
    auto& open = buttons.Add<nui::Button>("Open...", icon::kOpen);
    open.primary = true;
    open.Action("file.open");
    buttons.Add<nui::Button>("Unload", icon::kUnload).Action("file.unload");

    auto& path_box = root.Add<nui::Collapsible>("Load path", false, icon::kFile);
    auto& load = path_box.Add<nui::PropertyGrid>().Add<nui::TextBufferField>("Path", ui.load_path, sizeof(ui.load_path),
                                                                            "scene.nks");
    const auto load_now = [this] { Queue("file.open", Args({{"path", json::Value::Str(frame_.ui->load_path)}})); };
    load.on_enter = load_now;
    path_box.Add<nui::Button>("Load", icon::kOpen).on_click = load_now;

    auto& save_box = root.Add<nui::Collapsible>("Save", true, icon::kSave);
    save_box.visible_if = [this] { return frame_.ui->has_scene; };
    auto& target = save_box.Add<nui::PropertyGrid>().Add<nui::TextBufferField>("Path", ui.save_path, sizeof(ui.save_path),
                                                                              "scene.nks");
    const auto save_now = [this] { Queue("file.save", Args({{"path", json::Value::Str(frame_.ui->save_path)}})); };
    target.on_enter = save_now;
    auto& save_buttons = save_box.Add<nui::HStack>();
    auto& save = save_buttons.Add<nui::Button>("Save", icon::kSave);
    save.primary = true;
    save.on_click = save_now;
    save_buttons.Add<nui::Button>("Save As...", icon::kSaveAs).Action("file.save_as");
}

void ImGuiLayer::BuildStats(nui::VStack& root) {
    auto& hero = root.Add<nui::HStack>();
    auto& step = hero.Add<nui::Label>(std::string(), Tone::Accent, TextStyle::Hero);
    step.text_fn = [this] {
        return frame_.stats->cpu_timing_available ? Fmt("%.2f ms", frame_.stats->step_time_ms) : std::string(Tr("N/A"));
    };
    step.tone_fn = [this] { return frame_.stats->step_healthy ? Tone::Accent : Tone::Warn; };
    auto& live = hero.Add<nui::Badge>();
    live.text_fn = [this] { return std::string(Tr(frame_.ui->playing ? "live" : "idle")); };
    live.tone_fn = [this] { return frame_.ui->playing ? Tone::Ok : Tone::Dim; };
    root.Add<nui::Label>("Physics step", Tone::Dim, TextStyle::Label);
    auto& step_plot = root.Add<nui::Plot>();
    step_plot.sparkline = true;
    step_plot.height = 40.0f;
    step_plot.series.push_back(nui::Series{"step", {}, {}});
    step_plot_ = &step_plot;

    const auto stat = [&root](const char* label, std::function<std::string()> fn) -> nui::Stat& {
        return root.Add<nui::Stat>(label, std::move(fn));
    };
    root.Add<nui::Separator>("Timing");
    stat("fps", [this] { return frame_.stats->frame_timing_available ? Fmt("%.0f", frame_.stats->fps) : std::string(Tr("N/A")); });
    stat("frame", [this] {
        return frame_.stats->frame_timing_available ? Fmt("%.2f ms", frame_.stats->frame_time_ms) : std::string(Tr("N/A"));
    });
    auto& frame_plot = root.Add<nui::Plot>();
    frame_plot.sparkline = true;
    frame_plot.height = 32.0f;
    frame_plot.series.push_back(nui::Series{"frame", {}, {}});
    frame_plot_ = &frame_plot;
    stat("sub-steps", [this] { return Fmt("x%u", frame_.stats->sub_steps); });
    stat("ticks", [this] { return Fmt("%llu", static_cast<unsigned long long>(frame_.stats->sim_ticks)); });
    stat("snapshot age", [this] {
        return frame_.stats->cpu_timing_available ? Fmt("%.1f ms", frame_.stats->snapshot_age_ms) : std::string(Tr("N/A"));
    });
    stat("dropped", [this] { return Fmt("%llu", static_cast<unsigned long long>(frame_.stats->snapshot_drops)); });
    stat("capped batches", [this] { return Fmt("%llu", static_cast<unsigned long long>(frame_.stats->capped_batches)); })
        .tone_fn = [this] { return frame_.stats->capped_batches != 0u ? Tone::Warn : Tone::Normal; };
    // Publishing is zero-copy; only a fallback to host copies is shown, with its reason.
    auto& publish = stat("publish", [] { return std::string(Tr("HOST COPY")); });
    publish.tone = Tone::Warn;
    publish.visible_if = [this] { return !frame_.stats->zero_copy; };
    auto& reason = root.Add<nui::Label>(std::string(), Tone::Dim);
    reason.wrap = true;
    reason.text_fn = [this] { return frame_.stats->publish_reason; };
    reason.visible_if = [this] { return !frame_.stats->zero_copy && !frame_.stats->publish_reason.empty(); };

    root.Add<nui::Separator>("Model");
    stat("DOF", [this] { return Fmt("%u", frame_.stats->dof); }).tone = Tone::Accent;
    stat("links", [this] { return Fmt("%u", frame_.stats->links); });
    stat("bodies", [this] { return Fmt("%u", frame_.stats->bodies); });
    stat("contact cap", [this] { return Fmt("%u", frame_.stats->contact_cap); });
    stat("env index", [this] { return Fmt("%u", frame_.ui->env_index); });

    root.Add<nui::Separator>("Render");
    stat("draw calls", [this] {
        return frame_.stats->draw_calls_available ? Fmt("%u", frame_.stats->draw_calls) : std::string(Tr("N/A"));
    });
    stat("instances", [this] { return Fmt("%u", frame_.world->InstanceCount()); });
    stat("lit pixels", [this] {
        return frame_.stats->pixel_count_available ? Fmt("%llu", static_cast<unsigned long long>(frame_.stats->non_bg_pixels))
                                                   : std::string(Tr("N/A"));
    });
    stat("device", [this] { return frame_.stats->device_name; });
}

void ImGuiLayer::BuildDebug(nui::VStack& root) {
    const auto has_scene = [this] { return frame_.ui->has_scene; };
    root.Add<nui::Badge>("read only", Tone::Info, icon::kLock);
    auto& colliders = root.Add<nui::ToggleButton>("Collider proxies").Action("view.colliders");
    colliders.tooltip = "See-through wireframe; a translucent fill where wireframe is unsupported";
    colliders.enabled_if = has_scene;
    root.Add<nui::ToggleButton>("Contact points").Action("view.contacts").enabled_if = has_scene;
    root.Add<nui::Button>("Hide all", icon::kVisual).Action("view.hide_overlays").enabled_if = has_scene;
    auto& hint = root.Add<nui::Label>("Load a scene to inspect its published state.", Tone::Dim);
    hint.wrap = true;
    hint.visible_if = [this] { return !frame_.ui->has_scene; };

    root.Add<nui::Separator>("Legend");
    root.Add<nui::Custom>([] {
        const float s = nui::Scale();
        const std::pair<ImVec4, const char*> legend[] = {{theme::Hex(0x26D940), "Dynamic collider"},
                                                         {theme::Hex(0xD92ED9), "Static collider"},
                                                         {theme::Hex(0xFF730D), "Contact marker"}};
        for (const auto& [color, text] : legend) {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float h = ImGui::GetTextLineHeight();
            draw::Chamfer(ImGui::GetWindowDrawList(), ImVec2(p.x, p.y + 3.0f * s), ImVec2(p.x + 14.0f * s, p.y + h - 3.0f * s),
                          3.0f * s, U32(color), 0u);
            ImGui::Dummy(ImVec2(20.0f * s, h));
            ImGui::SameLine();
            ImGui::TextUnformatted(Tr(text));
        }
    });

    root.Add<nui::Separator>("Last overlay");
    root.Add<nui::Stat>("proxies", [this] {
        return frame_.stats->debug_colliders_available ? Fmt("%u", frame_.stats->debug_colliders) : std::string(Tr("N/A"));
    });
    root.Add<nui::Stat>("markers", [this] {
        return frame_.stats->debug_contacts_available ? Fmt("%u", frame_.stats->debug_contacts) : std::string(Tr("N/A"));
    });
    const auto warning = [&root, has_scene](std::function<std::string()> text, std::function<bool()> when) {
        auto& label = root.Add<nui::Label>(std::string(), Tone::Warn);
        label.wrap = true;
        label.text_fn = std::move(text);
        label.visible_if = [has_scene, when = std::move(when)] { return has_scene() && when(); };
    };
    warning([this] { return Fmt(Tr("%u unsupported collider shapes skipped"), frame_.stats->debug_skipped_shapes); },
            [this] { return frame_.stats->debug_skipped_shapes != 0u; });
    warning(
        [this] {
            return Fmt(Tr("Invalid data skipped: %u colliders, %u contact points"), frame_.stats->debug_invalid_colliders,
                       frame_.stats->debug_invalid_contacts);
        },
        [this] { return frame_.stats->debug_invalid_colliders != 0u || frame_.stats->debug_invalid_contacts != 0u; });
    warning(
        [this] {
            return Fmt(Tr("Overlay limit %u: %llu valid instances omitted"), frame_.stats->debug_capacity,
                       static_cast<unsigned long long>(frame_.stats->debug_omitted_instances));
        },
        [this] { return frame_.stats->debug_omitted_instances != 0u; });
    warning([] { return std::string(Tr("Contacts not read: collider proxies used the overlay budget.")); },
            [this] { return frame_.stats->debug_contacts_budget_skipped; });
    root.Add<nui::Label>("N/A: disabled, unavailable or not read. Markers show positions, not forces.", Tone::Dim).wrap = true;
}

void ImGuiLayer::BuildCamera(nui::VStack& root) {
    const auto vec = [](const math::Vec3& v) { return Fmt("%.2f  %.2f  %.2f", v.x, v.y, v.z); };
    root.Add<nui::Stat>("eye", [this, vec] { return vec(frame_.camera->ResolvedEye()); });
    root.Add<nui::Stat>("target", [this, vec] { return vec(frame_.camera->ResolvedTarget()); });
    root.Add<nui::Stat>("yaw", [this] { return Fmt("%.1f\xC2\xB0", frame_.camera->Yaw() * 57.29578f); }, Tone::Muted);
    root.Add<nui::Stat>("pitch", [this] { return Fmt("%.1f\xC2\xB0", frame_.camera->Pitch() * 57.29578f); }, Tone::Muted);
    root.Add<nui::Stat>("distance", [this] { return Fmt("%.2f", frame_.camera->Distance()); }, Tone::Muted);
    auto& fov = root.Add<nui::PropertyGrid>().Add<nui::FloatField>(
        "Field of view",
        nui::Bind<float>([this] { return frame_.camera->fov_degrees; }, [this](const float& v) { frame_.camera->fov_degrees = v; }),
        0.5f, 20.0f, 90.0f);
    fov.slider = true;
    fov.format = "%.0f\xC2\xB0";
    auto& buttons = root.Add<nui::HStack>();
    buttons.Add<nui::Button>("Frame all", icon::kFrame).Action("view.frame_all");
    buttons.Add<nui::Button>("Selected", icon::kTarget).Action("view.frame_selected");
    buttons.Add<nui::Button>("Reset", icon::kHome).Action("view.reset_camera");
    root.Add<nui::Separator>("Controls");
    const std::pair<const char*, const char*> keys[] = {{"LMB drag", "orbit"},           {"MMB / Shift", "pan"},
                                                        {"Wheel", "zoom"},               {"W A S D Q E", "fly"},
                                                        {"F / Home", "frame selection / all"}, {"Ctrl+LMB", "pick and drag a body"}};
    for (const auto& [key, what] : keys) root.Add<nui::Stat>(key, nullptr, Tone::Muted).value = what;
}

void ImGuiLayer::BuildScript(nui::VStack& root) {
    auto& bar = root.Add<nui::HStack>();
    auto& run = bar.Add<nui::Button>("Run", icon::kRun);
    run.primary = true;
    run.Action("script.run");
    bar.Add<nui::IconButton>(icon::kCursorText).Action("script.run", Args({{"selection", json::Value::Bool(true)}}));
    bar.Add<nui::IconButton>(icon::kStop).Action("script.stop");
    bar.Add<nui::IconButton>(icon::kNew).Action("script.new");
    bar.Add<nui::IconButton>(icon::kOpen).Action("script.open");
    bar.Add<nui::IconButton>(icon::kSave).Action("script.save");
    bar.Add<nui::IconButton>(icon::kScriptNode).Action("script.attach");
    bar.Add<nui::IconButton>(icon::kFile).Action("script.load");
    bar.Add<nui::IconButton>(icon::kRepeat).Action("script.toggle_rerun");
    auto& status = bar.Add<nui::Label>(std::string(), Tone::Dim, TextStyle::Label);
    status.text_fn = [this] {
        const ViewerUiState& ui = *frame_.ui;
        if (ui.script_busy_s > 0.0) return Fmt(Tr("running %.1f s"), ui.script_busy_s);
        return Fmt(Tr("%d script nodes"), ui.script_node_count);
    };
    root.Add<nui::Custom>([this] { DrawScriptTabs(); });
    auto& code = root.Add<nui::CodeEditor>([this] { return frame_.ui->scripts.Active().editor.get(); });
    code.height = -26.0f;
    code.on_run = [this] { Queue("script.run", Args({{"selection", json::Value::Bool(true)}})); };
    root.Add<nui::Custom>([this] { DrawScriptStatus(); });
}

void ImGuiLayer::DrawScriptTabs() {
    ViewerUiState& ui = *frame_.ui;
    ScriptTabs& tabs = ui.scripts;
    ScriptTab& active = tabs.Active();
    // Run, stop and save keys while the code has the keyboard; otherwise the global shortcuts act.
    if (ImGui::GetIO().WantCaptureKeyboard && ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        if (ImGui::IsKeyChordPressed(ImGuiKey_F5)) Queue("script.run");
        if (ImGui::IsKeyChordPressed(ImGuiMod_Shift | ImGuiKey_F5)) Queue("script.stop");
        if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S)) Queue("script.save");
    }
    for (const std::unique_ptr<ScriptTab>& tab : tabs.tabs) {
        if (tab->completion_set || !ui.complete) continue;
        TextEditor::AutoCompleteConfig config;
        config.triggerDelay = std::chrono::milliseconds(120);
        config.noSuggestionsLabel = "\xE2\x80\xA6";
        config.suggestionWidth = 40;
        // Asks the runner off the UI thread; the answer lands in a later frame.
        config.callback = [this, id = tab->id](TextEditor::AutoCompleteState& state) {
            ScriptTab* t = frame_.ui->scripts.Find(id);
            const TextEditor::DocPos pos = t->editor->GetMainCursorPosition();
            t->completion = frame_.ui->complete(t->Text(), static_cast<int>(pos.line) + 1, static_cast<int>(pos.index));
            std::vector<std::string>& shown = state.suggestions;
            shown.erase(std::remove_if(shown.begin(), shown.end(),
                                       [&](const std::string& s) { return s.rfind(state.searchTerm, 0) != 0; }),
                        shown.end());
            state.suggestionsPromise = true;
        };
        tab->editor->SetAutoCompleteConfig(&config);
        tab->completion_set = true;
    }
    std::string answer;
    if (active.completion != 0u && ui.take_completion(active.completion, &answer)) {
        std::vector<std::string> names;
        for (const json::Value& c : json::Value::Parse(answer).Elements()) names.push_back(c.At("name").AsString());
        active.editor->SetAutoCompleteSuggestions(names);
        active.completion = 0u;
    }

    const bool pending = shown_tab_ != tabs.active;
    if (ImGui::BeginTabBar("##scripts", ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (const std::unique_ptr<ScriptTab>& tab : tabs.tabs) {
            ImGuiTabItemFlags flags = tab->Dirty() ? ImGuiTabItemFlags_UnsavedDocument : 0;
            if (pending && tab->id == tabs.active) flags |= ImGuiTabItemFlags_SetSelected;
            const std::string glyph = tab->node.empty() ? icon::kFilePy : icon::kScriptNode;
            const std::string label = glyph + " " + tab->title + "###tab" + std::to_string(tab->id);
            bool open = true;
            if (ImGui::BeginTabItem(label.c_str(), &open, flags)) {
                // A switch the host asked for wins until the tab bar shows it.
                if (!pending && tab->id != tabs.active) tabs.Activate(tab->id);
                if (tab->id == tabs.active) shown_tab_ = tab->id;
                ImGui::EndTabItem();
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) ImGui::SetTooltip("%s", tab->Name().c_str());
            if (open) continue;
            if (tab->Dirty()) close_ask_ = tab->id;
            else Queue("script.close", Args({{"tab", json::Value::Int(static_cast<int64_t>(tab->id))}}));
        }
        if (ImGui::TabItemButton(icon::kAdd, ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) Queue("script.new");
        ImGui::EndTabBar();
    }

    ScriptTab& shown = tabs.Active();
    const json::Value tab_arg = Args({{"tab", json::Value::Int(static_cast<int64_t>(shown.id))}});
    if (shown.disk_changed) {
        const float s = nui::Scale();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::Alpha(theme::kWarn, 0.12f));
        if (ImGui::BeginChild("##disk", ImVec2(0.0f, ImGui::GetFrameHeightWithSpacing() + 4.0f * s), ImGuiChildFlags_None,
                              ImGuiWindowFlags_NoScrollbar)) {
            ImGui::SetCursorPos(ImVec2(6.0f * s, 2.0f * s));
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(theme::kWarn, "%s", Iconed(icon::kWarnCircle, Tr("Changed on disk")).c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton(Tr("Reload"))) Queue("script.reload", tab_arg);
            ImGui::SameLine();
            if (ImGui::SmallButton(Tr("Keep mine"))) {
                json::Value keep = tab_arg;
                keep.Set("keep", json::Value::Bool(true));
                Queue("script.reload", keep);
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    if (close_ask_ != 0u) ImGui::OpenPopup("##close_script");
    if (ImGui::BeginPopupModal("##close_script", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {
        const ScriptTab* asked = tabs.Find(close_ask_);
        const json::Value args = Args({{"tab", json::Value::Int(static_cast<int64_t>(close_ask_))}});
        ImGui::TextUnformatted(Fmt(Tr("%s has unsaved edits."), asked != nullptr ? asked->title.c_str() : "?").c_str());
        if (ImGui::Button(Tr("Save"))) {
            Queue("script.save", args);
            Queue("script.close", args);
            close_ask_ = 0u;
        }
        ImGui::SameLine();
        if (ImGui::Button(Tr("Discard"))) {
            json::Value discard = args;
            discard.Set("discard", json::Value::Bool(true));
            Queue("script.close", discard);
            close_ask_ = 0u;
        }
        ImGui::SameLine();
        if (ImGui::Button(Tr("Cancel")) || ImGui::IsKeyPressed(ImGuiKey_Escape)) close_ask_ = 0u;
        if (close_ask_ == 0u) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

void ImGuiLayer::DrawScriptStatus() {
    const ScriptTab& tab = frame_.ui->scripts.Active();
    const TextEditor::DocPos pos = tab.editor->GetMainCursorPosition();
    PushTextStyle(TextStyle::Label);
    ImGui::TextColored(theme::kFgDim, "%s", Fmt(Tr("Ln %zu, Col %zu"), pos.line + 1u, pos.index + 1u).c_str());
    ImGui::SameLine();
    ImGui::TextColored(theme::kFgDim, "\xC2\xB7  Python  \xC2\xB7  %s", tab.title.c_str());
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tab.Name().c_str());
    if (!tab.markers.empty()) {
        const ScriptMarker& first = tab.markers.front();
        ImGui::SameLine();
        ImGui::TextColored(theme::kErr, "%s", Iconed(icon::kWarnCircle, Fmt("%d: %s", first.line, first.message.c_str()).c_str()).c_str());
        if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (ImGui::IsItemClicked()) frame_.ui->scripts.Goto(frame_.ui->scripts.Active(), first.line);
    }
    nui::PopTextStyle();
}

void ImGuiLayer::BuildConsole(nui::VStack& root) {
    auto& bar = root.Add<nui::HStack>();
    auto& levels = bar.Add<nui::Segmented>(std::vector<std::string>{"All", "Info", "Warn", "Error"}, nui::Ref(&frame_.ui->console_level));
    levels.icons = {"", icon::kInfo, icon::kWarn, icon::kError};
    bar.Add<nui::Custom>([this] {
        const float s = nui::Scale();
        ImGui::SetNextItemWidth(std::max(ImGui::GetContentRegionAvail().x - ImGui::GetFrameHeight() - 8.0f * s, 60.0f * s));
        const std::string hint = Iconed(icon::kFilter, Tr("Filter output"));
        ImGui::InputTextWithHint("##filter", hint.c_str(), console_filter_, sizeof(console_filter_));
        ImGui::SameLine();
        if (draw::IconButton("##clear", icon::kClear, false, Tr("Clear console"))) Queue("script.clear_console");
    });
    auto& log = root.Add<nui::LogView>(&frame_.ui->console_log);
    log.filter = nui::Bind<std::string>([this] { return std::string(console_filter_); });
    log.level = nui::Ref(&frame_.ui->console_level);
    log.height = -30.0f;
    log.on_link = [this](const std::string& file, int line) {
        Queue("script.goto", Args({{"path", json::Value::Str(file)}, {"line", json::Value::Int(line)}}));
    };
    root.Add<nui::Custom>([this] { DrawRepl(); });
}

void ImGuiLayer::DrawRepl() {
    ViewerUiState& ui = *frame_.ui;
    std::string answer;
    if (repl_ticket_ != 0u && ui.take_completion(repl_ticket_, &answer)) {
        repl_ticket_ = 0u;
        repl_options_.clear();
        for (const json::Value& c : json::Value::Parse(answer).Elements()) repl_options_.push_back(c.At("name").AsString());
        // One option, or the longest prefix they share, goes straight into the line.
        if (!repl_options_.empty()) {
            std::string common = repl_options_.front();
            for (const std::string& o : repl_options_) {
                size_t n = 0u;
                while (n < common.size() && n < o.size() && common[n] == o[n]) ++n;
                common.resize(n);
            }
            if (common.size() > repl_word_) repl_insert_ = common;
            if (repl_options_.size() == 1u) repl_options_.clear();
        }
    }
    const float s = nui::Scale();
    PushTextStyle(TextStyle::Mono);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(theme::kAccent, ">>>");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (repl_focus_) {
        ImGui::SetKeyboardFocusHere();
        repl_focus_ = false;
    }
    const ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCompletion |
                                      ImGuiInputTextFlags_CallbackHistory | ImGuiInputTextFlags_CallbackAlways;
    const auto callback = [](ImGuiInputTextCallbackData* data) -> int {
        ImGuiLayer& self = *static_cast<ImGuiLayer*>(data->UserData);
        if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion && self.frame_.ui->complete) {
            size_t start = static_cast<size_t>(data->CursorPos);
            while (start > 0u && (std::isalnum(static_cast<unsigned char>(data->Buf[start - 1u])) || data->Buf[start - 1u] == '_')) --start;
            self.repl_word_ = static_cast<size_t>(data->CursorPos) - start;
            self.repl_ticket_ = self.frame_.ui->complete(std::string(data->Buf, static_cast<size_t>(data->CursorPos)), 1, data->CursorPos);
        } else if (data->EventFlag == ImGuiInputTextFlags_CallbackHistory && !self.repl_history_.empty()) {
            const size_t n = self.repl_history_.size();
            if (data->EventKey == ImGuiKey_UpArrow && self.repl_back_ < n) ++self.repl_back_;
            else if (data->EventKey == ImGuiKey_DownArrow && self.repl_back_ > 0u) --self.repl_back_;
            data->DeleteChars(0, data->BufTextLen);
            if (self.repl_back_ > 0u) data->InsertChars(0, self.repl_history_[n - self.repl_back_].c_str());
        } else if (data->EventFlag == ImGuiInputTextFlags_CallbackAlways && !self.repl_insert_.empty()) {
            const int word = static_cast<int>(self.repl_word_);
            data->DeleteChars(data->CursorPos - word, word);
            data->InsertChars(data->CursorPos, self.repl_insert_.c_str());
            self.repl_insert_.clear();
            self.repl_word_ = 0u;
        }
        return 0;
    };
    if (ImGui::InputTextWithHint("##repl", Tr("Python  (Enter runs, Tab completes, Up/Down history)"), repl_, sizeof(repl_), flags,
                                 callback, this)) {
        const std::string line = repl_;
        if (!line.empty()) {
            if (repl_history_.empty() || repl_history_.back() != line) repl_history_.push_back(line);
            Queue("script.eval", Args({{"code", json::Value::Str(line)}}));
        }
        repl_[0] = '\0';
        repl_back_ = 0u;
        repl_options_.clear();
        repl_focus_ = true;
    }
    const ImVec2 input = ImGui::GetItemRectMin();
    nui::PopTextStyle();
    if (repl_options_.empty()) return;
    // The choices float above the input line, so the log keeps its height.
    const float max_w = std::max(ImGui::GetContentRegionAvail().x, 120.0f * s);
    ImGui::SetNextWindowPos(ImVec2(input.x, input.y - 4.0f * s), ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    ImGui::BeginTooltip();
    PushTextStyle(TextStyle::Label);
    const float gap = 10.0f * s;
    float x = 0.0f;
    for (size_t i = 0; i < repl_options_.size() && i < 24u; ++i) {
        const float w = ImGui::CalcTextSize(repl_options_[i].c_str()).x;
        if (i > 0u && x + gap + w <= max_w) {
            ImGui::SameLine(0.0f, gap);
            x += gap + w;
        } else {
            x = w;
        }
        ImGui::TextColored(theme::kFgMuted, "%s", repl_options_[i].c_str());
    }
    nui::PopTextStyle();
    ImGui::EndTooltip();
}

void ImGuiLayer::BuildInspector(nui::VStack& root) {
    root.Add<nui::Label>("Every panel's nui tree; #N is the node's C API handle.", Tone::Dim).wrap = true;
    root.Add<nui::Custom>([this] {
        for (const auto& panel : host_.Panels()) {
            const std::string head = Fmt("%s  [%s]  %s  rebuilds %llu  draws %llu", Tr(panel->title).c_str(), panel->id.c_str(),
                                         panel->open ? "open" : "closed", static_cast<unsigned long long>(panel->rebuilds),
                                         static_cast<unsigned long long>(panel->draws));
            if (ImGui::TreeNode(panel->id.c_str(), "%s", head.c_str())) {
                DrawNodeTree(panel->Root());
                ImGui::TreePop();
            }
        }
    });
}

// ---- frame --------------------------------------------------------------------------------

void ImGuiLayer::RecordUi(const render::RenderWorld& world, const ViewerStats& stats, CameraController& camera,
                          ViewerUiState& ui_state, const nuka::scene::SceneIR* scene, uint32_t framebuffer_width,
                          uint32_t framebuffer_height) {
    frame_ = Frame{&world, &stats, &camera, &ui_state, scene};
    viewport_ = {};
    viewport_hovered_ = false;
    ui_state.assets.shown = false;
    InspectorState& ins = ui_state.inspector;
    ins.transform_changed = ins.transform_commit = ins.material_changed = ins.material_commit = false;
    if (ui_state.drive_dirty.size() != ui_state.drive_targets.size())
        ui_state.drive_dirty.assign(ui_state.drive_targets.size(), 0u);
    const float sample_x = static_cast<float>(stats.frame_index);
    if (frame_plot_ != nullptr && stats.frame_timing_available) frame_plot_->Push(0u, sample_x, stats.frame_time_ms, 240u);
    if (step_plot_ != nullptr && stats.cpu_timing_available) step_plot_->Push(0u, sample_x, stats.step_time_ms, 240u);

    // A new scene opens the stage on its top level; a selection made elsewhere scrolls into view.
    if (scene != stage_scene_ && stage_tree_ != nullptr) {
        stage_scene_ = scene;
        stage_tree_->CollapseAll();
        if (scene != nullptr && scene->Tree().Root()->first_child)
            stage_tree_->Reveal(scene->Tree().Root()->first_child->id + 1u);
    }
    if (stage_tree_ != nullptr && scene != nullptr && ui_state.selected_entity != revealed_) {
        revealed_ = ui_state.selected_entity;
        if (revealed_ != nuka::scene::kInvalidEntity)
            if (const auto node = scene->Ecs().NodeOf(revealed_)) stage_tree_->Reveal(node->id + 1u);
    }
    if (stage_tree_ != nullptr && stage_fold_ != 0) {
        if (stage_fold_ > 0) stage_tree_->ExpandAll();
        else stage_tree_->CollapseAll();
        stage_fold_ = 0;
    }

    ImGuizmo::SetOrthographic(false);
    ImGuizmo::BeginFrame();
    DrawMenuBar();
    DrawToolbar();
    DrawStatusBar();
    const ImGuiID dockspace = ImGui::GetID("##nuka_dockspace");
    if (layout_pending_) {
        DockLayout(dockspace);
        layout_pending_ = false;
        layout_settle_ = 3;
    }
    // Overflowing tab bars scroll to the selected tab once the docked tabs have their widths.
    if (layout_settle_ > 0 && --layout_settle_ == 0)
        for (const ImGuiID id : {dock_left_, dock_right_, dock_bottom_})
            if (ImGuiDockNode* node = ImGui::DockBuilderGetNode(id); node != nullptr && node->TabBar != nullptr)
                node->TabBar->NextSelectedTabId = node->TabBar->SelectedTabId;
    ImGui::DockSpaceOverViewport(dockspace, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    host_.DrawPanels([this](const nui::Panel& panel) -> ImGuiID {
        const ImGuiID node = panel.dock == "left"     ? dock_left_
                             : panel.dock == "right"  ? dock_right_
                             : panel.dock == "bottom" ? dock_bottom_
                                                      : 0u;
        return node != 0u && ImGui::DockBuilderGetNode(node) != nullptr ? node : 0u;
    });

    const ImGuiDockNode* central = ImGui::DockBuilderGetCentralNode(dockspace);
    const ImGuiIO& io = ImGui::GetIO();
    if (framebuffer_width == ~uint32_t(0))
        framebuffer_width = static_cast<uint32_t>(std::max(io.DisplaySize.x * io.DisplayFramebufferScale.x, 0.0f));
    if (framebuffer_height == ~uint32_t(0))
        framebuffer_height = static_cast<uint32_t>(std::max(io.DisplaySize.y * io.DisplayFramebufferScale.y, 0.0f));
    // The shown panes split the central node a divider apart, in rows; the active one is the viewport.
    pane_rects_ = {};
    const size_t pane_count = ui_state.panes.size();
    const size_t cols = pane_count > 1u ? 2u : 1u, rows = pane_count > 2u ? 2u : 1u;
    const float gap = pane_count > 1u ? 2.0f * nui::Scale() : 0.0f;
    ImVec2 cmin, cmax, pane;
    if (central != nullptr) {
        const ImGuiViewport* main = ImGui::GetMainViewport();
        cmin = central->Pos;
        cmax = ImVec2(central->Pos.x + central->Size.x, central->Pos.y + central->Size.y);
        pane = ImVec2((central->Size.x - gap * static_cast<float>(cols - 1u)) / static_cast<float>(cols),
                      (central->Size.y - gap * static_cast<float>(rows - 1u)) / static_cast<float>(rows));
        for (size_t k = 0; k < pane_count; ++k)
            pane_rects_[ui_state.panes[k].index] = SceneViewportRect::FromLogical(
                cmin.x + static_cast<float>(k % cols) * (pane.x + gap), cmin.y + static_cast<float>(k / cols) * (pane.y + gap),
                pane.x, pane.y, main->Pos.x, main->Pos.y, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y,
                framebuffer_width, framebuffer_height);
        viewport_ = pane_rects_[ui_state.active_view];
    }
    const ImVec2 mouse = ImGui::GetMousePos();
    pointer_free_ = ui_state.window_focused && !ui_state.layout_modifier_down && !io.WantCaptureMouse;
    viewport_hovered_ = pointer_free_ && viewport_.Contains(mouse.x, mouse.y);
    hovered_pane_ = -1;
    for (uint32_t i = 0; i < kViewPanes; ++i)
        if (pointer_free_ && pane_rects_[i].Contains(mouse.x, mouse.y)) hovered_pane_ = static_cast<int>(i);

    if (viewport_.Valid()) {
        const ImVec2 vmin(viewport_.x, viewport_.y), vmax(viewport_.x + viewport_.width, viewport_.y + viewport_.height);
        if (ui_state.has_scene) {
            DrawRtImage(vmin, vmax);
            if (pane_count > 1u) DrawPaneHeaders(cmin, cmax, pane, gap);
            std::vector<std::string> hud = {Fmt(Tr("FOV %.0f\xC2\xB0"), camera.fov_degrees),
                                            Tr(ui_state.playing ? "LIVE" : "PAUSED"),
                                            Fmt(Tr("TICK %llu"), static_cast<unsigned long long>(stats.sim_ticks))};
            const RtView& rt = ui_state.rt;
            if (rt.on)
                hud.push_back(rt.converged ? Fmt(Tr("RT %u SPP"), rt.spp) + " " + icon::kOk
                                           : Fmt(Tr("RT %u/%u SPP"), rt.spp, rt.max_spp));
            for (const ViewPane& p : ui_state.panes)
                if (p.index == ui_state.active_view && !p.camera.empty())
                    hud.insert(hud.begin(), Iconed(icon::kCameraNode, p.camera.substr(p.camera.rfind('/') + 1u).c_str()));
            // Below the pane header when there is one.
            draw::ViewportFrame(vmin, vmax, hud, pane_count > 1u ? ImGui::GetFrameHeight() + 8.0f * nui::Scale() : 0.0f);
            DrawSolverHud(vmin, vmax);
            DrawPerturbHud(vmin, vmax);
            DrawViewTools(vmin, vmax);
            DrawBookmarkBar(vmin, vmax);
        } else if (!ui_state.start.shown) {
            const float s = nui::Scale();
            ImFont* font = GetNukaFont(FontRole::Display);
            const float px = theme::kFontLabel * s * 1.3f;
            const char* text = Tr("NO SCENE   CTRL+O TO OPEN   CTRL+P FOR ACTIONS");
            const ImVec2 size = font->CalcTextSizeA(px, FLT_MAX, 0.0f, text);
            ImGui::GetBackgroundDrawList()->AddText(font, px,
                                                    ImVec2((vmin.x + vmax.x - size.x) * 0.5f, (vmin.y + vmax.y - size.y) * 0.5f),
                                                    U32(theme::kFgDim), text);
        }
        // Under every window, so popups, the palette and modals stay on top; across all panes, not the active one.
        if (!ui_state.start.shown) host_.DrawNotifications(ImGui::GetBackgroundDrawList(), cmin, cmax);
    }
    if (!ui_state.start.shown) DrawAssetDrop();
    DrawStartPage();
    DrawPalette();
    DrawAbout();
    DrawImportWizard();
    DrawCollisionDialog();
    DrawGainTuner();
    DrawAssembler();
    DrawPrompt();
}

void ImGuiLayer::DockLayout(ImGuiID dockspace) {
    const LayoutPreset& preset = *FindLayout(layout_);
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::DockBuilderRemoveNode(dockspace);
    ImGui::DockBuilderAddNode(dockspace, static_cast<ImGuiDockNodeFlags>(ImGuiDockNodeFlags_DockSpace) |
                                             ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::DockBuilderSetNodePos(dockspace, vp->WorkPos);
    ImGui::DockBuilderSetNodeSize(dockspace, vp->WorkSize);
    ImGuiID center = dockspace;
    dock_left_ = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.20f, nullptr, &center);
    dock_right_ = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.28f, nullptr, &center);
    dock_bottom_ = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.30f, nullptr, &center);
    std::lock_guard<std::recursive_mutex> lock(host_.Mutex());
    for (const PanelSpec& p : kPanels)
        if (nui::Panel* panel = host_.FindPanel(p.id)) panel->open = false;
    // Tabs follow the preset's order and a rebuilt node selects its saved tab, here the column's
    // first panel; disabled built-in extensions leave gaps.
    const auto place = [this](const std::vector<const char*>& ids, ImGuiID node) {
        const char* first = nullptr;
        short order = 0;
        for (const char* id : ids) {
            nui::Panel* panel = host_.FindPanel(id);
            if (panel == nullptr) continue;
            panel->open = true;
            if (first == nullptr) first = id;
            const std::string name = std::string("###") + id;
            ImGui::DockBuilderDockWindow(name.c_str(), node);
            if (ImGuiWindow* window = ImGui::FindWindowByName(name.c_str()))
                window->DockOrder = order++;
            else
                ImGui::FindWindowSettingsByID(ImHashStr(name.c_str()))->DockOrder = order++;
        }
        if (first == nullptr) return;
        const ImGuiID window = ImHashStr((std::string("###") + first).c_str());
        ImGui::DockBuilderGetNode(node)->SelectedTabId = ImHashStr("#TAB", 0, window);
    };
    place(preset.left, dock_left_);
    place(preset.right, dock_right_);
    place(preset.bottom, dock_bottom_);
    // Script panels keep their open state and return to the column they asked for.
    for (const auto& panel : host_.Panels()) {
        if (Builtin(panel->id)) continue;
        const ImGuiID node = panel->dock == "left" ? dock_left_ : panel->dock == "right" ? dock_right_
                             : panel->dock == "bottom"                                   ? dock_bottom_
                                                                                         : 0u;
        if (node != 0u) ImGui::DockBuilderDockWindow(("###" + panel->id).c_str(), node);
    }
    ImGui::DockBuilderFinish(dockspace);
}

bool ImGuiLayer::MenuAction(const std::string& id) {
    const Action* a = registry_->Find(id);
    if (a == nullptr || a->hidden) return false;
    const std::string keys = ShortcutText(a->shortcut);
    if (ImGui::MenuItemEx(Tr(a->label).c_str(), a->icon.empty() ? nullptr : a->icon.c_str(), keys.empty() ? nullptr : keys.c_str(),
                          a->checked && a->checked(), !a->enabled || a->enabled()))
        Queue(id);
    return true;
}

void ImGuiLayer::DrawMenuBar() {
    const float s = nui::Scale();
    const float bar_h = std::round(theme::kTitleBarHeight * s);
    // The menu bar is the window's title bar; a taller frame padding gives it the height and
    // centres the menu titles.
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                        ImVec2(ImGui::GetStyle().FramePadding.x, (bar_h - ImGui::GetFontSize()) * 0.5f));
    const bool open = ImGui::BeginMainMenuBar();
    ImGui::PopStyleVar();
    if (!open) return;
    TitleBar& bar = frame_.ui->title_bar;
    const bool focused = frame_.ui->window_focused;
    ImDrawList* list = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetWindowPos();
    const float width = ImGui::GetWindowWidth(), mid = origin.y + bar_h * 0.5f;
    ImFont* label_font = GetNukaFont(FontRole::Display);
    const float label_px = theme::kFontLabel * s;

    // The logo centres on the bar and the wordmark's capitals on the logo.
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float logo = std::round(20.0f * s);
    render::imgui::DrawNukaLogo(list, ImVec2(at.x, std::round(mid - logo * 0.5f)), logo);
    const float word_x = std::round(at.x + logo + 7.0f * s);
    list->AddText(label_font, label_px, ImVec2(word_x, CapTop(label_font, label_px, mid)),
                  U32(focused ? theme::kAccent : theme::kFgDim), "NUKA");
    const float brand_end = word_x + label_font->CalcTextSizeA(label_px, FLT_MAX, 0.0f, "NUKA").x;
    ImGui::Dummy(ImVec2(brand_end - at.x, ImGui::GetFontSize()));
    ImGui::SameLine(0.0f, 14.0f * s);
    const float menus_x = ImGui::GetCursorScreenPos().x;

    for (const MenuSpec& menu : Menus()) {
        const std::string title = Tr(menu.title) + "###" + menu.title;
        if (!ImGui::BeginMenu(title.c_str())) continue;
        bool any = false, separator = false;
        for (const std::string& id : menu.items) {
            if (id == "-") {
                separator = any;
                continue;
            }
            // Open Recent lists the recent scenes.
            if (id == "file.open_recent") {
                if (separator) ImGui::Separator();
                separator = false;
                const std::vector<StartEntry>& recent = frame_.ui->start.recent;
                if (ImGui::BeginMenuEx(Tr("Open Recent"), icon::kRecent, !recent.empty())) {
                    for (const StartEntry& r : recent)
                        if (ImGui::MenuItemEx(r.title.c_str(), nullptr, nullptr, false, r.available))
                            Queue("file.open_recent", Args({{"path", json::Value::Str(r.arg)}}));
                    ImGui::EndMenu();
                }
                any = true;
                continue;
            }
            const Action* a = registry_->Find(id);
            if (a == nullptr || a->hidden) continue;
            if (separator) ImGui::Separator();
            separator = false;
            any = MenuAction(id) || any;
        }
        // Actions of these categories the shell does not list, such as ones scripts register.
        bool extra = any;
        for (const Action& a : registry_->All()) {
            if (a.hidden || std::find(menu.categories.begin(), menu.categories.end(), a.category) == menu.categories.end() ||
                std::find(menu.items.begin(), menu.items.end(), a.id) != menu.items.end())
                continue;
            if (extra) ImGui::Separator();
            extra = false;
            any = MenuAction(a.id) || any;
        }
        if (menu.title == "Simulation") {
            ImGui::Separator();
            if (ImGui::BeginMenuEx(Tr("Speed"), icon::kSpeed)) {
                for (size_t i = 0; i < std::size(kSpeeds); ++i)
                    if (ImGui::MenuItem(kSpeedLabels[i], nullptr, std::fabs(frame_.ui->speed - kSpeeds[i]) < 1e-3f))
                        Queue("sim.speed", Args({{"value", json::Value::Float(kSpeeds[i])}}));
                ImGui::EndMenu();
            }
            any = true;
        }
        if (!any) ImGui::MenuItem(Tr("Nothing here yet"), nullptr, false, false);
        ImGui::EndMenu();
    }
    const float menus_end = ImGui::GetCursorScreenPos().x;

    // Minimize, maximize or restore, and close fill the bar's right end at its full height.
    const float bw = std::round(theme::kWindowButton * s);
    const float buttons_x = origin.x + width - 3.0f * bw;
    const float glyph = std::round(5.0f * s), thick = std::max(1.0f, std::round(s));
    const ImU32 ink = U32(focused ? theme::kFgMuted : theme::kFgDim);
    const auto window_button = [&](int i, const char* id, const char* tip, bool danger, auto&& draw) {
        const ImVec2 a(buttons_x + static_cast<float>(i) * bw, origin.y), b(a.x + bw, origin.y + bar_h);
        ImGui::SetCursorScreenPos(a);
        const bool pressed = ImGui::InvisibleButton(id, ImVec2(bw, bar_h));
        const bool hovered = ImGui::IsItemHovered(), held = ImGui::IsItemActive();
        if (hovered || held)
            list->AddRectFilled(a, b, U32(danger ? theme::Alpha(theme::kErr, held ? 0.7f : 0.9f) : held ? theme::kBg3 : theme::kBg2));
        ImGui::SetItemTooltip("%s", Tr(tip));
        draw(ImVec2(std::round((a.x + b.x) * 0.5f), std::round(mid)), danger && hovered ? U32(theme::kFg) : ink);
        return pressed;
    };
    if (window_button(0, "##minimize", "Minimize", false, [&](ImVec2 c, ImU32 col) {
            list->AddLine(ImVec2(c.x - glyph, c.y), ImVec2(c.x + glyph, c.y), col, thick);
        }))
        Queue("window.minimize");
    const float step = std::round(2.0f * s);
    if (window_button(1, "##maximize", bar.maximized ? "Restore" : "Maximize", false, [&](ImVec2 c, ImU32 col) {
            if (!bar.maximized) {
                list->AddRect(ImVec2(c.x - glyph, c.y - glyph), ImVec2(c.x + glyph, c.y + glyph), col, 0.0f, 0, thick);
                return;
            }
            list->AddRect(ImVec2(c.x - glyph, c.y - glyph + step), ImVec2(c.x + glyph - step, c.y + glyph), col, 0.0f, 0, thick);
            // AddLine and AddRect stroke on pixel centres themselves; the polyline is given them.
            const float l = c.x - glyph + step + 0.5f, r = c.x + glyph + 0.5f, t = c.y - glyph + 0.5f, d = c.y + glyph - step + 0.5f;
            const ImVec2 back[] = {ImVec2(l, t + step), ImVec2(l, t), ImVec2(r, t), ImVec2(r, d), ImVec2(r - step, d)};
            list->AddPolyline(back, 5, col, ImDrawFlags_None, thick);
        }))
        Queue("window.maximize");
    if (window_button(2, "##close", "Close", true, [&](ImVec2 c, ImU32 col) {
            list->AddLine(ImVec2(c.x - glyph, c.y - glyph), ImVec2(c.x + glyph, c.y + glyph), col, thick);
            list->AddLine(ImVec2(c.x + glyph, c.y - glyph), ImVec2(c.x - glyph, c.y + glyph), col, thick);
        }))
        Queue("app.exit");

    // The search chip shrinks to its icon when the menus leave no room for the text.
    const char* keys = "CTRL+P";
    const ImVec2 ksize = label_font->CalcTextSizeA(label_px, FLT_MAX, 0.0f, keys);
    const std::string full = Iconed(icon::kSearch, Tr("Search actions"));
    const float room = buttons_x - menus_end - 18.0f * s;
    const bool compact = ImGui::CalcTextSize(full.c_str()).x + ksize.x + 40.0f * s > room;
    const std::string text = compact ? std::string(icon::kSearch) : full;
    const ImVec2 tsize = ImGui::CalcTextSize(text.c_str());
    const float w = compact ? tsize.x + 20.0f * s : tsize.x + ksize.x + 40.0f * s, h = ImGui::GetFrameHeight() - 6.0f * s;
    const ImVec2 a(std::round(buttons_x - w - 10.0f * s), std::round(mid - h * 0.5f)), b(a.x + w, a.y + h);
    ImGui::SetCursorScreenPos(a);
    if (ImGui::InvisibleButton("##search", ImVec2(w, h))) Queue("app.command_palette");
    const bool hovered = ImGui::IsItemHovered();
    draw::Chamfer(list, a, b, 4.0f * s, U32(hovered ? theme::kBg2 : theme::kBg1), U32(hovered ? theme::kLineHi : theme::kLine));
    list->AddText(ImVec2(a.x + 10.0f * s, (a.y + b.y - tsize.y) * 0.5f), U32(theme::kFgDim), text.c_str());
    if (!compact)
        list->AddText(label_font, label_px, ImVec2(b.x - ksize.x - 10.0f * s, (a.y + b.y - ksize.y) * 0.5f), U32(theme::kFgDim),
                      keys);

    bar.height = bar_h;
    bar.edge = theme::kResizeEdge * s;
    bar.drag = {{origin.x, menus_x}, {menus_end, a.x}, {b.x, buttons_x}};
    ImGui::EndMainMenuBar();
    // A restored window has no system frame, so a hairline marks its edge.
    if (!bar.maximized) {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::GetForegroundDrawList()->AddRect(vp->Pos, ImVec2(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y),
                                                U32(focused ? theme::kLineHi : theme::kLine));
    }
}

void ImGuiLayer::DrawToolbar() {
    ViewerUiState& ui = *frame_.ui;
    const ViewerStats& stats = *frame_.stats;
    const float s = nui::Scale();
    const float height = theme::kToolbarHeight * s;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f * s, std::max((height - ImGui::GetFrameHeight()) * 0.5f, 0.0f)));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2.0f * s, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kBg1);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollWithMouse;
    if (ImGui::BeginViewportSideBar("##toolbar", ImGui::GetMainViewport(), ImGuiDir_Up, height, flags)) {
        ImDrawList* list = ImGui::GetWindowDrawList();
        const ImVec2 wpos = ImGui::GetWindowPos(), wsize = ImGui::GetWindowSize();
        list->AddLine(ImVec2(wpos.x, wpos.y + wsize.y - 1.0f), ImVec2(wpos.x + wsize.x, wpos.y + wsize.y - 1.0f), U32(theme::kLine));
        const auto action = [&](const char* id, const char* glyph, bool checked) {
            const Action* a = registry_->Find(id);
            const std::string keys = ShortcutText(a->shortcut);
            const std::string tip = keys.empty() ? Tr(a->label) : Tr(a->label) + "   " + keys;
            ImGui::BeginDisabled(a->enabled && !a->enabled());
            if (draw::IconButton(id, glyph, checked, tip.c_str())) Queue(id);
            ImGui::EndDisabled();
            ImGui::SameLine();
        };
        const auto toggle = [](const char* id, const char* glyph, bool on, const char* tip) {
            const bool pressed = draw::IconButton(id, glyph, on, tip);
            ImGui::SameLine();
            return pressed;
        };
        const auto divider = [&] {
            ImGui::SameLine(0.0f, 8.0f * s);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float fh = ImGui::GetFrameHeight();
            list->AddLine(ImVec2(p.x, p.y + 5.0f * s), ImVec2(p.x, p.y + fh - 5.0f * s), U32(theme::kLine));
            ImGui::Dummy(ImVec2(1.0f, fh));
            ImGui::SameLine(0.0f, 8.0f * s);
        };
        action("file.open", icon::kOpen, false);
        action("file.save", icon::kSave, false);
        divider();
        action("sim.toggle_play", ui.playing ? icon::kPause : icon::kPlay, ui.playing);
        action("sim.step", icon::kStep, false);
        action("sim.reset", icon::kReset, false);
        divider();
        const float frame_h = ImGui::GetFrameHeight();
        nui::PushTextStyle(TextStyle::Label);
        for (size_t i = 0; i < std::size(kSpeeds); ++i) {
            const bool on = std::fabs(ui.speed - kSpeeds[i]) < 1e-3f;
            ImGui::PushStyleColor(ImGuiCol_Button, on ? theme::Alpha(theme::kAccent, 0.16f) : ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, on ? theme::kAccent : theme::kFgDim);
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::Button(kSpeedLabels[i], ImVec2(0.0f, frame_h)))
                Queue("sim.speed", Args({{"value", json::Value::Float(kSpeeds[i])}}));
            ImGui::PopID();
            ImGui::PopStyleColor(2);
            ImGui::SameLine();
        }
        nui::PopTextStyle();
        divider();
        action("edit.undo", icon::kUndo, false);
        action("edit.redo", icon::kRedo, false);
        divider();
        GizmoState& gz = ui.gizmo;
        if (toggle("##gizmo", icon::kGizmo, gz.enabled, Tr("Gizmo"))) gz.enabled = !gz.enabled;
        ImGui::BeginDisabled(!gz.enabled);
        if (toggle("##move", icon::kMove, gz.op == GizmoState::Op::Translate, Tr("Move"))) gz.op = GizmoState::Op::Translate;
        if (toggle("##rotate", icon::kRotate, gz.op == GizmoState::Op::Rotate, Tr("Rotate"))) gz.op = GizmoState::Op::Rotate;
        if (toggle("##space", gz.local ? icon::kLocalSpace : icon::kWorldSpace, gz.local, Tr(gz.local ? "Local space" : "World space")))
            gz.local = !gz.local;
        if (toggle("##snap", icon::kSnap, gz.snap_on, Tr("Snap (right-click for the step)"))) gz.snap_on = !gz.snap_on;
        if (ImGui::BeginPopupContextItem("##snap_step")) {
            const bool rotate = gz.op == GizmoState::Op::Rotate;
            ImGui::SetNextItemWidth(110.0f * s);
            ImGui::DragFloat(Tr("Snap step"), rotate ? &gz.snap_rotate : &gz.snap_translate, rotate ? 0.5f : 0.01f, 0.0f, FLT_MAX,
                             rotate ? "%.1f\xC2\xB0" : "%.3f m", ImGuiSliderFlags_AlwaysClamp);
            ImGui::EndPopup();
        }
        ImGui::EndDisabled();
        divider();
        if (toggle("##perturb", icon::kGrab, ui.tool == ViewportTool::Perturb, Tr("Perturb (right-click for the stiffness)")))
            Queue("tool.perturb");
        if (ImGui::BeginPopupContextItem("##perturb_gain")) {
            ImGui::SetNextItemWidth(110.0f * s);
            ImGui::DragFloat(Tr("Stiffness"), &ui.perturb_gain, 0.02f, 0.05f, 20.0f, "%.2f\xC3\x97",
                             ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
            ImGui::EndPopup();
        }
        action("tool.pin", icon::kPin, ui.tool == ViewportTool::Pin);
        const float limit = ImGui::GetCursorScreenPos().x + 12.0f * s;

        // The language switch holds the right edge and the status chips line up to its left;
        // a chip that would cover the buttons is left out.
        float x = wpos.x + wsize.x - 10.0f * s;
        const float cy = wpos.y + wsize.y * 0.5f;
        ImFont* label_font = GetNukaFont(FontRole::Display);
        const float label_px = theme::kFontLabel * s;
        {
            const std::vector<nui::LocaleInfo> locales = nui::Locales();
            const std::string current = nui::Locale();
            const auto it = std::find_if(locales.begin(), locales.end(), [&](const auto& l) { return l.id == current; });
            const std::string tag = it != locales.end() ? it->tag : current;
            const ImVec2 ts = label_font->CalcTextSizeA(label_px, FLT_MAX, 0.0f, tag.c_str());
            const float bw = ts.x + 34.0f * s, bh = 22.0f * s;
            const ImVec2 b0(x - bw, cy - bh * 0.5f), b1(x, cy + bh * 0.5f);
            ImGui::SetCursorScreenPos(b0);
            if (ImGui::InvisibleButton("##language", ImVec2(bw, bh))) Queue("view.toggle_language");
            const bool hovered = ImGui::IsItemHovered();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip))
                ImGui::SetTooltip("%s   %s", Tr("Switch Language"), ShortcutText(registry_->Find("view.toggle_language")->shortcut).c_str());
            draw::Chamfer(list, b0, b1, 4.0f * s, U32(hovered ? theme::kBg2 : theme::Alpha(theme::kAccent, 0.08f)),
                          U32(hovered ? theme::kAccent : theme::kLineHi));
            draw::Glyph(list, ImVec2(b0.x + 13.0f * s, cy), icon::kTranslate, U32(theme::kAccent), false, 13.0f);
            list->AddText(label_font, label_px, ImVec2(b0.x + 24.0f * s, cy - ts.y * 0.5f), U32(theme::kFg), tag.c_str());
            x = b0.x - 6.0f * s;
        }
        // A chip with a tip is a button; it reports the click.
        const auto chip = [&](const std::string& text, const ImVec4& color, const char* glyph, bool pulse, const char* tip) {
            const ImVec2 ts = label_font->CalcTextSizeA(label_px, FLT_MAX, 0.0f, text.c_str());
            const float lead = glyph != nullptr ? 16.0f * s : 0.0f;
            const float cw = ts.x + lead + 16.0f * s, ch = 22.0f * s;
            if (x - cw < limit) return false;
            const ImVec2 c0(x - cw, cy - ch * 0.5f), c1(x, cy + ch * 0.5f);
            bool clicked = false, hovered = false;
            if (tip != nullptr) {
                ImGui::SetCursorScreenPos(c0);
                clicked = ImGui::InvisibleButton(tip, ImVec2(cw, ch));
                hovered = ImGui::IsItemHovered();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) ImGui::SetTooltip("%s", tip);
            }
            draw::Chamfer(list, c0, c1, 4.0f * s, U32(theme::Alpha(color, hovered ? 0.22f : 0.10f)),
                          U32(theme::Alpha(color, hovered ? 0.85f : 0.38f)));
            const ImVec2 mark(c0.x + 14.0f * s, cy);
            if (glyph != nullptr) draw::Glyph(list, mark, glyph, U32(color), false, 13.0f);
            if (pulse) {
                const float t = static_cast<float>(std::fmod(ImGui::GetTime(), 1.6) / 1.6);
                list->AddCircleFilled(mark, (3.0f + 4.0f * t) * s, U32(theme::Alpha(color, 0.35f * (1.0f - t))));
            }
            list->AddText(label_font, label_px, ImVec2(c0.x + 8.0f * s + lead, cy - ts.y * 0.5f), U32(color), text.c_str());
            x = c0.x - 6.0f * s;
            return clicked;
        };
        // A script session names its script; a script that steps the world drives the transport.
        if (ui.script_session)
            chip(std::string(Tr(ui.script_driven ? "SCRIPT DRIVEN" : "SCRIPT")) + "  " + Upper(ui.doc_title), theme::kInfo,
                 icon::kConsole, ui.script_driven, nullptr);
        if (ui.has_scene) {
            if (stats.capped_batches != 0u)
                chip(Fmt(Tr("CAP \xC3\x97%llu"), static_cast<unsigned long long>(stats.capped_batches)), theme::kWarn, icon::kWarn, false,
                     nullptr);
            if (ui.solver.recent_caps != 0u &&
                chip(Fmt(Tr("SOLVER CAP \xC3\x97%u"), ui.solver.recent_caps), theme::kErr, icon::kTarget, true,
                     Tr("Sweep budget spent above the alert residual; click to locate the worst element")))
                Queue("solver.locate");
            if (ui.save_dirty) chip(Tr("UNSAVED"), theme::kWarn, icon::kEdit, false, nullptr);
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

void ImGuiLayer::DrawStatusBar() {
    const ViewerStats& stats = *frame_.stats;
    const float s = nui::Scale();
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f * s, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kBg0);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs;
    if (ImGui::BeginViewportSideBar("##status", ImGui::GetMainViewport(), ImGuiDir_Down, 24.0f * s, flags)) {
        ImDrawList* list = ImGui::GetWindowDrawList();
        const ImVec2 wpos = ImGui::GetWindowPos(), wsize = ImGui::GetWindowSize();
        list->AddLine(wpos, ImVec2(wpos.x + wsize.x, wpos.y), U32(theme::kLine));
        ImFont* font = GetNukaFont(FontRole::Display);
        const float px = theme::kFontLabel * s;
        const float cy = wpos.y + wsize.y * 0.5f;
        // The scene's file and the selection on the left, frame timing on the right.
        std::string left = frame_.ui->has_scene ? Iconed(icon::kFile, Upper(frame_.ui->doc_title).c_str()) : std::string();
        const std::string path = SelectionPath(frame_);
        if (!path.empty()) left += (left.empty() ? "" : "      ") + Iconed(icon::kTarget, Upper(path).c_str());
        const ImVec2 lsize = font->CalcTextSizeA(px, FLT_MAX, 0.0f, left.c_str());
        list->AddText(font, px, ImVec2(wpos.x + 10.0f * s, cy - lsize.y * 0.5f), U32(theme::kFgDim), left.c_str());
        const std::string right = Fmt("%.0f FPS   %.2f MS", stats.fps, stats.frame_time_ms);
        const ImVec2 rsize = font->CalcTextSizeA(px, FLT_MAX, 0.0f, right.c_str());
        list->AddText(font, px, ImVec2(wpos.x + wsize.x - rsize.x - 10.0f * s, cy - rsize.y * 0.5f), U32(theme::kFgDim),
                      right.c_str());
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void ImGuiLayer::DrawPalette() {
    if (!palette_open_) return;
    const float s = nui::Scale();
    const bool first = palette_focus_;
    palette_focus_ = false;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float width = std::min(620.0f * s, vp->WorkSize.x - 32.0f * s);
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + (vp->WorkSize.x - width) * 0.5f, vp->WorkPos.y + 48.0f * s));
    ImGui::SetNextWindowSizeConstraints(ImVec2(width, 0.0f), ImVec2(width, FLT_MAX));
    if (first) ImGui::SetNextWindowFocus();
    ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::Alpha(theme::kBg0, 0.97f));
    ImGui::PushStyleColor(ImGuiCol_Border, theme::kLineHi);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f * s, 10.0f * s));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                   ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs;
    if (ImGui::Begin("##palette", nullptr, flags)) {
        ImDrawList* list = ImGui::GetWindowDrawList();
        const ImVec2 wpos = ImGui::GetWindowPos();
        list->AddLine(wpos, ImVec2(wpos.x + ImGui::GetWindowWidth(), wpos.y), U32(theme::kAccent), 2.0f * s);
        if (first) ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(-FLT_MIN);
        const std::string hint = Iconed(icon::kCommand, Tr("Run an action"));
        if (ImGui::InputTextWithHint("##query", hint.c_str(), palette_query_, sizeof(palette_query_))) palette_index_ = 0;

        const std::string query = Lower(palette_query_);
        std::vector<std::pair<int, const Action*>> matches;
        for (const Action& a : registry_->All()) {
            if (a.hidden) continue;
            int score = 0;
            if (query.empty()) {
                const auto it = std::find(palette_recent_.begin(), palette_recent_.end(), a.id);
                score = it == palette_recent_.end() ? 0 : 100 - static_cast<int>(it - palette_recent_.begin());
            } else {
                const char* category = CategoryTitle(a.category);
                score = Fuzzy(query, Lower(Tr(a.label) + " " + a.label + " " + Tr(category) + " " + category + " " + a.id));
                if (score < 0) continue;
            }
            matches.emplace_back(score, &a);
        }
        std::stable_sort(matches.begin(), matches.end(), [](const auto& l, const auto& r) { return l.first > r.first; });
        const int shown = static_cast<int>(std::min<size_t>(matches.size(), 12u));
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) ++palette_index_;
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) --palette_index_;
        palette_index_ = std::clamp(palette_index_, 0, std::max(shown - 1, 0));
        const bool enter = ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter);

        const Action* chosen = nullptr;
        ImGui::Dummy(ImVec2(0.0f, 4.0f * s));
        const float row_h = ImGui::GetTextLineHeight() + 12.0f * s;
        ImFont* label_font = GetNukaFont(FontRole::Display);
        const float label_px = theme::kFontLabel * s;
        ImFont* mono = GetNukaFont(FontRole::Mono);
        const float mono_px = theme::kFontMono * s;
        const ImVec2 mouse_delta = ImGui::GetIO().MouseDelta;
        ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
        for (int i = 0; i < shown; ++i) {
            const Action& a = *matches[static_cast<size_t>(i)].second;
            const bool enabled = !a.enabled || a.enabled();
            ImGui::PushID(i);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float w = ImGui::GetContentRegionAvail().x;
            if (ImGui::InvisibleButton("##row", ImVec2(w, row_h)) && enabled) chosen = &a;
            if (ImGui::IsItemHovered() && (mouse_delta.x != 0.0f || mouse_delta.y != 0.0f)) palette_index_ = i;
            const bool current = i == palette_index_;
            if (current) {
                list->AddRectFilled(p, ImVec2(p.x + w, p.y + row_h), U32(theme::Alpha(theme::kAccent, 0.12f)), theme::kRadius * s);
                list->AddRectFilled(p, ImVec2(p.x + 2.0f * s, p.y + row_h), U32(theme::kAccent));
            }
            const float mid = p.y + row_h * 0.5f;
            if (!a.icon.empty())
                draw::Glyph(list, ImVec2(p.x + 16.0f * s, mid), a.icon.c_str(), U32(current ? theme::kAccent : theme::kFgMuted), false,
                            15.0f);
            const std::string& label = Tr(a.label);
            list->AddText(ImVec2(p.x + 34.0f * s, mid - ImGui::GetTextLineHeight() * 0.5f), U32(enabled ? theme::kFg : theme::kFgDim),
                          label.c_str());
            const std::string category = Upper(Tr(CategoryTitle(a.category)));
            const ImVec2 csize = label_font->CalcTextSizeA(label_px, FLT_MAX, 0.0f, category.c_str());
            list->AddText(label_font, label_px,
                          ImVec2(p.x + 44.0f * s + ImGui::CalcTextSize(label.c_str()).x, mid - csize.y * 0.5f),
                          U32(theme::kFgDim), category.c_str());
            const std::string keys = ShortcutText(a.shortcut);
            if (!keys.empty()) {
                const ImVec2 ksize = mono->CalcTextSizeA(mono_px, FLT_MAX, 0.0f, keys.c_str());
                list->AddText(mono, mono_px, ImVec2(p.x + w - ksize.x - 10.0f * s, mid - ksize.y * 0.5f), U32(theme::kFgMuted),
                              keys.c_str());
            }
            ImGui::PopID();
        }
        ImGui::PopItemFlag();
        if (shown == 0) ImGui::TextColored(theme::kFgDim, "%s", Tr("No matching action"));
        if (enter && shown > 0) {
            const Action& a = *matches[static_cast<size_t>(palette_index_)].second;
            if (!a.enabled || a.enabled()) chosen = &a;
        }
        const bool lost = !first && !ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
        if (chosen != nullptr) {
            Queue(chosen->id);
            palette_recent_.erase(std::remove(palette_recent_.begin(), palette_recent_.end(), chosen->id), palette_recent_.end());
            palette_recent_.insert(palette_recent_.begin(), chosen->id);
            if (palette_recent_.size() > 8u) palette_recent_.resize(8u);
        }
        if (chosen != nullptr || lost || ImGui::IsKeyPressed(ImGuiKey_Escape)) palette_open_ = false;
    }
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
}

void ImGuiLayer::DrawAbout() {
    const std::string title = std::string(Tr("About Nuka Editor")) + "###about";
    if (about_open_) {
        ImGui::OpenPopup(title.c_str());
        about_open_ = false;
    }
    const float s = nui::Scale();
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(460.0f * s, 0.0f), ImVec2(460.0f * s, FLT_MAX));
    if (!ImGui::BeginPopupModal(title.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize))
        return;
    render::imgui::DrawNukaLogo(40.0f * s);
    ImGui::SameLine(0.0f, 12.0f * s);
    ImGui::BeginGroup();
    nui::PushTextStyle(TextStyle::Hero);
    ImGui::TextUnformatted("NUKA EDITOR");
    nui::PopTextStyle();
    ImGui::TextColored(theme::kFgMuted, "%s", Tr("Physics-first scene editor"));
    ImGui::EndGroup();
    draw::SectionHeader(Tr("Versions"));
    ImGui::TextColored(theme::kFgMuted, "nui API v%u    Dear ImGui %s", NUKA_NUI_API_VERSION, IMGUI_VERSION);
    draw::SectionHeader(Tr("Open source"));
    for (const char* credit : {"Phosphor Icons 2.1.1  MIT", "Nerd Fonts 3.4.0: Ubuntu Sans, Martian Mono, Geist Mono  OFL 1.1",
                               "Noto Sans CJK SC  OFL 1.1", "Dear ImGui, ImPlot, ImGuizmo  MIT", "SDL3  zlib"})
        ImGui::TextColored(theme::kFgMuted, "%s", credit);
    ImGui::Dummy(ImVec2(0.0f, 8.0f * s));
    if (ImGui::Button(Tr("Close"), ImVec2(-FLT_MIN, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

// ---- start page ---------------------------------------------------------------------------

namespace {

// One line of the start page: an icon or a picture, a title, an optional dim second line and a
// trailing note. True when clicked.
bool StartRow(const char* id, const char* glyph, const std::string& title, const std::string& detail, const std::string& note,
              bool enabled, const ImVec4& tint = theme::kFgMuted, ImTextureData* picture = nullptr) {
    const float s = nui::Scale();
    ImDrawList* list = ImGui::GetWindowDrawList();
    ImFont* body = ImGui::GetFont();
    const float body_px = ImGui::GetFontSize();
    ImFont* mono = GetNukaFont(FontRole::Mono);
    const float mono_px = theme::kFontMono * s;
    const float line = ImGui::GetTextLineHeight();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    const float h = line + (detail.empty() ? 0.0f : mono_px + 4.0f * s) + 14.0f * s;
    ImGui::PushID(id);
    const bool clicked = ImGui::InvisibleButton("##row", ImVec2(w, h)) && enabled;
    ImGui::PopID();
    const bool hot = enabled && ImGui::IsItemHovered();
    if (hot) {
        list->AddRectFilled(p, ImVec2(p.x + w, p.y + h), U32(theme::Alpha(theme::kAccent, 0.10f)), theme::kRadius * s);
        list->AddRectFilled(p, ImVec2(p.x + 2.0f * s, p.y + h), U32(theme::kAccent));
    }
    const float top = p.y + 7.0f * s;
    float x = p.x + 34.0f * s;
    if (picture != nullptr) {
        const ImVec2 a(p.x + 8.0f * s, p.y + 5.0f * s), b(a.x + (h - 10.0f * s) * 16.0f / 9.0f, p.y + h - 5.0f * s);
        list->AddImage(picture->GetTexRef(), a, b);
        list->AddRect(a, b, U32(hot ? theme::kAccent : theme::kLine));
        x = b.x + 10.0f * s;
    } else {
        draw::Glyph(list, ImVec2(p.x + 16.0f * s, top + line * 0.5f), glyph,
                    U32(!enabled ? theme::kFgDim : hot ? theme::kAccent : tint), hot, 16.0f);
    }
    const float right = p.x + w - 10.0f * s;
    float note_w = 0.0f;
    if (!note.empty()) {
        const ImVec2 size = mono->CalcTextSizeA(mono_px, FLT_MAX, 0.0f, note.c_str());
        note_w = size.x + 12.0f * s;
        list->AddText(mono, mono_px, ImVec2(right - size.x, top + (line - size.y) * 0.5f), U32(theme::kFgDim), note.c_str());
    }
    list->AddText(body, body_px, ImVec2(x, top), U32(enabled ? theme::kFg : theme::kFgDim),
                  Elide(body, body_px, title, right - note_w - x, false).c_str());
    if (!detail.empty())
        list->AddText(mono, mono_px, ImVec2(x, top + line + 4.0f * s), U32(theme::kFgDim),
                      Elide(mono, mono_px, detail, right - x, false).c_str());
    return clicked;
}

}  // namespace

// The editor's own document, a full-window page between the toolbar and the status bar while
// no scene is open: templates, recent scenes with their last viewport, examples and checks.
void ImGuiLayer::DrawStartPage() {
    const StartPage& page = frame_.ui->start;
    if (!page.shown) return;
    const float s = nui::Scale();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kBg0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(36.0f * s, 26.0f * s));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                   ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoScrollWithMouse;
    const bool open = ImGui::Begin("##start", nullptr, flags);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    const ImVec2 origin = ImGui::GetWindowPos(), size = ImGui::GetWindowSize();
    const ImVec2 corner(origin.x + size.x, origin.y + size.y);
    if (open) {
        ImDrawList* list = ImGui::GetWindowDrawList();
        // A faint blueprint grid under an accent glow from the top-left.
        const float cell = 32.0f * s;
        const ImU32 grid = U32(theme::Alpha(theme::kLine, 0.35f));
        for (float gx = origin.x + cell; gx < corner.x; gx += cell) list->AddLine(ImVec2(gx, origin.y), ImVec2(gx, corner.y), grid);
        for (float gy = origin.y + cell; gy < corner.y; gy += cell) list->AddLine(ImVec2(origin.x, gy), ImVec2(corner.x, gy), grid);
        list->AddRectFilledMultiColor(origin, corner, U32(theme::Alpha(theme::kAccent, 0.10f)), U32(theme::Alpha(theme::kAccent, 0.0f)),
                                      U32(theme::Alpha(theme::kBg0, 0.0f)), U32(theme::Alpha(theme::kAccent, 0.03f)));

        render::imgui::DrawNukaLogo(44.0f * s);
        ImGui::SameLine(0.0f, 14.0f * s);
        ImGui::BeginGroup();
        nui::PushTextStyle(TextStyle::Hero);
        ImGui::TextUnformatted("NUKA EDITOR");
        nui::PopTextStyle();
        ImGui::TextColored(theme::kFgMuted, "%s", Tr("Physics-first scene editor"));
        ImGui::EndGroup();
        const auto keys = [this](const char* id) { return ShortcutText(registry_->Find(id)->shortcut); };
        {
            ImFont* label_font = GetNukaFont(FontRole::Display);
            const float label_px = theme::kFontLabel * s;
            const std::string hint = Upper(keys("file.new") + "  " + Tr("New") + "     " + keys("file.open") + "  " + Tr("Open") +
                                           "     " + keys("app.command_palette") + "  " + Tr("Actions"));
            const ImVec2 hs = label_font->CalcTextSizeA(label_px, FLT_MAX, 0.0f, hint.c_str());
            const ImVec2 at(corner.x - 36.0f * s - hs.x, ImGui::GetItemRectMin().y + (ImGui::GetItemRectSize().y - hs.y) * 0.5f);
            list->AddText(label_font, label_px, at, U32(theme::kFgDim), hint.c_str());
        }
        const ImVec2 rule = ImGui::GetCursorScreenPos();
        const float full = ImGui::GetContentRegionAvail().x;
        list->AddRectFilled(ImVec2(rule.x, rule.y + 10.0f * s), ImVec2(rule.x + full, rule.y + 11.0f * s), U32(theme::kLine));
        list->AddRectFilled(ImVec2(rule.x, rule.y + 9.5f * s), ImVec2(rule.x + 96.0f * s, rule.y + 11.5f * s), U32(theme::kAccent));
        ImGui::Dummy(ImVec2(full, 26.0f * s));

        const float gap = 28.0f * s;
        const float line = ImGui::GetTextLineHeight();
        const float footer = line + 14.0f * s;
        const float column_h = ImGui::GetContentRegionAvail().y - footer;
        const float side = std::clamp(full * 0.24f, 220.0f * s, 300.0f * s);
        const float middle = full - side * 2.0f - gap * 2.0f;
        const auto path_args = [](const std::string& path) { return Args({{"path", json::Value::Str(path)}}); };

        ImGui::BeginChild("##start_new", ImVec2(side, column_h));
        draw::SectionHeader(Tr("New"));
        for (size_t i = 0; i < page.templates.size(); ++i) {
            const StartEntry& t = page.templates[i];
            if (StartRow(t.arg.c_str(), icon::kTemplate, Tr(t.title), Tr(t.detail), i == 0 ? keys("file.new") : "", true))
                Queue("file.new", Args({{"template", json::Value::Str(t.arg)}}));
        }
        ImGui::Dummy(ImVec2(0.0f, 6.0f * s));
        draw::SectionHeader(Tr("Open"));
        if (StartRow("open", icon::kOpen, Tr("Open Scene..."), "", keys("file.open"), true)) Queue("file.open");
        if (StartRow("window", icon::kNewWindow, Tr("Open in New Window..."), "", "", true)) Queue("file.open_window");
        if (StartRow("import", icon::kImport, Tr("Import Asset..."), Tr("URDF, MJCF, USD and mesh files"), keys("file.import"), true))
            Queue("file.import");
        ImGui::EndChild();

        ImGui::SameLine(0.0f, gap);
        ImGui::BeginChild("##start_recent", ImVec2(middle, column_h));
        // Work left by an editor that ended without saving comes first.
        if (!page.recovered.empty()) {
            draw::SectionHeader(Tr("Recover"));
            for (const StartEntry& r : page.recovered)
                if (StartRow(r.arg.c_str(), icon::kRecover, Fmt(Tr("%s (autosave)"), Tr(r.title).c_str()), r.detail, "", true,
                             theme::kWarn))
                    Queue("file.recover", path_args(r.arg));
            ImGui::Dummy(ImVec2(0.0f, 6.0f * s));
        }
        draw::SectionHeader(Tr("Recent"), Fmt("%zu", page.recent.size()).c_str());
        if (page.recent.empty()) ImGui::TextColored(theme::kFgDim, "%s", Tr("Scenes you open or save show up here."));
        {
            ImDrawList* cards = ImGui::GetWindowDrawList();
            ImFont* body = ImGui::GetFont();
            const float body_px = ImGui::GetFontSize();
            ImFont* mono = GetNukaFont(FontRole::Mono);
            const float mono_px = theme::kFontMono * s;
            const float card_gap = 14.0f * s, inset = 4.0f * s;
            const float avail = ImGui::GetContentRegionAvail().x;
            const size_t per_row = static_cast<size_t>(std::max(1.0f, std::floor((avail + card_gap) / (180.0f * s + card_gap))));
            const float card_w = (avail - card_gap * static_cast<float>(per_row - 1u)) / static_cast<float>(per_row);
            const float thumb_h = (card_w - inset * 2.0f) * 9.0f / 16.0f;
            const float card_h = inset + thumb_h + line + mono_px + 18.0f * s;
            for (size_t i = 0; i < page.recent.size(); ++i) {
                const StartEntry& r = page.recent[i];
                if (i % per_row != 0u) ImGui::SameLine(0.0f, card_gap);
                else if (i != 0u) ImGui::Dummy(ImVec2(0.0f, card_gap - ImGui::GetStyle().ItemSpacing.y));
                ImGui::PushID(static_cast<int>(i));
                ImGui::BeginGroup();
                const ImVec2 p = ImGui::GetCursorScreenPos(), q(p.x + card_w, p.y + card_h);
                ImGui::SetNextItemAllowOverlap();
                const bool clicked = ImGui::InvisibleButton("##card", ImVec2(card_w, card_h));
                const bool hot = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(p, q);
                draw::Chamfer(cards, p, q, theme::kChamfer * s, U32(hot ? theme::kBg2 : theme::kBg1),
                              U32(hot && r.available ? theme::kAccent : theme::kLine));
                const ImVec2 t0(p.x + inset, p.y + inset), t1(q.x - inset, p.y + inset + thumb_h);
                if (ImTextureData* tex = r.thumbnail.empty() ? nullptr : Thumbnail(r.thumbnail)) {
                    cards->AddImage(tex->GetTexRef(), t0, t1);
                } else {
                    cards->AddRectFilled(t0, t1, U32(theme::kBg0));
                    draw::Glyph(cards, ImVec2((t0.x + t1.x) * 0.5f, (t0.y + t1.y) * 0.5f), icon::kScene, U32(theme::kLineHi), false, 28.0f);
                }
                if (!r.available) {
                    cards->AddRectFilled(t0, t1, U32(theme::Alpha(theme::kBg0, 0.65f)));
                    const char* missing = Tr("MISSING");
                    const ImVec2 ms = mono->CalcTextSizeA(mono_px, FLT_MAX, 0.0f, missing);
                    cards->AddText(mono, mono_px, ImVec2((t0.x + t1.x - ms.x) * 0.5f, (t0.y + t1.y - ms.y) * 0.5f), U32(theme::kWarn),
                                   missing);
                }
                const float text_x = p.x + inset + 4.0f * s, text_w = card_w - inset * 2.0f - 8.0f * s;
                const float title_y = t1.y + 7.0f * s;
                const ImVec2 ts = mono->CalcTextSizeA(mono_px, FLT_MAX, 0.0f, r.detail.c_str());
                cards->AddText(mono, mono_px, ImVec2(text_x + text_w - ts.x, title_y + (line - ts.y) * 0.5f), U32(theme::kFgDim),
                               r.detail.c_str());
                cards->AddText(body, body_px, ImVec2(text_x, title_y), U32(r.available ? theme::kFg : theme::kFgDim),
                               Elide(body, body_px, r.title, text_w - ts.x - 10.0f * s, false).c_str());
                cards->AddText(mono, mono_px, ImVec2(text_x, title_y + line + 3.0f * s), U32(theme::kFgDim),
                               Elide(mono, mono_px, r.arg, text_w, true).c_str());
                // Remove from the list, over the picture's top-right corner while the card is hovered.
                const float b = 22.0f * s;
                const ImVec2 x0(t1.x - b - 4.0f * s, t0.y + 4.0f * s), x1(x0.x + b, x0.y + b);
                ImGui::SetCursorScreenPos(x0);
                const bool forget = ImGui::InvisibleButton("##forget", ImVec2(b, b));
                if (hot) {
                    const bool close_hot = ImGui::IsItemHovered();
                    cards->AddRectFilled(x0, x1, U32(theme::Alpha(theme::kBg0, 0.85f)), theme::kRadius * s);
                    draw::Glyph(cards, ImVec2((x0.x + x1.x) * 0.5f, (x0.y + x1.y) * 0.5f), icon::kClose,
                                U32(close_hot ? theme::kErr : theme::kFgMuted), false, 13.0f);
                    if (close_hot) ImGui::SetTooltip("%s", Tr("Remove from Recent"));
                }
                ImGui::EndGroup();
                ImGui::PopID();
                if (forget) Queue("file.forget_recent", path_args(r.arg));
                else if (clicked && r.available) Queue("file.open_recent", path_args(r.arg));
            }
        }
        ImGui::EndChild();

        ImGui::SameLine(0.0f, gap);
        ImGui::BeginChild("##start_more", ImVec2(side, column_h));
        draw::SectionHeader(Tr("Examples"));
        for (const StartEntry& e : page.examples)
            if (StartRow(e.arg.c_str(), icon::kExample, Tr(e.title), "", e.detail, e.available, theme::kFgMuted,
                         e.thumbnail.empty() ? nullptr : Thumbnail(e.thumbnail)))
                Queue("file.open", path_args(e.arg));
        ImGui::EndChild();

        const std::string drop = Iconed(icon::kImport, Tr("Drop a scene to open it, or an asset to add it"));
        const float foot_y = corner.y - 26.0f * s - line, foot_mid = foot_y + line * 0.5f;
        ImGui::SetCursorScreenPos(ImVec2(rule.x, foot_y));
        ImGui::TextColored(theme::kFgDim, "%s", drop.c_str());
        // The startup checks line up from the right of the footer; a hovered one shows its value.
        ImFont* label_font = GetNukaFont(FontRole::Display);
        const float label_px = theme::kFontLabel * s, dot = 3.0f * s;
        float right = rule.x + full;
        for (auto c = page.checks.rbegin(); c != page.checks.rend(); ++c) {
            const std::string name = Upper(Tr(c->title));
            const ImVec2 ns = label_font->CalcTextSizeA(label_px, FLT_MAX, 0.0f, name.c_str());
            const ImVec2 a(right - ns.x - 8.0f * s - dot * 2.0f, foot_y), b(right, foot_y + line);
            list->AddCircleFilled(ImVec2(a.x + dot, foot_mid), dot, U32(c->available ? theme::kOk : theme::kErr));
            list->AddText(label_font, label_px, ImVec2(right - ns.x, foot_mid - ns.y * 0.5f),
                          U32(c->available ? theme::kFgDim : theme::kErr), name.c_str());
            if (ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(a, b)) ImGui::SetTooltip("%s  %s", name.c_str(), c->detail.c_str());
            right = a.x - 18.0f * s;
        }
    }
    ImGui::End();
    // Toasts go over the page's columns, which leave no room under the windows.
    host_.DrawNotifications(ImGui::GetForegroundDrawList(ImGui::GetMainViewport()), origin, corner);
}

// The question a document-dropping action asks while there are unsaved changes.
void ImGuiLayer::DrawPrompt() {
    std::string& asked = frame_.ui->prompt;
    const std::string title = std::string(Tr("Unsaved Changes")) + "###prompt";
    if (!asked.empty() && !ImGui::IsPopupOpen(title.c_str())) ImGui::OpenPopup(title.c_str());
    const float s = nui::Scale();
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(440.0f * s, 0.0f), ImVec2(440.0f * s, FLT_MAX));
    if (!ImGui::BeginPopupModal(title.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize)) return;
    if (asked.empty()) {
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        return;
    }
    ImDrawList* list = ImGui::GetWindowDrawList();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    draw::Glyph(list, ImVec2(at.x + 14.0f * s, at.y + 14.0f * s), icon::kWarn, U32(theme::kWarn), true, 22.0f);
    ImGui::SetCursorScreenPos(ImVec2(at.x + 38.0f * s, at.y));
    ImGui::BeginGroup();
    nui::PushTextStyle(TextStyle::Strong);
    ImGui::TextWrapped("%s", Fmt(Tr("Save changes to %s?"), frame_.ui->doc_title.c_str()).c_str());
    nui::PopTextStyle();
    ImGui::TextColored(theme::kFgMuted, "%s", Tr("Your changes are lost if you don't save them."));
    ImGui::EndGroup();
    ImGui::Dummy(ImVec2(0.0f, 10.0f * s));

    const float bw = 112.0f * s, spacing = ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - bw * 3.0f - spacing * 2.0f);
    std::string choice;
    ImGui::PushStyleColor(ImGuiCol_Button, theme::kAccent);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, theme::Alpha(theme::kAccent, 0.85f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, theme::kAccentDim);
    ImGui::PushStyleColor(ImGuiCol_Text, theme::kAccentInk);
    if (ImGui::Button(Tr("Save"), ImVec2(bw, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter))
        choice = "save";
    ImGui::PopStyleColor(4);
    ImGui::SameLine();
    if (ImGui::Button(Tr("Don't Save"), ImVec2(bw, 0.0f))) choice = "discard";
    ImGui::SameLine();
    if (ImGui::Button(Tr("Cancel"), ImVec2(bw, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) choice = "cancel";
    if (!choice.empty()) {
        Queue("app.answer_prompt", Args({{"choice", json::Value::Str(choice)}}));
        asked.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

// Thumbnails share one size, so a newer file re-uploads into the texture it already has.
ImTextureData* ImGuiLayer::Thumbnail(const std::string& file) {
    const fs::path path = Utf8Path(file);
    std::error_code error;
    const fs::file_time_type stamp = fs::last_write_time(path, error);
    if (error) return nullptr;
    auto it = std::find_if(thumbs_.begin(), thumbs_.end(), [&](const Thumb& t) { return t.file == file; });
    if (it != thumbs_.end() && it->stamp == stamp) return it->texture.get();
    std::ifstream in(path, std::ios::binary);
    std::string magic;
    int w = 0, h = 0, max_value = 0;
    in >> magic >> w >> h >> max_value;
    in.get();
    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<unsigned char> rgb(count * 3u);
    in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    if (it == thumbs_.end()) {
        thumbs_.push_back(Thumb{file, stamp, std::make_unique<ImTextureData>()});
        it = std::prev(thumbs_.end());
        it->texture->Create(ImTextureFormat_RGBA32, w, h);
        ImGui::RegisterUserTexture(it->texture.get());
    } else {
        it->stamp = stamp;
        it->texture->UpdateRect = ImTextureRect{0, 0, static_cast<unsigned short>(w), static_cast<unsigned short>(h)};
        if (it->texture->Status == ImTextureStatus_OK) it->texture->SetStatus(ImTextureStatus_WantUpdates);
    }
    unsigned char* pixels = static_cast<unsigned char*>(it->texture->GetPixels());
    for (size_t i = 0; i < count; ++i) {
        pixels[i * 4u + 0u] = rgb[i * 3u + 0u];
        pixels[i * 4u + 1u] = rgb[i * 3u + 1u];
        pixels[i * 4u + 2u] = rgb[i * 3u + 2u];
        pixels[i * 4u + 3u] = 255u;
    }
    return it->texture.get();
}

void ImGuiLayer::DrawPerturbHud(const ImVec2& vmin, const ImVec2& vmax) {
    const ViewerUiState& ui = *frame_.ui;
    if (ui.held.empty()) return;
    float view_m[16], proj[16];
    CameraMatrices(*frame_.camera, view_m, proj);
    const auto project = [&](const float p[3], ImVec2* out) {
        float e[4], c[4];
        for (int r = 0; r < 4; ++r) e[r] = view_m[r] * p[0] + view_m[4 + r] * p[1] + view_m[8 + r] * p[2] + view_m[12 + r];
        for (int r = 0; r < 4; ++r) c[r] = proj[r] * e[0] + proj[4 + r] * e[1] + proj[8 + r] * e[2] + proj[12 + r] * e[3];
        if (c[3] <= 1.0e-6f) return false;
        *out = ImVec2(vmin.x + (c[0] / c[3] * 0.5f + 0.5f) * (vmax.x - vmin.x),
                      vmin.y + (0.5f - c[1] / c[3] * 0.5f) * (vmax.y - vmin.y));
        return true;
    };
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    list->PushClipRect(vmin, vmax, true);
    const float s = nui::Scale();
    ImFont* font = GetNukaFont(FontRole::Display);
    const float px = theme::kFontLabel * s;
    const ImU32 accent = U32(theme::kAccent);
    for (const HeldLoad& h : ui.held) {
        ImVec2 a, t;
        if (!project(h.at, &a) || !project(h.target, &t)) continue;
        // A taut band from the grab point to the target, its diamond and the pull it makes.
        list->AddLine(a, t, U32(theme::Alpha(theme::kAccent, 0.35f)), 5.0f * s);
        list->AddLine(a, t, accent, 1.5f * s);
        list->AddCircle(a, 6.0f * s, accent, 20, 1.5f * s);
        list->AddCircleFilled(a, 2.5f * s, accent);
        const float d = 5.0f * s;
        list->AddQuadFilled(ImVec2(t.x, t.y - d), ImVec2(t.x + d, t.y), ImVec2(t.x, t.y + d), ImVec2(t.x - d, t.y), accent);
        if (!h.body) continue;
        const std::string label = Fmt("%.1f N", static_cast<double>(h.force));
        const ImVec2 ts = font->CalcTextSizeA(px, FLT_MAX, 0.0f, label.c_str());
        const ImVec2 l0(t.x + 10.0f * s, t.y - ts.y * 0.5f - 3.0f * s);
        draw::Chamfer(list, l0, ImVec2(l0.x + ts.x + 12.0f * s, l0.y + ts.y + 6.0f * s), 3.0f * s,
                      U32(theme::Alpha(theme::kBg0, 0.85f)), U32(theme::Alpha(theme::kAccent, 0.7f)));
        list->AddText(font, px, ImVec2(l0.x + 6.0f * s, l0.y + 3.0f * s), accent, label.c_str());
    }
    list->PopClipRect();
}

// The floating windows over the viewport: no frame, no background, sized to their content.
constexpr ImGuiWindowFlags kOverlayFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
                                           ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                           ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                                           ImGuiWindowFlags_AlwaysAutoResize;

void ImGuiLayer::DrawViewButtons() {
    const ViewerUiState& ui = *frame_.ui;
    const float s = nui::Scale();
    if (draw::IconButton("##rt", icon::kSparkle, ui.rt.on, Tr("RT Preview"))) Queue("view.rt_preview");
    ImGui::SameLine(0.0f, 4.0f * s);
    if (draw::IconButton("##render", icon::kCamera, ui.rendering, Tr("Render This View..."))) ImGui::OpenPopup("##render_view");
    if (ImGui::BeginPopup("##render_view")) {
        ImGui::TextUnformatted(Tr("Render This View"));
        ImGui::SetNextItemWidth(280.0f * s);
        ImGui::InputTextWithHint("##path", Tr("Output PNG (empty asks)"), render_path_, sizeof(render_path_));
        ImGui::SetNextItemWidth(180.0f * s);
        ImGui::InputInt2(Tr("Size"), render_size_);
        ImGui::SetNextItemWidth(180.0f * s);
        ImGui::InputInt(Tr("Samples"), &render_spp_);
        ImGui::BeginDisabled(ui.rendering);
        if (ImGui::Button(Tr("Render"))) {
            Queue("view.render", Args({{"path", json::Value::Str(render_path_)},
                                       {"width", json::Value::Int(std::max(render_size_[0], 16))},
                                       {"height", json::Value::Int(std::max(render_size_[1], 16))},
                                       {"spp", json::Value::Int(std::max(render_spp_, 1))}}));
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndDisabled();
        ImGui::EndPopup();
    }
    ImGui::SameLine(0.0f, 4.0f * s);
    const char* layout_icon = ui.view_layout == ViewLayout::Quad    ? icon::kViewQuad
                              : ui.view_layout == ViewLayout::Split ? icon::kViewSplit
                                                                    : icon::kViewSingle;
    if (draw::IconButton("##layout", layout_icon, false, Tr("View Layout"))) ImGui::OpenPopup("##view_layout");
    if (ImGui::BeginPopup("##view_layout")) {
        for (const char* id : {"view.layout_single", "view.layout_split", "view.layout_quad", "view.maximize"}) MenuAction(id);
        ImGui::EndPopup();
    }
}

void ImGuiLayer::DrawViewTools(const ImVec2& vmin, const ImVec2& vmax) {
    const ViewerUiState& ui = *frame_.ui;
    const float s = nui::Scale();
    // With several panes these sit in the active pane's header.
    if (ui.panes.size() == 1u) {
        ImGui::SetNextWindowPos(ImVec2(vmax.x - 16.0f * s, vmin.y + 14.0f * s), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
        if (ImGui::Begin("##view_tools", nullptr, kOverlayFlags)) {
            DrawViewButtons();
            ImGui::SameLine(0.0f, 4.0f * s);
            ImGui::SetNextItemWidth(130.0f * s);
            const ViewModeInfo& shown = kViewModes[static_cast<size_t>(ui.view_mode)];
            if (ImGui::BeginCombo("##mode", Tr(shown.label))) {
                for (const ViewModeInfo& m : kViewModes)
                    if (ImGui::Selectable(Tr(m.label), m.mode == ui.view_mode))
                        Queue("view.mode", Args({{"mode", json::Value::Str(m.name)}}));
                ImGui::EndCombo();
            }
        }
        ImGui::End();
    }
    if (!ui.select_box) return;
    const ImVec2 a(std::min(ui.select_box_from[0], ui.select_box_to[0]), std::min(ui.select_box_from[1], ui.select_box_to[1]));
    const ImVec2 b(std::max(ui.select_box_from[0], ui.select_box_to[0]), std::max(ui.select_box_from[1], ui.select_box_to[1]));
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    list->PushClipRect(vmin, vmax, true);
    list->AddRectFilled(a, b, U32(theme::Alpha(theme::kAccent, 0.10f)));
    list->AddRect(a, b, U32(theme::Alpha(theme::kAccent, 0.8f)), 0.0f, 0, 1.0f * s);
    list->PopClipRect();
}

void ImGuiLayer::DrawRtImage(const ImVec2& vmin, const ImVec2& vmax) {
    const RtView& rt = frame_.ui->rt;
    // A replaced texture waits out the frames in flight before the backend destroys it.
    for (auto it = rt_retired_.begin(); it != rt_retired_.end();) {
        if ((*it)->Status == ImTextureStatus_Destroyed) {
            ImGui::UnregisterUserTexture(it->get());
            it = rt_retired_.erase(it);
        } else {
            ++(*it)->UnusedFrames;
            ++it;
        }
    }
    if (!rt.on || !rt.shown) return;
    if (rt_version_ != rt.version) {
        rt_version_ = rt.version;
        const int w = static_cast<int>(rt.width), h = static_cast<int>(rt.height);
        if (rt_texture_ && (rt_texture_->Width != w || rt_texture_->Height != h)) {
            rt_texture_->SetStatus(ImTextureStatus_WantDestroy);
            rt_retired_.push_back(std::move(rt_texture_));
        }
        if (!rt_texture_) {
            rt_texture_ = std::make_unique<ImTextureData>();
            rt_texture_->Create(ImTextureFormat_RGBA32, w, h);
            ImGui::RegisterUserTexture(rt_texture_.get());
        } else {
            rt_texture_->UpdateRect = ImTextureRect{0, 0, static_cast<unsigned short>(w), static_cast<unsigned short>(h)};
            if (rt_texture_->Status == ImTextureStatus_OK) rt_texture_->SetStatus(ImTextureStatus_WantUpdates);
        }
        unsigned char* pixels = static_cast<unsigned char*>(rt_texture_->GetPixels());
        for (size_t i = 0; i < static_cast<size_t>(w) * static_cast<size_t>(h); ++i) {
            pixels[i * 4u + 0u] = rt.rgb[i * 3u + 0u];
            pixels[i * 4u + 1u] = rt.rgb[i * 3u + 1u];
            pixels[i * 4u + 2u] = rt.rgb[i * 3u + 2u];
            pixels[i * 4u + 3u] = 255u;
        }
    }
    ImGui::GetBackgroundDrawList()->AddImage(rt_texture_->GetTexRef(), vmin, vmax);
}

void ImGuiLayer::DrawBookmarkBar(const ImVec2& vmin, const ImVec2& vmax) {
    const ViewerUiState& ui = *frame_.ui;
    const float s = nui::Scale();
    ImGui::SetNextWindowPos(ImVec2(vmin.x + 16.0f * s, vmax.y - 14.0f * s), ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    if (ImGui::Begin("##bookmarks", nullptr, kOverlayFlags)) {
        for (size_t i = 0; i < ui.bookmarks.size(); ++i) {
            const std::string& name = ui.bookmarks[i];
            ImGui::PushID(static_cast<int>(i));
            if (draw::Chip("bookmark", Iconed(icon::kCameraNode, name.c_str()), false))
                Queue("view.bookmark_goto", Args({{"name", json::Value::Str(name)}}));
            if (ImGui::BeginPopupContextItem("##bookmark_menu")) {
                if (ImGui::IsWindowAppearing()) std::snprintf(bookmark_name_, sizeof(bookmark_name_), "%s", name.c_str());
                ImGui::SetNextItemWidth(160.0f * s);
                const bool enter = ImGui::InputText("##name", bookmark_name_, sizeof(bookmark_name_),
                                                    ImGuiInputTextFlags_EnterReturnsTrue);
                ImGui::SameLine();
                if ((ImGui::SmallButton(Tr("Rename")) || enter) && bookmark_name_[0] != '\0') {
                    Queue("view.bookmark_rename", Args({{"name", json::Value::Str(name)}, {"to", json::Value::Str(bookmark_name_)}}));
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::MenuItemEx(Tr("Remove"), icon::kTrash)) Queue("view.bookmark_remove", Args({{"name", json::Value::Str(name)}}));
                ImGui::EndPopup();
            }
            ImGui::PopID();
            ImGui::SameLine(0.0f, 4.0f * s);
        }
        if (draw::IconButton("##add_bookmark", icon::kAdd, false, Tr("Add Camera Bookmark"))) Queue("view.bookmark_add");
    }
    ImGui::End();
}

void ImGuiLayer::DrawPaneHeaders(const ImVec2& cmin, const ImVec2& cmax, const ImVec2& pane, float gap) {
    const ViewerUiState& ui = *frame_.ui;
    const float s = nui::Scale();
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    list->AddRectFilled(ImVec2(cmin.x + pane.x, cmin.y), ImVec2(cmin.x + pane.x + gap, cmax.y), U32(theme::kBg0));
    if (ui.panes.size() > 2u)
        list->AddRectFilled(ImVec2(cmin.x, cmin.y + pane.y), ImVec2(cmax.x, cmin.y + pane.y + gap), U32(theme::kBg0));
    for (const ViewPane& p : ui.panes) {
        const SceneViewportRect& r = pane_rects_[p.index];
        if (!r.Valid()) continue;
        const ImVec2 pmin(r.x, r.y), pmax(r.x + r.width, r.y + r.height);
        const bool active = p.index == ui.active_view;
        const std::string camera = p.camera.substr(p.camera.rfind('/') + 1u);
        if (active) list->AddRect(pmin, pmax, U32(theme::kAccent), 0.0f, 0, 1.5f * s);
        else draw::ViewportFrame(pmin, pmax, camera.empty() ? std::vector<std::string>{}
                                                             : std::vector<std::string>{Iconed(icon::kCameraNode, camera.c_str())},
                                 ImGui::GetFrameHeight() + 8.0f * s);
        const json::Value index = json::Value::Int(p.index);
        ImGui::SetNextWindowPos(ImVec2(pmin.x + 18.0f * s, pmin.y + 14.0f * s), ImGuiCond_Always);
        if (ImGui::Begin(("##pane" + std::to_string(p.index)).c_str(), nullptr, kOverlayFlags)) {
            // The camera the pane looks through: lit while bound to a scene camera, which the HUD names.
            if (draw::IconButton("##camera", icon::kCameraNode, !camera.empty(), camera.empty() ? Tr("Free Camera") : camera.c_str()))
                ImGui::OpenPopup("##cameras");
            if (ImGui::BeginPopup("##cameras")) {
                if (ImGui::Selectable(Tr("Free Camera"), p.camera.empty()))
                    Queue("view.view", Args({{"index", index}, {"camera", json::Value::Str("")}}));
                for (const std::string& c : ui.scene_cameras)
                    if (ImGui::Selectable(c.c_str(), c == p.camera))
                        Queue("view.view", Args({{"index", index}, {"camera", json::Value::Str(c)}}));
                ImGui::EndPopup();
            }
            ImGui::SameLine(0.0f, 4.0f * s);
            ImGui::SetNextItemWidth(104.0f * s);
            if (ImGui::BeginCombo("##mode", Tr(kViewModes[static_cast<size_t>(p.mode)].label))) {
                for (const ViewModeInfo& m : kViewModes)
                    if (ImGui::Selectable(Tr(m.label), m.mode == p.mode))
                        Queue("view.view", Args({{"index", index}, {"mode", json::Value::Str(m.name)}}));
                ImGui::EndCombo();
            }
        }
        ImGui::End();
        // The active pane's view tools and every pane's maximize sit at its top right.
        ImGui::SetNextWindowPos(ImVec2(pmax.x - 16.0f * s, pmin.y + 14.0f * s), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
        if (ImGui::Begin(("##pane_tools" + std::to_string(p.index)).c_str(), nullptr, kOverlayFlags)) {
            if (active) {
                DrawViewButtons();
                ImGui::SameLine(0.0f, 4.0f * s);
            }
            if (draw::IconButton("##maximize", icon::kMaximize, false, Tr("Maximize View")))
                Queue("view.maximize", Args({{"index", index}}));
        }
        ImGui::End();
    }
}

void ImGuiLayer::SetActiveView(uint32_t index) {
    viewport_ = pane_rects_[index];
    const ImVec2 mouse = ImGui::GetMousePos();
    viewport_hovered_ = pointer_free_ && viewport_.Contains(mouse.x, mouse.y);
}

void ImGuiLayer::DrawGizmo(CameraController& camera, uint32_t vp_w, uint32_t vp_h, ViewerUiState& ui_state,
                           math::Transform& world, bool& changed) {
    changed = false;
    GizmoState& gz = ui_state.gizmo;
    if (!gz.enabled || ui_state.tool != ViewportTool::Select || ui_state.selected_entity == nuka::scene::kInvalidEntity ||
        vp_w == 0u || vp_h == 0u || !viewport_.Valid() || !ui_state.window_focused) {
        gz.active = false;
        gz.using_now = false;
        return;
    }
    float view[16], proj[16], model[16];
    CameraMatrices(camera, view, proj);
    ModelMatrix(world, model);

    ImGuizmo::Enable(viewport_hovered_ || ImGuizmo::IsUsing());
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    list->PushClipRect(ImVec2(viewport_.x, viewport_.y), ImVec2(viewport_.x + viewport_.width, viewport_.y + viewport_.height), true);
    ImGuizmo::SetDrawlist(list);
    ImGuizmo::SetRect(viewport_.x, viewport_.y, viewport_.width, viewport_.height);
    const ImGuizmo::OPERATION op = gz.op == GizmoState::Op::Rotate ? ImGuizmo::ROTATE : ImGuizmo::TRANSLATE;
    const ImGuizmo::MODE mode = gz.local ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
    // ImGuizmo reads snap[0]: world units for translate, degrees for rotate.
    const float step = gz.op == GizmoState::Op::Rotate ? gz.snap_rotate : gz.snap_translate;
    const float snap[3] = {step, step, step};
    ImGuizmo::Manipulate(view, proj, op, mode, model, nullptr, gz.snap_on ? snap : nullptr);
    list->PopClipRect();

    gz.active = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
    gz.using_now = ImGuizmo::IsUsing();
    if (gz.using_now) {
        world = TransformFromModel(model);
        changed = true;
    }
}

}  // namespace nuka::runtime::app::viewer
