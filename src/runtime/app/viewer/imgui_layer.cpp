// The editor shell. Panels are nui trees the host rebuilds only when their key changes; the
// menu bar, toolbar, status bar, palette and viewport HUD draw immediately each frame.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "imgui_internal.h"
#include "ImGuizmo.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/camera_controller.hpp"
#include "scene/ecs/components.hpp"
#include "scene/ecs/registry.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_ir.hpp"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <unordered_map>
#include <utility>

namespace nuka::runtime::app::viewer {

namespace json = nuka::scene::json;
namespace theme = nuka::render::imgui::theme;
namespace icon = nuka::render::imgui::icon;
namespace draw = nuka::nui::draw;
using nuka::nui::TextStyle;
using nuka::nui::Tone;
using nuka::nui::Tr;
using nuka::render::imgui::FontRole;
using nuka::render::imgui::GetNukaFont;

struct StageKind {
    const char* glyph;
    ImVec4 color;
    const char* badge;
};

namespace {

ImU32 U32(const ImVec4& c) { return ImGui::GetColorU32(c); }

template <class... T>
std::string Fmt(const char* format, T... values) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), format, values...);
    return buf;
}

json::Value Args(std::initializer_list<std::pair<const char*, json::Value>> items) {
    json::Value args = json::Value::Object();
    for (const auto& [key, value] : items) args.Set(key, value);
    return args;
}

std::string Lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

std::string Upper(std::string_view text) {
    std::string out(text);
    for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

std::string Iconed(const char* glyph, const char* text) { return std::string(glyph) + "  " + text; }

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
                             {"file", "File", icon::kFile},          {"stats", "Stats", icon::kStats},
                             {"debug", "Debug", icon::kDebug},       {"camera", "Camera", icon::kCamera},
                             {"drive", "Drive", icon::kDrive},       {"script", "Script", icon::kScript},
                             {"console", "Console", icon::kConsole}, {"nui", "UI Inspector", icon::kInspector}};

// Panels per dock column; the first of each column is its selected tab.
struct LayoutPreset {
    const char* name;
    const char* label;
    std::vector<const char*> left, right, bottom;
};
const LayoutPreset kLayouts[] = {
    {"authoring", "Authoring", {"stage", "file"}, {"property", "camera"}, {"console", "script", "stats"}},
    {"debug", "Debug", {"stage"}, {"debug", "stats", "property"}, {"console", "script"}},
    {"robot", "Robot", {"stage"}, {"drive", "property"}, {"console", "stats"}},
    {"data", "Data", {"file", "stage"}, {"stats", "camera", "property"}, {"console", "script"}},
    {"training", "Training", {"stats", "stage"}, {"drive", "camera"}, {"console", "script"}},
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
            {"File", {"file"}, {"file.open", "-", "file.save", "file.save_as", "-", "file.unload"}},
            {"Edit", {"edit"}, {"edit.undo", "edit.redo", "-", "edit.delete"}},
            {"Create", {"scene"}, {"scene.add_box", "scene.add_sphere", "scene.add_capsule", "scene.add_plane"}},
            {"View",
             {"view"},
             {"app.command_palette", "-", "view.frame_all", "view.frame_selected", "view.reset_camera", "-",
              "view.colliders", "view.contacts", "view.hide_overlays", "-", "scene.expand_all", "scene.collapse_all", "-",
              "view.ui_scale_up", "view.ui_scale_down", "view.ui_scale_reset", "view.toggle_language", "-", "view.screenshot"}},
            {"Simulation", {"sim"}, {"sim.toggle_play", "sim.step", "sim.reset"}},
            {"Tools", {"script", "tools"}, {"script.run", "script.attach", "script.load", "script.clear_console", "-", "tools.nui_inspector"}},
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

std::string MeshText(const ImGuiLayer::Frame& f) {
    for (const render::RenderInstance& ri : f.world->instances) {
        if (ri.entity != f.ui->selected_entity) continue;
        if (ri.mesh_id == render::kNoId || ri.mesh_id >= f.world->meshes.Count()) return Tr("none");
        return Tr(f.world->meshes.Source(ri.mesh_id) == render::MeshSource::NkaMesh ? "mesh" : "primitive");
    }
    return Tr("none");
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
        {"nuka.properties", "Properties", "Transform and material of the selection.", {"property"}},
        {"nuka.file", "File", "Open, save and load scenes.", {"file"}},
        {"nuka.stats", "Stats", "Frame and physics timings.", {"stats"}},
        {"nuka.debug", "Debug Draw", "Collider and contact overlays.", {"debug"}},
        {"nuka.camera", "Camera", "Camera pose, framing and controls.", {"camera"}},
        {"nuka.drive", "Drive", "Joint drive targets and keyboard teleop.", {"drive"}},
        {"nuka.script", "Script", "The Python script editor and the console.", {"script", "console"}},
        {"nuka.ui_inspector", "UI Inspector", "The nui component tree, for extension authors.", {"nui"}},
    };
    return builtins;
}

void ImGuiLayer::AddBuiltinPanel(const std::string& panel_id) {
    using Builder = void (ImGuiLayer::*)(nui::VStack&);
    const Builder builders[] = {&ImGuiLayer::BuildStage,  &ImGuiLayer::BuildProperties, &ImGuiLayer::BuildFile,
                                &ImGuiLayer::BuildStats,  &ImGuiLayer::BuildDebug,      &ImGuiLayer::BuildCamera,
                                &ImGuiLayer::BuildDrive,  &ImGuiLayer::BuildScript,     &ImGuiLayer::BuildConsole,
                                &ImGuiLayer::BuildInspector};
    const size_t i = static_cast<size_t>(
        std::find_if(std::begin(kPanels), std::end(kPanels), [&](const PanelSpec& p) { return panel_id == p.id; }) -
        std::begin(kPanels));
    std::lock_guard<std::recursive_mutex> lock(host_.Mutex());
    nui::Panel& panel = host_.AddPanel(kPanels[i].id, kPanels[i].title, kPanels[i].glyph);
    const Builder build = builders[i];
    panel.build = [this, build](nui::VStack& root) { (this->*build)(root); };
    panel.dock = LayoutColumn(*FindLayout(layout_), panel_id);
    panel.open = !panel.dock.empty();
    if (panel_id == "drive")
        panel.key = [this] {
            return static_cast<uint64_t>(frame_.ui->drive_targets.size()) * 7919u + frame_.ui->dof_labels.size();
        };
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
        {"file.open", icon::kOpen},         {"file.unload", icon::kUnload},           {"file.save", icon::kSave},
        {"file.save_as", icon::kSaveAs},    {"edit.undo", icon::kUndo},               {"edit.redo", icon::kRedo},
        {"edit.delete", icon::kTrash},      {"script.run", icon::kRun},               {"script.clear_console", icon::kClear},
        {"script.attach", icon::kScriptNode}, {"script.load", icon::kFile},           {"sim.toggle_play", icon::kPlay},
        {"sim.step", icon::kStep},          {"sim.reset", icon::kReset},              {"view.frame_all", icon::kFrame},
        {"view.frame_selected", icon::kTarget}, {"view.reset_camera", icon::kHome},   {"view.colliders", icon::kCollider},
        {"view.contacts", icon::kAtom},     {"view.hide_overlays", icon::kVisual},    {"view.screenshot", icon::kCamera}};
    for (const auto& [id, glyph] : icons) registry.Decorate(id, glyph, false);
    for (const char* id : {"edit.apply", "edit.rename", "edit.reparent", "scene.spawn", "scene.select", "sim.play",
                           "sim.pause", "sim.speed", "view.stats"})
        registry.Decorate(id, "", true);
    registry.Rebind("file.open", "Ctrl+O");
    registry.Rebind("file.save", "Ctrl+S");
    registry.Rebind("file.save_as", "Ctrl+Shift+S");
    registry.Rebind("edit.delete", "Delete");
    registry.Rebind("sim.toggle_play", "Space");

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
    auto& counts = root.Add<nui::Label>(std::string(), Tone::Dim, TextStyle::Label);
    counts.text_fn = [this] {
        const render::RenderWorld& w = *frame_.world;
        return Fmt(Tr("%u inst   %u mesh   %u mat"), w.InstanceCount(), w.meshes.Count(), w.MaterialCount());
    };
    counts.visible_if = [this] { return frame_.ui->has_scene; };
    auto& empty = root.Add<nui::Label>("No scene loaded. Open one with Ctrl+O.", Tone::Dim);
    empty.wrap = true;
    empty.visible_if = [this] { return !frame_.ui->has_scene; };

    auto& tree = root.Add<nui::TreeView>(stage_items_);
    stage_tree_ = &tree;
    tree.filter = nui::Bind<std::string>([this] { return std::string(stage_filter_); });
    tree.is_selected = [this](uint64_t item) {
        const nuka::scene::EntityId e = stage_items_->Entity(item);
        return e != nuka::scene::kInvalidEntity && e == frame_.ui->selected_entity;
    };
    tree.on_select = [this](uint64_t item) {
        frame_.ui->selected_entity = stage_items_->Entity(item);
        revealed_ = frame_.ui->selected_entity;
    };
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
        if (ImGui::MenuItemEx(Tr("Delete subtree"), icon::kTrash, "Delete"))
            Queue("edit.delete", Args({{"path", json::Value::Str(path)}}));
    };
}

void ImGuiLayer::BuildProperties(nui::VStack& root) {
    const auto selected = [this] { return frame_.ui->selected_entity != nuka::scene::kInvalidEntity; };
    const auto editable = [this, selected] { return selected() && frame_.ui->inspector.valid; };
    InspectorState& ins = frame_.ui->inspector;
    GizmoState* gz = &frame_.ui->gizmo;

    auto& none = root.Add<nui::Label>("Nothing selected. Click a node in the Stage, or Ctrl+click a body in the viewport.",
                                      Tone::Dim);
    none.wrap = true;
    none.visible_if = [selected] { return !selected(); };

    auto& head = root.Add<nui::HStack>();
    head.visible_if = selected;
    auto& name = head.Add<nui::Label>(std::string(), Tone::Normal, TextStyle::Strong);
    name.text_fn = [this] {
        const auto node = SelectedNode(frame_);
        return !node ? std::string() : node->name.empty() ? std::string(Tr("(unnamed)")) : node->name;
    };
    auto& mobility = head.Add<nui::Badge>();
    mobility.text_fn = [this] { return std::string(Tr(frame_.ui->inspector.movable ? "movable" : "fixed")); };
    mobility.tone_fn = [this] { return frame_.ui->inspector.movable ? Tone::Accent : Tone::Dim; };
    mobility.visible_if = editable;
    root.Add<nui::Stat>("Entity", [this] { return Fmt("%u", frame_.ui->selected_entity.index); }).visible_if = selected;
    root.Add<nui::Stat>("Path", [this] { return SelectionPath(frame_); }, Tone::Muted).visible_if = selected;
    root.Add<nui::Stat>("Mesh", [this] { return MeshText(frame_); }, Tone::Muted).visible_if = selected;
    auto& bare = root.Add<nui::Label>("No editable geometry", Tone::Dim);
    bare.visible_if = [this, selected] { return selected() && !frame_.ui->inspector.valid; };

    auto& transform = root.Add<nui::Collapsible>("Transform", true, icon::kMove);
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

    auto& gizmo = root.Add<nui::Collapsible>("Gizmo", false, icon::kGizmo);
    gizmo.visible_if = editable;
    auto& modes = gizmo.Add<nui::HStack>();
    auto& op = modes.Add<nui::Segmented>(std::vector<std::string>{"Move", "Rotate"},
                                         nui::Bind<int>([gz] { return gz->op == GizmoState::Op::Rotate ? 1 : 0; },
                                                        [gz](const int& v) {
                                                            gz->op = v == 1 ? GizmoState::Op::Rotate : GizmoState::Op::Translate;
                                                        }));
    op.icons = {icon::kMove, icon::kRotate};
    modes.Add<nui::ToggleButton>("Local", nui::Ref(&gz->local));
    modes.Add<nui::ToggleButton>("Snap", nui::Ref(&gz->snap_on));
    auto& ggrid = gizmo.Add<nui::PropertyGrid>();
    ggrid.Add<nui::BoolField>("Enabled", nui::Ref(&gz->enabled));
    ggrid.Add<nui::FloatField>("Snap step",
                               nui::Bind<float>(
                                   [gz] { return gz->op == GizmoState::Op::Rotate ? gz->snap_rotate : gz->snap_translate; },
                                   [gz](const float& v) {
                                       (gz->op == GizmoState::Op::Rotate ? gz->snap_rotate : gz->snap_translate) = v;
                                   }),
                               0.01f);

    auto& material = root.Add<nui::Collapsible>("Material", true, icon::kPalette);
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
    save_box.Add<nui::Badge>("unsaved", Tone::Warn, icon::kEdit).visible_if = [this] { return frame_.ui->save_dirty; };
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
    stat("publish", [this] { return std::string(Tr(frame_.stats->zero_copy ? "ZERO-COPY" : "HOST")); }).tone_fn = [this] {
        return frame_.stats->zero_copy ? Tone::Accent : Tone::Warn;
    };
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
    auto& device = root.Add<nui::Label>(std::string(), Tone::Dim, TextStyle::Label);
    device.text_fn = [this] { return frame_.stats->device_name; };
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

void ImGuiLayer::BuildDrive(nui::VStack& root) {
    ViewerUiState& ui = *frame_.ui;
    const size_t n = ui.drive_targets.size();
    root.Add<nui::Label>("No articulation DOFs", Tone::Dim).visible = n == 0u;
    auto& summary = root.Add<nui::Label>(std::string(), Tone::Dim, TextStyle::Label);
    summary.text_fn = [this, n] { return Fmt(Tr("%zu DOF   env %u"), n, frame_.ui->env_index); };
    summary.visible = n != 0u;
    auto& grid = root.Add<nui::PropertyGrid>();
    const bool labelled = ui.dof_labels.size() == n;
    for (size_t d = 0; d < n; ++d) {
        const std::string label = labelled && !ui.dof_labels[d].empty() ? ui.dof_labels[d] : Fmt("dof %zu", d);
        auto& field = grid.Add<nui::FloatField>(label,
                                                nui::Bind<float>([this, d] { return frame_.ui->drive_targets[d]; },
                                                                 [this, d](const float& v) {
                                                                     frame_.ui->drive_targets[d] = v;
                                                                     frame_.ui->drive_dirty[d] = 1u;
                                                                 }),
                                                0.01f, -3.1416f, 3.1416f);
        field.slider = true;
    }

    auto& teleop = root.Add<nui::Collapsible>("Teleop", false, icon::kKeyboard);
    auto& tgrid = teleop.Add<nui::PropertyGrid>();
    tgrid.Add<nui::BoolField>("Keyboard", nui::Ref(&ui.teleop_enabled));
    tgrid.Add<nui::IntField>("DOF", nui::Ref(&ui.teleop_dof), 0, std::max(static_cast<int>(n) - 1, 0));
    tgrid.Add<nui::FloatField>("Step size", nui::Ref(&ui.teleop_step), 0.005f);
    tgrid.Add<nui::BoolField>("PD hold", nui::Ref(&ui.teleop_hold));
    tgrid.Add<nui::FloatField>("kp", nui::Ref(&ui.teleop_kp), 0.5f);
    tgrid.Add<nui::FloatField>("kd", nui::Ref(&ui.teleop_kd), 0.05f);
    teleop.Add<nui::Label>("[ / ] nudge the DOF and - / = pick it. With a policy, W A S D move and Q / E turn.", Tone::Dim)
        .wrap = true;
    auto& cmd = teleop.Add<nui::Label>(std::string(), Tone::Accent, TextStyle::Mono);
    cmd.text_fn = [this] {
        const float* c = frame_.ui->teleop_cmd;
        return Fmt("cmd  vx %.2f  vy %.2f  wyaw %.2f", c[0], c[1], c[2]);
    };
}

void ImGuiLayer::BuildScript(nui::VStack& root) {
    ViewerUiState& ui = *frame_.ui;
    auto& bar = root.Add<nui::HStack>();
    auto& run = bar.Add<nui::Button>("Run", icon::kRun);
    run.primary = true;
    run.Action("script.run");
    bar.Add<nui::Button>("Attach", icon::kScriptNode).Action("script.attach");
    bar.Add<nui::Button>("Load", icon::kFile).Action("script.load");
    bar.Add<nui::Label>(std::string(), Tone::Dim, TextStyle::Label).text_fn = [this] {
        return Fmt(Tr("%d script nodes"), frame_.ui->script_node_count);
    };
    root.Add<nui::Label>(std::string(), Tone::Dim).text_fn = [this] {
        return std::string(Tr(frame_.ui->has_scene ? "Ctrl+Enter runs against the loaded world (nuka.*)"
                                                   : "Load a scene, then nuka.* drives it"));
    };
    auto& code = root.Add<nui::CodeEditor>(ui.script_buf, sizeof(ui.script_buf));
    code.height = -1.0f;
    code.on_run = [this] { Queue("script.run"); };
}

void ImGuiLayer::BuildConsole(nui::VStack& root) {
    root.Add<nui::Custom>([this] {
        const float s = nui::Scale();
        ImGui::SetNextItemWidth(std::max(ImGui::GetContentRegionAvail().x - ImGui::GetFrameHeight() - 8.0f * s, 60.0f * s));
        const std::string hint = Iconed(icon::kFilter, Tr("Filter output"));
        ImGui::InputTextWithHint("##filter", hint.c_str(), console_filter_, sizeof(console_filter_));
        ImGui::SameLine();
        if (draw::IconButton("##clear", icon::kClear, false, Tr("Clear console"))) Queue("script.clear_console");
    });
    auto& log = root.Add<nui::LogView>(&frame_.ui->console_log);
    log.filter = nui::Bind<std::string>([this] { return std::string(console_filter_); });
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
    }
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
    if (central != nullptr) {
        const ImGuiViewport* main = ImGui::GetMainViewport();
        viewport_ = SceneViewportRect::FromLogical(central->Pos.x, central->Pos.y, central->Size.x, central->Size.y,
                                                  main->Pos.x, main->Pos.y, io.DisplayFramebufferScale.x,
                                                  io.DisplayFramebufferScale.y, framebuffer_width, framebuffer_height);
    }
    const ImVec2 mouse = ImGui::GetMousePos();
    viewport_hovered_ = viewport_.Valid() && ui_state.window_focused && !ui_state.layout_modifier_down &&
                        !io.WantCaptureMouse && viewport_.Contains(mouse.x, mouse.y);

    if (viewport_.Valid()) {
        const ImVec2 vmin(viewport_.x, viewport_.y), vmax(viewport_.x + viewport_.width, viewport_.y + viewport_.height);
        if (ui_state.has_scene) {
            draw::ViewportFrame(vmin, vmax,
                                {Tr("PERSPECTIVE"), Fmt(Tr("FOV %.0f\xC2\xB0"), camera.fov_degrees),
                                 Tr(ui_state.playing ? "LIVE" : "PAUSED"),
                                 Fmt(Tr("TICK %llu"), static_cast<unsigned long long>(stats.sim_ticks))});
        } else {
            const float s = nui::Scale();
            ImFont* font = GetNukaFont(FontRole::Display);
            const float px = theme::kFontLabel * s * 1.3f;
            const char* text = Tr("NO SCENE   CTRL+O TO OPEN   CTRL+P FOR ACTIONS");
            const ImVec2 size = font->CalcTextSizeA(px, FLT_MAX, 0.0f, text);
            ImGui::GetBackgroundDrawList()->AddText(font, px,
                                                    ImVec2((vmin.x + vmax.x - size.x) * 0.5f, (vmin.y + vmax.y - size.y) * 0.5f),
                                                    U32(theme::kFgDim), text);
        }
        host_.DrawNotifications(vmin, vmax);
    }
    DrawPalette();
    DrawAbout();
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
    // Focusing a column's first panel selects its tab once the windows have docked; disabled
    // built-in extensions leave gaps.
    const auto place = [this](const std::vector<const char*>& ids, ImGuiID node) {
        nui::Panel* first = nullptr;
        for (const char* id : ids) {
            nui::Panel* panel = host_.FindPanel(id);
            if (panel == nullptr) continue;
            panel->open = true;
            if (first == nullptr) first = panel;
            ImGui::DockBuilderDockWindow((std::string("###") + id).c_str(), node);
        }
        if (first != nullptr) first->focus_request = true;
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
    if (!ImGui::BeginMainMenuBar()) return;
    const float s = nui::Scale();
    render::imgui::DrawNukaLogo(ImGui::GetFrameHeight() - 8.0f * s);
    ImGui::SameLine(0.0f, 6.0f * s);
    nui::PushTextStyle(TextStyle::Label);
    ImGui::TextColored(theme::kAccent, "NUKA");
    nui::PopTextStyle();
    ImGui::SameLine(0.0f, 14.0f * s);

    for (const MenuSpec& menu : Menus()) {
        const std::string title = Tr(menu.title) + "###" + menu.title;
        if (!ImGui::BeginMenu(title.c_str())) continue;
        bool any = false, separator = false;
        for (const std::string& id : menu.items) {
            if (id == "-") {
                separator = any;
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

    // The search chip shrinks to its icon when the menus leave no room for the text.
    const char* keys = "CTRL+P";
    ImFont* label_font = GetNukaFont(FontRole::Display);
    const float label_px = theme::kFontLabel * s;
    const ImVec2 ksize = label_font->CalcTextSizeA(label_px, FLT_MAX, 0.0f, keys);
    const std::string full = Iconed(icon::kSearch, Tr("Search actions"));
    const float room = ImGui::GetWindowWidth() - ImGui::GetCursorPosX() - 18.0f * s;
    const bool compact = ImGui::CalcTextSize(full.c_str()).x + ksize.x + 40.0f * s > room;
    const std::string text = compact ? std::string(icon::kSearch) : full;
    const ImVec2 tsize = ImGui::CalcTextSize(text.c_str());
    const float w = compact ? tsize.x + 20.0f * s : tsize.x + ksize.x + 40.0f * s, h = ImGui::GetFrameHeight() - 6.0f * s;
    ImGui::SameLine(ImGui::GetWindowWidth() - w - 10.0f * s);
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const ImVec2 a(at.x, at.y + (ImGui::GetFrameHeight() - h) * 0.5f), b(a.x + w, a.y + h);
    ImGui::SetCursorScreenPos(a);
    if (ImGui::InvisibleButton("##search", ImVec2(w, h))) Queue("app.command_palette");
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* list = ImGui::GetWindowDrawList();
    draw::Chamfer(list, a, b, 4.0f * s, U32(hovered ? theme::kBg2 : theme::kBg1), U32(hovered ? theme::kLineHi : theme::kLine));
    list->AddText(ImVec2(a.x + 10.0f * s, (a.y + b.y - tsize.y) * 0.5f), U32(theme::kFgDim), text.c_str());
    if (!compact)
        list->AddText(label_font, label_px, ImVec2(b.x - ksize.x - 10.0f * s, (a.y + b.y - ksize.y) * 0.5f), U32(theme::kFgDim),
                      keys);
    ImGui::EndMainMenuBar();
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
        if (toggle("##snap", icon::kSnap, gz.snap_on, Tr("Snap"))) gz.snap_on = !gz.snap_on;
        ImGui::EndDisabled();
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
        const auto chip = [&](const std::string& text, const ImVec4& color, const char* glyph, bool dot, bool pulse) {
            const ImVec2 ts = label_font->CalcTextSizeA(label_px, FLT_MAX, 0.0f, text.c_str());
            const float lead = glyph != nullptr || dot ? 16.0f * s : 0.0f;
            const float cw = ts.x + lead + 16.0f * s, ch = 22.0f * s;
            if (x - cw < limit) return;
            const ImVec2 c0(x - cw, cy - ch * 0.5f), c1(x, cy + ch * 0.5f);
            draw::Chamfer(list, c0, c1, 4.0f * s, U32(theme::Alpha(color, 0.10f)), U32(theme::Alpha(color, 0.38f)));
            const ImVec2 mark(c0.x + 14.0f * s, cy);
            if (glyph != nullptr) draw::Glyph(list, mark, glyph, U32(color), false, 13.0f);
            if (pulse) {
                const float t = static_cast<float>(std::fmod(ImGui::GetTime(), 1.6) / 1.6);
                list->AddCircleFilled(mark, (3.0f + 4.0f * t) * s, U32(theme::Alpha(color, 0.35f * (1.0f - t))));
            }
            if (dot) list->AddCircleFilled(mark, 3.0f * s, U32(color));
            list->AddText(label_font, label_px, ImVec2(c0.x + 8.0f * s + lead, cy - ts.y * 0.5f), U32(color), text.c_str());
            x = c0.x - 6.0f * s;
        };
        chip(Tr(!ui.has_scene ? "NO SCENE" : ui.playing ? "LIVE" : "PAUSED"),
             !ui.has_scene ? theme::kFgDim : ui.playing ? theme::kOk : theme::kFgMuted, nullptr, true, ui.playing);
        if (ui.has_scene) {
            chip(Fmt(Tr("T %llu"), static_cast<unsigned long long>(stats.sim_ticks)), theme::kFgMuted, icon::kTimer, false, false);
            chip(Tr(stats.zero_copy ? "ZERO-COPY" : "HOST COPY"), stats.zero_copy ? theme::kAccent : theme::kWarn,
                 stats.zero_copy ? icon::kZeroCopy : icon::kHostCopy, false, false);
            if (stats.capped_batches != 0u)
                chip(Fmt(Tr("CAP \xC3\x97%llu"), static_cast<unsigned long long>(stats.capped_batches)), theme::kWarn, icon::kWarn,
                     false, false);
            if (ui.save_dirty) chip(Tr("UNSAVED"), theme::kWarn, icon::kEdit, false, false);
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
        const std::string path = SelectionPath(frame_);
        const std::string left = Iconed(icon::kTarget, path.empty() ? Tr("NO SELECTION") : Upper(path).c_str()) + "      " +
                                 Iconed(icon::kLayout, Upper(Tr(FindLayout(layout_)->label)).c_str()) + "      " +
                                 Fmt(Tr("UI %.0f%%"), static_cast<double>(ui_scale_) * 100.0);
        const ImVec2 lsize = font->CalcTextSizeA(px, FLT_MAX, 0.0f, left.c_str());
        list->AddText(font, px, ImVec2(wpos.x + 10.0f * s, cy - lsize.y * 0.5f), U32(theme::kFgDim), left.c_str());
        const std::string right = Fmt("%.0f FPS   %.2f MS   ", stats.fps, stats.frame_time_ms) + Upper(stats.device_name);
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

void ImGuiLayer::DrawGizmo(CameraController& camera, uint32_t vp_w, uint32_t vp_h, ViewerUiState& ui_state,
                           math::Transform& world, bool& changed) {
    changed = false;
    GizmoState& gz = ui_state.gizmo;
    if (!gz.enabled || ui_state.selected_entity == nuka::scene::kInvalidEntity || vp_w == 0u || vp_h == 0u ||
        !viewport_.Valid() || !ui_state.window_focused) {
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
