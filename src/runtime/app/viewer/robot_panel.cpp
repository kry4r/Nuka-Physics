// The Joints panel: each moving joint's limits, value, drive target and effort, the selected
// joint's curves, drive conversion, the fixed base, the gizmo's IK and teleop; the
// Poses panel, the Policy panel, the Gain Tuner and System Identification.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/robot_tools.hpp"
#include "runtime/app/viewer/shell_util.hpp"
#include "runtime/app/viewer/teleop.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

namespace json = nuka::scene::json;
namespace theme = nuka::render::imgui::theme;
namespace icon = nuka::render::imgui::icon;
using nuka::nui::TextStyle;
using nuka::nui::Tone;
using nuka::nui::Tr;
using namespace shell;

namespace {

constexpr float kDegrees = 57.2957795f;
constexpr float kPi = 3.14159265f;
// The curves show this much of the newest simulated time.
constexpr double kCurveSeconds = 5.0;
constexpr int kTableRows = 12;

float At(const std::vector<float>& v, uint32_t i) { return i < v.size() ? v[i] : 0.0f; }

// What a joint's bar spans: its limits, else half a turn either way or a meter.
void Range(const RobotJoint& j, float* lo, float* hi) {
    if (j.limited && j.upper > j.lower) {
        *lo = j.lower;
        *hi = j.upper;
        return;
    }
    *hi = j.prismatic ? 1.0f : kPi;
    *lo = -*hi;
}

// The track from the lower to the upper limit across [a, b], the value as a fill from zero and
// the target as a caret; the value turns amber within 2% of a limit.
void LimitBar(const RobotJoint& j, float q, float target, ImVec2 a, ImVec2 b) {
    const float s = nui::Scale();
    const float w = b.x - a.x;
    float lo = 0.0f, hi = 0.0f;
    Range(j, &lo, &hi);
    const auto x = [&](float v) { return a.x + std::clamp((v - lo) / (hi - lo), 0.0f, 1.0f) * w; };
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float mid = a.y + (b.y - a.y) * 0.4f, y0 = mid - 2.5f * s, y1 = mid + 2.5f * s;
    const bool near = j.limited && (q - lo < 0.02f * (hi - lo) || hi - q < 0.02f * (hi - lo));
    const ImVec4 tone = near ? theme::kWarn : theme::kAccent;
    dl->AddRectFilled(ImVec2(a.x, y0), ImVec2(b.x, y1), U32(theme::kBg3), 2.0f * s);
    const float zero = x(std::clamp(0.0f, lo, hi)), at = x(q);
    dl->AddRectFilled(ImVec2(std::min(zero, at), y0), ImVec2(std::max(zero, at), y1), U32(theme::Alpha(tone, 0.45f)));
    dl->AddLine(ImVec2(at, y0 - 3.0f * s), ImVec2(at, y1 + 3.0f * s), U32(tone), 2.0f * s);
    if (j.limited)
        for (const float edge : {a.x, b.x}) dl->AddLine(ImVec2(edge, y0 - 2.0f * s), ImVec2(edge, y1 + 2.0f * s), U32(theme::kLineHi), s);
    const float tx = x(target), ty = y1 + 2.0f * s;
    dl->AddTriangleFilled(ImVec2(tx, ty), ImVec2(tx - 4.0f * s, ty + 5.0f * s), ImVec2(tx + 4.0f * s, ty + 5.0f * s),
                          U32(theme::kInfo));
}

ImPlotSpec Line(const ImVec4& color, float weight) {
    ImPlotSpec spec;
    spec.LineColor = color;
    spec.LineWeight = weight * nui::Scale();
    return spec;
}

// A figure, or a dash where it does not apply.
std::string Figure(const char* format, float v) {
    return std::isfinite(v) ? Fmt(format, static_cast<double>(v)) : std::string("\xE2\x80\x94");
}

// A label cell, then the value cell, of the tuner's option table.
void TunerLabel(const char* label) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(theme::kFgMuted, "%s", Tr(label));
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-FLT_MIN);
}

json::Value Floats(const float* v) {
    json::Value a = json::Value::Array();
    for (int i = 0; i < 3; ++i) a.PushBack(json::Value::Float(v[i]));
    return a;
}

constexpr const char* kAttachJoints[] = {"fixed", "revolute", "prismatic"};

// The built-in teleop modes' names, in kTeleopModes' order.
constexpr const char* kTeleopModeLabels[] = {"Velocity command", "Joint rate", "IK goal"};

std::string Leaf(const std::string& path) { return path.substr(path.rfind('/') + 1u); }

// An axis in [-1, 1] as a fill from the middle of [a, b].
void AxisBar(float value, ImVec2 a, ImVec2 b) {
    const float s = nui::Scale(), mid = (a.x + b.x) * 0.5f;
    const float at = mid + std::clamp(value, -1.0f, 1.0f) * (b.x - a.x) * 0.5f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(a, b, U32(theme::kBg3), 2.0f * s);
    dl->AddRectFilled(ImVec2(std::min(mid, at), a.y), ImVec2(std::max(mid, at), b.y), U32(theme::Alpha(theme::kAccent, 0.6f)),
                      2.0f * s);
    dl->AddLine(ImVec2(mid, a.y - s), ImVec2(mid, b.y + s), U32(theme::kLineHi), s);
}

const TeleopDeviceInfo* ActiveDevice(const TeleopView& t) {
    const auto it = std::find_if(t.devices.begin(), t.devices.end(), [&](const TeleopDeviceInfo& d) { return d.id == t.device; });
    return it != t.devices.end() ? &*it : nullptr;
}

}  // namespace

void ImGuiLayer::BuildJoints(nui::VStack& root) {
    ViewerUiState& ui = *frame_.ui;
    const auto robot = [this]() -> RobotView& { return frame_.ui->robot; };

    auto& hero = root.Add<nui::HStack>();
    hero.wrap = true;
    auto& count = hero.Add<nui::Label>(std::string(), Tone::Accent, TextStyle::Hero);
    count.text_fn = [robot] { return Fmt(Tr("%zu JOINTS"), robot().joints.size()); };
    auto& base = hero.Add<nui::Badge>();
    base.text_fn = [robot] { return std::string(Tr(robot().floating ? "floating base" : "fixed base")); };
    base.tone_fn = [robot] { return robot().floating ? Tone::Info : Tone::Dim; };
    auto& mass = hero.Add<nui::Badge>();
    mass.text_fn = [robot] {
        float kg = 0.0f;
        for (const RobotLink& l : robot().links) kg += l.mass;
        return Fmt("%.2f kg", static_cast<double>(kg));
    };
    mass.tone_fn = [] { return Tone::Dim; };
    hero.Add<nui::Spacer>();
    auto& overlays = hero.Add<nui::IconButton>(icon::kOverlays, "Robot Overlays");
    overlays.on_click = [this, robot] { Queue("view.robot_overlays", Args({{"on", json::Value::Bool(!robot().overlays.on)}})); };
    overlays.checked_fn = [robot] { return robot().overlays.on; };
    auto& tuner = hero.Add<nui::IconButton>(icon::kWaves, "Gain Tuner").Action("robot.gain_tuner");
    tuner.checked_fn = [robot] { return robot().tuner.open; };
    auto& sysid = hero.Add<nui::IconButton>(icon::kMagicWand, "System Identification").Action("sysid.panel");
    sysid.checked_fn = [robot] { return robot().sysid.open; };
    auto& assembler = hero.Add<nui::IconButton>(icon::kPlug, "Robot Assembler").Action("robot.assembler");
    assembler.checked_fn = [robot] { return robot().assembler.open; };
    auto& watch = hero.Add<nui::IconButton>(icon::kPlot, "Watch Joints").Action("robot.watch");
    watch.checked_fn = [robot] { return robot().watch; };
    hero.visible_if = [robot] { return !robot().joints.empty(); };
    root.Add<nui::Label>("No moving joints", Tone::Dim).visible_if = [robot] { return robot().joints.empty(); };

    auto& tools = root.Add<nui::HStack>();
    tools.wrap = true;
    tools.Add<nui::ToggleButton>("Fixed Base").Action("robot.fix_base");
    auto& convert = tools.Add<nui::Button>("Position Drive", icon::kDrive).Action("robot.drive");
    convert.tooltip = "Makes every joint a position drive with the gain below, in one undoable edit.";
    auto& reset = tools.Add<nui::Button>("Reset", icon::kReset).Action("robot.reset_targets");
    reset.tooltip = "Puts every joint's target back at its initial position.";
    tools.Add<nui::ToggleButton>("Radians").Action("robot.radians");
    tools.visible_if = [robot] { return !robot().joints.empty(); };
    auto& grid = root.Add<nui::PropertyGrid>();
    auto& kp = grid.Add<nui::FloatField>("Drive kp", nui::Ref(&ui.robot.drive_kp), 1.0f, 0.0f, 100000.0f);
    kp.unit = "N\xC2\xB7m/rad";
    kp.format = "%.1f";
    grid.Add<nui::StringField>("Filter", nui::Ref(&ui.robot.filter), "joint name");
    grid.visible_if = [robot] { return !robot().joints.empty(); };
    auto& hint = root.Add<nui::Label>("Targets move only position drives: convert the joints with Position Drive, or "
                                      "hold them with PD hold under Teleop.",
                                      Tone::Warn);
    hint.wrap = true;
    hint.visible_if = [robot] {
        const std::vector<RobotJoint>& joints = robot().joints;
        return std::any_of(joints.begin(), joints.end(), [](const RobotJoint& j) { return j.drive != "position"; });
    };

    // One row per joint: the name, value and effort over a limit bar; dragging the bar sets the
    // target, clicking the row selects the link.
    root.Add<nui::Custom>([this] {
        ViewerUiState& u = *frame_.ui;
        RobotView& v = u.robot;
        v.dragging = -1;
        if (v.joints.empty() || u.drive_targets.size() != u.drive_dirty.size()) return;
        const float s = nui::Scale();
        const std::string needle = Lower(v.filter);
        const float line = ImGui::GetTextLineHeight(), pad = 2.0f * theme::kSpace * s, bar = 14.0f * s;
        const float row = pad + line + bar, gap = ImGui::GetStyle().ItemSpacing.y;
        const int rows = std::min(static_cast<int>(v.joints.size()), kTableRows);
        if (ImGui::BeginChild("##joints", ImVec2(0.0f, static_cast<float>(rows) * (row + gap)))) {
            for (size_t i = 0; i < v.joints.size(); ++i) {
                const RobotJoint& j = v.joints[i];
                if (!needle.empty() && Lower(j.name).find(needle) == std::string::npos) continue;
                const float q = At(v.live.q, j.link);
                float& target = u.drive_targets[j.link];
                ImGui::PushID(static_cast<int>(i));
                const ImVec2 p = ImGui::GetCursorScreenPos();
                const float w = ImGui::GetContentRegionAvail().x;
                const bool focused = v.focus == static_cast<int>(i);
                if (ImGui::Selectable("##row", focused, ImGuiSelectableFlags_AllowOverlap, ImVec2(w, row)))
                    Queue("scene.select", Args({{"path", json::Value::Str(j.path)}}));
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    const std::string drive =
                        j.drive.empty()         ? std::string(Tr("no actuator"))
                        : j.drive == "position" ? Fmt(Tr("position drive, kp %.1f"), static_cast<double>(j.kp))
                                                : Fmt(Tr("%s actuator"), Tr(j.drive.c_str()));
                    const std::string limit = j.force_limit > 0.0f ? Fmt(Tr("limit %.1f"), static_cast<double>(j.force_limit))
                                                                   : std::string(Tr("no force limit"));
                    const char* velocity = j.prismatic ? Tr("velocity %+.3f m/s")
                                           : v.radians ? Tr("velocity %+.3f rad/s")
                                                       : Tr("velocity %+.1f\xC2\xB0/s");
                    const std::string speed = Fmt(velocity, static_cast<double>(JointShown(j, At(v.live.qd, j.link), v.radians)));
                    ImGui::SetTooltip("%s\n%s  \xC2\xB7  %s\n%s", j.path.c_str(), drive.c_str(), limit.c_str(), speed.c_str());
                }
                ImDrawList* dl = ImGui::GetWindowDrawList();
                const float ty = p.y + pad * 0.5f, effort = At(v.live.effort, j.link);
                const std::string force = Fmt(j.prismatic ? "%.1f N" : "%.1f N\xC2\xB7m", static_cast<double>(effort));
                const std::string value = JointText(j, q, v.radians);
                const float fx = p.x + w - pad - ImGui::CalcTextSize(force.c_str()).x;
                const float vx = fx - 2.0f * pad - ImGui::CalcTextSize(value.c_str()).x;
                dl->AddText(ImVec2(fx, ty), U32(At(v.live.saturated, j.link) != 0.0f ? theme::kWarn : theme::kFgDim),
                            force.c_str());
                dl->AddText(ImVec2(vx, ty), U32(theme::kFg), value.c_str());
                const std::string name = Iconed(j.prismatic ? icon::kMove : icon::kRotate, j.name.c_str());
                dl->PushClipRect(p, ImVec2(vx - pad, p.y + row), true);
                dl->AddText(ImVec2(p.x + pad, ty), U32(focused ? theme::kAccent : theme::kFgMuted), name.c_str());
                dl->PopClipRect();
                const ImVec2 a(p.x + pad, ty + line), b(p.x + w - pad, p.y + row);
                ImGui::SetCursorScreenPos(a);
                ImGui::InvisibleButton("##target", ImVec2(b.x - a.x, b.y - a.y));
                if (ImGui::IsItemActive()) {
                    float lo = 0.0f, hi = 0.0f;
                    Range(j, &lo, &hi);
                    target = lo + std::clamp((ImGui::GetIO().MousePos.x - a.x) / (b.x - a.x), 0.0f, 1.0f) * (hi - lo);
                    u.drive_dirty[j.link] = 1u;
                    v.dragging = static_cast<int>(i);
                }
                if (ImGui::IsItemHovered() || ImGui::IsItemActive())
                    ImGui::SetTooltip(Tr("target %s"), JointText(j, target, v.radians).c_str());
                LimitBar(j, q, target, a, b);
                ImGui::PopID();
            }
        }
        ImGui::EndChild();
    });

    // The selected joint's target, value and effort over the newest seconds of the watch.
    root.Add<nui::Custom>([this] {
        RobotView& v = frame_.ui->robot;
        if (v.focus < 0 || v.focus >= static_cast<int>(v.joints.size())) {
            v.plotted = -1;
            return;
        }
        if (v.plotted != v.focus && !v.watch) Queue("robot.watch", Args({{"on", json::Value::Bool(true)}}));
        v.plotted = v.focus;
        const RobotJoint& j = v.joints[static_cast<size_t>(v.focus)];
        std::vector<float> t, q, target, effort;
        const double end = v.history.empty() ? 0.0 : v.history.back().time;
        for (const JointSample& sample : v.history) {
            if (sample.time < end - kCurveSeconds || j.link >= sample.q.size()) continue;
            t.push_back(static_cast<float>(sample.time));
            q.push_back(JointShown(j, sample.q[j.link], v.radians));
            target.push_back(JointShown(j, sample.target[j.link], v.radians));
            effort.push_back(sample.effort[j.link]);
        }
        nui::draw::SectionHeader(j.name.c_str());
        if (t.empty()) {
            ImGui::TextDisabled("%s", Tr("Play or step the simulation to record this joint."));
            return;
        }
        const float s = nui::Scale();
        ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(theme::kSpace * s, theme::kSpace * s));
        if (ImPlot::BeginPlot("##joint_curve", ImVec2(-FLT_MIN, 170.0f * s),
                              ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoFrame)) {
            ImPlot::SetupAxes(Tr("s"), j.prismatic ? "m" : v.radians ? "rad" : "\xC2\xB0", ImPlotAxisFlags_AutoFit,
                              ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxis(ImAxis_Y2, j.prismatic ? "N" : "N\xC2\xB7m", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
            ImPlot::SetupLegend(ImPlotLocation_North,
                                ImPlotLegendFlags_NoButtons | ImPlotLegendFlags_Outside | ImPlotLegendFlags_Horizontal);
            const int n = static_cast<int>(t.size());
            ImPlot::PlotLine(Tr("target"), t.data(), target.data(), n, Line(theme::kInfo, 1.5f));
            ImPlot::PlotLine(Tr("actual"), t.data(), q.data(), n, Line(theme::kAccent, 1.5f));
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine(Tr("effort"), t.data(), effort.data(), n, Line(theme::Alpha(theme::kWarn, 0.5f), 1.0f));
            ImPlot::EndPlot();
        }
        ImPlot::PopStyleVar();
    });

    // How the gizmo poses the robot: the chain root, the links a drag pins and the orientation lock.
    auto& ik = root.Add<nui::Collapsible>("IK", false, icon::kJoint);
    auto& igrid = ik.Add<nui::PropertyGrid>();
    igrid.Add<nui::BoolField>("Gizmo drag", nui::Ref(&ui.robot.ik.on));
    igrid.Add<nui::BoolField>("Lock orientation", nui::Ref(&ui.robot.ik.lock_orientation));
    auto& weight = igrid.Add<nui::FloatField>("Rotation weight", nui::Ref(&ui.robot.ik.rotation_weight), 0.01f, 0.0f, 10.0f);
    weight.unit = "m/rad";
    weight.format = "%.2f";
    ik.Add<nui::Custom>([this] {
        const RobotView& v = frame_.ui->robot;
        const IkState& k = v.ik;
        const auto leaf = [](const std::string& path) { return path.substr(path.rfind('/') + 1u); };
        const auto set_pins = [this](const std::vector<std::string>& pins) {
            json::Value list = json::Value::Array();
            for (const std::string& p : pins) list.PushBack(json::Value::Str(p));
            Queue("robot.ik_settings", Args({{"pins", std::move(list)}}));
        };
        ImGui::TextDisabled("%s", Tr("Chain root"));
        ImGui::SetNextItemWidth(-FLT_MIN);
        const std::string root_name = k.root.empty() ? std::string(Tr("Articulation root")) : leaf(k.root);
        if (ImGui::BeginCombo("##ik_root", root_name.c_str())) {
            if (ImGui::Selectable(Tr("Articulation root"), k.root.empty()))
                Queue("robot.ik_settings", Args({{"root", json::Value::Str(std::string())}}));
            for (size_t i = 0; i < v.links.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::Selectable(leaf(v.links[i].path).c_str(), v.links[i].path == k.root))
                    Queue("robot.ik_settings", Args({{"root", json::Value::Str(v.links[i].path)}}));
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        ImGui::TextDisabled("%s", Tr("Pinned links"));
        for (size_t i = 0; i < k.pins.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            if (nui::draw::IconButton("##unpin", icon::kTrash, false, Tr("Unpin"))) {
                std::vector<std::string> pins = k.pins;
                pins.erase(pins.begin() + static_cast<std::ptrdiff_t>(i));
                set_pins(pins);
            }
            ImGui::SameLine();
            ImGui::TextUnformatted(leaf(k.pins[i]).c_str());
            ImGui::PopID();
        }
        if (k.pins.empty()) ImGui::TextDisabled("%s", Tr("None; a drag moves the whole chain."));
        const scene::SceneIR* sc = frame_.scene;
        const scene::EntityId e = frame_.ui->selected_entity;
        const std::string selected =
            sc == nullptr || e == scene::kInvalidEntity ? std::string() : sc->Tree().PathOf(sc->Ecs().NodeOf(e));
        const bool linked = std::any_of(v.links.begin(), v.links.end(), [&](const RobotLink& l) { return l.path == selected; });
        const bool pinned = std::find(k.pins.begin(), k.pins.end(), selected) != k.pins.end();
        ImGui::BeginDisabled(!linked || pinned);
        if (ImGui::Button(Iconed(icon::kPin, Tr("Pin Selected")).c_str())) {
            std::vector<std::string> pins = k.pins;
            pins.push_back(selected);
            set_pins(pins);
        }
        ImGui::EndDisabled();
    });

    // Which device drives which mode, the device's live reading, the mode's mapping and PD hold.
    const auto tele = [this]() -> TeleopView& { return frame_.ui->teleop; };
    auto& teleop = root.Add<nui::Collapsible>("Teleop", true, icon::kGamepad);
    auto& thead = teleop.Add<nui::HStack>();
    thead.wrap = true;
    thead.Add<nui::ToggleButton>("Enabled").Action("teleop.use");
    auto& link_state = thead.Add<nui::Badge>();
    link_state.text_fn = [tele] { return std::string(Tr(!tele().connected ? "disconnected" : tele().on ? "driving" : "idle")); };
    link_state.tone_fn = [tele] { return !tele().connected ? Tone::Warn : tele().on ? Tone::Accent : Tone::Dim; };
    teleop.Add<nui::Custom>([this] {
        const TeleopView& t = frame_.ui->teleop;
        const RobotView& v = frame_.ui->robot;
        const auto use = [this](const char* key, json::Value value) { Queue("teleop.use", Args({{key, std::move(value)}})); };
        const TeleopDeviceInfo* device = ActiveDevice(t);
        ImGui::TextDisabled("%s", Tr("Device"));
        ImGui::SetNextItemWidth(-FLT_MIN);
        const std::string current = device != nullptr ? std::string(Tr(device->label.c_str())) : Fmt(Tr("%s (gone)"), t.device.c_str());
        if (ImGui::BeginCombo("##teleop_device", current.c_str())) {
            for (const TeleopDeviceInfo& d : t.devices) {
                ImGui::PushID(d.id.c_str());
                if (ImGui::Selectable(Tr(d.label.c_str()), d.id == t.device)) use("device", json::Value::Str(d.id));
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                    ImGui::SetTooltip("%s  \xC2\xB7  %s%s%s", d.id.c_str(), d.kind.c_str(), d.owner.empty() ? "" : "  \xC2\xB7  ",
                                      d.owner.c_str());
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        const auto mode_label = [&](const std::string& id) -> std::string {
            const auto m = std::find(kTeleopModes.begin(), kTeleopModes.end(), id);
            if (m != kTeleopModes.end()) return Tr(kTeleopModeLabels[m - kTeleopModes.begin()]);
            const auto r = std::find_if(t.retargets.begin(), t.retargets.end(), [&](const TeleopRetarget& x) { return x.id == id; });
            return r != t.retargets.end() ? r->label : id;
        };
        ImGui::TextDisabled("%s", Tr("Mode"));
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::BeginCombo("##teleop_mode", mode_label(t.mode).c_str())) {
            std::vector<std::string> modes = kTeleopModes;
            for (const TeleopRetarget& r : t.retargets) modes.push_back(r.id);
            for (const std::string& m : modes) {
                ImGui::PushID(m.c_str());
                if (ImGui::Selectable(mode_label(m).c_str(), m == t.mode)) use("mode", json::Value::Str(m));
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        if (t.mode == "joint" && !v.joints.empty()) {
            const int j = std::clamp(t.joint, 0, static_cast<int>(v.joints.size()) - 1);
            ImGui::TextDisabled("%s", Tr("Joint"));
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (ImGui::BeginCombo("##teleop_joint", v.joints[static_cast<size_t>(j)].name.c_str())) {
                for (size_t i = 0; i < v.joints.size(); ++i) {
                    ImGui::PushID(static_cast<int>(i));
                    if (ImGui::Selectable(v.joints[i].name.c_str(), static_cast<int>(i) == j))
                        use("joint", json::Value::Str(v.joints[i].name));
                    ImGui::PopID();
                }
                ImGui::EndCombo();
            }
        }
        if (t.mode == "ik") {
            ImGui::TextDisabled("%s", Tr("Link"));
            ImGui::SetNextItemWidth(-FLT_MIN);
            const std::string link = t.link.empty() ? std::string(Tr("Selected link")) : Leaf(t.link);
            if (ImGui::BeginCombo("##teleop_link", link.c_str())) {
                if (ImGui::Selectable(Tr("Selected link"), t.link.empty())) use("link", json::Value::Str(std::string()));
                for (size_t i = 0; i < v.links.size(); ++i) {
                    if (v.links[i].parent < 0) continue;
                    ImGui::PushID(static_cast<int>(i));
                    if (ImGui::Selectable(Leaf(v.links[i].path).c_str(), v.links[i].path == t.link))
                        use("link", json::Value::Str(v.links[i].path));
                    ImGui::PopID();
                }
                ImGui::EndCombo();
            }
        }
        if (!t.error.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(theme::kWarn, "%s", t.error.c_str());
            ImGui::PopTextWrapPos();
        }
    });
    auto& out = teleop.Add<nui::Label>(std::string(), Tone::Accent, TextStyle::Mono);
    out.wrap = true;
    out.text_fn = [tele] {
        const TeleopView& t = tele();
        std::string line;
        for (const auto& [name, value] : t.quantities) line += Fmt("%s %+.2f  ", name.c_str(), static_cast<double>(value));
        if (t.mode == "command")
            line += Fmt("\n\xE2\x86\x92 vx %+.2f  vy %+.2f  wyaw %+.2f", static_cast<double>(t.command[0]),
                        static_cast<double>(t.command[1]), static_cast<double>(t.command[2]));
        if (t.mode == "ik" && t.goal_set)
            line += Fmt("\n\xE2\x86\x92 %.3f %.3f %.3f m", static_cast<double>(t.goal[0]), static_cast<double>(t.goal[1]),
                        static_cast<double>(t.goal[2]));
        return line;
    };
    auto& no_policy = teleop.Add<nui::Label>("No policy takes the command; load one, or pick the joint or IK mode.", Tone::Warn);
    no_policy.wrap = true;
    no_policy.visible_if = [tele] { return tele().on && tele().mode == "command" && !tele().controller; };

    // The active device's axes as bars from the middle, and its buttons as chips lit while down.
    teleop.Add<nui::Custom>([this] {
        const TeleopView& t = frame_.ui->teleop;
        const TeleopDeviceInfo* d = ActiveDevice(t);
        if (d == nullptr || !t.connected) {
            ImGui::TextDisabled("%s", Tr("The device is not connected."));
            return;
        }
        const float s = nui::Scale(), line = ImGui::GetTextLineHeight(), pad = theme::kSpace * s;
        float name_w = 0.0f;
        for (const std::string& a : d->axes) name_w = std::max(name_w, ImGui::CalcTextSize(a.c_str()).x);
        const float value_w = ImGui::CalcTextSize("+0.00").x;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (size_t i = 0; i < std::min(d->axes.size(), t.axes.size()); ++i) {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float w = ImGui::GetContentRegionAvail().x;
            ImGui::Dummy(ImVec2(w, line));
            dl->AddText(p, U32(theme::kFgMuted), d->axes[i].c_str());
            const std::string value = Fmt("%+.2f", static_cast<double>(t.axes[i]));
            dl->AddText(ImVec2(p.x + w - value_w, p.y), U32(theme::kFg), value.c_str());
            AxisBar(t.axes[i], ImVec2(p.x + name_w + pad, p.y + line * 0.3f), ImVec2(p.x + w - value_w - pad, p.y + line * 0.7f));
        }
        const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
        for (size_t i = 0; i < std::min(d->buttons.size(), t.buttons.size()); ++i) {
            const ImVec2 chip(ImGui::CalcTextSize(d->buttons[i].c_str()).x + 2.0f * pad, line + pad);
            if (i > 0) {
                ImGui::SameLine(0.0f, 0.5f * pad);
                if (ImGui::GetCursorScreenPos().x + chip.x > right) ImGui::NewLine();
            }
            const ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::Dummy(chip);
            const bool down = t.buttons[i] != 0u;
            dl->AddRectFilled(p, ImVec2(p.x + chip.x, p.y + chip.y), U32(down ? theme::Alpha(theme::kAccent, 0.55f) : theme::kBg3),
                              3.0f * s);
            dl->AddText(ImVec2(p.x + pad, p.y + 0.5f * pad), U32(down ? theme::kFg : theme::kFgDim), d->buttons[i].c_str());
        }
    });

    // The mode's bindings, each one input scaled into a quantity; edits act from the next frame.
    teleop.Add<nui::Custom>([this] {
        TeleopView& t = frame_.ui->teleop;
        std::vector<TeleopBinding>& bindings = t.maps[t.mode];
        std::vector<std::string> inputs;
        if (const TeleopDeviceInfo* d = ActiveDevice(t)) {
            inputs = d->axes;
            inputs.insert(inputs.end(), d->buttons.begin(), d->buttons.end());
        }
        const std::vector<std::string>& names = ModeQuantities(t.mode);
        const float s = nui::Scale();
        nui::draw::SectionHeader(Tr("Mapping"));
        int remove = -1;
        // The input and its quantity stack in one cell, which keeps both names readable in a narrow panel.
        if (ImGui::BeginTable("##teleop_map", 4, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
            const std::string pair = std::string(Tr("Input")) + " \xE2\x86\x92 " + Tr("Quantity");
            ImGui::TableSetupColumn(pair.c_str(), ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn(Tr("Gain"), ImGuiTableColumnFlags_WidthFixed, 40.0f * s);
            ImGui::TableSetupColumn("\xC2\xB1", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("##remove", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < bindings.size(); ++i) {
                TeleopBinding& b = bindings[i];
                ImGui::PushID(static_cast<int>(i));
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::BeginCombo("##input", b.input.c_str(), ImGuiComboFlags_NoArrowButton)) {
                    for (const std::string& in : inputs)
                        if (ImGui::Selectable(in.c_str(), in == b.input)) b.input = in;
                    ImGui::EndCombo();
                }
                ImGui::SetNextItemWidth(-FLT_MIN);
                // An extension's mode names its own quantities, so they are typed.
                if (names.empty()) {
                    char text[64];
                    std::snprintf(text, sizeof(text), "%s", b.quantity.c_str());
                    if (ImGui::InputText("##quantity", text, sizeof(text))) b.quantity = text;
                } else if (ImGui::BeginCombo("##quantity", ("\xE2\x86\x92 " + b.quantity).c_str(), ImGuiComboFlags_NoArrowButton)) {
                    for (const std::string& q : names)
                        if (ImGui::Selectable(q.c_str(), q == b.quantity)) b.quantity = q;
                    ImGui::EndCombo();
                }
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::DragFloat("##gain", &b.gain, 0.01f, 0.0f, 0.0f, "%.2f");
                ImGui::TableSetColumnIndex(2);
                ImGui::Checkbox("##invert", &b.invert);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", Tr("Invert"));
                ImGui::TableSetColumnIndex(3);
                if (nui::draw::IconButton("##remove", icon::kTrash, false, Tr("Remove"))) remove = static_cast<int>(i);
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        if (remove >= 0) bindings.erase(bindings.begin() + static_cast<std::ptrdiff_t>(remove));
        if (ImGui::Button(Iconed(icon::kAdd, Tr("Add Binding")).c_str()))
            bindings.push_back({inputs.empty() ? kPadAxes.front() : inputs.front(), names.empty() ? std::string() : names.front()});
        ImGui::SameLine();
        if (nui::draw::IconButton("##reset_map", icon::kReset, false, Tr("Reset Mapping")))
            Queue("teleop.map", Args({{"mode", json::Value::Str(t.mode)}, {"reset", json::Value::Bool(true)}}));
    });
    auto& tgrid = teleop.Add<nui::PropertyGrid>();
    auto& deadzone = tgrid.Add<nui::FloatField>("Deadzone", nui::Ref(&ui.teleop.deadzone), 0.005f, 0.0f, 0.95f);
    deadzone.format = "%.2f";
    tgrid.Add<nui::BoolField>("PD hold", nui::Ref(&ui.teleop.hold));
    tgrid.Add<nui::FloatField>("kp", nui::Ref(&ui.teleop.kp), 0.5f, 0.0f, 100000.0f);
    tgrid.Add<nui::FloatField>("kd", nui::Ref(&ui.teleop.kd), 0.05f, 0.0f, 10000.0f);
    auto& keys = teleop.Add<nui::Label>("W S and A D are the left stick, Q E the right stick across and Page Up / Page Down "
                                        "along it; - and = are the shoulders. Gamepads appear when plugged in.",
                                        Tone::Dim);
    keys.wrap = true;
    keys.visible_if = [tele] { return tele().device == "keyboard"; };
}

void ImGuiLayer::BuildPoses(nui::VStack& root) {
    ViewerUiState& ui = *frame_.ui;
    const auto robot = [this]() -> RobotView& { return frame_.ui->robot; };

    auto& hero = root.Add<nui::HStack>();
    auto& count = hero.Add<nui::Label>(std::string(), Tone::Accent, TextStyle::Hero);
    count.text_fn = [robot] { return Fmt(Tr("%zu POSES"), robot().poses.size()); };
    hero.Add<nui::Spacer>();
    hero.Add<nui::IconButton>(icon::kAdd, "Capture Pose").Action("robot.pose_capture");
    auto& grid = root.Add<nui::PropertyGrid>();
    auto& blend = grid.Add<nui::FloatField>("Blend time", nui::Ref(&ui.robot.pose_blend), 0.05f, 0.0f, 10.0f);
    blend.unit = "s";
    blend.format = "%.2f";
    root.Add<nui::Label>("No poses", Tone::Dim).visible_if = [robot] { return robot().poses.empty(); };

    // One row per pose: clicking blends the targets to it, the context menu holds the rest.
    root.Add<nui::Custom>([this] {
        RobotView& v = frame_.ui->robot;
        const float s = nui::Scale();
        for (size_t i = 0; i < v.poses.size(); ++i) {
            const JointPose& p = v.poses[i];
            const json::Value name = json::Value::Str(p.name);
            ImGui::PushID(static_cast<int>(i));
            const std::string label = Iconed(icon::kRobot, p.name.c_str());
            if (ImGui::Selectable(label.c_str(), false))
                Queue("robot.pose_apply", Args({{"name", name}, {"duration", json::Value::Float(v.pose_blend)}}));
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                ImGui::SetTooltip(Tr("Click to blend over %.2f s"), static_cast<double>(v.pose_blend));
            const std::string joints = Fmt(Tr("%zu joints"), p.joints.size());
            const ImVec2 end = ImGui::GetItemRectMax();
            ImGui::GetWindowDrawList()->AddText(
                ImVec2(end.x - ImGui::CalcTextSize(joints.c_str()).x - theme::kSpace * s, ImGui::GetItemRectMin().y),
                U32(theme::kFgDim), joints.c_str());
            if (ImGui::BeginPopupContextItem("##pose_menu")) {
                if (ImGui::IsWindowAppearing()) std::snprintf(pose_name_, sizeof(pose_name_), "%s", p.name.c_str());
                if (ImGui::MenuItemEx(Tr("Apply Instantly"), icon::kRun)) Queue("robot.pose_apply", Args({{"name", name}}));
                if (ImGui::MenuItemEx(Tr("Set as Initial"), icon::kHome)) Queue("robot.pose_initial", Args({{"name", name}}));
                if (ImGui::MenuItemEx(Tr("Copy as JSON"), icon::kCopy)) {
                    json::Value values = json::Value::Object();
                    for (const auto& [joint, q] : p.joints) values.Set(joint, json::Value::Float(q));
                    ImGui::SetClipboardText(Args({{"name", name}, {"joints", values}}).Dump(2).c_str());
                }
                ImGui::Separator();
                ImGui::SetNextItemWidth(160.0f * s);
                const bool enter = ImGui::InputText("##name", pose_name_, sizeof(pose_name_), ImGuiInputTextFlags_EnterReturnsTrue);
                ImGui::SameLine();
                if ((ImGui::SmallButton(Tr("Rename")) || enter) && pose_name_[0] != '\0') {
                    Queue("robot.pose_rename", Args({{"name", name}, {"to", json::Value::Str(pose_name_)}}));
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::MenuItemEx(Tr("Remove"), icon::kTrash)) Queue("robot.pose_remove", Args({{"name", name}}));
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
    });
}

void ImGuiLayer::BuildPolicy(nui::VStack& root) {
    const auto pol = [this]() -> PolicyView& { return frame_.ui->policy; };

    auto& head = root.Add<nui::HStack>();
    head.wrap = true;
    head.Add<nui::ToggleButton>("Run").Action("policy.enable");
    auto& state = head.Add<nui::Badge>();
    state.text_fn = [pol] { return std::string(Tr(!pol().loaded ? "no policy" : pol().on ? "running" : "stopped")); };
    state.tone_fn = [pol] { return !pol().loaded ? Tone::Dim : pol().on ? Tone::Accent : Tone::Warn; };
    auto& device = head.Add<nui::Badge>();
    device.text_fn = [pol] { return std::string(pol().gpu ? "GPU" : Tr("HOST")); };
    device.tone_fn = [] { return Tone::Info; };
    device.visible_if = [pol] { return pol().loaded; };

    // The file and where its MLP runs; Load replaces the policy before it.
    root.Add<nui::Custom>([this] {
        PolicyView& p = frame_.ui->policy;
        ImGui::TextDisabled("%s", Tr("Policy file"));
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputTextWithHint("##policy_path", Tr("NKML policy; empty browses"), p.path_edit, sizeof(p.path_edit));
        if (p.path_edit[0] != '\0' && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", p.path_edit);
        ImGui::Checkbox(Tr("Host MLP"), &p.host);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("%s", Tr("Run the MLP on the CPU instead of the GPU."));
        if (ImGui::Button(Iconed(icon::kOpen, Tr("Load")).c_str())) {
            json::Value args = Args({{"host", json::Value::Bool(p.host)}});
            if (p.path_edit[0] != '\0') args.Set("path", json::Value::Str(p.path_edit));
            Queue("policy.load", std::move(args));
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!p.loaded);
        if (ImGui::Button(Iconed(icon::kUnload, Tr("Unload")).c_str())) Queue("policy.unload", json::Value::Object());
        ImGui::EndDisabled();
        if (!p.error.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(theme::kWarn, "%s", p.error.c_str());
            ImGui::PopTextWrapPos();
        }
    });

    // The contract it runs under, the command it reads and its action norm over simulated time.
    auto& info = root.Add<nui::Custom>([this] {
        const PolicyView& p = frame_.ui->policy;
        const float s = nui::Scale();
        if (ImGui::BeginTable("##policy_info", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
            const auto row = [](const char* label, const std::string& value) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextColored(theme::kFgMuted, "%s", Tr(label));
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(value.c_str());
            };
            row("Robots", Fmt("%u", p.robots));
            row("Observations", Fmt("%u", p.obs_dim));
            row("Actions", Fmt(Tr("%u per robot"), p.act_dim));
            row("Decisions", Fmt(Tr("%.0f Hz (%u steps)"), static_cast<double>(p.rate_hz), p.decimation));
            row("Action norm", Fmt("%.3f", static_cast<double>(p.norm)));
            ImGui::EndTable();
        }
        ImGui::PushTextWrapPos(0.0f);
        nui::PushTextStyle(TextStyle::Mono);
        ImGui::TextColored(theme::kAccent, "\xE2\x86\x92 vx %+.2f  vy %+.2f  wyaw %+.2f", static_cast<double>(p.command[0]),
                           static_cast<double>(p.command[1]), static_cast<double>(p.command[2]));
        nui::PopTextStyle();
        ImGui::TextDisabled("%s", Tr("Teleop's velocity command mode steers it."));
        ImGui::PopTextWrapPos();
        nui::draw::SectionHeader(Tr("Action norm"));
        if (p.norm_time.empty()) {
            ImGui::TextDisabled("%s", Tr("Play to record its actions."));
            return;
        }
        ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(theme::kSpace * s, theme::kSpace * s));
        ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.0f, 0.1f));
        if (ImPlot::BeginPlot("##policy_norm", ImVec2(-FLT_MIN, 150.0f * s),
                              ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoFrame | ImPlotFlags_NoLegend)) {
            ImPlot::SetupAxes(Tr("s"), "|a|", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("##norm", p.norm_time.data(), p.norm_value.data(), static_cast<int>(p.norm_time.size()),
                             Line(theme::kAccent, 1.5f));
            ImPlot::EndPlot();
        }
        ImPlot::PopStyleVar(2);
    });
    info.visible_if = [pol] { return pol().loaded; };
}

void ImGuiLayer::BuildDemos(nui::VStack& root) {
    const auto demo = [this]() -> DemoView& { return frame_.ui->demos; };

    auto& head = root.Add<nui::HStack>();
    head.wrap = true;
    head.Add<nui::ToggleButton>("Record").Action("demo.record");
    auto& state = head.Add<nui::Badge>();
    state.text_fn = [demo] {
        const DemoView& d = demo();
        if (d.recording) return Fmt(Tr("REC %llu"), static_cast<unsigned long long>(d.samples));
        return std::string(Tr(d.playing >= 0 ? "replaying" : "idle"));
    };
    state.tone_fn = [demo] { return demo().recording ? Tone::Err : demo().playing >= 0 ? Tone::Accent : Tone::Dim; };
    auto& count = head.Add<nui::Badge>();
    count.text_fn = [demo] { return Fmt(Tr("%zu episodes"), demo().episodes.size()); };
    count.tone_fn = [] { return Tone::Info; };

    // Who the recording covers, the episodes, and the replay of the selected one.
    root.Add<nui::Custom>([this] {
        DemoView& d = frame_.ui->demos;
        const float s = nui::Scale();
        ImGui::BeginDisabled(d.recording);
        ImGui::Checkbox(Tr("All shown environments"), &d.all);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort | ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", Tr("Record every displayed environment, each as its own episode."));
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("%s", Tr("Each step at the head records the joints, drive targets, base, command and device axes."));
        ImGui::PopTextWrapPos();

        nui::draw::SectionHeader(Tr("Episodes"));
        const int n = static_cast<int>(d.episodes.size());
        if (n == 0) {
            ImGui::TextDisabled("%s", Tr("No episodes"));
        } else {
            const float rows = static_cast<float>(std::min(n, 8) + 1);
            const float row = ImGui::GetTextLineHeight() + 2.0f * ImGui::GetStyle().CellPadding.y;
            const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                                          ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY;
            if (ImGui::BeginTable("##demo_episodes", 4, flags, ImVec2(0.0f, rows * row + 2.0f * s))) {
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableSetupColumn("#");
                ImGui::TableSetupColumn(Tr("env"));
                ImGui::TableSetupColumn(Tr("steps"));
                ImGui::TableSetupColumn(Tr("seconds"));
                ImGui::TableHeadersRow();
                for (int k = 0; k < n; ++k) {
                    const DemoView::Episode& e = d.episodes[static_cast<size_t>(k)];
                    const json::Value episode = Args({{"episode", json::Value::Int(k)}});
                    ImGui::PushID(k);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    const std::string name = Iconed(k == d.playing ? icon::kPlay : icon::kRecord, Fmt("%d", k).c_str());
                    ImGui::PushStyleColor(ImGuiCol_Text, k == d.playing ? theme::kAccent : theme::kFg);
                    if (ImGui::Selectable(name.c_str(), k == d.selected, ImGuiSelectableFlags_SpanAllColumns)) d.selected = k;
                    ImGui::PopStyleColor();
                    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) Queue("demo.play", episode);
                    if (ImGui::BeginPopupContextItem("##episode_menu")) {
                        if (ImGui::MenuItemEx(Tr("Replay"), icon::kPlay)) Queue("demo.play", episode);
                        if (ImGui::MenuItemEx(Tr("Discard"), icon::kTrash)) Queue("demo.discard", episode);
                        ImGui::EndPopup();
                    }
                    ImGui::TableNextColumn();
                    ImGui::Text("%u", e.env);
                    ImGui::TableNextColumn();
                    ImGui::Text("%zu", e.steps);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.2f", e.seconds);
                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
        }
        // Buttons go on the line while they fit the panel.
        const auto beside = [](const std::string& label) {
            const float w = ImGui::CalcTextSize(label.c_str()).x + 2.0f * ImGui::GetStyle().FramePadding.x;
            ImGui::SameLine();
            if (ImGui::GetContentRegionAvail().x < w) ImGui::NewLine();
        };
        const bool picked = d.selected >= 0 && d.selected < n;
        const std::string replay = Iconed(icon::kPlay, Tr("Replay")), discard = Iconed(icon::kTrash, Tr("Discard")),
                          stop = Iconed(icon::kStop, Tr("Stop"));
        ImGui::BeginDisabled(!picked);
        if (ImGui::Button(replay.c_str())) Queue("demo.play", Args({{"episode", json::Value::Int(d.selected)}}));
        beside(discard);
        if (ImGui::Button(discard.c_str())) Queue("demo.discard", Args({{"episode", json::Value::Int(d.selected)}}));
        ImGui::EndDisabled();
        beside(stop);
        ImGui::BeginDisabled(d.playing < 0);
        if (ImGui::Button(stop.c_str())) Queue("demo.stop", json::Value::Object());
        ImGui::EndDisabled();
        if (d.playing >= 0) {
            const size_t steps = d.episodes[static_cast<size_t>(d.playing)].steps;
            const float done = 1.0f - static_cast<float>(d.left) / static_cast<float>(std::max<size_t>(steps, 1u));
            const std::string over = Fmt(Tr("episode %d: %u steps left"), d.playing, d.left);
            ImGui::ProgressBar(done, ImVec2(-FLT_MIN, 0.0f), over.c_str());
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("%s", Tr("A replay drives the targets from the state now; reset first to replay from its start."));
        ImGui::PopTextWrapPos();

        // The HDF5 file, in robomimic's layout.
        nui::draw::SectionHeader(Tr("File"));
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputTextWithHint("##demo_path", Tr("HDF5 file; empty browses"), d.path_edit, sizeof(d.path_edit));
        if (d.path_edit[0] != '\0' && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", d.path_edit);
        const auto path_args = [&d] {
            json::Value args = json::Value::Object();
            if (d.path_edit[0] != '\0') args.Set("path", json::Value::Str(d.path_edit));
            return args;
        };
        ImGui::BeginDisabled(n == 0);
        if (ImGui::Button(Iconed(icon::kSave, Tr("Save")).c_str())) Queue("demo.save", path_args());
        ImGui::EndDisabled();
        ImGui::SameLine(0.0f, theme::kSpace * s);
        ImGui::BeginDisabled(d.recording);
        if (ImGui::Button(Iconed(icon::kOpen, Tr("Load")).c_str())) Queue("demo.load", path_args());
        ImGui::EndDisabled();
        if (!d.error.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(theme::kWarn, "%s", d.error.c_str());
            ImGui::PopTextWrapPos();
        }
    });
}

void ImGuiLayer::DrawGainTuner() {
    RobotView& v = frame_.ui->robot;
    GainTuner& g = v.tuner;
    if (!g.open || v.joints.empty()) return;
    const float s = nui::Scale();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowSize(ImVec2(std::min(600.0f * s, vp->WorkSize.x * 0.9f), std::min(600.0f * s, vp->WorkSize.y * 0.9f)),
                             ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(vp->GetWorkCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    bool open = true;
    const std::string title = std::string(Tr("Gain Tuner")) + "###gain_tuner";
    if (ImGui::Begin(title.c_str(), &open, ImGuiWindowFlags_NoCollapse)) {
        int j = FindJoint(v.joints, g.joint);
        if (j < 0) {
            j = std::max(v.focus, 0);
            g.joint = v.joints[static_cast<size_t>(j)].name;
        }
        const RobotJoint& joint = v.joints[static_cast<size_t>(j)];
        const bool position = joint.drive == "position";
        if (ImGui::BeginTable("##gain_options", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, 120.0f * s);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
            TunerLabel("Joint");
            if (ImGui::BeginCombo("##joint", joint.name.c_str())) {
                for (size_t i = 0; i < v.joints.size(); ++i) {
                    const std::string name = Iconed(v.joints[i].prismatic ? icon::kMove : icon::kRotate, v.joints[i].name.c_str());
                    if (ImGui::Selectable(name.c_str(), static_cast<int>(i) == j)) g.joint = v.joints[i].name;
                }
                ImGui::EndCombo();
            }
            TunerLabel("Test");
            if (ImGui::RadioButton(Tr("Step test"), !g.sine)) g.sine = false;
            ImGui::SameLine();
            if (ImGui::RadioButton(Tr("Sine test"), g.sine)) g.sine = true;
            TunerLabel("Amplitude");
            const bool linear = joint.prismatic || v.radians;
            float amplitude = JointShown(joint, g.amplitude, v.radians);
            if (ImGui::DragFloat("##amplitude", &amplitude, linear ? 0.001f : 0.1f, 0.0f, 0.0f,
                                 joint.prismatic ? "%.3f m" : v.radians ? "%.3f rad" : "%.1f\xC2\xB0"))
                g.amplitude = linear ? amplitude : amplitude / kDegrees;
            if (g.sine) {
                TunerLabel("Frequency");
                ImGui::DragFloat("##frequency", &g.frequency, 0.05f, 0.05f, 50.0f, "%.2f Hz", ImGuiSliderFlags_AlwaysClamp);
            }
            TunerLabel("Duration");
            ImGui::DragFloat("##duration", &g.duration, 0.05f, 0.05f, 10.0f, "%.2f s", ImGuiSliderFlags_AlwaysClamp);
            // The joint's gains are document fields: each drag ends in one undoable edit.
            const char* const labels[3] = {"kp", "Damping", "Force limit"};
            const char* const fields[3] = {"actuator.gain", "joint.damping", "actuator.force_limit"};
            const char* const formats[3][2] = {{"%.1f N\xC2\xB7m/rad", "%.1f N/m"},
                                               {"%.3f N\xC2\xB7m\xC2\xB7s/rad", "%.3f N\xC2\xB7s/m"},
                                               {"%.1f N\xC2\xB7m", "%.1f N"}};
            const float live[3] = {joint.kp, joint.damping, joint.force_limit};
            for (int k = 0; k < 3; ++k) {
                TunerLabel(labels[k]);
                if (tuner_active_ != k) tuner_gains_[k] = live[k];
                ImGui::BeginDisabled(k != 1 && !position);
                ImGui::PushID(k);
                ImGui::DragFloat("##gain", &tuner_gains_[k], std::max(0.01f, live[k] * 0.005f), 0.0f, FLT_MAX,
                                 formats[k][joint.prismatic ? 1 : 0], ImGuiSliderFlags_AlwaysClamp);
                if (ImGui::IsItemActive()) tuner_active_ = k;
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    json::Value op = Args({{"target", json::Value::Str(joint.path)},
                                           {"field", json::Value::Str(fields[k])},
                                           {"after", json::Value::Float(tuner_gains_[k])}});
                    json::Value ops = json::Value::Array();
                    ops.PushBack(std::move(op));
                    Queue("edit.apply", Args({{"label", json::Value::Str(Tr(labels[k]))}, {"ops", std::move(ops)}}));
                }
                if (ImGui::IsItemDeactivated()) tuner_active_ = -1;
                ImGui::PopID();
                ImGui::EndDisabled();
            }
            ImGui::EndTable();
        }
        if (!position) {
            ImGui::TextColored(theme::kWarn, "%s", Tr("The test drives a position target: make this joint a position drive."));
            json::Value names = json::Value::Array();
            names.PushBack(json::Value::Str(joint.name));
            if (ImGui::Button(Iconed(icon::kDrive, Tr("Position Drive")).c_str()))
                Queue("robot.drive", Args({{"joints", std::move(names)}}));
        }
        if (v.floating) {
            ImGui::TextColored(theme::kWarn, "%s", Tr("The base is floating, so the robot falls during the test: fix it first."));
            if (ImGui::Button(Iconed(icon::kStatic, Tr("Fixed Base")).c_str()))
                Queue("robot.fix_base", Args({{"on", json::Value::Bool(true)}}));
        }
        ImGui::BeginDisabled(g.running != 0u || !position);
        if (ImGui::Button(Iconed(icon::kRun, Tr("Run")).c_str()))
            Queue("robot.gain_test", Args({{"joint", json::Value::Str(joint.name)},
                                           {"kind", json::Value::Str(g.sine ? "sine" : "step")},
                                           {"amplitude", json::Value::Float(g.amplitude)},
                                           {"frequency", json::Value::Float(g.frequency)},
                                           {"duration", json::Value::Float(g.duration)}}));
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(g.runs.empty());
        if (ImGui::Button(Iconed(icon::kTrash, Tr("Clear")).c_str())) Queue("robot.gain_clear");
        ImGui::EndDisabled();
        if (g.running != 0u) {
            ImGui::SameLine();
            ImGui::TextColored(theme::kInfo, "%s", Tr("Running..."));
        }

        nui::draw::SectionHeader(Tr("Response"), Fmt("%zu / %zu", g.runs.size(), kGainRuns).c_str());
        if (g.runs.empty()) {
            ImGui::TextDisabled("%s", Tr("Run a test to see the joint's response; the world is put back afterwards."));
        } else {
            const GainRun& newest = g.runs.back();
            ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(theme::kSpace * s, theme::kSpace * s));
            if (ImPlot::BeginPlot("##gain_plot", ImVec2(-FLT_MIN, 220.0f * s),
                                  ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoFrame)) {
                ImPlot::SetupAxes(Tr("s"), newest.prismatic ? "m" : "\xC2\xB0", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupLegend(ImPlotLocation_South, ImPlotLegendFlags_NoButtons | ImPlotLegendFlags_Outside |
                                                              ImPlotLegendFlags_Horizontal);
                for (size_t k = 0; k < g.runs.size(); ++k) {
                    const GainRun& run = g.runs[k];
                    const float scale = run.prismatic ? 1.0f : kDegrees;
                    std::vector<float> target(run.trace.target), q(run.trace.q);
                    for (float& x : target) x *= scale;
                    for (float& x : q) x *= scale;
                    const ImVec4 color = ImPlot::GetColormapColor(static_cast<int>(k));
                    const int n = static_cast<int>(q.size());
                    ImPlotSpec faint = Line(theme::Alpha(color, 0.45f), 1.0f);
                    faint.Flags = ImPlotItemFlags_NoLegend;
                    ImPlot::PlotLine(Fmt("##target%zu", k).c_str(), run.trace.time.data(), target.data(), n, faint);
                    const std::string label = Fmt(Tr("#%zu kp %.0f"), k + 1u, static_cast<double>(run.kp)) + Fmt("##run%zu", k);
                    ImPlot::PlotLine(label.c_str(), run.trace.time.data(), q.data(), n, Line(color, 1.5f));
                }
                ImPlot::EndPlot();
            }
            ImPlot::PopStyleVar();
            const char* const heads[] = {"#",          "Joint",    "Test",      "kp",       "Damping", "Rise",
                                         "Overshoot",  "Settling", "Steady error", "\xCE\xB6", "\xCF\x89n", "Gain",
                                         "Lag",        "RMSE",     "Saturated"};
            constexpr int kColumns = static_cast<int>(sizeof(heads) / sizeof(heads[0]));
            if (ImGui::BeginTable("##gain_runs", kColumns,
                                  ImGuiTableFlags_ScrollX | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                      ImGuiTableFlags_SizingFixedFit)) {
                for (const char* head : heads) ImGui::TableSetupColumn(Tr(head));
                ImGui::TableHeadersRow();
                for (size_t k = 0; k < g.runs.size(); ++k) {
                    const GainRun& run = g.runs[k];
                    const GainMetrics& m = run.metrics;
                    const float scale = run.prismatic ? 1.0f : kDegrees;
                    const std::string cells[kColumns] = {
                        Fmt("%zu", k + 1u),
                        run.joint,
                        run.trace.sine ? Fmt(Tr("sine %.2f Hz"), static_cast<double>(run.trace.frequency))
                                       : std::string(Tr("step test")),
                        Fmt("%.1f", static_cast<double>(run.kp)),
                        Fmt("%.3f", static_cast<double>(run.damping)),
                        Figure("%.3f s", m.rise),
                        Figure("%.1f%%", m.overshoot),
                        Figure("%.3f s", m.settling),
                        Figure(run.prismatic ? "%+.4f m" : "%+.2f\xC2\xB0", m.steady_error * scale),
                        Figure("%.2f", m.zeta),
                        Figure("%.1f", m.omega),
                        Figure("%+.1f dB", m.gain_db),
                        Figure("%.1f\xC2\xB0", m.lag),
                        Figure(run.prismatic ? "%.4f m" : "%.2f\xC2\xB0", m.rmse * scale),
                        Figure("%.0f%%", m.saturation * 100.0f)};
                    ImGui::TableNextRow();
                    for (int c = 0; c < kColumns; ++c) {
                        ImGui::TableSetColumnIndex(c);
                        if (c == 0)
                            ImGui::TextColored(ImPlot::GetColormapColor(static_cast<int>(k)), "%s", cells[c].c_str());
                        else
                            ImGui::TextUnformatted(cells[c].c_str());
                    }
                }
                ImGui::EndTable();
            }
        }
    }
    ImGui::End();
    if (!open) Queue("robot.gain_tuner", Args({{"on", json::Value::Bool(false)}}));
}

void ImGuiLayer::DrawSysId() {
    RobotView& v = frame_.ui->robot;
    SysIdView& id = v.sysid;
    if (!id.open || v.joints.empty()) return;
    const DemoView& demos = frame_.ui->demos;
    const float s = nui::Scale();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const ImVec2 size(std::min(560.0f * s, vp->WorkSize.x * 0.9f), std::min(620.0f * s, vp->WorkSize.y * 0.9f));
    ImGui::SetNextWindowSize(size, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(vp->GetWorkCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    bool open = true;
    const std::string title = std::string(Tr("System Identification")) + "###sysid";
    if (ImGui::Begin(title.c_str(), &open, ImGuiWindowFlags_NoCollapse)) {
        const int episodes = static_cast<int>(demos.episodes.size());
        if (episodes == 0) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextDisabled("%s", Tr("Record or load a demonstration in the Demos panel first: identification "
                                         "replays its targets and fits the joint positions it recorded."));
            ImGui::PopTextWrapPos();
        } else {
            if (id.episode >= episodes) id.episode = -1;
            const int shown = id.episode < 0 ? episodes - 1 : id.episode;
            const auto episode_name = [&demos](int k) {
                const DemoView::Episode& e = demos.episodes[static_cast<size_t>(k)];
                return Fmt(Tr("episode %d: env %u, %zu steps"), k, e.env, e.steps);
            };
            if (ImGui::BeginTable("##sysid_options", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, 120.0f * s);
                ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
                TunerLabel("Episode");
                if (ImGui::BeginCombo("##episode", episode_name(shown).c_str())) {
                    for (int k = 0; k < episodes; ++k)
                        if (ImGui::Selectable(episode_name(k).c_str(), k == shown)) id.episode = k;
                    ImGui::EndCombo();
                }
                TunerLabel("Generations");
                ImGui::DragInt("##generations", &id.generations, 0.2f, 1, 1000, "%d", ImGuiSliderFlags_AlwaysClamp);
                TunerLabel("Population");
                ImGui::DragInt("##population", &id.population, 0.1f, 2, 256, "%d", ImGuiSliderFlags_AlwaysClamp);
                ImGui::EndTable();
            }

            // Each row searches one quantity over [min, max]; a scale multiplies the links' masses.
            nui::draw::SectionHeader(Tr("Parameters"));
            static const char* const kKeys[] = {"joint_damping", "joint_armature", "joint_frictionloss",
                                                "link_mass_scale"};
            static const char* const kNames[] = {"Damping", "Armature", "Friction loss", "Mass scale"};
            const auto name_of = [](const std::string& key) {
                for (size_t k = 0; k < 4u; ++k)
                    if (key == kKeys[k]) return Tr(kNames[k]);
                return key.c_str();
            };
            int remove = -1;
            if (ImGui::BeginTable("##sysid_params", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                                           ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn(Tr("Parameter"));
                ImGui::TableSetupColumn(Tr("Joint"));
                ImGui::TableSetupColumn(Tr("Min"));
                ImGui::TableSetupColumn(Tr("Max"));
                ImGui::TableSetupColumn("##remove", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFrameHeight());
                ImGui::TableHeadersRow();
                for (size_t i = 0; i < id.params.size(); ++i) {
                    SysIdParam& p = id.params[i];
                    ImGui::PushID(static_cast<int>(i));
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    if (ImGui::BeginCombo("##param", name_of(p.param))) {
                        for (size_t k = 0; k < 4u; ++k) {
                            if (!ImGui::Selectable(Tr(kNames[k]), p.param == kKeys[k]) || p.param == kKeys[k]) continue;
                            p.param = kKeys[k];
                            const bool scale = k == 3u;
                            p.min = scale ? 0.5f : 0.0f;
                            p.max = scale ? 2.0f : 1.0f;
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    if (ImGui::BeginCombo("##joint", p.joint.empty() ? Tr("All joints") : p.joint.c_str())) {
                        if (ImGui::Selectable(Tr("All joints"), p.joint.empty())) p.joint.clear();
                        for (const RobotJoint& joint : v.joints)
                            if (ImGui::Selectable(joint.name.c_str(), p.joint == joint.name)) p.joint = joint.name;
                        ImGui::EndCombo();
                    }
                    ImGui::TableSetColumnIndex(2);
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::DragFloat("##min", &p.min, 0.01f, 0.0f, FLT_MAX, "%.4g", ImGuiSliderFlags_AlwaysClamp);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::DragFloat("##max", &p.max, 0.01f, 0.0f, FLT_MAX, "%.4g", ImGuiSliderFlags_AlwaysClamp);
                    ImGui::TableSetColumnIndex(4);
                    ImGui::BeginDisabled(id.params.size() == 1u);
                    if (ImGui::SmallButton(icon::kTrash)) remove = static_cast<int>(i);
                    ImGui::EndDisabled();
                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
            if (remove >= 0) id.params.erase(id.params.begin() + remove);
            if (ImGui::SmallButton(Iconed(icon::kAdd, Tr("Add Parameter")).c_str())) id.params.push_back(SysIdParam{});

            ImGui::Spacing();
            ImGui::BeginDisabled(id.running);
            if (ImGui::Button(Iconed(icon::kRun, Tr("Identify")).c_str())) {
                json::Value params = json::Value::Array();
                for (const SysIdParam& p : id.params) {
                    json::Value joints = json::Value::Array();
                    if (!p.joint.empty()) joints.PushBack(json::Value::Str(p.joint));
                    params.PushBack(Args({{"param", json::Value::Str(p.param)},
                                          {"joints", std::move(joints)},
                                          {"min", json::Value::Float(p.min)},
                                          {"max", json::Value::Float(p.max)}}));
                }
                Queue("sysid.run", Args({{"episode", json::Value::Int(shown)},
                                         {"params", std::move(params)},
                                         {"generations", json::Value::Int(id.generations)},
                                         {"population", json::Value::Int(id.population)}}));
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::BeginDisabled(!id.running);
            if (ImGui::Button(Iconed(icon::kStop, Tr("Stop")).c_str())) Queue("sysid.stop");
            ImGui::EndDisabled();
            ImGui::SameLine();
            const bool improved = !id.running && !id.found.empty() && id.best_loss < id.start_loss;
            ImGui::BeginDisabled(!improved);
            if (ImGui::Button(Iconed(icon::kOk, Tr("Apply")).c_str())) Queue("sysid.apply");
            ImGui::EndDisabled();
            if (id.running) {
                const float done = static_cast<float>(id.generation) / static_cast<float>(std::max(id.total, 1u));
                const std::string over =
                    Fmt(Tr("generation %u / %u, %u replays"), id.generation, id.total, id.evaluations);
                ImGui::ProgressBar(done, ImVec2(-FLT_MIN, 0.0f), over.c_str());
            }
            if (!id.error.empty()) {
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(theme::kWarn, "%s", id.error.c_str());
                ImGui::PopTextWrapPos();
            }

            nui::draw::SectionHeader(Tr("Fit"));
            if (id.found.empty()) {
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextDisabled("%s", Tr("Each replay starts from the episode's first step in a copy of the world "
                                             "and scores the mean squared joint error; the world is put back."));
                ImGui::PopTextWrapPos();
            } else {
                const double cut = id.start_loss > 0.0 ? 100.0 * (1.0 - id.best_loss / id.start_loss) : 0.0;
                const std::string fit = Fmt(Tr("Joint error %.3g \xE2\x86\x92 %.3g (%.1f%% lower)"), id.start_loss,
                                            id.best_loss, cut);
                ImGui::TextUnformatted(fit.c_str());
                if (!id.curve.empty()) {
                    // Generation 0 is the values in force.
                    std::vector<double> x(id.curve.size() + 1u), y{id.start_loss};
                    for (size_t k = 0; k < x.size(); ++k) x[k] = static_cast<double>(k);
                    y.insert(y.end(), id.curve.begin(), id.curve.end());
                    // A factor of two of room on the log axis; a perfect fit's zero stays above its floor.
                    const auto [lo, hi] = std::minmax_element(y.begin(), y.end());
                    const double bottom = std::max(*lo, *hi * 1e-15);
                    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(theme::kSpace * s, theme::kSpace * s));
                    if (ImPlot::BeginPlot("##sysid_plot", ImVec2(-FLT_MIN, 180.0f * s),
                                          ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoFrame |
                                              ImPlotFlags_NoLegend)) {
                        ImPlot::SetupAxes(Tr("generation"), Tr("best loss"), ImPlotAxisFlags_AutoFit);
                        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
                        ImPlot::SetupAxisLimits(ImAxis_Y1, bottom * 0.5, *hi * 2.0, ImPlotCond_Always);
                        ImPlot::PlotLine(Tr("best"), x.data(), y.data(), static_cast<int>(x.size()),
                                         Line(ImPlot::GetColormapColor(0), 1.5f));
                        ImPlot::EndPlot();
                    }
                    ImPlot::PopStyleVar();
                }
                if (ImGui::BeginTable("##sysid_found", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                                              ImGuiTableFlags_SizingStretchProp)) {
                    for (const char* head : {"Parameter", "Joints", "Before", "Identified"})
                        ImGui::TableSetupColumn(Tr(head));
                    ImGui::TableHeadersRow();
                    for (const SysIdFound& f : id.found) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::TextUnformatted(name_of(f.param));
                        ImGui::TableSetColumnIndex(1);
                        const std::string joints =
                            f.joints.size() == 1u ? f.joints.front() : Fmt(Tr("%zu joints"), f.joints.size());
                        ImGui::TextUnformatted(joints.c_str());
                        ImGui::TableSetColumnIndex(2);
                        ImGui::Text("%.4g", f.nominal);
                        ImGui::TableSetColumnIndex(3);
                        ImGui::TextColored(f.value == f.nominal ? theme::kFgMuted : theme::kInfo, "%.4g", f.value);
                    }
                    ImGui::EndTable();
                }
            }
        }
    }
    ImGui::End();
    if (!open) Queue("sysid.panel", Args({{"on", json::Value::Bool(false)}}));
}

void ImGuiLayer::DrawAssembler() {
    RobotView& v = frame_.ui->robot;
    Assembler& a = v.assembler;
    if (!a.open || !frame_.ui->has_scene) return;
    const float s = nui::Scale();
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetWorkCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    bool open = true;
    const std::string title = std::string(Tr("Robot Assembler")) + "###robot_assembler";
    if (ImGui::Begin(title.c_str(), &open, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) {
        const std::string pick = Iconed(icon::kTarget, Tr("Use Selection"));
        const ImGuiStyle& style = ImGui::GetStyle();
        const float pick_w = ImGui::CalcTextSize(pick.c_str()).x + style.FramePadding.x * 2.0f + style.ItemSpacing.x;
        const auto use_selection = [&](const char* id, std::string* out) {
            ImGui::SameLine();
            ImGui::BeginDisabled(a.selected.empty());
            if (ImGui::SmallButton(Fmt("%s##%s", pick.c_str(), id).c_str())) *out = a.selected;
            ImGui::EndDisabled();
        };
        const auto node = [](const std::string& path) { return path.empty() ? std::string(Tr("none")) : path; };
        if (ImGui::BeginTable("##assembler", 2, ImGuiTableFlags_SizingStretchProp, ImVec2(440.0f * s, 0.0f))) {
            ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, 110.0f * s);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
            TunerLabel("Attach");
            if (ImGui::RadioButton(Tr("Asset file"), a.from_asset)) a.from_asset = true;
            ImGui::SameLine();
            if (ImGui::RadioButton(Tr("Scene body"), !a.from_asset)) a.from_asset = false;
            if (a.from_asset) {
                TunerLabel("Asset");
                if (ImGui::SmallButton(Iconed(icon::kOpen, Tr("Browse...")).c_str()))
                    Queue("robot.asset_links", Args({{"profile", json::Value::Str(a.profile)}}));
                ImGui::SameLine();
                const std::string file = a.path.substr(a.path.find_last_of("/\\") + 1u);
                ImGui::TextUnformatted(a.path.empty() ? Tr("none") : file.c_str());
                if (!a.path.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", a.path.c_str());
                TunerLabel("Profile");
                for (const char* profile : {"strict", "legacy"}) {
                    if (profile[0] == 'l') ImGui::SameLine();
                    if (!ImGui::RadioButton(Tr(profile[0] == 'l' ? "Legacy" : "Strict"), a.profile == profile) ||
                        a.profile == profile)
                        continue;
                    a.profile = profile;
                    if (!a.path.empty())
                        Queue("robot.asset_links", Args({{"path", json::Value::Str(a.path)}, {"profile", json::Value::Str(profile)}}));
                }
                TunerLabel("Mount");
                const std::string shown = a.mount.empty() ? std::string(Tr("its root")) : a.mount;
                if (ImGui::BeginCombo("##mount", shown.c_str())) {
                    if (ImGui::Selectable(Tr("its root"), a.mount.empty())) a.mount.clear();
                    for (const std::string& link : a.asset_links)
                        if (ImGui::Selectable(link.c_str(), a.mount == link)) a.mount = link;
                    ImGui::EndCombo();
                }
            } else {
                TunerLabel("Body");
                ImGui::TextUnformatted(node(a.child).c_str());
                use_selection("child", &a.child);
            }
            TunerLabel("Parent link");
            ImGui::SetNextItemWidth(-pick_w);
            if (ImGui::BeginCombo("##parent", node(a.parent).c_str())) {
                for (const RobotLink& link : v.links)
                    if (ImGui::Selectable(link.path.c_str(), a.parent == link.path)) a.parent = link.path;
                ImGui::EndCombo();
            }
            use_selection("parent", &a.parent);
            TunerLabel("Joint");
            ImGui::Combo("##joint", &a.joint, [](void*, int i) { return Tr(kAttachJoints[i]); }, nullptr, 3);
            if (a.joint != 0) {
                TunerLabel("Axis");
                ImGui::DragFloat3("##axis", a.axis, 0.01f, -1.0f, 1.0f, "%.2f");
            }
            if (!a.from_asset) {
                TunerLabel("Placement");
                if (ImGui::RadioButton(Tr("Keep where it is"), a.keep_pose)) a.keep_pose = true;
                ImGui::SameLine();
                if (ImGui::RadioButton(Tr("Offset"), !a.keep_pose)) a.keep_pose = false;
            }
            if (a.from_asset || !a.keep_pose) {
                TunerLabel("Position");
                ImGui::DragFloat3("##pos", a.pos, 0.001f, 0.0f, 0.0f, "%.3f m");
                TunerLabel("Rotation");
                ImGui::DragFloat3("##rot", a.rot, 0.5f, -360.0f, 360.0f, "%.1f\xC2\xB0");
            } else if (a.posed) {
                TunerLabel("Offset");
                ImGui::TextColored(theme::kFgMuted, "%s",
                                   Fmt("%.3f %.3f %.3f m  %.1f %.1f %.1f\xC2\xB0", static_cast<double>(a.offset_pos[0]),
                                       static_cast<double>(a.offset_pos[1]), static_cast<double>(a.offset_pos[2]),
                                       static_cast<double>(a.offset_rot[0]), static_cast<double>(a.offset_rot[1]),
                                       static_cast<double>(a.offset_rot[2]))
                                       .c_str());
            }
            ImGui::EndTable();
        }
        if (!a.asset_error.empty() && a.from_asset) ImGui::TextColored(theme::kWarn, "%s", a.asset_error.c_str());
        const bool mounted = a.from_asset && !a.mount.empty() && (a.asset_links.empty() || a.mount != a.asset_links[0]);
        if (mounted && a.joint != 0)
            ImGui::TextColored(theme::kWarn, "%s", Tr("Only a fixed joint mounts a link other than the root."));
        ImGui::TextDisabled("%s", Tr(a.from_asset ? "The mount takes the offset in the parent link's frame."
                                                  : "The body and what its joints carry move with the parent link."));
        const bool ready = !a.parent.empty() && (a.from_asset ? !a.path.empty() && a.asset_error.empty() : !a.child.empty());
        ImGui::BeginDisabled(!ready);
        if (ImGui::Button(Iconed(icon::kJoint, Tr("Attach")).c_str())) {
            json::Value args = Args({{"parent", json::Value::Str(a.parent)}, {"joint", json::Value::Str(kAttachJoints[a.joint])}});
            args.Set("axis", Floats(a.axis));
            if (a.from_asset || !a.keep_pose)
                args.Set("frame", Args({{"pos", Floats(a.pos)}, {"rot", Floats(a.rot)}}));
            if (!a.mount.empty() && a.from_asset) args.Set("mount", json::Value::Str(a.mount));
            if (a.from_asset) {
                args.Set("path", json::Value::Str(a.path));
                args.Set("profile", json::Value::Str(a.profile));
                Queue("robot.attach_asset", std::move(args));
            } else {
                args.Set("child", json::Value::Str(a.child));
                Queue("robot.attach", std::move(args));
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button(Tr("Cancel"))) open = false;
    }
    ImGui::End();
    if (!open) Queue("robot.assembler", Args({{"on", json::Value::Bool(false)}}));
}

}  // namespace nuka::runtime::app::viewer
