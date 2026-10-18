// The Joints panel: each moving joint's limits, value, drive target and effort, the selected
// joint's curves, drive conversion, the fixed base and keyboard teleop; and the Poses panel.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/shell_util.hpp"

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

// Revolute joints show degrees, prismatic ones meters.
float Shown(const RobotJoint& j, float v) { return j.prismatic ? v : v * kDegrees; }
const char* Unit(const RobotJoint& j) { return j.prismatic ? " m" : "\xC2\xB0"; }

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

std::string Value(const RobotJoint& j, float v) {
    return Fmt(j.prismatic ? "%+.3f%s" : "%+.1f%s", static_cast<double>(Shown(j, v)), Unit(j));
}

ImPlotSpec Line(const ImVec4& color, float weight) {
    ImPlotSpec spec;
    spec.LineColor = color;
    spec.LineWeight = weight * nui::Scale();
    return spec;
}

}  // namespace

void ImGuiLayer::BuildJoints(nui::VStack& root) {
    ViewerUiState& ui = *frame_.ui;
    const auto robot = [this]() -> RobotView& { return frame_.ui->robot; };

    auto& hero = root.Add<nui::HStack>();
    auto& count = hero.Add<nui::Label>(std::string(), Tone::Accent, TextStyle::Hero);
    count.text_fn = [robot] { return Fmt(Tr("%zu JOINTS"), robot().joints.size()); };
    auto& base = hero.Add<nui::Badge>();
    base.text_fn = [robot] { return std::string(Tr(robot().floating ? "floating base" : "fixed base")); };
    base.tone_fn = [robot] { return robot().floating ? Tone::Info : Tone::Dim; };
    hero.Add<nui::Spacer>();
    auto& watch = hero.Add<nui::IconButton>(icon::kPlot, "Watch Joints").Action("robot.watch");
    watch.checked_fn = [robot] { return robot().watch; };
    hero.visible_if = [robot] { return !robot().joints.empty(); };
    root.Add<nui::Label>("No moving joints", Tone::Dim).visible_if = [robot] { return robot().joints.empty(); };

    auto& tools = root.Add<nui::HStack>();
    tools.Add<nui::ToggleButton>("Fixed Base").Action("robot.fix_base");
    auto& convert = tools.Add<nui::Button>("Position Drive", icon::kDrive).Action("robot.drive");
    convert.tooltip = "Makes every joint a position drive with the gain below, in one undoable edit.";
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
                    const std::string speed = Fmt(j.prismatic ? Tr("velocity %+.3f m/s") : Tr("velocity %+.1f\xC2\xB0/s"),
                                                  static_cast<double>(Shown(j, At(v.live.qd, j.link))));
                    ImGui::SetTooltip("%s\n%s  \xC2\xB7  %s\n%s", j.path.c_str(), drive.c_str(), limit.c_str(), speed.c_str());
                }
                ImDrawList* dl = ImGui::GetWindowDrawList();
                const float ty = p.y + pad * 0.5f, effort = At(v.live.effort, j.link);
                const std::string force = Fmt(j.prismatic ? "%.1f N" : "%.1f N\xC2\xB7m", static_cast<double>(effort));
                const std::string value = Value(j, q);
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
                }
                if (ImGui::IsItemHovered() || ImGui::IsItemActive())
                    ImGui::SetTooltip(Tr("target %s"), Value(j, target).c_str());
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
            q.push_back(Shown(j, sample.q[j.link]));
            target.push_back(Shown(j, sample.target[j.link]));
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
            ImPlot::SetupAxes(Tr("s"), j.prismatic ? "m" : "\xC2\xB0", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
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

    const size_t n = ui.drive_targets.size();
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

}  // namespace nuka::runtime::app::viewer
