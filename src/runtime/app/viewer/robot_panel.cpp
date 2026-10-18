// The Joints panel: each moving joint's limits, value, drive target and effort, the selected
// joint's curves, drive conversion, the fixed base and keyboard teleop; the Poses panel and
// the Gain Tuner.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/robot_tools.hpp"
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
    auto& tuner = hero.Add<nui::IconButton>(icon::kWaves, "Gain Tuner").Action("robot.gain_tuner");
    tuner.checked_fn = [robot] { return robot().tuner.open; };
    auto& assembler = hero.Add<nui::IconButton>(icon::kPlug, "Robot Assembler").Action("robot.assembler");
    assembler.checked_fn = [robot] { return robot().assembler.open; };
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
            float amplitude = Shown(joint, g.amplitude);
            if (ImGui::DragFloat("##amplitude", &amplitude, joint.prismatic ? 0.001f : 0.1f, 0.0f, 0.0f,
                                 joint.prismatic ? "%.3f m" : "%.1f\xC2\xB0"))
                g.amplitude = joint.prismatic ? amplitude : amplitude / kDegrees;
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
