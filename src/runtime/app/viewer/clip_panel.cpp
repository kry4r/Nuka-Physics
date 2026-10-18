// The Clip Editor: the robot's joint clips as a dope sheet of keys over a curve editor of the
// chosen joint, with the playhead, playback and pose keying above them.

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
using nuka::nui::Tr;
using namespace shell;

namespace {

constexpr float kDegrees = 57.2957795f;
constexpr int kCurveSamples = 241;
constexpr int kPlayheadId = 1 << 30;  // the curve plot's playhead line, past every key's drag id

ImU32 Col(const ImVec4& c) { return ImGui::GetColorU32(c); }

ImPlotSpec Line(const ImVec4& color, float weight) {
    ImPlotSpec spec;
    spec.LineColor = color;
    spec.LineWeight = weight * nui::Scale();
    return spec;
}

// A time on the step grid; without a step it stays.
float Snap(float t, float dt) { return dt > 0.0f ? std::round(t / dt) * dt : t; }

// The ruler's label step for `px` pixels a second, labels at least `gap` pixels apart.
float RulerStep(float px, float gap) {
    for (const float step : {0.01f, 0.02f, 0.05f, 0.1f, 0.2f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f, 30.0f})
        if (step * px >= gap) return step;
    return 60.0f;
}

bool Selected(const ClipView& c, const std::string& joint, float time) {
    return std::any_of(c.selected.begin(), c.selected.end(),
                       [&](const ClipKeyRef& r) { return r.joint == joint && std::abs(r.time - time) < kClipKeyEpsilon; });
}

json::Value KeyRefs(const std::vector<ClipKeyRef>& keys) {
    json::Value list = json::Value::Array();
    for (const ClipKeyRef& k : keys)
        list.PushBack(Args({{"joint", json::Value::Str(k.joint)}, {"time", json::Value::Float(k.time)}}));
    return list;
}

// A key as clip.key reads it.
json::Value KeyArg(const std::string& joint, const ClipKey& key) {
    json::Value v = KeyJson(key);
    v.Set("joint", json::Value::Str(joint));
    return v;
}

// The selected keys with interpolation `interp`, as clip.key's `keys`.
json::Value ShapedKeys(const ClipView& c, Clip* clip, ClipInterp interp) {
    json::Value keys = json::Value::Array();
    for (const ClipKeyRef& r : c.selected) {
        const ClipKey* k = FindKey(clip, r.joint, r.time);
        if (k == nullptr) continue;
        ClipKey key = *k;
        key.interp = interp;
        keys.PushBack(KeyArg(r.joint, key));
    }
    return keys;
}

// The selected keys into the clipboard, their times from the earliest.
void CopyKeys(ClipView* c, Clip* clip) {
    c->clipboard.clear();
    float first = FLT_MAX;
    for (const ClipKeyRef& r : c->selected) first = std::min(first, r.time);
    for (const ClipKeyRef& r : c->selected) {
        const ClipKey* k = FindKey(clip, r.joint, r.time);
        if (k == nullptr) continue;
        ClipKey key = *k;
        key.time -= first;
        c->clipboard.emplace_back(r.joint, key);
    }
}

// The clipboard's keys from the playhead as clip.key's `keys`, those past the clip's end left out.
json::Value PastedKeys(const ClipView& c, const Clip& clip) {
    json::Value keys = json::Value::Array();
    for (const auto& [joint, k] : c.clipboard) {
        ClipKey key = k;
        key.time += c.playhead;
        if (key.time <= clip.duration) keys.PushBack(KeyArg(joint, key));
    }
    return keys;
}

// A key's mark: a diamond for linear, a square for step and a circle for Bezier.
void KeyMark(ImDrawList* dl, ImVec2 at, float r, ClipInterp interp, ImU32 fill, ImU32 edge) {
    const float w = nui::Scale();
    if (interp == ClipInterp::Step) {
        const ImVec2 a(at.x - r * 0.75f, at.y - r * 0.75f), b(at.x + r * 0.75f, at.y + r * 0.75f);
        dl->AddRectFilled(a, b, fill);
        dl->AddRect(a, b, edge, 0.0f, 0, w);
    } else if (interp == ClipInterp::Bezier) {
        dl->AddCircleFilled(at, r * 0.85f, fill);
        dl->AddCircle(at, r * 0.85f, edge, 0, w);
    } else {
        const ImVec2 p[4] = {ImVec2(at.x, at.y - r), ImVec2(at.x + r, at.y), ImVec2(at.x, at.y + r), ImVec2(at.x - r, at.y)};
        dl->AddConvexPolyFilled(p, 4, fill);
        dl->AddPolyline(p, 4, edge, ImDrawFlags_Closed, w);
    }
}

}  // namespace

void ImGuiLayer::DrawClipEditor() {
    ViewerUiState& ui = *frame_.ui;
    ClipView& c = ui.clips;
    const RobotView& v = ui.robot;
    c.focused = false;
    if (!c.open || v.joints.empty()) return;
    const float s = nui::Scale();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowSize(ImVec2(std::min(780.0f * s, vp->WorkSize.x * 0.9f), std::min(580.0f * s, vp->WorkSize.y * 0.9f)),
                             ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(vp->GetWorkCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    bool open = true;
    const bool was_scrubbing = clip_scrubbing_;
    clip_scrubbing_ = false;
    const std::string title = std::string(Tr("Clip Editor")) + "###clip_editor";
    if (ImGui::Begin(title.c_str(), &open, ImGuiWindowFlags_NoCollapse)) {
        c.focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
        Clip* clip = c.edited >= 0 ? &c.clips[static_cast<size_t>(c.edited)] : nullptr;
        const json::Value name = json::Value::Str(clip != nullptr ? clip->name : std::string());

        ImGui::SetNextItemWidth(170.0f * s);
        if (ImGui::BeginCombo("##clip", clip != nullptr ? clip->name.c_str() : Tr("No clip"))) {
            for (size_t k = 0; k < c.clips.size(); ++k)
                if (ImGui::Selectable(c.clips[k].name.c_str(), static_cast<int>(k) == c.edited))
                    Queue("clip.editor", Args({{"clip", json::Value::Str(c.clips[k].name)}}));
            ImGui::EndCombo();
        }
        // Right-clicked, the clip renames or goes.
        if (clip != nullptr && ImGui::BeginPopupContextItem("##clip_menu")) {
            if (ImGui::IsWindowAppearing()) {
                std::snprintf(c.name_edit, sizeof(c.name_edit), "%s", clip->name.c_str());
                ImGui::SetKeyboardFocusHere();
            }
            ImGui::SetNextItemWidth(180.0f * s);
            if (ImGui::InputText("##rename", c.name_edit, sizeof(c.name_edit), ImGuiInputTextFlags_EnterReturnsTrue)) {
                Queue("clip.rename", Args({{"clip", name}, {"to", json::Value::Str(c.name_edit)}}));
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem(Iconed(icon::kTrash, Tr("Remove Clip")).c_str())) Queue("clip.remove", Args({{"clip", name}}));
            ImGui::EndPopup();
        }
        ImGui::SetItemTooltip("%s", Tr("The clip being edited; right-click to rename or remove it."));
        ImGui::SameLine();
        if (ImGui::Button(Iconed(icon::kAdd, Tr("New")).c_str())) Queue("clip.new");
        if (clip != nullptr) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(90.0f * s);
            ImGui::DragFloat("##duration", &clip->duration, 0.01f, 0.05f, 600.0f, "%.2f s", ImGuiSliderFlags_AlwaysClamp);
            if (ImGui::IsItemDeactivatedAfterEdit()) c.commit = true;
            ImGui::SetItemTooltip("%s", Tr("Duration"));
            ImGui::SameLine();
            bool loop = clip->loop;
            if (ImGui::Checkbox(Tr("Loop"), &loop)) Queue("clip.set", Args({{"clip", name}, {"loop", json::Value::Bool(loop)}}));
            ImGui::SameLine();
            if (c.playing >= 0) {
                if (ImGui::Button(Iconed(icon::kStop, Tr("Stop")).c_str())) Queue("clip.stop");
            } else if (ImGui::Button(Iconed(icon::kPlay, Tr("Play")).c_str())) {
                Queue("clip.play", Args({{"clip", name}}));
            }
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!c.can_undo);
        if (ImGui::Button(icon::kUndo)) Queue("clip.undo");
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("%s", Tr("Undo Clip Edit"));
        ImGui::SameLine();
        ImGui::BeginDisabled(!c.can_redo);
        if (ImGui::Button(icon::kRedo)) Queue("clip.redo");
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("%s", Tr("Redo Clip Edit"));
        ImGui::SameLine();
        ImGui::BeginDisabled(clip == nullptr);
        if (ImGui::Button(icon::kSave)) Queue("clip.save", Args({{"clip", name}}));
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("%s", Tr("Export Clip..."));
        ImGui::SameLine();
        if (ImGui::Button(icon::kOpen)) Queue("clip.load");
        ImGui::SetItemTooltip("%s", Tr("Import Clip..."));

        if (clip == nullptr) {
            ImGui::TextDisabled("%s", Tr("A clip keys the robot's joint targets over time: make one to start."));
        } else {
            if (ImGui::Button(Iconed(icon::kKeyframe, Tr("Key Pose")).c_str())) Queue("clip.key_pose", Args({{"clip", name}}));
            ImGui::SetItemTooltip("%s", Tr("Keys every joint at the playhead where the robot stands now."));
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f * s);
            if (ImGui::BeginCombo("##key_library", Tr("Key a saved pose"))) {
                if (v.poses.empty()) ImGui::TextDisabled("%s", Tr("No saved poses"));
                for (const JointPose& p : v.poses)
                    if (ImGui::Selectable(p.name.c_str()))
                        Queue("clip.key_pose", Args({{"clip", name}, {"pose", json::Value::Str(p.name)}}));
                ImGui::EndCombo();
            }
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, c.auto_key ? theme::kErr : theme::kFgMuted);
            if (ImGui::Button(Iconed(icon::kRecord, Tr("Auto-key")).c_str())) c.auto_key = !c.auto_key;
            ImGui::PopStyleColor();
            ImGui::SetItemTooltip("%s", Tr("While on, moving a joint target or dragging IK keys it at the playhead."));
            ImGui::SameLine();
            const char* const shapes[] = {Tr("Step"), Tr("Linear"), Tr("Bezier")};
            int shape = static_cast<int>(c.interp);
            ImGui::SetNextItemWidth(100.0f * s);
            if (ImGui::Combo("##interp", &shape, shapes, 3)) c.interp = static_cast<ClipInterp>(shape);
            ImGui::SetItemTooltip("%s", Tr("The interpolation new keys take."));
            ImGui::SameLine();
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(c.playing >= 0 ? theme::kAccent : theme::kFgMuted, "%s",
                               Fmt(Tr("%.3f / %.2f s"), static_cast<double>(c.playhead), static_cast<double>(clip->duration)).c_str());
            if (c.playing >= 0) {
                ImGui::SameLine();
                ImGui::TextColored(theme::kAccent, "%s", c.looping ? Tr("looping") : Tr("playing"));
            }

            const float row = ImGui::GetFrameHeight();
            const float rows = static_cast<float>(v.joints.size() + 2u);
            DrawDopeSheet(std::min(rows * row + 2.0f * s, ImGui::GetContentRegionAvail().y * 0.5f));
            if (FindTrack(*clip, c.curve) == nullptr)
                ImGui::TextDisabled("%s", Tr("No keys on this joint: double-click its row or key a pose."));
            DrawClipCurves();
            ClipKeys();
        }
    }
    ImGui::End();
    // A released playhead puts the robot's targets where the clip has them there.
    if (was_scrubbing && !clip_scrubbing_ && c.edited >= 0 && c.playing < 0)
        Queue("clip.sample", Args({{"time", json::Value::Float(c.playhead)}, {"apply", json::Value::Bool(true)}}));
    if (!open) Queue("clip.editor", Args({{"on", json::Value::Bool(false)}}));
}

void ImGuiLayer::DrawDopeSheet(float height) {
    ViewerUiState& ui = *frame_.ui;
    ClipView& c = ui.clips;
    const RobotView& v = ui.robot;
    Clip& clip = c.clips[static_cast<size_t>(c.edited)];
    const json::Value name = json::Value::Str(clip.name);
    const ImGuiIO& io = ImGui::GetIO();
    const float s = nui::Scale();
    const float row = ImGui::GetFrameHeight(), r = 5.0f * s;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    const float x0 = origin.x + 150.0f * s, x1 = origin.x + width - ImGui::GetStyle().ScrollbarSize - 8.0f * s;
    const float px = (x1 - x0) / clip.duration;
    const auto x_of = [&](float t) { return x0 + t * px; };
    const auto t_of = [&](float x) { return std::clamp(Snap((x - x0) / px, c.dt), 0.0f, clip.duration); };
    const float step = RulerStep(px, 56.0f * s);
    const int ticks = static_cast<int>(std::floor(clip.duration / step + 1e-3f));
    const char* const tick_format = step < 0.1f ? "%.2f" : step < 1.0f ? "%.1f" : "%.0f";

    // The ruler: pressed or dragged, it moves the playhead.
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGui::InvisibleButton("##ruler", ImVec2(width, row));
    if (ImGui::IsItemActive()) {
        c.playhead = t_of(io.MousePos.x);
        clip_scrubbing_ = true;
    }
    dl->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + row), Col(theme::kBg2));
    for (int i = 0; i <= ticks; ++i) {
        const float t = static_cast<float>(i) * step, x = x_of(t);
        dl->AddLine(ImVec2(x, origin.y + row * 0.55f), ImVec2(x, origin.y + row), Col(theme::kLineHi), s);
        dl->AddText(ImVec2(x + 3.0f * s, origin.y + 2.0f * s), Col(theme::kFgDim), Fmt(tick_format, static_cast<double>(t)).c_str());
    }
    dl->AddText(ImVec2(origin.x + 6.0f * s, origin.y + 2.0f * s), Col(theme::kFgDim), Tr("Joint"));

    const size_t n = v.joints.size() + 1u;
    ImGui::BeginChild("##dope_rows", ImVec2(width, height - row), ImGuiChildFlags_None);
    ImDrawList* rl = ImGui::GetWindowDrawList();
    const ImVec2 top = ImGui::GetCursorScreenPos();
    const float canvas_h = std::max(static_cast<float>(n) * row, ImGui::GetContentRegionAvail().y);
    ImGui::InvisibleButton("##dope_canvas", ImVec2(width - ImGui::GetStyle().ScrollbarSize, canvas_h),
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const bool hovered = ImGui::IsItemHovered(), active = ImGui::IsItemActive();
    const ImVec2 mouse = io.MousePos;
    const int hover_row = static_cast<int>(std::floor((mouse.y - top.y) / row));
    const auto joint_of = [&](int i) -> const std::string& { return v.joints[static_cast<size_t>(i - 1)].name; };

    // The key under the pointer: its row (0 the summary) and time.
    int hit_row = -1;
    float hit_time = 0.0f;
    if (hovered && hover_row >= 0 && hover_row < static_cast<int>(n)) {
        float best = r + 3.0f * s;
        for (const ClipTrack& t : clip.tracks) {
            if (hover_row > 0 && t.joint != joint_of(hover_row)) continue;
            for (const ClipKey& k : t.keys) {
                const float d = std::abs(x_of(k.time) - mouse.x);
                if (d >= best) continue;
                best = d;
                hit_row = hover_row;
                hit_time = k.time;
            }
        }
    }
    // A key of the summary row stands for every key at its time.
    const auto refs_of = [&](int i, float time) {
        std::vector<ClipKeyRef> refs;
        for (const ClipTrack& t : clip.tracks)
            if (i == 0 || t.joint == joint_of(i))
                for (const ClipKey& k : t.keys)
                    if (std::abs(k.time - time) < kClipKeyEpsilon) refs.push_back(ClipKeyRef{t.joint, k.time});
        return refs;
    };
    const auto all_selected = [&](const std::vector<ClipKeyRef>& refs) {
        return std::all_of(refs.begin(), refs.end(), [&](const ClipKeyRef& k) { return Selected(c, k.joint, k.time); });
    };
    const auto add = [&](const std::vector<ClipKeyRef>& refs) {
        for (const ClipKeyRef& k : refs)
            if (!Selected(c, k.joint, k.time)) c.selected.push_back(k);
    };

    // A press on a key selects it (Shift adds, Ctrl toggles) and starts moving the selection; on
    // a joint's name it shows that joint's curve; elsewhere it starts a box selection.
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        const bool extend = io.KeyCtrl || io.KeyShift;
        if (hit_row >= 0) {
            const std::vector<ClipKeyRef> refs = refs_of(hit_row, hit_time);
            const bool was = all_selected(refs);
            if (io.KeyCtrl && was) {
                std::erase_if(c.selected, [&](const ClipKeyRef& k) {
                    return std::any_of(refs.begin(), refs.end(), [&](const ClipKeyRef& h) {
                        return h.joint == k.joint && std::abs(h.time - k.time) < kClipKeyEpsilon;
                    });
                });
            } else {
                if (!extend && !was) c.selected.clear();
                add(refs);
            }
            if (hit_row > 0) c.curve = joint_of(hit_row);
            clip_dragging_ = !(io.KeyCtrl && was);
            clip_drag_base_ = clip;
            clip_drag_keys_ = c.selected;
            clip_drag_x_ = mouse.x;
        } else if (mouse.x < x0 - r && hover_row > 0 && hover_row < static_cast<int>(n)) {
            c.curve = joint_of(hover_row);
        } else {
            if (!extend) c.selected.clear();
            clip_boxing_ = true;
            clip_box_from_ = mouse;
        }
    }
    // A double-click on a joint's row keys it there at the value its curve has.
    if (hovered && hit_row < 0 && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && mouse.x >= x0 - r &&
        hover_row > 0 && hover_row < static_cast<int>(n)) {
        const RobotJoint& j = v.joints[static_cast<size_t>(hover_row - 1)];
        const ClipTrack* t = FindTrack(clip, j.name);
        const float time = t_of(mouse.x);
        const float value = t != nullptr ? SampleTrack(*t, time) : ui.drive_targets[j.link];
        Queue("clip.key", Args({{"clip", name},
                                {"joint", json::Value::Str(j.name)},
                                {"time", json::Value::Float(time)},
                                {"value", json::Value::Float(value)}}));
    }
    const float shift = Snap((mouse.x - clip_drag_x_) / px, c.dt);
    if (active && clip_dragging_) {
        clip = clip_drag_base_;
        std::vector<ClipKeyRef> moved = clip_drag_keys_;
        MoveKeys(&clip, &moved, shift, 0.0f);
        c.selected = std::move(moved);
    }
    const ImVec2 box_a(std::min(clip_box_from_.x, mouse.x), std::min(clip_box_from_.y, mouse.y));
    const ImVec2 box_b(std::max(clip_box_from_.x, mouse.x), std::max(clip_box_from_.y, mouse.y));
    if (ImGui::IsItemDeactivated()) {
        if (clip_dragging_ && shift != 0.0f) c.commit = true;
        if (clip_boxing_) {
            for (size_t i = 0; i < n; ++i) {
                const float cy = top.y + (static_cast<float>(i) + 0.5f) * row;
                if (cy < box_a.y || cy > box_b.y) continue;
                for (const ClipTrack& t : clip.tracks)
                    if (i == 0 || t.joint == joint_of(static_cast<int>(i)))
                        for (const ClipKey& k : t.keys)
                            if (x_of(k.time) >= box_a.x && x_of(k.time) <= box_b.x) add({ClipKeyRef{t.joint, k.time}});
            }
        }
        clip_dragging_ = false;
        clip_boxing_ = false;
    }
    // Right-clicked, a key is selected and the selection's menu opens.
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
        if (hit_row >= 0) {
            const std::vector<ClipKeyRef> refs = refs_of(hit_row, hit_time);
            if (!all_selected(refs)) c.selected = refs;
        }
        ImGui::OpenPopup("##key_menu");
    }
    if (ImGui::BeginPopup("##key_menu")) {
        ImGui::BeginDisabled(c.selected.empty());
        const char* const shapes[] = {"Step", "Linear", "Bezier"};
        for (int i = 0; i < 3; ++i)
            if (ImGui::MenuItem(Tr(shapes[i])))
                Queue("clip.key", Args({{"clip", name}, {"keys", ShapedKeys(c, &clip, static_cast<ClipInterp>(i))}}));
        ImGui::Separator();
        if (ImGui::MenuItem(Iconed(icon::kCopy, Tr("Copy")).c_str(), "Ctrl+C")) CopyKeys(&c, &clip);
        ImGui::EndDisabled();
        if (ImGui::MenuItem(Iconed(icon::kPaste, Tr("Paste")).c_str(), "Ctrl+V", false, !c.clipboard.empty()))
            Queue("clip.key", Args({{"clip", name}, {"keys", PastedKeys(c, clip)}}));
        if (ImGui::MenuItem(Iconed(icon::kTrash, Tr("Delete")).c_str(), "Delete", false, !c.selected.empty()))
            Queue("clip.remove_keys", Args({{"clip", name}, {"keys", KeyRefs(c.selected)}}));
        ImGui::EndPopup();
    }

    // Rows, the ruler's grid, names, keys, then the playhead and the box over them.
    const float bottom = top.y + canvas_h;
    for (size_t i = 0; i < n; ++i) {
        const float y0 = top.y + static_cast<float>(i) * row, y1 = y0 + row;
        const bool curve = i > 0 && joint_of(static_cast<int>(i)) == c.curve;
        const ImVec4 fill = i == 0 ? theme::kBg2 : curve ? theme::Alpha(theme::kAccent, 0.10f) : theme::Alpha(theme::kBg2, i % 2u == 0u ? 0.55f : 0.0f);
        rl->AddRectFilled(ImVec2(top.x, y0), ImVec2(top.x + width, y1), Col(fill));
    }
    for (int i = 0; i <= ticks; ++i) {
        const float x = x_of(static_cast<float>(i) * step);
        rl->AddLine(ImVec2(x, top.y), ImVec2(x, bottom), Col(theme::Alpha(theme::kLine, 0.8f)), s);
    }
    rl->AddRectFilled(ImVec2(x_of(clip.duration), top.y), ImVec2(top.x + width, bottom), Col(theme::Alpha(theme::kBg0, 0.5f)));
    const float text_dy = (row - ImGui::GetTextLineHeight()) * 0.5f;
    for (size_t i = 0; i < n; ++i) {
        const float y0 = top.y + static_cast<float>(i) * row, cy = y0 + row * 0.5f;
        const bool curve = i > 0 && joint_of(static_cast<int>(i)) == c.curve;
        rl->PushClipRect(ImVec2(top.x, y0), ImVec2(x0 - r - 2.0f * s, y0 + row), true);
        rl->AddText(ImVec2(top.x + 6.0f * s, y0 + text_dy), Col(i == 0 ? theme::kFgMuted : curve ? theme::kAccent : theme::kFg),
                    i == 0 ? Tr("Summary") : joint_of(static_cast<int>(i)).c_str());
        rl->PopClipRect();
        if (i == 0) {
            std::vector<float> times;
            for (const ClipTrack& t : clip.tracks)
                for (const ClipKey& k : t.keys) times.push_back(k.time);
            std::sort(times.begin(), times.end());
            times.erase(std::unique(times.begin(), times.end(), [](float a, float b) { return b - a < kClipKeyEpsilon; }), times.end());
            for (const float t : times) {
                const bool sel = all_selected(refs_of(0, t));
                KeyMark(rl, ImVec2(x_of(t), cy), r, ClipInterp::Linear, Col(sel ? theme::kAccent : theme::kFgMuted),
                        Col(theme::kBg0));
            }
            continue;
        }
        const ClipTrack* t = FindTrack(clip, joint_of(static_cast<int>(i)));
        if (t == nullptr) continue;
        for (const ClipKey& k : t->keys) {
            const bool sel = Selected(c, t->joint, k.time);
            KeyMark(rl, ImVec2(x_of(k.time), cy), r, k.interp, Col(sel ? theme::kAccent : theme::kFg), Col(theme::kBg0));
        }
    }
    const float head = x_of(c.playhead);
    rl->AddLine(ImVec2(head, top.y), ImVec2(head, bottom), Col(theme::kAccent), 1.5f * s);
    if (clip_boxing_ && active) {
        rl->AddRectFilled(box_a, box_b, Col(theme::Alpha(theme::kAccent, 0.10f)));
        rl->AddRect(box_a, box_b, Col(theme::kAccent), 0.0f, 0, s);
    }
    ImGui::EndChild();
    dl->AddTriangleFilled(ImVec2(head - 5.0f * s, origin.y), ImVec2(head + 5.0f * s, origin.y), ImVec2(head, origin.y + 7.0f * s),
                          Col(theme::kAccent));
    dl->AddLine(ImVec2(head, origin.y), ImVec2(head, origin.y + row), Col(theme::kAccent), 1.5f * s);
    // Auto-key frames the sheet red, as recording does.
    if (c.auto_key)
        dl->AddRect(origin, ImVec2(origin.x + width, origin.y + height), Col(theme::Alpha(theme::kErr, 0.8f)), 0.0f, 0, 1.5f * s);
}

void ImGuiLayer::DrawClipCurves() {
    ViewerUiState& ui = *frame_.ui;
    ClipView& c = ui.clips;
    const RobotView& v = ui.robot;
    Clip& clip = c.clips[static_cast<size_t>(c.edited)];
    const float s = nui::Scale();
    int j = FindJoint(v.joints, c.curve);
    if (j < 0) {
        j = 0;
        for (const ClipTrack& t : clip.tracks)
            if (const int k = FindJoint(v.joints, t.joint); k >= 0) {
                j = k;
                break;
            }
        c.curve = v.joints[static_cast<size_t>(j)].name;
    }
    const RobotJoint& joint = v.joints[static_cast<size_t>(j)];
    const auto scale_of = [&](const RobotJoint& o) { return o.prismatic || v.radians ? 1.0f : kDegrees; };
    const float scale = scale_of(joint);
    ClipTrack* track = FindTrack(&clip, joint.name);
    const bool was_held = clip_curve_held_;
    clip_curve_held_ = false;
    std::vector<float> xs(kCurveSamples);
    for (int i = 0; i < kCurveSamples; ++i)
        xs[static_cast<size_t>(i)] = clip.duration * static_cast<float>(i) / static_cast<float>(kCurveSamples - 1);
    // Every keyed joint's samples, and the value range they and the curve joint's handles span.
    std::vector<std::pair<const ClipTrack*, std::vector<float>>> lines;
    float ymin = FLT_MAX, ymax = -FLT_MAX;
    const auto span = [&](float y) {
        ymin = std::min(ymin, y);
        ymax = std::max(ymax, y);
    };
    for (const ClipTrack& t : clip.tracks) {
        const int o = FindJoint(v.joints, t.joint);
        if (o < 0) continue;
        const float sc = scale_of(v.joints[static_cast<size_t>(o)]);
        std::vector<float>& ys = lines.emplace_back(&t, std::vector<float>(xs.size())).second;
        for (size_t i = 0; i < xs.size(); ++i) span(ys[i] = SampleTrack(t, xs[i]) * sc);
    }
    if (track != nullptr)
        for (size_t i = 0; i < track->keys.size(); ++i) {
            const ClipKey& k = track->keys[i];
            if (i + 1u < track->keys.size() && k.interp == ClipInterp::Bezier)
                span((k.value + k.out_slope * (track->keys[i + 1u].time - k.time) / 3.0f) * scale);
            if (i > 0u && track->keys[i - 1u].interp == ClipInterp::Bezier)
                span((k.value - k.in_slope * (k.time - track->keys[i - 1u].time) / 3.0f) * scale);
        }
    if (ymin > ymax) ymin = ymax = 0.0f;
    const float pad = std::max(0.08f * (ymax - ymin), 0.1f * scale);
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(theme::kSpace * s, theme::kSpace * s));
    if (ImPlot::BeginPlot("##clip_curves", ImVec2(-FLT_MIN, -FLT_MIN), ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoFrame)) {
        ImPlot::SetupAxes(Tr("s"), joint.prismatic ? "m" : v.radians ? "rad" : "\xC2\xB0");
        const double margin = 0.02 * static_cast<double>(clip.duration);
        ImPlot::SetupAxisLimits(ImAxis_X1, -margin, static_cast<double>(clip.duration) + margin, ImPlotCond_Always);
        // The values hold still while a point is dragged.
        if (!was_held) ImPlot::SetupAxisLimits(ImAxis_Y1, ymin - pad, ymax + pad, ImPlotCond_Always);
        ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);
        // The other keyed joints faintly under the curve joint.
        for (const auto& [t, ys] : lines) {
            if (t == track) continue;
            ImPlotSpec faint = Line(theme::Alpha(theme::kFgDim, 0.6f), 1.0f);
            faint.Flags = ImPlotItemFlags_NoLegend;
            ImPlot::PlotLine(("##" + t->joint).c_str(), xs.data(), ys.data(), kCurveSamples, faint);
        }
        if (track != nullptr) {
            const auto own = std::find_if(lines.begin(), lines.end(), [&](const auto& l) { return l.first == track; });
            ImPlot::PlotLine(joint.name.c_str(), xs.data(), own->second.data(), kCurveSamples, Line(theme::kAccent, 2.0f));
            std::vector<ClipKey>& keys = track->keys;
            ImDrawList* pl = ImPlot::GetPlotDrawList();
            ImPlot::PushPlotClipRect();
            for (size_t i = 0; i < keys.size(); ++i) {
                ClipKey& k = keys[i];
                const int id = static_cast<int>(i) * 3;
                bool held = false;
                // Bezier handles sit a third of the way to each neighbour along the key's slopes.
                if (i + 1u < keys.size() && k.interp == ClipInterp::Bezier) {
                    const float h = (keys[i + 1u].time - k.time) / 3.0f;
                    double hx = static_cast<double>(k.time + h), hy = static_cast<double>((k.value + k.out_slope * h) * scale);
                    pl->AddLine(ImPlot::PlotToPixels(k.time, k.value * scale), ImPlot::PlotToPixels(hx, hy), Col(theme::kInfo), s);
                    if (ImPlot::DragPoint(id + 1, &hx, &hy, theme::kInfo, 4.0f * s, ImPlotDragToolFlags_NoFit, nullptr, nullptr, &held)) {
                        k.out_slope = (static_cast<float>(hy) / scale - k.value) / h;
                        clip_curve_changed_ = true;
                    }
                    clip_curve_held_ = clip_curve_held_ || held;
                }
                if (i > 0u && keys[i - 1u].interp == ClipInterp::Bezier) {
                    const float h = (k.time - keys[i - 1u].time) / 3.0f;
                    double hx = static_cast<double>(k.time - h), hy = static_cast<double>((k.value - k.in_slope * h) * scale);
                    pl->AddLine(ImPlot::PlotToPixels(k.time, k.value * scale), ImPlot::PlotToPixels(hx, hy), Col(theme::kInfo), s);
                    if (ImPlot::DragPoint(id + 2, &hx, &hy, theme::kInfo, 4.0f * s, ImPlotDragToolFlags_NoFit, nullptr, nullptr, &held)) {
                        k.in_slope = (k.value - static_cast<float>(hy) / scale) / h;
                        clip_curve_changed_ = true;
                    }
                    clip_curve_held_ = clip_curve_held_ || held;
                }
                // A key moves between its neighbours on the step grid.
                const float lo = i > 0u ? keys[i - 1u].time + 2.0f * kClipKeyEpsilon : 0.0f;
                const float hi = i + 1u < keys.size() ? keys[i + 1u].time - 2.0f * kClipKeyEpsilon : clip.duration;
                const bool sel = Selected(c, joint.name, k.time);
                double x = static_cast<double>(k.time), y = static_cast<double>(k.value * scale);
                bool clicked = false;
                if (ImPlot::DragPoint(id, &x, &y, sel ? theme::kAccent : theme::kFg, 5.0f * s, ImPlotDragToolFlags_None, &clicked,
                                      nullptr, &held)) {
                    const float before = k.time;
                    k.time = std::clamp(Snap(static_cast<float>(x), c.dt), lo, hi);
                    k.value = static_cast<float>(y) / scale;
                    for (ClipKeyRef& ref : c.selected)
                        if (ref.joint == joint.name && std::abs(ref.time - before) < kClipKeyEpsilon) ref.time = k.time;
                    clip_curve_changed_ = true;
                }
                if (clicked) {
                    if (!ImGui::GetIO().KeyCtrl && !ImGui::GetIO().KeyShift) c.selected.clear();
                    if (!Selected(c, joint.name, k.time)) c.selected.push_back(ClipKeyRef{joint.name, k.time});
                }
                clip_curve_held_ = clip_curve_held_ || held;
            }
            ImPlot::PopPlotClipRect();
        }
        double head = static_cast<double>(c.playhead);
        bool scrub = false;
        if (ImPlot::DragLineX(kPlayheadId, &head, theme::kAccent, 1.5f * s, ImPlotDragToolFlags_NoFit, nullptr, nullptr, &scrub))
            c.playhead = std::clamp(Snap(static_cast<float>(head), c.dt), 0.0f, clip.duration);
        clip_scrubbing_ = clip_scrubbing_ || scrub;
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    // A finished drag that moved something is one undo point.
    if (was_held && !clip_curve_held_ && clip_curve_changed_) c.commit = true;
    if (!clip_curve_held_) clip_curve_changed_ = false;
}

void ImGuiLayer::ClipKeys() {
    ClipView& c = frame_.ui->clips;
    const ImGuiIO& io = ImGui::GetIO();
    if (!c.focused || io.WantTextInput) return;
    Clip& clip = c.clips[static_cast<size_t>(c.edited)];
    const json::Value name = json::Value::Str(clip.name);
    if (ImGui::IsKeyPressed(ImGuiKey_Delete, false) && !c.selected.empty())
        Queue("clip.remove_keys", Args({{"clip", name}, {"keys", KeyRefs(c.selected)}}));
    if (!io.KeyCtrl) return;
    if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) Queue(io.KeyShift ? "clip.redo" : "clip.undo");
    if (ImGui::IsKeyPressed(ImGuiKey_Y, false)) Queue("clip.redo");
    if (ImGui::IsKeyPressed(ImGuiKey_C, false) && !c.selected.empty()) CopyKeys(&c, &clip);
    if (ImGui::IsKeyPressed(ImGuiKey_V, false) && !c.clipboard.empty()) {
        json::Value keys = PastedKeys(c, clip);
        if (keys.Size() > 0u) Queue("clip.key", Args({{"clip", name}, {"keys", std::move(keys)}}));
    }
    if (ImGui::IsKeyPressed(ImGuiKey_A, false)) {
        c.selected.clear();
        for (const ClipTrack& t : clip.tracks)
            for (const ClipKey& k : t.keys) c.selected.push_back(ClipKeyRef{t.joint, k.time});
    }
}

}  // namespace nuka::runtime::app::viewer
