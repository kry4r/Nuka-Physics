// The Timeline panel: the scrubber over the current branch's recorded ticks with its keyframes
// and forks, the transport of the history, the branch list with the ghost, keyframe memory, and
// saving, loading and exporting the history.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/shell_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

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
using namespace shell;

namespace {

constexpr uint32_t kIntervals[] = {4u, 16u, 64u};
constexpr double kBudgetsMb[] = {256.0, 1024.0, 4096.0};

std::string BranchName(const TimelineBranchInfo& b) {
    if (!b.label.empty()) return b.label;
    return b.id == 0u ? std::string(Tr("main")) : Fmt(Tr("branch %u"), b.id);
}

const TimelineBranchInfo* Find(const TimelineInfo& t, uint32_t id) {
    const auto it = std::find_if(t.branches.begin(), t.branches.end(), [&](const TimelineBranchInfo& b) { return b.id == id; });
    return it == t.branches.end() ? nullptr : &*it;
}

std::string Ticks(uint64_t tick) { return Fmt("%llu", static_cast<unsigned long long>(tick)); }

}  // namespace

void ImGuiLayer::BuildTimeline(nui::VStack& root) {
    const auto has_scene = [this] { return frame_.ui->has_scene; };
    const auto view = [this]() -> const TimelineInfo& { return frame_.ui->timeline; };
    const auto exact = [view] {
        const TimelineBranchInfo* b = Find(view(), view().branch);
        return b == nullptr || b->exact;
    };

    auto& hero = root.Add<nui::HStack>();
    auto& tick = hero.Add<nui::Label>(std::string(), Tone::Accent, TextStyle::Hero);
    tick.text_fn = [view] { return Ticks(view().tick); };
    auto& head = hero.Add<nui::Label>(std::string(), Tone::Dim);
    head.text_fn = [view] { return Fmt(Tr("/ %llu ticks"), static_cast<unsigned long long>(view().head)); };
    auto& mode = hero.Add<nui::Badge>();
    mode.text_fn = [view] { return std::string(Tr(view().tick < view().head ? "REPLAY" : "LIVE")); };
    mode.tone_fn = [view] { return view().tick < view().head ? Tone::Info : Tone::Accent; };
    auto& fidelity = hero.Add<nui::Badge>();
    fidelity.text_fn = [exact] { return std::string(Tr(exact() ? "EXACT" : "APPROX")); };
    fidelity.tone_fn = [exact] { return exact() ? Tone::Ok : Tone::Warn; };
    fidelity.tooltip = "Exact: replay repeats the recording bit for bit. Approx: a controller or script ran, and "
                       "replay snaps back to each keyframe.";
    hero.visible_if = has_scene;

    auto& transport = root.Add<nui::HStack>();
    transport.Add<nui::IconButton>(icon::kSkipBack, "To start")
        .Action("timeline.seek", Args({{"to", json::Value::Str("start")}}));
    transport.Add<nui::IconButton>(icon::kCaretLeft, "Back one tick")
        .Action("timeline.seek", Args({{"delta", json::Value::Int(-1)}}));
    transport.Add<nui::IconButton>(icon::kChevronRight, "Forward one tick")
        .Action("timeline.seek", Args({{"delta", json::Value::Int(1)}}));
    transport.Add<nui::IconButton>(icon::kStep, "To the head").Action("timeline.seek", Args({{"to", json::Value::Str("head")}}));
    transport.Add<nui::Button>("Fork", icon::kFork).Action("timeline.fork");
    transport.Add<nui::ToggleButton>("Ghost").Action("timeline.ghost");
    transport.Add<nui::IconButton>(icon::kSave, "Save the recording").Action("timeline.save");
    transport.Add<nui::IconButton>(icon::kOpen, "Load a recording").Action("timeline.load");
    transport.Add<nui::IconButton>(icon::kFilm, "Export an MP4 video").Action("timeline.export");
    transport.visible_if = has_scene;
    auto& progress = root.Add<nui::ProgressBar>(nui::Bind<float>([this] { return frame_.ui->export_done; }));
    progress.label = "Exporting video";
    progress.visible_if = [this] { return frame_.ui->exporting; };

    // The scrubber: dragging seeks, so the world replays to the tick under the pointer.
    root.Add<nui::Custom>([this, view] {
        const TimelineInfo& t = view();
        const float s = nui::Scale();
        const float w = std::max(ImGui::GetContentRegionAvail().x, 1.0f), h = 46.0f * s;
        const ImVec2 p0 = ImGui::GetCursorScreenPos(), p1(p0.x + w, p0.y + h);
        ImGui::InvisibleButton("##scrub", ImVec2(w, h));
        const bool live = frame_.ui->has_scene && t.head > t.start;
        const float pad = 12.0f * s, x0 = p0.x + pad, x1 = std::max(p1.x - pad, x0 + 1.0f), mid = p0.y + h * 0.42f;
        const double span = static_cast<double>(std::max<uint64_t>(t.head - t.start, 1u));
        const auto x_of = [&](uint64_t at) {
            return x0 + static_cast<float>((static_cast<double>(at) - static_cast<double>(t.start)) / span) * (x1 - x0);
        };
        const auto tick_at = [&](float x) {
            return t.start + static_cast<uint64_t>(std::llround(std::clamp((x - x0) / (x1 - x0), 0.0f, 1.0f) * span));
        };
        ImDrawList* list = ImGui::GetWindowDrawList();
        draw::Chamfer(list, p0, p1, theme::kChamfer * s, U32(theme::kBg0), U32(ImGui::IsItemHovered() ? theme::kLineHi : theme::kLine));
        list->AddLine(ImVec2(x0, mid), ImVec2(x1, mid), U32(theme::kLineHi), 2.0f * s);
        const float px = x_of(t.tick);
        list->AddLine(ImVec2(x0, mid), ImVec2(px, mid), U32(theme::kAccentDim), 2.0f * s);
        for (const uint64_t k : t.keys) list->AddLine(ImVec2(x_of(k), mid - 4.0f * s), ImVec2(x_of(k), mid + 4.0f * s), U32(theme::kFgDim), s);
        for (const uint64_t k : t.pinned) {
            const float x = x_of(k), r = 4.0f * s;
            list->AddQuadFilled(ImVec2(x, mid - r), ImVec2(x + r, mid), ImVec2(x, mid + r), ImVec2(x - r, mid), U32(theme::kWarn));
        }
        // Forks: where this branch left its parent, and where its children leave it.
        const TimelineBranchInfo* current = Find(t, t.branch);
        for (const TimelineBranchInfo& b : t.branches) {
            const bool own = &b == current && b.parent != kNoBranch;
            if (!own && b.parent != t.branch) continue;
            if (b.fork_tick < t.start || b.fork_tick > t.head) continue;
            const float x = x_of(b.fork_tick), y = mid - 9.0f * s, r = 4.0f * s;
            list->AddTriangleFilled(ImVec2(x - r, y - r), ImVec2(x + r, y - r), ImVec2(x, y + r),
                                    U32(own ? theme::kAccent : theme::kInfo));
        }
        list->AddLine(ImVec2(px, p0.y + 5.0f * s), ImVec2(px, mid + 8.0f * s), U32(theme::kAccent), 2.0f * s);
        list->AddCircleFilled(ImVec2(px, mid), 4.5f * s, U32(theme::kAccent));
        ImFont* font = GetNukaFont(FontRole::Mono);
        const float fpx = theme::kFontLabel * s, ty = p1.y - fpx - 5.0f * s;
        const std::string start = Ticks(t.start), end = Ticks(t.head);
        list->AddText(font, fpx, ImVec2(x0, ty), U32(theme::kFgMuted), start.c_str());
        const float end_w = font->CalcTextSizeA(fpx, FLT_MAX, 0.0f, end.c_str()).x;
        list->AddText(font, fpx, ImVec2(x1 - end_w, ty), U32(theme::kFgMuted), end.c_str());
        if (!live) return;
        if (ImGui::IsItemActive()) {
            const uint64_t want = tick_at(ImGui::GetIO().MousePos.x);
            if (want != scrub_sent_) {
                scrub_sent_ = want;
                Queue("timeline.seek", Args({{"tick", json::Value::Int(static_cast<int64_t>(want))}}));
            }
        } else {
            scrub_sent_ = ~0ull;
        }
        if (ImGui::IsItemHovered() && !ImGui::IsItemActive())
            ImGui::SetTooltip("%s", Fmt(Tr("tick %llu"), static_cast<unsigned long long>(tick_at(ImGui::GetIO().MousePos.x))).c_str());
    }).visible_if = has_scene;

    root.Add<nui::Separator>("Branches").visible_if = has_scene;
    root.Add<nui::Custom>([this, view] {
        const TimelineInfo& t = view();
        const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp;
        if (!ImGui::BeginTable("##branches", 5, flags)) return;
        ImGui::TableSetupColumn(Tr("branch"), ImGuiTableColumnFlags_WidthStretch, 3.0f);
        ImGui::TableSetupColumn(Tr("fork"));
        ImGui::TableSetupColumn(Tr("head"));
        ImGui::TableSetupColumn(Tr("replay"));
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableHeadersRow();
        for (const TimelineBranchInfo& b : t.branches) {
            const bool current = b.id == t.branch;
            const bool ghosted = t.ghost_on && b.id == t.ghost && !current;
            ImGui::PushID(static_cast<int>(b.id));
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const std::string name = Iconed(icon::kBranch, BranchName(b).c_str());
            ImGui::PushStyleColor(ImGuiCol_Text, current ? theme::kAccent : theme::kFg);
            if (ImGui::Selectable(name.c_str(), current) && !current)
                Queue("timeline.branch", Args({{"id", json::Value::Int(b.id)}}));
            ImGui::PopStyleColor();
            if (ImGui::BeginPopupContextItem("##rename")) {
                if (ImGui::IsWindowAppearing()) {
                    const std::string now = BranchName(b);
                    std::snprintf(branch_name_, sizeof(branch_name_), "%s", now.c_str());
                    ImGui::SetKeyboardFocusHere();
                }
                if (ImGui::InputText("##name", branch_name_, sizeof(branch_name_), ImGuiInputTextFlags_EnterReturnsTrue)) {
                    Queue("timeline.rename", Args({{"id", json::Value::Int(b.id)}, {"label", json::Value::Str(branch_name_)}}));
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(b.parent == kNoBranch ? "-" : Ticks(b.fork_tick).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(Ticks(b.head).c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(b.exact ? theme::kOk : theme::kWarn, "%s", Tr(b.exact ? "exact" : "approx"));
            ImGui::TableNextColumn();
            ImGui::BeginDisabled(current);
            if (draw::IconButton("##ghost", icon::kGhost, ghosted, Tr("Show as the ghost")))
                Queue("timeline.ghost", Args({{"on", json::Value::Bool(true)}, {"branch", json::Value::Int(b.id)}}));
            ImGui::EndDisabled();
            ImGui::PopID();
        }
        ImGui::EndTable();
    }).visible_if = has_scene;

    root.Add<nui::Separator>("Keyframes").visible_if = has_scene;
    const auto stat = [&root, has_scene](const char* label, std::function<std::string()> fn) -> nui::Stat& {
        auto& s = root.Add<nui::Stat>(label, std::move(fn));
        s.visible_if = has_scene;
        return s;
    };
    stat("kept", [view] { return Fmt("%u", view().keyframes); });
    stat("memory", [view] {
        return Fmt("%.1f / %.0f MB", static_cast<double>(view().bytes) / 1048576.0, static_cast<double>(view().budget) / 1048576.0);
    }).tone_fn = [view] { return view().bytes * 10u > view().budget * 9u ? Tone::Warn : Tone::Normal; };
    std::vector<std::string> intervals;
    for (const uint32_t n : kIntervals) intervals.push_back(Fmt(Tr("every %u"), n));
    auto& interval = root.Add<nui::Segmented>(
        intervals, nui::Bind<int>(
                       [view] {
                           const auto it = std::find(std::begin(kIntervals), std::end(kIntervals), view().interval);
                           return it == std::end(kIntervals) ? -1 : static_cast<int>(it - std::begin(kIntervals));
                       },
                       [this](const int& i) {
                           Queue("timeline.configure", Args({{"interval", json::Value::Int(kIntervals[static_cast<size_t>(i)])}}));
                       }));
    interval.tooltip = "Ticks between keyframes: fewer seek faster, more keep a longer history in the budget.";
    interval.visible_if = has_scene;
    auto& budget = root.Add<nui::Segmented>(
        std::vector<std::string>{"256 MB", "1 GB", "4 GB"},
        nui::Bind<int>(
            [view] {
                const double mb = static_cast<double>(view().budget) / 1048576.0;
                for (size_t i = 0; i < std::size(kBudgetsMb); ++i)
                    if (std::fabs(mb - kBudgetsMb[i]) < 0.5) return static_cast<int>(i);
                return -1;
            },
            [this](const int& i) {
                Queue("timeline.configure", Args({{"budget_mb", json::Value::Float(kBudgetsMb[static_cast<size_t>(i)])}}));
            }));
    budget.tooltip = "Keyframe memory; past it the oldest detail thins out first.";
    budget.visible_if = has_scene;
    root.Add<nui::Button>("Clear history", icon::kTrash).Action("timeline.clear").visible_if = has_scene;
}

}  // namespace nuka::runtime::app::viewer
