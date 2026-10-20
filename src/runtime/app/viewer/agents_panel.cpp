// The Agents panel: who reaches the editor from outside and what they may do, the proposals and
// approvals waiting on the user, the recent agent calls, and the dialog asking about an approval.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/shell_util.hpp"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <ctime>
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
using namespace shell;

namespace {

struct ModeSpec {
    const char* mode;
    const char* label;
    const char* badge;
    const char* text;
};
constexpr ModeSpec kModes[] = {
    {"off", "Off", "OFF", "No program can reach the editor."},
    {"read", "Read", "READ", "Agents read the editor and move the view; anything else is refused."},
    {"propose", "Propose", "PROPOSE",
     "Agents read, move the view and simulate. Their edits wait for you to accept them; files, scripts, "
     "programs and editor settings wait for your approval."},
    {"full", "Full", "FULL", "Agents run every action except those on the deny list."},
};

const ModeSpec& ModeOf(const std::string& mode) {
    const auto it =
        std::find_if(std::begin(kModes), std::end(kModes), [&](const ModeSpec& m) { return mode == m.mode; });
    return it != std::end(kModes) ? *it : kModes[3];
}

std::string ClockText(int64_t seconds) {
    const std::time_t when = static_cast<std::time_t>(seconds);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &when);
#else
    localtime_r(&when, &local);
#endif
    char text[16];
    std::strftime(text, sizeof(text), "%H:%M:%S", &local);
    return text;
}

// "sim.*, view.screenshot" as a JSON list of the ids.
json::Value Patterns(const char* text) {
    json::Value list = json::Value::Array();
    std::string item;
    for (const char* c = text;; ++c) {
        if (*c == ',' || *c == '\0') {
            const size_t a = item.find_first_not_of(" \t"), b = item.find_last_not_of(" \t");
            if (a != std::string::npos) list.PushBack(json::Value::Str(item.substr(a, b + 1u - a)));
            item.clear();
            if (*c == '\0') break;
        } else {
            item += *c;
        }
    }
    return list;
}

ImVec4 StateColor(const std::string& state) {
    if (state == "pending") return theme::kWarn;
    if (state == "accepted" || state == "approved") return theme::kOk;
    return state == "failed" ? theme::kErr : theme::kFgDim;
}

}  // namespace

void ImGuiLayer::BuildAgents(nui::VStack& root) {
    const auto agents = [this]() -> const AgentView& { return frame_.ui->agents; };

    auto& hero = root.Add<nui::HStack>();
    auto& count = hero.Add<nui::Label>(std::string(), Tone::Accent, TextStyle::Hero);
    count.text_fn = [agents] { return Fmt(Tr("%zu CONNECTED"), agents().clients.size()); };
    hero.Add<nui::Spacer>();
    auto& mode = hero.Add<nui::Badge>();
    mode.text_fn = [agents] { return std::string(Tr(ModeOf(agents().mode).badge)); };
    mode.tone_fn = [agents] {
        const std::string& m = agents().mode;
        return m == "off" ? Tone::Dim : m == "read" ? Tone::Info : m == "propose" ? Tone::Warn : Tone::Accent;
    };
    auto& pending = hero.Add<nui::Badge>();
    pending.text_fn = [agents] { return Fmt(Tr("%zu WAITING"), agents().pending); };
    pending.tone_fn = [] { return Tone::Warn; };
    pending.visible_if = [agents] { return agents().pending > 0u; };

    // The access mode and the lists, each list applied on Enter.
    root.Add<nui::Custom>([this] {
        AgentView& a = frame_.ui->agents;
        draw::SectionHeader(Tr("Access"));
        for (const ModeSpec& m : kModes) {
            if (ImGui::RadioButton(Tr(m.label), a.mode == m.mode) && a.mode != m.mode)
                Queue("agent.mode", Args({{"mode", json::Value::Str(m.mode)}}));
            if (&m != &kModes[3]) ImGui::SameLine();
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("%s", Tr(ModeOf(a.mode).text));
        ImGui::PopTextWrapPos();
        const char* hint = Tr("action ids, comma separated; * matches anything");
        ImGui::TextColored(theme::kFgMuted, "%s", Tr("Always allow"));
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputTextWithHint("##agent_allow", hint, a.allow_edit, sizeof(a.allow_edit),
                                     ImGuiInputTextFlags_EnterReturnsTrue))
            Queue("agent.mode", Args({{"allow", Patterns(a.allow_edit)}}));
        ImGui::TextColored(theme::kFgMuted, "%s", Tr("Never allow"));
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputTextWithHint("##agent_deny", hint, a.deny_edit, sizeof(a.deny_edit),
                                     ImGuiInputTextFlags_EnterReturnsTrue))
            Queue("agent.mode", Args({{"deny", Patterns(a.deny_edit)}}));
        if (ImGui::Button(Iconed(icon::kPlug, Tr("Disconnect All and Turn Off")).c_str()))
            Queue("agent.disconnect", Args({{"off", json::Value::Bool(true)}}));

        draw::SectionHeader(Tr("Connected"));
        if (a.clients.empty()) ImGui::TextDisabled("%s", Tr("No program is connected."));
        if (!a.clients.empty() &&
            ImGui::BeginTable("##agent_clients", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn(Tr("client"));
            ImGui::TableSetupColumn(Tr("pid"));
            ImGui::TableSetupColumn(Tr("since"));
            ImGui::TableSetupColumn(Tr("calls"));
            ImGui::TableHeadersRow();
            for (const AgentClientView& c : a.clients) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(theme::kAccent, "%s", Iconed(icon::kSparkle, c.name.c_str()).c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%lld", static_cast<long long>(c.pid));
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(ClockText(c.since).c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%llu", static_cast<unsigned long long>(c.calls));
            }
            ImGui::EndTable();
        }
    });

    // Each request with what it changes; pending ones take an answer.
    root.Add<nui::Custom>([this] {
        AgentView& a = frame_.ui->agents;
        draw::SectionHeader(Tr("Requests"));
        if (a.requests.empty()) ImGui::TextDisabled("%s", Tr("No requests"));
        for (AgentRequestView& r : a.requests) {
            ImGui::PushID(static_cast<int>(r.id));
            const bool open = r.state == "pending";
            const bool proposal = r.kind == "proposal";
            nui::PushTextStyle(TextStyle::Strong);
            ImGui::TextColored(open ? theme::kFg : theme::kFgMuted, "%s",
                               Iconed(proposal ? icon::kEdit : icon::kRun, r.label.c_str()).c_str());
            nui::PopTextStyle();
            ImGui::SameLine();
            ImGui::TextColored(StateColor(r.state), "%s", Tr(r.state.c_str()));
            ImGui::TextDisabled("#%llu  %s  %s  %s", static_cast<unsigned long long>(r.id), r.client.c_str(),
                                Tr(r.kind.c_str()), ClockText(r.time).c_str());
            ImGui::PushTextWrapPos(0.0f);
            if (!r.note.empty()) ImGui::TextColored(theme::kFgMuted, "%s", r.note.c_str());
            nui::PushTextStyle(TextStyle::Mono);
            for (const AgentChange& c : r.changes) {
                const std::string where = c.target.empty() ? c.field : c.target + "  " + c.field;
                ImGui::TextUnformatted(where.c_str());
                ImGui::SameLine();
                if (!c.before.empty()) {
                    ImGui::TextColored(theme::kFgDim, "%s", c.before.c_str());
                    ImGui::SameLine();
                }
                ImGui::TextColored(theme::kAccent, "\xE2\x86\x92 %s", c.after.c_str());
            }
            if (!proposal) ImGui::TextColored(theme::kFgDim, "%s %s", r.action.c_str(), r.args.c_str());
            nui::PopTextStyle();
            if (!r.error.empty()) ImGui::TextColored(theme::kErr, "%s", r.error.c_str());
            ImGui::PopTextWrapPos();
            if (open) {
                const json::Value id = json::Value::Int(static_cast<int64_t>(r.id));
                if (proposal && ImGui::Button(Iconed(icon::kOk, Tr("Accept")).c_str()))
                    Queue("agent.accept", Args({{"id", id}}));
                if (!proposal && ImGui::Button(Iconed(icon::kOk, Tr("Allow Once")).c_str()))
                    Queue("agent.approve", Args({{"request", id}}));
                ImGui::SameLine();
                if (ImGui::Button(Iconed(icon::kProhibit, Tr(proposal ? "Reject" : "Deny")).c_str()))
                    Queue("agent.reject", Args({{"id", id}}));
            }
            ImGui::Separator();
            ImGui::PopID();
        }

        draw::SectionHeader(Tr("Recent Calls"));
        if (a.calls.empty()) ImGui::TextDisabled("%s", Tr("No calls yet"));
        if (!a.calls.empty() &&
            ImGui::BeginTable("##agent_calls", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn(Tr("time"));
            ImGui::TableSetupColumn(Tr("client"));
            ImGui::TableSetupColumn(Tr("action"));
            ImGui::TableSetupColumn(Tr("outcome"));
            ImGui::TableHeadersRow();
            for (const AgentCallView& c : a.calls) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextDisabled("%s", ClockText(c.time).c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(c.client.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(c.action.c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(c.ok ? theme::kFgMuted : theme::kErr, "%s", c.outcome.c_str());
            }
            ImGui::EndTable();
        }
    });
}

void ImGuiLayer::DrawAgentApproval() {
    AgentView& a = frame_.ui->agents;
    const auto asked = std::find_if(a.requests.begin(), a.requests.end(), [](const AgentRequestView& r) {
        return r.kind == "approval" && r.state == "pending";
    });
    const std::string title = std::string(Tr("Agent Request")) + "###agent_approval";
    if (asked != a.requests.end() && !ImGui::IsPopupOpen(title.c_str())) ImGui::OpenPopup(title.c_str());
    const float s = nui::Scale();
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(480.0f * s, 0.0f), ImVec2(480.0f * s, FLT_MAX));
    if (!ImGui::BeginPopupModal(title.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize))
        return;
    if (asked == a.requests.end()) {
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        return;
    }
    ImDrawList* list = ImGui::GetWindowDrawList();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    draw::Glyph(list, ImVec2(at.x + 14.0f * s, at.y + 14.0f * s), icon::kSparkle, U32(theme::kWarn), true, 22.0f);
    ImGui::SetCursorScreenPos(ImVec2(at.x + 38.0f * s, at.y));
    ImGui::BeginGroup();
    ImGui::PushTextWrapPos(0.0f);
    nui::PushTextStyle(TextStyle::Strong);
    ImGui::TextUnformatted(Fmt(Tr("%s asks to run %s"), asked->client.c_str(), asked->action.c_str()).c_str());
    nui::PopTextStyle();
    ImGui::TextColored(theme::kFgMuted, "%s", Tr("It runs as soon as you allow it; Always Allow adds it to the list."));
    nui::PushTextStyle(TextStyle::Mono);
    ImGui::TextColored(theme::kFgDim, "%s", asked->args.c_str());
    nui::PopTextStyle();
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
    ImGui::Dummy(ImVec2(0.0f, 10.0f * s));

    const json::Value id = json::Value::Int(static_cast<int64_t>(asked->id));
    const float bw = 120.0f * s, spacing = ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - bw * 3.0f - spacing * 2.0f);
    bool answered = false;
    ImGui::PushStyleColor(ImGuiCol_Button, theme::kAccent);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, theme::Alpha(theme::kAccent, 0.85f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, theme::kAccentDim);
    ImGui::PushStyleColor(ImGuiCol_Text, theme::kAccentInk);
    if (ImGui::Button(Tr("Allow Once"), ImVec2(bw, 0.0f))) {
        Queue("agent.approve", Args({{"request", id}}));
        answered = true;
    }
    ImGui::PopStyleColor(4);
    ImGui::SameLine();
    if (ImGui::Button(Tr("Always Allow"), ImVec2(bw, 0.0f))) {
        Queue("agent.approve", Args({{"request", id}, {"always", json::Value::Bool(true)}}));
        answered = true;
    }
    ImGui::SameLine();
    if (ImGui::Button(Tr("Deny"), ImVec2(bw, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        Queue("agent.reject", Args({{"id", id}}));
        answered = true;
    }
    // The answer runs at the next frame top; until then the request reads as answered here.
    if (answered) {
        asked->state = "answered";
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

}  // namespace nuka::runtime::app::viewer
