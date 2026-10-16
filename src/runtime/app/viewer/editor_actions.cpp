#include "runtime/app/viewer/editor_actions.hpp"

#include "imgui.h"

#include <exception>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

namespace json = nuka::scene::json;

std::string_view Trim(std::string_view s) {
    while (!s.empty() && s.front() == ' ') s.remove_prefix(1u);
    while (!s.empty() && s.back() == ' ') s.remove_suffix(1u);
    return s;
}

int KeyOf(std::string_view name) {
    for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END; ++k)
        if (name == ImGui::GetKeyName(static_cast<ImGuiKey>(k))) return k;
    return ImGuiKey_None;
}

// "Ctrl+Shift+Z" as an ImGui key chord; ImGuiKey_None in the key bits when the key is unknown.
int ChordOf(std::string_view text) {
    int chord = 0;
    while (!text.empty()) {
        const size_t plus = text.find('+');
        const std::string_view token = Trim(text.substr(0u, plus));
        text = plus == std::string_view::npos ? std::string_view() : text.substr(plus + 1u);
        if (token == "Ctrl") chord |= ImGuiMod_Ctrl;
        else if (token == "Shift") chord |= ImGuiMod_Shift;
        else if (token == "Alt") chord |= ImGuiMod_Alt;
        else if (token == "Super") chord |= ImGuiMod_Super;
        else chord |= KeyOf(token);
    }
    return chord;
}

bool ParseShortcut(const std::string& text, std::vector<int>* chords) {
    chords->clear();
    std::string_view rest = text;
    while (!Trim(rest).empty()) {
        const size_t comma = rest.find(',');
        const int chord = ChordOf(Trim(rest.substr(0u, comma)));
        if ((chord & ~ImGuiMod_Mask_) == ImGuiKey_None) return false;
        chords->push_back(chord);
        rest = comma == std::string_view::npos ? std::string_view() : rest.substr(comma + 1u);
    }
    return true;
}

}  // namespace

ActionResult ActionResult::Fail(std::string error) {
    ActionResult result;
    result.ok = false;
    result.error = std::move(error);
    return result;
}

void ActionRegistry::Add(Action action) {
    std::vector<int> chords;
    ParseShortcut(action.shortcut, &chords);
    actions_.push_back(std::move(action));
    chords_.push_back(std::move(chords));
}

const Action* ActionRegistry::Find(std::string_view id) const {
    for (const Action& a : actions_)
        if (a.id == id) return &a;
    return nullptr;
}

bool ActionRegistry::Rebind(std::string_view id, const std::string& shortcut) {
    for (size_t i = 0; i < actions_.size(); ++i) {
        if (actions_[i].id != id) continue;
        std::vector<int> chords;
        if (!ParseShortcut(shortcut, &chords)) return false;
        actions_[i].shortcut = shortcut;
        chords_[i] = std::move(chords);
        return true;
    }
    return false;
}

// Arguments come from scripts and remote callers, so a run that throws on them fails instead.
ActionResult ActionRegistry::Run(std::string_view id, const json::Value& args, const std::string& source,
                                 uint64_t frame) {
    ActionResult result;
    const Action* action = Find(id);
    if (action == nullptr) {
        result = ActionResult::Fail("unknown action '" + std::string(id) + "'");
    } else if (action->enabled && !action->enabled()) {
        result = ActionResult::Fail("'" + std::string(id) + "' is disabled now");
    } else {
        try {
            result = action->run(args);
        } catch (const std::exception& e) {
            result = ActionResult::Fail(e.what());
        }
    }
    log_.push_back(ActionLogEntry{next_seq_++, frame, std::string(id), source, args, result.ok, result.error});
    return result;
}

std::vector<std::string> ActionRegistry::Pressed(bool global_ok, bool viewport_ok) const {
    std::vector<std::string> ids;
    for (size_t i = 0; i < actions_.size(); ++i) {
        if (!(actions_[i].scope == ShortcutScope::Global ? global_ok : viewport_ok)) continue;
        for (const int chord : chords_[i]) {
            if (ImGui::IsKeyChordPressed(chord)) {
                ids.push_back(actions_[i].id);
                break;
            }
        }
    }
    return ids;
}

json::Value ActionRegistry::List() const {
    json::Value list = json::Value::Array();
    for (const Action& a : actions_) {
        json::Value v = json::Value::Object();
        v.Set("id", json::Value::Str(a.id));
        v.Set("label", json::Value::Str(a.label));
        v.Set("category", json::Value::Str(a.category));
        v.Set("shortcut", json::Value::Str(a.shortcut));
        v.Set("enabled", json::Value::Bool(!a.enabled || a.enabled()));
        if (a.checked) v.Set("checked", json::Value::Bool(a.checked()));
        list.PushBack(std::move(v));
    }
    return list;
}

json::Value ActionRegistry::LogSince(uint64_t seq) const {
    json::Value list = json::Value::Array();
    for (const ActionLogEntry& e : log_) {
        if (e.seq <= seq) continue;
        json::Value v = json::Value::Object();
        v.Set("seq", json::Value::Int(static_cast<int64_t>(e.seq)));
        v.Set("frame", json::Value::Int(static_cast<int64_t>(e.frame)));
        v.Set("id", json::Value::Str(e.id));
        v.Set("source", json::Value::Str(e.source));
        v.Set("args", e.args);
        v.Set("ok", json::Value::Bool(e.ok));
        v.Set("error", json::Value::Str(e.error));
        list.PushBack(std::move(v));
    }
    return list;
}

}  // namespace nuka::runtime::app::viewer
