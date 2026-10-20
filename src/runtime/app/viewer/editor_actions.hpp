#pragma once
// The editor's commands. Menus, buttons, shortcuts, Python, the CLI and smoke scenarios all
// run them through one registry, which logs every run.

#include "scene/format/json.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nuka::runtime::app::viewer {

struct ActionResult {
    bool ok = true;
    scene::json::Value value;
    std::string error;
    static ActionResult Fail(std::string error);
};

// A global shortcut fires unless ImGui owns the keyboard; a viewport one also needs the
// pointer over the viewport.
enum class ShortcutScope : uint8_t { Global, Viewport };

// What an action touches, as its doc's `kind` names it (assets/editor/actions.toml explains each).
inline constexpr std::string_view kActionKinds[] = {"read", "view", "edit", "sim", "file", "process", "app"};

struct Action {
    std::string id;
    std::string label;
    std::string category;
    std::string shortcut;  // "Ctrl+Shift+Z"; alternatives separated by ", "
    ShortcutScope scope = ShortcutScope::Global;
    std::function<bool()> enabled;  // empty: always enabled
    std::function<bool()> checked;  // empty: not a toggle
    std::function<ActionResult(const scene::json::Value& args)> run;
    std::string icon;     // a UTF-8 icon glyph for menus, toolbars and the palette
    bool hidden = false;  // takes arguments or backs a widget, so menus and the palette skip it
    std::string owner;    // the extension that registered it; empty for the editor's own
    scene::json::Value doc;  // {summary, kind, args, returns, example}; null: the catalog's entry
};

// A run queued for the next frame top, where the UI's own calls execute.
struct ActionCall {
    std::string id;
    scene::json::Value args;
    std::string source;
};

struct ActionLogEntry {
    uint64_t seq = 0u;
    uint64_t frame = 0u;
    std::string id;
    std::string source;
    scene::json::Value args;
    bool ok = false;
    std::string error;
};

class ActionRegistry {
public:
    void Add(Action action);
    void RemoveOwned(std::string_view owner);
    const Action* Find(std::string_view id) const;
    bool Rebind(std::string_view id, const std::string& shortcut);
    bool Decorate(std::string_view id, std::string icon, bool hidden);
    // Unknown and disabled actions fail without running; every call lands in the log.
    ActionResult Run(std::string_view id, const scene::json::Value& args, const std::string& source,
                     uint64_t frame);
    // Actions whose shortcut was pressed this ImGui frame, in registration order.
    std::vector<std::string> Pressed(bool global_ok, bool viewport_ok) const;
    const std::vector<Action>& All() const { return actions_; }
    // The editor's documented actions, {id: doc}, for those registered without a doc.
    void SetDocs(scene::json::Value docs) { docs_ = std::move(docs); }
    // The action's doc, else its catalog entry; nullptr when it has neither.
    const scene::json::Value* Doc(const Action& action) const;
    // An undocumented action's kind is "process", so access modes treat it as running code.
    std::string Kind(const Action& action) const;
    scene::json::Value List() const;
    // One action as List() reports it, plus its doc's args, returns and example.
    scene::json::Value Help(const Action& action) const;
    scene::json::Value LogSince(uint64_t seq) const;

private:
    scene::json::Value Describe(const Action& action, bool full) const;

    std::vector<Action> actions_;
    std::vector<std::vector<int>> chords_;
    std::vector<ActionLogEntry> log_;
    scene::json::Value docs_ = scene::json::Value::Object();
    uint64_t next_seq_ = 1u;
};

}  // namespace nuka::runtime::app::viewer
