#pragma once
// The editor's commands. Menus, buttons, shortcuts, Python, the CLI and smoke scenarios all
// run them through one registry, which logs every run.

#include "scene/format/json.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
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

struct Action {
    std::string id;
    std::string label;
    std::string category;
    std::string shortcut;  // "Ctrl+Shift+Z"; alternatives separated by ", "
    ShortcutScope scope = ShortcutScope::Global;
    std::function<bool()> enabled;  // empty: always enabled
    std::function<bool()> checked;  // empty: not a toggle
    std::function<ActionResult(const scene::json::Value& args)> run;
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
    const Action* Find(std::string_view id) const;
    bool Rebind(std::string_view id, const std::string& shortcut);
    // Unknown and disabled actions fail without running; every call lands in the log.
    ActionResult Run(std::string_view id, const scene::json::Value& args, const std::string& source,
                     uint64_t frame);
    // Actions whose shortcut was pressed this ImGui frame, in registration order.
    std::vector<std::string> Pressed(bool global_ok, bool viewport_ok) const;
    scene::json::Value List() const;
    scene::json::Value LogSince(uint64_t seq) const;

private:
    std::vector<Action> actions_;
    std::vector<std::vector<int>> chords_;
    std::vector<ActionLogEntry> log_;
    uint64_t next_seq_ = 1u;
};

}  // namespace nuka::runtime::app::viewer
