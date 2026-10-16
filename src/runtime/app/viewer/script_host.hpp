#pragma once
// Console and lifecycle scripts run in the host process through the hooks it installs;
// a host that installs none leaves the editor without scripting.

#include "nuka/nuka_editor.h"

#include <cstdint>
#include <string>

namespace nuka::runtime::app::viewer {

class ScriptHost {
public:
    void SetHooks(const nuka_editor_script_hooks_t& hooks) { hooks_ = hooks; }
    bool Valid() const { return hooks_.run != nullptr; }

    std::string RunString(const std::string& code) {
        if (!Valid()) return {};
        return Text(hooks_.run(hooks_.user_data, code.c_str()));
    }
    std::string RegisterScript(uint64_t script_id, const std::string& source) {
        if (!Valid()) return {};
        return Text(hooks_.register_script(hooks_.user_data, script_id, source.c_str()));
    }
    void UnregisterScript(uint64_t script_id) {
        if (Valid()) hooks_.unregister_script(hooks_.user_data, script_id);
    }
    void ClearScripts() {
        if (Valid()) hooks_.clear_scripts(hooks_.user_data);
    }
    std::string DispatchReady() { return Dispatch(NUKA_EDITOR_CALLBACK_READY); }
    std::string DispatchStep() { return Dispatch(NUKA_EDITOR_CALLBACK_STEP); }
    std::string DispatchReset() { return Dispatch(NUKA_EDITOR_CALLBACK_RESET); }

private:
    static std::string Text(const char* text) { return text != nullptr ? text : ""; }
    std::string Dispatch(nuka_editor_callback_t callback) {
        if (!Valid()) return {};
        return Text(hooks_.dispatch(hooks_.user_data, callback));
    }

    nuka_editor_script_hooks_t hooks_{};
};

}  // namespace nuka::runtime::app::viewer
