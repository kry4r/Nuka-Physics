#pragma once
// The editor application over one world: surface, ImGui shell, scene edits and the
// frame loop, advanced one frame per Update.

#include "nuka/nuka_editor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

class EditorHost {
public:
    // A non-null world is borrowed; otherwise the first load creates one. Throws
    // std::runtime_error when the surface, renderer or ImGui cannot start.
    EditorHost(const std::vector<std::string>& args, nuka_world_handle world, float dt);
    ~EditorHost();
    EditorHost(const EditorHost&) = delete;
    EditorHost& operator=(const EditorHost&) = delete;

    void SetScriptHooks(const nuka_editor_script_hooks_t& hooks);
    void Update(bool step_physics);
    void StopPhysics();
    bool Running() const;
    bool Playing() const;
    void Close();
    nuka_world_handle World() const;
    float Dt() const;
    void ConsoleAppend(const std::string& text);
    // Runs one registered action now; replies {"ok","value","error"} as JSON text.
    std::string RunAction(const std::string& id, const std::string& args_json, const std::string& source, bool* ok);
    std::string Actions() const;
    std::string ActionLog(uint64_t since) const;
    std::string DocumentState();
    // The shell's nui C API; valid for the editor's lifetime.
    const nuka_nui_api_t* Nui();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace nuka::runtime::app::viewer
