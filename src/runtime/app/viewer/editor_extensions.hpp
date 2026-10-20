#pragma once
// Editor extensions on the search path: dependencies, enabling and the scope each one registers
// into. Native plugins load from shadow copies; all of it runs on the UI thread.

#include "nuka/nuka_editor.h"
#include "runtime/app/viewer/editor_actions.hpp"
#include "runtime/app/viewer/teleop_view.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

class CameraController;
class ImGuiLayer;

// A TOML file as JSON: tables become objects. Throws on a syntax error.
scene::json::Value ReadToml(const std::filesystem::path& file);

// What the manager needs from the editor around it.
struct ExtensionServices {
    ActionRegistry* actions = nullptr;
    ImGuiLayer* ui = nullptr;
    std::function<ActionResult(const std::string& id, const scene::json::Value& args, const std::string& source)> run;
    std::function<void(const std::string& text)> console;
    std::filesystem::path state_file;  // extensions.toml: the user's enable and disable choices
};

class ExtensionManager {
public:
    ExtensionManager();
    ~ExtensionManager();
    ExtensionManager(const ExtensionManager&) = delete;
    ExtensionManager& operator=(const ExtensionManager&) = delete;

    // Adds the extensions.* actions and the Extensions panel, then reads the manifests in
    // `folders`; `startup` names the extensions to enable besides the built-ins.
    void Configure(ExtensionServices services, std::vector<std::filesystem::path> folders,
                   std::vector<std::string> startup);
    void SetPythonHooks(const nuka_editor_python_hooks_t& hooks);
    // Enables the built-ins and the startup set; the first frame calls it, once hooks are in.
    void Start();
    // Disables every extension, dependents first.
    void Shutdown();
    // Reloads extensions whose files changed, and retries failed ones; checks once a second.
    void Poll();
    void DrawOverlays(const CameraController& camera, uint64_t tick);
    // An extension's action ran; a callback that keeps failing disables its extension.
    void Report(const std::string& owner, const std::string& callback, bool ok);
    const nuka_editor_host_api_t* HostApi(const std::string& id) const;
    // The teleop devices (appended to `devices`) and modes the enabled extensions registered.
    void TeleopEntries(std::vector<TeleopDeviceInfo>* devices, std::vector<TeleopRetarget>* retargets) const;
    // One frame of extension device `id`; false when it failed or is gone.
    bool ReadTeleopDevice(const std::string& id, std::vector<float>* axes, std::vector<uint8_t>* buttons);
    // Runs extension mode `id` on the state JSON: true with *out its targets, false with the reason.
    bool Retarget(const std::string& id, const std::string& state_json, std::string* out);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace nuka::runtime::app::viewer
