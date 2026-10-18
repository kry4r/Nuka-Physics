#pragma once
// A header-only C++ face over nuka/nuka_editor_plugin.h, free of exceptions and RTTI; see
// examples/editor_extensions/hello_cpp. NUKA_EDITOR_PLUGIN_IMGUI turns on Dear ImGui panels.

#include "nuka/nuka_editor_plugin.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef NUKA_EDITOR_PLUGIN_IMGUI
#include "imgui.h"
#endif

namespace nuka::editor {

// An action's outcome: a JSON value, or the reason it failed.
struct Reply {
    bool ok = true;
    std::string text = "null";
    static Reply Value(std::string json) { return Reply{true, std::move(json)}; }
    static Reply Fail(std::string reason) { return Reply{false, std::move(reason)}; }
};

// `text` as a JSON string literal.
inline std::string Quote(const std::string& text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else if (static_cast<unsigned char>(c) < 0x20u) {
            char code[8];
            std::snprintf(code, sizeof(code), "\\u%04x", static_cast<unsigned>(c));
            out += code;
        } else {
            out += c;
        }
    }
    return out + "\"";
}

// One frame of drawing over the scene viewport, in world coordinates; rgba is 0xRRGGBBAA.
class Overlay {
public:
    explicit Overlay(const nuka_editor_overlay_frame_t& frame) : frame_(frame) {}
    void Line(const float a[3], const float b[3], uint32_t rgba, float thickness = 1.5f) const {
        frame_.line(frame_.host, a, b, rgba, thickness);
    }
    void Circle(const float center[3], float radius_px, uint32_t rgba, float thickness = 1.5f) const {
        frame_.circle(frame_.host, center, radius_px, rgba, thickness);
    }
    void Text(const float at[3], uint32_t rgba, const char* text) const { frame_.text(frame_.host, at, rgba, text); }
    uint64_t Tick() const { return frame_.tick; }
    const nuka_editor_overlay_frame_t& Frame() const { return frame_; }

private:
    const nuka_editor_overlay_frame_t& frame_;
};

struct ActionInfo {
    ActionInfo(const char* action_id, const char* action_label, const char* menu = nullptr, const char* keys = nullptr,
               const char* glyph = nullptr, bool skip_menus = false)
        : id(action_id), label(action_label), category(menu), shortcut(keys), icon(glyph), hidden(skip_menus) {}
    const char* id;
    const char* label;
    const char* category;  // "extensions" when null
    const char* shortcut;
    const char* icon;
    bool hidden;
};

using ActionFn = std::function<Reply(const std::string& args_json)>;
using OverlayFn = std::function<bool(const Overlay& overlay)>;
using EventFn = std::function<void(uint32_t node, const std::string& event_json)>;
// Fills one frame of a teleop device's axes and buttons in the order it named them.
using TeleopReadFn = std::function<bool(float* axes, uint8_t* buttons)>;

// The editor as this extension sees it. What it registers goes away with the extension.
class Host {
public:
    explicit Host(const nuka_editor_host_api_t* api) : api_(api) {}
    Host(const Host&) = delete;
    Host& operator=(const Host&) = delete;

    const nuka_editor_host_api_t* Api() const { return api_; }
    std::string Id() const { return api_->extension_id; }
    std::string Dir() const { return api_->extension_dir; }
    std::string SettingsJson() const { return api_->settings_json; }

    bool AddAction(const ActionInfo& info, ActionFn fn) {
        nuka_editor_action_desc_t desc{};
        desc.struct_size = static_cast<uint32_t>(sizeof(desc));
        desc.id = info.id;
        desc.label = info.label;
        desc.category = info.category;
        desc.shortcut = info.shortcut;
        desc.icon = info.icon;
        desc.hidden = info.hidden ? 1u : 0u;
        desc.run = &RunAction;
        desc.user = new ActionSlot{std::move(fn), std::string()};
        desc.release = [](void* user) { delete static_cast<ActionSlot*>(user); };
        return api_->register_action(api_->host, &desc) == NUKA_RESULT_OK;
    }

    // `fn` returns false when it failed; repeated failures disable the extension.
    bool AddOverlay(const char* id, OverlayFn fn) {
        nuka_editor_overlay_desc_t desc{};
        desc.struct_size = static_cast<uint32_t>(sizeof(desc));
        desc.id = id;
        desc.draw = [](void* user, const nuka_editor_overlay_frame_t* frame) -> int32_t {
            return (*static_cast<OverlayFn*>(user))(Overlay(*frame)) ? 1 : 0;
        };
        desc.user = new OverlayFn(std::move(fn));
        desc.release = [](void* user) { delete static_cast<OverlayFn*>(user); };
        return api_->register_overlay(api_->host, &desc) == NUKA_RESULT_OK;
    }

    // A teleop input device, active once the Teleop panel or teleop.use picks it by `id`.
    bool AddTeleopDevice(const char* id, const char* label, const char* kind, const std::vector<const char*>& axes,
                         const std::vector<const char*>& buttons, TeleopReadFn read) {
        nuka_editor_teleop_device_desc_t desc{};
        desc.struct_size = static_cast<uint32_t>(sizeof(desc));
        desc.id = id;
        desc.label = label;
        desc.kind = kind;
        desc.axis_count = static_cast<uint32_t>(axes.size());
        desc.axis_names = axes.data();
        desc.button_count = static_cast<uint32_t>(buttons.size());
        desc.button_names = buttons.data();
        desc.read = [](void* user, float* a, uint8_t* b) -> int32_t {
            return (*static_cast<TeleopReadFn*>(user))(a, b) ? 1 : 0;
        };
        desc.user = new TeleopReadFn(std::move(read));
        desc.release = [](void* user) { delete static_cast<TeleopReadFn*>(user); };
        return api_->register_teleop_device(api_->host, &desc) == NUKA_RESULT_OK;
    }

    // A teleop mode: `map` gets the state JSON (see nuka_editor_teleop_retarget_desc_t) and
    // replies {joint name: target}.
    bool AddTeleopMode(const char* id, const char* label, ActionFn map) {
        nuka_editor_teleop_retarget_desc_t desc{};
        desc.struct_size = static_cast<uint32_t>(sizeof(desc));
        desc.id = id;
        desc.label = label;
        desc.map = &RunAction;
        desc.user = new ActionSlot{std::move(map), std::string()};
        desc.release = [](void* user) { delete static_cast<ActionSlot*>(user); };
        return api_->register_teleop_retarget(api_->host, &desc) == NUKA_RESULT_OK;
    }

#ifdef NUKA_EDITOR_PLUGIN_IMGUI
    // A panel whose content `draw` paints with Dear ImGui, inside the editor's window.
    bool AddImGuiPanel(const char* id, const char* title, const char* dock, std::function<void()> draw) {
        return api_->register_imgui_panel(
                   api_->host, id, title, dock, [](void* user) { (*static_cast<std::function<void()>*>(user))(); },
                   new std::function<void()>(std::move(draw)),
                   [](void* user) { delete static_cast<std::function<void()>*>(user); }) == NUKA_RESULT_OK;
    }
#endif

    // Runs any editor action now and returns its {"ok", "value", "error"} reply.
    std::string Run(const char* id, const std::string& args_json = "{}") const {
        const char* out = nullptr;
        api_->run_action(api_->host, id, args_json.c_str(), &out);
        return out != nullptr ? out : "";
    }

    void Log(nuka_nui_level_t level, const std::string& text) const { api_->log(api_->host, level, text.c_str()); }
    void Toast(nuka_nui_level_t level, const std::string& text) const {
        api_->nui->toast(api_->nui->host, level, text.c_str());
    }

    // The nui tree: a panel's root, then nodes with JSON props.
    uint32_t Panel(const char* id, const char* title, const char* dock = "right") const {
        return api_->nui->panel(api_->nui->host, id, title, dock);
    }
    uint32_t Add(uint32_t parent, const char* kind, const std::string& props = "{}") const {
        return api_->nui->add(api_->nui->host, parent, kind, props.c_str());
    }
    bool Set(uint32_t node, const std::string& props) const {
        return api_->nui->set(api_->nui->host, node, props.c_str()) != 0;
    }
    std::string Get(uint32_t node) const {
        const char* value = api_->nui->get(api_->nui->host, node);
        return value != nullptr ? value : "null";
    }
    // `fn` runs on the UI thread at the next frame top; the host keeps it while the plugin runs.
    bool OnEvent(uint32_t node, EventFn fn) {
        events_.push_back(std::make_unique<EventFn>(std::move(fn)));
        return api_->nui->on_event(
                   api_->nui->host, node,
                   [](void* user, uint32_t id, const char* event) { (*static_cast<EventFn*>(user))(id, event); },
                   events_.back().get()) != 0;
    }

private:
    struct ActionSlot {
        ActionFn fn;
        std::string reply;  // valid until the action's next run
    };

    static int32_t RunAction(void* user, const char* args_json, const char** out_json) {
        ActionSlot& slot = *static_cast<ActionSlot*>(user);
        Reply reply = slot.fn(args_json != nullptr ? args_json : "{}");
        slot.reply = std::move(reply.text);
        *out_json = slot.reply.c_str();
        return reply.ok ? 1 : 0;
    }

    const nuka_editor_host_api_t* api_;
    std::vector<std::unique_ptr<EventFn>> events_;
};

class Plugin {
public:
    virtual ~Plugin() = default;
    // False fails the load; say why through host.Log first.
    virtual bool OnStartup(Host& host) = 0;
    virtual void OnShutdown() {}
};

namespace detail {

struct Instance {
    explicit Instance(const nuka_editor_host_api_t* api) : host(api) {}
    Host host;
    std::unique_ptr<Plugin> plugin;
};

template <class T>
nuka_result_t Load(const nuka_editor_host_api_t* api, void** state) {
#ifdef NUKA_EDITOR_PLUGIN_IMGUI
    if (api->imgui_context != nullptr) {
        ImGui::SetCurrentContext(static_cast<ImGuiContext*>(api->imgui_context));
        ImGui::SetAllocatorFunctions(api->imgui_alloc, api->imgui_free, api->imgui_alloc_user);
    }
#endif
    auto* instance = new Instance(api);
    instance->plugin = std::make_unique<T>();
    if (!instance->plugin->OnStartup(instance->host)) {
        delete instance;
        return NUKA_RESULT_INTERNAL;
    }
    *state = instance;
    return NUKA_RESULT_OK;
}

inline void Unload(void* state) {
    auto* instance = static_cast<Instance*>(state);
    instance->plugin->OnShutdown();
    delete instance;
}

#ifdef NUKA_EDITOR_PLUGIN_IMGUI
inline void ImGuiCheck(nuka_editor_imgui_check_t* out) {
    out->version_num = IMGUI_VERSION_NUM;
    out->size_io = static_cast<uint32_t>(sizeof(ImGuiIO));
    out->size_style = static_cast<uint32_t>(sizeof(ImGuiStyle));
    out->size_vec2 = static_cast<uint32_t>(sizeof(ImVec2));
    out->size_vec4 = static_cast<uint32_t>(sizeof(ImVec4));
    out->size_draw_vert = static_cast<uint32_t>(sizeof(ImDrawVert));
    out->size_draw_idx = static_cast<uint32_t>(sizeof(ImDrawIdx));
    out->size_wchar = static_cast<uint32_t>(sizeof(ImWchar));
}
#define NUKA_EDITOR_PLUGIN_IMGUI_EXPORT                                                          \
    extern "C" NUKA_EDITOR_PLUGIN_EXPORT void nuka_editor_plugin_imgui(nuka_editor_imgui_check_t* out) { \
        ::nuka::editor::detail::ImGuiCheck(out);                                                  \
    }
#else
#define NUKA_EDITOR_PLUGIN_IMGUI_EXPORT
#endif

}  // namespace detail
}  // namespace nuka::editor

// Defines the plugin's exports around one Plugin subclass.
#define NUKA_EDITOR_PLUGIN(Type)                                                                        \
    extern "C" NUKA_EDITOR_PLUGIN_EXPORT uint32_t nuka_editor_plugin_abi(void) { return NUKA_EDITOR_PLUGIN_ABI; } \
    extern "C" NUKA_EDITOR_PLUGIN_EXPORT nuka_result_t nuka_editor_plugin_load(const nuka_editor_host_api_t* host, \
                                                                               void** state) {           \
        return ::nuka::editor::detail::Load<Type>(host, state);                                         \
    }                                                                                                    \
    extern "C" NUKA_EDITOR_PLUGIN_EXPORT void nuka_editor_plugin_unload(void* state) {                  \
        ::nuka::editor::detail::Unload(state);                                                           \
    }                                                                                                    \
    NUKA_EDITOR_PLUGIN_IMGUI_EXPORT
