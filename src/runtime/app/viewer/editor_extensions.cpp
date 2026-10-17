#include "runtime/app/viewer/editor_extensions.hpp"

#include "imgui.h"

#include <toml.hpp>

#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/camera_controller.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"
#include "runtime/app/viewer/nui/nui.hpp"
#include "scene/format/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace nuka::runtime::app::viewer {
namespace {

namespace fs = std::filesystem;
namespace json = nuka::scene::json;
namespace theme = nuka::render::imgui::theme;
namespace icon = nuka::render::imgui::icon;
namespace draw = nuka::nui::draw;
using nuka::nui::TextStyle;
using nuka::nui::Tone;
using nuka::nui::Tr;
using Clock = std::chrono::steady_clock;

// A callback that fails this many times in a row disables its extension.
constexpr int kMaxFailures = 3;
constexpr auto kPollInterval = std::chrono::seconds(1);
constexpr size_t kLogCap = 64u * 1024u;
constexpr float kNearW = 1e-3f;

#if defined(_WIN32)
constexpr const char* kPlatform = "windows-x64";
constexpr const char* kLibrarySuffix = ".dll";
#else
constexpr const char* kPlatform = "linux-x64";
constexpr const char* kLibrarySuffix = ".so";
#endif

enum class State : uint8_t { Disabled, Enabled, Failed };

template <class... T>
std::string Fmt(const char* format, T... values) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), format, values...);
    return buf;
}

fs::path Utf8Path(const std::string& text) { return fs::path(std::u8string(text.begin(), text.end())); }

std::string PathText(const fs::path& path) {
    const std::u8string text = path.u8string();
    return std::string(text.begin(), text.end());
}

std::string LastLine(const std::string& text) {
    size_t end = text.find_last_not_of("\r\n");
    if (end == std::string::npos) return text;
    const size_t start = text.find_last_of('\n', end);
    return text.substr(start == std::string::npos ? 0u : start + 1u, end - (start == std::string::npos ? 0u : start + 1u) + 1u);
}

uint64_t ProcessId() {
#if defined(_WIN32)
    return GetCurrentProcessId();
#else
    return static_cast<uint64_t>(getpid());
#endif
}

void* OpenLibrary(const fs::path& path, std::string* reason) {
#if defined(_WIN32)
    HMODULE module = LoadLibraryW(path.c_str());
    if (module == nullptr) *reason = Fmt("cannot load %s (Windows error %lu)", PathText(path).c_str(), GetLastError());
    return module;
#else
    void* module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module == nullptr) *reason = std::string("cannot load ") + dlerror();
    return module;
#endif
}

template <class Fn>
Fn Symbol(void* library, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return reinterpret_cast<Fn>(dlsym(library, name));
#endif
}

void CloseLibrary(void* library) {
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(library));
#else
    dlclose(library);
#endif
}

struct Version {
    int major = 0;
    int minor = 0;
    int patch = 0;
    auto operator<=>(const Version&) const = default;
};

Version ParseVersion(std::string_view text) {
    Version v;
    int* parts[] = {&v.major, &v.minor, &v.patch};
    size_t i = 0u;
    for (int* part : parts) {
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') *part = *part * 10 + (text[i++] - '0');
        if (i >= text.size() || text[i] != '.') break;
        ++i;
    }
    return v;
}

// "*", "^1.2" (same major; same minor below 1.0), ">=1.2", or an exact "1.2.3".
bool Satisfies(const Version& v, std::string_view req) {
    if (req.empty() || req == "*") return true;
    if (req.front() == '^') {
        const Version base = ParseVersion(req.substr(1u));
        const bool same = base.major > 0   ? v.major == base.major
                          : base.minor > 0 ? v.major == 0 && v.minor == base.minor
                                           : v.major == 0 && v.minor == 0 && v.patch == base.patch;
        return same && v >= base;
    }
    if (req.starts_with(">=")) return v >= ParseVersion(req.substr(2u));
    return v == ParseVersion(req.front() == '=' ? req.substr(1u) : req);
}

json::Value FromToml(const toml::node& node) {
    if (const toml::table* table = node.as_table()) {
        json::Value out = json::Value::Object();
        for (const auto& [key, value] : *table) out.Set(std::string(key.str()), FromToml(value));
        return out;
    }
    if (const toml::array* array = node.as_array()) {
        json::Value out = json::Value::Array();
        for (const toml::node& item : *array) out.PushBack(FromToml(item));
        return out;
    }
    if (const auto* text = node.as_string()) return json::Value::Str(text->get());
    if (const auto* integer = node.as_integer()) return json::Value::Int(integer->get());
    if (const auto* real = node.as_floating_point()) return json::Value::Float(real->get());
    if (const auto* flag = node.as_boolean()) return json::Value::Bool(flag->get());
    std::ostringstream text;
    node.visit([&text](const auto& value) { text << value; });
    return json::Value::Str(text.str());
}

// "bin/${platform}/name" next to the manifest; the platform's suffix when it has none.
fs::path ResolveLibrary(const fs::path& dir, std::string text) {
    const std::string token = "${platform}";
    for (size_t at = text.find(token); at != std::string::npos; at = text.find(token)) text.replace(at, token.size(), kPlatform);
    fs::path path = dir / Utf8Path(text);
    if (!path.has_extension()) path += kLibrarySuffix;
    return path;
}

// IMGUI_VERSION_NUM is major*10000 + minor*100 + patch*10; the last digit counts WIP builds.
std::string ImGuiVersionText(uint32_t num) {
    return Fmt("%u.%u.%u", num / 10000u, (num / 100u) % 100u, (num % 100u) / 10u);
}

// Why a plugin's Dear ImGui build cannot share the editor's context; empty when it can.
std::string ImGuiMismatch(const nuka_editor_imgui_check_t& theirs) {
    if (theirs.version_num != IMGUI_VERSION_NUM)
        return "built against Dear ImGui " + ImGuiVersionText(theirs.version_num) + "; the editor runs " +
               ImGuiVersionText(IMGUI_VERSION_NUM) + ", so the plugin cannot draw with it";
    const std::pair<const char*, std::pair<uint32_t, size_t>> sizes[] = {
        {"ImGuiIO", {theirs.size_io, sizeof(ImGuiIO)}},
        {"ImGuiStyle", {theirs.size_style, sizeof(ImGuiStyle)}},
        {"ImVec2", {theirs.size_vec2, sizeof(ImVec2)}},
        {"ImVec4", {theirs.size_vec4, sizeof(ImVec4)}},
        {"ImDrawVert", {theirs.size_draw_vert, sizeof(ImDrawVert)}},
        {"ImDrawIdx", {theirs.size_draw_idx, sizeof(ImDrawIdx)}},
        {"ImWchar", {theirs.size_wchar, sizeof(ImWchar)}}};
    for (const auto& [name, size] : sizes)
        if (size.first != size.second)
            return Fmt("built with another Dear ImGui configuration: sizeof(%s) is %u there and %u here", name,
                       size.first, static_cast<unsigned>(size.second));
    return std::string();
}

toml::table ParseToml(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    return toml::parse(text.str(), PathText(file));
}

// ---- overlay canvas -----------------------------------------------------------------------

struct Canvas {
    float clip[16];  // projection * view, column-major
    float x, y, w, h;
    ImDrawList* list;
};

struct ClipPoint {
    float x, y, z, w;
};

ClipPoint ToClip(const Canvas& c, const float p[3]) {
    const float* m = c.clip;
    return {m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
            m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14], m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15]};
}

ImVec2 ToScreen(const Canvas& c, const ClipPoint& q) {
    return ImVec2(c.x + (q.x / q.w * 0.5f + 0.5f) * c.w, c.y + (0.5f - q.y / q.w * 0.5f) * c.h);
}

ImU32 Rgba(uint32_t rgba) { return IM_COL32((rgba >> 24) & 0xFFu, (rgba >> 16) & 0xFFu, (rgba >> 8) & 0xFFu, rgba & 0xFFu); }

// Segments crossing the camera plane are cut at it.
void CanvasLine(void* host, const float a[3], const float b[3], uint32_t rgba, float thickness) {
    const Canvas& c = *static_cast<const Canvas*>(host);
    ClipPoint p = ToClip(c, a), q = ToClip(c, b);
    if (p.w < kNearW && q.w < kNearW) return;
    const auto cut = [](const ClipPoint& u, const ClipPoint& v) {
        const float t = (kNearW - u.w) / (v.w - u.w);
        return ClipPoint{u.x + (v.x - u.x) * t, u.y + (v.y - u.y) * t, u.z + (v.z - u.z) * t, kNearW};
    };
    if (p.w < kNearW) p = cut(p, q);
    else if (q.w < kNearW) q = cut(q, p);
    c.list->AddLine(ToScreen(c, p), ToScreen(c, q), Rgba(rgba), thickness * nui::Scale());
}

void CanvasCircle(void* host, const float center[3], float radius_px, uint32_t rgba, float thickness) {
    const Canvas& c = *static_cast<const Canvas*>(host);
    const ClipPoint q = ToClip(c, center);
    if (q.w < kNearW) return;
    c.list->AddCircle(ToScreen(c, q), radius_px * nui::Scale(), Rgba(rgba), 0, thickness * nui::Scale());
}

void CanvasText(void* host, const float at[3], uint32_t rgba, const char* text) {
    const Canvas& c = *static_cast<const Canvas*>(host);
    const ClipPoint q = ToClip(c, at);
    if (q.w < kNearW) return;
    c.list->AddText(ToScreen(c, q), Rgba(rgba), text);
}

std::shared_ptr<void> Owned(void* user, void (*release)(void*)) {
    return std::shared_ptr<void>(user, [release](void* u) {
        if (release != nullptr) release(u);
    });
}

// ---- records and scopes -------------------------------------------------------------------

struct Overlay {
    std::string id;
    int32_t (*draw)(void* user, const nuka_editor_overlay_frame_t* frame);
    std::shared_ptr<void> user;
};

struct Record;

// What one enabled extension registered, and the host table it registered it through.
struct Scope {
    ExtensionManager::Impl* impl = nullptr;
    Record* record = nullptr;
    std::string dir_text;
    nuka_editor_host_api_t api{};
    nuka_nui_api_t nui{};
    std::vector<std::string> panels;
    std::vector<Overlay> overlays;
    std::string reply;
    void* library = nullptr;
    fs::path shadow;
    void* plugin_state = nullptr;
    nuka_editor_plugin_unload_fn unload = nullptr;
    bool python = false;
};

struct Record {
    std::string id;
    std::string version = "0.0.0";
    std::string title;
    std::string description;
    fs::path dir;
    fs::path manifest;
    std::string manifest_error;
    std::vector<std::pair<std::string, std::string>> deps;
    fs::path library;
    std::string module;
    std::string settings_json = "{}";
    const ImGuiLayer::BuiltinExtension* builtin = nullptr;
    bool wanted = false;  // asked for, so a changed file retries it after a failure
    State state = State::Disabled;
    bool auto_disabled = false;
    std::string error;
    std::string log;
    std::map<std::string, int> failures;
    fs::file_time_type stamp{};
    std::unique_ptr<Scope> scope;
};

}  // namespace

struct ExtensionManager::Impl {
    ExtensionServices services;
    std::vector<fs::path> folders;
    std::vector<std::string> startup;
    std::vector<std::unique_ptr<Record>> records;
    nuka_editor_python_hooks_t python{};
    uint64_t revision = 1u;
    std::string selected;
    Clock::time_point next_poll{};
    uint64_t shadows = 0u;
    std::mutex log_mu;
    std::vector<std::pair<std::string, std::string>> pending_logs;  // extension id, line

    Record* Find(std::string_view id) {
        for (const auto& r : records)
            if (r->id == id) return r.get();
        return nullptr;
    }

    // Lines may come from any thread; the UI thread files them at the next frame top.
    void Log(const Record& r, nuka_nui_level_t level, const std::string& text) {
        std::string line = level == NUKA_NUI_ERROR ? "error: " + text : level == NUKA_NUI_WARN ? "warning: " + text : text;
        if (!line.empty() && line.back() != '\n') line += '\n';
        std::lock_guard<std::mutex> lock(log_mu);
        pending_logs.emplace_back(r.id, std::move(line));
    }

    void DrainLogs() {
        std::vector<std::pair<std::string, std::string>> lines;
        {
            std::lock_guard<std::mutex> lock(log_mu);
            lines.swap(pending_logs);
        }
        for (const auto& [id, line] : lines) {
            Record& r = *Find(id);
            r.log += line;
            if (r.log.size() > kLogCap) r.log.erase(0u, r.log.size() - kLogCap);
            services.console("[" + id + "] " + line);
        }
    }

    void Queue(const char* id, const std::string& extension_id) {
        json::Value args = json::Value::Object();
        args.Set("id", json::Value::Str(extension_id));
        services.ui->Nui().actions.run(id, args);
    }

    void Discover();
    void ReadManifest(Record& r);
    fs::file_time_type Stamp(const Record& r) const;
    bool Enable(Record& r, std::vector<std::string>& chain);
    void Disable(Record& r);
    void Reload(Record& r);
    void Fail(Record& r, const std::string& reason);
    bool StartScope(Record& r);
    bool LoadNative(Record& r, Scope& s, std::string* reason);
    void Teardown(Record& r);
    json::Value Describe(const Record& r) const;
    void RegisterActions();
    void BuildPanel(nui::VStack& root);
    void DrawHeader();
    void DrawCards();
};

namespace {

using Impl = ExtensionManager::Impl;

Scope& S(void* host) { return *static_cast<Scope*>(host); }
const nuka_nui_api_t& Real(void* host) { return *S(host).impl->services.ui->Nui().Api(); }

// ---- the host table ----------------------------------------------------------------------

nuka_result_t ApiRegisterAction(void* host, const nuka_editor_action_desc_t* desc) {
    Scope& s = S(host);
    std::shared_ptr<void> user = Owned(desc->user, desc->release);
    if (s.impl->services.actions->Find(desc->id) != nullptr) {
        s.impl->Log(*s.record, NUKA_NUI_ERROR, std::string("action '") + desc->id + "' is taken");
        return NUKA_RESULT_INVALID_ARG;
    }
    Action action;
    action.id = desc->id;
    action.label = desc->label;
    action.category = desc->category != nullptr ? desc->category : "extensions";
    action.shortcut = desc->shortcut != nullptr ? desc->shortcut : "";
    action.icon = desc->icon != nullptr ? nui::IconOf(desc->icon) : "";
    action.hidden = desc->hidden != 0u;
    action.owner = s.record->id;
    const nuka_editor_action_fn run = desc->run;
    action.run = [run, user](const json::Value& args) {
        const char* out = nullptr;
        if (run(user.get(), args.Dump(0).c_str(), &out) == 0) return ActionResult::Fail(out != nullptr ? out : "failed");
        ActionResult result;
        if (out != nullptr) result.value = json::Value::Parse(out);
        return result;
    };
    s.impl->services.actions->Add(std::move(action));
    return NUKA_RESULT_OK;
}

nuka_result_t ApiRegisterOverlay(void* host, const nuka_editor_overlay_desc_t* desc) {
    S(host).overlays.push_back(Overlay{desc->id, desc->draw, Owned(desc->user, desc->release)});
    return NUKA_RESULT_OK;
}

nuka_result_t ApiRunAction(void* host, const char* id, const char* args_json, const char** out_json) {
    Scope& s = S(host);
    ActionResult result;
    try {
        const json::Value args =
            args_json != nullptr && *args_json != '\0' ? json::Value::Parse(args_json) : json::Value::Object();
        result = s.impl->services.run(id, args, "extension:" + s.record->id);
    } catch (const json::ParseError& e) {
        result = ActionResult::Fail(std::string("bad action arguments: ") + e.what());
    }
    json::Value reply = json::Value::Object();
    reply.Set("ok", json::Value::Bool(result.ok));
    reply.Set("value", result.value);
    reply.Set("error", json::Value::Str(result.error));
    s.reply = reply.Dump(0);
    *out_json = s.reply.c_str();
    return result.ok ? NUKA_RESULT_OK : NUKA_RESULT_INVALID_ARG;
}

void ApiLog(void* host, nuka_nui_level_t level, const char* text) { S(host).impl->Log(*S(host).record, level, text); }

nuka_result_t ApiImGuiPanel(void* host, const char* id, const char* title, const char* dock, void (*draw_fn)(void*),
                            void* user, void (*release)(void*)) {
    Scope& s = S(host);
    std::shared_ptr<void> owned = Owned(user, release);
    if (s.api.imgui_context == nullptr) {
        s.impl->Log(*s.record, NUKA_NUI_ERROR, "an ImGui panel needs the nuka_editor_plugin_imgui export");
        return NUKA_RESULT_INVALID_ARG;
    }
    nui::Host& ui = s.impl->services.ui->Nui();
    std::lock_guard<std::recursive_mutex> lock(ui.Mutex());
    ui.PanelNode(id, title, dock);
    ui.FindPanel(id)->Root().Add<nui::Custom>([draw_fn, owned] { draw_fn(owned.get()); });
    s.panels.push_back(id);
    return NUKA_RESULT_OK;
}

// The extension's nui table: the editor's, with its panels recorded for removal.
uint32_t NuiPanel(void* host, const char* id, const char* title, const char* dock) {
    const nuka_nui_api_t& real = Real(host);
    const uint32_t root = real.panel(real.host, id, title, dock);
    std::vector<std::string>& panels = S(host).panels;
    if (root != 0u && std::find(panels.begin(), panels.end(), id) == panels.end()) panels.push_back(id);
    return root;
}
uint32_t NuiAdd(void* host, uint32_t parent, const char* kind, const char* props) {
    return Real(host).add(Real(host).host, parent, kind, props);
}
int32_t NuiSet(void* host, uint32_t node, const char* props) { return Real(host).set(Real(host).host, node, props); }
int32_t NuiRemove(void* host, uint32_t node) { return Real(host).remove(Real(host).host, node); }
int32_t NuiClear(void* host, uint32_t node) { return Real(host).clear(Real(host).host, node); }
const char* NuiGet(void* host, uint32_t node) { return Real(host).get(Real(host).host, node); }
int32_t NuiOnEvent(void* host, uint32_t node, nuka_nui_event_fn fn, void* user) {
    return Real(host).on_event(Real(host).host, node, fn, user);
}
void NuiToast(void* host, nuka_nui_level_t level, const char* text) { Real(host).toast(Real(host).host, level, text); }
void NuiProgress(void* host, const char* id, const char* label, float fraction) {
    Real(host).progress(Real(host).host, id, label, fraction);
}
const char* NuiLastError(void* host) { return Real(host).last_error(Real(host).host); }

void FillApi(Scope& s) {
    nuka_editor_host_api_t& a = s.api;
    a.struct_size = static_cast<uint32_t>(sizeof(nuka_editor_host_api_t));
    a.abi_version = NUKA_EDITOR_PLUGIN_ABI;
    a.host = &s;
    a.extension_id = s.record->id.c_str();
    a.extension_dir = s.dir_text.c_str();
    a.settings_json = s.record->settings_json.c_str();
    a.nui = &s.nui;
    a.register_action = &ApiRegisterAction;
    a.register_overlay = &ApiRegisterOverlay;
    a.run_action = &ApiRunAction;
    a.log = &ApiLog;
    a.register_imgui_panel = &ApiImGuiPanel;
    nuka_nui_api_t& n = s.nui;
    n.struct_size = static_cast<uint32_t>(sizeof(nuka_nui_api_t));
    n.version = NUKA_NUI_API_VERSION;
    n.host = &s;
    n.panel = &NuiPanel;
    n.add = &NuiAdd;
    n.set = &NuiSet;
    n.remove = &NuiRemove;
    n.clear = &NuiClear;
    n.get = &NuiGet;
    n.on_event = &NuiOnEvent;
    n.toast = &NuiToast;
    n.progress = &NuiProgress;
    n.last_error = &NuiLastError;
}

const char* StateName(State state) {
    return state == State::Enabled ? "enabled" : state == State::Failed ? "failed" : "disabled";
}

const char* KindName(const Record& r) {
    if (r.builtin != nullptr) return "built-in";
    if (!r.library.empty()) return r.module.empty() ? "native" : "native + python";
    return r.module.empty() ? "data" : "python";
}

bool DependsOn(const Record& r, const std::string& id) {
    return std::any_of(r.deps.begin(), r.deps.end(), [&](const auto& dep) { return dep.first == id; });
}

}  // namespace

// ---- discovery ----------------------------------------------------------------------------

// A folder holding extension.toml is one extension; otherwise each subfolder holding one is.
// The first folder to name an id keeps it.
void Impl::Discover() {
    for (const fs::path& folder : folders) {
        std::error_code ec;
        std::vector<fs::path> manifests;
        if (fs::exists(folder / "extension.toml", ec)) {
            manifests.push_back(folder / "extension.toml");
        } else if (fs::is_directory(folder, ec)) {
            for (const fs::directory_entry& entry : fs::directory_iterator(folder, ec))
                if (fs::exists(entry.path() / "extension.toml", ec)) manifests.push_back(entry.path() / "extension.toml");
            std::sort(manifests.begin(), manifests.end());
        }
        for (const fs::path& manifest : manifests) {
            auto record = std::make_unique<Record>();
            record->manifest = manifest;
            record->dir = manifest.parent_path();
            record->id = PathText(record->dir.filename());
            ReadManifest(*record);
            if (Find(record->id) != nullptr) {
                services.console("[extensions] " + PathText(record->dir) + " repeats the id " + record->id + "; skipped\n");
                continue;
            }
            records.push_back(std::move(record));
        }
    }
}

void Impl::ReadManifest(Record& r) {
    r.manifest_error.clear();
    r.deps.clear();
    r.library.clear();
    r.module.clear();
    r.settings_json = "{}";
    try {
        const toml::table t = ParseToml(r.manifest);
        r.id = t["package"]["id"].value_or(r.id);
        r.version = t["package"]["version"].value_or(std::string("0.0.0"));
        r.title = t["package"]["title"].value_or(r.id);
        r.description = t["package"]["description"].value_or(std::string());
        if (const toml::table* deps = t["dependencies"].as_table())
            for (const auto& [key, value] : *deps) r.deps.emplace_back(std::string(key.str()), value.value_or(std::string("*")));
        if (const std::optional<std::string> library = t["native"]["library"].value<std::string>())
            r.library = ResolveLibrary(r.dir, *library);
        r.module = t["python"]["module"].value_or(std::string());
        if (const toml::table* settings = t["settings"].as_table()) r.settings_json = FromToml(*settings).Dump(0);
        // Its string tables load with the manifest, so its card reads translated while disabled.
        nui::RemoveTranslations(r.id);
        nui::LoadTranslationDir(r.dir / "i18n", r.id);
    } catch (const toml::parse_error& e) {
        r.manifest_error = PathText(Utf8Path(*e.source().path).filename()) + Fmt(":%u: ", e.source().begin.line) +
                           std::string(e.description());
    }
    if (r.title.empty()) r.title = r.id;
    r.stamp = Stamp(r);
}

// The newest file under the extension's folder, bytecode caches aside.
fs::file_time_type Impl::Stamp(const Record& r) const {
    fs::file_time_type newest{};
    std::error_code ec;
    for (fs::recursive_directory_iterator it(r.dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().filename() == "__pycache__") {
            it.disable_recursion_pending();
            continue;
        }
        if (it->is_regular_file(ec)) newest = std::max(newest, it->last_write_time(ec));
    }
    return newest;
}

// ---- lifecycle ----------------------------------------------------------------------------

void Impl::Fail(Record& r, const std::string& reason) {
    r.state = State::Failed;
    r.error = LastLine(reason);
    Log(r, NUKA_NUI_ERROR, reason);
    ++revision;
}

// Dependencies start first; a cycle, a missing one or a version out of range fails the chain.
bool Impl::Enable(Record& r, std::vector<std::string>& chain) {
    if (r.state == State::Enabled) return true;
    if (!r.manifest_error.empty()) {
        Fail(r, r.manifest_error);
        return false;
    }
    chain.push_back(r.id);
    for (const auto& [dep_id, req] : r.deps) {
        Record* dep = Find(dep_id);
        std::string reason;
        if (std::find(chain.begin(), chain.end(), dep_id) != chain.end()) {
            reason = "dependency cycle: ";
            for (const std::string& id : chain) reason += id + " -> ";
            reason += dep_id;
        } else if (dep == nullptr) {
            reason = "needs " + dep_id + ", which is not installed";
        } else if (!Satisfies(ParseVersion(dep->version), req)) {
            reason = "needs " + dep_id + " " + req + ", found " + dep->version;
        } else if (!Enable(*dep, chain)) {
            reason = "needs " + dep_id + ": " + dep->error;
        }
        if (!reason.empty()) {
            chain.pop_back();
            Fail(r, reason);
            return false;
        }
    }
    chain.pop_back();
    return StartScope(r);
}

// Native first, then Python; the scope exists before either so both can reach it.
bool Impl::StartScope(Record& r) {
    r.scope = std::make_unique<Scope>();
    Scope& s = *r.scope;
    s.impl = this;
    s.record = &r;
    s.dir_text = PathText(r.dir);
    FillApi(s);
    r.state = State::Enabled;
    r.error.clear();
    r.failures.clear();
    r.auto_disabled = false;
    if (r.builtin != nullptr) {
        for (const std::string& pid : r.builtin->panels) {
            services.ui->AddBuiltinPanel(pid);
            s.panels.push_back(pid);
        }
        for (Action& action : services.ui->BuiltinActions(*r.builtin)) {
            action.owner = r.id;
            services.actions->Add(std::move(action));
        }
    }
    std::string reason;
    if (!r.library.empty() && !LoadNative(r, s, &reason)) {
        Teardown(r);
        Fail(r, reason);
        return false;
    }
    if (!r.module.empty()) {
        if (python.load == nullptr) {
            Teardown(r);
            Fail(r, "a Python extension, and this host runs no Python");
            return false;
        }
        int32_t failed = 0;
        const char* out = python.load(python.user_data, r.id.c_str(), r.module.c_str(), s.dir_text.c_str(), &failed);
        if (failed != 0) {
            const std::string text = out != nullptr ? out : "the import failed";
            Teardown(r);
            Fail(r, text);
            return false;
        }
        s.python = true;
        if (out != nullptr) Log(r, NUKA_NUI_INFO, out);
    }
    if (r.builtin == nullptr) Log(r, NUKA_NUI_OK, "enabled " + r.version);
    ++revision;
    return true;
}

// The plugin loads from a private copy, so a rebuild can replace the original while it runs.
bool Impl::LoadNative(Record& r, Scope& s, std::string* reason) {
    std::error_code ec;
    if (!fs::exists(r.library, ec)) {
        *reason = "no library at " + PathText(r.library);
        return false;
    }
    const fs::path dir = fs::temp_directory_path(ec) / "nuka_editor_ext" / std::to_string(ProcessId());
    fs::create_directories(dir, ec);
    s.shadow = dir / Utf8Path(r.id + "-" + std::to_string(++shadows) + PathText(r.library.extension()));
    fs::copy_file(r.library, s.shadow, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        *reason = "cannot copy " + PathText(r.library) + ": " + ec.message();
        return false;
    }
    s.library = OpenLibrary(s.shadow, reason);
    if (s.library == nullptr) return false;
    const auto abi = Symbol<nuka_editor_plugin_abi_fn>(s.library, "nuka_editor_plugin_abi");
    const auto load = Symbol<nuka_editor_plugin_load_fn>(s.library, "nuka_editor_plugin_load");
    const auto unload = Symbol<nuka_editor_plugin_unload_fn>(s.library, "nuka_editor_plugin_unload");
    if (abi == nullptr || load == nullptr || unload == nullptr) {
        *reason = "not an editor plugin: nuka_editor_plugin_abi, _load or _unload is not exported";
        return false;
    }
    if (abi() != NUKA_EDITOR_PLUGIN_ABI) {
        *reason = Fmt("built for editor plugin ABI %u; this editor speaks %u", abi(), NUKA_EDITOR_PLUGIN_ABI);
        return false;
    }
    if (const auto check = Symbol<nuka_editor_plugin_imgui_fn>(s.library, "nuka_editor_plugin_imgui")) {
        nuka_editor_imgui_check_t theirs{};
        theirs.struct_size = static_cast<uint32_t>(sizeof(theirs));
        check(&theirs);
        *reason = ImGuiMismatch(theirs);
        if (!reason->empty()) return false;
        ImGuiMemAllocFunc alloc = nullptr;
        ImGuiMemFreeFunc free_fn = nullptr;
        void* user = nullptr;
        ImGui::GetAllocatorFunctions(&alloc, &free_fn, &user);
        s.api.imgui_context = ImGui::GetCurrentContext();
        s.api.imgui_alloc = alloc;
        s.api.imgui_free = free_fn;
        s.api.imgui_alloc_user = user;
    }
    void* state = nullptr;
    const nuka_result_t result = load(&s.api, &state);
    if (result != NUKA_RESULT_OK) {
        *reason = Fmt("its load returned nuka_result_t %d", static_cast<int>(result));
        return false;
    }
    s.plugin_state = state;
    s.unload = unload;
    return true;
}

// Python stops first, then the plugin; what they registered goes before their code unloads.
void Impl::Teardown(Record& r) {
    Scope& s = *r.scope;
    if (s.python) {
        s.python = false;
        if (const char* out = python.unload(python.user_data, r.id.c_str())) Log(r, NUKA_NUI_INFO, out);
    }
    if (s.unload != nullptr) {
        s.unload(s.plugin_state);
        s.unload = nullptr;
    }
    s.overlays.clear();
    services.actions->RemoveOwned(r.id);
    for (const std::string& pid : s.panels) {
        if (r.builtin != nullptr) services.ui->RemoveBuiltinPanel(pid);
        else services.ui->Nui().RemovePanel(pid);
    }
    if (s.library != nullptr) CloseLibrary(s.library);
    std::error_code ec;
    if (!s.shadow.empty()) fs::remove(s.shadow, ec);
    r.scope.reset();
    r.state = State::Disabled;
    ++revision;
}

// Dependents go first.
void Impl::Disable(Record& r) {
    for (const auto& other : records)
        if (other->state == State::Enabled && DependsOn(*other, r.id)) Disable(*other);
    if (r.state == State::Enabled) {
        Teardown(r);
        if (r.builtin == nullptr) Log(r, NUKA_NUI_INFO, "disabled");
    }
    r.state = State::Disabled;
    r.error.clear();
    ++revision;
}

// Re-reads the manifest and restarts the extension, and the dependents it took down.
void Impl::Reload(Record& r) {
    std::vector<Record*> running;
    for (const auto& other : records)
        if (other->state == State::Enabled && other.get() != &r) running.push_back(other.get());
    Disable(r);
    if (r.builtin == nullptr) ReadManifest(r);
    std::vector<std::string> chain;
    if (Enable(r, chain)) Log(r, NUKA_NUI_OK, "reloaded");
    for (Record* other : running) {
        chain.clear();
        if (other->state != State::Enabled) Enable(*other, chain);
    }
}

json::Value Impl::Describe(const Record& r) const {
    json::Value v = json::Value::Object();
    v.Set("id", json::Value::Str(r.id));
    v.Set("title", json::Value::Str(r.title));
    v.Set("version", json::Value::Str(r.version));
    v.Set("description", json::Value::Str(r.description));
    v.Set("kind", json::Value::Str(KindName(r)));
    v.Set("state", json::Value::Str(StateName(r.state)));
    v.Set("error", json::Value::Str(r.error));
    v.Set("auto_disabled", json::Value::Bool(r.auto_disabled));
    v.Set("dir", json::Value::Str(PathText(r.dir)));
    json::Value actions = json::Value::Array();
    for (const Action& a : services.actions->All())
        if (a.owner == r.id) actions.PushBack(json::Value::Str(a.id));
    v.Set("actions", std::move(actions));
    json::Value panels = json::Value::Array();
    json::Value overlays = json::Value::Array();
    if (r.scope) {
        for (const std::string& pid : r.scope->panels) panels.PushBack(json::Value::Str(pid));
        for (const Overlay& o : r.scope->overlays) overlays.PushBack(json::Value::Str(o.id));
    }
    v.Set("panels", std::move(panels));
    v.Set("overlays", std::move(overlays));
    return v;
}

void Impl::RegisterActions() {
    ActionRegistry& registry = *services.actions;
    const auto record = [this](const json::Value& args) -> Record& {
        const std::string id = args.At("id").AsString();
        Record* r = Find(id);
        if (r == nullptr) throw std::runtime_error("no extension '" + id + "'");
        return *r;
    };
    registry.Add({"extensions.manager", "Extension Manager", "extensions", "", ShortcutScope::Global, {},
                  [this] {
                      std::lock_guard<std::recursive_mutex> lock(services.ui->Nui().Mutex());
                      return services.ui->Nui().FindPanel("extensions")->open;
                  },
                  [this](const json::Value&) {
                      std::lock_guard<std::recursive_mutex> lock(services.ui->Nui().Mutex());
                      nui::Panel* panel = services.ui->Nui().FindPanel("extensions");
                      panel->open = !panel->open;
                      panel->focus_request = panel->open;
                      return ActionResult{};
                  },
                  icon::kPlug});
    registry.Add({"extensions.rescan", "Rescan Extensions", "extensions", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      const size_t before = records.size();
                      Discover();
                      ++revision;
                      return ActionResult{true, json::Value::Int(static_cast<int64_t>(records.size() - before)), {}};
                  },
                  icon::kReset});
    registry.Add({"extensions.list", "List Extensions", "extensions", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      json::Value list = json::Value::Array();
                      for (const auto& r : records) list.PushBack(Describe(*r));
                      return ActionResult{true, std::move(list), {}};
                  },
                  "", true});
    registry.Add({"extensions.enable", "Enable Extension", "extensions", "", ShortcutScope::Global, {}, {},
                  [this, record](const json::Value& args) {
                      Record& r = record(args);
                      r.wanted = true;
                      std::vector<std::string> chain;
                      if (!Enable(r, chain)) return ActionResult::Fail(r.error);
                      return ActionResult{true, Describe(r), {}};
                  },
                  "", true});
    registry.Add({"extensions.disable", "Disable Extension", "extensions", "", ShortcutScope::Global, {}, {},
                  [this, record](const json::Value& args) {
                      Record& r = record(args);
                      r.wanted = false;
                      Disable(r);
                      return ActionResult{true, Describe(r), {}};
                  },
                  "", true});
    registry.Add({"extensions.reload", "Reload Extension", "extensions", "", ShortcutScope::Global, {}, {},
                  [this, record](const json::Value& args) {
                      Record& r = record(args);
                      r.wanted = true;
                      Reload(r);
                      if (r.state != State::Enabled) return ActionResult::Fail(r.error);
                      return ActionResult{true, Describe(r), {}};
                  },
                  "", true});
    registry.Add({"extensions.log", "Extension Log", "extensions", "", ShortcutScope::Global, {}, {},
                  [this, record](const json::Value& args) {
                      DrainLogs();
                      return ActionResult{true, json::Value::Str(record(args).log), {}};
                  },
                  "", true});
}

// ---- the Extensions panel -----------------------------------------------------------------

void Impl::BuildPanel(nui::VStack& root) {
    root.Add<nui::Custom>([this] { DrawHeader(); });
    root.Add<nui::Custom>([this] { DrawCards(); });
    Record* r = Find(selected);
    if (r == nullptr) return;
    root.Add<nui::Separator>("Log");
    root.Add<nui::Label>(r->title, Tone::Muted, TextStyle::Label);
    root.Add<nui::LogView>(&r->log);
}

void Impl::DrawHeader() {
    const ImVec2 start = ImGui::GetCursorScreenPos();
    const float right = start.x + ImGui::GetContentRegionAvail().x;
    const auto count = [this](State state) {
        return static_cast<int>(std::count_if(records.begin(), records.end(), [state](const auto& r) { return r->state == state; }));
    };
    const int failed = count(State::Failed);
    nui::PushTextStyle(TextStyle::Label);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(theme::kFgMuted, "%s", Fmt(Tr("%d of %d enabled"), count(State::Enabled), static_cast<int>(records.size())).c_str());
    if (failed > 0) {
        ImGui::SameLine();
        ImGui::TextColored(theme::kErr, "%s", Fmt(Tr("%d failed"), failed).c_str());
    }
    nui::PopTextStyle();
    ImGui::SameLine();
    ImGui::SetCursorScreenPos(ImVec2(right - ImGui::GetFrameHeight() - 2.0f * nui::Scale(), start.y));
    if (draw::IconButton("##rescan", icon::kReset, false, Tr("Rescan Extensions")))
        services.ui->Nui().actions.run("extensions.rescan", json::Value::Object());
}

// One chamfered card per extension: state, kind, the power and reload switches, and the error.
void Impl::DrawCards() {
    const float s = nui::Scale();
    ImDrawList* list = ImGui::GetWindowDrawList();
    const float pad = 8.0f * s;
    for (const auto& rec : records) {
        Record& r = *rec;
        ImGui::PushID(r.id.c_str());
        const ImVec2 min = ImGui::GetCursorScreenPos();
        const float width = ImGui::GetContentRegionAvail().x;
        const float right = min.x + width - pad;
        list->ChannelsSplit(2);
        list->ChannelsSetCurrent(1);
        ImGui::SetCursorScreenPos(ImVec2(min.x + pad, min.y + pad));
        ImGui::BeginGroup();
        const Tone tone = r.state == State::Enabled ? Tone::Ok : r.state == State::Failed ? Tone::Err : Tone::Dim;
        draw::StatusDot(tone, false);
        ImGui::SameLine(0.0f, 2.0f * s);
        nui::PushTextStyle(TextStyle::Strong);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(Tr(r.title).c_str());
        nui::PopTextStyle();
        ImGui::SameLine();
        draw::Badge(Tr(KindName(r)), r.builtin != nullptr ? Tone::Dim : Tone::Info);
        const float button = ImGui::GetFrameHeight() + 2.0f * s;
        const float buttons = r.builtin != nullptr ? 1.0f : 2.0f;
        ImGui::SameLine();
        ImGui::SetCursorScreenPos(ImVec2(right - buttons * button - (buttons - 1.0f) * 2.0f * s, min.y + pad));
        if (r.builtin == nullptr) {
            if (draw::IconButton("##reload", icon::kReset, false, Tr("Reload"))) Queue("extensions.reload", r.id);
            ImGui::SameLine(0.0f, 2.0f * s);
        }
        const bool on = r.state == State::Enabled;
        if (draw::IconButton("##power", icon::kPower, on, Tr(on ? "Disable" : "Enable")))
            Queue(on ? "extensions.disable" : "extensions.enable", r.id);
        nui::PushTextStyle(TextStyle::Mono);
        ImGui::TextColored(theme::kFgDim, "%s  v%s", r.id.c_str(), r.version.c_str());
        nui::PopTextStyle();
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width - 2.0f * pad);
        if (!r.description.empty()) ImGui::TextColored(theme::kFgMuted, "%s", Tr(r.description).c_str());
        if (!r.error.empty()) ImGui::TextColored(theme::kErr, "%s", r.error.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
        const ImVec2 max(min.x + width, ImGui::GetItemRectMax().y + pad);
        list->ChannelsSetCurrent(0);
        const bool chosen = r.id == selected;
        draw::Chamfer(list, min, max, theme::kChamfer * s, ImGui::GetColorU32(chosen ? theme::kBg2 : theme::kBg0),
                      ImGui::GetColorU32(chosen ? theme::kAccentDim : theme::kLine));
        list->AddRectFilled(ImVec2(min.x, min.y + theme::kChamfer * s), ImVec2(min.x + 2.0f * s, max.y),
                            ImGui::GetColorU32(nui::ToneColor(tone)));
        list->ChannelsMerge();
        if (ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(min, max) && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
            selected != r.id) {
            selected = r.id;
            ++revision;
        }
        ImGui::SetCursorScreenPos(ImVec2(min.x, max.y + theme::kSpace * s));
        ImGui::Dummy(ImVec2(width, 0.0f));
        ImGui::PopID();
    }
}

// ---- the manager --------------------------------------------------------------------------

ExtensionManager::ExtensionManager() : impl_(std::make_unique<Impl>()) {}

ExtensionManager::~ExtensionManager() = default;

void ExtensionManager::Configure(ExtensionServices services, std::vector<fs::path> folders,
                                 std::vector<std::string> startup) {
    Impl& m = *impl_;
    m.services = std::move(services);
    m.folders = std::move(folders);
    m.startup = std::move(startup);
    for (const ImGuiLayer::BuiltinExtension& builtin : ImGuiLayer::Builtins()) {
        auto record = std::make_unique<Record>();
        record->id = builtin.id;
        record->title = builtin.title;
        record->description = builtin.description;
        record->version = "1.0.0";
        record->builtin = &builtin;
        record->wanted = true;
        m.records.push_back(std::move(record));
    }
    m.Discover();
    m.RegisterActions();
    // Built-ins start now, so their panels and actions exist before the first frame.
    for (const auto& r : m.records) {
        std::vector<std::string> chain;
        if (r->builtin != nullptr) m.Enable(*r, chain);
    }
    nui::Host& host = m.services.ui->Nui();
    std::lock_guard<std::recursive_mutex> lock(host.Mutex());
    nui::Panel& panel = host.AddPanel("extensions", "Extensions", icon::kPlug);
    panel.dock = "right";
    panel.open = false;
    panel.key = [&m] { return m.revision; };
    panel.build = [&m](nui::VStack& root) { m.BuildPanel(root); };
}

void ExtensionManager::SetPythonHooks(const nuka_editor_python_hooks_t& hooks) { impl_->python = hooks; }

void ExtensionManager::Start() {
    Impl& m = *impl_;
    for (const std::string& id : m.startup) {
        if (Record* r = m.Find(id)) r->wanted = true;
        else m.services.console("[extensions] --ext " + id + ": no such extension\n");
    }
    for (const auto& r : m.records) {
        if (!r->wanted) continue;
        std::vector<std::string> chain;
        m.Enable(*r, chain);
    }
}

void ExtensionManager::Shutdown() {
    Impl& m = *impl_;
    for (auto it = m.records.rbegin(); it != m.records.rend(); ++it) m.Disable(**it);
}

void ExtensionManager::Poll() {
    Impl& m = *impl_;
    m.DrainLogs();
    const Clock::time_point now = Clock::now();
    if (now < m.next_poll) return;
    m.next_poll = now + kPollInterval;
    for (size_t i = 0; i < m.records.size(); ++i) {
        Record& r = *m.records[i];
        if (r.builtin != nullptr || !r.wanted || r.state == State::Disabled) continue;
        const fs::file_time_type stamp = m.Stamp(r);
        if (stamp == r.stamp) continue;
        m.Log(r, NUKA_NUI_INFO, "files changed; reloading");
        m.Reload(r);
    }
}

void ExtensionManager::DrawOverlays(const CameraController& camera, uint64_t tick) {
    Impl& m = *impl_;
    const SceneViewportRect& vp = m.services.ui->Viewport();
    if (!vp.Valid()) return;
    nuka_editor_overlay_frame_t frame{};
    frame.struct_size = static_cast<uint32_t>(sizeof(frame));
    m.services.ui->CameraMatrices(camera, frame.view, frame.proj);
    Canvas canvas{};
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum += frame.proj[k * 4 + row] * frame.view[col * 4 + k];
            canvas.clip[col * 4 + row] = sum;
        }
    canvas.x = vp.x;
    canvas.y = vp.y;
    canvas.w = vp.width;
    canvas.h = vp.height;
    canvas.list = ImGui::GetBackgroundDrawList();
    frame.viewport[0] = vp.x;
    frame.viewport[1] = vp.y;
    frame.viewport[2] = vp.width;
    frame.viewport[3] = vp.height;
    frame.tick = tick;
    frame.host = &canvas;
    frame.line = &CanvasLine;
    frame.circle = &CanvasCircle;
    frame.text = &CanvasText;
    struct Outcome {
        std::string owner, callback;
        bool ok;
    };
    std::vector<Outcome> outcomes;
    canvas.list->PushClipRect(ImVec2(vp.x, vp.y), ImVec2(vp.x + vp.width, vp.y + vp.height), true);
    for (const auto& r : m.records) {
        if (!r->scope) continue;
        for (const Overlay& o : r->scope->overlays)
            outcomes.push_back({r->id, "overlay " + o.id, o.draw(o.user.get(), &frame) != 0});
    }
    canvas.list->PopClipRect();
    for (const Outcome& o : outcomes) Report(o.owner, o.callback, o.ok);
}

void ExtensionManager::Report(const std::string& owner, const std::string& callback, bool ok) {
    Impl& m = *impl_;
    Record* r = m.Find(owner);
    if (r == nullptr || r->state != State::Enabled) return;
    int& count = r->failures[callback];
    count = ok ? 0 : count + 1;
    if (count < kMaxFailures) return;
    const std::string reason = "'" + callback + "' failed " + std::to_string(kMaxFailures) + " times in a row; disabled";
    m.Disable(*r);
    r->state = State::Failed;
    r->auto_disabled = true;
    r->error = reason;
    m.Log(*r, NUKA_NUI_WARN, reason);
    m.services.ui->Nui().Toast(nui::Level::Warn, Tr(r->title) + ": " + Tr("disabled after repeated failures"));
}

const nuka_editor_host_api_t* ExtensionManager::HostApi(const std::string& id) const {
    for (const auto& r : impl_->records)
        if (r->id == id && r->scope) return &r->scope->api;
    return nullptr;
}

}  // namespace nuka::runtime::app::viewer
