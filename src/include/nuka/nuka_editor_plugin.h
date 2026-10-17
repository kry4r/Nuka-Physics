#ifndef NUKA_NUKA_EDITOR_PLUGIN_H
#define NUKA_NUKA_EDITOR_PLUGIN_H

// The native editor extension ABI: plain C with fixed-width fields and sized, append-only
// structs, so a plugin built by another compiler, runtime or flag set loads.

#include "nuka/nuka.h"
#include "nuka/nuka_nui.h"

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define NUKA_EDITOR_PLUGIN_EXPORT __declspec(dllexport)
#else
#define NUKA_EDITOR_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NUKA_EDITOR_PLUGIN_ABI 1u

// Runs an action: 1 with *out_json a JSON value (NULL: null), or 0 with *out_json the reason.
// The text stays valid until the plugin's next callback.
typedef int32_t (*nuka_editor_action_fn)(void* user, const char* args_json, const char** out_json);

typedef struct nuka_editor_action_desc_t {
    uint32_t struct_size;
    const char* id;        // unique across the editor, e.g. "hello_cpp.greet"
    const char* label;     // English; string tables translate it
    const char* category;  // its menu: "extensions" (NULL), "tools", "view", ...
    const char* shortcut;  // "Ctrl+Shift+H", or NULL
    const char* icon;      // a shell icon name ("robot", "play", ...), or NULL
    uint32_t hidden;       // nonzero: it takes arguments, so menus and the palette skip it
    nuka_editor_action_fn run;
    void* user;
    void (*release)(void* user);  // once the action is gone; may be NULL
} nuka_editor_action_desc_t;

// One frame over the scene viewport: column-major view and GL-style projection, the
// viewport in window pixels, and draw calls taking world points. rgba is 0xRRGGBBAA.
typedef struct nuka_editor_overlay_frame_t {
    uint32_t struct_size;
    float view[16];
    float proj[16];
    float viewport[4];  // x, y, width, height
    uint64_t tick;
    void* host;
    void (*line)(void* host, const float a[3], const float b[3], uint32_t rgba, float thickness);
    void (*circle)(void* host, const float center[3], float radius_px, uint32_t rgba, float thickness);
    void (*text)(void* host, const float at[3], uint32_t rgba, const char* text);
} nuka_editor_overlay_frame_t;

// draw returns 0 when it failed; repeated failures disable the extension.
typedef struct nuka_editor_overlay_desc_t {
    uint32_t struct_size;
    const char* id;
    int32_t (*draw)(void* user, const nuka_editor_overlay_frame_t* frame);
    void* user;
    void (*release)(void* user);
} nuka_editor_overlay_desc_t;

// The Dear ImGui build a plugin compiled against; see nuka_editor_plugin_imgui.
typedef struct nuka_editor_imgui_check_t {
    uint32_t struct_size;
    uint32_t version_num;  // IMGUI_VERSION_NUM
    uint32_t size_io;
    uint32_t size_style;
    uint32_t size_vec2;
    uint32_t size_vec4;
    uint32_t size_draw_vert;
    uint32_t size_draw_idx;
    uint32_t size_wchar;
} nuka_editor_imgui_check_t;

// The host as one extension sees it. Everything registered through it belongs to the
// extension and goes away when it is disabled or reloaded.
typedef struct nuka_editor_host_api_t {
    uint32_t struct_size;
    uint32_t abi_version;
    void* host;
    const char* extension_id;
    const char* extension_dir;  // the folder holding extension.toml
    const char* settings_json;  // the manifest's [settings] table
    const nuka_nui_api_t* nui;  // panels made here belong to the extension
    nuka_result_t (*register_action)(void* host, const nuka_editor_action_desc_t* desc);
    nuka_result_t (*register_overlay)(void* host, const nuka_editor_overlay_desc_t* desc);
    // Runs any editor action now, like nuka_editor_run_action; the reply lives until the next call.
    nuka_result_t (*run_action)(void* host, const char* id, const char* args_json, const char** out_json);
    // To the extension's log and the console.
    void (*log)(void* host, nuka_nui_level_t level, const char* text);
    // Set only when the plugin exported nuka_editor_plugin_imgui and its build matched.
    void* imgui_context;
    void* (*imgui_alloc)(size_t size, void* user);
    void (*imgui_free)(void* ptr, void* user);
    void* imgui_alloc_user;
    // A panel whose content the plugin draws with Dear ImGui; needs the fields above.
    nuka_result_t (*register_imgui_panel)(void* host, const char* id, const char* title, const char* dock,
                                          void (*draw)(void* user), void* user, void (*release)(void* user));
} nuka_editor_host_api_t;

typedef uint32_t (*nuka_editor_plugin_abi_fn)(void);
typedef nuka_result_t (*nuka_editor_plugin_load_fn)(const nuka_editor_host_api_t* host, void** state);
typedef void (*nuka_editor_plugin_unload_fn)(void* state);
typedef void (*nuka_editor_plugin_imgui_fn)(nuka_editor_imgui_check_t* out);

// A plugin's exports; the host table lives until unload. One that calls Dear ImGui also exports
// nuka_editor_plugin_imgui, and a build that differs from the editor's is refused.
NUKA_EDITOR_PLUGIN_EXPORT uint32_t nuka_editor_plugin_abi(void);
NUKA_EDITOR_PLUGIN_EXPORT nuka_result_t nuka_editor_plugin_load(const nuka_editor_host_api_t* host, void** state);
NUKA_EDITOR_PLUGIN_EXPORT void nuka_editor_plugin_unload(void* state);
NUKA_EDITOR_PLUGIN_EXPORT void nuka_editor_plugin_imgui(nuka_editor_imgui_check_t* out);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_EDITOR_PLUGIN_H */
