#ifndef NUKA_NUKA_NUI_H
#define NUKA_NUKA_NUI_H

// The editor's UI component tree as a C function table, for Python and native extensions.
// Nodes are handles; props and values travel as JSON. Calls may come from any thread.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUKA_NUI_API_VERSION 1u

// event_json is {"type": "click" | "change" | "commit" | "select" | "run" | ..., "value": ...}.
// Events are delivered on the UI thread at the top of the next frame, outside any draw.
typedef void (*nuka_nui_event_fn)(void* user, uint32_t node, const char* event_json);

typedef enum nuka_nui_level_t {
    NUKA_NUI_INFO = 0,
    NUKA_NUI_OK = 1,
    NUKA_NUI_WARN = 2,
    NUKA_NUI_ERROR = 3
} nuka_nui_level_t;

typedef struct nuka_nui_api_t {
    uint32_t struct_size;
    uint32_t version;
    void* host;
    // A dockable panel ("left", "right", "bottom" or "float"); returns its root node, reusing
    // the panel when the id exists. 0 on failure.
    uint32_t (*panel)(void* host, const char* id, const char* title, const char* dock);
    // Appends a node of `kind` ("label", "button", "vstack", ...) under `parent`; 0 on failure.
    uint32_t (*add)(void* host, uint32_t parent, const char* kind, const char* props_json);
    int32_t (*set)(void* host, uint32_t node, const char* props_json);
    // Removes a node and its subtree; a panel root removes the panel.
    int32_t (*remove)(void* host, uint32_t node);
    int32_t (*clear)(void* host, uint32_t node);
    // The node's value as JSON, valid until the next call on this table.
    const char* (*get)(void* host, uint32_t node);
    int32_t (*on_event)(void* host, uint32_t node, nuka_nui_event_fn fn, void* user);
    void (*toast)(void* host, nuka_nui_level_t level, const char* text);
    // fraction in [0, 1], or < 0 to remove the entry.
    void (*progress)(void* host, const char* id, const char* label, float fraction);
    const char* (*last_error)(void* host);
} nuka_nui_api_t;

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_NUI_H */
