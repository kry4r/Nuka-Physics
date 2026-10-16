#ifndef NUKA_NUKA_EDITOR_H
#define NUKA_NUKA_EDITOR_H

// The editor shared library: a window (or an offscreen surface) over a world the
// host owns, or one the editor creates from a scene file. Plain C crosses the boundary.

#include "nuka/nuka.h"

#if defined(_WIN32)
#if defined(NUKA_EDITOR_BUILD)
#define NUKA_EDITOR_API __declspec(dllexport)
#else
#define NUKA_EDITOR_API __declspec(dllimport)
#endif
#else
#define NUKA_EDITOR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nuka_editor_t* nuka_editor_handle;

// READY, STEP and RESET reach the scene's scripts; TICK reaches the host's tick handlers at
// every tick boundary (each transport step, or each update when the host steps the world).
typedef enum nuka_editor_callback_t {
    NUKA_EDITOR_CALLBACK_READY = 0,
    NUKA_EDITOR_CALLBACK_STEP = 1,
    NUKA_EDITOR_CALLBACK_RESET = 2,
    NUKA_EDITOR_CALLBACK_TICK = 3
} nuka_editor_callback_t;

// Hooks run in the host on the physics thread, else the updating thread. Returned text is
// console output valid until that thread's next hook call; NULL means none.
typedef struct nuka_editor_script_hooks_t {
    void* user_data;
    const char* (*run)(void* user_data, const char* code);
    const char* (*register_script)(void* user_data, uint64_t script_id, const char* source);
    void (*unregister_script)(void* user_data, uint64_t script_id);
    void (*clear_scripts)(void* user_data);
    const char* (*dispatch)(void* user_data, nuka_editor_callback_t callback);
} nuka_editor_script_hooks_t;

// argv holds the launcher flags and an optional scene path. A non-NULL world is borrowed and
// must outlive the editor; otherwise the first load creates a world the editor owns.
NUKA_EDITOR_API nuka_result_t nuka_editor_create(int argc, const char* const* argv,
                                                 nuka_world_handle world, float dt,
                                                 nuka_editor_handle* out);
NUKA_EDITOR_API void nuka_editor_destroy(nuka_editor_handle editor);
NUKA_EDITOR_API nuka_result_t nuka_editor_set_script_hooks(nuka_editor_handle editor,
                                                           const nuka_editor_script_hooks_t* hooks);
// One frame of input, edits and presentation. With step_physics nonzero a physics thread
// advances the transport; with zero none runs and the host steps the world between updates.
NUKA_EDITOR_API nuka_result_t nuka_editor_update(nuka_editor_handle editor, uint32_t step_physics);
// Joins the physics thread, so the host may call the world until the next update with step_physics.
NUKA_EDITOR_API void nuka_editor_stop_physics(nuka_editor_handle editor);
NUKA_EDITOR_API uint32_t nuka_editor_running(nuka_editor_handle editor);
NUKA_EDITOR_API uint32_t nuka_editor_playing(nuka_editor_handle editor);
NUKA_EDITOR_API void nuka_editor_close(nuka_editor_handle editor);
// NULL until a world exists; loads and structural edits rebuild it in place.
NUKA_EDITOR_API nuka_world_handle nuka_editor_world(nuka_editor_handle editor);
NUKA_EDITOR_API float nuka_editor_dt(nuka_editor_handle editor);
NUKA_EDITOR_API void nuka_editor_console_append(nuka_editor_handle editor, const char* text);

// Applies a JSON EditTxn {"label", "ops": [{"target", "field", "before"?, "after"}]}; live
// parameters land at the next tick and other edits rebuild the world. INVALID_ARG on failure.
NUKA_EDITOR_API nuka_result_t nuka_editor_apply_edit(nuka_editor_handle editor, const char* txn_json);
// NOT_SUPPORTED when the history is empty or the world refused the rebuild.
NUKA_EDITOR_API nuka_result_t nuka_editor_undo(nuka_editor_handle editor);
NUKA_EDITOR_API nuka_result_t nuka_editor_redo(nuka_editor_handle editor);
// The document state as JSON. *out_length gets its length; a NULL out only queries it, and
// capacity must exceed it.
NUKA_EDITOR_API nuka_result_t nuka_editor_document_state(nuka_editor_handle editor, char* out,
                                                         size_t capacity, size_t* out_length);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_EDITOR_H */
