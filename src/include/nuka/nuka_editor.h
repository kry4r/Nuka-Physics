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

typedef enum nuka_editor_callback_t {
    NUKA_EDITOR_CALLBACK_READY = 0,
    NUKA_EDITOR_CALLBACK_STEP = 1,
    NUKA_EDITOR_CALLBACK_RESET = 2
} nuka_editor_callback_t;

// Scripts run in the host process; the editor links no interpreter. Returned text is
// console output that stays valid until the next hook call, and NULL means none.
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
// One frame of input, edits, pose publishing and presentation. Physics advances with the
// transport only when step_physics is nonzero; otherwise the host steps the world.
NUKA_EDITOR_API nuka_result_t nuka_editor_update(nuka_editor_handle editor, uint32_t step_physics);
NUKA_EDITOR_API uint32_t nuka_editor_running(nuka_editor_handle editor);
NUKA_EDITOR_API uint32_t nuka_editor_playing(nuka_editor_handle editor);
NUKA_EDITOR_API void nuka_editor_close(nuka_editor_handle editor);
// NULL until a world exists; loads and structural edits rebuild it in place.
NUKA_EDITOR_API nuka_world_handle nuka_editor_world(nuka_editor_handle editor);
NUKA_EDITOR_API float nuka_editor_dt(nuka_editor_handle editor);
NUKA_EDITOR_API void nuka_editor_console_append(nuka_editor_handle editor, const char* text);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_EDITOR_H */
