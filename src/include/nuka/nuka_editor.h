#ifndef NUKA_NUKA_EDITOR_H
#define NUKA_NUKA_EDITOR_H

// The editor shared library: a window (or an offscreen surface) over a world the
// host owns, or one the editor creates from a scene file. Plain C crosses the boundary.

#include "nuka/nuka.h"
#include "nuka/nuka_editor_plugin.h"
#include "nuka/nuka_nui.h"

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

// Hooks run in the host on the physics thread, else the updating thread; interrupt and complete
// may come from any thread. Returned text is valid until that thread's next hook call; NULL: none.
typedef struct nuka_editor_script_hooks_t {
    void* user_data;
    // Runs `code` compiled as `filename` ("" for the console); nonzero `echo` prints the value of
    // a final expression. What it prints goes to nuka_editor_console_append line by line as it
    // prints, and comes back whole; `failed` turns nonzero when the code raised.
    const char* (*run)(void* user_data, const char* code, const char* filename, uint32_t echo,
                       int32_t* failed);
    const char* (*register_script)(void* user_data, uint64_t script_id, const char* source);
    void (*unregister_script)(void* user_data, uint64_t script_id);
    void (*clear_scripts)(void* user_data);
    const char* (*dispatch)(void* user_data, nuka_editor_callback_t callback);
    // Raises ScriptInterrupted in every hook call in progress.
    void (*interrupt)(void* user_data);
    // Completions of `code` at 1-based `line` and 0-based `column` in the console's namespace,
    // as a JSON array of {"name", "type", "signature"}.
    const char* (*complete)(void* user_data, const char* code, int32_t line, int32_t column);
} nuka_editor_script_hooks_t;

// Hooks the host's Python runs extensions through, called on the UI thread. Returned text is
// the call's output or failure, valid until the next hook call; NULL means none.
typedef struct nuka_editor_python_hooks_t {
    void* user_data;
    // Imports `module` from `dir` and starts its extension; nonzero `failed` when it did not.
    const char* (*load)(void* user_data, const char* extension_id, const char* module, const char* dir,
                        int32_t* failed);
    // Stops the extension and forgets its modules.
    const char* (*unload)(void* user_data, const char* extension_id);
} nuka_editor_python_hooks_t;

// Hooks the host's endpoint for other programs (agents) runs through, called on the UI thread.
// The editor opens or closes it as the agent access mode leaves or enters "off".
typedef struct nuka_editor_rpc_hooks_t {
    void* user_data;
    // Opens the endpoint when `on` is nonzero; zero closes it and drops its clients.
    void (*listen)(void* user_data, uint32_t on);
    // Drops every connected client; the endpoint stays open.
    void (*disconnect)(void* user_data);
} nuka_editor_rpc_hooks_t;

// argv holds the launcher flags and an optional scene path. A non-NULL world is borrowed and
// must outlive the editor; otherwise the first load creates a world the editor owns.
NUKA_EDITOR_API nuka_result_t nuka_editor_create(int argc, const char* const* argv,
                                                 nuka_world_handle world, float dt,
                                                 nuka_editor_handle* out);
// Why this thread's last nuka_editor_create failed, naming the session log; "" after a success.
NUKA_EDITOR_API const char* nuka_editor_create_error(void);
// Hands the scene argv names to the editor already running on the same profile, which opens it
// and comes to the front. 0 when argv names no scene or no such editor runs.
NUKA_EDITOR_API uint32_t nuka_editor_forward(int argc, const char* const* argv);
NUKA_EDITOR_API void nuka_editor_destroy(nuka_editor_handle editor);
NUKA_EDITOR_API nuka_result_t nuka_editor_set_script_hooks(nuka_editor_handle editor,
                                                           const nuka_editor_script_hooks_t* hooks);
NUKA_EDITOR_API nuka_result_t nuka_editor_set_python_hooks(nuka_editor_handle editor,
                                                           const nuka_editor_python_hooks_t* hooks);
// Calls listen at once with the current access mode.
NUKA_EDITOR_API nuka_result_t nuka_editor_set_rpc_hooks(nuka_editor_handle editor,
                                                        const nuka_editor_rpc_hooks_t* hooks);
// The endpoint's clients, a JSON array of {"client", "pid", "since"}; any thread may report them.
NUKA_EDITOR_API void nuka_editor_rpc_clients(nuka_editor_handle editor, const char* clients_json);
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

// Every editor command is an action ("edit.apply", "sim.play", ...) with JSON args (NULL: none).
// The reply {"ok","value","error"} lives until the calling thread's next text-returning call; INVALID_ARG if not ok.
NUKA_EDITOR_API nuka_result_t nuka_editor_run_action(nuka_editor_handle editor, const char* id,
                                                     const char* args_json, const char* source,
                                                     const char** out_json);
// The registered actions with their labels, shortcuts and enabled/checked state.
NUKA_EDITOR_API nuka_result_t nuka_editor_actions(nuka_editor_handle editor, const char** out_json);
// The action runs logged after sequence number `since`, oldest first.
NUKA_EDITOR_API nuka_result_t nuka_editor_action_log(nuka_editor_handle editor, uint64_t since,
                                                     const char** out_json);
NUKA_EDITOR_API nuka_result_t nuka_editor_document_state(nuka_editor_handle editor, const char** out_json);

// The UI component table for panels built by Python or extensions; lives as long as the editor.
NUKA_EDITOR_API const nuka_nui_api_t* nuka_editor_nui(nuka_editor_handle editor);
// The host table of an enabled extension, for the Python half of its scope; NULL otherwise.
// It lives until the extension is disabled.
NUKA_EDITOR_API const nuka_editor_host_api_t* nuka_editor_extension_host(nuka_editor_handle editor,
                                                                         const char* extension_id);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_EDITOR_H */
