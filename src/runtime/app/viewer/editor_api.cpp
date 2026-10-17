#include "nuka/nuka_editor.h"

#include "runtime/app/viewer/editor_host.hpp"

#include <cstddef>
#include <cstdio>
#include <exception>
#include <new>
#include <string>
#include <vector>

struct nuka_editor_t : nuka::runtime::app::viewer::EditorHost {
    using EditorHost::EditorHost;
    std::string reply_;  // the text the last text-returning call handed out
};

namespace {

template <class Fn>
nuka_result_t Guard(Fn&& fn) {
    try {
        fn();
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[nuka_editor] %s\n", e.what());
        return NUKA_RESULT_INTERNAL;
    }
}

}  // namespace

extern "C" {

nuka_result_t nuka_editor_create(int argc, const char* const* argv, nuka_world_handle world, float dt,
                                 nuka_editor_handle* out) {
    if (out == nullptr || argc < 0 || (argc > 0 && argv == nullptr)) return NUKA_RESULT_INVALID_ARG;
    *out = nullptr;
    return Guard([&] { *out = new nuka_editor_t(std::vector<std::string>(argv, argv + argc), world, dt); });
}

void nuka_editor_destroy(nuka_editor_handle editor) { delete editor; }

nuka_result_t nuka_editor_set_script_hooks(nuka_editor_handle editor, const nuka_editor_script_hooks_t* hooks) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    return Guard([&] { editor->SetScriptHooks(hooks != nullptr ? *hooks : nuka_editor_script_hooks_t{}); });
}

nuka_result_t nuka_editor_update(nuka_editor_handle editor, uint32_t step_physics) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    return Guard([&] { editor->Update(step_physics != 0u); });
}

void nuka_editor_stop_physics(nuka_editor_handle editor) { editor->StopPhysics(); }

uint32_t nuka_editor_running(nuka_editor_handle editor) { return editor->Running() ? 1u : 0u; }

uint32_t nuka_editor_playing(nuka_editor_handle editor) { return editor->Playing() ? 1u : 0u; }

void nuka_editor_close(nuka_editor_handle editor) { editor->Close(); }

nuka_world_handle nuka_editor_world(nuka_editor_handle editor) { return editor->World(); }

float nuka_editor_dt(nuka_editor_handle editor) { return editor->Dt(); }

void nuka_editor_console_append(nuka_editor_handle editor, const char* text) {
    if (text != nullptr) editor->ConsoleAppend(text);
}

nuka_result_t nuka_editor_run_action(nuka_editor_handle editor, const char* id, const char* args_json,
                                     const char* source, const char** out_json) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (id == nullptr || out_json == nullptr) return NUKA_RESULT_INVALID_ARG;
    bool ok = false;
    const nuka_result_t result = Guard([&] {
        editor->reply_ = editor->RunAction(id, args_json != nullptr ? args_json : "", source != nullptr ? source : "api",
                                           &ok);
        *out_json = editor->reply_.c_str();
    });
    return result != NUKA_RESULT_OK ? result : ok ? NUKA_RESULT_OK : NUKA_RESULT_INVALID_ARG;
}

nuka_result_t nuka_editor_actions(nuka_editor_handle editor, const char** out_json) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (out_json == nullptr) return NUKA_RESULT_INVALID_ARG;
    return Guard([&] {
        editor->reply_ = editor->Actions();
        *out_json = editor->reply_.c_str();
    });
}

nuka_result_t nuka_editor_action_log(nuka_editor_handle editor, uint64_t since, const char** out_json) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (out_json == nullptr) return NUKA_RESULT_INVALID_ARG;
    return Guard([&] {
        editor->reply_ = editor->ActionLog(since);
        *out_json = editor->reply_.c_str();
    });
}

nuka_result_t nuka_editor_document_state(nuka_editor_handle editor, const char** out_json) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (out_json == nullptr) return NUKA_RESULT_INVALID_ARG;
    return Guard([&] {
        editor->reply_ = editor->DocumentState();
        *out_json = editor->reply_.c_str();
    });
}

const nuka_nui_api_t* nuka_editor_nui(nuka_editor_handle editor) { return editor->Nui(); }

nuka_result_t nuka_editor_set_python_hooks(nuka_editor_handle editor, const nuka_editor_python_hooks_t* hooks) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    return Guard([&] { editor->SetPythonHooks(hooks != nullptr ? *hooks : nuka_editor_python_hooks_t{}); });
}

const nuka_editor_host_api_t* nuka_editor_extension_host(nuka_editor_handle editor, const char* extension_id) {
    return editor->ExtensionHost(extension_id);
}

}  // extern "C"
