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
};

namespace {

thread_local std::string create_error;  // why this thread's last nuka_editor_create failed
thread_local std::string reply;         // the text this thread's last text-returning call handed out
thread_local std::string stream;        // the samples this thread's last nuka_editor_stream_take handed out

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
    create_error.clear();
    try {
        *out = new nuka_editor_t(std::vector<std::string>(argv, argv + argc), world, dt);
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        create_error = "out of memory";
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        create_error = e.what();
        return NUKA_RESULT_INTERNAL;
    }
}

const char* nuka_editor_create_error(void) { return create_error.c_str(); }

uint32_t nuka_editor_forward(int argc, const char* const* argv) {
    return nuka::runtime::app::viewer::EditorHost::Forward(std::vector<std::string>(argv, argv + argc)) ? 1u : 0u;
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
        reply = editor->RunAction(id, args_json != nullptr ? args_json : "", source != nullptr ? source : "api", &ok);
        *out_json = reply.c_str();
    });
    return result != NUKA_RESULT_OK ? result : ok ? NUKA_RESULT_OK : NUKA_RESULT_INVALID_ARG;
}

nuka_result_t nuka_editor_actions(nuka_editor_handle editor, const char** out_json) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (out_json == nullptr) return NUKA_RESULT_INVALID_ARG;
    return Guard([&] {
        reply = editor->Actions();
        *out_json = reply.c_str();
    });
}

nuka_result_t nuka_editor_action_log(nuka_editor_handle editor, uint64_t since, const char** out_json) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (out_json == nullptr) return NUKA_RESULT_INVALID_ARG;
    return Guard([&] {
        reply = editor->ActionLog(since);
        *out_json = reply.c_str();
    });
}

nuka_result_t nuka_editor_document_state(nuka_editor_handle editor, const char** out_json) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (out_json == nullptr) return NUKA_RESULT_INVALID_ARG;
    return Guard([&] {
        reply = editor->DocumentState();
        *out_json = reply.c_str();
    });
}

nuka_result_t nuka_editor_stream_take(nuka_editor_handle editor, const void** out_data, size_t* out_size) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (out_data == nullptr || out_size == nullptr) return NUKA_RESULT_INVALID_ARG;
    return Guard([&] {
        editor->StreamTake(&stream);
        *out_data = stream.data();
        *out_size = stream.size();
    });
}

const nuka_nui_api_t* nuka_editor_nui(nuka_editor_handle editor) { return editor->Nui(); }

nuka_result_t nuka_editor_set_python_hooks(nuka_editor_handle editor, const nuka_editor_python_hooks_t* hooks) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    return Guard([&] { editor->SetPythonHooks(hooks != nullptr ? *hooks : nuka_editor_python_hooks_t{}); });
}

nuka_result_t nuka_editor_set_rpc_hooks(nuka_editor_handle editor, const nuka_editor_rpc_hooks_t* hooks) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    return Guard([&] { editor->SetRpcHooks(hooks != nullptr ? *hooks : nuka_editor_rpc_hooks_t{}); });
}

void nuka_editor_rpc_clients(nuka_editor_handle editor, const char* clients_json) {
    if (clients_json != nullptr) editor->RpcClients(clients_json);
}

const nuka_editor_host_api_t* nuka_editor_extension_host(nuka_editor_handle editor, const char* extension_id) {
    return editor->ExtensionHost(extension_id);
}

}  // extern "C"
