#include "nuka/nuka_editor.h"

#include "runtime/app/viewer/editor_host.hpp"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

struct nuka_editor_t : nuka::runtime::app::viewer::EditorHost {
    using EditorHost::EditorHost;
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

nuka_result_t nuka_editor_apply_edit(nuka_editor_handle editor, const char* txn_json) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (txn_json == nullptr) return NUKA_RESULT_INVALID_ARG;
    bool ok = false;
    const nuka_result_t result = Guard([&] { ok = editor->ApplyEdit(txn_json); });
    return result != NUKA_RESULT_OK ? result : ok ? NUKA_RESULT_OK : NUKA_RESULT_INVALID_ARG;
}

nuka_result_t nuka_editor_undo(nuka_editor_handle editor) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    bool ok = false;
    const nuka_result_t result = Guard([&] { ok = editor->Undo(); });
    return result != NUKA_RESULT_OK ? result : ok ? NUKA_RESULT_OK : NUKA_RESULT_NOT_SUPPORTED;
}

nuka_result_t nuka_editor_redo(nuka_editor_handle editor) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    bool ok = false;
    const nuka_result_t result = Guard([&] { ok = editor->Redo(); });
    return result != NUKA_RESULT_OK ? result : ok ? NUKA_RESULT_OK : NUKA_RESULT_NOT_SUPPORTED;
}

nuka_result_t nuka_editor_document_state(nuka_editor_handle editor, char* out, size_t capacity,
                                         size_t* out_length) {
    if (editor == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (out_length == nullptr) return NUKA_RESULT_INVALID_ARG;
    std::string state;
    const nuka_result_t result = Guard([&] { state = editor->DocumentState(); });
    if (result != NUKA_RESULT_OK) return result;
    *out_length = state.size();
    if (out == nullptr) return NUKA_RESULT_OK;
    if (capacity <= state.size()) return NUKA_RESULT_INVALID_ARG;
    std::memcpy(out, state.c_str(), state.size() + 1u);
    return NUKA_RESULT_OK;
}

}  // extern "C"
