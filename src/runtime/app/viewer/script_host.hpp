#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer::ScriptHost -- the editor's embedded CPython.
//
// Owns one in-process interpreter plus a built-in `nuka` module (registered via
// PyImport_AppendInittab before init) bound to the editor's LIVE objects through
// a ScriptContext of raw pointers. Scripts therefore drive the ON-SCREEN world --
// every mutation routes through the SAME CommandQueue / editor-edit seams the
// panels use, never a second engine. Runs on the interactive frame loop, OUTSIDE
// the offscreen determinism path.
//
// HOST-ONLY / zero-CUDA-token: pure host C++ + libpython. Built ONLY into the
// gated nuka_editor exe (NK_BUILD_VULKAN_VALIDATION).
// ---------------------------------------------------------------------------

#include <string>

namespace nuka::runtime::app::viewer {

struct EditorScene;
struct EntityRecordIndex;
struct ViewerUiState;

// The live objects the built-in `nuka` module mutates. The viewer points it at the
// active world on every load / unload; `scene` null means the editor is empty
// (scene-gated calls raise a Python error rather than spinning a second engine).
struct ScriptContext {
    EditorScene*       scene = nullptr;
    EntityRecordIndex* index = nullptr;
    ViewerUiState*     ui    = nullptr;
};

class ScriptHost {
public:
    // Registers the built-in module, then initializes the interpreter against the
    // CMake-pinned Python home. On any failure the host stays !Valid() and the
    // editor runs without scripting (never aborts).
    ScriptHost();
    ~ScriptHost();
    ScriptHost(const ScriptHost&) = delete;
    ScriptHost& operator=(const ScriptHost&) = delete;

    bool Valid() const { return valid_; }

    // Re-point the built-in module at the active world (or an empty editor: nulls).
    void SetContext(const ScriptContext& ctx);

    // Execute `code` in the embedded interpreter and return the captured
    // stdout+stderr (including any traceback). NEVER throws and never crashes the
    // editor: a script exception is isolated and reported as text; a runaway loop
    // is interrupted by a watchdog after a wall-clock budget.
    std::string RunString(const std::string& code);

    // -- lifecycle scripts (/script nodes) ----------------------------------
    // Which lifecycle hook a dispatch invokes / a query reports on.
    enum class Callback { Ready, Step, Reset };

    // Register (or REPLACE, keyed by stable_id) a lifecycle script: exec `source`
    // once in its OWN namespace, defining optional on_ready/on_step/on_reset. The
    // body runs through the same isolation + budget as RunString; the captured
    // output (incl. any traceback) is returned. Does NOT itself fire on_ready.
    std::string RegisterScript(uint64_t stable_id, const std::string& source);

    // Forget every registered lifecycle script (called on unload / before reload).
    void ClearScripts();

    // Forget ONE registered lifecycle script (a deleted /script node); no-op for
    // an unknown id.
    void UnregisterScript(uint64_t stable_id);

    // Registered lifecycle-script count (editor / test introspection).
    size_t ScriptCount() const;

    // Invoke a lifecycle hook on every registered script, each isolated + held to a
    // tight per-call budget. on_ready fires AT MOST ONCE per script. A throwing
    // callback is caught (its first traceback returned) and DISABLED after a bounded
    // number of consecutive failures, so it never spins or spams the editor. Returns
    // any captured output (callbacks are usually silent).
    std::string DispatchReady();
    std::string DispatchStep();
    std::string DispatchReset();

    // True iff `cb` has been disabled (after the bounded retry) for the script with
    // `stable_id`. False for an unknown id or a still-live callback.
    bool CallbackDisabled(uint64_t stable_id, Callback cb) const;

private:
    bool valid_ = false;
};

}  // namespace nuka::runtime::app::viewer
