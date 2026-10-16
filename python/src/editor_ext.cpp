// _editor_ext: the editor library (nuka/nuka_editor.h) behind nuka.editor. Frames run
// with the GIL released; script hooks take it back to call the Python runner.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "nuka/nuka_editor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

void Check(nuka_result_t result, const char* what) {
    if (result != NUKA_RESULT_OK)
        throw std::runtime_error(std::string(what) + " failed with nuka_result_t " +
                                 std::to_string(static_cast<int>(result)));
}

class App {
public:
    App(const std::vector<std::string>& args, std::uintptr_t world, float dt, nb::object runner)
        : runner_(std::move(runner)) {
        std::vector<const char*> argv;
        argv.reserve(args.size());
        for (const std::string& arg : args) argv.push_back(arg.c_str());
        nuka_result_t result = NUKA_RESULT_OK;
        {
            nb::gil_scoped_release release;
            result = nuka_editor_create(static_cast<int>(argv.size()), argv.data(),
                                        reinterpret_cast<nuka_world_handle>(world), dt, &editor_);
        }
        Check(result, "nuka_editor_create");
        const nuka_editor_script_hooks_t hooks{this, &Run, &Register, &Unregister, &Clear, &Dispatch};
        Check(nuka_editor_set_script_hooks(editor_, &hooks), "nuka_editor_set_script_hooks");
    }
    ~App() { destroy(); }
    App(const App&) = delete;
    App& operator=(const App&) = delete;

    void update(bool step) {
        nuka_editor_handle editor = Editor();
        nuka_result_t result = NUKA_RESULT_OK;
        {
            nb::gil_scoped_release release;
            result = nuka_editor_update(editor, step ? 1u : 0u);
        }
        Check(result, "nuka_editor_update");
    }
    void stop_physics() {
        nuka_editor_handle editor = Editor();
        nb::gil_scoped_release release;
        nuka_editor_stop_physics(editor);
    }
    bool running() const { return editor_ != nullptr && nuka_editor_running(editor_) != 0u; }
    bool playing() const { return nuka_editor_playing(Editor()) != 0u; }
    std::uintptr_t world_handle() const { return reinterpret_cast<std::uintptr_t>(nuka_editor_world(Editor())); }
    float dt() const { return nuka_editor_dt(Editor()); }
    void close() { nuka_editor_close(Editor()); }
    void console_append(const std::string& text) { nuka_editor_console_append(Editor(), text.c_str()); }

    // Edits park the physics thread, whose script callbacks need the GIL.
    bool edit(const std::string& txn_json) {
        nuka_editor_handle editor = Editor();
        nuka_result_t result = NUKA_RESULT_OK;
        {
            nb::gil_scoped_release release;
            result = nuka_editor_apply_edit(editor, txn_json.c_str());
        }
        if (result == NUKA_RESULT_INVALID_ARG) return false;
        Check(result, "nuka_editor_apply_edit");
        return true;
    }
    bool history(bool redo) {
        nuka_editor_handle editor = Editor();
        nuka_result_t result = NUKA_RESULT_OK;
        {
            nb::gil_scoped_release release;
            result = redo ? nuka_editor_redo(editor) : nuka_editor_undo(editor);
        }
        if (result == NUKA_RESULT_NOT_SUPPORTED) return false;
        Check(result, redo ? "nuka_editor_redo" : "nuka_editor_undo");
        return true;
    }
    std::string document() const {
        nuka_editor_handle editor = Editor();
        size_t length = 0u;
        Check(nuka_editor_document_state(editor, nullptr, 0u, &length), "nuka_editor_document_state");
        std::string text(length + 1u, '\0');
        Check(nuka_editor_document_state(editor, text.data(), text.size(), &length), "nuka_editor_document_state");
        text.resize(length);
        return text;
    }

    void destroy() {
        if (editor_ == nullptr) return;
        nuka_editor_handle editor = std::exchange(editor_, nullptr);
        nb::gil_scoped_release release;
        nuka_editor_destroy(editor);
    }

private:
    nuka_editor_handle Editor() const {
        if (editor_ == nullptr) throw std::runtime_error("the editor was destroyed");
        return editor_;
    }

    // Hook text stays valid until the thread's next hook call, as nuka_editor.h requires.
    template <class Fn>
    static const char* Call(void* self, Fn&& fn) {
        App& app = *static_cast<App*>(self);
        static thread_local std::string text;
        nb::gil_scoped_acquire acquire;
        try {
            nb::object result = fn(app.runner_);
            if (result.is_none()) return nullptr;
            text = nb::str(result).c_str();
            return text.c_str();
        } catch (nb::python_error& error) {
            error.discard_as_unraisable("nuka.editor script hook");
            return nullptr;
        }
    }
    static const char* Run(void* self, const char* code) {
        return Call(self, [&](nb::object& runner) { return runner.attr("run")(code); });
    }
    static const char* Register(void* self, uint64_t script_id, const char* source) {
        return Call(self, [&](nb::object& runner) { return runner.attr("register")(script_id, source); });
    }
    static void Unregister(void* self, uint64_t script_id) {
        Call(self, [&](nb::object& runner) { return runner.attr("unregister")(script_id); });
    }
    static void Clear(void* self) {
        Call(self, [](nb::object& runner) { return runner.attr("clear")(); });
    }
    static const char* Dispatch(void* self, nuka_editor_callback_t callback) {
        return Call(self, [&](nb::object& runner) { return runner.attr("dispatch")(static_cast<int>(callback)); });
    }

    nb::object runner_;
    nuka_editor_handle editor_ = nullptr;
};

}  // namespace

NB_MODULE(_editor_ext, m) {
    m.doc() = "The Nuka editor library; use nuka.editor.";

    nb::class_<App>(m, "App")
        .def(nb::init<const std::vector<std::string>&, std::uintptr_t, float, nb::object>(),
             nb::arg("args"), nb::arg("world"), nb::arg("dt"), nb::arg("runner"),
             "Open the editor over a world handle (0: it creates one at the first load). "
             "runner receives run/register/unregister/clear/dispatch calls for scripts.")
        .def("update", &App::update, nb::arg("step"),
             "One frame. Physics advances with the editor transport only when step is True.")
        .def("stop_physics", &App::stop_physics,
             "Join the physics thread; the world is the caller's until the next update(True).")
        .def_prop_ro("running", &App::running)
        .def_prop_ro("playing", &App::playing)
        .def_prop_ro("world_handle", &App::world_handle,
                     "The shown world's handle as an integer, 0 before the first load.")
        .def_prop_ro("dt", &App::dt)
        .def("close", &App::close, "Ask the editor to stop; running turns False.")
        .def("console_append", &App::console_append, nb::arg("text"))
        .def("edit", &App::edit, nb::arg("txn_json"),
             "Apply a JSON edit transaction through the document; False when an op failed.")
        .def("undo", [](App& app) { return app.history(false); })
        .def("redo", [](App& app) { return app.history(true); })
        .def("document", &App::document, "The document state as JSON text.")
        .def("destroy", &App::destroy, "Close the window and destroy any world the editor created.");
}
