// _editor_ext: the editor library (nuka/nuka_editor.h) behind nuka.editor. Frames run
// with the GIL released; script hooks and UI events take it back to call into Python.

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "nuka/nuka_editor.h"

#include <array>
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

using Point = std::array<float, 3>;

std::vector<const char*> Argv(const std::vector<std::string>& args) {
    std::vector<const char*> argv;
    argv.reserve(args.size());
    for (const std::string& arg : args) argv.push_back(arg.c_str());
    return argv;
}

// One overlay draw call's frame; it stops working when the call returns.
class OverlayFrame {
public:
    explicit OverlayFrame(const nuka_editor_overlay_frame_t* frame) : frame_(frame) {}
    std::vector<float> view() const { return std::vector<float>(Frame().view, Frame().view + 16); }
    std::vector<float> proj() const { return std::vector<float>(Frame().proj, Frame().proj + 16); }
    std::vector<float> viewport() const { return std::vector<float>(Frame().viewport, Frame().viewport + 4); }
    uint64_t tick() const { return Frame().tick; }
    void line(const Point& a, const Point& b, uint32_t rgba, float thickness) const {
        Frame().line(Frame().host, a.data(), b.data(), rgba, thickness);
    }
    void circle(const Point& center, float radius_px, uint32_t rgba, float thickness) const {
        Frame().circle(Frame().host, center.data(), radius_px, rgba, thickness);
    }
    void text(const Point& at, uint32_t rgba, const std::string& content) const {
        Frame().text(Frame().host, at.data(), rgba, content.c_str());
    }
    void Drop() { frame_ = nullptr; }

private:
    const nuka_editor_overlay_frame_t& Frame() const {
        if (frame_ == nullptr) throw std::runtime_error("an overlay frame draws only during its call");
        return *frame_;
    }
    const nuka_editor_overlay_frame_t* frame_;
};

// A Python callable the editor holds for an extension; released with the GIL taken.
struct Callback {
    nb::object fn;
    std::string reply;
    static void Release(void* self) {
        nb::gil_scoped_acquire acquire;
        delete static_cast<Callback*>(self);
    }
};

// The editor as one extension sees it (nuka_editor_host_api_t): what it builds or registers
// here goes away when the extension is disabled, and the table with it.
class ExtHost {
public:
    explicit ExtHost(const nuka_editor_host_api_t* api) : api_(api) {}

    std::string id() const { return Api().extension_id; }
    std::string dir() const { return Api().extension_dir; }
    std::string settings_json() const { return Api().settings_json; }

    uint32_t nui_panel(const std::string& panel_id, const std::string& title, const std::string& dock) {
        return Made(Nui().panel(Nui().host, panel_id.c_str(), title.c_str(), dock.c_str()));
    }
    uint32_t nui_add(uint32_t parent, const std::string& kind, const std::string& props_json) {
        return Made(Nui().add(Nui().host, parent, kind.c_str(), props_json.c_str()));
    }
    void nui_set(uint32_t node, const std::string& props_json) { Done(Nui().set(Nui().host, node, props_json.c_str())); }
    void nui_remove(uint32_t node) { Done(Nui().remove(Nui().host, node)); }
    void nui_clear(uint32_t node) { Done(Nui().clear(Nui().host, node)); }
    std::string nui_get(uint32_t node) {
        const char* text = Nui().get(Nui().host, node);
        if (text == nullptr) throw std::runtime_error(Nui().last_error(Nui().host));
        return text;
    }
    void nui_listen(uint32_t node) { Done(Nui().on_event(Nui().host, node, &UiEvent, this)); }
    void set_ui_handler(nb::object handler) { ui_handler_ = std::move(handler); }
    void toast(int level, const std::string& text) {
        Nui().toast(Nui().host, static_cast<nuka_nui_level_t>(level), text.c_str());
    }
    void progress(const std::string& entry_id, const std::string& label, float fraction) {
        Nui().progress(Nui().host, entry_id.c_str(), label.c_str(), fraction);
    }

    // fn(args_json) returns (ok, value_json or reason).
    void register_action(const std::string& action_id, const std::string& label, const std::string& category,
                         const std::string& shortcut, const std::string& icon, bool hidden, nb::object fn) {
        nuka_editor_action_desc_t desc{};
        desc.struct_size = static_cast<uint32_t>(sizeof(desc));
        desc.id = action_id.c_str();
        desc.label = label.c_str();
        desc.category = category.empty() ? nullptr : category.c_str();
        desc.shortcut = shortcut.empty() ? nullptr : shortcut.c_str();
        desc.icon = icon.empty() ? nullptr : icon.c_str();
        desc.hidden = hidden ? 1u : 0u;
        desc.run = &RunAction;
        desc.user = new Callback{std::move(fn), std::string()};
        desc.release = &Callback::Release;
        if (Api().register_action(Api().host, &desc) != NUKA_RESULT_OK)
            throw std::runtime_error("cannot register action '" + action_id + "'; see the extension log");
    }
    // fn(frame) returns whether it drew.
    void register_overlay(const std::string& overlay_id, nb::object fn) {
        nuka_editor_overlay_desc_t desc{};
        desc.struct_size = static_cast<uint32_t>(sizeof(desc));
        desc.id = overlay_id.c_str();
        desc.draw = &DrawOverlay;
        desc.user = new Callback{std::move(fn), std::string()};
        desc.release = &Callback::Release;
        Check(Api().register_overlay(Api().host, &desc), "register_overlay");
    }
    void log(int level, const std::string& text) {
        Api().log(Api().host, static_cast<nuka_nui_level_t>(level), text.c_str());
    }
    // The extension stops; the table goes with it.
    void close() {
        api_ = nullptr;
        ui_handler_ = nb::object();
    }

private:
    const nuka_editor_host_api_t& Api() const {
        if (api_ == nullptr) throw std::runtime_error("the extension is disabled");
        return *api_;
    }
    const nuka_nui_api_t& Nui() const { return *Api().nui; }
    uint32_t Made(uint32_t node) const {
        if (node == 0u) throw std::runtime_error(Nui().last_error(Nui().host));
        return node;
    }
    void Done(int32_t ok) const {
        if (ok == 0) throw std::runtime_error(Nui().last_error(Nui().host));
    }

    static void UiEvent(void* self, uint32_t node, const char* event_json) {
        ExtHost& host = *static_cast<ExtHost*>(self);
        nb::gil_scoped_acquire acquire;
        if (!host.ui_handler_.is_valid()) return;
        try {
            host.ui_handler_(node, event_json);
        } catch (nb::python_error& error) {
            error.discard_as_unraisable("nuka.editor extension ui handler");
        }
    }
    static int32_t RunAction(void* user, const char* args_json, const char** out_json) {
        Callback& callback = *static_cast<Callback*>(user);
        nb::gil_scoped_acquire acquire;
        bool ok = false;
        try {
            nb::tuple reply = nb::cast<nb::tuple>(callback.fn(args_json));
            ok = nb::cast<bool>(reply[0]);
            callback.reply = nb::cast<std::string>(reply[1]);
        } catch (nb::python_error& error) {
            callback.reply = error.what();
        }
        *out_json = callback.reply.c_str();
        return ok ? 1 : 0;
    }
    static int32_t DrawOverlay(void* user, const nuka_editor_overlay_frame_t* frame) {
        Callback& callback = *static_cast<Callback*>(user);
        nb::gil_scoped_acquire acquire;
        nb::object view = nb::cast(OverlayFrame(frame));
        bool ok = false;
        try {
            ok = nb::cast<bool>(callback.fn(view));
        } catch (nb::python_error& error) {
            error.discard_as_unraisable("nuka.editor overlay");
        }
        nb::cast<OverlayFrame&>(view).Drop();
        return ok ? 1 : 0;
    }

    const nuka_editor_host_api_t* api_;
    nb::object ui_handler_;
};

class App {
public:
    App(const std::vector<std::string>& args, std::uintptr_t world, float dt, nb::object runner, nb::object extensions)
        : runner_(std::move(runner)), extensions_(std::move(extensions)) {
        const std::vector<const char*> argv = Argv(args);
        nuka_result_t result = NUKA_RESULT_OK;
        {
            nb::gil_scoped_release release;
            result = nuka_editor_create(static_cast<int>(argv.size()), argv.data(),
                                        reinterpret_cast<nuka_world_handle>(world), dt, &editor_);
        }
        if (result != NUKA_RESULT_OK) throw std::runtime_error(nuka_editor_create_error());
        const nuka_editor_script_hooks_t hooks{this, &Run, &Register, &Unregister, &Clear, &Dispatch, &Interrupt, &Complete};
        Check(nuka_editor_set_script_hooks(editor_, &hooks), "nuka_editor_set_script_hooks");
        const nuka_editor_python_hooks_t python{this, &LoadExtension, &UnloadExtension};
        Check(nuka_editor_set_python_hooks(editor_, &python), "nuka_editor_set_python_hooks");
        api_ = nuka_editor_nui(editor_);
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

    // Actions may park the physics thread, whose script callbacks need the GIL. A failed
    // action still replies; only a broken call raises.
    std::string run_action(const std::string& id, const std::string& args_json, const std::string& source) {
        nuka_editor_handle editor = Editor();
        nuka_result_t result = NUKA_RESULT_OK;
        const char* reply = nullptr;
        {
            nb::gil_scoped_release release;
            result = nuka_editor_run_action(editor, id.c_str(), args_json.c_str(), source.c_str(), &reply);
        }
        if (result != NUKA_RESULT_INVALID_ARG) Check(result, "nuka_editor_run_action");
        return reply;
    }
    // Queries run on the editor's UI thread like actions, so they wait without the GIL too.
    std::string actions() const {
        nuka_editor_handle editor = Editor();
        const char* text = nullptr;
        nuka_result_t result = NUKA_RESULT_OK;
        {
            nb::gil_scoped_release release;
            result = nuka_editor_actions(editor, &text);
        }
        Check(result, "nuka_editor_actions");
        return text;
    }
    std::string action_log(uint64_t since) const {
        nuka_editor_handle editor = Editor();
        const char* text = nullptr;
        nuka_result_t result = NUKA_RESULT_OK;
        {
            nb::gil_scoped_release release;
            result = nuka_editor_action_log(editor, since, &text);
        }
        Check(result, "nuka_editor_action_log");
        return text;
    }
    std::string document() const {
        nuka_editor_handle editor = Editor();
        const char* text = nullptr;
        nuka_result_t result = NUKA_RESULT_OK;
        {
            nb::gil_scoped_release release;
            result = nuka_editor_document_state(editor, &text);
        }
        Check(result, "nuka_editor_document_state");
        return text;
    }

    // The nui table: node handles in, the editor's reason raised on failure.
    uint32_t nui_panel(const std::string& id, const std::string& title, const std::string& dock) {
        return Made(Nui().panel(api_->host, id.c_str(), title.c_str(), dock.c_str()));
    }
    uint32_t nui_add(uint32_t parent, const std::string& kind, const std::string& props_json) {
        return Made(Nui().add(api_->host, parent, kind.c_str(), props_json.c_str()));
    }
    void nui_set(uint32_t node, const std::string& props_json) {
        Done(Nui().set(api_->host, node, props_json.c_str()));
    }
    void nui_remove(uint32_t node) { Done(Nui().remove(api_->host, node)); }
    void nui_clear(uint32_t node) { Done(Nui().clear(api_->host, node)); }
    std::string nui_get(uint32_t node) {
        const char* text = Nui().get(api_->host, node);
        if (text == nullptr) throw std::runtime_error(api_->last_error(api_->host));
        return text;
    }
    void nui_listen(uint32_t node) { Done(Nui().on_event(api_->host, node, &UiEvent, this)); }
    void set_ui_handler(nb::object handler) { ui_handler_ = std::move(handler); }
    void toast(int level, const std::string& text) {
        Nui().toast(api_->host, static_cast<nuka_nui_level_t>(level), text.c_str());
    }
    void progress(const std::string& id, const std::string& label, float fraction) {
        Nui().progress(api_->host, id.c_str(), label.c_str(), fraction);
    }
    // An enabled extension's scoped host; the loader asks for it while the extension loads.
    ExtHost extension_host(const std::string& id) const {
        const nuka_editor_host_api_t* host = nuka_editor_extension_host(Editor(), id.c_str());
        if (host == nullptr) throw std::runtime_error("no enabled extension '" + id + "'");
        return ExtHost(host);
    }

    void destroy() {
        if (editor_ == nullptr) return;
        nuka_editor_handle editor = std::exchange(editor_, nullptr);
        api_ = nullptr;
        ui_handler_ = nb::object();
        nb::gil_scoped_release release;
        nuka_editor_destroy(editor);
    }

private:
    nuka_editor_handle Editor() const {
        if (editor_ == nullptr) throw std::runtime_error("the editor was destroyed");
        return editor_;
    }
    const nuka_nui_api_t& Nui() const {
        Editor();
        return *api_;
    }
    uint32_t Made(uint32_t node) const {
        if (node == 0u) throw std::runtime_error(api_->last_error(api_->host));
        return node;
    }
    void Done(int32_t ok) const {
        if (ok == 0) throw std::runtime_error(api_->last_error(api_->host));
    }

    // Events arrive at the top of a frame, on the thread running update().
    static void UiEvent(void* self, uint32_t node, const char* event_json) {
        App& app = *static_cast<App*>(self);
        nb::gil_scoped_acquire acquire;
        try {
            app.ui_handler_(node, event_json);
        } catch (nb::python_error& error) {
            error.discard_as_unraisable("nuka.editor ui handler");
        }
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
    // The runner answers (failed, text).
    static const char* Run(void* self, const char* code, const char* filename, uint32_t echo, int32_t* failed) {
        return Call(self, [&](nb::object& runner) {
            nb::tuple reply = nb::cast<nb::tuple>(runner.attr("run")(code, filename, echo != 0u));
            *failed = nb::cast<bool>(reply[0]) ? 1 : 0;
            return nb::object(reply[1]);
        });
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
    static void Interrupt(void* self) {
        Call(self, [](nb::object& runner) { return runner.attr("interrupt")(); });
    }
    static const char* Complete(void* self, const char* code, int32_t line, int32_t column) {
        return Call(self, [&](nb::object& runner) { return runner.attr("complete")(code, line, column); });
    }

    // The loader replies (ok, text): the log on success, the traceback on failure.
    static const char* LoadExtension(void* self, const char* id, const char* module, const char* dir,
                                     int32_t* failed) {
        App& app = *static_cast<App*>(self);
        static thread_local std::string text;
        nb::gil_scoped_acquire acquire;
        *failed = 1;
        try {
            nb::tuple reply = nb::cast<nb::tuple>(app.extensions_.attr("load")(id, module, dir));
            *failed = nb::cast<bool>(reply[0]) ? 0 : 1;
            text = nb::cast<std::string>(reply[1]);
        } catch (nb::python_error& error) {
            text = error.what();
        }
        return text.empty() ? nullptr : text.c_str();
    }
    static const char* UnloadExtension(void* self, const char* id) {
        App& app = *static_cast<App*>(self);
        static thread_local std::string text;
        nb::gil_scoped_acquire acquire;
        try {
            nb::object reply = app.extensions_.attr("unload")(id);
            if (reply.is_none()) return nullptr;
            text = nb::cast<std::string>(reply);
            return text.c_str();
        } catch (nb::python_error& error) {
            error.discard_as_unraisable("nuka.editor extension unload");
            return nullptr;
        }
    }

    nb::object runner_;
    nb::object extensions_;
    nb::object ui_handler_;
    nuka_editor_handle editor_ = nullptr;
    const nuka_nui_api_t* api_ = nullptr;
};

}  // namespace

NB_MODULE(_editor_ext, m) {
    m.doc() = "The Nuka editor library; use nuka.editor.";

    m.def(
        "forward",
        [](const std::vector<std::string>& args) {
            const std::vector<const char*> argv = Argv(args);
            return nuka_editor_forward(static_cast<int>(argv.size()), argv.data()) != 0u;
        },
        nb::arg("args"),
        "Hand the scene in args to the editor already running on the same profile; False when none runs.");

    nb::class_<OverlayFrame>(m, "OverlayFrame", "One overlay draw over the scene viewport; rgba is 0xRRGGBBAA.")
        .def_prop_ro("view", &OverlayFrame::view, "The column-major view matrix.")
        .def_prop_ro("proj", &OverlayFrame::proj, "The column-major GL-style projection.")
        .def_prop_ro("viewport", &OverlayFrame::viewport, "x, y, width, height in window pixels.")
        .def_prop_ro("tick", &OverlayFrame::tick)
        .def("line", &OverlayFrame::line, nb::arg("a"), nb::arg("b"), nb::arg("rgba"), nb::arg("thickness") = 1.5f)
        .def("circle", &OverlayFrame::circle, nb::arg("center"), nb::arg("radius_px"), nb::arg("rgba"),
             nb::arg("thickness") = 1.5f)
        .def("text", &OverlayFrame::text, nb::arg("at"), nb::arg("rgba"), nb::arg("text"));

    nb::class_<ExtHost>(m, "ExtHost", "One extension's host table; use nuka.editor.extension.Context.")
        .def_prop_ro("id", &ExtHost::id)
        .def_prop_ro("dir", &ExtHost::dir)
        .def_prop_ro("settings_json", &ExtHost::settings_json)
        .def("nui_panel", &ExtHost::nui_panel, nb::arg("id"), nb::arg("title"), nb::arg("dock"))
        .def("nui_add", &ExtHost::nui_add, nb::arg("parent"), nb::arg("kind"), nb::arg("props_json"))
        .def("nui_set", &ExtHost::nui_set, nb::arg("node"), nb::arg("props_json"))
        .def("nui_remove", &ExtHost::nui_remove, nb::arg("node"))
        .def("nui_clear", &ExtHost::nui_clear, nb::arg("node"))
        .def("nui_get", &ExtHost::nui_get, nb::arg("node"))
        .def("nui_listen", &ExtHost::nui_listen, nb::arg("node"))
        .def("set_ui_handler", &ExtHost::set_ui_handler, nb::arg("handler"))
        .def("toast", &ExtHost::toast, nb::arg("level"), nb::arg("text"))
        .def("progress", &ExtHost::progress, nb::arg("id"), nb::arg("label"), nb::arg("fraction"))
        .def("register_action", &ExtHost::register_action, nb::arg("id"), nb::arg("label"), nb::arg("category"),
             nb::arg("shortcut"), nb::arg("icon"), nb::arg("hidden"), nb::arg("fn"))
        .def("register_overlay", &ExtHost::register_overlay, nb::arg("id"), nb::arg("fn"))
        .def("log", &ExtHost::log, nb::arg("level"), nb::arg("text"))
        .def("close", &ExtHost::close);

    nb::class_<App>(m, "App")
        .def(nb::init<const std::vector<std::string>&, std::uintptr_t, float, nb::object, nb::object>(),
             nb::arg("args"), nb::arg("world"), nb::arg("dt"), nb::arg("runner"), nb::arg("extensions"),
             "Open the editor over a world handle (0: it creates one at the first load). "
             "runner receives run/register/unregister/clear/dispatch calls for scripts, extensions "
             "load/unload calls for Python extensions.")
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
        .def("run_action", &App::run_action, nb::arg("id"), nb::arg("args_json"), nb::arg("source"),
             "Run one editor action; the JSON reply holds ok, value and error.")
        .def("actions", &App::actions, "The registered actions as JSON text.")
        .def("action_log", &App::action_log, nb::arg("since"), "Action runs after sequence since, as JSON text.")
        .def("document", &App::document, "The document state as JSON text.")
        .def("nui_panel", &App::nui_panel, nb::arg("id"), nb::arg("title"), nb::arg("dock"),
             "A dockable panel's root node, reusing the panel when the id exists.")
        .def("nui_add", &App::nui_add, nb::arg("parent"), nb::arg("kind"), nb::arg("props_json"))
        .def("nui_set", &App::nui_set, nb::arg("node"), nb::arg("props_json"))
        .def("nui_remove", &App::nui_remove, nb::arg("node"))
        .def("nui_clear", &App::nui_clear, nb::arg("node"))
        .def("nui_get", &App::nui_get, nb::arg("node"), "The node's value as JSON text.")
        .def("nui_listen", &App::nui_listen, nb::arg("node"),
             "Route the node's events to the ui handler as (node, event_json).")
        .def("set_ui_handler", &App::set_ui_handler, nb::arg("handler"))
        .def("toast", &App::toast, nb::arg("level"), nb::arg("text"))
        .def("progress", &App::progress, nb::arg("id"), nb::arg("label"), nb::arg("fraction"))
        .def("extension_host", &App::extension_host, nb::arg("id"))
        .def("destroy", &App::destroy, "Close the window and destroy any world the editor created.");
}
