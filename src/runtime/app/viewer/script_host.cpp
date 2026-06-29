// ---------------------------------------------------------------------------
// The editor's embedded CPython + built-in `nuka` module (see script_host.hpp).
//
// The module wraps the editor's live EditorScene / EntityRecordIndex / ViewerUiState
// and routes every mutation through the existing general seams (ApplyTransformEdit,
// ApplyMaterialEdit, the per-DOF DriveTarget upload). It is statically linked into
// the exe and registered via PyImport_AppendInittab, so it is a built-in module on
// the editor's own symbols -- no second libnuka, no -rdynamic (Windows-ready).
//
// HOST-ONLY / zero-CUDA-token.
// ---------------------------------------------------------------------------

#include <Python.h>  // precedes standard headers (CPython requirement)

#include "runtime/app/viewer/script_host.hpp"

#include "math/transform.hpp"
#include "render/render_world.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/editor_edits.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_ir.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace nuka::runtime::app::viewer {
namespace {

namespace scene = nuka::scene;
namespace render = nuka::render;

// The live world the built-in module mutates; updated by ScriptHost::SetContext.
ScriptContext g_ctx;

// Wall-clock budget one Run gets before the hang guard interrupts it (env override
// NUKA_SCRIPT_BUDGET_S). A pure-Python runaway unwinds; a blocked C call does not.
constexpr double kDefaultBudgetSeconds = 5.0;
// Console text returned by one Run is capped to this many trailing bytes.
constexpr size_t kRunOutputCap = 256u * 1024u;

// Steady-clock deadline (epoch count) the per-line trace hook enforces; 0 = inactive.
// A pure-Python runaway unwinds at the budget; a blocked C call is not traced.
std::atomic<long long> g_deadline{0};

unsigned long long PackEntity(const scene::EntityId& e) {
    return (static_cast<unsigned long long>(e.index) << 32) |
           static_cast<unsigned long long>(e.gen);
}
scene::EntityId UnpackEntity(unsigned long long v) {
    return scene::EntityId{static_cast<uint32_t>(v >> 32),
                           static_cast<uint32_t>(v & 0xffffffffull)};
}

EditorScene* RequireScene() {
    if (g_ctx.scene == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "no scene loaded");
        return nullptr;
    }
    return g_ctx.scene;
}

// Resolve a node PATH to its entity (first pre-order match), or kInvalidEntity.
scene::EntityId EntityOfPath(const EditorScene& es, const std::string& path) {
    scene::EntityId found = scene::kInvalidEntity;
    const scene::SceneGraph& tree = es.scene.Tree();
    tree.Traverse(tree.Root(), [&](const std::shared_ptr<scene::SceneNode>& n) {
        if (found == scene::kInvalidEntity && tree.PathOf(n) == path) found = n->entity;
    });
    return found;
}

// A Python str (path) or int (packed id) -> entity. Sets a Python error + returns
// false on a bad type or an unknown path.
bool ResolveTarget(PyObject* arg, scene::EntityId* out) {
    if (PyLong_Check(arg)) {
        const unsigned long long v = PyLong_AsUnsignedLongLong(arg);
        if (v == static_cast<unsigned long long>(-1) && PyErr_Occurred()) return false;
        *out = UnpackEntity(v);
        return true;
    }
    if (PyUnicode_Check(arg)) {
        const char* p = PyUnicode_AsUTF8(arg);
        if (p == nullptr) return false;
        EditorScene* es = RequireScene();
        if (es == nullptr) return false;
        const scene::EntityId e = EntityOfPath(*es, p);
        if (e == scene::kInvalidEntity) {
            PyErr_Format(PyExc_RuntimeError, "no node at path '%s'", p);
            return false;
        }
        *out = e;
        return true;
    }
    PyErr_SetString(PyExc_TypeError, "expected a node path (str) or id (int)");
    return false;
}

// Write a line through the live sys.stdout (the redirected sink during a Run), so it
// interleaves with print() output; falls back to the process stdout otherwise.
void WriteStdout(const std::string& line) {
    PyObject* out = PySys_GetObject("stdout");  // borrowed; may be null
    if (out != nullptr) {
        PyObject* r = PyObject_CallMethod(out, "write", "s", line.c_str());
        Py_XDECREF(r);
    } else {
        std::fputs(line.c_str(), stdout);
    }
}

PyObject* Nk_log(PyObject*, PyObject* args) {
    PyObject* o = nullptr;
    if (!PyArg_ParseTuple(args, "O", &o)) return nullptr;
    PyObject* s = PyObject_Str(o);
    if (s == nullptr) return nullptr;
    const char* c = PyUnicode_AsUTF8(s);
    if (c != nullptr) WriteStdout(std::string(c) + "\n");
    Py_DECREF(s);
    if (c == nullptr) return nullptr;
    Py_RETURN_NONE;
}

PyObject* Nk_entity_count(PyObject*, PyObject*) {
    EditorScene* es = g_ctx.scene;
    if (es == nullptr) return PyLong_FromLong(0);
    long count = 0;
    const scene::SceneGraph& tree = es->scene.Tree();
    tree.Traverse(tree.Root(), [&](const std::shared_ptr<scene::SceneNode>& n) {
        if (n->entity != scene::kInvalidEntity) ++count;
    });
    return PyLong_FromLong(count);
}

PyObject* Nk_find(PyObject*, PyObject* args) {
    const char* path = nullptr;
    if (!PyArg_ParseTuple(args, "s", &path)) return nullptr;
    EditorScene* es = RequireScene();
    if (es == nullptr) return nullptr;
    const scene::EntityId e = EntityOfPath(*es, path);
    if (e == scene::kInvalidEntity) Py_RETURN_NONE;
    return PyLong_FromUnsignedLongLong(PackEntity(e));
}

PyObject* Nk_selected(PyObject*, PyObject*) {
    if (g_ctx.ui == nullptr || g_ctx.ui->selected_entity == scene::kInvalidEntity)
        Py_RETURN_NONE;
    return PyLong_FromUnsignedLongLong(PackEntity(g_ctx.ui->selected_entity));
}

PyObject* Nk_position(PyObject*, PyObject* args) {
    PyObject* target = nullptr;
    if (!PyArg_ParseTuple(args, "O", &target)) return nullptr;
    EditorScene* es = RequireScene();
    if (es == nullptr) return nullptr;
    scene::EntityId e;
    if (!ResolveTarget(target, &e)) return nullptr;
    const render::RenderInstance* inst =
        InstanceOfEntity(es->sim->GetRenderWorld(), e);
    if (inst == nullptr) Py_RETURN_NONE;
    const math::Vec3 p = inst->world_xform.position;
    return Py_BuildValue("(ddd)", static_cast<double>(p.x), static_cast<double>(p.y),
                         static_cast<double>(p.z));
}

PyObject* Nk_move(PyObject*, PyObject* args) {
    PyObject* target = nullptr;
    double x = 0.0, y = 0.0, z = 0.0, qw = 1.0, qx = 0.0, qy = 0.0, qz = 0.0;
    if (!PyArg_ParseTuple(args, "Oddd|dddd", &target, &x, &y, &z, &qw, &qx, &qy, &qz))
        return nullptr;
    const Py_ssize_t argc = PyTuple_Size(args);
    if (argc != 4 && argc != 8) {
        PyErr_SetString(PyExc_TypeError,
                        "move expects (target,x,y,z) or (target,x,y,z,qw,qx,qy,qz)");
        return nullptr;
    }
    EditorScene* es = RequireScene();
    if (es == nullptr) return nullptr;
    if (g_ctx.index == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "no edit index");
        return nullptr;
    }
    scene::EntityId e;
    if (!ResolveTarget(target, &e)) return nullptr;

    math::Transform world = math::Transform::Identity();
    world.position = math::Vec3{static_cast<float>(x), static_cast<float>(y),
                                static_cast<float>(z)};
    if (argc == 8) {
        math::Quat q;
        q.w = static_cast<float>(qw); q.x = static_cast<float>(qx);
        q.y = static_cast<float>(qy); q.z = static_cast<float>(qz);
        world.rotation = q.Normalized();
    } else if (const render::RenderInstance* inst =
                   InstanceOfEntity(es->sim->GetRenderWorld(), e)) {
        world.rotation = inst->world_xform.rotation;  // keep current orientation
    }
    ApplyTransformEdit(*es, *g_ctx.index, e, world, /*commit=*/true);
    if (g_ctx.ui != nullptr) g_ctx.ui->save_dirty = true;
    Py_RETURN_NONE;
}

PyObject* Nk_set_drive(PyObject*, PyObject* args) {
    int dof = 0;
    double value = 0.0;
    if (!PyArg_ParseTuple(args, "id", &dof, &value)) return nullptr;
    if (g_ctx.ui == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "no editor state");
        return nullptr;
    }
    ViewerUiState& ui = *g_ctx.ui;
    if (dof < 0 || static_cast<size_t>(dof) >= ui.drive_targets.size()) {
        PyErr_Format(PyExc_IndexError, "drive dof %d out of range [0,%zu)", dof,
                     ui.drive_targets.size());
        return nullptr;
    }
    if (ui.drive_dirty.size() != ui.drive_targets.size())
        ui.drive_dirty.assign(ui.drive_targets.size(), 0u);
    ui.drive_targets[static_cast<size_t>(dof)] = static_cast<float>(value);
    ui.drive_dirty[static_cast<size_t>(dof)] = 1u;  // viewer uploads only this DOF
    Py_RETURN_NONE;
}

PyObject* Nk_material(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* kw[] = {"target", "base", "roughness", "metallic", nullptr};
    PyObject* target = nullptr;
    PyObject* base = nullptr;
    PyObject* roughness = nullptr;
    PyObject* metallic = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOO", const_cast<char**>(kw),
                                     &target, &base, &roughness, &metallic))
        return nullptr;
    EditorScene* es = RequireScene();
    if (es == nullptr) return nullptr;
    if (g_ctx.index == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "no edit index");
        return nullptr;
    }
    scene::EntityId e;
    if (!ResolveTarget(target, &e)) return nullptr;

    const MaterialBinding mb = ResolveMaterial(*es, *g_ctx.index, e);
    if (!mb.valid) {
        PyErr_SetString(PyExc_RuntimeError, "node has no editable material");
        return nullptr;
    }
    scene::RenderMaterial m = es->sim->GetRenderWorld().materials[mb.render_slot];
    if (base != nullptr && base != Py_None) {
        PyObject* seq = PySequence_Fast(base, "base must be a sequence of 3 or 4 numbers");
        if (seq == nullptr) return nullptr;
        const Py_ssize_t bn = PySequence_Fast_GET_SIZE(seq);
        if (bn != 3 && bn != 4) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "base needs 3 (rgb) or 4 (rgba) numbers");
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < bn; ++i) {
            const double c = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
            m.base_color[i] = static_cast<float>(c);
        }
        if (bn == 4) m.opacity = m.base_color[3];
        Py_DECREF(seq);
        if (PyErr_Occurred()) return nullptr;
    }
    if (roughness != nullptr && roughness != Py_None) {
        m.roughness = static_cast<float>(PyFloat_AsDouble(roughness));
        if (PyErr_Occurred()) return nullptr;
    }
    if (metallic != nullptr && metallic != Py_None) {
        m.metallic = static_cast<float>(PyFloat_AsDouble(metallic));
        if (PyErr_Occurred()) return nullptr;
    }
    ApplyMaterialEdit(*es, mb, m, /*commit=*/true);
    if (g_ctx.ui != nullptr) g_ctx.ui->save_dirty = true;
    Py_RETURN_NONE;
}

// METH_KEYWORDS stores a 3-arg function through PyCFunction; the function-pointer
// cast is the documented idiom (the warning is local to this table).
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
PyMethodDef kNukaMethods[] = {
    {"log", Nk_log, METH_VARARGS, "log(msg): write a line to the editor console"},
    {"entity_count", Nk_entity_count, METH_NOARGS, "entity_count(): addressable scene nodes"},
    {"find", Nk_find, METH_VARARGS, "find(path) -> id or None"},
    {"selected", Nk_selected, METH_NOARGS, "selected() -> id or None"},
    {"position", Nk_position, METH_VARARGS, "position(target) -> (x,y,z) or None"},
    {"move", Nk_move, METH_VARARGS,
     "move(target, x, y, z[, qw, qx, qy, qz]): reposition through the edit seam"},
    {"set_drive", Nk_set_drive, METH_VARARGS, "set_drive(dof, value): per-DOF drive target"},
    {"material", reinterpret_cast<PyCFunction>(Nk_material), METH_VARARGS | METH_KEYWORDS,
     "material(target, base=[r,g,b,a], roughness=, metallic=)"},
    {nullptr, nullptr, 0, nullptr},
};
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

PyModuleDef kNukaModule = {
    PyModuleDef_HEAD_INIT, "nuka", "Nuka editor live-world bridge", -1,
    kNukaMethods, nullptr, nullptr, nullptr, nullptr};

PyObject* PyInit_nuka() { return PyModule_Create(&kNukaModule); }

// Per-line trace hook: raise once the Run's budget is spent so a runaway loop ends.
// Installed only for the duration of a Run; fires on every Python line/call event.
int TraceHangGuard(PyObject*, PyFrameObject*, int, PyObject*) {
    const long long dl = g_deadline.load();
    if (dl != 0 &&
        std::chrono::steady_clock::now().time_since_epoch().count() >= dl) {
        PyErr_SetString(PyExc_RuntimeError, "script exceeded its time budget");
        return -1;
    }
    return 0;
}

double BudgetSeconds() {
    if (const char* s = std::getenv("NUKA_SCRIPT_BUDGET_S")) {
        const double v = std::atof(s);
        if (v > 0.0) return v;
    }
    return kDefaultBudgetSeconds;
}

}  // namespace

ScriptHost::ScriptHost() {
    if (PyImport_AppendInittab("nuka", &PyInit_nuka) != 0) {
        std::fprintf(stderr, "[nuka_editor] PyImport_AppendInittab(nuka) failed\n");
        return;
    }
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.install_signal_handlers = 0;  // do not steal the host's SIGINT
#ifdef NUKA_PY_HOME
    {
        const char* home = NUKA_PY_HOME;
        if (home != nullptr && home[0] != '\0') {
            wchar_t* whome = Py_DecodeLocale(home, nullptr);
            if (whome != nullptr) {
                const PyStatus hs = PyConfig_SetString(&config, &config.home, whome);
                PyMem_RawFree(whome);
                if (PyStatus_Exception(hs)) {
                    std::fprintf(stderr, "[nuka_editor] python home set failed\n");
                    PyConfig_Clear(&config);
                    return;
                }
            }
        }
    }
#endif
    const PyStatus status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        std::fprintf(stderr, "[nuka_editor] Py_InitializeFromConfig failed\n");
        return;
    }
    valid_ = true;
    // Cache OUR built-in `nuka` in sys.modules so it wins over any on-disk `nuka`
    // package a PEP 660 editable finder would otherwise shadow it with (second engine).
    const char* kForceBuiltin =
        "import sys\n"
        "from importlib.machinery import BuiltinImporter\n"
        "import importlib.util as _u\n"
        "_spec = BuiltinImporter.find_spec('nuka')\n"
        "if _spec is not None:\n"
        "    _m = _u.module_from_spec(_spec)\n"
        "    _spec.loader.exec_module(_m)\n"
        "    sys.modules['nuka'] = _m\n";
    if (PyRun_SimpleString(kForceBuiltin) != 0)
        std::fprintf(stderr, "[nuka_editor] could not pin the built-in nuka module\n");
    // Inject `nuka` into __main__ so scripts call it without an explicit import.
    PyObject* main_mod = PyImport_AddModule("__main__");  // borrowed
    if (main_mod != nullptr) {
        PyObject* g = PyModule_GetDict(main_mod);  // borrowed
        PyObject* mod = PyImport_ImportModule("nuka");
        if (mod != nullptr) {
            PyDict_SetItemString(g, "nuka", mod);
            Py_DECREF(mod);
        }
    }
    std::printf("[nuka_editor] embedded python %s (built-in module `nuka`)\n",
                Py_GetVersion());
    std::fflush(stdout);
}

ScriptHost::~ScriptHost() {
    if (valid_) Py_FinalizeEx();
}

void ScriptHost::SetContext(const ScriptContext& ctx) { g_ctx = ctx; }

std::string ScriptHost::RunString(const std::string& code) {
    if (!valid_) return "[scripting] interpreter unavailable\n";

    // Redirect sys.stdout/stderr to a fresh StringIO so print() + any traceback are
    // captured into the returned console text.
    PyObject* io_mod = PyImport_ImportModule("io");
    PyObject* sink = (io_mod != nullptr) ? PyObject_CallMethod(io_mod, "StringIO", nullptr)
                                         : nullptr;
    PyObject* sys_mod = PyImport_ImportModule("sys");
    PyObject* old_out = (sys_mod != nullptr) ? PyObject_GetAttrString(sys_mod, "stdout") : nullptr;
    PyObject* old_err = (sys_mod != nullptr) ? PyObject_GetAttrString(sys_mod, "stderr") : nullptr;
    if (sys_mod != nullptr && sink != nullptr) {
        PyObject_SetAttrString(sys_mod, "stdout", sink);
        PyObject_SetAttrString(sys_mod, "stderr", sink);
    }

    PyObject* main_mod = PyImport_AddModule("__main__");  // borrowed
    PyObject* globals = (main_mod != nullptr) ? PyModule_GetDict(main_mod) : nullptr;

    // Hang guard: a per-line trace hook raises once the budget is spent, so a runaway
    // pure-Python loop unwinds (a thread blocked in a single C call is not traced).
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(BudgetSeconds()));
    g_deadline.store(deadline.time_since_epoch().count());
    PyEval_SetTrace(&TraceHangGuard, nullptr);

    PyObject* result = nullptr;
    if (globals != nullptr) {
        result = PyRun_String(code.c_str(), Py_file_input, globals, globals);
    } else {
        PyErr_SetString(PyExc_RuntimeError, "no __main__ namespace");
    }
    PyEval_SetTrace(nullptr, nullptr);
    g_deadline.store(0);

    if (result == nullptr) {
        // SystemExit must NOT propagate (it would terminate the editor); report it.
        if (PyErr_ExceptionMatches(PyExc_SystemExit)) {
            PyErr_Clear();
            if (sink != nullptr) {
                PyObject* w = PyObject_CallMethod(
                    sink, "write", "s",
                    "SystemExit ignored (a script cannot exit the editor)\n");
                Py_XDECREF(w);
            }
        } else {
            PyErr_Print();  // traceback -> the redirected sys.stderr (sink)
        }
    } else {
        Py_DECREF(result);
    }

    // Restore the original streams and drain the sink.
    if (sys_mod != nullptr) {
        if (old_out != nullptr) PyObject_SetAttrString(sys_mod, "stdout", old_out);
        if (old_err != nullptr) PyObject_SetAttrString(sys_mod, "stderr", old_err);
    }
    std::string out;
    if (sink != nullptr) {
        PyObject* val = PyObject_CallMethod(sink, "getvalue", nullptr);
        if (val != nullptr) {
            Py_ssize_t len = 0;
            const char* s = PyUnicode_AsUTF8AndSize(val, &len);
            if (s != nullptr) out.assign(s, static_cast<size_t>(len));
            Py_DECREF(val);
        }
    }
    Py_XDECREF(old_out);
    Py_XDECREF(old_err);
    Py_XDECREF(sink);
    Py_XDECREF(sys_mod);
    Py_XDECREF(io_mod);
    if (out.size() > kRunOutputCap) out.erase(0, out.size() - kRunOutputCap);
    return out;
}

}  // namespace nuka::runtime::app::viewer
