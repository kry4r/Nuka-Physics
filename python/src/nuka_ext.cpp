// nuka_ext.cpp -- nanobind extension binding the Nuka C ABI.
//
// ABI CONTRACT (read python/README.md):
//   * This extension is compiled with g++-10, _GLIBCXX_USE_CXX11_ABI=1, exactly
//     like the engine static/shared libs it links (libnuka.so). It must NOT link
//     libtorch (pip libtorch is the OLD pre-CXX11 ABI=0 -> symbol clash).
//   * torch interop is therefore done purely through DLPack, an ABI-neutral C
//     struct protocol. `buffer_view(field)` returns an nb::ndarray that exposes
//     __dlpack__ / __dlpack_device__; `torch.from_dlpack(...)` consumes it as a
//     ZERO-COPY CUDA tensor aliasing the engine's live device buffer.
//
// We bind ONLY the C ABI (nuka/nuka.h) -- not the C++ wrapper -- to stay
// dependency-light. The single CUDA call we need (cudaMemcpy for the ergonomic
// set_drive_targets path) is declared locally and resolved against the same
// libcudart.so.12 that libnuka.so already links.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nuka/nuka.h"

namespace nb = nanobind;

// --- minimal CUDA runtime decls (resolved via libcudart.so.12 at link time) --
// We avoid including <cuda_runtime.h> / linking the full toolkit headers; only
// these two symbols are needed for the host/device copy convenience path.
extern "C" {
int cudaMemcpy(void* dst, const void* src, size_t count, int kind);
int cudaDeviceSynchronize(void);
}
namespace {
constexpr int kCudaMemcpyHostToDevice = 1;
constexpr int kCudaMemcpyDeviceToDevice = 3;
}  // namespace

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------
namespace {

void check(nuka_result_t r, const char* what) {
    if (r != NUKA_RESULT_OK) {
        const char* msg = nuka_result_message(r);
        throw std::runtime_error(std::string(what) + " failed: " +
                                 (msg ? msg : "?") + " (" + std::to_string((int)r) + ")");
    }
}

// The writable float ndarray type we hand back for zero-copy views.
//
// We tag it nb::pytorch so buffer_view(field) returns a torch.Tensor that
// ALIASES the engine device buffer (exposes __dlpack__ / __dlpack_device__,
// data_ptr() == engine device_ptr -> genuinely zero-copy). IMPORTANT for the
// ABI rule: this only makes nanobind *import* the torch PYTHON module to build
// the Tensor wrapper; it does NOT link libtorch into this .so (readelf confirms
// no torch/c10 in NEEDED). Importing torch alongside nuka in one process is the
// task's premise; the forbidden thing is LINKING the ABI=0 libtorch. A
// framework-neutral nb::ndarray<float> (raw DLPack capsule, no torch import) is
// a future option if a non-torch consumer (cupy/jax) ever appears.
//
// No shape / device is baked into the type: both are supplied at construction so
// the helper is fully GENERIC across fields (a future field works unchanged).
using FloatArray = nb::ndarray<nb::pytorch, float>;

}  // namespace

// ---------------------------------------------------------------------------
// Device wrapper (RAII)
// ---------------------------------------------------------------------------
class Device {
public:
    // stream_ptr: optional caller-supplied cudaStream_t (as a uintptr_t). 0 =>
    // null => the engine creates+owns a default stream. A non-zero value is
    // adopted verbatim as the device-context stream (see src/c_abi/device.cpp:
    // when desc.cuda_stream != nullptr the engine skips its OwnedStream and runs
    // every kernel on the supplied stream) -- this is how a torch
    // cuda_stream flows in so torch ops + physics share ordering with no
    // explicit cudaStreamSynchronize.
    static Device* create(uint32_t ordinal, uintptr_t stream_ptr) {
        nuka_device_desc_t desc{};
        desc.gpu_index = ordinal;
        desc.cuda_stream = reinterpret_cast<void*>(stream_ptr);
        desc.backend_selection_layer_enabled = 1u;
        nuka_device_handle h = nullptr;
        check(nuka_device_create(&desc, &h), "nuka_device_create");
        return new Device(h);
    }

    void close() {
        if (h_ != nullptr) {
            nuka_device_destroy(h_);
            h_ = nullptr;
        }
    }
    ~Device() { close(); }

    nuka_device_handle raw() const { return h_; }
    bool valid() const { return h_ != nullptr; }

private:
    explicit Device(nuka_device_handle h) : h_(h) {}
    nuka_device_handle h_ = nullptr;
};

// ---------------------------------------------------------------------------
// World wrapper (RAII)
// ---------------------------------------------------------------------------
class World {
public:
    static World* create_from_scene(Device* device, const std::string& scene_path,
                                    uint32_t env_count, float dt) {
        if (device == nullptr || !device->valid()) {
            throw std::runtime_error("create_from_scene: invalid device");
        }
        nuka_world_desc_t desc{};
        desc.scene_path = scene_path.c_str();
        desc.env_count = env_count;
        desc.fixed_dt = dt;
        nuka_world_handle h = nullptr;
        check(nuka_world_create_from_scene(device->raw(), &desc, &h),
              "nuka_world_create_from_scene");

        // Derive base_link_count = total_link_count / env_count from the
        // JOINT_POSITION view (the C ABI exposes no explicit accessor; this is
        // the generic, correct derivation -- see README known-gap).
        nuka_buffer_view_t qv{};
        check(nuka_world_get_buffer_view(h, NUKA_FIELD_JOINT_POSITION, &qv),
              "nuka_world_get_buffer_view(JOINT_POSITION)");
        uint32_t base_link_count =
            (env_count > 0u) ? (uint32_t)(qv.element_count / env_count) : 0u;
        return new World(h, env_count, base_link_count, dt);
    }

    void destroy() {
        if (h_ != nullptr) {
            nuka_world_destroy(h_);
            h_ = nullptr;
        }
    }
    ~World() { destroy(); }

    void step() { check(nuka_world_step(h_), "nuka_world_step"); }
    void step_n(uint32_t n) { check(nuka_world_step_n(h_, n), "nuka_world_step_n"); }

    // p03 RL autoreset. reset() restores ALL envs to the creation-time initial
    // pose; reset_envs(ids) restores only the listed envs (the masked autoreset
    // path) and leaves every other env byte-for-byte unchanged.
    void reset() { check(nuka_world_reset(h_), "nuka_world_reset"); }
    void reset_envs(const uint32_t* ids, uint32_t count) {
        check(nuka_world_reset_envs(h_, ids, count), "nuka_world_reset_envs");
    }

    uint32_t env_count() const { return env_count_; }
    uint32_t base_link_count() const { return base_link_count_; }
    // action_dim == the number of ACTUATED joint DOFs a policy controls ==
    // base_link_count_ - 1. The engine's JOINT_POSITION / DRIVE_TARGET buffers are
    // base_link_count_ wide; slot 0 is the root link, which on a floating base
    // carries no actuated DOF and is inert under the PD drive (its Kp/Kd are 0).
    // The actuated joints (12 for Go2) occupy slots [1 .. base_link_count_). A
    // policy emits an (env, action_dim) tensor (12-wide for Go2) and the
    // drive/autograd path writes it into DRIVE_TARGET slots [1:], leaving the root
    // slot untouched -- matching the proven go2_policy_drive harness (GO2_BLC=13,
    // targets/gains set on slots 1:13). Use `base_link_count` for the raw buffer
    // width; use `action_dim` for the policy/RL action width.
    uint32_t action_dim() const {
        return base_link_count_ > 0u ? base_link_count_ - 1u : 0u;
    }
    float dt() const { return dt_; }

    nuka_world_handle raw() const { return h_; }

    nuka_buffer_view_t get_view(nuka_state_field_t field) const {
        nuka_buffer_view_t view{};
        check(nuka_world_get_buffer_view(h_, field, &view),
              "nuka_world_get_buffer_view");
        if (view.device_ptr == nullptr) {
            throw std::runtime_error("buffer_view: engine returned null device_ptr");
        }
        return view;
    }

private:
    World(nuka_world_handle h, uint32_t ec, uint32_t blc, float dt)
        : h_(h), env_count_(ec), base_link_count_(blc), dt_(dt) {}
    nuka_world_handle h_ = nullptr;
    uint32_t env_count_ = 0u;
    uint32_t base_link_count_ = 0u;
    float dt_ = 0.0f;
};

// ---------------------------------------------------------------------------
// buffer_view(field) -> zero-copy DLPack-capable nb::ndarray (CUDA, float32)
// ---------------------------------------------------------------------------
namespace {

// Build a CUDA float ndarray aliasing `view.device_ptr` with the per-field
// natural shape. The returned ndarray is reference-policy (no copy) and `owner`
// (the Python World object) keeps the engine buffer alive for the array's life.
FloatArray make_view_array(World* world, nuka_state_field_t field, nb::handle owner) {
    nuka_buffer_view_t view = world->get_view(field);
    if (view.dtype != 0u) {
        throw std::runtime_error("buffer_view: non-float32 dtype not supported");
    }

    const uint32_t ec = world->env_count();
    const uint32_t blc = world->base_link_count();
    const int dev_id = 0;  // CUDA_VISIBLE_DEVICES=0 -> ordinal 0 in this process.

    // floats-per-element from the view stride (1 for q/qd/drive; 7 for link pose).
    const size_t fpe = (view.element_stride_bytes >= sizeof(float))
                           ? (view.element_stride_bytes / sizeof(float))
                           : 1u;

    // Per-ENV (not per-link) multi-float field: BASE_POSE (7 floats, Transform)
    // has element_count == env_count -- ONE root pose per env, NOT env_count*blc.
    // Shape it (env, fpe) so Python reads base_pose[:, 3:7] for the root quat,
    // directly parallel to ARTICULATION_LINK_POSE[:, 0, 3:7]. Must precede the
    // generic fpe>1 branch (whose ec*blc==element_count check would fail here and
    // silently drop BASE_POSE to the flat (env*7,) fallback).
    if (fpe > 1u && ec > 0u && (size_t)ec == view.element_count) {
        size_t shape[2] = {ec, fpe};
        return FloatArray(view.device_ptr, 2, shape, owner, nullptr,
                          nb::dtype<float>(), nb::device::cuda::value, dev_id);
    }

    // Multi-float-per-link fields (fpe > 1): ARTICULATION_LINK_POSE (7 floats,
    // Transform) and LINK_VELOCITY (6 floats, omega-first spatial velocity).
    // element_count == env_count*base_link_count; shape (env, base_link, fpe).
    if (fpe > 1u) {
        if ((size_t)ec * blc != view.element_count) {
            // Fall back to flat (shouldn't happen for go2).
            size_t shape[1] = {view.element_count * fpe};
            return FloatArray(view.device_ptr, 1, shape, owner, nullptr,
                              nb::dtype<float>(), nb::device::cuda::value, dev_id);
        }
        size_t shape[3] = {ec, blc, fpe};
        return FloatArray(view.device_ptr, 3, shape, owner, nullptr,
                          nb::dtype<float>(), nb::device::cuda::value, dev_id);
    }

    // q / qd / drive-target / drive-gains (and any future flat per-(env,link)
    // field):
    // element_count == env_count*base_link_count, fpe == 1 -> shape (ec, blc).
    if (fpe == 1u && ec > 0u && (size_t)ec * blc == view.element_count) {
        size_t shape[2] = {ec, blc};
        return FloatArray(view.device_ptr, 2, shape, owner, nullptr,
                          nb::dtype<float>(), nb::device::cuda::value, dev_id);
    }

    // Generic fallback: 1D over all floats. Keeps buffer_view() field-agnostic so
    // a future field with a different layout still yields a usable DLPack tensor.
    size_t shape[1] = {view.element_count * fpe};
    return FloatArray(view.device_ptr, 1, shape, owner, nullptr,
                      nb::dtype<float>(), nb::device::cuda::value, dev_id);
}

}  // namespace

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
NB_MODULE(_nuka_ext, m) {
    m.doc() = "Nuka physics engine -- nanobind binding (zero-copy DLPack interop)";

    nuka_version_t v = nuka_get_version();
    m.attr("__engine_version__") =
        std::to_string(v.major) + "." + std::to_string(v.minor) + "." + std::to_string(v.patch);

    // Field enum (kept identical to nuka_state_field_t).
    nb::enum_<nuka_state_field_t>(m, "Field")
        .value("RIGID_BODY_TRANSFORM", NUKA_FIELD_RIGID_BODY_TRANSFORM)
        .value("ARTICULATION_LINK_POSE", NUKA_FIELD_ARTICULATION_LINK_POSE)
        .value("JOINT_POSITION", NUKA_FIELD_JOINT_POSITION)
        .value("JOINT_VELOCITY", NUKA_FIELD_JOINT_VELOCITY)
        .value("OBSERVATIONS", NUKA_FIELD_OBSERVATIONS)
        .value("CONTACT_POINTS", NUKA_FIELD_CONTACT_POINTS)
        .value("DRIVE_TARGET", NUKA_FIELD_DRIVE_TARGET)
        .value("LINK_VELOCITY", NUKA_FIELD_LINK_VELOCITY)
        .value("DRIVE_STIFFNESS", NUKA_FIELD_DRIVE_STIFFNESS)
        .value("DRIVE_DAMPING", NUKA_FIELD_DRIVE_DAMPING)
        .value("DRIVE_FORCE_LIMIT", NUKA_FIELD_DRIVE_FORCE_LIMIT)
        .value("BASE_POSE", NUKA_FIELD_BASE_POSE)
        .export_values();

    nb::class_<Device>(m, "Device")
        .def_static("create", &Device::create, nb::arg("ordinal") = 0,
                    nb::arg("stream_ptr") = uintptr_t{0},
                    nb::rv_policy::take_ownership,
                    "Create a CUDA device handle for the given ordinal. "
                    "stream_ptr (uintptr_t, default 0): when non-zero it is the "
                    "caller-supplied cudaStream_t the engine runs on (e.g. "
                    "torch.cuda.current_stream().cuda_stream); 0 lets the engine "
                    "own a default stream. Device.create(0) is unchanged.")
        .def("close", &Device::close, "Destroy the device handle.")
        .def("__enter__", [](Device& d) -> Device& { return d; })
        .def("__exit__",
             [](Device& d, nb::object, nb::object, nb::object) { d.close(); },
             nb::arg("exc_type").none(), nb::arg("exc_value").none(),
             nb::arg("traceback").none());

    nb::class_<World>(m, "World")
        .def_static("create_from_scene", &World::create_from_scene,
                    nb::arg("device"), nb::arg("scene_path"),
                    nb::arg("env_count"), nb::arg("dt") = 1.0f / 240.0f,
                    nb::rv_policy::take_ownership,
                    "Create a batched world from a USDA scene.")
        .def("step", &World::step, "Advance the world one fixed step.")
        .def("step_n", &World::step_n, nb::arg("n"),
             "Advance the world n fixed steps.")
        .def("reset", &World::reset,
             "Reset ALL envs to the deterministic creation-time initial pose "
             "(internal floating-base pose, base/joint velocities, joint "
             "positions; contact warm-start cleared). GPU-only, D1-deterministic. "
             "Batched (env_count>1) worlds only.")
        .def(
            "reset_envs",
            [](World& w, nb::object env_ids) {
                // Accept a 1-D int array (numpy / torch / list) -> host uint32[].
                // A control-plane call (a few ids); a host round-trip is fine.
                // `.tolist()` (when present) pulls a CUDA torch tensor to host
                // python ints uniformly; a plain list/tuple is iterated directly.
                nb::object seq = env_ids;
                if (nb::hasattr(env_ids, "tolist")) {
                    seq = env_ids.attr("tolist")();
                }
                std::vector<uint32_t> ids;
                for (nb::handle item : seq) {
                    const long long v = nb::cast<long long>(item);
                    if (v < 0) {
                        throw std::runtime_error(
                            "reset_envs: env_id must be non-negative");
                    }
                    ids.push_back(static_cast<uint32_t>(v));
                }
                w.reset_envs(ids.empty() ? nullptr : ids.data(),
                             static_cast<uint32_t>(ids.size()));
            },
            nb::arg("env_ids"),
            "Reset only the listed envs to the creation-time initial pose (the "
            "masked RL autoreset path). env_ids: a 1-D int array (numpy / torch / "
            "list) of env indices in [0, env_count). Un-listed envs are left "
            "byte-for-byte unchanged. GPU-only, D1-deterministic.")
        .def("destroy", &World::destroy, "Destroy the world.")
        .def_prop_ro("env_count", &World::env_count)
        .def_prop_ro("base_link_count", &World::base_link_count)
        .def_prop_ro("action_dim", &World::action_dim,
                     "Actuated joint DOFs a policy controls (12 for Go2) == "
                     "base_link_count - 1. The DRIVE_TARGET / JOINT_POSITION "
                     "buffers are base_link_count wide (slot 0 = the root link, "
                     "inert under the PD drive); the actuated joints occupy slots "
                     "[1:]. A policy emits an (env_count, action_dim) tensor and "
                     "the drive/autograd path writes it into slots [1:].")
        .def_prop_ro("dt", &World::dt)
        // Raw engine device pointer for a field (as a Python int). Lets a test
        // prove DLPack zero-copy: torch tensor.data_ptr() == this value.
        .def(
            "buffer_device_ptr",
            [](World& w, nuka_state_field_t field) -> uintptr_t {
                return reinterpret_cast<uintptr_t>(w.get_view(field).device_ptr);
            },
            nb::arg("field"),
            "Raw engine device pointer (int) backing `field` -- for zero-copy "
            "verification (torch.from_dlpack(...).data_ptr() must equal it).")
        .def("__enter__", [](World& w) -> World& { return w; })
        .def("__exit__",
             [](World& w, nb::object, nb::object, nb::object) { w.destroy(); },
             nb::arg("exc_type").none(), nb::arg("exc_value").none(),
             nb::arg("traceback").none())
        // Zero-copy DLPack view. `self` (the Python World) is the owner -> the
        // engine buffer stays alive for as long as the returned array does.
        .def(
            "buffer_view",
            [](nb::handle self, nuka_state_field_t field) {
                World* w = nb::cast<World*>(self);
                return make_view_array(w, field, self);
            },
            nb::arg("field"), nb::rv_policy::reference,
            "Return a zero-copy DLPack-capable CUDA float32 ndarray aliasing the "
            "engine's live device buffer for `field`. DRIVE_TARGET / "
            "DRIVE_STIFFNESS / DRIVE_DAMPING / DRIVE_FORCE_LIMIT are WRITABLE: "
            "torch writes in place and the next step() applies them. LINK_VELOCITY "
            "is read-only, shape (env, base_link, 6), omega-first [wx,wy,wz,vx,vy,"
            "vz]; the root slot is the live base spatial velocity in the root-link "
            "body frame.")
        // Ergonomic alternative to raw DLPack writes: copy a host/device float
        // array into the DRIVE_TARGET buffer.
        .def(
            "set_drive_targets",
            [](World& w,
               nb::ndarray<float, nb::ndim<1>, nb::c_contig> arr) {
                nuka_buffer_view_t view = w.get_view(NUKA_FIELD_DRIVE_TARGET);
                const size_t n = view.element_count;
                if (arr.shape(0) != n) {
                    throw std::runtime_error(
                        "set_drive_targets: array length " +
                        std::to_string(arr.shape(0)) + " != DRIVE_TARGET count " +
                        std::to_string(n));
                }
                const int dt = (arr.device_type() == nb::device::cuda::value)
                                   ? kCudaMemcpyDeviceToDevice
                                   : kCudaMemcpyHostToDevice;
                int rc = cudaMemcpy(view.device_ptr, arr.data(),
                                    n * sizeof(float), dt);
                if (rc != 0) {
                    throw std::runtime_error(
                        "set_drive_targets: cudaMemcpy failed (" +
                        std::to_string(rc) + ")");
                }
            },
            nb::arg("array"),
            "Copy a 1D contiguous float array (host or CUDA) into the live "
            "DRIVE_TARGET device buffer; picked up by the next step().");

    m.def("sync", []() {
        int rc = cudaDeviceSynchronize();
        if (rc != 0) {
            throw std::runtime_error("cudaDeviceSynchronize failed (" +
                                     std::to_string(rc) + ")");
        }
    }, "Block until all queued CUDA work completes (test/debug helper).");
}
