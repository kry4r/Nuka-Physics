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
#include "nuka/nuka_diffsim.h"
#include "nuka/nuka_grasp.h"  // v0.8 A2: the batched grasp world C ABI (BatchedUnifiedWorld)
#include "nuka/nuka_noise.h"

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

// p14a: the uint32 sibling for the ONE non-float32 field, CONTACT_LINK (per-slot
// owning global link index). Same nb::pytorch tag -> torch.from_dlpack yields a
// zero-copy torch.uint32 CUDA tensor (verified supported on torch>=2.6). Kept a
// distinct type because nb::ndarray bakes the scalar type into its C++ type.
using Uint32Array = nb::ndarray<nb::pytorch, uint32_t>;

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
                                    uint32_t env_count, float dt,
                                    uint32_t determinism, uint32_t control_mode,
                                    uint32_t osc_task_link) {
        if (device == nullptr || !device->valid()) {
            throw std::runtime_error("create_from_scene: invalid device");
        }
        if (determinism > 1u) {
            throw std::runtime_error(
                "create_from_scene: determinism must be 0 (Strong/D1) or "
                "1 (Weak/D2)");
        }
        // v0.5 C-fwd: 0 = PDPosition (default), 1 = Torque, 2 = Velocity, 3 =
        // ComputedTorque, 4 = Osc (operational-space control, forward position
        // task), 5 = Actuator. Only a value above 5 is rejected. Non-PD modes
        // require the batched (env_count > 1) path. (Mirror the engine's
        // IsControlModeImplemented set {0,1,2,3,4,5}.)
        if (control_mode > 5u) {
            throw std::runtime_error(
                "create_from_scene: control_mode must be 0 (PDPosition), "
                "1 (Torque), 2 (Velocity), 3 (ComputedTorque), 4 (Osc) or "
                "5 (Actuator)");
        }
        nuka_world_desc_t desc{};
        desc.scene_path = scene_path.c_str();
        desc.env_count = env_count;
        desc.fixed_dt = dt;
        // p01-W4: 0 = D1/Strong (default), 1 = D2/Weak. Plain uint8_t keeps the
        // desc C-compatible (the engine maps it to gpu::DeterminismLevel).
        desc.determinism = static_cast<uint8_t>(determinism);
        // v0.5 C-fwd: stage-1 control mode (0=PDPosition default, 1=Torque,
        // 2=Velocity, 3=ComputedTorque, 4=Osc, 5=Actuator). Zero-init default
        // already maps to PDPosition.
        desc.control_mode = static_cast<uint8_t>(control_mode);
        // v0.5 C-fwd slice 3: Osc task link (articulation-local). Read only in Osc
        // mode; ignored otherwise (zero-init default == root, a no-op task).
        desc.osc_task_link = osc_task_link;
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

    // v0.5 p04 §4 PARAMETER spine: set one articulation link's scalar mass at
    // runtime (GLOBAL link index in [0, total_link_count)). Rebuilds the link's
    // 6x6 spatial inertia from the new mass via the SAME affine MakeSpatialInertia
    // parameterization the mass-gradient adjoint assumes. The Python autograd layer
    // calls this BEFORE stepping so the sim actually uses the param tensor's value
    // (otherwise the gradient would be silently wrong). See nuka_diffsim.h.
    void set_link_mass(uint32_t link_index, float mass) {
        check(nuka_world_set_link_mass(h_, link_index, mass),
              "nuka_world_set_link_mass");
    }

    // v0.5 p04 A4 floating-gradcheck enabler: set the world's uniform gravity
    // (Z component, m/s^2). NO behavioral change until called (default stays
    // -9.81); a world that never calls this steps byte-identically. MUST be called
    // BEFORE nuka_tape_create -- the tape captures gravity_z at create time.
    void set_gravity_z(float gravity_z) {
        check(nuka_world_set_gravity_z(h_, gravity_z), "nuka_world_set_gravity_z");
    }

    // v0.5 p04 N1 sim-to-real sensor noise (Task 5.4.8/5.4.9). Register a noise
    // descriptor on a per-field basis. kind: 0=NONE (clears), 1=GAUSSIAN
    // (param1=mean, param2=stddev), 2=POISSON (param1=lambda). Recording resets
    // that field's per-field sequence counter to 0. See nuka_noise.h.
    void set_sensor_noise(nuka_state_field_t field, int kind, float param1,
                          float param2, uint64_t seed) {
        nuka_sensor_noise_desc_t desc{};
        desc.kind = static_cast<nuka_noise_kind_t>(kind);
        desc.param1 = param1;
        desc.param2 = param2;
        desc.seed = seed;
        check(nuka_world_set_sensor_noise(h_, field, &desc),
              "nuka_world_set_sensor_noise");
    }

    // Apply the registered noise to `field`'s device buffer in place ONCE, then
    // advance that field's sequence counter (next apply is independent noise).
    // NONE / no registered desc -> byte no-op. Non-float-stride field (pose) ->
    // NUKA_RESULT_NOT_SUPPORTED (raises).
    void apply_sensor_noise(nuka_state_field_t field) {
        check(nuka_world_apply_sensor_noise(h_, field),
              "nuka_world_apply_sensor_noise");
    }

    // v0.5 p04 N2 per-episode domain randomization (Task 5.4.7/5.4.9). mass /
    // friction are MULTIPLIER ranges; restitution / armature / gravity are OFFSET
    // ranges. enabled == 0 -> apply is a byte no-op. See nuka_noise.h.
    void set_domain_randomization(float mass_mul_lo, float mass_mul_hi,
                                  float friction_mul_lo, float friction_mul_hi,
                                  float restitution_off_lo,
                                  float restitution_off_hi, float armature_off_lo,
                                  float armature_off_hi, float gravity_off_lo,
                                  float gravity_off_hi, uint64_t seed,
                                  int enabled) {
        nuka_domain_randomization_desc_t desc{};
        desc.mass_mul_lo = mass_mul_lo;
        desc.mass_mul_hi = mass_mul_hi;
        desc.friction_mul_lo = friction_mul_lo;
        desc.friction_mul_hi = friction_mul_hi;
        desc.restitution_off_lo = restitution_off_lo;
        desc.restitution_off_hi = restitution_off_hi;
        desc.armature_off_lo = armature_off_lo;
        desc.armature_off_hi = armature_off_hi;
        desc.gravity_off_lo = gravity_off_lo;
        desc.gravity_off_hi = gravity_off_hi;
        desc.seed = seed;
        desc.enabled = enabled;
        check(nuka_world_set_domain_randomization(h_, &desc),
              "nuka_world_set_domain_randomization");
    }

    // Sample + apply the stored randomization for ALL envs (call at episode
    // reset, BEFORE Tape.create -- the tape captures gravity at create time and
    // mass must be in place before the first step). DR disabled -> byte no-op.
    void apply_domain_randomization() {
        check(nuka_world_apply_domain_randomization(h_),
              "nuka_world_apply_domain_randomization");
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
// Tape wrapper (RAII) -- the multi-step differentiable rollout (nuka_diffsim.h).
// ---------------------------------------------------------------------------
//
// Bound for the v0.5 p04 Python autograd layer. The tape records a contact-free
// PD rollout in place on the world's articulation state; backward() reverses it
// to per-step action gradients + the per-link mass-parameter gradient.
//
// ABI / DLPack note: nuka_tape_backward's C contract is HOST pointers (the engine
// uploads the seed to device and downloads the grads internally -- the device
// reduction order is fixed inside the engine, which is where D1 bit-exactness
// lives). So backward() here accepts a HOST-contiguous seed array and returns
// HOST grad arrays; the Python autograd layer moves them to CUDA with a plain
// (deterministic) .cuda() copy. This is NOT a Python-side float reduction over
// device data -- it is a straight copy of grads the engine already reduced
// deterministically, so D1 (two-run byte-identity) is preserved. The zero-copy
// DLPack buffer_view pattern is used for the FORWARD state channels (q'/qd'/
// v_root') the autograd layer reads from the live engine buffers.
class Tape {
public:
    static Tape* create(World* world, uint32_t checkpoint_interval,
                        uint32_t max_tape_entries, uint32_t max_checkpoints,
                        uint32_t recompute_on_backward) {
        if (world == nullptr) {
            throw std::runtime_error("Tape.create: null world");
        }
        nuka_tape_desc_t desc{};
        desc.checkpoint_interval = checkpoint_interval;
        desc.max_tape_entries = max_tape_entries;
        desc.max_checkpoints = max_checkpoints;
        desc.recompute_on_backward = recompute_on_backward;
        nuka_tape_handle h = nullptr;
        check(nuka_tape_create(world->raw(), &desc, &h), "nuka_tape_create");
        Tape* t = new Tape(h, world->raw());
        // Stash the C++ World wrapper so state_view can (a) shape the returned
        // ndarray with env_count/base_link_count and (b) use the World Python
        // object as the array's owner (it owns the aliased device buffer).
        t->world_wrapper_ = world;
        return t;
    }

    void destroy() {
        if (h_ != nullptr) {
            nuka_tape_destroy(h_);
            h_ = nullptr;
        }
    }
    ~Tape() { destroy(); }

    // Advance the differentiable rollout one contact-free PD step in place,
    // recording the step (action == the world's current DRIVE_TARGET buffer).
    void step_with_tape() {
        check(nuka_world_step_with_tape(world_, h_), "nuka_world_step_with_tape");
    }

    void reset() { check(nuka_tape_reset(h_), "nuka_tape_reset"); }

    uint32_t step_count() const { return nuka_tape_step_count(h_); }
    uint32_t link_count() const { return nuka_tape_link_count(h_); }

    // Zero-copy DLPack view into the tape's LIVE differentiable state (the
    // OBSERVATION side: q / qdot / link_velocity the tape forward evolves in
    // place). Defined out-of-line below make_array_from_view (the shaping helper
    // is declared after this class). field: JOINT_POSITION / JOINT_VELOCITY /
    // LINK_VELOCITY. Returns a FloatArray (zero-copy) exactly like buffer_view.
    FloatArray state_view(nuka_state_field_t field);

    // Reverse pass. `seed` is a HOST contiguous float array in the layout
    // [dL/dq'(n) | dL/dqdot'(n) | dL/dv_root'(6n)] (length 8n) or empty -> all-zero
    // seed. Returns a tuple (grad_actions [step_count, n], grad_parameters [n]) as
    // HOST numpy float32 arrays. The autograd layer slices/moves them to CUDA.
    nb::object backward(nb::ndarray<float, nb::c_contig> seed) {
        const uint32_t n = link_count();
        const uint32_t steps = step_count();
        if (n == 0u) {
            throw std::runtime_error("Tape.backward: empty tape (link_count==0)");
        }
        const float* seed_ptr = nullptr;
        if (seed.size() != 0u) {
            const size_t want = static_cast<size_t>(8u) * n;  // 2n + 6n
            if (seed.size() != want) {
                throw std::runtime_error(
                    "Tape.backward: seed length " + std::to_string(seed.size()) +
                    " != 8*link_count (" + std::to_string(want) + ")");
            }
            if (seed.device_type() != nb::device::cpu::value) {
                throw std::runtime_error(
                    "Tape.backward: seed must be a HOST (CPU) contiguous array");
            }
            seed_ptr = seed.data();
        }
        // Host output buffers. We allocate with `new[]` + a capsule deleter so the
        // returned nb::ndarray owns the memory (no dangling).
        const size_t na = static_cast<size_t>(steps) * n;
        auto* ga = new float[na == 0u ? 1u : na];
        auto* gp = new float[n];
        for (size_t i = 0u; i < na; ++i) ga[i] = 0.0f;
        for (uint32_t i = 0u; i < n; ++i) gp[i] = 0.0f;
        nuka_result_t r = nuka_tape_backward(h_, seed_ptr, ga, gp);
        if (r != NUKA_RESULT_OK) {
            delete[] ga;
            delete[] gp;
            check(r, "nuka_tape_backward");
        }
        nb::capsule ga_owner(ga, [](void* p) noexcept { delete[] (float*)p; });
        nb::capsule gp_owner(gp, [](void* p) noexcept { delete[] (float*)p; });
        size_t ga_shape[2] = {steps, n};
        size_t gp_shape[1] = {n};
        nb::ndarray<nb::numpy, float> ga_arr(ga, 2, ga_shape, ga_owner);
        nb::ndarray<nb::numpy, float> gp_arr(gp, 1, gp_shape, gp_owner);
        return nb::make_tuple(ga_arr, gp_arr);
    }

private:
    Tape(nuka_tape_handle h, nuka_world_handle w) : h_(h), world_(w) {}
    nuka_tape_handle h_ = nullptr;
    nuka_world_handle world_ = nullptr;
    // The owning C++ World wrapper (set in create()). state_view reads its
    // env_count/base_link_count for shaping and uses its Python object as the
    // returned array's owner (the World owns the aliased articulation device
    // buffer; the Tape only holds a raw handle). Friend so state_view (a member,
    // defined out-of-line) and create (static) reach it -- both are members so
    // no friend needed; documented here for clarity.
    World* world_wrapper_ = nullptr;
};

// ---------------------------------------------------------------------------
// buffer_view(field) -> zero-copy DLPack-capable nb::ndarray (CUDA, float32)
// ---------------------------------------------------------------------------
namespace {

// Build a CUDA float ndarray aliasing `view.device_ptr` with the per-field
// natural shape. The returned ndarray is reference-policy (no copy) and `owner`
// (the Python World object) keeps the engine buffer alive for the array's life.
// Shape a CUDA float ndarray aliasing `view.device_ptr` given the per-env
// dimensions (ec == env_count, blc == base_link_count). Called directly by BOTH
// World.buffer_view AND Tape.state_view so they shape identically
// (q/qd/drive -> (ec,blc); LINK_VELOCITY fpe=6 -> (ec,blc,6);
// BASE_POSE -> (ec,7)). `owner` keeps the engine buffer alive for the array's
// life. For single-env tape state, total_link_count == base_link_count so
// ec*blc == element_count holds and the natural (1,blc) / (1,blc,6) shapes apply.
FloatArray make_array_from_view(const nuka_buffer_view_t& view, uint32_t ec,
                                uint32_t blc, nb::handle owner) {
    if (view.dtype != 0u) {
        throw std::runtime_error("buffer_view: non-float32 dtype not supported");
    }
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

    // p14a: per-CONTACT-SLOT multi-float fields (fpe > 1, env-major at the FIXED
    // slot stride kMaxFootContactsPerEnv == 4): CONTACT_NORMAL / CONTACT_FORCE.
    // element_count == env_count * 4 (slot_count), NOT env_count*base_link_count,
    // so the per-link branch below would misroute these to the flat fallback. Shape
    // (env, slots_per_env, fpe) == (ec, 4, 3) -- the same (env, 4, 3) shape
    // CONTACT_POINTS gets. Discriminate by element_count being a per-ENV multiple
    // that is NOT the per-link count (element_count % ec == 0 && != ec*blc). Must
    // precede the per-link fpe>1 branch. ASSUMES base_link_count !=
    // kMaxFootContactsPerEnv (4): if an articulation had exactly 4 links a per-slot
    // field (element_count == env*4) would equal ec*blc and misroute to the
    // per-link branch -- not the case for go2 (blc=13); revisit if a 4-link
    // articulation is ever added.
    if (fpe > 1u && ec > 0u && view.element_count % ec == 0u &&
        view.element_count != (size_t)ec * blc) {
        const size_t slots_per_env = view.element_count / ec;
        size_t shape[3] = {ec, slots_per_env, fpe};
        return FloatArray(view.device_ptr, 3, shape, owner, nullptr,
                          nb::dtype<float>(), nb::device::cuda::value, dev_id);
    }

    // Multi-float-per-link fields (fpe > 1): ARTICULATION_LINK_POSE (7 floats,
    // Transform), LINK_VELOCITY (6 floats, omega-first spatial velocity), and p14a
    // LINK_CONTACT_WRENCH (6 floats, [F(3), tau(3)] per GLOBAL link).
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

// p14a: shape a per-CONTACT-SLOT uint32 view (CONTACT_LINK, dtype==1) as a
// zero-copy CUDA torch.uint32 ndarray, env-major at the fixed slot stride: shape
// (env, slots_per_env) == (ec, 4). element_count == ec*4 (slot_count), stride ==
// sizeof(uint32). owner keeps the engine buffer alive. Falls back to flat 1D if
// the per-env divisibility does not hold (it always does for go2: slot_count ==
// env_count * kMaxFootContactsPerEnv).
Uint32Array make_uint32_view(const nuka_buffer_view_t& view, uint32_t ec,
                             nb::handle owner) {
    const int dev_id = 0;
    if (ec > 0u && view.element_count % ec == 0u) {
        size_t shape[2] = {ec, view.element_count / ec};
        return Uint32Array(view.device_ptr, 2, shape, owner, nullptr,
                           nb::dtype<uint32_t>(), nb::device::cuda::value, dev_id);
    }
    size_t shape[1] = {view.element_count};
    return Uint32Array(view.device_ptr, 1, shape, owner, nullptr,
                       nb::dtype<uint32_t>(), nb::device::cuda::value, dev_id);
}

}  // namespace

// Tape.state_view: a zero-copy DLPack-capable CUDA float32 ndarray aliasing the
// tape's LIVE differentiable state (q / qdot / link_velocity), shaped per-env via
// the SAME helper World.buffer_view uses. Defined out-of-line (after the anon-ns
// helpers) so it can reach make_array_from_view, which is declared below Tape.
// The owner is the WORLD Python object (nb::find(world_wrapper_)) -- the engine
// buffer is owned by the World's articulation_device, NOT by the Tape; using the
// World as owner keeps that buffer alive for the array's life (the Tape holds only
// a raw handle). For single-env, total_link_count == base_link_count so the
// (1,n) / (1,n,6) shaping applies.
FloatArray Tape::state_view(nuka_state_field_t field) {
    if (world_wrapper_ == nullptr) {
        throw std::runtime_error("Tape.state_view: tape has no world wrapper");
    }
    nuka_buffer_view_t view{};
    check(nuka_tape_state_view(h_, field, &view), "nuka_tape_state_view");
    if (view.device_ptr == nullptr) {
        throw std::runtime_error(
            "Tape.state_view: engine returned null device_ptr");
    }
    // owner = the WORLD Python object (it owns the aliased articulation device
    // buffer; the Tape holds only a raw handle). Returns the zero-copy FloatArray
    // straight through the SAME nanobind ndarray caster World.buffer_view uses.
    nb::object owner = nb::find(world_wrapper_);
    return make_array_from_view(view, world_wrapper_->env_count(),
                                world_wrapper_->base_link_count(), owner);
}

// ---------------------------------------------------------------------------
// GraspWorld wrapper (RAII) -- the BATCHED GRASP WORLD (v0.8 A2, nuka_grasp.h).
// ---------------------------------------------------------------------------
//
// Wraps the nuka_grasp_world_* C ABI over BatchedUnifiedWorld so a Python PPO env
// (H1GraspEnv) can drive N parallel grasp envs. Mirrors the World wrapper shape: a
// static create() factory (take_ownership), step / set_actions, and readout methods
// that return HOST numpy arrays. UNLIKE World.buffer_view (zero-copy DEVICE DLPack),
// the obs here are HOST float* fills (new[] + capsule deleter -> nb::ndarray<nb::numpy>),
// the SAME host-array pattern Tape.backward uses -- the engine's ExportObsState already
// did the ONE bulk device->host download per buffer, so the Python side gets numpy.
// ---------------------------------------------------------------------------
namespace {

// Allocate a host float[] of `n` and wrap it as a 1-D numpy ndarray owning the
// memory (capsule deleter). Mirrors the Tape.backward() host-array pattern.
nb::ndarray<nb::numpy, float> make_host_float_array(size_t n, size_t d0) {
    auto* p = new float[n == 0u ? 1u : n];
    for (size_t i = 0u; i < n; ++i) p[i] = 0.0f;
    nb::capsule owner(p, [](void* q) noexcept { delete[] reinterpret_cast<float*>(q); });
    size_t shape[1] = {d0};
    return nb::ndarray<nb::numpy, float>(p, 1, shape, owner);
}

}  // namespace

class GraspWorld {
public:
    static GraspWorld* create(Device* device, const std::string& cup_asset_path,
                              uint32_t env_count, float grip_force, float friction_mu,
                              float gravity_z, float dt, float cup_start_z_offset,
                              float reset_jitter_x, float reset_jitter_y) {
        if (device == nullptr) {
            throw std::runtime_error("GraspWorld.create: device is None");
        }
        nuka_grasp_world_desc_t desc{};
        desc.cup_asset_path = cup_asset_path.c_str();
        desc.env_count = env_count;
        desc.grip_force = grip_force;
        desc.friction_mu = friction_mu;
        desc.gravity_z = gravity_z;
        desc.fixed_dt = dt;
        desc.cup_start_z_offset = cup_start_z_offset;
        desc.reset_jitter_x = reset_jitter_x;
        desc.reset_jitter_y = reset_jitter_y;
        nuka_grasp_world_handle h = nullptr;
        check(nuka_grasp_world_create(device->raw(), &desc, &h),
              "nuka_grasp_world_create");
        uint32_t ec = 0u, ad = 0u, nf = 0u, bpe = 0u;
        check(nuka_grasp_world_env_count(h, &ec), "nuka_grasp_world_env_count");
        check(nuka_grasp_world_action_dim(h, &ad), "nuka_grasp_world_action_dim");
        check(nuka_grasp_world_num_fingertips(h, &nf), "nuka_grasp_world_num_fingertips");
        check(nuka_grasp_world_bodies_per_env(h, &bpe), "nuka_grasp_world_bodies_per_env");
        return new GraspWorld(h, ec, ad, nf, bpe);
    }

    void destroy() {
        if (h_ != nullptr) {
            nuka_grasp_world_destroy(h_);
            h_ = nullptr;
        }
    }
    ~GraspWorld() { destroy(); }

    void step() { check(nuka_grasp_world_step(h_), "nuka_grasp_world_step"); }

    // set_actions: a HOST contiguous float32 array of length env_count*action_dim
    // (env-major). Accepts numpy / a CPU torch tensor (c_contig).
    void set_actions(nb::ndarray<float, nb::c_contig> actions) {
        const size_t want = static_cast<size_t>(env_count_) * action_dim_;
        if (actions.size() != want) {
            throw std::runtime_error(
                "GraspWorld.set_actions: length " + std::to_string(actions.size()) +
                " != env_count*action_dim (" + std::to_string(want) + ")");
        }
        if (actions.device_type() != nb::device::cpu::value) {
            throw std::runtime_error(
                "GraspWorld.set_actions: actions must be a HOST (CPU) contiguous array");
        }
        check(nuka_grasp_world_set_actions(h_, actions.data(), actions.size()),
              "nuka_grasp_world_set_actions");
    }

    // step_with_actions: set_actions then step in one call (the RL per-step driver).
    void step_with_actions(nb::ndarray<float, nb::c_contig> actions) {
        set_actions(actions);
        step();
    }

    // export_obs -> dict of HOST numpy arrays (env-major), shaped per the C ABI:
    //   q                    : (env_count, action_dim)
    //   qdot                 : (env_count, action_dim)
    //   fingertip_world_pos  : (env_count, 3*num_fingertips)
    //   finger_normal_impulse: (env_count, num_fingertips)
    nb::object export_obs() {
        const size_t n = env_count_;
        const size_t nq = n * action_dim_;
        const size_t nft = n * 3u * num_fingertips_;
        const size_t nimp = n * num_fingertips_;
        nb::ndarray<nb::numpy, float> q = make_host_float_array(nq, nq);
        nb::ndarray<nb::numpy, float> qdot = make_host_float_array(nq, nq);
        nb::ndarray<nb::numpy, float> ft = make_host_float_array(nft, nft);
        nb::ndarray<nb::numpy, float> imp = make_host_float_array(nimp, nimp);
        check(nuka_grasp_world_export_obs(
                  h_, q.data(), nq, qdot.data(), nq, ft.data(), nft, imp.data(), nimp),
              "nuka_grasp_world_export_obs");
        // Reshape the 1-D buffers to their env-major 2-D shapes via numpy.reshape.
        nb::dict out;
        out["q"] = q.cast().attr("reshape")(n, action_dim_);
        out["qdot"] = qdot.cast().attr("reshape")(n, action_dim_);
        out["fingertip_world_pos"] = ft.cast().attr("reshape")(n, 3u * num_fingertips_);
        out["finger_normal_impulse"] = imp.cast().attr("reshape")(n, num_fingertips_);
        return out;
    }

    // read_cup -> dict of HOST numpy arrays (env-major): cup pose/vel + per-env
    // contact signals (the reward inputs).
    //   cup_pose       : (env_count, 7)  [px,py,pz, qw,qx,qy,qz]
    //   cup_vel        : (env_count, 6)  [vx,vy,vz, wx,wy,wz]
    //   cup_z          : (env_count,)
    //   cup_vz         : (env_count,)
    //   finger_contacts: (env_count,) uint32
    nb::object read_cup() {
        const size_t n = env_count_;
        nb::ndarray<nb::numpy, float> pose = make_host_float_array(n * 7u, n * 7u);
        nb::ndarray<nb::numpy, float> vel = make_host_float_array(n * 6u, n * 6u);
        nb::ndarray<nb::numpy, float> cz = make_host_float_array(n, n);
        nb::ndarray<nb::numpy, float> cvz = make_host_float_array(n, n);
        auto* fc = new uint32_t[n == 0u ? 1u : n];
        for (size_t i = 0u; i < n; ++i) fc[i] = 0u;
        nb::capsule fc_owner(fc, [](void* q) noexcept {
            delete[] reinterpret_cast<uint32_t*>(q);
        });
        size_t fc_shape[1] = {n};
        nb::ndarray<nb::numpy, uint32_t> fc_arr(fc, 1, fc_shape, fc_owner);
        check(nuka_grasp_world_read_cup(h_, pose.data(), n * 7u, vel.data(), n * 6u,
                                        cz.data(), n, cvz.data(), n, fc, n),
              "nuka_grasp_world_read_cup");
        nb::dict out;
        out["cup_pose"] = pose.cast().attr("reshape")(n, 7u);
        out["cup_vel"] = vel.cast().attr("reshape")(n, 6u);
        out["cup_z"] = cz;
        out["cup_vz"] = cvz;
        out["finger_contacts"] = fc_arr;
        return out;
    }

    // reset_envs: restore the listed envs to a seed-randomized cup pose + nominal
    // gripper. env_ids: a 1-D int array (numpy / torch / list). Deterministic given seed.
    void reset_envs(nb::object env_ids, uint64_t seed) {
        nb::object seq = env_ids;
        if (nb::hasattr(env_ids, "tolist")) {
            seq = env_ids.attr("tolist")();
        }
        std::vector<uint32_t> ids;
        for (nb::handle item : seq) {
            const long long v = nb::cast<long long>(item);
            if (v < 0) {
                throw std::runtime_error("GraspWorld.reset_envs: env_id must be non-negative");
            }
            ids.push_back(static_cast<uint32_t>(v));
        }
        check(nuka_grasp_world_reset_envs(h_, ids.empty() ? nullptr : ids.data(),
                                          static_cast<uint32_t>(ids.size()), seed),
              "nuka_grasp_world_reset_envs");
    }

    uint32_t env_count() const { return env_count_; }
    uint32_t action_dim() const { return action_dim_; }
    uint32_t num_fingertips() const { return num_fingertips_; }
    uint32_t bodies_per_env() const { return bodies_per_env_; }

private:
    GraspWorld(nuka_grasp_world_handle h, uint32_t ec, uint32_t ad, uint32_t nf,
               uint32_t bpe)
        : h_(h), env_count_(ec), action_dim_(ad), num_fingertips_(nf), bodies_per_env_(bpe) {}
    nuka_grasp_world_handle h_ = nullptr;
    uint32_t env_count_ = 0u;
    uint32_t action_dim_ = 0u;
    uint32_t num_fingertips_ = 0u;
    uint32_t bodies_per_env_ = 0u;
};

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
NB_MODULE(_nuka_ext, m) {
    m.doc() = "Nuka physics engine -- nanobind binding (zero-copy DLPack interop)";

    nuka_version_t v = nuka_get_version();
    m.attr("__engine_version__") =
        std::to_string(v.major) + "." + std::to_string(v.minor) + "." + std::to_string(v.patch);

    // p01-W4 determinism levels (the int values nuka_world_desc_t.determinism /
    // World.create_from_scene(determinism=...) accept). STRONG (0, default) is
    // D1: bit-exact + reproducible. WEAK (1) is D2: the reserved atomic-fast-path
    // escape hatch -- today it shares the D1 kernels and is NOT held to the D1
    // bit-exact bar.
    m.attr("DETERMINISM_STRONG") = uint32_t{0};
    m.attr("DETERMINISM_WEAK") = uint32_t{1};

    // v0.5 C-fwd stage-1 control modes (the int values
    // nuka_world_desc_t.control_mode / World.create_from_scene(control_mode=...)
    // accept). PD_POSITION (0, default) is the legacy PD position drive,
    // byte-for-byte unchanged. TORQUE (1) reads NUKA_FIELD_TORQUE_INPUT; VELOCITY
    // (2) reads NUKA_FIELD_VELOCITY_TARGET (servo gain == DRIVE_STIFFNESS).
    // COMPUTED_TORQUE (3) is inverse-dynamics PD (tau = M*(Kp*e - Kd*qdot) + bias,
    // Kp/Kd == DRIVE_STIFFNESS/DRIVE_DAMPING, target == DRIVE_TARGET). ACTUATOR (5)
    // is a DC-motor torque-speed envelope on TORQUE_INPUT (tau_stall ==
    // DRIVE_FORCE_LIMIT, no-load speed == ACTUATOR_NOLOAD_SPEED). OSC (4) is
    // operational-space control (forward position task on the osc_task_link link
    // toward Field.TASK_TARGET; task Kp/Kd reuse DRIVE_STIFFNESS/DRIVE_DAMPING at
    // the task link). Non-PD modes need the batched (env_count > 1) path.
    m.attr("CONTROL_MODE_PD_POSITION") = uint32_t{0};
    m.attr("CONTROL_MODE_TORQUE") = uint32_t{1};
    m.attr("CONTROL_MODE_VELOCITY") = uint32_t{2};
    m.attr("CONTROL_MODE_COMPUTED_TORQUE") = uint32_t{3};
    m.attr("CONTROL_MODE_OSC") = uint32_t{4};
    m.attr("CONTROL_MODE_ACTUATOR") = uint32_t{5};

    // v0.5 p04 N1 sim-to-real sensor-noise kinds (the int values
    // World.set_sensor_noise(kind=...) accepts). NONE (0, default) clears the
    // field's noise -> apply is a byte no-op. GAUSSIAN (1): param1=mean,
    // param2=stddev. POISSON (2): param1=lambda. Mirrors nuka_noise_kind_t.
    m.attr("NOISE_NONE") = int{0};
    m.attr("NOISE_GAUSSIAN") = int{1};
    m.attr("NOISE_POISSON") = int{2};

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
        .value("TORQUE_INPUT", NUKA_FIELD_TORQUE_INPUT)
        .value("VELOCITY_TARGET", NUKA_FIELD_VELOCITY_TARGET)
        .value("ACTUATOR_NOLOAD_SPEED", NUKA_FIELD_ACTUATOR_NOLOAD_SPEED)
        .value("TASK_TARGET", NUKA_FIELD_TASK_TARGET)
        // p14a (v0.7) contact-force readout surface (batched/multi-env only).
        .value("LINK_CONTACT_WRENCH", NUKA_FIELD_LINK_CONTACT_WRENCH)
        .value("CONTACT_NORMAL", NUKA_FIELD_CONTACT_NORMAL)
        .value("CONTACT_FORCE", NUKA_FIELD_CONTACT_FORCE)
        .value("CONTACT_LINK", NUKA_FIELD_CONTACT_LINK)
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
                    nb::arg("determinism") = uint32_t{0},
                    nb::arg("control_mode") = uint32_t{0},
                    nb::arg("osc_task_link") = uint32_t{0},
                    nb::rv_policy::take_ownership,
                    "Create a batched world from a USDA scene. determinism "
                    "(p01-W4, default 0): 0 = DETERMINISM_STRONG (D1, bit-exact "
                    "and reproducible -- the default) or 1 = DETERMINISM_WEAK (D2, "
                    "the reserved atomic-fast-path escape hatch; today it selects "
                    "the SAME kernels as D1 and is NOT held to the D1 bit-exact "
                    "bar). control_mode (v0.5 C-fwd, default 0): 0 = "
                    "CONTROL_MODE_PD_POSITION (legacy PD drive, byte-for-byte "
                    "unchanged), 1 = CONTROL_MODE_TORQUE (writes Field.TORQUE_INPUT), "
                    "2 = CONTROL_MODE_VELOCITY (writes Field.VELOCITY_TARGET; servo "
                    "gain == DRIVE_STIFFNESS), 3 = CONTROL_MODE_COMPUTED_TORQUE "
                    "(inverse-dynamics PD on DRIVE_TARGET; gains == "
                    "DRIVE_STIFFNESS/DRIVE_DAMPING), 4 = CONTROL_MODE_OSC "
                    "(operational-space control, forward position task on the world "
                    "position of the osc_task_link link toward Field.TASK_TARGET; "
                    "task gains Kp/Kd reuse DRIVE_STIFFNESS/DRIVE_DAMPING at the "
                    "task link), 5 = CONTROL_MODE_ACTUATOR "
                    "(DC-motor envelope on Field.TORQUE_INPUT; tau_stall == "
                    "DRIVE_FORCE_LIMIT, no-load speed == Field.ACTUATOR_NOLOAD_SPEED). "
                    "Non-PD modes require env_count > 1. osc_task_link (default 0, "
                    "Osc only): the articulation-local link index of the task link; "
                    "for a fixed-base scene the root (0) is a no-op, so set a real "
                    "end-effector (e.g. a foot/calf local link).")
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
            [](nb::handle self, nuka_state_field_t field) -> nb::object {
                World* w = nb::cast<World*>(self);
                // p14a: CONTACT_LINK is the ONE uint32 field (dtype==1); every other
                // field is float32 (dtype==0). Inspect the engine view's dtype and
                // build the matching-typed zero-copy ndarray. Both share the
                // nb::pytorch tag -> torch.from_dlpack(...) yields a CUDA tensor
                // aliasing the engine buffer (float32 or uint32 respectively).
                nuka_buffer_view_t view = w->get_view(field);
                if (view.dtype == 1u) {
                    return nb::cast(make_uint32_view(view, w->env_count(), self));
                }
                return nb::cast(make_array_from_view(
                    view, w->env_count(), w->base_link_count(), self));
            },
            nb::arg("field"), nb::rv_policy::reference,
            "Return a zero-copy DLPack-capable CUDA ndarray aliasing the engine's "
            "live device buffer for `field` (float32 for every field except "
            "CONTACT_LINK, which is uint32). DRIVE_TARGET / DRIVE_STIFFNESS / "
            "DRIVE_DAMPING / DRIVE_FORCE_LIMIT are WRITABLE: torch writes in place "
            "and the next step() applies them. LINK_VELOCITY is read-only, shape "
            "(env, base_link, 6), omega-first [wx,wy,wz,vx,vy,vz]; the root slot is "
            "the live base spatial velocity in the root-link body frame. p14a "
            "contact-force readout (batched only): LINK_CONTACT_WRENCH (env, "
            "base_link, 6) [F(3),tau(3)] world, tau about the link frame origin; "
            "CONTACT_NORMAL / CONTACT_FORCE (env, 4, 3) per-slot; CONTACT_LINK "
            "(env, 4) uint32 owning global link index.")
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
            "DRIVE_TARGET device buffer; picked up by the next step().")
        // v0.5 p04 §4 PARAMETER spine: runtime link-mass setter (the forward write
        // the autograd layer uses to push a mass param tensor INTO the sim).
        .def("set_link_mass", &World::set_link_mass, nb::arg("link_index"),
             nb::arg("mass"),
             "Set one articulation link's scalar mass (GLOBAL link index in "
             "[0, base_link_count) for single-env). Rebuilds the link's 6x6 spatial "
             "inertia affinely from the new mass via the SAME MakeSpatialInertia "
             "parameterization the mass-gradient adjoint assumes; the next step()/"
             "step_with_tape() reads it fresh. Used by nuka.autograd."
             "differentiable_step to push a mass param tensor's value into the sim "
             "BEFORE stepping (so the true d(output)/d(mass) is nonzero). mass > 0; "
             "an out-of-range index or non-positive mass raises.")
        // v0.5 p04 A4 enabler: world gravity setter (MUST precede Tape.create).
        .def("set_gravity_z", &World::set_gravity_z, nb::arg("gravity_z"),
             "Set the world's uniform gravity Z (m/s^2). Default is -9.81; call "
             "BEFORE creating a Tape (the tape captures gravity at create time). "
             "Used by the floating mass-gradcheck, which needs g=0 so the deferred "
             "floating-base orientation channel is inert. A world that never calls "
             "this steps byte-identically (default unchanged).")
        // v0.5 p04 N1 sim-to-real sensor noise (Task 5.4.9). Register/clear a
        // per-field noise descriptor; apply it in place to the live device
        // buffer. Counter-based (Philox) -> D1 two-run bit-exact.
        .def("set_sensor_noise", &World::set_sensor_noise, nb::arg("field"),
             nb::arg("kind"), nb::arg("param1") = 0.0f, nb::arg("param2") = 0.0f,
             nb::arg("seed") = uint64_t{0},
             "Register sim-to-real noise on `field`. kind: 0 = NOISE_NONE (clears "
             "-> apply becomes a byte no-op), 1 = NOISE_GAUSSIAN (param1=mean, "
             "param2=stddev), 2 = NOISE_POISSON (param1=lambda). Recording resets "
             "the field's per-field sequence counter to 0. Only float-stride "
             "fields (q/qd/drive/etc.) are supported by a later apply; a "
             "non-float-stride field (e.g. ARTICULATION_LINK_POSE) registers OK "
             "but raises NOT_SUPPORTED on apply. An out-of-range field or unknown "
             "kind raises.")
        .def("apply_sensor_noise", &World::apply_sensor_noise, nb::arg("field"),
             "Apply the registered noise to `field`'s device buffer ONCE, in "
             "place, then advance that field's sequence counter (so the next "
             "apply is independent noise across steps). NONE / no registered desc "
             "-> byte no-op. Non-float-stride field -> raises NOT_SUPPORTED. The "
             "kernel is async; call nuka.sync() before reading the buffer back.")
        // v0.5 p04 N2 per-episode domain randomization (Task 5.4.9). mass /
        // friction are MULTIPLIER ranges; restitution / armature / gravity are
        // OFFSET ranges. enabled == 0 -> apply is a byte no-op (oracle safe).
        .def("set_domain_randomization", &World::set_domain_randomization,
             nb::arg("mass_mul_lo"), nb::arg("mass_mul_hi"),
             nb::arg("friction_mul_lo"), nb::arg("friction_mul_hi"),
             nb::arg("restitution_off_lo"), nb::arg("restitution_off_hi"),
             nb::arg("armature_off_lo"), nb::arg("armature_off_hi"),
             nb::arg("gravity_off_lo"), nb::arg("gravity_off_hi"),
             nb::arg("seed") = uint64_t{0}, nb::arg("enabled") = int{1},
             "Record the per-episode domain-randomization descriptor. mass / "
             "friction are MULTIPLIER ranges [lo, hi] (nominal * mult); "
             "restitution / armature / gravity are OFFSET ranges (nominal + "
             "offset). Each is sampled ONCE per env per episode-reset as a pure "
             "function of (seed, env_idx, param) via counter-based Philox -> D1 "
             "two-run bit-exact backward. enabled == 0 (or never set) -> apply is "
             "a byte no-op. Recording does NOT sample/apply -- call "
             "apply_domain_randomization at episode reset.")
        .def("apply_domain_randomization", &World::apply_domain_randomization,
             "Sample + apply the stored randomization for ALL envs (call at "
             "episode reset). On the FIRST enabled apply it snapshots a nominal "
             "baseline (per-link mass, gravity.z, per-DOF armature) so the apply "
             "is idempotent across resets. DR disabled -> byte no-op. MUST be "
             "called BEFORE Tape.create (the tape captures gravity at create "
             "time; mass must be in place before the first step_with_tape). A "
             "non-articulated world raises NOT_SUPPORTED.");

    // -----------------------------------------------------------------------
    // Tape -- the multi-step differentiable rollout (nuka_diffsim.h).
    // -----------------------------------------------------------------------
    nb::class_<Tape>(m, "Tape")
        .def_static("create", &Tape::create, nb::arg("world"),
                    nb::arg("checkpoint_interval") = uint32_t{3},
                    nb::arg("max_tape_entries") = uint32_t{1024},
                    nb::arg("max_checkpoints") = uint32_t{128},
                    nb::arg("recompute_on_backward") = uint32_t{1},
                    nb::rv_policy::take_ownership,
                    "Create a differentiable-rollout tape bound to `world`'s "
                    "articulation state + drive gains + link masses. The world must "
                    "be articulated and SINGLE-ENV (env_count==1) for the "
                    "contact-free p02 slice. recompute_on_backward=1 is the "
                    "checkpointed reverse (memory ~ N/K); 0 is full-tape debug mode "
                    "(a checkpoint every step). If World.set_gravity_z was called, "
                    "call it BEFORE this -- gravity is captured here.")
        .def("step_with_tape", &Tape::step_with_tape,
             "Advance the differentiable rollout ONE contact-free PD step in place "
             "on the world's articulation state, recording the step. The action is "
             "the world's current DRIVE_TARGET buffer (write it via buffer_view "
             "before each call). NOTE: this runs the differentiable forward, NOT "
             "the production contact pipeline of World.step().")
        .def("reset", &Tape::reset,
             "Clear the recorded rollout (steps + checkpoints) for reuse. Does NOT "
             "reset the world's articulation state.")
        .def_prop_ro("step_count", &Tape::step_count,
                     "Number of forward steps currently recorded.")
        .def_prop_ro("link_count", &Tape::link_count,
                     "Per-step action / parameter width == articulation "
                     "total_link_count (== base_link_count for single-env).")
        .def("state_view", &Tape::state_view, nb::arg("field"),
             nb::rv_policy::reference,
             "Return a zero-copy DLPack-capable CUDA float32 ndarray aliasing the "
             "tape's LIVE differentiable state for `field` (the OBSERVATION side). "
             "JOINT_POSITION -> q (1, n); JOINT_VELOCITY -> qdot (1, n); "
             "LINK_VELOCITY -> per-link spatial velocity (1, n, 6), omega-first "
             "[wx,wy,wz,vx,vy,vz] (root slot = base spatial velocity in the "
             "root-link body frame). UNLIKE World.buffer_view's single-env arm "
             "(which reads the host mirror, NOT updated by step_with_tape and so "
             "stale 0.0), this reflects the state step_with_tape evolves in place. "
             "Read-only; write the per-step action via World.buffer_view"
             "(DRIVE_TARGET).")
        .def("backward", &Tape::backward, nb::arg("seed"),
             "Run the reverse pass. `seed` is a HOST (CPU) contiguous float32 array "
             "in the layout [dL/dq'(n) | dL/dqdot'(n) | dL/dv_root'(6n)] (length "
             "8*link_count), or an empty array for an all-zero seed. Returns a tuple "
             "(grad_actions, grad_parameters) of HOST numpy float32 arrays: "
             "grad_actions is (step_count, link_count) per-step per-link action "
             "gradient; grad_parameters is (link_count,) the dL/d(link mass) summed "
             "over steps. Bit-exact across two runs (D1).")
        .def("destroy", &Tape::destroy, "Destroy the tape.")
        .def("__enter__", [](Tape& t) -> Tape& { return t; })
        .def("__exit__",
             [](Tape& t, nb::object, nb::object, nb::object) { t.destroy(); },
             nb::arg("exc_type").none(), nb::arg("exc_value").none(),
             nb::arg("traceback").none());

    // v0.8 A2: the BATCHED GRASP WORLD (BatchedUnifiedWorld) -- the RL-grasp substrate.
    nb::class_<GraspWorld>(m, "GraspWorld")
        .def_static("create", &GraspWorld::create, nb::arg("device"),
                    nb::arg("cup_asset_path"), nb::arg("env_count"),
                    nb::arg("grip_force") = 8.0f, nb::arg("friction_mu") = 0.8f,
                    nb::arg("gravity_z") = -9.81f, nb::arg("dt") = 1.0f / 240.0f,
                    nb::arg("cup_start_z_offset") = 0.0f,
                    nb::arg("reset_jitter_x") = 0.025f,
                    nb::arg("reset_jitter_y") = 0.025f,
                    nb::rv_policy::take_ownership,
                    "Create a batched grasp world: env_count parallel fixed-base "
                    "2-finger grippers each pinching the C7a cup (loaded+cooked from "
                    "cup_asset_path) by friction alone (NO table). grip_force is the "
                    "constant per-finger grip torque the template seeds (overridden "
                    "per-step by set_actions); friction_mu is the contact friction; "
                    "gravity_z/dt are the integrator params. cup_start_z_offset (A3, "
                    "default 0 == the byte-identical validated scene) raises the cup's "
                    "INITIAL Z above the fixed fingertip catch plane (z=0.20) so the cup "
                    "descends into OPEN fingers -> the discriminative timing IC. "
                    "reset_jitter_x / reset_jitter_y (A5a, default 0.025 == the legacy "
                    "isotropic +/-2.5 cm jitter, byte-identical) are the per-axis cup "
                    "RESET-jitter half-boxes ResetEnvs draws -- set Y small/0 so the "
                    "X-only gripper can catch the jittered cups (the un-actuated Y jitter "
                    "is unsaveable). Built via the SAME validated scene factory the 21 "
                    "BatchedUnifiedWorld gates exercise (defaults keep them byte-identical).")
        .def("step", &GraspWorld::step,
             "Advance EVERY env one fixed step (applies the last set_actions, or the "
             "template grip until set_actions is first called).")
        .def("set_actions", &GraspWorld::set_actions, nb::arg("actions"),
             "Set the per-env per-finger drive torque from a HOST contiguous float32 "
             "array of length env_count*action_dim (env-major). The next step() "
             "applies it (replacing the constant template grip).")
        .def("step_with_actions", &GraspWorld::step_with_actions, nb::arg("actions"),
             "set_actions(actions) then step() -- the RL per-step driver.")
        .def("export_obs", &GraspWorld::export_obs,
             "Export the batched per-step obs as a dict of HOST numpy arrays "
             "(env-major): q (env,action_dim), qdot (env,action_dim), "
             "fingertip_world_pos (env,3*num_fingertips), finger_normal_impulse "
             "(env,num_fingertips). The engine does ONE bulk device->host download "
             "per buffer (no per-env sync loop).")
        .def("read_cup", &GraspWorld::read_cup,
             "Read cup pose/vel + per-env contact signals as a dict of HOST numpy "
             "arrays (env-major): cup_pose (env,7) [px,py,pz,qw,qx,qy,qz], cup_vel "
             "(env,6) [vx,vy,vz,wx,wy,wz], cup_z (env,), cup_vz (env,), "
             "finger_contacts (env,) uint32 (# fingertip<->cup contacts per env).")
        .def("reset_envs", &GraspWorld::reset_envs, nb::arg("env_ids"),
             nb::arg("seed"),
             "Reset the listed envs to a seed-randomized cup pose + nominal open "
             "gripper. env_ids: a 1-D int array (numpy / torch / list) in "
             "[0,env_count). Deterministic given seed (a re-run with the same seed "
             "produces identical states).")
        .def("destroy", &GraspWorld::destroy, "Destroy the grasp world.")
        .def_prop_ro("env_count", &GraspWorld::env_count)
        .def_prop_ro("action_dim", &GraspWorld::action_dim,
                     "Per-env per-finger DOF count (== gripper joint DOFs). Actions "
                     "are (env_count, action_dim) env-major.")
        .def_prop_ro("num_fingertips", &GraspWorld::num_fingertips)
        .def_prop_ro("bodies_per_env", &GraspWorld::bodies_per_env)
        .def("__enter__", [](GraspWorld& w) -> GraspWorld& { return w; })
        .def("__exit__",
             [](GraspWorld& w, nb::object, nb::object, nb::object) { w.destroy(); },
             nb::arg("exc_type").none(), nb::arg("exc_value").none(),
             nb::arg("traceback").none());

    m.def("sync", []() {
        int rc = cudaDeviceSynchronize();
        if (rc != 0) {
            throw std::runtime_error("cudaDeviceSynchronize failed (" +
                                     std::to_string(rc) + ")");
        }
    }, "Block until all queued CUDA work completes (test/debug helper).");
}
