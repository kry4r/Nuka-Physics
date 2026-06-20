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
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nuka/nuka.h"
#include "nuka/nuka_diffsim.h"
#include "nuka/nuka_noise.h"
#include "nuka/nuka_recorder.h"  // M8 T5: the offscreen recorder C ABI (PPM/mp4)
#include "nuka/nuka_scene.h"  // M9 T4: the GENERIC scene-authoring C ABI

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
    // null => the legacy orchestration runs on stream 0 (the NULL/default
    // stream; see src/c_abi/device.cpp). The desc.cuda_stream field is accepted
    // by the ABI but BUF-14 routes the legacy paths through stream 0, which
    // serializes with the phi-v2 backend's blocking `main` stream via NULL-stream
    // implicit sync -- so torch ops + physics still share ordering with no
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
                                    uint32_t osc_task_link,
                                    float terrain_step_height,
                                    float terrain_step_width,
                                    float terrain_platform_width,
                                    float terrain_grid_width,
                                    float terrain_grid_height_max,
                                    uint32_t instance_count,
                                    float instance_spacing,
                                    uint32_t contact_family,
                                    uint32_t heightfield_terrain_type,
                                    uint32_t heightfield_nrow,
                                    uint32_t heightfield_ncol,
                                    float heightfield_cell,
                                    const std::string& heightfield_image_path,
                                    float heightfield_radius_x,
                                    float heightfield_radius_y,
                                    float heightfield_elevation_z,
                                    float heightfield_base_z,
                                    uint32_t curric_levels,
                                    uint32_t curric_types,
                                    float terrain_feature_cell) {
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
        // Go2-on-stairs Phase 2a: procedural-terrain cook config (model-level, NOT
        // per env). All-zero default (the kwargs default to 0.0) == flat terrain ==
        // byte-identical to a world created without terrain. The per-env terrain
        // TYPE/DIFFICULTY are seeded (0 / 1.0) and written via
        // buffer_view(Field.ENV_TERRAIN_TYPE / ENV_TERRAIN_DIFFICULTY).
        desc.terrain_step_height = terrain_step_height;
        desc.terrain_step_width = terrain_step_width;
        desc.terrain_platform_width = terrain_platform_width;
        desc.terrain_grid_width = terrain_grid_width;
        desc.terrain_grid_height_max = terrain_grid_height_max;
        // M10 co-residence: compose K replicas of the scene before cook -> ONE env
        // holds K co-resident articulations that physically collide (dog-dog +
        // body-vs-terrain). Default 1 (single instance) == byte-identical legacy.
        desc.instance_count = instance_count;
        desc.instance_spacing = instance_spacing;
        // ONE-GENERAL-SOLVER landing (L1): 0 (default) = legacy FusedFoot
        // (byte-identical); 1 = general PairDriven path + a baked static heightfield
        // collidable (heightfield_terrain_type/nrow/ncol/cell; 0s => engine
        // defaults). Used by the L1 acceptance rollout (trained policy on the general
        // path) before FusedFoot is deleted.
        desc.contact_family = contact_family;
        desc.heightfield_terrain_type = heightfield_terrain_type;
        desc.heightfield_nrow = heightfield_nrow;
        desc.heightfield_ncol = heightfield_ncol;
        desc.heightfield_cell = heightfield_cell;
        // Terrain SOURCE: a non-empty image path loads a grayscale height map
        // (placed by radius/elevation/base); empty => the parametric generator.
        desc.heightfield_image_path =
            heightfield_image_path.empty() ? nullptr : heightfield_image_path.c_str();
        desc.heightfield_radius_x = heightfield_radius_x;
        desc.heightfield_radius_y = heightfield_radius_y;
        desc.heightfield_elevation_z = heightfield_elevation_z;
        desc.heightfield_base_z = heightfield_base_z;
        // Per-env curriculum tile grid (heightfield_terrain_type == 5). Zero-init
        // default (curric_levels == 0) leaves every other terrain type unchanged.
        desc.curric_levels = curric_levels;
        desc.curric_types = curric_types;
        desc.terrain_feature_cell = terrain_feature_cell;
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

    // Batched terrain height over the world's COOKED grid (the ONE source obs and
    // physics share, at full relief). xs/ys are CPU float32 (length n); returns a
    // (n,) numpy float32 array of absolute surface z -- the exact grid physics uses.
    nb::object sample_terrain_height(
        nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> xs,
        nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> ys) {
        const size_t n = xs.shape(0);
        if (ys.shape(0) != n) {
            throw std::runtime_error("sample_terrain_height: xs/ys length mismatch");
        }
        float* out = new float[n == 0 ? 1 : n];
        nuka_result_t rc = nuka_world_sample_terrain_height_batch(
            h_, static_cast<uint32_t>(n), xs.data(), ys.data(), out);
        if (rc != NUKA_RESULT_OK) {
            delete[] out;
            throw std::runtime_error(
                std::string("sample_terrain_height failed: ") +
                nuka_result_message(rc));
        }
        nb::capsule owner(out, [](void* p) noexcept {
            delete[] static_cast<float*>(p);
        });
        size_t shape[1] = {n};
        return nb::cast(nb::ndarray<nb::numpy, float>(out, 1, shape, owner));
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

    // Device-resident batched camera sensor: S cameras per env into a single
    // (E,S,H,W,ch) device tensor (each attach at one size appends a camera).
    // mount_frame 0=Link 1=Body 2=Base; offset=pos3+quat4.
    void attach_camera_sensor(uint32_t mount_frame, uint32_t mount_index,
                              const std::array<float, 7>& local_offset,
                              float vfov_deg, uint32_t width, uint32_t height) {
        check(nuka_world_attach_camera_sensor(
                  h_, static_cast<nuka_sensor_mount_t>(mount_frame), mount_index,
                  local_offset.data(), vfov_deg, width, height),
              "nuka_world_attach_camera_sensor");
        sensor_width_ = width;
        sensor_height_ = height;
        uint32_t s = 1u;
        nuka_world_get_sensor_dims(h_, nullptr, &s, nullptr, nullptr, nullptr);
        sensor_count_ = s;
    }

    // (env_count, sensors_per_env, height, width, channels) of the attached block.
    std::array<uint32_t, 5> sensor_dims() const {
        uint32_t e = 0u, s = 0u, hh = 0u, ww = 0u, ch = 0u;
        check(nuka_world_get_sensor_dims(h_, &e, &s, &hh, &ww, &ch),
              "nuka_world_get_sensor_dims");
        return {e, s, hh, ww, ch};
    }

    // Device-resident batched lidar: S lidars per env, each an (az,el) ray fan, into
    // a single (E,S,az,el) device RANGE tensor on the SAME RT TLAS the cameras use.
    // mount_frame 0=Link 1=Body 2=Base; offset=pos3+quat4; angles in radians.
    void attach_lidar_sensor(uint32_t mount_frame, uint32_t mount_index,
                             const std::array<float, 7>& local_offset,
                             uint32_t az_count, uint32_t el_count, float az_min,
                             float az_max, float el_min, float el_max,
                             float min_range, float max_range) {
        check(nuka_world_attach_lidar_sensor(
                  h_, static_cast<nuka_sensor_mount_t>(mount_frame), mount_index,
                  local_offset.data(), az_count, el_count, az_min, az_max, el_min,
                  el_max, min_range, max_range),
              "nuka_world_attach_lidar_sensor");
        uint32_t e = 0u, s = 0u, az = 0u, el = 0u;
        nuka_world_get_lidar_dims(h_, &e, &s, &az, &el);
        lidar_count_ = s;
        lidar_az_ = az;
        lidar_el_ = el;
    }

    // (env_count, sensors_per_env, az_count, el_count) of the attached lidar fan.
    std::array<uint32_t, 4> lidar_dims() const {
        uint32_t e = 0u, s = 0u, az = 0u, el = 0u;
        check(nuka_world_get_lidar_dims(h_, &e, &s, &az, &el),
              "nuka_world_get_lidar_dims");
        return {e, s, az, el};
    }

    void render_sensors() {
        check(nuka_world_render_sensors(h_), "nuka_world_render_sensors");
    }

    nuka_buffer_view_t get_sensor_view(nuka_sensor_channel_t channel) const {
        nuka_buffer_view_t view{};
        check(nuka_world_get_sensor_view(h_, channel, &view),
              "nuka_world_get_sensor_view");
        if (view.device_ptr == nullptr) {
            throw std::runtime_error(
                "get_sensor_view: engine returned null device_ptr");
        }
        return view;
    }

    uint32_t sensor_width() const { return sensor_width_; }
    uint32_t sensor_height() const { return sensor_height_; }
    uint32_t sensor_count() const { return sensor_count_; }
    uint32_t lidar_count() const { return lidar_count_; }
    uint32_t lidar_az() const { return lidar_az_; }
    uint32_t lidar_el() const { return lidar_el_; }

    // Per-env render domain randomization: give each env its OWN appearance
    // (material color/roughness/metallic, light dir/intensity/color, ambient). The
    // trace always reads BY ENV; enabled=0 -> base replicas (cross-env identical).
    // Same seed -> same bytes. Requires a camera attached.
    void set_render_dr(float color_jitter, float roughness_jitter,
                       float metallic_jitter, float light_dir_jitter,
                       float light_intensity_jitter, float light_color_jitter,
                       float ambient_intensity_jitter, uint64_t seed, int enabled) {
        nuka_render_dr_desc_t desc{};
        desc.color_jitter = color_jitter;
        desc.roughness_jitter = roughness_jitter;
        desc.metallic_jitter = metallic_jitter;
        desc.light_dir_jitter = light_dir_jitter;
        desc.light_intensity_jitter = light_intensity_jitter;
        desc.light_color_jitter = light_color_jitter;
        desc.ambient_intensity_jitter = ambient_intensity_jitter;
        desc.seed = seed;
        desc.enabled = enabled;
        check(nuka_world_set_render_randomization(h_, &desc),
              "nuka_world_set_render_randomization");
    }

    // Opt-in sensor RGB shading fidelity: lift the batched RGB to the beauty look
    // (MSAA spp + soft shadow + AO + GI + tonemap) for a vision policy. Default ==
    // cheap shade (a byte no-op). Deterministic + seeded. Requires a camera attached.
    void set_sensor_fidelity(uint32_t spp, uint32_t shadow_samples,
                             float sun_angular_radius, bool ao_enabled,
                             uint32_t ao_samples, float ao_radius, bool gi_enabled,
                             bool tonemap_enabled, float sky_intensity,
                             float fog_density, uint64_t seed) {
        nuka_sensor_fidelity_desc_t desc{};
        desc.spp = spp;
        desc.shadow_samples = shadow_samples;
        desc.sun_angular_radius = sun_angular_radius;
        desc.ao_enabled = ao_enabled ? 1 : 0;
        desc.ao_samples = ao_samples;
        desc.ao_radius = ao_radius;
        desc.gi_enabled = gi_enabled ? 1 : 0;
        desc.tonemap_enabled = tonemap_enabled ? 1 : 0;
        desc.sky_intensity = sky_intensity;
        desc.fog_density = fog_density;
        desc.seed = seed;
        check(nuka_world_set_sensor_fidelity(h_, &desc),
              "nuka_world_set_sensor_fidelity");
    }

    // Camera lens model: radial (Brown-Conrady) distortion + a clipped depth range
    // on every attached camera. Default (distortion off, wide-open clip) is a byte
    // no-op. fx/fy/cx/cy are renderer-side, not in the scene schema -> not exposed.
    void set_camera_intrinsics(bool distortion, float k1, float k2, float near_clip,
                               float far_clip) {
        nuka_camera_intrinsics_desc_t desc{};
        desc.distortion = distortion ? 1 : 0;
        desc.k1 = k1;
        desc.k2 = k2;
        desc.near_clip = near_clip;
        desc.far_clip = far_clip;
        check(nuka_world_set_camera_intrinsics(h_, &desc),
              "nuka_world_set_camera_intrinsics");
    }

private:
    World(nuka_world_handle h, uint32_t ec, uint32_t blc, float dt)
        : h_(h), env_count_(ec), base_link_count_(blc), dt_(dt) {}
    nuka_world_handle h_ = nullptr;
    uint32_t env_count_ = 0u;
    uint32_t base_link_count_ = 0u;
    float dt_ = 0.0f;
    // Last attached camera image size + cameras per env (the (E,S,H,W,ch) shaping).
    uint32_t sensor_width_ = 0u;
    uint32_t sensor_height_ = 0u;
    uint32_t sensor_count_ = 1u;
    // Last attached lidar fan (the (E,S,az,el) range shaping).
    uint32_t lidar_count_ = 0u;
    uint32_t lidar_az_ = 0u;
    uint32_t lidar_el_ = 0u;
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

// Shape a batched-sensor AOV plane zero-copy: ch = element_count / (E*S*H*W) (3
// color/normal/albedo, 1 depth). S>1 -> (E,S,H,W,ch); S==1 -> (E,H,W,ch) for the
// one-camera-per-env contract. owner keeps the engine buffer alive.
FloatArray make_sensor_float_view(const nuka_buffer_view_t& view, uint32_t n,
                                  uint32_t s, uint32_t h, uint32_t w,
                                  nb::handle owner) {
    const int dev_id = 0;
    const size_t pixels = static_cast<size_t>(n) * s * h * w;
    if (n > 0u && s > 0u && h > 0u && w > 0u && pixels > 0u &&
        view.element_count % pixels == 0u) {
        const size_t ch = view.element_count / pixels;
        if (s > 1u) {
            size_t shape[5] = {n, s, h, w, ch};
            return FloatArray(view.device_ptr, 5, shape, owner, nullptr,
                              nb::dtype<float>(), nb::device::cuda::value, dev_id);
        }
        size_t shape[4] = {n, h, w, ch};
        return FloatArray(view.device_ptr, 4, shape, owner, nullptr,
                          nb::dtype<float>(), nb::device::cuda::value, dev_id);
    }
    size_t shape[1] = {view.element_count};
    return FloatArray(view.device_ptr, 1, shape, owner, nullptr, nb::dtype<float>(),
                      nb::device::cuda::value, dev_id);
}

// Shape the lidar RANGE tensor zero-copy: S>1 -> (E,S,az,el); S==1 -> (E,az,el) for
// the one-lidar-per-env contract. owner keeps the engine buffer alive.
FloatArray make_lidar_range_view(const nuka_buffer_view_t& view, uint32_t n,
                                 uint32_t s, uint32_t az, uint32_t el,
                                 nb::handle owner) {
    const int dev_id = 0;
    const size_t cells = static_cast<size_t>(n) * s * az * el;
    if (n > 0u && s > 0u && az > 0u && el > 0u && cells > 0u &&
        view.element_count == cells) {
        if (s > 1u) {
            size_t shape[4] = {n, s, az, el};
            return FloatArray(view.device_ptr, 4, shape, owner, nullptr,
                              nb::dtype<float>(), nb::device::cuda::value, dev_id);
        }
        size_t shape[3] = {n, az, el};
        return FloatArray(view.device_ptr, 3, shape, owner, nullptr,
                          nb::dtype<float>(), nb::device::cuda::value, dev_id);
    }
    size_t shape[1] = {view.element_count};
    return FloatArray(view.device_ptr, 1, shape, owner, nullptr, nb::dtype<float>(),
                      nb::device::cuda::value, dev_id);
}

// The uint32 (prim) plane as the same (E,S,H,W,1) / (E,H,W,1) view (one channel).
Uint32Array make_sensor_uint32_view(const nuka_buffer_view_t& view, uint32_t n,
                                    uint32_t s, uint32_t h, uint32_t w,
                                    nb::handle owner) {
    const int dev_id = 0;
    const size_t pixels = static_cast<size_t>(n) * s * h * w;
    if (n > 0u && s > 0u && h > 0u && w > 0u && pixels > 0u &&
        view.element_count % pixels == 0u) {
        const size_t ch = view.element_count / pixels;
        if (s > 1u) {
            size_t shape[5] = {n, s, h, w, ch};
            return Uint32Array(view.device_ptr, 5, shape, owner, nullptr,
                               nb::dtype<uint32_t>(), nb::device::cuda::value, dev_id);
        }
        size_t shape[4] = {n, h, w, ch};
        return Uint32Array(view.device_ptr, 4, shape, owner, nullptr,
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
// Recorder wrapper (RAII) -- M8 T5. Drives the host frame loop over an nk::World
// cooked from `scene_path`, through the offscreen Vulkan raster renderer, and
// captures frames to PPM P6 + muxes an mp4 via ffmpeg. On a Vulkan-less libnuka
// every C entry returns NUKA_RESULT_NOT_SUPPORTED -> create() raises cleanly, so
// the binding still IMPORTS but Recorder is unavailable at runtime (Decision D5).
// ---------------------------------------------------------------------------
class Recorder {
public:
    static Recorder* create(Device* device, const std::string& scene_path,
                            uint32_t width, uint32_t height, uint32_t env_index,
                            float dt) {
        if (device == nullptr) {
            throw std::runtime_error("Recorder.create: device is None");
        }
        nuka_recorder_desc_t desc{};
        desc.scene_path = scene_path.empty() ? nullptr : scene_path.c_str();
        desc.width = width;          // 0 -> 1920 (D6 default)
        desc.height = height;        // 0 -> 1080
        desc.env_index = env_index;  // D4: default 0
        desc.dt = dt;                // <= 0 -> 1/240f
        desc.use_camera_override = 0u;  // use the scene camera / auto-frame
        nuka_recorder_handle h = nullptr;
        check(nuka_recorder_create(device->raw(), &desc, &h),
              "nuka_recorder_create");
        return new Recorder(h);
    }

    void destroy() {
        if (h_ != nullptr) {
            nuka_recorder_destroy(h_);
            h_ = nullptr;
        }
    }
    ~Recorder() { destroy(); }

    // Drive n_frames Frame()s, writing frame_%06d.ppm into out_dir.
    void capture(uint32_t n_frames, const std::string& out_dir) {
        check(nuka_recorder_capture(h_, n_frames, out_dir.c_str()),
              "nuka_recorder_capture");
    }

    // Mux out_dir/frame_%06d.ppm -> out_mp4 at fps. Raises on missing/failing
    // ffmpeg (the C side returns NOT_SUPPORTED / INTERNAL -- never crashes).
    void to_video(const std::string& out_dir, const std::string& out_mp4,
                  uint32_t fps) {
        check(nuka_recorder_to_video(h_, out_dir.c_str(), out_mp4.c_str(), fps),
              "nuka_recorder_to_video");
    }

    uint32_t frame_count() const { return nuka_recorder_frame_count(h_); }

private:
    explicit Recorder(nuka_recorder_handle h) : h_(h) {}
    nuka_recorder_handle h_ = nullptr;
};

// ---------------------------------------------------------------------------
// Scene wrapper (RAII) -- M9 T4. The GENERIC scene-authoring surface over
// nuka_scene.h: load any format (mjcf/urdf/usd/nks) through ONE entry, compose /
// find / set_local / set_physics_material / settle / save. NOT a special
// grasp/union type (owner [[unified-world-no-special-grasp-binding]]): behavior
// comes from the imported scene DATA + a per-scene control script. Nodes are
// addressed by their DERIVED tree-path STRING (stable across edits); quaternions
// are (w,x,y,z). Only `settle` touches a device (CUDA-gated -> RuntimeError on a
// CUDA-less build). Plain C crosses the g++-14/g++-10 boundary, like Recorder.
// ---------------------------------------------------------------------------
class Scene {
public:
    static Scene* load(const std::string& path) {
        nuka_scene_handle h = nullptr;
        check(nuka_scene_load(path.c_str(), &h), "nuka_scene_load");
        return new Scene(h);
    }

    // Graft `addon` into this scene at (pos xyz, quat w,x,y,z), via Compose.
    // Empty pos/quat -> identity component. attach_at is an optional name prefix.
    void compose(Scene* addon, const std::vector<float>& pos,
                 const std::vector<float>& quat, const std::string& attach_at) {
        if (addon == nullptr) {
            throw std::runtime_error("Scene.compose: addon is None");
        }
        if (!pos.empty() && pos.size() != 3) {
            throw std::runtime_error("Scene.compose: pos must be 3 floats (xyz)");
        }
        if (!quat.empty() && quat.size() != 4) {
            throw std::runtime_error(
                "Scene.compose: quat must be 4 floats (w,x,y,z)");
        }
        check(nuka_scene_compose(h_, addon->h_,
                                 pos.empty() ? nullptr : pos.data(),
                                 quat.empty() ? nullptr : quat.data(),
                                 attach_at.empty() ? nullptr : attach_at.c_str()),
              "nuka_scene_compose");
    }

    bool find(const std::string& path) {
        int found = 0;
        check(nuka_scene_find(h_, path.c_str(), &found), "nuka_scene_find");
        return found != 0;
    }

    // Set a node's local transform; empty pos/quat leaves that component as-is.
    void set_local(const std::string& path, const std::vector<float>& pos,
                   const std::vector<float>& quat) {
        if (!pos.empty() && pos.size() != 3) {
            throw std::runtime_error("Scene.set_local: pos must be 3 floats (xyz)");
        }
        if (!quat.empty() && quat.size() != 4) {
            throw std::runtime_error(
                "Scene.set_local: quat must be 4 floats (w,x,y,z)");
        }
        check(nuka_scene_set_local(h_, path.c_str(),
                                   pos.empty() ? nullptr : pos.data(),
                                   quat.empty() ? nullptr : quat.data()),
              "nuka_scene_set_local");
    }

    // Set physics-material params on every collision-shape whose derived path
    // matches `path_regex`. A param < 0 leaves it untouched; static_friction and
    // dynamic_friction must be equal when both >= 0 (the record is isotropic).
    // Returns the number of shapes touched. restitution does NOT persist to .nks.
    uint32_t set_physics_material(const std::string& path_regex,
                                  float static_friction, float dynamic_friction,
                                  float restitution) {
        uint32_t matched = 0u;
        check(nuka_scene_set_physics_material(h_, path_regex.c_str(),
                                              static_friction, dynamic_friction,
                                              restitution, &matched),
              "nuka_scene_set_physics_material");
        return matched;
    }

    // Cook to an nk::World on `device`'s backend, settle `steps` steps, write the
    // settled state back into the scene. Raises on a CUDA-less build (NOT_SUPPORTED).
    void settle(Device* device, uint32_t steps, float dt) {
        if (device == nullptr) {
            throw std::runtime_error("Scene.settle: device is None");
        }
        check(nuka_scene_settle(h_, device->raw(), steps, dt),
              "nuka_scene_settle");
    }

    void save(const std::string& nks_path) {
        check(nuka_scene_save(h_, nks_path.c_str()), "nuka_scene_save");
    }

    void destroy() {
        if (h_ != nullptr) {
            nuka_scene_destroy(h_);
            h_ = nullptr;
        }
    }
    ~Scene() { destroy(); }

private:
    explicit Scene(nuka_scene_handle h) : h_(h) {}
    nuka_scene_handle h_ = nullptr;
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
        // Go2-on-stairs Phase 2a: per-env procedural-terrain controls (writable,
        // one value per env). ENV_TERRAIN_TYPE (uint32, default seed 0=Flat):
        // 0=Flat, 1=PyramidStairs, 2=InvertedPyramid, 3=RandomBoxes. buffer_view
        // shape (env_count, 1), dtype uint32. ENV_TERRAIN_DIFFICULTY (float32,
        // default seed 1.0): scales the terrain's vertical feature height per env
        // for the locomotion curriculum; buffer_view shape (env_count,).
        .value("ENV_TERRAIN_TYPE", NUKA_FIELD_ENV_TERRAIN_TYPE)
        .value("ENV_TERRAIN_DIFFICULTY", NUKA_FIELD_ENV_TERRAIN_DIFFICULTY)
        // T3 unified-actuator feed-forward joint force (per-link f32, default 0):
        // tau += joint_f in every control mode, after actuator saturation. Same
        // layout/slot map as DRIVE_TARGET; writable zero-copy. Default 0 => no-op.
        .value("JOINT_FEEDFORWARD", NUKA_FIELD_JOINT_FEEDFORWARD)
        .export_values();

    // Batched camera-sensor AOV plane for World.get_sensor_view: COLOR/NORMAL/ALBEDO
    // = (N,H,W,3) float32, DEPTH = (N,H,W,1) float32, PRIM = (N,H,W,1) uint32. RANGE
    // is the lidar plane = (N,az,el) float32 (reshape with World.lidar_dims).
    nb::enum_<nuka_sensor_channel_t>(m, "SensorChannel")
        .value("COLOR", NUKA_SENSOR_CHANNEL_COLOR)
        .value("DEPTH", NUKA_SENSOR_CHANNEL_DEPTH)
        .value("NORMAL", NUKA_SENSOR_CHANNEL_NORMAL)
        .value("ALBEDO", NUKA_SENSOR_CHANNEL_ALBEDO)
        .value("PRIM", NUKA_SENSOR_CHANNEL_PRIM)
        .value("RANGE", NUKA_SENSOR_CHANNEL_RANGE)
        .export_values();

    // Which FK frame a camera mounts on for World.attach_camera_sensor.
    nb::enum_<nuka_sensor_mount_t>(m, "SensorMount")
        .value("LINK", NUKA_SENSOR_MOUNT_LINK)
        .value("BODY", NUKA_SENSOR_MOUNT_BODY)
        .value("BASE", NUKA_SENSOR_MOUNT_BASE)
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
                    nb::arg("terrain_step_height") = 0.0f,
                    nb::arg("terrain_step_width") = 0.0f,
                    nb::arg("terrain_platform_width") = 0.0f,
                    nb::arg("terrain_grid_width") = 0.0f,
                    nb::arg("terrain_grid_height_max") = 0.0f,
                    nb::arg("instance_count") = uint32_t{1},
                    nb::arg("instance_spacing") = 1.5f,
                    nb::arg("contact_family") = uint32_t{0},
                    nb::arg("heightfield_terrain_type") = uint32_t{0},
                    nb::arg("heightfield_nrow") = uint32_t{0},
                    nb::arg("heightfield_ncol") = uint32_t{0},
                    nb::arg("heightfield_cell") = 0.0f,
                    nb::arg("heightfield_image_path") = std::string{},
                    nb::arg("heightfield_radius_x") = 0.0f,
                    nb::arg("heightfield_radius_y") = 0.0f,
                    nb::arg("heightfield_elevation_z") = 0.0f,
                    nb::arg("heightfield_base_z") = 0.0f,
                    nb::arg("curric_levels") = uint32_t{0},
                    nb::arg("curric_types") = uint32_t{0},
                    nb::arg("terrain_feature_cell") = 0.0f,
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
                    "end-effector (e.g. a foot/calf local link). "
                    "terrain_* (Go2-on-stairs Phase 2a, all default 0.0 == flat): "
                    "the procedural-terrain cook config (model-level, NOT per env). "
                    "Set terrain_step_height/terrain_step_width/terrain_platform_width "
                    "for PyramidStairs/InvertedPyramid or terrain_grid_width/"
                    "terrain_grid_height_max for RandomBoxes, then write per-env "
                    "Field.ENV_TERRAIN_TYPE (0=Flat,1=PyramidStairs,2=InvertedPyramid,"
                    "3=RandomBoxes) and Field.ENV_TERRAIN_DIFFICULTY (default 1.0, "
                    "scales the vertical feature height) via buffer_view. All-zero "
                    "terrain == byte-identical to a world created without terrain.")
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
        // Device-resident batched camera sensor (the in-the-loop obs surface):
        // attach the camera, render the (N,H,W,ch) tensor, alias a plane zero-copy.
        .def(
            "attach_camera_sensor",
            [](World& w, uint32_t mount_frame, uint32_t mount_index,
               std::array<float, 7> local_offset, float vfov_deg, uint32_t width,
               uint32_t height) {
                w.attach_camera_sensor(mount_frame, mount_index, local_offset,
                                       vfov_deg, width, height);
            },
            nb::arg("mount_frame"), nb::arg("mount_index"), nb::arg("local_offset"),
            nb::arg("vfov_deg"), nb::arg("width"), nb::arg("height"),
            "Attach a camera mounted on mount_frame (0=Link, 1=Body, 2=Base) "
            "link/body/base index mount_index, offset by local_offset (pos3 + quat4: "
            "px,py,pz, qw,qx,qy,qz) in that frame; camera-local axes are -Z forward, "
            "+Y up. vfov_deg is the vertical field of view; width/height size every "
            "env image. Builds the sensor scene from the world's cooked visual "
            "meshes. Re-callable: a call at the SAME width/height APPENDS another "
            "camera (S cameras per env, each its own mount); a different width/height "
            "resets to a fresh single-camera set. Raises NOT_SUPPORTED if the engine "
            "has no RT sensor backend or the scene has no renderable geometry.")
        .def("render_sensors", &World::render_sensors,
             "Render every (env, sensor) camera into the persistent (E,S,H,W,ch) "
             "device AOV tensor IN PLACE on the world's backend stream (no host "
             "download), driven by the live link poses. Call after step() and after "
             "attach_camera_sensor.")
        .def(
            "get_sensor_view",
            [](nb::handle self, nuka_sensor_channel_t channel) -> nb::object {
                World* w = nb::cast<World*>(self);
                nuka_buffer_view_t view = w->get_sensor_view(channel);
                // RANGE is the lidar plane: float32, shaped by the (E,S,az,el) fan.
                if (channel == NUKA_SENSOR_CHANNEL_RANGE) {
                    return nb::cast(make_lidar_range_view(
                        view, w->env_count(), w->lidar_count(), w->lidar_az(),
                        w->lidar_el(), self));
                }
                // PRIM is the ONE uint32 plane (dtype==1); the rest are float32.
                // (E,S,H,W,ch) for S>1, (E,H,W,ch) for S==1; zero-copy DLPack.
                if (view.dtype == 1u) {
                    return nb::cast(make_sensor_uint32_view(
                        view, w->env_count(), w->sensor_count(), w->sensor_height(),
                        w->sensor_width(), self));
                }
                return nb::cast(make_sensor_float_view(
                    view, w->env_count(), w->sensor_count(), w->sensor_height(),
                    w->sensor_width(), self));
            },
            nb::arg("channel"), nb::rv_policy::reference,
            "Return a zero-copy DLPack-capable CUDA ndarray aliasing the batched "
            "sensor tensor for `channel`. Cameras (COLOR/DEPTH/NORMAL/ALBEDO float32, "
            "PRIM uint32) are shaped (env_count, sensors_per_env, height, width, "
            "channels) for >1 camera else (env_count, height, width, channels): "
            "COLOR/NORMAL/ALBEDO ch=3, DEPTH/PRIM ch=1. RANGE (lidar, float32) is "
            "shaped (env_count, sensors_per_env, az_count, el_count) for >1 lidar "
            "else (env_count, az_count, el_count). torch.from_dlpack(...) yields a "
            "CUDA tensor aliasing the engine buffer (no copy). Raises NOT_SUPPORTED "
            "before the first render_sensors().")
        .def("sensor_dims", &World::sensor_dims,
             "The attached sensor block shape as a 5-tuple (env_count, "
             "sensors_per_env, height, width, channels). channels is 3 (the "
             "color/normal/albedo planes; depth/prim are 1). Valid after "
             "attach_camera_sensor; raises NOT_SUPPORTED if no camera is attached.")
        .def_prop_ro("sensor_width", &World::sensor_width,
                     "Width of the last attached camera image (0 if none).")
        .def_prop_ro("sensor_height", &World::sensor_height,
                     "Height of the last attached camera image (0 if none).")
        .def_prop_ro("sensor_count", &World::sensor_count,
                     "Cameras per env (S) in the attached sensor block (1 default).")
        // Device-resident batched lidar: an (az,el) ray fan per env into a single
        // (E,S,az,el) device RANGE tensor on the SAME RT TLAS the cameras use.
        .def(
            "attach_lidar_sensor",
            [](World& w, uint32_t mount_frame, uint32_t mount_index,
               const std::array<float, 7>& local_offset, uint32_t az_count,
               uint32_t el_count, float az_min, float az_max, float el_min,
               float el_max, float min_range, float max_range) {
                w.attach_lidar_sensor(mount_frame, mount_index, local_offset, az_count,
                                      el_count, az_min, az_max, el_min, el_max,
                                      min_range, max_range);
            },
            nb::arg("mount_frame"), nb::arg("mount_index"), nb::arg("local_offset"),
            nb::arg("az_count"), nb::arg("el_count"), nb::arg("az_min"),
            nb::arg("az_max"), nb::arg("el_min"), nb::arg("el_max"),
            nb::arg("min_range") = 0.0f, nb::arg("max_range") = 100.0f,
            "Attach a lidar mounted on mount_frame (0=Link, 1=Body, 2=Base) "
            "link/body/base index mount_index, offset by local_offset (pos3 + quat4: "
            "px,py,pz, qw,qx,qy,qz). The fan sweeps az_count*el_count rays: az in "
            "[az_min,az_max] (radians) about the local +Z axis, el in [el_min,el_max] "
            "toward local +Z; az=el=0 is local +X. Ranges clamp to "
            "[min_range,max_range] (a miss reads max_range). Read RANGE via "
            "get_sensor_view(SensorChannel.RANGE) shaped by lidar_dims. Re-callable: "
            "a lidar attach REPLACES the lidar set. Raises NOT_SUPPORTED with no RT "
            "backend / no geometry; ValueError on a zero fan or bad range.")
        .def("lidar_dims", &World::lidar_dims,
             "The attached lidar fan shape as a 4-tuple (env_count, sensors_per_env, "
             "az_count, el_count). Valid after attach_lidar_sensor; raises "
             "NOT_SUPPORTED if no lidar is attached.")
        // Per-env render domain randomization (the vision-policy / sim2real lever):
        // each env gets its OWN appearance set, deterministic + seeded.
        .def(
            "set_render_dr",
            [](World& w, float color_jitter, float roughness_jitter,
               float metallic_jitter, float light_dir_jitter,
               float light_intensity_jitter, float light_color_jitter,
               float ambient_intensity_jitter, uint64_t seed, bool enabled) {
                w.set_render_dr(color_jitter, roughness_jitter, metallic_jitter,
                                light_dir_jitter, light_intensity_jitter,
                                light_color_jitter, ambient_intensity_jitter, seed,
                                enabled ? 1 : 0);
            },
            nb::arg("color_jitter") = 0.0f, nb::arg("roughness_jitter") = 0.0f,
            nb::arg("metallic_jitter") = 0.0f, nb::arg("light_dir_jitter") = 0.0f,
            nb::arg("light_intensity_jitter") = 0.0f,
            nb::arg("light_color_jitter") = 0.0f,
            nb::arg("ambient_intensity_jitter") = 0.0f, nb::arg("seed") = 0u,
            nb::arg("enabled") = true,
            "Enable per-env render domain randomization: each env gets its OWN "
            "material color/roughness/metallic, light direction/intensity/color, and "
            "ambient intensity, so the (E,S,H,W,ch) sensor tensor shows appearance "
            "variety. Each *_jitter is a symmetric band about the base (additive for "
            "color/roughness/metallic/light_dir, fractional for the intensities); a "
            "zero jitter leaves that axis at base. enabled=False restores base "
            "replicas (cross-env byte-identical). A pure function of (seed, env, "
            "axis) -- the same seed yields the same bytes (RL-reproducible). Requires "
            "a camera attached; safe to re-call per reset.")
        // Opt-in sensor RGB shading fidelity (the sim2real lever for a vision
        // policy): lift the batched RGB to the single-camera beauty look.
        .def(
            "set_sensor_fidelity",
            [](World& w, uint32_t spp, uint32_t shadow_samples,
               float sun_angular_radius, bool ao_enabled, uint32_t ao_samples,
               float ao_radius, bool gi_enabled, bool tonemap_enabled,
               float sky_intensity, float fog_density, uint64_t seed) {
                w.set_sensor_fidelity(spp, shadow_samples, sun_angular_radius,
                                      ao_enabled, ao_samples, ao_radius, gi_enabled,
                                      tonemap_enabled, sky_intensity, fog_density,
                                      seed);
            },
            nb::arg("spp") = 4u, nb::arg("shadow_samples") = 4u,
            nb::arg("sun_angular_radius") = 0.04f, nb::arg("ao_enabled") = true,
            nb::arg("ao_samples") = 3u, nb::arg("ao_radius") = 0.6f,
            nb::arg("gi_enabled") = true, nb::arg("tonemap_enabled") = true,
            nb::arg("sky_intensity") = 0.0f, nb::arg("fog_density") = 0.0f,
            nb::arg("seed") = 0x9e3779b9u,
            "Enable opt-in sensor RGB shading fidelity: the batched RGB shades flat "
            "(1 spp, 1 hard shadow) by default; this lifts it to the single-camera "
            "beauty look -- MSAA (spp jittered samples), soft sun shadow "
            "(shadow_samples across sun_angular_radius), ambient occlusion + "
            "one-bounce GI (ao_*/gi_enabled), and an ACES-ish tonemap. The default "
            "World shade (no call, or spp<=1 with everything off) is the cheap shade "
            "(the AOV bytes are unchanged). depth/normal/albedo/prim always come from "
            "the center ray. Deterministic + seeded: the samples derive from "
            "Philox(seed, env, sensor, pixel, sample), so the same seed yields the "
            "same bytes. spp/shadow_samples/ao_samples are capped at 256. Requires a "
            "camera attached; safe to re-call per reset. Textures are not yet "
            "applied (a follow-on); this is lighting/shading + anti-aliasing.")
        // Camera lens model: radial distortion + a clipped depth range on every
        // attached camera. Default (distortion off, wide-open clip) is a byte no-op.
        .def(
            "set_camera_intrinsics",
            [](World& w, bool distortion, float k1, float k2, float near_clip,
               float far_clip) {
                w.set_camera_intrinsics(distortion, k1, k2, near_clip, far_clip);
            },
            nb::arg("distortion") = false, nb::arg("k1") = 0.0f, nb::arg("k2") = 0.0f,
            nb::arg("near_clip") = 0.0f, nb::arg("far_clip") = 0.0f,
            "Apply the camera lens model to every attached camera: radial "
            "(Brown-Conrady) distortion (distortion=True enables d=1+k1*r2+k2*r2*r2 "
            "on each ray's normalized sensor coord) and a depth clip (a hit outside "
            "[near_clip, far_clip] reads as a miss in the depth AOV). A non-positive "
            "near/far reads as the wide-open default (0.01 / 1000), so the default "
            "call (distortion off, clip 0/0) is a byte no-op. The principal point and "
            "per-axis focal are renderer-side knobs not carried by the scene schema, "
            "so they are not exposed here. Requires a camera attached; rebuilds the "
            "sensor scene (per-env DR + shading fidelity are preserved).")
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
             "non-articulated world raises NOT_SUPPORTED.")
        .def("sample_terrain_height", &World::sample_terrain_height,
             nb::arg("xs"), nb::arg("ys"),
             "Batched terrain height over the world's COOKED heightfield grid (the "
             "ONE full-relief source obs/spawn/render and physics share, sampled "
             "bilinearly). xs/ys: (n,) float32 CPU world columns. Returns (n,) numpy "
             "float32 absolute surface z. No cooked heightfield -> all 0.");

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

    // M8 T5: the offscreen Recorder. On a Vulkan-less libnuka create() raises
    // (the C entries return NUKA_RESULT_NOT_SUPPORTED) -- the binding still
    // imports; only the runtime call fails (Decision D5).
    nb::class_<Recorder>(m, "Recorder")
        .def_static("create", &Recorder::create, nb::arg("device"),
                    nb::arg("scene_path"), nb::arg("width") = uint32_t{0},
                    nb::arg("height") = uint32_t{0},
                    nb::arg("env_index") = uint32_t{0}, nb::arg("dt") = 0.0f,
                    nb::rv_policy::take_ownership,
                    "Create an offscreen recorder over an nk::World cooked from "
                    "scene_path (an authored .nks). It builds the RenderWorld from "
                    "the SAME cook's Registry + SceneMap and stands up the Vulkan "
                    "forward raster renderer. width/height=0 -> 1920x1080 (D6); "
                    "env_index selects the rendered env (D4, default 0); dt<=0 -> "
                    "1/240. Raises if libnuka was built without Vulkan, if the "
                    "scene/assets are missing, or if no Vulkan graphics device is "
                    "present.")
        .def("capture", &Recorder::capture, nb::arg("n_frames"),
             nb::arg("out_dir"),
             "Drive n_frames frames (step + publish + render) and write each "
             "frame to out_dir/frame_%06d.ppm (binary PPM P6). The frame index "
             "continues across capture() calls so a multi-segment rollout muxes "
             "contiguously. out_dir is created if absent.")
        .def("to_video", &Recorder::to_video, nb::arg("out_dir"),
             nb::arg("out_mp4"), nb::arg("fps") = uint32_t{30},
             "Mux out_dir/frame_%06d.ppm -> out_mp4 via ffmpeg at fps (default "
             "30; libx264/yuv420p). Raises cleanly if ffmpeg is absent or the "
             "mux fails (the C side never crashes).")
        .def("destroy", &Recorder::destroy, "Destroy the recorder.")
        .def_prop_ro("frame_count", &Recorder::frame_count,
                     "Total frames captured so far on this recorder.")
        .def("__enter__", [](Recorder& r) -> Recorder& { return r; })
        .def("__exit__",
             [](Recorder& r, nb::object, nb::object, nb::object) { r.destroy(); },
             nb::arg("exc_type").none(), nb::arg("exc_value").none(),
             nb::arg("traceback").none());

    // M9 T4: the GENERIC scene-authoring surface (nuka_scene.h). ONE load entry
    // for any format; uniform compose/find/set_local/set_physics_material/settle/
    // save. NOT a special grasp/union type.
    nb::class_<Scene>(m, "Scene")
        .def_static("load", &Scene::load, nb::arg("path"),
                    nb::rv_policy::take_ownership,
                    "Load a scene from `path`, dispatching by file extension "
                    "(.nks / .xml / .mjcf / .urdf / .usd / .usda). .nks resolves "
                    "its `imports` internally. Raises on a missing file, an "
                    "unknown extension, or malformed input.")
        .def("compose", &Scene::compose, nb::arg("addon"),
             nb::arg("pos") = std::vector<float>{},
             nb::arg("quat") = std::vector<float>{},
             nb::arg("attach_at") = std::string(),
             "Graft `addon` (a Scene) into this scene at the placement (pos "
             "[x,y,z] + quat [w,x,y,z], in this scene's frame; empty -> identity "
             "component). attach_at is an optional name prefix for the addon's "
             "appended records (disambiguates duplicate names). Mutates this "
             "scene in place; addon is unchanged.")
        .def("find", &Scene::find, nb::arg("path"),
             "Return True if a node at the derived tree path `path` exists "
             "(e.g. 'cup/body', 'h1/right_hand_link'; a leading 'Scene' segment "
             "is tolerated).")
        .def("set_local", &Scene::set_local, nb::arg("path"),
             nb::arg("pos") = std::vector<float>{},
             nb::arg("quat") = std::vector<float>{},
             "Set the LOCAL transform of the node at `path` (its body/shape "
             "record). pos [x,y,z] + quat [w,x,y,z]; an empty list leaves that "
             "component unchanged. Raises if no node matches `path`.")
        .def("set_physics_material", &Scene::set_physics_material,
             nb::arg("path_regex"), nb::arg("static_friction") = -1.0f,
             nb::arg("dynamic_friction") = -1.0f, nb::arg("restitution") = -1.0f,
             "Set physics-material params on every COLLISION-SHAPE node whose "
             "derived path matches the regex `path_regex` (full-match). A param "
             "< 0 leaves it untouched. The record carries a SINGLE isotropic "
             "friction, so static_friction and dynamic_friction must be EQUAL "
             "when both are given. restitution is applied to the resolved "
             "in-memory PhysicsMaterial but is NOT persisted by save() (the .nks "
             "schema has no restitution field). Returns the number of shapes "
             "touched.")
        .def("settle", &Scene::settle, nb::arg("device"),
             nb::arg("steps") = uint32_t{200}, nb::arg("dt") = 0.0f,
             "Cook the scene to an nk::World on `device`'s backend, run "
             "cook::Settle for `steps` deterministic device steps (dt<=0 -> "
             "1/240), and write the settled state back into the scene so save() "
             "persists it (the articulation IC -> SceneInitialState; settled "
             "movable-body poses -> their body records). Raises NOT_SUPPORTED on "
             "a CUDA-less build (no phi v2 backend) or if the cooked world is "
             "not Ready.")
        .def("save", &Scene::save, nb::arg("nks_path"),
             "Save the scene to `nks_path` (+ a sibling <base>.nka).")
        .def("destroy", &Scene::destroy, "Destroy the scene handle.")
        .def("__enter__", [](Scene& s) -> Scene& { return s; })
        .def("__exit__",
             [](Scene& s, nb::object, nb::object, nb::object) { s.destroy(); },
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
