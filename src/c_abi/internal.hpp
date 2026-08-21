#pragma once

#include "nuka/nuka.h"

#include "core/diagnostics/invariants.hpp"
#include "sensor/noise/noise_config.hpp"
#include "rt/render_dr.hpp"
#include "rt/sensor_fidelity.hpp"
#include "phi/backend.hpp"
// M9 T5/T6: the C-ABI world is now ONE generic nk::World (Scene->CookToModel->
// nk::World). The diffsim + noise + set_link_mass paths still consume the legacy
// ArticulationHostState/ArticulationDeviceState VIEW types as a kernel-parameter
// surface (the M3b seam MakeArticulationDeviceStateFromViews builds that view
// over the nk arena fields — algorithm/kernels untouched, proven bit-exact by
// AbaReverse.NkArenaSeamForwardReverseBitExact), so articulation_state.hpp stays.
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/control_mode.hpp"
#include "runtime/step_options.hpp"
#include "scene/cook/media_render_surface.hpp"
#include "scene/terrain/heightfield.hpp"

#include <memory>
#include <string>
#include <vector>

namespace nuka::scene {
class SceneIR;     // scene/scene_ir.hpp -- fwd-declared (held by unique_ptr below).
struct SensorDesc; // scene/scene_ir.hpp -- accumulated mount list (vector member).
}  // namespace nuka::scene

namespace nuka::nk {
class World;  // nk/pipeline/world.hpp -- fwd-declared (held by unique_ptr below;
              // complete in world.cpp where the World ctor runs).
class Model;  // nk/model/model.hpp -- a cooked Model (passed by &&/& to the
              // shared world-create helpers below; complete at each call site).
}  // namespace nuka::nk

namespace nuka::math { struct Vec3; }  // math/vec3.hpp (gravity vector param).

namespace nuka::render {
class SensorBackendI;     // render/sensor_backend.hpp -- the batched sensor facet.
struct SensorSceneHandle; // backend-owned built scene; freed via FreeSensorScene.
struct StudioScene;       // render/studio_beauty.hpp -- the shared studio render scene.
class StudioRtRenderer;   // render/studio_beauty.hpp -- the offline RT beauty tracer.
}  // namespace nuka::render

namespace nuka::c_abi {

struct DeviceRecord {
    // --- CUDA device ordinal (BUF-14) ---------------------------------------
    // The validated CUDA device this handle was created on. Read by the legacy
    // diffsim/noise orchestration (ScopedDeviceGuard) + recorder.cpp's
    // SelectDeviceByOrdinal. The legacy orchestration that used to own a created
    // stream now runs on STREAM 0 (the NULL/default stream): single-stream
    // ordering + NULL-stream implicit serialization preserve the exact
    // happens-before the owned stream gave, and the v2 backward op (on
    // backend->main) coordinates with stream 0 via that same implicit sync.
    int device_id = 0;

    // --- phi v2 owned device/backend (M9 T2) -------------------------------
    // An OWNED phi v2 Backend (and the registry-owned Device it was init'd on),
    // acquired in nuka_device_create. These feed the nk::World ctor + cook::Settle
    // from the C-ABI. M9 COMPLETE: T2 (phi_device/backend acquisition here), T4
    // (scene.cpp settle calls cook::Settle on this backend) and T5 (world.cpp
    // build nk::World(model, n, phi_device, backend, cfg)) are all wired as of the
    // M9 push.
    // LIFETIME (mirrors RecorderRecord, commit fcae2a6): the Backend is OWNED
    // and freed in the destructor; nk::World only BORROWS it
    // (World::~World -> BackendPlanFree(backend_), never frees the backend), so
    // any future borrowing member must be torn down BEFORE the BackendFree. The
    // Device is registry-OWNED (RegistryEntryGetDevice / InitBestDevice) and is
    // NOT freed here. Null when no phi v2 device/backend is available (e.g. no
    // CUDA backend registered); the legacy paths still function in that case.
    phi::Device* phi_device = nullptr;   // registry-owned; do NOT free
    phi::Backend* backend = nullptr;     // OWNED; BackendFree'd in dtor

    ~DeviceRecord() {
        // The nk::World that BORROWS this backend lives on the WorldRecord, which
        // the API contract destroys before its DeviceRecord (nuka_world_destroy
        // before nuka_device_destroy), so the backend has no live borrower here.
        if (backend != nullptr) {
            phi::ResetActiveBackend(backend);  // drop the global selection first.
            phi::BackendFree(backend);
            backend = nullptr;
        }
        // phi_device is registry-owned; not freed.
    }
};

// The world's batched sensors: the backend + the backend-owned scene handle + the
// image size + the accumulated mount list. Cameras AND lidars ride ONE handle/TLAS;
// `sensors` holds both (the backend splits by type). Each camera attach appends one
// camera (S cameras per env); a lidar attach replaces the lidar set (one (E,S,az,el)
// fan). The out-of-line dtor frees the handle through the backend.
struct SensorAttachment {
    std::unique_ptr<nuka::render::SensorBackendI> backend;
    nuka::render::SensorSceneHandle* handle = nullptr;
    std::vector<nuka::scene::SensorDesc> sensors;  // appended per attach (cameras + lidars).
    uint32_t width = 0u;
    uint32_t height = 0u;
    bool rendered = false;  // a device AOV/range tensor exists only after the first render.
    // Lidar fan dims (0 until a lidar is attached). The (E,S,az,el) RANGE tensor's
    // az/el extent; lidars_per_env*env_count*az*el == the range cell count.
    uint32_t lidar_az = 0u;
    uint32_t lidar_el = 0u;
    uint32_t lidars_per_env = 0u;
    // Per-env appearance DR (disabled -> base replicas). Retained so a re-attach
    // (which rebuilds the sensor scene) re-applies it onto the fresh tables.
    nuka::rt::RenderDrConfig render_dr;
    // Opt-in shading fidelity (default -> cheap shade). Retained so a re-attach
    // re-applies it onto the rebuilt sensor scene.
    nuka::rt::SensorFidelityConfig fidelity;
    // Camera AOV selection (0 => legacy all-AOV profile). Retained across a
    // camera/lidar re-attach so an observation pipeline does not silently fall
    // back to the expensive shaded path.
    uint32_t aov_mask = 0u;
    SensorAttachment();     // out-of-line (the SensorDesc vector member is incomplete here).
    ~SensorAttachment();    // defined in c_abi/sensor.cpp (FreeSensorScene then backend).
    SensorAttachment(SensorAttachment&&) = delete;
    SensorAttachment& operator=(SensorAttachment&&) = delete;
    SensorAttachment(const SensorAttachment&) = delete;
    SensorAttachment& operator=(const SensorAttachment&) = delete;
};

// The world's offline beauty render bridge (lazily built by the first render_beauty
// call): the shared studio render scene built once from the world's cooked scene +
// the live RT beauty tracer. Held by unique_ptrs to fwd-declared render types so this
// header stays free of the render headers; the out-of-line dtor (render_beauty.cpp)
// destroys them where they are complete. Rebuilt if the image size changes.
struct BeautyRender {
    std::unique_ptr<nuka::render::StudioScene> scene;
    std::unique_ptr<nuka::render::StudioRtRenderer> renderer;
    BeautyRender();     // out-of-line in render_beauty.cpp (StudioScene incomplete here).
    ~BeautyRender();    // out-of-line in render_beauty.cpp.
    BeautyRender(BeautyRender&&) = delete;
    BeautyRender& operator=(BeautyRender&&) = delete;
    BeautyRender(const BeautyRender&) = delete;
    BeautyRender& operator=(const BeautyRender&) = delete;
};

struct WorldRecord {
    // M9 T5/T6: the ONE generic live sim — Scene->CookToModel->nk::World. Owns
    // the device-resident Model+Data+Pipeline. BORROWS the DeviceRecord's phi v2
    // backend (World::~World -> BackendPlanFree, never frees the backend), so it
    // MUST be torn down BEFORE the DeviceRecord frees the backend. The two are
    // separate handles/tables; by the public API contract nuka_world_destroy is
    // called before nuka_device_destroy. Held by unique_ptr behind a forward
    // declaration; the ctor/dtor are out-of-line in world.cpp (where nk::World is
    // complete) so the other C-ABI TUs that touch WorldTable need not see it.
    std::unique_ptr<nuka::nk::World> world;

    // The FINAL composed scene this world cooked (co-residence replicas included),
    // RETAINED so the sensor attach builds the per-env visual binding from one cook.
    std::unique_ptr<nuka::scene::SceneIR> scene;

    // Directory of the scene file this world loaded from (empty for in-memory built
    // scenes). Relative material-map / hdri paths resolve against it at render time.
    std::string scene_dir;

    // The batched camera sensor (lazily built by attach). Declared AFTER `world` so
    // it is destroyed BEFORE the World/backend drop. Null until a sensor is attached.
    std::unique_ptr<SensorAttachment> sensor;

    // The offline beauty render bridge (lazily built by the first render_beauty call).
    // Declared AFTER `world` so it (and the RT backend/scene it owns) tears down BEFORE
    // the World/backend drop. Null until render_beauty runs.
    std::unique_ptr<BeautyRender> beauty;

    // One render surface per cooked particle medium (cloth lattice faces, soft-tet
    // boundary faces), retained at world-create so a live beauty render rebuilds each
    // deforming surface from the live particle positions. Empty when no surface media.
    std::vector<nuka::scene::cook::MediaRenderSurface> particle_surfaces;

    DeviceRecord* device = nullptr;

    // dt / gravity scalars (formerly the legacy WorldStepOptions). Read by the
    // diffsim RolloutParams + the InvariantWorldView; written by set_gravity_z /
    // the DR gravity poke. enable_contacts/iters fields are inert here (the nk
    // Pipeline owns its op schedule) but kept so the InvariantWorldView ctor and
    // the diffsim/noise paths compile unchanged.
    runtime::WorldStepOptions step_options;

    // --- Transitional diffsim/noise host mirror (M9 named gap; deleted at T7) -
    // The diffsim backward family + set_link_mass + per-episode DR still consume
    // the cooked per-link inertia params (mass / diagonal_inertia / inertial_frame
    // / joint_armature) as HOST data to build dI/dmass + rebuild a link's spatial
    // inertia in place. These are sourced from the SAME cook the legacy create
    // used (CookArticulations -> BuildArticulationHostState), so BuildMassParams /
    // ResolveLinkInertiaParams / CaptureNominalBaseline stay BYTE-IDENTICAL. The
    // device writes (set_link_mass / DR) go to the nk arena's LinkInertia field
    // (FieldPtr), NOT a legacy device buffer. Retired when the backward op-ifies
    // at T7 (the dI/dmass slope then rides the Model). The articulations[] topo +
    // link_inertia + joint_armature are the only members consumed.
    runtime::articulation::ArticulationHostState articulation_host;

    uint32_t simulated_step_count = 0u;

    // --- Sparse solver backend selection (v0.7 p01) -----------------------
    // Which SparseLinearSolver backend the diff-sim / general solver path uses.
    // Default 0 == NUKA_SOLVER_BACKEND_SELF_CG (the only shipping backend); the
    // reserved MINRES/GMRES values are rejected at the setter until their phases
    // land. Read at solver-construction time. A plain host scalar; no device state.
    uint32_t sparse_solver_backend = 0u;

    // Env count the World was cooked/built at (== nk::World::EnvCount()). Cached
    // for the DR env-tiling math + the diffsim total_link_count derivation.
    uint32_t env_count = 1u;
    // Stage-1 control mode chosen at world creation. M9: only PDPosition is wired
    // onto the generic nk::World; non-PD modes (Torque/Velocity/ComputedTorque/
    // Osc/Actuator) are an M10 named gap (create rejects them with NOT_SUPPORTED).
    runtime::articulation::ControlMode control_mode =
        runtime::articulation::ControlMode::PDPosition;
    core::diagnostics::InvariantSampler invariant_sampler{
        core::diagnostics::InvariantConfig{
            true,
            1u,
            {},
            true,
            true,
        }
    };
    std::vector<core::diagnostics::InvariantSample> last_invariant_violations;
    std::vector<float> empty_float_storage;

    // The cooked terrain HeightField (the ONE grid physics rests feet on). Empty
    // when no heightfield was baked. The obs/spawn height sampler reads THIS grid
    // via the bilinear sampler, so obs and physics share one height definition.
    nuka::terrain::HeightField cooked_terrain;

    // --- v0.5 p04 N1: per-sensor-field domain-randomization noise -----------
    // Fixed array indexed by nuka_state_field_t. Default is None so a field with
    // no registered noise is a byte no-op on apply and V1 oracle scenes stay
    // byte-identical. `noise_seq[f]` is that field's monotonically advancing
    // per-apply sequence index (the Philox counter seq lane), giving independent
    // noise across steps; the SAME (seed, seq) replays bit-exact on the reverse
    // pass (no RNG state to checkpoint -- exit #6). Sized to cover EVERY public
    // field: derived from the current enum maximum so adding a field never leaves
    // it silently out of range (the old hardcoded 16 rejected terrain fields).
    static constexpr uint32_t kNoiseFieldCount =
        static_cast<uint32_t>(NUKA_FIELD_ACTUATOR_SATURATED) + 1u;
    static_assert(NUKA_FIELD_ACTUATOR_SATURATED >=
                      NUKA_FIELD_ENV_TERRAIN_DIFFICULTY,
                  "kNoiseFieldCount must derive from the maximum field enum");
    nuka::sensor::noise::SensorNoiseConfig noise_config[kNoiseFieldCount];
    uint64_t noise_seq[kNoiseFieldCount] = {};

    // --- v0.5 p04 N2: per-episode domain randomization ----------------------
    // The DR descriptor (default disabled -> apply is a byte no-op, oracle safe).
    // A per-env multiplier/offset is sampled as a PURE FUNCTION of (seed, env_idx,
    // param) at apply time (no mutable sampled state to checkpoint), so the
    // diff-sim backward stays D1 byte-exact two-run with DR on.
    nuka::sensor::noise::DomainRandomizationConfig dr_config;
    // NOMINAL baseline, snapshotted ONCE on the first enabled apply so repeated
    // resets re-randomize AROUND nominal (idempotent) rather than compounding a
    // random walk. `dr_baseline_captured` guards the one-time snapshot.
    bool dr_baseline_captured = false;
    std::vector<float> dr_nominal_link_mass;       // per global link (kg)
    std::vector<float> dr_nominal_joint_armature;  // per global DOF
    float dr_nominal_gravity_z = 0.0f;
    // Nominal contact friction baseline. M9: the legacy batched contact step
    // params are gone, so the DR friction multiplier no longer has a host scalar
    // to poke (contact material lives on the nk Model material buckets); kept so
    // CaptureNominalBaseline + the friction sample compile and stay inert (M10
    // named gap — RL contact DR is rebuilt on the nk world at M10).
    float dr_nominal_friction = 0.0f;

    // Out-of-line so the unique_ptr<nk::World> member is destroyed where nk::World
    // is complete (world.cpp). Declared (not defaulted here) for the same reason
    // the SceneRecord uses out-of-line special members.
    WorldRecord();
    ~WorldRecord();
    WorldRecord(WorldRecord&&) noexcept;
    WorldRecord& operator=(WorldRecord&&) noexcept;
    WorldRecord(const WorldRecord&) = delete;
    WorldRecord& operator=(const WorldRecord&) = delete;
};

// --- M9 T4: the GENERIC authored-scene record (nuka_scene.h) ----------------
// Wraps ONE in-memory nuka::scene::SceneIR (the M2b facade). The scene-authoring
// C-ABI (c_abi/scene.cpp) loads/composes/edits/settles/saves this; the LATER
// world entry (World.create over Scene->CookToModel->nk::World, M9 T5) builds a
// world from the same in-memory SceneIR. The SceneIR is held by unique_ptr
// behind a forward declaration so this header stays free of scene_ir.hpp's
// include weight (the ctor/dtor are out-of-line in scene.cpp where SceneIR is
// complete). NOT a special grasp/union type -- it carries whatever was imported.
struct SceneRecord {
    std::unique_ptr<nuka::scene::SceneIR> scene;

    SceneRecord();
    ~SceneRecord();
    SceneRecord(SceneRecord&&) noexcept;
    SceneRecord& operator=(SceneRecord&&) noexcept;
    SceneRecord(const SceneRecord&) = delete;
    SceneRecord& operator=(const SceneRecord&) = delete;
};

// --- shared world-create helpers (defined in world.cpp) ---------------------
// The pieces nuka_world_create_from_scene assembles, factored so the built-scene
// create (c_abi/scene_builder.cpp) cooks the SAME way without a second world path.

// Validate the world desc + resolve the device/backend and control mode. When
// require_scene_path is true the desc MUST carry a scene_path (the file-load
// path); the built-scene path passes false (its scene comes from a handle).
nuka_result_t ValidateWorldDescAndDevice(
    nuka_device_handle device, const nuka_world_desc_t* desc,
    bool require_scene_path, DeviceRecord** out_device,
    runtime::articulation::ControlMode* out_mode);

// Route the control mode onto the cooked Model's drive_mode, bake the optional
// heightfield collidable (desc->contact_family == 1) and resolve world gravity.
// Mutates `model`; writes the baked terrain + gravity. Returns INVALID_ARG on a
// malformed terrain source (else OK). The post-cook step both create paths share.
nuka_result_t ApplyControlTerrainGravity(
    const nuka_world_desc_t* desc,
    runtime::articulation::ControlMode control_mode, nuka::nk::Model& model,
    nuka::terrain::HeightField* out_terrain, nuka::math::Vec3* out_gravity);

// Build the live nk::World from a cooked Model + scene and insert it into the
// WorldTable (the ONE record-assembly path). The solver_* override the world
// SolverConfig (each 0 keeps the engine default). On success writes *out and
// returns OK; on a failed build returns INTERNAL and leaves *out null.
nuka_result_t FinishWorldCreate(
    nuka::nk::Model&& cooked_model, nuka::scene::SceneIR&& scene,
    nuka::terrain::HeightField&& cooked_terrain, DeviceRecord* device_record,
    float fixed_dt, uint32_t env_count,
    runtime::articulation::ControlMode control_mode,
    const nuka::math::Vec3& gravity, nuka_world_handle* out,
    uint32_t solver_vel_iters = 0u, uint32_t solver_pos_iters = 0u,
    float solver_contact_margin = 0.0f, uint32_t solver_max_pairs = 0u);

} // namespace nuka::c_abi
