#pragma once

#include "nuka/nuka.h"

#include "core/diagnostics/invariants.hpp"
#include "sensor/noise/noise_config.hpp"
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
#include "scene/terrain/heightfield.hpp"

#include <memory>
#include <string>
#include <vector>

namespace nuka::scene {
class SceneIR;  // scene/scene_ir.hpp -- fwd-declared (held by unique_ptr below).
}  // namespace nuka::scene

namespace nuka::nk {
class World;  // nk/pipeline/world.hpp -- fwd-declared (held by unique_ptr below;
              // complete in world.cpp where the World ctor runs).
}  // namespace nuka::nk

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
            phi::BackendFree(backend);
            backend = nullptr;
        }
        // phi_device is registry-owned; not freed.
    }
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
        static_cast<uint32_t>(NUKA_FIELD_JOINT_FEEDFORWARD) + 1u;
    // JOINT_FEEDFORWARD is the highest field enumerator; a new field above it must
    // bump this anchor so kNoiseFieldCount still covers the whole enum.
    static_assert(NUKA_FIELD_JOINT_FEEDFORWARD >= NUKA_FIELD_ENV_TERRAIN_DIFFICULTY,
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

} // namespace nuka::c_abi
