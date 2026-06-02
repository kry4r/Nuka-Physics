#pragma once

#include "nuka/nuka.h"

#include "core/diagnostics/invariants.hpp"
#include "sensor/noise/noise_config.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "phi/owned_stream.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/control_mode.hpp"
#include "runtime/gpu/batched_articulated_world.hpp"
#include "runtime/world_builder.hpp"
#include "runtime/world_stepper.hpp"

#include <memory>
#include <string>
#include <vector>

namespace nuka::c_abi {

struct DeviceRecord {
    phi::DeviceContext context;
    std::unique_ptr<phi::OwnedStream> owned_stream;
};

struct WorldRecord {
    DeviceRecord* device = nullptr;
    runtime::BuiltWorld world;
    runtime::WorldStepOptions step_options;
    runtime::articulation::ArticulationHostState articulation_host;
    runtime::articulation::ArticulationDeviceBuffers articulation_device;
    std::vector<float> drive_targets_host;
    std::vector<float> drive_stiffness_host;
    std::vector<float> drive_damping_host;
    std::vector<float> drive_force_limits_host;
    phi::Buffer drive_targets_device;
    phi::Buffer drive_stiffness_device;
    phi::Buffer drive_damping_device;
    phi::Buffer drive_force_limits_device;
    phi::Buffer joint_position_buffer;
    phi::Buffer joint_velocity_buffer;
    uint32_t simulated_step_count = 0u;

    // --- Sparse solver backend selection (v0.7 p01) -----------------------
    // Which SparseLinearSolver backend the diff-sim / general solver path uses.
    // Default 0 == NUKA_SOLVER_BACKEND_SELF_CG (the only shipping backend); the
    // reserved MINRES/GMRES values are rejected at the setter until their phases
    // land. Read at solver-construction time. A plain host scalar; no device state.
    uint32_t sparse_solver_backend = 0u;

    // --- Single-env implicit joint-damping scratch (Option B unification) ---
    // The single-env oracle path (StepWorldGpu) now runs the SAME general
    // implicit joint damping as the batched contacts path. It needs the
    // per-articulation joint-space inertia M and its inverse (M+dt*C)^-1 plus a
    // composite-inertia scratch buffer for the CRBA. Allocated lazily on the
    // first step (mirroring the lazy articulation_device upload) and reused for
    // every step. `single_env_max_dof` is the articulation's DOF count (== the
    // M tile stride); 0 until allocated.
    uint32_t single_env_max_dof = 0u;
    phi::Buffer single_env_inertia_m;
    phi::Buffer single_env_inertia_m_inv;
    phi::Buffer single_env_composite;

    // --- Multi-env batched articulated path (p01-F T7) --------------------
    // When env_count > 1 the world is driven through the batched articulated-
    // with-contacts step path (T6) instead of the single-env Featherstone
    // sequence. `batched` is null for the single-env (env_count == 1) oracle
    // path, which stays byte-for-byte unchanged. The replicated drive buffers
    // tile the base hold drives across all envs (link-major, length
    // env_count * base_link_count) as BatchedArticulatedStepParams requires.
    uint32_t env_count = 1u;
    // v0.5 C-fwd: stage-1 control mode chosen at world creation (default
    // PDPosition). Non-PD modes require the batched path; the single-env oracle
    // is PDPosition-only.
    runtime::articulation::ControlMode control_mode =
        runtime::articulation::ControlMode::PDPosition;
    std::unique_ptr<runtime::gpu::BatchedArticulatedWorld> batched;
    runtime::gpu::BatchedArticulatedStepParams batched_step_params;
    phi::Buffer batched_drive_targets_device;
    phi::Buffer batched_drive_stiffness_device;
    phi::Buffer batched_drive_damping_device;
    phi::Buffer batched_drive_force_limits_device;
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

    // --- v0.5 p04 N1: per-sensor-field domain-randomization noise -----------
    // Fixed array indexed by nuka_state_field_t (0..15). Default is None so a
    // field with no registered noise is a byte no-op on apply and V1 oracle
    // scenes stay byte-identical. `noise_seq[f]` is that field's monotonically
    // advancing per-apply sequence index (the Philox counter seq lane), giving
    // independent noise across steps; the SAME (seed, seq) replays bit-exact on
    // the reverse pass (no RNG state to checkpoint -- exit #6). 16 == one past
    // the max field enum (NUKA_FIELD_TASK_TARGET == 15).
    static constexpr uint32_t kNoiseFieldCount = 16u;
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
    float dr_nominal_friction = 0.0f;  // batched contact path scalar (inert here)
};

struct BufferRecord {
    phi::Buffer buffer;
};

nuka_result_t RefreshWorldBuffers(WorldRecord& record) noexcept;

} // namespace nuka::c_abi
