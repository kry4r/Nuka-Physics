#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::gpu -- batched articulated-with-contacts GPU step path
// ---------------------------------------------------------------------------
//
// p01-F T6. The production N-env (e.g. 4096 Unitree Go2) articulated step path.
// It orchestrates the already-built-and-validated p01-F T1..T5 pieces into one
// deterministic per-step pipeline with per-stage timing, ALONGSIDE the existing
// single-env path (nuka::c_abi::StepWorldGpu); it does NOT alter that path.
//
// Per-step pipeline (exactly the sequence validated in the T5 test):
//   1. ApplyPositionDrives                -> tau
//   2. ComputeAccelerations(gravity_z)    -> qddot                (featherstone_aba)
//   3. velocity-integrate qdot += qddot*dt                         (integrator)
//   4. UpdateWorldLinkPoses -> world_pose; copy into state.link_pose
//      (the chain Jacobian reads state.link_pose, which UploadArticulationState
//      leaves at the static cooked rest pose -- it must be refreshed from the
//      current q each step for the contact geometry to be correct).
//   5. DetectFootGroundContacts                                    (contact_generation)
//   6. tangent basis (t1,t2); chain Jacobians normal + t1 + t2     (row_builder)
//   7. ComputeArticulationInertiaM -> M; FactorArticulationInertiaM -> M_inv
//   8. ComputeContactEffectiveMass for normal + t1 + t2
//   9. AssembleArticulatedContactRows                              (row_builder)
//  10. SolveArticulatedContactRows (fused apply Delta_qdot; persistent lambda)  (row_solver)
//  11. position-integrate q += qdot*dt                             (integrator)
//
// The split velocity/position integration (FeatherstoneAba::IntegrateVelocity /
// IntegratePosition) is bit-for-bit equal to the combined Integrate(), so with
// contacts INACTIVE (ground far away => the solve is a genuine no-op) the
// batched q/qdot are bit-identical to the single-env engine path. That is the
// PRIMARY validation gate.
//
// Determinism (D1): the underlying kernels use no float atomics and a fixed
// loop/grid order, so all envs are bit-identical to each other (same initial
// state) and bit-identical across runs.
// ---------------------------------------------------------------------------

#include "core/perf/perf_recorder.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace nuka::runtime::gpu {

// Per-step inputs for the batched articulated step. The drive buffers are device
// pointers of length state.total_link_count (one drive descriptor per link, the
// same convention ApplyPositionDrives uses). They may be null only if no drives
// are desired (then ApplyPositionDrives is a no-op and tau is whatever the prior
// step left it -- callers that want drives must supply all four).
struct BatchedArticulatedStepParams {
    const float* drive_targets = nullptr;
    const float* drive_stiffness = nullptr;
    const float* drive_damping = nullptr;
    const float* drive_force_limits = nullptr;
    float gravity_z = -9.81f;
    float dt = 1.0f / 240.0f;
    float friction_coefficient = articulation::kContactFriction;
    // Caps the normal-row Baumgarte bias at min(beta/dt*max(depth-slop,0), this).
    // Default +inf == non-binding (matches T5). The production step passes a
    // finite value (~3 m/s) to bound the one-step velocity transient at large
    // penetrations.
    float baumgarte_max_velocity = std::numeric_limits<float>::infinity();
};

// Owns the device-resident articulation state + all reused scratch buffers + a
// PerfRecorder. Created once (scratch sized from the world), then Step() advances
// one step and reuses every buffer. The persistent lambda buffer is carried
// across steps for warm-start.
class BatchedArticulatedWorld {
public:
    // Builds the device state from `host` (which must hold env_count replicated
    // articulations, e.g. from ReplicateArticulationHostState) and allocates all
    // scratch sized from the world. `max_dof` must equal every articulation's DOF
    // count and the chain Jacobian dof_stride (precondition shared by T3/T4/T5).
    BatchedArticulatedWorld(const phi::DeviceContext& context,
                            const articulation::ArticulationHostState& host,
                            const std::vector<articulation::FootShape>& feet,
                            uint32_t max_dof,
                            float ground_height);

    // Advances one deterministic step in place on the device state, recording
    // per-stage timings under the canonical perf tags. The persistent lambda
    // buffer carries across calls for warm-start (warm-start affects convergence
    // only -- never determinism).
    void Step(const BatchedArticulatedStepParams& params);

    // Number of replicated environments (== articulation count).
    uint32_t EnvCount() const { return env_count_; }
    uint32_t BaseLinkCount() const { return base_link_count_; }
    uint32_t MaxDof() const { return max_dof_; }

    // Device-state view (q, qdot, ... live here). Stable for the object's life.
    articulation::ArticulationDeviceState View() { return device_.View(); }

    // Downloads the current device state into `host` (q, qdot, ...).
    void Download(articulation::ArticulationHostState* host) const;

    // Per-step timing aggregator (canonical tags: featherstone_aba, integrator,
    // contact_generation, row_builder, row_solver, buffer_mgmt, step_total).
    core::perf::PerfRecorder& Perf() { return perf_; }
    const core::perf::PerfRecorder& Perf() const { return perf_; }

    // Reads back this step's converged per-slot impulses (length
    // env_count * kMaxFootContactsPerEnv * 3, {normal,t1,t2} per slot).
    std::vector<float> DownloadLambda() const;
    // Reads back the most-recent per-slot detected contact depths (length
    // env_count * kMaxFootContactsPerEnv); inactive slots are 0.
    std::vector<float> DownloadContactDepth() const;
    std::vector<uint32_t> DownloadContactLink() const;
    std::vector<articulation::ArticulatedContactRow> DownloadRows() const;

private:
    const phi::DeviceContext& context_;
    articulation::ArticulationDeviceBuffers device_;
    core::perf::PerfRecorder perf_;

    uint32_t env_count_ = 0u;
    uint32_t base_link_count_ = 0u;
    uint32_t total_link_count_ = 0u;
    uint32_t articulation_count_ = 0u;
    uint32_t max_dof_ = 0u;
    uint32_t foot_count_ = 0u;
    uint32_t slot_count_ = 0u;
    float ground_height_ = 0.0f;

    // Scratch (allocated once, reused every step).
    phi::Buffer feet_;            // FootShape[foot_count]
    phi::Buffer composite_;       // LinkSpatialInertia[total_link_count]
    phi::Buffer m_;               // float[articulation_count * max_dof * max_dof]
    phi::Buffer m_inv_;           // float[articulation_count * max_dof * max_dof]
    phi::Buffer world_pose_;      // Transform[total_link_count]
    phi::Buffer contact_link_;    // uint32_t[slot_count]
    phi::Buffer contact_point_;   // Vec3[slot_count]
    phi::Buffer contact_normal_;  // Vec3[slot_count]
    phi::Buffer contact_depth_;   // float[slot_count]
    phi::Buffer contact_count_;   // uint32_t[env_count]
    phi::Buffer tangent1_;        // Vec3[slot_count]
    phi::Buffer tangent2_;        // Vec3[slot_count]
    phi::Buffer jac_normal_;      // float[slot_count * max_dof]
    phi::Buffer jac_tangent1_;    // float[slot_count * max_dof]
    phi::Buffer jac_tangent2_;    // float[slot_count * max_dof]
    phi::Buffer meff_normal_;     // float[slot_count]
    phi::Buffer meff_tangent1_;   // float[slot_count]
    phi::Buffer meff_tangent2_;   // float[slot_count]
    phi::Buffer rows_;            // ArticulatedContactRow[slot_count]
    phi::Buffer lambda_;          // float[slot_count * 3] (persistent warm-start)
};

} // namespace nuka::runtime::gpu
