#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- contact-free forward + replay orchestrator (p02-B/C)
// ---------------------------------------------------------------------------
//
// THE forward this whole differentiable rollout is built on. The p02-A adjoint
// (StepBackward) is the reverse of ONE contact-free PD step. It is EXACT on the
// joint channel (q/qdot/tau/mass) AND the floating-base VELOCITY channel (the
// root spatial velocity v_root', whose gyroscopic bias is linearized at the
// PRE-integration v_root_pre snapshot). The base-ORIENTATION channel is DEFERRED
// (NOT differentiated): the a_grav = R(base_rot)^T g dependence on base rotation
// and the quaternion POSE integrator -- base orientation is held fixed, the
// supported loss seeds are q'/qdot'/v_root', NOT base_pose'. The step:
//
//     tau = clamp(Kp*(target-q) - Kd*qdot)          (ApplyPositionDrives, EXPLICIT
//                                                     damping: defer_velocity_damping=false)
//   -> qddot / a_base = ABA(state, tau, mass, g)     (ComputeAccelerations)
//   -> qdot' = qdot + qddot*dt                       (IntegrateVelocity)
//      v_root' = v_root + (a_base - a_grav)*dt        (IntegrateFloatingBaseVelocity)
//   -> q' = q + qdot'*dt                             (IntegratePosition)
//      base_pose' advanced                            (IntegrateFloatingBasePose)
//
// CRITICAL: this is NOT the full batched articulated Step(). That Step() applies DEFERRED
// drive damping folded into an implicit (M+dt*C)^-1 solve and runs the full
// contact/CRBA pipeline unconditionally -- a DIFFERENT function the p02-A adjoint
// does not model. Using Step() as the forward would pass replay-bit-exact but
// FAIL grad-vs-FD. So we compose the SAME contact-free kernel sequence the p02-A
// FD test (rung e / floating) validates against, here for ALL THREE roles:
// record (the live forward), replay (regenerate a window's intermediates), and
// the FD oracle. One forward function -> the adjoint it pairs with is the one
// p02-A FD-validated exact on the joint + floating-VELOCITY channels (the
// base-orientation channel stays deferred -- see above).
//
// The orchestrator owns the per-step drive descriptors (gains / force-limits,
// rollout-constant) as device buffers, the dt / gravity, and the K-cadence
// checkpoint capture. StepOnce advances the live state one step IN PLACE
// (reusing the engine kernels) and is bit-exact across record vs replay (D1: the
// kernels have no float atomics + a fixed order; identical inputs -> identical
// bits). ReplayWindow restores the nearest checkpoint <= a target step and
// re-rolls StepOnce to refill a window. ReconstructInputs fills a
// StepBackwardInputs for the reverse of step k from the tape + the orchestrator's
// constants (q_pre/qdot_pre are snapshotted by the backward runner).
// ---------------------------------------------------------------------------

#include "diffsim/checkpoint.hpp"
#include "diffsim/step_backward.hpp"
#include "diffsim/tape.hpp"
#include "phi/buffer.hpp"
#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>
#include <vector>

namespace nuka::diffsim {

namespace articulation = nuka::runtime::articulation;

// Rollout-constant forward parameters (the non-action inputs).
struct RolloutParams {
    float dt = 1.0f / 240.0f;
    float gravity_z = -9.81f;
};

class RecomputeOrchestrator {
public:
    // Binds the orchestrator to a live device state + per-link drive gains
    // (host, length total_link_count, env-major). The gains / force-limits /
    // dI_dmass are uploaded once and reused every step; the per-step actions come
    // from the tape. `mass_params` (length total_link_count) builds the
    // representation-consistent dI/dmass slope for grad_mass.
    RecomputeOrchestrator(cudaStream_t stream, int device_id,
                          articulation::ArticulationDeviceState state,
                          const RolloutParams& params,
                          const std::vector<float>& drive_stiffness,
                          const std::vector<float>& drive_damping,
                          const std::vector<float>& drive_force_limits,
                          const std::vector<LinkMassParams>& mass_params);

    uint32_t TotalLinkCount() const { return total_link_count_; }
    const RolloutParams& Params() const { return params_; }

    // Runs ONE contact-free PD step IN PLACE on the live state, with the action
    // (drive_targets) at `device_actions` (length total_link_count). The SAME
    // function backs record + replay -> bit-exact. Does NOT synchronize. Replay
    // of a window is driven by the backward runner as a checkpoint Restore +
    // a sequence of StepOnce calls, so the runner controls exactly when it
    // snapshots q_pre/qdot_pre (immediately before the step it is reconstructing).
    void StepOnce(const float* device_actions);

    // Build the StepBackwardInputs for the reverse of step `step`. Points
    // drive_targets at the tape's action slice, the gains/force-limits/dI_dmass at
    // the orchestrator's uploaded buffers, and sets dt/gravity. q_pre/qdot_pre are
    // filled by the backward runner (it owns the pre-step snapshot buffers).
    StepBackwardInputs ReconstructInputs(const Tape& tape, uint32_t step) const;

    // Device pointers to the uploaded drive descriptors (const), for the runner.
    const float* DriveStiffness() const {
        return static_cast<const float*>(phi::BufferBase(stiffness_));
    }
    const float* DriveDamping() const {
        return static_cast<const float*>(phi::BufferBase(damping_));
    }
    const float* DriveForceLimits() const {
        return static_cast<const float*>(phi::BufferBase(force_limits_));
    }
    const float* DiDMass() const {
        return static_cast<const float*>(phi::BufferBase(dI_dmass_));
    }

    articulation::ArticulationDeviceState State() const { return state_; }

    ~RecomputeOrchestrator();
    RecomputeOrchestrator(const RecomputeOrchestrator&) = delete;
    RecomputeOrchestrator& operator=(const RecomputeOrchestrator&) = delete;

private:
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    articulation::ArticulationDeviceState state_;
    RolloutParams params_;
    uint32_t total_link_count_ = 0u;

    phi::Buffer* stiffness_ = nullptr;     // float[total_link_count]
    phi::Buffer* damping_ = nullptr;       // float[total_link_count]
    phi::Buffer* force_limits_ = nullptr;  // float[total_link_count]
    phi::Buffer* dI_dmass_ = nullptr;      // float[total_link_count * 36]
};

}  // namespace nuka::diffsim
