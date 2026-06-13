#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- single-step reverse-mode adjoint (CONTACT-FREE PD path)
// ---------------------------------------------------------------------------
//
// p02-A: the differentiable single-step spine. Hand-written reverse-mode
// adjoint of ONE forward step on the contact-free path:
//
//     drive(action) -> ABA(mass via spatial inertia) -> integrate -> state'
//
// composing, in forward order:
//   1. ApplyPositionDrives        tau = clamp(Kp*(target-q) - Kd*qdot)
//   2. ComputeAccelerations       3-pass Featherstone ABA -> qddot / a_base
//   3. IntegrateVelocity          qdot' = qdot + qddot*dt   (joints)
//      IntegrateFloatingBaseVel   v_root' = v_root + (a_base - a_grav)*dt
//   4. IntegratePosition          q'   = q + qdot'*dt       (joints)
//      IntegrateFloatingBasePose  base_pose' advanced (the base-pose / quaternion
//                                 integrator is NOT differentiated here -- the
//                                 loss seeds we support are on q' / qdot' (joints)
//                                 and v_root' (the root spatial velocity))
//
// This is the connective adjoint spine the whole tape/backward (p02-B/C) and the
// exit demo (mass system-ID) ride on. It is CONTACT-FREE: it composes drive ->
// ABA -> integrate ONLY (no contact detect/solve/CRBA-M). The reverse mirrors
// the forward's single-thread-per-articulation structure EXACTLY, so the tree
// adjoint accumulation is fixed-order plain `+=` (NO float atomics) -- a
// correctness + D1 + lint property by construction.
//
// SCOPE: this is an INTERNAL / test-facing C++ API (NOT the tape, checkpoint, or
// the nuka_diffsim.h C-ABI -- those are p02-B). The forward must already have run
// (its intermediates -- link_xup, joint_motion_subspace, link_articulated_I,
// joint_diagonal, link_u_spatial, joint_force, link_velocity, link_velocity_bias,
// link_bias_force, link_acceleration, qddot -- persist in the device buffers and
// are read by the reverse). The caller snapshots the PRE-step q / qdot AND the
// PRE-integration root spatial velocity v_root_pre (the integrators overwrite all
// three in place; the drive + ABA pass-1/2 reverses, and the root gyroscopic
// adjoint's linearization point, need the pre-step values).
//
// COVERAGE (FD-validated, see tests/diffsim/test_aba_reverse_fd.cpp):
//   * Fixed-base revolute chain: d/d(tau,qdot,q,mass), the EXACT M^-1 cross-check
//     (~1e-7), integration, and the full drive step d/d(action) + d/d(mass).
//   * Floating-base root (FloatingBase joint): the VELOCITY channel is EXACT --
//     the 6x6 LDLT-solve adjoint at the root, the root gyroscopic pass-1 reverse,
//     and the floating-base linear-velocity integrator reverse, seeded via
//     grad_link_velocity_out and FD-validated d/d(tau) + d/d(mass) + d/d(v_root).
//     CRITICAL: the root gyroscopic bias p[root] = ForceCross(v, I*v) is QUADRATIC
//     in the root spatial velocity v, so its adjoint (and the grad_I[root] ->
//     grad_mass[root] outer-product path) is linearized at the PRE-integration
//     v_root (StepBackwardInputs::v_root_pre), NOT the live post-integration value
//     that IntegrateFloatingBaseVelocity leaves in link_velocity[root]. The
//     large-dt twin-variant FD test guards this linearization point.
// ORIENTATION channel (p08b -- now EXACT, FD-validated): the base-ORIENTATION
// channel is differentiated end-to-end. Two coupled forward sites are reversed:
//   (1) the gravity-rotation a_grav = R(base_rot)^T g in the floating-base
//       VELOCITY integrator (v_root'[3:6] picks up +R(base_rot_pre)^T g * dt), and
//   (2) the quaternion POSE integrator: position' = position + R(base_rot_pre) *
//       v_root'[3:6] * dt ; rotation' = normalize(QuatMul(base_rot_pre, dq)),
//       dq = (1, 0.5*v_root'[0:3]*dt).
// CRITICAL: all three orientation reads are at the PRE-step base orientation
// (StepBackwardInputs::base_pose_pre) -- the pose integrator overwrites
// base_pose in place (pre->post) BEFORE StepBackward runs, mirroring v_root_pre.
// base_pose' is now a supported loss seed (StepBackwardGrads::grad_base_pose_out,
// PER-ARTICULATION: 3 position + 4 quaternion floats). The pose reverse runs
// BEFORE the velocity reverse so its contribution into grad v_root' (via
// v_lin_body / omega) is consumed by the velocity-integrator reverse in one pass.
// Still deferred: the prismatic-joint q-channel and contacts.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"

#include <vector>

namespace nuka::diffsim {

namespace articulation = nuka::runtime::articulation;

// Differentiable inputs to one contact-free PD step, as device pointers. The
// drive descriptors are per-link (env-major, total_link_count). q_pre / qdot_pre
// are the PRE-step joint positions / velocities (snapshotted by the caller before
// IntegrateVelocity/IntegratePosition overwrote the live q/qdot buffers).
struct StepBackwardInputs {
    // Pre-step joint state (device, total_link_count).
    const float* q_pre = nullptr;
    const float* qdot_pre = nullptr;
    // Pre-INTEGRATION root spatial velocity (device, total_link_count*6, one
    // 6-vector per link -- only the floating root entries are read). Snapshotted
    // by the caller BEFORE IntegrateFloatingBaseVelocity overwrote the live
    // link_velocity[root] (v_pre -> v_post). The forward evaluated the root
    // gyroscopic bias p[root] = ForceCross(v, I*v) at v_pre; that bias is
    // QUADRATIC in v, so its adjoint (and the grad_I[root] -> grad_mass[root]
    // path) must linearize at v_pre, NOT the live post-integration value. When
    // null the root gyroscopic reverse falls back to the live state (legal only
    // for fixed-base models, which have no floating root).
    const float* v_root_pre = nullptr;
    // PRE-step base orientation snapshot, per articulation (device,
    // articulation_count Transforms == articulation_count*7 floats laid out as
    // math::Transform{Vec3 position; Quat rotation}). Snapshotted by the caller
    // BEFORE IntegrateFloatingBasePose overwrote the live base_pose (pre->post).
    // The three orientation reads -- a_grav = R(base_rot)^T g, v_lin_world =
    // R(base_rot)*v_lin_body, and rotation' = normalize(QuatMul(base_rot, dq)) --
    // all evaluate at this PRE-step orientation, so the adjoint linearizes here,
    // NOT at the clobbered live (post) base_pose. When null the orientation channel
    // is OFF (no a_grav-rotation grad, no pose-integrator reverse) -- the legacy
    // velocity-only floating behavior, byte-identical.
    const math::Transform* base_pose_pre = nullptr;
    // Drive (action) descriptors (device, total_link_count).
    const float* drive_targets = nullptr;    // the ACTION input we differentiate
    const float* drive_stiffness = nullptr;  // Kp
    const float* drive_damping = nullptr;    // Kd
    const float* drive_force_limits = nullptr;
    // Per-link d(spatial inertia I[36])/d(mass), device, total_link_count*36.
    // The reverse contracts grad_I against this to get grad_mass. Built host-side
    // from the SAME MakeSpatialInertia mapping the forward uses (so the mass
    // perturbation is representation-consistent: perturbing the scalar mass
    // updates the full 6x6 spatial inertia -- parallel-axis + COM coupling
    // included). See diffsim::BuildSpatialInertiaMassJacobian.
    const float* dI_dmass = nullptr;
    float dt = 0.0f;
    float gravity_z = 0.0f;
    // When true, the reverse converts the accumulated dL/dtau through the PD
    // drive (tau = clamp(Kp*(target-q) - Kd*qdot)) into grad_target / grad_q /
    // grad_qdot. When false (bare-ABA rungs a/b/c where tau is a DIRECT input),
    // the drive reverse is skipped and the raw dL/dtau is left in
    // StepBackwardGrads::grad_tau_out.
    bool has_drive = true;
    // When true, STAGE A reverses the velocity+position integrators (the loss is
    // seeded on the POST-step q'/qdot' via grad_q_out/grad_qdot_out). When false,
    // the output is qddot itself (rungs a/b/c + the M^-1 cross-check): the loss is
    // seeded directly on qddot via grad_qddot_seed, and grad_q_out/grad_qdot_out
    // start at the caller's value (typically 0) -- no integrator coupling / dt
    // truncation contaminates the qddot Jacobian.
    bool has_integrate = true;
    // Per-DOF dL/dqddot seed (device, total_link_count). Read ONLY when
    // has_integrate=false. nullptr otherwise.
    const float* grad_qddot_seed = nullptr;
    // When true, also back-propagate through the link_xup = JointTransform(q)
    // configuration path (the d/d(q) X-path, revolute only). grad_X feeds ONLY
    // grad_q (STAGE E), so grad_tau / grad_qdot / grad_mass are independent of it
    // -- the M^-1 and mass checks are isolated from any q-channel bug. Turn off to
    // validate the spine before the rotation-derivative chain.
    bool enable_q_channel = true;
};

// Output gradient buffers (device pointers), all sized total_link_count. The
// caller seeds the loss adjoint into grad_q_out / grad_qdot_out (= dL/dq',
// dL/dqdot' on the POST-step state) BEFORE calling StepBackward; StepBackward
// OVERWRITES them with the back-propagated input gradients in place, and writes
// grad_target / grad_mass. Floating-base seeds (dL/d v_root') live in
// grad_link_velocity_out (per-link spatial 6-vector adjoint), also seeded by the
// caller and overwritten with the propagated grad.
struct StepBackwardGrads {
    float* grad_q_out = nullptr;        // in: dL/dq'   out: dL/dq   (pre-step)
    float* grad_qdot_out = nullptr;     // in: dL/dqdot' out: dL/dqdot (pre-step)
    float* grad_target_out = nullptr;   // out: dL/d(drive_targets) (the action)
    float* grad_mass_out = nullptr;     // out: dL/d(link mass), per link
    // out: dL/dtau (the accumulated torque cotangent). Always written. When
    // has_drive=true it is ALSO consumed to fill grad_target_out; when false it
    // is the bare-ABA d/d(tau) deliverable (rungs a/b/c). Caller pre-zeroes it.
    float* grad_tau_out = nullptr;
    // Per-link spatial-velocity adjoint (6 floats/link). in: dL/d v_root' seed
    // on the floating root (zero elsewhere); out: dL/d v_root (pre-step) on root.
    float* grad_link_velocity_out = nullptr;
    // PER-ARTICULATION base-pose adjoint, articulation_count Transforms (7 floats
    // each: 3 position + 4 quaternion, matching math::Transform layout). in:
    // dL/d base_pose' seed (the POST-step base pose); out: dL/d base_pose_pre
    // (pre-step). Overwritten in place like grad_link_velocity_out. When null (or
    // base_pose_pre null) the orientation channel is OFF. The position adjoint is
    // an identity passthrough (position does not feed the step's dynamics); the
    // rotation adjoint flows through the pose-integrator + a_grav reverse.
    float* grad_base_pose_out = nullptr;
};

// Run the single-step reverse. `state` is the device view AFTER the forward step
// (its intermediate buffers hold the forward's checkpointed values). Seeds the
// loss adjoint from grads (grad_q_out / grad_qdot_out / grad_link_velocity_out),
// back-propagates through integrate -> ABA(3-pass reverse) -> drive, and writes
// grad_q_out / grad_qdot_out / grad_target_out / grad_mass_out /
// grad_link_velocity_out. ONE thread per articulation (mirrors the forward),
// fixed-order atomic-free accumulation -> D1 bit-exact.
void StepBackward(const phi::DeviceContext& context,
                  articulation::ArticulationDeviceState state,
                  const StepBackwardInputs& inputs,
                  const StepBackwardGrads& grads);

// M9 T7: the SINGLE StepBackward kernel launcher, DEFINED in the phi v2 op TU
// (phi/backend_cuda/ops/diffsim_backward.cu) where the kernel now lives (as
// NkOp::StepBackward). The host StepBackward() above + the NkOp dispatch both
// drive THIS one launcher, so the op path and the direct path are byte-identical
// by single-source. SAME launch config as the old in-target kernel
// (<<<articulation_count, 32, 0, stream>>>). enable_q_channel mirrors
// inputs.enable_q_channel (passed explicitly to match the old call site).
void LaunchStepBackwardKernel(articulation::ArticulationDeviceState state,
                              const StepBackwardInputs& inputs,
                              const StepBackwardGrads& grads,
                              bool enable_q_channel, cudaStream_t stream);

// M9 T7: the backward runner's cross-step / cross-window grad_mass accumulate
// (acc[i] += step[i], one thread/link, NO atomics -> D1). DEFINED alongside the
// StepBackward kernel in the phi v2 op TU so backward_runner stays pure-host
// (.cpp). SAME launch geometry (256 threads/block) as the deleted
// backward_runner.cu kernel.
void LaunchAddInPlace(float* acc, const float* step, uint32_t n,
                      cudaStream_t stream);

// Builds the per-link d(I[36])/d(mass) Jacobian on the host, from the SAME
// MakeSpatialInertia mapping the engine uses. Because MakeSpatialInertia is
// exactly affine in mass for fixed (diagonal_inertia, inertial_frame), the slope
// is dI/dmass = MakeSpatialInertia(m+1) - MakeSpatialInertia(m), independent of
// m. Returns total_link_count*36 floats (row-major 6x6 per link). The caller
// uploads this to device and points StepBackwardInputs::dI_dmass at it.
struct LinkMassParams {
    float mass = 0.0f;
    nuka::math::Vec3 diagonal_inertia{0.0f, 0.0f, 0.0f};
    nuka::math::Transform inertial_frame = nuka::math::Transform::Identity();
};
std::vector<float> BuildSpatialInertiaMassJacobian(
    const std::vector<LinkMassParams>& links);

}  // namespace nuka::diffsim
