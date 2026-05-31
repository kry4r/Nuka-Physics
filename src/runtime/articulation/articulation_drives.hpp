#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- NON-PD control-mode drive launchers (v0.5)
// ---------------------------------------------------------------------------
//
// Stage-1 control-law sub-kernels for the control modes ADDED in v0.5 (C-fwd):
// slice 1 -- Torque and Velocity; slice 2 -- ComputedTorque (inverse dynamics)
// and Actuator (DC-motor torque-speed envelope). The legacy PDPosition mode
// stays in featherstone_aba.{cu,hpp} (ApplyPositionDrives) and is NOT touched, so
// the PD path's instruction / FP order -- and the go2_stand golden -- is
// byte-for-byte unchanged. The HOST-SIDE dispatcher in
// batched_articulated_world.cu / c_abi/world.cpp selects PDPosition (existing
// launcher) vs these by ControlMode (control_mode.hpp).
//
// Both kernels mirror ApplyPositionDriveKernel's shape EXACTLY: a flat per-link
// grid, each thread owns one link and writes only its own tau[link] -- no
// cross-thread interaction, no float atomics, fixed order => D1-deterministic.
// Fixed joints are skipped (their tau is left untouched, same as the PD path).
// The clamp is the same |tau| <= drive_force_limits[link] (when limit > 0).
//
// The implicit joint damping (#43, driven by the drive_damping / joint_damping
// buffer) is a JOINT PROPERTY orthogonal to the control law and still runs
// downstream for BOTH modes; neither kernel applies a control Kd.
// ---------------------------------------------------------------------------

#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"

namespace nuka::runtime::articulation {

// Torque mode: tau[link] = clamp(torque_input[link], +/- drive_force_limits[link]).
// `torque_input` is a per-link device buffer (length state.total_link_count,
// same layout as drive_targets). `drive_force_limits` may be null (no clamp);
// `torque_input` null is a no-op (tau left as-is). No control Kd is applied --
// physical joint damping is handled by the orthogonal implicit-damping path.
void LaunchApplyTorqueDriveKernels(const phi::DeviceContext& context,
                                   ArticulationDeviceState state,
                                   const float* torque_input,
                                   const float* drive_force_limits);

// Velocity mode: tau[link] = clamp(drive_stiffness[link] *
// (velocity_target[link] - qdot[link]), +/- drive_force_limits[link]).
// drive_stiffness is reused as the velocity gain Kp_v. `velocity_target` /
// `drive_stiffness` null is a no-op; `drive_force_limits` null disables the
// clamp. No control Kd here either (the implicit joint damping still runs).
void LaunchApplyVelocityDriveKernels(const phi::DeviceContext& context,
                                     ArticulationDeviceState state,
                                     const float* velocity_target,
                                     const float* drive_stiffness,
                                     const float* drive_force_limits);

// ComputedTorque mode (slice 2): inverse-dynamics / computed-torque control.
//   tau = M*(qddot_des + Kp*e + Kd*edot) + bias,
//     e    = (q_target - q),   edot = (qdot_target - qdot),
//     bias = C(q,qdot)*qdot + G(q)   (Coriolis/centrifugal + gravity).
// With qddot_des = qdot_target = 0 this is the common "computed-torque PD" form;
// Kp = Kd = 0 reduces it to pure gravity/bias compensation (tau = bias).
//
// OPERAND ORDERING (the key challenge). `bias` is NOT formed explicitly. The
// caller runs the EXISTING engine dynamics EARLY this step (a physics-M CRBA, its
// full-tile LDL^T inverse `inertia_M_inv`, plus a tau=0 ABA pass whose qddot is
// `qddot_free`). The exact engine forward map for the JOINT DOFs of a possibly
// floating-base articulation is qddot_j = D*tau_j + qddot_free_j, where D is the
// JOINT sub-block of the physics-M inverse (the Schur-complement admittance --
// NOT M_jj, which would ignore the base reaction). This kernel SOLVES
// D*tau_j = (a_des_j - qddot_free_j) for tau_j, a_des = qddot_des + Kp*e + Kd*edot
// (== tau = M*a_des + bias algebraically): stage-2 ABA then realizes
// qddot_j = a_des_j and the bias -- whatever Coriolis/gravity/model joint-damping
// it contains, all inside qddot_free -- CANCELS exactly (FP-close: CRBA-M vs ABA-
// implicit-M differ only in the last bits). No bias is ever named or decomposed.
//
// `inertia_M_inv` is the per-articulation full-tile physics-M inverse (stride
// max_dof*max_dof, indexed by LocalDofIndex -- the SAME tile/layout the contact
// solve consumes, but built from the physics M WITHOUT the dt*C implicit-damping
// fold). The joint sub-block D starts at dof offset base_dof (6 for a floating
// root, 0 for a fixed base). `q_target` = drive_targets, `kp` = drive_stiffness,
// `kd` = drive_damping (reused). qddot_des / qdot_target default to 0 (null kp/kd
// drop the corresponding term). The grid is one block per articulation (lane 0
// owns the dense joint-block LDL^T solve) -- per-articulation, fixed order, no
// float atomics => D1. The floating-base 6 DOFs are never actuated; the clamp is
// the same |tau| <= drive_force_limits[link]. Any null required pointer
// (q_target/inertia_M_inv/qddot_free) is a no-op (tau left as-is).
void LaunchApplyComputedTorqueDriveKernels(const phi::DeviceContext& context,
                                           ArticulationDeviceState state,
                                           uint32_t max_dof,
                                           const float* inertia_M_inv,
                                           const float* qddot_free,
                                           const float* q_target,
                                           const float* kp,
                                           const float* kd,
                                           const float* drive_force_limits);

// Actuator mode (slice 2): a torque command clamped by a DC-motor torque-speed
// envelope. tau_cmd = torque_input[link]; the max deliverable torque falls
// linearly with speed toward the no-load speed:
//   tau_max(qdot) = clamp(tau_stall*(1 - |qdot|/qdot_noload), 0, tau_stall),
//   tau = clamp(tau_cmd, -tau_max, +tau_max).
// |qdot| is used so the envelope is symmetric for both rotation directions (a
// motor can deliver up to tau_stall at standstill and 0 at the no-load speed
// regardless of which way it spins). `tau_stall` reuses drive_force_limits
// (per-link stall torque); `qdot_noload` is a new per-link buffer. A link with
// qdot_noload <= 0 (uninitialized / fixed) falls back to the plain
// drive_force_limits clamp (== Torque mode), so a zero-init buffer is safe.
// torque_input null is a no-op. Flat per-link grid, no atomics => D1.
void LaunchApplyActuatorDriveKernels(const phi::DeviceContext& context,
                                     ArticulationDeviceState state,
                                     const float* torque_input,
                                     const float* drive_force_limits,
                                     const float* actuator_noload_speed);

} // namespace nuka::runtime::articulation
