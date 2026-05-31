#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- NON-PD control-mode drive launchers (v0.5)
// ---------------------------------------------------------------------------
//
// Stage-1 control-law sub-kernels for the control modes ADDED in v0.5 (C-fwd
// slice 1): Torque and Velocity. The legacy PDPosition mode stays in
// featherstone_aba.{cu,hpp} (ApplyPositionDrives) and is NOT touched, so the PD
// path's instruction / FP order -- and the go2_stand golden -- is byte-for-byte
// unchanged. The HOST-SIDE dispatcher in batched_articulated_world.cu /
// c_abi/world.cpp selects PDPosition (existing launcher) vs these by
// ControlMode (control_mode.hpp).
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

} // namespace nuka::runtime::articulation
