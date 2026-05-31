#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- joint control-mode enumeration (v0.5 C-fwd)
// ---------------------------------------------------------------------------
//
// The control law that stage 1 of the articulated step applies to convert the
// per-link drive/control inputs into the joint torque `tau` consumed by ABA.
// Historically the engine HARD-WIRED a PD position drive; v0.5 makes stage 1
// a host-side DISPATCH on this enum so control laws beyond PD can be added.
//
// Slice 1 implements PDPosition (byte-for-byte the legacy path), Torque, and
// Velocity. ComputedTorque / Osc / Actuator are RESERVED enumerators for later
// slices -- the dispatcher hits an unimplemented path for them (the C ABI / the
// BatchedArticulatedWorld constructor reject values > Velocity so an
// unimplemented mode is never silently mis-actuated).
//
// The dispatch is HOST-SIDE: PDPosition routes to the existing, untouched
// ApplyPositionDriveKernel (so the PD instruction / FP order -- and thus the
// go2_stand golden trajectory -- is bit-for-bit unchanged); the new modes route
// to separate kernels in articulation_drives.{cu,hpp}. The implicit joint
// damping (#43, driven by the `joint_damping` / drive_damping buffer) is a JOINT
// PROPERTY orthogonal to the control law and still runs for every mode.
//
// The plain uint8_t underlying type matches the C ABI nuka_world_desc_t
// .control_mode so a zero-initialized desc maps to PDPosition (backward compat).
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::runtime::articulation {

enum class ControlMode : uint8_t {
    PDPosition = 0,     // tau = Kp*(target-q) [- Kd*qdot]  (legacy, unchanged)
    Torque = 1,         // tau = torque_input               (direct torque)
    Velocity = 2,       // tau = Kp_v*(velocity_target-qdot)(velocity servo)
    ComputedTorque = 3, // RESERVED (later slice)
    Osc = 4,            // RESERVED (later slice)
    Actuator = 5,       // RESERVED (later slice)
};

// The highest control mode IMPLEMENTED in this slice. Values above this are
// reserved enumerators and are rejected at the ABI / world-construction boundary
// rather than silently falling back to PD.
inline constexpr uint8_t kMaxImplementedControlMode =
    static_cast<uint8_t>(ControlMode::Velocity);

inline constexpr bool IsControlModeImplemented(uint8_t mode) {
    return mode <= kMaxImplementedControlMode;
}

} // namespace nuka::runtime::articulation
