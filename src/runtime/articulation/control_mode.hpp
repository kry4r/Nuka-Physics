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
// Velocity. Slice 2 ADDS ComputedTorque (inverse-dynamics / computed-torque PD)
// and Actuator (DC-motor torque-speed envelope). Osc remains a RESERVED
// enumerator for the Phase-3 solver slice -- the C ABI / the
// BatchedArticulatedWorld constructor reject Osc (and any value above Actuator)
// so an unimplemented mode is never silently mis-actuated.
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
    ComputedTorque = 3, // tau = M*(qddot_des+Kp*e+Kd*edot)+bias (inverse dynamics)
    Osc = 4,            // RESERVED (Phase-3 solver slice)
    Actuator = 5,       // tau = clamp(torque_cmd, +/-tau_max(qdot)) (DC motor)
};

// Whether a control mode is IMPLEMENTED. PDPosition/Torque/Velocity (slice 1),
// ComputedTorque + Actuator (slice 2) are implemented; Osc (4) is still RESERVED
// for the Phase-3 solver slice and is rejected at the ABI / world-construction
// boundary rather than silently falling back to PD. The set is non-contiguous
// ({0,1,2,3,5}), so this is NOT a simple `<= threshold` (Osc=4 is the hole).
inline constexpr bool IsControlModeImplemented(uint8_t mode) {
    return mode <= static_cast<uint8_t>(ControlMode::ComputedTorque) ||
           mode == static_cast<uint8_t>(ControlMode::Actuator);
}

} // namespace nuka::runtime::articulation
