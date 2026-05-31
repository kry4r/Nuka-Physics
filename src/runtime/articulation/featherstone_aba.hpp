#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- CUDA Featherstone ABA public API
// ---------------------------------------------------------------------------

#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"

namespace nuka::runtime::articulation {

class FeatherstoneAba {
public:
    // `defer_velocity_damping`: when true, the drive emits ONLY the stiffness
    // torque Kp*(target-q); the -Kd*qdot damping is deferred and applied IMPLICITLY
    // in the constrained-velocity solve (general implicit joint damping). The
    // batched contact stepper sets this true so the soft-gain stance is stable at
    // the native dt; the single-env oracle path leaves it false (explicit damping,
    // byte-for-byte unchanged trajectory).
    static void ApplyPositionDrives(const phi::DeviceContext& context,
                                    ArticulationDeviceState state,
                                    const float* drive_targets,
                                    const float* drive_stiffness,
                                    const float* drive_damping,
                                    const float* drive_force_limits,
                                    bool defer_velocity_damping = false);
    static void ComputeAccelerations(const phi::DeviceContext& context,
                                     ArticulationDeviceState state,
                                     float gravity_z);
    static void Integrate(const phi::DeviceContext& context,
                          ArticulationDeviceState state,
                          float dt);
    // Split halves of Integrate(): IntegrateVelocity does qdot += qddot*dt,
    // IntegratePosition does q += qdot*dt. Running velocity then position is
    // bit-for-bit equal to the combined Integrate(); the batched stepper uses
    // the split so the contact solve sits between the two halves. The single-env
    // path keeps using the combined Integrate().
    static void IntegrateVelocity(const phi::DeviceContext& context,
                                  ArticulationDeviceState state,
                                  float dt);
    static void IntegratePosition(const phi::DeviceContext& context,
                                  ArticulationDeviceState state,
                                  float dt);
    // T8a: floating-base integrators. For a FloatingBase root (parent ==
    // kInvalidLink), IntegrateFloatingBaseVelocity does
    // link_velocity[root] += link_acceleration[root]*dt (the 6-DOF base spatial
    // velocity); IntegrateFloatingBasePose advances base_pose[articulation]
    // (position from the body-frame linear velocity rotated to world, orientation
    // via a normalized first-order quaternion update). Both are no-ops for fixed/
    // kinematic roots, so callers may invoke them unconditionally. Placed
    // analogously to the velocity/position split: velocity pre-contact, pose
    // post-contact.
    static void IntegrateFloatingBaseVelocity(const phi::DeviceContext& context,
                                              ArticulationDeviceState state,
                                              float dt,
                                              float gravity_z);
    static void IntegrateFloatingBasePose(const phi::DeviceContext& context,
                                          ArticulationDeviceState state,
                                          float dt);
};

} // namespace nuka::runtime::articulation
