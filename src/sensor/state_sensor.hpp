#pragma once
// ---------------------------------------------------------------------------
// nuka::sensor -- State-based sensor queries (IMU, joint state)
// ---------------------------------------------------------------------------

#include "sensor/sensor_packet.hpp"
#include "runtime/rigid/body_state.hpp"

namespace nuka::sensor {

/// Build a simple test IMU packet with default values.
SensorPacket BuildTestImuPacket();

/// Query an IMU sensor from a rigid body state. Returns the accelerometer
/// SPECIFIC FORCE = applied_force * inv_mass - gravity (reaction reads +g up at
/// rest). The integrator adds gravity straight to velocity, so body.force never
/// holds it; callers MUST pass the world gravity to get the true IMU signal.
/// Default gravity {0,0,0} yields coordinate acceleration (no gravity term).
SensorPacket QueryImuSensor(const runtime::rigid::BodyState& body,
                            math::Vec3 gravity = math::Vec3::Zero());

/// Query a joint state sensor from angle and velocity.
SensorPacket QueryJointStateSensor(float joint_angle, float joint_velocity);

} // namespace nuka::sensor
