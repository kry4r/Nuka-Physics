#pragma once
// ---------------------------------------------------------------------------
// nuka::sensor -- State-based sensor queries (IMU, joint state)
// ---------------------------------------------------------------------------

#include "sensor/sensor_packet.hpp"
#include "runtime/rigid/body_state.hpp"

namespace nuka::sensor {

/// Build a simple test IMU packet with default values.
SensorPacket BuildTestImuPacket();

/// Query an IMU sensor from a rigid body state.
SensorPacket QueryImuSensor(const runtime::rigid::BodyState& body);

/// Query a joint state sensor from angle and velocity.
SensorPacket QueryJointStateSensor(float joint_angle, float joint_velocity);

} // namespace nuka::sensor
