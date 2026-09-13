#pragma once
// Host sensor queries use solved frame motion with explicit physical time intervals.

#include "sensor/sensor_packet.hpp"
#include "sensor/state_types.hpp"

namespace nuka::sensor {

// Returns interval-average specific force and angular velocity in the mounted sensor frame.
// Input motion velocities are world-space at each frame origin and include solved contact response.
SensorPacket QueryImuSensor(const MotionFrame& before, const MotionFrame& after,
    double interval, math::Vec3 gravity, math::Transform local_offset = math::Transform::Identity());

/// Query a joint state sensor from angle and velocity.
SensorPacket QueryJointStateSensor(float joint_angle, float joint_velocity);

} // namespace nuka::sensor
