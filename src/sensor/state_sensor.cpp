#include "sensor/state_sensor.hpp"

#include <cmath>
#include <stdexcept>

namespace nuka::sensor {

SensorPacket QueryImuSensor(const MotionFrame& before, const MotionFrame& after,
    double interval, math::Vec3 gravity, math::Transform local_offset) {
    if (!(interval > 0.0) || !std::isfinite(interval)) throw std::invalid_argument("IMU interval must be positive and finite");
    const auto first_pose = before.pose * local_offset;
    const auto last_pose = after.pose * local_offset;
    const auto first_velocity = before.linear_velocity + before.angular_velocity.Cross(
        before.pose.rotation.Rotate(local_offset.position));
    const auto last_velocity = after.linear_velocity + after.angular_velocity.Cross(
        after.pose.rotation.Rotate(local_offset.position));
    const auto a = first_pose.rotation;
    const auto b = last_pose.rotation;
    const float sign = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z < 0.0f ? -1.0f : 1.0f;
    const auto midpoint = math::Quat{a.w + sign * b.w, a.x + sign * b.x,
                                    a.y + sign * b.y, a.z + sign * b.z}.Normalized();
    SensorPacket packet;
    packet.has_linear_acceleration = packet.has_angular_velocity = packet.has_position = true;
    packet.linear_acceleration = midpoint.Conjugate().Rotate(
        (last_velocity - first_velocity) / static_cast<float>(interval) - gravity);
    packet.angular_velocity = midpoint.Conjugate().Rotate(
        (before.angular_velocity + after.angular_velocity) * 0.5f);
    packet.position = last_pose.position;
    return packet;
}

SensorPacket QueryJointStateSensor(float joint_angle, float joint_velocity) {
    SensorPacket packet;
    packet.has_position = packet.has_angular_velocity = true;
    packet.position = {joint_angle, 0.0f, 0.0f};
    packet.angular_velocity = {joint_velocity, 0.0f, 0.0f};
    return packet;
}

}  // namespace nuka::sensor
