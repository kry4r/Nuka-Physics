// ---------------------------------------------------------------------------
// nuka::sensor -- State-based sensor implementations
// ---------------------------------------------------------------------------

#include "sensor/state_sensor.hpp"

namespace nuka::sensor {

SensorPacket BuildTestImuPacket() {
    SensorPacket pkt;
    pkt.has_linear_acceleration = true;
    pkt.has_angular_velocity    = true;
    pkt.has_position            = true;
    pkt.linear_acceleration     = {0.0f, -9.81f, 0.0f};
    pkt.angular_velocity        = {0.0f, 0.0f, 0.0f};
    pkt.position                = {0.0f, 0.0f, 0.0f};
    return pkt;
}

SensorPacket QueryImuSensor(const runtime::rigid::BodyState& body) {
    SensorPacket pkt;
    pkt.has_linear_acceleration = true;
    pkt.has_angular_velocity    = true;
    pkt.has_position            = true;

    // Linear acceleration = force * inv_mass  (simplified; gravity not added)
    pkt.linear_acceleration = body.force * body.inv_mass;
    pkt.angular_velocity    = body.angular_velocity;
    pkt.position            = body.position;
    return pkt;
}

SensorPacket QueryJointStateSensor(float joint_angle, float joint_velocity) {
    SensorPacket pkt;
    pkt.has_position         = true;
    pkt.has_angular_velocity = true;

    // Encode joint angle in position.x and velocity in angular_velocity.x
    pkt.position         = {joint_angle, 0.0f, 0.0f};
    pkt.angular_velocity = {joint_velocity, 0.0f, 0.0f};
    return pkt;
}

} // namespace nuka::sensor
