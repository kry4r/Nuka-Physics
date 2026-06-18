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

SensorPacket QueryImuSensor(const runtime::rigid::BodyState& body,
                            math::Vec3 gravity) {
    SensorPacket pkt;
    pkt.has_linear_acceleration = true;
    pkt.has_angular_velocity    = true;
    pkt.has_position            = true;

    // Accelerometer specific force: applied force minus gravity (reads +g up at
    // rest). Gravity bypasses body.force in the integrator, so subtract it here.
    pkt.linear_acceleration = body.force * body.inv_mass - gravity;
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
