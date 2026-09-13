#include "sensor/state_sensor.hpp"

#include <gtest/gtest.h>

using namespace nuka::sensor;

TEST(ImuSensor, FreeFallHasZeroSpecificForce) {
    MotionFrame before, after;
    const nuka::math::Vec3 gravity{0.0f, 0.0f, -9.81f};
    after.linear_velocity = gravity * 0.01f;
    const auto packet = QueryImuSensor(before, after, 0.01, gravity);
    EXPECT_NEAR(packet.linear_acceleration.Length(), 0.0f, 1.0e-5f);
}

TEST(ImuSensor, SupportedRestUsesSensorAxes) {
    MotionFrame frame;
    const auto rotation = nuka::math::Quat::FromAxisAngle({1.0f, 0.0f, 0.0f}, 1.57079632679f);
    const auto packet = QueryImuSensor(frame, frame, 0.01, {0.0f, 0.0f, -9.81f}, {{}, rotation});
    EXPECT_NEAR(packet.linear_acceleration.y, 9.81f, 1.0e-4f);
    EXPECT_NEAR(packet.linear_acceleration.z, 0.0f, 1.0e-4f);
}

TEST(ImuSensor, JointStateSensor) {
    const auto packet = QueryJointStateSensor(1.57f, 0.5f);
    EXPECT_TRUE(packet.has_position);
    EXPECT_TRUE(packet.has_angular_velocity);
    EXPECT_NEAR(packet.position.x, 1.57f, 1.0e-5f);
    EXPECT_NEAR(packet.angular_velocity.x, 0.5f, 1.0e-5f);
}
