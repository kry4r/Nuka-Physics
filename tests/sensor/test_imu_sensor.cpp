// ---------------------------------------------------------------------------
// Tests for nuka::sensor -- IMU and joint state sensors
// ---------------------------------------------------------------------------

#include "sensor/state_sensor.hpp"

#include <gtest/gtest.h>

using namespace nuka::sensor;

// ---------------------------------------------------------------------------
// BuildTestImuPacket has linear acceleration
// ---------------------------------------------------------------------------

TEST(ImuSensor, BuildTestImuPacketHasLinearAcceleration) {
    const auto pkt = BuildTestImuPacket();
    EXPECT_TRUE(pkt.has_linear_acceleration);
    EXPECT_NEAR(pkt.linear_acceleration.y, -9.81f, 0.01f);
}

// ---------------------------------------------------------------------------
// IMU from body state populates all fields
// ---------------------------------------------------------------------------

TEST(ImuSensor, QueryImuFromBodyState) {
    nuka::runtime::rigid::BodyState body;
    body.inv_mass         = 1.0f;
    body.position         = {1.0f, 2.0f, 3.0f};
    body.angular_velocity = {0.1f, 0.2f, 0.3f};
    body.force            = {0.0f, -9.81f, 0.0f};

    const auto pkt = QueryImuSensor(body);

    EXPECT_TRUE(pkt.has_linear_acceleration);
    EXPECT_TRUE(pkt.has_angular_velocity);
    EXPECT_TRUE(pkt.has_position);
    EXPECT_NEAR(pkt.linear_acceleration.y, -9.81f, 0.01f);
    EXPECT_NEAR(pkt.angular_velocity.y, 0.2f, 1e-5f);
    EXPECT_NEAR(pkt.position.x, 1.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Joint state sensor encodes angle and velocity
// ---------------------------------------------------------------------------

TEST(ImuSensor, JointStateSensor) {
    const auto pkt = QueryJointStateSensor(1.57f, 0.5f);
    EXPECT_TRUE(pkt.has_position);
    EXPECT_TRUE(pkt.has_angular_velocity);
    EXPECT_NEAR(pkt.position.x, 1.57f, 1e-5f);
    EXPECT_NEAR(pkt.angular_velocity.x, 0.5f, 1e-5f);
}
