// ---------------------------------------------------------------------------
// Tests for nuka::sensor -- Lidar / ray sensors
// ---------------------------------------------------------------------------

#include "sensor/ray_sensor.hpp"

#include <gtest/gtest.h>

using namespace nuka::sensor;
using namespace nuka::math;

// ---------------------------------------------------------------------------
// Lidar packet has depth data
// ---------------------------------------------------------------------------

TEST(LidarSensor, PacketHasDepthData) {
    const auto pkt = QueryLidarSensor(
        Vec3{0.0f, 1.0f, 0.0f},
        Vec3{1.0f, 0.0f, 0.0f},
        64,
        50.0f
    );

    EXPECT_TRUE(pkt.has_depth_data);
    EXPECT_TRUE(pkt.has_position);
}

// ---------------------------------------------------------------------------
// Ray count matches requested count
// ---------------------------------------------------------------------------

TEST(LidarSensor, RayCountMatches) {
    const uint32_t ray_count = 128;
    const auto pkt = QueryLidarSensor(
        Vec3::Zero(),
        Vec3::UnitX(),
        ray_count,
        100.0f
    );

    ASSERT_EQ(pkt.depth_buffer.size(), static_cast<size_t>(ray_count));
}

// ---------------------------------------------------------------------------
// Default depth values equal range (no intersections)
// ---------------------------------------------------------------------------

TEST(LidarSensor, DefaultDepthEqualsRange) {
    const float range = 75.0f;
    const auto pkt = QueryLidarSensor(Vec3::Zero(), Vec3::UnitZ(), 16, range);

    for (const auto& d : pkt.depth_buffer) {
        EXPECT_FLOAT_EQ(d, range);
    }
}
