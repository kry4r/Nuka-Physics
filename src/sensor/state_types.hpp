#pragma once

#include <cstdint>

#include "math/transform.hpp"
#include "sensor/observation_types.hpp"
#include "sensor/tactile.hpp"

namespace nuka::sensor {

enum class StateSensorKind : uint32_t { Imu, FramePose, JointState, LinearVelocity, ContactWrench, ForceTorque, Touch, Tactile };
enum class StateSensorMount : uint32_t { Link, Body, Base };

inline constexpr uint32_t kStateSensorErrorChannels = 6u;
inline constexpr uint32_t kStateSensorMaxValues = 7u;

struct StateSensorDesc {
    StateSensorKind kind = StateSensorKind::Imu;
    StateSensorMount mount = StateSensorMount::Link;
    uint32_t index = ~0u;
    uint32_t update_period = 1u;
    math::Transform local_offset;
    double sample_period = 0.0;
    double latency = 0.0;
    double latency_jitter = 0.0;
    float dropout_probability = 0.0f;
    float temperature = 25.0f;
    uint64_t seed = 0u;
    ObservationConfig errors[kStateSensorErrorChannels];
    TactileConfig tactile;
};

inline bool IsContactRegionSensor(StateSensorKind kind) {
    return kind == StateSensorKind::Touch || kind == StateSensorKind::Tactile;
}

// World velocities refer to the frame origin; sensor offsets use rigid-body point kinematics.
struct MotionFrame {
    math::Transform pose;
    math::Vec3 linear_velocity;
    math::Vec3 angular_velocity;
};

// World-space impulse and angular impulse about the midpoint frame origin.
struct WrenchImpulse {
    math::Vec3 linear;
    math::Vec3 angular;
};

struct StateSensorStamp {
    uint64_t sequence = 0u;
    uint64_t acquisitions = 0u;
    uint64_t dropped = 0u;
    double sample_time = 0.0;
    double delivery_time = 0.0;
    uint32_t valid = 0u;
    uint32_t reserved = 0u;
};

struct StateSensorPacket {
    double sample_time = 0.0;
    double available_time = 0.0;
    uint64_t sequence = 0u;
    float values[kStateSensorMaxValues] = {};
    uint32_t reserved = 0u;
};

struct StateSensorRuntime {
    StateSensorStamp stamp;
    double integral[kStateSensorErrorChannels] = {};
    double exposure = 0.0;
    double next_sample_time = 0.0;
    double last_available_time = 0.0;
    math::Quat filtered_orientation;
    uint32_t initialized = 0u;
    uint32_t queue_begin = 0u;
    uint32_t queue_size = 0u;
    uint32_t reserved = 0u;
};

uint32_t StateSensorValueCount(StateSensorKind kind);
bool ValidStateSensorDesc(const StateSensorDesc& desc);

}  // namespace nuka::sensor
