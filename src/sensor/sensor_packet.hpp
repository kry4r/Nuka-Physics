#pragma once
// ---------------------------------------------------------------------------
// nuka::sensor::SensorPacket -- Generic sensor data container
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <vector>

namespace nuka::sensor {

struct SensorPacket {
    bool has_linear_acceleration = false;
    bool has_angular_velocity    = false;
    bool has_position            = false;
    bool has_depth_data          = false;

    math::Vec3 linear_acceleration = {};
    math::Vec3 angular_velocity    = {};
    math::Vec3 position            = {};

    std::vector<float> depth_buffer;
};

} // namespace nuka::sensor
