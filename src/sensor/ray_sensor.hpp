#pragma once
// ---------------------------------------------------------------------------
// nuka::sensor -- Ray-based sensor queries (lidar, depth)
// ---------------------------------------------------------------------------

#include "sensor/sensor_packet.hpp"

#include <cstdint>

namespace nuka::sensor {

/// Query a lidar sensor, producing a depth buffer of ray_count entries.
SensorPacket QueryLidarSensor(math::Vec3 origin,
                              math::Vec3 direction,
                              uint32_t   ray_count,
                              float      range);

} // namespace nuka::sensor
