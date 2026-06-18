#pragma once
// ---------------------------------------------------------------------------
// nuka::sensor -- Ray-based sensor queries (lidar, depth)
// ---------------------------------------------------------------------------

#include "sensor/sensor_packet.hpp"
#include "collision/static_bvh.hpp"

#include <cstdint>

namespace nuka::sensor {

/// Query a lidar sensor, producing a depth buffer of ray_count entries by
/// casting a horizontal fan of rays centred on `direction` against `scene`
/// (real slab-based ray-vs-AABB). Each entry is the genuine hit distance,
/// clamped to `range`, or `range` on a miss. With scene == nullptr the world is
/// empty, so every ray honestly reaches max range (not a canned fake).
SensorPacket QueryLidarSensor(math::Vec3 origin,
                              math::Vec3 direction,
                              uint32_t   ray_count,
                              float      range,
                              const collision::StaticBVH* scene = nullptr);

} // namespace nuka::sensor
