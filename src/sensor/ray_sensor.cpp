// ---------------------------------------------------------------------------
// nuka::sensor -- Ray-based sensor implementations
// ---------------------------------------------------------------------------

#include "sensor/ray_sensor.hpp"

namespace nuka::sensor {

SensorPacket QueryLidarSensor(math::Vec3 origin,
                              math::Vec3 direction,
                              uint32_t   ray_count,
                              float      range) {
    SensorPacket pkt;
    pkt.has_depth_data = true;
    pkt.has_position   = true;
    pkt.position       = origin;

    // Placeholder: fill depth buffer with max range (no geometry intersection).
    pkt.depth_buffer.resize(ray_count, range);

    (void)direction;  // direction used for orientation in a full implementation
    return pkt;
}

} // namespace nuka::sensor
