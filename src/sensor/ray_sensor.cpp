// ---------------------------------------------------------------------------
// nuka::sensor -- Ray-based sensor implementations
// ---------------------------------------------------------------------------

#include "sensor/ray_sensor.hpp"

#include <cmath>

namespace nuka::sensor {

SensorPacket QueryLidarSensor(math::Vec3 origin,
                              math::Vec3 direction,
                              uint32_t   ray_count,
                              float      range,
                              const collision::StaticBVH* scene) {
    SensorPacket pkt;
    pkt.has_depth_data = true;
    pkt.has_position   = true;
    pkt.position       = origin;
    pkt.depth_buffer.resize(ray_count, range);

    // Build a horizontal scan basis: forward = `direction`, side = forward x up.
    math::Vec3 fwd = direction.Normalized();
    if (fwd.LengthSq() < 1e-12f) fwd = math::Vec3::UnitX();
    math::Vec3 up = math::Vec3::UnitY();
    math::Vec3 side = fwd.Cross(up);
    if (side.LengthSq() < 1e-12f) side = math::Vec3::UnitX();  // fwd ∥ up
    side = side.Normalized();

    // 360-degree horizontal fan; single ray points straight along `direction`.
    constexpr float kTwoPi = 6.2831853071795864769f;
    for (uint32_t i = 0; i < ray_count; ++i) {
        float ang = (ray_count > 1u)
                        ? (static_cast<float>(i) / static_cast<float>(ray_count)) * kTwoPi
                        : 0.0f;
        math::Vec3 dir = fwd * std::cos(ang) + side * std::sin(ang);

        float depth = range;
        if (scene != nullptr) {
            auto hit = scene->Raycast(origin, dir, range);
            if (hit) depth = hit->distance;  // genuine slab hit distance
        }
        pkt.depth_buffer[i] = depth;
    }
    return pkt;
}

} // namespace nuka::sensor
