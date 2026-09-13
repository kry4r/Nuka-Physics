#pragma once

#include "rt/render_dr.hpp"
#include "rt/sensor_fidelity.hpp"
#include "sensor/imaging.hpp"

namespace nuka::rt {

struct SensorStateSnapshot {
    sensor::ImagingState imaging;
    RenderDrConfig render_dr;
    SensorFidelityConfig fidelity;
    uint32_t aov_mask = 0u;
    uint32_t width = 0u, height = 0u;
    uint32_t lidar_az = 0u, lidar_el = 0u;
    std::vector<float> color, depth, normal, albedo, range;
    std::vector<uint32_t> prim;
};

}  // namespace nuka::rt
