#pragma once
// The scene's sensors as the Stage, the Properties panel and the viewport overlays show them.

#include "math/transform.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

struct SensorItem {
    std::string name;
    std::string type;   // "camera", "lidar", "imu", ...
    std::string mount;  // the body's node path; empty when it rides no body
    std::string frame;  // "body", "world", "link" or "base"
    bool camera = false;  // a camera or depth camera
    bool lidar = false;   // a lidar or range scan
    math::Transform offset;
    math::Vec3 rotation_deg;  // the offset's rotation as intrinsic XYZ degrees
    float rate_hz = 0.0f;
    uint32_t update_period = 1u;
    uint32_t aov_mask = 0u;
    uint32_t width = 0u, height = 0u;
    float vfov = 0.0f, near_clip = 0.0f, far_clip = 0.0f;
    uint32_t az_count = 0u, el_count = 0u;
    float az_min = 0.0f, az_max = 0.0f, el_min = 0.0f, el_max = 0.0f;
    float min_range = 0.0f, max_range = 0.0f;
    bool posed = false;  // `pose` holds where the shown environment has it this frame
    math::Transform pose;
};

struct SensorOverlays {
    bool on = true;
    bool frustums = true;  // camera view volumes
    bool fans = true;      // lidar ray fans
    bool frames = true;    // state sensors' axes
    bool labels = true;
};

struct SensorView {
    std::vector<SensorItem> items;  // in the scene's sensor order
    std::vector<std::string> bodies;  // every body's node path, the mounts a sensor can take
    uint64_t version = 0u;          // bumped whenever `items` is rebuilt
    int selected = -1;              // by index into `items`
    SensorOverlays overlays;
};

// A sensor type the editor offers: its record name, menu label and icon.
struct SensorKind {
    const char* type;
    const char* label;
    const char* glyph;
};
const std::vector<SensorKind>& SensorKinds();
// The entry of a record type name, which every sensor type has.
const SensorKind& FindSensorKind(const std::string& type);

}  // namespace nuka::runtime::app::viewer
