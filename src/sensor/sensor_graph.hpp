#pragma once
// ---------------------------------------------------------------------------
// nuka::sensor::SensorGraph -- Collection of sensors attached to bodies
// ---------------------------------------------------------------------------

#include "sensor/sensor_packet.hpp"
#include "math/transform.hpp"
#include "runtime/rigid/body_state.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::sensor {

enum class SensorType : uint8_t {
    IMU,
    JointState,
    ContactSummary,
    Lidar,
    Depth
};

struct SensorDescriptor {
    std::string name;
    SensorType  type;
    uint32_t    attached_body   = 0;
    math::Transform local_transform = math::Transform::Identity();
    uint32_t    ray_count       = 0;        // for lidar
    float       ray_range       = 100.0f;
};

class SensorGraph {
public:
    uint32_t AddSensor(SensorDescriptor desc);
    SensorPacket Query(uint32_t sensor_id) const;
    size_t SensorCount() const;

    // Set body state for querying
    void SetBodyStates(const std::vector<runtime::rigid::BodyState>& states);

private:
    std::vector<SensorDescriptor>       descriptors_;
    std::vector<runtime::rigid::BodyState> body_states_;
};

} // namespace nuka::sensor
