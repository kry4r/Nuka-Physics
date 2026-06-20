#pragma once
// ---------------------------------------------------------------------------
// nuka::sensor::SensorGraph -- Collection of sensors attached to bodies
// ---------------------------------------------------------------------------

#include "sensor/sensor_packet.hpp"
#include "scene/scene_ir.hpp"
#include "runtime/rigid/body_state.hpp"

#include <cstdint>
#include <vector>

namespace nuka::sensor {

// The sensor kind + descriptor are the ONE unified scene IR; the runtime sensor
// graph names them by alias so there is no parallel sensor type.
using SensorType       = scene::SensorType;
using SensorDescriptor = scene::SensorDesc;

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
