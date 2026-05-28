#pragma once

#include "nuka/nuka.h"

#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "phi/owned_stream.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/world_builder.hpp"
#include "runtime/world_stepper.hpp"

#include <memory>
#include <string>
#include <vector>

namespace nuka::c_abi {

struct DeviceRecord {
    phi::DeviceContext context;
    std::unique_ptr<phi::OwnedStream> owned_stream;
};

struct WorldRecord {
    DeviceRecord* device = nullptr;
    runtime::BuiltWorld world;
    runtime::WorldStepOptions step_options;
    runtime::articulation::ArticulationHostState articulation_host;
    runtime::articulation::ArticulationDeviceBuffers articulation_device;
    phi::Buffer joint_position_buffer;
    phi::Buffer joint_velocity_buffer;
    std::vector<float> empty_float_storage;
};

struct BufferRecord {
    phi::Buffer buffer;
};

nuka_result_t RefreshWorldBuffers(WorldRecord& record) noexcept;

} // namespace nuka::c_abi
