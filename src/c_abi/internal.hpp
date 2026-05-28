#pragma once

#include "nuka/nuka.h"

#include "core/diagnostics/invariants.hpp"
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
    uint32_t simulated_step_count = 0u;
    core::diagnostics::InvariantSampler invariant_sampler{
        core::diagnostics::InvariantConfig{
            true,
            1u,
            {},
            true,
            true,
        }
    };
    std::vector<core::diagnostics::InvariantSample> last_invariant_violations;
    std::vector<float> empty_float_storage;
};

struct BufferRecord {
    phi::Buffer buffer;
};

nuka_result_t RefreshWorldBuffers(WorldRecord& record) noexcept;

} // namespace nuka::c_abi
