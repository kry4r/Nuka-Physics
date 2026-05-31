#pragma once

#include "nuka/nuka.h"

#include "core/diagnostics/invariants.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "phi/owned_stream.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/gpu/batched_articulated_world.hpp"
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
    std::vector<float> drive_targets_host;
    std::vector<float> drive_stiffness_host;
    std::vector<float> drive_damping_host;
    std::vector<float> drive_force_limits_host;
    phi::Buffer drive_targets_device;
    phi::Buffer drive_stiffness_device;
    phi::Buffer drive_damping_device;
    phi::Buffer drive_force_limits_device;
    phi::Buffer joint_position_buffer;
    phi::Buffer joint_velocity_buffer;
    uint32_t simulated_step_count = 0u;

    // --- Single-env implicit joint-damping scratch (Option B unification) ---
    // The single-env oracle path (StepWorldGpu) now runs the SAME general
    // implicit joint damping as the batched contacts path. It needs the
    // per-articulation joint-space inertia M and its inverse (M+dt*C)^-1 plus a
    // composite-inertia scratch buffer for the CRBA. Allocated lazily on the
    // first step (mirroring the lazy articulation_device upload) and reused for
    // every step. `single_env_max_dof` is the articulation's DOF count (== the
    // M tile stride); 0 until allocated.
    uint32_t single_env_max_dof = 0u;
    phi::Buffer single_env_inertia_m;
    phi::Buffer single_env_inertia_m_inv;
    phi::Buffer single_env_composite;

    // --- Multi-env batched articulated path (p01-F T7) --------------------
    // When env_count > 1 the world is driven through the batched articulated-
    // with-contacts step path (T6) instead of the single-env Featherstone
    // sequence. `batched` is null for the single-env (env_count == 1) oracle
    // path, which stays byte-for-byte unchanged. The replicated drive buffers
    // tile the base hold drives across all envs (link-major, length
    // env_count * base_link_count) as BatchedArticulatedStepParams requires.
    uint32_t env_count = 1u;
    std::unique_ptr<runtime::gpu::BatchedArticulatedWorld> batched;
    runtime::gpu::BatchedArticulatedStepParams batched_step_params;
    phi::Buffer batched_drive_targets_device;
    phi::Buffer batched_drive_stiffness_device;
    phi::Buffer batched_drive_damping_device;
    phi::Buffer batched_drive_force_limits_device;
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
