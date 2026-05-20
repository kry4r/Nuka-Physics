#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::gpu::cuda_world_stepper -- CUDA rigid integration stage
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "runtime/gpu/device_world.hpp"

#include <cstdint>

namespace nuka::runtime::gpu {

struct CudaWorldStepOptions {
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    float dt = 1.0f / 60.0f;
    uint32_t step_count = 1;
    bool clear_forces_after_step = true;
};

struct CudaWorldStepReport {
    uint32_t simulated_step_count = 0;
    uint32_t body_count = 0;
    uint32_t kernel_launch_count = 0;
};

CudaWorldStepReport StepCudaWorld(DeviceWorld& device_world,
                                  const CudaWorldStepOptions& options = {});

} // namespace nuka::runtime::gpu
