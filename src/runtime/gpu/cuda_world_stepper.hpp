#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::gpu::cuda_world_stepper -- CUDA rigid integration stage
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/gpu/device_world.hpp"
#include "runtime/soft/xpbd_world.hpp"

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
    // v0.7 p09-A: XPBD soft co-simulation. 0 when no XpbdWorld is attached or it
    // has zero particles (the rigid path is then byte-identical to the legacy
    // overloads below -- the inert-when-empty guarantee).
    uint32_t xpbd_particle_count = 0;
    uint32_t xpbd_kernel_launch_count = 0;
};

CudaWorldStepReport StepCudaWorld(const phi::DeviceContext& context,
                                  DeviceWorld& device_world,
                                  const CudaWorldStepOptions& options = {});
CudaWorldStepReport StepCudaWorld(DeviceWorld& device_world,
                                  const CudaWorldStepOptions& options = {});

// v0.7 p09-A: additive rigid + XPBD-soft co-step. Runs the EXACT rigid step
// above, then -- only if the attached XpbdWorld has particles -- runs the GPU
// XPBD predict/solve/correct loop over the same device context. With an empty
// XpbdWorld (ParticleCount()==0) NO XPBD kernels launch and the rigid result /
// kernel_launch_count are byte-identical to the rigid-only overloads (the
// inert-when-empty contract; verified by the D1 + no-regression suites).
CudaWorldStepReport StepCudaWorld(const phi::DeviceContext& context,
                                  DeviceWorld& device_world,
                                  soft::XpbdWorld& xpbd_world,
                                  const CudaWorldStepOptions& options = {});

} // namespace nuka::runtime::gpu
