// ---------------------------------------------------------------------------
// nuka::runtime::gpu::cuda_world_stepper implementation
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_world_stepper.hpp"

#include "math/cuda_vec_ops.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::runtime::gpu {

namespace {

// ZeroVec3 / Add / Scale now come from the shared device math library
// (math/cuda_vec_ops.cuh); the former field-assignment local copies were
// bit-identical to the shared brace-init bodies.
namespace mg = ::nuka::math::gpu;
using mg::Add;
using mg::Scale;
using mg::ZeroVec3;

__global__ void IntegrateRigidBodiesKernel(uint32_t body_count,
                                           math::Transform* poses,
                                           math::Vec3* linear_velocities,
                                           math::Vec3* angular_velocities,
                                           math::Vec3* forces,
                                           math::Vec3* torques,
                                           const float* inv_masses,
                                           const math::Vec3* inv_inertias,
                                           math::Vec3 gravity,
                                           float dt,
                                           bool clear_forces_after_step) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= body_count) {
        return;
    }

    const float inv_mass = inv_masses[index];
    if (inv_mass <= 0.0f) {
        if (clear_forces_after_step) {
            forces[index] = ZeroVec3();
            torques[index] = ZeroVec3();
        }
        return;
    }

    math::Vec3 linear_velocity = linear_velocities[index];
    linear_velocity = Add(linear_velocity, Scale(gravity, dt));
    linear_velocity = Add(linear_velocity, Scale(forces[index], inv_mass * dt));

    math::Vec3 angular_velocity = angular_velocities[index];
    const math::Vec3 inv_inertia = inv_inertias[index];
    angular_velocity.x += torques[index].x * inv_inertia.x * dt;
    angular_velocity.y += torques[index].y * inv_inertia.y * dt;
    angular_velocity.z += torques[index].z * inv_inertia.z * dt;

    poses[index].position = Add(poses[index].position, Scale(linear_velocity, dt));

    linear_velocities[index] = linear_velocity;
    angular_velocities[index] = angular_velocity;

    if (clear_forces_after_step) {
        forces[index] = ZeroVec3();
        torques[index] = ZeroVec3();
    }
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) +
                                 " failed: " +
                                 cudaGetErrorString(result));
    }
}

} // namespace

CudaWorldStepReport StepCudaWorld(const phi::DeviceContext& context,
                                  DeviceWorld& device_world,
                                  const CudaWorldStepOptions& options) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    CudaWorldStepReport report;
    report.body_count = device_world.BodyCount();

    if (options.dt <= 0.0f || options.step_count == 0 || device_world.BodyCount() == 0) {
        return report;
    }

    if (!device_world.HasUploadedState()) {
        throw std::runtime_error(
            "StepCudaWorld requires UploadDeviceState before stepping a non-empty DeviceWorld");
    }

    constexpr uint32_t kBlockSize = 128;
    const uint32_t block_count =
        (device_world.BodyCount() + kBlockSize - 1u) / kBlockSize;

    for (uint32_t step = 0; step < options.step_count; ++step) {
        IntegrateRigidBodiesKernel<<<block_count, kBlockSize, 0, stream>>>(
            device_world.BodyCount(),
            device_world.DevicePoses(),
            device_world.DeviceLinearVelocities(),
            device_world.DeviceAngularVelocities(),
            device_world.DeviceForces(),
            device_world.DeviceTorques(),
            device_world.DeviceInvMasses(),
            device_world.DeviceInvInertias(),
            options.gravity,
            options.dt,
            options.clear_forces_after_step);

        CheckCuda(cudaGetLastError(), "IntegrateRigidBodiesKernel launch");
        ++report.kernel_launch_count;
    }

    context.stream.Synchronize();
    report.simulated_step_count = options.step_count;
    return report;
}

CudaWorldStepReport StepCudaWorld(DeviceWorld& device_world,
                                  const CudaWorldStepOptions& options) {
    auto context = phi::MakeDefaultDeviceContext();
    return StepCudaWorld(context, device_world, options);
}

CudaWorldStepReport StepCudaWorld(const phi::DeviceContext& context,
                                  DeviceWorld& device_world,
                                  soft::XpbdWorld& xpbd_world,
                                  const CudaWorldStepOptions& options) {
    // The rigid step is run UNCHANGED -- bit-for-bit identical to the rigid-only
    // overload above. The XPBD co-step is purely additive and runs only when the
    // attached world actually has particles (inert-when-empty).
    CudaWorldStepReport report = StepCudaWorld(context, device_world, options);
    report.xpbd_particle_count = xpbd_world.ParticleCount();

    if (xpbd_world.ParticleCount() == 0u || options.dt <= 0.0f ||
        options.step_count == 0u) {
        return report;  // no XPBD work: rigid result is byte-identical.
    }

    soft::XpbdStepOptions xpbd_options;
    xpbd_options.gravity = options.gravity;
    xpbd_options.dt = options.dt;
    xpbd_options.step_count = options.step_count;
    const soft::XpbdStepReport xpbd_report =
        soft::StepXpbdWorld(context, xpbd_world, xpbd_options);
    report.xpbd_kernel_launch_count = xpbd_report.kernel_launch_count;
    return report;
}

} // namespace nuka::runtime::gpu
