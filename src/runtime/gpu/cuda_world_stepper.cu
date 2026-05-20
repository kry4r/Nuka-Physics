// ---------------------------------------------------------------------------
// nuka::runtime::gpu::cuda_world_stepper implementation
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_world_stepper.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::runtime::gpu {

namespace {

__device__ math::Vec3 ZeroVec3() {
    math::Vec3 v;
    v.x = 0.0f;
    v.y = 0.0f;
    v.z = 0.0f;
    return v;
}

__device__ math::Vec3 Add(math::Vec3 a, math::Vec3 b) {
    math::Vec3 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
}

__device__ math::Vec3 Scale(math::Vec3 v, float s) {
    math::Vec3 result;
    result.x = v.x * s;
    result.y = v.y * s;
    result.z = v.z * s;
    return result;
}

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

CudaWorldStepReport StepCudaWorld(DeviceWorld& device_world,
                                  const CudaWorldStepOptions& options) {
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
        IntegrateRigidBodiesKernel<<<block_count, kBlockSize>>>(
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

    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    report.simulated_step_count = options.step_count;
    return report;
}

} // namespace nuka::runtime::gpu
