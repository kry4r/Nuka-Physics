// ---------------------------------------------------------------------------
// nuka::runtime::gpu::BatchedDeviceWorld implementation
// ---------------------------------------------------------------------------

#include "runtime/gpu/batched_device_world.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace nuka::runtime::gpu {

namespace {

template <typename T>
phi::Buffer UploadVector(const std::vector<T>& values) {
    phi::Buffer buffer(values.size() * sizeof(T), phi::MemoryKind::Device);
    if (!values.empty()) {
        buffer.CopyFromHost(values.data(), values.size() * sizeof(T));
    }
    return buffer;
}

template <typename T>
std::vector<T> DownloadVector(const phi::Buffer& buffer, uint32_t count) {
    std::vector<T> values(count);
    if (!values.empty()) {
        buffer.CopyToHost(values.data(), values.size() * sizeof(T));
    }
    return values;
}

template <typename T>
std::vector<T> DefaultedCopy(const std::vector<T>& values, uint32_t count, T fallback) {
    std::vector<T> result(count, fallback);
    const uint32_t copied_count =
        std::min<uint32_t>(count, static_cast<uint32_t>(values.size()));
    for (uint32_t index = 0; index < copied_count; ++index) {
        result[index] = values[index];
    }
    return result;
}

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

__global__ void IntegrateBatchedRigidBodiesKernel(uint32_t total_body_count,
                                                  uint32_t body_count_per_instance,
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
    const uint32_t flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_index >= total_body_count) {
        return;
    }

    const uint32_t body_index = flat_index % body_count_per_instance;
    const float inv_mass = inv_masses[body_index];
    if (inv_mass <= 0.0f) {
        if (clear_forces_after_step) {
            forces[flat_index] = ZeroVec3();
            torques[flat_index] = ZeroVec3();
        }
        return;
    }

    math::Vec3 linear_velocity = linear_velocities[flat_index];
    linear_velocity = Add(linear_velocity, Scale(gravity, dt));
    linear_velocity = Add(linear_velocity, Scale(forces[flat_index], inv_mass * dt));

    math::Vec3 angular_velocity = angular_velocities[flat_index];
    const math::Vec3 inv_inertia = inv_inertias[body_index];
    angular_velocity.x += torques[flat_index].x * inv_inertia.x * dt;
    angular_velocity.y += torques[flat_index].y * inv_inertia.y * dt;
    angular_velocity.z += torques[flat_index].z * inv_inertia.z * dt;

    poses[flat_index].position =
        Add(poses[flat_index].position, Scale(linear_velocity, dt));

    linear_velocities[flat_index] = linear_velocity;
    angular_velocities[flat_index] = angular_velocity;

    if (clear_forces_after_step) {
        forces[flat_index] = ZeroVec3();
        torques[flat_index] = ZeroVec3();
    }
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

void ValidateInstanceShape(const WorldTemplate& world_template,
                           const WorldInstance& instance) {
    if (instance.body_count != world_template.body_count ||
        instance.poses.size() != world_template.body_count ||
        instance.linear_velocities.size() != world_template.body_count ||
        instance.angular_velocities.size() != world_template.body_count ||
        instance.forces.size() != world_template.body_count ||
        instance.torques.size() != world_template.body_count) {
        throw std::invalid_argument(
            "UploadBatchedDeviceWorld requires every instance to match the shared template body count");
    }
}

} // namespace

BatchedDeviceWorld::BatchedDeviceWorld(uint32_t instance_count,
                                       uint32_t body_count_per_instance,
                                       phi::Buffer body_inv_masses,
                                       phi::Buffer body_inv_inertias,
                                       phi::Buffer poses,
                                       phi::Buffer linear_velocities,
                                       phi::Buffer angular_velocities,
                                       phi::Buffer forces,
                                       phi::Buffer torques)
    : instance_count_(instance_count)
    , body_count_per_instance_(body_count_per_instance)
    , body_inv_masses_(std::move(body_inv_masses))
    , body_inv_inertias_(std::move(body_inv_inertias))
    , poses_(std::move(poses))
    , linear_velocities_(std::move(linear_velocities))
    , angular_velocities_(std::move(angular_velocities))
    , forces_(std::move(forces))
    , torques_(std::move(torques)) {}

bool BatchedDeviceWorld::HasUploadedState() const {
    const auto total_body_count = TotalBodyCount();
    if (total_body_count == 0) {
        return true;
    }

    return poses_.Size() == total_body_count * sizeof(math::Transform)
        && linear_velocities_.Size() == total_body_count * sizeof(math::Vec3)
        && angular_velocities_.Size() == total_body_count * sizeof(math::Vec3)
        && forces_.Size() == total_body_count * sizeof(math::Vec3)
        && torques_.Size() == total_body_count * sizeof(math::Vec3);
}

BatchedDeviceState BatchedDeviceWorld::DownloadState() const {
    BatchedDeviceState state;
    state.instance_count = instance_count_;
    state.body_count_per_instance = body_count_per_instance_;
    const uint32_t total_body_count = TotalBodyCount();
    if (total_body_count == 0) {
        return state;
    }

    state.poses = DownloadVector<math::Transform>(poses_, total_body_count);
    state.linear_velocities =
        DownloadVector<math::Vec3>(linear_velocities_, total_body_count);
    state.angular_velocities =
        DownloadVector<math::Vec3>(angular_velocities_, total_body_count);
    state.forces = DownloadVector<math::Vec3>(forces_, total_body_count);
    state.torques = DownloadVector<math::Vec3>(torques_, total_body_count);
    return state;
}

math::Transform* BatchedDeviceWorld::DevicePoses() {
    return static_cast<math::Transform*>(poses_.Data());
}

math::Vec3* BatchedDeviceWorld::DeviceLinearVelocities() {
    return static_cast<math::Vec3*>(linear_velocities_.Data());
}

math::Vec3* BatchedDeviceWorld::DeviceAngularVelocities() {
    return static_cast<math::Vec3*>(angular_velocities_.Data());
}

math::Vec3* BatchedDeviceWorld::DeviceForces() {
    return static_cast<math::Vec3*>(forces_.Data());
}

math::Vec3* BatchedDeviceWorld::DeviceTorques() {
    return static_cast<math::Vec3*>(torques_.Data());
}

const float* BatchedDeviceWorld::DeviceInvMasses() const {
    return static_cast<const float*>(body_inv_masses_.Data());
}

const math::Vec3* BatchedDeviceWorld::DeviceInvInertias() const {
    return static_cast<const math::Vec3*>(body_inv_inertias_.Data());
}

BatchedDeviceWorld UploadBatchedDeviceWorld(
    const WorldTemplate& world_template,
    const std::vector<WorldInstance>& instances) {
    for (const auto& instance : instances) {
        ValidateInstanceShape(world_template, instance);
    }

    const uint32_t instance_count = static_cast<uint32_t>(instances.size());
    const uint32_t body_count = world_template.body_count;
    const uint32_t total_body_count = instance_count * body_count;

    std::vector<math::Transform> poses;
    std::vector<math::Vec3> linear_velocities;
    std::vector<math::Vec3> angular_velocities;
    std::vector<math::Vec3> forces;
    std::vector<math::Vec3> torques;
    poses.reserve(total_body_count);
    linear_velocities.reserve(total_body_count);
    angular_velocities.reserve(total_body_count);
    forces.reserve(total_body_count);
    torques.reserve(total_body_count);

    for (const auto& instance : instances) {
        poses.insert(poses.end(), instance.poses.begin(), instance.poses.end());
        linear_velocities.insert(linear_velocities.end(),
                                 instance.linear_velocities.begin(),
                                 instance.linear_velocities.end());
        angular_velocities.insert(angular_velocities.end(),
                                  instance.angular_velocities.begin(),
                                  instance.angular_velocities.end());
        forces.insert(forces.end(), instance.forces.begin(), instance.forces.end());
        torques.insert(torques.end(), instance.torques.begin(), instance.torques.end());
    }

    const auto inv_masses =
        DefaultedCopy(world_template.body_table.inv_masses, body_count, 0.0f);
    const auto inv_inertias =
        DefaultedCopy(world_template.body_table.inv_inertias,
                      body_count,
                      math::Vec3::Zero());

    return BatchedDeviceWorld(instance_count,
                              body_count,
                              UploadVector(inv_masses),
                              UploadVector(inv_inertias),
                              UploadVector(poses),
                              UploadVector(linear_velocities),
                              UploadVector(angular_velocities),
                              UploadVector(forces),
                              UploadVector(torques));
}

CudaBatchedWorldStepReport StepBatchedCudaWorld(
    BatchedDeviceWorld& batch,
    const CudaBatchedWorldStepOptions& options) {
    CudaBatchedWorldStepReport report;
    report.instance_count = batch.InstanceCount();
    report.body_count_per_instance = batch.BodyCountPerInstance();
    report.total_body_count = batch.TotalBodyCount();

    if (options.dt <= 0.0f || options.step_count == 0 || batch.TotalBodyCount() == 0) {
        return report;
    }

    if (!batch.HasUploadedState()) {
        throw std::runtime_error(
            "StepBatchedCudaWorld requires uploaded BatchedDeviceWorld state");
    }

    constexpr uint32_t kBlockSize = 128;
    const uint32_t block_count = (batch.TotalBodyCount() + kBlockSize - 1u) / kBlockSize;

    for (uint32_t step = 0; step < options.step_count; ++step) {
        IntegrateBatchedRigidBodiesKernel<<<block_count, kBlockSize>>>(
            batch.TotalBodyCount(),
            batch.BodyCountPerInstance(),
            batch.DevicePoses(),
            batch.DeviceLinearVelocities(),
            batch.DeviceAngularVelocities(),
            batch.DeviceForces(),
            batch.DeviceTorques(),
            batch.DeviceInvMasses(),
            batch.DeviceInvInertias(),
            options.gravity,
            options.dt,
            options.clear_forces_after_step);

        CheckCuda(cudaGetLastError(), "IntegrateBatchedRigidBodiesKernel launch");
        ++report.kernel_launch_count;
    }

    CheckCuda(cudaDeviceSynchronize(), "IntegrateBatchedRigidBodiesKernel synchronize");
    report.simulated_step_count = options.step_count;
    return report;
}

} // namespace nuka::runtime::gpu
