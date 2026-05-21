#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::gpu::BatchedDeviceWorld -- CUDA batched mutable world state
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "runtime/world_instance.hpp"
#include "runtime/world_template.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::gpu {

struct CudaBatchedWorldStepOptions {
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    float dt = 1.0f / 60.0f;
    uint32_t step_count = 1;
    bool clear_forces_after_step = true;
};

struct CudaBatchedWorldStepReport {
    uint32_t simulated_step_count = 0;
    uint32_t instance_count = 0;
    uint32_t body_count_per_instance = 0;
    uint32_t total_body_count = 0;
    uint32_t kernel_launch_count = 0;
};

struct BatchedDeviceState {
    uint32_t instance_count = 0;
    uint32_t body_count_per_instance = 0;
    std::vector<math::Transform> poses;
    std::vector<math::Vec3> linear_velocities;
    std::vector<math::Vec3> angular_velocities;
    std::vector<math::Vec3> forces;
    std::vector<math::Vec3> torques;
};

class BatchedDeviceWorld {
public:
    BatchedDeviceWorld() = default;
    BatchedDeviceWorld(uint32_t instance_count,
                       uint32_t body_count_per_instance,
                       phi::Buffer body_inv_masses,
                       phi::Buffer body_inv_inertias,
                       phi::Buffer poses,
                       phi::Buffer linear_velocities,
                       phi::Buffer angular_velocities,
                       phi::Buffer forces,
                       phi::Buffer torques);

    BatchedDeviceWorld(const BatchedDeviceWorld&) = delete;
    BatchedDeviceWorld& operator=(const BatchedDeviceWorld&) = delete;
    BatchedDeviceWorld(BatchedDeviceWorld&&) noexcept = default;
    BatchedDeviceWorld& operator=(BatchedDeviceWorld&&) noexcept = default;

    uint32_t InstanceCount() const { return instance_count_; }
    uint32_t BodyCountPerInstance() const { return body_count_per_instance_; }
    uint32_t TotalBodyCount() const { return instance_count_ * body_count_per_instance_; }
    bool HasUploadedState() const;

    BatchedDeviceState DownloadState() const;

    math::Transform* DevicePoses();
    math::Vec3* DeviceLinearVelocities();
    math::Vec3* DeviceAngularVelocities();
    math::Vec3* DeviceForces();
    math::Vec3* DeviceTorques();
    const float* DeviceInvMasses() const;
    const math::Vec3* DeviceInvInertias() const;

private:
    uint32_t instance_count_ = 0;
    uint32_t body_count_per_instance_ = 0;

    phi::Buffer body_inv_masses_;
    phi::Buffer body_inv_inertias_;
    phi::Buffer poses_;
    phi::Buffer linear_velocities_;
    phi::Buffer angular_velocities_;
    phi::Buffer forces_;
    phi::Buffer torques_;
};

BatchedDeviceWorld UploadBatchedDeviceWorld(
    const WorldTemplate& world_template,
    const std::vector<WorldInstance>& instances);

CudaBatchedWorldStepReport StepBatchedCudaWorld(
    BatchedDeviceWorld& batch,
    const CudaBatchedWorldStepOptions& options = {});

} // namespace nuka::runtime::gpu
