#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::gpu::DeviceWorld -- CUDA-resident cooked runtime tables
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "runtime/world_instance.hpp"
#include "runtime/world_template.hpp"
#include "scene/canonical_types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nuka::runtime::gpu {

struct DeviceWorldSummary {
    uint32_t body_count = 0;
    uint32_t shape_count = 0;
    uint32_t joint_count = 0;
    uint32_t actuator_count = 0;

    math::Vec3 first_body_position = math::Vec3::Zero();
    float second_body_inv_mass = 0.0f;
    scene::BodyId first_shape_body_id = 0;
    float first_shape_radius = 0.0f;
    float first_shape_half_height = 0.0f;
    scene::BodyId first_joint_child_body = 0;
    float first_actuator_gain = 0.0f;
    float first_actuator_force_limit = 0.0f;
};

struct DeviceStateSnapshot {
    uint32_t body_count = 0;
    math::Vec3 first_body_position = math::Vec3::Zero();
    math::Vec3 first_linear_velocity = math::Vec3::Zero();
    math::Vec3 first_angular_velocity = math::Vec3::Zero();
    math::Vec3 first_force = math::Vec3::Zero();
    math::Vec3 first_torque = math::Vec3::Zero();
};

struct DeviceState {
    uint32_t body_count = 0;
    std::vector<math::Transform> poses;
    std::vector<math::Vec3> linear_velocities;
    std::vector<math::Vec3> angular_velocities;
    std::vector<math::Vec3> forces;
    std::vector<math::Vec3> torques;
};

class DeviceWorld {
public:
    DeviceWorld() = default;

    DeviceWorld(uint32_t body_count,
                uint32_t shape_count,
                uint32_t joint_count,
                uint32_t actuator_count,
                phi::Buffer body_poses,
                phi::Buffer body_inv_masses,
                phi::Buffer body_inv_inertias,
                phi::Buffer shape_types,
                phi::Buffer shape_body_ids,
                phi::Buffer shape_local_transforms,
                phi::Buffer shape_half_extents,
                phi::Buffer shape_radii,
                phi::Buffer shape_half_heights,
                phi::Buffer joint_types,
                phi::Buffer joint_parent_bodies,
                phi::Buffer joint_child_bodies,
                phi::Buffer joint_axes,
                phi::Buffer joint_parent_frames,
                phi::Buffer joint_child_frames,
                phi::Buffer actuator_types,
                phi::Buffer actuator_joint_ids,
                phi::Buffer actuator_gains,
                phi::Buffer actuator_force_limits);

    DeviceWorld(const DeviceWorld&) = delete;
    DeviceWorld& operator=(const DeviceWorld&) = delete;
    DeviceWorld(DeviceWorld&&) noexcept = default;
    DeviceWorld& operator=(DeviceWorld&&) noexcept = default;

    uint32_t BodyCount() const { return body_count_; }
    uint32_t ShapeCount() const { return shape_count_; }
    uint32_t JointCount() const { return joint_count_; }
    uint32_t ActuatorCount() const { return actuator_count_; }

    std::size_t DeviceBytes() const;
    bool HasUploadedState() const;
    DeviceWorldSummary DownloadSummary() const;
    DeviceStateSnapshot DownloadStateSnapshot() const;
    DeviceState DownloadState() const;

    void UploadState(const WorldInstance& instance);

    math::Transform* DevicePoses();
    const math::Transform* DevicePoses() const;
    math::Vec3* DeviceLinearVelocities();
    const math::Vec3* DeviceLinearVelocities() const;
    math::Vec3* DeviceAngularVelocities();
    const math::Vec3* DeviceAngularVelocities() const;
    math::Vec3* DeviceForces();
    const math::Vec3* DeviceForces() const;
    math::Vec3* DeviceTorques();
    const math::Vec3* DeviceTorques() const;
    const float* DeviceInvMasses() const;
    const math::Vec3* DeviceInvInertias() const;
    const scene::ShapeType* DeviceShapeTypes() const;
    const scene::BodyId* DeviceShapeBodyIds() const;
    const math::Transform* DeviceShapeLocalTransforms() const;
    const math::Vec3* DeviceShapeHalfExtents() const;
    const float* DeviceShapeRadii() const;
    const float* DeviceShapeHalfHeights() const;
    const scene::JointType* DeviceJointTypes() const;
    const scene::BodyId* DeviceJointParentBodies() const;
    const scene::BodyId* DeviceJointChildBodies() const;
    const math::Vec3* DeviceJointAxes() const;
    const math::Transform* DeviceJointParentFrames() const;
    const math::Transform* DeviceJointChildFrames() const;
    const scene::ActuatorType* DeviceActuatorTypes() const;
    const scene::JointId* DeviceActuatorJointIds() const;
    const float* DeviceActuatorGains() const;
    const float* DeviceActuatorForceLimits() const;

private:
    uint32_t body_count_ = 0;
    uint32_t shape_count_ = 0;
    uint32_t joint_count_ = 0;
    uint32_t actuator_count_ = 0;

    phi::Buffer body_poses_;
    phi::Buffer body_inv_masses_;
    phi::Buffer body_inv_inertias_;
    phi::Buffer shape_types_;
    phi::Buffer shape_body_ids_;
    phi::Buffer shape_local_transforms_;
    phi::Buffer shape_half_extents_;
    phi::Buffer shape_radii_;
    phi::Buffer shape_half_heights_;
    phi::Buffer joint_types_;
    phi::Buffer joint_parent_bodies_;
    phi::Buffer joint_child_bodies_;
    phi::Buffer joint_axes_;
    phi::Buffer joint_parent_frames_;
    phi::Buffer joint_child_frames_;
    phi::Buffer actuator_types_;
    phi::Buffer actuator_joint_ids_;
    phi::Buffer actuator_gains_;
    phi::Buffer actuator_force_limits_;
    phi::Buffer state_poses_;
    phi::Buffer state_linear_velocities_;
    phi::Buffer state_angular_velocities_;
    phi::Buffer state_forces_;
    phi::Buffer state_torques_;
};

DeviceWorld UploadDeviceWorld(const WorldTemplate& world_template);
void UploadDeviceState(DeviceWorld& device_world, const WorldInstance& instance);

} // namespace nuka::runtime::gpu
