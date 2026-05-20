// ---------------------------------------------------------------------------
// nuka::runtime::gpu::DeviceWorld implementation
// ---------------------------------------------------------------------------

#include "runtime/gpu/device_world.hpp"

#include <algorithm>
#include <vector>

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

} // namespace

DeviceWorld::DeviceWorld(uint32_t body_count,
                         uint32_t shape_count,
                         uint32_t joint_count,
                         uint32_t actuator_count,
                         phi::Buffer body_poses,
                         phi::Buffer body_inv_masses,
                         phi::Buffer body_inv_inertias,
                         phi::Buffer shape_body_ids,
                         phi::Buffer joint_child_bodies,
                         phi::Buffer actuator_gains,
                         phi::Buffer actuator_force_limits)
    : body_count_(body_count)
    , shape_count_(shape_count)
    , joint_count_(joint_count)
    , actuator_count_(actuator_count)
    , body_poses_(std::move(body_poses))
    , body_inv_masses_(std::move(body_inv_masses))
    , body_inv_inertias_(std::move(body_inv_inertias))
    , shape_body_ids_(std::move(shape_body_ids))
    , joint_child_bodies_(std::move(joint_child_bodies))
    , actuator_gains_(std::move(actuator_gains))
    , actuator_force_limits_(std::move(actuator_force_limits)) {}

std::size_t DeviceWorld::DeviceBytes() const {
    return body_poses_.Size()
        + body_inv_masses_.Size()
        + body_inv_inertias_.Size()
        + shape_body_ids_.Size()
        + joint_child_bodies_.Size()
        + actuator_gains_.Size()
        + actuator_force_limits_.Size()
        + state_poses_.Size()
        + state_linear_velocities_.Size()
        + state_angular_velocities_.Size()
        + state_forces_.Size()
        + state_torques_.Size();
}

bool DeviceWorld::HasUploadedState() const {
    if (body_count_ == 0) {
        return true;
    }

    return state_poses_.Size() == body_count_ * sizeof(math::Transform)
        && state_linear_velocities_.Size() == body_count_ * sizeof(math::Vec3)
        && state_angular_velocities_.Size() == body_count_ * sizeof(math::Vec3)
        && state_forces_.Size() == body_count_ * sizeof(math::Vec3)
        && state_torques_.Size() == body_count_ * sizeof(math::Vec3);
}

DeviceWorldSummary DeviceWorld::DownloadSummary() const {
    DeviceWorldSummary summary;
    summary.body_count = body_count_;
    summary.shape_count = shape_count_;
    summary.joint_count = joint_count_;
    summary.actuator_count = actuator_count_;

    if (body_count_ > 0) {
        const auto poses = DownloadVector<math::Transform>(body_poses_, body_count_);
        summary.first_body_position = poses.front().position;
    }

    if (body_count_ > 1) {
        const auto inv_masses = DownloadVector<float>(body_inv_masses_, body_count_);
        summary.second_body_inv_mass = inv_masses[1];
    }

    if (shape_count_ > 0) {
        const auto body_ids = DownloadVector<scene::BodyId>(shape_body_ids_, shape_count_);
        summary.first_shape_body_id = body_ids.front();
    }

    if (joint_count_ > 0) {
        const auto child_bodies =
            DownloadVector<scene::BodyId>(joint_child_bodies_, joint_count_);
        summary.first_joint_child_body = child_bodies.front();
    }

    if (actuator_count_ > 0) {
        const auto gains = DownloadVector<float>(actuator_gains_, actuator_count_);
        const auto force_limits =
            DownloadVector<float>(actuator_force_limits_, actuator_count_);
        summary.first_actuator_gain = gains.front();
        summary.first_actuator_force_limit = force_limits.front();
    }

    return summary;
}

DeviceStateSnapshot DeviceWorld::DownloadStateSnapshot() const {
    DeviceStateSnapshot snapshot;
    snapshot.body_count = body_count_;
    if (body_count_ == 0) {
        return snapshot;
    }

    const auto poses = DownloadVector<math::Transform>(state_poses_, body_count_);
    const auto linear_velocities =
        DownloadVector<math::Vec3>(state_linear_velocities_, body_count_);
    const auto angular_velocities =
        DownloadVector<math::Vec3>(state_angular_velocities_, body_count_);
    const auto forces = DownloadVector<math::Vec3>(state_forces_, body_count_);
    const auto torques = DownloadVector<math::Vec3>(state_torques_, body_count_);

    snapshot.first_body_position = poses.front().position;
    snapshot.first_linear_velocity = linear_velocities.front();
    snapshot.first_angular_velocity = angular_velocities.front();
    snapshot.first_force = forces.front();
    snapshot.first_torque = torques.front();
    return snapshot;
}

DeviceState DeviceWorld::DownloadState() const {
    DeviceState state;
    state.body_count = body_count_;
    if (body_count_ == 0) {
        return state;
    }

    state.poses = DownloadVector<math::Transform>(state_poses_, body_count_);
    state.linear_velocities =
        DownloadVector<math::Vec3>(state_linear_velocities_, body_count_);
    state.angular_velocities =
        DownloadVector<math::Vec3>(state_angular_velocities_, body_count_);
    state.forces = DownloadVector<math::Vec3>(state_forces_, body_count_);
    state.torques = DownloadVector<math::Vec3>(state_torques_, body_count_);
    return state;
}

void DeviceWorld::UploadState(const WorldInstance& instance) {
    state_poses_ = UploadVector(instance.poses);
    state_linear_velocities_ = UploadVector(instance.linear_velocities);
    state_angular_velocities_ = UploadVector(instance.angular_velocities);
    state_forces_ = UploadVector(instance.forces);
    state_torques_ = UploadVector(instance.torques);
}

math::Transform* DeviceWorld::DevicePoses() {
    return static_cast<math::Transform*>(state_poses_.Data());
}

math::Vec3* DeviceWorld::DeviceLinearVelocities() {
    return static_cast<math::Vec3*>(state_linear_velocities_.Data());
}

math::Vec3* DeviceWorld::DeviceAngularVelocities() {
    return static_cast<math::Vec3*>(state_angular_velocities_.Data());
}

math::Vec3* DeviceWorld::DeviceForces() {
    return static_cast<math::Vec3*>(state_forces_.Data());
}

math::Vec3* DeviceWorld::DeviceTorques() {
    return static_cast<math::Vec3*>(state_torques_.Data());
}

const float* DeviceWorld::DeviceInvMasses() const {
    return static_cast<const float*>(body_inv_masses_.Data());
}

const math::Vec3* DeviceWorld::DeviceInvInertias() const {
    return static_cast<const math::Vec3*>(body_inv_inertias_.Data());
}

DeviceWorld UploadDeviceWorld(const WorldTemplate& world_template) {
    return DeviceWorld(
        world_template.body_count,
        world_template.shape_count,
        world_template.joint_count,
        world_template.actuator_count,
        UploadVector(world_template.body_table.poses),
        UploadVector(world_template.body_table.inv_masses),
        UploadVector(world_template.body_table.inv_inertias),
        UploadVector(world_template.shape_table.body_ids),
        UploadVector(world_template.joint_table.child_bodies),
        UploadVector(world_template.actuator_table.gains),
        UploadVector(world_template.actuator_table.force_limits));
}

void UploadDeviceState(DeviceWorld& device_world, const WorldInstance& instance) {
    device_world.UploadState(instance);
}

} // namespace nuka::runtime::gpu
