// ---------------------------------------------------------------------------
// nuka::runtime::gpu::DeviceWorld implementation
// ---------------------------------------------------------------------------

#include "runtime/gpu/device_world.hpp"

#include <algorithm>
#include <utility>
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
std::vector<T> DefaultedCopy(const std::vector<T>& values, uint32_t count, T fallback) {
    std::vector<T> result(count, fallback);
    const uint32_t copied_count =
        std::min<uint32_t>(count, static_cast<uint32_t>(values.size()));
    for (uint32_t index = 0; index < copied_count; ++index) {
        result[index] = values[index];
    }
    return result;
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
                         phi::Buffer actuator_force_limits)
    : body_count_(body_count)
    , shape_count_(shape_count)
    , joint_count_(joint_count)
    , actuator_count_(actuator_count)
    , body_poses_(std::move(body_poses))
    , body_inv_masses_(std::move(body_inv_masses))
    , body_inv_inertias_(std::move(body_inv_inertias))
    , shape_types_(std::move(shape_types))
    , shape_body_ids_(std::move(shape_body_ids))
    , shape_local_transforms_(std::move(shape_local_transforms))
    , shape_half_extents_(std::move(shape_half_extents))
    , shape_radii_(std::move(shape_radii))
    , shape_half_heights_(std::move(shape_half_heights))
    , joint_types_(std::move(joint_types))
    , joint_parent_bodies_(std::move(joint_parent_bodies))
    , joint_child_bodies_(std::move(joint_child_bodies))
    , joint_axes_(std::move(joint_axes))
    , joint_parent_frames_(std::move(joint_parent_frames))
    , joint_child_frames_(std::move(joint_child_frames))
    , actuator_types_(std::move(actuator_types))
    , actuator_joint_ids_(std::move(actuator_joint_ids))
    , actuator_gains_(std::move(actuator_gains))
    , actuator_force_limits_(std::move(actuator_force_limits)) {}

std::size_t DeviceWorld::DeviceBytes() const {
    return body_poses_.Size()
        + body_inv_masses_.Size()
        + body_inv_inertias_.Size()
        + shape_types_.Size()
        + shape_body_ids_.Size()
        + shape_local_transforms_.Size()
        + shape_half_extents_.Size()
        + shape_radii_.Size()
        + shape_half_heights_.Size()
        + joint_types_.Size()
        + joint_parent_bodies_.Size()
        + joint_child_bodies_.Size()
        + joint_axes_.Size()
        + joint_parent_frames_.Size()
        + joint_child_frames_.Size()
        + actuator_types_.Size()
        + actuator_joint_ids_.Size()
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
        const auto radii = DownloadVector<float>(shape_radii_, shape_count_);
        const auto half_heights = DownloadVector<float>(shape_half_heights_, shape_count_);
        summary.first_shape_body_id = body_ids.front();
        summary.first_shape_radius = radii.front();
        summary.first_shape_half_height = half_heights.front();
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

void DeviceWorld::UploadState(const phi::DeviceContext& context,
                              const WorldInstance& instance) {
    phi::ScopedDeviceGuard guard(context.device_id);
    state_poses_ = UploadVector(instance.poses);
    state_linear_velocities_ = UploadVector(instance.linear_velocities);
    state_angular_velocities_ = UploadVector(instance.angular_velocities);
    state_forces_ = UploadVector(instance.forces);
    state_torques_ = UploadVector(instance.torques);
}

void DeviceWorld::UploadState(const WorldInstance& instance) {
    auto context = phi::MakeDefaultDeviceContext();
    UploadState(context, instance);
}

math::Transform* DeviceWorld::DevicePoses() {
    return static_cast<math::Transform*>(state_poses_.Data());
}

const math::Transform* DeviceWorld::DevicePoses() const {
    return static_cast<const math::Transform*>(state_poses_.Data());
}

math::Vec3* DeviceWorld::DeviceLinearVelocities() {
    return static_cast<math::Vec3*>(state_linear_velocities_.Data());
}

const math::Vec3* DeviceWorld::DeviceLinearVelocities() const {
    return static_cast<const math::Vec3*>(state_linear_velocities_.Data());
}

math::Vec3* DeviceWorld::DeviceAngularVelocities() {
    return static_cast<math::Vec3*>(state_angular_velocities_.Data());
}

const math::Vec3* DeviceWorld::DeviceAngularVelocities() const {
    return static_cast<const math::Vec3*>(state_angular_velocities_.Data());
}

math::Vec3* DeviceWorld::DeviceForces() {
    return static_cast<math::Vec3*>(state_forces_.Data());
}

const math::Vec3* DeviceWorld::DeviceForces() const {
    return static_cast<const math::Vec3*>(state_forces_.Data());
}

math::Vec3* DeviceWorld::DeviceTorques() {
    return static_cast<math::Vec3*>(state_torques_.Data());
}

const math::Vec3* DeviceWorld::DeviceTorques() const {
    return static_cast<const math::Vec3*>(state_torques_.Data());
}

const float* DeviceWorld::DeviceInvMasses() const {
    return static_cast<const float*>(body_inv_masses_.Data());
}

const math::Vec3* DeviceWorld::DeviceInvInertias() const {
    return static_cast<const math::Vec3*>(body_inv_inertias_.Data());
}

const scene::ShapeType* DeviceWorld::DeviceShapeTypes() const {
    return static_cast<const scene::ShapeType*>(shape_types_.Data());
}

const scene::BodyId* DeviceWorld::DeviceShapeBodyIds() const {
    return static_cast<const scene::BodyId*>(shape_body_ids_.Data());
}

const math::Transform* DeviceWorld::DeviceShapeLocalTransforms() const {
    return static_cast<const math::Transform*>(shape_local_transforms_.Data());
}

const math::Vec3* DeviceWorld::DeviceShapeHalfExtents() const {
    return static_cast<const math::Vec3*>(shape_half_extents_.Data());
}

const float* DeviceWorld::DeviceShapeRadii() const {
    return static_cast<const float*>(shape_radii_.Data());
}

const float* DeviceWorld::DeviceShapeHalfHeights() const {
    return static_cast<const float*>(shape_half_heights_.Data());
}

const scene::JointType* DeviceWorld::DeviceJointTypes() const {
    return static_cast<const scene::JointType*>(joint_types_.Data());
}

const scene::BodyId* DeviceWorld::DeviceJointParentBodies() const {
    return static_cast<const scene::BodyId*>(joint_parent_bodies_.Data());
}

const scene::BodyId* DeviceWorld::DeviceJointChildBodies() const {
    return static_cast<const scene::BodyId*>(joint_child_bodies_.Data());
}

const math::Vec3* DeviceWorld::DeviceJointAxes() const {
    return static_cast<const math::Vec3*>(joint_axes_.Data());
}

const math::Transform* DeviceWorld::DeviceJointParentFrames() const {
    return static_cast<const math::Transform*>(joint_parent_frames_.Data());
}

const math::Transform* DeviceWorld::DeviceJointChildFrames() const {
    return static_cast<const math::Transform*>(joint_child_frames_.Data());
}

const scene::ActuatorType* DeviceWorld::DeviceActuatorTypes() const {
    return static_cast<const scene::ActuatorType*>(actuator_types_.Data());
}

const scene::JointId* DeviceWorld::DeviceActuatorJointIds() const {
    return static_cast<const scene::JointId*>(actuator_joint_ids_.Data());
}

const float* DeviceWorld::DeviceActuatorGains() const {
    return static_cast<const float*>(actuator_gains_.Data());
}

const float* DeviceWorld::DeviceActuatorForceLimits() const {
    return static_cast<const float*>(actuator_force_limits_.Data());
}

DeviceWorld UploadDeviceWorld(const phi::DeviceContext& context,
                              const WorldTemplate& world_template) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const auto shape_types =
        DefaultedCopy(world_template.shape_table.types,
                      world_template.shape_count,
                      scene::ShapeType::Box);
    const auto shape_body_ids =
        DefaultedCopy(world_template.shape_table.body_ids,
                      world_template.shape_count,
                      scene::kInvalidBody);
    const auto shape_local_transforms =
        DefaultedCopy(world_template.shape_table.local_transforms,
                      world_template.shape_count,
                      math::Transform::Identity());
    const auto shape_half_extents =
        DefaultedCopy(world_template.shape_table.half_extents,
                      world_template.shape_count,
                      math::Vec3{0.5f, 0.5f, 0.5f});
    const auto shape_radii =
        DefaultedCopy(world_template.shape_table.radii,
                      world_template.shape_count,
                      0.5f);
    const auto shape_half_heights =
        DefaultedCopy(world_template.shape_table.half_heights,
                      world_template.shape_count,
                      0.5f);
    const auto joint_types =
        DefaultedCopy(world_template.joint_table.types,
                      world_template.joint_count,
                      scene::JointType::Revolute);
    const auto joint_parent_bodies =
        DefaultedCopy(world_template.joint_table.parent_bodies,
                      world_template.joint_count,
                      scene::kInvalidBody);
    const auto joint_child_bodies =
        DefaultedCopy(world_template.joint_table.child_bodies,
                      world_template.joint_count,
                      scene::kInvalidBody);
    const auto joint_axes =
        DefaultedCopy(world_template.joint_table.axes,
                      world_template.joint_count,
                      math::Vec3::UnitZ());
    const auto joint_parent_frames =
        DefaultedCopy(world_template.joint_table.parent_frames,
                      world_template.joint_count,
                      math::Transform::Identity());
    const auto joint_child_frames =
        DefaultedCopy(world_template.joint_table.child_frames,
                      world_template.joint_count,
                      math::Transform::Identity());
    const auto actuator_types =
        DefaultedCopy(world_template.actuator_table.types,
                      world_template.actuator_count,
                      scene::ActuatorType::Motor);
    const auto actuator_joint_ids =
        DefaultedCopy(world_template.actuator_table.joint_ids,
                      world_template.actuator_count,
                      scene::kInvalidJoint);
    const auto actuator_gains =
        DefaultedCopy(world_template.actuator_table.gains,
                      world_template.actuator_count,
                      0.0f);
    const auto actuator_force_limits =
        DefaultedCopy(world_template.actuator_table.force_limits,
                      world_template.actuator_count,
                      1.0e6f);

    return DeviceWorld(
        world_template.body_count,
        world_template.shape_count,
        world_template.joint_count,
        world_template.actuator_count,
        UploadVector(world_template.body_table.poses),
        UploadVector(world_template.body_table.inv_masses),
        UploadVector(world_template.body_table.inv_inertias),
        UploadVector(shape_types),
        UploadVector(shape_body_ids),
        UploadVector(shape_local_transforms),
        UploadVector(shape_half_extents),
        UploadVector(shape_radii),
        UploadVector(shape_half_heights),
        UploadVector(joint_types),
        UploadVector(joint_parent_bodies),
        UploadVector(joint_child_bodies),
        UploadVector(joint_axes),
        UploadVector(joint_parent_frames),
        UploadVector(joint_child_frames),
        UploadVector(actuator_types),
        UploadVector(actuator_joint_ids),
        UploadVector(actuator_gains),
        UploadVector(actuator_force_limits));
}

DeviceWorld UploadDeviceWorld(const WorldTemplate& world_template) {
    auto context = phi::MakeDefaultDeviceContext();
    return UploadDeviceWorld(context, world_template);
}

void UploadDeviceState(const phi::DeviceContext& context,
                       DeviceWorld& device_world,
                       const WorldInstance& instance) {
    device_world.UploadState(context, instance);
}

void UploadDeviceState(DeviceWorld& device_world, const WorldInstance& instance) {
    auto context = phi::MakeDefaultDeviceContext();
    UploadDeviceState(context, device_world, instance);
}

} // namespace nuka::runtime::gpu
