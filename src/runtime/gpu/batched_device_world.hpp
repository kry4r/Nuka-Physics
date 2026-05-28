#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::gpu::BatchedDeviceWorld -- CUDA batched mutable world state
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/dynamic_broadphase.hpp"
#include "constraint/constraint_block.hpp"
#include "constraint/contact_manifold.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/gpu/cuda_constraint_row_buffer.hpp"
#include "runtime/world_instance.hpp"
#include "runtime/world_template.hpp"
#include "scene/canonical_types.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::gpu {

struct CudaBatchedWorldStepOptions {
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    float dt = 1.0f / 60.0f;
    uint32_t step_count = 1;
    bool clear_forces_after_step = true;
    bool enable_contacts = false;
    bool enable_joints = false;
    bool enable_drives = false;
    uint32_t solver_velocity_iterations = 10;
    uint32_t solver_position_iterations = 4;
    float solver_slop = 0.005f;
    float solver_baumgarte = 0.2f;
};

struct CudaBatchedWorldStepReport {
    uint32_t simulated_step_count = 0;
    uint32_t instance_count = 0;
    uint32_t body_count_per_instance = 0;
    uint32_t shape_count_per_instance = 0;
    uint32_t total_body_count = 0;
    uint32_t kernel_launch_count = 0;
    uint32_t broadphase_pair_count = 0;
    uint32_t contact_manifold_count = 0;
    uint32_t contact_point_count = 0;
    uint32_t contact_constraint_count = 0;
    uint32_t joint_constraint_count = 0;
    uint32_t drive_constraint_count = 0;
    uint32_t constraint_block_count = 0;
    uint32_t constraint_row_count = 0;
    uint32_t solver_iterations_used = 0;
    float max_constraint_error = 0.0f;
    CudaConstraintRowSchedulerReport row_scheduler_report;
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

struct BatchedShapeTables {
    std::vector<scene::ShapeType> types;
    std::vector<scene::BodyId> body_ids;
    std::vector<math::Transform> local_transforms;
    std::vector<math::Vec3> half_extents;
    std::vector<float> radii;
    std::vector<float> half_heights;
};

struct CudaBatchedContactManifold {
    uint32_t instance_index = 0;
    uint32_t body_a = ~0u;
    uint32_t body_b = ~0u;
    uint32_t point_count = 0;
    float friction = 0.5f;
    float restitution = 0.0f;
    constraint::ContactPoint points[constraint::ContactManifold::kMaxPoints];
};

struct CudaBatchedContactReport {
    uint32_t instance_count = 0;
    uint32_t pair_count = 0;
    uint32_t contact_manifold_count = 0;
    uint32_t contact_point_count = 0;
};

struct CudaBatchedConstraintSolverConfig {
    uint32_t velocity_iterations = 10;
    uint32_t position_iterations = 4;
    float slop = 0.005f;
    float baumgarte = 0.2f;
};

struct CudaBatchedConstraintSolverReport {
    uint32_t constraint_block_count = 0;
    uint32_t constraint_row_count = 0;
    uint32_t contact_constraint_count = 0;
    uint32_t joint_constraint_count = 0;
    uint32_t drive_constraint_count = 0;
    uint32_t velocity_iterations = 0;
    uint32_t position_iterations = 0;
    float max_position_error = 0.0f;
    CudaConstraintRowSchedulerReport row_scheduler_report;
};

class CudaBatchedBroadphaseResult {
public:
    CudaBatchedBroadphaseResult() = default;
    CudaBatchedBroadphaseResult(uint32_t instance_count,
                                uint32_t shape_count_per_instance,
                                uint32_t pair_slot_count_per_instance,
                                phi::Buffer aabbs,
                                phi::Buffer pairs,
                                phi::Buffer pair_instance_indices,
                                phi::Buffer pair_active_flags,
                                phi::Buffer pair_count);

    CudaBatchedBroadphaseResult(const CudaBatchedBroadphaseResult&) = delete;
    CudaBatchedBroadphaseResult& operator=(const CudaBatchedBroadphaseResult&) = delete;
    CudaBatchedBroadphaseResult(CudaBatchedBroadphaseResult&&) noexcept = default;
    CudaBatchedBroadphaseResult& operator=(CudaBatchedBroadphaseResult&&) noexcept = default;

    uint32_t InstanceCount() const { return instance_count_; }
    uint32_t ShapeCountPerInstance() const { return shape_count_per_instance_; }
    uint32_t PairSlotCountPerInstance() const { return pair_slot_count_per_instance_; }
    uint32_t TotalPairSlotCount() const { return instance_count_ * pair_slot_count_per_instance_; }
    uint32_t PairCount() const;

    const collision::AABB* DeviceAabbs() const;
    const collision::CollisionPair* DevicePairs() const;
    const uint32_t* DevicePairInstanceIndices() const;
    const uint8_t* DevicePairActiveFlags() const;
    const uint32_t* DevicePairCount() const;

private:
    uint32_t instance_count_ = 0;
    uint32_t shape_count_per_instance_ = 0;
    uint32_t pair_slot_count_per_instance_ = 0;
    phi::Buffer aabbs_;
    phi::Buffer pairs_;
    phi::Buffer pair_instance_indices_;
    phi::Buffer pair_active_flags_;
    phi::Buffer pair_count_;
};

class CudaBatchedContactResult {
public:
    CudaBatchedContactResult() = default;
    CudaBatchedContactResult(uint32_t total_pair_slot_count,
                             phi::Buffer manifolds,
                             phi::Buffer report);

    CudaBatchedContactResult(const CudaBatchedContactResult&) = delete;
    CudaBatchedContactResult& operator=(const CudaBatchedContactResult&) = delete;
    CudaBatchedContactResult(CudaBatchedContactResult&&) noexcept = default;
    CudaBatchedContactResult& operator=(CudaBatchedContactResult&&) noexcept = default;

    uint32_t TotalPairSlotCount() const { return total_pair_slot_count_; }
    CudaBatchedContactReport DownloadReport() const;
    std::vector<CudaBatchedContactManifold> DownloadManifolds() const;

    const CudaBatchedContactManifold* DeviceManifolds() const;
    const CudaBatchedContactReport* DeviceReport() const;

private:
    uint32_t total_pair_slot_count_ = 0;
    phi::Buffer manifolds_;
    phi::Buffer report_;
};

class CudaBatchedConstraintSolverResult {
public:
    CudaBatchedConstraintSolverResult() = default;
    CudaBatchedConstraintSolverResult(uint32_t block_capacity,
                                      phi::Buffer blocks,
                                      phi::Buffer block_count,
                                      phi::Buffer report);

    CudaBatchedConstraintSolverResult(const CudaBatchedConstraintSolverResult&) = delete;
    CudaBatchedConstraintSolverResult& operator=(const CudaBatchedConstraintSolverResult&) = delete;
    CudaBatchedConstraintSolverResult(CudaBatchedConstraintSolverResult&&) noexcept = default;
    CudaBatchedConstraintSolverResult& operator=(CudaBatchedConstraintSolverResult&&) noexcept = default;

    CudaBatchedConstraintSolverReport DownloadReport() const;
    CudaConstraintRowBufferView ConstraintRowBuffer() const;

private:
    uint32_t block_capacity_ = 0;
    phi::Buffer blocks_;
    phi::Buffer block_count_;
    phi::Buffer report_;
};

class BatchedDeviceWorld {
public:
    BatchedDeviceWorld() = default;
    BatchedDeviceWorld(uint32_t instance_count,
                       uint32_t body_count_per_instance,
                       uint32_t shape_count_per_instance,
                       uint32_t joint_count_per_instance,
                       uint32_t actuator_count_per_instance,
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
                       phi::Buffer actuator_force_limits,
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
    uint32_t ShapeCountPerInstance() const { return shape_count_per_instance_; }
    uint32_t JointCountPerInstance() const { return joint_count_per_instance_; }
    uint32_t ActuatorCountPerInstance() const { return actuator_count_per_instance_; }
    uint32_t TotalBodyCount() const { return instance_count_ * body_count_per_instance_; }
    uint32_t TotalShapeCount() const { return instance_count_ * shape_count_per_instance_; }
    uint32_t TotalJointCount() const { return instance_count_ * joint_count_per_instance_; }
    uint32_t TotalActuatorCount() const { return instance_count_ * actuator_count_per_instance_; }
    bool HasUploadedState() const;
    bool HasUploadedShapeTables() const;
    bool HasUploadedJointTables() const;
    bool HasUploadedActuatorTables() const;

    BatchedDeviceState DownloadState() const;
    BatchedShapeTables DownloadShapeTables() const;

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
    uint32_t instance_count_ = 0;
    uint32_t body_count_per_instance_ = 0;
    uint32_t shape_count_per_instance_ = 0;
    uint32_t joint_count_per_instance_ = 0;
    uint32_t actuator_count_per_instance_ = 0;

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
    phi::Buffer poses_;
    phi::Buffer linear_velocities_;
    phi::Buffer angular_velocities_;
    phi::Buffer forces_;
    phi::Buffer torques_;
};

BatchedDeviceWorld UploadBatchedDeviceWorld(
    const phi::DeviceContext& context,
    const WorldTemplate& world_template,
    const std::vector<WorldInstance>& instances);
BatchedDeviceWorld UploadBatchedDeviceWorld(
    const WorldTemplate& world_template,
    const std::vector<WorldInstance>& instances);

CudaBatchedBroadphaseResult BuildBatchedCudaBroadphase(
    const phi::DeviceContext& context,
    const BatchedDeviceWorld& batch);
CudaBatchedBroadphaseResult BuildBatchedCudaBroadphase(
    const BatchedDeviceWorld& batch);

CudaBatchedContactResult GenerateBatchedCudaContacts(
    const phi::DeviceContext& context,
    const BatchedDeviceWorld& batch,
    const CudaBatchedBroadphaseResult& broadphase);
CudaBatchedContactResult GenerateBatchedCudaContacts(
    const BatchedDeviceWorld& batch,
    const CudaBatchedBroadphaseResult& broadphase);

CudaBatchedConstraintSolverResult SolveBatchedCudaContactConstraints(
    const phi::DeviceContext& context,
    BatchedDeviceWorld& batch,
    const CudaBatchedContactResult& contacts,
    const CudaBatchedConstraintSolverConfig& config = {});
CudaBatchedConstraintSolverResult SolveBatchedCudaContactConstraints(
    BatchedDeviceWorld& batch,
    const CudaBatchedContactResult& contacts,
    const CudaBatchedConstraintSolverConfig& config = {});

CudaBatchedConstraintSolverResult SolveBatchedCudaConstraints(
    const phi::DeviceContext& context,
    BatchedDeviceWorld& batch,
    const CudaBatchedContactResult* contacts,
    const CudaBatchedConstraintSolverConfig& config = {});
CudaBatchedConstraintSolverResult SolveBatchedCudaConstraints(
    BatchedDeviceWorld& batch,
    const CudaBatchedContactResult* contacts,
    const CudaBatchedConstraintSolverConfig& config = {});

CudaBatchedWorldStepReport StepBatchedCudaWorld(
    const phi::DeviceContext& context,
    BatchedDeviceWorld& batch,
    const CudaBatchedWorldStepOptions& options = {});
CudaBatchedWorldStepReport StepBatchedCudaWorld(
    BatchedDeviceWorld& batch,
    const CudaBatchedWorldStepOptions& options = {});

} // namespace nuka::runtime::gpu
