// ---------------------------------------------------------------------------
// nuka::solver::gpu::cuda_constraint_solver implementation
// ---------------------------------------------------------------------------

#include "solver/gpu/cuda_constraint_solver.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace nuka::solver::gpu {

namespace {

constexpr uint32_t kInvalidBody = ~0u;
constexpr float kHugeLimit = 1.0e6f;
constexpr uint32_t kRigidConstraintSchedulerDiagnosticSlots = 4u;

using runtime::gpu::AccumulateCudaConstraintRowSchedulerIteration;
using runtime::gpu::CudaConstraintRowBufferKind;
using runtime::gpu::CudaConstraintRowBufferView;
using runtime::gpu::CudaConstraintRowLayout;
using runtime::gpu::CudaConstraintRowSchedulerConfig;
using runtime::gpu::CudaConstraintRowSchedulerIterationReport;
using runtime::gpu::CudaConstraintRowSchedulerReport;
using runtime::gpu::CudaConstraintRowScheduleMode;
using runtime::gpu::SetCudaConstraintRowSchedulerMetadata;

struct RigidConstraintBlockSchedulerInput {
    cudaStream_t stream = nullptr;
    CudaConstraintRowBufferView row_buffer;
    CudaConstraintRowSchedulerConfig config;
    uint32_t body_count = 0u;
    uint32_t block_capacity = 0u;
    constraint::ConstraintBlock* blocks = nullptr;
    const uint32_t* block_count = nullptr;
    const float* inv_masses = nullptr;
    const math::Vec3* inv_inertias = nullptr;
    math::Vec3* linear_velocities = nullptr;
    math::Vec3* angular_velocities = nullptr;
    CudaConstraintSolverReport* report = nullptr;
};

__device__ math::Vec3 MakeVec3(float x, float y, float z) {
    math::Vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

__device__ math::Vec3 ZeroVec3() {
    return MakeVec3(0.0f, 0.0f, 0.0f);
}

__device__ math::Vec3 UnitX() {
    return MakeVec3(1.0f, 0.0f, 0.0f);
}

__device__ math::Vec3 UnitY() {
    return MakeVec3(0.0f, 1.0f, 0.0f);
}

__device__ math::Vec3 UnitZ() {
    return MakeVec3(0.0f, 0.0f, 1.0f);
}

__device__ math::Quat MakeQuat(float w, float x, float y, float z) {
    math::Quat q;
    q.w = w;
    q.x = x;
    q.y = y;
    q.z = z;
    return q;
}

__device__ math::Vec3 Add(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ math::Vec3 Sub(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ math::Vec3 Neg(math::Vec3 v) {
    return MakeVec3(-v.x, -v.y, -v.z);
}

__device__ math::Vec3 Scale(math::Vec3 v, float s) {
    return MakeVec3(v.x * s, v.y * s, v.z * s);
}

__device__ float Dot(math::Vec3 a, math::Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ math::Vec3 Cross(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ float LengthSq(math::Vec3 v) {
    return Dot(v, v);
}

__device__ float Length(math::Vec3 v) {
    return sqrtf(LengthSq(v));
}

__device__ math::Vec3 Normalize(math::Vec3 v) {
    const float length = Length(v);
    if (length < 1.0e-8f) {
        return UnitZ();
    }
    return Scale(v, 1.0f / length);
}

__device__ math::Vec3 Rotate(math::Quat q, math::Vec3 v) {
    const math::Vec3 qv = MakeVec3(q.x, q.y, q.z);
    const math::Vec3 t = Scale(Cross(qv, v), 2.0f);
    return Add(Add(v, Scale(t, q.w)), Cross(qv, t));
}

__device__ math::Quat Mul(math::Quat a, math::Quat b) {
    return MakeQuat(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

__device__ math::Quat NormalizeQuat(math::Quat q) {
    const float norm = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (norm < 1.0e-8f) {
        return MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const float inv_norm = 1.0f / norm;
    return MakeQuat(q.w * inv_norm, q.x * inv_norm, q.y * inv_norm, q.z * inv_norm);
}

__device__ math::Quat FromAxisAngle(math::Vec3 axis, float angle) {
    const math::Vec3 normalized_axis = Normalize(axis);
    const float half = 0.5f * angle;
    const float s = sinf(half);
    return MakeQuat(cosf(half),
                    normalized_axis.x * s,
                    normalized_axis.y * s,
                    normalized_axis.z * s);
}

__device__ void ClearBlock(constraint::ConstraintBlock* block) {
    *block = constraint::ConstraintBlock{};
}

__device__ math::Vec3 ChooseTangent(math::Vec3 normal) {
    if (fabsf(normal.x) < 0.9f) {
        return Normalize(Cross(normal, UnitX()));
    }
    return Normalize(Cross(normal, UnitY()));
}

__device__ uint32_t ContactNormalRowCount(constraint::ConstraintBlock block) {
    if (block.type != constraint::ConstraintType::Contact) {
        return 0u;
    }
    return block.normal_row_count > 0u ? block.normal_row_count : block.row_count;
}

__device__ bool IsContactFrictionRow(constraint::ConstraintBlock block, uint32_t row) {
    return block.type == constraint::ConstraintType::Contact
        && block.friction_row_count > 0u
        && row >= block.first_friction_row
        && row < block.first_friction_row + block.friction_row_count;
}

__device__ float TotalNormalImpulse(constraint::ConstraintBlock block) {
    float impulse = 0.0f;
    const uint32_t normal_rows = ContactNormalRowCount(block);
    for (uint32_t row = 0; row < normal_rows; ++row) {
        impulse += fmaxf(block.impulse[row], 0.0f);
    }
    return impulse;
}

__device__ bool IsNormalImpulseRow(constraint::ConstraintBlock block, uint32_t row) {
    if (row >= block.row_count) {
        return false;
    }
    if (block.type == constraint::ConstraintType::Contact) {
        return row < ContactNormalRowCount(block);
    }
    return true;
}

__device__ float ProjectedRowResidual(float velocity_error,
                                      float effective_mass,
                                      float impulse,
                                      float lower_limit,
                                      float upper_limit) {
    const float projected_impulse =
        fminf(fmaxf(impulse + effective_mass * velocity_error, lower_limit),
              upper_limit);
    return fabsf(projected_impulse - impulse);
}

__device__ float BodyInvMass(uint32_t body, uint32_t body_count, const float* inv_masses) {
    if (body >= body_count || body == kInvalidBody) {
        return 0.0f;
    }
    return inv_masses[body];
}

__device__ math::Vec3 BodyInvInertia(uint32_t body,
                                     uint32_t body_count,
                                     const math::Vec3* inv_inertias) {
    if (body >= body_count || body == kInvalidBody) {
        return ZeroVec3();
    }
    return inv_inertias[body];
}

__device__ math::Vec3 BodyLinearVelocity(uint32_t body,
                                         uint32_t body_count,
                                         const math::Vec3* velocities) {
    if (body >= body_count || body == kInvalidBody) {
        return ZeroVec3();
    }
    return velocities[body];
}

__device__ math::Vec3 BodyAngularVelocity(uint32_t body,
                                          uint32_t body_count,
                                          const math::Vec3* velocities) {
    if (body >= body_count || body == kInvalidBody) {
        return ZeroVec3();
    }
    return velocities[body];
}

__device__ math::Transform BodyPose(uint32_t body,
                                    uint32_t body_count,
                                    const math::Transform* poses) {
    math::Transform transform;
    transform.position = ZeroVec3();
    transform.rotation = MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
    if (body < body_count && body != kInvalidBody) {
        transform = poses[body];
    }
    return transform;
}

__device__ float ComputeJv(constraint::ConstraintBlock block,
                           uint32_t row,
                           uint32_t body_count,
                           const math::Vec3* linear_velocities,
                           const math::Vec3* angular_velocities) {
    float jv = 0.0f;
    jv += Dot(block.jacobian_linear_a[row],
              BodyLinearVelocity(block.body_a, body_count, linear_velocities));
    jv += Dot(block.jacobian_angular_a[row],
              BodyAngularVelocity(block.body_a, body_count, angular_velocities));
    jv += Dot(block.jacobian_linear_b[row],
              BodyLinearVelocity(block.body_b, body_count, linear_velocities));
    jv += Dot(block.jacobian_angular_b[row],
              BodyAngularVelocity(block.body_b, body_count, angular_velocities));
    return jv;
}

__device__ float ComputeEffectiveMass(constraint::ConstraintBlock block,
                                      uint32_t row,
                                      uint32_t body_count,
                                      const float* inv_masses,
                                      const math::Vec3* inv_inertias) {
    float diag = 0.0f;
    const float inv_mass_a = BodyInvMass(block.body_a, body_count, inv_masses);
    const float inv_mass_b = BodyInvMass(block.body_b, body_count, inv_masses);
    const math::Vec3 inv_inertia_a = BodyInvInertia(block.body_a, body_count, inv_inertias);
    const math::Vec3 inv_inertia_b = BodyInvInertia(block.body_b, body_count, inv_inertias);

    diag += inv_mass_a * Dot(block.jacobian_linear_a[row], block.jacobian_linear_a[row]);
    diag += inv_mass_b * Dot(block.jacobian_linear_b[row], block.jacobian_linear_b[row]);

    const math::Vec3 ja = block.jacobian_angular_a[row];
    diag += ja.x * ja.x * inv_inertia_a.x
          + ja.y * ja.y * inv_inertia_a.y
          + ja.z * ja.z * inv_inertia_a.z;

    const math::Vec3 jb = block.jacobian_angular_b[row];
    diag += jb.x * jb.x * inv_inertia_b.x
          + jb.y * jb.y * inv_inertia_b.y
          + jb.z * jb.z * inv_inertia_b.z;

    return diag > 1.0e-12f ? 1.0f / diag : 0.0f;
}

__device__ void ApplyImpulse(constraint::ConstraintBlock block,
                             uint32_t row,
                             float delta_impulse,
                             uint32_t body_count,
                             const float* inv_masses,
                             const math::Vec3* inv_inertias,
                             math::Vec3* linear_velocities,
                             math::Vec3* angular_velocities) {
    if (block.body_a < body_count && block.body_a != kInvalidBody) {
        const float inv_mass = inv_masses[block.body_a];
        const math::Vec3 inv_inertia = inv_inertias[block.body_a];
        linear_velocities[block.body_a] =
            Add(linear_velocities[block.body_a],
                Scale(block.jacobian_linear_a[row], inv_mass * delta_impulse));
        angular_velocities[block.body_a] = Add(
            angular_velocities[block.body_a],
            Scale(MakeVec3(block.jacobian_angular_a[row].x * inv_inertia.x,
                           block.jacobian_angular_a[row].y * inv_inertia.y,
                           block.jacobian_angular_a[row].z * inv_inertia.z),
                  delta_impulse));
    }

    if (block.body_b < body_count && block.body_b != kInvalidBody) {
        const float inv_mass = inv_masses[block.body_b];
        const math::Vec3 inv_inertia = inv_inertias[block.body_b];
        linear_velocities[block.body_b] =
            Add(linear_velocities[block.body_b],
                Scale(block.jacobian_linear_b[row], inv_mass * delta_impulse));
        angular_velocities[block.body_b] = Add(
            angular_velocities[block.body_b],
            Scale(MakeVec3(block.jacobian_angular_b[row].x * inv_inertia.x,
                           block.jacobian_angular_b[row].y * inv_inertia.y,
                           block.jacobian_angular_b[row].z * inv_inertia.z),
                  delta_impulse));
    }
}

__device__ void ApplyAngularCorrection(math::Transform* pose,
                                       math::Vec3 angular_delta) {
    const float angle = Length(angular_delta);
    if (angle <= 1.0e-8f) {
        return;
    }
    const math::Quat dq = FromAxisAngle(Scale(angular_delta, 1.0f / angle), angle);
    pose->rotation = NormalizeQuat(Mul(dq, pose->rotation));
}

__device__ void ApplyPositionCorrection(uint32_t body,
                                        uint32_t body_count,
                                        math::Vec3 linear_jacobian,
                                        math::Vec3 angular_jacobian,
                                        float position_impulse,
                                        const float* inv_masses,
                                        const math::Vec3* inv_inertias,
                                        math::Transform* poses) {
    if (body >= body_count || body == kInvalidBody) {
        return;
    }

    const float inv_mass = inv_masses[body];
    const math::Vec3 inv_inertia = inv_inertias[body];
    if (inv_mass > 0.0f) {
        poses[body].position =
            Add(poses[body].position, Scale(linear_jacobian, inv_mass * position_impulse));
    }

    const math::Vec3 angular_delta = MakeVec3(
        angular_jacobian.x * inv_inertia.x * position_impulse,
        angular_jacobian.y * inv_inertia.y * position_impulse,
        angular_jacobian.z * inv_inertia.z * position_impulse);
    ApplyAngularCorrection(&poses[body], angular_delta);
}

__device__ float JointRowMass(math::Vec3 linear_a,
                              math::Vec3 angular_a,
                              math::Vec3 linear_b,
                              math::Vec3 angular_b,
                              uint32_t body_a,
                              uint32_t body_b,
                              uint32_t body_count,
                              const float* inv_masses,
                              const math::Vec3* inv_inertias) {
    float diag = 0.0f;
    const float inv_mass_a = BodyInvMass(body_a, body_count, inv_masses);
    const float inv_mass_b = BodyInvMass(body_b, body_count, inv_masses);
    const math::Vec3 inv_inertia_a = BodyInvInertia(body_a, body_count, inv_inertias);
    const math::Vec3 inv_inertia_b = BodyInvInertia(body_b, body_count, inv_inertias);
    diag += inv_mass_a * Dot(linear_a, linear_a);
    diag += inv_mass_b * Dot(linear_b, linear_b);
    diag += angular_a.x * angular_a.x * inv_inertia_a.x
          + angular_a.y * angular_a.y * inv_inertia_a.y
          + angular_a.z * angular_a.z * inv_inertia_a.z;
    diag += angular_b.x * angular_b.x * inv_inertia_b.x
          + angular_b.y * angular_b.y * inv_inertia_b.y
          + angular_b.z * angular_b.z * inv_inertia_b.z;
    return diag > 1.0e-12f ? 1.0f / diag : 0.0f;
}

__device__ constraint::ConstraintBlock BuildContactBlock(
    const constraint::ContactManifold& manifold) {
    constraint::ConstraintBlock block;
    ClearBlock(&block);
    block.type = constraint::ConstraintType::Contact;
    block.body_a = manifold.body_a;
    block.body_b = manifold.body_b;
    block.normal_row_count =
        manifold.point_count < constraint::ConstraintBlock::kMaxRows - 2u
            ? manifold.point_count
            : constraint::ConstraintBlock::kMaxRows - 2u;
    block.first_friction_row = block.normal_row_count;
    block.friction_row_count = block.normal_row_count > 0u ? 2u : 0u;
    block.row_count = block.normal_row_count + block.friction_row_count;
    block.friction = manifold.friction;
    block.restitution = manifold.restitution;

    if (block.normal_row_count == 0u) {
        return block;
    }

    const math::Vec3 normal = Normalize(manifold.points[0].normal);
    const math::Vec3 tangent0 = ChooseTangent(normal);
    const math::Vec3 tangent1 = Normalize(Cross(normal, tangent0));

    for (uint32_t row = 0; row < block.normal_row_count; ++row) {
        const math::Vec3 row_normal = Normalize(manifold.points[row].normal);
        block.jacobian_linear_a[row] = row_normal;
        block.jacobian_linear_b[row] = Neg(row_normal);
        block.lower_limit[row] = 0.0f;
        block.upper_limit[row] = kHugeLimit;
        block.position_error[row] = manifold.points[row].penetration;
        block.impulse[row] = manifold.points[row].normal_impulse;
    }

    if (block.friction_row_count == 2u) {
        const uint32_t first = block.first_friction_row;
        block.jacobian_linear_a[first] = tangent0;
        block.jacobian_linear_b[first] = Neg(tangent0);
        block.impulse[first] = manifold.points[0].friction_impulse_1;
        block.jacobian_linear_a[first + 1u] = tangent1;
        block.jacobian_linear_b[first + 1u] = Neg(tangent1);
        block.impulse[first + 1u] = manifold.points[0].friction_impulse_2;
    }

    return block;
}

__device__ constraint::ConstraintBlock BuildRevoluteBlock(uint32_t parent,
                                                          uint32_t child,
                                                          math::Vec3 axis,
                                                          math::Transform parent_frame,
                                                          math::Transform child_frame,
                                                          bool fixed_joint) {
    constraint::ConstraintBlock block;
    ClearBlock(&block);
    block.type = constraint::ConstraintType::Joint;
    block.body_a = parent;
    block.body_b = child;
    block.row_count = fixed_joint ? 6u : 5u;
    block.anchor_local_a = parent_frame.position;
    block.anchor_local_b = child_frame.position;

    const math::Vec3 norm_axis = Normalize(axis);
    const math::Vec3 perp1 =
        fabsf(norm_axis.x) < 0.9f
            ? Normalize(Cross(norm_axis, UnitX()))
            : Normalize(Cross(norm_axis, UnitY()));
    const math::Vec3 perp2 = Normalize(Cross(norm_axis, perp1));
    const math::Vec3 dirs[3] = {UnitX(), UnitY(), UnitZ()};
    const math::Vec3 r_a = parent_frame.position;
    const math::Vec3 r_b = child_frame.position;

    for (uint32_t row = 0; row < 3u; ++row) {
        block.jacobian_linear_a[row] = dirs[row];
        block.jacobian_angular_a[row] = Cross(r_a, dirs[row]);
        block.jacobian_linear_b[row] = Neg(dirs[row]);
        block.jacobian_angular_b[row] = Neg(Cross(r_b, dirs[row]));
        block.lower_limit[row] = -kHugeLimit;
        block.upper_limit[row] = kHugeLimit;
    }

    const math::Vec3 rot_axes[3] = {perp1, perp2, norm_axis};
    for (uint32_t i = 0; i < block.row_count - 3u; ++i) {
        const uint32_t row = 3u + i;
        block.jacobian_angular_a[row] = rot_axes[i];
        block.jacobian_angular_b[row] = Neg(rot_axes[i]);
        block.lower_limit[row] = -kHugeLimit;
        block.upper_limit[row] = kHugeLimit;
    }

    return block;
}

__device__ constraint::ConstraintBlock BuildDriveBlock(uint32_t child,
                                                       uint32_t parent,
                                                       math::Vec3 axis,
                                                       scene::ActuatorType type,
                                                       float gain,
                                                       float force_limit) {
    constraint::ConstraintBlock block;
    ClearBlock(&block);
    block.type = constraint::ConstraintType::Drive;
    block.body_a = child;
    block.body_b = parent;
    block.row_count = 1u;
    const math::Vec3 norm_axis = Normalize(axis);
    block.jacobian_angular_a[0] = norm_axis;
    block.jacobian_angular_b[0] = Neg(norm_axis);
    block.rhs[0] =
        (type == scene::ActuatorType::Velocity
         || type == scene::ActuatorType::Motor
         || type == scene::ActuatorType::Force)
            ? gain
            : 0.0f;
    const float limit = force_limit > 0.0f ? force_limit : kHugeLimit;
    block.lower_limit[0] = -limit;
    block.upper_limit[0] = limit;
    return block;
}

__global__ void ClearAssemblyKernel(uint32_t* block_count,
                                    CudaConstraintSolverReport* report) {
    *block_count = 0u;
    report->constraint_block_count = 0u;
    report->constraint_row_count = 0u;
    report->contact_constraint_count = 0u;
    report->joint_constraint_count = 0u;
    report->drive_constraint_count = 0u;
    report->velocity_iterations = 0u;
    report->position_iterations = 0u;
    report->max_position_error = 0.0f;
    report->row_scheduler_report = CudaConstraintRowSchedulerReport{};
}

__global__ void AssembleContactBlocksKernel(uint32_t pair_slot_count,
                                            const constraint::ContactManifold* manifolds,
                                            constraint::ConstraintBlock* blocks,
                                            uint32_t* block_count) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= pair_slot_count || manifolds[slot].point_count == 0u) {
        return;
    }

    const uint32_t out_index = atomicAdd(block_count, 1u);
    blocks[out_index] = BuildContactBlock(manifolds[slot]);
}

__global__ void AssembleJointBlocksKernel(uint32_t joint_count,
                                          const scene::JointType* joint_types,
                                          const scene::BodyId* joint_parent_bodies,
                                          const scene::BodyId* joint_child_bodies,
                                          const math::Vec3* joint_axes,
                                          const math::Transform* joint_parent_frames,
                                          const math::Transform* joint_child_frames,
                                          constraint::ConstraintBlock* blocks,
                                          uint32_t* block_count) {
    const uint32_t joint_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (joint_index >= joint_count) {
        return;
    }

    const scene::BodyId parent = joint_parent_bodies[joint_index];
    const scene::BodyId child = joint_child_bodies[joint_index];
    if (parent == scene::kInvalidBody && child == scene::kInvalidBody) {
        return;
    }

    const bool fixed_joint = joint_types[joint_index] == scene::JointType::Fixed;
    const uint32_t out_index = atomicAdd(block_count, 1u);
    blocks[out_index] = BuildRevoluteBlock(parent,
                                           child,
                                           joint_axes[joint_index],
                                           joint_parent_frames[joint_index],
                                           joint_child_frames[joint_index],
                                           fixed_joint);
}

__global__ void AssembleDriveBlocksKernel(uint32_t actuator_count,
                                          uint32_t joint_count,
                                          const scene::JointId* actuator_joint_ids,
                                          const scene::ActuatorType* actuator_types,
                                          const float* actuator_gains,
                                          const float* actuator_force_limits,
                                          const scene::BodyId* joint_parent_bodies,
                                          const scene::BodyId* joint_child_bodies,
                                          const math::Vec3* joint_axes,
                                          constraint::ConstraintBlock* blocks,
                                          uint32_t* block_count) {
    const uint32_t actuator_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (actuator_index >= actuator_count) {
        return;
    }

    const scene::JointId joint_id = actuator_joint_ids[actuator_index];
    if (joint_id >= joint_count || joint_id == scene::kInvalidJoint) {
        return;
    }

    const scene::BodyId parent = joint_parent_bodies[joint_id];
    const scene::BodyId child = joint_child_bodies[joint_id];
    if (parent == scene::kInvalidBody && child == scene::kInvalidBody) {
        return;
    }

    const uint32_t out_index = atomicAdd(block_count, 1u);
    blocks[out_index] = BuildDriveBlock(child,
                                        parent,
                                        joint_axes[joint_id],
                                        actuator_types[actuator_index],
                                        actuator_gains[actuator_index],
                                        actuator_force_limits[actuator_index]);
}

__global__ void FinalizeAssemblyReportKernel(const constraint::ConstraintBlock* blocks,
                                             const uint32_t* block_count,
                                             CudaConstraintSolverReport* report) {
    const uint32_t count = *block_count;
    report->constraint_block_count = count;
    report->constraint_row_count = 0u;
    report->contact_constraint_count = 0u;
    report->joint_constraint_count = 0u;
    report->drive_constraint_count = 0u;
    for (uint32_t index = 0; index < count; ++index) {
        report->constraint_row_count += blocks[index].row_count;
        if (blocks[index].type == constraint::ConstraintType::Contact) {
            ++report->contact_constraint_count;
        } else if (blocks[index].type == constraint::ConstraintType::Joint) {
            ++report->joint_constraint_count;
        } else if (blocks[index].type == constraint::ConstraintType::Drive) {
            ++report->drive_constraint_count;
        }
    }
}

__global__ void PrecomputeBlocksKernel(uint32_t body_count,
                                       constraint::ConstraintBlock* blocks,
                                       const uint32_t* block_count,
                                       const float* inv_masses,
                                       const math::Vec3* inv_inertias,
                                       const math::Vec3* linear_velocities,
                                       const math::Vec3* angular_velocities) {
    const uint32_t block_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_index >= *block_count) {
        return;
    }

    constraint::ConstraintBlock block = blocks[block_index];
    for (uint32_t row = 0; row < block.row_count; ++row) {
        block.effective_mass[row] =
            ComputeEffectiveMass(block, row, body_count, inv_masses, inv_inertias);
        if (block.type == constraint::ConstraintType::Contact
            && row < ContactNormalRowCount(block)
            && block.restitution > 0.0f) {
            const float jv = ComputeJv(block,
                                       row,
                                       body_count,
                                       linear_velocities,
                                       angular_velocities);
            if (jv < 0.0f) {
                block.rhs[row] = fmaxf(block.rhs[row], -block.restitution * jv);
            }
        }
    }
    blocks[block_index] = block;
}

__global__ void SolveVelocityIterationKernel(uint32_t body_count,
                                             constraint::ConstraintBlock* blocks,
                                             const uint32_t* block_count,
                                             const float* inv_masses,
                                             const math::Vec3* inv_inertias,
                                             math::Vec3* linear_velocities,
                                             math::Vec3* angular_velocities,
                                             uint32_t iteration_index,
                                             CudaConstraintSolverReport* report) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) {
        return;
    }

    const uint32_t count = *block_count;
    uint32_t active_row_count = 0u;
    uint32_t normal_impulse_count = 0u;
    uint32_t tangent_impulse_count = 0u;
    float normal_delta_impulse_magnitude = 0.0f;
    float tangent_delta_impulse_magnitude = 0.0f;
    float max_normal_delta_impulse = 0.0f;
    float max_tangent_delta_impulse = 0.0f;
    float max_residual = 0.0f;

    for (uint32_t block_index = 0; block_index < count; ++block_index) {
        constraint::ConstraintBlock block = blocks[block_index];
        for (uint32_t row = 0; row < block.row_count; ++row) {
            ++active_row_count;
            const float jv = ComputeJv(block,
                                       row,
                                       body_count,
                                       linear_velocities,
                                       angular_velocities);
            const float lambda = block.effective_mass[row] * (block.rhs[row] - jv);
            const float old_impulse = block.impulse[row];
            float new_impulse = old_impulse + lambda;
            float lower_limit = block.lower_limit[row];
            float upper_limit = block.upper_limit[row];
            if (IsContactFrictionRow(block, row)) {
                const float max_friction = fmaxf(block.friction, 0.0f) * TotalNormalImpulse(block);
                lower_limit = -max_friction;
                upper_limit = max_friction;
            }
            new_impulse = fminf(fmaxf(new_impulse, lower_limit), upper_limit);
            block.impulse[row] = new_impulse;
            const float delta = new_impulse - old_impulse;
            if (fabsf(delta) > 1.0e-12f) {
                ApplyImpulse(block,
                             row,
                             delta,
                             body_count,
                             inv_masses,
                             inv_inertias,
                             linear_velocities,
                             angular_velocities);
            }
            const bool tangent_row = IsContactFrictionRow(block, row);
            const float abs_delta = fabsf(delta);
            if (abs_delta > 0.0f) {
                if (tangent_row) {
                    ++tangent_impulse_count;
                    tangent_delta_impulse_magnitude += abs_delta;
                    max_tangent_delta_impulse =
                        fmaxf(max_tangent_delta_impulse, abs_delta);
                } else if (IsNormalImpulseRow(block, row)) {
                    ++normal_impulse_count;
                    normal_delta_impulse_magnitude += abs_delta;
                    max_normal_delta_impulse =
                        fmaxf(max_normal_delta_impulse, abs_delta);
                }
            }
            const float solved_jv = ComputeJv(block,
                                             row,
                                             body_count,
                                             linear_velocities,
                                             angular_velocities);
            max_residual =
                fmaxf(max_residual,
                      ProjectedRowResidual(block.rhs[row] - solved_jv,
                                           block.effective_mass[row],
                                           new_impulse,
                                           lower_limit,
                                           upper_limit));
        }
        blocks[block_index] = block;
    }

    CudaConstraintRowSchedulerIterationReport iteration_report;
    iteration_report.active_row_count = active_row_count;
    iteration_report.normal_impulse_count = normal_impulse_count;
    iteration_report.tangent_impulse_count = tangent_impulse_count;
    iteration_report.diagnostic_slot_count =
        iteration_index < kRigidConstraintSchedulerDiagnosticSlots
            ? iteration_index + 1u
            : kRigidConstraintSchedulerDiagnosticSlots;
    iteration_report.normal_delta_impulse_magnitude =
        normal_delta_impulse_magnitude;
    iteration_report.tangent_delta_impulse_magnitude =
        tangent_delta_impulse_magnitude;
    iteration_report.max_normal_delta_impulse = max_normal_delta_impulse;
    iteration_report.max_tangent_delta_impulse = max_tangent_delta_impulse;
    iteration_report.max_residual = max_residual;
    CudaConstraintRowSchedulerReport updated_scheduler =
        report->row_scheduler_report;
    AccumulateCudaConstraintRowSchedulerIteration(updated_scheduler,
                                                 iteration_report);
    updated_scheduler.executed_iterations = iteration_index + 1u;
    updated_scheduler.solver_launch_count = iteration_index + 1u;
    updated_scheduler.diagnostic_launch_count = iteration_index + 1u;
    report->row_scheduler_report = updated_scheduler;
}

__global__ void SolveContactPositionIterationKernel(uint32_t body_count,
                                                    constraint::ConstraintBlock* blocks,
                                                    const uint32_t* block_count,
                                                    const float* inv_masses,
                                                    const math::Vec3* inv_inertias,
                                                    math::Transform* poses,
                                                    float slop,
                                                    float baumgarte,
                                                    CudaConstraintSolverReport* report) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) {
        return;
    }

    float max_error = 0.0f;
    const uint32_t count = *block_count;
    for (uint32_t block_index = 0; block_index < count; ++block_index) {
        const constraint::ConstraintBlock block = blocks[block_index];
        if (block.type != constraint::ConstraintType::Contact) {
            continue;
        }
        const uint32_t normal_rows = ContactNormalRowCount(block);
        for (uint32_t row = 0; row < normal_rows; ++row) {
            const float penetration =
                block.position_error[row] > 0.0f
                    ? block.position_error[row]
                    : fabsf(block.rhs[row]);
            max_error = fmaxf(max_error, penetration);
            const float correction = baumgarte * fmaxf(penetration - slop, 0.0f);
            if (correction <= 1.0e-8f) {
                continue;
            }
            const float position_impulse = correction * block.effective_mass[row];
            ApplyPositionCorrection(block.body_a,
                                    body_count,
                                    block.jacobian_linear_a[row],
                                    ZeroVec3(),
                                    position_impulse,
                                    inv_masses,
                                    inv_inertias,
                                    poses);
            ApplyPositionCorrection(block.body_b,
                                    body_count,
                                    block.jacobian_linear_b[row],
                                    ZeroVec3(),
                                    position_impulse,
                                    inv_masses,
                                    inv_inertias,
                                    poses);
        }
    }
    report->max_position_error = max_error;
}

__global__ void SolveJointPositionIterationKernel(uint32_t body_count,
                                                  const constraint::ConstraintBlock* blocks,
                                                  const uint32_t* block_count,
                                                  const float* inv_masses,
                                                  const math::Vec3* inv_inertias,
                                                  math::Transform* poses,
                                                  float slop,
                                                  float baumgarte,
                                                  CudaConstraintSolverReport* report) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) {
        return;
    }

    float max_error = report->max_position_error;
    const math::Vec3 axes[3] = {UnitX(), UnitY(), UnitZ()};
    const uint32_t count = *block_count;
    for (uint32_t block_index = 0; block_index < count; ++block_index) {
        const constraint::ConstraintBlock block = blocks[block_index];
        if (block.type != constraint::ConstraintType::Joint) {
            continue;
        }

        for (uint32_t axis_index = 0; axis_index < 3u; ++axis_index) {
            const math::Vec3 axis = axes[axis_index];
            const math::Transform pose_a = BodyPose(block.body_a, body_count, poses);
            const math::Transform pose_b = BodyPose(block.body_b, body_count, poses);
            const math::Vec3 r_a = Rotate(pose_a.rotation, block.anchor_local_a);
            const math::Vec3 r_b = Rotate(pose_b.rotation, block.anchor_local_b);
            const math::Vec3 error =
                Sub(Add(pose_a.position, r_a), Add(pose_b.position, r_b));
            const float row_error = Dot(error, axis);
            max_error = fmaxf(max_error, fabsf(row_error));
            const float correction = baumgarte * row_error;
            if (fabsf(correction) <= slop) {
                continue;
            }

            const math::Vec3 linear_a = Neg(axis);
            const math::Vec3 linear_b = axis;
            const math::Vec3 angular_a = Neg(Cross(r_a, axis));
            const math::Vec3 angular_b = Cross(r_b, axis);
            const float effective_mass = JointRowMass(linear_a,
                                                      angular_a,
                                                      linear_b,
                                                      angular_b,
                                                      block.body_a,
                                                      block.body_b,
                                                      body_count,
                                                      inv_masses,
                                                      inv_inertias);
            if (effective_mass <= 0.0f) {
                continue;
            }

            const float position_impulse = correction * effective_mass;
            ApplyPositionCorrection(block.body_a,
                                    body_count,
                                    linear_a,
                                    angular_a,
                                    position_impulse,
                                    inv_masses,
                                    inv_inertias,
                                    poses);
            ApplyPositionCorrection(block.body_b,
                                    body_count,
                                    linear_b,
                                    angular_b,
                                    position_impulse,
                                    inv_masses,
                                    inv_inertias,
                                    poses);
        }
    }
    report->max_position_error = max_error;
}

__global__ void SetIterationReportKernel(uint32_t velocity_iterations,
                                         uint32_t position_iterations,
                                         CudaConstraintSolverReport* report) {
    report->velocity_iterations = velocity_iterations;
    report->position_iterations = position_iterations;
}

__global__ void SetRigidSchedulerMetadataKernel(
    CudaConstraintRowBufferView row_buffer,
    CudaConstraintRowSchedulerConfig config,
    CudaConstraintSolverReport* report) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) {
        return;
    }
    SetCudaConstraintRowSchedulerMetadata(report->row_scheduler_report,
                                          row_buffer,
                                          config);
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

template <typename T>
std::vector<T> DownloadVector(const phi::Buffer& buffer, uint32_t count) {
    std::vector<T> values(count);
    if (!values.empty()) {
        buffer.CopyToHost(values.data(), values.size() * sizeof(T));
    }
    return values;
}

void RunCudaRigidConstraintBlockScheduler(
    const RigidConstraintBlockSchedulerInput& input) {
    if (input.row_buffer.kind != CudaConstraintRowBufferKind::RigidConstraintBlock ||
        input.row_buffer.layout != CudaConstraintRowLayout::ConstraintBlock ||
        input.row_buffer.schedule_mode != CudaConstraintRowScheduleMode::GlobalRowSweep ||
        input.block_capacity == 0u ||
        input.body_count == 0u ||
        input.blocks == nullptr ||
        input.block_count == nullptr ||
        input.report == nullptr) {
        return;
    }

    constexpr uint32_t kBlockSize = 128u;
    const uint32_t blocks_grid =
        (input.block_capacity + kBlockSize - 1u) / kBlockSize;
    PrecomputeBlocksKernel<<<blocks_grid, kBlockSize, 0, input.stream>>>(
        input.body_count,
        input.blocks,
        input.block_count,
        input.inv_masses,
        input.inv_inertias,
        input.linear_velocities,
        input.angular_velocities);
    CheckCuda(cudaGetLastError(), "PrecomputeBlocksKernel launch");

    SetRigidSchedulerMetadataKernel<<<1, 1, 0, input.stream>>>(
        input.row_buffer,
        input.config,
        input.report);
    CheckCuda(cudaGetLastError(), "SetRigidSchedulerMetadataKernel launch");

    if (input.config.iterations == 0u) {
        return;
    }

    for (uint32_t iter = 0; iter < input.config.iterations; ++iter) {
        SolveVelocityIterationKernel<<<1, 1, 0, input.stream>>>(
            input.body_count,
            input.blocks,
            input.block_count,
            input.inv_masses,
            input.inv_inertias,
            input.linear_velocities,
            input.angular_velocities,
            iter,
            input.report);
        CheckCuda(cudaGetLastError(), "SolveVelocityIterationKernel launch");
    }
}

void RunSolverKernels(runtime::gpu::DeviceWorld& device_world,
                      uint32_t block_capacity,
                      phi::Buffer& blocks,
                      phi::Buffer& block_count,
                      phi::Buffer& report,
                      cudaStream_t stream,
                      const CudaConstraintSolverConfig& config) {
    if (block_capacity == 0u || device_world.BodyCount() == 0u) {
        return;
    }

    CudaConstraintRowBufferView row_buffer;
    row_buffer.kind = CudaConstraintRowBufferKind::RigidConstraintBlock;
    row_buffer.layout = CudaConstraintRowLayout::ConstraintBlock;
    row_buffer.schedule_mode = CudaConstraintRowScheduleMode::GlobalRowSweep;
    row_buffer.device_rows = blocks.Data();
    row_buffer.row_count =
        block_capacity * constraint::ConstraintBlock::kMaxRows;
    row_buffer.owner_count = block_capacity;
    row_buffer.rows_per_owner = constraint::ConstraintBlock::kMaxRows;
    row_buffer.row_stride_bytes = sizeof(constraint::ConstraintBlock);

    CudaConstraintRowSchedulerConfig scheduler_config;
    scheduler_config.iterations = config.velocity_iterations;
    scheduler_config.enable_warm_start = true;
    scheduler_config.reduce_diagnostics = true;

    RigidConstraintBlockSchedulerInput scheduler_input;
    scheduler_input.stream = stream;
    scheduler_input.row_buffer = row_buffer;
    scheduler_input.config = scheduler_config;
    scheduler_input.body_count = device_world.BodyCount();
    scheduler_input.block_capacity = block_capacity;
    scheduler_input.blocks = static_cast<constraint::ConstraintBlock*>(blocks.Data());
    scheduler_input.block_count = static_cast<const uint32_t*>(block_count.Data());
    scheduler_input.inv_masses = device_world.DeviceInvMasses();
    scheduler_input.inv_inertias = device_world.DeviceInvInertias();
    scheduler_input.linear_velocities = device_world.DeviceLinearVelocities();
    scheduler_input.angular_velocities = device_world.DeviceAngularVelocities();
    scheduler_input.report =
        static_cast<CudaConstraintSolverReport*>(report.Data());
    RunCudaRigidConstraintBlockScheduler(scheduler_input);

    for (uint32_t iter = 0; iter < config.position_iterations; ++iter) {
        SolveContactPositionIterationKernel<<<1, 1, 0, stream>>>(
            device_world.BodyCount(),
            static_cast<constraint::ConstraintBlock*>(blocks.Data()),
            static_cast<const uint32_t*>(block_count.Data()),
            device_world.DeviceInvMasses(),
            device_world.DeviceInvInertias(),
            device_world.DevicePoses(),
            config.slop,
            config.baumgarte,
            static_cast<CudaConstraintSolverReport*>(report.Data()));
        CheckCuda(cudaGetLastError(), "SolveContactPositionIterationKernel launch");

        SolveJointPositionIterationKernel<<<1, 1, 0, stream>>>(
            device_world.BodyCount(),
            static_cast<constraint::ConstraintBlock*>(blocks.Data()),
            static_cast<const uint32_t*>(block_count.Data()),
            device_world.DeviceInvMasses(),
            device_world.DeviceInvInertias(),
            device_world.DevicePoses(),
            config.slop,
            config.baumgarte,
            static_cast<CudaConstraintSolverReport*>(report.Data()));
        CheckCuda(cudaGetLastError(), "SolveJointPositionIterationKernel launch");
    }

    SetIterationReportKernel<<<1, 1, 0, stream>>>(
        config.velocity_iterations,
        config.position_iterations,
        static_cast<CudaConstraintSolverReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "SetIterationReportKernel launch");
}

} // namespace

CudaConstraintSolverResult::CudaConstraintSolverResult(uint32_t block_capacity,
                                                       phi::Buffer blocks,
                                                       phi::Buffer block_count,
                                                       phi::Buffer report)
    : block_capacity_(block_capacity)
    , blocks_(std::move(blocks))
    , block_count_(std::move(block_count))
    , report_(std::move(report)) {}

CudaConstraintSolverReport CudaConstraintSolverResult::DownloadReport() const {
    CudaConstraintSolverReport report;
    if (report_.Size() > 0u) {
        report_.CopyToHost(&report, sizeof(report));
    }
    return report;
}

std::vector<constraint::ConstraintBlock> CudaConstraintSolverResult::DownloadBlocks() const {
    uint32_t block_count = 0;
    if (block_count_.Size() > 0u) {
        block_count_.CopyToHost(&block_count, sizeof(block_count));
    }
    if (block_count > block_capacity_) {
        block_count = block_capacity_;
    }
    return DownloadVector<constraint::ConstraintBlock>(blocks_, block_count);
}

runtime::gpu::CudaConstraintRowBufferView
CudaConstraintSolverResult::ConstraintRowBuffer() const {
    runtime::gpu::CudaConstraintRowBufferView view;
    view.kind = runtime::gpu::CudaConstraintRowBufferKind::RigidConstraintBlock;
    view.layout = runtime::gpu::CudaConstraintRowLayout::ConstraintBlock;
    view.schedule_mode =
        runtime::gpu::CudaConstraintRowScheduleMode::GlobalRowSweep;
    view.device_rows = const_cast<void*>(blocks_.Data());
    view.row_count = block_capacity_ * constraint::ConstraintBlock::kMaxRows;
    view.owner_count = block_capacity_;
    view.rows_per_owner = constraint::ConstraintBlock::kMaxRows;
    view.row_stride_bytes = sizeof(constraint::ConstraintBlock);
    return view;
}

const constraint::ConstraintBlock* CudaConstraintSolverResult::DeviceBlocks() const {
    return static_cast<const constraint::ConstraintBlock*>(blocks_.Data());
}

const uint32_t* CudaConstraintSolverResult::DeviceBlockCount() const {
    return static_cast<const uint32_t*>(block_count_.Data());
}

CudaConstraintSolverResult SolveCudaConstraints(
    const phi::DeviceContext& context,
    runtime::gpu::DeviceWorld& device_world,
    const constraint::gpu::CudaContactResult* contacts,
    const CudaConstraintSolverConfig& config) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    if (!device_world.HasUploadedState()) {
        throw std::runtime_error(
            "SolveCudaConstraints requires UploadDeviceState before solving");
    }

    const uint32_t contact_capacity = contacts != nullptr ? contacts->PairSlotCount() : 0u;
    const uint32_t block_capacity =
        contact_capacity + device_world.JointCount() + device_world.ActuatorCount();
    phi::Buffer blocks(block_capacity * sizeof(constraint::ConstraintBlock),
                       phi::MemoryKind::Device);
    phi::Buffer block_count(sizeof(uint32_t), phi::MemoryKind::Device);
    phi::Buffer report(sizeof(CudaConstraintSolverReport), phi::MemoryKind::Device);

    ClearAssemblyKernel<<<1, 1, 0, stream>>>(
        static_cast<uint32_t*>(block_count.Data()),
        static_cast<CudaConstraintSolverReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "ClearAssemblyKernel launch");

    constexpr uint32_t kBlockSize = 128u;
    if (contacts != nullptr && contacts->PairSlotCount() > 0u) {
        const uint32_t contact_blocks =
            (contacts->PairSlotCount() + kBlockSize - 1u) / kBlockSize;
        AssembleContactBlocksKernel<<<contact_blocks, kBlockSize, 0, stream>>>(
            contacts->PairSlotCount(),
            contacts->DeviceManifolds(),
            static_cast<constraint::ConstraintBlock*>(blocks.Data()),
            static_cast<uint32_t*>(block_count.Data()));
        CheckCuda(cudaGetLastError(), "AssembleContactBlocksKernel launch");
    }

    if (device_world.JointCount() > 0u) {
        const uint32_t joint_blocks = (device_world.JointCount() + kBlockSize - 1u) / kBlockSize;
        AssembleJointBlocksKernel<<<joint_blocks, kBlockSize, 0, stream>>>(
            device_world.JointCount(),
            device_world.DeviceJointTypes(),
            device_world.DeviceJointParentBodies(),
            device_world.DeviceJointChildBodies(),
            device_world.DeviceJointAxes(),
            device_world.DeviceJointParentFrames(),
            device_world.DeviceJointChildFrames(),
            static_cast<constraint::ConstraintBlock*>(blocks.Data()),
            static_cast<uint32_t*>(block_count.Data()));
        CheckCuda(cudaGetLastError(), "AssembleJointBlocksKernel launch");
    }

    if (device_world.ActuatorCount() > 0u) {
        const uint32_t actuator_blocks =
            (device_world.ActuatorCount() + kBlockSize - 1u) / kBlockSize;
        AssembleDriveBlocksKernel<<<actuator_blocks, kBlockSize, 0, stream>>>(
            device_world.ActuatorCount(),
            device_world.JointCount(),
            device_world.DeviceActuatorJointIds(),
            device_world.DeviceActuatorTypes(),
            device_world.DeviceActuatorGains(),
            device_world.DeviceActuatorForceLimits(),
            device_world.DeviceJointParentBodies(),
            device_world.DeviceJointChildBodies(),
            device_world.DeviceJointAxes(),
            static_cast<constraint::ConstraintBlock*>(blocks.Data()),
            static_cast<uint32_t*>(block_count.Data()));
        CheckCuda(cudaGetLastError(), "AssembleDriveBlocksKernel launch");
    }

    FinalizeAssemblyReportKernel<<<1, 1, 0, stream>>>(
        static_cast<constraint::ConstraintBlock*>(blocks.Data()),
        static_cast<const uint32_t*>(block_count.Data()),
        static_cast<CudaConstraintSolverReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "FinalizeAssemblyReportKernel launch");

    RunSolverKernels(device_world, block_capacity, blocks, block_count, report, stream, config);
    context.stream.Synchronize();

    return CudaConstraintSolverResult(block_capacity,
                                      std::move(blocks),
                                      std::move(block_count),
                                      std::move(report));
}

CudaConstraintSolverResult SolveCudaConstraintBlocks(
    const phi::DeviceContext& context,
    runtime::gpu::DeviceWorld& device_world,
    const std::vector<constraint::ConstraintBlock>& host_blocks,
    const CudaConstraintSolverConfig& config) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    if (!device_world.HasUploadedState()) {
        throw std::runtime_error(
            "SolveCudaConstraintBlocks requires UploadDeviceState before solving");
    }

    const uint32_t block_capacity = static_cast<uint32_t>(host_blocks.size());
    phi::Buffer blocks(block_capacity * sizeof(constraint::ConstraintBlock),
                       phi::MemoryKind::Device);
    phi::Buffer block_count(sizeof(uint32_t), phi::MemoryKind::Device);
    phi::Buffer report(sizeof(CudaConstraintSolverReport), phi::MemoryKind::Device);

    if (!host_blocks.empty()) {
        blocks.CopyFromHost(host_blocks.data(),
                            host_blocks.size() * sizeof(constraint::ConstraintBlock));
    }
    block_count.CopyFromHost(&block_capacity, sizeof(block_capacity));

    ClearAssemblyKernel<<<1, 1, 0, stream>>>(
        static_cast<uint32_t*>(block_count.Data()),
        static_cast<CudaConstraintSolverReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "ClearAssemblyKernel launch");
    if (block_capacity > 0u) {
        blocks.CopyFromHost(host_blocks.data(),
                            host_blocks.size() * sizeof(constraint::ConstraintBlock));
        block_count.CopyFromHost(&block_capacity, sizeof(block_capacity));
    }
    FinalizeAssemblyReportKernel<<<1, 1, 0, stream>>>(
        static_cast<constraint::ConstraintBlock*>(blocks.Data()),
        static_cast<const uint32_t*>(block_count.Data()),
        static_cast<CudaConstraintSolverReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "FinalizeAssemblyReportKernel launch");

    RunSolverKernels(device_world, block_capacity, blocks, block_count, report, stream, config);
    context.stream.Synchronize();

    return CudaConstraintSolverResult(block_capacity,
                                      std::move(blocks),
                                      std::move(block_count),
                                      std::move(report));
}

CudaConstraintSolverResult SolveCudaConstraints(
    runtime::gpu::DeviceWorld& device_world,
    const constraint::gpu::CudaContactResult* contacts,
    const CudaConstraintSolverConfig& config) {
    auto context = phi::MakeDefaultDeviceContext();
    return SolveCudaConstraints(context, device_world, contacts, config);
}

CudaConstraintSolverResult SolveCudaConstraintBlocks(
    runtime::gpu::DeviceWorld& device_world,
    const std::vector<constraint::ConstraintBlock>& host_blocks,
    const CudaConstraintSolverConfig& config) {
    auto context = phi::MakeDefaultDeviceContext();
    return SolveCudaConstraintBlocks(context, device_world, host_blocks, config);
}

} // namespace nuka::solver::gpu
