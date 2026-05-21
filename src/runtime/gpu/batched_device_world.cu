// ---------------------------------------------------------------------------
// nuka::runtime::gpu::BatchedDeviceWorld implementation
// ---------------------------------------------------------------------------

#include "runtime/gpu/batched_device_world.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace nuka::runtime::gpu {

namespace {

constexpr uint32_t kInvalidBody = ~0u;
constexpr float kHugeLimit = 1.0e6f;

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

__host__ __device__ math::Vec3 MakeVec3(float x, float y, float z) {
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

__device__ float Length(math::Vec3 v) {
    return sqrtf(Dot(v, v));
}

__device__ math::Vec3 Normalize(math::Vec3 v) {
    const float length = Length(v);
    if (length < 1.0e-8f) {
        return UnitY();
    }
    return Scale(v, 1.0f / length);
}

__device__ math::Vec3 ChooseTangent(math::Vec3 normal) {
    if (fabsf(normal.x) < 0.9f) {
        return Normalize(Cross(normal, UnitX()));
    }
    return Normalize(Cross(normal, UnitY()));
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

__device__ math::Vec3 TransformPoint(math::Transform transform, math::Vec3 point) {
    return Add(Rotate(transform.rotation, point), transform.position);
}

__device__ math::Transform Compose(math::Transform a, math::Transform b) {
    math::Transform result;
    result.position = TransformPoint(a, b.position);
    result.rotation = Mul(a.rotation, b.rotation);
    return result;
}

__device__ void Expand(collision::AABB& aabb, math::Vec3 point) {
    aabb.min.x = fminf(aabb.min.x, point.x);
    aabb.min.y = fminf(aabb.min.y, point.y);
    aabb.min.z = fminf(aabb.min.z, point.z);
    aabb.max.x = fmaxf(aabb.max.x, point.x);
    aabb.max.y = fmaxf(aabb.max.y, point.y);
    aabb.max.z = fmaxf(aabb.max.z, point.z);
}

__device__ collision::AABB SphereAabb(math::Vec3 center, float radius) {
    collision::AABB aabb;
    const math::Vec3 extents = MakeVec3(radius, radius, radius);
    aabb.min = Sub(center, extents);
    aabb.max = Add(center, extents);
    return aabb;
}

__device__ collision::AABB PlaneAabb(float plane_y) {
    collision::AABB aabb;
    aabb.min = MakeVec3(-1.0e6f, plane_y - 0.01f, -1.0e6f);
    aabb.max = MakeVec3(1.0e6f, plane_y + 0.01f, 1.0e6f);
    return aabb;
}

__device__ collision::AABB BoxAabb(math::Transform transform, math::Vec3 half_extents) {
    collision::AABB aabb;
    aabb.min = MakeVec3(FLT_MAX, FLT_MAX, FLT_MAX);
    aabb.max = MakeVec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int corner_index = 0; corner_index < 8; ++corner_index) {
        const math::Vec3 corner = MakeVec3(
            (corner_index & 1) ? half_extents.x : -half_extents.x,
            (corner_index & 2) ? half_extents.y : -half_extents.y,
            (corner_index & 4) ? half_extents.z : -half_extents.z);
        Expand(aabb, TransformPoint(transform, corner));
    }
    return aabb;
}

__device__ math::Vec3 AabbCenter(collision::AABB aabb) {
    return Scale(Add(aabb.min, aabb.max), 0.5f);
}

__device__ bool Overlaps(collision::AABB a, collision::AABB b) {
    if (a.max.x < b.min.x || a.min.x > b.max.x) return false;
    if (a.max.y < b.min.y || a.min.y > b.max.y) return false;
    if (a.max.z < b.min.z || a.min.z > b.max.z) return false;
    return true;
}

__device__ uint32_t PairSlot(uint32_t shape_count, uint32_t i, uint32_t j) {
    return (i * (2u * shape_count - i - 1u)) / 2u + (j - i - 1u);
}

__device__ collision::CollisionPair LocalPairFromSlot(uint32_t shape_count,
                                                      uint32_t local_pair_slot) {
    for (uint32_t i = 0; i < shape_count; ++i) {
        for (uint32_t j = i + 1u; j < shape_count; ++j) {
            if (PairSlot(shape_count, i, j) == local_pair_slot) {
                return {i, j};
            }
        }
    }
    return {0u, 0u};
}

__device__ uint32_t FlattenBody(uint32_t instance_index,
                                uint32_t body_count_per_instance,
                                uint32_t local_body) {
    if (local_body == kInvalidBody) {
        return kInvalidBody;
    }
    return instance_index * body_count_per_instance + local_body;
}

__device__ float BodyInvMass(uint32_t flat_body,
                             uint32_t total_body_count,
                             uint32_t body_count_per_instance,
                             const float* inv_masses) {
    if (flat_body >= total_body_count || flat_body == kInvalidBody) {
        return 0.0f;
    }
    return inv_masses[flat_body % body_count_per_instance];
}

__device__ math::Vec3 BodyInvInertia(uint32_t flat_body,
                                     uint32_t total_body_count,
                                     uint32_t body_count_per_instance,
                                     const math::Vec3* inv_inertias) {
    if (flat_body >= total_body_count || flat_body == kInvalidBody) {
        return ZeroVec3();
    }
    return inv_inertias[flat_body % body_count_per_instance];
}

__device__ bool IsDynamic(uint32_t local_body, const float* inv_masses) {
    return local_body != kInvalidBody && inv_masses[local_body] > 0.0f;
}

__device__ bool IsPlane(scene::ShapeType type) {
    return type == scene::ShapeType::Plane;
}

__device__ bool IsSphere(scene::ShapeType type) {
    return type == scene::ShapeType::Sphere;
}

__device__ CudaBatchedContactManifold EmptyBatchedManifold() {
    CudaBatchedContactManifold manifold;
    manifold.instance_index = 0u;
    manifold.body_a = kInvalidBody;
    manifold.body_b = kInvalidBody;
    manifold.point_count = 0u;
    manifold.friction = 0.5f;
    manifold.restitution = 0.0f;
    for (uint32_t point = 0; point < constraint::ContactManifold::kMaxPoints; ++point) {
        manifold.points[point] = constraint::ContactPoint{};
    }
    return manifold;
}

__device__ void AddSinglePointManifold(uint32_t instance_index,
                                       uint32_t body_a,
                                       uint32_t body_b,
                                       math::Vec3 position,
                                       math::Vec3 normal,
                                       float penetration,
                                       CudaBatchedContactManifold* manifold) {
    if (penetration <= 0.0f) {
        return;
    }

    manifold->instance_index = instance_index;
    manifold->body_a = body_a;
    manifold->body_b = body_b;
    manifold->point_count = 1u;
    manifold->friction = 0.5f;
    manifold->restitution = 0.0f;
    manifold->points[0].position = position;
    manifold->points[0].normal = Normalize(normal);
    manifold->points[0].penetration = penetration;
    manifold->points[0].normal_impulse = 0.0f;
    manifold->points[0].friction_impulse_1 = 0.0f;
    manifold->points[0].friction_impulse_2 = 0.0f;
}

__device__ void GeneratePlaneContact(uint32_t instance_index,
                                     uint32_t dynamic_body,
                                     uint32_t plane_body,
                                     scene::ShapeType dynamic_type,
                                     math::Transform dynamic_transform,
                                     math::Transform plane_transform,
                                     collision::AABB dynamic_aabb,
                                     float dynamic_radius,
                                     CudaBatchedContactManifold* manifold) {
    const float plane_y = plane_transform.position.y;
    if (dynamic_type == scene::ShapeType::Sphere) {
        const float bottom = dynamic_transform.position.y - dynamic_radius;
        AddSinglePointManifold(
            instance_index,
            dynamic_body,
            plane_body,
            MakeVec3(dynamic_transform.position.x, plane_y, dynamic_transform.position.z),
            UnitY(),
            plane_y - bottom,
            manifold);
        return;
    }

    const float bottom = dynamic_aabb.min.y;
    const math::Vec3 center = AabbCenter(dynamic_aabb);
    AddSinglePointManifold(
        instance_index,
        dynamic_body,
        plane_body,
        MakeVec3(center.x, plane_y, center.z),
        UnitY(),
        plane_y - bottom,
        manifold);
}

__device__ void GenerateSphereSphereContact(uint32_t instance_index,
                                            uint32_t body_a,
                                            uint32_t body_b,
                                            math::Transform transform_a,
                                            math::Transform transform_b,
                                            float radius_a,
                                            float radius_b,
                                            CudaBatchedContactManifold* manifold) {
    const math::Vec3 delta = Sub(transform_a.position, transform_b.position);
    const float distance = Length(delta);
    const float penetration = radius_a + radius_b - distance;
    if (penetration <= 0.0f) {
        return;
    }

    const math::Vec3 normal =
        distance > 1.0e-6f ? Scale(delta, 1.0f / distance) : UnitY();
    const math::Vec3 position = Add(transform_b.position, Scale(normal, radius_b));
    AddSinglePointManifold(instance_index, body_a, body_b, position, normal, penetration, manifold);
}

__device__ void GenerateBoxBoxContact(uint32_t instance_index,
                                      uint32_t body_a,
                                      uint32_t body_b,
                                      collision::AABB a,
                                      collision::AABB b,
                                      CudaBatchedContactManifold* manifold) {
    const float overlap_x = fminf(a.max.x, b.max.x) - fmaxf(a.min.x, b.min.x);
    const float overlap_y = fminf(a.max.y, b.max.y) - fmaxf(a.min.y, b.min.y);
    const float overlap_z = fminf(a.max.z, b.max.z) - fmaxf(a.min.z, b.min.z);
    if (overlap_x <= 0.0f || overlap_y <= 0.0f || overlap_z <= 0.0f) {
        return;
    }

    math::Vec3 normal = UnitX();
    float penetration = overlap_x;
    if (overlap_y < penetration) {
        normal = UnitY();
        penetration = overlap_y;
    }
    if (overlap_z < penetration) {
        normal = UnitZ();
        penetration = overlap_z;
    }

    if (Dot(Sub(AabbCenter(a), AabbCenter(b)), normal) < 0.0f) {
        normal = Scale(normal, -1.0f);
    }

    const math::Vec3 position = MakeVec3(
        (fmaxf(a.min.x, b.min.x) + fminf(a.max.x, b.max.x)) * 0.5f,
        (fmaxf(a.min.y, b.min.y) + fminf(a.max.y, b.max.y)) * 0.5f,
        (fmaxf(a.min.z, b.min.z) + fminf(a.max.z, b.max.z)) * 0.5f);
    AddSinglePointManifold(instance_index, body_a, body_b, position, normal, penetration, manifold);
}

__device__ void ClearBlock(constraint::ConstraintBlock* block) {
    *block = constraint::ConstraintBlock{};
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

__device__ constraint::ConstraintBlock BuildBatchedContactBlock(
    const CudaBatchedContactManifold& manifold,
    uint32_t body_count_per_instance) {
    constraint::ConstraintBlock block;
    ClearBlock(&block);
    block.type = constraint::ConstraintType::Contact;
    block.body_a = FlattenBody(manifold.instance_index, body_count_per_instance, manifold.body_a);
    block.body_b = FlattenBody(manifold.instance_index, body_count_per_instance, manifold.body_b);
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

__device__ constraint::ConstraintBlock BuildBatchedRevoluteBlock(uint32_t instance_index,
                                                                 uint32_t body_count_per_instance,
                                                                 uint32_t parent,
                                                                 uint32_t child,
                                                                 math::Vec3 axis,
                                                                 math::Transform parent_frame,
                                                                 math::Transform child_frame,
                                                                 bool fixed_joint) {
    constraint::ConstraintBlock block;
    ClearBlock(&block);
    block.type = constraint::ConstraintType::Joint;
    block.body_a = FlattenBody(instance_index, body_count_per_instance, parent);
    block.body_b = FlattenBody(instance_index, body_count_per_instance, child);
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

__device__ constraint::ConstraintBlock BuildBatchedDriveBlock(uint32_t instance_index,
                                                              uint32_t body_count_per_instance,
                                                              uint32_t child,
                                                              uint32_t parent,
                                                              math::Vec3 axis,
                                                              scene::ActuatorType type,
                                                              float gain,
                                                              float force_limit) {
    constraint::ConstraintBlock block;
    ClearBlock(&block);
    block.type = constraint::ConstraintType::Drive;
    block.body_a = FlattenBody(instance_index, body_count_per_instance, child);
    block.body_b = FlattenBody(instance_index, body_count_per_instance, parent);
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

__device__ math::Vec3 BodyLinearVelocity(uint32_t flat_body,
                                         uint32_t total_body_count,
                                         const math::Vec3* velocities) {
    if (flat_body >= total_body_count || flat_body == kInvalidBody) {
        return ZeroVec3();
    }
    return velocities[flat_body];
}

__device__ math::Vec3 BodyAngularVelocity(uint32_t flat_body,
                                          uint32_t total_body_count,
                                          const math::Vec3* velocities) {
    if (flat_body >= total_body_count || flat_body == kInvalidBody) {
        return ZeroVec3();
    }
    return velocities[flat_body];
}

__device__ math::Transform BodyPose(uint32_t flat_body,
                                    uint32_t total_body_count,
                                    const math::Transform* poses) {
    math::Transform transform;
    transform.position = ZeroVec3();
    transform.rotation = MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
    if (flat_body < total_body_count && flat_body != kInvalidBody) {
        transform = poses[flat_body];
    }
    return transform;
}

__device__ float ComputeJv(constraint::ConstraintBlock block,
                           uint32_t row,
                           uint32_t total_body_count,
                           const math::Vec3* linear_velocities,
                           const math::Vec3* angular_velocities) {
    float jv = 0.0f;
    jv += Dot(block.jacobian_linear_a[row],
              BodyLinearVelocity(block.body_a, total_body_count, linear_velocities));
    jv += Dot(block.jacobian_angular_a[row],
              BodyAngularVelocity(block.body_a, total_body_count, angular_velocities));
    jv += Dot(block.jacobian_linear_b[row],
              BodyLinearVelocity(block.body_b, total_body_count, linear_velocities));
    jv += Dot(block.jacobian_angular_b[row],
              BodyAngularVelocity(block.body_b, total_body_count, angular_velocities));
    return jv;
}

__device__ float ComputeEffectiveMass(constraint::ConstraintBlock block,
                                      uint32_t row,
                                      uint32_t total_body_count,
                                      uint32_t body_count_per_instance,
                                      const float* inv_masses,
                                      const math::Vec3* inv_inertias) {
    float diag = 0.0f;
    const float inv_mass_a =
        BodyInvMass(block.body_a, total_body_count, body_count_per_instance, inv_masses);
    const float inv_mass_b =
        BodyInvMass(block.body_b, total_body_count, body_count_per_instance, inv_masses);
    const math::Vec3 inv_inertia_a =
        BodyInvInertia(block.body_a, total_body_count, body_count_per_instance, inv_inertias);
    const math::Vec3 inv_inertia_b =
        BodyInvInertia(block.body_b, total_body_count, body_count_per_instance, inv_inertias);

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

__device__ float JointRowMass(math::Vec3 linear_a,
                              math::Vec3 angular_a,
                              math::Vec3 linear_b,
                              math::Vec3 angular_b,
                              uint32_t body_a,
                              uint32_t body_b,
                              uint32_t total_body_count,
                              uint32_t body_count_per_instance,
                              const float* inv_masses,
                              const math::Vec3* inv_inertias) {
    float diag = 0.0f;
    const float inv_mass_a =
        BodyInvMass(body_a, total_body_count, body_count_per_instance, inv_masses);
    const float inv_mass_b =
        BodyInvMass(body_b, total_body_count, body_count_per_instance, inv_masses);
    const math::Vec3 inv_inertia_a =
        BodyInvInertia(body_a, total_body_count, body_count_per_instance, inv_inertias);
    const math::Vec3 inv_inertia_b =
        BodyInvInertia(body_b, total_body_count, body_count_per_instance, inv_inertias);
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

__device__ void ApplyImpulse(constraint::ConstraintBlock block,
                             uint32_t row,
                             float delta_impulse,
                             uint32_t total_body_count,
                             uint32_t body_count_per_instance,
                             const float* inv_masses,
                             const math::Vec3* inv_inertias,
                             math::Vec3* linear_velocities,
                             math::Vec3* angular_velocities) {
    if (block.body_a < total_body_count && block.body_a != kInvalidBody) {
        const float inv_mass =
            BodyInvMass(block.body_a, total_body_count, body_count_per_instance, inv_masses);
        const math::Vec3 inv_inertia =
            BodyInvInertia(block.body_a, total_body_count, body_count_per_instance, inv_inertias);
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

    if (block.body_b < total_body_count && block.body_b != kInvalidBody) {
        const float inv_mass =
            BodyInvMass(block.body_b, total_body_count, body_count_per_instance, inv_masses);
        const math::Vec3 inv_inertia =
            BodyInvInertia(block.body_b, total_body_count, body_count_per_instance, inv_inertias);
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

__device__ void ApplyPositionCorrection(uint32_t flat_body,
                                        uint32_t total_body_count,
                                        uint32_t body_count_per_instance,
                                        math::Vec3 linear_jacobian,
                                        math::Vec3 angular_jacobian,
                                        float position_impulse,
                                        const float* inv_masses,
                                        const math::Vec3* inv_inertias,
                                        math::Transform* poses) {
    if (flat_body >= total_body_count || flat_body == kInvalidBody) {
        return;
    }

    const float inv_mass =
        BodyInvMass(flat_body, total_body_count, body_count_per_instance, inv_masses);
    const math::Vec3 inv_inertia =
        BodyInvInertia(flat_body, total_body_count, body_count_per_instance, inv_inertias);
    if (inv_mass > 0.0f) {
        poses[flat_body].position =
            Add(poses[flat_body].position, Scale(linear_jacobian, inv_mass * position_impulse));
    }

    const math::Vec3 angular_delta = MakeVec3(
        angular_jacobian.x * inv_inertia.x * position_impulse,
        angular_jacobian.y * inv_inertia.y * position_impulse,
        angular_jacobian.z * inv_inertia.z * position_impulse);
    ApplyAngularCorrection(&poses[flat_body], angular_delta);
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

__global__ void GenerateBatchedAabbsKernel(uint32_t total_shape_count,
                                           uint32_t shape_count_per_instance,
                                           uint32_t body_count_per_instance,
                                           const math::Transform* poses,
                                           const scene::BodyId* shape_body_ids,
                                           const scene::ShapeType* shape_types,
                                           const math::Transform* shape_local_transforms,
                                           const math::Vec3* shape_half_extents,
                                           const float* shape_radii,
                                           collision::AABB* aabbs) {
    const uint32_t flat_shape = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_shape >= total_shape_count || shape_count_per_instance == 0u) {
        return;
    }

    const uint32_t instance_index = flat_shape / shape_count_per_instance;
    const uint32_t local_shape = flat_shape % shape_count_per_instance;
    const scene::BodyId local_body = shape_body_ids[local_shape];
    const uint32_t flat_body = FlattenBody(instance_index, body_count_per_instance, local_body);
    const math::Transform world_transform =
        Compose(poses[flat_body], shape_local_transforms[local_shape]);
    const scene::ShapeType type = shape_types[local_shape];

    if (type == scene::ShapeType::Sphere) {
        aabbs[flat_shape] = SphereAabb(world_transform.position, shape_radii[local_shape]);
        return;
    }
    if (type == scene::ShapeType::Plane) {
        aabbs[flat_shape] = PlaneAabb(world_transform.position.y);
        return;
    }

    aabbs[flat_shape] = BoxAabb(world_transform, shape_half_extents[local_shape]);
}

__global__ void GenerateBatchedPairSlotsKernel(uint32_t total_pair_slot_count,
                                               uint32_t pair_slot_count_per_instance,
                                               uint32_t shape_count_per_instance,
                                               const collision::AABB* aabbs,
                                               collision::CollisionPair* pairs,
                                               uint32_t* pair_instance_indices,
                                               uint8_t* pair_active_flags,
                                               uint32_t* pair_count) {
    const uint32_t flat_pair_slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_pair_slot >= total_pair_slot_count || pair_slot_count_per_instance == 0u) {
        return;
    }

    const uint32_t instance_index = flat_pair_slot / pair_slot_count_per_instance;
    const uint32_t local_pair_slot = flat_pair_slot % pair_slot_count_per_instance;
    const collision::CollisionPair pair =
        LocalPairFromSlot(shape_count_per_instance, local_pair_slot);
    const uint32_t shape_base = instance_index * shape_count_per_instance;

    pairs[flat_pair_slot] = pair;
    pair_instance_indices[flat_pair_slot] = instance_index;

    const bool overlaps = Overlaps(aabbs[shape_base + pair.body_a],
                                   aabbs[shape_base + pair.body_b]);
    pair_active_flags[flat_pair_slot] = overlaps ? 1u : 0u;
    if (overlaps) {
        atomicAdd(pair_count, 1u);
    }
}

__global__ void InitializeBatchedContactReportKernel(uint32_t instance_count,
                                                     const uint32_t* pair_count,
                                                     CudaBatchedContactReport* report) {
    report->instance_count = instance_count;
    report->pair_count = *pair_count;
    report->contact_manifold_count = 0u;
    report->contact_point_count = 0u;
}

__global__ void GenerateBatchedContactsKernel(
    uint32_t total_pair_slot_count,
    uint32_t shape_count_per_instance,
    uint32_t body_count_per_instance,
    const collision::CollisionPair* pairs,
    const uint32_t* pair_instance_indices,
    const uint8_t* pair_active_flags,
    const collision::AABB* aabbs,
    const math::Transform* poses,
    const float* inv_masses,
    const scene::BodyId* shape_body_ids,
    const scene::ShapeType* shape_types,
    const math::Transform* shape_local_transforms,
    const float* shape_radii,
    CudaBatchedContactManifold* manifolds) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_pair_slot_count) {
        return;
    }

    manifolds[slot] = EmptyBatchedManifold();

    if (pair_active_flags[slot] == 0u) {
        return;
    }

    const uint32_t instance_index = pair_instance_indices[slot];
    const collision::CollisionPair pair = pairs[slot];
    const scene::BodyId first_body = shape_body_ids[pair.body_a];
    const scene::BodyId second_body = shape_body_ids[pair.body_b];
    if (first_body == second_body) {
        return;
    }

    const bool first_dynamic = IsDynamic(first_body, inv_masses);
    const bool second_dynamic = IsDynamic(second_body, inv_masses);
    if (!first_dynamic && !second_dynamic) {
        return;
    }

    uint32_t dynamic_shape = pair.body_a;
    uint32_t other_shape = pair.body_b;
    if (!first_dynamic && second_dynamic) {
        dynamic_shape = pair.body_b;
        other_shape = pair.body_a;
    }

    const scene::BodyId dynamic_body = shape_body_ids[dynamic_shape];
    const scene::BodyId other_body = shape_body_ids[other_shape];
    const scene::ShapeType dynamic_type = shape_types[dynamic_shape];
    const scene::ShapeType other_type = shape_types[other_shape];
    const uint32_t flat_dynamic_body =
        FlattenBody(instance_index, body_count_per_instance, dynamic_body);
    const uint32_t flat_other_body =
        FlattenBody(instance_index, body_count_per_instance, other_body);
    const math::Transform dynamic_transform =
        Compose(poses[flat_dynamic_body], shape_local_transforms[dynamic_shape]);
    const math::Transform other_transform =
        Compose(poses[flat_other_body], shape_local_transforms[other_shape]);
    const uint32_t shape_base = instance_index * shape_count_per_instance;

    if (IsPlane(other_type)) {
        GeneratePlaneContact(instance_index,
                             dynamic_body,
                             other_body,
                             dynamic_type,
                             dynamic_transform,
                             other_transform,
                             aabbs[shape_base + dynamic_shape],
                             shape_radii[dynamic_shape],
                             &manifolds[slot]);
        return;
    }

    if (IsPlane(dynamic_type)) {
        GeneratePlaneContact(instance_index,
                             other_body,
                             dynamic_body,
                             other_type,
                             other_transform,
                             dynamic_transform,
                             aabbs[shape_base + other_shape],
                             shape_radii[other_shape],
                             &manifolds[slot]);
        return;
    }

    if (IsSphere(dynamic_type) && IsSphere(other_type)) {
        GenerateSphereSphereContact(instance_index,
                                    dynamic_body,
                                    other_body,
                                    dynamic_transform,
                                    other_transform,
                                    shape_radii[dynamic_shape],
                                    shape_radii[other_shape],
                                    &manifolds[slot]);
        return;
    }

    GenerateBoxBoxContact(instance_index,
                          dynamic_body,
                          other_body,
                          aabbs[shape_base + dynamic_shape],
                          aabbs[shape_base + other_shape],
                          &manifolds[slot]);
}

__global__ void CountBatchedContactsKernel(uint32_t total_pair_slot_count,
                                           const CudaBatchedContactManifold* manifolds,
                                           CudaBatchedContactReport* report) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_pair_slot_count || manifolds[slot].point_count == 0u) {
        return;
    }

    atomicAdd(&report->contact_manifold_count, 1u);
    atomicAdd(&report->contact_point_count, manifolds[slot].point_count);
}

__global__ void ClearBatchedSolverReportKernel(uint32_t* block_count,
                                               CudaBatchedConstraintSolverReport* report) {
    *block_count = 0u;
    report->constraint_block_count = 0u;
    report->constraint_row_count = 0u;
    report->contact_constraint_count = 0u;
    report->joint_constraint_count = 0u;
    report->drive_constraint_count = 0u;
    report->velocity_iterations = 0u;
    report->position_iterations = 0u;
    report->max_position_error = 0.0f;
}

__global__ void AssembleBatchedContactBlocksKernel(
    uint32_t total_pair_slot_count,
    uint32_t body_count_per_instance,
    const CudaBatchedContactManifold* manifolds,
    constraint::ConstraintBlock* blocks,
    uint32_t* block_count) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_pair_slot_count || manifolds[slot].point_count == 0u) {
        return;
    }

    const uint32_t out_index = atomicAdd(block_count, 1u);
    blocks[out_index] = BuildBatchedContactBlock(manifolds[slot], body_count_per_instance);
}

__global__ void AssembleBatchedJointBlocksKernel(uint32_t total_joint_count,
                                                 uint32_t joint_count_per_instance,
                                                 uint32_t body_count_per_instance,
                                                 const scene::JointType* joint_types,
                                                 const scene::BodyId* joint_parent_bodies,
                                                 const scene::BodyId* joint_child_bodies,
                                                 const math::Vec3* joint_axes,
                                                 const math::Transform* joint_parent_frames,
                                                 const math::Transform* joint_child_frames,
                                                 constraint::ConstraintBlock* blocks,
                                                 uint32_t* block_count) {
    const uint32_t flat_joint = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_joint >= total_joint_count || joint_count_per_instance == 0u) {
        return;
    }

    const uint32_t instance_index = flat_joint / joint_count_per_instance;
    const uint32_t local_joint = flat_joint % joint_count_per_instance;
    const scene::BodyId parent = joint_parent_bodies[local_joint];
    const scene::BodyId child = joint_child_bodies[local_joint];
    if (parent == scene::kInvalidBody && child == scene::kInvalidBody) {
        return;
    }

    const bool fixed_joint = joint_types[local_joint] == scene::JointType::Fixed;
    const uint32_t out_index = atomicAdd(block_count, 1u);
    blocks[out_index] = BuildBatchedRevoluteBlock(instance_index,
                                                  body_count_per_instance,
                                                  parent,
                                                  child,
                                                  joint_axes[local_joint],
                                                  joint_parent_frames[local_joint],
                                                  joint_child_frames[local_joint],
                                                  fixed_joint);
}

__global__ void AssembleBatchedDriveBlocksKernel(uint32_t total_actuator_count,
                                                 uint32_t actuator_count_per_instance,
                                                 uint32_t joint_count_per_instance,
                                                 uint32_t body_count_per_instance,
                                                 const scene::JointId* actuator_joint_ids,
                                                 const scene::ActuatorType* actuator_types,
                                                 const float* actuator_gains,
                                                 const float* actuator_force_limits,
                                                 const scene::BodyId* joint_parent_bodies,
                                                 const scene::BodyId* joint_child_bodies,
                                                 const math::Vec3* joint_axes,
                                                 constraint::ConstraintBlock* blocks,
                                                 uint32_t* block_count) {
    const uint32_t flat_actuator = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_actuator >= total_actuator_count || actuator_count_per_instance == 0u) {
        return;
    }

    const uint32_t instance_index = flat_actuator / actuator_count_per_instance;
    const uint32_t local_actuator = flat_actuator % actuator_count_per_instance;
    const scene::JointId joint_id = actuator_joint_ids[local_actuator];
    if (joint_id >= joint_count_per_instance || joint_id == scene::kInvalidJoint) {
        return;
    }

    const scene::BodyId parent = joint_parent_bodies[joint_id];
    const scene::BodyId child = joint_child_bodies[joint_id];
    if (parent == scene::kInvalidBody && child == scene::kInvalidBody) {
        return;
    }

    const uint32_t out_index = atomicAdd(block_count, 1u);
    blocks[out_index] = BuildBatchedDriveBlock(instance_index,
                                               body_count_per_instance,
                                               child,
                                               parent,
                                               joint_axes[joint_id],
                                               actuator_types[local_actuator],
                                               actuator_gains[local_actuator],
                                               actuator_force_limits[local_actuator]);
}

__global__ void FinalizeBatchedSolverReportKernel(
    const constraint::ConstraintBlock* blocks,
    const uint32_t* block_count,
    CudaBatchedConstraintSolverReport* report) {
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

__global__ void PrecomputeBatchedBlocksKernel(uint32_t total_body_count,
                                              uint32_t body_count_per_instance,
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
            ComputeEffectiveMass(block,
                                 row,
                                 total_body_count,
                                 body_count_per_instance,
                                 inv_masses,
                                 inv_inertias);
        if (block.type == constraint::ConstraintType::Contact
            && row < ContactNormalRowCount(block)
            && block.restitution > 0.0f) {
            const float jv = ComputeJv(block,
                                       row,
                                       total_body_count,
                                       linear_velocities,
                                       angular_velocities);
            if (jv < 0.0f) {
                block.rhs[row] = fmaxf(block.rhs[row], -block.restitution * jv);
            }
        }
    }
    blocks[block_index] = block;
}

__global__ void SolveBatchedVelocityIterationKernel(uint32_t total_body_count,
                                                    uint32_t body_count_per_instance,
                                                    constraint::ConstraintBlock* blocks,
                                                    const uint32_t* block_count,
                                                    const float* inv_masses,
                                                    const math::Vec3* inv_inertias,
                                                    math::Vec3* linear_velocities,
                                                    math::Vec3* angular_velocities) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) {
        return;
    }

    const uint32_t count = *block_count;
    for (uint32_t block_index = 0; block_index < count; ++block_index) {
        constraint::ConstraintBlock block = blocks[block_index];
        for (uint32_t row = 0; row < block.row_count; ++row) {
            const float jv =
                ComputeJv(block, row, total_body_count, linear_velocities, angular_velocities);
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
                             total_body_count,
                             body_count_per_instance,
                             inv_masses,
                             inv_inertias,
                             linear_velocities,
                             angular_velocities);
            }
        }
        blocks[block_index] = block;
    }
}

__global__ void SolveBatchedContactPositionIterationKernel(
    uint32_t total_body_count,
    uint32_t body_count_per_instance,
    constraint::ConstraintBlock* blocks,
    const uint32_t* block_count,
    const float* inv_masses,
    const math::Vec3* inv_inertias,
    math::Transform* poses,
    float slop,
    float baumgarte,
    CudaBatchedConstraintSolverReport* report) {
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
                                    total_body_count,
                                    body_count_per_instance,
                                    block.jacobian_linear_a[row],
                                    ZeroVec3(),
                                    position_impulse,
                                    inv_masses,
                                    inv_inertias,
                                    poses);
            ApplyPositionCorrection(block.body_b,
                                    total_body_count,
                                    body_count_per_instance,
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

__global__ void SolveBatchedJointPositionIterationKernel(
    uint32_t total_body_count,
    uint32_t body_count_per_instance,
    const constraint::ConstraintBlock* blocks,
    const uint32_t* block_count,
    const float* inv_masses,
    const math::Vec3* inv_inertias,
    math::Transform* poses,
    float slop,
    float baumgarte,
    CudaBatchedConstraintSolverReport* report) {
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
            const math::Transform pose_a = BodyPose(block.body_a, total_body_count, poses);
            const math::Transform pose_b = BodyPose(block.body_b, total_body_count, poses);
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
                                                      total_body_count,
                                                      body_count_per_instance,
                                                      inv_masses,
                                                      inv_inertias);
            if (effective_mass <= 0.0f) {
                continue;
            }

            const float position_impulse = correction * effective_mass;
            ApplyPositionCorrection(block.body_a,
                                    total_body_count,
                                    body_count_per_instance,
                                    linear_a,
                                    angular_a,
                                    position_impulse,
                                    inv_masses,
                                    inv_inertias,
                                    poses);
            ApplyPositionCorrection(block.body_b,
                                    total_body_count,
                                    body_count_per_instance,
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

__global__ void SetBatchedSolverIterationReportKernel(
    uint32_t velocity_iterations,
    uint32_t position_iterations,
    CudaBatchedConstraintSolverReport* report) {
    report->velocity_iterations = velocity_iterations;
    report->position_iterations = position_iterations;
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

CudaBatchedBroadphaseResult::CudaBatchedBroadphaseResult(
    uint32_t instance_count,
    uint32_t shape_count_per_instance,
    uint32_t pair_slot_count_per_instance,
    phi::Buffer aabbs,
    phi::Buffer pairs,
    phi::Buffer pair_instance_indices,
    phi::Buffer pair_active_flags,
    phi::Buffer pair_count)
    : instance_count_(instance_count)
    , shape_count_per_instance_(shape_count_per_instance)
    , pair_slot_count_per_instance_(pair_slot_count_per_instance)
    , aabbs_(std::move(aabbs))
    , pairs_(std::move(pairs))
    , pair_instance_indices_(std::move(pair_instance_indices))
    , pair_active_flags_(std::move(pair_active_flags))
    , pair_count_(std::move(pair_count)) {}

uint32_t CudaBatchedBroadphaseResult::PairCount() const {
    if (pair_count_.Size() == 0u) {
        return 0u;
    }
    uint32_t pair_count = 0;
    pair_count_.CopyToHost(&pair_count, sizeof(pair_count));
    return pair_count;
}

const collision::AABB* CudaBatchedBroadphaseResult::DeviceAabbs() const {
    return static_cast<const collision::AABB*>(aabbs_.Data());
}

const collision::CollisionPair* CudaBatchedBroadphaseResult::DevicePairs() const {
    return static_cast<const collision::CollisionPair*>(pairs_.Data());
}

const uint32_t* CudaBatchedBroadphaseResult::DevicePairInstanceIndices() const {
    return static_cast<const uint32_t*>(pair_instance_indices_.Data());
}

const uint8_t* CudaBatchedBroadphaseResult::DevicePairActiveFlags() const {
    return static_cast<const uint8_t*>(pair_active_flags_.Data());
}

const uint32_t* CudaBatchedBroadphaseResult::DevicePairCount() const {
    return static_cast<const uint32_t*>(pair_count_.Data());
}

CudaBatchedContactResult::CudaBatchedContactResult(uint32_t total_pair_slot_count,
                                                   phi::Buffer manifolds,
                                                   phi::Buffer report)
    : total_pair_slot_count_(total_pair_slot_count)
    , manifolds_(std::move(manifolds))
    , report_(std::move(report)) {}

CudaBatchedContactReport CudaBatchedContactResult::DownloadReport() const {
    CudaBatchedContactReport report;
    if (report_.Size() > 0u) {
        report_.CopyToHost(&report, sizeof(report));
    }
    return report;
}

std::vector<CudaBatchedContactManifold> CudaBatchedContactResult::DownloadManifolds() const {
    const auto all_manifolds =
        DownloadVector<CudaBatchedContactManifold>(manifolds_, total_pair_slot_count_);

    std::vector<CudaBatchedContactManifold> contacts;
    for (const auto& manifold : all_manifolds) {
        if (manifold.point_count > 0u) {
            contacts.push_back(manifold);
        }
    }
    return contacts;
}

const CudaBatchedContactManifold* CudaBatchedContactResult::DeviceManifolds() const {
    return static_cast<const CudaBatchedContactManifold*>(manifolds_.Data());
}

const CudaBatchedContactReport* CudaBatchedContactResult::DeviceReport() const {
    return static_cast<const CudaBatchedContactReport*>(report_.Data());
}

CudaBatchedConstraintSolverResult::CudaBatchedConstraintSolverResult(
    uint32_t block_capacity,
    phi::Buffer blocks,
    phi::Buffer block_count,
    phi::Buffer report)
    : block_capacity_(block_capacity)
    , blocks_(std::move(blocks))
    , block_count_(std::move(block_count))
    , report_(std::move(report)) {}

CudaBatchedConstraintSolverReport CudaBatchedConstraintSolverResult::DownloadReport() const {
    CudaBatchedConstraintSolverReport report;
    if (report_.Size() > 0u) {
        report_.CopyToHost(&report, sizeof(report));
    }
    return report;
}

BatchedDeviceWorld::BatchedDeviceWorld(uint32_t instance_count,
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
                                       phi::Buffer torques)
    : instance_count_(instance_count)
    , body_count_per_instance_(body_count_per_instance)
    , shape_count_per_instance_(shape_count_per_instance)
    , joint_count_per_instance_(joint_count_per_instance)
    , actuator_count_per_instance_(actuator_count_per_instance)
    , body_inv_masses_(std::move(body_inv_masses))
    , body_inv_inertias_(std::move(body_inv_inertias))
    , shape_types_(std::move(shape_types))
    , shape_body_ids_(std::move(shape_body_ids))
    , shape_local_transforms_(std::move(shape_local_transforms))
    , shape_half_extents_(std::move(shape_half_extents))
    , shape_radii_(std::move(shape_radii))
    , joint_types_(std::move(joint_types))
    , joint_parent_bodies_(std::move(joint_parent_bodies))
    , joint_child_bodies_(std::move(joint_child_bodies))
    , joint_axes_(std::move(joint_axes))
    , joint_parent_frames_(std::move(joint_parent_frames))
    , joint_child_frames_(std::move(joint_child_frames))
    , actuator_types_(std::move(actuator_types))
    , actuator_joint_ids_(std::move(actuator_joint_ids))
    , actuator_gains_(std::move(actuator_gains))
    , actuator_force_limits_(std::move(actuator_force_limits))
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

bool BatchedDeviceWorld::HasUploadedShapeTables() const {
    const auto shape_count = ShapeCountPerInstance();
    if (shape_count == 0u) {
        return true;
    }

    return shape_types_.Size() == shape_count * sizeof(scene::ShapeType)
        && shape_body_ids_.Size() == shape_count * sizeof(scene::BodyId)
        && shape_local_transforms_.Size() == shape_count * sizeof(math::Transform)
        && shape_half_extents_.Size() == shape_count * sizeof(math::Vec3)
        && shape_radii_.Size() == shape_count * sizeof(float);
}

bool BatchedDeviceWorld::HasUploadedJointTables() const {
    const auto joint_count = JointCountPerInstance();
    if (joint_count == 0u) {
        return true;
    }

    return joint_types_.Size() == joint_count * sizeof(scene::JointType)
        && joint_parent_bodies_.Size() == joint_count * sizeof(scene::BodyId)
        && joint_child_bodies_.Size() == joint_count * sizeof(scene::BodyId)
        && joint_axes_.Size() == joint_count * sizeof(math::Vec3)
        && joint_parent_frames_.Size() == joint_count * sizeof(math::Transform)
        && joint_child_frames_.Size() == joint_count * sizeof(math::Transform);
}

bool BatchedDeviceWorld::HasUploadedActuatorTables() const {
    const auto actuator_count = ActuatorCountPerInstance();
    if (actuator_count == 0u) {
        return true;
    }

    return actuator_types_.Size() == actuator_count * sizeof(scene::ActuatorType)
        && actuator_joint_ids_.Size() == actuator_count * sizeof(scene::JointId)
        && actuator_gains_.Size() == actuator_count * sizeof(float)
        && actuator_force_limits_.Size() == actuator_count * sizeof(float);
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

const math::Transform* BatchedDeviceWorld::DevicePoses() const {
    return static_cast<const math::Transform*>(poses_.Data());
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

const scene::ShapeType* BatchedDeviceWorld::DeviceShapeTypes() const {
    return static_cast<const scene::ShapeType*>(shape_types_.Data());
}

const scene::BodyId* BatchedDeviceWorld::DeviceShapeBodyIds() const {
    return static_cast<const scene::BodyId*>(shape_body_ids_.Data());
}

const math::Transform* BatchedDeviceWorld::DeviceShapeLocalTransforms() const {
    return static_cast<const math::Transform*>(shape_local_transforms_.Data());
}

const math::Vec3* BatchedDeviceWorld::DeviceShapeHalfExtents() const {
    return static_cast<const math::Vec3*>(shape_half_extents_.Data());
}

const float* BatchedDeviceWorld::DeviceShapeRadii() const {
    return static_cast<const float*>(shape_radii_.Data());
}

const scene::JointType* BatchedDeviceWorld::DeviceJointTypes() const {
    return static_cast<const scene::JointType*>(joint_types_.Data());
}

const scene::BodyId* BatchedDeviceWorld::DeviceJointParentBodies() const {
    return static_cast<const scene::BodyId*>(joint_parent_bodies_.Data());
}

const scene::BodyId* BatchedDeviceWorld::DeviceJointChildBodies() const {
    return static_cast<const scene::BodyId*>(joint_child_bodies_.Data());
}

const math::Vec3* BatchedDeviceWorld::DeviceJointAxes() const {
    return static_cast<const math::Vec3*>(joint_axes_.Data());
}

const math::Transform* BatchedDeviceWorld::DeviceJointParentFrames() const {
    return static_cast<const math::Transform*>(joint_parent_frames_.Data());
}

const math::Transform* BatchedDeviceWorld::DeviceJointChildFrames() const {
    return static_cast<const math::Transform*>(joint_child_frames_.Data());
}

const scene::ActuatorType* BatchedDeviceWorld::DeviceActuatorTypes() const {
    return static_cast<const scene::ActuatorType*>(actuator_types_.Data());
}

const scene::JointId* BatchedDeviceWorld::DeviceActuatorJointIds() const {
    return static_cast<const scene::JointId*>(actuator_joint_ids_.Data());
}

const float* BatchedDeviceWorld::DeviceActuatorGains() const {
    return static_cast<const float*>(actuator_gains_.Data());
}

const float* BatchedDeviceWorld::DeviceActuatorForceLimits() const {
    return static_cast<const float*>(actuator_force_limits_.Data());
}

BatchedDeviceWorld UploadBatchedDeviceWorld(
    const WorldTemplate& world_template,
    const std::vector<WorldInstance>& instances) {
    for (const auto& instance : instances) {
        ValidateInstanceShape(world_template, instance);
    }

    const uint32_t instance_count = static_cast<uint32_t>(instances.size());
    const uint32_t body_count = world_template.body_count;
    const uint32_t shape_count = world_template.shape_count;
    const uint32_t joint_count = world_template.joint_count;
    const uint32_t actuator_count = world_template.actuator_count;
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
    const auto shape_types =
        DefaultedCopy(world_template.shape_table.types, shape_count, scene::ShapeType::Box);
    const auto shape_body_ids =
        DefaultedCopy(world_template.shape_table.body_ids, shape_count, scene::kInvalidBody);
    const auto shape_local_transforms =
        DefaultedCopy(world_template.shape_table.local_transforms,
                      shape_count,
                      math::Transform::Identity());
    const auto shape_half_extents =
        DefaultedCopy(world_template.shape_table.half_extents,
                      shape_count,
                      math::Vec3::Zero());
    const auto shape_radii =
        DefaultedCopy(world_template.shape_table.radii, shape_count, 0.0f);
    const auto joint_types =
        DefaultedCopy(world_template.joint_table.types,
                      joint_count,
                      scene::JointType::Revolute);
    const auto joint_parent_bodies =
        DefaultedCopy(world_template.joint_table.parent_bodies,
                      joint_count,
                      scene::kInvalidBody);
    const auto joint_child_bodies =
        DefaultedCopy(world_template.joint_table.child_bodies,
                      joint_count,
                      scene::kInvalidBody);
    const auto joint_axes =
        DefaultedCopy(world_template.joint_table.axes,
                      joint_count,
                      math::Vec3::UnitZ());
    const auto joint_parent_frames =
        DefaultedCopy(world_template.joint_table.parent_frames,
                      joint_count,
                      math::Transform::Identity());
    const auto joint_child_frames =
        DefaultedCopy(world_template.joint_table.child_frames,
                      joint_count,
                      math::Transform::Identity());
    const auto actuator_types =
        DefaultedCopy(world_template.actuator_table.types,
                      actuator_count,
                      scene::ActuatorType::Motor);
    const auto actuator_joint_ids =
        DefaultedCopy(world_template.actuator_table.joint_ids,
                      actuator_count,
                      scene::kInvalidJoint);
    const auto actuator_gains =
        DefaultedCopy(world_template.actuator_table.gains,
                      actuator_count,
                      0.0f);
    const auto actuator_force_limits =
        DefaultedCopy(world_template.actuator_table.force_limits,
                      actuator_count,
                      kHugeLimit);

    return BatchedDeviceWorld(instance_count,
                              body_count,
                              shape_count,
                              joint_count,
                              actuator_count,
                              UploadVector(inv_masses),
                              UploadVector(inv_inertias),
                              UploadVector(shape_types),
                              UploadVector(shape_body_ids),
                              UploadVector(shape_local_transforms),
                              UploadVector(shape_half_extents),
                              UploadVector(shape_radii),
                              UploadVector(joint_types),
                              UploadVector(joint_parent_bodies),
                              UploadVector(joint_child_bodies),
                              UploadVector(joint_axes),
                              UploadVector(joint_parent_frames),
                              UploadVector(joint_child_frames),
                              UploadVector(actuator_types),
                              UploadVector(actuator_joint_ids),
                              UploadVector(actuator_gains),
                              UploadVector(actuator_force_limits),
                              UploadVector(poses),
                              UploadVector(linear_velocities),
                              UploadVector(angular_velocities),
                              UploadVector(forces),
                              UploadVector(torques));
}

CudaBatchedBroadphaseResult BuildBatchedCudaBroadphase(
    const BatchedDeviceWorld& batch) {
    if (!batch.HasUploadedState() || !batch.HasUploadedShapeTables()) {
        throw std::runtime_error(
            "BuildBatchedCudaBroadphase requires uploaded batched state and shape tables");
    }

    const uint32_t instance_count = batch.InstanceCount();
    const uint32_t shape_count = batch.ShapeCountPerInstance();
    const uint32_t pair_slot_count =
        shape_count > 1u ? (shape_count * (shape_count - 1u)) / 2u : 0u;
    const uint32_t total_shape_count = batch.TotalShapeCount();
    const uint32_t total_pair_slot_count = instance_count * pair_slot_count;

    phi::Buffer aabbs(total_shape_count * sizeof(collision::AABB), phi::MemoryKind::Device);
    phi::Buffer pairs(total_pair_slot_count * sizeof(collision::CollisionPair),
                      phi::MemoryKind::Device);
    phi::Buffer pair_instance_indices(total_pair_slot_count * sizeof(uint32_t),
                                      phi::MemoryKind::Device);
    phi::Buffer active_flags(total_pair_slot_count * sizeof(uint8_t), phi::MemoryKind::Device);
    phi::Buffer pair_count(sizeof(uint32_t), phi::MemoryKind::Device);
    CheckCuda(cudaMemset(pair_count.Data(), 0, sizeof(uint32_t)), "cudaMemset batched pair count");

    if (total_shape_count == 0u) {
        return CudaBatchedBroadphaseResult(instance_count,
                                          shape_count,
                                          pair_slot_count,
                                          std::move(aabbs),
                                          std::move(pairs),
                                          std::move(pair_instance_indices),
                                          std::move(active_flags),
                                          std::move(pair_count));
    }

    constexpr uint32_t kBlockSize = 128u;
    const uint32_t shape_blocks = (total_shape_count + kBlockSize - 1u) / kBlockSize;
    GenerateBatchedAabbsKernel<<<shape_blocks, kBlockSize>>>(
        total_shape_count,
        shape_count,
        batch.BodyCountPerInstance(),
        batch.DevicePoses(),
        batch.DeviceShapeBodyIds(),
        batch.DeviceShapeTypes(),
        batch.DeviceShapeLocalTransforms(),
        batch.DeviceShapeHalfExtents(),
        batch.DeviceShapeRadii(),
        static_cast<collision::AABB*>(aabbs.Data()));
    CheckCuda(cudaGetLastError(), "GenerateBatchedAabbsKernel launch");

    if (total_pair_slot_count > 0u) {
        const uint32_t pair_blocks =
            (total_pair_slot_count + kBlockSize - 1u) / kBlockSize;
        GenerateBatchedPairSlotsKernel<<<pair_blocks, kBlockSize>>>(
            total_pair_slot_count,
            pair_slot_count,
            shape_count,
            static_cast<const collision::AABB*>(aabbs.Data()),
            static_cast<collision::CollisionPair*>(pairs.Data()),
            static_cast<uint32_t*>(pair_instance_indices.Data()),
            static_cast<uint8_t*>(active_flags.Data()),
            static_cast<uint32_t*>(pair_count.Data()));
        CheckCuda(cudaGetLastError(), "GenerateBatchedPairSlotsKernel launch");
    }

    return CudaBatchedBroadphaseResult(instance_count,
                                      shape_count,
                                      pair_slot_count,
                                      std::move(aabbs),
                                      std::move(pairs),
                                      std::move(pair_instance_indices),
                                      std::move(active_flags),
                                      std::move(pair_count));
}

CudaBatchedContactResult GenerateBatchedCudaContacts(
    const BatchedDeviceWorld& batch,
    const CudaBatchedBroadphaseResult& broadphase) {
    if (!batch.HasUploadedState() || !batch.HasUploadedShapeTables()) {
        throw std::runtime_error(
            "GenerateBatchedCudaContacts requires uploaded batched state and shape tables");
    }

    phi::Buffer manifolds(
        broadphase.TotalPairSlotCount() * sizeof(CudaBatchedContactManifold),
        phi::MemoryKind::Device);
    phi::Buffer report(sizeof(CudaBatchedContactReport), phi::MemoryKind::Device);

    InitializeBatchedContactReportKernel<<<1, 1>>>(
        batch.InstanceCount(),
        broadphase.DevicePairCount(),
        static_cast<CudaBatchedContactReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "InitializeBatchedContactReportKernel launch");

    if (broadphase.TotalPairSlotCount() == 0u) {
        return CudaBatchedContactResult(0u, std::move(manifolds), std::move(report));
    }

    constexpr uint32_t kBlockSize = 128u;
    const uint32_t blocks =
        (broadphase.TotalPairSlotCount() + kBlockSize - 1u) / kBlockSize;
    GenerateBatchedContactsKernel<<<blocks, kBlockSize>>>(
        broadphase.TotalPairSlotCount(),
        batch.ShapeCountPerInstance(),
        batch.BodyCountPerInstance(),
        broadphase.DevicePairs(),
        broadphase.DevicePairInstanceIndices(),
        broadphase.DevicePairActiveFlags(),
        broadphase.DeviceAabbs(),
        batch.DevicePoses(),
        batch.DeviceInvMasses(),
        batch.DeviceShapeBodyIds(),
        batch.DeviceShapeTypes(),
        batch.DeviceShapeLocalTransforms(),
        batch.DeviceShapeRadii(),
        static_cast<CudaBatchedContactManifold*>(manifolds.Data()));
    CheckCuda(cudaGetLastError(), "GenerateBatchedContactsKernel launch");

    CountBatchedContactsKernel<<<blocks, kBlockSize>>>(
        broadphase.TotalPairSlotCount(),
        static_cast<CudaBatchedContactManifold*>(manifolds.Data()),
        static_cast<CudaBatchedContactReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "CountBatchedContactsKernel launch");

    return CudaBatchedContactResult(broadphase.TotalPairSlotCount(),
                                   std::move(manifolds),
                                   std::move(report));
}

namespace {

CudaBatchedConstraintSolverResult SolveBatchedCudaConstraintsImpl(
    BatchedDeviceWorld& batch,
    const CudaBatchedContactResult* contacts,
    bool include_joints,
    bool include_drives,
    const CudaBatchedConstraintSolverConfig& config) {
    if (!batch.HasUploadedState()) {
        throw std::runtime_error(
            "SolveBatchedCudaConstraints requires uploaded batched state");
    }
    if (include_joints && !batch.HasUploadedJointTables()) {
        throw std::runtime_error(
            "SolveBatchedCudaConstraints requires uploaded batched joint tables");
    }
    if (include_drives
        && (!batch.HasUploadedJointTables() || !batch.HasUploadedActuatorTables())) {
        throw std::runtime_error(
            "SolveBatchedCudaConstraints requires uploaded batched actuator tables");
    }

    const uint32_t contact_capacity = contacts != nullptr ? contacts->TotalPairSlotCount() : 0u;
    const uint32_t joint_capacity = include_joints ? batch.TotalJointCount() : 0u;
    const uint32_t actuator_capacity = include_drives ? batch.TotalActuatorCount() : 0u;
    const uint32_t block_capacity = contact_capacity + joint_capacity + actuator_capacity;
    phi::Buffer blocks(block_capacity * sizeof(constraint::ConstraintBlock),
                       phi::MemoryKind::Device);
    phi::Buffer block_count(sizeof(uint32_t), phi::MemoryKind::Device);
    phi::Buffer report(sizeof(CudaBatchedConstraintSolverReport), phi::MemoryKind::Device);

    ClearBatchedSolverReportKernel<<<1, 1>>>(
        static_cast<uint32_t*>(block_count.Data()),
        static_cast<CudaBatchedConstraintSolverReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "ClearBatchedSolverReportKernel launch");

    constexpr uint32_t kBlockSize = 128u;
    if (contacts != nullptr && contacts->TotalPairSlotCount() > 0u) {
        const uint32_t contact_blocks =
            (contacts->TotalPairSlotCount() + kBlockSize - 1u) / kBlockSize;
        AssembleBatchedContactBlocksKernel<<<contact_blocks, kBlockSize>>>(
            contacts->TotalPairSlotCount(),
            batch.BodyCountPerInstance(),
            contacts->DeviceManifolds(),
            static_cast<constraint::ConstraintBlock*>(blocks.Data()),
            static_cast<uint32_t*>(block_count.Data()));
        CheckCuda(cudaGetLastError(), "AssembleBatchedContactBlocksKernel launch");
    }

    if (include_joints && batch.TotalJointCount() > 0u) {
        const uint32_t joint_blocks =
            (batch.TotalJointCount() + kBlockSize - 1u) / kBlockSize;
        AssembleBatchedJointBlocksKernel<<<joint_blocks, kBlockSize>>>(
            batch.TotalJointCount(),
            batch.JointCountPerInstance(),
            batch.BodyCountPerInstance(),
            batch.DeviceJointTypes(),
            batch.DeviceJointParentBodies(),
            batch.DeviceJointChildBodies(),
            batch.DeviceJointAxes(),
            batch.DeviceJointParentFrames(),
            batch.DeviceJointChildFrames(),
            static_cast<constraint::ConstraintBlock*>(blocks.Data()),
            static_cast<uint32_t*>(block_count.Data()));
        CheckCuda(cudaGetLastError(), "AssembleBatchedJointBlocksKernel launch");
    }

    if (include_drives && batch.TotalActuatorCount() > 0u) {
        const uint32_t drive_blocks =
            (batch.TotalActuatorCount() + kBlockSize - 1u) / kBlockSize;
        AssembleBatchedDriveBlocksKernel<<<drive_blocks, kBlockSize>>>(
            batch.TotalActuatorCount(),
            batch.ActuatorCountPerInstance(),
            batch.JointCountPerInstance(),
            batch.BodyCountPerInstance(),
            batch.DeviceActuatorJointIds(),
            batch.DeviceActuatorTypes(),
            batch.DeviceActuatorGains(),
            batch.DeviceActuatorForceLimits(),
            batch.DeviceJointParentBodies(),
            batch.DeviceJointChildBodies(),
            batch.DeviceJointAxes(),
            static_cast<constraint::ConstraintBlock*>(blocks.Data()),
            static_cast<uint32_t*>(block_count.Data()));
        CheckCuda(cudaGetLastError(), "AssembleBatchedDriveBlocksKernel launch");
    }

    FinalizeBatchedSolverReportKernel<<<1, 1>>>(
        static_cast<constraint::ConstraintBlock*>(blocks.Data()),
        static_cast<const uint32_t*>(block_count.Data()),
        static_cast<CudaBatchedConstraintSolverReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "FinalizeBatchedSolverReportKernel launch");

    if (block_capacity > 0u) {
        const uint32_t blocks_grid = (block_capacity + kBlockSize - 1u) / kBlockSize;
        PrecomputeBatchedBlocksKernel<<<blocks_grid, kBlockSize>>>(
            batch.TotalBodyCount(),
            batch.BodyCountPerInstance(),
            static_cast<constraint::ConstraintBlock*>(blocks.Data()),
            static_cast<const uint32_t*>(block_count.Data()),
            batch.DeviceInvMasses(),
            batch.DeviceInvInertias(),
            batch.DeviceLinearVelocities(),
            batch.DeviceAngularVelocities());
        CheckCuda(cudaGetLastError(), "PrecomputeBatchedBlocksKernel launch");

        for (uint32_t iter = 0; iter < config.velocity_iterations; ++iter) {
            SolveBatchedVelocityIterationKernel<<<1, 1>>>(
                batch.TotalBodyCount(),
                batch.BodyCountPerInstance(),
                static_cast<constraint::ConstraintBlock*>(blocks.Data()),
                static_cast<const uint32_t*>(block_count.Data()),
                batch.DeviceInvMasses(),
                batch.DeviceInvInertias(),
                batch.DeviceLinearVelocities(),
                batch.DeviceAngularVelocities());
            CheckCuda(cudaGetLastError(), "SolveBatchedVelocityIterationKernel launch");
        }

        for (uint32_t iter = 0; iter < config.position_iterations; ++iter) {
            SolveBatchedContactPositionIterationKernel<<<1, 1>>>(
                batch.TotalBodyCount(),
                batch.BodyCountPerInstance(),
                static_cast<constraint::ConstraintBlock*>(blocks.Data()),
                static_cast<const uint32_t*>(block_count.Data()),
                batch.DeviceInvMasses(),
                batch.DeviceInvInertias(),
                batch.DevicePoses(),
                config.slop,
                config.baumgarte,
                static_cast<CudaBatchedConstraintSolverReport*>(report.Data()));
            CheckCuda(cudaGetLastError(), "SolveBatchedContactPositionIterationKernel launch");

            SolveBatchedJointPositionIterationKernel<<<1, 1>>>(
                batch.TotalBodyCount(),
                batch.BodyCountPerInstance(),
                static_cast<constraint::ConstraintBlock*>(blocks.Data()),
                static_cast<const uint32_t*>(block_count.Data()),
                batch.DeviceInvMasses(),
                batch.DeviceInvInertias(),
                batch.DevicePoses(),
                config.slop,
                config.baumgarte,
                static_cast<CudaBatchedConstraintSolverReport*>(report.Data()));
            CheckCuda(cudaGetLastError(), "SolveBatchedJointPositionIterationKernel launch");
        }
    }

    SetBatchedSolverIterationReportKernel<<<1, 1>>>(
        config.velocity_iterations,
        config.position_iterations,
        static_cast<CudaBatchedConstraintSolverReport*>(report.Data()));
    CheckCuda(cudaGetLastError(), "SetBatchedSolverIterationReportKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "SolveBatchedCudaConstraints synchronize");

    return CudaBatchedConstraintSolverResult(block_capacity,
                                            std::move(blocks),
                                            std::move(block_count),
                                            std::move(report));
}

} // namespace

CudaBatchedConstraintSolverResult SolveBatchedCudaContactConstraints(
    BatchedDeviceWorld& batch,
    const CudaBatchedContactResult& contacts,
    const CudaBatchedConstraintSolverConfig& config) {
    return SolveBatchedCudaConstraintsImpl(batch, &contacts, false, false, config);
}

CudaBatchedConstraintSolverResult SolveBatchedCudaConstraints(
    BatchedDeviceWorld& batch,
    const CudaBatchedContactResult* contacts,
    const CudaBatchedConstraintSolverConfig& config) {
    return SolveBatchedCudaConstraintsImpl(batch,
                                           contacts,
                                           true,
                                           config.velocity_iterations > 0u,
                                           config);
}

CudaBatchedWorldStepReport StepBatchedCudaWorld(
    BatchedDeviceWorld& batch,
    const CudaBatchedWorldStepOptions& options) {
    CudaBatchedWorldStepReport report;
    report.instance_count = batch.InstanceCount();
    report.body_count_per_instance = batch.BodyCountPerInstance();
    report.shape_count_per_instance = batch.ShapeCountPerInstance();
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

        if (options.enable_contacts || options.enable_joints || options.enable_drives) {
            CudaBatchedConstraintSolverConfig config;
            config.velocity_iterations = options.solver_velocity_iterations;
            config.position_iterations = options.solver_position_iterations;
            config.slop = options.solver_slop;
            config.baumgarte = options.solver_baumgarte;

            if (options.enable_contacts) {
                auto broadphase = BuildBatchedCudaBroadphase(batch);
                auto contacts = GenerateBatchedCudaContacts(batch, broadphase);
                auto solver = SolveBatchedCudaConstraintsImpl(batch,
                                                              &contacts,
                                                              options.enable_joints,
                                                              options.enable_drives,
                                                              config);

                const auto contact_report = contacts.DownloadReport();
                report.broadphase_pair_count += contact_report.pair_count;
                report.contact_manifold_count += contact_report.contact_manifold_count;
                report.contact_point_count += contact_report.contact_point_count;

                const auto solver_report = solver.DownloadReport();
                report.contact_constraint_count += solver_report.contact_constraint_count;
                report.joint_constraint_count += solver_report.joint_constraint_count;
                report.drive_constraint_count += solver_report.drive_constraint_count;
                report.constraint_block_count += solver_report.constraint_block_count;
                report.constraint_row_count += solver_report.constraint_row_count;
                report.solver_iterations_used +=
                    solver_report.velocity_iterations + solver_report.position_iterations;
                report.max_constraint_error =
                    std::max(report.max_constraint_error, solver_report.max_position_error);
                continue;
            }

            auto solver = SolveBatchedCudaConstraintsImpl(batch,
                                                          nullptr,
                                                          options.enable_joints,
                                                          options.enable_drives,
                                                          config);
            const auto solver_report = solver.DownloadReport();
            report.contact_constraint_count += solver_report.contact_constraint_count;
            report.joint_constraint_count += solver_report.joint_constraint_count;
            report.drive_constraint_count += solver_report.drive_constraint_count;
            report.constraint_block_count += solver_report.constraint_block_count;
            report.constraint_row_count += solver_report.constraint_row_count;
            report.solver_iterations_used +=
                solver_report.velocity_iterations + solver_report.position_iterations;
            report.max_constraint_error =
                std::max(report.max_constraint_error, solver_report.max_position_error);
        }
    }

    CheckCuda(cudaDeviceSynchronize(), "IntegrateBatchedRigidBodiesKernel synchronize");
    report.simulated_step_count = options.step_count;
    return report;
}

} // namespace nuka::runtime::gpu
