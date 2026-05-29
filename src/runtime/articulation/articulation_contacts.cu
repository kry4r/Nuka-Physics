// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- world-pose FK + foot-vs-ground contacts
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_contacts.hpp"

#include "math/quat.hpp"
#include "math/transform.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::runtime::articulation {

namespace {

constexpr uint32_t kInvalidLink = ~0u;

// -- device vec/quat helpers ------------------------------------------------
// math::Quat::operator*, FromAxisAngle and Rotate are host-only (constexpr /
// inline, not __device__), so the device path hand-rolls them here, mirroring
// the w-first Hamilton-product and half-angle conventions in math/quat.hpp and
// the RotateByQuat short form already used in articulation_jacobian.cu.

__device__ math::Vec3 AddVec(math::Vec3 a, math::Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ math::Vec3 ScaleVec(math::Vec3 v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

__device__ float Dot3(math::Vec3 a, math::Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ math::Vec3 Cross3(math::Vec3 a, math::Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// math::Quat's constructors are host-only constexpr, so the device path builds
// quaternions by mutating a default-constructed value (field assignment is
// device-legal) rather than via brace/Identity()/operator* construction.
__device__ math::Quat MakeQuat(float w, float x, float y, float z) {
    math::Quat q;
    q.w = w;
    q.x = x;
    q.y = y;
    q.z = z;
    return q;
}

__device__ math::Quat QuatIdentity() {
    return MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
}

// Hamilton product this = a * b (w-first), mirroring math::Quat::operator*.
__device__ math::Quat QuatMul(math::Quat a, math::Quat b) {
    return MakeQuat(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

__device__ math::Quat QuatNormalize(math::Quat q) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq <= 1.0e-12f) {
        return QuatIdentity();
    }
    const float inv_norm = rsqrtf(norm_sq);
    return MakeQuat(q.w * inv_norm, q.x * inv_norm, q.y * inv_norm, q.z * inv_norm);
}

// Unit quaternion for a rotation of `angle` about `axis`, mirroring
// math::Quat::FromAxisAngle (half-angle, normalized axis).
__device__ math::Quat QuatFromAxisAngle(math::Vec3 axis, float angle) {
    const float length_sq = Dot3(axis, axis);
    if (length_sq <= 1.0e-12f) {
        return QuatIdentity();
    }
    const math::Vec3 a = ScaleVec(axis, rsqrtf(length_sq));
    const float half = angle * 0.5f;
    const float s = sinf(half);
    const float c = cosf(half);
    return MakeQuat(c, a.x * s, a.y * s, a.z * s);
}

// Rotates `v` by (near-)unit quaternion `q` (q * v * q^-1 short form). Mirrors
// the device RotateByQuat in articulation_jacobian.cu / math::Quat::Rotate.
__device__ math::Vec3 RotateByQuat(math::Quat q, math::Vec3 v) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq > 1.0e-12f) {
        const float inv_norm = rsqrtf(norm_sq);
        q.w *= inv_norm;
        q.x *= inv_norm;
        q.y *= inv_norm;
        q.z *= inv_norm;
    }
    const math::Vec3 qv{q.x, q.y, q.z};
    const math::Vec3 t = ScaleVec(Cross3(qv, v), 2.0f);
    const math::Vec3 wt = ScaleVec(t, q.w);
    const math::Vec3 qvt = Cross3(qv, t);
    return AddVec(AddVec(v, wt), qvt);
}

// Composes lhs o rhs (apply rhs first), mirroring math::Transform::operator*.
__device__ math::Transform ComposeTransform(const math::Transform& lhs,
                                            const math::Transform& rhs) {
    math::Transform out;
    out.position = AddVec(lhs.position, RotateByQuat(lhs.rotation, rhs.position));
    out.rotation = QuatNormalize(QuatMul(lhs.rotation, rhs.rotation));
    return out;
}

// SE(3) parent->child relative transform for one link. Mirrors JointTransform
// in featherstone_aba.cu (lines 349-365), but in position+quaternion form
// rather than the spatial 6x6 motion form (which transposes the rotation):
//   translation = local_pose.position + parent_offset (+ axis * q for prismatic)
//   rotation    = local_pose.rotation * jointRotation(axis, q)
// Revolute rotates about joint_axis by q; prismatic/fixed have identity rotation.
__device__ math::Transform RelativeTransform(const ArticulationDeviceState& state,
                                             uint32_t link) {
    const ArticulationJointType type = state.joint_type[link];
    const math::Vec3 axis = state.joint_axis[link];
    const math::Transform local_pose = state.link_local_pose[link];
    const math::Vec3 parent_offset = state.parent_offset[link];

    math::Transform relative;
    relative.position = AddVec(local_pose.position, parent_offset);
    relative.rotation = local_pose.rotation;
    if (type == ArticulationJointType::Revolute) {
        relative.rotation =
            QuatNormalize(QuatMul(local_pose.rotation,
                                  QuatFromAxisAngle(axis, state.q[link])));
    } else if (type == ArticulationJointType::Prismatic) {
        relative.position =
            AddVec(relative.position, ScaleVec(axis, state.q[link]));
    }
    return relative;
}

// (A) One block per articulation, single lane. Forward pass over the links in
// local order; parent-local-index < child is guaranteed by the cooker, so each
// parent's world pose is already final when its child is processed.
__global__ void UpdateWorldLinkPosesKernel(ArticulationDeviceState state,
                                           math::Transform* out_world_pose) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        const math::Transform relative = RelativeTransform(state, link);
        const uint32_t parent_local = state.parent_link[link];
        if (parent_local == kInvalidLink) {
            // Root: world = identity o relative = relative.
            out_world_pose[link] = relative;
        } else {
            out_world_pose[link] =
                ComposeTransform(out_world_pose[offset + parent_local], relative);
        }
    }
}

// (B) One thread per environment. Walks the env's feet in fixed order, compacts
// penetrating feet into the env's own [env*stride, env*stride+count) slots, and
// writes the env's contact count. No atomics; each env owns its slot block.
__global__ void DetectFootGroundContactsKernel(const math::Transform* world_pose,
                                               const FootShape* feet,
                                               uint32_t foot_count,
                                               uint32_t env_count,
                                               uint32_t base_link_count,
                                               float ground_height,
                                               uint32_t* out_contact_link,
                                               math::Vec3* out_contact_point,
                                               math::Vec3* out_contact_normal,
                                               float* out_contact_depth,
                                               uint32_t* out_contact_count) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= env_count) {
        return;
    }

    const uint32_t base_slot = env * kMaxFootContactsPerEnv;
    uint32_t count = 0u;
    for (uint32_t foot = 0u; foot < foot_count; ++foot) {
        const FootShape shape = feet[foot];
        const uint32_t link = env * base_link_count + shape.calf_local_link;
        const math::Transform calf = world_pose[link];
        // Foot sphere center in world = calf_world o local_offset.
        const math::Vec3 center =
            AddVec(calf.position, RotateByQuat(calf.rotation, shape.local_offset));
        const float depth = (ground_height + shape.radius) - center.z;
        if (depth > 0.0f) {
            const uint32_t slot = base_slot + count;
            out_contact_link[slot] = link;
            // Contact point: foot center projected onto the ground surface.
            out_contact_point[slot] = {center.x, center.y, ground_height};
            out_contact_normal[slot] = {0.0f, 0.0f, 1.0f};
            out_contact_depth[slot] = depth;
            ++count;
        }
    }
    // Zero-fill the env's unused trailing slots so stale data never leaks.
    for (uint32_t slot = count; slot < kMaxFootContactsPerEnv; ++slot) {
        const uint32_t index = base_slot + slot;
        out_contact_link[index] = kInvalidLink;
        out_contact_point[index] = {0.0f, 0.0f, 0.0f};
        out_contact_normal[index] = {0.0f, 0.0f, 0.0f};
        out_contact_depth[index] = 0.0f;
    }
    out_contact_count[env] = count;
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

} // namespace

void UpdateWorldLinkPoses(const phi::DeviceContext& context,
                          ArticulationDeviceState state,
                          math::Transform* out_world_pose) {
    if (state.articulation_count == 0u || state.total_link_count == 0u) {
        return;
    }
    if (out_world_pose == nullptr) {
        throw std::runtime_error("UpdateWorldLinkPoses requires an output buffer");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    // One block per articulation; the forward pass runs on a single lane (the
    // parent->child dependency is sequential within an articulation).
    UpdateWorldLinkPosesKernel<<<state.articulation_count, 32u, 0u, stream>>>(
        state, out_world_pose);
    CheckCuda(cudaGetLastError(), "UpdateWorldLinkPosesKernel launch");
}

void DetectFootGroundContacts(const phi::DeviceContext& context,
                              const math::Transform* world_pose,
                              const FootShape* feet,
                              uint32_t foot_count,
                              uint32_t env_count,
                              uint32_t base_link_count,
                              float ground_height,
                              uint32_t* out_contact_link,
                              math::Vec3* out_contact_point,
                              math::Vec3* out_contact_normal,
                              float* out_contact_depth,
                              uint32_t* out_contact_count) {
    if (env_count == 0u) {
        return;
    }
    if (foot_count > kMaxFootContactsPerEnv) {
        throw std::runtime_error(
            "DetectFootGroundContacts: foot_count exceeds kMaxFootContactsPerEnv");
    }
    if (world_pose == nullptr || feet == nullptr || out_contact_link == nullptr ||
        out_contact_point == nullptr || out_contact_normal == nullptr ||
        out_contact_depth == nullptr || out_contact_count == nullptr) {
        throw std::runtime_error(
            "DetectFootGroundContacts requires device input and output buffers");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t block_count = (env_count + kBlockSize - 1u) / kBlockSize;
    DetectFootGroundContactsKernel<<<block_count, kBlockSize, 0u, stream>>>(
        world_pose,
        feet,
        foot_count,
        env_count,
        base_link_count,
        ground_height,
        out_contact_link,
        out_contact_point,
        out_contact_normal,
        out_contact_depth,
        out_contact_count);
    CheckCuda(cudaGetLastError(), "DetectFootGroundContactsKernel launch");
}

} // namespace nuka::runtime::articulation
