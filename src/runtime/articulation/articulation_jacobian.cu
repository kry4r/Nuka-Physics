// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- Featherstone generalized Jacobian builder
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_jacobian.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::runtime::articulation {

namespace {

__device__ math::Vec3 Sub(math::Vec3 a, math::Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ math::Vec3 Cross(math::Vec3 a, math::Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__device__ float Dot(math::Vec3 a, math::Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ math::Vec3 ContactNormalOrDefault(const math::Vec3* points, uint32_t index) {
    const math::Vec3 point = points[index];
    const float length_sq = point.x * point.x + point.y * point.y + point.z * point.z;
    if (length_sq > 1.0e-10f) {
        const float inv_length = rsqrtf(length_sq);
        return {point.x * inv_length, point.y * inv_length, point.z * inv_length};
    }
    return {0.0f, 0.0f, 1.0f};
}

// Normalizes a contact normal, falling back to +Z for a degenerate input.
__device__ math::Vec3 NormalizeOrUp(math::Vec3 normal) {
    const float length_sq = Dot(normal, normal);
    if (length_sq > 1.0e-10f) {
        const float inv_length = rsqrtf(length_sq);
        return {normal.x * inv_length, normal.y * inv_length, normal.z * inv_length};
    }
    return {0.0f, 0.0f, 1.0f};
}

// Rotates `v` by unit-ish quaternion `q` (w-first). Device-side reimplementation
// of math::Quat::Rotate, which is host-only. Uses the q*v*q^-1 short form and
// normalizes defensively so unnormalized poses do not scale the axis.
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
    const math::Vec3 t = Cross(qv, v);
    const math::Vec3 t2{2.0f * t.x, 2.0f * t.y, 2.0f * t.z};
    const math::Vec3 wt{q.w * t2.x, q.w * t2.y, q.w * t2.z};
    const math::Vec3 qvt = Cross(qv, t2);
    return {v.x + wt.x + qvt.x, v.y + wt.y + qvt.y, v.z + wt.z + qvt.z};
}

// Generalized-coordinate count contributed by a joint of the given type. This
// is the single source of truth for both the per-contact dof_stride and the
// dof_index() prefix sum, so the chain Jacobian transparently picks up a future
// 6-DOF floating-base root without touching the walk logic.
__device__ uint32_t JointDofCount(ArticulationJointType type) {
    switch (type) {
        case ArticulationJointType::Revolute:
        case ArticulationJointType::Prismatic:
            return 1u;
        case ArticulationJointType::Fixed:
            return 0u;
        case ArticulationJointType::FloatingBase:
            return 6u;  // T8a: free-floating root contributes 6 DOF.
    }
    return 0u;
}

__global__ void ComputeLinkPointJacobianKernel(
    ArticulationDeviceState state,
    const uint32_t* contact_link_indices,
    const math::Vec3* contact_point_world,
    uint32_t contact_count,
    float* out_jacobian_scalars) {
    const uint32_t contact = blockIdx.x * blockDim.x + threadIdx.x;
    if (contact >= contact_count) {
        return;
    }

    const uint32_t link = contact_link_indices[contact];
    if (link >= state.total_link_count) {
        out_jacobian_scalars[contact] = 0.0f;
        return;
    }

    const math::Vec3 normal = ContactNormalOrDefault(contact_point_world, contact);
    const math::Vec3 lever = Sub(contact_point_world[contact], state.link_pose[link].position);
    const math::Vec3 axis = state.joint_axis[link];
    if (state.joint_type[link] == ArticulationJointType::Prismatic) {
        out_jacobian_scalars[contact] = Dot(axis, normal);
    } else if (state.joint_type[link] == ArticulationJointType::Revolute) {
        out_jacobian_scalars[contact] = Dot(Cross(axis, lever), normal);
    } else {
        out_jacobian_scalars[contact] = 0.0f;
    }
}

// One thread per contact. Walks the ancestor-joint chain (contact link -> root)
// in fixed order, writing each ancestor DOF's Jacobian entry into the contact's
// own dof_stride-wide output slice. No atomics, no cross-contact aliasing.
__global__ void ComputeContactChainJacobianKernel(
    ArticulationDeviceState state,
    const uint32_t* contact_link_indices,
    const math::Vec3* contact_point_world,
    const math::Vec3* contact_normal_world,
    uint32_t contact_count,
    uint32_t dof_stride,
    float* out_chain_jacobian) {
    const uint32_t contact = blockIdx.x * blockDim.x + threadIdx.x;
    if (contact >= contact_count) {
        return;
    }

    const uint32_t contact_link = contact_link_indices[contact];
    if (contact_link >= state.total_link_count) {
        return;
    }

    const uint32_t articulation = state.link_to_articulation[contact_link];
    const uint32_t offset = state.articulation_link_offset[articulation];

    const math::Vec3 point = contact_point_world[contact];
    const math::Vec3 normal = NormalizeOrUp(contact_normal_world[contact]);
    float* const out_row = out_chain_jacobian + static_cast<size_t>(contact) * dof_stride;

    // Walk from the contact's link up to the root. parent_link is articulation-
    // local; the root's parent is the ~0u sentinel (scene::kInvalidBody).
    uint32_t link = contact_link;
    constexpr uint32_t kInvalidLink = ~0u;
    while (link != kInvalidLink) {
        const ArticulationJointType type = state.joint_type[link];
        const uint32_t dof_count = JointDofCount(type);
        if (dof_count != 0u) {
            // dof_index(link): base-inclusive prefix sum of per-joint DOF counts
            // across the articulation's links in [offset, link).
            uint32_t dof_index = 0u;
            for (uint32_t k = offset; k < link; ++k) {
                dof_index += JointDofCount(state.joint_type[k]);
            }

            const math::Vec3 axis_world =
                RotateByQuat(state.link_pose[link].rotation, state.joint_axis[link]);
            float entry = 0.0f;
            if (type == ArticulationJointType::Prismatic) {
                entry = Dot(axis_world, normal);
            } else {  // Revolute
                const math::Vec3 lever = Sub(point, state.link_pose[link].position);
                entry = Dot(Cross(axis_world, lever), normal);
            }
            if (dof_index < dof_stride) {
                out_row[dof_index] = entry;
            }
        }

        const uint32_t parent_local = state.parent_link[link];
        link = (parent_local == kInvalidLink) ? kInvalidLink : (offset + parent_local);
    }
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

} // namespace

void ComputeLinkPointJacobians(const phi::DeviceContext& context,
                               ArticulationDeviceState state,
                               const uint32_t* contact_link_indices,
                               const math::Vec3* contact_point_world,
                               uint32_t contact_count,
                               float* out_jacobian_scalars) {
    if (contact_count == 0u) {
        return;
    }
    if (contact_link_indices == nullptr ||
        contact_point_world == nullptr ||
        out_jacobian_scalars == nullptr) {
        throw std::runtime_error("ComputeLinkPointJacobians requires device input and output buffers");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t block_count = (contact_count + kBlockSize - 1u) / kBlockSize;
    ComputeLinkPointJacobianKernel<<<block_count, kBlockSize, 0u, stream>>>(
        state,
        contact_link_indices,
        contact_point_world,
        contact_count,
        out_jacobian_scalars);
    CheckCuda(cudaGetLastError(), "ComputeLinkPointJacobianKernel launch");
}

void ComputeContactChainJacobians(const phi::DeviceContext& context,
                                  ArticulationDeviceState state,
                                  const uint32_t* contact_link_indices,
                                  const math::Vec3* contact_point_world,
                                  const math::Vec3* contact_normal_world,
                                  uint32_t contact_count,
                                  uint32_t dof_stride,
                                  float* out_chain_jacobian) {
    if (contact_count == 0u || dof_stride == 0u) {
        return;
    }
    if (contact_link_indices == nullptr ||
        contact_point_world == nullptr ||
        contact_normal_world == nullptr ||
        out_chain_jacobian == nullptr) {
        throw std::runtime_error(
            "ComputeContactChainJacobians requires device input and output buffers");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    // Each contact owns its dof_stride-wide slice; the kernel only writes the
    // ancestor-chain columns, so zero the full output up front to give the
    // "untouched columns are zero" guarantee the callers rely on.
    const size_t out_bytes =
        static_cast<size_t>(contact_count) * dof_stride * sizeof(float);
    CheckCuda(cudaMemsetAsync(out_chain_jacobian, 0, out_bytes, stream),
              "ComputeContactChainJacobians memset");
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t block_count = (contact_count + kBlockSize - 1u) / kBlockSize;
    ComputeContactChainJacobianKernel<<<block_count, kBlockSize, 0u, stream>>>(
        state,
        contact_link_indices,
        contact_point_world,
        contact_normal_world,
        contact_count,
        dof_stride,
        out_chain_jacobian);
    CheckCuda(cudaGetLastError(), "ComputeContactChainJacobianKernel launch");
}

} // namespace nuka::runtime::articulation
