// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- Featherstone generalized Jacobian builder
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_jacobian.hpp"

#include "math/cuda_vec_ops.cuh"
#include "runtime/articulation/articulation_device_helpers.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::runtime::articulation {

namespace {

// Sub / Cross / Dot / NormalizeOrUp now come from the shared device math library
// (math/cuda_vec_ops.cuh); the bodies are bit-identical (NormalizeOrUp uses the
// same rsqrtf, 1e-10, fallback +Z recipe). The file-local ContactNormalOrDefault
// and RotateByQuat (differently named / defensive-normalize variants, not in the
// shared inventory) stay local.
namespace mg = ::nuka::math::gpu;
using mg::Cross;
using mg::Dot;
using mg::NormalizeOrUp;
using mg::Sub;

__device__ math::Vec3 ContactNormalOrDefault(const math::Vec3* points, uint32_t index) {
    const math::Vec3 point = points[index];
    const float length_sq = point.x * point.x + point.y * point.y + point.z * point.z;
    if (length_sq > 1.0e-10f) {
        const float inv_length = rsqrtf(length_sq);
        return {point.x * inv_length, point.y * inv_length, point.z * inv_length};
    }
    return {0.0f, 0.0f, 1.0f};
}

// Rotates `v` by unit-ish quaternion `q` (w-first). Device-side reimplementation
// of math::Quat::Rotate, which is host-only. Uses the q*v*q^-1 short form and
// normalizes defensively so unnormalized poses do not scale the axis. Now routed
// through the shared mg::RotateByQuatNormalized (byte-identical body) via a thin
// forwarder that keeps the call site verbatim.
__forceinline__ __device__ math::Vec3 RotateByQuat(math::Quat q, math::Vec3 v) {
    return mg::RotateByQuatNormalized(q, v);
}

// Generalized-coordinate count contributed by a joint of the given type. This
// is the single source of truth for both the per-contact dof_stride and the
// dof_index() prefix sum, so the chain Jacobian transparently picks up a future
// 6-DOF floating-base root without touching the walk logic. Now routed through
// the shared JointDofCountDevice (articulation_device_helpers.cuh; byte-identical
// integer body) via a thin forwarder that keeps the local JointDofCount(...)
// call sites verbatim.
__forceinline__ __device__ uint32_t JointDofCount(ArticulationJointType type) {
    return JointDofCountDevice(type);
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

            if (type == ArticulationJointType::FloatingBase) {
                // T8b: floating-base root contributes 6 columns (dof_index 0..5),
                // omega-first [omega(0:2), lin(3:5)] in the SAME body frame and at
                // the SAME origin as link_velocity[root] (verified against the T8a
                // ABA / pose integrator). Read the LIVE base pose directly so R and
                // the origin match exactly the frame link_velocity[root] references.
                //   base spatial velocity v = [omega_body; vlin_body]  (body frame)
                //   contact-point world vel = (R*omega)x(point-origin) + R*vlin
                //   J_d[k] = d . d(point vel)/d v_k
                // angular column k: axis_world = R*e_k (k-th column of R);
                //   entry = dot( cross(axis_world, point-origin), d )  (== Revolute)
                // linear column k:  entry = dot( R*e_k, d )            (== Prismatic)
                const math::Quat base_rot = state.base_pose[articulation].rotation;
                const math::Vec3 base_origin = state.base_pose[articulation].position;
                const math::Vec3 lever = Sub(point, base_origin);
                const math::Vec3 ex = RotateByQuat(base_rot, {1.0f, 0.0f, 0.0f});
                const math::Vec3 ey = RotateByQuat(base_rot, {0.0f, 1.0f, 0.0f});
                const math::Vec3 ez = RotateByQuat(base_rot, {0.0f, 0.0f, 1.0f});
                const float ang[3] = {Dot(Cross(ex, lever), normal),
                                      Dot(Cross(ey, lever), normal),
                                      Dot(Cross(ez, lever), normal)};
                const float lin[3] = {Dot(ex, normal), Dot(ey, normal),
                                      Dot(ez, normal)};
                for (uint32_t b = 0u; b < 3u; ++b) {
                    if (dof_index + b < dof_stride) {
                        out_row[dof_index + b] = ang[b];
                    }
                    if (dof_index + 3u + b < dof_stride) {
                        out_row[dof_index + 3u + b] = lin[b];
                    }
                }
            } else {
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
