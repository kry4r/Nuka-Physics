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

} // namespace nuka::runtime::articulation
