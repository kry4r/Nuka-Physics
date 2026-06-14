// ---------------------------------------------------------------------------
// nuka::core::diagnostics - CUDA invariant reductions
// ---------------------------------------------------------------------------

#include "core/diagnostics/invariants_gpu.cuh"

#include "math/cuda_vec_ops.cuh"
#include "phi/backend.hpp"  // InitBestDevice / DeviceBufferType
#include "phi/buffer.hpp"   // Buffer* / BufferAlloc / BufferDownload / BufferFree

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::core::diagnostics::gpu {

namespace {

struct DeviceReductionSlot {
    float energy = 0.0f;
    math::Vec3 linear_momentum = {};
    math::Vec3 angular_momentum = {};
    uint32_t nan_inf_count = 0;
    float max_speed = 0.0f;
    float max_position_abs = 0.0f;
};

__device__ bool IsFinite(float value) {
    return isfinite(value) != 0;
}

__device__ bool IsFinite(math::Vec3 value) {
    return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
}

__device__ bool IsFinite(math::Quat value) {
    return IsFinite(value.w) && IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
}

__device__ float AbsMax(math::Vec3 value) {
    return fmaxf(fabsf(value.x), fmaxf(fabsf(value.y), fabsf(value.z)));
}

// Vec3 Add/Dot/Cross now come from the shared device math library
// (math/cuda_vec_ops.cuh). Bodies are bit-identical to the former local copies.
namespace mg = ::nuka::math::gpu;
using mg::Add;
using mg::Cross;
using mg::Dot;

__global__ void ComputeRigidBodyInvariantKernel(
    uint32_t body_count,
    const math::Transform* poses,
    const math::Vec3* linear_velocities,
    const math::Vec3* angular_velocities,
    const float* inverse_masses,
    const math::Vec3* inverse_inertias,
    math::Vec3 gravity,
    DeviceReductionSlot* out_slot) {
    extern __shared__ DeviceReductionSlot shared[];
    const uint32_t tid = threadIdx.x;
    DeviceReductionSlot local;

    for (uint32_t index = tid; index < body_count; index += blockDim.x) {
        const math::Transform pose = poses[index];
        const math::Vec3 linear_velocity = linear_velocities[index];
        const math::Vec3 angular_velocity = angular_velocities[index];
        const float inv_mass = inverse_masses[index];
        const math::Vec3 inv_inertia = inverse_inertias[index];

        if (!IsFinite(pose.position) || !IsFinite(pose.rotation) ||
            !IsFinite(linear_velocity) || !IsFinite(angular_velocity) ||
            !IsFinite(inv_mass) || !IsFinite(inv_inertia)) {
            ++local.nan_inf_count;
            continue;
        }

        local.max_speed = fmaxf(local.max_speed,
                                sqrtf(linear_velocity.x * linear_velocity.x +
                                      linear_velocity.y * linear_velocity.y +
                                      linear_velocity.z * linear_velocity.z));
        local.max_position_abs = fmaxf(local.max_position_abs, AbsMax(pose.position));

        if (inv_mass <= 0.0f) {
            continue;
        }

        const float mass = 1.0f / inv_mass;
        const math::Vec3 momentum = {
            linear_velocity.x * mass,
            linear_velocity.y * mass,
            linear_velocity.z * mass
        };
        local.linear_momentum = Add(local.linear_momentum, momentum);
        local.angular_momentum = Add(local.angular_momentum, Cross(pose.position, momentum));

        const float speed_sq = linear_velocity.x * linear_velocity.x +
                               linear_velocity.y * linear_velocity.y +
                               linear_velocity.z * linear_velocity.z;
        local.energy += 0.5f * mass * speed_sq - mass * Dot(gravity, pose.position);

        if (inv_inertia.x > 0.0f) {
            local.energy += 0.5f * angular_velocity.x * angular_velocity.x / inv_inertia.x;
        }
        if (inv_inertia.y > 0.0f) {
            local.energy += 0.5f * angular_velocity.y * angular_velocity.y / inv_inertia.y;
        }
        if (inv_inertia.z > 0.0f) {
            local.energy += 0.5f * angular_velocity.z * angular_velocity.z / inv_inertia.z;
        }
    }

    shared[tid] = local;
    __syncthreads();

    for (uint32_t offset = blockDim.x / 2u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            shared[tid].energy += shared[tid + offset].energy;
            shared[tid].linear_momentum =
                Add(shared[tid].linear_momentum, shared[tid + offset].linear_momentum);
            shared[tid].angular_momentum =
                Add(shared[tid].angular_momentum, shared[tid + offset].angular_momentum);
            shared[tid].nan_inf_count += shared[tid + offset].nan_inf_count;
            shared[tid].max_speed =
                fmaxf(shared[tid].max_speed, shared[tid + offset].max_speed);
            shared[tid].max_position_abs =
                fmaxf(shared[tid].max_position_abs, shared[tid + offset].max_position_abs);
        }
        __syncthreads();
    }

    if (tid == 0u) {
        out_slot[0] = shared[0];
    }
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

} // namespace

void ComputeRigidBodyInvariantSnapshot(cudaStream_t stream, int device_id,
                                       uint32_t body_count,
                                       const math::Transform* poses,
                                       const math::Vec3* linear_velocities,
                                       const math::Vec3* angular_velocities,
                                       const float* inverse_masses,
                                       const math::Vec3* inverse_inertias,
                                       math::Vec3 gravity,
                                       DeviceInvariantSnapshot* out_snapshot) {
    if (out_snapshot == nullptr) {
        throw std::runtime_error("ComputeRigidBodyInvariantSnapshot requires output storage");
    }
    *out_snapshot = {};
    if (body_count == 0u) {
        return;
    }

    phi::ScopedDeviceGuard guard(device_id);
    // BUF-10/BUF-13: the reduction slot is a phi-v2 opaque Buffer* bound to the
    // stream-0 DeviceBufferType (NULL/default stream -> the download below is
    // byte+ordering identical to the legacy synchronous cudaMemcpy). The kernel
    // still runs on context.stream EXACTLY as before; the explicit
    // cudaStreamSynchronize(stream) guarantees the kernel completes before the
    // (self-syncing) BufferDownload reads the slot.
    phi::Buffer* reduction =
        phi::BufferAlloc(phi::DeviceBufferType(phi::InitBestDevice()),
                         sizeof(DeviceReductionSlot));
    constexpr uint32_t kBlockSize = 256u;
    ComputeRigidBodyInvariantKernel<<<1u,
                                      kBlockSize,
                                      kBlockSize * sizeof(DeviceReductionSlot),
                                      stream>>>(
        body_count,
        poses,
        linear_velocities,
        angular_velocities,
        inverse_masses,
        inverse_inertias,
        gravity,
        static_cast<DeviceReductionSlot*>(phi::BufferBase(reduction)));
    const cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        phi::BufferFree(reduction);
        CheckCuda(launch_err, "ComputeRigidBodyInvariantKernel launch");
    }
    cudaStreamSynchronize(stream);

    DeviceReductionSlot host_slot;
    phi::BufferDownload(reduction, &host_slot, 0, sizeof(host_slot));
    phi::BufferFree(reduction);
    out_snapshot->energy = host_slot.energy;
    out_snapshot->linear_momentum = host_slot.linear_momentum;
    out_snapshot->angular_momentum = host_slot.angular_momentum;
    out_snapshot->nan_inf_count = host_slot.nan_inf_count;
    out_snapshot->max_speed = host_slot.max_speed;
    out_snapshot->max_position_abs = host_slot.max_position_abs;
}

} // namespace nuka::core::diagnostics::gpu
