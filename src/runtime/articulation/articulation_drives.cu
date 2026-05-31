// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- NON-PD control-mode drive kernels (v0.5 C-fwd)
// ---------------------------------------------------------------------------
//
// Stage-1 Torque / Velocity control laws (slice 1). See articulation_drives.hpp
// for the contract. PDPosition stays in featherstone_aba.cu (untouched).
//
// D1 determinism: both kernels are a flat per-link grid (link =
// blockIdx*blockDim + threadIdx), each thread writes ONLY its own tau[link].
// No cross-thread reduction, no float atomics, fixed grid/loop order => bit-
// identical across replicas and across runs (matches ApplyPositionDriveKernel).
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_drives.hpp"

#include "runtime/articulation/articulation_state.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::runtime::articulation {

namespace {

// Mirror featherstone_aba.cu's block size so the per-link grid is identical.
constexpr uint32_t kDriveBlockSize = 32u;

void CheckCudaDrive(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// tau = clamp(torque_input). No control Kd; the implicit joint damping (#43)
// still runs downstream for the physical joint damping.
__global__ void ApplyTorqueDriveKernel(ArticulationDeviceState state,
                                       const float* torque_input,
                                       const float* drive_force_limits) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count ||
        state.joint_type[link] == ArticulationJointType::Fixed) {
        return;
    }
    float tau = torque_input[link];
    if (drive_force_limits != nullptr) {
        const float limit = drive_force_limits[link];
        if (limit > 0.0f) {
            tau = fminf(fmaxf(tau, -limit), limit);
        }
    }
    state.tau[link] = tau;
}

// tau = clamp(Kp_v * (velocity_target - qdot)), Kp_v reusing drive_stiffness.
// No control Kd; implicit joint damping still runs downstream.
__global__ void ApplyVelocityDriveKernel(ArticulationDeviceState state,
                                         const float* velocity_target,
                                         const float* drive_stiffness,
                                         const float* drive_force_limits) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count ||
        state.joint_type[link] == ArticulationJointType::Fixed) {
        return;
    }
    float tau = drive_stiffness[link] * (velocity_target[link] - state.qdot[link]);
    if (drive_force_limits != nullptr) {
        const float limit = drive_force_limits[link];
        if (limit > 0.0f) {
            tau = fminf(fmaxf(tau, -limit), limit);
        }
    }
    state.tau[link] = tau;
}

}  // namespace

void LaunchApplyTorqueDriveKernels(const phi::DeviceContext& context,
                                   ArticulationDeviceState state,
                                   const float* torque_input,
                                   const float* drive_force_limits) {
    if (state.total_link_count == 0u || torque_input == nullptr) {
        return;
    }
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    const uint32_t blocks =
        (state.total_link_count + kDriveBlockSize - 1u) / kDriveBlockSize;
    ApplyTorqueDriveKernel<<<blocks, kDriveBlockSize, 0u, stream>>>(
        state, torque_input, drive_force_limits);
    CheckCudaDrive(cudaGetLastError(), "ApplyTorqueDriveKernel launch");
}

void LaunchApplyVelocityDriveKernels(const phi::DeviceContext& context,
                                     ArticulationDeviceState state,
                                     const float* velocity_target,
                                     const float* drive_stiffness,
                                     const float* drive_force_limits) {
    if (state.total_link_count == 0u || velocity_target == nullptr ||
        drive_stiffness == nullptr) {
        return;
    }
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    const uint32_t blocks =
        (state.total_link_count + kDriveBlockSize - 1u) / kDriveBlockSize;
    ApplyVelocityDriveKernel<<<blocks, kDriveBlockSize, 0u, stream>>>(
        state, velocity_target, drive_stiffness, drive_force_limits);
    CheckCudaDrive(cudaGetLastError(), "ApplyVelocityDriveKernel launch");
}

} // namespace nuka::runtime::articulation
