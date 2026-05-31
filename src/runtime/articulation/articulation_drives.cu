// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- NON-PD control-mode drive kernels (v0.5 C-fwd)
// ---------------------------------------------------------------------------
//
// Stage-1 Torque / Velocity (slice 1) + ComputedTorque / Actuator (slice 2)
// control laws. See articulation_drives.hpp for the contract. PDPosition stays in
// featherstone_aba.cu (untouched).
//
// D1 determinism: the Torque/Velocity/Actuator kernels are a flat per-link grid
// (link = blockIdx*blockDim + threadIdx), each thread writes ONLY its own
// tau[link]. ComputedTorque is one block per articulation (lane 0 owns the dense
// DOF-space matvec) -- a per-articulation, fixed-order, atomic-free reduction.
// No cross-thread float reduction, no float atomics, fixed grid/loop order =>
// bit-identical across replicas and across runs (matches ApplyPositionDriveKernel
// / the per-articulation CRBA kernel).
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_drives.hpp"

#include "runtime/articulation/articulation_contacts.hpp"  // kMaxContactSolverDof
#include "runtime/articulation/articulation_state.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::runtime::articulation {

namespace {

// Mirror featherstone_aba.cu's block size so the per-link grid is identical.
constexpr uint32_t kDriveBlockSize = 32u;

// SPD floor for the ComputedTorque joint-block LDL^T solve. Matches the
// kMinDiagonal used in FactorArticulationInertiaMKernel / the ABA factorization.
constexpr float kMinDiagonalDrive = 1.0e-6f;

void CheckCudaDrive(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// Per-joint DOF count + base-inclusive local dof_index. Bit-identical to the
// file-local helpers in articulation_contacts.cu (the CRBA tile uses the SAME
// indexing, so the ComputedTorque matvec's M rows/cols line up with the M built
// there). Replicated here because those are translation-unit-local.
__device__ uint32_t JointDofCountDriveDevice(ArticulationJointType type) {
    switch (type) {
        case ArticulationJointType::Revolute:
        case ArticulationJointType::Prismatic:
            return 1u;
        case ArticulationJointType::Fixed:
            return 0u;
        case ArticulationJointType::FloatingBase:
            return 6u;
    }
    return 0u;
}

__device__ uint32_t LocalDofIndexDrive(const ArticulationDeviceState& state,
                                       uint32_t offset, uint32_t link) {
    uint32_t index = 0u;
    for (uint32_t k = offset; k < link; ++k) {
        index += JointDofCountDriveDevice(state.joint_type[k]);
    }
    return index;
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

// ComputedTorque mode: solve D*tau_j = (a_des_j - qddot_free_j) for the joint
// torque, where D is the JOINT sub-block of the physics-M INVERSE (inertia_M_inv,
// the m_inv_ tile). a_des = qddot_des + Kp*e + Kd*edot (qddot_des/qdot_target
// default 0). One block per articulation, lane 0 owns the dense DOF-space solve
// (fixed order, atomic-free => D1).
//
// WHY the inverse sub-block, not M_jj. On a FLOATING base the engine's effective
// joint admittance is qddot_j = D*tau_j + qddot_free_j, with D the Schur-
// complement inverse = exactly the joint sub-block of the full-tile physics-M
// inverse (M_jj alone would IGNORE the base reaction, giving qddot != a_des).
// Solving D*tau_j = a_des_j - qddot_free_j therefore makes stage-2 ABA realize
// qddot_j = a_des_j, and the engine bias (Coriolis/gravity/model joint damping,
// all inside qddot_free) cancels exactly. inertia_M_inv is indexed by
// LocalDofIndex (same tile layout as M); the joint block starts at dof offset
// base_dof (6 for a floating root, 0 for a fixed base).
__global__ void ApplyComputedTorqueDriveKernel(ArticulationDeviceState state,
                                               uint32_t max_dof,
                                               const float* inertia_M_inv,
                                               const float* qddot_free,
                                               const float* q_target,
                                               const float* kp,
                                               const float* kd,
                                               const float* drive_force_limits) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    const size_t tile_stride = static_cast<size_t>(max_dof) * max_dof;
    const float* const Minv =
        inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;

    const bool floating_root =
        (count > 0u) &&
        (state.joint_type[offset] == ArticulationJointType::FloatingBase);
    const uint32_t base_dof = floating_root ? 6u : 0u;

    // Build the joint-DOF right-hand side rhs[j] = a_des_j - qddot_free_j (in
    // JOINT-local index, i.e. global dof_i - base_dof), and remember which link
    // each joint DOF maps to (for the scatter). All revolute/prismatic joints are
    // 1 DOF, so the joint DOFs are contiguous after the base block.
    float rhs[kMaxContactSolverDof];
    uint32_t dof_to_link[kMaxContactSolverDof];
    uint32_t n_j = 0u;  // joint DOF count = total dof - base_dof.
    for (uint32_t i = 0u; i < max_dof; ++i) {
        rhs[i] = 0.0f;
        dof_to_link[i] = 0u;
    }
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (JointDofCountDriveDevice(state.joint_type[link]) == 0u) {
            continue;  // fixed joint: no DOF column.
        }
        if (local == 0u && floating_root) {
            continue;  // free base: not actuated.
        }
        const uint32_t dof_i = LocalDofIndexDrive(state, offset, link);
        if (dof_i < base_dof || dof_i >= max_dof) {
            continue;
        }
        const uint32_t jdof = dof_i - base_dof;
        if (jdof >= max_dof) {
            continue;
        }
        // a_des = qddot_des + Kp*(q_target - q) + Kd*(qdot_target - qdot), with
        // qddot_des = qdot_target = 0 (the common computed-torque PD form).
        const float e = q_target[link] - state.q[link];
        const float edot = -state.qdot[link];
        float a_des = 0.0f;
        if (kp != nullptr) {
            a_des += kp[link] * e;
        }
        if (kd != nullptr) {
            a_des += kd[link] * edot;
        }
        rhs[jdof] = a_des - qddot_free[link];
        dof_to_link[jdof] = link;
        if (jdof + 1u > n_j) {
            n_j = jdof + 1u;
        }
    }
    if (n_j == 0u) {
        return;
    }

    // Copy the joint sub-block D = Minv[base_dof.., base_dof..] into local scratch
    // and LDL^T-factor it (mirrors FactorArticulationInertiaMKernel; n_j <=
    // kMaxContactSolverDof). D is SPD (a principal sub-block of an SPD inverse).
    float a[kMaxContactSolverDof * kMaxContactSolverDof];
    for (uint32_t r = 0u; r < n_j; ++r) {
        for (uint32_t c = 0u; c < n_j; ++c) {
            a[r * kMaxContactSolverDof + c] =
                Minv[static_cast<size_t>(base_dof + r) * max_dof + (base_dof + c)];
        }
    }
    float d[kMaxContactSolverDof];
    for (uint32_t j = 0u; j < n_j; ++j) {
        float djj = a[j * kMaxContactSolverDof + j];
        for (uint32_t k = 0u; k < j; ++k) {
            djj -= a[j * kMaxContactSolverDof + k] *
                   a[j * kMaxContactSolverDof + k] * d[k];
        }
        if (djj < kMinDiagonalDrive) {
            djj = kMinDiagonalDrive;  // SPD floor (guards a degenerate config).
        }
        d[j] = djj;
        for (uint32_t i = j + 1u; i < n_j; ++i) {
            float lij = a[i * kMaxContactSolverDof + j];
            for (uint32_t k = 0u; k < j; ++k) {
                lij -= a[i * kMaxContactSolverDof + k] *
                       a[j * kMaxContactSolverDof + k] * d[k];
            }
            a[i * kMaxContactSolverDof + j] = lij / djj;
        }
    }
    // Solve D * tau_j = rhs (forward L, diagonal D, backward L^T) -- one RHS.
    float y[kMaxContactSolverDof];
    for (uint32_t i = 0u; i < n_j; ++i) {
        float value = rhs[i];
        for (uint32_t k = 0u; k < i; ++k) {
            value -= a[i * kMaxContactSolverDof + k] * y[k];
        }
        y[i] = value;
    }
    for (uint32_t i = 0u; i < n_j; ++i) {
        y[i] /= d[i];
    }
    float tau_j[kMaxContactSolverDof];
    for (uint32_t ii = n_j; ii > 0u; --ii) {
        const uint32_t i = ii - 1u;
        float value = y[i];
        for (uint32_t k = i + 1u; k < n_j; ++k) {
            value -= a[k * kMaxContactSolverDof + i] * tau_j[k];
        }
        tau_j[i] = value;
    }

    // Scatter to link-indexed tau with the force-limit clamp. Fixed joints / the
    // free base keep whatever tau they had (ABA ignores tau for them).
    for (uint32_t jdof = 0u; jdof < n_j; ++jdof) {
        const uint32_t link = dof_to_link[jdof];
        float tau = tau_j[jdof];
        if (drive_force_limits != nullptr) {
            const float limit = drive_force_limits[link];
            if (limit > 0.0f) {
                tau = fminf(fmaxf(tau, -limit), limit);
            }
        }
        state.tau[link] = tau;
    }
}

// Actuator mode: tau = clamp(torque_input, +/- tau_max(qdot)), where the DC-motor
// torque-speed envelope is tau_max = clamp(tau_stall*(1-|qdot|/qdot_noload), 0,
// tau_stall). tau_stall reuses drive_force_limits; qdot_noload is per-link. A
// link with qdot_noload <= 0 falls back to the plain drive_force_limits clamp.
__global__ void ApplyActuatorDriveKernel(ArticulationDeviceState state,
                                         const float* torque_input,
                                         const float* drive_force_limits,
                                         const float* actuator_noload_speed) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count ||
        state.joint_type[link] == ArticulationJointType::Fixed) {
        return;
    }
    float tau = torque_input[link];
    if (drive_force_limits != nullptr) {
        const float tau_stall = drive_force_limits[link];
        if (tau_stall > 0.0f) {
            float tau_max = tau_stall;  // standstill: full stall torque available.
            if (actuator_noload_speed != nullptr) {
                const float qdot_noload = actuator_noload_speed[link];
                if (qdot_noload > 0.0f) {
                    // Linear DC-motor envelope; |qdot| -> symmetric for both
                    // rotation directions. clamp(.,0,tau_stall): no regeneration
                    // past the no-load speed, never exceeds stall.
                    const float frac =
                        1.0f - fabsf(state.qdot[link]) / qdot_noload;
                    tau_max = tau_stall * fminf(fmaxf(frac, 0.0f), 1.0f);
                }
            }
            tau = fminf(fmaxf(tau, -tau_max), tau_max);
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

void LaunchApplyComputedTorqueDriveKernels(const phi::DeviceContext& context,
                                           ArticulationDeviceState state,
                                           uint32_t max_dof,
                                           const float* inertia_M_inv,
                                           const float* qddot_free,
                                           const float* q_target,
                                           const float* kp,
                                           const float* kd,
                                           const float* drive_force_limits) {
    // Require the dynamics operands (M^-1, qddot_free) and the position target;
    // missing any is a no-op (tau left as-is). The grid is one block per
    // articulation (lane 0 owns the dense joint-block solve), matching the CRBA
    // kernel.
    if (state.articulation_count == 0u || max_dof == 0u ||
        max_dof > kMaxContactSolverDof || inertia_M_inv == nullptr ||
        qddot_free == nullptr || q_target == nullptr) {
        return;
    }
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    ApplyComputedTorqueDriveKernel<<<state.articulation_count, 32u, 0u, stream>>>(
        state, max_dof, inertia_M_inv, qddot_free, q_target, kp, kd,
        drive_force_limits);
    CheckCudaDrive(cudaGetLastError(), "ApplyComputedTorqueDriveKernel launch");
}

void LaunchApplyActuatorDriveKernels(const phi::DeviceContext& context,
                                     ArticulationDeviceState state,
                                     const float* torque_input,
                                     const float* drive_force_limits,
                                     const float* actuator_noload_speed) {
    if (state.total_link_count == 0u || torque_input == nullptr) {
        return;
    }
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    const uint32_t blocks =
        (state.total_link_count + kDriveBlockSize - 1u) / kDriveBlockSize;
    ApplyActuatorDriveKernel<<<blocks, kDriveBlockSize, 0u, stream>>>(
        state, torque_input, drive_force_limits, actuator_noload_speed);
    CheckCudaDrive(cudaGetLastError(), "ApplyActuatorDriveKernel launch");
}

} // namespace nuka::runtime::articulation
