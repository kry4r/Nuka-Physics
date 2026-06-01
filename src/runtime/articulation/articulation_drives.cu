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

#include "math/cuda_vec_ops.cuh"                          // Cross/Dot device math
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

// Osc task-Jacobian device math. Sub/Cross/Dot come from the shared device math
// library (math/cuda_vec_ops.cuh). RotateByQuat is the SAME defensive-normalize
// q*v*q^-1 recipe ComputeContactChainJacobianKernel uses to take a joint axis to
// world (replicated -- that one is translation-unit-local), so the Osc task
// Jacobian's columns are built with bit-identical geometry to the contact-row J.
namespace mg = ::nuka::math::gpu;

__device__ math::Vec3 RotateByQuatDrive(math::Quat q, math::Vec3 v) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq > 1.0e-12f) {
        const float inv_norm = rsqrtf(norm_sq);
        q.w *= inv_norm;
        q.x *= inv_norm;
        q.y *= inv_norm;
        q.z *= inv_norm;
    }
    const math::Vec3 qv{q.x, q.y, q.z};
    const math::Vec3 t = mg::Cross(qv, v);
    const math::Vec3 t2{2.0f * t.x, 2.0f * t.y, 2.0f * t.z};
    const math::Vec3 wt{q.w * t2.x, q.w * t2.y, q.w * t2.z};
    const math::Vec3 qvt = mg::Cross(qv, t2);
    return {v.x + wt.x + qvt.x, v.y + wt.y + qvt.y, v.z + wt.z + qvt.z};
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

// Osc mode (slice 3, p03 R2): operational-space control, FORWARD, 3-DOF position
// task on the world position of one task link. One block per articulation, lane 0
// owns the dense 3xndof task-Jacobian build + the 3x3 LDL^T solve (fixed order,
// no float atomics => D1). See articulation_drives.hpp for the full law.
//
// Builds J (3 rows = the 3 spatial components of the geometric point Jacobian)
// by walking the task link's ancestor chain ONCE -- the SAME per-joint geometry
// as ComputeContactChainJacobianKernel: revolute column = cross(axis_world,
// x_task - joint_origin), prismatic column = axis_world. Forms A = J M^-1 J^T
// (3x3, SPD), solves A*y = (Kp*e_x + Kd*edot_x), and scatters tau_j = J_j^T*y.
__global__ void ApplyOscDriveKernel(ArticulationDeviceState state,
                                    uint32_t max_dof,
                                    uint32_t osc_task_link,
                                    const float* inertia_M_inv,
                                    const float* task_target,
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
    if (osc_task_link >= count) {
        return;  // task link out of range for this articulation: no-op.
    }
    const uint32_t task_link = offset + osc_task_link;

    // J is 3 x ndof, stored row-major in registers as J[row*max_dof + dof].
    // Zero it first: only the task link's ancestor-chain columns are written.
    float J[3u * kMaxContactSolverDof];
    for (uint32_t i = 0u; i < 3u * max_dof; ++i) {
        J[i] = 0.0f;
    }

    // x_current = task link world position (link_pose refreshed from the current q
    // THIS step by the caller, mirroring stage 4 -- otherwise it is one step stale).
    const math::Vec3 x_task = state.link_pose[task_link].position;

    // Walk the ancestor chain task_link -> root, writing each ancestor DOF's 3
    // Jacobian components. FloatingBase columns are intentionally NOT built (scope:
    // fixed-base scenes only); a floating root contributes no task columns here.
    uint32_t n = 0u;  // ndof actually present (high-water mark of dof_index+1).
    {
        uint32_t link = task_link;
        constexpr uint32_t kInvalidLink = ~0u;
        while (link != kInvalidLink) {
            const ArticulationJointType type = state.joint_type[link];
            const uint32_t dof_count = JointDofCountDriveDevice(type);
            if (dof_count == 1u) {  // Revolute / Prismatic (1 DOF).
                const uint32_t dof = LocalDofIndexDrive(state, offset, link);
                if (dof < max_dof) {
                    const math::Vec3 axis_world = RotateByQuatDrive(
                        state.link_pose[link].rotation, state.joint_axis[link]);
                    math::Vec3 col;
                    if (type == ArticulationJointType::Prismatic) {
                        col = axis_world;  // d(point)/dq = axis (translation).
                    } else {               // Revolute: d(point)/dq = axis x lever.
                        const math::Vec3 lever =
                            mg::Sub(x_task, state.link_pose[link].position);
                        col = mg::Cross(axis_world, lever);
                    }
                    J[0u * max_dof + dof] = col.x;
                    J[1u * max_dof + dof] = col.y;
                    J[2u * max_dof + dof] = col.z;
                    if (dof + 1u > n) {
                        n = dof + 1u;
                    }
                }
            }
            const uint32_t parent_local = state.parent_link[link];
            link = (parent_local == kInvalidLink) ? kInvalidLink
                                                  : (offset + parent_local);
        }
    }
    if (n == 0u || inertia_M_inv == nullptr || task_target == nullptr) {
        return;  // degenerate (e.g. task link at root) or missing inputs: no-op.
    }

    const size_t tile_stride = static_cast<size_t>(max_dof) * max_dof;
    const float* const Minv =
        inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;

    // xdot_current = J * qdot (3-vector), over the present DOF columns. qdot is
    // link-indexed; column dof maps back to its link, but J already encodes the
    // chain, so we accumulate per task row by scanning the chain again would be
    // wasteful -- instead read qdot through a column->link gather. The chain DOFs
    // are sparse in [0,n); J's zero columns contribute nothing, so summing over
    // all columns is correct and fixed-order (D1). Map column dof -> global link
    // via a single forward scan of the articulation's links (same prefix-sum
    // convention as LocalDofIndexDrive).
    float xdot[3] = {0.0f, 0.0f, 0.0f};
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (JointDofCountDriveDevice(state.joint_type[link]) != 1u) {
            continue;  // fixed (0) / floating (skipped above) -> no column.
        }
        const uint32_t dof = LocalDofIndexDrive(state, offset, link);
        if (dof >= n) {
            continue;  // off-chain joint: its J columns are zero.
        }
        const float qd = state.qdot[link];
        xdot[0] += J[0u * max_dof + dof] * qd;
        xdot[1] += J[1u * max_dof + dof] * qd;
        xdot[2] += J[2u * max_dof + dof] * qd;
    }

    // A = J M^-1 J^T (3x3). MinvJt[dof][r] = sum_c Minv[dof][c] * J[r][c] over the
    // present columns; A[r][s] = sum_dof J[r][dof] * MinvJt[dof][s]. Fixed loop
    // order over [0,n) DOFs => D1 (zero columns of J contribute exact zeros).
    float MinvJt[kMaxContactSolverDof * 3u];
    for (uint32_t dof = 0u; dof < n; ++dof) {
        for (uint32_t r = 0u; r < 3u; ++r) {
            float acc = 0.0f;
            for (uint32_t c = 0u; c < n; ++c) {
                acc += Minv[static_cast<size_t>(dof) * max_dof + c] *
                       J[r * max_dof + c];
            }
            MinvJt[dof * 3u + r] = acc;
        }
    }
    float A[9];
    for (uint32_t r = 0u; r < 3u; ++r) {
        for (uint32_t s = 0u; s < 3u; ++s) {
            float acc = 0.0f;
            for (uint32_t dof = 0u; dof < n; ++dof) {
                acc += J[r * max_dof + dof] * MinvJt[dof * 3u + s];
            }
            A[r * 3u + s] = acc;
        }
    }

    // rhs = Kp*e_x + Kd*edot_x = Kp*(x_target - x_task) + Kd*(-xdot). Gains are
    // SCALAR per articulation (reused drive_stiffness/drive_damping at task_link),
    // applied to all 3 task DOFs; null kp/kd drop that term.
    const float kp_s = (kp != nullptr) ? kp[task_link] : 0.0f;
    const float kd_s = (kd != nullptr) ? kd[task_link] : 0.0f;
    const math::Vec3 target{task_target[articulation * 3u + 0u],
                            task_target[articulation * 3u + 1u],
                            task_target[articulation * 3u + 2u]};
    const math::Vec3 e_x = mg::Sub(target, x_task);
    float rhs[3];
    for (uint32_t r = 0u; r < 3u; ++r) {
        const float ex = (r == 0u) ? e_x.x : (r == 1u) ? e_x.y : e_x.z;
        rhs[r] = kp_s * ex - kd_s * xdot[r];
    }

    // Solve A*y = rhs with a fixed-order 3x3 LDL^T (A SPD on a non-degenerate J;
    // the SPD floor kMinDiagonalDrive guards a rank-deficient/collinear J -> a
    // large-but-finite y, never NaN/Inf). Mirrors the ComputedTorque LDL^T solve.
    float L[9];  // unit-lower (only strictly-lower entries used) + scratch copy.
    for (uint32_t i = 0u; i < 9u; ++i) {
        L[i] = A[i];
    }
    float d[3];
    for (uint32_t j = 0u; j < 3u; ++j) {
        float djj = L[j * 3u + j];
        for (uint32_t k = 0u; k < j; ++k) {
            djj -= L[j * 3u + k] * L[j * 3u + k] * d[k];
        }
        if (djj < kMinDiagonalDrive) {
            djj = kMinDiagonalDrive;
        }
        d[j] = djj;
        for (uint32_t i = j + 1u; i < 3u; ++i) {
            float lij = L[i * 3u + j];
            for (uint32_t k = 0u; k < j; ++k) {
                lij -= L[i * 3u + k] * L[j * 3u + k] * d[k];
            }
            L[i * 3u + j] = lij / djj;
        }
    }
    float y[3];
    for (uint32_t i = 0u; i < 3u; ++i) {  // forward L.
        float value = rhs[i];
        for (uint32_t k = 0u; k < i; ++k) {
            value -= L[i * 3u + k] * y[k];
        }
        y[i] = value;
    }
    for (uint32_t i = 0u; i < 3u; ++i) {  // diagonal D.
        y[i] /= d[i];
    }
    for (uint32_t ii = 3u; ii > 0u; --ii) {  // backward L^T.
        const uint32_t i = ii - 1u;
        float value = y[i];
        for (uint32_t k = i + 1u; k < 3u; ++k) {
            value -= L[k * 3u + i] * y[k];
        }
        y[i] = value;
    }

    // Scatter tau_j = J_j^T * y per joint DOF (== J^T Lambda (Kp*e_x+Kd*edot_x)),
    // with the force-limit clamp. Column dof -> global link via the same forward
    // prefix-sum scan. A 1-DOF joint NOT on the task link's chain has all-zero J
    // columns, so its tau is exactly 0 -- the task exerts no torque on it. We
    // write EVERY actuated (1-DOF) joint (zero off the chain) rather than leaving
    // a stale prior-step tau; fixed joints / the (unbuilt) floating base keep
    // theirs (ABA ignores tau for them). Fixed order, no atomics => D1.
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (JointDofCountDriveDevice(state.joint_type[link]) != 1u) {
            continue;  // fixed (0) / floating (unbuilt) -> not actuated here.
        }
        const uint32_t dof = LocalDofIndexDrive(state, offset, link);
        float tau = 0.0f;
        if (dof < n) {
            tau = J[0u * max_dof + dof] * y[0] +
                  J[1u * max_dof + dof] * y[1] +
                  J[2u * max_dof + dof] * y[2];
        }
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

void LaunchApplyOscDriveKernels(const phi::DeviceContext& context,
                                ArticulationDeviceState state,
                                uint32_t max_dof,
                                uint32_t osc_task_link,
                                const float* inertia_M_inv,
                                const float* task_target,
                                const float* kp,
                                const float* kd,
                                const float* drive_force_limits) {
    // Require the physics M^-1 tile and the per-env task target; missing either is
    // a no-op (tau left as-is). One block per articulation (lane 0 owns the dense
    // task-Jacobian build + 3x3 solve), matching the ComputedTorque / CRBA grid.
    if (state.articulation_count == 0u || max_dof == 0u ||
        max_dof > kMaxContactSolverDof || inertia_M_inv == nullptr ||
        task_target == nullptr) {
        return;
    }
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    ApplyOscDriveKernel<<<state.articulation_count, 32u, 0u, stream>>>(
        state, max_dof, osc_task_link, inertia_M_inv, task_target, kp, kd,
        drive_force_limits);
    CheckCudaDrive(cudaGetLastError(), "ApplyOscDriveKernel launch");
}

} // namespace nuka::runtime::articulation
