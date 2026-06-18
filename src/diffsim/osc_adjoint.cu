// ---------------------------------------------------------------------------
// nuka::diffsim -- OSC differentiable adjoint kernel (v0.5 p03 R3b, R8 / R9)
// ---------------------------------------------------------------------------
//
// Implements LaunchOscAdjointGainTarget (see osc_adjoint.hpp for the full adjoint
// derivation + the D1 / clamp / scope contract). One warp per articulation, lane
// 0 owns the dense build + 3x3 LDL^T solve REPLICATED BIT-FOR-BIT from
// ApplyOscDriveKernel (articulation_drives.cu) so the adjoint differentiates the
// REAL (possibly floored) forward map. Zero float atomics; each articulation
// writes only its own grad slots => two-run byte-identical (D1).
//
// The gain/target channels are LINEAR in the params:
//     w = J g_tau,   s = A^-1 w,
//     dL/dKp = e_x.s,   dL/dKd = -(xdot.s),   dL/dx_target = Kp*s.
// A / Lambda is a fixed linear operator w.r.t. these params, so NO matrix-inverse
// derivative is needed -- we just reuse the forward's OWN factorization of A to
// apply A^-1 (A symmetric => A^-1 is its own transpose). g_tau is masked to zero
// on any joint whose forward tau saturated against the force limit (R9 stop-grad).
// ---------------------------------------------------------------------------

#include "diffsim/osc_adjoint.hpp"

#include "math/cuda_vec_ops.cuh"
#include "runtime/articulation/articulation_contacts.hpp"  // kMaxContactSolverDof
#include "runtime/articulation/articulation_drives.hpp"    // kMinDiagonalDrive

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::diffsim {

namespace {

namespace art = ::nuka::runtime::articulation;
namespace mg = ::nuka::math::gpu;
using art::ArticulationDeviceState;
using art::ArticulationJointType;

// SPD floor for the LDL^T -- the SAME exported art::kMinDiagonalDrive the forward
// uses, so the floored factorization is bit-identical (the adjoint-of-forward
// contract); a forward-side change to the floor now propagates here automatically.
using art::kMinDiagonalDrive;
constexpr uint32_t kMaxDof = art::kMaxContactSolverDof;  // 18

// Compile-time ceiling on the task-space dimension; the on-thread scratch
// (A[kMaxTaskDim^2], MinvJt rows, J row count) is sized by this. The ACTUAL
// task_dim is a RUNTIME launcher argument (3 == the current world-POSITION task,
// matching the forward; 6 == a future pose task), validated <= kMaxTaskDim in the
// launcher (LOUD throw on overflow, never a silent clamp).
constexpr uint32_t kMaxTaskDim = kMaxOscTaskDim;  // headroom for a 6-DOF pose task

void CheckCudaOsc(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("nuka::diffsim OSC adjoint ") +
                                 operation + ": " + cudaGetErrorString(result));
    }
}

// --- Forward-replicated device helpers (BIT-FOR-BIT with articulation_drives.cu).
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

// Same defensive-normalize q*v*q^-1 the forward uses to take a joint axis to world
// (replicated -- the forward's is translation-unit-local), so J's columns are
// built with bit-identical geometry to the forward.
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

// One warp per articulation, lane 0 owns the dense build+solve (forward-replicated)
// plus the adjoint solve + param-grad scatter. Disjoint output slots => D1.
__global__ void OscAdjointGainTargetKernel(ArticulationDeviceState state,
                                           uint32_t max_dof,
                                           uint32_t task_dim,
                                           uint32_t osc_task_link,
                                           const float* __restrict__ inertia_M_inv,
                                           const float* __restrict__ task_target,
                                           const float* __restrict__ kp,
                                           const float* __restrict__ kd,
                                           const float* __restrict__ drive_force_limits,
                                           const float* __restrict__ grad_tau,
                                           float* __restrict__ grad_kp,
                                           float* __restrict__ grad_kd,
                                           float* __restrict__ grad_target) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    // Default the outputs to 0 (degenerate / no-op articulations stay 0, which is
    // a deterministic, two-run-stable value).
    if (grad_kp != nullptr) grad_kp[articulation] = 0.0f;
    if (grad_kd != nullptr) grad_kd[articulation] = 0.0f;
    if (grad_target != nullptr) {
        for (uint32_t r = 0u; r < task_dim; ++r) {
            grad_target[articulation * task_dim + r] = 0.0f;
        }
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    if (osc_task_link >= count) {
        return;
    }
    const uint32_t task_link = offset + osc_task_link;

    // --- Build J (task_dim x ndof), row-major J[row*max_dof + dof]. Scratch sized
    //     by the named kMaxTaskDim cap. The geometry fills the 3 POSITION rows the
    //     forward emits (task_dim == 3 matches it exactly; a task_dim > 3 pose task
    //     leaves the extra rows zero until a pose-task forward exists).
    float J[kMaxTaskDim * kMaxDof];
    for (uint32_t i = 0u; i < task_dim * max_dof; ++i) {
        J[i] = 0.0f;
    }
    const math::Vec3 x_task = state.link_pose[task_link].position;

    uint32_t n = 0u;
    {
        uint32_t link = task_link;
        constexpr uint32_t kInvalidLink = ~0u;
        while (link != kInvalidLink) {
            const ArticulationJointType type = state.joint_type[link];
            const uint32_t dof_count = JointDofCountDriveDevice(type);
            if (dof_count == 1u) {
                const uint32_t dof = LocalDofIndexDrive(state, offset, link);
                if (dof < max_dof) {
                    const math::Vec3 axis_world = RotateByQuatDrive(
                        state.link_pose[link].rotation, state.joint_axis[link]);
                    math::Vec3 col;
                    if (type == ArticulationJointType::Prismatic) {
                        col = axis_world;
                    } else {
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
    if (n == 0u || inertia_M_inv == nullptr || task_target == nullptr ||
        grad_tau == nullptr) {
        return;  // degenerate / missing inputs: grads stay 0.
    }

    const size_t tile_stride = static_cast<size_t>(max_dof) * max_dof;
    const float* __restrict__ Minv =
        inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;

    // --- xdot = J*qdot (forward-replicated, fixed-order). Needed for the dL/dKd
    //     channel: edot_x = -xdot. (task_dim == 3 position task; rows 0..2 = x,y,z.)
    float xdot[kMaxTaskDim] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (JointDofCountDriveDevice(state.joint_type[link]) != 1u) {
            continue;
        }
        const uint32_t dof = LocalDofIndexDrive(state, offset, link);
        if (dof >= n) {
            continue;
        }
        const float qd = state.qdot[link];
        xdot[0] += J[0u * max_dof + dof] * qd;
        xdot[1] += J[1u * max_dof + dof] * qd;
        xdot[2] += J[2u * max_dof + dof] * qd;
    }

    // --- A = J M^-1 J^T (task_dim x task_dim), forward-replicated (same MinvJt
    //     intermediate + identical loop order). Scratch sized by kMaxTaskDim.
    float MinvJt[kMaxDof * kMaxTaskDim];
    for (uint32_t dof = 0u; dof < n; ++dof) {
        for (uint32_t r = 0u; r < task_dim; ++r) {
            float acc = 0.0f;
            for (uint32_t c = 0u; c < n; ++c) {
                acc += Minv[static_cast<size_t>(dof) * max_dof + c] *
                       J[r * max_dof + c];
            }
            MinvJt[dof * task_dim + r] = acc;
        }
    }
    float A[kMaxTaskDim * kMaxTaskDim];
    for (uint32_t r = 0u; r < task_dim; ++r) {
        for (uint32_t s = 0u; s < task_dim; ++s) {
            float acc = 0.0f;
            for (uint32_t dof = 0u; dof < n; ++dof) {
                acc += J[r * max_dof + dof] * MinvJt[dof * task_dim + s];
            }
            A[r * task_dim + s] = acc;
        }
    }

    // --- Factor A = L D L^T, forward-replicated (same kMinDiagonalDrive floor,
    //     same order). The SAME factorization is reused to apply A^-1 to BOTH the
    //     forward rhs (to recover y = A^-1 rhs, only needed if dL/dq were shipped)
    //     and the adjoint w (s = A^-1 w). A symmetric => A^-1 symmetric, so the
    //     adjoint solve is the exact transpose of the forward solve.
    float L[kMaxTaskDim * kMaxTaskDim];
    for (uint32_t i = 0u; i < task_dim * task_dim; ++i) {
        L[i] = A[i];
    }
    float d[kMaxTaskDim];
    for (uint32_t j = 0u; j < task_dim; ++j) {
        float djj = L[j * task_dim + j];
        for (uint32_t k = 0u; k < j; ++k) {
            djj -= L[j * task_dim + k] * L[j * task_dim + k] * d[k];
        }
        if (djj < kMinDiagonalDrive) {
            djj = kMinDiagonalDrive;
        }
        d[j] = djj;
        for (uint32_t i = j + 1u; i < task_dim; ++i) {
            float lij = L[i * task_dim + j];
            for (uint32_t k = 0u; k < j; ++k) {
                lij -= L[i * task_dim + k] * L[j * task_dim + k] * d[k];
            }
            L[i * task_dim + j] = lij / djj;
        }
    }

    // --- w = J * g_tau (3-vector). g_tau is link-indexed; scan the actuated 1-DOF
    //     joints in the SAME fixed forward order. R9: MASK a joint's g_tau to 0 if
    //     its forward tau saturated against the force limit (stop-grad sub-gradient
    //     at the clamp). The forward computes the UNCLAMPED tau_j = J_j^T y; we
    //     recompute that here (needs y = A^-1 rhs) to test saturation.
    //
    //   First recover y = A^-1 rhs (the forward's solve) so we can detect which
    //   joints saturated. rhs = Kp*e_x + Kd*edot_x = Kp*e_x - Kd*xdot.
    const float kp_s = (kp != nullptr) ? kp[task_link] : 0.0f;
    const float kd_s = (kd != nullptr) ? kd[task_link] : 0.0f;
    const math::Vec3 target{task_target[articulation * task_dim + 0u],
                            task_target[articulation * task_dim + 1u],
                            task_target[articulation * task_dim + 2u]};
    const math::Vec3 e_x = mg::Sub(target, x_task);
    // rhs row->component mapping is the position task's x,y,z (task_dim == 3).
    float rhs[kMaxTaskDim];
    for (uint32_t r = 0u; r < task_dim; ++r) {
        const float ex = (r == 0u) ? e_x.x : (r == 1u) ? e_x.y : e_x.z;
        rhs[r] = kp_s * ex - kd_s * xdot[r];
    }
    // Solve A*y = rhs (forward L, diagonal D, backward L^T) -- forward-replicated.
    float y[kMaxTaskDim];
    for (uint32_t i = 0u; i < task_dim; ++i) {
        float value = rhs[i];
        for (uint32_t k = 0u; k < i; ++k) {
            value -= L[i * task_dim + k] * y[k];
        }
        y[i] = value;
    }
    for (uint32_t i = 0u; i < task_dim; ++i) {
        y[i] /= d[i];
    }
    for (uint32_t ii = task_dim; ii > 0u; --ii) {
        const uint32_t i = ii - 1u;
        float value = y[i];
        for (uint32_t k = i + 1u; k < task_dim; ++k) {
            value -= L[k * task_dim + i] * y[k];
        }
        y[i] = value;
    }

    // w = J * g_tau, masking saturated joints. tau_j (unclamped) = J_j^T y.
    float w[kMaxTaskDim] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (JointDofCountDriveDevice(state.joint_type[link]) != 1u) {
            continue;
        }
        const uint32_t dof = LocalDofIndexDrive(state, offset, link);
        if (dof >= n) {
            continue;  // off-chain joint: J column is zero, contributes nothing.
        }
        float g = grad_tau[link];
        // R9: zero the gradient through a saturated component (stop-grad). The
        // forward applies tau = clamp(tau_j, +/- limit); a saturated joint's tau
        // is constant w.r.t. the params, so dL/dparam through it is 0.
        if (drive_force_limits != nullptr) {
            const float limit = drive_force_limits[link];
            if (limit > 0.0f) {
                float tau_j = 0.0f;
                for (uint32_t r = 0u; r < task_dim; ++r) {
                    tau_j += J[r * max_dof + dof] * y[r];
                }
                if (tau_j > limit || tau_j < -limit) {
                    g = 0.0f;  // saturated => stop-grad.
                }
            }
        }
        for (uint32_t r = 0u; r < task_dim; ++r) {
            w[r] += J[r * max_dof + dof] * g;
        }
    }

    // --- s = A^-1 w, reusing the forward's L D L^T factorization (forward solve
    //     shape, applied to w). Same fixed order => deterministic.
    float s[kMaxTaskDim];
    for (uint32_t i = 0u; i < task_dim; ++i) {
        float value = w[i];
        for (uint32_t k = 0u; k < i; ++k) {
            value -= L[i * task_dim + k] * s[k];
        }
        s[i] = value;
    }
    for (uint32_t i = 0u; i < task_dim; ++i) {
        s[i] /= d[i];
    }
    for (uint32_t ii = task_dim; ii > 0u; --ii) {
        const uint32_t i = ii - 1u;
        float value = s[i];
        for (uint32_t k = i + 1u; k < task_dim; ++k) {
            value -= L[k * task_dim + i] * s[k];
        }
        s[i] = value;
    }

    // --- Scatter the param gradients.
    //   dL/dKp       = e_x . s          (rhs = Kp*e_x + ...)
    //   dL/dKd       = edot_x . s = -(xdot . s)
    //   dL/dx_target = Kp * s           (e_x = x_target - x_task, d e_x/d x_target = I)
    if (grad_kp != nullptr) {
        grad_kp[articulation] = e_x.x * s[0] + e_x.y * s[1] + e_x.z * s[2];
    }
    if (grad_kd != nullptr) {
        float acc = 0.0f;
        for (uint32_t r = 0u; r < task_dim; ++r) acc += xdot[r] * s[r];
        grad_kd[articulation] = -acc;
    }
    if (grad_target != nullptr) {
        for (uint32_t r = 0u; r < task_dim; ++r) {
            grad_target[articulation * task_dim + r] = kp_s * s[r];
        }
    }
}

}  // namespace

void LaunchOscAdjointGainTarget(cudaStream_t stream, int device_id,
                                ArticulationDeviceState state,
                                uint32_t max_dof,
                                uint32_t task_dim,
                                uint32_t osc_task_link,
                                const float* inertia_M_inv,
                                const float* task_target,
                                const float* kp,
                                const float* kd,
                                const float* drive_force_limits,
                                const float* grad_tau,
                                OscAdjointParamGrads out_grads) {
    if (state.articulation_count == 0u || max_dof == 0u ||
        inertia_M_inv == nullptr || task_target == nullptr ||
        grad_tau == nullptr) {
        return;
    }
    // Honesty: the adjoint scratch is pinned at kMaxDof (==kMaxContactSolverDof, in
    // lockstep with the forward Osc drive); beyond it FAIL LOUDLY -- mirrors the
    // forward LaunchApplyOscDriveKernels throw -- NOT the old silent no-op that
    // would emit all-zero (silently wrong) gradients for a larger articulation.
    if (max_dof > kMaxDof) {
        throw std::runtime_error(
            "LaunchOscAdjointGainTarget: max_dof (" + std::to_string(max_dof) +
            ") exceeds kMaxDof (" + std::to_string(kMaxDof) +
            "), the Osc adjoint scratch cap");
    }
    // task_dim is the runtime task-space dimension (3 == the current position task).
    // Bounded by kMaxTaskDim (the on-thread A/J/MinvJt scratch cap); 0 or overflow
    // is a LOUD throw, never a silent clamp.
    if (task_dim == 0u || task_dim > kMaxTaskDim) {
        throw std::runtime_error(
            "LaunchOscAdjointGainTarget: task_dim (" + std::to_string(task_dim) +
            ") out of range (1.." + std::to_string(kMaxTaskDim) + ")");
    }
    if (out_grads.grad_kp == nullptr && out_grads.grad_kd == nullptr &&
        out_grads.grad_target == nullptr) {
        return;
    }
    phi::ScopedDeviceGuard guard(device_id);
    OscAdjointGainTargetKernel<<<state.articulation_count, 32u, 0u, stream>>>(
        state, max_dof, task_dim, osc_task_link, inertia_M_inv, task_target, kp, kd,
        drive_force_limits, grad_tau, out_grads.grad_kp, out_grads.grad_kd,
        out_grads.grad_target);
    CheckCudaOsc(cudaGetLastError(), "OscAdjointGainTargetKernel launch");
}

}  // namespace nuka::diffsim
