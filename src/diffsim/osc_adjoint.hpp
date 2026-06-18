#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- OSC (operational-space control) differentiable adjoint
//                  (v0.5 p03 R3b, risk R8 / R9)
// ---------------------------------------------------------------------------
//
// Reverse-mode adjoint of the FORWARD OSC drive kernel committed in
// runtime/articulation/articulation_drives.cu (`ApplyOscDriveKernel`). The
// forward law, per articulation, on a FIXED-base 3-DOF world-POSITION task at
// link `osc_task_link`, is
//
//     tau_j = J_j^T * y,    y = A^-1 * rhs,    A = J M^-1 J^T  (3x3, SPD),
//     rhs   = Kp*e_x + Kd*edot_x,
//     e_x   = x_target - x_task,    edot_x = -xdot,   xdot = J*qdot,
//
// with J the 3 x ndof geometric point Jacobian of the task link (revolute col =
// cross(axis_world, x_task - joint_origin), prismatic col = axis_world), and Kp,
// Kd SCALAR per-articulation gains (drive_stiffness/drive_damping at the task
// link). See articulation_drives.hpp for the authoritative forward contract.
//
// GIVEN the upstream gradient g_tau = dL/dtau (per joint DOF, link-indexed, in
// the SAME convention the forward writes state.tau), this runner produces the
// gradients w.r.t. the DIFFERENTIABLE CONTROL PARAMETERS. The gain and target
// channels enter rhs LINEARLY (A / Lambda is a FIXED linear operator w.r.t.
// these params -- no matrix-inverse derivative needed), which is exactly what
// makes them tractable and FD-tight (~float precision):
//
//     w        = J * g_tau          (3-vector; since tau = J^T y, dL/dy = J g_tau)
//     s        = A^-1 * w           (A symmetric => SAME operator as the forward;
//                                    solved with the forward's OWN 3x3 LDL^T)
//     dL/dKp        = e_x   . s     (scalar)
//     dL/dKd        = edot_x . s  = -(xdot . s)   (scalar; note the SIGN)
//     dL/dx_target  = Kp * s        (3-vector, env-major like task_target)
//
// CHANNELS SHIPPED:  dL/dKp, dL/dKd, dL/dx_target  (gain + task-target).
// CHANNEL DEFERRED:  dL/dq through J(q) and A=A(q). That is the genuine
//   matrix-inverse-derivative (d(A^-1) = -A^-1 dA A^-1) showcase; it is NOT on
//   the v0.5 exit-demo critical path (the demo's parameter spine is link MASS
//   through the direct-dynamics adjoint, not OSC) and the validated task point
//   (calf-link origin) has a RANK-2 J -> the floored A makes the null-direction
//   dL/dq derivative ill-posed. Deferred to v0.7 with this honest note, mirroring
//   ift_runner.cu's deferral of its d/dM, d/dJ channels.
//
// R9 (clamp nonsmoothness): if the forward clamps tau_j to +/- force-limit, a
// SATURATED joint component is treated stop-grad-like -- its g_tau entry is
// MASKED to zero before w = J g_tau (sub-gradient at the clamp). With null /
// high force limits (the clean primary validation) no joint saturates and the
// mask is inert. This mirrors the R5/R9 row-class clamp pattern from p01.
//
// D1 / PERF (mirrors kkt_builder.cu / ift_runner.cu / the forward kernel):
//   * ONE warp per articulation, grid = articulation_count -> independent warps,
//     no global sync, no float atomics. Each articulation writes only ITS OWN
//     grad_kp[a] / grad_kd[a] / grad_target[a*3..] slots (disjoint) => race-free,
//     two-run byte-identical (D1).
//   * lane 0 owns the dense J-build + 3x3 LDL^T solve, BIT-FOR-BIT the forward's
//     code path (same RotateByQuat, same chain walk, same kMinDiagonalDrive floor,
//     same fixed loop order), so `s = A^-1 w` is the exact transpose of the REAL
//     (possibly floored) forward map -- the adjoint differentiates the real
//     forward, not an idealized one. (We deliberately do NOT cooperative-ize the
//     build across lanes a la kkt_builder: matching the forward's lane-0 FP order
//     exactly is worth more than the parallelism for a 3-row task here.)
//   * J + M^-1 read coalesced; __restrict__ pointers; fixed-order fp32 reductions.
// ---------------------------------------------------------------------------

#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>

namespace nuka::diffsim {

namespace articulation = nuka::runtime::articulation;

// Named ceiling on the OSC task dimension (headroom for a future 6-DOF pose task).
// The on-thread adjoint scratch (A[kMaxOscTaskDim^2], MinvJt rows) is sized by this;
// the RUNTIME task_dim passed to the launcher must satisfy 1 <= task_dim <=
// kMaxOscTaskDim or the launcher throws LOUDLY (never a silent clamp).
inline constexpr uint32_t kMaxOscTaskDim = 6u;

// Output param-gradient buffers for the OSC adjoint. The caller owns these; they
// must be sized for `articulation_count` articulations (grad_kp / grad_kd:
// 1 float each; grad_target: `task_dim` floats each, env-major like the forward's
// task_target). This is the clean callable the diff-sim parameter spine consumes
// (mirrors how StepBackwardGrads / the tape's grad_parameters_out slot expose
// param gradients); C-ABI wiring is intentionally NOT done this round.
struct OscAdjointParamGrads {
    float* grad_kp = nullptr;      // float[articulation_count]          dL/dKp per articulation
    float* grad_kd = nullptr;      // float[articulation_count]          dL/dKd per articulation
    float* grad_target = nullptr;  // float[articulation_count*task_dim] dL/dx_target (env-major)
};

// Reverse-mode adjoint of LaunchApplyOscDriveKernels. Differentiates the gain
// (Kp, Kd) and task-target channels of the FORWARD OSC drive.
//
//   state            : the SAME ArticulationDeviceState the forward consumed.
//                      link_pose / q / qdot MUST be the values at which the
//                      forward was evaluated (the forward reads x_task from
//                      link_pose[task_link] and xdot from qdot) -- i.e. do NOT
//                      re-step between the forward and this adjoint.
//   max_dof          : the M^-1 tile / Jacobian stride (== forward's max_dof).
//   task_dim         : the task-space dimension (3 == world-POSITION task, matching
//                      the current forward; bounded by kMaxOscTaskDim). Sizes the
//                      J row count, the task inertia A, and the grad_target stride.
//   osc_task_link    : articulation-LOCAL task link index (== forward's).
//   inertia_M_inv    : the SAME per-articulation full-tile physics-M inverse the
//                      forward used (stride max_dof*max_dof, LocalDofIndex layout).
//   task_target      : the SAME per-env 3-float world target the forward used
//                      (element env*3). Needed for e_x in the dL/dKp channel.
//   kp / kd          : the SAME scalar gain buffers (per-link; read at task_link).
//                      May be null (that term's gradient is then 0).
//   drive_force_limits: the SAME clamp buffer the forward used (may be null). Used
//                      ONLY to MASK saturated joint components out of g_tau (R9
//                      stop-grad). When null / limit<=0 nothing is masked.
//   grad_tau         : float[total_link_count] upstream dL/dtau, link-indexed in
//                      the forward's tau convention.
//   out_grads        : freshly populated param gradients (see OscAdjointParamGrads).
//                      The caller pre-allocates the three device buffers; this
//                      launcher overwrites them in full (every articulation slot,
//                      so a two-run memcmp is well-defined).
//
// Enqueues on the context stream; the caller synchronizes (mirrors diffsim).
// Any null required pointer (inertia_M_inv / task_target / grad_tau, or all three
// out buffers) is a no-op.
void LaunchOscAdjointGainTarget(cudaStream_t stream, int device_id,
                                articulation::ArticulationDeviceState state,
                                uint32_t max_dof,
                                uint32_t task_dim,
                                uint32_t osc_task_link,
                                const float* inertia_M_inv,
                                const float* task_target,
                                const float* kp,
                                const float* kd,
                                const float* drive_force_limits,
                                const float* grad_tau,
                                OscAdjointParamGrads out_grads);

}  // namespace nuka::diffsim
