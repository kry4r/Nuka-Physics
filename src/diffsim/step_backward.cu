// ---------------------------------------------------------------------------
// nuka::diffsim -- single-step reverse-mode adjoint (contact-free PD path)
// ---------------------------------------------------------------------------
//
// Hand-written reverse of one forward step:  drive -> ABA(3-pass) -> integrate.
// ONE thread per articulation, mirroring the forward kernels' single-lane
// per-articulation loops. The reverse runs the passes in reverse order:
//
//   integrate-reverse -> pass3-reverse -> pass2-reverse -> pass1-reverse
//                        -> drive-reverse
//
// Every cross-link adjoint (a child feeding a parent's bias/inertia in forward
// pass 2; a parent feeding a child's acceleration in forward pass 3) is
// accumulated in fixed local-link order with plain `+=` -- NO float atomics
// (correctness + D1 + lint, by construction of the single-thread design).
//
// The forward intermediates (link_xup, joint_motion_subspace, link_articulated_I,
// joint_diagonal, link_u_spatial, joint_force, link_velocity, link_velocity_bias,
// link_bias_force, link_acceleration, qddot) persist in the device buffers; the
// reverse reads them. The PRE-step q / qdot are snapshotted by the caller.
//
// FIXED-BASE (revolute / fixed joints) is fully implemented incl. the q-channel
// (the link_xup = JointTransform(q) path). FLOATING-BASE root reverse is
// implemented for the VELOCITY channel: the 6x6 LDLT-solve adjoint of
// a_free = Ia^-1(-p) (STAGE B-float) + the root gyroscopic pass-1 reverse
// (p = v x* (I v), STAGE D-float) + the floating-base linear velocity integrator
// reverse (v_root' = v_root + (a_free - a_grav)dt, STAGE A-float), seeded through
// grad_link_velocity_out. NOT yet differentiated: the base ORIENTATION channel --
// the q-of-base path through a_grav = R(base_rot)^T g and the quaternion POSE
// integrator (base_pose is held fixed; the supported loss seeds are on q'/qdot'
// (joints) and v_root' (root spatial velocity), not on base_pose'). The prismatic
// joint q-channel is also not wired (revolute-only STAGE E).
// ---------------------------------------------------------------------------

#include "diffsim/step_backward.hpp"
#include "diffsim/aba_reverse.cuh"

#include "codegen/generated/maximal_drive_adjoint.cuh"
#include "math/cuda_spatial_ops.cuh"
#include "math/cuda_vec_ops.cuh"
#include "runtime/articulation/articulation_state.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::diffsim {

namespace {

namespace mg = ::nuka::math::gpu;
using articulation::ArticulationDeviceState;
using articulation::ArticulationJointType;

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kMinDiagonal = 1.0e-6f;
constexpr uint32_t kMaxRevLinks = 24u;   // covers go2 (13) + base; bound for scratch

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// Reproduce JointTransform's rotation (R = local_rot * joint_rot) and translation
// exactly enough to recompute the inputs the X-adjoint needs. We only need this
// for the q-channel (revolute), where the q-dependence is through
// RotationFromAxisAngle(axis, q). We recompute the same Mat3 the forward built.
struct Mat3R { float m[9]; };

__device__ Mat3R Mat3IdentityR() {
    return Mat3R{{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
}

// Deterministic fixed-pivot 6x6 LDL^T solve A x = b (A SPD). BYTE-IDENTICAL to
// the forward Solve6x6Ldlt in featherstone_aba.cu (same kMinDiagonal floor, same
// fixed loop order) so the floating-base reverse stays D1. Used for the adjoint
// of a_free = Ia^-1 (-p): since Ia is SPD-symmetric, grad_b = Ia^-1 grad_a_free
// is the SAME solve on the seed. Now routed through the shared mg::Solve6x6Ldlt
// (byte-identical body, kMinDiagonal parameterized) via a thin forwarder.
__forceinline__ __device__ void Solve6x6LdltRev(const float* matrix, const float* rhs, float* out) {
    mg::Solve6x6Ldlt(matrix, rhs, out, kMinDiagonal);
}

__device__ Mat3R RotationFromQuatR(math::Quat q) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq <= 1.0e-12f) return Mat3IdentityR();
    const float inv_norm = rsqrtf(norm_sq);
    q.w *= inv_norm; q.x *= inv_norm; q.y *= inv_norm; q.z *= inv_norm;
    const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    return Mat3R{{1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy),
                  2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx),
                  2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy)}};
}

// RotationFromAxisAngle and its derivative w.r.t. angle (axis fixed, normalized).
__device__ Mat3R RotationFromAxisAngleR(math::Vec3 axis, float angle, Mat3R* d_dangle) {
    const float length_sq = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
    if (length_sq <= 1.0e-12f) {
        if (d_dangle) *d_dangle = Mat3R{{0,0,0,0,0,0,0,0,0}};
        return Mat3IdentityR();
    }
    const float inv = rsqrtf(length_sq);
    axis.x *= inv; axis.y *= inv; axis.z *= inv;
    const float c = cosf(angle), s = sinf(angle), t = 1.0f - c;
    const float ax = axis.x, ay = axis.y, az = axis.z;
    Mat3R R{{t*ax*ax + c,    t*ax*ay - s*az, t*ax*az + s*ay,
             t*ax*ay + s*az, t*ay*ay + c,    t*ay*az - s*ax,
             t*ax*az - s*ay, t*ay*az + s*ax, t*az*az + c}};
    if (d_dangle) {
        // dt/dangle = s, dc/dangle = -s, ds/dangle = c.
        const float dt = s, dc = -s, ds = c;
        *d_dangle = Mat3R{{dt*ax*ax + dc,     dt*ax*ay - ds*az, dt*ax*az + ds*ay,
                           dt*ax*ay + ds*az,  dt*ay*ay + dc,    dt*ay*az - ds*ax,
                           dt*ax*az - ds*ay,  dt*ay*az + ds*ax, dt*az*az + dc}};
    }
    return R;
}

// MakeMotionTransform forward (rotation already TRANSPOSED by JointTransform):
// X[0:3,0:3] = R, X[3:6,3:6] = R, X[3:6,0:3] = -(R * skew(translation)).
// We need grad_R from grad_X (translation held fixed -- it is q-independent for
// revolute). skew(t) = [[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]].
__device__ void MakeMotionTransform_RevRot(const float* grad_X,
                                            const math::Vec3 translation,
                                            float* grad_R /*9*/) {
    for (uint32_t i = 0u; i < 9u; ++i) grad_R[i] = 0.0f;
    // upper-left + lower-right blocks: X[r,c]=R[r,c], X[r+3,c+3]=R[r,c]
    for (uint32_t r = 0u; r < 3u; ++r) {
        for (uint32_t c = 0u; c < 3u; ++c) {
            grad_R[r * 3u + c] += grad_X[r * 6u + c];
            grad_R[r * 3u + c] += grad_X[(r + 3u) * 6u + (c + 3u)];
        }
    }
    // lower-left block: X[r+3,c] = -(R*skew)[r,c] = -sum_k R[r,k]*skew[k,c]
    // grad_R[r,k] += -grad_X[r+3,c]*skew[k,c]  (sum over c)
    const float skew[9] = {0.0f, -translation.z, translation.y,
                           translation.z, 0.0f, -translation.x,
                           -translation.y, translation.x, 0.0f};
    for (uint32_t r = 0u; r < 3u; ++r) {
        for (uint32_t k = 0u; k < 3u; ++k) {
            float s = 0.0f;
            for (uint32_t c = 0u; c < 3u; ++c) {
                s += grad_X[(r + 3u) * 6u + c] * skew[k * 3u + c];
            }
            grad_R[r * 3u + k] += -s;
        }
    }
}

// ---------------------------------------------------------------------------
// The single-articulation reverse kernel. One thread (blockIdx.x ==
// articulation, lane 0). enable_q_channel: also back-propagate through the
// link_xup = JointTransform(q) configuration path (rung-(a) d/d(q) X-path).
// ---------------------------------------------------------------------------
__global__ void StepBackwardKernel(ArticulationDeviceState state,
                                   StepBackwardInputs in,
                                   StepBackwardGrads g,
                                   bool enable_q_channel) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) return;

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    if (count > kMaxRevLinks) return;  // defensive; tests stay within bound

    const float dt = in.dt;

    // --- Per-link adjoint scratch (local index 0..count-1). ----------------
    float grad_accel[kMaxRevLinks][6];      // dL/d link_acceleration[link]
    float grad_p[kMaxRevLinks][6];          // dL/d link_bias_force[link] (post pass2)
    float grad_Ia[kMaxRevLinks][36];        // dL/d link_articulated_I[link] (post pass2)
    float grad_v[kMaxRevLinks][6];          // dL/d link_velocity[link]
    float grad_vbias[kMaxRevLinks][6];      // dL/d link_velocity_bias[link]
    float grad_jf[kMaxRevLinks];            // dL/d joint_force[link]
    float grad_qddot_dof[kMaxRevLinks];     // dL/d qddot[link]
    float grad_X[kMaxRevLinks][36];         // dL/d link_xup[link] (q-channel)
    // Floating-base root: dL/d a_free (the LDLT-solved root spatial accel). Seeded
    // by the velocity-integrator reverse (STAGE A-float) and accumulated by the
    // children's pass-3 reverse; consumed by the root LDLT reverse (STAGE B-float).
    float grad_a_free_root[6];
    Zero6Rev(grad_a_free_root);

    for (uint32_t i = 0u; i < count; ++i) {
        Zero6Rev(grad_accel[i]);
        Zero6Rev(grad_p[i]);
        Zero36Rev(grad_Ia[i]);
        Zero6Rev(grad_v[i]);
        Zero6Rev(grad_vbias[i]);
        grad_jf[i] = 0.0f;
        grad_qddot_dof[i] = 0.0f;
        Zero36Rev(grad_X[i]);
    }

    // =======================================================================
    // STAGE A: reverse the integrators (per DOF). Seed grad_q'/grad_qdot'.
    //   forward: qdot' = qdot + qddot*dt ; q' = q + qdot'*dt
    //   grad_qddot += dt*grad_qdot' + dt*dt*grad_q'
    //   grad_qdot(pre) = grad_qdot' + dt*grad_q'
    //   grad_q(pre)    = grad_q'  (+ drive path, below)
    // The caller seeded g.grad_q_out = dL/dq', g.grad_qdot_out = dL/dqdot'. We
    // read those, produce grad_qddot_dof + the pre-step grad_q/grad_qdot, and
    // OVERWRITE g.grad_q_out/g.grad_qdot_out with the pre-step values.
    // =======================================================================
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (in.has_integrate) {
            const float gq_post = g.grad_q_out[link];
            const float gqd_post = g.grad_qdot_out[link];
            grad_qddot_dof[local] += dt * gqd_post + dt * dt * gq_post;
            // pre-step grad_qdot and grad_q (q-pre also gets a drive contribution
            // later); store provisional pre-step values back.
            g.grad_qdot_out[link] = gqd_post + dt * gq_post;
            g.grad_q_out[link] = gq_post;  // pre-step grad_q (drive adds below)
        } else {
            // Output is qddot itself: seed the qddot DOF adjoint directly. Leave
            // grad_q_out / grad_qdot_out at the caller's value (the velocity /
            // drive reverses add their pre-step contributions below).
            grad_qddot_dof[local] +=
                in.grad_qddot_seed ? in.grad_qddot_seed[link] : 0.0f;
        }
    }

    // --- STAGE A-float: reverse the floating-base velocity integrator. -------
    // forward: v_root'[i] = v_root[i] + (a_free[i] - a_grav[i])*dt
    // (a_grav depends on base_pose.rotation -- held fixed; base orientation is NOT
    // differentiated, the supported seed is dL/d v_root').  Seed:
    // g.grad_link_velocity_out[root] = dL/d v_root'.
    //   grad_a_free_root += dt * grad_vroot'      (integrate case)
    //   grad_v_root(pre)  = grad_vroot'           (write back to the same slot)
    // When !has_integrate the floating "output" is the root spatial accel a_free
    // itself (the floating analogue of seeding qddot): seed grad_a_free directly.
    {
        const uint32_t root = offset;  // local 0
        if (state.joint_type[root] == ArticulationJointType::FloatingBase &&
            g.grad_link_velocity_out != nullptr) {
            float* gvr = g.grad_link_velocity_out + static_cast<size_t>(root) * 6u;
            if (in.has_integrate) {
                for (uint32_t i = 0u; i < 6u; ++i) {
                    grad_a_free_root[i] += dt * gvr[i];
                    // grad_v_root(pre) = grad_vroot' (identity); stays in gvr[i].
                }
            } else {
                for (uint32_t i = 0u; i < 6u; ++i) {
                    grad_a_free_root[i] += gvr[i];  // seed dL/d a_free
                    gvr[i] = 0.0f;                  // no pre-step velocity passthrough
                }
            }
        }
    }

    // =======================================================================
    // STAGE B: reverse ABA Pass 3 (forward was root->leaf; reverse leaf->root).
    //   forward (non-root, non-fixed):
    //     parent_accel = X[link] * a[parent]   (or gravity for root parent)
    //     accel0 = parent_accel + vbias[link]
    //     qddot = (jf - u . accel0) / D
    //     accel = accel0 + s * qddot
    //     a[link] = accel
    //   Adjoint (seed: grad_accel[local] holds dL/d a[link], plus the qddot DOF
    //   seed grad_qddot_dof[local] from STAGE A):
    //     from accel = accel0 + s*qddot:
    //        grad_accel0 += grad_accel ; grad_qddot += s . grad_accel
    //        grad_s += qddot * grad_accel   (q-channel: s is q-independent here)
    //     from qddot = (jf - u.accel0)/D:
    //        c = grad_qddot / D
    //        grad_jf += c ; grad_u += -c*accel0 ; grad_accel0 += -c*u
    //        grad_D  += -qddot * c            (D depends on Ia,s -> pass2)
    //     from accel0 = parent_accel + vbias:
    //        grad_vbias[local] += grad_accel0 ; grad_parent_accel = grad_accel0
    //     from parent_accel = X*a[parent]:
    //        grad_accel[parent] += X^T grad_parent_accel
    //        grad_X[local] += grad_parent_accel * a[parent]^T   (q-channel)
    // =======================================================================
    float grad_u[kMaxRevLinks][6];
    float grad_D[kMaxRevLinks];
    for (uint32_t i = 0u; i < count; ++i) { Zero6Rev(grad_u[i]); grad_D[i] = 0.0f; }

    for (uint32_t rev = count; rev > 0u; --rev) {
        const uint32_t local = rev - 1u;
        const uint32_t link = offset + local;
        const ArticulationJointType jt = state.joint_type[link];
        const uint32_t parent = state.parent_link[link];

        // --- STAGE B-float: floating-base root pass-3 reverse (LDLT solve). ---
        // forward: a_free = Ia^-1 (-p) = Solve6x6Ldlt(Ia, -p);
        //          link_acceleration[root] = a_free.
        // The children's pass-3 reverse has ALREADY added X^T grad_parent_accel
        // into grad_accel[local==0] (leaf->root order guarantees the root is last);
        // grad_a_free_root carries the velocity-integrator seed. Total cotangent of
        // a_free = grad_accel[0] + grad_a_free_root.
        //   Let b = -p, a_free = Ia^-1 b. Ia SPD-symmetric:
        //     grad_b   = Ia^-1 grad_a_free        (same Solve6x6Ldlt on the seed)
        //     grad_p   = -grad_b
        //     grad_Ia += -grad_b * a_free^T       (outer product)
        if (parent == kInvalidLink && jt == ArticulationJointType::FloatingBase) {
            float grad_a_free[6];
            for (uint32_t i = 0u; i < 6u; ++i)
                grad_a_free[i] = grad_accel[local][i] + grad_a_free_root[i];
            float grad_b[6];
            Zero6Rev(grad_b);
            Solve6x6LdltRev(state.link_articulated_I[link].Ia, grad_a_free, grad_b);
            const float* a_free = state.link_acceleration[link].a;
            for (uint32_t i = 0u; i < 6u; ++i) grad_p[local][i] += -grad_b[i];
            for (uint32_t r = 0u; r < 6u; ++r)
                for (uint32_t c = 0u; c < 6u; ++c)
                    grad_Ia[local][r * 6u + c] += -grad_b[r] * a_free[c];
            continue;
        }

        // grad of accel (== a[link]) already in grad_accel[local].
        float g_accel[6];
        for (uint32_t i = 0u; i < 6u; ++i) g_accel[i] = grad_accel[local][i];

        // Recompute forward accel0 (parent_accel + vbias). We need accel0, u,
        // qddot, D, jf -- all persisted (qddot[link], joint_diagonal, joint_force,
        // link_u_spatial, link_velocity_bias) or recomputable from link_xup.
        float parent_accel[6];
        if (parent != kInvalidLink) {
            mg::TransformMotion(state.link_xup[link].X,
                                state.link_acceleration[offset + parent].a, parent_accel);
        } else {
            Zero6Rev(parent_accel);
            parent_accel[5] = -in.gravity_z;  // GravityAcceleration
        }
        float accel0[6];
        for (uint32_t i = 0u; i < 6u; ++i)
            accel0[i] = parent_accel[i] + state.link_velocity_bias[link].v[i];

        float g_accel0[6];
        for (uint32_t i = 0u; i < 6u; ++i) g_accel0[i] = 0.0f;
        float g_qddot = grad_qddot_dof[local];

        if (jt == ArticulationJointType::Fixed) {
            // qddot=0 forced; accel = accel0 (no s*qddot term).
            for (uint32_t i = 0u; i < 6u; ++i) g_accel0[i] += g_accel[i];
        } else {
            const float D = fmaxf(state.joint_diagonal[link], kMinDiagonal);
            const float qddot = state.qddot[link];
            const float* s = state.joint_motion_subspace[link].s;
            const float* u = state.link_u_spatial[link].p;
            // accel = accel0 + s*qddot
            for (uint32_t i = 0u; i < 6u; ++i) {
                g_accel0[i] += g_accel[i];
                g_qddot += s[i] * g_accel[i];
            }
            // qddot = (jf - u.accel0)/D
            const float c = g_qddot / D;
            grad_jf[local] += c;
            for (uint32_t i = 0u; i < 6u; ++i) {
                grad_u[local][i] += -c * accel0[i];
                g_accel0[i] += -c * u[i];
            }
            grad_D[local] += -qddot * c;
        }

        // accel0 = parent_accel + vbias
        for (uint32_t i = 0u; i < 6u; ++i) grad_vbias[local][i] += g_accel0[i];
        // parent_accel = X * a[parent]  (or gravity const -> no grad)
        if (parent != kInvalidLink) {
            TransformMotion_RevIn(state.link_xup[link].X, g_accel0,
                                  grad_accel[parent]);
            if (enable_q_channel) {
                TransformMotion_RevX(g_accel0,
                                     state.link_acceleration[offset + parent].a,
                                     grad_X[local]);
            }
        }
    }

    // =======================================================================
    // STAGE C: reverse ABA Pass 2 (forward leaf->root; reverse root->leaf).
    // Forward per link (non-root, non-fixed):
    //   u = Ia * s
    //   D = s.u (+armature); jf = tau - damping*qdot - s.p
    //   reduced = Ia - (u u^T)/D
    //   Ia[parent] += X^T reduced X
    //   pa = p + reduced*vbias + (u*jf)/D
    //   p[parent] += X^T pa
    // The adjoints grad_Ia[link], grad_p[link] hold the cotangent of the link's
    // articulated inertia / bias force AS LEFT after pass2 (i.e. AFTER children
    // added into it). For the parent, the contributions from THIS child are added
    // here in reverse root->leaf order. grad_u/grad_D/grad_jf seeded by pass3.
    // =======================================================================
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        const ArticulationJointType jt = state.joint_type[link];
        const uint32_t parent = state.parent_link[link];

        if (parent == kInvalidLink && jt == ArticulationJointType::FloatingBase) {
            continue;  // root inertia/bias adjoint resolved in STAGE B-float
        }

        const float* s = state.joint_motion_subspace[link].s;
        const float* Ia = state.link_articulated_I[link].Ia;
        const float* u = state.link_u_spatial[link].p;
        const float D = fmaxf(state.joint_diagonal[link], kMinDiagonal);
        const float inv_D = 1.0f / D;

        // ---- Recompute reduced inertia (forward) for the reverse. ----
        float reduced[36];
        mg::Copy36(Ia, reduced);
        if (jt != ArticulationJointType::Fixed) {
            mg::SubtractOuterProduct36(reduced, u, D, kMinDiagonal);
        }

        float g_reduced[36];
        Zero36Rev(g_reduced);
        float g_pa[6];
        Zero6Rev(g_pa);

        // ---- (reverse of) p[parent] += X^T pa ----
        if (parent != kInvalidLink) {
            // forward: p[parent] += X^T * pa  -> grad_pa += X * grad_p[parent]
            TransformForceTranspose_RevIn(state.link_xup[link].X, grad_p[parent],
                                          g_pa);
            // q-channel: grad_X from this transform (operand "pa").
            // computed after pa is reconstructed below.
        }

        // ---- reconstruct pa to back-prop its constituents ----
        // pa = p + reduced*vbias + u*jf/D
        const float* p_link = state.link_bias_force[link].p;  // p AFTER children
        const float* vbias = state.link_velocity_bias[link].v;
        const float jf = state.joint_force[link];
        // pa = p + reduced*vbias + u*(jf/D)
        // grad_p[link] += g_pa (identity); grad over reduced*vbias and u*jf/D:
        for (uint32_t i = 0u; i < 6u; ++i) grad_p[local][i] += g_pa[i];
        // reduced*vbias term:
        Mat66MulVec6_RevVec(reduced, g_pa, grad_vbias[local]);   // grad_vbias
        Mat66MulVec6_RevMat(g_pa, vbias, g_reduced);             // grad_reduced
        // u*(jf/D) term:
        if (jt != ArticulationJointType::Fixed) {
            const float jf_over_D = jf * inv_D;
            for (uint32_t i = 0u; i < 6u; ++i) grad_u[local][i] += g_pa[i] * jf_over_D;
            float g_jfD = 0.0f;  // grad of (jf/D)
            for (uint32_t i = 0u; i < 6u; ++i) g_jfD += g_pa[i] * u[i];
            grad_jf[local] += g_jfD * inv_D;
            grad_D[local] += -g_jfD * jf * inv_D * inv_D;
        }

        // q-channel: grad_X from forward p[parent] += X^T * pa (operand pa).
        if (parent != kInvalidLink && enable_q_channel) {
            // recompute pa forward value for the RevX operand.
            float pa[6];
            for (uint32_t i = 0u; i < 6u; ++i) pa[i] = p_link[i];
            float inertia_bias[6];
            mg::Mat66MulVec6(reduced, vbias, inertia_bias);
            for (uint32_t i = 0u; i < 6u; ++i) pa[i] += inertia_bias[i];
            if (jt != ArticulationJointType::Fixed) {
                for (uint32_t i = 0u; i < 6u; ++i) pa[i] += u[i] * jf * inv_D;
            }
            TransformForceTranspose_RevX(pa, grad_p[parent], grad_X[local]);
        }

        // ---- (reverse of) Ia[parent] += X^T reduced X ----
        if (parent != kInvalidLink) {
            TransformInertiaToParent_RevC(state.link_xup[link].X, grad_Ia[parent],
                                          g_reduced);
            if (enable_q_channel) {
                TransformInertiaToParent_RevX(state.link_xup[link].X, reduced,
                                              grad_Ia[parent], grad_X[local]);
            }
        }

        // ---- reduced = Ia - (u u^T)/D  ----
        // grad_Ia[link] += g_reduced (passthrough), plus the outer-product u/D rev.
        for (uint32_t i = 0u; i < 36u; ++i) grad_Ia[local][i] += g_reduced[i];
        if (jt != ArticulationJointType::Fixed) {
            // M_out = M_in - u u^T * inv_D. grad_u and grad_inv_d:
            SubtractOuterProduct36_RevU(g_reduced, u, inv_D, grad_u[local]);
            const float g_inv_d = SubtractOuterProduct36_RevInvD(g_reduced, u);
            // inv_D = 1/D, D = joint_diagonal (>= eps). d inv_D/dD = -inv_D^2.
            grad_D[local] += g_inv_d * (-inv_D * inv_D);
        }

        // ---- jf = tau - damping*qdot - s.p  ----
        // grad_tau += grad_jf ; grad_qdot += -damping*grad_jf ; grad_p += -s*grad_jf
        // grad_tau is STAGED in g.grad_tau_out (caller pre-zeroed it); STAGE G
        // reads it as the drive seed dL/dtau and converts to the input grads.
        const float g_jf = grad_jf[local];
        if (jt != ArticulationJointType::Fixed) {
            g.grad_tau_out[link] += g_jf;  // accumulate dL/dtau
        }
        g.grad_qdot_out[link] += -state.joint_damping[link] * g_jf;
        for (uint32_t i = 0u; i < 6u; ++i) {
            grad_p[local][i] += -s[i] * g_jf;
        }

        // ---- D = s.u (+armature, +floor). grad_D -> grad_u (and grad_s, q-chan)
        // D_pre = s.u; armature is constant; the fmax floor is inactive in tests
        // (D > eps), so dD/dD_pre = 1.
        if (jt != ArticulationJointType::Fixed) {
            const float gD = grad_D[local];
            for (uint32_t i = 0u; i < 6u; ++i) grad_u[local][i] += gD * s[i];
            // grad_s += gD*u  (s is q-independent: axis only -> no q-channel here)
        }
        // (Fixed joint forces D=1 const; grad_D discarded.)

        // ---- u = Ia * s  ----
        // grad_Ia[link] += grad_u * s^T ; grad_s += Ia^T grad_u (s q-independent).
        Mat66MulVec6_RevMat(grad_u[local], s, grad_Ia[local]);
    }

    // =======================================================================
    // STAGE D: reverse ABA Pass 1 (forward root->leaf; reverse leaf->root).
    // Forward per link (non-root): copy I->Ia ; v = X*v_parent + s*qdot ;
    //   vbias = v x* (s*qdot) ; p = v x* (I v).
    // Adjoints to resolve: grad_Ia[link] -> grad over link_inertia (mass!) AND
    // the I-in-bias-force path; grad_vbias, grad_v -> qdot, v_parent (-> grad_v
    // [parent], q-channel via X), and link_inertia again (p = v x* (I v)).
    // =======================================================================
    float grad_I[kMaxRevLinks][36];
    for (uint32_t i = 0u; i < count; ++i) Zero36Rev(grad_I[i]);

    for (uint32_t rev = count; rev > 0u; --rev) {
        const uint32_t local = rev - 1u;
        const uint32_t link = offset + local;
        const ArticulationJointType jt = state.joint_type[link];
        const uint32_t parent = state.parent_link[link];

        // --- Floating-base root pass-1 reverse. ---
        // forward (root): Ia = copy(I) ; velocity_bias = 0 ; p = ForceCross(v, I*v)
        // where v = link_velocity[root] (the STATE -- no parent/joint term). So:
        //   grad_I  += grad_Ia (copy)  +  the I-in-(I*v) gyroscopic path
        //   grad_v(root) += the ForceCross + I^T paths -> grad_link_velocity_out[root]
        if (parent == kInvalidLink && jt == ArticulationJointType::FloatingBase) {
            const float* I = state.link_inertia[link].I;
            // The root gyroscopic bias p[root] = ForceCross(v, I*v) is QUADRATIC in
            // v, so its adjoint AND the grad_I[root]->grad_mass[root] outer-product
            // path must linearize at the PRE-integration root velocity v_pre. The
            // forward's IntegrateFloatingBaseVelocity overwrote link_velocity[root]
            // in place (v_pre -> v_post) BEFORE StepBackward runs, so we read the
            // caller's v_pre snapshot (in.v_root_pre[link*6 .. +6)) rather than the
            // clobbered live state. Fall back to the live state only when no
            // snapshot was provided (fixed-base models never hit this branch).
            const float* v = (in.v_root_pre != nullptr)
                                 ? (in.v_root_pre + static_cast<size_t>(link) * 6u)
                                 : state.link_velocity[link].v;
            // copy I -> Ia
            for (uint32_t i = 0u; i < 36u; ++i) grad_I[local][i] += grad_Ia[local][i];
            // p = ForceCross(v, I*v)
            float w[6];
            mg::Mat66MulVec6(I, v, w);
            float gv_from_p[6], gw[6];
            Zero6Rev(gv_from_p); Zero6Rev(gw);
            ForceCross_Rev(v, w, grad_p[local], gv_from_p, gw);
            Mat66MulVec6_RevMat(gw, v, grad_I[local]);
            Mat66MulVec6_RevVec(I, gw, grad_v[local]);
            for (uint32_t i = 0u; i < 6u; ++i) grad_v[local][i] += gv_from_p[i];
            // v is the root STATE -> route grad_v(root) to grad_link_velocity_out.
            if (g.grad_link_velocity_out != nullptr) {
                float* gvr = g.grad_link_velocity_out + static_cast<size_t>(link) * 6u;
                for (uint32_t i = 0u; i < 6u; ++i) gvr[i] += grad_v[local][i];
            }
            continue;
        }

        const float* I = state.link_inertia[link].I;
        const float* v = state.link_velocity[link].v;
        const float* s = state.joint_motion_subspace[link].s;
        const float qd = in.qdot_pre[link];

        // ---- copy I -> Ia : grad_I += grad_Ia ----
        for (uint32_t i = 0u; i < 36u; ++i) grad_I[local][i] += grad_Ia[local][i];

        // ---- p = v x* (I v)  =  ForceCross(v, I*v) ----
        // Let w = I*v. p = ForceCross(v, w). grad_p[link] is the cotangent.
        // ForceCross is bilinear: reverse gives grad_v (from the motion arg) and
        // grad_w (from the force arg); then grad_w -> grad_I (outer) + grad_v.
        {
            float w[6];
            mg::Mat66MulVec6(I, v, w);
            float gv_from_p[6], gw[6];
            Zero6Rev(gv_from_p); Zero6Rev(gw);
            // ForceCross adjoint (see helper below).
            ForceCross_Rev(v, w, grad_p[local], gv_from_p, gw);
            // w = I*v : grad_I += gw * v^T ; grad_v += I^T gw
            Mat66MulVec6_RevMat(gw, v, grad_I[local]);
            Mat66MulVec6_RevVec(I, gw, grad_v[local]);
            for (uint32_t i = 0u; i < 6u; ++i) grad_v[local][i] += gv_from_p[i];
        }

        // ---- vbias = v x* (s*qdot) = MotionCross(v, jv), jv = s*qdot ----
        {
            float jv[6];
            for (uint32_t i = 0u; i < 6u; ++i) jv[i] = s[i] * qd;
            float gv_from_vb[6], gjv[6];
            Zero6Rev(gv_from_vb); Zero6Rev(gjv);
            MotionCross_Rev(v, jv, grad_vbias[local], gv_from_vb, gjv);
            for (uint32_t i = 0u; i < 6u; ++i) grad_v[local][i] += gv_from_vb[i];
            // jv = s*qdot : grad_qdot += s.gjv
            float gqd = 0.0f;
            for (uint32_t i = 0u; i < 6u; ++i) gqd += s[i] * gjv[i];
            g.grad_qdot_out[link] += gqd;
        }

        // ---- v = transformed_parent + s*qdot, transformed_parent = X*v_parent --
        // grad_transformed_parent = grad_v ; grad_qdot += s . grad_v
        {
            float g_tp[6];
            for (uint32_t i = 0u; i < 6u; ++i) g_tp[i] = grad_v[local][i];
            float gqd = 0.0f;
            for (uint32_t i = 0u; i < 6u; ++i) gqd += s[i] * grad_v[local][i];
            g.grad_qdot_out[link] += gqd;
            if (parent != kInvalidLink) {
                float v_parent[6];
                for (uint32_t i = 0u; i < 6u; ++i)
                    v_parent[i] = state.link_velocity[offset + parent].v[i];
                TransformMotion_RevIn(state.link_xup[link].X, g_tp,
                                      grad_v[parent]);
                if (enable_q_channel) {
                    TransformMotion_RevX(g_tp, v_parent, grad_X[local]);
                }
            }
        }
    }

    // =======================================================================
    // STAGE E: q-channel -- grad_X[link] -> grad_q. (revolute only; prismatic /
    // floating excluded from the q-channel here.) X = MakeMotionTransform(
    // transpose(R), translation), R = local_rot * joint_rot(q). For revolute the
    // q-dependence is joint_rot = RotationFromAxisAngle(axis, q); translation is
    // q-independent. d X / d q chains: grad_X -> grad_Rt(transposed rot) ->
    // grad_R -> grad(joint_rot) -> grad_q.
    // =======================================================================
    if (enable_q_channel) {
        for (uint32_t local = 0u; local < count; ++local) {
            const uint32_t link = offset + local;
            const ArticulationJointType jt = state.joint_type[link];
            if (jt != ArticulationJointType::Revolute) continue;
            const uint32_t parent = state.parent_link[link];
            (void)parent;

            const math::Vec3 axis = state.joint_axis[link];
            const float q = in.q_pre[link];
            const math::Transform local_pose = state.link_local_pose[link];
            const math::Vec3 parent_offset = state.parent_offset[link];
            const math::Vec3 translation = mg::Add(local_pose.position, parent_offset);

            // Forward: Rt = transpose(local_rot * joint_rot). MakeMotionTransform
            // received Rt. grad_X -> grad_Rt.
            float grad_Rt[9];
            MakeMotionTransform_RevRot(grad_X[local], translation, grad_Rt);
            // Rt = transpose(R): grad_R[i,j] = grad_Rt[j,i].
            float grad_R[9];
            for (uint32_t r = 0u; r < 3u; ++r)
                for (uint32_t c = 0u; c < 3u; ++c)
                    grad_R[r * 3u + c] = grad_Rt[c * 3u + r];
            // R = local_rot * joint_rot. grad_joint_rot = local_rot^T * grad_R.
            Mat3R local_rot = RotationFromQuatR(local_pose.rotation);
            float grad_jr[9];
            for (uint32_t r = 0u; r < 3u; ++r) {
                for (uint32_t c = 0u; c < 3u; ++c) {
                    float ssum = 0.0f;
                    for (uint32_t k = 0u; k < 3u; ++k)
                        ssum += local_rot.m[k * 3u + r] * grad_R[k * 3u + c];
                    grad_jr[r * 3u + c] = ssum;
                }
            }
            // joint_rot = RotationFromAxisAngle(axis, q). grad_q = <d jr/dq, grad_jr>.
            Mat3R d_dangle;
            RotationFromAxisAngleR(axis, q, &d_dangle);
            float gq = 0.0f;
            for (uint32_t i = 0u; i < 9u; ++i) gq += d_dangle.m[i] * grad_jr[i];
            g.grad_q_out[link] += gq;
        }
    }

    // =======================================================================
    // STAGE F: contract grad_I -> grad_mass (per link). grad_mass = <grad_I,
    // dI/dmass>. dI_dmass is the representation-consistent slope (parallel-axis +
    // COM coupling included), built host-side from MakeSpatialInertia.
    // =======================================================================
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        const float* dIdm = in.dI_dmass + static_cast<size_t>(link) * 36u;
        float gm = 0.0f;
        for (uint32_t i = 0u; i < 36u; ++i) gm += grad_I[local][i] * dIdm[i];
        g.grad_mass_out[link] = gm;
    }

    // =======================================================================
    // STAGE G: reverse the drive (only when has_drive). tau = clamp(Kp*(target-q)
    // - Kd*qdot). grad_tau was accumulated (STAGE C) into g.grad_tau_out; convert
    // it to the drive input grads (REUSING the committed p01 maximal_drive
    // adjoint). When !has_drive, tau is a direct ABA input: g.grad_tau_out is the
    // d/d(tau) deliverable and grad_target_out is left zero.
    // =======================================================================
    if (in.has_drive) {
        for (uint32_t local = 0u; local < count; ++local) {
            const uint32_t link = offset + local;
            if (state.joint_type[link] == ArticulationJointType::Fixed) {
                g.grad_target_out[link] = 0.0f;
                continue;
            }
            const float seed = g.grad_tau_out[link];  // grad_tau staged here
            nuka::solver::generated::MaximalDriveAdjInputs di;
            di.q = in.q_pre[link];
            di.qdot = in.qdot_pre[link];
            di.drive_target = in.drive_targets ? in.drive_targets[link] : 0.0f;
            di.drive_stiffness = in.drive_stiffness ? in.drive_stiffness[link] : 0.0f;
            di.drive_damping = in.drive_damping ? in.drive_damping[link] : 0.0f;
            di.force_limit = in.drive_force_limits ? in.drive_force_limits[link] : 0.0f;
            const auto gg =
                nuka::solver::generated::maximal_drive_adjoint_eval(di, seed);
            // grad_target is the action gradient.
            g.grad_target_out[link] = gg.grad_drive_target;
            // grad_q / grad_qdot pick up the drive contributions.
            g.grad_q_out[link] += gg.grad_q;
            g.grad_qdot_out[link] += gg.grad_qdot;
        }
    }
}

}  // namespace

void StepBackward(const phi::DeviceContext& context,
                  ArticulationDeviceState state,
                  const StepBackwardInputs& inputs,
                  const StepBackwardGrads& grads) {
    if (state.articulation_count == 0u) return;
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    // ONE thread per articulation (lane 0); grid over articulations.
    StepBackwardKernel<<<state.articulation_count, 32u, 0u, stream>>>(
        state, inputs, grads, inputs.enable_q_channel);
    CheckCuda(cudaGetLastError(), "StepBackwardKernel launch");
}

}  // namespace nuka::diffsim
