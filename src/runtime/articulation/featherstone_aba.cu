// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- CUDA Featherstone ABA implementation
// ---------------------------------------------------------------------------

#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/articulation/featherstone_aba.cuh"

#include "math/cuda_spatial_ops.cuh"
#include "math/cuda_vec_ops.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::runtime::articulation {

namespace {

constexpr uint32_t kAbaBlockSize = 32u;
constexpr uint32_t kInvalidLink = ~0u;
constexpr float kMinDiagonal = 1.0e-6f;

struct Mat3 {
    float m[9];
};

// Small-vector and spatial (6-vector / 6x6) primitives now come from the shared
// device math libraries (math/cuda_vec_ops.cuh, math/cuda_spatial_ops.cuh).
// Bodies are bit-identical to the former local copies (the loop counters are the
// same fixed 0..6 / 0..36 integer ranges, which never touch the float
// accumulation order). Name-matched symbols are pulled in directly; symbols that
// differ only in name (Dot3->Dot, Cross3->Cross) or signature (the Transform*
// helpers take const float* X per the §1.3 decoupling; SubtractOuterProduct36
// takes the min-diagonal as a parameter) are routed via thin __forceinline__
// forwarders that keep every call site verbatim. QuatNormalizeForward maps to
// QuatNormalizeForwardRsqrt(., 1e-24f).
namespace mg = ::nuka::math::gpu;
using mg::Add;
using mg::Add6InPlace;
using mg::Copy36;
using mg::Dot6;
using mg::MakeVec3;
using mg::Mat66MulVec6;
using mg::Scale;
using mg::Zero6;
using mg::Zero36;

__forceinline__ __device__ float Dot3(math::Vec3 a, math::Vec3 b) {
    return mg::Dot(a, b);
}

__forceinline__ __device__ math::Vec3 Cross3(math::Vec3 a, math::Vec3 b) {
    return mg::Cross(a, b);
}

__forceinline__ __device__ void SubtractOuterProduct36(float* matrix,
                                                       const float* u,
                                                       float diagonal) {
    mg::SubtractOuterProduct36(matrix, u, diagonal, kMinDiagonal);
}

__device__ Mat3 Mat3Transpose(const Mat3& matrix) {
    Mat3 out;
    out.m[0] = matrix.m[0];
    out.m[1] = matrix.m[3];
    out.m[2] = matrix.m[6];
    out.m[3] = matrix.m[1];
    out.m[4] = matrix.m[4];
    out.m[5] = matrix.m[7];
    out.m[6] = matrix.m[2];
    out.m[7] = matrix.m[5];
    out.m[8] = matrix.m[8];
    return out;
}

__device__ Mat3 Mat3Mul(const Mat3& a, const Mat3& b) {
    Mat3 out;
    for (uint32_t row = 0u; row < 3u; ++row) {
        for (uint32_t col = 0u; col < 3u; ++col) {
            out.m[row * 3u + col] =
                a.m[row * 3u + 0u] * b.m[col + 0u] +
                a.m[row * 3u + 1u] * b.m[col + 3u] +
                a.m[row * 3u + 2u] * b.m[col + 6u];
        }
    }
    return out;
}

__device__ Mat3 Mat3Identity() {
    Mat3 identity{{1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f,
                   0.0f, 0.0f, 1.0f}};
    return identity;
}

__device__ Mat3 RotationFromQuat(math::Quat q) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq <= 1.0e-12f) {
        return Mat3Identity();
    }
    const float inv_norm = rsqrtf(norm_sq);
    q.w *= inv_norm;
    q.x *= inv_norm;
    q.y *= inv_norm;
    q.z *= inv_norm;

    const float xx = q.x * q.x;
    const float yy = q.y * q.y;
    const float zz = q.z * q.z;
    const float xy = q.x * q.y;
    const float xz = q.x * q.z;
    const float yz = q.y * q.z;
    const float wx = q.w * q.x;
    const float wy = q.w * q.y;
    const float wz = q.w * q.z;

    Mat3 out{{
        1.0f - 2.0f * (yy + zz),
        2.0f * (xy - wz),
        2.0f * (xz + wy),
        2.0f * (xy + wz),
        1.0f - 2.0f * (xx + zz),
        2.0f * (yz - wx),
        2.0f * (xz - wy),
        2.0f * (yz + wx),
        1.0f - 2.0f * (xx + yy)
    }};
    return out;
}

__device__ Mat3 RotationFromAxisAngle(math::Vec3 axis, float angle) {
    const float length_sq = Dot3(axis, axis);
    if (length_sq <= 1.0e-12f) {
        return Mat3Identity();
    }
    axis = Scale(axis, rsqrtf(length_sq));
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float t = 1.0f - c;
    Mat3 out{{
        t * axis.x * axis.x + c,
        t * axis.x * axis.y - s * axis.z,
        t * axis.x * axis.z + s * axis.y,
        t * axis.x * axis.y + s * axis.z,
        t * axis.y * axis.y + c,
        t * axis.y * axis.z - s * axis.x,
        t * axis.x * axis.z - s * axis.y,
        t * axis.y * axis.z + s * axis.x,
        t * axis.z * axis.z + c
    }};
    return out;
}

__device__ LinkSpatialTransform MakeMotionTransform(const Mat3& rotation,
                                                    math::Vec3 translation) {
    LinkSpatialTransform transform;
    Zero36(transform.X);
    for (uint32_t row = 0u; row < 3u; ++row) {
        for (uint32_t col = 0u; col < 3u; ++col) {
            transform.X[row * 6u + col] = rotation.m[row * 3u + col];
            transform.X[(row + 3u) * 6u + (col + 3u)] = rotation.m[row * 3u + col];
        }
    }
    const Mat3 skew{{
        0.0f, -translation.z, translation.y,
        translation.z, 0.0f, -translation.x,
        -translation.y, translation.x, 0.0f
    }};
    const Mat3 lower_left = Mat3Mul(rotation, skew);
    for (uint32_t row = 0u; row < 3u; ++row) {
        for (uint32_t col = 0u; col < 3u; ++col) {
            transform.X[(row + 3u) * 6u + col] = -lower_left.m[row * 3u + col];
        }
    }
    return transform;
}

// The Transform* helpers read only transform.X (a float[36]); the shared library
// versions take that raw pointer (§1.3 T8b decoupling). These thin forwarders
// keep every call site -- f(transform, ...) -- verbatim while routing to the
// shared, byte-identical bodies.
__forceinline__ __device__ void TransformMotion(const LinkSpatialTransform& transform,
                                                const float* in,
                                                float* out) {
    mg::TransformMotion(transform.X, in, out);
}

__forceinline__ __device__ void TransformForceTranspose(const LinkSpatialTransform& transform,
                                                        const float* in,
                                                        float* out) {
    mg::TransformForceTranspose(transform.X, in, out);
}

__forceinline__ __device__ void TransformInertiaToParent(const LinkSpatialTransform& transform,
                                                         const float* child_inertia,
                                                         float* parent_delta) {
    mg::TransformInertiaToParent(transform.X, child_inertia, parent_delta);
}

// Mat66MulVec6 / Copy36 / MotionCross / ForceCross are name- and signature-
// identical to the shared library symbols (pulled in via `using` below); the
// former local copies are removed.
using mg::ForceCross;
using mg::MotionCross;

__device__ void MotionSubspaceForJoint(ArticulationJointType type,
                                       math::Vec3 axis,
                                       float* out) {
    Zero6(out);
    const float length_sq = Dot3(axis, axis);
    if (length_sq > 1.0e-12f) {
        axis = Scale(axis, rsqrtf(length_sq));
    }
    if (type == ArticulationJointType::Revolute) {
        out[0] = axis.x;
        out[1] = axis.y;
        out[2] = axis.z;
    } else if (type == ArticulationJointType::Prismatic) {
        out[3] = axis.x;
        out[4] = axis.y;
        out[5] = axis.z;
    }
}

__device__ void GravityAcceleration(float gravity_z, float* out) {
    Zero6(out);
    out[5] = -gravity_z;
}

// T8a helpers --------------------------------------------------------------

// Rotates a world-frame vector into a body frame given the body's world
// orientation quaternion: v_body = R(q)^T v_world. Uses the same w-first
// Hamilton convention as math::Quat; v' = v + 2*(-w)*(qv x v) + 2*qv x (qv x v)
// for the conjugate rotation, written out directly to stay __device__.
__device__ math::Vec3 RotateByQuatInverse(math::Quat q, math::Vec3 v) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq > 1.0e-12f) {
        const float inv = rsqrtf(norm_sq);
        q.w *= inv;
        q.x *= inv;
        q.y *= inv;
        q.z *= inv;
    }
    // Conjugate (inverse for a unit quaternion) negates the vector part.
    const math::Vec3 qv = MakeVec3(-q.x, -q.y, -q.z);
    const math::Vec3 t = Scale(Cross3(qv, v), 2.0f);
    return Add(v, Add(Scale(t, q.w), Cross3(qv, t)));
}

// Deterministic fixed-pivot (unpivoted) 6x6 LDL^T solve of A x = b, with A SPD.
// Mirrors FactorArticulationInertiaMKernel's LDL^T + kMinDiagonal floor exactly
// (no atomics, fixed order) so determinism (D1) holds. Now routed through the
// shared mg::Solve6x6Ldlt (byte-identical body, kMinDiagonal parameterized) via
// a thin __forceinline__ forwarder that keeps the call site verbatim.
__forceinline__ __device__ void Solve6x6Ldlt(const float* matrix, const float* rhs, float* out) {
    mg::Solve6x6Ldlt(matrix, rhs, out, kMinDiagonal);
}

__device__ void CopySpatialInertia(const LinkSpatialInertia& src,
                                   LinkArticulatedInertia* dst) {
    for (uint32_t i = 0u; i < 36u; ++i) {
        dst->Ia[i] = src.I[i];
    }
}

__device__ void SpatialToArray(const LinkSpatialVel& value, float* out) {
    for (uint32_t i = 0u; i < 6u; ++i) {
        out[i] = value.v[i];
    }
}

__device__ void ArrayToSpatial(const float* in, LinkSpatialVel* out) {
    for (uint32_t i = 0u; i < 6u; ++i) {
        out->v[i] = in[i];
    }
}

__device__ void ArrayToAccel(const float* in, LinkSpatialAccel* out) {
    for (uint32_t i = 0u; i < 6u; ++i) {
        out->a[i] = in[i];
    }
}

__device__ void ArrayToForce(const float* in, LinkBiasForce* out) {
    for (uint32_t i = 0u; i < 6u; ++i) {
        out->p[i] = in[i];
    }
}

__device__ LinkSpatialTransform JointTransform(const ArticulationDeviceState& state,
                                               uint32_t link) {
    const ArticulationJointType type = state.joint_type[link];
    const math::Vec3 axis = state.joint_axis[link];
    const math::Transform local_pose = state.link_local_pose[link];
    const math::Vec3 parent_offset = state.parent_offset[link];
    Mat3 joint_rotation = Mat3Identity();
    math::Vec3 translation = Add(local_pose.position, parent_offset);
    if (type == ArticulationJointType::Revolute) {
        joint_rotation = RotationFromAxisAngle(axis, state.q[link]);
    } else if (type == ArticulationJointType::Prismatic) {
        translation = Add(parent_offset, Scale(axis, state.q[link]));
        translation = Add(local_pose.position, translation);
    }
    const Mat3 local_rotation = RotationFromQuat(local_pose.rotation);
    const Mat3 rotation = Mat3Mul(local_rotation, joint_rotation);
    return MakeMotionTransform(Mat3Transpose(rotation), translation);
}

__global__ void AbaPass1KinematicsKernel(ArticulationDeviceState state,
                                         float gravity_z) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;

        // T8a: free-floating root. Its 6-DOF spatial velocity IS state held in
        // link_velocity[root] (integrated each step), so Pass-1 must NOT recompute
        // and clobber it with the scalar-qdot formula (which would zero it). The
        // root frame is the base body frame; xup[root] is identity (no parent
        // joint), the subspace is conceptually I6 (handled by branching, not
        // stored), velocity_bias = 0, and the bias force is the pure gyroscopic
        // term p = v x* (I v) computed from the PRESERVED velocity.
        if (state.parent_link[link] == kInvalidLink &&
            state.joint_type[link] == ArticulationJointType::FloatingBase) {
            state.link_xup[link] = MakeMotionTransform(Mat3Identity(),
                                                       MakeVec3(0.0f, 0.0f, 0.0f));
            Zero6(state.joint_motion_subspace[link].s);
            CopySpatialInertia(state.link_inertia[link], &state.link_articulated_I[link]);

            float velocity[6];
            SpatialToArray(state.link_velocity[link], velocity);

            float velocity_bias[6];
            Zero6(velocity_bias);
            ArrayToSpatial(velocity_bias, &state.link_velocity_bias[link]);

            float inertia_velocity[6];
            Mat66MulVec6(state.link_inertia[link].I, velocity, inertia_velocity);
            float bias_force[6];
            ForceCross(velocity, inertia_velocity, bias_force);
            ArrayToForce(bias_force, &state.link_bias_force[link]);
            state.joint_diagonal[link] = 0.0f;
            state.joint_force[link] = 0.0f;
            Zero6(state.link_u_spatial[link].p);
            continue;
        }

        state.link_xup[link] = JointTransform(state, link);
        MotionSubspaceForJoint(state.joint_type[link],
                               state.joint_axis[link],
                               state.joint_motion_subspace[link].s);
        CopySpatialInertia(state.link_inertia[link], &state.link_articulated_I[link]);

        float v_parent[6];
        Zero6(v_parent);
        const uint32_t parent = state.parent_link[link];
        if (parent != kInvalidLink) {
            SpatialToArray(state.link_velocity[offset + parent], v_parent);
        }

        float transformed_parent[6];
        TransformMotion(state.link_xup[link], v_parent, transformed_parent);
        float joint_velocity[6];
        for (uint32_t i = 0u; i < 6u; ++i) {
            joint_velocity[i] = state.joint_motion_subspace[link].s[i] * state.qdot[link];
        }

        float velocity[6];
        for (uint32_t i = 0u; i < 6u; ++i) {
            velocity[i] = transformed_parent[i] + joint_velocity[i];
        }
        ArrayToSpatial(velocity, &state.link_velocity[link]);

        float velocity_bias[6];
        MotionCross(velocity, joint_velocity, velocity_bias);
        ArrayToSpatial(velocity_bias, &state.link_velocity_bias[link]);

        float inertia_velocity[6];
        Mat66MulVec6(state.link_inertia[link].I, velocity, inertia_velocity);
        float bias_force[6];
        ForceCross(velocity, inertia_velocity, bias_force);
        ArrayToForce(bias_force, &state.link_bias_force[link]);
        state.joint_diagonal[link] = 0.0f;
        state.joint_force[link] = 0.0f;
        Zero6(state.link_u_spatial[link].p);
    }
    (void)gravity_z;
}

__global__ void AbaPass2ArticulatedInertiaKernel(ArticulationDeviceState state) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    for (uint32_t reverse = count; reverse > 0u; --reverse) {
        const uint32_t link = offset + reverse - 1u;

        // T8a: free-floating root. Children have already accumulated their
        // reduced articulated inertia into Ia[root] and their bias force into
        // bias_force[root] (the leaf->root order guarantees the root is last).
        // The base 6x6 articulated inertia is Ia[root] itself: there is no scalar
        // joint U/D rank-1 reduction and (as for any root) no parent propagation.
        // Pass-3 solves a_base = Ia[root]^-1 (-p[root]) directly.
        if (state.parent_link[link] == kInvalidLink &&
            state.joint_type[link] == ArticulationJointType::FloatingBase) {
            Zero6(state.link_u_spatial[link].p);
            state.joint_diagonal[link] = 0.0f;
            state.joint_force[link] = 0.0f;
            continue;
        }

        float u_spatial[6];
        Mat66MulVec6(state.link_articulated_I[link].Ia,
                     state.joint_motion_subspace[link].s,
                     u_spatial);
        ArrayToForce(u_spatial, &state.link_u_spatial[link]);

        float diagonal = Dot6(state.joint_motion_subspace[link].s, u_spatial);
        if (state.joint_type[link] == ArticulationJointType::Fixed) {
            diagonal = 1.0f;
        }
        diagonal += state.joint_armature[link];
        diagonal = fmaxf(diagonal, kMinDiagonal);
        state.joint_diagonal[link] = diagonal;
        state.joint_force[link] =
            state.tau[link] -
            state.joint_damping[link] * state.qdot[link] -
            Dot6(state.joint_motion_subspace[link].s,
                 state.link_bias_force[link].p);

        const uint32_t parent = state.parent_link[link];
        if (parent == kInvalidLink) {
            continue;
        }

        float reduced_inertia[36];
        Copy36(state.link_articulated_I[link].Ia, reduced_inertia);
        if (state.joint_type[link] != ArticulationJointType::Fixed) {
            SubtractOuterProduct36(reduced_inertia, u_spatial, diagonal);
        }

        float parent_delta[36];
        TransformInertiaToParent(state.link_xup[link], reduced_inertia, parent_delta);
        const uint32_t parent_link = offset + parent;
        for (uint32_t i = 0u; i < 36u; ++i) {
            state.link_articulated_I[parent_link].Ia[i] += parent_delta[i];
        }

        float pa[6];
        for (uint32_t i = 0u; i < 6u; ++i) {
            pa[i] = state.link_bias_force[link].p[i];
        }
        float inertia_bias[6];
        Mat66MulVec6(reduced_inertia,
                     state.link_velocity_bias[link].v,
                     inertia_bias);
        Add6InPlace(pa, inertia_bias);
        if (state.joint_type[link] != ArticulationJointType::Fixed) {
            const float inv_diagonal = 1.0f / diagonal;
            for (uint32_t i = 0u; i < 6u; ++i) {
                pa[i] += u_spatial[i] * state.joint_force[link] * inv_diagonal;
            }
        }

        float parent_force[6];
        TransformForceTranspose(state.link_xup[link], pa, parent_force);
        Add6InPlace(state.link_bias_force[parent_link].p, parent_force);
    }
}

__global__ void AbaPass3AccelerationKernel(ArticulationDeviceState state,
                                           float gravity_z) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;

        // T8a: free-floating root. Treat the base as a 6-DOF joint (S = I6) with
        // the world (parent) acceleration seeded to the Featherstone gravity
        // pseudo-acceleration a_grav = GravityAcceleration(gravity_z) (the same
        // (0,0,-gravity_z) the FIXED base propagates to its children). For the
        // 6-DOF base joint:
        //   qddot_0 = (Ia)^-1 (-p_0)  -  a_grav            (base generalized accel)
        //   a_0     = a_grav + qddot_0 = (Ia)^-1 (-p_0)    (SPATIAL accel for kids)
        // so:
        //   * link_acceleration[root] stores the APPARENT spatial accel a_0 =
        //     (Ia)^-1(-p): this is what the existing child TransformMotion
        //     propagation consumes, and it is the gravity-frame accel (at rest in
        //     free fall a_0 = 0, so the legs feel zero apparent accel and stay
        //     put -- physically a free-falling robot's joints do not move).
        //   * the base VELOCITY is integrated with the REAL accel qddot_0 =
        //     a_0 - a_grav (computed in IntegrateFloatingBaseVelocityKernel, which
        //     re-derives a_grav from gravity_z + base_pose), so at rest the base
        //     falls at the real g (qddot_0 = -a_grav = (0,0,gravity_z)).
        // Gravity thus enters exactly ONCE, with the CORRECT sign for both the
        // base free-fall and the children -- verified by the free-fall test
        // (a_base ~ (0,0,gravity_z), CoM drop ~ 1/2 |g| t^2). p_0 carries ONLY the
        // velocity-product / children bias force (no gravity), so in zero-g
        // a_0 = (Ia)^-1(-p) is pure momentum coupling.
        if (state.parent_link[link] == kInvalidLink &&
            state.joint_type[link] == ArticulationJointType::FloatingBase) {
            float neg_p[6];
            for (uint32_t i = 0u; i < 6u; ++i) {
                neg_p[i] = -state.link_bias_force[link].p[i];
            }
            float a_free[6];
            Zero6(a_free);
            Solve6x6Ldlt(state.link_articulated_I[link].Ia, neg_p, a_free);
            // a_0 (apparent spatial accel) = (Ia)^-1(-p), NO gravity seed: children
            // propagate from this, and the velocity integrator subtracts a_grav.
            ArrayToAccel(a_free, &state.link_acceleration[link]);
            state.qddot[link] = 0.0f;
            (void)gravity_z;
            continue;
        }

        float parent_accel[6];
        const uint32_t parent = state.parent_link[link];
        if (parent != kInvalidLink) {
            TransformMotion(state.link_xup[link],
                            state.link_acceleration[offset + parent].a,
                            parent_accel);
        } else {
            GravityAcceleration(gravity_z, parent_accel);
        }

        float accel[6];
        for (uint32_t i = 0u; i < 6u; ++i) {
            accel[i] = parent_accel[i] + state.link_velocity_bias[link].v[i];
        }

        if (state.joint_type[link] == ArticulationJointType::Fixed) {
            state.qddot[link] = 0.0f;
        } else {
            const float u_dot_a = Dot6(state.link_u_spatial[link].p, accel);
            state.qddot[link] =
                (state.joint_force[link] - u_dot_a) /
                fmaxf(state.joint_diagonal[link], kMinDiagonal);
            for (uint32_t i = 0u; i < 6u; ++i) {
                accel[i] += state.joint_motion_subspace[link].s[i] * state.qddot[link];
            }
        }
        ArrayToAccel(accel, &state.link_acceleration[link]);
    }
}

__global__ void IntegrateArticulationKernel(ArticulationDeviceState state, float dt) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count) {
        return;
    }
    state.qdot[link] += state.qddot[link] * dt;
    state.q[link] += state.qdot[link] * dt;
}

// Split halves of IntegrateArticulationKernel. The arithmetic of each half is
// textually identical to the combined kernel above, so running velocity then
// position is bit-for-bit equal to the combined Integrate (the batched stepper
// needs the contact solve to sit between the two halves). The single-env path
// keeps using the combined kernel; these are additive.
__global__ void IntegrateVelocityArticulationKernel(ArticulationDeviceState state,
                                                    float dt) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count) {
        return;
    }
    state.qdot[link] += state.qddot[link] * dt;
}

__global__ void IntegratePositionArticulationKernel(ArticulationDeviceState state,
                                                    float dt) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count) {
        return;
    }
    state.q[link] += state.qdot[link] * dt;
}

// T8a: forward quaternion rotation v' = R(q) v (w-first Hamilton), __device__.
__device__ math::Vec3 RotateByQuatForward(math::Quat q, math::Vec3 v) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq > 1.0e-12f) {
        const float inv = rsqrtf(norm_sq);
        q.w *= inv;
        q.x *= inv;
        q.y *= inv;
        q.z *= inv;
    }
    const math::Vec3 qv = MakeVec3(q.x, q.y, q.z);
    const math::Vec3 t = Scale(Cross3(qv, v), 2.0f);
    return Add(v, Add(Scale(t, q.w), Cross3(qv, t)));
}

// T8a: Hamilton quaternion product (w-first), __device__. Now routed through the
// shared mg::QuatMul: the four per-component expressions are character-identical
// (MakeQuat assigns its args to the same fields this body did), so the float DAG
// is byte-identical. The forwarder keeps the call site verbatim.
__forceinline__ __device__ math::Quat QuatMulForward(math::Quat a, math::Quat b) {
    return mg::QuatMul(a, b);
}

// QuatNormalizeForward maps to the shared QuatNormalizeForwardRsqrt with eps
// 1e-24f (norm_sq, strict `<`, rsqrtf in-place). Body is byte-identical; the
// forwarder keeps the call site verbatim.
__forceinline__ __device__ math::Quat QuatNormalizeForward(math::Quat q) {
    return mg::QuatNormalizeForwardRsqrt(q, 1.0e-24f);
}

// T8a: floating-base velocity update. One block per articulation, lane 0.
// link_acceleration[root] holds the APPARENT (gravity-frame) spatial accel a_0 =
// (Ia)^-1(-p) that Pass-3 stored for child propagation. The REAL base
// generalized acceleration is qddot_0 = a_0 - a_grav, where a_grav is the
// Featherstone gravity pseudo-acceleration (0,0,-gravity_z) rotated into the
// base body frame. We integrate the REAL accel so the base falls at the correct
// signed g (at rest a_0 = 0 => qddot_0 = -a_grav = (0,0,gravity_z) in world).
// Non-floating roots are left untouched (their scalar DOFs integrate elsewhere).
__global__ void IntegrateFloatingBaseVelocityKernel(ArticulationDeviceState state,
                                                    float dt,
                                                    float gravity_z) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }
    const uint32_t root = state.articulation_link_offset[articulation];
    if (state.parent_link[root] != kInvalidLink ||
        state.joint_type[root] != ArticulationJointType::FloatingBase) {
        return;
    }
    // Body-frame gravity pseudo-accel a_grav: world (0,0,-gravity_z) on the linear
    // axes, rotated by the inverse base orientation. Angular part is 0.
    float g_world_arr[6];
    GravityAcceleration(gravity_z, g_world_arr);
    const math::Quat base_rot = state.base_pose[articulation].rotation;
    const math::Vec3 g_body =
        RotateByQuatInverse(base_rot, MakeVec3(g_world_arr[3], g_world_arr[4],
                                               g_world_arr[5]));
    float a_grav[6] = {0.0f, 0.0f, 0.0f, g_body.x, g_body.y, g_body.z};
    for (uint32_t i = 0u; i < 6u; ++i) {
        const float real_accel = state.link_acceleration[root].a[i] - a_grav[i];
        state.link_velocity[root].v[i] += real_accel * dt;
    }
}

// T8a: floating-base pose update. One block per articulation, lane 0. The base
// spatial velocity is body-frame [omega(0:3), v_lin(3:6)]:
//   position += R(base_rot) * v_lin * dt   (rotate body-frame linear vel to world)
//   base_rot  = normalize(base_rot (x) dq), dq = (1, 0.5*omega*dt)  (first order)
// Stable first-order quaternion integration with renormalization each step.
__global__ void IntegrateFloatingBasePoseKernel(ArticulationDeviceState state,
                                                float dt) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }
    const uint32_t root = state.articulation_link_offset[articulation];
    if (state.parent_link[root] != kInvalidLink ||
        state.joint_type[root] != ArticulationJointType::FloatingBase) {
        return;
    }
    const float* v = state.link_velocity[root].v;
    math::Transform pose = state.base_pose[articulation];

    const math::Vec3 v_lin_body = MakeVec3(v[3], v[4], v[5]);
    const math::Vec3 v_lin_world = RotateByQuatForward(pose.rotation, v_lin_body);
    pose.position = Add(pose.position, Scale(v_lin_world, dt));

    math::Quat dq;
    dq.w = 1.0f;
    dq.x = 0.5f * v[0] * dt;
    dq.y = 0.5f * v[1] * dt;
    dq.z = 0.5f * v[2] * dt;
    pose.rotation = QuatNormalizeForward(QuatMulForward(pose.rotation, dq));

    state.base_pose[articulation] = pose;
}

__global__ void ApplyPositionDriveKernel(ArticulationDeviceState state,
                                         const float* drive_targets,
                                         const float* drive_stiffness,
                                         const float* drive_damping,
                                         const float* drive_force_limits,
                                         bool defer_velocity_damping) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count ||
        state.joint_type[link] == ArticulationJointType::Fixed) {
        return;
    }
    // tau = Kp*(target-q) - Kd*qdot. When `defer_velocity_damping` is set the
    // -Kd*qdot term is OMITTED here and instead applied IMPLICITLY downstream in
    // the constrained-velocity solve (SolveArticulatedContactRows seeds qdot with
    // -dt*(M+dt*C)^-1*(C*qdot), C the per-DOF joint-damping diag). Explicit -Kd*qdot
    // integrated by the semi-implicit Euler step is only conditionally stable
    // (instability ~ dt*Kd/m_eff, blows up at coarse dt / contact-shrunk m_eff);
    // the deferred implicit form is unconditionally stable. The batched contact
    // stepper sets this true; the single-env oracle path leaves it false so its
    // trajectory (and the go2_stand_5s golden) is byte-for-byte unchanged.
    float tau = drive_stiffness[link] * (drive_targets[link] - state.q[link]);
    if (!defer_velocity_damping) {
        tau -= drive_damping[link] * state.qdot[link];
    }
    const float limit = drive_force_limits[link];
    if (limit > 0.0f) {
        tau = fminf(fmaxf(tau, -limit), limit);
    }
    state.tau[link] = tau;
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

} // namespace

void LaunchFeatherstoneAbaKernels(cudaStream_t stream, int device_id,
                                  ArticulationDeviceState state,
                                  float gravity_z) {
    if (state.articulation_count == 0u || state.total_link_count == 0u) {
        return;
    }

    phi::ScopedDeviceGuard guard(device_id);
    dim3 grid(state.articulation_count);
    dim3 block(kAbaBlockSize);

    AbaPass1KinematicsKernel<<<grid, block, 0u, stream>>>(state, gravity_z);
    CheckCuda(cudaGetLastError(), "AbaPass1KinematicsKernel launch");
    AbaPass2ArticulatedInertiaKernel<<<grid, block, 0u, stream>>>(state);
    CheckCuda(cudaGetLastError(), "AbaPass2ArticulatedInertiaKernel launch");
    AbaPass3AccelerationKernel<<<grid, block, 0u, stream>>>(state, gravity_z);
    CheckCuda(cudaGetLastError(), "AbaPass3AccelerationKernel launch");
}

void LaunchApplyPositionDriveKernels(cudaStream_t stream, int device_id,
                                     ArticulationDeviceState state,
                                     const float* drive_targets,
                                     const float* drive_stiffness,
                                     const float* drive_damping,
                                     const float* drive_force_limits,
                                     bool defer_velocity_damping) {
    if (state.total_link_count == 0u ||
        drive_targets == nullptr ||
        drive_stiffness == nullptr ||
        drive_damping == nullptr ||
        drive_force_limits == nullptr) {
        return;
    }

    phi::ScopedDeviceGuard guard(device_id);
    const uint32_t blocks = (state.total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
    ApplyPositionDriveKernel<<<blocks, kAbaBlockSize, 0u, stream>>>(
        state,
        drive_targets,
        drive_stiffness,
        drive_damping,
        drive_force_limits,
        defer_velocity_damping);
    CheckCuda(cudaGetLastError(), "ApplyPositionDriveKernel launch");
}

void LaunchIntegrateArticulationKernels(cudaStream_t stream, int device_id,
                                        ArticulationDeviceState state,
                                        float dt) {
    if (state.total_link_count == 0u || dt <= 0.0f) {
        return;
    }

    phi::ScopedDeviceGuard guard(device_id);
    const uint32_t blocks = (state.total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
    IntegrateArticulationKernel<<<blocks, kAbaBlockSize, 0u, stream>>>(state, dt);
    CheckCuda(cudaGetLastError(), "IntegrateArticulationKernel launch");
}

void LaunchIntegrateVelocityArticulationKernels(cudaStream_t stream, int device_id,
                                                ArticulationDeviceState state,
                                                float dt) {
    if (state.total_link_count == 0u || dt <= 0.0f) {
        return;
    }

    phi::ScopedDeviceGuard guard(device_id);
    const uint32_t blocks = (state.total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
    IntegrateVelocityArticulationKernel<<<blocks, kAbaBlockSize, 0u, stream>>>(state, dt);
    CheckCuda(cudaGetLastError(), "IntegrateVelocityArticulationKernel launch");
}

void LaunchIntegratePositionArticulationKernels(cudaStream_t stream, int device_id,
                                                ArticulationDeviceState state,
                                                float dt) {
    if (state.total_link_count == 0u || dt <= 0.0f) {
        return;
    }

    phi::ScopedDeviceGuard guard(device_id);
    const uint32_t blocks = (state.total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
    IntegratePositionArticulationKernel<<<blocks, kAbaBlockSize, 0u, stream>>>(state, dt);
    CheckCuda(cudaGetLastError(), "IntegratePositionArticulationKernel launch");
}

// T8a: floating-base integrators are per-articulation (lane-0) kernels like the
// ABA Pass kernels, so the grid is articulation_count -- NOT total_link_count.
// Both kernels early-return for non-floating roots, so launching them
// unconditionally leaves the fixed-base path byte-for-byte untouched.
void LaunchIntegrateFloatingBaseVelocityKernels(cudaStream_t stream, int device_id,
                                                ArticulationDeviceState state,
                                                float dt,
                                                float gravity_z) {
    if (state.articulation_count == 0u || dt <= 0.0f) {
        return;
    }

    phi::ScopedDeviceGuard guard(device_id);
    IntegrateFloatingBaseVelocityKernel<<<state.articulation_count, kAbaBlockSize, 0u, stream>>>(
        state, dt, gravity_z);
    CheckCuda(cudaGetLastError(), "IntegrateFloatingBaseVelocityKernel launch");
}

void LaunchIntegrateFloatingBasePoseKernels(cudaStream_t stream, int device_id,
                                            ArticulationDeviceState state,
                                            float dt) {
    if (state.articulation_count == 0u || dt <= 0.0f) {
        return;
    }

    phi::ScopedDeviceGuard guard(device_id);
    IntegrateFloatingBasePoseKernel<<<state.articulation_count, kAbaBlockSize, 0u, stream>>>(
        state, dt);
    CheckCuda(cudaGetLastError(), "IntegrateFloatingBasePoseKernel launch");
}

void FeatherstoneAba::ApplyPositionDrives(cudaStream_t stream, int device_id,
                                          ArticulationDeviceState state,
                                          const float* drive_targets,
                                          const float* drive_stiffness,
                                          const float* drive_damping,
                                          const float* drive_force_limits,
                                          bool defer_velocity_damping) {
    LaunchApplyPositionDriveKernels(stream, device_id,
                                    state,
                                    drive_targets,
                                    drive_stiffness,
                                    drive_damping,
                                    drive_force_limits,
                                    defer_velocity_damping);
}

void FeatherstoneAba::ComputeAccelerations(cudaStream_t stream, int device_id,
                                           ArticulationDeviceState state,
                                           float gravity_z) {
    LaunchFeatherstoneAbaKernels(stream, device_id, state, gravity_z);
}

void FeatherstoneAba::Integrate(cudaStream_t stream, int device_id,
                                ArticulationDeviceState state,
                                float dt) {
    LaunchIntegrateArticulationKernels(stream, device_id, state, dt);
}

void FeatherstoneAba::IntegrateVelocity(cudaStream_t stream, int device_id,
                                        ArticulationDeviceState state,
                                        float dt) {
    LaunchIntegrateVelocityArticulationKernels(stream, device_id, state, dt);
}

void FeatherstoneAba::IntegratePosition(cudaStream_t stream, int device_id,
                                        ArticulationDeviceState state,
                                        float dt) {
    LaunchIntegratePositionArticulationKernels(stream, device_id, state, dt);
}

void FeatherstoneAba::IntegrateFloatingBaseVelocity(cudaStream_t stream, int device_id,
                                                    ArticulationDeviceState state,
                                                    float dt,
                                                    float gravity_z) {
    LaunchIntegrateFloatingBaseVelocityKernels(stream, device_id, state, dt, gravity_z);
}

void FeatherstoneAba::IntegrateFloatingBasePose(cudaStream_t stream, int device_id,
                                                ArticulationDeviceState state,
                                                float dt) {
    LaunchIntegrateFloatingBasePoseKernels(stream, device_id, state, dt);
}

} // namespace nuka::runtime::articulation
