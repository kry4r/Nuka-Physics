// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M3b articulation dynamics ops:
//   ApplyDrives / AbaForward / IntegrateVelocity / FkWorldPoses /
//   IntegratePosition
//
// KERNEL BODIES ARE LINE-BY-LINE PORTS (D1 byte-exact contract) of
//   src/runtime/articulation/featherstone_aba.cu      (drives / ABA / integrate)
//   src/runtime/articulation/articulation_contacts.cu (UpdateWorldLinkPosesKernel)
// The ONLY change is the input wiring: pointers come from ModelView/DataView
// (via nkops::MakeArticulationDeviceState) instead of ArticulationDeviceBuffers,
// and launches go through phi::LaunchCuda on the dispatch/capture stream.
// Floating-point operation order, fma patterns, block/grid shapes and launch
// geometry are UNCHANGED. The legacy files stay alive (production) until M9.
//
// FkWorldPoses note: the legacy pipeline computes world poses into a separate
// world_pose_ scratch buffer and then D2D-copies it over state.link_pose. The
// FK kernel never READS link_pose (it reads q/topology/base_pose and its own
// output for parents), so writing straight into the link_pose field produces
// byte-identical link_pose/world-pose content with one buffer and no copy.
//
// NO allocation in this TU (lint hot_path_cuda_malloc covers ops/**); all
// staging is init-time (Model::UploadTo / Arena).
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "math/cuda_spatial_ops.cuh"
#include "math/cuda_vec_ops.cuh"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

constexpr uint32_t kAbaBlockSize = 32u;
constexpr uint32_t kInvalidLink = ~0u;
constexpr float kMinDiagonal = 1.0e-6f;

struct Mat3 {
    float m[9];
};

// Small-vector and spatial (6-vector / 6x6) primitives come from the shared
// device math libraries (math/cuda_vec_ops.cuh, math/cuda_spatial_ops.cuh) —
// the SAME bodies the legacy featherstone_aba.cu consumes. The thin
// __forceinline__ forwarders below are copied verbatim from that file so every
// call site stays textually identical.
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

        // T8a: free-floating root (see featherstone_aba.cu for the full note).
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

        // T8a: free-floating root (see featherstone_aba.cu).
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

        // T8a: free-floating root (see featherstone_aba.cu for the gravity
        // bookkeeping rationale).
        if (state.parent_link[link] == kInvalidLink &&
            state.joint_type[link] == ArticulationJointType::FloatingBase) {
            float neg_p[6];
            for (uint32_t i = 0u; i < 6u; ++i) {
                neg_p[i] = -state.link_bias_force[link].p[i];
            }
            float a_free[6];
            Zero6(a_free);
            Solve6x6Ldlt(state.link_articulated_I[link].Ia, neg_p, a_free);
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

__forceinline__ __device__ math::Quat QuatMulForward(math::Quat a, math::Quat b) {
    return mg::QuatMul(a, b);
}

__forceinline__ __device__ math::Quat QuatNormalizeForward(math::Quat q) {
    return mg::QuatNormalizeForwardRsqrt(q, 1.0e-24f);
}

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
    // tau = Kp*(target-q) - Kd*qdot; the -Kd*qdot term is deferred to the
    // implicit joint-damping solve when defer_velocity_damping is set (see
    // featherstone_aba.cu for the stability note).
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

// M4: the union world's torque drive — LINE-BY-LINE port of
// articulation_drives.cu ApplyTorqueDriveKernel (tau = clamp(torque_input,
// +/-limit); no control Kd — physical joint damping runs downstream). The
// torque input is the drive_target field (the union action surface).
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

// M4: movable rigid-body arms (the union world's cup path, ports of
// BatchedUnifiedWorld's gravity kick + IntegrateBodyPosition).

// linear_velocity.z += g*dt for movable bodies — the legacy per-body kick.
__global__ void BodyGravityKickKernel(math::Vec3* body_linear_velocity,
                                      const float* body_inv_mass,
                                      uint32_t total_body_count,
                                      float gravity_z,
                                      float dt) {
    const uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= total_body_count) {
        return;
    }
    if (body_inv_mass[body] <= 0.0f) {
        return;
    }
    body_linear_velocity[body].z += gravity_z * dt;
}

// Symplectic-Euler body position step — BYTE-FAITHFUL port of
// BatchedUnifiedWorld::IntegrateBodyPosition (position += v*dt; the small-angle
// quaternion advance dq = {1, 0.5*w*dt}; orientation = (q*dq).Normalized() with
// the host Normalized expression: n = sqrt(.), <1e-12 -> identity, else q/n).
__global__ void BodyIntegratePositionKernel(math::Transform* body_pose,
                                            const math::Vec3* body_linear_velocity,
                                            const math::Vec3* body_angular_velocity,
                                            const float* body_inv_mass,
                                            uint32_t total_body_count,
                                            float dt) {
    const uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= total_body_count) {
        return;
    }
    if (body_inv_mass[body] <= 0.0f) {
        return;
    }
    math::Transform pose = body_pose[body];
    const math::Vec3 v = body_linear_velocity[body];
    pose.position.x += v.x * dt;
    pose.position.y += v.y * dt;
    pose.position.z += v.z * dt;
    const math::Vec3 w = body_angular_velocity[body];
    math::Quat dq;
    dq.w = 1.0f;
    dq.x = 0.5f * w.x * dt;
    dq.y = 0.5f * w.y * dt;
    dq.z = 0.5f * w.z * dt;
    // Hamilton product pose.rotation * dq — the host constexpr operator*'s
    // exact expression (the operator itself is host-only constexpr).
    const math::Quat lq = pose.rotation;
    math::Quat q;
    q.w = lq.w * dq.w - lq.x * dq.x - lq.y * dq.y - lq.z * dq.z;
    q.x = lq.w * dq.x + lq.x * dq.w + lq.y * dq.z - lq.z * dq.y;
    q.y = lq.w * dq.y - lq.x * dq.z + lq.y * dq.w + lq.z * dq.x;
    q.z = lq.w * dq.z + lq.x * dq.y - lq.y * dq.x + lq.z * dq.w;
    const float n = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (n < 1e-12f) {
        q.w = 1.0f; q.x = 0.0f; q.y = 0.0f; q.z = 0.0f;
    } else {
        q.w /= n; q.x /= n; q.y /= n; q.z /= n;
    }
    pose.rotation = q;
    body_pose[body] = pose;
}

// --- FkWorldPoses (port of articulation_contacts.cu UpdateWorldLinkPoses) ---

__device__ math::Vec3 AddVec(math::Vec3 a, math::Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ math::Vec3 ScaleVec(math::Vec3 v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

using mg::MakeQuat;
using mg::QuatIdentity;
using mg::QuatMul;

__forceinline__ __device__ math::Quat QuatNormalize(math::Quat q) {
    return mg::QuatNormalizeRsqrt(q, 1.0e-12f);
}

__device__ math::Quat QuatFromAxisAngle(math::Vec3 axis, float angle) {
    const float length_sq = Dot3(axis, axis);
    if (length_sq <= 1.0e-12f) {
        return QuatIdentity();
    }
    const math::Vec3 a = ScaleVec(axis, rsqrtf(length_sq));
    const float half = angle * 0.5f;
    const float s = sinf(half);
    const float c = cosf(half);
    return MakeQuat(c, a.x * s, a.y * s, a.z * s);
}

__device__ math::Vec3 RotateByQuat(math::Quat q, math::Vec3 v) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq > 1.0e-12f) {
        const float inv_norm = rsqrtf(norm_sq);
        q.w *= inv_norm;
        q.x *= inv_norm;
        q.y *= inv_norm;
        q.z *= inv_norm;
    }
    const math::Vec3 qv{q.x, q.y, q.z};
    const math::Vec3 t = ScaleVec(Cross3(qv, v), 2.0f);
    const math::Vec3 wt = ScaleVec(t, q.w);
    const math::Vec3 qvt = Cross3(qv, t);
    return AddVec(AddVec(v, wt), qvt);
}

__device__ math::Transform ComposeTransform(const math::Transform& lhs,
                                            const math::Transform& rhs) {
    math::Transform out;
    out.position = AddVec(lhs.position, RotateByQuat(lhs.rotation, rhs.position));
    out.rotation = QuatNormalize(QuatMul(lhs.rotation, rhs.rotation));
    return out;
}

__device__ math::Transform RelativeTransform(const ArticulationDeviceState& state,
                                             uint32_t link) {
    const ArticulationJointType type = state.joint_type[link];
    const math::Vec3 axis = state.joint_axis[link];
    const math::Transform local_pose = state.link_local_pose[link];
    const math::Vec3 parent_offset = state.parent_offset[link];

    math::Transform relative;
    relative.position = AddVec(local_pose.position, parent_offset);
    relative.rotation = local_pose.rotation;
    if (type == ArticulationJointType::Revolute) {
        relative.rotation =
            QuatNormalize(QuatMul(local_pose.rotation,
                                  QuatFromAxisAngle(axis, state.q[link])));
    } else if (type == ArticulationJointType::Prismatic) {
        relative.position =
            AddVec(relative.position, ScaleVec(axis, state.q[link]));
    }
    return relative;
}

__global__ void UpdateWorldLinkPosesKernel(ArticulationDeviceState state,
                                           math::Transform* out_world_pose) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        const math::Transform relative = RelativeTransform(state, link);
        const uint32_t parent_local = state.parent_link[link];
        if (parent_local == kInvalidLink) {
            if (state.joint_type[link] == ArticulationJointType::FloatingBase) {
                out_world_pose[link] = state.base_pose[articulation];
            } else {
                // Root: world = identity o relative = relative.
                out_world_pose[link] = relative;
            }
        } else {
            out_world_pose[link] =
                ComposeTransform(out_world_pose[offset + parent_local], relative);
        }
    }
}

// --- op entry points ---------------------------------------------------------

Status LaunchOk(cudaStream_t /*stream*/) {
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpApplyDrives(const ModelView& model, const DataView& data,
                     const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ApplyDrivesParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->total_link_count == 0u) {
        return Status::Ok;
    }
    const ArticulationDeviceState state =
        MakeArticulationDeviceState(model, data, p->total_link_count, 0u);
    const uint32_t blocks = (p->total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
    if (p->mode == 1u) {
        // M4 union path: direct torque drive (drive_target carries the torque).
        LaunchCuda(ApplyTorqueDriveKernel, dim3(blocks), dim3(kAbaBlockSize), 0u,
                   stream, state,
                   static_cast<const float*>(data.drive_target),
                   static_cast<const float*>(data.drive_force_limit));
        return LaunchOk(stream);
    }
    LaunchCuda(ApplyPositionDriveKernel, dim3(blocks), dim3(kAbaBlockSize), 0u, stream,
               state,
               static_cast<const float*>(data.drive_target),
               static_cast<const float*>(data.drive_stiffness),
               static_cast<const float*>(data.drive_damping),
               static_cast<const float*>(data.drive_force_limit),
               p->defer_velocity_damping != 0u);
    return LaunchOk(stream);
}

Status OpAbaForward(const ModelView& model, const DataView& data,
                    const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const AbaForwardParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->articulation_count == 0u || p->total_link_count == 0u) {
        return Status::Ok;
    }
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, p->total_link_count, p->articulation_count);
    const float gravity_z = p->gravity[2];
    dim3 grid(p->articulation_count);
    dim3 block(kAbaBlockSize);
    LaunchCuda(AbaPass1KinematicsKernel, grid, block, 0u, stream, state, gravity_z);
    LaunchCuda(AbaPass2ArticulatedInertiaKernel, grid, block, 0u, stream, state);
    LaunchCuda(AbaPass3AccelerationKernel, grid, block, 0u, stream, state, gravity_z);
    return LaunchOk(stream);
}

Status OpIntegrateVelocity(const ModelView& model, const DataView& data,
                           const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const IntegrateVelocityParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->dt <= 0.0f) {
        return Status::Ok;
    }
    if (p->total_link_count > 0u) {
        const ArticulationDeviceState state = MakeArticulationDeviceState(
            model, data, p->total_link_count, p->articulation_count);
        const uint32_t blocks =
            (p->total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
        // Joint velocity first, then the floating-base velocity — the production
        // BatchedArticulatedWorld order (the two write DISJOINT state: qdot vs
        // link_velocity[root], so the single-env FB-first order is byte-equal).
        LaunchCuda(IntegrateVelocityArticulationKernel, dim3(blocks),
                   dim3(kAbaBlockSize), 0u, stream, state, p->dt);
        if (p->articulation_count > 0u) {
            LaunchCuda(IntegrateFloatingBaseVelocityKernel, dim3(p->articulation_count),
                       dim3(kAbaBlockSize), 0u, stream, state, p->dt, p->gravity_z);
        }
    }
    // M4: the movable rigid-body gravity kick (the union world's cup kick —
    // applied exactly once per step, before the contact solve).
    if (p->total_body_count > 0u) {
        const uint32_t blocks =
            (p->total_body_count + kAbaBlockSize - 1u) / kAbaBlockSize;
        LaunchCuda(BodyGravityKickKernel, dim3(blocks), dim3(kAbaBlockSize), 0u,
                   stream, data.body_linear_velocity,
                   static_cast<const float*>(data.body_inv_mass),
                   p->total_body_count, p->gravity_z, p->dt);
    }
    return LaunchOk(stream);
}

Status OpFkWorldPoses(const ModelView& model, const DataView& data,
                      const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const FkWorldPosesParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->articulation_count == 0u || p->total_link_count == 0u) {
        return Status::Ok;
    }
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, p->total_link_count, p->articulation_count);
    // out_world_pose = the link_pose field itself: the kernel never reads
    // link_pose (header note), so this is byte-identical to the legacy
    // world_pose_-then-copy and saves the D2D copy.
    LaunchCuda(UpdateWorldLinkPosesKernel, dim3(p->articulation_count), dim3(32u),
               0u, stream, state, data.link_pose);
    return LaunchOk(stream);
}

Status OpIntegratePosition(const ModelView& model, const DataView& data,
                           const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const IntegratePositionParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->dt <= 0.0f) {
        return Status::Ok;
    }
    if (p->total_link_count > 0u) {
        const ArticulationDeviceState state = MakeArticulationDeviceState(
            model, data, p->total_link_count, p->articulation_count);
        const uint32_t blocks =
            (p->total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
        LaunchCuda(IntegratePositionArticulationKernel, dim3(blocks),
                   dim3(kAbaBlockSize), 0u, stream, state, p->dt);
        if (p->articulation_count > 0u) {
            LaunchCuda(IntegrateFloatingBasePoseKernel, dim3(p->articulation_count),
                       dim3(kAbaBlockSize), 0u, stream, state, p->dt);
        }
    }
    // M4: the movable rigid-body symplectic-Euler position step (the union
    // world's IntegrateBodyPosition — runs AFTER the contact solve).
    if (p->total_body_count > 0u) {
        const uint32_t blocks =
            (p->total_body_count + kAbaBlockSize - 1u) / kAbaBlockSize;
        LaunchCuda(BodyIntegratePositionKernel, dim3(blocks), dim3(kAbaBlockSize), 0u,
                   stream, data.body_pose, data.body_linear_velocity,
                   data.body_angular_velocity,
                   static_cast<const float*>(data.body_inv_mass),
                   p->total_body_count, p->dt);
    }
    return LaunchOk(stream);
}

} // namespace

void RegisterNkAbaOps() {
    SetCudaOp(NkOp::ApplyDrives, &OpApplyDrives);
    SetCudaOp(NkOp::AbaForward, &OpAbaForward);
    SetCudaOp(NkOp::IntegrateVelocity, &OpIntegrateVelocity);
    SetCudaOp(NkOp::FkWorldPoses, &OpFkWorldPoses);
    SetCudaOp(NkOp::IntegratePosition, &OpIntegratePosition);
}

// Umbrella: the explicit static-lib-safe registration of the whole M3b
// articulation pipeline, called from RegisterCudaBackendEntry(). Idempotent
// (SetCudaOp last-wins; re-registering the same fns is a no-op).
void RegisterNkArticulationPipelineOps() {
    RegisterNkAbaOps();
    RegisterNkCrbaOps();
    RegisterNkContactsFootOps();
    RegisterNkAssembleRowsOps();
    RegisterNkSolveRowsOps();
    RegisterNkReadoutOps();
}

} // namespace nuka::phi
