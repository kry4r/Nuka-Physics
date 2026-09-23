// Articulation dynamics and bounded control share the model/data arenas.
// Controller mass response and contact impulses use the same physical inertia.

#include <cuda_runtime.h>

#include <cfloat>

#include "math/cuda_spatial_ops.cuh"
#include "math/cuda_vec_ops.cuh"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/kinematics.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/rigid_types.cuh"

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

__device__ void GravityAcceleration(math::Vec3 gravity, float* out) {
    Zero6(out);
    out[3] = -gravity.x;
    out[4] = -gravity.y;
    out[5] = -gravity.z;
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
        translation = JointRelativeFrame(type, axis, local_pose, parent_offset, state.q[link]).position;
    }
    const Mat3 local_rotation = RotationFromQuat(local_pose.rotation);
    const Mat3 rotation = Mat3Mul(local_rotation, joint_rotation);
    return MakeMotionTransform(Mat3Transpose(rotation), translation);
}

__global__ void AbaPass1KinematicsKernel(ArticulationDeviceState state) {
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
}

__global__ void FkLinkVelocitiesKernel(ArticulationDeviceState state) {
    const uint32_t articulation = blockIdx.x;
    if (articulation >= state.articulation_count || threadIdx.x != 0u) return;
    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        const uint32_t parent = state.parent_link[link];
        if (parent == kInvalidLink && state.joint_type[link] == ArticulationJointType::FloatingBase)
            continue;
        float parent_velocity[6] = {};
        if (parent != kInvalidLink) SpatialToArray(state.link_velocity[offset + parent], parent_velocity);
        float velocity[6];
        TransformMotion(JointTransform(state, link), parent_velocity, velocity);
        float subspace[6];
        MotionSubspaceForJoint(state.joint_type[link], state.joint_axis[link], subspace);
        for (uint32_t component = 0u; component < 6u; ++component)
            velocity[component] += subspace[component] * state.qdot[link];
        ArrayToSpatial(velocity, &state.link_velocity[link]);
    }
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
                                           math::Vec3 gravity) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;

        // Uniform gravity is added to floating-root velocity after this bias solve.
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
            continue;
        }

        float parent_accel[6];
        const uint32_t parent = state.parent_link[link];
        if (parent != kInvalidLink) {
            TransformMotion(state.link_xup[link],
                            state.link_acceleration[offset + parent].a,
                            parent_accel);
        } else {
            float world_accel[6];
            GravityAcceleration(gravity, world_accel);
            TransformMotion(state.link_xup[link], world_accel, parent_accel);
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
                                                    const float* __restrict__ qdot_pseudo,
                                                    float dt) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count) {
        return;
    }
    // Split-impulse: advance by (real+pseudo)*dt; the persisted qdot stays = real.
    // The null branch is the EXACT velocity-only expression (byte-identical; no
    // +0.0f that could flip a signed-zero).
    if (qdot_pseudo != nullptr) {
        state.q[link] += (state.qdot[link] + qdot_pseudo[link]) * dt;
    } else {
        state.q[link] += state.qdot[link] * dt;
    }
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
                                                    math::Vec3 gravity) {
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
    GravityAcceleration(gravity, g_world_arr);
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
                                                const LinkSpatialVel* __restrict__ link_velocity_pseudo,
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
    const float* rv = state.link_velocity[root].v;
    math::Transform pose = state.base_pose[articulation];

    // Split-impulse: the root pose integrates from (real+pseudo) root velocity.
    // The null branch is the EXACT velocity-only expression (byte-identical; no
    // +0.0f that could flip a signed-zero).
    float v[6];
    if (link_velocity_pseudo != nullptr) {
        const float* pv = link_velocity_pseudo[root].v;
        for (uint32_t i = 0u; i < 6u; ++i) v[i] = rv[i] + pv[i];
    } else {
        for (uint32_t i = 0u; i < 6u; ++i) v[i] = rv[i];
    }

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

struct DriveControlView {
    const float* target;
    const float* velocity_target;
    const float* acceleration_target;
    const float* stiffness;
    const float* damping;
    const float* force_limit;
    const float* noload_speed;
    const float* feedforward;
    const math::Vec3* task_target;
    const math::Quat* task_rotation;
    const math::Transform* task_frame;
    const float* null_stiffness;
    const float* null_damping;
    const float* inverse_mass;
    float* command;
    float* dissipation;
    float* lower;
    float* upper;
    float* mass;
    float* factor;
    float* jacobian;
    float* response;
    float* task_map;
    uint32_t* status;
};

DriveControlView MakeDriveControlView(const DataView& data) {
    return {data.drive_target, data.velocity_target, data.acceleration_target,
            data.drive_stiffness, data.drive_damping, data.drive_force_limit,
            data.actuator_noload_speed, data.joint_f, data.task_target,
            data.task_rotation_target, data.task_local_pose,
            data.task_nullspace_stiffness, data.task_nullspace_damping, data.m_inv,
            data.drive_command, data.drive_dissipation, data.drive_lower,
            data.drive_upper, data.control_mass, data.control_factor,
            data.control_jacobian, data.control_response, data.control_task_map,
            data.env_status};
}

// Controllers provide an affine effort and its bounds to the common row solver.
// Passive joint damping stays in ABA; feedforward is an independent generalized load.
__global__ void ApplyAffineDriveKernel(ArticulationDeviceState state,
                                       DriveControlView drive, uint32_t mode,
                                       uint32_t links_per_env, bool implicit) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count) return;
    drive.command[link] = drive.dissipation[link] = 0.0f;
    drive.lower[link] = drive.upper[link] = 0.0f;
    state.tau[link] = 0.0f;
    if (JointDofCountDevice(state.joint_type[link]) != 1u) return;

    const auto control = static_cast<ArticulationControlMode>(mode);
    const bool inverse_dynamics = control == ArticulationControlMode::ComputedTorque ||
                                  control == ArticulationControlMode::Osc;
    const bool position_control = control == ArticulationControlMode::PDPosition || inverse_dynamics;
    const bool velocity_control = control == ArticulationControlMode::Velocity;
    if ((position_control && (!isfinite(drive.target[link]) ||
            !isfinite(drive.stiffness[link]) || drive.stiffness[link] < 0.0f ||
            !isfinite(drive.damping[link]) || drive.damping[link] < 0.0f)) ||
        ((position_control || velocity_control) && !isfinite(drive.velocity_target[link])) ||
        (velocity_control && (!isfinite(drive.stiffness[link]) || drive.stiffness[link] < 0.0f)) ||
        (control == ArticulationControlMode::ComputedTorque && !isfinite(drive.acceleration_target[link])) ||
        (control == ArticulationControlMode::Actuator && !isfinite(drive.noload_speed[link]))) {
        atomicOr(drive.status + link / links_per_env, kEnvStatusControlFailure);
        return;
    }
    const float velocity = state.qdot[link];
    const float limit = drive.force_limit[link];
    float lower = limit > 0.0f ? -limit : -FLT_MAX;
    float upper = limit > 0.0f ? limit : FLT_MAX;
    float bias = 0.0f;
    float damping = 0.0f;
    switch (control) {
        case ArticulationControlMode::PDPosition:
            damping = drive.damping[link];
            bias = drive.stiffness[link] * (drive.target[link] - state.q[link]) +
                   damping * drive.velocity_target[link];
            break;
        case ArticulationControlMode::Velocity:
            damping = drive.stiffness[link];
            bias = damping * drive.velocity_target[link];
            break;
        case ArticulationControlMode::Torque:
        case ArticulationControlMode::Actuator:
            bias = drive.target[link];
            if (control == ArticulationControlMode::Actuator && limit > 0.0f &&
                drive.noload_speed[link] > 0.0f) {
                const float speed_bias = limit * velocity / drive.noload_speed[link];
                lower = fminf(fmaxf(-limit - speed_bias, -limit), limit);
                upper = fminf(fmaxf(limit - speed_bias, -limit), limit);
            }
            break;
        case ArticulationControlMode::ComputedTorque:
        case ArticulationControlMode::Osc:
            break;
    }
    if (!implicit) {
        bias -= damping * velocity;
        damping = 0.0f;
    }
    const float feedforward = drive.feedforward[link];
    if (!isfinite(bias) || !isfinite(damping) || damping < 0.0f ||
        !isfinite(limit) || !isfinite(feedforward) || !isfinite(velocity) || !isfinite(state.q[link])) {
        atomicOr(drive.status + link / links_per_env, kEnvStatusControlFailure);
        return;
    }
    drive.command[link] = bias;
    drive.dissipation[link] = damping;
    drive.lower[link] = lower;
    drive.upper[link] = upper;
    state.tau[link] = inverse_dynamics ? 0.0f : feedforward;
}

__global__ void ReadoutDrivesKernel(ArticulationDeviceState state,
                                    DriveControlView drive, const float* lambda,
                                    ReadoutDrivesParams params, float* requested,
                                    float* applied, float* saturated) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count) return;
    requested[link] = applied[link] = saturated[link] = 0.0f;
    if (JointDofCountDevice(state.joint_type[link]) != 1u) return;
    const uint32_t env = link / params.links_per_env;
    const uint32_t local = link - env * params.links_per_env;
    const uint32_t row = env * params.rows_per_env + params.first_drive_row + local;
    const float effort = drive.command[link] - drive.dissipation[link] * state.qdot[link];
    requested[link] = effort;
    applied[link] = lambda[row] / params.dt;
    saturated[link] = effort < drive.lower[link] || effort > drive.upper[link] ? 1.0f : 0.0f;
    state.tau[link] += applied[link];
}

// The world inertia tensor is refreshed before any contact impulses are applied.
__global__ void BodyIntegrateVelocityKernel(math::Vec3* body_linear_velocity,
                                      math::Vec3* body_angular_velocity,
                                      math::Vec3* body_force,
                                      math::Vec3* body_torque,
                                      const float* body_inv_mass,
                                      const math::Transform* body_pose,
                                      const math::Transform* body_inertial_frame,
                                      const math::Vec3* body_inv_inertia,
                                      math::SymmetricMat3* body_world_inv_inertia,
                                      uint32_t total_body_count,
                                      math::Vec3 gravity,
                                      float dt, uint32_t clear_forces) {
    const uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= total_body_count) {
        return;
    }
    const math::Vec3 force = body_force[body];
    const math::Vec3 torque = body_torque[body];
    if (clear_forces != 0u) {
        body_force[body] = {};
        body_torque[body] = {};
    }
    if (body_inv_mass[body] <= 0.0f) {
        body_world_inv_inertia[body] = math::SymmetricMat3{};
        return;
    }
    body_world_inv_inertia[body] = BodyWorldInverseInertia(
        body_pose[body], body_inertial_frame[body], body_inv_inertia[body]);
    body_linear_velocity[body] += (gravity + force * body_inv_mass[body]) * dt;
    body_angular_velocity[body] += body_world_inv_inertia[body].Multiply(torque) * dt;
}

__global__ void ClearGyroStatusKernel(uint32_t* status, uint32_t env_count) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < env_count) status[env] &= ~kEnvStatusGyroFailure;
}

// The midpoint momentum equation and Cayley pose update share one time layer.
__global__ void BodyIntegratePositionKernel(math::Transform* body_pose,
                                            const math::Vec3* body_linear_velocity,
                                            math::Vec3* body_angular_velocity,
                                            const math::Vec3* body_pseudo_lin_vel,
                                            const math::Vec3* body_pseudo_ang_vel,
                                            const float* body_inv_mass,
                                            const math::Transform* body_inertial_frame,
                                            const math::Vec3* body_inv_inertia,
                                            math::SymmetricMat3* body_world_inv_inertia,
                                            float* gyro_residual,
                                            uint32_t* gyro_iterations,
                                            uint32_t* gyro_status,
                                            uint32_t* env_status,
                                            uint32_t bodies_per_env,
                                            uint32_t total_body_count,
                                            float dt) {
    const uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= total_body_count) {
        return;
    }
    gyro_residual[body] = 0.0f;
    gyro_iterations[body] = gyro_status[body] = 0u;
    if (body_inv_mass[body] <= 0.0f) return;
    math::Transform pose = body_pose[body];
    const auto& frame = body_inertial_frame[body];
    const math::Vec3 inv_i = body_inv_inertia[body];
    const math::Quat principal = mg::QuatNormalizeRsqrt(mg::QuatMul(pose.rotation, frame.rotation), 1.0e-12f);
    const math::Quat inverse = mg::MakeQuat(principal.w, -principal.x, -principal.y, -principal.z);
    const math::Vec3 w0 = mg::RotateByQuatNormalized(inverse, body_angular_velocity[body]);
    const GyroResult gyro = SolveFreeRotation(w0, inv_i, dt);
    gyro_residual[body] = gyro.residual;
    gyro_iterations[body] = gyro.iterations;
    gyro_status[body] = gyro.status;
    if (gyro.status != 0u) {
        atomicOr(env_status + body / bodies_per_env, kEnvStatusGyroFailure);
        return;
    }
    const math::Vec3 momentum = mg::RotateByQuatNormalized(principal,
        {w0.x / inv_i.x, w0.y / inv_i.y, w0.z / inv_i.z});
    math::Vec3 v = body_linear_velocity[body];
    const math::Vec3 w = mg::RotateByQuatNormalized(principal, gyro.midpoint);
    const math::Vec3 center = BodyCenterOfMass(pose, frame);
    const math::Quat dq = mg::MakeQuat(1.0f, 0.5f * dt * w.x, 0.5f * dt * w.y, 0.5f * dt * w.z);
    math::Quat q = mg::QuatNormalizeRsqrt(mg::QuatMul(dq, pose.rotation), 1.0e-12f);
    if (body_pseudo_lin_vel != nullptr) {
        v += body_pseudo_lin_vel[body];
        const math::Vec3 wp = body_pseudo_ang_vel[body];
        const math::Quat pseudo = mg::MakeQuat(1.0f, 0.5f * dt * wp.x, 0.5f * dt * wp.y, 0.5f * dt * wp.z);
        q = mg::QuatNormalizeRsqrt(mg::QuatMul(pseudo, q), 1.0e-12f);
    }
    pose.rotation = q;
    const math::Vec3 offset = mg::RotateByQuatNormalized(q, frame.position);
    pose.position = mg::Sub(mg::Add(center, mg::Scale(v, dt)), offset);
    body_pose[body] = pose;
    const auto inverse_inertia = BodyWorldInverseInertia(pose, frame, inv_i);
    body_world_inv_inertia[body] = inverse_inertia;
    body_angular_velocity[body] = inverse_inertia.Multiply(momentum);
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
    return JointRelativeFrame(state.joint_type[link], state.joint_axis[link],
        state.link_local_pose[link], state.parent_offset[link], state.q[link]);
}

constexpr uint32_t kTaskDimension = 6u;

// LDL factors use arena storage sized from the articulation, without a controller-specific DOF cap.
__device__ bool FactorControlMass(float* matrix, uint32_t n, uint32_t stride,
                                   float* diagonal) {
    for (uint32_t j = 0u; j < n; ++j) {
        double pivot = matrix[j * stride + j];
        for (uint32_t k = 0u; k < j; ++k) {
            const double value = matrix[j * stride + k];
            pivot -= value * value * diagonal[k];
        }
        if (!(pivot > 0.0) || !isfinite(pivot)) return false;
        diagonal[j] = static_cast<float>(pivot);
        for (uint32_t i = j + 1u; i < n; ++i) {
            double value = matrix[i * stride + j];
            for (uint32_t k = 0u; k < j; ++k)
                value -= static_cast<double>(matrix[i * stride + k]) *
                         matrix[j * stride + k] * diagonal[k];
            matrix[i * stride + j] = static_cast<float>(value / pivot);
        }
    }
    return true;
}

__device__ void SolveControlMass(const float* factor, const float* diagonal,
                                  uint32_t n, uint32_t stride,
                                  const float* rhs, float* solution) {
    for (uint32_t i = 0u; i < n; ++i) {
        double value = rhs[i];
        for (uint32_t j = 0u; j < i; ++j)
            value -= static_cast<double>(factor[i * stride + j]) * solution[j];
        solution[i] = static_cast<float>(value);
    }
    for (uint32_t i = 0u; i < n; ++i) solution[i] /= diagonal[i];
    for (uint32_t ii = n; ii > 0u; --ii) {
        const uint32_t i = ii - 1u;
        double value = solution[i];
        for (uint32_t j = i + 1u; j < n; ++j)
            value -= static_cast<double>(factor[j * stride + i]) * solution[j];
        solution[i] = static_cast<float>(value);
    }
}

// A symmetric pseudoinverse discards unreachable task directions at the float precision floor.
__device__ void InvertTaskResponse(float* matrix, uint32_t n, float* inverse) {
    float vectors[kTaskDimension * kTaskDimension] = {};
    for (uint32_t i = 0u; i < n; ++i) vectors[i * kTaskDimension + i] = 1.0f;
    for (uint32_t sweep = 0u; sweep < 24u; ++sweep) {
        bool changed = false;
        for (uint32_t p = 0u; p < n; ++p) {
            for (uint32_t q = p + 1u; q < n; ++q) {
                const float app = matrix[p * kTaskDimension + p];
                const float aqq = matrix[q * kTaskDimension + q];
                const float apq = matrix[p * kTaskDimension + q];
                if (fabsf(apq) <= FLT_EPSILON * (fabsf(app) + fabsf(aqq))) continue;
                changed = true;
                const double theta = (static_cast<double>(aqq) - app) / (2.0 * apq);
                const float tangent = static_cast<float>(copysign(1.0, theta) /
                    (fabs(theta) + sqrt(1.0 + theta * theta)));
                const float cosine = rsqrtf(1.0f + tangent * tangent);
                const float sine = tangent * cosine;
                for (uint32_t k = 0u; k < n; ++k) {
                    if (k != p && k != q) {
                        const float kp = matrix[k * kTaskDimension + p];
                        const float kq = matrix[k * kTaskDimension + q];
                        matrix[k * kTaskDimension + p] = matrix[p * kTaskDimension + k] = cosine * kp - sine * kq;
                        matrix[k * kTaskDimension + q] = matrix[q * kTaskDimension + k] = sine * kp + cosine * kq;
                    }
                    const float vp = vectors[k * kTaskDimension + p];
                    const float vq = vectors[k * kTaskDimension + q];
                    vectors[k * kTaskDimension + p] = cosine * vp - sine * vq;
                    vectors[k * kTaskDimension + q] = sine * vp + cosine * vq;
                }
                matrix[p * kTaskDimension + p] = app - tangent * apq;
                matrix[q * kTaskDimension + q] = aqq + tangent * apq;
                matrix[p * kTaskDimension + q] = matrix[q * kTaskDimension + p] = 0.0f;
            }
        }
        if (!changed) break;
    }
    float largest = 0.0f;
    for (uint32_t i = 0u; i < n; ++i)
        largest = fmaxf(largest, matrix[i * kTaskDimension + i]);
    const float tolerance = largest * (8.0f * FLT_EPSILON);
    for (uint32_t r = 0u; r < n; ++r) {
        for (uint32_t c = 0u; c < n; ++c) {
            double value = 0.0;
            for (uint32_t k = 0u; k < n; ++k) {
                const float eigenvalue = matrix[k * kTaskDimension + k];
                if (eigenvalue > tolerance)
                    value += static_cast<double>(vectors[r * kTaskDimension + k]) *
                             vectors[c * kTaskDimension + k] / eigenvalue;
            }
            inverse[r * kTaskDimension + c] = static_cast<float>(value);
        }
    }
}

__device__ void StoreTaskColumn(float* jacobian, uint32_t stride, uint32_t dof,
                                 math::Vec3 linear, math::Vec3 angular) {
    jacobian[dof] = linear.x;
    jacobian[stride + dof] = linear.y;
    jacobian[2u * stride + dof] = linear.z;
    jacobian[3u * stride + dof] = angular.x;
    jacobian[4u * stride + dof] = angular.y;
    jacobian[5u * stride + dof] = angular.z;
}

__device__ math::Vec3 TaskOrientationError(math::Quat desired, math::Quat current) {
    current = QuatNormalize(current);
    desired = QuatNormalize(desired);
    math::Quat conjugate = current;
    conjugate.x = -conjugate.x;
    conjugate.y = -conjugate.y;
    conjugate.z = -conjugate.z;
    math::Quat error = QuatMul(desired, conjugate);
    const float sign = error.w < 0.0f ? -1.0f : 1.0f;
    const float length = sqrtf(error.x * error.x + error.y * error.y + error.z * error.z);
    const float scale = length > 1.0e-7f
        ? sign * 2.0f * atan2f(length, fabsf(error.w)) / length : sign * 2.0f;
    return {error.x * scale, error.y * scale, error.z * scale};
}

// S selects scalar actuators: H = (S M^-1 S^T)^-1 includes floating-base reaction.
// OSC uses B = J M^-1 S^T and projects posture forces through its dynamic nullspace.
__global__ void ApplyDynamicsDriveKernel(ArticulationDeviceState state,
                                          DriveControlView drive,
                                          ApplyDynamicsDrivesParams params) {
    const uint32_t articulation = blockIdx.x;
    if (articulation >= state.articulation_count || threadIdx.x != 0u) return;
    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    const uint32_t stride = params.max_dof;
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (JointDofCountDevice(state.joint_type[link]) == 1u &&
            drive.lower[link] == 0.0f && drive.upper[link] == 0.0f)
            return;
    }
    if (params.mode == static_cast<uint32_t>(ArticulationControlMode::Osc)) {
        const auto target = drive.task_target[articulation];
        const auto rotation = drive.task_rotation[articulation];
        const auto frame = drive.task_frame[articulation];
        const uint32_t link = offset + params.task_link;
        if (!isfinite(target.x) || !isfinite(target.y) || !isfinite(target.z) ||
            !isfinite(rotation.w) || !isfinite(rotation.x) || !isfinite(rotation.y) || !isfinite(rotation.z) ||
            !isfinite(frame.position.x) || !isfinite(frame.position.y) || !isfinite(frame.position.z) ||
            !isfinite(frame.rotation.w) || !isfinite(frame.rotation.x) ||
            !isfinite(frame.rotation.y) || !isfinite(frame.rotation.z) ||
            !isfinite(drive.stiffness[link]) || drive.stiffness[link] < 0.0f ||
            !isfinite(drive.damping[link]) || drive.damping[link] < 0.0f ||
            !isfinite(drive.null_stiffness[articulation]) || drive.null_stiffness[articulation] < 0.0f ||
            !isfinite(drive.null_damping[articulation]) || drive.null_damping[articulation] < 0.0f) {
            atomicOr(drive.status + offset / params.links_per_env, kEnvStatusControlFailure);
            return;
        }
    }
    uint32_t links[kMaxArticulationDof];
    uint32_t dofs[kMaxArticulationDof];
    uint32_t active = 0u;
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (JointDofCountDevice(state.joint_type[link]) == 1u) {
            links[active] = link;
            dofs[active] = LocalDofIndexDevice(state, offset, link);
            state.tau[link] = drive.feedforward[link];
            ++active;
        }
    }
    if (active == 0u) return;
    const size_t matrix_offset = static_cast<size_t>(articulation) * stride * stride;
    const float* inverse_mass = drive.inverse_mass + matrix_offset;
    float* factor = drive.factor + matrix_offset;
    float* mass = drive.mass + matrix_offset;
    float diagonal[kMaxArticulationDof];
    float rhs[kMaxArticulationDof];
    float solution[kMaxArticulationDof];
    for (uint32_t r = 0u; r < active; ++r)
        for (uint32_t c = 0u; c < active; ++c)
            factor[r * stride + c] = 0.5f *
                (inverse_mass[dofs[r] * stride + dofs[c]] + inverse_mass[dofs[c] * stride + dofs[r]]);
    if (!FactorControlMass(factor, active, stride, diagonal)) {
        atomicOr(drive.status + offset / params.links_per_env, kEnvStatusControlFailure);
        return;
    }
    for (uint32_t c = 0u; c < active; ++c) {
        for (uint32_t r = 0u; r < active; ++r) rhs[r] = r == c ? 1.0f : 0.0f;
        SolveControlMass(factor, diagonal, active, stride, rhs, solution);
        for (uint32_t r = 0u; r < active; ++r) mass[r * stride + c] = solution[r];
    }
    float bias[kMaxArticulationDof];
    for (uint32_t r = 0u; r < active; ++r) {
        double value = -state.joint_damping[links[r]] * state.qdot[links[r]];
        for (uint32_t c = 0u; c < active; ++c)
            value -= static_cast<double>(mass[r * stride + c]) * state.qddot[links[c]];
        bias[r] = static_cast<float>(value);
    }
    if (params.mode == static_cast<uint32_t>(ArticulationControlMode::ComputedTorque)) {
        for (uint32_t c = 0u; c < active; ++c) {
            const uint32_t link = links[c];
            rhs[c] = drive.acceleration_target[link] +
                drive.stiffness[link] * (drive.target[link] - state.q[link]) +
                drive.damping[link] * (drive.velocity_target[link] - state.qdot[link]);
        }
        for (uint32_t r = 0u; r < active; ++r) {
            double value = bias[r];
            for (uint32_t c = 0u; c < active; ++c)
                value += static_cast<double>(mass[r * stride + c]) * rhs[c];
            drive.command[links[r]] = static_cast<float>(value);
        }
    } else {
        const uint32_t task_link = offset + params.task_link;
        math::Transform local_task = drive.task_frame[articulation];
        local_task.rotation = QuatNormalize(local_task.rotation);
        const math::Transform task_pose = ComposeTransform(state.link_pose[task_link], local_task);
        const size_t task_offset = static_cast<size_t>(articulation) * stride * kTaskDimension;
        float* jacobian = drive.jacobian + task_offset;
        float* response = drive.response + task_offset;
        float* task_map = drive.task_map + task_offset;
        for (uint32_t i = 0u; i < stride * kTaskDimension; ++i) jacobian[i] = 0.0f;
        uint8_t chain[kMaxArticulationDof] = {};
        uint32_t walk = task_link;
        while (walk != kInvalidLink) {
            const auto type = state.joint_type[walk];
            const uint32_t dof = LocalDofIndexDevice(state, offset, walk);
            const math::Vec3 lever = mg::Sub(task_pose.position, state.link_pose[walk].position);
            if (type == ArticulationJointType::FloatingBase) {
                const math::Vec3 basis[3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
                for (uint32_t c = 0u; c < 3u; ++c) {
                    const auto axis = RotateByQuat(state.base_pose[articulation].rotation, basis[c]);
                    StoreTaskColumn(jacobian, stride, dof + c, Cross3(axis, lever), axis);
                    StoreTaskColumn(jacobian, stride, dof + 3u + c, axis, {});
                }
            } else if (JointDofCountDevice(type) == 1u) {
                const auto axis = RotateByQuat(state.link_pose[walk].rotation, state.joint_axis[walk]);
                const bool revolute = type == ArticulationJointType::Revolute;
                StoreTaskColumn(jacobian, stride, dof, revolute ? Cross3(axis, lever) : axis,
                                revolute ? axis : math::Vec3{});
                chain[dof] = 1u;
            }
            const uint32_t parent = state.parent_link[walk];
            walk = parent == kInvalidLink ? kInvalidLink : offset + parent;
        }
        const auto desired_rotation = drive.task_rotation[articulation];
        const float rotation_norm = desired_rotation.w * desired_rotation.w +
            desired_rotation.x * desired_rotation.x + desired_rotation.y * desired_rotation.y +
            desired_rotation.z * desired_rotation.z;
        const uint32_t task_rows = rotation_norm > 1.0e-12f ? 6u : 3u;
        for (uint32_t r = 0u; r < task_rows; ++r) {
            for (uint32_t c = 0u; c < active; ++c) {
                double value = 0.0;
                for (uint32_t d = 0u; d < stride; ++d)
                    value += static_cast<double>(jacobian[r * stride + d]) * inverse_mass[d * stride + dofs[c]];
                response[r * stride + c] = static_cast<float>(value);
            }
        }
        for (uint32_t r = 0u; r < task_rows; ++r) {
            for (uint32_t c = 0u; c < active; ++c) {
                double value = 0.0;
                for (uint32_t a = 0u; a < active; ++a)
                    value += static_cast<double>(mass[c * stride + a]) * response[r * stride + a];
                task_map[r * stride + c] = static_cast<float>(value);
            }
        }
        float metric[kTaskDimension * kTaskDimension] = {};
        float inverse_metric[kTaskDimension * kTaskDimension] = {};
        for (uint32_t r = 0u; r < task_rows; ++r) {
            for (uint32_t c = 0u; c < task_rows; ++c) {
                double value = 0.0;
                for (uint32_t a = 0u; a < active; ++a)
                    value += static_cast<double>(response[r * stride + a]) * task_map[c * stride + a];
                metric[r * kTaskDimension + c] = static_cast<float>(value);
            }
        }
        InvertTaskResponse(metric, task_rows, inverse_metric);

        const float* velocity = state.link_velocity[task_link].v;
        const float* acceleration = state.link_acceleration[task_link].a;
        const math::Vec3 omega{velocity[0], velocity[1], velocity[2]};
        const math::Vec3 linear{velocity[3], velocity[4], velocity[5]};
        const math::Vec3 alpha{acceleration[0], acceleration[1], acceleration[2]};
        const math::Vec3 linear_accel{acceleration[3], acceleration[4], acceleration[5]};
        const auto rotation = state.link_pose[task_link].rotation;
        const auto task_velocity = RotateByQuat(rotation, AddVec(linear, Cross3(omega, local_task.position)));
        const auto task_angular = RotateByQuat(rotation, omega);
        const auto task_accel = AddVec(RotateByQuat(rotation,
            AddVec(AddVec(linear_accel, Cross3(omega, linear)),
                   AddVec(Cross3(alpha, local_task.position), Cross3(omega, Cross3(omega, local_task.position))))),
            {params.gravity[0], params.gravity[1], params.gravity[2]});
        const auto task_angular_accel = RotateByQuat(rotation, alpha);
        const auto position_error = mg::Sub(drive.task_target[articulation], task_pose.position);
        const auto orientation_error = task_rows == 6u ? TaskOrientationError(desired_rotation, task_pose.rotation) : math::Vec3{};
        const float kp = drive.stiffness[task_link], kd = drive.damping[task_link];
        float task_rhs[kTaskDimension] = {
            kp * position_error.x - kd * task_velocity.x - task_accel.x,
            kp * position_error.y - kd * task_velocity.y - task_accel.y,
            kp * position_error.z - kd * task_velocity.z - task_accel.z,
            kp * orientation_error.x - kd * task_angular.x - task_angular_accel.x,
            kp * orientation_error.y - kd * task_angular.y - task_angular_accel.y,
            kp * orientation_error.z - kd * task_angular.z - task_angular_accel.z};

        const float null_kp = drive.null_stiffness[articulation];
        const float null_kd = drive.null_damping[articulation];
        for (uint32_t c = 0u; c < active; ++c) {
            const uint32_t link = links[c];
            rhs[c] = chain[dofs[c]] ?
                null_kp * (drive.target[link] - state.q[link]) +
                null_kd * (drive.velocity_target[link] - state.qdot[link]) : 0.0f;
        }
        float posture[kMaxArticulationDof];
        for (uint32_t r = 0u; r < active; ++r) {
            double value = 0.0;
            for (uint32_t c = 0u; c < active; ++c)
                value += static_cast<double>(mass[r * stride + c]) * rhs[c];
            const uint32_t link = links[r];
            if (!chain[dofs[r]])
                value += drive.stiffness[link] * (drive.target[link] - state.q[link]) +
                         drive.damping[link] * (drive.velocity_target[link] - state.qdot[link]);
            posture[r] = static_cast<float>(value);
        }
        for (uint32_t r = 0u; r < task_rows; ++r) {
            for (uint32_t a = 0u; a < active; ++a) {
                const uint32_t link = links[a];
                task_rhs[r] -= response[r * stride + a] *
                    (bias[a] + state.joint_damping[link] * state.qdot[link] + posture[a]);
            }
        }
        float task_force[kTaskDimension] = {};
        for (uint32_t r = 0u; r < task_rows; ++r)
            for (uint32_t c = 0u; c < task_rows; ++c)
                task_force[r] += inverse_metric[r * kTaskDimension + c] * task_rhs[c];
        for (uint32_t a = 0u; a < active; ++a) {
            double value = bias[a] + posture[a];
            for (uint32_t r = 0u; r < task_rows; ++r)
                value += static_cast<double>(task_map[r * stride + a]) * task_force[r];
            drive.command[links[a]] = static_cast<float>(value);
        }
    }
    for (uint32_t a = 0u; a < active; ++a) {
        if (!isfinite(drive.command[links[a]])) {
            for (uint32_t c = 0u; c < active; ++c) drive.command[links[c]] = 0.0f;
            atomicOr(drive.status + offset / params.links_per_env, kEnvStatusControlFailure);
            return;
        }
    }
}

__global__ void UpdateWorldLinkPosesKernel(ArticulationDeviceState state,
                                           math::Transform* out_world_pose,
                                           const uint32_t* env_ids,
                                           uint32_t articulations_per_env) {
    const uint32_t articulation = env_ids
        ? env_ids[blockIdx.x / articulations_per_env] * articulations_per_env +
              blockIdx.x % articulations_per_env
        : blockIdx.x;
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
    if (!p || p->mode > static_cast<uint32_t>(ArticulationControlMode::Actuator))
        return Status::InvalidArgument;
    if (p->total_link_count == 0u) return Status::Ok;
    if (p->links_per_env == 0u) return Status::InvalidArgument;
    const auto state = MakeArticulationDeviceState(model, data, p->total_link_count, 0u);
    const uint32_t blocks = (p->total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
    LaunchCuda(ApplyAffineDriveKernel, dim3(blocks), dim3(kAbaBlockSize), 0u, stream,
               state, MakeDriveControlView(data), p->mode, p->links_per_env,
               p->defer_velocity_damping != 0u);
    return LaunchOk(stream);
}

Status OpApplyDynamicsDrives(const ModelView& model, const DataView& data,
                             const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ApplyDynamicsDrivesParams*>(params);
    if (!p || p->max_dof > kMaxArticulationDof ||
        (p->mode != static_cast<uint32_t>(ArticulationControlMode::ComputedTorque) &&
         p->mode != static_cast<uint32_t>(ArticulationControlMode::Osc)))
        return Status::InvalidArgument;
    if (p->articulation_count == 0u || p->total_link_count == 0u || p->max_dof == 0u)
        return Status::Ok;
    if (!data.m_inv || !data.control_mass || !data.control_factor || !data.control_jacobian ||
        !data.control_response || !data.control_task_map || p->links_per_env == 0u)
        return Status::InvalidArgument;
    const auto state = MakeArticulationDeviceState(model, data, p->total_link_count, p->articulation_count);
    LaunchCuda(ApplyDynamicsDriveKernel, dim3(p->articulation_count), dim3(32u), 0u,
               stream, state, MakeDriveControlView(data), *p);
    return LaunchOk(stream);
}

Status OpReadoutDrives(const ModelView& model, const DataView& data,
                       const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ReadoutDrivesParams*>(params);
    if (!p || p->dt <= 0.0f) return Status::InvalidArgument;
    if (p->total_link_count == 0u) return Status::Ok;
    if (p->links_per_env == 0u || p->first_drive_row + p->links_per_env > p->rows_per_env)
        return Status::InvalidArgument;
    const auto state = MakeArticulationDeviceState(model, data, p->total_link_count, 0u);
    const uint32_t blocks = (p->total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
    LaunchCuda(ReadoutDrivesKernel, dim3(blocks), dim3(kAbaBlockSize), 0u, stream,
               state, MakeDriveControlView(data), data.lambda, *p,
               data.actuator_effort_requested, data.actuator_effort, data.actuator_saturated);
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
    const math::Vec3 gravity{p->gravity[0], p->gravity[1], p->gravity[2]};
    dim3 grid(p->articulation_count);
    dim3 block(kAbaBlockSize);
    LaunchCuda(AbaPass1KinematicsKernel, grid, block, 0u, stream, state);
    LaunchCuda(AbaPass2ArticulatedInertiaKernel, grid, block, 0u, stream, state);
    LaunchCuda(AbaPass3AccelerationKernel, grid, block, 0u, stream, state, gravity);
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
    const math::Vec3 gravity{p->gravity[0], p->gravity[1], p->gravity[2]};
    if (p->total_link_count > 0u) {
        const ArticulationDeviceState state = MakeArticulationDeviceState(
            model, data, p->total_link_count, p->articulation_count);
        const uint32_t blocks =
            (p->total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
        // Joint velocity first, then the floating-base velocity — the production
        // batched articulated order (the two write DISJOINT state: qdot vs
        // link_velocity[root], so the single-env FB-first order is byte-equal).
        LaunchCuda(IntegrateVelocityArticulationKernel, dim3(blocks),
                   dim3(kAbaBlockSize), 0u, stream, state, p->dt);
        if (p->articulation_count > 0u) {
            LaunchCuda(IntegrateFloatingBaseVelocityKernel, dim3(p->articulation_count),
                       dim3(kAbaBlockSize), 0u, stream, state, p->dt, gravity);
        }
    }
    // Free-body inertia and velocity are updated before the shared contact solve.
    if (p->total_body_count > 0u) {
        const uint32_t blocks =
            (p->total_body_count + kAbaBlockSize - 1u) / kAbaBlockSize;
        LaunchCuda(BodyIntegrateVelocityKernel, dim3(blocks), dim3(kAbaBlockSize), 0u,
                   stream, data.body_linear_velocity, data.body_angular_velocity,
                   data.body_force, data.body_torque,
                   static_cast<const float*>(data.body_inv_mass),
                   data.body_pose, data.body_inertial_frame, data.body_inv_inertia,
                   data.body_world_inv_inertia,
                   p->total_body_count, gravity, p->dt, p->clear_body_forces);
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
    if (p->selected_env_count > 0u &&
        (p->articulations_per_env == 0u || data.reset_env_ids == nullptr ||
         p->articulation_count % p->articulations_per_env != 0u ||
         p->selected_env_count > p->articulation_count / p->articulations_per_env))
        return Status::Failed;
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, p->total_link_count, p->articulation_count);
    const uint32_t count = p->selected_env_count > 0u
        ? p->selected_env_count * p->articulations_per_env : p->articulation_count;
    const uint32_t* env_ids = p->selected_env_count > 0u ? data.reset_env_ids : nullptr;
    LaunchCuda(UpdateWorldLinkPosesKernel, dim3(count), dim3(32u),
               0u, stream, state, data.link_pose, env_ids, p->articulations_per_env);
    return LaunchOk(stream);
}

Status OpFkLinkVelocities(const ModelView& model, const DataView& data,
                           const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const FkLinkVelocitiesParams*>(params);
    if (p == nullptr) return Status::InvalidArgument;
    if (p->articulation_count == 0u || p->total_link_count == 0u) return Status::Ok;
    if (!model.articulation_link_offset || !model.articulation_link_count ||
        !model.parent_link || !model.joint_type || !data.link_velocity || !data.qdot ||
        !data.q || !model.joint_axis || !model.link_local_pose || !model.parent_offset)
        return Status::InvalidArgument;
    const auto state = MakeArticulationDeviceState(model, data, p->total_link_count, p->articulation_count);
    LaunchCuda(FkLinkVelocitiesKernel, dim3(p->articulation_count), dim3(32u),
               0u, stream, state);
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
    // Split-impulse: read the pseudo velocity additively when the position pass is
    // active. A null pseudo pointer makes every integrate kernel byte-identical.
    const bool pos_pass = p->pos_pass != 0u;
    if (p->total_link_count > 0u) {
        const ArticulationDeviceState state = MakeArticulationDeviceState(
            model, data, p->total_link_count, p->articulation_count);
        const uint32_t blocks =
            (p->total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
        const float* qdot_pseudo =
            pos_pass ? static_cast<const float*>(data.qdot_pseudo) : nullptr;
        const LinkSpatialVel* link_vel_pseudo =
            pos_pass ? reinterpret_cast<const LinkSpatialVel*>(data.link_velocity_pseudo)
                     : nullptr;
        LaunchCuda(IntegratePositionArticulationKernel, dim3(blocks),
                   dim3(kAbaBlockSize), 0u, stream, state, qdot_pseudo, p->dt);
        if (p->articulation_count > 0u) {
            LaunchCuda(IntegrateFloatingBasePoseKernel, dim3(p->articulation_count),
                       dim3(kAbaBlockSize), 0u, stream, state, link_vel_pseudo, p->dt);
        }
    }
    // Advance free bodies using the velocities from the shared contact solve.
    if (p->total_body_count > 0u) {
        if (p->env_count == 0u || p->total_body_count % p->env_count != 0u)
            return Status::InvalidArgument;
        LaunchCuda(ClearGyroStatusKernel,
                   dim3((p->env_count + kAbaBlockSize - 1u) / kAbaBlockSize),
                   dim3(kAbaBlockSize), 0u, stream, data.env_status, p->env_count);
        const uint32_t blocks =
            (p->total_body_count + kAbaBlockSize - 1u) / kAbaBlockSize;
        const math::Vec3* body_pseudo_lin =
            pos_pass ? static_cast<const math::Vec3*>(data.body_pseudo_linear_velocity)
                     : nullptr;
        const math::Vec3* body_pseudo_ang =
            pos_pass ? static_cast<const math::Vec3*>(data.body_pseudo_angular_velocity)
                     : nullptr;
        LaunchCuda(BodyIntegratePositionKernel, dim3(blocks), dim3(kAbaBlockSize), 0u,
                   stream, data.body_pose, data.body_linear_velocity,
                   data.body_angular_velocity, body_pseudo_lin, body_pseudo_ang,
                   static_cast<const float*>(data.body_inv_mass),
                   data.body_inertial_frame, data.body_inv_inertia,
                   data.body_world_inv_inertia, data.body_gyro_residual,
                   data.body_gyro_iterations, data.body_gyro_status, data.env_status,
                   p->total_body_count / p->env_count,
                   p->total_body_count, p->dt);
    }
    return LaunchOk(stream);
}

} // namespace

void RegisterNkAbaOps() {
    SetCudaOp(NkOp::ApplyDrives, &OpApplyDrives);
    SetCudaOp(NkOp::ApplyDynamicsDrives, &OpApplyDynamicsDrives);
    SetCudaOp(NkOp::ReadoutDrives, &OpReadoutDrives);
    SetCudaOp(NkOp::AbaForward, &OpAbaForward);
    SetCudaOp(NkOp::IntegrateVelocity, &OpIntegrateVelocity);
    SetCudaOp(NkOp::FkWorldPoses, &OpFkWorldPoses);
    SetCudaOp(NkOp::FkLinkVelocities, &OpFkLinkVelocities);
    SetCudaOp(NkOp::IntegratePosition, &OpIntegratePosition);
}

// Umbrella: the explicit static-lib-safe registration of the whole M3b
// articulation pipeline, called from RegisterCudaBackendEntry(). Idempotent
// (SetCudaOp last-wins; re-registering the same fns is a no-op).
void RegisterNkArticulationPipelineOps() {
    RegisterNkAbaOps();
    RegisterNkCrbaOps();
    RegisterNkContactsFootOps();
    RegisterNkSyncBodyPoseOps();     // general contact B2: SyncLinkBodyPose
    RegisterNkNarrowphaseHeightfieldOps();  // general contact H3: heightfield midphase
    RegisterNkNarrowphaseBodyParticleOps(); // body/artic <-> particle narrowphase
    RegisterNkBroadphaseOps();
    RegisterNkNarrowphaseSdfOps();
    RegisterNkAssembleRowsOps();
    RegisterNkBuildSolveIslandsOps(); // dynamic connected-component solve schedule
    RegisterNkSolveRowsOps();
    RegisterNkParticleOps();
    RegisterNkMpmOps();              // MLS-MPM transfers (inert round-trip scaffold)
    RegisterNkReadoutOps();
    RegisterNkSensorOps();
    RegisterNkDiffsimBackwardOps();  // M9 T7: NkOp::StepBackward
}

} // namespace nuka::phi
