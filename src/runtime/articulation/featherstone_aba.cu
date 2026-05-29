// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- CUDA Featherstone ABA implementation
// ---------------------------------------------------------------------------

#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/articulation/featherstone_aba.cuh"

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

__device__ math::Vec3 MakeVec3(float x, float y, float z) {
    return {x, y, z};
}

__device__ math::Vec3 Add(math::Vec3 a, math::Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ math::Vec3 Scale(math::Vec3 v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

__device__ float Dot3(math::Vec3 a, math::Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ math::Vec3 Cross3(math::Vec3 a, math::Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__device__ void Zero6(float* value) {
    for (uint32_t i = 0u; i < 6u; ++i) {
        value[i] = 0.0f;
    }
}

__device__ void Zero36(float* value) {
    for (uint32_t i = 0u; i < 36u; ++i) {
        value[i] = 0.0f;
    }
}

__device__ float Dot6(const float* a, const float* b) {
    float sum = 0.0f;
    for (uint32_t i = 0u; i < 6u; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

__device__ void Add6InPlace(float* lhs, const float* rhs) {
    for (uint32_t i = 0u; i < 6u; ++i) {
        lhs[i] += rhs[i];
    }
}

__device__ void SubtractOuterProduct36(float* matrix, const float* u, float diagonal) {
    const float inv_diagonal = 1.0f / fmaxf(diagonal, kMinDiagonal);
    for (uint32_t row = 0u; row < 6u; ++row) {
        for (uint32_t col = 0u; col < 6u; ++col) {
            matrix[row * 6u + col] -= u[row] * u[col] * inv_diagonal;
        }
    }
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

__device__ void TransformMotion(const LinkSpatialTransform& transform,
                                const float* in,
                                float* out) {
    for (uint32_t row = 0u; row < 6u; ++row) {
        float value = 0.0f;
        for (uint32_t col = 0u; col < 6u; ++col) {
            value += transform.X[row * 6u + col] * in[col];
        }
        out[row] = value;
    }
}

__device__ void TransformForceTranspose(const LinkSpatialTransform& transform,
                                        const float* in,
                                        float* out) {
    for (uint32_t row = 0u; row < 6u; ++row) {
        float value = 0.0f;
        for (uint32_t col = 0u; col < 6u; ++col) {
            value += transform.X[col * 6u + row] * in[col];
        }
        out[row] = value;
    }
}

__device__ void TransformInertiaToParent(const LinkSpatialTransform& transform,
                                         const float* child_inertia,
                                         float* parent_delta) {
    float temp[36];
    for (uint32_t row = 0u; row < 6u; ++row) {
        for (uint32_t col = 0u; col < 6u; ++col) {
            float value = 0.0f;
            for (uint32_t k = 0u; k < 6u; ++k) {
                value += child_inertia[row * 6u + k] * transform.X[k * 6u + col];
            }
            temp[row * 6u + col] = value;
        }
    }
    for (uint32_t row = 0u; row < 6u; ++row) {
        for (uint32_t col = 0u; col < 6u; ++col) {
            float value = 0.0f;
            for (uint32_t k = 0u; k < 6u; ++k) {
                value += transform.X[k * 6u + row] * temp[k * 6u + col];
            }
            parent_delta[row * 6u + col] = value;
        }
    }
}

__device__ void Mat66MulVec6(const float* matrix, const float* vector, float* out) {
    for (uint32_t row = 0u; row < 6u; ++row) {
        float value = 0.0f;
        for (uint32_t col = 0u; col < 6u; ++col) {
            value += matrix[row * 6u + col] * vector[col];
        }
        out[row] = value;
    }
}

__device__ void Copy36(const float* src, float* dst) {
    for (uint32_t i = 0u; i < 36u; ++i) {
        dst[i] = src[i];
    }
}

__device__ void MotionCross(const float* lhs, const float* rhs, float* out) {
    const math::Vec3 lw = MakeVec3(lhs[0], lhs[1], lhs[2]);
    const math::Vec3 lv = MakeVec3(lhs[3], lhs[4], lhs[5]);
    const math::Vec3 rw = MakeVec3(rhs[0], rhs[1], rhs[2]);
    const math::Vec3 rv = MakeVec3(rhs[3], rhs[4], rhs[5]);
    const math::Vec3 angular = Cross3(lw, rw);
    const math::Vec3 linear = Add(Cross3(lw, rv), Cross3(lv, rw));
    out[0] = angular.x;
    out[1] = angular.y;
    out[2] = angular.z;
    out[3] = linear.x;
    out[4] = linear.y;
    out[5] = linear.z;
}

__device__ void ForceCross(const float* motion, const float* force, float* out) {
    const math::Vec3 w = MakeVec3(motion[0], motion[1], motion[2]);
    const math::Vec3 v = MakeVec3(motion[3], motion[4], motion[5]);
    const math::Vec3 n = MakeVec3(force[0], force[1], force[2]);
    const math::Vec3 f = MakeVec3(force[3], force[4], force[5]);
    const math::Vec3 angular = Add(Cross3(w, n), Cross3(v, f));
    const math::Vec3 linear = Cross3(w, f);
    out[0] = angular.x;
    out[1] = angular.y;
    out[2] = angular.z;
    out[3] = linear.x;
    out[4] = linear.y;
    out[5] = linear.z;
}

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

__global__ void ApplyPositionDriveKernel(ArticulationDeviceState state,
                                         const float* drive_targets,
                                         const float* drive_stiffness,
                                         const float* drive_damping,
                                         const float* drive_force_limits) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= state.total_link_count ||
        state.joint_type[link] == ArticulationJointType::Fixed) {
        return;
    }
    float tau = drive_stiffness[link] * (drive_targets[link] - state.q[link]) -
                drive_damping[link] * state.qdot[link];
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

void LaunchFeatherstoneAbaKernels(const phi::DeviceContext& context,
                                  ArticulationDeviceState state,
                                  float gravity_z) {
    if (state.articulation_count == 0u || state.total_link_count == 0u) {
        return;
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    dim3 grid(state.articulation_count);
    dim3 block(kAbaBlockSize);

    AbaPass1KinematicsKernel<<<grid, block, 0u, stream>>>(state, gravity_z);
    CheckCuda(cudaGetLastError(), "AbaPass1KinematicsKernel launch");
    AbaPass2ArticulatedInertiaKernel<<<grid, block, 0u, stream>>>(state);
    CheckCuda(cudaGetLastError(), "AbaPass2ArticulatedInertiaKernel launch");
    AbaPass3AccelerationKernel<<<grid, block, 0u, stream>>>(state, gravity_z);
    CheckCuda(cudaGetLastError(), "AbaPass3AccelerationKernel launch");
}

void LaunchApplyPositionDriveKernels(const phi::DeviceContext& context,
                                     ArticulationDeviceState state,
                                     const float* drive_targets,
                                     const float* drive_stiffness,
                                     const float* drive_damping,
                                     const float* drive_force_limits) {
    if (state.total_link_count == 0u ||
        drive_targets == nullptr ||
        drive_stiffness == nullptr ||
        drive_damping == nullptr ||
        drive_force_limits == nullptr) {
        return;
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    const uint32_t blocks = (state.total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
    ApplyPositionDriveKernel<<<blocks, kAbaBlockSize, 0u, stream>>>(
        state,
        drive_targets,
        drive_stiffness,
        drive_damping,
        drive_force_limits);
    CheckCuda(cudaGetLastError(), "ApplyPositionDriveKernel launch");
}

void LaunchIntegrateArticulationKernels(const phi::DeviceContext& context,
                                        ArticulationDeviceState state,
                                        float dt) {
    if (state.total_link_count == 0u || dt <= 0.0f) {
        return;
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    const uint32_t blocks = (state.total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
    IntegrateArticulationKernel<<<blocks, kAbaBlockSize, 0u, stream>>>(state, dt);
    CheckCuda(cudaGetLastError(), "IntegrateArticulationKernel launch");
}

void LaunchIntegrateVelocityArticulationKernels(const phi::DeviceContext& context,
                                                ArticulationDeviceState state,
                                                float dt) {
    if (state.total_link_count == 0u || dt <= 0.0f) {
        return;
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    const uint32_t blocks = (state.total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
    IntegrateVelocityArticulationKernel<<<blocks, kAbaBlockSize, 0u, stream>>>(state, dt);
    CheckCuda(cudaGetLastError(), "IntegrateVelocityArticulationKernel launch");
}

void LaunchIntegratePositionArticulationKernels(const phi::DeviceContext& context,
                                                ArticulationDeviceState state,
                                                float dt) {
    if (state.total_link_count == 0u || dt <= 0.0f) {
        return;
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    const uint32_t blocks = (state.total_link_count + kAbaBlockSize - 1u) / kAbaBlockSize;
    IntegratePositionArticulationKernel<<<blocks, kAbaBlockSize, 0u, stream>>>(state, dt);
    CheckCuda(cudaGetLastError(), "IntegratePositionArticulationKernel launch");
}

void FeatherstoneAba::ApplyPositionDrives(const phi::DeviceContext& context,
                                          ArticulationDeviceState state,
                                          const float* drive_targets,
                                          const float* drive_stiffness,
                                          const float* drive_damping,
                                          const float* drive_force_limits) {
    LaunchApplyPositionDriveKernels(context,
                                    state,
                                    drive_targets,
                                    drive_stiffness,
                                    drive_damping,
                                    drive_force_limits);
}

void FeatherstoneAba::ComputeAccelerations(const phi::DeviceContext& context,
                                           ArticulationDeviceState state,
                                           float gravity_z) {
    LaunchFeatherstoneAbaKernels(context, state, gravity_z);
}

void FeatherstoneAba::Integrate(const phi::DeviceContext& context,
                                ArticulationDeviceState state,
                                float dt) {
    LaunchIntegrateArticulationKernels(context, state, dt);
}

void FeatherstoneAba::IntegrateVelocity(const phi::DeviceContext& context,
                                        ArticulationDeviceState state,
                                        float dt) {
    LaunchIntegrateVelocityArticulationKernels(context, state, dt);
}

void FeatherstoneAba::IntegratePosition(const phi::DeviceContext& context,
                                        ArticulationDeviceState state,
                                        float dt) {
    LaunchIntegratePositionArticulationKernels(context, state, dt);
}

} // namespace nuka::runtime::articulation
