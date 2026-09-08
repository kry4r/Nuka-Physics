#pragma once

#include "math/cuda_vec_ops.cuh"
#include "math/symmetric_mat3.hpp"
#include "math/transform.hpp"

namespace nuka::phi::nkops {

__forceinline__ __device__ math::Vec3 BodyCenterOfMass(
    const math::Transform& pose, const math::Transform& inertial_frame) {
    return math::gpu::Add(pose.position,
        math::gpu::RotateByQuatNormalized(pose.rotation, inertial_frame.position));
}

__forceinline__ __device__ math::SymmetricMat3 BodyWorldInverseInertia(
    const math::Transform& pose, const math::Transform& inertial_frame,
    math::Vec3 principal_inverse_inertia) {
    return math::SymmetricMat3::FromRotatedDiagonal(principal_inverse_inertia,
        math::gpu::QuatMul(pose.rotation, inertial_frame.rotation));
}

}  // namespace nuka::phi::nkops
