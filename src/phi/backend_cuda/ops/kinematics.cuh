#pragma once

#include "math/cuda_vec_ops.cuh"
#include "phi/articulation_contract.hpp"
#include "math/transform.hpp"

namespace nuka::phi::nkops {

__device__ inline math::Vec3 JointUnitAxis(math::Vec3 axis) {
    const float length_sq = math::gpu::Dot(axis, axis);
    return length_sq > 1.0e-12f ? math::gpu::Scale(axis, rsqrtf(length_sq)) : math::Vec3{};
}

__device__ inline math::Transform ComposeFrame(math::Transform parent, math::Transform local) {
    math::Transform result;
    result.position = math::gpu::Add(parent.position,
        math::gpu::RotateByQuatNormalized(parent.rotation, local.position));
    result.rotation = math::gpu::QuatNormalizeRsqrt(
        math::gpu::QuatMul(parent.rotation, local.rotation), 1.0e-12f);
    return result;
}

// Joint axes are expressed in the child joint frame for both motion and spatial transforms.
__device__ inline math::Transform JointRelativeFrame(ArticulationJointType type, math::Vec3 axis,
    math::Transform local, math::Vec3 parent_offset, float coordinate) {
    math::Transform result = local;
    result.position = math::gpu::Add(local.position, parent_offset);
    axis = JointUnitAxis(axis);
    if (type == ArticulationJointType::Revolute) {
        const float half = 0.5f * coordinate;
        const float sine = sinf(half);
        const auto rotation = math::gpu::MakeQuat(cosf(half), axis.x * sine, axis.y * sine, axis.z * sine);
        result.rotation = math::gpu::QuatNormalizeRsqrt(math::gpu::QuatMul(local.rotation, rotation), 1.0e-12f);
    } else if (type == ArticulationJointType::Prismatic) {
        result.position = math::gpu::Add(result.position, math::gpu::RotateByQuatNormalized(
            local.rotation, math::gpu::Scale(axis, coordinate)));
    }
    return result;
}

}  // namespace nuka::phi::nkops
