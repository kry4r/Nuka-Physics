#pragma once

#include "math/cuda_vec_ops.cuh"
#include "math/symmetric_mat3.hpp"
#include "math/transform.hpp"
#include "phi/op_schema.hpp"

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

__forceinline__ __device__ float MaxAbs(math::Vec3 v) {
    return fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
}

__forceinline__ __device__ bool FiniteVector(math::Vec3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

struct GyroResult {
    math::Vec3 midpoint{};
    float residual = 0.0f;
    uint32_t iterations = 0u;
    uint32_t status = 0u;
};

__device__ inline GyroResult SolveFreeRotation(math::Vec3 w0, math::Vec3 inv_i, float dt) {
    namespace mg = math::gpu;
    GyroResult out;
    if (!FiniteVector(w0) || !FiniteVector(inv_i) ||
        inv_i.x <= 0.0f || inv_i.y <= 0.0f || inv_i.z <= 0.0f) {
        out.status = kBodyGyroInvalidInput;
        out.residual = INFINITY;
        return out;
    }
    const math::Vec3 m0{w0.x / inv_i.x, w0.y / inv_i.y, w0.z / inv_i.z};
    const float scale = MaxAbs(m0);
    if (!FiniteVector(m0)) {
        out.status = kBodyGyroInvalidInput;
        out.residual = INFINITY;
        return out;
    }
    if (scale == 0.0f) return out;
    const math::Vec3 u0 = mg::Scale(m0, 1.0f / scale);
    const math::Vec3 a = mg::Scale(inv_i, scale);
    const math::Vec3 c{dt * (a.y - a.z), dt * (a.z - a.x), dt * (a.x - a.y)};
    auto residual = [&](math::Vec3 u) {
        const math::Vec3 mid = mg::Scale(mg::Add(u0, u), 0.5f);
        return math::Vec3{u.x - u0.x + c.x * mid.y * mid.z,
                          u.y - u0.y + c.y * mid.z * mid.x,
                          u.z - u0.z + c.z * mid.x * mid.y};
    };
    constexpr float tolerance = 1.0e-6f;
    constexpr uint32_t max_iterations = 12u, max_backtracks = 8u;
    math::Vec3 u = u0;
    for (uint32_t iteration = 0u; iteration <= max_iterations; ++iteration) {
        const math::Vec3 r = residual(u);
        out.residual = FiniteVector(r) ? MaxAbs(r) : INFINITY;
        if (out.residual <= tolerance) {
            const math::Vec3 mid = mg::Scale(mg::Add(u0, u), 0.5f);
            out.midpoint = {a.x * mid.x, a.y * mid.y, a.z * mid.z};
            return out;
        }
        if (iteration == max_iterations || !isfinite(out.residual)) break;
        const math::Vec3 mid = mg::Scale(mg::Add(u0, u), 0.5f);
        float j[3][4] = {{1.0f, 0.5f * c.x * mid.z, 0.5f * c.x * mid.y, r.x},
                         {0.5f * c.y * mid.z, 1.0f, 0.5f * c.y * mid.x, r.y},
                         {0.5f * c.z * mid.y, 0.5f * c.z * mid.x, 1.0f, r.z}};
        bool valid = true;
        for (uint32_t column = 0u; column < 3u; ++column) {
            uint32_t pivot = column;
            for (uint32_t row = column + 1u; row < 3u; ++row)
                if (fabsf(j[row][column]) > fabsf(j[pivot][column])) pivot = row;
            if (!isfinite(j[pivot][column]) || fabsf(j[pivot][column]) < 1.0e-12f) {
                valid = false;
                break;
            }
            for (uint32_t col = column; col < 4u; ++col) {
                const float value = j[column][col];
                j[column][col] = j[pivot][col];
                j[pivot][col] = value;
            }
            for (uint32_t row = column + 1u; row < 3u; ++row) {
                const float factor = j[row][column] / j[column][column];
                for (uint32_t col = column + 1u; col < 4u; ++col)
                    j[row][col] -= factor * j[column][col];
            }
        }
        ++out.iterations;
        if (!valid) break;
        math::Vec3 delta;
        delta.z = j[2][3] / j[2][2];
        delta.y = (j[1][3] - j[1][2] * delta.z) / j[1][1];
        delta.x = (j[0][3] - j[0][1] * delta.y - j[0][2] * delta.z) / j[0][0];
        bool accepted = false;
        float alpha = 1.0f;
        for (uint32_t backtrack = 0u; backtrack < max_backtracks; ++backtrack) {
            const math::Vec3 trial = mg::Sub(u, mg::Scale(delta, alpha));
            const math::Vec3 trial_r = residual(trial);
            if (FiniteVector(trial_r) && MaxAbs(trial_r) < out.residual) {
                u = trial;
                accepted = true;
                break;
            }
            alpha *= 0.5f;
        }
        if (!accepted) break;
    }
    out.status = kBodyGyroNotConverged;
    return out;
}

}  // namespace nuka::phi::nkops
