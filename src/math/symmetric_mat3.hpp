#pragma once

#include "math/quat.hpp"
#include "math/vec3.hpp"

#if defined(__CUDACC__)
#define NUKA_SYMMETRIC_HD __host__ __device__
#else
#define NUKA_SYMMETRIC_HD
#endif

namespace nuka::math {

// Symmetric tensor stored as xx, yy, zz, xy, xz, yz.
struct SymmetricMat3 {
    float xx = 0.0f, yy = 0.0f, zz = 0.0f;
    float xy = 0.0f, xz = 0.0f, yz = 0.0f;

    NUKA_SYMMETRIC_HD constexpr SymmetricMat3() = default;
    NUKA_SYMMETRIC_HD constexpr SymmetricMat3(
        float xx_, float yy_, float zz_, float xy_, float xz_, float yz_)
        : xx(xx_), yy(yy_), zz(zz_), xy(xy_), xz(xz_), yz(yz_) {}

    NUKA_SYMMETRIC_HD constexpr Vec3 Multiply(Vec3 v) const {
        return {xx * v.x + xy * v.y + xz * v.z,
                xy * v.x + yy * v.y + yz * v.z,
                xz * v.x + yz * v.y + zz * v.z};
    }

    NUKA_SYMMETRIC_HD static SymmetricMat3 FromRotatedDiagonal(Vec3 d, Quat q) {
        const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
        const float s = norm_sq > 1.0e-12f ? 2.0f / norm_sq : 0.0f;
        const Vec3 x{1.0f - s * (q.y * q.y + q.z * q.z),
                     s * (q.x * q.y - q.w * q.z), s * (q.x * q.z + q.w * q.y)};
        const Vec3 y{s * (q.x * q.y + q.w * q.z),
                     1.0f - s * (q.x * q.x + q.z * q.z), s * (q.y * q.z - q.w * q.x)};
        const Vec3 z{s * (q.x * q.z - q.w * q.y), s * (q.y * q.z + q.w * q.x),
                     1.0f - s * (q.x * q.x + q.y * q.y)};
        return {d.x * x.x * x.x + d.y * x.y * x.y + d.z * x.z * x.z,
                d.x * y.x * y.x + d.y * y.y * y.y + d.z * y.z * y.z,
                d.x * z.x * z.x + d.y * z.y * z.y + d.z * z.z * z.z,
                d.x * x.x * y.x + d.y * x.y * y.y + d.z * x.z * y.z,
                d.x * x.x * z.x + d.y * x.y * z.y + d.z * x.z * z.z,
                d.x * y.x * z.x + d.y * y.y * z.y + d.z * y.z * z.z};
    }
};

static_assert(sizeof(SymmetricMat3) == 6u * sizeof(float));

}  // namespace nuka::math

#undef NUKA_SYMMETRIC_HD
