#pragma once
// ---------------------------------------------------------------------------
// nuka::math::Quat – unit quaternion (w-first convention)
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include <cmath>

namespace nuka::math {

struct Quat {
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    // -- constructors -------------------------------------------------------
    constexpr Quat() = default;
    constexpr Quat(float w_, float x_, float y_, float z_)
        : w(w_), x(x_), y(y_), z(z_) {}

    // -- factories ----------------------------------------------------------
    static constexpr Quat Identity() { return {1.0f, 0.0f, 0.0f, 0.0f}; }

    static inline Quat FromAxisAngle(Vec3 axis, float angle) {
        const float half = angle * 0.5f;
        const float s = std::sin(half);
        const float c = std::cos(half);
        const Vec3 a = axis.Normalized();
        return {c, a.x * s, a.y * s, a.z * s};
    }

    // -- operations ---------------------------------------------------------

    /// Hamilton product: this * rhs
    constexpr Quat operator*(const Quat& rhs) const {
        return {
            w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,
            w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
            w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
            w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w
        };
    }

    constexpr Quat Conjugate() const {
        return {w, -x, -y, -z};
    }

    inline float Norm() const {
        return std::sqrt(w * w + x * x + y * y + z * z);
    }

    inline Quat Normalized() const {
        const float n = Norm();
        if (n < 1e-12f) return Identity();
        return {w / n, x / n, y / n, z / n};
    }

    /// Rotate a vector v by this quaternion: q * v * q^-1
    inline Vec3 Rotate(Vec3 v) const {
        // Optimised quaternion-vector rotation (avoids full quat multiply)
        const Vec3 qv{x, y, z};
        const Vec3 t = 2.0f * qv.Cross(v);
        return v + w * t + qv.Cross(t);
    }

    // -- comparison ---------------------------------------------------------
    constexpr bool operator==(const Quat& rhs) const {
        return w == rhs.w && x == rhs.x && y == rhs.y && z == rhs.z;
    }
    constexpr bool operator!=(const Quat& rhs) const {
        return !(*this == rhs);
    }
};

} // namespace nuka::math
