#pragma once
// ---------------------------------------------------------------------------
// nuka::math::Vec3 – 3-component float vector
// ---------------------------------------------------------------------------

#include <cmath>

#if defined(__CUDACC__)
#define NUKA_MATH_HD __host__ __device__
#else
#define NUKA_MATH_HD
#endif

namespace nuka::math {

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    // -- constructors -------------------------------------------------------
    NUKA_MATH_HD constexpr Vec3() = default;
    NUKA_MATH_HD constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // -- arithmetic ---------------------------------------------------------
    NUKA_MATH_HD constexpr Vec3 operator+(const Vec3& rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }
    NUKA_MATH_HD constexpr Vec3 operator-(const Vec3& rhs) const {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }
    NUKA_MATH_HD constexpr Vec3 operator-() const {
        return {-x, -y, -z};
    }
    NUKA_MATH_HD constexpr Vec3 operator*(float s) const {
        return {x * s, y * s, z * s};
    }
    NUKA_MATH_HD constexpr Vec3 operator/(float s) const {
        return {x / s, y / s, z / s};
    }

    NUKA_MATH_HD constexpr Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x; y += rhs.y; z += rhs.z;
        return *this;
    }
    NUKA_MATH_HD constexpr Vec3& operator-=(const Vec3& rhs) {
        x -= rhs.x; y -= rhs.y; z -= rhs.z;
        return *this;
    }
    NUKA_MATH_HD constexpr Vec3& operator*=(float s) {
        x *= s; y *= s; z *= s;
        return *this;
    }
    NUKA_MATH_HD constexpr Vec3& operator/=(float s) {
        x /= s; y /= s; z /= s;
        return *this;
    }

    // -- products -----------------------------------------------------------
    NUKA_MATH_HD constexpr float Dot(const Vec3& rhs) const {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    NUKA_MATH_HD constexpr Vec3 Cross(const Vec3& rhs) const {
        return {
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        };
    }

    // -- length / normalisation ---------------------------------------------
    NUKA_MATH_HD constexpr float LengthSq() const { return Dot(*this); }

    inline float Length() const { return std::sqrt(LengthSq()); }

    inline Vec3 Normalized() const {
        const float len = Length();
        if (len < 1e-12f) return {};
        return *this / len;
    }

    // -- comparison ---------------------------------------------------------
    NUKA_MATH_HD constexpr bool operator==(const Vec3& rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
    NUKA_MATH_HD constexpr bool operator!=(const Vec3& rhs) const {
        return !(*this == rhs);
    }

    // -- named constants ----------------------------------------------------
    NUKA_MATH_HD static constexpr Vec3 Zero()  { return {0.0f, 0.0f, 0.0f}; }
    NUKA_MATH_HD static constexpr Vec3 UnitX() { return {1.0f, 0.0f, 0.0f}; }
    NUKA_MATH_HD static constexpr Vec3 UnitY() { return {0.0f, 1.0f, 0.0f}; }
    NUKA_MATH_HD static constexpr Vec3 UnitZ() { return {0.0f, 0.0f, 1.0f}; }
};

// scalar * vec (commutative convenience)
NUKA_MATH_HD constexpr Vec3 operator*(float s, const Vec3& v) { return v * s; }

} // namespace nuka::math

#undef NUKA_MATH_HD
