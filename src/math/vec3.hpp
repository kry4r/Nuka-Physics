#pragma once
// ---------------------------------------------------------------------------
// nuka::math::Vec3 – 3-component float vector
// ---------------------------------------------------------------------------

#include <cmath>

namespace nuka::math {

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    // -- constructors -------------------------------------------------------
    constexpr Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // -- arithmetic ---------------------------------------------------------
    constexpr Vec3 operator+(const Vec3& rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }
    constexpr Vec3 operator-(const Vec3& rhs) const {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }
    constexpr Vec3 operator-() const {
        return {-x, -y, -z};
    }
    constexpr Vec3 operator*(float s) const {
        return {x * s, y * s, z * s};
    }
    constexpr Vec3 operator/(float s) const {
        return {x / s, y / s, z / s};
    }

    constexpr Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x; y += rhs.y; z += rhs.z;
        return *this;
    }
    constexpr Vec3& operator-=(const Vec3& rhs) {
        x -= rhs.x; y -= rhs.y; z -= rhs.z;
        return *this;
    }
    constexpr Vec3& operator*=(float s) {
        x *= s; y *= s; z *= s;
        return *this;
    }
    constexpr Vec3& operator/=(float s) {
        x /= s; y /= s; z /= s;
        return *this;
    }

    // -- products -----------------------------------------------------------
    constexpr float Dot(const Vec3& rhs) const {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    constexpr Vec3 Cross(const Vec3& rhs) const {
        return {
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        };
    }

    // -- length / normalisation ---------------------------------------------
    constexpr float LengthSq() const { return Dot(*this); }

    inline float Length() const { return std::sqrt(LengthSq()); }

    inline Vec3 Normalized() const {
        const float len = Length();
        if (len < 1e-12f) return {};
        return *this / len;
    }

    // -- comparison ---------------------------------------------------------
    constexpr bool operator==(const Vec3& rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
    constexpr bool operator!=(const Vec3& rhs) const {
        return !(*this == rhs);
    }

    // -- named constants ----------------------------------------------------
    static constexpr Vec3 Zero()  { return {0.0f, 0.0f, 0.0f}; }
    static constexpr Vec3 UnitX() { return {1.0f, 0.0f, 0.0f}; }
    static constexpr Vec3 UnitY() { return {0.0f, 1.0f, 0.0f}; }
    static constexpr Vec3 UnitZ() { return {0.0f, 0.0f, 1.0f}; }
};

// scalar * vec (commutative convenience)
constexpr Vec3 operator*(float s, const Vec3& v) { return v * s; }

} // namespace nuka::math
