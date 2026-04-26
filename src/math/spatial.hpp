#pragma once
// ---------------------------------------------------------------------------
// nuka::math – Spatial algebra types for rigid-body dynamics
//
// SpatialVector  : 6D vector  (angular[3] + linear[3])
// SpatialTransform : 6D rigid transform for Pluecker coordinates
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "math/quat.hpp"

namespace nuka::math {

// ---------------------------------------------------------------------------
// SpatialVector – 6-element vector (angular, linear)
// ---------------------------------------------------------------------------
struct SpatialVector {
    Vec3 angular{};
    Vec3 linear{};

    constexpr SpatialVector() = default;
    constexpr SpatialVector(Vec3 ang, Vec3 lin) : angular(ang), linear(lin) {}

    // -- arithmetic ---------------------------------------------------------
    constexpr SpatialVector operator+(const SpatialVector& rhs) const {
        return {angular + rhs.angular, linear + rhs.linear};
    }
    constexpr SpatialVector operator-(const SpatialVector& rhs) const {
        return {angular - rhs.angular, linear - rhs.linear};
    }
    constexpr SpatialVector operator-() const {
        return {-angular, -linear};
    }
    constexpr SpatialVector operator*(float s) const {
        return {angular * s, linear * s};
    }

    constexpr SpatialVector& operator+=(const SpatialVector& rhs) {
        angular += rhs.angular; linear += rhs.linear;
        return *this;
    }
    constexpr SpatialVector& operator-=(const SpatialVector& rhs) {
        angular -= rhs.angular; linear -= rhs.linear;
        return *this;
    }
    constexpr SpatialVector& operator*=(float s) {
        angular *= s; linear *= s;
        return *this;
    }

    /// Spatial dot product (used for power: force^T * motion).
    constexpr float Dot(const SpatialVector& rhs) const {
        return angular.Dot(rhs.angular) + linear.Dot(rhs.linear);
    }

    /// Motion cross product:  v_m x* v  (Featherstone notation).
    /// For motion vector m = (w, v0) acting on spatial vector s = (s_w, s_v):
    ///   m x* s = (w x s_w,  w x s_v + v0 x s_w)
    constexpr SpatialVector CrossMotion(const SpatialVector& s) const {
        return {
            angular.Cross(s.angular),
            angular.Cross(s.linear) + linear.Cross(s.angular)
        };
    }

    /// Force cross product:  v_m x  f  (Featherstone notation).
    /// For motion vector m = (w, v0) acting on force vector f = (n, f0):
    ///   m x f = (w x n + v0 x f0,  w x f0)
    constexpr SpatialVector CrossForce(const SpatialVector& f) const {
        return {
            angular.Cross(f.angular) + linear.Cross(f.linear),
            angular.Cross(f.linear)
        };
    }

    static constexpr SpatialVector Zero() {
        return {Vec3::Zero(), Vec3::Zero()};
    }
};

constexpr SpatialVector operator*(float s, const SpatialVector& v) {
    return v * s;
}

// ---------------------------------------------------------------------------
// SpatialTransform – rigid transform in Pluecker coordinates
//
// Represents X = (E, r) where E is rotation and r is translation.
// Transforms motion vectors as:  X * v = (E * w,  E * (v0 - r x w))
// Transforms force  vectors as:  X^* f = (E * (n - r x f0),  E * f0)
// ---------------------------------------------------------------------------
struct SpatialTransform {
    Quat rotation{Quat::Identity()};
    Vec3 translation{};

    SpatialTransform() = default;
    constexpr SpatialTransform(Quat rot, Vec3 trans)
        : rotation(rot), translation(trans) {}

    static SpatialTransform Identity() {
        return {Quat::Identity(), Vec3::Zero()};
    }

    /// Transform a motion vector.
    inline SpatialVector TransformMotion(const SpatialVector& v) const {
        const Vec3 Ew = rotation.Rotate(v.angular);
        const Vec3 Ev = rotation.Rotate(v.linear - translation.Cross(v.angular));
        return {Ew, Ev};
    }

    /// Transform a force vector (dual / co-vector transform).
    inline SpatialVector TransformForce(const SpatialVector& f) const {
        const Vec3 Ef0 = rotation.Rotate(f.linear);
        const Vec3 En  = rotation.Rotate(f.angular - translation.Cross(f.linear));
        return {En, Ef0};
    }

    /// Compose two spatial transforms: this * rhs.
    inline SpatialTransform operator*(const SpatialTransform& rhs) const {
        return {
            (rotation * rhs.rotation).Normalized(),
            rhs.translation + Quat{rhs.rotation.Conjugate()}.Rotate(translation)
        };
    }

    /// Inverse spatial transform.
    inline SpatialTransform Inverse() const {
        const Quat invRot = rotation.Conjugate().Normalized();
        const Vec3 invTrans = -(rotation.Rotate(translation));
        return {invRot, invTrans};
    }
};

} // namespace nuka::math
