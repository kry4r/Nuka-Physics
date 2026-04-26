#pragma once
// ---------------------------------------------------------------------------
// nuka::math::Transform – rigid body transform (rotation + translation)
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "math/quat.hpp"

namespace nuka::math {

struct Transform {
    Vec3 position{};
    Quat rotation{Quat::Identity()};

    // -- constructors -------------------------------------------------------
    Transform() = default;
    constexpr Transform(Vec3 pos, Quat rot) : position(pos), rotation(rot) {}

    // -- factories ----------------------------------------------------------
    static Transform Identity() {
        return {Vec3::Zero(), Quat::Identity()};
    }

    // -- operations ---------------------------------------------------------

    /// Transform a point: rotate then translate.
    inline Vec3 TransformPoint(Vec3 p) const {
        return rotation.Rotate(p) + position;
    }

    /// Transform a direction (rotation only, no translation).
    inline Vec3 TransformDirection(Vec3 d) const {
        return rotation.Rotate(d);
    }

    /// Compose transforms: this * rhs  (apply rhs first, then this).
    inline Transform operator*(const Transform& rhs) const {
        return {
            TransformPoint(rhs.position),
            (rotation * rhs.rotation).Normalized()
        };
    }

    /// Return the inverse transform.
    inline Transform Inverse() const {
        const Quat invRot = rotation.Conjugate().Normalized();
        const Vec3 invPos = -(Quat{invRot}.Rotate(position));
        return {invPos, invRot};
    }
};

} // namespace nuka::math
