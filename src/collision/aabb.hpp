#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::AABB -- Axis-aligned bounding box
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "math/transform.hpp"

#include <algorithm>
#include <cfloat>

namespace nuka::collision {

struct AABB {
    math::Vec3 min = {FLT_MAX, FLT_MAX, FLT_MAX};
    math::Vec3 max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

    // -- mutators -----------------------------------------------------------

    /// Expand the AABB to include the given point.
    void Expand(math::Vec3 point) {
        min.x = std::min(min.x, point.x);
        min.y = std::min(min.y, point.y);
        min.z = std::min(min.z, point.z);
        max.x = std::max(max.x, point.x);
        max.y = std::max(max.y, point.y);
        max.z = std::max(max.z, point.z);
    }

    /// Merge another AABB into this one.
    void Merge(const AABB& other) {
        min.x = std::min(min.x, other.min.x);
        min.y = std::min(min.y, other.min.y);
        min.z = std::min(min.z, other.min.z);
        max.x = std::max(max.x, other.max.x);
        max.y = std::max(max.y, other.max.y);
        max.z = std::max(max.z, other.max.z);
    }

    // -- queries ------------------------------------------------------------

    /// Test overlap with another AABB.
    bool Overlaps(const AABB& other) const {
        if (max.x < other.min.x || min.x > other.max.x) return false;
        if (max.y < other.min.y || min.y > other.max.y) return false;
        if (max.z < other.min.z || min.z > other.max.z) return false;
        return true;
    }

    /// Return the center of the AABB.
    math::Vec3 Center() const {
        return (min + max) * 0.5f;
    }

    /// Return the half-extents of the AABB.
    math::Vec3 Extents() const {
        return (max - min) * 0.5f;
    }

    // -- factories ----------------------------------------------------------

    /// Create an AABB enclosing a sphere.
    static AABB FromSphere(math::Vec3 center, float radius) {
        AABB result;
        result.min = center - math::Vec3{radius, radius, radius};
        result.max = center + math::Vec3{radius, radius, radius};
        return result;
    }

    /// Create an AABB enclosing an oriented box.
    static AABB FromBox(math::Transform transform, math::Vec3 half_extents) {
        AABB result;
        // Enumerate all 8 corners of the oriented box and expand.
        for (int i = 0; i < 8; ++i) {
            math::Vec3 corner{
                (i & 1) ? half_extents.x : -half_extents.x,
                (i & 2) ? half_extents.y : -half_extents.y,
                (i & 4) ? half_extents.z : -half_extents.z
            };
            result.Expand(transform.TransformPoint(corner));
        }
        return result;
    }
};

} // namespace nuka::collision
