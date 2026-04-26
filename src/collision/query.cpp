// ---------------------------------------------------------------------------
// Scene query implementations
// ---------------------------------------------------------------------------

#include "collision/query.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>

namespace nuka::collision {

// ---------------------------------------------------------------------------
// Ground plane (y = 0, normal = +Y)
// ---------------------------------------------------------------------------

std::optional<RayHit> RaycastPlane(math::Vec3 origin, math::Vec3 direction) {
    // Plane equation: y = 0, normal (0, 1, 0).
    if (std::abs(direction.y) < 1e-12f) {
        return std::nullopt; // parallel to the plane
    }
    float t = -origin.y / direction.y;
    if (t < 0.0f) {
        return std::nullopt; // behind the ray
    }
    return RayHit{t, {0.0f, 1.0f, 0.0f}};
}

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

std::optional<RayHit> RaycastSphere(math::Vec3 origin, math::Vec3 direction,
                                     math::Vec3 center, float radius) {
    math::Vec3 oc = origin - center;
    float a = direction.Dot(direction);
    float b = 2.0f * oc.Dot(direction);
    float c = oc.Dot(oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) {
        return std::nullopt;
    }

    float sqrt_disc = std::sqrt(discriminant);
    float t = (-b - sqrt_disc) / (2.0f * a);
    if (t < 0.0f) {
        t = (-b + sqrt_disc) / (2.0f * a);
        if (t < 0.0f) {
            return std::nullopt;
        }
    }

    math::Vec3 hit_point = origin + direction * t;
    math::Vec3 normal    = (hit_point - center).Normalized();
    return RayHit{t, normal};
}

// ---------------------------------------------------------------------------
// AABB (slab method)
// ---------------------------------------------------------------------------

std::optional<RayHit> RaycastAABB(math::Vec3 origin, math::Vec3 direction,
                                   const AABB& box) {
    float tmin = 0.0f;
    float tmax = FLT_MAX;
    int   axis_min = -1;    // which axis set tmin
    float sign_min = 1.0f;  // +1 or -1 for the face normal

    const float* o    = &origin.x;
    const float* d    = &direction.x;
    const float* bmin = &box.min.x;
    const float* bmax = &box.max.x;

    for (int i = 0; i < 3; ++i) {
        if (std::abs(d[i]) < 1e-12f) {
            if (o[i] < bmin[i] || o[i] > bmax[i]) return std::nullopt;
        } else {
            float inv_d = 1.0f / d[i];
            float t1 = (bmin[i] - o[i]) * inv_d;
            float t2 = (bmax[i] - o[i]) * inv_d;

            float sign = -1.0f; // entry face normal direction
            if (t1 > t2) {
                std::swap(t1, t2);
                sign = 1.0f;
            }

            if (t1 > tmin) {
                tmin     = t1;
                axis_min = i;
                sign_min = sign;
            }
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return std::nullopt;
        }
    }

    // Build face normal from the entry axis.
    math::Vec3 normal{};
    if (axis_min >= 0) {
        float* n = &normal.x;
        n[axis_min] = sign_min;
    }

    return RayHit{tmin, normal};
}

} // namespace nuka::collision
