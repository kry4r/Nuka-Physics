#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint::ContactManifold -- contact point storage
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>

namespace nuka::constraint {

struct ContactPoint {
    math::Vec3 position;
    math::Vec3 normal;
    float penetration       = 0.0f;
    float normal_impulse    = 0.0f;   // for warm start
    float friction_impulse_1 = 0.0f;
    float friction_impulse_2 = 0.0f;
};

struct ContactManifold {
    uint32_t body_a      = ~0u;
    uint32_t body_b      = ~0u;
    uint32_t point_count = 0;

    static constexpr uint32_t kMaxPoints = 4;
    ContactPoint points[kMaxPoints];

    /// Add a contact point. If the manifold is full the point is dropped.
    void AddPoint(const ContactPoint& pt) {
        if (point_count < kMaxPoints) {
            points[point_count] = pt;
            ++point_count;
        }
    }

    /// Reset all points.
    void Clear() {
        point_count = 0;
    }
};

} // namespace nuka::constraint
