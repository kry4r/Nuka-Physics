#pragma once
// ---------------------------------------------------------------------------
// nuka::collision -- scene query utilities (ray, overlap, sweep)
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "math/vec3.hpp"

#include <optional>

namespace nuka::collision {

struct RayHit {
    float       distance;
    math::Vec3  normal;
    uint32_t    shape_index = ~0u;
};

/// Raycast against the XZ ground plane (y = 0, normal = +Y).
std::optional<RayHit> RaycastPlane(math::Vec3 origin, math::Vec3 direction);

/// Raycast against a sphere centred at `center` with given `radius`.
std::optional<RayHit> RaycastSphere(math::Vec3 origin, math::Vec3 direction,
                                     math::Vec3 center, float radius);

/// Raycast against an axis-aligned bounding box.
std::optional<RayHit> RaycastAABB(math::Vec3 origin, math::Vec3 direction,
                                   const AABB& box);

} // namespace nuka::collision
