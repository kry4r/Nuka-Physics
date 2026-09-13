#pragma once

#include <cmath>
#include <cstdint>

#include "collision/shape_kind.hpp"
#include "math/vec3.hpp"

#if defined(__CUDACC__)
#define NUKA_SURFACE_HD __host__ __device__
#else
#define NUKA_SURFACE_HD
#endif

namespace nuka::collision {

// Local signed distance is positive outside; normal points out of the solid.
struct PrimitiveSurface {
    float distance = 0.0f;
    math::Vec3 normal{1.0f, 0.0f, 0.0f};
    math::Vec3 point{};
    uint32_t feature = 0u;
    bool valid = false;
};

// Capsules extend along local Z. Planes bound the solid half-space local z <= 0.
// Parameters are radius, (radius, half-height), or box half-extents.
NUKA_SURFACE_HD inline PrimitiveSurface QueryPrimitiveSurface(
    uint32_t kind, math::Vec3 parameters, math::Vec3 position) {
    using math::Vec3;
    PrimitiveSurface result;
    switch (kind) {
        case kShapeSphere:
        case kShapeCapsule: {
            if (!(parameters.x >= 0.0f)) return result;
            Vec3 offset = position;
            if (kind == kShapeCapsule) {
                if (!(parameters.y >= 0.0f)) return result;
                const float axis = fmaxf(-parameters.y, fminf(position.z, parameters.y));
                offset.z -= axis;
                result.feature = position.z < -parameters.y ? 0u
                    : position.z > parameters.y ? 2u : 1u;
            }
            const float length = sqrtf(offset.Dot(offset));
            if (length > 0.0f) result.normal = offset / length;
            result.distance = length - parameters.x;
            break;
        }
        case kShapeBox: {
            if (!(parameters.x >= 0.0f && parameters.y >= 0.0f && parameters.z >= 0.0f))
                return result;
            const Vec3 closest{
                fmaxf(-parameters.x, fminf(position.x, parameters.x)),
                fmaxf(-parameters.y, fminf(position.y, parameters.y)),
                fmaxf(-parameters.z, fminf(position.z, parameters.z))};
            const Vec3 offset = position - closest;
            const float length = sqrtf(offset.Dot(offset));
            if (length > 0.0f) {
                result.distance = length;
                result.normal = offset / length;
                if (offset.x != 0.0f) result.feature |= 1u << (position.x < 0.0f ? 0u : 1u);
                if (offset.y != 0.0f) result.feature |= 1u << (position.y < 0.0f ? 2u : 3u);
                if (offset.z != 0.0f) result.feature |= 1u << (position.z < 0.0f ? 4u : 5u);
            } else {
                const Vec3 distance{fabsf(position.x) - parameters.x,
                                    fabsf(position.y) - parameters.y,
                                    fabsf(position.z) - parameters.z};
                if (distance.x >= distance.y && distance.x >= distance.z) {
                    result.distance = distance.x;
                    result.normal = {position.x < 0.0f ? -1.0f : 1.0f, 0.0f, 0.0f};
                    result.feature = 1u << (position.x < 0.0f ? 0u : 1u);
                } else if (distance.y >= distance.z) {
                    result.distance = distance.y;
                    result.normal = {0.0f, position.y < 0.0f ? -1.0f : 1.0f, 0.0f};
                    result.feature = 1u << (position.y < 0.0f ? 2u : 3u);
                } else {
                    result.distance = distance.z;
                    result.normal = {0.0f, 0.0f, position.z < 0.0f ? -1.0f : 1.0f};
                    result.feature = 1u << (position.z < 0.0f ? 4u : 5u);
                }
            }
            break;
        }
        case kShapePlane:
            result.distance = position.z;
            result.normal = {0.0f, 0.0f, 1.0f};
            break;
        default:
            return result;
    }
    result.point = position - result.normal * result.distance;
    result.valid = true;
    return result;
}

}  // namespace nuka::collision

#undef NUKA_SURFACE_HD
