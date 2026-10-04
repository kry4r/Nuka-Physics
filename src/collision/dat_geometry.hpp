#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "collision/ogc_geometry.hpp"

#if defined(__CUDACC__)
#define NUKA_DAT_HD __host__ __device__
#else
#define NUKA_DAT_HD
#endif

namespace nuka::collision {

constexpr uint32_t kDatPrimitiveVertices = 3u;
constexpr float kDatQueryRadiusMax = 0.02f;
constexpr uint32_t kDatFailureRadius = 1u, kDatFailureMotion = 2u, kDatFailureDegenerate = 3u,
                   kDatFailureOverlap = 4u, kDatFailureReasons = 4u, kDatWitnessSlots = 128u;
// Words: reason totals, unrecorded events, then slots of (reason << 60 | owner << 30 | owner,
// count, deepest overlap as float bits).
constexpr uint32_t kDatWitnessWords = kDatFailureReasons + 1u + 3u * kDatWitnessSlots;
// Owner codes hold the kind in bits 28-29 above an env-local index; all ones means no owner.
constexpr uint32_t kDatWitnessBody = 0u, kDatWitnessLink = 1u, kDatWitnessSurface = 2u,
                   kDatWitnessStatic = 3u, kDatWitnessNone = 0x3FFFFFFFu;

NUKA_DAT_HD inline uint32_t DatWitnessOwner(uint32_t kind, uint32_t index) {
    return kind <= kDatWitnessStatic && index < 0x0FFFFFFFu ? (kind << 28u) | index
                                                            : kDatWitnessNone;
}

NUKA_DAT_HD inline float DatQueryRadius(float radius) {
    return fminf(radius, kDatQueryRadiusMax);
}

// Uncapped mixed query radius whose relaxed half covers each side's predicted step displacement.
NUKA_DAT_HD inline float DatMotionRadius(float base, float dt, float speed_a, float speed_b,
                                         float relaxation) {
    return base + dt * (speed_a + speed_b) * (2.0f / relaxation);
}

struct DatPrimitive {
    math::Vec3 vertex[kDatPrimitiveVertices]{};
    uint32_t count = 0u;
};

struct DatSeparator {
    math::Vec3 normal{};
    math::Vec3 negative_support{};
    float gap = 0.0f;
};

struct DatSweptPrimitive {
    math::Vec3 vertex[2u * kDatPrimitiveVertices]{};
    uint32_t count = 0u;
};

// Float intervals decide the certificate below when they clear its threshold by a slack that
// bounds every float and double rounding difference; -1 leaves the decision to double precision.
NUKA_DAT_HD inline int DatSweptAxisFilter(math::Vec3 axis, const DatSweptPrimitive& a,
                                          const DatSweptPrimitive& b, float minimum_gap) {
    float minimum[2] = {FLT_MAX, FLT_MAX}, maximum[2] = {-FLT_MAX, -FLT_MAX};
    float scale = 0.0f;
    const math::Vec3 origin = b.vertex[0];
    constexpr float interpolation_roundoff = 3.0f * FLT_EPSILON / (1.0f - 3.0f * FLT_EPSILON);
    for (uint32_t side = 0u; side < 2u; ++side) {
        const DatSweptPrimitive& primitive = side == 0u ? a : b;
        const uint32_t half = primitive.count / 2u;
        for (uint32_t i = 0u; i < primitive.count; ++i) {
            const auto point = primitive.vertex[i];
            const auto start = primitive.vertex[i % half];
            const auto end = primitive.vertex[i % half + half];
            const float x = (point.x - origin.x) * axis.x;
            const float y = (point.y - origin.y) * axis.y;
            const float z = (point.z - origin.z) * axis.z;
            const float value = (x + y) + z;
            const float interpolation = interpolation_roundoff * (
                fabsf(axis.x) * (fabsf(start.x) + fabsf(end.x)) +
                fabsf(axis.y) * (fabsf(start.y) + fabsf(end.y)) +
                fabsf(axis.z) * (fabsf(start.z) + fabsf(end.z)));
            minimum[side] = fminf(minimum[side], value - interpolation);
            maximum[side] = fmaxf(maximum[side], value + interpolation);
            scale = fmaxf(scale, fabsf(x) + fabsf(y) + fabsf(z) + interpolation);
        }
    }
    const float required = minimum_gap * sqrtf(axis.LengthSq());
    const float slack = 64.0f * (FLT_EPSILON * (scale + required) + FLT_MIN);
    const float gap0 = minimum[0] - maximum[1], gap1 = minimum[1] - maximum[0];
    if (!(fabsf(gap0) <= FLT_MAX && fabsf(gap1) <= FLT_MAX && slack <= FLT_MAX)) return -1;
    if (gap0 - slack > required || gap1 - slack > required) return 1;
    if (gap0 + slack < required && gap1 + slack < required) return 0;
    return -1;
}

// Outward projection intervals certify a fixed plane for every convex combination of the endpoints.
NUKA_DAT_HD inline bool DatSweptAxisSeparates(math::Vec3 axis, const DatSweptPrimitive& a,
                                              const DatSweptPrimitive& b, float minimum_gap) {
    if (!(axis.LengthSq() > 0.0f && axis.LengthSq() <= FLT_MAX)) return false;
    const int filtered = DatSweptAxisFilter(axis, a, b, minimum_gap);
    if (filtered >= 0) return filtered == 1;
    double minimum[2] = {DBL_MAX, DBL_MAX}, maximum[2] = {-DBL_MAX, -DBL_MAX};
    const math::Vec3 origin = b.vertex[0];
    constexpr double roundoff = 8.0 * DBL_EPSILON / (1.0 - 8.0 * DBL_EPSILON);
    constexpr double interpolation_roundoff = 3.0 * FLT_EPSILON / (1.0 - 3.0 * FLT_EPSILON);
    for (uint32_t side = 0u; side < 2u; ++side) {
        const DatSweptPrimitive& primitive = side == 0u ? a : b;
        for (uint32_t i = 0u; i < primitive.count; ++i) {
            const auto point = primitive.vertex[i];
            const uint32_t half = primitive.count / 2u;
            const auto start = primitive.vertex[i % half];
            const auto end = primitive.vertex[i % half + half];
            const double x = (double(point.x) - double(origin.x)) * double(axis.x);
            const double y = (double(point.y) - double(origin.y)) * double(axis.y);
            const double z = (double(point.z) - double(origin.z)) * double(axis.z);
            const double value = (x + y) + z;
            const double interpolation_error = interpolation_roundoff * (
                fabs(double(axis.x)) * (fabs(double(start.x)) + fabs(double(end.x))) +
                fabs(double(axis.y)) * (fabs(double(start.y)) + fabs(double(end.y))) +
                fabs(double(axis.z)) * (fabs(double(start.z)) + fabs(double(end.z))));
            const double error = roundoff * (fabs(x) + fabs(y) + fabs(z)) + interpolation_error;
            minimum[side] = fmin(minimum[side], value - error);
            maximum[side] = fmax(maximum[side], value + error);
        }
    }
    const double axis_length = sqrt(double(axis.x) * axis.x + double(axis.y) * axis.y +
                                    double(axis.z) * axis.z);
    const double required_gap = double(minimum_gap) * axis_length * (1.0 + roundoff);
    return minimum[0] - maximum[1] > required_gap || minimum[1] - maximum[0] > required_gap;
}

// Candidate axes come from the endpoint hulls; an unsuccessful search keeps the original DAT guard.
NUKA_DAT_HD inline bool DatSweptSeparated(const DatSweptPrimitive& a,
                                          const DatSweptPrimitive& b, math::Vec3 first_axis,
                                          float minimum_gap) {
    if (a.count == 0u || a.count > 2u * kDatPrimitiveVertices ||
        b.count == 0u || b.count > 2u * kDatPrimitiveVertices ||
        (a.count & 1u) != 0u || (b.count & 1u) != 0u ||
        !(minimum_gap >= 0.0f && minimum_gap <= FLT_MAX)) return false;
    for (uint32_t side = 0u; side < 2u; ++side) {
        const DatSweptPrimitive& primitive = side == 0u ? a : b;
        for (uint32_t i = 0u; i < primitive.count; ++i)
            if (!(primitive.vertex[i].LengthSq() <= FLT_MAX)) return false;
    }
    if (DatSweptAxisSeparates(first_axis, a, b, minimum_gap)) return true;
    for (uint32_t i = 0u; i < a.count; ++i)
        for (uint32_t j = 0u; j < b.count; ++j)
            if (DatSweptAxisSeparates(a.vertex[i] - b.vertex[j], a, b, minimum_gap)) return true;
    for (uint32_t side = 0u; side < 2u; ++side) {
        const DatSweptPrimitive& primitive = side == 0u ? a : b;
        for (uint32_t i = 0u; i < primitive.count; ++i)
            for (uint32_t j = i + 1u; j < primitive.count; ++j)
                for (uint32_t k = j + 1u; k < primitive.count; ++k)
                    if (DatSweptAxisSeparates((primitive.vertex[j] - primitive.vertex[i]).Cross(
                            primitive.vertex[k] - primitive.vertex[i]), a, b, minimum_gap)) return true;
    }
    for (uint32_t i = 0u; i < a.count; ++i)
        for (uint32_t j = i + 1u; j < a.count; ++j)
            for (uint32_t k = 0u; k < b.count; ++k)
                for (uint32_t l = k + 1u; l < b.count; ++l)
                    if (DatSweptAxisSeparates((a.vertex[j] - a.vertex[i]).Cross(
                            b.vertex[l] - b.vertex[k]), a, b, minimum_gap)) return true;
    return false;
}

NUKA_DAT_HD inline bool DatPrimitiveValid(const DatPrimitive& p) {
    if (p.count == 0u || p.count > kDatPrimitiveVertices) return false;
    for (uint32_t i = 0u; i < p.count; ++i)
        if (!(p.vertex[i].LengthSq() <= FLT_MAX)) return false;
    if (p.count >= 2u && !((p.vertex[1] - p.vertex[0]).LengthSq() > 0.0f)) return false;
    return p.count < 3u ||
        (p.vertex[1] - p.vertex[0]).Cross(p.vertex[2] - p.vertex[0]).LengthSq() > 0.0f;
}

NUKA_DAT_HD inline void DatConsiderAxis(math::Vec3 axis,
                                       const DatPrimitive& a,
                                       const DatPrimitive& b,
                                       DatSeparator* best) {
    const float length_sq = axis.LengthSq();
    if (!(length_sq > 0.0f)) return;
    const math::Vec3 unit = axis / sqrtf(length_sq);
    for (int orientation = -1; orientation <= 1; orientation += 2) {
        const math::Vec3 normal = unit * static_cast<float>(orientation);
        const math::Vec3 origin = b.vertex[0];
        float positive_min = (a.vertex[0] - origin).Dot(normal);
        for (uint32_t i = 1u; i < a.count; ++i)
            positive_min = fminf(positive_min, (a.vertex[i] - origin).Dot(normal));
        float negative_max = 0.0f;
        math::Vec3 support = origin;
        for (uint32_t i = 1u; i < b.count; ++i) {
            const float projection = (b.vertex[i] - origin).Dot(normal);
            if (projection > negative_max) {
                negative_max = projection;
                support = b.vertex[i];
            }
        }
        const float gap = positive_min - negative_max;
        if (gap > best->gap) *best = {normal, support, gap};
    }
}

// A negative floor keeps the signed overlap of intersecting primitives.
NUKA_DAT_HD inline DatSeparator DatFindSeparator(const DatPrimitive& a,
                                                 const DatPrimitive& b, float floor = 0.0f) {
    DatSeparator result;
    result.gap = floor;
    if (!DatPrimitiveValid(a) || !DatPrimitiveValid(b)) return result;
    for (uint32_t i = 0u; i < a.count; ++i)
        for (uint32_t j = 0u; j < b.count; ++j)
            DatConsiderAxis(a.vertex[i] - b.vertex[j], a, b, &result);
    const uint32_t edges_a = a.count == 1u ? 0u : a.count == 2u ? 1u : a.count;
    const uint32_t edges_b = b.count == 1u ? 0u : b.count == 2u ? 1u : b.count;
    for (uint32_t i = 0u; i < edges_a; ++i) {
        const math::Vec3 edge = a.vertex[(i + 1u) % a.count] - a.vertex[i];
        for (uint32_t j = 0u; j < b.count; ++j) {
            const math::Vec3 offset = a.vertex[i] - b.vertex[j];
            DatConsiderAxis(edge.Cross(offset.Cross(edge)), a, b, &result);
        }
        for (uint32_t j = 0u; j < edges_b; ++j) {
            const math::Vec3 other = b.vertex[(j + 1u) % b.count] - b.vertex[j];
            DatConsiderAxis(edge.Cross(other), a, b, &result);
        }
    }
    for (uint32_t j = 0u; j < edges_b; ++j) {
        const math::Vec3 edge = b.vertex[(j + 1u) % b.count] - b.vertex[j];
        for (uint32_t i = 0u; i < a.count; ++i) {
            const math::Vec3 offset = a.vertex[i] - b.vertex[j];
            DatConsiderAxis(edge.Cross(offset.Cross(edge)), a, b, &result);
        }
    }
    if (a.count == kDatPrimitiveVertices)
        DatConsiderAxis((a.vertex[1] - a.vertex[0]).Cross(a.vertex[2] - a.vertex[0]),
                        a, b, &result);
    if (b.count == kDatPrimitiveVertices)
        DatConsiderAxis((b.vertex[1] - b.vertex[0]).Cross(b.vertex[2] - b.vertex[0]),
                        a, b, &result);
    return result;
}

}  // namespace nuka::collision

#undef NUKA_DAT_HD
