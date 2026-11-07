#pragma once

#include <cfloat>
#include <cmath>

#include "math/vec3.hpp"

#if defined(__CUDACC__)
#define NUKA_CONTACT_HD __host__ __device__
#else
#define NUKA_CONTACT_HD
#endif

namespace nuka::collision {

constexpr float kContactQueryReachMin = 0.02f;

// The query band covers particle thickness and relative motion at contact detection.
NUKA_CONTACT_HD inline float ContactQueryReach(float thickness, float margin) {
    return fmaxf(kContactQueryReachMin, 2.0f * (2.0f * thickness + margin));
}

NUKA_CONTACT_HD inline float ContactQueryRadius(float radius, float reach) {
    return fminf(radius, reach);
}

NUKA_CONTACT_HD inline float ContactMotionRadius(float base, float dt, float speed_a,
                                                float speed_b, float relaxation) {
    return base + dt * (speed_a + speed_b) * (2.0f / relaxation);
}

// Roundoff bound for a float subtraction, multiply and addition along a fixed axis.
NUKA_CONTACT_HD inline double ContactProjectionRoundoff(math::Vec3 start, math::Vec3 end,
                                                       math::Vec3 axis) {
    constexpr double gamma = 3.0 * FLT_EPSILON / (1.0 - 3.0 * FLT_EPSILON);
    return gamma * (fabs(double(axis.x)) * (fabs(double(start.x)) + fabs(double(end.x))) +
        fabs(double(axis.y)) * (fabs(double(start.y)) + fabs(double(end.y))) +
        fabs(double(axis.z)) * (fabs(double(start.z)) + fabs(double(end.z))));
}

}  // namespace nuka::collision

#undef NUKA_CONTACT_HD
