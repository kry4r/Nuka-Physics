#pragma once

#include <cfloat>
#include <cmath>

#include "math/vec3.hpp"

#if defined(__CUDACC__)
#define NUKA_BEND_HD __host__ __device__
#else
#define NUKA_BEND_HD
#endif

namespace nuka::constraint {

struct DihedralBend {
    float angle = 0.0f;
    math::Vec3 gradients[4]{};
    bool valid = false;
};

// The shared edge is (a,b); c and d are the opposite vertices of its two faces.
// atan2 retains a finite derivative at the flat rest configuration.
NUKA_BEND_HD inline DihedralBend EvaluateDihedralBend(
    math::Vec3 a, math::Vec3 b, math::Vec3 c, math::Vec3 d) {
    DihedralBend out;
    const math::Vec3 edge = b - a;
    const math::Vec3 first = c - a;
    const math::Vec3 second = d - a;
    const math::Vec3 n0 = edge.Cross(first);
    const math::Vec3 n1 = second.Cross(edge);
    const float edge2 = edge.LengthSq();
    const float n02 = n0.LengthSq();
    const float n12 = n1.LengthSq();
    constexpr float angular_epsilon2 = FLT_EPSILON * FLT_EPSILON;
    if (!(edge2 > 0.0f &&
          n02 > angular_epsilon2 * edge2 * first.LengthSq() &&
          n12 > angular_epsilon2 * edge2 * second.LengthSq())) return out;
    const float length = sqrtf(edge2);
    out.angle = atan2f((edge / length).Dot(n0.Cross(n1)), n0.Dot(n1));
    out.gradients[2] = n0 * (-length / n02);
    out.gradients[3] = n1 * (-length / n12);
    out.gradients[1] = out.gradients[2] * (-first.Dot(edge) / edge2) +
                       out.gradients[3] * (-second.Dot(edge) / edge2);
    out.gradients[0] = -out.gradients[1] - out.gradients[2] - out.gradients[3];
    out.valid = true;
    return out;
}

NUKA_BEND_HD inline float DihedralBendError(float angle, float rest_angle) {
    const float delta = angle - rest_angle;
    return atan2f(sinf(delta), cosf(delta));
}

}  // namespace nuka::constraint

#undef NUKA_BEND_HD
