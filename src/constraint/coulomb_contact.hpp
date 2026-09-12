#pragma once

#include <cfloat>
#include <cmath>

#include "math/symmetric_mat3.hpp"

#if defined(__CUDACC__)
#define NUKA_COULOMB_HD __host__ __device__
#else
#define NUKA_COULOMB_HD
#endif

namespace nuka::constraint {

// Normal complementarity and tangential maximum dissipation have separate projections.
// A scalar spectral step in scaled tangent coordinates preserves the friction ellipse.
NUKA_COULOMB_HD inline math::Vec3 ProjectedCoulombStep(
    const math::SymmetricMat3& response, math::Vec3 residual, math::Vec3 impulse,
    float mu_first, float mu_second) {
    if (!(response.xx > 0.0f) || !(response.xx <= FLT_MAX)) return {};
    const float normal = fmaxf(impulse.x + residual.x / response.xx, 0.0f);
    const float mu = fmaxf(mu_first, mu_second);
    if (!(normal > 0.0f) || !(mu > 0.0f)) return {normal, 0.0f, 0.0f};
    const float first_scale = fmaxf(mu_first, 0.0f) / mu;
    const float second_scale = fmaxf(mu_second, 0.0f) / mu;
    const float h11 = response.yy * first_scale * first_scale;
    const float h22 = response.zz * second_scale * second_scale;
    const float h12 = response.yz * first_scale * second_scale;
    const float largest = 0.5f * (h11 + h22 + hypotf(h11 - h22, 2.0f * h12));
    if (!(largest > FLT_MIN) || !(largest <= FLT_MAX)) return {normal, 0.0f, 0.0f};
    const float delta_normal = normal - impulse.x;
    float first = first_scale > 0.0f ? impulse.y / first_scale +
        first_scale * (residual.y - response.xy * delta_normal) / largest : 0.0f;
    float second = second_scale > 0.0f ? impulse.z / second_scale +
        second_scale * (residual.z - response.xz * delta_normal) / largest : 0.0f;
    const float length = hypotf(first, second);
    const float radius = normal * mu;
    if (length > radius) {
        const float scale = radius / length;
        first *= scale;
        second *= scale;
    }
    return {normal, first * first_scale, second * second_scale};
}

}  // namespace nuka::constraint

#undef NUKA_COULOMB_HD
