#pragma once

#include <cfloat>
#include <cstdint>

#include "math/vec3.hpp"

#if defined(__CUDACC__)
#define NUKA_CHEBYSHEV_HD __host__ __device__
#else
#define NUKA_CHEBYSHEV_HD
#endif

namespace nuka::nk::solve {

NUKA_CHEBYSHEV_HD inline float ChebyshevIterationRatio(
    uint32_t iteration, float previous, float spectral_radius) {
    if (!(spectral_radius > 0.0f && spectral_radius < 1.0f) || iteration == 0u)
        return 1.0f;
    const double radius = static_cast<double>(spectral_radius);
    const double radius_squared = radius * radius;
    const double numerator = iteration == 1u ? 2.0 : 4.0;
    const double denominator = iteration == 1u ? 2.0 - radius_squared
        : 4.0 - radius_squared * static_cast<double>(previous);
    if (!(denominator > 0.0)) return 1.0f;
    const double ratio = numerator / denominator;
    if (!(ratio >= -static_cast<double>(FLT_MAX) && ratio <= static_cast<double>(FLT_MAX)))
        return 1.0f;
    return static_cast<float>(ratio);
}

NUKA_CHEBYSHEV_HD inline math::Vec3 ChebyshevExtrapolate(
    math::Vec3 current, math::Vec3 older, float ratio) {
    if (ratio == 1.0f) return current;
    return older + (current - older) * ratio;
}

NUKA_CHEBYSHEV_HD inline bool ChebyshevAccept(double candidate_merit, double descent_merit) {
    return candidate_merit >= 0.0 && candidate_merit <= descent_merit && descent_merit <= DBL_MAX;
}

}  // namespace nuka::nk::solve

#undef NUKA_CHEBYSHEV_HD
