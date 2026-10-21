#pragma once

#include <cfloat>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NUKA_FINITE_HD __host__ __device__
#else
#define NUKA_FINITE_HD
#endif

namespace nuka::math {

// Ordered bounds reject infinities and NaNs without a host-library classification call.
NUKA_FINITE_HD inline bool IsFinite(float value) {
    return value >= -FLT_MAX && value <= FLT_MAX;
}

NUKA_FINITE_HD inline bool IsFinite(double value) {
    return value >= -DBL_MAX && value <= DBL_MAX;
}

}  // namespace nuka::math

#undef NUKA_FINITE_HD
