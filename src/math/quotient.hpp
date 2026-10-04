#pragma once

#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define NUKA_QUOTIENT_HD __host__ __device__
#else
#define NUKA_QUOTIENT_HD
#endif

namespace nuka::math {

// The IEEE quotient n / d. Only a nonzero numerator over a finite nonzero divisor is divided;
// the remaining cases are exact products, which keeps device division off its software path.
NUKA_QUOTIENT_HD inline double Quotient(double n, double d) {
    const bool divisor_regular = fabs(d) > 0.0 && fabs(d) <= DBL_MAX;
    const double q = (n != 0.0 ? n : 1.0) / (divisor_regular ? d : 1.0);
    const double factor = d == 0.0 ? copysign(INFINITY, d)
        : divisor_regular ? copysign(1.0, d) : d == d ? copysign(0.0, d) : d;
    return n != 0.0 && divisor_regular ? q : n * factor;
}

}  // namespace nuka::math

#undef NUKA_QUOTIENT_HD
