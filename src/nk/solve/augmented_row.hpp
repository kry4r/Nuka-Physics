#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "math/quotient.hpp"
#include "math/symmetric_mat3.hpp"
#include "math/vec3.hpp"

#if defined(__CUDACC__)
#define NUKA_AUGMENTED_HD __host__ __device__
#else
#define NUKA_AUGMENTED_HD
#endif

namespace nuka::nk::augmented {

struct ScalarTerm {
    float impulse = 0.0f;
    float curvature = 0.0f;
    double potential = 0.0;
};

struct ScalarResponse {
    double dual_scale;
    double residual_scale;
};

NUKA_AUGMENTED_HD inline ScalarResponse ComputeScalarResponse(float rho, float compliance) {
    const double denominator = 1.0 + double(rho) * compliance;
    const double s = 1.0 / denominator;
    const double k = double(rho) / denominator;
    return {s, k};
}

struct TangentTerm {
    math::Vec3 impulse{};
    math::SymmetricMat3 curvature{};
    double potential = 0.0;
};

// Residual is rhs * dt * damping_scale - Ju; compliance includes damping_scale.
NUKA_AUGMENTED_HD inline ScalarTerm EvaluateScalar(
    float dual, float rho, float residual, float compliance, float lower, float upper,
    const ScalarResponse& response) {
    if (rho <= 0.0f) return {};
    if (dual != dual || rho != rho || residual != residual || compliance != compliance ||
        lower != lower || upper != upper) {
        const float invalid = dual + rho + residual + compliance + lower + upper;
        return {invalid, invalid, double(invalid)};
    }
    const double s = response.dual_scale;
    const double k = response.residual_scale;
    const double x = s * dual + k * residual;
    const double f = x < lower ? double(lower) : (x > upper ? double(upper) : x);
    const double curvature = x != x ? x :
        (lower < upper && x >= lower && x <= upper ? k : 0.0);
    return {float(f), float(curvature), math::Quotient(f * (x - 0.5 * f), k)};
}

NUKA_AUGMENTED_HD inline ScalarTerm EvaluateScalar(
    float dual, float rho, float residual, float compliance, float lower, float upper) {
    if (rho <= 0.0f) return {};
    if (dual != dual || rho != rho || residual != residual || compliance != compliance ||
        lower != lower || upper != upper)
        return EvaluateScalar(dual, rho, residual, compliance, lower, upper, ScalarResponse{});
    return EvaluateScalar(dual, rho, residual, compliance, lower, upper, ComputeScalarResponse(rho, compliance));
}

// The caller holds normal_impulse fixed throughout each local Newton solve and line search.
NUKA_AUGMENTED_HD inline TangentTerm EvaluateTangent(
    math::Vec3 dual, float rho, math::Vec3 residual, float normal_impulse,
    float mu_first, float mu_second) {
    if (rho <= 0.0f) return {};
    const double mu_y = mu_first <= 0.0f ? 0.0 : double(mu_first);
    const double mu_z = mu_second <= 0.0f ? 0.0 : double(mu_second);
    const double q_y = mu_y == 0.0 ? 0.0
        : math::Quotient(double(dual.y), mu_y) + double(rho) * mu_y * residual.y;
    const double q_z = mu_z == 0.0 ? 0.0
        : math::Quotient(double(dual.z), mu_z) + double(rho) * mu_z * residual.z;
    const double radius = normal_impulse < 0.0f ? 0.0 : double(normal_impulse);
    if (q_y != q_y || q_z != q_z || radius != radius || rho != rho) {
        const float invalid = float(q_y + q_z + radius + rho);
        return {{0.0f, invalid, invalid},
                {0.0f, invalid, invalid, 0.0f, 0.0f, invalid}, double(invalid)};
    }
    const double norm_sq = q_y * q_y + q_z * q_z;
    const double norm = sqrt(norm_sq);
    if (norm <= radius) {
        return {{0.0f, float(mu_y * q_y), float(mu_z * q_z)},
                {0.0f, float(double(rho) * mu_y * mu_y), float(double(rho) * mu_z * mu_z),
                 0.0f, 0.0f, 0.0f},
                math::Quotient(0.5 * norm_sq, double(rho))};
    }
    const double direction_y = math::Quotient(q_y, norm);
    const double direction_z = math::Quotient(q_z, norm);
    const double curvature_scale = math::Quotient(double(rho) * radius, norm);
    return {{0.0f, float(mu_y * radius * direction_y), float(mu_z * radius * direction_z)},
            {0.0f, float(curvature_scale * mu_y * mu_y * direction_z * direction_z),
             float(curvature_scale * mu_z * mu_z * direction_y * direction_y),
             0.0f, 0.0f, float(-curvature_scale * mu_y * mu_z * direction_y * direction_z)},
            math::Quotient(radius * (norm - 0.5 * radius), double(rho))};
}

// EvaluateScalar's impulse and curvature without its potential.
NUKA_AUGMENTED_HD inline ScalarTerm EvaluateScalarImpulse(
    float dual, float rho, float residual, float compliance, float lower, float upper,
    const ScalarResponse& response) {
    if (rho <= 0.0f) return {};
    if (dual != dual || rho != rho || residual != residual || compliance != compliance ||
        lower != lower || upper != upper) {
        const float invalid = dual + rho + residual + compliance + lower + upper;
        return {invalid, invalid, double(invalid)};
    }
    const double s = response.dual_scale;
    const double k = response.residual_scale;
    const double x = s * dual + k * residual;
    const double f = x < lower ? double(lower) : (x > upper ? double(upper) : x);
    const double curvature = x != x ? x :
        (lower < upper && x >= lower && x <= upper ? k : 0.0);
    return {float(f), float(curvature), 0.0};
}

// Change of EvaluateScalar's potential when the residual drops by `drop`: (1/k) times the
// integral of the clamped impulse argument along the drop, so no two potentials are subtracted.
NUKA_AUGMENTED_HD inline float ScalarPotentialChange(
    float dual, float rho, float residual, float drop, float compliance, float lower, float upper,
    const ScalarResponse& response) {
    if (rho <= 0.0f) return 0.0f;
    if (dual != dual || rho != rho || residual != residual || drop != drop ||
        compliance != compliance || lower != lower || upper != upper)
        return dual + rho + residual + drop + compliance + lower + upper;
    const double s = response.dual_scale;
    const double k = response.residual_scale;
    if (!(fabs(dual) <= FLT_MAX && fabs(residual) <= FLT_MAX && fabs(drop) <= FLT_MAX) ||
        !(k > 0.0 && k <= DBL_MAX && fabs(s) <= DBL_MAX)) return NAN;
    const double x = s * dual + k * residual;
    const double y = x - k * drop;
    if (lower > upper) {
        const double fx = x < lower ? double(lower) : (x > upper ? double(upper) : x);
        const double fy = y < lower ? double(lower) : (y > upper ? double(upper) : y);
        return float(fy * (y - 0.5 * fy) / k - fx * (x - 0.5 * fx) / k);
    }
    if (x >= lower && x <= upper && y >= lower && y <= upper)
        return float(-double(drop) * (x - 0.5 * k * drop));
    if (x <= lower && y <= lower) return float(-double(lower) * drop);
    if (x >= upper && y >= upper) return float(-double(upper) * drop);
    const double a = fmin(x, y), b = fmax(x, y);
    double integral = 0.0;
    if (a < lower) integral += double(lower) * (fmin(b, double(lower)) - a);
    const double low = fmax(a, double(lower)), high = fmin(b, double(upper));
    if (low < high) integral += 0.5 * (high - low) * (high + low);
    if (b > upper) integral += double(upper) * (b - fmax(a, double(upper)));
    return float((y >= x ? integral : -integral) / k);
}

// dual / mu + rho * mu * residual to about one float ulp: the quotient's and products' rounding
// errors are recovered with fma and added back after an error-free sum.
NUKA_AUGMENTED_HD inline float ConeArgument(float dual, float rho, float mu, float residual) {
    const float d = dual / mu;
    const float d_low = fmaf(-d, mu, dual) / mu;
    const float p = rho * mu;
    const float t = p * residual;
    const float t_low = fmaf(p, residual, -t) + fmaf(rho, mu, -p) * residual;
    const float sum = d + t;
    const float t_part = sum - d;
    const float sum_low = (d - (sum - t_part)) + (t - t_part);
    return sum + (sum_low + d_low + t_low);
}

// EvaluateTangent's impulse and curvature, evaluated in float without the potential.
NUKA_AUGMENTED_HD inline TangentTerm EvaluateTangentImpulse(
    math::Vec3 dual, float rho, math::Vec3 residual, float normal_impulse,
    float mu_first, float mu_second) {
    if (rho <= 0.0f) return {};
    const float mu_y = mu_first <= 0.0f ? 0.0f : mu_first;
    const float mu_z = mu_second <= 0.0f ? 0.0f : mu_second;
    const float q_y = mu_y == 0.0f ? 0.0f : ConeArgument(dual.y, rho, mu_y, residual.y);
    const float q_z = mu_z == 0.0f ? 0.0f : ConeArgument(dual.z, rho, mu_z, residual.z);
    const float radius = normal_impulse < 0.0f ? 0.0f : normal_impulse;
    if (q_y != q_y || q_z != q_z || radius != radius || rho != rho) {
        const float invalid = q_y + q_z + radius + rho;
        return {{0.0f, invalid, invalid}, {0.0f, invalid, invalid, 0.0f, 0.0f, invalid}, double(invalid)};
    }
    const float norm = hypotf(q_y, q_z);
    if (norm <= radius)
        return {{0.0f, mu_y * q_y, mu_z * q_z},
                {0.0f, rho * mu_y * mu_y, rho * mu_z * mu_z, 0.0f, 0.0f, 0.0f}, 0.0};
    const float direction_y = q_y / norm;
    const float direction_z = q_z / norm;
    const float curvature_scale = rho * radius / norm;
    return {{0.0f, mu_y * radius * direction_y, mu_z * radius * direction_z},
            {0.0f, curvature_scale * mu_y * mu_y * direction_z * direction_z,
             curvature_scale * mu_z * mu_z * direction_y * direction_y,
             0.0f, 0.0f, -curvature_scale * mu_y * mu_z * direction_y * direction_z}, 0.0};
}

// Change of EvaluateTangent's potential when the tangent residuals drop by `drop`: (1/rho) times
// the integral of min(t, radius) between the two cone arguments' norms.
NUKA_AUGMENTED_HD inline float TangentPotentialChange(
    math::Vec3 dual, float rho, math::Vec3 residual, math::Vec3 drop, float normal_impulse,
    float mu_first, float mu_second) {
    if (rho <= 0.0f) return 0.0f;
    const float mu_y = mu_first <= 0.0f ? 0.0f : mu_first;
    const float mu_z = mu_second <= 0.0f ? 0.0f : mu_second;
    const float q_y = mu_y == 0.0f ? 0.0f : ConeArgument(dual.y, rho, mu_y, residual.y);
    const float q_z = mu_z == 0.0f ? 0.0f : ConeArgument(dual.z, rho, mu_z, residual.z);
    const float slide_y = mu_y == 0.0f ? 0.0f : mu_y * drop.y;
    const float slide_z = mu_z == 0.0f ? 0.0f : mu_z * drop.z;
    const float step_y = -rho * slide_y, step_z = -rho * slide_z;
    const float radius = normal_impulse < 0.0f ? 0.0f : normal_impulse;
    if (!(fabsf(q_y) <= FLT_MAX && fabsf(q_z) <= FLT_MAX && fabsf(step_y) <= FLT_MAX &&
          fabsf(step_z) <= FLT_MAX) || radius != radius || rho != rho) return NAN;
    const float before = hypotf(q_y, q_z);
    const float after = hypotf(q_y + step_y, q_z + step_z);
    const float stick = -(slide_y * fmaf(0.5f, step_y, q_y) + slide_z * fmaf(0.5f, step_z, q_z));
    if (before <= radius && after <= radius) return stick;
    if (before >= radius && after >= radius) {
        const float sum = before + after;
        if (!(sum > 0.0f)) return 0.0f;
        return -radius * (slide_y * fmaf(2.0f, q_y, step_y) + slide_z * fmaf(2.0f, q_z, step_z)) / sum;
    }
    // Crossing the radius: the stick quadratic less the slip side's (norm - radius)^2 / (2 rho).
    const float over = fmaxf(before, after) - radius;
    const float slip = 0.5f * over * over / rho;
    return after > before ? stick - slip : stick + slip;
}

template <uint32_t N>
struct BallTerm {
    static_assert(N > 0u);
    float impulse[N]{};
    float curvature[N][N]{};
    double potential = 0.0;
};

// The caller holds radius fixed throughout each local Newton solve and line search.
template <uint32_t N>
NUKA_AUGMENTED_HD inline BallTerm<N> EvaluateBall(
    const double (&dual)[N], double rho, const double (&residual)[N],
    double compliance, double radius) {
    BallTerm<N> term{};
    if ((rho <= 0.0 && fabs(rho) <= DBL_MAX) || compliance == double(FLT_MAX)) return term;
    bool valid = rho > 0.0 && rho <= DBL_MAX && compliance >= 0.0 && compliance <= DBL_MAX &&
                 radius >= 0.0 && radius <= DBL_MAX;
    const double denominator = valid ? 1.0 + rho * compliance : double(NAN);
    const double s = 1.0 / denominator;
    const double k = rho / denominator;
    valid = valid && denominator <= DBL_MAX && k > 0.0 && k <= DBL_MAX;
    double x[N]{};
    double scale = 0.0;
    if (valid) {
        for (uint32_t i = 0u; i < N; ++i) {
            if (!(fabs(dual[i]) <= DBL_MAX) || !(fabs(residual[i]) <= DBL_MAX)) {
                valid = false;
                break;
            }
            x[i] = s * dual[i] + k * residual[i];
            const double magnitude = fabs(x[i]);
            if (!(magnitude <= DBL_MAX)) {
                valid = false;
                break;
            }
            if (magnitude > scale) scale = magnitude;
        }
    }
    if (!valid) {
        term.potential = double(NAN);
        for (uint32_t i = 0u; i < N; ++i) {
            term.impulse[i] = NAN;
            for (uint32_t j = 0u; j < N; ++j) term.curvature[i][j] = NAN;
        }
        return term;
    }
    double norm_sq_scaled = 0.0;
    if (scale > 0.0) {
        for (uint32_t i = 0u; i < N; ++i) {
            const double component = x[i] / scale;
            norm_sq_scaled += component * component;
        }
    }
    const double norm_scaled = sqrt(norm_sq_scaled);
    if (radius == DBL_MAX || scale == 0.0 || scale <= radius / norm_scaled) {
        for (uint32_t i = 0u; i < N; ++i) {
            term.impulse[i] = float(x[i]);
            term.curvature[i][i] = float(k);
        }
        const double energy_norm = (scale / sqrt(k)) * norm_scaled;
        term.potential = (0.5 * energy_norm) * energy_norm;
        return term;
    }
    const double curvature_scale = k * ((radius / scale) / norm_scaled);
    for (uint32_t i = 0u; i < N; ++i) {
        x[i] = (x[i] / scale) / norm_scaled;
        term.impulse[i] = float(radius * x[i]);
    }
    for (uint32_t i = 0u; i < N; ++i)
        for (uint32_t j = 0u; j < N; ++j)
            term.curvature[i][j] = float(curvature_scale * ((i == j ? 1.0 : 0.0) - x[i] * x[j]));
    if (radius > 0.0) {
        const double energy_norm = (sqrt(radius) / sqrt(k)) * sqrt(scale);
        term.potential = (energy_norm * (norm_scaled - 0.5 * (radius / scale))) * energy_norm;
    }
    return term;
}

}  // namespace nuka::nk::augmented

#undef NUKA_AUGMENTED_HD
