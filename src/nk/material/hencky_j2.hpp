#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "nk/material/mpm_material.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NUKA_MATERIAL_HD __host__ __device__
#else
#define NUKA_MATERIAL_HD
#endif

namespace nuka::nk::material {

enum class ConstitutiveStatus : uint32_t {
    Ok, InvalidParameters, InvalidState, SingularDeformation, DecompositionFailure, Overflow
};

struct HenckyJ2Parameters {
    float youngs, poisson, yield_stress, hardening_modulus;
};

struct HenckyResponse {
    float kirchhoff[9] = {};
    float elastic_energy = 0.0f;
    float equivalent_stress = 0.0f;
};

NUKA_MATERIAL_HD inline bool ValidHenckyJ2(const HenckyJ2Parameters& p) {
    if (!std::isfinite(p.youngs) || !(p.youngs > 0.0f) ||
        !std::isfinite(p.poisson) || !(p.poisson > -1.0f && p.poisson < 0.5f) ||
        !std::isfinite(p.yield_stress) || !(p.yield_stress > 0.0f) ||
        !std::isfinite(p.hardening_modulus) || p.hardening_modulus < 0.0f) return false;
    const float mu = p.youngs / (2.0f * (1.0f + p.poisson));
    const float bulk = p.youngs / (3.0f * (1.0f - 2.0f * p.poisson));
    return mu > 0.0f && bulk > 0.0f && std::isfinite(mu) && std::isfinite(bulk) &&
           std::isfinite(3.0f * mu + p.hardening_modulus);
}

NUKA_MATERIAL_HD inline bool ValidMpmMaterial(const MpmMaterial& m) {
    if (m.model_kind == MpmMaterial::kHenckyJ2)
        return std::isfinite(m.density) && m.density > 0.0f &&
               ValidHenckyJ2({m.youngs, m.poisson, m.yield_stress, m.hardening_modulus});
    return m.model_kind == 0.0f || m.model_kind == 2.0f ||
           m.model_kind == 3.0f || m.model_kind == 4.0f;
}

namespace detail {

NUKA_MATERIAL_HD inline float Determinant(const float* a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7]) -
           a[1] * (a[3] * a[8] - a[5] * a[6]) +
           a[2] * (a[3] * a[7] - a[4] * a[6]);
}

NUKA_MATERIAL_HD inline bool FiniteMatrix(const float* a) {
    for (int i = 0; i < 9; ++i) if (!std::isfinite(a[i])) return false;
    return true;
}

NUKA_MATERIAL_HD inline void Multiply(const float* a, const float* b, float* result) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            result[r * 3 + c] = static_cast<float>(double(a[r * 3]) * b[c] +
                double(a[r * 3 + 1]) * b[3 + c] + double(a[r * 3 + 2]) * b[6 + c]);
}

struct PrincipalStretch {
    double left[9], right[9], log_stretch[3];
};

// Double-precision Jacobi preserves small stretches without F^T F or clipping.
NUKA_MATERIAL_HD inline ConstitutiveStatus Decompose(const float* f, PrincipalStretch& result) {
    double scale = 0.0;
    for (int i = 0; i < 9; ++i) {
        if (!std::isfinite(f[i])) return ConstitutiveStatus::InvalidState;
        scale = std::fmax(scale, std::fabs(double(f[i])));
    }
    if (!(scale > 0.0f)) return ConstitutiveStatus::SingularDeformation;
    double a[9];
    for (int i = 0; i < 9; ++i) {
        a[i] = f[i] / scale;
        result.right[i] = i % 4 == 0 ? 1.0f : 0.0f;
    }
    const double determinant = a[0] * (a[4] * a[8] - a[5] * a[7]) -
        a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (!(determinant > 0.0)) return ConstitutiveStatus::SingularDeformation;
    for (int sweep = 0; sweep < 16; ++sweep) {
        bool changed = false;
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                double pp = 0.0, qq = 0.0, pq = 0.0;
                for (int r = 0; r < 3; ++r) {
                    pp += a[r * 3 + p] * a[r * 3 + p];
                    qq += a[r * 3 + q] * a[r * 3 + q];
                    pq += a[r * 3 + p] * a[r * 3 + q];
                }
                if (std::fabs(pq) <= 4.0 * DBL_EPSILON * std::sqrt(pp * qq)) continue;
                const double delta = qq - pp;
                const double radius = std::hypot(delta, 2.0 * pq);
                const double t = 2.0 * pq / (delta >= 0.0 ? delta + radius : delta - radius);
                const double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                for (int r = 0; r < 3; ++r) {
                    const double ap = a[r * 3 + p], aq = a[r * 3 + q];
                    const double vp = result.right[r * 3 + p], vq = result.right[r * 3 + q];
                    a[r * 3 + p] = c * ap - s * aq;
                    a[r * 3 + q] = s * ap + c * aq;
                    result.right[r * 3 + p] = c * vp - s * vq;
                    result.right[r * 3 + q] = s * vp + c * vq;
                }
                changed = true;
            }
        }
        if (!changed) break;
    }
    for (int c = 0; c < 3; ++c) {
        const double length = std::sqrt(a[c] * a[c] + a[3 + c] * a[3 + c] + a[6 + c] * a[6 + c]);
        const double stretch = length * scale;
        if (!(stretch > 0.0f) || !std::isfinite(stretch)) return ConstitutiveStatus::SingularDeformation;
        result.log_stretch[c] = std::log(stretch);
        for (int r = 0; r < 3; ++r) result.left[r * 3 + c] = a[r * 3 + c] / length;
    }
    for (int p = 0; p < 2; ++p)
        for (int q = p + 1; q < 3; ++q) {
            double dot = 0.0;
            for (int r = 0; r < 3; ++r) dot += result.left[r * 3 + p] * result.left[r * 3 + q];
            if (!std::isfinite(dot) || std::fabs(dot) > 32.0 * DBL_EPSILON)
                return ConstitutiveStatus::DecompositionFailure;
        }
    return ConstitutiveStatus::Ok;
}

NUKA_MATERIAL_HD inline void SpectralProduct(const double* left, const double* diagonal,
                                            const double* right, float* result) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            result[r * 3 + c] = static_cast<float>(left[r * 3] * diagonal[0] * right[c * 3] +
                left[r * 3 + 1] * diagonal[1] * right[c * 3 + 1] +
                left[r * 3 + 2] * diagonal[2] * right[c * 3 + 2]);
}

}  // namespace detail

NUKA_MATERIAL_HD inline ConstitutiveStatus EvaluateHenckyJ2(
    const float* elastic_f, const HenckyJ2Parameters& p, HenckyResponse& response) {
    if (!ValidHenckyJ2(p)) return ConstitutiveStatus::InvalidParameters;
    detail::PrincipalStretch principal;
    const auto status = detail::Decompose(elastic_f, principal);
    if (status != ConstitutiveStatus::Ok) return status;
    const double mu = p.youngs / (2.0 * (1.0 + p.poisson));
    const double bulk = p.youngs / (3.0 * (1.0 - 2.0 * p.poisson));
    const double trace = principal.log_stretch[0] + principal.log_stretch[1] + principal.log_stretch[2];
    double tau[3], dev2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double dev = principal.log_stretch[i] - trace / 3.0;
        dev2 += dev * dev;
        tau[i] = 2.0f * mu * dev + bulk * trace;
    }
    HenckyResponse output;
    detail::SpectralProduct(principal.left, tau, principal.left, output.kirchhoff);
    output.elastic_energy = static_cast<float>(mu * dev2 + 0.5 * bulk * trace * trace);
    output.equivalent_stress = static_cast<float>(2.0 * mu * std::sqrt(1.5 * dev2));
    if (!detail::FiniteMatrix(output.kirchhoff) || !std::isfinite(output.elastic_energy) ||
        !std::isfinite(output.equivalent_stress)) return ConstitutiveStatus::Overflow;
    response = output;
    return ConstitutiveStatus::Ok;
}

// The return preserves total deformation and plastic volume; failure commits no state.
NUKA_MATERIAL_HD inline ConstitutiveStatus ReturnHenckyJ2(
    const float* trial_f, const HenckyJ2Parameters& p, float* elastic_f,
    float* plastic_f, float& alpha) {
    if (!ValidHenckyJ2(p)) return ConstitutiveStatus::InvalidParameters;
    const float plastic_volume = detail::Determinant(plastic_f);
    if (!std::isfinite(alpha) || alpha < 0.0f || !detail::FiniteMatrix(plastic_f) ||
        !std::isfinite(plastic_volume) || !(plastic_volume > 0.0f))
        return ConstitutiveStatus::InvalidState;
    detail::PrincipalStretch principal;
    const auto status = detail::Decompose(trial_f, principal);
    if (status != ConstitutiveStatus::Ok) return status;
    const double mu = p.youngs / (2.0 * (1.0 + p.poisson));
    const double trace = principal.log_stretch[0] + principal.log_stretch[1] + principal.log_stretch[2];
    double dev[3], dev2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        dev[i] = principal.log_stretch[i] - trace / 3.0;
        dev2 += dev[i] * dev[i];
    }
    const double q = 2.0 * mu * std::sqrt(1.5 * dev2);
    const double yield = p.yield_stress + double(p.hardening_modulus) * alpha;
    if (!std::isfinite(q) || !std::isfinite(yield)) return ConstitutiveStatus::Overflow;
    if (q <= yield) {
        for (int i = 0; i < 9; ++i) elastic_f[i] = trial_f[i];
        return ConstitutiveStatus::Ok;
    }
    const double increment = (q - yield) / (3.0 * mu + p.hardening_modulus);
    const float next_alpha = static_cast<float>(alpha + increment);
    const double retained = (p.yield_stress + double(p.hardening_modulus) * next_alpha) / q;
    double stretch[3], plastic_stretch[3];
    for (int i = 0; i < 3; ++i) {
        const double strain = trace / 3.0 + retained * dev[i];
        stretch[i] = std::exp(strain);
        plastic_stretch[i] = std::exp(principal.log_stretch[i] - strain);
    }
    float next_elastic[9], next_plastic[9], plastic_increment[9];
    detail::SpectralProduct(principal.left, stretch, principal.right, next_elastic);
    detail::SpectralProduct(principal.right, plastic_stretch, principal.right, plastic_increment);
    detail::Multiply(plastic_increment, plastic_f, next_plastic);
    const float elastic_volume = detail::Determinant(next_elastic);
    const float next_plastic_volume = detail::Determinant(next_plastic);
    if (!std::isfinite(next_alpha) || !detail::FiniteMatrix(next_elastic) ||
        !detail::FiniteMatrix(next_plastic) || !std::isfinite(elastic_volume) ||
        !std::isfinite(next_plastic_volume) || !(elastic_volume > 0.0f) ||
        !(next_plastic_volume > 0.0f)) return ConstitutiveStatus::Overflow;
    for (int i = 0; i < 9; ++i) {
        elastic_f[i] = next_elastic[i];
        plastic_f[i] = next_plastic[i];
    }
    alpha = next_alpha;
    return ConstitutiveStatus::Ok;
}

}  // namespace nuka::nk::material

#undef NUKA_MATERIAL_HD
