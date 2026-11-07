#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "math/finite.hpp"

#include "nk/material/hencky_j2.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NUKA_MPM_CONSTITUTIVE_HD __host__ __device__
#else
#define NUKA_MPM_CONSTITUTIVE_HD
#endif

namespace nuka::nk::material {

struct MpmArithmetic {
    NUKA_MPM_CONSTITUTIVE_HD static inline float Sum3(float a, float b, float c) {
        return (a + b) + c;
    }
};

// A trial reads accepted history; repeated evaluations never advance that history.
struct MpmMaterialHistory {
    const float* elastic_f = nullptr;
    const float* plastic_f = nullptr;
    float equivalent_plastic_strain = 0.0f;
};

struct MpmMaterialTrial {
    float elastic_f[9]{};
    float plastic_f[9]{};
    float equivalent_plastic_strain = 0.0f;
    bool has_plastic_history = false;
};

namespace mpm_detail {

// C = A * B (row-major 3x3).
template <class Arithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline void Mat3Mul(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            C[r * 3 + c] = Arithmetic::Sum3(A[r * 3] * B[c],
                A[r * 3 + 1] * B[3 + c], A[r * 3 + 2] * B[6 + c]);
}

// C = A * B^T (row-major 3x3).
template <class Arithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline void Mat3MulT(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            C[r * 3 + c] = Arithmetic::Sum3(A[r * 3] * B[c * 3],
                A[r * 3 + 1] * B[c * 3 + 1], A[r * 3 + 2] * B[c * 3 + 2]);
}

NUKA_MPM_CONSTITUTIVE_HD inline float Mat3Det(const float* F) {
    return F[0] * (F[4] * F[8] - F[5] * F[7]) -
           F[1] * (F[3] * F[8] - F[5] * F[6]) +
           F[2] * (F[3] * F[7] - F[4] * F[6]);
}

NUKA_MPM_CONSTITUTIVE_HD inline double Mat3DetDouble(const float* F) {
    return double(F[0]) * (double(F[4]) * F[8] - double(F[5]) * F[7]) -
           double(F[1]) * (double(F[3]) * F[8] - double(F[5]) * F[6]) +
           double(F[2]) * (double(F[3]) * F[7] - double(F[4]) * F[6]);
}

// One-sided Jacobi keeps the singular vectors orthonormal without clipping stretches.
template <class Arithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline ConstitutiveStatus Svd3(const float* F, float* U, float* sig, float* V) {
    detail::PrincipalStretch principal;
    const auto status = detail::Decompose(F, principal);
    if (status != ConstitutiveStatus::Ok) return status;
    for (int c = 0; c < 3; ++c) {
        sig[c] = static_cast<float>(std::exp(principal.log_stretch[c]));
        if (!math::IsFinite(sig[c])) return ConstitutiveStatus::Overflow;
        if (!(sig[c] > 0.0f)) return ConstitutiveStatus::SingularDeformation;
        for (int r = 0; r < 3; ++r) {
            U[3 * r + c] = static_cast<float>(principal.left[3 * r + c]);
            V[3 * r + c] = static_cast<float>(principal.right[3 * r + c]);
        }
    }
    return ConstitutiveStatus::Ok;
}

// Hencky elasticity: tau = U diag(2*mu*eps + lambda*tr(eps)) U^T.
// eps_i = ln(sig_i) uses the stored elastic deformation.
template <class Arithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline ConstitutiveStatus GranularKirchhoff(const float* F, float mu,
                                                  float lambda, float* stress) {
    float U[9], sig[3], V[9];
    const auto status = Svd3<Arithmetic>(F, U, sig, V);
    if (status != ConstitutiveStatus::Ok) return status;
    float eps[3];
    for (int i = 0; i < 3; ++i) eps[i] = logf(sig[i]);
    const float tr = eps[0] + eps[1] + eps[2];
    float tau[3];
    for (int i = 0; i < 3; ++i) tau[i] = 2.0f * mu * eps[i] + lambda * tr;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            stress[r * 3 + c] = U[r * 3 + 0] * tau[0] * U[c * 3 + 0] +
                                U[r * 3 + 1] * tau[1] * U[c * 3 + 1] +
                                U[r * 3 + 2] * tau[2] * U[c * 3 + 2];
    return ConstitutiveStatus::Ok;
}

// Drucker-Prager return mapping of principal Hencky strains (Klar et al. 2016).
// Cohesion shifts the tensile apex; friction_deg sets the yield-cone angle.
template <class Arithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline ConstitutiveStatus SandReturnMap(float* F, float mu, float lambda,
                                             float friction_deg, float cohesion) {
    float U[9], sig[3], V[9];
    const auto status = Svd3<Arithmetic>(F, U, sig, V);
    if (status != ConstitutiveStatus::Ok) return status;
    float eps[3];
    for (int i = 0; i < 3; ++i) eps[i] = logf(sig[i]);
    const float tr = eps[0] + eps[1] + eps[2];
    const float kappa = 3.0f * lambda + 2.0f * mu;             // d*lambda + 2*mu, d=3.
    const float c0 = cohesion / kappa;  // apex tensile strain.
    float dev[3];
    for (int i = 0; i < 3; ++i) dev[i] = eps[i] - tr * (1.0f / 3.0f);
    const float devn = sqrtf(dev[0] * dev[0] + dev[1] * dev[1] + dev[2] * dev[2]);
    const float sinp = sinf(friction_deg * 0.017453292519943295f);
    const float alpha = 1.632993161855452f * sinp / fmaxf(3.0f - sinp, 1.0e-6f);
    const float tr_shift = tr - c0;
    float en[3];
    if (tr_shift > 0.0f) {
        for (int i = 0; i < 3; ++i) en[i] = c0 * (1.0f / 3.0f);  // return to the apex.
    } else if (devn < 1.0e-12f) {
        for (int i = 0; i < 3; ++i) en[i] = eps[i];
    } else {
        const float dgamma = devn + (kappa / (2.0f * mu)) * tr_shift * alpha;
        if (dgamma <= 0.0f) {
            for (int i = 0; i < 3; ++i) en[i] = eps[i];           // inside the cone.
        } else {
            const float inv = 1.0f / devn;                        // radial return.
            for (int i = 0; i < 3; ++i) en[i] = eps[i] - dgamma * dev[i] * inv;
        }
    }
    float s2[3];
    for (int i = 0; i < 3; ++i) s2[i] = expf(en[i]);
    float US[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) US[r * 3 + c] = U[r * 3 + c] * s2[c];
    Mat3MulT<Arithmetic>(US, V, F);
    return ConstitutiveStatus::Ok;
}

}  // namespace mpm_detail

// The Kirchhoff stress uses accepted or trial elastic history and the supplied velocity gradient.
template <class Arithmetic = MpmArithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline ConstitutiveStatus EvaluateMpmKirchhoff(
    const MpmMaterial& material, const float* F, const float* C, float* stress) {
    if (!ValidMpmConstitutiveParameters(material)) return ConstitutiveStatus::InvalidParameters;
    if (F == nullptr || stress == nullptr) return ConstitutiveStatus::InvalidState;
    if (!detail::FiniteMatrix(F) || (C != nullptr && !detail::FiniteMatrix(C)))
        return ConstitutiveStatus::InvalidState;
    const float youngs = material.youngs, poisson = material.poisson, kind = material.model_kind;
    const float bulk = material.bulk_modulus, tait_gamma = material.tait_gamma, visc = material.viscosity;
    if (kind == MpmMaterial::kHenckyJ2 || kind == MpmMaterial::kHenckyElastic) {
        HenckyResponse response;
        const auto status = kind == MpmMaterial::kHenckyElastic
            ? EvaluateHenckyElastic(F, {youngs, poisson}, response)
            : EvaluateHenckyJ2(F,
                {youngs, poisson, material.yield_stress, material.hardening_modulus}, response);
        if (status != ConstitutiveStatus::Ok) return status;
        for (int k = 0; k < 9; ++k) stress[k] = response.kirchhoff[k];
    } else if (kind == 4.0f) {
        const float denom = (1.0f + poisson) * (1.0f - 2.0f * poisson);
        const float mu = youngs / (2.0f * (1.0f + poisson));
        const float lambda = (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
        const auto status = mpm_detail::GranularKirchhoff<Arithmetic>(F, mu, lambda, stress);
        if (status != ConstitutiveStatus::Ok) return status;
    } else if (kind == 3.0f) {
        const double determinant = mpm_detail::Mat3DetDouble(F);
        if (!(determinant > 0.0) || !math::IsFinite(determinant))
            return ConstitutiveStatus::SingularDeformation;
        const float J = static_cast<float>(determinant);
        const float pr = fmaxf(bulk * (powf(J, -tait_gamma) - 1.0f), 0.0f);
        const float diag = -pr * J;
        stress[0] = diag; stress[1] = 0.0f; stress[2] = 0.0f;
        stress[3] = 0.0f; stress[4] = diag; stress[5] = 0.0f;
        stress[6] = 0.0f; stress[7] = 0.0f; stress[8] = diag;
    } else if (kind == 2.0f) {
        const double J = mpm_detail::Mat3DetDouble(F);
        if (!(J > 0.0) || !math::IsFinite(J)) return ConstitutiveStatus::SingularDeformation;
        const double mu = youngs / (2.0 * (1.0 + poisson));
        const double lambda = youngs * double(poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
        const double volumetric = lambda * std::log(J);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                const double product = double(F[3*r]) * F[3*c] + double(F[3*r+1]) * F[3*c+1] +
                    double(F[3*r+2]) * F[3*c+2];
                stress[3*r+c] = static_cast<float>(mu * (product - (r == c ? 1.0 : 0.0)) +
                    (r == c ? volumetric : 0.0));
            }
    } else if (kind == 0.0f) {
        detail::PrincipalStretch principal;
        const auto status = detail::Decompose(F, principal);
        if (status != ConstitutiveStatus::Ok) return status;
        const double mu = youngs / (2.0 * (1.0 + poisson));
        const double lambda = youngs * double(poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
        const double log_j = principal.log_stretch[0] + principal.log_stretch[1] + principal.log_stretch[2];
        const double J = std::exp(log_j);
        const double volumetric = lambda * (J - 1.0) * J;
        double diagonal[3];
        for (int k = 0; k < 3; ++k) {
            const double stretch = std::exp(principal.log_stretch[k]);
            diagonal[k] = 2.0 * mu * (stretch - 1.0) * stretch + volumetric;
        }
        detail::SpectralProduct(principal.left, diagonal, principal.left, stress);
    } else {
        return ConstitutiveStatus::InvalidParameters;
    }
    // Every model carries the same Newtonian rate term, J visc (grad v + grad v^T).
    if (visc > 0.0f && C != nullptr) {
        const float Jv = mpm_detail::Mat3Det(F) * visc;
        stress[0] += Jv * 2.0f * C[0];
        stress[4] += Jv * 2.0f * C[4];
        stress[8] += Jv * 2.0f * C[8];
        const float s01 = Jv * (C[1] + C[3]);
        const float s02 = Jv * (C[2] + C[6]);
        const float s12 = Jv * (C[5] + C[7]);
        stress[1] += s01; stress[3] += s01;
        stress[2] += s02; stress[6] += s02;
        stress[5] += s12; stress[7] += s12;
    }
    if (!detail::FiniteMatrix(stress)) return ConstitutiveStatus::Overflow;
    return ConstitutiveStatus::Ok;
}

// Stress rows linearize the Kirchhoff stress: pressure = -tr(tau) / 3 with bulk = -dp / d ln J,
// and the deviator grows at 2 shear per unit strain rate plus 2 viscosity per unit strain rate.
struct MpmStressRowResponse {
    float pressure = 0.0f;
    float bulk = 0.0f;
    float tension = 0.0f;
    float shear = 0.0f;
    float viscosity = 0.0f;
    // The deviator norm stays within cone_slope * pressure + cone_offset.
    float cone_slope = FLT_MAX;
    float cone_offset = FLT_MAX;
};

// Fluids report their unclamped pressure, so a unilateral row opens once J exceeds one.
NUKA_MPM_CONSTITUTIVE_HD inline MpmStressRowResponse EvaluateMpmStressRows(
    const MpmMaterial& material, const float* F, const float* stress,
    float equivalent_plastic_strain = 0.0f) {
    MpmStressRowResponse response;
    const float J = mpm_detail::Mat3Det(F);
    if (!(J > 0.0f) || !math::IsFinite(J)) return response;
    response.pressure = -(stress[0] + stress[4] + stress[8]) / 3.0f;
    response.viscosity = J * fmaxf(material.viscosity, 0.0f);
    const float kind = material.model_kind;
    if (kind == 3.0f) {
        const float gamma = material.tait_gamma, compression = powf(J, -gamma);
        response.pressure += J * fminf(material.bulk_modulus * (compression - 1.0f), 0.0f);
        response.bulk = J * material.bulk_modulus * ((gamma - 1.0f) * compression + 1.0f);
        return response;
    }
    const float youngs = material.youngs, poisson = material.poisson;
    const float denom = (1.0f + poisson) * (1.0f - 2.0f * poisson);
    const float mu = youngs / (2.0f * (1.0f + poisson));
    const float lambda = (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
    if (kind == 0.0f) {
        const float stretch = cbrtf(J);
        response.bulk = lambda * J * (2.0f * J - 1.0f) +
                        (2.0f / 3.0f) * mu * stretch * (2.0f * stretch - 1.0f);
        response.shear = mu * stretch * stretch;
    } else if (kind == 2.0f) {
        response.bulk = lambda + (2.0f / 3.0f) * mu * cbrtf(J * J);
        response.shear = mu * cbrtf(J * J);
    } else {
        response.bulk = lambda + (2.0f / 3.0f) * mu;
        response.shear = mu;
    }
    response.bulk = fmaxf(response.bulk, 0.0f);
    response.shear = fmaxf(response.shear, 0.0f);
    response.tension = FLT_MAX;
    if (kind == 4.0f) {
        // Drucker-Prager in Kirchhoff stress: |dev tau| <= alpha (3 pressure + cohesion).
        const float sine = sinf(material.dp_friction * 0.017453292519943295f);
        const float alpha = 1.632993161855452f * sine / fmaxf(3.0f - sine, 1.0e-6f);
        response.tension = material.dp_cohesion / 3.0f;
        response.cone_slope = 3.0f * alpha;
        response.cone_offset = alpha * material.dp_cohesion;
    } else if (kind == MpmMaterial::kHenckyJ2 && material.yield_stress > 0.0f) {
        response.cone_slope = 0.0f;
        response.cone_offset = static_cast<float>(0.816496580927726 *
            (material.yield_stress + double(material.hardening_modulus) * equivalent_plastic_strain));
    }
    return response;
}

// Orthonormal traceless symmetric basis B_k: component k of a tensor T is B_k : T.
NUKA_MPM_CONSTITUTIVE_HD inline void MpmDeviatorComponents(const float* T, float* out) {
    constexpr float kHalfRoot = 0.70710678118654752f, kSixthRoot = 0.40824829046386302f;
    out[0] = kHalfRoot * (T[1] + T[3]);
    out[1] = kHalfRoot * (T[2] + T[6]);
    out[2] = kHalfRoot * (T[5] + T[7]);
    out[3] = kHalfRoot * (T[0] - T[4]);
    out[4] = kSixthRoot * (T[0] + T[4] - 2.0f * T[8]);
}

// B_k g for a node gradient g, so a row of component k reads B_k : grad v.
NUKA_MPM_CONSTITUTIVE_HD inline void MpmDeviatorGradient(uint32_t k, float gx, float gy, float gz,
                                                         float* out) {
    constexpr float kHalfRoot = 0.70710678118654752f, kSixthRoot = 0.40824829046386302f;
    switch (k) {
    case 0u: out[0] = kHalfRoot * gy; out[1] = kHalfRoot * gx; out[2] = 0.0f; break;
    case 1u: out[0] = kHalfRoot * gz; out[1] = 0.0f; out[2] = kHalfRoot * gx; break;
    case 2u: out[0] = 0.0f; out[1] = kHalfRoot * gz; out[2] = kHalfRoot * gy; break;
    case 3u: out[0] = kHalfRoot * gx; out[1] = -kHalfRoot * gy; out[2] = 0.0f; break;
    default: out[0] = kSixthRoot * gx; out[1] = kSixthRoot * gy; out[2] = -2.0f * kSixthRoot * gz;
    }
}

// A stress block reads B_0 = I, then B_1..B_5 above. With S = sum m g g^T (xx yy zz xy xz yz),
// its response is A_kl = tr(B_k B_l S).
NUKA_MPM_CONSTITUTIVE_HD inline void MpmStressBlockResponse(const float* s, float (&A)[6][6]) {
    constexpr float kRoot2 = 1.41421356237309505f, kHalfRoot = 0.70710678118654752f;
    constexpr float kSixthRoot = 0.40824829046386302f, kThirdRoot = 0.57735026918962576f;
    constexpr float kTwelfthRoot = 0.28867513459481288f;
    const float a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5];
    const float row[21] = {a + b + c, kRoot2 * d, kRoot2 * e, kRoot2 * f, kHalfRoot * (a - b),
        kSixthRoot * (a + b - 2.0f * c), 0.5f * (a + b), 0.5f * f, 0.5f * e, 0.0f,
        kThirdRoot * d, 0.5f * (a + c), 0.5f * d, 0.5f * e, -kTwelfthRoot * e, 0.5f * (b + c),
        -0.5f * f, -kTwelfthRoot * f, 0.5f * (a + b), kTwelfthRoot * (a - b),
        (a + b + 4.0f * c) / 6.0f};
    uint32_t at = 0u;
    for (uint32_t k = 0u; k < 6u; ++k)
        for (uint32_t l = k; l < 6u; ++l) A[k][l] = A[l][k] = row[at++];
}

// Rates B_k : G of a symmetric velocity gradient G (xx yy zz xy xz yz).
NUKA_MPM_CONSTITUTIVE_HD inline void MpmStressBlockRates(const float* g, float (&rate)[6]) {
    const float T[9] = {g[0], g[3], g[4], g[3], g[1], g[5], g[4], g[5], g[2]};
    rate[0] = g[0] + g[1] + g[2];
    MpmDeviatorComponents(T, rate + 1);
}

// The symmetric tensor sum_k d_k B_k (xx yy zz xy xz yz).
NUKA_MPM_CONSTITUTIVE_HD inline void MpmStressBlockTensor(const float (&d)[6], float* out) {
    constexpr float kHalfRoot = 0.70710678118654752f, kSixthRoot = 0.40824829046386302f;
    out[0] = d[0] + kHalfRoot * d[4] + kSixthRoot * d[5];
    out[1] = d[0] - kHalfRoot * d[4] + kSixthRoot * d[5];
    out[2] = d[0] - 2.0f * kSixthRoot * d[5];
    out[3] = kHalfRoot * d[1];
    out[4] = kHalfRoot * d[2];
    out[5] = kHalfRoot * d[3];
}

// A failed evaluation leaves the output intact; plastic history belongs to the trial until commit.
template <class Arithmetic = MpmArithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline ConstitutiveStatus EvaluateMpmMaterialTrial(
    const MpmMaterial& material, MpmMaterialHistory history, const float* C, float dt,
    MpmMaterialTrial& trial) {
    if (!ValidMpmConstitutiveParameters(material)) return ConstitutiveStatus::InvalidParameters;
    if (history.elastic_f == nullptr || C == nullptr || !(dt > 0.0f) || !math::IsFinite(dt))
        return ConstitutiveStatus::InvalidState;
    if (!detail::FiniteMatrix(history.elastic_f) || !detail::FiniteMatrix(C))
        return ConstitutiveStatus::InvalidState;
    if (!(mpm_detail::Mat3DetDouble(history.elastic_f) > 0.0))
        return ConstitutiveStatus::SingularDeformation;
    const float kind = material.model_kind;
    MpmMaterialTrial next;
    if (kind == MpmMaterial::kHenckyJ2) {
        if (history.plastic_f == nullptr) return ConstitutiveStatus::InvalidState;
        next.has_plastic_history = true;
        next.equivalent_plastic_strain = history.equivalent_plastic_strain;
        for (int k = 0; k < 9; ++k) next.plastic_f[k] = history.plastic_f[k];
        float map[9], elastic[9];
        for (int k = 0; k < 9; ++k) map[k] = dt * C[k] + (k % 4 == 0 ? 1.0f : 0.0f);
        mpm_detail::Mat3Mul<Arithmetic>(map, history.elastic_f, elastic);
        const auto status = ReturnHenckyJ2(elastic,
            {material.youngs, material.poisson, material.yield_stress, material.hardening_modulus},
            next.elastic_f, next.plastic_f, next.equivalent_plastic_strain);
        if (status != ConstitutiveStatus::Ok) return status;
    } else if (kind == 3.0f) {
        const float volume = mpm_detail::Mat3Det(history.elastic_f) *
            (1.0f + dt * (C[0] + C[4] + C[8]));
        if (!(volume > 0.0f) || !math::IsFinite(volume)) return ConstitutiveStatus::SingularDeformation;
        // Liquid bears no tension: separation opens voids instead of stretching the liquid.
        const float stretch = cbrtf(fminf(volume, 1.0f));
        for (int k = 0; k < 9; ++k) next.elastic_f[k] = (k % 4 == 0) ? stretch : 0.0f;
    } else {
        float map[9];
        for (int k = 0; k < 9; ++k) map[k] = dt * C[k];
        map[0] += 1.0f; map[4] += 1.0f; map[8] += 1.0f;
        mpm_detail::Mat3Mul<Arithmetic>(map, history.elastic_f, next.elastic_f);
        if (!detail::FiniteMatrix(next.elastic_f)) return ConstitutiveStatus::Overflow;
        if (kind == MpmMaterial::kHenckyElastic) {
            HenckyResponse response;
            const auto status = EvaluateHenckyElastic(next.elastic_f,
                {material.youngs, material.poisson}, response);
            if (status != ConstitutiveStatus::Ok) return status;
        }
        if (kind == 4.0f) {
            const float youngs = material.youngs, poisson = material.poisson;
            const float denom = (1.0f + poisson) * (1.0f - 2.0f * poisson);
            const float mu = youngs / (2.0f * (1.0f + poisson));
            const float lambda = (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
            const auto status = mpm_detail::SandReturnMap<Arithmetic>(next.elastic_f, mu, lambda,
                material.dp_friction, material.dp_cohesion);
            if (status != ConstitutiveStatus::Ok) return status;
        }
    }
    if (!detail::FiniteMatrix(next.elastic_f) ||
        (next.has_plastic_history && (!detail::FiniteMatrix(next.plastic_f) ||
            !math::IsFinite(next.equivalent_plastic_strain)))) return ConstitutiveStatus::Overflow;
    if (!(mpm_detail::Mat3DetDouble(next.elastic_f) > 0.0))
        return ConstitutiveStatus::SingularDeformation;
    trial = next;
    return ConstitutiveStatus::Ok;
}

// The caller commits only an accepted trial, with all required destination fields present.
NUKA_MPM_CONSTITUTIVE_HD inline bool CommitMpmMaterialTrial(
    const MpmMaterialTrial& trial, float* elastic_f, float* plastic_f, float* plastic_strain) {
    if (elastic_f == nullptr || (trial.has_plastic_history &&
        (plastic_f == nullptr || plastic_strain == nullptr))) return false;
    for (int k = 0; k < 9; ++k) elastic_f[k] = trial.elastic_f[k];
    if (trial.has_plastic_history) {
        for (int k = 0; k < 9; ++k) plastic_f[k] = trial.plastic_f[k];
        *plastic_strain = trial.equivalent_plastic_strain;
    }
    return true;
}

}  // namespace nuka::nk::material

#undef NUKA_MPM_CONSTITUTIVE_HD
