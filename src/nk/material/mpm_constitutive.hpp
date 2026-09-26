#pragma once

#include <cmath>
#include <cstdint>

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

// Transposed inverse F^{-T} (row-major). Returns false (and leaves out unset) for a
// near-singular F so the caller can fall back to a zero stress (degenerate cell).
NUKA_MPM_CONSTITUTIVE_HD inline bool Mat3InvTranspose(const float* F, float* out, float det) {
    if (fabsf(det) < 1e-12f) return false;
    const float inv = 1.0f / det;
    // Cofactor matrix C; F^{-1} = C^T/det, so F^{-T} = C/det.
    out[0] = (F[4] * F[8] - F[5] * F[7]) * inv;
    out[1] = (F[5] * F[6] - F[3] * F[8]) * inv;
    out[2] = (F[3] * F[7] - F[4] * F[6]) * inv;
    out[3] = (F[2] * F[7] - F[1] * F[8]) * inv;
    out[4] = (F[0] * F[8] - F[2] * F[6]) * inv;
    out[5] = (F[1] * F[6] - F[0] * F[7]) * inv;
    out[6] = (F[1] * F[5] - F[2] * F[4]) * inv;
    out[7] = (F[2] * F[3] - F[0] * F[5]) * inv;
    out[8] = (F[0] * F[4] - F[1] * F[3]) * inv;
    return true;
}

// One symmetric Jacobi rotation eliminates S(p,q) and accumulates eigenvectors in V.
NUKA_MPM_CONSTITUTIVE_HD inline void JacobiRotate(float* S, float* V, int p, int q) {
    const float spq = S[p * 3 + q];
    if (spq == 0.0f) return;
    const float spp = S[p * 3 + p], sqq = S[q * 3 + q];
    const float theta = (sqq - spp) / (2.0f * spq);
    const float sign = theta >= 0.0f ? 1.0f : -1.0f;
    const float t = sign / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
    const float c = 1.0f / sqrtf(t * t + 1.0f);
    const float s = t * c;
    for (int k = 0; k < 3; ++k) {
        const float sik = S[k * 3 + p], siq = S[k * 3 + q];
        S[k * 3 + p] = c * sik - s * siq;
        S[k * 3 + q] = s * sik + c * siq;
    }
    for (int k = 0; k < 3; ++k) {
        const float skp = S[p * 3 + k], skq = S[q * 3 + k];
        S[p * 3 + k] = c * skp - s * skq;
        S[q * 3 + k] = s * skp + c * skq;
        const float vkp = V[k * 3 + p], vkq = V[k * 3 + q];
        V[k * 3 + p] = c * vkp - s * vkq;
        V[k * 3 + q] = s * vkp + c * vkq;
    }
}

// Eight Jacobi sweeps decompose F^T F; U = F V / sig completes the SVD.
// Near-zero singular values use the corresponding V column.
template <class Arithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline void Svd3(const float* F, float* U, float* sig, float* V) {
    float A[9];
    // A := F^T F (symmetric); its eigenvectors are the right singular vectors V.
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            A[r * 3 + c] = Arithmetic::Sum3(F[r] * F[c],
                F[3 + r] * F[3 + c], F[6 + r] * F[6 + c]);
    for (int k = 0; k < 9; ++k) V[k] = (k % 4 == 0) ? 1.0f : 0.0f;  // V = I.
    for (int sweep = 0; sweep < 8; ++sweep) {
        JacobiRotate(A, V, 0, 1);
        JacobiRotate(A, V, 0, 2);
        JacobiRotate(A, V, 1, 2);
    }
    float s2[3] = {A[0], A[4], A[8]};
    // Floor the singular values so volumetric stress stays bounded as J -> 0.
    for (int i = 0; i < 3; ++i) sig[i] = fmaxf(sqrtf(fmaxf(s2[i], 0.0f)), 0.05f);
    // U columns = F * V_col / sig (fall back to V_col when sig ~ 0).
    for (int c = 0; c < 3; ++c) {
        float fc[3];
        for (int r = 0; r < 3; ++r)
            fc[r] = F[r * 3 + 0] * V[0 * 3 + c] + F[r * 3 + 1] * V[1 * 3 + c] +
                    F[r * 3 + 2] * V[2 * 3 + c];
        if (sig[c] > 1e-8f) {
            const float inv = 1.0f / sig[c];
            for (int r = 0; r < 3; ++r) U[r * 3 + c] = fc[r] * inv;
        } else {
            for (int r = 0; r < 3; ++r) U[r * 3 + c] = V[r * 3 + c];
        }
    }
    // Reflect (not just rotate): if det(U) < 0 flip the smallest-magnitude singular
    // value so R = U V^T is the closest proper rotation (handles inverted elements).
    if (Mat3Det(U) < 0.0f) {
        int kmin = 0;
        if (fabsf(sig[1]) < fabsf(sig[kmin])) kmin = 1;
        if (fabsf(sig[2]) < fabsf(sig[kmin])) kmin = 2;
        for (int r = 0; r < 3; ++r) U[r * 3 + kmin] = -U[r * 3 + kmin];
        sig[kmin] = -sig[kmin];
    }
}

// First Piola-Kirchhoff stress for fixed-corotated and Neo-Hookean elasticity.
// mu/lambda are the Lame moduli; matrices use row-major float[9] storage.
template <class Arithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline void FirstPiola(const float* F, float mu, float lambda,
                                           float model_kind, float* P) {
    const float J = Mat3Det(F);
    float FinvT[9];
    const bool ok = Mat3InvTranspose(F, FinvT, J);
    if (!ok) { for (int k = 0; k < 9; ++k) P[k] = 0.0f; return; }
    if (model_kind > 1.5f) {  // Neo-Hookean elastic (kind 2).
        const float lj = logf(fmaxf(J, 1e-8f));
        for (int k = 0; k < 9; ++k)
            P[k] = mu * (F[k] - FinvT[k]) + lambda * lj * FinvT[k];
        return;
    }
    float U[9], sig[3], V[9], R[9];
    Svd3<Arithmetic>(F, U, sig, V);
    Mat3MulT<Arithmetic>(U, V, R);  // R = U * V^T (proper rotation).
    const float coef = lambda * (J - 1.0f) * J;
    for (int k = 0; k < 9; ++k)
        P[k] = 2.0f * mu * (F[k] - R[k]) + coef * FinvT[k];
}

// Granular stress and stored elastic deformation share this Hencky-strain bound.
constexpr float kSandHenckyCap = 0.15f;

// Hencky elasticity: tau = U diag(2*mu*eps + lambda*tr(eps)) U^T.
// eps_i = ln(sig_i) uses the stored elastic deformation.
template <class Arithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline void GranularKirchhoff(const float* F, float mu,
                                                  float lambda, float* stress) {
    float U[9], sig[3], V[9];
    Svd3<Arithmetic>(F, U, sig, V);
    float eps[3];
    for (int i = 0; i < 3; ++i) {
        eps[i] = logf(fmaxf(fabsf(sig[i]), 1.0e-6f));
        eps[i] = fminf(fmaxf(eps[i], -kSandHenckyCap), kSandHenckyCap);
    }
    const float tr = eps[0] + eps[1] + eps[2];
    float tau[3];
    for (int i = 0; i < 3; ++i) tau[i] = 2.0f * mu * eps[i] + lambda * tr;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            stress[r * 3 + c] = U[r * 3 + 0] * tau[0] * U[c * 3 + 0] +
                                U[r * 3 + 1] * tau[1] * U[c * 3 + 1] +
                                U[r * 3 + 2] * tau[2] * U[c * 3 + 2];
}

// Drucker-Prager return mapping of principal Hencky strains (Klar et al. 2016).
// Cohesion shifts the tensile apex; friction_deg sets the yield-cone angle.
template <class Arithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline void SandReturnMap(float* F, float mu, float lambda,
                                             float friction_deg, float cohesion) {
    float U[9], sig[3], V[9];
    Svd3<Arithmetic>(F, U, sig, V);
    float eps[3], sgn[3];
    for (int i = 0; i < 3; ++i) {
        sgn[i] = sig[i] < 0.0f ? -1.0f : 1.0f;
        eps[i] = logf(fmaxf(fabsf(sig[i]), 1.0e-6f));
    }
    const float tr = eps[0] + eps[1] + eps[2];
    const float kappa = 3.0f * lambda + 2.0f * mu;             // d*lambda + 2*mu, d=3.
    const float c0 = kappa > 1.0e-9f ? cohesion / kappa : 0.0f;  // apex tensile strain.
    float dev[3];
    for (int i = 0; i < 3; ++i) dev[i] = eps[i] - tr * (1.0f / 3.0f);
    const float devn = sqrtf(dev[0] * dev[0] + dev[1] * dev[1] + dev[2] * dev[2]);
    const float sinp = sinf(friction_deg * 0.017453292519943295f);
    const float alpha = 1.632993161855452f * sinp / fmaxf(3.0f - sinp, 1.0e-6f);
    const float tr_shift = tr - c0;
    float en[3];
    if (devn < 1.0e-12f || tr_shift > 0.0f) {
        for (int i = 0; i < 3; ++i) en[i] = c0 * (1.0f / 3.0f);  // return to the apex.
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
    for (int i = 0; i < 3; ++i) {
        // Cap the STORED elastic strain: the overflow is plastic densification, so
        // the state the next substep stresses can never spiral (bounded restoring).
        en[i] = fminf(fmaxf(en[i], -kSandHenckyCap), kSandHenckyCap);
        s2[i] = sgn[i] * expf(en[i]);
    }
    float US[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) US[r * 3 + c] = U[r * 3 + c] * s2[c];
    Mat3MulT<Arithmetic>(US, V, F);  // F = US * V^T = U diag(s2) V^T.
}

}  // namespace mpm_detail

// The Kirchhoff stress uses accepted or trial elastic history and the supplied velocity gradient.
template <class Arithmetic = MpmArithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline ConstitutiveStatus EvaluateMpmKirchhoff(
    const MpmMaterial& material, const float* F, const float* C, float* stress) {
    if (F == nullptr || stress == nullptr) return ConstitutiveStatus::InvalidState;
    const float youngs = material.youngs, poisson = material.poisson, kind = material.model_kind;
    const float bulk = material.bulk_modulus, tait_gamma = material.tait_gamma, visc = material.viscosity;
    if (kind == MpmMaterial::kHenckyJ2) {
        HenckyResponse response;
        const auto status = EvaluateHenckyJ2(F,
            {youngs, poisson, material.yield_stress, material.hardening_modulus}, response);
        if (status != ConstitutiveStatus::Ok) return status;
        for (int k = 0; k < 9; ++k) stress[k] = response.kirchhoff[k];
    } else if (kind == 4.0f) {
        const float denom = (1.0f + poisson) * (1.0f - 2.0f * poisson);
        const float mu = youngs / (2.0f * (1.0f + poisson));
        const float lambda = (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
        mpm_detail::GranularKirchhoff<Arithmetic>(F, mu, lambda, stress);
    } else if (kind == 3.0f) {
        const float J = mpm_detail::Mat3Det(F);
        const float pr = fmaxf(bulk * (powf(J, -tait_gamma) - 1.0f), 0.0f);
        const float diag = -pr * J;
        stress[0] = diag; stress[1] = 0.0f; stress[2] = 0.0f;
        stress[3] = 0.0f; stress[4] = diag; stress[5] = 0.0f;
        stress[6] = 0.0f; stress[7] = 0.0f; stress[8] = diag;
        if (visc > 0.0f && C != nullptr) {
            const float Jv = J * visc;
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
    } else if (kind == 0.0f || kind == 2.0f) {
        const float denom = (1.0f + poisson) * (1.0f - 2.0f * poisson);
        const float mu = youngs / (2.0f * (1.0f + poisson));
        const float lambda = (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
        float P[9];
        mpm_detail::FirstPiola<Arithmetic>(F, mu, lambda, kind, P);
        mpm_detail::Mat3MulT<Arithmetic>(P, F, stress);
    } else {
        return ConstitutiveStatus::InvalidParameters;
    }
    return ConstitutiveStatus::Ok;
}

// A failed evaluation leaves the output intact; plastic history belongs to the trial until commit.
template <class Arithmetic = MpmArithmetic>
NUKA_MPM_CONSTITUTIVE_HD inline ConstitutiveStatus EvaluateMpmMaterialTrial(
    const MpmMaterial& material, MpmMaterialHistory history, const float* C, float dt,
    MpmMaterialTrial& trial) {
    if (history.elastic_f == nullptr || C == nullptr || !(dt > 0.0f) || !std::isfinite(dt))
        return ConstitutiveStatus::InvalidState;
    const float kind = material.model_kind;
    if (!(kind == 0.0f || kind == 2.0f || kind == 3.0f || kind == 4.0f || kind == MpmMaterial::kHenckyJ2))
        return ConstitutiveStatus::InvalidParameters;
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
        if (!(volume > 0.0f) || !std::isfinite(volume)) return ConstitutiveStatus::SingularDeformation;
        const float stretch = cbrtf(volume);
        for (int k = 0; k < 9; ++k) next.elastic_f[k] = (k % 4 == 0) ? stretch : 0.0f;
    } else {
        float map[9];
        for (int k = 0; k < 9; ++k) map[k] = dt * C[k];
        map[0] += 1.0f; map[4] += 1.0f; map[8] += 1.0f;
        mpm_detail::Mat3Mul<Arithmetic>(map, history.elastic_f, next.elastic_f);
        if (kind == 4.0f) {
            const float youngs = material.youngs, poisson = material.poisson;
            const float denom = (1.0f + poisson) * (1.0f - 2.0f * poisson);
            const float mu = youngs / (2.0f * (1.0f + poisson));
            const float lambda = (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
            mpm_detail::SandReturnMap<Arithmetic>(next.elastic_f, mu, lambda,
                material.dp_friction, material.dp_cohesion);
        }
    }
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
