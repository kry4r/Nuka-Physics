#pragma once

#include <cmath>

#include "nk/material/hencky_j2.hpp"
#include "nk/material/mpm_transfer.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NUKA_MPM_ENERGY_HD __host__ __device__
#else
#define NUKA_MPM_ENERGY_HD
#endif

namespace nuka::nk::material {

// Stored energy per reference volume, including the recoverable isotropic hardening energy.
NUKA_MPM_ENERGY_HD inline ConstitutiveStatus EvaluateMpmStoredEnergy(
    const MpmMaterial& material, const float* deformation, float plastic_strain, double& energy) {
    if (!ValidMpmConstitutiveParameters(material)) return ConstitutiveStatus::InvalidParameters;
    if (deformation == nullptr) return ConstitutiveStatus::InvalidState;
    const double determinant = double(deformation[0]) *
        (double(deformation[4]) * deformation[8] - double(deformation[5]) * deformation[7]) -
        double(deformation[1]) *
        (double(deformation[3]) * deformation[8] - double(deformation[5]) * deformation[6]) +
        double(deformation[2]) *
        (double(deformation[3]) * deformation[7] - double(deformation[4]) * deformation[6]);
    if (!(determinant > 0.0) || !math::IsFinite(determinant))
        return ConstitutiveStatus::SingularDeformation;
    double value = 0.0;
    if (material.model_kind == 3.0f) {
        if (determinant < 1.0) {
            const double exponent = 1.0 - material.tait_gamma;
            const double integral = exponent == 0.0 ? std::log(determinant)
                : std::expm1(exponent * std::log(determinant)) / exponent;
            value = material.bulk_modulus * (determinant - 1.0 - integral);
        }
    } else {
        const double mu = material.youngs / (2.0 * (1.0 + material.poisson));
        const double lambda = material.youngs * material.poisson /
            ((1.0 + material.poisson) * (1.0 - 2.0 * material.poisson));
        if (material.model_kind == 2.0f) {
            double norm = 0.0;
            for (int k = 0; k < 9; ++k) norm += double(deformation[k]) * deformation[k];
            const double log_j = std::log(determinant);
            value = 0.5 * mu * (norm - 3.0) - mu * log_j + 0.5 * lambda * log_j * log_j;
        } else {
            detail::PrincipalStretch principal;
            const auto status = detail::Decompose(deformation, principal);
            if (status != ConstitutiveStatus::Ok) return status;
            double trace = 0.0, norm = 0.0;
            for (double log_stretch : principal.log_stretch) {
                const double strain = material.model_kind == 0.0f ? std::expm1(log_stretch) : log_stretch;
                trace += strain;
                norm += strain * strain;
            }
            if (material.model_kind == 0.0f)
                value = mu * norm + 0.5 * lambda * (determinant - 1.0) * (determinant - 1.0);
            else if (material.model_kind == 4.0f || material.model_kind == MpmMaterial::kHenckyJ2 ||
                     material.model_kind == MpmMaterial::kHenckyElastic)
                value = mu * norm + 0.5 * lambda * trace * trace;
            else return ConstitutiveStatus::InvalidParameters;
        }
        if (material.model_kind == MpmMaterial::kHenckyJ2) {
            if (!(plastic_strain >= 0.0f) || !(material.hardening_modulus >= 0.0f))
                return ConstitutiveStatus::InvalidState;
            value += 0.5 * material.hardening_modulus * double(plastic_strain) * plastic_strain;
        }
    }
    if (!math::IsFinite(value)) return ConstitutiveStatus::Overflow;
    energy = value;
    return ConstitutiveStatus::Ok;
}

// The compact dual-lattice covariance includes off-diagonal moments as well as its diagonal.
NUKA_MPM_ENERGY_HD inline void MpmCovariance(const MpmCompactAxis (&axes)[3], float dx,
                                            double (&covariance)[3][3]) {
    for (int a = 0; a < 3; ++a) {
        covariance[a][a] = double(dx) * dx * axes[a].moment;
        for (int b = 0; b < a; ++b) {
            double moment = 0.0;
            for (uint32_t lattice = 0u; lattice < kMpmLattices; ++lattice)
                moment += double(kMpmLatticeShare) * axes[a].first[lattice] * axes[b].first[lattice];
            covariance[a][b] = covariance[b][a] = double(dx) * dx * moment;
        }
    }
}

NUKA_MPM_ENERGY_HD inline double MpmAffineKinetic(const float* affine, double mass,
    const MpmCompactAxis (&axes)[3], float dx) {
    double covariance[3][3]{};
    MpmCovariance(axes, dx, covariance);
    double value = 0.0;
    for (int r = 0; r < 3; ++r)
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                value += double(affine[3 * r + a]) * covariance[a][b] * affine[3 * r + b];
    return 0.5 * mass * value;
}

NUKA_MPM_ENERGY_HD inline void MpmAffineAngularMomentum(const float* affine, double mass,
    const MpmCompactAxis (&axes)[3], float dx, double (&spin)[3]) {
    double covariance[3][3]{};
    MpmCovariance(axes, dx, covariance);
    for (int a = 0; a < 3; ++a) {
        const int b = (a + 1) % 3, c = (a + 2) % 3;
        spin[a] = 0.0;
        for (int k = 0; k < 3; ++k)
            spin[a] += mass * (covariance[b][k] * affine[3 * c + k] -
                              covariance[c][k] * affine[3 * b + k]);
    }
}

}  // namespace nuka::nk::material

#undef NUKA_MPM_ENERGY_HD
