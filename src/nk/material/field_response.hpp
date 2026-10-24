#pragma once
// Material-field shell response: one plane-stress law over membrane strain and plate curvature, with J2
// plasticity and two Maxwell branches at fixed committed history; the device evaluates float, host oracles double.

#include <cmath>
#include <cstdint>

#include "math/finite.hpp"
#include "nk/solve/residual_contract.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NUKA_MF_HD __host__ __device__
#else
#define NUKA_MF_HD
#endif

namespace nuka::nk {

// Registered field laws: the small-strain isotropic adapter, and Lagrangian log-strain orthotropic Hill plasticity.
inline constexpr uint32_t kMfSmallStrainLaw = 1u;
inline constexpr uint32_t kMfLogStrainOrthotropicLaw = 2u;
// Registered seam laws: the isotropic cohesive, and the mixed-mode cohesive with its own sliding strength and toughness.
inline constexpr uint32_t kMfIsotropicCohesive = 1u;
inline constexpr uint32_t kMfMixedModeCohesive = 2u;
// Registered section laws: the homogeneous plate, and an effective section with its own bending stiffness and
// declared curvature range.
inline constexpr uint32_t kMfPlateSection = 1u;
inline constexpr uint32_t kMfEffectiveSection = 2u;

// SI shell material: volumetric density, thickness, moduli, yield stress/moment and fracture strength/toughness.
// Maxwell moduli are elastic stiffness fractions, with relaxation times in seconds. The log-strain law reads youngs
// and poisson along the grain, plus the cross-grain modulus, shear modulus, Hill yield stresses and its strain range.
struct MfMaterial {
    float density = 0.0f;
    float thickness = 0.0f;
    float youngs = 0.0f;
    float poisson = 0.0f;
    float yield_stress = 0.0f;
    float hardening = 0.0f;
    float yield_moment = 0.0f;
    float bend_hardening = 0.0f;
    float maxwell_modulus[2]{};
    float maxwell_time[2]{};
    float bend_maxwell_modulus[2]{};
    float bend_maxwell_time[2]{};
    float tear_strength = 0.0f;
    float tear_energy = 0.0f;
    float shear_strength = 0.0f;
    uint32_t law = kMfSmallStrainLaw;
    float youngs_cd = 0.0f;
    float shear_modulus = 0.0f;
    float yield_stress_cd = 0.0f;  // zero: yield_stress
    float yield_shear = 0.0f;      // zero: yield_stress / sqrt(3)
    float yield_biaxial = 0.0f;    // zero: yield_stress
    float strain_range = 0.0f;     // largest log strain the law is calibrated for
    uint32_t fracture_law = kMfIsotropicCohesive;
    float shear_tear_energy = 0.0f;  // zero: tear_energy
    float mixity_exponent = 0.0f;    // zero: 1
    uint32_t section_law = kMfPlateSection;
    float bend_stiffness = 0.0f;   // N m along the grain; zero: the plate value
    float curvature_range = 0.0f;  // largest principal curvature, 1/m; zero: 0.1 / thickness
};
static_assert(sizeof(MfMaterial) == 32u * sizeof(float), "MfMaterial packs to 128 bytes");

// Each registered response declares its own packed state; element offsets address disjoint typed spans.
inline constexpr uint32_t kMfStateVersion = 2u;
inline constexpr uint32_t kMfPlaneState = 10u;
inline constexpr uint32_t kMfPlateState = 13u;
inline constexpr uint32_t kMfCohesiveState = 3u;
// Mixed-mode seams keep damage per pair, the cracked flag, the mode mixity per pair and the interval-start crack normal.
inline constexpr uint32_t kMfMixedCohesiveState = 8u;
inline constexpr uint32_t kMfSeamMixity = 3u;
inline constexpr uint32_t kMfSeamNormal = 5u;
inline constexpr uint32_t kMfResponseStateCapacity = kMfPlateState;
// A trial span appends the step's plastic or cohesive loss and its viscous loss to the state.
inline constexpr uint32_t kMfTrialLosses = 2u;
inline constexpr uint32_t kMfHingeDamage = 10u;
// A fracture seam is 1 here once its edge has cracked.
inline constexpr uint32_t kMfSeamCracked = 2u;

enum class MfStatus : uint32_t { Ok, InvalidParameters, LocalNonConvergence, NonFinite };

namespace mf {

// Plane-stress stiffness `modulus` C(nu) over a strain triple, J2 yield with linear hardening (none at zero
// yield), two Maxwell branches, and the strain beyond which the law no longer holds. `shape` (2h, 2a, 2n/3)
// scales the Hill yield metric (2/3) [[1, -h, 0], [-h, a, 0], [0, 0, n]]; (1, 2, 2) is J2.
struct PlaneStressLaw {
    float modulus;
    float poisson;
    float yield;
    float hardening;
    float maxwell[2];
    float time[2];
    float range;
    float shape[3]{1.0f, 2.0f, 2.0f};
};

template <class Real> struct LocalMaterialReport {
    MfStatus status = MfStatus::Ok;
    uint32_t iterations = 0u;
    Real normalized_residual = Real(0);
    bool converged = true;
    bool beyond_range = false;
};

// Work conjugacy uses reference-frame (11, 22, 2 12); membrane resultants are N/m and plate moments N.
// Energy and dissipation are J/m^2 of reference midsurface; tangent stores (11, 22, 33, 12, 13, 23).
template <class Real> struct FieldResponse {
    static constexpr DerivativeKind derivative = DerivativeKind::Consistent;
    Real psi;  // incremental potential from the committed state
    Real stress[3];
    Real tangent[6];
    Real history[kMfResponseStateCapacity];
    Real stored;   // elastic, Maxwell and hardening energy at the trial history
    Real plastic;  // plastic dissipation of the step
    Real viscous;  // viscous dissipation of the step
    bool yielded;
    bool beyond_range;
    LocalMaterialReport<Real> local;
};

NUKA_MF_HD inline bool ValidMaterial(const MfMaterial& m) {
    const float values[] = {m.yield_stress, m.hardening, m.yield_moment, m.bend_hardening,
        m.maxwell_modulus[0], m.maxwell_modulus[1], m.maxwell_time[0], m.maxwell_time[1],
        m.bend_maxwell_modulus[0], m.bend_maxwell_modulus[1], m.bend_maxwell_time[0], m.bend_maxwell_time[1],
        m.youngs_cd, m.shear_modulus, m.yield_stress_cd, m.yield_shear, m.yield_biaxial, m.strain_range,
        m.shear_tear_energy, m.mixity_exponent, m.bend_stiffness, m.curvature_range};
    for (float value : values)
        if (!math::IsFinite(value) || value < 0.0f) return false;
    const bool orthotropic = m.law == kMfLogStrainOrthotropicLaw;
    bool valid = math::IsFinite(m.density) && m.density > 0.0f && math::IsFinite(m.thickness) &&
                 m.thickness > 0.0f && math::IsFinite(m.youngs) && m.youngs > 0.0f &&
                 math::IsFinite(m.poisson) && m.poisson >= 0.0f &&
                 (m.law == kMfSmallStrainLaw || orthotropic) &&
                 m.yield_stress >= 0.0f && m.hardening >= 0.0f && m.yield_moment >= 0.0f &&
                 m.bend_hardening >= 0.0f && math::IsFinite(m.tear_strength) && m.tear_strength >= 0.0f &&
                 math::IsFinite(m.tear_energy) && m.tear_energy >= 0.0f &&
                 math::IsFinite(m.shear_strength) && m.shear_strength >= 0.0f &&
                 (m.fracture_law == kMfIsotropicCohesive || m.fracture_law == kMfMixedModeCohesive) &&
                 (m.section_law == kMfPlateSection ? m.bend_stiffness == 0.0f && m.curvature_range == 0.0f
                                                   : m.section_law == kMfEffectiveSection && m.curvature_range > 0.0f);
    // The orthotropic stiffness needs nu12^2 E2 < E1; the isotropic adapter needs nu < 1/2.
    if (orthotropic)
        valid = valid && m.youngs_cd > 0.0f && m.shear_modulus > 0.0f && m.strain_range > 0.0f &&
                double(m.poisson) * m.poisson * m.youngs_cd < double(m.youngs);
    else
        valid = valid && m.poisson < 0.5f;
    if (orthotropic && m.yield_stress > 0.0f) {
        const double y = m.yield_stress;
        const double a = m.yield_stress_cd > 0.0f ? y * y / (double(m.yield_stress_cd) * m.yield_stress_cd) : 1.0;
        const double b = m.yield_biaxial > 0.0f ? y * y / (double(m.yield_biaxial) * m.yield_biaxial) : 1.0;
        const double h = 0.5 * (1.0 + a - b);
        valid = valid && h * h < a;
    }
    for (uint32_t k = 0u; k < 2u; ++k) {
        valid = valid && m.maxwell_modulus[k] >= 0.0f && (m.maxwell_modulus[k] == 0.0f || m.maxwell_time[k] > 0.0f);
        valid = valid && m.bend_maxwell_modulus[k] >= 0.0f &&
                (m.bend_maxwell_modulus[k] == 0.0f || m.bend_maxwell_time[k] > 0.0f);
    }
    return valid;
}

// Plane-stress membrane stiffness Y t / (1 - nu^2) in N/m; along the grain E1 t / (1 - nu12 nu21) for law 2.
template <class Real> NUKA_MF_HD inline Real MembraneModulus(const MfMaterial& m) {
    const Real nu = m.poisson;
    if (m.law == kMfLogStrainOrthotropicLaw)
        return Real(m.youngs) * Real(m.thickness) / (Real(1) - nu * nu * Real(m.youngs_cd) / Real(m.youngs));
    return Real(m.youngs) * Real(m.thickness) / (Real(1) - nu * nu);
}

// Membrane shear stiffness Y t / (2 (1 + nu)), or G12 t, in N/m.
template <class Real> NUKA_MF_HD inline Real ShearModulus(const MfMaterial& m) {
    if (m.law == kMfLogStrainOrthotropicLaw) return Real(m.shear_modulus) * Real(m.thickness);
    return Real(m.youngs) * Real(m.thickness) / (Real(2) * (Real(1) + Real(m.poisson)));
}

// Plate bending stiffness D = Y t^3 / (12 (1 - nu^2)) in N m, or the effective section's own.
template <class Real> NUKA_MF_HD inline Real BendingModulus(const MfMaterial& m) {
    if (m.bend_stiffness > 0.0f) return Real(m.bend_stiffness);
    const Real t = m.thickness;
    return MembraneModulus<Real>(m) * t * t / Real(12);
}

// Orthotropic plane-stress stiffness of law 2 over (11, 22, 2 12) in the grain frame, times `scale`.
template <class Real> NUKA_MF_HD inline void OrthotropicEntries(const MfMaterial& m, Real scale, Real (&c)[6]) {
    const Real e1 = m.youngs, e2 = m.youngs_cd, nu = m.poisson;
    const Real den = Real(1) - nu * nu * e2 / e1;
    c[0] = scale * e1 / den;
    c[1] = scale * e2 / den;
    c[2] = scale * Real(m.shear_modulus);
    c[3] = scale * nu * e2 / den;
    c[4] = c[5] = Real(0);
}

// Hill yield shape of law 2 normalized to the grain yield stress; J2 otherwise.
NUKA_MF_HD inline void YieldShape(const MfMaterial& m, float (&shape)[3]) {
    shape[0] = 1.0f;
    shape[1] = 2.0f;
    shape[2] = 2.0f;
    if (m.law != kMfLogStrainOrthotropicLaw || !(m.yield_stress > 0.0f)) return;
    const double y = m.yield_stress;
    const double a = m.yield_stress_cd > 0.0f ? y * y / (double(m.yield_stress_cd) * m.yield_stress_cd) : 1.0;
    const double n = m.yield_shear > 0.0f ? y * y / (double(m.yield_shear) * m.yield_shear) : 3.0;
    const double b = m.yield_biaxial > 0.0f ? y * y / (double(m.yield_biaxial) * m.yield_biaxial) : 1.0;
    shape[0] = static_cast<float>(1.0 + a - b);
    shape[1] = static_cast<float>(2.0 * a);
    shape[2] = static_cast<float>(2.0 * n / 3.0);
}

// Membranes yield on stress resultants; strains past 5% leave the small-strain range, log strains past
// strain_range the log-strain law's.
NUKA_MF_HD inline PlaneStressLaw MembraneLaw(const MfMaterial& m) {
    const float t = m.thickness;
    const bool orthotropic = m.law == kMfLogStrainOrthotropicLaw;
    PlaneStressLaw law{MembraneModulus<float>(m), orthotropic ? 0.0f : m.poisson, m.yield_stress * t, m.hardening * t,
                       {m.maxwell_modulus[0], m.maxwell_modulus[1]}, {m.maxwell_time[0], m.maxwell_time[1]},
                       orthotropic ? m.strain_range : 0.05f};
    YieldShape(m, law.shape);
    return law;
}

// Plates yield on moments (Ilyushin); a curvature past the section's declared range, or for the plate one whose
// surface strain passes 5%, leaves the range.
NUKA_MF_HD inline PlaneStressLaw BendLaw(const MfMaterial& m) {
    PlaneStressLaw law{BendingModulus<float>(m), m.law == kMfLogStrainOrthotropicLaw ? 0.0f : m.poisson,
                       m.yield_moment, m.bend_hardening, {m.bend_maxwell_modulus[0], m.bend_maxwell_modulus[1]},
                       {m.bend_maxwell_time[0], m.bend_maxwell_time[1]},
                       m.curvature_range > 0.0f ? m.curvature_range : 0.1f / m.thickness};
    YieldShape(m, law.shape);
    return law;
}

template <class Real> NUKA_MF_HD inline void TangentProduct(const Real (&c)[6], const Real (&x)[3], Real (&y)[3]) {
    y[0] = c[0] * x[0] + c[3] * x[1] + c[4] * x[2];
    y[1] = c[3] * x[0] + c[1] * x[1] + c[5] * x[2];
    y[2] = c[4] * x[0] + c[5] * x[1] + c[2] * x[2];
}

// The local stress Jacobian includes the condensed plastic and Maxwell history response.
template <class Real>
NUKA_MF_HD inline void StressJacobianProduct(const FieldResponse<Real>& response,
                                            const Real (&direction)[3], Real (&product)[3]) {
    TangentProduct(response.tangent, direction, product);
}

// 1/2 x^T C(nu) x of a strain triple at unit modulus.
template <class Real> NUKA_MF_HD inline Real UnitEnergy(Real nu, const Real (&x)[3]) {
    return Real(0.5) * (x[0] * x[0] + x[1] * x[1] + Real(2) * nu * x[0] * x[1] +
                        Real(0.5) * (Real(1) - nu) * x[2] * x[2]);
}


// The eigenbasis shared by C(nu) and the J2 metric P is (11 + 22)/sqrt2, (11 - 22)/sqrt2, 12, where C has
// eigenvalues (1 + nu, 1 - nu, (1 - nu)/2) and P (1/3, 1, 2); the rotation is its own inverse.
inline constexpr double kHalfSqrt2 = 0.70710678118654752440;

template <class Real> struct StressMetric {
    Real value[3][3]{};
};

template <class Real>
NUKA_MF_HD inline StressMetric<Real> ElasticMetric(const PlaneStressLaw& law, const Real* entries = nullptr) {
    StressMetric<Real> c{};
    if (entries) {
        c.value[0][0] = entries[0]; c.value[1][1] = entries[1]; c.value[2][2] = entries[2];
        c.value[0][1] = c.value[1][0] = entries[3];
        c.value[0][2] = c.value[2][0] = entries[4];
        c.value[1][2] = c.value[2][1] = entries[5];
    } else {
        c.value[0][0] = c.value[1][1] = Real(law.modulus);
        c.value[0][1] = c.value[1][0] = Real(law.modulus) * Real(law.poisson);
        c.value[2][2] = Real(0.5) * Real(law.modulus) * (Real(1) - Real(law.poisson));
    }
    return c;
}

template <class Real>
NUKA_MF_HD inline void MetricProduct(const StressMetric<Real>& c, const Real (&x)[3], Real (&out)[3]) {
    for (uint32_t i = 0u; i < 3u; ++i)
        out[i] = c.value[i][0] * x[0] + c.value[i][1] * x[1] + c.value[i][2] * x[2];
}

template <class Real>
NUKA_MF_HD inline Real MetricEnergy(const StressMetric<Real>& c, const Real (&x)[3]) {
    Real force[3];
    MetricProduct(c, x, force);
    return Real(0.5) * (x[0] * force[0] + x[1] * force[1] + x[2] * force[2]);
}

// Stored energy at a committed history, using the same metric as the trial response.
template <class Real>
NUKA_MF_HD inline Real StoredEnergy(const PlaneStressLaw& law, const float* committed, const Real (&E)[3],
                                    const Real* elastic_entries = nullptr) {
    const StressMetric<Real> c = ElasticMetric(law, elastic_entries);
    Real z[kMfPlaneState];
    for (uint32_t k = 0u; k < kMfPlaneState; ++k) z[k] = committed != nullptr ? Real(committed[k]) : Real(0);
    const Real e[3] = {E[0] - z[0], E[1] - z[1], E[2] - z[2]};
    Real energy = MetricEnergy(c, e);
    for (uint32_t k = 0u; k < 2u; ++k) {
        const Real lag[3] = {e[0] - z[4u + 3u * k], e[1] - z[5u + 3u * k], e[2] - z[6u + 3u * k]};
        energy += Real(law.maxwell[k]) * MetricEnergy(c, lag);
    }
    return energy + Real(0.5) * Real(law.hardening) * z[3] * z[3];
}

template <class Real>
NUKA_MF_HD inline void YieldNormal(const float (&shape)[3], const Real (&s)[3], Real (&out)[3]) {
    out[0] = (Real(2) * s[0] - Real(shape[0]) * s[1]) / Real(3);
    out[1] = (Real(shape[1]) * s[1] - Real(shape[0]) * s[0]) / Real(3);
    out[2] = Real(shape[2]) * s[2];
}

// The inverse acts on I + gamma C P, which stays invertible for a positive semidefinite C.
template <class Real>
NUKA_MF_HD inline bool CondensedMetric(const float (&shape)[3], const StressMetric<Real>& c, Real gamma,
                                      const Real (&trial)[3], Real (&stress)[3], StressMetric<Real>& xi) {
    Real a[3][3], inverse[3][3];
    for (uint32_t i = 0u; i < 3u; ++i) {
        Real row[3] = {c.value[i][0], c.value[i][1], c.value[i][2]}, cp[3];
        YieldNormal(shape, row, cp);
        for (uint32_t j = 0u; j < 3u; ++j) a[i][j] = (i == j ? Real(1) : Real(0)) + gamma * cp[j];
    }
    const Real det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                     a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                     a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if (!(det > Real(0)) || !math::IsFinite(det)) return false;
    for (uint32_t i = 0u; i < 3u; ++i)
        for (uint32_t j = 0u; j < 3u; ++j)
            inverse[i][j] = (a[(j + 1u) % 3u][(i + 1u) % 3u] * a[(j + 2u) % 3u][(i + 2u) % 3u] -
                             a[(j + 1u) % 3u][(i + 2u) % 3u] * a[(j + 2u) % 3u][(i + 1u) % 3u]) / det;
    for (uint32_t i = 0u; i < 3u; ++i) {
        stress[i] = Real(0);
        for (uint32_t k = 0u; k < 3u; ++k) stress[i] += inverse[i][k] * trial[k];
        for (uint32_t j = 0u; j < 3u; ++j) {
            xi.value[i][j] = Real(0);
            for (uint32_t k = 0u; k < 3u; ++k) xi.value[i][j] += inverse[i][k] * c.value[k][j];
        }
    }
    return true;
}

// Plastic flow and Maxwell relaxation use the same elastic metric, including directional degradation.
template <class Real>
NUKA_MF_HD inline MfStatus EvaluatePlaneStress(const PlaneStressLaw& law, const float* committed, const Real (&E)[3],
                                               Real h, FieldResponse<Real>& out,
                                               LocalMaterialReport<Real>* report = nullptr,
                                               const Real* elastic_entries = nullptr) {
    const auto fail = [&](MfStatus status) {
        if (report) { *report = {}; report->status = status; report->converged = false; }
        return status;
    };
    if (!(h > Real(0)) || !math::IsFinite(h) || !(law.modulus > 0.0f) ||
        !math::IsFinite(law.modulus) || !math::IsFinite(law.poisson) || law.poisson < 0.0f ||
        law.poisson >= 0.5f || !math::IsFinite(law.yield) || law.yield < 0.0f ||
        !math::IsFinite(law.hardening) || law.hardening < 0.0f ||
        !math::IsFinite(law.range) || !(law.range > 0.0f)) return fail(MfStatus::InvalidParameters);
    for (uint32_t i = 0u; i < 3u; ++i)
        if (!math::IsFinite(E[i])) return fail(MfStatus::NonFinite);
    for (uint32_t i = 0u; committed && i < kMfPlaneState; ++i)
        if (!math::IsFinite(committed[i])) return fail(MfStatus::NonFinite);
    for (uint32_t i = 0u; i < 2u; ++i)
        if (!math::IsFinite(law.maxwell[i]) || law.maxwell[i] < 0.0f ||
            !math::IsFinite(law.time[i]) || law.time[i] < 0.0f ||
            (law.maxwell[i] > 0.0f && !(law.time[i] > 0.0f)))
            return fail(MfStatus::InvalidParameters);
    const StressMetric<Real> base = ElasticMetric(law, elastic_entries);
    for (uint32_t i = 0u; i < 3u; ++i)
        for (uint32_t j = 0u; j < 3u; ++j)
            if (!math::IsFinite(base.value[i][j])) return fail(MfStatus::NonFinite);
    FieldResponse<Real> r{};
    Real z[kMfPlaneState];
    for (uint32_t k = 0u; k < kMfPlaneState; ++k) z[k] = committed != nullptr ? Real(committed[k]) : Real(0);
    Real branch[2], total = Real(1), shift[3]{};
    for (uint32_t k = 0u; k < 2u; ++k) {
        branch[k] = law.maxwell[k] > 0.0f ? Real(law.maxwell[k]) / (Real(1) + h / Real(law.time[k])) : Real(0);
        total += branch[k];
        for (uint32_t i = 0u; i < 3u; ++i) shift[i] += branch[k] * z[4u + 3u * k + i];
    }
    StressMetric<Real> c = base, tangent;
    for (uint32_t i = 0u; i < 3u; ++i)
        for (uint32_t j = 0u; j < 3u; ++j) c.value[i][j] *= total;
    Real e[3] = {E[0] - z[0], E[1] - z[1], E[2] - z[2]};
    const Real d[3] = {e[0] - shift[0] / total, e[1] - shift[1] / total, e[2] - shift[2] / total};
    Real s[3], normal[3];
    MetricProduct(c, d, s);
    YieldNormal(law.shape, s, normal);
    const Real trial[3] = {s[0], s[1], s[2]};
    const Real yield = Real(law.yield) + Real(law.hardening) * z[3];
    const Real size2 = s[0] * normal[0] + s[1] * normal[1] + s[2] * normal[2];
    r.yielded = law.yield > 0.0f && size2 > Real(2.0 / 3.0) * yield * yield;
    tangent = c;
    Real gamma = Real(0), next_yield = yield;
    if (r.yielded) {
        const Real root = Real(0.81649658092772603273), slope = Real(2.0 / 3.0) * Real(law.hardening);
        const Real tolerance = sizeof(Real) == sizeof(double) ? Real(1e-12) : Real(4e-7);
        Real lo = Real(0), hi = slope > Real(0) ? (Real(1) - Real(1e-6)) / slope : Real(1);
        if (slope == Real(0)) {
            Real rate = Real(0);
            for (uint32_t i = 0u; i < 3u; ++i)
                for (uint32_t j = 0u; j < 3u; ++j) rate += std::fabs(c.value[i][j]);
            hi = Real(1) / rate;
            for (uint32_t bracket = 0u; bracket < 64u; ++bracket) {
                if (!CondensedMetric(law.shape, c, hi, trial, s, tangent)) return fail(MfStatus::LocalNonConvergence);
                ++r.local.iterations;
                YieldNormal(law.shape, s, normal);
                const Real norm = std::sqrt(s[0] * normal[0] + s[1] * normal[1] + s[2] * normal[2]);
                if (norm <= root * yield) break;
                hi *= Real(2);
                if (bracket == 63u) return fail(MfStatus::LocalNonConvergence);
            }
        }
        r.local.converged = false;
        for (uint32_t iteration = 0u; iteration < 32u; ++iteration) {
            if (!CondensedMetric(law.shape, c, gamma, trial, s, tangent)) return fail(MfStatus::LocalNonConvergence);
            ++r.local.iterations;
            YieldNormal(law.shape, s, normal);
            Real n[3];
            MetricProduct(tangent, normal, n);
            const Real size = std::sqrt(s[0] * normal[0] + s[1] * normal[1] + s[2] * normal[2]);
            const Real residual = Real(1) / size - (Real(1) - slope * gamma) / (root * yield);
            r.local.normalized_residual = std::fabs(residual) * size;
            if (r.local.normalized_residual <= tolerance) { r.local.converged = true; break; }
            if (residual < Real(0)) lo = gamma;
            else hi = gamma;
            const Real derivative = (normal[0] * n[0] + normal[1] * n[1] + normal[2] * n[2]) /
                                        (size * size * size) + slope / (root * yield);
            const Real next = gamma - residual / derivative;
            gamma = next > lo && next < hi ? next : Real(0.5) * (lo + hi);
        }
        if (!r.local.converged) {
            r.local.status = MfStatus::LocalNonConvergence;
            if (report) *report = r.local;
            return MfStatus::LocalNonConvergence;
        }
        next_yield = yield / (Real(1) - slope * gamma);
        Real n[3];
        MetricProduct(tangent, normal, n);
        const Real den = normal[0] * n[0] + normal[1] * n[1] + normal[2] * n[2] +
            Real(4.0 / 9.0) * Real(law.hardening) * next_yield * next_yield / (Real(1) - slope * gamma);
        for (uint32_t i = 0u; i < 3u; ++i) {
            e[i] -= gamma * normal[i];
            for (uint32_t j = 0u; j < 3u; ++j) tangent.value[i][j] -= n[i] * n[j] / den;
        }
    }
    r.history[3] = z[3] + Real(2.0 / 3.0) * gamma * next_yield;
    for (uint32_t i = 0u; i < 3u; ++i) {
        r.history[i] = E[i] - e[i];
        r.stress[i] = s[i];
        r.tangent[i] = tangent.value[i][i];
    }
    r.tangent[3] = Real(0.5) * (tangent.value[0][1] + tangent.value[1][0]);
    r.tangent[4] = Real(0.5) * (tangent.value[0][2] + tangent.value[2][0]);
    r.tangent[5] = Real(0.5) * (tangent.value[1][2] + tangent.value[2][1]);
    Real psi = MetricEnergy(base, e), stored = psi, viscous = Real(0);
    for (uint32_t k = 0u; k < 2u; ++k) {
        if (!(law.maxwell[k] > 0.0f)) continue;
        const Real relax = h / Real(law.time[k]);
        Real lag[3], from[3], moved[3];
        for (uint32_t i = 0u; i < 3u; ++i) {
            const Real q = z[4u + 3u * k + i], next = (q + relax * e[i]) / (Real(1) + relax);
            r.history[4u + 3u * k + i] = next;
            lag[i] = e[i] - next;
            from[i] = e[i] - q;
            moved[i] = next - q;
        }
        psi += branch[k] * MetricEnergy(base, from);
        stored += Real(law.maxwell[k]) * MetricEnergy(base, lag);
        viscous += Real(2) * Real(law.maxwell[k]) / relax * MetricEnergy(base, moved);
    }
    const Real hardening = law.hardening, grown = r.history[3] - z[3];
    r.psi = psi + Real(0.5) * hardening * (r.history[3] * r.history[3] - z[3] * z[3]) + Real(law.yield) * grown;
    r.stored = stored + Real(0.5) * hardening * r.history[3] * r.history[3];
    r.plastic = Real(law.yield) * grown;
    r.viscous = viscous;
    r.beyond_range =
        std::fmax(std::fmax(std::fabs(E[0]), std::fabs(E[1])), Real(0.5) * std::fabs(E[2])) > Real(law.range);
    r.local.beyond_range = r.beyond_range;
    if (!math::IsFinite(r.psi) || !math::IsFinite(r.stored) || !math::IsFinite(r.plastic) ||
        !math::IsFinite(r.viscous)) return fail(MfStatus::NonFinite);
    for (uint32_t i = 0u; i < 3u; ++i)
        if (!math::IsFinite(r.stress[i])) return fail(MfStatus::NonFinite);
    for (uint32_t i = 0u; i < 6u; ++i)
        if (!math::IsFinite(r.tangent[i])) return fail(MfStatus::NonFinite);
    for (uint32_t i = 0u; i < kMfPlaneState; ++i)
        if (!math::IsFinite(r.history[i])) return fail(MfStatus::NonFinite);
    out = r;
    if (report) *report = r.local;
    return MfStatus::Ok;
}

// Plate stiffness over Voigt curvature: C(nu) D, or the law-2 orthotropic section stiffness times t^2 / 12, scaled
// for an effective section to its bending stiffness along the grain.
template <class Real> NUKA_MF_HD inline StressMetric<Real> BendBase(const MfMaterial& m) {
    if (m.law != kMfLogStrainOrthotropicLaw) return ElasticMetric<Real>(BendLaw(m));
    const Real t = m.thickness, e1 = m.youngs;
    const Real den = Real(1) - Real(m.poisson) * Real(m.poisson) * Real(m.youngs_cd) / e1;
    Real entries[6];
    OrthotropicEntries<Real>(m, m.bend_stiffness > 0.0f ? Real(m.bend_stiffness) * den / e1 : t * t * t / Real(12),
                             entries);
    return ElasticMetric<Real>(BendLaw(m), entries);
}

// Lagrangian log strain (1/2) ln C of a membrane Green strain (11, 22, 2 12), with C = I + 2 E, over the eigenbasis
// (c, s), (-s, c) of C. first[a][b] is the divided difference f[l_a, l_b] of f(l) = (ln l) / 2 and third[k] the
// second divided difference with k of its three nodes at l_2.
template <class Real> struct LogStrainMap {
    Real strain[3];
    Real c, s;
    Real first[2][2];
    Real third[4];
};

// log1p(u) / u, (u / (1 + u) - log1p(u)) / u^2 and (log1p(u) - u) / u^2; series below u = 1/4 keep close
// eigenvalue pairs at full precision.
template <class Real> NUKA_MF_HD inline void LogDifferences(Real u, Real& first, Real& lead, Real& trail) {
    if (u < Real(0.25)) {
        const int terms = sizeof(Real) == sizeof(double) ? 28 : 14;
        first = lead = trail = Real(0);
        for (int j = terms - 1; j >= 0; --j) {
            const Real sign = (j & 1) != 0 ? Real(1) : Real(-1);
            first = first * u - sign / Real(j + 1);
            lead = lead * u + sign * Real(j + 1) / Real(j + 2);
            trail = trail * u + sign / Real(j + 2);
        }
        return;
    }
    const Real lg = std::log1p(u);
    first = lg / u;
    lead = (u / (Real(1) + u) - lg) / (u * u);
    trail = (lg - u) / (u * u);
}

// Eigenvalues come from the strain itself, so small strains keep their relative precision.
template <class Real> NUKA_MF_HD inline bool MapLogStrain(const Real (&E)[3], LogStrainMap<Real>& map) {
    const Real half = E[0] - E[1], shear = E[2], grow = E[0] + E[1];
    const Real r = std::sqrt(half * half + shear * shear);
    const Real l1m = grow + r, l1 = Real(1) + l1m;
    const Real l2m = (Real(2) * grow + Real(4) * E[0] * E[1] - shear * shear - l1m) / l1, l2 = Real(1) + l2m;
    if (!(l2 > Real(0)) || !math::IsFinite(l1) || !math::IsFinite(l2)) return false;
    const Real u = Real(2) * r / l2, angle = Real(0.5) * std::atan2(shear, half);
    map.c = std::cos(angle);
    map.s = std::sin(angle);
    Real first, lead, trail;
    LogDifferences(u, first, lead, trail);
    const Real low = Real(0.5) * std::log1p(l2m), gap = Real(0.5) * u * first;
    map.strain[0] = low + gap * map.c * map.c;
    map.strain[1] = low + gap * map.s * map.s;
    map.strain[2] = Real(2) * gap * map.c * map.s;
    map.first[0][0] = Real(0.5) / l1;
    map.first[1][1] = Real(0.5) / l2;
    map.first[0][1] = map.first[1][0] = Real(0.5) * first / l2;
    map.third[0] = Real(-0.25) / (l1 * l1);
    map.third[1] = Real(0.5) * lead / (l2 * l2);
    map.third[2] = Real(0.5) * trail / (l2 * l2);
    map.third[3] = Real(-0.25) / (l2 * l2);
    return true;
}

// A symmetric tensor (11, 22, 12) in and out of the eigenbasis.
template <class Real>
NUKA_MF_HD inline void ToEigen(const LogStrainMap<Real>& m, const Real (&x)[3], Real (&y)[3]) {
    const Real cc = m.c * m.c, ss = m.s * m.s, cs = m.c * m.s;
    y[0] = cc * x[0] + ss * x[1] + Real(2) * cs * x[2];
    y[1] = ss * x[0] + cc * x[1] - Real(2) * cs * x[2];
    y[2] = cs * (x[1] - x[0]) + (cc - ss) * x[2];
}

template <class Real>
NUKA_MF_HD inline void FromEigen(const LogStrainMap<Real>& m, const Real (&y)[3], Real (&x)[3]) {
    const Real cc = m.c * m.c, ss = m.s * m.s, cs = m.c * m.s;
    x[0] = cc * y[0] + ss * y[1] - Real(2) * cs * y[2];
    x[1] = ss * y[0] + cc * y[1] + Real(2) * cs * y[2];
    x[2] = cs * (y[0] - y[1]) + (cc - ss) * y[2];
}

// Second Piola stress S = 2 Df[T] of a log-space stress T and its stiffness dS/dE = 4 Df C Df + 4 D2f[T];
// without `second_order` the stiffness keeps 4 Df C Df, positive definite with the log-space tangent C.
template <class Real>
NUKA_MF_HD inline void PullBackLogResponse(const LogStrainMap<Real>& m, FieldResponse<Real>& r, bool second_order) {
    const Real k[3] = {m.first[0][0], m.first[1][1], m.first[0][1]};
    Real stress[3], t[3];
    for (uint32_t i = 0u; i < 3u; ++i) stress[i] = r.stress[i];
    ToEigen(m, stress, t);
    const Real pulled[3] = {Real(2) * k[0] * t[0], Real(2) * k[1] * t[1], Real(2) * k[2] * t[2]};
    FromEigen(m, pulled, stress);
    Real column[3][3];
    for (uint32_t j = 0u; j < 3u; ++j) {
        const Real unit[3] = {j == 0u ? Real(1) : Real(0), j == 1u ? Real(1) : Real(0), j == 2u ? Real(0.5) : Real(0)};
        Real e[3], d[3], dt[3], dte[3];
        ToEigen(m, unit, e);
        const Real de[3] = {Real(2) * k[0] * e[0], Real(2) * k[1] * e[1], Real(2) * k[2] * e[2]};
        FromEigen(m, de, d);
        const Real voigt[3] = {d[0], d[1], Real(2) * d[2]};
        TangentProduct(r.tangent, voigt, dt);
        ToEigen(m, dt, dte);
        Real out[3] = {Real(2) * k[0] * dte[0], Real(2) * k[1] * dte[1], Real(2) * k[2] * dte[2]};
        if (second_order) {
            out[0] += Real(8) * (m.third[0] * t[0] * e[0] + m.third[1] * t[2] * e[2]);
            out[1] += Real(8) * (m.third[2] * t[2] * e[2] + m.third[3] * t[1] * e[1]);
            out[2] += Real(4) * (m.third[1] * (t[0] * e[2] + e[0] * t[2]) + m.third[2] * (t[2] * e[1] + e[2] * t[1]));
        }
        FromEigen(m, out, column[j]);
    }
    for (uint32_t i = 0u; i < 3u; ++i) {
        r.stress[i] = stress[i];
        r.tangent[i] = column[i][i];
    }
    r.tangent[3] = Real(0.5) * (column[1][0] + column[0][1]);
    r.tangent[4] = Real(0.5) * (column[2][0] + column[0][2]);
    r.tangent[5] = Real(0.5) * (column[2][1] + column[1][2]);
}

// Membrane response over Green strain (11, 22, 2 12): the small-strain law directly, or the orthotropic law in
// Lagrangian log strain with stress and stiffness carried back; energies and history need no map.
template <class Real>
NUKA_MF_HD inline MfStatus MembraneResponse(const MfMaterial& m, const float* committed, const Real (&E)[3], Real h,
                                            FieldResponse<Real>& out, LocalMaterialReport<Real>* report = nullptr,
                                            bool second_order = true) {
    if (m.law != kMfLogStrainOrthotropicLaw) return EvaluatePlaneStress(MembraneLaw(m), committed, E, h, out, report);
    LogStrainMap<Real> map;
    if (!MapLogStrain(E, map)) {
        if (report) { *report = {}; report->status = MfStatus::NonFinite; report->converged = false; }
        return MfStatus::NonFinite;
    }
    Real entries[6];
    OrthotropicEntries<Real>(m, Real(m.thickness), entries);
    FieldResponse<Real> r;
    const MfStatus status = EvaluatePlaneStress(MembraneLaw(m), committed, map.strain, h, r, report, entries);
    if (status != MfStatus::Ok) return status;
    PullBackLogResponse(map, r, second_order);
    // The range bounds the principal log strains (1/2) ln l.
    const Real split = map.strain[0] - map.strain[1], spread = std::sqrt(split * split + map.strain[2] * map.strain[2]);
    const Real low = Real(0.5) * (map.strain[0] + map.strain[1] - spread);
    r.beyond_range = r.local.beyond_range = std::fmax(std::fabs(low), std::fabs(low + spread)) > Real(m.strain_range);
    if (report) report->beyond_range = r.beyond_range;
    out = r;
    return MfStatus::Ok;
}

// A vanishing eigenvalue of C takes the log-strain energy without bound.
template <class Real>
NUKA_MF_HD inline Real MembraneStoredEnergy(const MfMaterial& m, const float* committed, const Real (&E)[3]) {
    if (m.law != kMfLogStrainOrthotropicLaw) return StoredEnergy(MembraneLaw(m), committed, E);
    LogStrainMap<Real> map;
    if (!MapLogStrain(E, map)) return -std::log(Real(0));
    Real entries[6];
    OrthotropicEntries<Real>(m, Real(m.thickness), entries);
    return StoredEnergy(MembraneLaw(m), committed, map.strain, entries);
}

// Bilinear traction-separation of a seam per unit area: stiffness up to the strength, then linear softening
// that has spent the toughness when the separation parts. Damage follows the largest separation reached.
struct CohesiveLaw {
    double stiffness;  // Pa/m
    double strength;   // Pa
    double toughness;  // J/m^2
    NUKA_MF_HD double Onset() const { return strength / stiffness; }
    NUKA_MF_HD double Parting() const { return 2.0 * toughness / strength; }
};

// Damage D at a largest separation, where (1 - D) K r meets the softening line.
NUKA_MF_HD inline double SeamDamage(const CohesiveLaw& law, double reached) {
    const double onset = law.Onset(), parting = law.Parting();
    if (!(reached > onset)) return 0.0;
    if (reached >= parting) return 1.0;
    return parting * (reached - onset) / (reached * (parting - onset));
}

// Share of the strength a seam has lost at its largest separation: zero up to the peak, one once parted.
NUKA_MF_HD inline double SeamSpent(const CohesiveLaw& law, double reached) {
    const double onset = law.Onset(), parting = law.Parting();
    if (!(reached > onset)) return 0.0;
    return reached >= parting ? 1.0 : (reached - onset) / (parting - onset);
}

// Incremental potential, secant stiffness T / r, and the stored energy and loss once the step's
// separation is committed.
struct SeamResponse {
    double psi;
    double secant;
    double reached;
    double stored;
    double loss;
};

NUKA_MF_HD inline SeamResponse EvaluateSeam(const CohesiveLaw& law, double reached, double r) {
    const double onset = law.Onset(), parting = law.Parting(), slope = law.strength / (parting - onset);
    const double knee = fmax(reached, onset);
    SeamResponse s{0.0, 0.0, fmax(reached, r), 0.0, 0.0};
    if (knee >= parting) return s;
    const double held = (1.0 - SeamDamage(law, knee)) * law.stiffness;
    auto soften = [&](double x) { return slope * (parting * x - 0.5 * x * x); };
    if (r <= knee) {
        s.psi = 0.5 * held * r * r;
        s.secant = held;
    } else {
        const double end = fmin(r, parting);
        s.psi = 0.5 * held * knee * knee + soften(end) - soften(knee);
        s.secant = r < parting ? slope * (parting - r) / r : 0.0;
    }
    if (s.reached < parting) s.stored = 0.5 * (1.0 - SeamDamage(law, s.reached)) * law.stiffness * r * r;
    s.loss = s.psi - s.stored;
    return s;
}

// Mixed-mode seam per unit area. Opening along the crack normal and sliding share one damage D on the energy
// q = kn <dn>+^2 / 2 + ks |dt|^2 / 2, with ks = kn (tau / sigma)^2 so (Tn / sigma)^2 + (Ts / tau)^2 = 1 at onset.
struct MixedCohesiveLaw {
    double normal_stiffness;  // Pa/m
    double normal_strength;   // Pa
    double shear_strength;    // Pa
    double normal_toughness;  // J/m^2
    double shear_toughness;   // J/m^2
    double exponent;          // Benzeggagh-Kenane exponent on the sliding energy fraction
    NUKA_MF_HD double ShearStiffness() const {
        const double ratio = shear_strength / normal_strength;
        return normal_stiffness * ratio * ratio;
    }
    NUKA_MF_HD double Onset() const { return 0.5 * normal_strength * normal_strength / normal_stiffness; }
    NUKA_MF_HD double Toughness(double mixity) const {
        return normal_toughness + (shear_toughness - normal_toughness) * pow(fmin(fmax(mixity, 0.0), 1.0), exponent);
    }
};

// Bilinear in lambda = sqrt(q / onset energy): T falls linearly from the peak and spends Gc(mixity) when lambda
// reaches Gc / onset energy. A pair's state is its knee, the lambda of its damage under its mixity: 1 - D =
// (parting - knee) / (knee (parting - 1)); zero is intact and a negative knee marks a parted pair.
NUKA_MF_HD inline double MixedHeld(const MixedCohesiveLaw& law, double knee, double mixity) {
    if (knee < 0.0) return 0.0;
    if (!(knee > 1.0)) return 1.0;
    const double parting = law.Toughness(mixity) / law.Onset();
    return knee < parting ? (parting - knee) / (knee * (parting - 1.0)) : 0.0;
}

// Share of the strength a mixed pair has lost at its knee: the traction falls linearly in lambda past the peak.
NUKA_MF_HD inline double MixedSpent(const MixedCohesiveLaw& law, double knee, double mixity) {
    if (knee < 0.0) return 1.0;
    if (!(knee > 1.0)) return 0.0;
    const double parting = law.Toughness(mixity) / law.Onset();
    return knee < parting ? (knee - 1.0) / (parting - 1.0) : 1.0;
}

// The incremental potential integrates min(1 - D_committed, 1 - D(lambda)) dq at the committed mixity, so each step
// dissipates q dD and damage never heals. The new mixity re-expresses the knee for the same integrity in float
// storage, one step up where rounding would heal it. Compression is undamaged until the pair parts.
struct MixedSeamResponse {
    double psi;      // incremental potential from zero separation
    double keep;     // 1 - D at the step's end, scaling opening and sliding tractions
    double knee;     // committed knee under the new mixity
    double mixity;   // sliding share of q, the committed one while q is zero
    double energy;   // undamaged opening and sliding energy q
    double contact;  // compression energy
    double squeeze;  // compression stiffness, zero once parted
};

NUKA_MF_HD inline MixedSeamResponse EvaluateMixedSeam(const MixedCohesiveLaw& law, double knee, double mixity,
                                                      double opening, double sliding_sq) {
    const double onset = law.Onset(), kn = law.normal_stiffness, ks = law.ShearStiffness();
    const double held = MixedHeld(law, knee, mixity), tension = fmax(opening, 0.0), closing = fmin(opening, 0.0);
    MixedSeamResponse s{};
    s.energy = 0.5 * kn * tension * tension + 0.5 * ks * sliding_sq;
    s.squeeze = held > 0.0 ? kn : 0.0;
    s.contact = 0.5 * s.squeeze * closing * closing;
    s.psi = s.contact;
    s.knee = knee;
    s.mixity = mixity;
    if (!(held > 0.0)) return s;
    if (s.energy > 0.0) s.mixity = double(float(0.5 * ks * sliding_sq / s.energy));
    const double parting = law.Toughness(mixity) / onset, lambda = sqrt(s.energy / onset), reached = fmax(knee, 1.0);
    auto soften = [&](double x) { return 2.0 * onset * (parting * x - 0.5 * x * x) / (parting - 1.0); };
    if (lambda <= reached) {
        s.keep = held;
        s.psi += held * s.energy;
    } else {
        const double end = fmin(lambda, parting);
        s.keep = lambda < parting ? (parting - lambda) / (lambda * (parting - 1.0)) : 0.0;
        s.psi += held * onset * reached * reached + soften(end) - soften(reached);
    }
    if (s.keep >= 1.0) return s;
    if (!(s.keep > 0.0)) {
        s.knee = -1.0;
        return s;
    }
    const double next = law.Toughness(s.mixity) / onset;
    float stored = float(next / (1.0 + s.keep * (next - 1.0)));
    if (MixedHeld(law, stored, s.mixity) > s.keep) stored = nextafterf(stored, 2.0f * stored);
    s.knee = stored;
    return s;
}

// Stored energy of a pair at a committed knee and mixity.
NUKA_MF_HD inline double MixedStored(const MixedCohesiveLaw& law, double knee, double mixity,
                                     const MixedSeamResponse& s) {
    const double held = MixedHeld(law, knee, mixity);
    return held * s.energy + (held > 0.0 ? s.contact : 0.0);
}

}  // namespace mf

}  // namespace nuka::nk

#undef NUKA_MF_HD
