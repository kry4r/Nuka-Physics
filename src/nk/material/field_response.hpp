#pragma once
// Material-field shell response: one plane-stress law over membrane strain and plate curvature, with J2
// plasticity and two Maxwell branches, evaluated at fixed committed history free of side effects. The device
// evaluates Real = float, the host oracles Real = double; the plastic return mapping always runs in double.

#include <cmath>
#include <cstdint>

#include "math/finite.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NUKA_MF_HD __host__ __device__
#else
#define NUKA_MF_HD
#endif

namespace nuka::nk {

// Density kg/m^3, thickness m, moduli and yield stress Pa, yield moment N m per m, bend hardening N m,
// Maxwell moduli as multiples of the elastic stiffness and their relaxation times in s.
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
};
static_assert(sizeof(MfMaterial) == 16u * sizeof(float), "MfMaterial packs to 64 bytes");

// Committed history of an element's strain triple: plastic strain, equivalent plastic strain and two Maxwell
// strains, all in the (11, 22, 2 12) order of the strain.
inline constexpr uint32_t kMfHistory = 10u;

enum class MfStatus : uint32_t { Ok, InvalidParameters, LocalNonConvergence, NonFinite };

namespace mf {

// Plane-stress stiffness `modulus` C(nu) over a strain triple, J2 yield with linear hardening (none at zero
// yield), two Maxwell branches, and the strain beyond which the small-strain assumption no longer holds.
struct PlaneStressLaw {
    float modulus;
    float poisson;
    float yield;
    float hardening;
    float maxwell[2];
    float time[2];
    float range;
};

// Stress and tangent follow the strain triple; the tangent stores (11, 22, 33, 12, 13, 23).
template <class Real> struct FieldResponse {
    Real psi;  // incremental potential from the committed state
    Real stress[3];
    Real tangent[6];
    Real history[kMfHistory];
    Real stored;   // elastic, Maxwell and hardening energy at the trial history
    Real plastic;  // plastic dissipation of the step
    Real viscous;  // viscous dissipation of the step
    bool yielded;
    bool beyond_range;
};

NUKA_MF_HD inline bool ValidMaterial(const MfMaterial& m) {
    bool valid = math::IsFinite(m.density) && m.density > 0.0f && math::IsFinite(m.thickness) &&
                 m.thickness > 0.0f && math::IsFinite(m.youngs) && m.youngs > 0.0f && m.poisson >= 0.0f &&
                 m.poisson < 0.5f && m.yield_stress >= 0.0f && m.hardening >= 0.0f && m.yield_moment >= 0.0f &&
                 m.bend_hardening >= 0.0f;
    for (uint32_t k = 0u; k < 2u; ++k) {
        valid = valid && m.maxwell_modulus[k] >= 0.0f && (m.maxwell_modulus[k] == 0.0f || m.maxwell_time[k] > 0.0f);
        valid = valid && m.bend_maxwell_modulus[k] >= 0.0f &&
                (m.bend_maxwell_modulus[k] == 0.0f || m.bend_maxwell_time[k] > 0.0f);
    }
    return valid;
}

// Plane-stress membrane stiffness Y t / (1 - nu^2) in N/m.
template <class Real> NUKA_MF_HD inline Real MembraneModulus(const MfMaterial& m) {
    const Real nu = m.poisson;
    return Real(m.youngs) * Real(m.thickness) / (Real(1) - nu * nu);
}

// Membrane shear stiffness Y t / (2 (1 + nu)) in N/m.
template <class Real> NUKA_MF_HD inline Real ShearModulus(const MfMaterial& m) {
    return Real(m.youngs) * Real(m.thickness) / (Real(2) * (Real(1) + Real(m.poisson)));
}

// Plate bending stiffness D = Y t^3 / (12 (1 - nu^2)) in N m.
template <class Real> NUKA_MF_HD inline Real BendingModulus(const MfMaterial& m) {
    const Real t = m.thickness;
    return MembraneModulus<Real>(m) * t * t / Real(12);
}

// Membranes yield on stress resultants; strains past 5% leave the small-strain range.
NUKA_MF_HD inline PlaneStressLaw MembraneLaw(const MfMaterial& m) {
    const float t = m.thickness;
    return {MembraneModulus<float>(m), m.poisson, m.yield_stress * t, m.hardening * t,
            {m.maxwell_modulus[0], m.maxwell_modulus[1]}, {m.maxwell_time[0], m.maxwell_time[1]}, 0.05f};
}

// Plates yield on moments (Ilyushin); a curvature whose surface strain passes 5% leaves the range.
NUKA_MF_HD inline PlaneStressLaw BendLaw(const MfMaterial& m) {
    return {BendingModulus<float>(m), m.poisson, m.yield_moment, m.bend_hardening,
            {m.bend_maxwell_modulus[0], m.bend_maxwell_modulus[1]},
            {m.bend_maxwell_time[0], m.bend_maxwell_time[1]}, 0.1f / m.thickness};
}

template <class Real> NUKA_MF_HD inline void TangentProduct(const Real (&c)[6], const Real (&x)[3], Real (&y)[3]) {
    y[0] = c[0] * x[0] + c[3] * x[1] + c[4] * x[2];
    y[1] = c[3] * x[0] + c[1] * x[1] + c[5] * x[2];
    y[2] = c[4] * x[0] + c[5] * x[1] + c[2] * x[2];
}

// 1/2 x^T C(nu) x of a strain triple at unit modulus.
template <class Real> NUKA_MF_HD inline Real UnitEnergy(Real nu, const Real (&x)[3]) {
    return Real(0.5) * (x[0] * x[0] + x[1] * x[1] + Real(2) * nu * x[0] * x[1] +
                        Real(0.5) * (Real(1) - nu) * x[2] * x[2]);
}

// Stored energy at a committed history (null: virgin): elastic and Maxwell strain energy plus hardening.
template <class Real>
NUKA_MF_HD inline Real StoredEnergy(const PlaneStressLaw& law, const float* committed, const Real (&E)[3]) {
    Real z[kMfHistory];
    for (uint32_t k = 0u; k < kMfHistory; ++k) z[k] = committed != nullptr ? Real(committed[k]) : Real(0);
    const Real nu = law.poisson;
    const Real e[3] = {E[0] - z[0], E[1] - z[1], E[2] - z[2]};
    Real energy = UnitEnergy(nu, e);
    for (uint32_t k = 0u; k < 2u; ++k) {
        const Real lag[3] = {e[0] - z[4u + 3u * k], e[1] - z[5u + 3u * k], e[2] - z[6u + 3u * k]};
        energy += Real(law.maxwell[k]) * UnitEnergy(nu, lag);
    }
    return Real(law.modulus) * energy + Real(0.5) * Real(law.hardening) * z[3] * z[3];
}

// The eigenbasis shared by C(nu) and the J2 metric P is (11 + 22)/sqrt2, (11 - 22)/sqrt2, 12, where C has
// eigenvalues (1 + nu, 1 - nu, (1 - nu)/2) and P (1/3, 1, 2); the rotation is its own inverse.
inline constexpr double kHalfSqrt2 = 0.70710678118654752440;

// Return-mapping result in the eigenbasis: stress, plastic multiplier and the hardened yield.
struct J2Return {
    double stress[3];
    double gamma;
    double yield;
    bool converged;
};

// Simo-Taylor plane-stress return: Newton on 1/|S(g)|_P - (1 - 2/3 H g)/(sqrt(2/3) k_n), nearly linear in g,
// bracketed between zero and the step that would reach the yield at constant hardening.
NUKA_MF_HD inline J2Return ReturnMap(const double (&trial)[3], const double (&c)[3], double yield, double hardening) {
    const double p[3] = {1.0 / 3.0, 1.0, 2.0}, root = 0.81649658092772603273;
    const double slope = 2.0 / 3.0 * hardening;
    double norm = 0.0, rate = c[0] * p[0];
    for (uint32_t i = 0u; i < 3u; ++i) {
        norm += p[i] * trial[i] * trial[i];
        rate = fmin(rate, c[i] * p[i]);
    }
    double lo = 0.0, hi = (sqrt(norm) / (root * yield) - 1.0) / rate, gamma = 0.0;
    if (slope > 0.0) hi = fmin(hi, (1.0 - 1e-9) / slope);
    J2Return r{};
    for (uint32_t it = 0u; it < 8u; ++it) {
        double size = 0.0, change = 0.0;
        for (uint32_t i = 0u; i < 3u; ++i) {
            const double scale = 1.0 + gamma * c[i] * p[i], s = trial[i] / scale;
            r.stress[i] = s;
            size += p[i] * s * s;
            change -= p[i] * p[i] * c[i] * s * s / scale;
        }
        size = sqrt(size);
        const double residual = 1.0 / size - (1.0 - slope * gamma) / (root * yield);
        if (fabs(residual) * size <= 1e-12) {
            r.converged = true;
            break;
        }
        if (residual < 0.0) lo = gamma;
        else hi = gamma;
        const double next = gamma - residual / (-change / (size * size * size) + slope / (root * yield));
        gamma = next > lo && next < hi ? next : 0.5 * (lo + hi);
    }
    r.gamma = gamma;
    r.yield = yield / (1.0 - slope * gamma);
    return r;
}

// Response at strain E from the committed history (null: virgin). The Maxwell branches enter the step as
// elastic stiffness about their committed strains, so plasticity sees one shifted elastic law. A failed
// evaluation leaves `out` unchanged.
template <class Real>
NUKA_MF_HD inline MfStatus EvaluatePlaneStress(const PlaneStressLaw& law, const float* committed, const Real (&E)[3],
                                               Real h, FieldResponse<Real>& out) {
    if (!(h > Real(0))) return MfStatus::InvalidParameters;
    FieldResponse<Real> r{};
    Real z[kMfHistory];
    for (uint32_t k = 0u; k < kMfHistory; ++k) z[k] = committed != nullptr ? Real(committed[k]) : Real(0);
    const Real nu = law.poisson, modulus = law.modulus;
    Real branch[2], total = Real(1), shift[3] = {Real(0), Real(0), Real(0)};
    for (uint32_t k = 0u; k < 2u; ++k) {
        branch[k] = law.maxwell[k] > 0.0f ? Real(law.maxwell[k]) / (Real(1) + h / Real(law.time[k])) : Real(0);
        total += branch[k];
        for (uint32_t i = 0u; i < 3u; ++i) shift[i] += branch[k] * z[4u + 3u * k + i];
    }
    Real e[3] = {E[0] - z[0], E[1] - z[1], E[2] - z[2]};
    const Real d[3] = {e[0] - shift[0] / total, e[1] - shift[1] / total, e[2] - shift[2] / total};
    const Real c[3] = {modulus * total * (Real(1) + nu), modulus * total * (Real(1) - nu),
                       Real(0.5) * modulus * total * (Real(1) - nu)};
    Real s[3] = {c[0] * Real(kHalfSqrt2) * (d[0] + d[1]), c[1] * Real(kHalfSqrt2) * (d[0] - d[1]), c[2] * d[2]};
    Real t[6] = {c[0], c[1], c[2], Real(0), Real(0), Real(0)};
    r.history[3] = z[3];
    const Real kappa = Real(law.yield) + Real(law.hardening) * z[3];
    const Real size = s[0] * s[0] / Real(3) + s[1] * s[1] + Real(2) * s[2] * s[2];
    r.yielded = law.yield > 0.0f && size > Real(2.0 / 3.0) * kappa * kappa;
    if (r.yielded) {
        const double trial[3] = {double(s[0]), double(s[1]), double(s[2])};
        const double cd[3] = {double(c[0]), double(c[1]), double(c[2])}, p[3] = {1.0 / 3.0, 1.0, 2.0};
        const J2Return j = ReturnMap(trial, cd, double(kappa), double(law.hardening));
        if (!j.converged) return MfStatus::LocalNonConvergence;
        // Consistent tangent Xi - n n^T / (n^T P S + beta), Xi = (C^-1 + g P)^-1, n = Xi P S.
        double xi[3], n[3], den = 4.0 / 9.0 * double(law.hardening) * j.yield * j.yield /
                                       (1.0 - 2.0 / 3.0 * double(law.hardening) * j.gamma);
        for (uint32_t i = 0u; i < 3u; ++i) {
            xi[i] = cd[i] / (1.0 + j.gamma * cd[i] * p[i]);
            n[i] = xi[i] * p[i] * j.stress[i];
            den += p[i] * j.stress[i] * n[i];
            s[i] = Real(j.stress[i]);
        }
        t[0] = Real(xi[0] - n[0] * n[0] / den);
        t[1] = Real(xi[1] - n[1] * n[1] / den);
        t[2] = Real(xi[2] - n[2] * n[2] / den);
        t[3] = Real(-n[0] * n[1] / den);
        t[4] = Real(-n[0] * n[2] / den);
        t[5] = Real(-n[1] * n[2] / den);
        const Real flow[3] = {Real(j.gamma * p[0] * j.stress[0]), Real(j.gamma * p[1] * j.stress[1]),
                              Real(j.gamma * p[2] * j.stress[2])};
        e[0] -= Real(kHalfSqrt2) * (flow[0] + flow[1]);
        e[1] -= Real(kHalfSqrt2) * (flow[0] - flow[1]);
        e[2] -= flow[2];
        r.history[3] = z[3] + Real(2.0 / 3.0 * j.gamma * j.yield);
    }
    for (uint32_t i = 0u; i < 3u; ++i) r.history[i] = E[i] - e[i];
    r.stress[0] = Real(kHalfSqrt2) * (s[0] + s[1]);
    r.stress[1] = Real(kHalfSqrt2) * (s[0] - s[1]);
    r.stress[2] = s[2];
    r.tangent[0] = Real(0.5) * (t[0] + Real(2) * t[3] + t[1]);
    r.tangent[1] = Real(0.5) * (t[0] - Real(2) * t[3] + t[1]);
    r.tangent[2] = t[2];
    r.tangent[3] = Real(0.5) * (t[0] - t[1]);
    r.tangent[4] = Real(kHalfSqrt2) * (t[4] + t[5]);
    r.tangent[5] = Real(kHalfSqrt2) * (t[4] - t[5]);
    Real psi = UnitEnergy(nu, e), stored = psi, viscous = Real(0);
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
        psi += branch[k] * UnitEnergy(nu, from);
        stored += Real(law.maxwell[k]) * UnitEnergy(nu, lag);
        viscous += Real(2) * Real(law.maxwell[k]) / relax * UnitEnergy(nu, moved);
    }
    const Real hardening = law.hardening, grown = r.history[3] - z[3];
    r.psi = modulus * psi + Real(0.5) * hardening * (r.history[3] * r.history[3] - z[3] * z[3]) +
            Real(law.yield) * grown;
    r.stored = modulus * stored + Real(0.5) * hardening * r.history[3] * r.history[3];
    r.plastic = Real(law.yield) * grown;
    r.viscous = modulus * viscous;
    r.beyond_range = fmax(fmax(fabs(double(E[0])), fabs(double(E[1]))), 0.5 * fabs(double(E[2]))) > law.range;
    if (!math::IsFinite(r.psi)) return MfStatus::NonFinite;
    out = r;
    return MfStatus::Ok;
}

}  // namespace mf

}  // namespace nuka::nk

#undef NUKA_MF_HD
