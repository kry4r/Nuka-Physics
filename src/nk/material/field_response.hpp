#pragma once
// Material-field shell response: plane-stress membranes and triangle bending evaluated at fixed committed
// history, free of side effects. The device evaluates Real = float, the host oracles Real = double.

#include <cstdint>

#include "math/finite.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NUKA_MF_HD __host__ __device__
#else
#define NUKA_MF_HD
#endif

namespace nuka::nk {

// Density kg/m^3, thickness m, moduli and yield stress Pa, yield moment N m per m,
// Maxwell moduli as fractions of the instantaneous stiffness and their relaxation times in s.
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

// Committed history of a triangle: plastic strain (Voigt), hardening and two Maxwell strains;
// of a bend: plastic turns of its three hinges, hardening and two Maxwell turn triples.
inline constexpr uint32_t kMfTriangleHistory = 12u;
inline constexpr uint32_t kMfBendHistory = 10u;

enum class MfStatus : uint32_t { Ok, InvalidParameters, LocalNonConvergence, NonFinite, OutOfRange };

namespace mf {

// Strain and stress use E = (E11, E22, 2 E12); the tangent stores (11, 22, 33, 12, 13, 23).
template <class Real> struct MembraneResult {
    Real psi;
    Real stress[3];
    Real tangent[6];
    Real history[kMfTriangleHistory];
    Real dissipation;
};

// Turns and moments follow a triangle's three hinges; the tangent stores (00, 11, 22, 01, 02, 12).
template <class Real> struct BendResult {
    Real psi;
    Real moment[3];
    Real tangent[6];
    Real history[kMfBendHistory];
    Real dissipation;
};

NUKA_MF_HD inline bool ValidMaterial(const MfMaterial& m) {
    return math::IsFinite(m.density) && m.density > 0.0f && math::IsFinite(m.thickness) && m.thickness > 0.0f &&
           math::IsFinite(m.youngs) && m.youngs > 0.0f && m.poisson >= 0.0f && m.poisson < 0.5f;
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

template <class Real> NUKA_MF_HD inline void ElasticTangent(const MfMaterial& m, Real (&c)[6]) {
    const Real c11 = MembraneModulus<Real>(m), nu = m.poisson;
    c[0] = c11;
    c[1] = c11;
    c[2] = Real(0.5) * (Real(1) - nu) * c11;
    c[3] = nu * c11;
    c[4] = Real(0);
    c[5] = Real(0);
}

template <class Real> NUKA_MF_HD inline void TangentProduct(const Real (&c)[6], const Real (&x)[3], Real (&y)[3]) {
    y[0] = c[0] * x[0] + c[3] * x[1] + c[4] * x[2];
    y[1] = c[3] * x[0] + c[1] * x[1] + c[5] * x[2];
    y[2] = c[4] * x[0] + c[5] * x[1] + c[2] * x[2];
}

// Energy per rest area 1/2 (E - Ep)^T C (E - Ep) at the committed plastic strain Ep; a null history
// is the virgin state. A failed evaluation leaves `out` unchanged.
template <class Real>
NUKA_MF_HD inline MfStatus EvaluateMembrane(const MfMaterial& m, const float* committed, const Real (&E)[3],
                                            Real h, MembraneResult<Real>& out) {
    if (!ValidMaterial(m) || !(h > Real(0))) return MfStatus::InvalidParameters;
    MembraneResult<Real> r{};
    for (uint32_t k = 0u; k < kMfTriangleHistory; ++k) r.history[k] = committed != nullptr ? Real(committed[k]) : Real(0);
    const Real elastic[3] = {E[0] - r.history[0], E[1] - r.history[1], E[2] - r.history[2]};
    ElasticTangent(m, r.tangent);
    TangentProduct(r.tangent, elastic, r.stress);
    r.psi = Real(0.5) * (elastic[0] * r.stress[0] + elastic[1] * r.stress[1] + elastic[2] * r.stress[2]);
    if (!math::IsFinite(r.psi)) return MfStatus::NonFinite;
    out = r;
    return MfStatus::Ok;
}

// Bending energy 1/2 (turn - turn_p)^T K (turn - turn_p) over the hinge turns from their rest angles, with the
// cooked plate stiffness K of the triangle in hinge space, stored like the tangent.
template <class Real>
NUKA_MF_HD inline MfStatus EvaluateBend(const MfMaterial& m, const float* committed, const Real (&turn)[3],
                                        const float* stiffness, Real h, BendResult<Real>& out) {
    if (!ValidMaterial(m) || !(h > Real(0))) return MfStatus::InvalidParameters;
    BendResult<Real> r{};
    for (uint32_t k = 0u; k < kMfBendHistory; ++k) r.history[k] = committed != nullptr ? Real(committed[k]) : Real(0);
    const Real elastic[3] = {turn[0] - r.history[0], turn[1] - r.history[1], turn[2] - r.history[2]};
    for (uint32_t k = 0u; k < 6u; ++k) r.tangent[k] = Real(stiffness[k]);
    TangentProduct(r.tangent, elastic, r.moment);
    r.psi = Real(0.5) * (elastic[0] * r.moment[0] + elastic[1] * r.moment[1] + elastic[2] * r.moment[2]);
    if (!math::IsFinite(r.psi)) return MfStatus::NonFinite;
    out = r;
    return MfStatus::Ok;
}

}  // namespace mf

}  // namespace nuka::nk

#undef NUKA_MF_HD
