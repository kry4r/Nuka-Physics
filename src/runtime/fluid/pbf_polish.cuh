#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::fluid -- p10-B PBF POLISH kernels: the Akinci et al. 2013
// surface-tension COHESION spline C(r). (XSPH viscosity reuses Poly6FromR2 from
// pbf_kernels.cuh, so it lives directly in the nk PBF op TU; only the cohesion
// spline is novel enough to warrant its own coefficient + device helper here.)
//
// Akinci, Akinci & Teschner 2013, "Versatile Surface Tension and Adhesion for
// SPH Fluids", eq.2: the cohesion kernel is a piecewise C^1 spline supported on
// [0, h], shaped so the cohesion FORCE is attractive at medium range and
// repulsive at VERY short range, which keeps the blob from collapsing onto itself
// while pulling the surface inward. NOTE r = h/2 is the spline PEAK / branch
// boundary (both branches meet there, where C is at its MAXIMUM, > 0 = attractive)
// -- NOT the sign change. C(r) actually turns negative (repulsive) only for
// r <~ 0.28 h (solve 2(h-r)^3 r^3 = h^6/64); so cohesion is attractive across
// almost the whole support and repulsive only in a thin short-range core:
//
//   C(r) = k * { (h - r)^3 * r^3                      , 2r >  h , r <= h
//             { 2 (h - r)^3 * r^3 - h^6 / 64          , r   >  0 , 2r <= h
//             { 0                                     , otherwise
//
// We pick k = 64 / h^6 (NOT the textbook physical normalization 32/(pi*h^9)).
// This NORMALIZES the spline so its MAXIMUM over [0, h] is exactly 1 (the max sits
// at r = h/2, where both branches meet at k * h^6/64 = 1). The textbook constant
// 32/(pi*h^9) is ~3.1e2 at h=0.08, which would make the cohesion force ~300x the
// intended scale -- gamma would then have to be ~1e-3 and the stable window
// razor-thin. With the max normalized to 1, GAMMA is an O(1) knob (the documented
// "gamma absorbs the constant"): it is the cohesion force scale directly. The
// SHAPE -- repulsive core inside h/2, attractive lobe beyond -- is unchanged
// (that is the Akinci physics; only the absolute scale is folded into gamma).
//
// The surface-tension cohesion acceleration on particle i (Akinci eq.1, the
// cohesion term only -- the curvature term is DEFERRED to p13) is
//   a_i^coh = -gamma * sum_j m_j * C(|r_ij|) * (r_ij / |r_ij|)
// with r_ij = p_i - p_j. C(r) > 0 over most of the support, so -C * r_hat points
// p_i toward its neighbors (attraction); the short-range sign change gives the
// repulsive core. (We do NOT divide by the symmetrized density 2*rho0/(rho_i+rho_j)
// here: that factor is a near-unity normalization in the bulk and folding it in
// would couple this pass to a density read; gamma absorbs the constant. The pass
// is a demonstrative cohesion model, gated inert when gamma == 0.)
//
// DETERMINISM (D1): fixed float expression, no reassociation, no intrinsics; the
// h-dependent constant k is precomputed on the host and passed by value, so the
// device op order is the identical instruction stream for every particle.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#if !defined(__CUDACC__)
#error "runtime/fluid/pbf_polish.cuh is a CUDA device header; compile the including TU with nvcc."
#endif

namespace nuka::runtime::fluid {

// Host-computed, h-dependent cohesion-spline coefficients. Built once per world
// (h fixed) and passed by value into the device kernel.
struct PbfCohesionCoeffs {
    float h = 1.0f;        // support radius.
    float half_h = 0.5f;   // h/2 (the spline's peak / sign-shape point; precomputed).
    float k = 0.0f;        // 64 / h^6  (max-normalized: max C(r) over [0,h] == 1)
    float h6_over_64 = 0.0f;  // h^6 / 64 (the short-range offset; precomputed).
};

inline PbfCohesionCoeffs MakePbfCohesionCoeffs(float h) {
    PbfCohesionCoeffs c;
    c.h = h;
    c.half_h = 0.5f * h;
    const float h3 = h * h * h;
    const float h6 = h3 * h3;
    // k = 64/h^6 normalizes the spline peak (at r=h/2) to 1, so gamma is O(1).
    c.k = 64.0f / h6;
    c.h6_over_64 = h6 / 64.0f;
    return c;
}

// Akinci cohesion spline C(r) at separation r (r = |r_vec| >= 0). Zero outside
// [0, h]; the two-branch interior gives the short-range repulsive / medium-range
// attractive shape. r == 0 returns 0 (coincident particles exert no cohesion
// direction; the caller also guards the 1/r normalization).
__device__ __forceinline__ float CohesionSpline(float r, const PbfCohesionCoeffs& c) {
    if (r >= c.h || r <= 0.0f) {
        return 0.0f;
    }
    const float hr = c.h - r;          // (h - r) > 0 here.
    const float hr3 = hr * hr * hr;    // (h - r)^3
    const float r3 = r * r * r;        // r^3
    if (2.0f * r > c.h) {
        // Far half (h/2, h]: single positive lobe (all attractive).
        return c.k * hr3 * r3;
    }
    // Near half (0, h/2]: doubled lobe minus the constant offset. This is still
    // POSITIVE (attractive) near r = h/2 and only goes negative (the repulsive
    // core) for r <~ 0.28 h, where 2*hr3*r3 < h6/64.
    return c.k * (2.0f * hr3 * r3 - c.h6_over_64);
}

} // namespace nuka::runtime::fluid
