#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::fluid -- SPH smoothing kernels for Position-Based Fluids
// (Macklin & Mueller 2013; kernel forms from Mueller et al. 2003 "Particle-Based
// Fluid Simulation for Interactive Applications").
//
// Two kernels are used by PBF:
//   * Poly6   W(r,h) -- density estimation (the SCALAR kernel value).
//   * Spiky   gradient of the spiky kernel -- the pressure / position-correction
//             gradient (Poly6's gradient vanishes at r->0, which gives clumping;
//             the spiky kernel has a non-zero gradient near the origin, so M&M use
//             it for the gradient terms in lambda and the position correction).
//
// Normalization constants (3D, both kernels supported on [0, h]):
//   Poly6:        W(r,h)        = 315/(64*pi*h^9) * (h^2 - r^2)^3,  0 <= r <= h
//   Spiky grad:   grad W(r,h)   = -45/(pi*h^6)    * (h - r)^2 * (r_vec / r)
// Both are zero for r > h and (for Poly6) clamped to >= 0. The h-dependent scalar
// coefficients are PRECOMPUTED on the host (PbfKernelCoeffs) so the device kernels
// only multiply -- this keeps the per-particle hot path free of pow()/division by
// h and makes the float recipe identical for every particle (D1).
//
// DETERMINISM (D1): every function reproduces a FIXED float expression with no
// reassociation and no intrinsics beyond sqrtf (matching the rest of the tree's
// math posture, math/cuda_vec_ops.cuh). The coefficients are passed in as plain
// floats, so the device op order is literally the same instruction stream for all
// particles regardless of warp scheduling.
//
// rho0 NOTE: the ABSOLUTE Poly6 constant only sets physical units. The PBF
// equilibrium is C_i = rho_i/rho0 - 1 == 0, i.e. rho_i == rho0; a wrong constant
// is absorbed when rho0 is calibrated numerically from this SAME kernel over a
// rest lattice (see pbf_world.cu / the tests). The constants below are the
// textbook values so that a future real-units (rho0 = 1000 kg/m^3) Flex oracle
// (deferred to p15) lines up without re-derivation.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#if !defined(__CUDACC__)
#error "runtime/fluid/pbf_kernels.cuh is a CUDA device header; compile the including TU with nvcc."
#endif

namespace nuka::runtime::fluid {

// Host-computed, h-dependent kernel coefficients. Built once per world (h fixed)
// and passed by value into the device kernels. kPi is the literal used in the
// textbook normalizations.
struct PbfKernelCoeffs {
    float h = 1.0f;          // support radius.
    float h2 = 1.0f;         // h*h (precomputed; Poly6 uses (h^2 - r^2)).
    float poly6 = 0.0f;      // 315 / (64*pi*h^9)
    float spiky_grad = 0.0f; // -45 / (pi*h^6)  (sign folded in)
};

// Build the coefficients for support radius h (host-side; constexpr-friendly math
// kept as runtime floats so the same recipe runs on host and is *stored*, not
// recomputed, on the device).
inline PbfKernelCoeffs MakePbfKernelCoeffs(float h) {
    constexpr float kPi = 3.14159265358979323846f;
    PbfKernelCoeffs c;
    c.h = h;
    c.h2 = h * h;
    const float h3 = h * h * h;
    const float h6 = h3 * h3;
    const float h9 = h6 * h3;
    c.poly6 = 315.0f / (64.0f * kPi * h9);
    c.spiky_grad = -45.0f / (kPi * h6);
    return c;
}

// Poly6 SCALAR kernel value at squared distance r2 (r2 = |r_vec|^2). Zero outside
// the support; clamped non-negative inside. Taking r2 directly avoids a sqrt in
// the density sum (Poly6 depends only on r^2).
__device__ __forceinline__ float Poly6FromR2(float r2, const PbfKernelCoeffs& c) {
    if (r2 >= c.h2) {
        return 0.0f;
    }
    const float diff = c.h2 - r2;     // (h^2 - r^2) >= 0 here.
    return c.poly6 * diff * diff * diff;
}

// Spiky kernel GRADIENT wrt the first position, given the separation vector
// r_vec = p_i - p_j and its length r. Returns the vector
//   grad W = spiky_grad * (h - r)^2 * (r_vec / r)
// which points along +r_vec (away from p_j) with the negative coefficient folded
// in (spiky_grad < 0), matching the M&M sign convention. Returns zero for r >= h
// or r ~ 0 (gradient direction undefined at coincidence; r==0 is excluded because
// self is not in the neighbor list and distinct particles rarely coincide -- the
// guard keeps it finite if they do).
__device__ __forceinline__ math::Vec3 SpikyGradient(math::Vec3 r_vec,
                                                     float r,
                                                     const PbfKernelCoeffs& c) {
    if (r >= c.h || r <= 0.0f) {
        return math::Vec3{0.0f, 0.0f, 0.0f};
    }
    const float hr = c.h - r;             // (h - r) > 0 here.
    const float coeff = c.spiky_grad * hr * hr / r;  // includes 1/r for r_vec/r.
    return math::Vec3{r_vec.x * coeff, r_vec.y * coeff, r_vec.z * coeff};
}

} // namespace nuka::runtime::fluid
