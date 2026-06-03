// V3 finite-difference validation for the ParticleParticleContactRow reverse-mode
// adjoint (v0.7 p11, K3 cross-system coupling infrastructure). The row is a
// dense_adjoint UNILATERAL contact row whose dispatchable per-row adjoint is the
// IDENTICAL multilinear XPBD multiplier-update law as the XPBDDistanceRow (only the
// geometric grad C -- the contact normal, downstream of this local adjoint -- and
// the unilateral box differ). This harness therefore mirrors the xpbd_distance
// path: it central-differences the GENERATED primal
//     lambda_new = clamp(u, lower, upper),
//     u = lambda + effective_mass*(-C - alpha_tilde*lambda)
// and compares to the GENERATED analytical adjoint at seed v=1.
//
// UNILATERAL box (lower = 0, upper = +huge): the narrow phase only emits
// PENETRATING pairs (C < 0), for which u = effective_mass*(-C) (+ the small
// lambda/alpha_tilde terms) is strictly POSITIVE -- i.e. strictly inside the
// contact box, so the clamp sub-gradient indicator is 1 and the adjoint matches
// the central difference. This is the genuine active-contact regime; the harness
// samples C strongly negative (penetration depth in [pen_min, pen_max]) with a
// comfortable margin so the +/- fd_delta perturbations never cross the lower
// active-set edge (where the gradient would correctly stop). The separating /
// at-the-edge case is NOT a smooth point and is deliberately not sampled here (it
// is never emitted as a row).
//
// The law is MULTILINEAR (degree <= 1 in each differentiable field), so the
// central-difference truncation error is exactly zero in real arithmetic; the only
// residual is float32 cancellation -- the property that lets the maximal /
// distance / volume rows clear the 1e-3 gate. The geometric chain (the contact
// normal n as a function of the two positions, the inverse masses) rides DOWNSTREAM
// on the position chain, validated separately by the coupling sim oracles
// (tests/runtime/test_particle_particle_contact), NOT here.

#pragma once

#include <cstdint>

namespace nuka::codegen::v3 {

struct ParticleParticleContactFdResult {
    float max_rel_err = 0.0f;        // worst element-wise relative error
    uint32_t worst_case_index = 0u;  // batch index achieving max_rel_err
    uint32_t worst_param_index = 0u; // which grad component (0..3) was worst
    uint32_t case_count = 0u;        // number of cases evaluated
};

struct ParticleParticleContactFdConfig {
    uint32_t num_cases = 100u;
    uint64_t seed = 0xC0FFEEu;
    float fd_delta = 1.0e-2f;

    // C is the signed gap; an ACTIVE contact is penetrating (C < 0). Sample C
    // strongly negative (penetration magnitude in [pen_min, pen_max]) so
    // u = effective_mass*(-C) (+ small lambda/alpha_tilde terms) stays comfortably
    // > 0 -- strictly inside the unilateral box [0, +huge) -- even after the
    // +/- fd_delta perturbations. The (-C) gap is also what keeps grad_lambda /
    // grad_alpha_tilde well-conditioned against the f32 FD cancellation floor
    // (the law is multilinear, so CD has zero truncation error -- only roundoff).
    float pen_min = 0.5f;              // |C| floor (well-conditioned penetration)
    float pen_max = 2.0f;              // |C| ceiling
    // lambda small + away from zero (the position-based co-step warm-starts
    // lambda = 0; a small non-zero band keeps grad_alpha_tilde = -m_eff*lambda
    // observable while keeping u dominated by the (-C) term, so the row stays
    // in-band). Random sign, magnitude in [lambda_min, lambda_max].
    float lambda_min = 0.05f;
    float lambda_max = 0.5f;
    float meff_min = 0.2f;             // effective_mass = 1/(w_i+w_j+a~) in (0,1]
    float meff_max = 1.0f;
    float alpha_tilde_min = 0.0f;      // XPBD compliance a~ (>= 0; 0 == rigid)
    float alpha_tilde_max = 0.2f;

    // Unilateral contact box: lower = 0 (a contact pushes, never sticks), upper
    // = +huge. An active penetrating contact is strictly inside this band.
    float lower = 0.0f;
    float upper = 1.0e6f;
};

// Away-from-edge FD vs analytical-adjoint batch (ParticleParticleContactRow). All
// cases are active penetrating contacts strictly inside the unilateral box.
ParticleParticleContactFdResult
ValidateParticleParticleContactFd(const ParticleParticleContactFdConfig& config);

} // namespace nuka::codegen::v3
