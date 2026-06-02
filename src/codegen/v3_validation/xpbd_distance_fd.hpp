// V3 finite-difference validation for the XPBDDistanceRow reverse-mode adjoint
// (v0.7 p09-A). XPBDDistanceRow is a dense_adjoint bilateral EQUALITY row (the
// box is unbounded -- clamp never active), so this harness mirrors the
// maximal_joint path in box_pgs_fd: it central-differences the GENERATED primal
//     lambda_new = clamp(u, lower, upper),
//     u = lambda + effective_mass*(-C - alpha_tilde*lambda)
// and compares to the GENERATED analytical adjoint at seed v=1.
//
// A SEPARATE harness (rather than reusing box_pgs_fd) is required because the
// XPBD law's four differentiable fields are {C, lambda, effective_mass,
// alpha_tilde} -- a different field set from the box-PGS rows' {jv, rhs, lambda,
// effective_mass}, so box_pgs_fd's binding (which hardcodes the latter names)
// does not compile against XPBDDistanceAdjInputs.
//
// The law is MULTILINEAR (degree <= 1 in each differentiable field), so the
// central-difference truncation error is exactly zero in real arithmetic; the
// only residual is float32 cancellation -- the same property that lets the
// maximal rows clear the 1e-3 gate.

#pragma once

#include <cstdint>

namespace nuka::codegen::v3 {

struct XpbdDistanceFdResult {
    float max_rel_err = 0.0f;        // worst element-wise relative error
    uint32_t worst_case_index = 0u;  // batch index achieving max_rel_err
    uint32_t worst_param_index = 0u; // which grad component (0..3) was worst
    uint32_t case_count = 0u;        // number of cases evaluated
};

struct XpbdDistanceFdConfig {
    uint32_t num_cases = 100u;
    uint64_t seed = 0xC0FFEEu;
    float fd_delta = 1.0e-2f;

    // Sample ranges. effective_mass = 1/(w_a+w_b+alpha_tilde) is positive and
    // bounded; alpha_tilde = compliance_alpha/dt^2 spans rigid (~0) to compliant.
    // lambda is sampled away from zero (magnitude in [lambda_min, lambda_max],
    // random sign), mirroring box_pgs_fd's lambda sampling: the partial
    // grad_alpha_tilde = -effective_mass*lambda vanishes as lambda -> 0, and a
    // near-zero partial differenced from the float32 primal is dominated by
    // float32 cancellation (the law is multilinear, so CD has zero truncation
    // error -- only the f32 roundoff floor remains). Keeping lambda away from
    // zero keeps every partial well-conditioned, exactly as the maximal rows do.
    float C_abs_max = 0.5f;            // signed distance error
    float lambda_min = 0.5f;          // |lambda| floor (well-conditioned grads)
    float lambda_max = 2.0f;          // |lambda| ceiling
    float meff_min = 0.25f;            // effective inverse mass (well-conditioned)
    float meff_max = 2.0f;
    float alpha_tilde_min = 0.0f;
    float alpha_tilde_max = 1.0f;

    // Unbounded equality box -- wide enough that clamp_active is always 1.
    float lower = -1.0e6f;
    float upper = 1.0e6f;
};

// Away-from-boundary FD vs analytical-adjoint batch (XPBDDistanceRow).
XpbdDistanceFdResult ValidateXpbdDistanceFd(const XpbdDistanceFdConfig& config);

} // namespace nuka::codegen::v3
