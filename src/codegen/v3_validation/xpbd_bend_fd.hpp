// V3 finite-difference validation for the XPBDBendRow reverse-mode adjoint
// (v0.7 p09-B). XPBDBendRow is a dense_adjoint bilateral EQUALITY row (the box is
// unbounded -- clamp never active), so this harness mirrors the xpbd_distance
// path: it central-differences the GENERATED primal
//     lambda_new = clamp(u, lower, upper),
//     u = lambda + effective_mass*(-C - alpha_tilde*lambda)
// and compares to the GENERATED analytical adjoint at seed v=1.
//
// The bend row's LOCAL projection law is IDENTICAL in form to the distance row's
// (the XPBD multiplier update is constraint-type-agnostic once the reduced
// scalars C/lambda/effective_mass/alpha_tilde are treated as independent inputs).
// The constraint-specific Bergou bending GRADIENT is DOWNSTREAM of this local
// adjoint (it rides the position chain) and is validated separately, by a direct
// host FD check of grad C in the cloth/tet sim tests -- NOT here.
//
// The law is MULTILINEAR (degree <= 1 in each differentiable field), so the
// central-difference truncation error is exactly zero in real arithmetic; the
// only residual is float32 cancellation -- the property that lets the maximal /
// distance rows clear the 1e-3 gate.

#pragma once

#include <cstdint>

namespace nuka::codegen::v3 {

struct XpbdBendFdResult {
    float max_rel_err = 0.0f;        // worst element-wise relative error
    uint32_t worst_case_index = 0u;  // batch index achieving max_rel_err
    uint32_t worst_param_index = 0u; // which grad component (0..3) was worst
    uint32_t case_count = 0u;        // number of cases evaluated
};

struct XpbdBendFdConfig {
    uint32_t num_cases = 100u;
    uint64_t seed = 0xC0FFEEu;
    float fd_delta = 1.0e-2f;

    // Sample ranges. effective_mass = 1/(sum_i w_i|k_i|^2 + alpha_tilde) is
    // positive and bounded; alpha_tilde = compliance_alpha/dt^2 spans rigid (~0)
    // to compliant. lambda is sampled away from zero (magnitude in
    // [lambda_min, lambda_max], random sign): the partial
    // grad_alpha_tilde = -effective_mass*lambda vanishes as lambda -> 0, and a
    // near-zero partial differenced from the float32 primal is dominated by
    // float32 cancellation (the law is multilinear, so CD has zero truncation
    // error -- only the f32 roundoff floor remains). Keeping lambda away from
    // zero keeps every partial well-conditioned, exactly as the distance row does.
    float C_abs_max = 0.5f;            // bend constraint value (Bergou stencil)
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

// Away-from-boundary FD vs analytical-adjoint batch (XPBDBendRow).
XpbdBendFdResult ValidateXpbdBendFd(const XpbdBendFdConfig& config);

} // namespace nuka::codegen::v3
