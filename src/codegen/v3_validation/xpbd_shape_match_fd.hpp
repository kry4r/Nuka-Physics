// V3 finite-difference validation for the XPBDShapeMatchRow reverse-mode adjoint
// (v0.7 p09-C). XPBDShapeMatchRow is a dense_adjoint bilateral EQUALITY row (the
// box is unbounded -- clamp never active), so this harness mirrors the
// xpbd_distance / xpbd_volume paths: it central-differences the GENERATED primal
//     position_new = clamp(u, lower, upper),
//     u = position + stiffness*(goal - position)
// and compares to the GENERATED analytical adjoint at seed v=1.
//
// Unlike the distance/bend/volume rows, shape matching has NO scalar Lagrange
// multiplier: this is the PER-PARTICLE, PER-AXIS GOAL PROJECTION (one
// representative scalar channel; the 3 axes of g_i, p_i are independent
// applications of the SAME law). The cluster size n NEVER enters this local
// adjoint -- it lives entirely in computing the goal (centroid/covariance/polar).
// The constraint-specific GEOMETRY (the polar-decomposition rotation R that forms
// goal = c + R(x_i^0 - c0), and its derivative dR/dA) is DOWNSTREAM of this local
// adjoint and is validated separately, by a direct host FD check in the cluster
// sim test (tests/runtime/test_xpbd_shape_match) -- NOT here.
//
// The law is MULTILINEAR (degree <= 1 in each differentiable field), so the
// central-difference truncation error is exactly zero in real arithmetic; the
// only residual is float32 cancellation -- the property that lets the maximal /
// distance / bend / volume rows clear the 1e-3 gate.

#pragma once

#include <cstdint>

namespace nuka::codegen::v3 {

struct XpbdShapeMatchFdResult {
    float max_rel_err = 0.0f;        // worst element-wise relative error
    uint32_t worst_case_index = 0u;  // batch index achieving max_rel_err
    uint32_t worst_param_index = 0u; // which grad component (0..2) was worst
    uint32_t case_count = 0u;        // number of cases evaluated
};

struct XpbdShapeMatchFdConfig {
    uint32_t num_cases = 100u;
    uint64_t seed = 0x5A1ADu;
    float fd_delta = 1.0e-2f;

    // position is the current particle coordinate (one axis). goal is the
    // shape-matched goal coordinate. stiffness in (0,1) is the goal-pull fraction.
    // The (goal - position) gap is kept away from zero (random sign, magnitude in
    // [gap_min, gap_max]) so grad_stiffness = (goal - position) stays
    // well-conditioned against the float32 FD cancellation floor (the law is
    // multilinear, so CD has zero truncation error -- only the f32 roundoff floor
    // remains). Same discipline the volume row uses for lambda.
    float position_abs_max = 2.0f;     // current coordinate magnitude
    float gap_min = 0.5f;              // |goal - position| floor (well-conditioned)
    float gap_max = 2.0f;              // |goal - position| ceiling
    float stiffness_min = 0.05f;       // goal-pull fraction (away from 0 and 1)
    float stiffness_max = 0.95f;

    // Unbounded equality box -- wide enough that clamp_active is always 1.
    float lower = -1.0e6f;
    float upper = 1.0e6f;
};

// Away-from-boundary FD vs analytical-adjoint batch (XPBDShapeMatchRow).
XpbdShapeMatchFdResult ValidateXpbdShapeMatchFd(const XpbdShapeMatchFdConfig& config);

} // namespace nuka::codegen::v3
