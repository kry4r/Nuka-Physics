// ---------------------------------------------------------------------------
// V3 finite-difference validation of the generated XPBDDistanceRow reverse-mode
// adjoint (v0.7 p09-A, dense_adjoint -- the XPBD soft-body distance projection).
//
// Proves the analytical adjoint of one local distance-constraint application
// matches a central-difference numerical Jacobian of the SAME generated primal
// (lambda_new = lambda + effective_mass*(-C - alpha_tilde*lambda) for the
// unbounded equality box) within 1e-3 relative on 100 random cases. This is the
// exit-crit-6 gate: XPBDDistanceRow (id 6) ships a GENUINE dispatchable per-row
// adjoint. As a bilateral equality row (like maximal_joint) there is no event /
// box-saturation case -- only the away-from-boundary + determinism checks.
// ---------------------------------------------------------------------------

#include "codegen/v3_validation/xpbd_distance_fd.hpp"

#include <gtest/gtest.h>

namespace {

using nuka::codegen::v3::ValidateXpbdDistanceFd;
using nuka::codegen::v3::XpbdDistanceFdConfig;
using nuka::codegen::v3::XpbdDistanceFdResult;

// The analytical adjoint must match the numerical Jacobian within 1e-3 relative
// on 100 random cases (row strictly unclamped -- the equality box is unbounded).
TEST(AdjointFdXpbdDistance, MatchesNumericalJacobian) {
    XpbdDistanceFdConfig config;
    config.num_cases = 100u;

    const XpbdDistanceFdResult result = ValidateXpbdDistanceFd(config);

    ASSERT_EQ(result.case_count, 100u);
    EXPECT_LT(result.max_rel_err, 1.0e-3f)
        << "worst case index=" << result.worst_case_index
        << " param=" << result.worst_param_index
        << " max_rel_err=" << result.max_rel_err;
}

// Determinism (D1): identical seed -> bit-identical worst-case error.
TEST(AdjointFdXpbdDistance, IsDeterministicAcrossRuns) {
    XpbdDistanceFdConfig config;
    config.num_cases = 100u;

    const XpbdDistanceFdResult a = ValidateXpbdDistanceFd(config);
    const XpbdDistanceFdResult b = ValidateXpbdDistanceFd(config);

    EXPECT_EQ(a.max_rel_err, b.max_rel_err);
    EXPECT_EQ(a.worst_case_index, b.worst_case_index);
    EXPECT_EQ(a.worst_param_index, b.worst_param_index);
}

} // namespace
