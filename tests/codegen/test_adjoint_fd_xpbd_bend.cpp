// ---------------------------------------------------------------------------
// V3 finite-difference validation of the generated XPBDBendRow reverse-mode
// adjoint (v0.7 p09-B, dense_adjoint -- the XPBD soft-body bend projection).
//
// Proves the analytical adjoint of one local bend-constraint application matches
// a central-difference numerical Jacobian of the SAME generated primal
// (lambda_new = lambda + effective_mass*(-C - alpha_tilde*lambda) for the
// unbounded equality box) within 1e-3 relative on 100 random cases. This is the
// exit-crit-6 gate: XPBDBendRow (id 7) ships a GENUINE dispatchable per-row
// adjoint. As a bilateral equality row (like xpbd_distance / maximal_joint) there
// is no event / box-saturation case -- only the away-from-boundary + determinism
// checks. The constraint-specific Bergou bending gradient is downstream of this
// local adjoint and is validated separately (direct host FD of grad C in the
// cloth sim test).
// ---------------------------------------------------------------------------

#include "codegen/v3_validation/xpbd_bend_fd.hpp"

#include <gtest/gtest.h>

namespace {

using nuka::codegen::v3::ValidateXpbdBendFd;
using nuka::codegen::v3::XpbdBendFdConfig;
using nuka::codegen::v3::XpbdBendFdResult;

// The analytical adjoint must match the numerical Jacobian within 1e-3 relative
// on 100 random cases (row strictly unclamped -- the equality box is unbounded).
TEST(AdjointFdXpbdBend, MatchesNumericalJacobian) {
    XpbdBendFdConfig config;
    config.num_cases = 100u;

    const XpbdBendFdResult result = ValidateXpbdBendFd(config);

    ASSERT_EQ(result.case_count, 100u);
    EXPECT_LT(result.max_rel_err, 1.0e-3f)
        << "worst case index=" << result.worst_case_index
        << " param=" << result.worst_param_index
        << " max_rel_err=" << result.max_rel_err;
}

// Determinism (D1): identical seed -> bit-identical worst-case error.
TEST(AdjointFdXpbdBend, IsDeterministicAcrossRuns) {
    XpbdBendFdConfig config;
    config.num_cases = 100u;

    const XpbdBendFdResult a = ValidateXpbdBendFd(config);
    const XpbdBendFdResult b = ValidateXpbdBendFd(config);

    EXPECT_EQ(a.max_rel_err, b.max_rel_err);
    EXPECT_EQ(a.worst_case_index, b.worst_case_index);
    EXPECT_EQ(a.worst_param_index, b.worst_param_index);
}

} // namespace
