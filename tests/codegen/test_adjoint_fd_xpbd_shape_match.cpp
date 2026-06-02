// ---------------------------------------------------------------------------
// V3 finite-difference validation of the generated XPBDShapeMatchRow reverse-mode
// adjoint (v0.7 p09-C, dense_adjoint -- the XPBD soft-body shape-match goal
// projection).
//
// Proves the analytical adjoint of one local per-particle goal-projection
// application matches a central-difference numerical Jacobian of the SAME
// generated primal
//     position_new = position + stiffness*(goal - position)  (unbounded equality
//                                                              box -- clamp inactive)
// within 1e-3 relative on 100 random cases. This is the exit-crit-6 gate:
// XPBDShapeMatchRow (id 9) ships a GENUINE dispatchable per-row adjoint. As a
// bilateral equality row (like xpbd_distance / xpbd_volume / maximal_joint) there
// is no event / box-saturation case -- only the away-from-boundary + determinism
// checks.
//
// IMPORTANT (the adjoint-granularity decision): shape matching has NO scalar
// constraint C and NO Lagrange multiplier. The dispatchable per-row adjoint is
// the per-particle, per-axis goal projection (multilinear in position / goal /
// stiffness). The polar-decomposition rotation R that forms the goal
// (goal = c + R(x_i^0 - c0)) and its derivative dR/dA are DOWNSTREAM geometry,
// validated separately by direct host FD in the cluster sim test
// (test_xpbd_shape_match), the analog of the volume row's grad-C check.
// ---------------------------------------------------------------------------

#include "codegen/v3_validation/xpbd_shape_match_fd.hpp"

#include <gtest/gtest.h>

namespace {

using nuka::codegen::v3::ValidateXpbdShapeMatchFd;
using nuka::codegen::v3::XpbdShapeMatchFdConfig;
using nuka::codegen::v3::XpbdShapeMatchFdResult;

// The analytical adjoint must match the numerical Jacobian within 1e-3 relative
// on 100 random cases (row strictly unclamped -- the equality box is unbounded).
TEST(AdjointFdXpbdShapeMatch, MatchesNumericalJacobian) {
    XpbdShapeMatchFdConfig config;
    config.num_cases = 100u;

    const XpbdShapeMatchFdResult result = ValidateXpbdShapeMatchFd(config);

    ASSERT_EQ(result.case_count, 100u);
    EXPECT_LT(result.max_rel_err, 1.0e-3f)
        << "worst case index=" << result.worst_case_index
        << " param=" << result.worst_param_index
        << " max_rel_err=" << result.max_rel_err;
}

// Determinism (D1): identical seed -> bit-identical worst-case error.
TEST(AdjointFdXpbdShapeMatch, IsDeterministicAcrossRuns) {
    XpbdShapeMatchFdConfig config;
    config.num_cases = 100u;

    const XpbdShapeMatchFdResult a = ValidateXpbdShapeMatchFd(config);
    const XpbdShapeMatchFdResult b = ValidateXpbdShapeMatchFd(config);

    EXPECT_EQ(a.max_rel_err, b.max_rel_err);
    EXPECT_EQ(a.worst_case_index, b.worst_case_index);
    EXPECT_EQ(a.worst_param_index, b.worst_param_index);
}

} // namespace
