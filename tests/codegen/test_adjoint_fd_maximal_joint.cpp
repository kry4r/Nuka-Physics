// ---------------------------------------------------------------------------
// V3 finite-difference validation of the generated maximal_joint reverse-mode
// adjoint (dense_adjoint -- the bilateral equality PGS row).
//
// Proves the analytical adjoint emitted by the codegen pipeline matches a
// central-difference numerical Jacobian of the SAME generated primal. A joint is
// a bilateral equality row (unbounded box), so the clamp is never active and the
// adjoint is the pure smooth Jacobian of the linear update -- away-from-clamp +
// determinism only, exactly like the maximal_drive dense slice (no event case).
// ---------------------------------------------------------------------------

#include "codegen/v3_validation/box_pgs_fd.hpp"

#include <gtest/gtest.h>

namespace {

using nuka::codegen::v3::BoxPgsFdConfig;
using nuka::codegen::v3::BoxPgsFdResult;
using nuka::codegen::v3::ValidateMaximalJointFd;

// The analytical adjoint must match the numerical Jacobian within 1e-3 relative
// on 100 random cases sampled strictly inside the (unbounded) box.
TEST(AdjointFdMaximalJoint, MatchesNumericalJacobianAwayFromClamp) {
    BoxPgsFdConfig config;
    config.num_cases = 100u;

    const BoxPgsFdResult result = ValidateMaximalJointFd(config);

    ASSERT_EQ(result.case_count, 100u);
    EXPECT_LT(result.max_rel_err, 1.0e-3f)
        << "worst case index=" << result.worst_case_index
        << " param=" << result.worst_param_index
        << " max_rel_err=" << result.max_rel_err;
}

// Determinism (D1): identical seed -> bit-identical worst-case error.
TEST(AdjointFdMaximalJoint, IsDeterministicAcrossRuns) {
    BoxPgsFdConfig config;
    config.num_cases = 100u;

    const BoxPgsFdResult a = ValidateMaximalJointFd(config);
    const BoxPgsFdResult b = ValidateMaximalJointFd(config);

    EXPECT_EQ(a.max_rel_err, b.max_rel_err);
    EXPECT_EQ(a.worst_case_index, b.worst_case_index);
    EXPECT_EQ(a.worst_param_index, b.worst_param_index);
}

} // namespace
