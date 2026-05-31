// ---------------------------------------------------------------------------
// V3 finite-difference validation of the generated maximal_drive reverse-mode
// adjoint (dense_adjoint vertical slice).
//
// Proves the analytical adjoint emitted by the codegen pipeline matches a
// central-difference numerical Jacobian of the SAME generated primal, and that
// at/over the force-limit clamp the adjoint returns the correct sub-gradient
// (exactly zero) -- the R9 non-smoothness handling.
// ---------------------------------------------------------------------------

#include "codegen/v3_validation/fd_harness.hpp"

#include <gtest/gtest.h>

namespace {

using nuka::codegen::v3::FdValidationResult;
using nuka::codegen::v3::MaximalDriveFdConfig;
using nuka::codegen::v3::MaxSaturatedAdjointMagnitude;
using nuka::codegen::v3::ValidateMaximalDriveFd;

// The analytical adjoint must match the numerical Jacobian within 1e-3 relative
// on 100 random cases sampled strictly inside the force-limit band.
TEST(AdjointFdMaximalDrive, MatchesNumericalJacobianAwayFromClamp) {
    MaximalDriveFdConfig config;
    config.num_cases = 100u;

    const FdValidationResult result = ValidateMaximalDriveFd(config);

    ASSERT_EQ(result.case_count, 100u);
    EXPECT_LT(result.max_rel_err, 1.0e-3f)
        << "worst case index=" << result.worst_case_index
        << " param=" << result.worst_param_index
        << " max_rel_err=" << result.max_rel_err;
}

// Determinism (D1): identical seed -> bit-identical worst-case error.
TEST(AdjointFdMaximalDrive, IsDeterministicAcrossRuns) {
    MaximalDriveFdConfig config;
    config.num_cases = 100u;

    const FdValidationResult a = ValidateMaximalDriveFd(config);
    const FdValidationResult b = ValidateMaximalDriveFd(config);

    EXPECT_EQ(a.max_rel_err, b.max_rel_err);
    EXPECT_EQ(a.worst_case_index, b.worst_case_index);
    EXPECT_EQ(a.worst_param_index, b.worst_param_index);
}

// R9: across a SATURATED clamp the analytical adjoint is the correct
// sub-gradient -- exactly zero on every input and gain.
TEST(AdjointFdMaximalDrive, SaturatedClampSubgradientIsZero) {
    MaximalDriveFdConfig config;
    config.num_cases = 100u;

    const float worst = MaxSaturatedAdjointMagnitude(config);

    EXPECT_EQ(worst, 0.0f)
        << "saturated drive must yield zero gradient; got max |grad|=" << worst;
}

} // namespace
