// ---------------------------------------------------------------------------
// V3 finite-difference validation of the generated maximal_contact reverse-mode
// adjoint (stop_grad_on_event -- the PGS contact row).
//
// Proves the analytical adjoint matches a central-difference numerical Jacobian
// of the SAME generated primal AWAY from the box boundary (where the active-set /
// friction-cone gradient genuinely flows), and that AT/over the box boundary --
// the active-set / friction-cone EVENT -- the adjoint returns exactly zero (the
// stop-gradient). This is the R5 event analog of the drive slice's R9 clamp.
// ---------------------------------------------------------------------------

#include "codegen/v3_validation/box_pgs_fd.hpp"

#include <gtest/gtest.h>

namespace {

using nuka::codegen::v3::BoxPgsFdConfig;
using nuka::codegen::v3::BoxPgsFdResult;
using nuka::codegen::v3::MaxEventStoppedMaximalContactMagnitude;
using nuka::codegen::v3::ValidateMaximalContactFd;

// Away from the event boundary the analytical adjoint must match the numerical
// Jacobian within 1e-3 relative on 100 random cases (row strictly unclamped).
TEST(AdjointFdMaximalContact, MatchesNumericalJacobianAwayFromEvent) {
    BoxPgsFdConfig config;
    config.num_cases = 100u;

    const BoxPgsFdResult result = ValidateMaximalContactFd(config);

    ASSERT_EQ(result.case_count, 100u);
    EXPECT_LT(result.max_rel_err, 1.0e-3f)
        << "worst case index=" << result.worst_case_index
        << " param=" << result.worst_param_index
        << " max_rel_err=" << result.max_rel_err;
}

// Determinism (D1): identical seed -> bit-identical worst-case error.
TEST(AdjointFdMaximalContact, IsDeterministicAcrossRuns) {
    BoxPgsFdConfig config;
    config.num_cases = 100u;

    const BoxPgsFdResult a = ValidateMaximalContactFd(config);
    const BoxPgsFdResult b = ValidateMaximalContactFd(config);

    EXPECT_EQ(a.max_rel_err, b.max_rel_err);
    EXPECT_EQ(a.worst_case_index, b.worst_case_index);
    EXPECT_EQ(a.worst_param_index, b.worst_param_index);
}

// R5: at/over the active-set / friction-cone EVENT boundary the analytical
// adjoint is the stop-gradient -- exactly zero on every input and param.
TEST(AdjointFdMaximalContact, EventBoundaryStopsGradient) {
    BoxPgsFdConfig config;
    config.num_cases = 100u;

    const float worst = MaxEventStoppedMaximalContactMagnitude(config);

    EXPECT_EQ(worst, 0.0f)
        << "an event-clamped contact row must yield zero gradient; got max |grad|="
        << worst;
}

} // namespace
