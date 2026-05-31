// ---------------------------------------------------------------------------
// V3 finite-difference validation of the generated featherstone_contact
// reverse-mode adjoint (stop_grad_on_event -- the ARTICULATED PGS contact row).
//
// Same PGS contact-row law as maximal_contact, but the constraint Jacobian is the
// per-articulation chain Jacobian (contracted into the scalar jv). This test
// validates the CONTACT-ROW adjoint only: away-from-event 1e-3 FD match +
// determinism + zero-at-event stop-gradient. The ABA forward-dynamics 3-pass
// reverse (the qddot adjoint) is NOT in scope here -- it belongs to v0.5 Phase 2's
// backward_runner.
// ---------------------------------------------------------------------------

#include "codegen/v3_validation/box_pgs_fd.hpp"

#include <gtest/gtest.h>

namespace {

using nuka::codegen::v3::BoxPgsFdConfig;
using nuka::codegen::v3::BoxPgsFdResult;
using nuka::codegen::v3::MaxEventStoppedFeatherstoneContactMagnitude;
using nuka::codegen::v3::ValidateFeatherstoneContactFd;

// Away from the event boundary the analytical adjoint must match the numerical
// Jacobian within 1e-3 relative on 100 random cases (row strictly unclamped).
TEST(AdjointFdFeatherstoneContact, MatchesNumericalJacobianAwayFromEvent) {
    BoxPgsFdConfig config;
    config.num_cases = 100u;

    const BoxPgsFdResult result = ValidateFeatherstoneContactFd(config);

    ASSERT_EQ(result.case_count, 100u);
    EXPECT_LT(result.max_rel_err, 1.0e-3f)
        << "worst case index=" << result.worst_case_index
        << " param=" << result.worst_param_index
        << " max_rel_err=" << result.max_rel_err;
}

// Determinism (D1): identical seed -> bit-identical worst-case error.
TEST(AdjointFdFeatherstoneContact, IsDeterministicAcrossRuns) {
    BoxPgsFdConfig config;
    config.num_cases = 100u;

    const BoxPgsFdResult a = ValidateFeatherstoneContactFd(config);
    const BoxPgsFdResult b = ValidateFeatherstoneContactFd(config);

    EXPECT_EQ(a.max_rel_err, b.max_rel_err);
    EXPECT_EQ(a.worst_case_index, b.worst_case_index);
    EXPECT_EQ(a.worst_param_index, b.worst_param_index);
}

// R5: at/over the active-set / friction-cone EVENT boundary the analytical
// adjoint is the stop-gradient -- exactly zero on every input and param.
TEST(AdjointFdFeatherstoneContact, EventBoundaryStopsGradient) {
    BoxPgsFdConfig config;
    config.num_cases = 100u;

    const float worst = MaxEventStoppedFeatherstoneContactMagnitude(config);

    EXPECT_EQ(worst, 0.0f)
        << "an event-clamped articulated contact row must yield zero gradient; "
           "got max |grad|=" << worst;
}

} // namespace
