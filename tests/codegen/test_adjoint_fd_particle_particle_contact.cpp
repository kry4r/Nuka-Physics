// ---------------------------------------------------------------------------
// V3 finite-difference validation of the generated ParticleParticleContactRow
// reverse-mode adjoint (v0.7 p11, K3 cross-system coupling -- dense_adjoint, the
// cross-system particle-particle CONTACT multiplier law).
//
// Proves the analytical adjoint of one local contact-projection multiplier update
// matches a central-difference numerical Jacobian of the SAME generated primal
//     lambda_new = clamp(u, lower, upper),
//     u = lambda + effective_mass*(-C - alpha_tilde*lambda)
// within 1e-3 relative on 100 random ACTIVE penetrating contacts (C < 0), each
// sampled strictly inside the UNILATERAL contact box [0, +huge) so the clamp
// sub-gradient indicator is 1. This is the exit-crit gate for every new row class:
// ParticleParticleContactRow (id 10) ships a GENUINE dispatchable per-row adjoint
// -- the SAME multilinear XPBD multiplier law as xpbd_distance / xpbd_volume.
//
// IMPORTANT (the adjoint-granularity decision, mirroring the XPBD rows): the
// dispatchable per-row adjoint is the scalar multiplier law; the geometric chain
// (the contact normal n as a function of the two positions, the inverse masses) is
// DOWNSTREAM and validated separately by the coupling sim oracles
// (tests/runtime/test_particle_particle_contact), the analog of the volume row's
// grad-C check. The separating / at-the-edge case is non-smooth and never emitted
// as a row, so it is deliberately not sampled here.
// ---------------------------------------------------------------------------

#include "codegen/v3_validation/particle_particle_contact_fd.hpp"

#include <gtest/gtest.h>

namespace {

using nuka::codegen::v3::ParticleParticleContactFdConfig;
using nuka::codegen::v3::ParticleParticleContactFdResult;
using nuka::codegen::v3::ValidateParticleParticleContactFd;

// The analytical adjoint must match the numerical Jacobian within 1e-3 relative
// on 100 random active penetrating contacts (strictly inside the unilateral box).
TEST(AdjointFdParticleParticleContact, MatchesNumericalJacobian) {
    ParticleParticleContactFdConfig config;
    config.num_cases = 100u;

    const ParticleParticleContactFdResult result =
        ValidateParticleParticleContactFd(config);

    ASSERT_EQ(result.case_count, 100u);
    EXPECT_LT(result.max_rel_err, 1.0e-3f)
        << "worst case index=" << result.worst_case_index
        << " param=" << result.worst_param_index
        << " max_rel_err=" << result.max_rel_err;
}

// Determinism (D1): identical seed -> bit-identical worst-case error.
TEST(AdjointFdParticleParticleContact, IsDeterministicAcrossRuns) {
    ParticleParticleContactFdConfig config;
    config.num_cases = 100u;

    const ParticleParticleContactFdResult a = ValidateParticleParticleContactFd(config);
    const ParticleParticleContactFdResult b = ValidateParticleParticleContactFd(config);

    EXPECT_EQ(a.max_rel_err, b.max_rel_err);
    EXPECT_EQ(a.worst_case_index, b.worst_case_index);
    EXPECT_EQ(a.worst_param_index, b.worst_param_index);
}

} // namespace
