// ---------------------------------------------------------------------------
// v0.7 p09-A: XPBD distance-constraint ANALYTIC oscillation oracle + D1 two-run.
//
// Spec validation item 1: a single particle-pair under an XPBD distance
// constraint with compliance alpha > 0 behaves as a damped harmonic oscillator
// with a KNOWN discrete period. We derive the period analytically from the EXACT
// discrete scheme (no fabricated trajectory) and assert the GPU sim agrees.
//
// Reduced 1-D analysis (gravity = 0, motion along a single axis). Let x = signed
// displacement from rest length and xd the relative velocity. One step of the
// shipped predict -> single-GS-solve -> correct loop is, for a free particle
// against a pinned anchor (w_sum = w_a, lambda reset to 0 each step):
//     predict :  x_pred  = x + xd*dt
//     solve   :  C = x_pred ;  dl = -x_pred/(w_sum + alpha_tilde)
//                x_corr  = x_pred + w_sum*dl = gamma * x_pred
//     correct :  xd_new  = (x_corr - x)/dt
//   with  gamma = alpha/(alpha + w_sum*dt^2) = alpha_tilde * effective_mass.
// The resulting linear map has |determinant| = gamma < 1 (XPBD's intrinsic
// numerical damping, even at damping_beta = 0), so the AMPLITUDE decays at
// sqrt(gamma) per step while the OSCILLATION period is exactly
//     T_discrete = 2*pi*dt / acos(sqrt(gamma)).
// In the continuous limit (dt -> 0) this tends to 2*pi*sqrt(alpha/w_sum) -- the
// SHM period of a spring of stiffness 1/alpha and reduced mass 1/w_sum -- which
// is the sanity anchor. Because amplitude decays, the period is measured from
// ZERO-CROSSINGS of (distance - rest_length), never from an amplitude return.
//
// NOTE (deferred): a full Houdini/Vellum golden trajectory oracle is NOT used --
// Vellum is not reproducible in this environment. The complete Vellum oracle is
// DEFERRED to p15 (the oracle catalog). This analytic period is the p09-A oracle.
// ---------------------------------------------------------------------------

#include "runtime/soft/xpbd_world.hpp"

#include "math/vec3.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

using nuka::math::Vec3;
using nuka::runtime::soft::StepXpbdWorld;
using nuka::runtime::soft::UploadXpbdWorld;
using nuka::runtime::soft::XpbdConstraintSet;
using nuka::runtime::soft::XpbdDistanceConstraint;
using nuka::runtime::soft::XpbdParticleSet;
using nuka::runtime::soft::XpbdStepOptions;
using nuka::runtime::soft::XpbdWorld;
using nuka::runtime::soft::XpbdWorldState;

constexpr float kPi = 3.14159265358979323846f;

// A free particle (b, inv_mass = 1) tethered to a pinned anchor (a, inv_mass = 0)
// by an XPBD distance constraint of rest length L and compliance alpha, stretched
// to L + stretch along +x at t=0 with zero velocity.
struct OscillatorSetup {
    float rest_length = 1.0f;
    float stretch = 0.1f;
    float inv_mass = 1.0f;      // w_b (w_a = 0, pinned anchor)
    float alpha = 0.02f;        // compliance (1/stiffness)
    float dt = 1.0f / 240.0f;
};

XpbdWorld BuildOscillator(const OscillatorSetup& s) {
    XpbdParticleSet particles;
    particles.positions = {Vec3{0.0f, 0.0f, 0.0f},
                           Vec3{s.rest_length + s.stretch, 0.0f, 0.0f}};
    particles.velocities = {Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 0.0f}};
    particles.inv_masses = {0.0f, s.inv_mass};  // particle 0 pinned, 1 free.

    XpbdConstraintSet constraints;
    XpbdDistanceConstraint dc;
    dc.particle_a = 0u;
    dc.particle_b = 1u;
    dc.rest_length = s.rest_length;
    dc.compliance_alpha = s.alpha;
    constraints.distance = {dc};

    return UploadXpbdWorld(particles, constraints);
}

} // namespace

// Gate 3: the observed zero-crossing period matches the analytic discrete period.
TEST(XpbdDistanceOracle, OscillationPeriodMatchesAnalyticDiscrete) {
    OscillatorSetup s;
    XpbdWorld world = BuildOscillator(s);

    const float w_sum = s.inv_mass;  // w_a = 0.
    const float gamma = s.alpha / (s.alpha + w_sum * s.dt * s.dt);
    const float t_discrete = 2.0f * kPi * s.dt / std::acos(std::sqrt(gamma));
    const float t_continuous = 2.0f * kPi * std::sqrt(s.alpha / w_sum);

    // Sanity: discrete period is close to the continuous SHM anchor (dt is small).
    EXPECT_NEAR(t_discrete, t_continuous, 0.02f * t_continuous);

    // Step one dt at a time, recording the signed displacement x = dist - L.
    XpbdStepOptions options;
    options.gravity = Vec3{0.0f, 0.0f, 0.0f};  // pure SHM about rest length.
    options.dt = s.dt;
    options.step_count = 1u;
    options.solver_iterations = 1u;

    const uint32_t kSteps = 6000u;  // many periods at ~213 steps/period.
    std::vector<float> x;
    x.reserve(kSteps);
    for (uint32_t i = 0; i < kSteps; ++i) {
        StepXpbdWorld(world, options);
        const XpbdWorldState st = world.DownloadState();
        const Vec3 r = st.positions[1] - st.positions[0];
        const float dist = std::sqrt(r.Dot(r));
        x.push_back(dist - s.rest_length);
    }

    // Count zero-crossings; consecutive crossings are half-periods.
    int first = -1;
    int last = -1;
    int crossings = 0;
    for (uint32_t i = 1; i < kSteps; ++i) {
        const bool up = x[i - 1] <= 0.0f && x[i] > 0.0f;
        const bool down = x[i - 1] >= 0.0f && x[i] < 0.0f;
        if (up || down) {
            if (first < 0) {
                first = static_cast<int>(i);
            }
            last = static_cast<int>(i);
            ++crossings;
        }
    }
    ASSERT_GE(crossings, 3) << "too few zero-crossings to estimate a period";

    const float half_period_steps =
        static_cast<float>(last - first) / static_cast<float>(crossings - 1);
    const float t_observed = 2.0f * half_period_steps * s.dt;

    const float rel_err = std::fabs(t_observed - t_discrete) / t_discrete;
    EXPECT_LT(rel_err, 0.02f)
        << "T_discrete=" << t_discrete << " T_observed=" << t_observed
        << " T_continuous=" << t_continuous << " rel_err=" << rel_err
        << " crossings=" << crossings;
}

// Gate 4: D1 two-run byte-exactness of the XPBD forward (predict+solve+correct).
// Two independent uploads + identical step schedules must yield byte-identical
// position AND velocity buffers (memcmp on the raw float storage).
TEST(XpbdDistanceOracle, ForwardIsByteExactAcrossRuns) {
    OscillatorSetup s;

    XpbdStepOptions options;
    options.gravity = Vec3{0.0f, -9.81f, 0.0f};  // exercise the gravity predict too.
    options.dt = s.dt;
    options.step_count = 1u;
    options.solver_iterations = 4u;  // multiple GS iterations in the serial sweep.

    const uint32_t kSteps = 500u;

    auto run = [&]() {
        XpbdWorld world = BuildOscillator(s);
        for (uint32_t i = 0; i < kSteps; ++i) {
            StepXpbdWorld(world, options);
        }
        return world.DownloadState();
    };

    const XpbdWorldState a = run();
    const XpbdWorldState b = run();

    ASSERT_EQ(a.positions.size(), b.positions.size());
    ASSERT_EQ(a.velocities.size(), b.velocities.size());
    EXPECT_EQ(std::memcmp(a.positions.data(), b.positions.data(),
                          a.positions.size() * sizeof(Vec3)),
              0)
        << "XPBD position buffer is not two-run byte-identical (D1 violation)";
    EXPECT_EQ(std::memcmp(a.velocities.data(), b.velocities.data(),
                          a.velocities.size() * sizeof(Vec3)),
              0)
        << "XPBD velocity buffer is not two-run byte-identical (D1 violation)";
}

// Inert-when-empty sanity at the soft-world level: an empty XpbdWorld steps to a
// no-op (zero kernel launches, zero particles) -- the building block of the
// cuda_world_stepper inert-when-empty contract.
TEST(XpbdDistanceOracle, EmptyWorldStepsToNoOp) {
    XpbdParticleSet particles;  // no particles.
    XpbdConstraintSet constraints;
    XpbdWorld world = UploadXpbdWorld(particles, constraints);

    XpbdStepOptions options;
    options.step_count = 5u;
    const auto report = StepXpbdWorld(world, options);

    EXPECT_EQ(report.particle_count, 0u);
    EXPECT_EQ(report.kernel_launch_count, 0u);
    EXPECT_EQ(report.simulated_step_count, 0u);
}
