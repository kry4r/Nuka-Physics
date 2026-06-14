// ---------------------------------------------------------------------------
// v0.7 p09-A: XPBD distance-constraint ANALYTIC oscillation oracle + D1 two-run,
// M9 T11 Phase-2b RE-POINTED to nk (the legacy XPBD soft stepper is
// deleted; the distance solve is op-ified onto the unified nk core -- particles.cu
// XpbdDistanceKernel, run inside nk::World's particle solve). The analytic period
// derivation and the D1 / empty-no-op invariants are preserved verbatim; only the
// stepper underneath them changes.
//
// Spec validation item 1: a single particle-pair under an XPBD distance
// constraint with compliance alpha > 0 behaves as a damped harmonic oscillator
// with a KNOWN discrete period. We derive the period analytically from the EXACT
// discrete scheme (no fabricated trajectory) and assert the nk sim agrees.
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

#include "import/cooker/xpbd_cooker_types.hpp"  // XpbdConstraintSet/XpbdParticleSet (CUDA-free PODs)

#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/device_context.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Vec3;
using nuka::runtime::soft::XpbdConstraintSet;
using nuka::runtime::soft::XpbdDistanceConstraint;
using nuka::runtime::soft::XpbdParticleSet;

constexpr float kPi = 3.14159265358979323846f;

// nk backend context (shared singleton, the nk_particle_equivalence pattern).
struct NkCtx { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
NkCtx GetNkCtx() {
    static NkCtx c = [] {
        NkCtx r;
        static nuka::phi::DeviceContext keep = nuka::phi::MakeDefaultDeviceContext();
        (void)keep;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return c;
}

// Downloaded particle state (replaces the legacy soft world state).
struct XpbdState {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
};

// Cook an XpbdParticleSet + XpbdConstraintSet into an nk::Model (mode = Xpbd),
// transcribing the de-interleaved distance SoA exactly as CookXpbdParticles does.
nk::Model BuildNkXpbdModel(const XpbdParticleSet& ps, const XpbdConstraintSet& cs,
                           uint16_t iters) {
    nk::Model model;
    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Xpbd;
    mp.initial_pos = ps.positions;
    mp.initial_vel = ps.velocities;
    mp.inv_mass = ps.inv_masses;
    mp.xpbd_iters = iters == 0u ? 1u : iters;
    const uint32_t dn = static_cast<uint32_t>(cs.distance.size());
    for (uint32_t c = 0; c < dn; ++c) {
        mp.dist_a.push_back(cs.distance[c].particle_a);
        mp.dist_b.push_back(cs.distance[c].particle_b);
        mp.dist_rest.push_back(cs.distance[c].rest_length);
        mp.dist_alpha.push_back(cs.distance[c].compliance_alpha);
    }
    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = 1;
    cap.particles_per_env = static_cast<uint32_t>(ps.positions.size());
    cap.dist_cons_per_env = dn;
    return model;
}

// An nk XPBD world wrapper that exposes per-step download for the oscillation
// trajectory (the oracle needs the distance after EACH step). Cook once, Step()
// one dt at a time, download position after each.
struct NkXpbdRunner {
    nk::World world;
    uint32_t P = 0u;
    NkXpbdRunner(const XpbdParticleSet& ps, const XpbdConstraintSet& cs,
                 uint16_t iters, Vec3 gravity, float dt)
        : world([&] {
              NkCtx c = GetNkCtx();
              nk::Pipeline::SolverConfig cfg;
              cfg.dt = dt;
              cfg.gravity[0] = gravity.x; cfg.gravity[1] = gravity.y;
              cfg.gravity[2] = gravity.z;
              return nk::World(BuildNkXpbdModel(ps, cs, iters), 1u, c.dev, c.backend,
                               cfg);
          }()) {
        P = world.GetModel().capacities.particles_per_env;
    }
    void Step() { world.Step(); }
    XpbdState Download() {
        XpbdState st;
        st.positions.resize(P);
        st.velocities.resize(P);
        world.GetData().DownloadField(nk::FieldId::ParticlePos, st.positions.data(),
                                      P * sizeof(Vec3));
        world.GetData().DownloadField(nk::FieldId::ParticleVel, st.velocities.data(),
                                      P * sizeof(Vec3));
        return st;
    }
};

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

void BuildOscillator(const OscillatorSetup& s, XpbdParticleSet& particles,
                     XpbdConstraintSet& constraints) {
    particles.positions = {Vec3{0.0f, 0.0f, 0.0f},
                           Vec3{s.rest_length + s.stretch, 0.0f, 0.0f}};
    particles.velocities = {Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 0.0f}};
    particles.inv_masses = {0.0f, s.inv_mass};  // particle 0 pinned, 1 free.

    XpbdDistanceConstraint dc;
    dc.particle_a = 0u;
    dc.particle_b = 1u;
    dc.rest_length = s.rest_length;
    dc.compliance_alpha = s.alpha;
    constraints.distance = {dc};
}

} // namespace

// Gate 3: the observed zero-crossing period matches the analytic discrete period.
TEST(XpbdDistanceOracle, OscillationPeriodMatchesAnalyticDiscrete) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    OscillatorSetup s;
    XpbdParticleSet particles;
    XpbdConstraintSet constraints;
    BuildOscillator(s, particles, constraints);

    const float w_sum = s.inv_mass;  // w_a = 0.
    const float gamma = s.alpha / (s.alpha + w_sum * s.dt * s.dt);
    const float t_discrete = 2.0f * kPi * s.dt / std::acos(std::sqrt(gamma));
    const float t_continuous = 2.0f * kPi * std::sqrt(s.alpha / w_sum);

    // Sanity: discrete period is close to the continuous SHM anchor (dt is small).
    EXPECT_NEAR(t_discrete, t_continuous, 0.02f * t_continuous);

    // Step one dt at a time, recording the signed displacement x = dist - L. nk
    // single-GS sweep == the legacy serial sweep (solver_iterations = 1).
    NkXpbdRunner runner(particles, constraints, /*iters=*/1u,
                        Vec3{0.0f, 0.0f, 0.0f}, s.dt);

    const uint32_t kSteps = 6000u;  // many periods at ~213 steps/period.
    std::vector<float> x;
    x.reserve(kSteps);
    for (uint32_t i = 0; i < kSteps; ++i) {
        runner.Step();
        const XpbdState st = runner.Download();
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

// Gate 4: D1 two-run byte-exactness of the XPBD forward (predict+solve+correct) on
// nk. Two independent cooks + identical step schedules must yield byte-identical
// position AND velocity buffers (memcmp on the raw float storage).
TEST(XpbdDistanceOracle, ForwardIsByteExactAcrossRuns) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    OscillatorSetup s;

    const uint32_t kSteps = 500u;

    auto run = [&]() {
        XpbdParticleSet particles;
        XpbdConstraintSet constraints;
        BuildOscillator(s, particles, constraints);
        // gravity ON to exercise the predict; multiple GS iterations in the sweep.
        NkXpbdRunner runner(particles, constraints, /*iters=*/4u,
                            Vec3{0.0f, -9.81f, 0.0f}, s.dt);
        for (uint32_t i = 0; i < kSteps; ++i) {
            runner.Step();
        }
        return runner.Download();
    };

    const XpbdState a = run();
    const XpbdState b = run();

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

// Inert-when-empty sanity at the nk soft path: a world with ZERO particles steps to
// a no-op (no particle solve, no constraints) -- the building block of the
// world-level inert-when-empty contract. Cooking an empty particle model and
// stepping it must not crash and must keep zero particles.
TEST(XpbdDistanceOracle, EmptyWorldStepsToNoOp) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    XpbdParticleSet particles;     // no particles.
    XpbdConstraintSet constraints; // no constraints.
    NkXpbdRunner runner(particles, constraints, /*iters=*/1u,
                        Vec3{0.0f, -9.81f, 0.0f}, 1.0f / 240.0f);
    EXPECT_EQ(runner.P, 0u);
    for (uint32_t i = 0; i < 5u; ++i) {
        runner.Step();  // must be a clean no-op (zero particles).
    }
    const XpbdState st = runner.Download();
    EXPECT_TRUE(st.positions.empty());
    EXPECT_TRUE(st.velocities.empty());
}
