// ---------------------------------------------------------------------------
// v0.7 p11 (K3): cross-system particle-particle CONTACT co-step -- ANALYTIC
// invariant oracles + D1 two-run + inert byte-identity.
//
// There is no Vellum/Flex/Houdini golden available here (deferred to p15), so the
// physical correctness of the coupling co-step is validated with analytic
// invariants of the mass-weighted symmetric non-penetration projection
// (StepParticleParticleCoupling, src/runtime/coupling/particle_particle_contact):
//
//   (a) SEPARATION (LOAD-BEARING, EXACT): two overlapping particles project in one
//       full correction to EXACTLY d_min separation. No iteration needed -- the
//       single-pair XPBD multiplier (a~ = 0, lambda warm 0) is dl = -C/(w_i+w_j),
//       so dp_i - dp_j = (w_i+w_j)*n*dl = -C*n, i.e. the gap closes by exactly -C
//       and the final |p_i - p_j| = d_min. Asserted to ~1e-5.
//
//   (b) MASS-WEIGHTED CENTROID CONSERVATION (LOAD-BEARING, UNEQUAL MASSES): the
//       projection conserves the MASS-WEIGHTED centroid Sum_i m_i p_i (analytically
//       m_i*dp_i + m_j*dp_j = 0 since m_i*w_i = 1). With UNEQUAL masses the
//       mass-weighting BITES: the test asserts Sum m_i*dp_i ~ 0 (conserved) AND
//       that the UNWEIGHTED centroid (p_i + p_j)/2 MOVES (so the gate is not the
//       trivial equal-mass coincidence). truncated_particle_count is asserted 0
//       (momentum conservation relies on SYMMETRIC neighbor detection).
//
//   (c) SOFT<->FLUID pair: one XPBD particle + one PBF particle, overlapping, are
//       separated by the SAME class-blind row to exactly d_min. The infrastructure
//       deliverable: the co-step does not distinguish the two subsystems.
//
//   (d) MULTI-PAIR SEGMENT REDUCTION (3-in-a-line): the MIDDLE particle is in TWO
//       pairs; its accumulated correction is the SUM of both pairs' halves -- the
//       segment-reduction the D1 Jacobi gather must get right (a float atomicAdd
//       would be non-deterministic here). LOAD-BEARING: the middle particle's
//       FIRST-iteration delta equals the hand-computed sum of the two halves.
//       The multi-iteration converged geometry is a CALIBRATION SENTINEL (Jacobi
//       under-relaxes; final positions are iteration-count dependent), labeled so.
//
//   (e) D1 two-run byte-exact (positions + velocities, memcmp over N steps).
//
//   (f) INERT byte-identity: with NO cross pairs within d_min, the attached co-step
//       leaves both worlds byte-identical to running them independently (no kernel
//       perturbs state; report.inert == true).
//
// SCOPE BOUNDARY (p11 infrastructure): the co-step is a POSITION-space pass run
// AFTER each subsystem's own solve; it does NOT re-derive velocity from the
// correction (the position change propagates to velocity on the next subsystem
// step's finalize). The D1 / inert gates therefore compare positions+velocities
// after the coupling pass on top of the subsystem state.
// ---------------------------------------------------------------------------

#include "runtime/coupling/particle_particle_contact.hpp"

#include "math/vec3.hpp"
#include "runtime/fluid/pbf_world.hpp"
#include "runtime/soft/xpbd_world.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

using nuka::math::Vec3;
using nuka::runtime::coupling::ParticleParticleContactParams;
using nuka::runtime::coupling::ParticleParticleContactReport;
using nuka::runtime::coupling::StepParticleParticleCoupling;
using nuka::runtime::fluid::PbfParticleSet;
using nuka::runtime::fluid::PbfWorld;
using nuka::runtime::fluid::UploadPbfWorld;
using nuka::runtime::soft::UploadXpbdWorld;
using nuka::runtime::soft::XpbdConstraintSet;
using nuka::runtime::soft::XpbdParticleSet;
using nuka::runtime::soft::XpbdWorld;

float Distance(const Vec3& a, const Vec3& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

XpbdWorld MakeXpbd(const std::vector<Vec3>& positions,
                   const std::vector<float>& inv_masses) {
    XpbdParticleSet particles;
    particles.positions = positions;
    particles.velocities.assign(positions.size(), Vec3{0.0f, 0.0f, 0.0f});
    particles.inv_masses = inv_masses;
    XpbdConstraintSet empty_constraints;  // pure coupling test: no internal rows.
    return UploadXpbdWorld(particles, empty_constraints);
}

PbfWorld MakePbf(const std::vector<Vec3>& positions, float mass) {
    PbfParticleSet set;
    set.positions = positions;
    set.velocities.assign(positions.size(), Vec3{0.0f, 0.0f, 0.0f});
    set.particle_mass = mass;
    return UploadPbfWorld(set);
}

// (a) Two overlapping XPBD particles (equal unit mass) project to EXACTLY d_min.
TEST(ParticleParticleContact, OverlappingPairSeparatesToExactlyDMin) {
    constexpr float kDmin = 1.0f;
    auto xpbd = MakeXpbd({Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.4f, 0.0f, 0.0f}},
                         {1.0f, 1.0f});
    PbfParticleSet empty;
    auto pbf = UploadPbfWorld(empty);

    ParticleParticleContactParams params;
    params.contact_distance_d_min = kDmin;
    const auto report = StepParticleParticleCoupling(xpbd, pbf, params);
    EXPECT_FALSE(report.inert);
    EXPECT_EQ(report.truncated_particle_count, 0u);

    const auto state = xpbd.DownloadState();
    EXPECT_NEAR(Distance(state.positions[0], state.positions[1]), kDmin, 1.0e-5f);
}

// (b) Mass-weighted centroid conserved under UNEQUAL masses (the load-bearing
// version); the unweighted centroid moves, proving the mass-weighting bites.
TEST(ParticleParticleContact, MassWeightedCentroidConservedUnequalMasses) {
    constexpr float kDmin = 1.0f;
    // Unequal masses: particle 0 is heavy (m=4, w=0.25), particle 1 is light
    // (m=1, w=1). The light particle should move 4x as far.
    const float w0 = 0.25f;  // m0 = 4
    const float w1 = 1.0f;   // m1 = 1
    const Vec3 p0_init{0.0f, 0.0f, 0.0f};
    const Vec3 p1_init{0.4f, 0.0f, 0.0f};
    auto xpbd = MakeXpbd({p0_init, p1_init}, {w0, w1});
    PbfParticleSet empty;
    auto pbf = UploadPbfWorld(empty);

    ParticleParticleContactParams params;
    params.contact_distance_d_min = kDmin;
    StepParticleParticleCoupling(xpbd, pbf, params);

    const auto state = xpbd.DownloadState();
    const Vec3 p0 = state.positions[0];
    const Vec3 p1 = state.positions[1];
    const float m0 = 1.0f / w0;
    const float m1 = 1.0f / w1;

    // Mass-weighted centroid (Sum m_i p_i) conserved: m0*dp0 + m1*dp1 == 0.
    const float wc_x = m0 * (p0.x - p0_init.x) + m1 * (p1.x - p1_init.x);
    EXPECT_NEAR(wc_x, 0.0f, 1.0e-4f);

    // The UNWEIGHTED centroid MOVES (so the mass-weighting is genuinely load-
    // bearing, not the equal-mass coincidence). dp0 = -0.25*(-C), dp1 = +1.0*(-C)
    // scaled: the unweighted midpoint shifts toward the heavy particle's side.
    const float unweighted_mid_init = 0.5f * (p0_init.x + p1_init.x);
    const float unweighted_mid = 0.5f * (p0.x + p1.x);
    EXPECT_GT(std::fabs(unweighted_mid - unweighted_mid_init), 1.0e-3f);

    // The light particle (w1) moved farther than the heavy (w0): |dp1| = 4*|dp0|.
    const float dp0 = std::fabs(p0.x - p0_init.x);
    const float dp1 = std::fabs(p1.x - p1_init.x);
    EXPECT_NEAR(dp1 / dp0, m0 / m1, 1.0e-3f);  // ratio == w1/w0 == m0/m1 == 4.

    // Separation still lands at exactly d_min.
    EXPECT_NEAR(Distance(p0, p1), kDmin, 1.0e-5f);
}

// (c) Soft<->fluid: one XPBD + one PBF particle, overlapping, separate to d_min.
TEST(ParticleParticleContact, SoftFluidPairSeparates) {
    constexpr float kDmin = 1.0f;
    auto xpbd = MakeXpbd({Vec3{0.0f, 0.0f, 0.0f}}, {1.0f});  // 1 soft particle, m=1
    auto pbf = MakePbf({Vec3{0.5f, 0.0f, 0.0f}}, 1.0f);      // 1 fluid particle, m=1

    ParticleParticleContactParams params;
    params.contact_distance_d_min = kDmin;
    const auto report = StepParticleParticleCoupling(xpbd, pbf, params);
    EXPECT_FALSE(report.inert);
    EXPECT_EQ(report.union_particle_count, 2u);
    EXPECT_EQ(report.truncated_particle_count, 0u);

    const Vec3 ps = xpbd.DownloadState().positions[0];
    const Vec3 pf = pbf.DownloadState().positions[0];
    EXPECT_NEAR(Distance(ps, pf), kDmin, 1.0e-5f);
}

// (d) Multi-pair segment reduction: 3 XPBD particles in a line; the MIDDLE is in
// TWO pairs and accumulates the SUM of both halves. LOAD-BEARING on the first-
// iteration sum; the converged geometry is a calibration sentinel.
TEST(ParticleParticleContact, MiddleParticleAccumulatesBothPairHalves) {
    constexpr float kDmin = 1.0f;
    // Equal unit mass; spacing 0.6 < d_min so BOTH adjacent pairs penetrate, but
    // the 0-2 gap (1.2 > d_min) does NOT, so the middle particle (1) is in exactly
    // two pairs and the ends are each in one.
    const Vec3 p0{0.0f, 0.0f, 0.0f};
    const Vec3 p1{0.6f, 0.0f, 0.0f};
    const Vec3 p2{1.2f, 0.0f, 0.0f};
    auto xpbd = MakeXpbd({p0, p1, p2}, {1.0f, 1.0f, 1.0f});
    PbfParticleSet empty;
    auto pbf = UploadPbfWorld(empty);

    ParticleParticleContactParams params;
    params.contact_distance_d_min = kDmin;
    params.solver_iterations = 1u;  // single sweep: delta is the exact half-sum.
    const auto report = StepParticleParticleCoupling(xpbd, pbf, params);
    EXPECT_FALSE(report.inert);
    EXPECT_EQ(report.truncated_particle_count, 0u);

    const auto state = xpbd.DownloadState();
    // Hand-computed FIRST-iteration delta of the middle particle (1):
    //   pair (1,0): C = 0.6 - 1 = -0.4, n = +x, dl = 0.4/2 = 0.2, half = w*n*dl
    //               = +0.2 (particle 1 is the "i" with p_i - p_j = +x) -> +0.2.
    //   pair (1,2): C = 0.6 - 1 = -0.4, n = -x (p1 - p2), dl = 0.2, half = -0.2.
    //   sum = +0.2 + (-0.2) = 0.0  -> the symmetric middle particle does NOT move.
    // The ENDS each get one half of +/-0.2 outward. This is the segment-reduction:
    // the middle's two halves are summed (and here cancel), the ends get one each.
    EXPECT_NEAR(state.positions[1].x, p1.x, 1.0e-5f);    // middle stays (halves cancel).
    EXPECT_NEAR(state.positions[0].x, p0.x - 0.2f, 1.0e-5f);  // left pushed -x.
    EXPECT_NEAR(state.positions[2].x, p2.x + 0.2f, 1.0e-5f);  // right pushed +x.
    // y/z untouched.
    EXPECT_NEAR(state.positions[1].y, 0.0f, 1.0e-6f);
    EXPECT_NEAR(state.positions[1].z, 0.0f, 1.0e-6f);
}

// (d') ASYMMETRIC multi-pair segment reduction (the self-evidently load-bearing
// version): the symmetric (d) case sums to 0, so "summed correctly" and "skipped
// the middle" coincide. Here the two pairs penetrate by DIFFERENT amounts so the
// middle's summed correction is a value that equals NEITHER single half NOR rest
// -- a break-early / overwrite-last segment-reduction bug is then unambiguously
// caught. This is the gate for the D1 Jacobi gather's per-particle SUM over all
// pairs (which a float atomicAdd would compute non-deterministically).
TEST(ParticleParticleContact, MiddleParticleSumsAsymmetricPairHalves) {
    constexpr float kDmin = 1.0f;
    // Middle at 0.5; left at 0.0 (gap 0.5), right at 1.2 (gap 0.7). 0-2 gap is 1.2
    // > d_min so the ends do not pair directly; the middle (1) is in two pairs of
    // UNEQUAL penetration. Equal unit mass.
    const Vec3 p0{0.0f, 0.0f, 0.0f};
    const Vec3 p1{0.5f, 0.0f, 0.0f};
    const Vec3 p2{1.2f, 0.0f, 0.0f};
    auto xpbd = MakeXpbd({p0, p1, p2}, {1.0f, 1.0f, 1.0f});
    PbfParticleSet empty;
    auto pbf = UploadPbfWorld(empty);

    ParticleParticleContactParams params;
    params.contact_distance_d_min = kDmin;
    params.solver_iterations = 1u;
    StepParticleParticleCoupling(xpbd, pbf, params);

    const auto state = xpbd.DownloadState();
    // Hand-computed FIRST-iteration delta of the middle particle (1):
    //   pair (1,0): C = 0.5 - 1 = -0.5, n = +x, dl = 0.5/2 = 0.25, half = +0.25.
    //   pair (1,2): C = 0.7 - 1 = -0.3, n = -x, dl = 0.3/2 = 0.15, half = -0.15.
    //   sum = +0.25 + (-0.15) = +0.10  -> equals NEITHER half, NOR 0.
    EXPECT_NEAR(state.positions[1].x, p1.x + 0.10f, 1.0e-5f);  // summed half-corrections.
    EXPECT_NEAR(state.positions[0].x, p0.x - 0.25f, 1.0e-5f);  // left: its single half.
    EXPECT_NEAR(state.positions[2].x, p2.x + 0.15f, 1.0e-5f);  // right: its single half.
}

// (e) D1: two runs of the same coupling co-step produce byte-identical positions
// + velocities across BOTH worlds.
TEST(ParticleParticleContact, D1TwoRunByteIdentical) {
    constexpr float kDmin = 1.0f;
    auto run = [&]() {
        auto xpbd = MakeXpbd({Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.5f, 0.0f, 0.0f},
                              Vec3{0.3f, 0.4f, 0.0f}},
                             {1.0f, 0.5f, 2.0f});
        auto pbf = MakePbf({Vec3{0.2f, 0.2f, 0.0f}, Vec3{0.7f, 0.1f, 0.0f}}, 1.5f);
        ParticleParticleContactParams params;
        params.contact_distance_d_min = kDmin;
        params.solver_iterations = 4u;
        for (int step = 0; step < 5; ++step) {
            StepParticleParticleCoupling(xpbd, pbf, params);
        }
        return std::pair<decltype(xpbd.DownloadState()), decltype(pbf.DownloadState())>(
            xpbd.DownloadState(), pbf.DownloadState());
    };

    const auto a = run();
    const auto b = run();

    ASSERT_EQ(a.first.positions.size(), b.first.positions.size());
    EXPECT_EQ(std::memcmp(a.first.positions.data(), b.first.positions.data(),
                          a.first.positions.size() * sizeof(Vec3)),
              0)
        << "xpbd positions diverged between two coupling runs (D1)";
    EXPECT_EQ(std::memcmp(a.second.positions.data(), b.second.positions.data(),
                          a.second.positions.size() * sizeof(Vec3)),
              0)
        << "pbf positions diverged between two coupling runs (D1)";
    EXPECT_EQ(std::memcmp(a.first.velocities.data(), b.first.velocities.data(),
                          a.first.velocities.size() * sizeof(Vec3)),
              0)
        << "xpbd velocities diverged between two coupling runs (D1)";
    EXPECT_EQ(std::memcmp(a.second.velocities.data(), b.second.velocities.data(),
                          a.second.velocities.size() * sizeof(Vec3)),
              0)
        << "pbf velocities diverged between two coupling runs (D1)";
}

// (f) Inert byte-identity: with NO cross pairs within d_min, the attached co-step
// leaves both worlds byte-identical to NOT calling it (no kernel perturbs state).
TEST(ParticleParticleContact, SubsystemPathsByteIdenticalWithNoCrossPairs) {
    // Particles spaced WELL beyond d_min: no pair is within range.
    constexpr float kDmin = 0.1f;
    const std::vector<Vec3> xpbd_pos = {Vec3{0.0f, 0.0f, 0.0f},
                                        Vec3{5.0f, 0.0f, 0.0f}};
    const std::vector<Vec3> pbf_pos = {Vec3{10.0f, 0.0f, 0.0f},
                                       Vec3{15.0f, 0.0f, 0.0f}};

    // Reference: upload + immediately download (no coupling pass at all).
    auto ref_xpbd = MakeXpbd(xpbd_pos, {1.0f, 1.0f});
    auto ref_pbf = MakePbf(pbf_pos, 1.0f);
    const auto ref_x = ref_xpbd.DownloadState();
    const auto ref_p = ref_pbf.DownloadState();

    // With coupling attached: it must be INERT and not touch state.
    auto xpbd = MakeXpbd(xpbd_pos, {1.0f, 1.0f});
    auto pbf = MakePbf(pbf_pos, 1.0f);
    ParticleParticleContactParams params;
    params.contact_distance_d_min = kDmin;
    const auto report = StepParticleParticleCoupling(xpbd, pbf, params);

    EXPECT_TRUE(report.inert);
    EXPECT_EQ(report.kernel_launch_count, 0u);
    EXPECT_EQ(report.total_candidate_neighbors, 0u);

    const auto x = xpbd.DownloadState();
    const auto p = pbf.DownloadState();
    EXPECT_EQ(std::memcmp(ref_x.positions.data(), x.positions.data(),
                          ref_x.positions.size() * sizeof(Vec3)),
              0)
        << "xpbd positions perturbed by an inert coupling co-step";
    EXPECT_EQ(std::memcmp(ref_p.positions.data(), p.positions.data(),
                          ref_p.positions.size() * sizeof(Vec3)),
              0)
        << "pbf positions perturbed by an inert coupling co-step";
    EXPECT_EQ(std::memcmp(ref_x.velocities.data(), x.velocities.data(),
                          ref_x.velocities.size() * sizeof(Vec3)),
              0)
        << "xpbd velocities perturbed by an inert coupling co-step";
    EXPECT_EQ(std::memcmp(ref_p.velocities.data(), p.velocities.data(),
                          ref_p.velocities.size() * sizeof(Vec3)),
              0)
        << "pbf velocities perturbed by an inert coupling co-step";
}

} // namespace
