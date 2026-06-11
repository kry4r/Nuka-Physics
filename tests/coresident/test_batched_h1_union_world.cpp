// ---------------------------------------------------------------------------
// G1 -- the BATCHED UNION WORLD parity TU (spec docs/specs/2026-06-10-h1-whole-
// body-rl-grasp-spec.md gate G1; brief subagent-plans/g1-union-world-brief.md).
// ---------------------------------------------------------------------------
// G1a (THIS increment) -- FLOATING-BASE batched articulation parity, NO CONTACT.
// The whole-body H1 (nuka::demo::LoadFloating, 51 DOF: 6 floating-base + 45
// revolute) runs through BatchedUnifiedWorld's env-major machinery and must be
// BYTE-EXACT (tol 0) vs the N=1 UnifiedCoResidentStepper oracle on the IDENTICAL
// proto. NO contact rows are ever emitted (the cup sits 5 m away and free-falls
// alongside the robot), which ISOLATES exactly the floating-base surfaces H1.2
// never exercised (its hand cook was Fixed-root, base_dof_=0):
//   * floating root replication (ReplicateArticulationHostState base_pose tiling),
//   * per-env ACTIVATION of IntegrateFloatingBaseVelocity / IntegrateFloatingBase-
//     Pose (one block per articulation; no-ops on a Fixed root, LIVE here),
//   * SetActions over a dof_stride that now contains 6 base columns (dof_to_link_
//     maps columns 0..5 to the FloatingBase root -- they must be DEAD, gate d),
//   * the no-contact early-return path at base_dof_=6 (no qdot pack/scatter runs;
//     the contact-path base prefix-sum lanes are G1b's surface, gated on contact).
// NOTE the chaos caveat of H1.2 does NOT apply here: with ZERO contact rows both
// sides run the SAME kernels in the SAME order on the SAME state -- full-run
// byte-exactness (tol 0) is the honest bar, and the scene is DISCRIMINATIVE (the
// base genuinely free-falls while the legs PD-swing to the bent stance; asserted),
// so two frozen robots cannot fake a pass.
//
// G1b..G1d will EXTEND this TU (feet x ground, finger x cup, cup x table union
// rows); the shared scene/drive/snapshot helpers below are written for that.
// ---------------------------------------------------------------------------
// G1b (THIS increment) -- FEET x GROUND union rows at the floating base.
// The grasp branch now ALSO emits foot-sphere x ground-plane rows (authored ankle
// spheres, HOST SpherePlane narrowphase, foot chain-J slots into the SAME gather
// list as fingers) into the ONE per-env append -> one-solve loop. This is the
// FIRST contact at base_dof_=6: the contact-path base prefix-sum qdot pack
// (batched_unified_world.cpp pack lanes) and scatter lanes G1a could not reach
// are exactly the surface under test.
//   ORACLE CHOICE (documented per the brief): a StandGraspConfig with the SAME
// feet + ground + 30 fingertips + FAR cup (5m, never touched) + has_table=false,
// NOT a StandConfig. Why: (1) StepStandGrasp emits EXACTLY feet x ground rows in
// this config (every finger pair narrowphases to nothing 5m away; the table pair
// is gated off) -- asserted per step via finger_contacts==0 on BOTH sides; (2) it
// is structurally IDENTICAL to the batched grasp branch: ONE movable cup
// (body_count=1, gravity-kicked + integrated), so the foot-side coloring key is
// body_count+kArtIndex==1 on both sides at N=1 AND the free-falling cup can be
// byte-compared -- a StandConfig oracle has NO movable body (body_count=0,
// foot key 0, no cup to compare) and aliases its foot count into finger_contacts;
// (3) it is the SAME oracle mode G1c/G1d extend, so the whole G1 series compares
// against one code path. tol-0 is the bar and the expectation: feet narrowphase
// is HOST code on BOTH sides, the finger GPU narrowphase finds NOTHING (no ULP
// surface), and every device kernel runs in the same order on the same state.
// ---------------------------------------------------------------------------
// ADDITIVE TU. G1b's production edits live in src/runtime/coresident/
// batched_unified_world.{hpp,cpp} (has_feet template fields + foot row emission),
// byte-inert when has_feet=false.
// ---------------------------------------------------------------------------
// G1c (THIS increment) -- + FINGER x CUP at the floating base: the FULL grasp
// while standing (no table). The SAME world now steps feet x ground AND
// fingertip x cup rows in ONE solve: the G1b feet/ground/stance scene + the
// H1.1/H1.2 grasp pose (cup ~10.9cm nestled in the curled right hand via the
// proven BestPlacementShallowNoPalmCaging search, PD-close Kp=4/Kd=0.4/+0.18),
// cup held against gravity by friction force-closure ALONE while the CoP-
// stabilized legs carry robot+cup. ZERO production change expected -- G1b
// already restructured the grasp-branch emission to COMPOSE (feet rows first,
// then finger rows, shared chain-J gather list, per-category friction stamp);
// G1c's job is to PROVE the composed union parity vs the SAME StandGraspConfig
// oracle mode G1b used (now with the cup in the hand instead of parked 5m) and
// flush out composition bugs (row ordering / friction stamping / report
// splitting / qdot pack-scatter with BOTH row classes live in one env).
//   THE PARITY BAR (G1b tol-0 does NOT carry over -- documented, not relaxed):
// bringing the cup into the hand REOPENS the GPU-vs-host SphereHull NP seam
// (the batched finger narrowphase is the GPU launcher; the oracle narrowphases
// on host) and the 30-contact grasp on SHARED links is CHAOTIC at the FP floor
// (H1.2's evidenced finding). So gate (a) HARD-asserts the H1.2 reformulated
// bar: a leading FP-floor window (cup pos/vel <=1e-6, gripper qdot <=5e-6,
// contact-count match through the window) + a CHAOS CONTROL (two same-code
// oracles, 1e-7 m cup IC nudge, must amplify to the same order as the batched-
// vs-oracle delta) + the full-run deltas REPORTED. See the gate-(a) body for
// the full written justification (spec §G1(a) requires it in-test).
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/coresident/batched_unified_world.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"
#include "runtime/rigid/body_state.hpp"

#include "h1_demo_shared.hpp"
// G1e: the union scene authoring + PD/CoP drive seams (UnionScene /
// BuildGraspStandTableScene / DrivePdOracle / ...) were EXTRACTED to this shared
// test header so the G1e union throughput perf cases build from the SAME path.
// `using namespace` below brings every symbol in unqualified -> the 13 union gates
// keep referencing them exactly as before (byte-identical scene + drive).
#include "h1_union_scene_shared.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace coresident = nuka::runtime::coresident;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;

// G1e: the union scene authoring + drive seams now live in the shared header
// included above (EXTRACTED verbatim). Bring every symbol in unqualified so the
// snapshot helpers + the 13 union gates below reference them exactly as before
// (kInvalidLink / kGravityZ / kDt / UnionScene / BuildGraspStandTableScene /
// DrivePdOracle / DrivePdBatched / CopCtl / kRestBackOffset / ...).
using namespace nuka::test::h1union;  // NOLINT(build/namespaces) -- test-side only.

// ===========================================================================
// Snapshots + the byte-exact comparator (tol 0: memcmp, never an epsilon).
// ===========================================================================
struct EnvSnap {
    BodyState cup;
    std::vector<float> q;
    std::vector<float> qdot;
    Transform base_pose{};
    std::vector<articulation::LinkSpatialVel> linkvel;  // ALL links' spatial vel.
};

EnvSnap SnapBatched(const coresident::BatchedUnifiedWorld& world, uint32_t env) {
    EnvSnap s;
    s.cup = world.Body(env, 0u);
    articulation::ArticulationHostState st;
    world.DownloadGripper(env, &st);
    s.q = st.q;
    s.qdot = st.qdot;
    s.base_pose = st.base_pose[0];
    s.linkvel = st.link_velocity;
    return s;
}

EnvSnap SnapOracle(const coresident::UnifiedCoResidentStepper& stepper) {
    EnvSnap s;
    s.cup = stepper.Cup();
    articulation::ArticulationHostState st;
    stepper.Download(&st);
    s.q = st.q;
    s.qdot = st.qdot;
    s.base_pose = st.base_pose[0];
    s.linkvel = st.link_velocity;
    return s;
}

bool SnapByteEqual(const EnvSnap& a, const EnvSnap& b) {
    if (std::memcmp(&a.cup, &b.cup, sizeof(BodyState)) != 0) return false;
    if (a.q.size() != b.q.size() ||
        std::memcmp(a.q.data(), b.q.data(), a.q.size() * sizeof(float)) != 0)
        return false;
    if (a.qdot.size() != b.qdot.size() ||
        std::memcmp(a.qdot.data(), b.qdot.data(), a.qdot.size() * sizeof(float)) != 0)
        return false;
    if (std::memcmp(&a.base_pose, &b.base_pose, sizeof(Transform)) != 0) return false;
    if (a.linkvel.size() != b.linkvel.size() ||
        std::memcmp(a.linkvel.data(), b.linkvel.data(),
                    a.linkvel.size() * sizeof(articulation::LinkSpatialVel)) != 0)
        return false;
    return true;
}

// Float-magnitude diagnostics for a mismatch (printed; the ASSERT is the memcmp).
struct SnapDiff {
    double dq = 0.0, dqdot = 0.0, dbase = 0.0, dcup = 0.0;
};
SnapDiff SnapMaxDiff(const EnvSnap& a, const EnvSnap& b) {
    SnapDiff d;
    for (size_t i = 0u; i < a.q.size() && i < b.q.size(); ++i)
        d.dq = std::max(d.dq, std::fabs(double(a.q[i]) - double(b.q[i])));
    for (size_t i = 0u; i < a.qdot.size() && i < b.qdot.size(); ++i)
        d.dqdot = std::max(d.dqdot, std::fabs(double(a.qdot[i]) - double(b.qdot[i])));
    d.dbase = std::max({std::fabs(double(a.base_pose.position.x) - b.base_pose.position.x),
                        std::fabs(double(a.base_pose.position.y) - b.base_pose.position.y),
                        std::fabs(double(a.base_pose.position.z) - b.base_pose.position.z)});
    d.dcup = (a.cup.position - b.cup.position).Length();
    return d;
}

}  // namespace

// ===========================================================================
// ★ G1a GATE (a) -- N=1 BYTE-EXACT (tol 0) floating-base parity, NO contact.
// 120 lockstep steps: drive BOTH sides from their OWN downloaded state (equivalent
// seams), step both, memcmp base pose + q + qdot + ALL link spatial velocities +
// the (far, free-falling) cup. The no-contact invariant (finger_contacts == 0 on
// BOTH sides, every step) keeps the increment honest; the discriminativeness
// asserts (base z genuinely DROPS >0.3 m, legs genuinely SWING >0.05 rad, base
// genuinely CARRIES speed) make a two-frozen-robots pass impossible.
// ===========================================================================
TEST(BatchedH1UnionWorld, G1a_FloatingBaseNoContact_N1_ByteExactVsOracle) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildFloatingNoContactScene();
    ASSERT_EQ(sc.base_dof, 6u) << "the cook is NOT floating-base -- G1a is vacuous";
    ASSERT_EQ(sc.dof_stride, 51u)
        << "whole-body H1 DOF changed (expected 6 base + 45 revolute)";
    ASSERT_LE(sc.dof_stride, articulation::kMaxArticulationDof);
    ASSERT_GE(sc.config.fingertips.size(), 30u);
    std::printf("[G1a GATE-A] floating H1: links=%u dof_stride=%u base_dof=%u "
                "fingertips=%zu cup@+%.1fm (never touched)\n",
                sc.link_count, sc.dof_stride, sc.base_dof,
                sc.config.fingertips.size(), kCupFarOffsetX);

    const auto tmpl = MakeUnionTemplate(sc);
    const auto drive = BuildDriveSet(sc.h1);
    for (const DriveLink& d : drive) ASSERT_NE(d.link, kInvalidLink);

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGravityZ, kDt);
    coresident::UnifiedCoResidentStepper oracle(context, sc.host, sc.config,
                                                kGravityZ, kDt);

    const EnvSnap snap0 = SnapOracle(oracle);
    const float base_z0 = snap0.base_pose.position.z;

    constexpr uint32_t kRun = 120u;
    const std::vector<float> one_scale = {1.0f};
    for (uint32_t s = 0u; s < kRun; ++s) {
        DrivePdOracle(oracle, sc, drive, 1.0f);
        DrivePdBatched(world, sc, drive, one_scale);
        const auto rep = oracle.Step();
        world.Step();

        // The NO-CONTACT invariant (both sides, every step).
        ASSERT_EQ(rep.finger_contacts, 0u) << "oracle found a contact at step " << s
                                           << " -- the G1a scene is not contact-free";
        ASSERT_EQ(world.GraspReports()[0].finger_contacts, 0u)
            << "batched world found a contact at step " << s;

        const EnvSnap so = SnapOracle(oracle);
        const EnvSnap sb = SnapBatched(world, 0u);
        const SnapDiff d = SnapMaxDiff(so, sb);
        if (s < 3u || (s % 20u) == 19u || s == kRun - 1u) {
            std::printf("[G1a GATE-A] step %3u: |dq|=%.3e |dqdot|=%.3e |dbase|=%.3e "
                        "|dcup|=%.3e base_z(o=%.5f b=%.5f)\n",
                        s, d.dq, d.dqdot, d.dbase, d.dcup, so.base_pose.position.z,
                        sb.base_pose.position.z);
        }
        ASSERT_TRUE(SnapByteEqual(so, sb))
            << "step " << s << " NOT byte-exact: |dq|=" << d.dq << " |dqdot|=" << d.dqdot
            << " |dbase|=" << d.dbase << " |dcup|=" << d.dcup
            << " -- a floating-base batching defect (replication / base lanes / "
               "IntegrateFloatingBase* / action seam)";
    }

    // ----- DISCRIMINATIVENESS (two frozen robots cannot fake the byte-equality) ----
    const EnvSnap send = SnapOracle(oracle);
    const float base_drop = base_z0 - send.base_pose.position.z;
    double max_leg_dq = 0.0;
    const auto legs = nuka::demo::ResolveLegLinks(sc.h1);
    for (uint32_t l : legs)
        max_leg_dq = std::max(max_leg_dq, std::fabs(double(send.q[l]) - snap0.q[l]));
    const float base_vz_body = send.linkvel[sc.root_link].v[5];  // body-frame lin z.
    std::printf("[G1a GATE-A RESULT] %u steps byte-exact (tol 0). base drop=%.3f m "
                "(expect ~%.3f free-fall) max leg |dq|=%.3f rad base vz=%.3f m/s\n",
                kRun, base_drop, 0.5 * 9.81 * (kRun * kDt) * (kRun * kDt), max_leg_dq,
                base_vz_body);
    EXPECT_GT(base_drop, 0.3f)
        << "the base did NOT fall -- the floating root never integrated (vacuous gate)";
    EXPECT_GT(max_leg_dq, 0.05)
        << "the legs did NOT swing -- the PD drive never reached the joints (vacuous)";
    EXPECT_GT(std::fabs(base_vz_body), 0.5f)
        << "the base carries no speed -- IntegrateFloatingBaseVelocity never ran";
}

// ===========================================================================
// G1a GATE (b)+(c) -- N=8 INDEPENDENCE (mixed per-env actions) + D1.
// Eight envs run MIXED per-env PD target scales (env 0 holds rest, the rest swing
// by distinct amounts -- the H1.2 Gate-2 pattern on the action seam). Each env
// must be BYTE-EXACT vs its OWN N=1 batched run under the same scale; adjacent
// envs must DIFFER; and the full N=8 trajectory must be two-run byte-identical (D1).
// ===========================================================================
TEST(BatchedH1UnionWorld, G1a_FloatingBaseNoContact_N8_IndependenceAndD1) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildFloatingNoContactScene();
    ASSERT_EQ(sc.base_dof, 6u);
    const auto tmpl = MakeUnionTemplate(sc);
    const auto drive = BuildDriveSet(sc.h1);
    for (const DriveLink& d : drive) ASSERT_NE(d.link, kInvalidLink);

    constexpr uint32_t kEnvs = 8u;
    constexpr uint32_t kRun = 60u;
    // MIXED per-env action scales: env 0 holds the rest pose (zero targets), the
    // others swing toward distinctly-scaled stance/arm targets.
    const std::vector<float> scales = {0.0f, 0.6f, 0.75f, 0.9f,
                                       1.0f, 1.15f, 1.3f, 1.45f};

    // ---- (b) PER-ENV INDEPENDENCE ----
    coresident::BatchedUnifiedWorld world(context, tmpl, kEnvs, kGravityZ, kDt);
    for (uint32_t s = 0u; s < kRun; ++s) {
        DrivePdBatched(world, sc, drive, scales);
        world.Step();
        ASSERT_EQ(world.GraspReports()[0].finger_contacts, 0u);
    }
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        coresident::BatchedUnifiedWorld solo(context, tmpl, 1u, kGravityZ, kDt);
        const std::vector<float> solo_scale = {scales[e]};
        for (uint32_t s = 0u; s < kRun; ++s) {
            DrivePdBatched(solo, sc, drive, solo_scale);
            solo.Step();
        }
        EXPECT_TRUE(SnapByteEqual(SnapBatched(world, e), SnapBatched(solo, 0u)))
            << "env " << e << " (N=8, scale " << scales[e]
            << ") NOT byte-exact vs its own N=1 run -- floating-base env-major "
               "cross-contamination";
    }
    for (uint32_t e = 1u; e < kEnvs; ++e) {
        EXPECT_FALSE(SnapByteEqual(SnapBatched(world, e), SnapBatched(world, e - 1u)))
            << "env " << e << " collapsed onto its neighbor (mixed actions ignored)";
    }
    // Non-vacuous: every env's base must have genuinely fallen.
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        const EnvSnap s = SnapBatched(world, e);
        EXPECT_LT(s.base_pose.position.z, sc.host.base_pose[0].position.z - 0.05f)
            << "env " << e << " base never fell";
    }
    std::printf("[G1a GATE-B] N=%u mixed scales: every env byte-exact vs its own N=1 "
                "run; adjacent envs differ; all bases fell\n", kEnvs);

    // ---- (c) D1 two-run byte-identity of the FULL N=8 trajectory ----
    auto run = [&]() {
        coresident::BatchedUnifiedWorld w(context, tmpl, kEnvs, kGravityZ, kDt);
        for (uint32_t s = 0u; s < kRun; ++s) {
            DrivePdBatched(w, sc, drive, scales);
            w.Step();
        }
        std::vector<EnvSnap> out(kEnvs);
        for (uint32_t e = 0u; e < kEnvs; ++e) out[e] = SnapBatched(w, e);
        return out;
    };
    const auto a = run();
    const auto b = run();
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        EXPECT_TRUE(SnapByteEqual(a[e], b[e]))
            << "D1: env " << e << " differed between two identical floating-base runs";
    }
    std::printf("[G1a GATE-C D1] N=%u: two identical rollouts byte-exact (base pose + "
                "q/qdot/linkvel + cup)\n", kEnvs);
}

// ===========================================================================
// G1a GATE (d) -- the FloatingBase root's action columns (0..5) are DEAD.
// Two N=1 worlds run the IDENTICAL leg/arm PD; world B additionally stuffs ALL SIX
// base action columns with 1000.0 every step. The trajectories must stay byte-
// identical: dof_to_link_ maps columns 0..5 to the root link, the torque-drive
// kernel DOES write the garbage into device tau[root] (FloatingBase != Fixed; we
// CONFIRM tau[root]==1000 post-run so the probe provably reached the device), but
// ABA pass-2 skips the FloatingBase root (joint_force forced 0) and nothing else
// reads root tau -- the base is dynamically un-actuated, as the brief requires.
// If this FAILS, base torque LEAKS into the dynamics -- a loud FINDING, not a
// tolerance to widen.
// NOTE (scatter semantics, documented not asserted): all six base columns map to
// the ONE root link slot, so the per-link scatter keeps only the LAST base column
// (index 5); columns 0..4 are dead even before the device. Writing all six makes
// the probe insensitive to that overwrite order.
// ===========================================================================
TEST(BatchedH1UnionWorld, G1a_FloatingBaseNoContact_BaseActionColumnsDead) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildFloatingNoContactScene();
    ASSERT_EQ(sc.base_dof, 6u);
    const auto tmpl = MakeUnionTemplate(sc);
    const auto drive = BuildDriveSet(sc.h1);
    for (const DriveLink& d : drive) ASSERT_NE(d.link, kInvalidLink);

    constexpr float kBaseGarbage = 1000.0f;
    constexpr uint32_t kRun = 80u;
    coresident::BatchedUnifiedWorld world_a(context, tmpl, 1u, kGravityZ, kDt);
    coresident::BatchedUnifiedWorld world_b(context, tmpl, 1u, kGravityZ, kDt);
    const std::vector<float> one_scale = {1.0f};
    const float base_z0 = sc.host.base_pose[0].position.z;

    for (uint32_t s = 0u; s < kRun; ++s) {
        DrivePdBatched(world_a, sc, drive, one_scale, 0.0f);
        DrivePdBatched(world_b, sc, drive, one_scale, kBaseGarbage);
        world_a.Step();
        world_b.Step();
        const EnvSnap sa = SnapBatched(world_a, 0u);
        const EnvSnap sb = SnapBatched(world_b, 0u);
        const SnapDiff d = SnapMaxDiff(sa, sb);
        ASSERT_TRUE(SnapByteEqual(sa, sb))
            << "step " << s << ": NONZERO BASE ACTION COLUMNS CHANGED THE TRAJECTORY "
               "(|dq|=" << d.dq << " |dqdot|=" << d.dqdot << " |dbase|=" << d.dbase
            << ") -- base torque REACHES the floating root: a FINDING to surface, "
               "not to paper over";
    }

    // Prove the garbage genuinely reached the device drive (tau[root] == 1000 on B,
    // 0 on A) -- so the byte-equality above is dynamics-deadness, not a dropped write.
    articulation::ArticulationHostState st_a, st_b;
    world_a.DownloadGripper(0u, &st_a);
    world_b.DownloadGripper(0u, &st_b);
    EXPECT_EQ(st_a.tau[sc.root_link], 0.0f);
    EXPECT_EQ(st_b.tau[sc.root_link], kBaseGarbage)
        << "the base-column garbage never reached device tau[root] -- the dead-column "
           "probe is vacuous (SetActions dropped it before the device)";

    // Non-vacuous: the shared trajectory must be a real falling+swinging one.
    const EnvSnap send = SnapBatched(world_a, 0u);
    EXPECT_LT(send.base_pose.position.z, base_z0 - 0.2f) << "base never fell (vacuous)";
    std::printf("[G1a GATE-D] %u steps: base action columns 0..5 stuffed with %.0f on "
                "world B -> trajectory byte-identical to world A; device tau[root] "
                "(a=%.1f b=%.1f) confirms the probe reached the drive kernel and ABA "
                "ignored it (FloatingBase root has no scalar tau slot)\n",
                kRun, kBaseGarbage, st_a.tau[sc.root_link], st_b.tau[sc.root_link]);
}

// ===========================================================================
// ★ G1b GATE (a) -- N=1 BYTE-EXACT (tol 0) FEET x GROUND union parity, 300 steps,
// feet LOADED. Oracle = UnifiedCoResidentStepper(StandGraspConfig) -- see the TU
// header for why StandGraspConfig (fingerless-in-effect: the 30 finger pairs find
// NOTHING 5m away) and not StandConfig. Per step: identical full-body stance-hold
// PD on both sides from each side's OWN state -> step -> assert (i) the row-class
// invariant (finger_contacts==0 BOTH sides, foot rows >0 BOTH sides -- exactly
// feet x ground), (ii) tol-0 report parity (foot_normal_rows + bit-equal
// foot_normal_impulse_sum), (iii) byte-equal state (base pose + q + qdot + all
// link spatial velocities + the free-falling cup). FORCE BALANCE: the late-window
// mean of Σλ_n/(m*g*dt) must sit near 1 (the feet carry the whole robot,
// quasi-static). tol-0 is the bar: host foot narrowphase on BOTH sides, zero
// finger manifolds (no GPU-vs-host ULP surface), same kernels in the same order.
// ===========================================================================
TEST(BatchedH1UnionWorld, G1b_FeetGroundUnion_N1_ByteExactVsStandGraspOracle) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildFeetGroundScene(context);
    ASSERT_EQ(sc.base_dof, 6u) << "the cook is NOT floating-base";
    ASSERT_EQ(sc.dof_stride, 51u);
    ASSERT_EQ(sc.sg.feet.size(), 4u) << "expected toe/heel x 2 ankles";
    ASSERT_GE(sc.config.fingertips.size(), 30u);
    const double mg_dt = sc.total_mass * 9.81 * kDt;
    std::printf("[G1b GATE-A] feet x ground union: mass=%.2f kg seat_z=%.4f "
                "m*g*dt=%.4f feet=%zu fingertips=%zu cup@+%.1fm (never touched)\n",
                sc.total_mass, sc.seat_pose.position.z, mg_dt, sc.sg.feet.size(),
                sc.config.fingertips.size(), kCupFarOffsetX);

    const auto tmpl = MakeUnionTemplate(sc);
    const auto drive = BuildStanceHoldDriveSet(sc);
    for (const DriveLink& d : drive) ASSERT_NE(d.link, kInvalidLink);

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGravityZ, kDt);
    coresident::UnifiedCoResidentStepper oracle(context, sc.host, sc.sg, kGravityZ, kDt);
    CopCtl cop_o(context, sc, 1u);  // one CoP state per world instance (per side).
    CopCtl cop_b(context, sc, 1u);

    constexpr uint32_t kRun = 300u;
    const std::vector<float> one_scale = {1.0f};
    double ratio_sum = 0.0;
    uint32_t ratio_n = 0u;
    uint32_t min_rows = ~0u, max_rows = 0u;
    for (uint32_t s = 0u; s < kRun; ++s) {
        DrivePdOracle(oracle, sc, drive, 1.0f, &cop_o);
        DrivePdBatched(world, sc, drive, one_scale, 0.0f, &cop_b);
        const auto rep = oracle.Step();
        world.Step();
        const auto& brep = world.GraspReports()[0];

        // (i) the ROW-CLASS invariant: EXACTLY feet x ground, both sides, every step.
        ASSERT_EQ(rep.finger_contacts, 0u)
            << "oracle found a finger contact at step " << s << " -- not feet-only";
        ASSERT_EQ(brep.finger_contacts, 0u)
            << "batched world found a finger contact at step " << s;
        ASSERT_GT(rep.foot_normal_rows, 0u) << "oracle lost foot contact at step " << s;
        ASSERT_GT(brep.foot_normal_rows, 0u)
            << "batched world lost foot contact at step " << s
            << " (foot rows never emitted? has_feet support missing)";

        // (ii) tol-0 REPORT parity (identical rows in identical order -> bit-equal).
        ASSERT_EQ(brep.foot_normal_rows, rep.foot_normal_rows) << "step " << s;
        ASSERT_EQ(brep.foot_normal_impulse_sum, rep.foot_normal_impulse_sum)
            << "step " << s << " foot impulse sum not bit-equal (batched="
            << brep.foot_normal_impulse_sum << " oracle=" << rep.foot_normal_impulse_sum
            << ")";

        min_rows = std::min(min_rows, brep.foot_normal_rows);
        max_rows = std::max(max_rows, brep.foot_normal_rows);
        if (s >= kRun - 100u) {  // the late (settled) force-balance window.
            ratio_sum += brep.foot_normal_impulse_sum / mg_dt;
            ++ratio_n;
        }

        // (iii) tol-0 STATE parity.
        const EnvSnap so = SnapOracle(oracle);
        const EnvSnap sb = SnapBatched(world, 0u);
        const SnapDiff d = SnapMaxDiff(so, sb);
        if (s < 3u || (s % 50u) == 49u || s == kRun - 1u) {
            std::printf("[G1b GATE-A] step %3u: |dq|=%.3e |dqdot|=%.3e |dbase|=%.3e "
                        "rows=%u Σλ/mgdt=%.4f base_z=%.4f\n",
                        s, d.dq, d.dqdot, d.dbase, brep.foot_normal_rows,
                        brep.foot_normal_impulse_sum / mg_dt, so.base_pose.position.z);
        }
        ASSERT_TRUE(SnapByteEqual(so, sb))
            << "step " << s << " NOT byte-exact: |dq|=" << d.dq
            << " |dqdot|=" << d.dqdot << " |dbase|=" << d.dbase << " |dcup|=" << d.dcup
            << " -- a feet-row batching defect (host narrowphase / foot chain-J slot / "
               "base prefix-sum contact lanes / scatter)";
    }

    // ----- FORCE BALANCE: the feet carry the whole robot at the settled stance. ----
    const double mean_ratio = ratio_sum / std::max(1u, ratio_n);
    const EnvSnap send = SnapOracle(oracle);
    std::printf("[G1b GATE-A RESULT] %u steps byte-exact (tol 0). foot rows [%u..%u]; "
                "late-window mean Σλ_n/(m*g*dt)=%.4f (m*g*dt=%.4f); base z %.4f -> %.4f\n",
                kRun, min_rows, max_rows, mean_ratio, mg_dt, sc.seat_pose.position.z,
                send.base_pose.position.z);
    EXPECT_GT(mean_ratio, 0.7) << "feet NOT carrying the robot weight (Σλ too small "
                                  "-- the stance is not genuinely loaded)";
    EXPECT_LT(mean_ratio, 1.3) << "foot impulse far above the weight (non-quasi-static "
                                  "or a contact-model defect)";
    // Discriminative: genuinely SUPPORTED (a no-contact run free-falls ~7.7m in 300
    // steps; the loaded stance must stay near the seat).
    EXPECT_GT(send.base_pose.position.z, sc.seat_pose.position.z - 0.2f)
        << "the base sank/fell -- the feet did not support the robot";
}

// ===========================================================================
// G1b GATE (b)+(c) -- N=8 MIXED independence (incl. a FOOT-AIRBORNE env) + D1.
// Eight envs run per-env stance-target scales; env 6 is seated 0.8 m HIGH via the
// SetGripperBasePose IC seam so its feet NEVER touch within the run (free-fall
// 0.31 m < 0.8 m) -- the env-major TILE-GAP + scatter-guard env with FEET rows
// (brief risk #4): its qdot tile is never packed/solved/scattered while its
// neighbors' are. Each env must be BYTE-EXACT vs its OWN N=1 run (same scale,
// same IC seam); adjacent envs differ; D1 two-run byte-identity over the full
// N=8 trajectory.
// ===========================================================================
TEST(BatchedH1UnionWorld, G1b_FeetGroundUnion_N8_MixedAirborneIndependenceAndD1) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildFeetGroundScene(context);
    ASSERT_EQ(sc.base_dof, 6u);
    const auto tmpl = MakeUnionTemplate(sc);
    const auto drive = BuildStanceHoldDriveSet(sc);
    for (const DriveLink& d : drive) ASSERT_NE(d.link, kInvalidLink);

    constexpr uint32_t kEnvs = 8u;
    constexpr uint32_t kRun = 60u;
    constexpr uint32_t kAirborneEnv = 6u;
    // MIXED per-env stance-target scales (env 0 drives the legs STRAIGHT, the others
    // toward distinctly-scaled bends; env 6 shares scale 1.0 with env 3 but is
    // AIRBORNE -- trajectory-distinct by the IC, not the action).
    const std::vector<float> scales = {0.0f, 0.6f, 0.8f, 1.0f, 1.15f, 1.3f, 1.0f, 1.45f};
    Transform high_pose = sc.seat_pose;
    high_pose.position.z += 0.8f;

    // ---- (b) PER-ENV INDEPENDENCE ----
    coresident::BatchedUnifiedWorld world(context, tmpl, kEnvs, kGravityZ, kDt);
    world.SetGripperBasePose(kAirborneEnv, high_pose);
    CopCtl cop(context, sc, kEnvs);
    uint32_t airborne_rows_total = 0u;
    std::vector<uint32_t> loaded_steps(kEnvs, 0u);
    for (uint32_t s = 0u; s < kRun; ++s) {
        DrivePdBatched(world, sc, drive, scales, 0.0f, &cop);
        world.Step();
        airborne_rows_total += world.GraspReports()[kAirborneEnv].foot_normal_rows;
        for (uint32_t e = 0u; e < kEnvs; ++e) {
            if (world.GraspReports()[e].foot_normal_rows > 0u) ++loaded_steps[e];
        }
    }
    // The MIXED shape is REAL: the airborne env NEVER emitted a foot row (the tile
    // gap), every seated env stayed loaded for >=70% of the run (a hard-driven env --
    // e.g. scale 0.0 straightens the legs -- may briefly HOP and re-land, which itself
    // exercises the row-gap -> re-entry path; sustained-majority load is the honest
    // MIXED-shape bar).
    EXPECT_EQ(airborne_rows_total, 0u)
        << "the 'airborne' env touched the ground -- the tile-gap env is vacuous";
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        if (e == kAirborneEnv) {
            EXPECT_EQ(loaded_steps[e], 0u);
            continue;
        }
        EXPECT_GE(loaded_steps[e], (kRun * 7u) / 10u)
            << "seated env " << e << " (scale " << scales[e]
            << ") was loaded only " << loaded_steps[e] << "/" << kRun << " steps";
    }
    // The airborne env genuinely free-falls (its base carries the drop).
    {
        const EnvSnap sa = SnapBatched(world, kAirborneEnv);
        EXPECT_LT(sa.base_pose.position.z, high_pose.position.z - 0.2f)
            << "airborne env base never fell";
    }

    for (uint32_t e = 0u; e < kEnvs; ++e) {
        coresident::BatchedUnifiedWorld solo(context, tmpl, 1u, kGravityZ, kDt);
        if (e == kAirborneEnv) solo.SetGripperBasePose(0u, high_pose);
        CopCtl solo_cop(context, sc, 1u);
        const std::vector<float> solo_scale = {scales[e]};
        for (uint32_t s = 0u; s < kRun; ++s) {
            DrivePdBatched(solo, sc, drive, solo_scale, 0.0f, &solo_cop);
            solo.Step();
        }
        EXPECT_TRUE(SnapByteEqual(SnapBatched(world, e), SnapBatched(solo, 0u)))
            << "env " << e << " (N=8, scale " << scales[e]
            << (e == kAirborneEnv ? ", AIRBORNE" : "")
            << ") NOT byte-exact vs its own N=1 run -- env-major cross-contamination "
               "through the feet rows / tile gap";
    }
    for (uint32_t e = 1u; e < kEnvs; ++e) {
        EXPECT_FALSE(SnapByteEqual(SnapBatched(world, e), SnapBatched(world, e - 1u)))
            << "env " << e << " collapsed onto its neighbor";
    }
    std::printf("[G1b GATE-B] N=%u MIXED (7 seated + env %u airborne): every env "
                "byte-exact vs its own N=1 run; airborne emitted 0 foot rows; seated "
                "envs sustained contact\n", kEnvs, kAirborneEnv);

    // ---- (c) D1 two-run byte-identity of the FULL N=8 trajectory ----
    auto run = [&]() {
        coresident::BatchedUnifiedWorld w(context, tmpl, kEnvs, kGravityZ, kDt);
        w.SetGripperBasePose(kAirborneEnv, high_pose);
        CopCtl run_cop(context, sc, kEnvs);
        for (uint32_t s = 0u; s < kRun; ++s) {
            DrivePdBatched(w, sc, drive, scales, 0.0f, &run_cop);
            w.Step();
        }
        std::vector<EnvSnap> out(kEnvs);
        for (uint32_t e = 0u; e < kEnvs; ++e) out[e] = SnapBatched(w, e);
        return out;
    };
    const auto a = run();
    const auto b = run();
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        EXPECT_TRUE(SnapByteEqual(a[e], b[e]))
            << "D1: env " << e << " differed between two identical feet-loaded runs";
    }
    std::printf("[G1b GATE-C D1] N=%u: two identical feet x ground rollouts byte-exact\n",
                kEnvs);
}

// ===========================================================================
// G1b GATE (d) -- HONESTY/BITE: kill the leg drive mid-run -> the robot COLLAPSES.
// 120 driven steps hold the bent stance near the seat (feet loaded), then 240
// steps with ALL-ZERO actions: feet contact alone CANNOT hold the bent stance
// (the knees buckle, the unactuated base falls). Asserts the base z drops far
// below the driven-phase level. This rejects any silently-welded-legs fake: a
// weld would hold the stance with zero drive.
// ===========================================================================
TEST(BatchedH1UnionWorld, G1b_FeetGroundUnion_DriveKillCollapse_BITE) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildFeetGroundScene(context);
    const auto tmpl = MakeUnionTemplate(sc);
    const auto drive = BuildStanceHoldDriveSet(sc);
    for (const DriveLink& d : drive) ASSERT_NE(d.link, kInvalidLink);

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGravityZ, kDt);
    CopCtl cop(context, sc, 1u);
    const std::vector<float> one_scale = {1.0f};

    // ----- phase 1: DRIVEN -- the stance holds near the seat, feet loaded. --------
    constexpr uint32_t kDriven = 120u;
    for (uint32_t s = 0u; s < kDriven; ++s) {
        DrivePdBatched(world, sc, drive, one_scale, 0.0f, &cop);
        world.Step();
        ASSERT_GT(world.GraspReports()[0].foot_normal_rows, 0u)
            << "feet unloaded during the driven phase at step " << s;
    }
    const float z_driven = SnapBatched(world, 0u).base_pose.position.z;
    EXPECT_GT(z_driven, sc.seat_pose.position.z - 0.15f)
        << "the DRIVEN stance already collapsed -- the BITE has no baseline";

    // ----- phase 2: DRIVE KILLED -- all-zero actions; the stance must buckle. -----
    constexpr uint32_t kDead = 240u;
    std::vector<float> zeros(sc.dof_stride, 0.0f);
    for (uint32_t s = 0u; s < kDead; ++s) {
        world.SetActions(zeros.data(), zeros.size());
        world.Step();
    }
    const float z_dead = SnapBatched(world, 0u).base_pose.position.z;
    std::printf("[G1b GATE-D BITE] seat z=%.4f driven(%u)=%.4f -> drive killed(%u)="
                "%.4f (drop %.3f m)\n",
                sc.seat_pose.position.z, kDriven, z_driven, kDead, z_dead,
                z_driven - z_dead);
    EXPECT_LT(z_dead, z_driven - 0.25f)
        << "the robot did NOT collapse when the drive was killed -- the bent stance "
           "held itself up: silently-welded legs or faked foot support (a FINDING)";
}

// ===========================================================================
// ★ G1c GATE (a) -- N=1 COMPOSED union parity (feet x ground AND finger x cup in
// ONE solve), 300 steps, feet loaded AND fingers gripping the whole run.
// ---------------------------------------------------------------------------
// THE HONEST BAR (the H1.2 reformulated bar, an EVIDENCED reformulation -- NOT a
// relaxation to hide a bug; spec §G1(a) allows <=1e-5 with a written justification,
// this is it). G1b held tol-0 because the feet path has ZERO divergence surface
// (host narrowphase on BOTH sides, zero finger manifolds). G1c brings the cup into
// the hand, which REOPENS the two H1.2 surfaces simultaneously:
//   (1) the batched finger narrowphase is the GPU SphereHull launcher while the
//       oracle narrowphases on HOST -- a documented ULP-scale float seam, and
//   (2) the 30-contact condim=3 grasp on 12 SHARED links is CHAOTIC at the FP
//       floor (H1.2's perturbation control PROVED a 1e-7 m IC seed in IDENTICAL
//       code amplifies past 1e-4 within ~70 steps).
// Both classes feed the SAME UnifiedSolve here, so the ULP seam perturbs the whole
// union state (base, legs, foot lambdas included) -- full-run byte-parity is
// impossible for ANY two code paths, correct or not. The IRREDUCIBLE FP-floor
// window is the LEADING steps before the seed amplifies. So the HARD assert =
// the first kFloorSteps at the FP floor (cup pos/vel <=1e-6, gripper qdot <=5e-6,
// base pos <=1e-6) AND the contact-set COUNTS (finger points + foot rows) matching
// through the window -- a real STRUCTURAL-bug discriminator: a composition bug
// (wrong row order, wrong friction stamp, wrong body key, qdot pack/scatter
// corruption with both classes live) corrupts step 0 by orders of magnitude
// and/or forks the contact set immediately. Steps beyond the window are REPORTED
// (chaotic), and the CHAOS CONTROL below proves the full-run delta is the scene,
// not the batched path. If tol-0 happens to hold it would be asserted instead --
// it does NOT (measured: the GPU-vs-host NP seam is real), hence this bar.
// COMPOSITION asserts (the point of G1c): every step has foot rows AND finger
// contacts SIMULTANEOUSLY on both sides; the two force balances close --
//   feet:    Σλ_foot_normal ~= (m_robot + m_cup)*g*dt (the robot carries the cup),
//   fingers: Σλ_finger*j_z  ~= m_cup*g*dt             (the hand carries the cup)
// -- the H1.1/C7b-1 balance pattern, late-window means.
// ===========================================================================
TEST(BatchedH1UnionWorld, G1c_GraspStandUnion_N1_FpFloorWindowVsStandGraspOracle) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildGraspStandScene(context);
    ASSERT_EQ(sc.base_dof, 6u) << "the cook is NOT floating-base";
    ASSERT_EQ(sc.dof_stride, 51u);
    ASSERT_EQ(sc.sg.feet.size(), 4u);
    ASSERT_GE(sc.config.fingertips.size(), 30u);
    ASSERT_EQ(sc.grip_links.size(), 12u) << "a wrap-driven link failed to resolve";
    ASSERT_TRUE(sc.place.found)
        << "the 10.9cm finger-only shallow caging placement vanished on the floating "
           "stance cook -- the G1c grasp pose does not exist";
    const double mg_dt_feet = (sc.total_mass + kCupMass) * 9.81 * kDt;
    const double mg_dt_cup = kCupMass * 9.81 * kDt;
    std::printf("[G1c GATE-A] grasp-while-standing union: m_robot=%.2f kg m_cup=%.2f kg "
                "cup@(%.3f,%.3f,%.3f) arc=%.0fdeg genuine=%u pen=%.4f feet=%zu "
                "fingertips=%zu (m_r+m_c)*g*dt=%.4f m_c*g*dt=%.5f\n",
                sc.total_mass, kCupMass, sc.place.center.x, sc.place.center.y,
                sc.place.center.z, sc.place.covered_arc, sc.place.genuine,
                sc.place.max_pen, sc.sg.feet.size(), sc.config.fingertips.size(),
                mg_dt_feet, mg_dt_cup);

    const auto tmpl = MakeUnionTemplate(sc);
    const auto drive = BuildGraspStanceDriveSet(sc);
    for (const DriveLink& d : drive) ASSERT_NE(d.link, kInvalidLink);

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGravityZ, kDt);
    coresident::UnifiedCoResidentStepper oracle(context, sc.host, sc.sg, kGravityZ, kDt);
    CopCtl cop_o(context, sc, 1u);
    CopCtl cop_b(context, sc, 1u);

    constexpr uint32_t kRun = 300u;
    // The H1.2 FP-floor window bar (the evidenced reformulation; see the test header).
    constexpr uint32_t kFloorSteps = 10u;
    constexpr double kFloorTol = 1.0e-6;      // cup pos/vel + base pos in-window.
    constexpr double kFloorTolQdot = 5.0e-6;  // gripper qdot (51 DOF) in-window.
    const std::vector<float> one_scale = {1.0f};
    double floor_max_dpos = 0.0, floor_max_dvel = 0.0, floor_max_dqdot = 0.0,
           floor_max_dbase = 0.0;
    double full_max_dpos = 0.0, full_max_dvel = 0.0, full_max_dqdot = 0.0;
    uint32_t count_match_onset = kRun;  // first step a contact COUNT forks (chaos marker).
    bool count_open = true;
    uint32_t both_steps = 0u;           // steps with feet AND fingers loaded (batched).
    double feet_ratio_sum = 0.0, cup_ratio_sum = 0.0;
    uint32_t ratio_n = 0u;
    for (uint32_t s = 0u; s < kRun; ++s) {
        DrivePdOracle(oracle, sc, drive, 1.0f, &cop_o);
        DrivePdBatched(world, sc, drive, one_scale, 0.0f, &cop_b);
        const auto rep = oracle.Step();
        world.Step();
        const auto& brep = world.GraspReports()[0];

        // ----- (i) the COMPOSITION invariant: feet rows AND finger contacts exist
        // SIMULTANEOUSLY, every step, on BOTH sides (the union is genuinely composed,
        // not two disjoint regimes). -----
        ASSERT_GT(rep.foot_normal_rows, 0u) << "oracle lost foot contact at step " << s;
        ASSERT_GT(brep.foot_normal_rows, 0u)
            << "batched world lost foot contact at step " << s;
        ASSERT_GT(rep.finger_contacts, 0u)
            << "oracle lost the cup at step " << s << " -- the grasp did not hold";
        ASSERT_GT(brep.finger_contacts, 0u)
            << "batched world lost the cup at step " << s;
        if (brep.foot_normal_rows > 0u && brep.finger_contacts > 0u) ++both_steps;

        // ----- (ii) contact-set COUNT parity through the FP-floor window (asserted
        // after the loop via count_match_onset; the fork step is REPORTED). -----
        if (count_open && (rep.finger_contacts != brep.finger_contacts ||
                           rep.foot_normal_rows != brep.foot_normal_rows)) {
            count_open = false;
            count_match_onset = s;
        }

        // ----- (iii) state deltas: HARD in-window, REPORTED full-run. -----
        const EnvSnap so = SnapOracle(oracle);
        const EnvSnap sb = SnapBatched(world, 0u);
        const SnapDiff d = SnapMaxDiff(so, sb);
        const double dvel =
            (so.cup.linear_velocity - sb.cup.linear_velocity).Length();
        full_max_dpos = std::max(full_max_dpos, d.dcup);
        full_max_dvel = std::max(full_max_dvel, dvel);
        full_max_dqdot = std::max(full_max_dqdot, d.dqdot);
        if (s < kFloorSteps) {
            floor_max_dpos = std::max(floor_max_dpos, d.dcup);
            floor_max_dvel = std::max(floor_max_dvel, dvel);
            floor_max_dqdot = std::max(floor_max_dqdot, d.dqdot);
            floor_max_dbase = std::max(floor_max_dbase, d.dbase);
        }

        // ----- (iv) the two force balances (late settled window, batched side). -----
        if (s >= kRun - 100u) {
            feet_ratio_sum += brep.foot_normal_impulse_sum / mg_dt_feet;
            cup_ratio_sum += brep.cup_vertical_impulse / mg_dt_cup;
            ++ratio_n;
        }
        if (s <= 14u || (s % 50u) == 49u || s == kRun - 1u) {
            std::printf("[G1c GATE-A] step %3u: |dcup|=%.3e |dvel|=%.3e |dqdot|=%.3e "
                        "|dbase|=%.3e contacts(o=%u b=%u) feet(o=%u b=%u) "
                        "Σλf/mgdt=%.3f Σλc/mgdt=%.3f cup_z=%.4f base_z=%.4f\n",
                        s, d.dcup, dvel, d.dqdot, d.dbase, rep.finger_contacts,
                        brep.finger_contacts, rep.foot_normal_rows,
                        brep.foot_normal_rows,
                        brep.foot_normal_impulse_sum / mg_dt_feet,
                        brep.cup_vertical_impulse / mg_dt_cup, sb.cup.position.z,
                        sb.base_pose.position.z);
        }
    }

    const double feet_ratio = feet_ratio_sum / std::max(1u, ratio_n);
    const double cup_ratio = cup_ratio_sum / std::max(1u, ratio_n);
    const EnvSnap send = SnapBatched(world, 0u);
    std::printf("[G1c GATE-A RESULT] FP-FLOOR WINDOW (steps 0-%u): max|dcup|=%.3e "
                "max|dvel|=%.3e max|dqdot|=%.3e max|dbase|=%.3e (HARD tol pos/vel/base "
                "%.0e qdot %.0e). FULL RUN (%u steps): max|dcup|=%.3e max|dvel|=%.3e "
                "max|dqdot|=%.3e (REPORTED, chaotic). contact-count fork @ step %u. "
                "both-classes %u/%u steps. late-window Σλ_feet/((m_r+m_c)g dt)=%.4f "
                "Σλ_cup/(m_c g dt)=%.4f. cup_z %.4f->%.4f base_z %.4f->%.4f\n",
                kFloorSteps - 1u, floor_max_dpos, floor_max_dvel, floor_max_dqdot,
                floor_max_dbase, kFloorTol, kFloorTolQdot, kRun, full_max_dpos,
                full_max_dvel, full_max_dqdot, count_match_onset, both_steps, kRun,
                feet_ratio, cup_ratio, sc.cup0.position.z, send.cup.position.z,
                sc.seat_pose.position.z, send.base_pose.position.z);

    // ★ THE HEADLINE (HARD): the FP-floor window. A structural composition bug
    // (row order, friction stamp, body key, pack/scatter with both classes live)
    // corrupts step 0 by ~1e-3+ and/or forks the contact set immediately -- this
    // window catches it; Gate (b)'s tol-0 batching proof is BLIND to N-independent
    // resolver bugs, this window is not.
    EXPECT_LE(floor_max_dpos, kFloorTol)
        << "batched cup POSITION diverged in the FP-floor window -- a composition "
           "defect in the union row assembly (not the NP seam)";
    EXPECT_LE(floor_max_dvel, kFloorTol)
        << "batched cup VELOCITY diverged in the FP-floor window";
    EXPECT_LE(floor_max_dqdot, kFloorTolQdot)
        << "batched gripper qdot diverged in the FP-floor window";
    EXPECT_LE(floor_max_dbase, kFloorTol)
        << "batched BASE position diverged in the FP-floor window -- the base "
           "prefix-sum lanes corrupt under the composed row classes";
    EXPECT_GE(count_match_onset, kFloorSteps)
        << "the contact-set COUNT forked INSIDE the FP-floor window -- a structural "
           "composition defect, not the NP seam";

    // The COMPOSITION was total: every step carried both row classes.
    EXPECT_EQ(both_steps, kRun);

    // The two force balances (the robot carries the cup; the hand carries the cup).
    EXPECT_GT(feet_ratio, 0.7) << "feet NOT carrying robot+cup";
    EXPECT_LT(feet_ratio, 1.3) << "foot impulse far above the robot+cup weight";
    EXPECT_GT(cup_ratio, 0.7) << "fingers NOT carrying the cup weight (no honest hold)";
    EXPECT_LT(cup_ratio, 1.3) << "finger vertical impulse far above the cup weight";

    // Non-vacuous: genuinely standing AND genuinely holding (a dropped cup falls
    // ~7.7 m in 300 steps; an unsupported robot likewise).
    EXPECT_GT(send.base_pose.position.z, sc.seat_pose.position.z - 0.2f)
        << "the base sank -- the feet did not support the robot";
    EXPECT_GT(send.cup.position.z, sc.cup0.position.z - 0.25f)
        << "the cup fell -- the hand did not hold it";

    // ------------------------------------------------------------------------------
    // THE CHAOS CONTROL (proves the full-run divergence is the SCENE, not the batched
    // path). Two SAME-CODE oracles on the IDENTICAL scene, one with the cup IC nudged
    // 1e-7 m in x, each with its OWN CopCtl. If the 1e-7 seed amplifies to the SAME
    // order as the batched-vs-oracle full-run delta, full-run byte-parity is
    // impossible for ANY two code paths, correct or not -- the cross-path delta is
    // explained by scene chaos, not a batched defect.
    // ------------------------------------------------------------------------------
    coresident::StandGraspConfig sg_a = sc.sg;
    coresident::StandGraspConfig sg_b = sc.sg;
    sg_b.cup_state.position.x += 1.0e-7f;
    coresident::UnifiedCoResidentStepper o_a(context, sc.host, sg_a, kGravityZ, kDt);
    coresident::UnifiedCoResidentStepper o_b(context, sc.host, sg_b, kGravityZ, kDt);
    CopCtl cop_a2(context, sc, 1u);
    CopCtl cop_b2(context, sc, 1u);
    double self_max_dpos = 0.0, self_max_dvel = 0.0;
    for (uint32_t s = 0u; s < kRun; ++s) {
        DrivePdOracle(o_a, sc, drive, 1.0f, &cop_a2);
        DrivePdOracle(o_b, sc, drive, 1.0f, &cop_b2);
        o_a.Step();
        o_b.Step();
        const double dpos = (o_a.Cup().position - o_b.Cup().position).Length();
        const double dvel =
            (o_a.Cup().linear_velocity - o_b.Cup().linear_velocity).Length();
        self_max_dpos = std::max(self_max_dpos, dpos);
        self_max_dvel = std::max(self_max_dvel, dvel);
    }
    std::printf("[G1c GATE-A CHAOS CONTROL] two SAME-code oracles, cup IC nudged "
                "1e-7 m: self max|dpos|=%.3e max|dvel|=%.3e over %u steps (vs batched-"
                "vs-oracle full-run max|dpos|=%.3e)\n",
                self_max_dpos, self_max_dvel, kRun, full_max_dpos);
    EXPECT_GT(self_max_dpos, 1.0e-4)
        << "a 1e-7 m IC seed did NOT amplify -- the scene is NOT chaotic, so the "
           "batched-vs-oracle full-run divergence is unexplained (investigate)";
    EXPECT_LT(full_max_dpos, 10.0 * self_max_dpos)
        << "the batched-vs-oracle delta is >10x a 1e-7 m IC nudge's amplification -- "
           "larger than scene chaos explains (investigate the batched path)";
}

// ===========================================================================
// G1c GATE (b)+(c) -- N=8 MIXED independence + D1, with BOTH gap kinds (brief
// risk #4): (i) env 6 FOOT-AIRBORNE (seated +0.8m via SetGripperBasePose, cup
// raised with it via BodyMut -> FINGER rows only, NO foot rows -- the foot-class
// tile gap while the finger class is live); (ii) env 2 CUP-DROPPED (grip torque
// ZERO from step 0 -> the cup falls free of the open hand -> foot rows only, no
// finger rows after separation -- the finger-class gap while the foot class is
// live). The remaining envs carry BOTH classes at per-env distinct stance scales.
// Each env must be BYTE-EXACT vs its OWN N=1 run (same IC + grip seams); adjacent
// envs differ; D1 two-run byte-identity over the full N=8 trajectory.
// ===========================================================================
TEST(BatchedH1UnionWorld, G1c_GraspStandUnion_N8_MixedDroppedAirborneIndependenceAndD1) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildGraspStandScene(context);
    ASSERT_EQ(sc.base_dof, 6u);
    ASSERT_TRUE(sc.place.found);
    const auto tmpl = MakeUnionTemplate(sc);
    const auto drive = BuildGraspStanceDriveSet(sc);
    for (const DriveLink& d : drive) ASSERT_NE(d.link, kInvalidLink);

    constexpr uint32_t kEnvs = 8u;
    constexpr uint32_t kRun = 80u;
    constexpr uint32_t kDroppedEnv = 2u;   // grip killed from step 0 -> cup falls.
    constexpr uint32_t kAirborneEnv = 6u;  // base+cup seated 0.8m high -> no foot rows.
    // Per-env stance-target scales (grip targets are NOT scaled; envs 3 and 6 share
    // scale 1.0 -- env 6 is distinct by its IC, the G1b pattern).
    const std::vector<float> scales = {0.0f, 0.6f, 0.8f, 1.0f, 1.15f, 1.3f, 1.0f, 1.45f};
    std::vector<uint8_t> grip_on(kEnvs, 1u);
    grip_on[kDroppedEnv] = 0u;
    Transform high_pose = sc.seat_pose;
    high_pose.position.z += 0.8f;

    // ---- (b) PER-ENV INDEPENDENCE ----
    coresident::BatchedUnifiedWorld world(context, tmpl, kEnvs, kGravityZ, kDt);
    world.SetGripperBasePose(kAirborneEnv, high_pose);
    world.BodyMut(kAirborneEnv, 0u).position.z += 0.8f;  // the cup rides up with the hand.
    CopCtl cop(context, sc, kEnvs);
    uint32_t airborne_foot_rows = 0u;
    std::vector<uint32_t> foot_steps(kEnvs, 0u), finger_steps(kEnvs, 0u);
    uint32_t dropped_late_contacts = 0u;  // finger contacts in the dropped env's last 15.
    for (uint32_t s = 0u; s < kRun; ++s) {
        DrivePdBatched(world, sc, drive, scales, 0.0f, &cop, &grip_on);
        world.Step();
        for (uint32_t e = 0u; e < kEnvs; ++e) {
            const auto& r = world.GraspReports()[e];
            if (r.foot_normal_rows > 0u) ++foot_steps[e];
            if (r.finger_contacts > 0u) ++finger_steps[e];
        }
        airborne_foot_rows += world.GraspReports()[kAirborneEnv].foot_normal_rows;
        if (s >= kRun - 15u)
            dropped_late_contacts += world.GraspReports()[kDroppedEnv].finger_contacts;
    }
    // The MIXED shape is REAL -- both gap kinds present:
    //   * airborne env: ZERO foot rows the whole run (foot-class tile gap), finger
    //     rows LIVE for the sustained majority (the hand holds the cup in free-fall);
    //   * dropped env: feet loaded the sustained majority, finger contacts GONE by
    //     the end (the cup separated and free-falls);
    //   * every other env: BOTH classes for the sustained majority.
    EXPECT_EQ(airborne_foot_rows, 0u)
        << "the 'airborne' env touched the ground -- the foot-class gap is vacuous";
    EXPECT_GE(finger_steps[kAirborneEnv], (kRun * 7u) / 10u)
        << "the airborne env's hand lost the cup -- the finger-only shape is vacuous";
    EXPECT_GE(foot_steps[kDroppedEnv], (kRun * 7u) / 10u)
        << "the dropped-cup env's feet unloaded";
    EXPECT_EQ(dropped_late_contacts, 0u)
        << "the dropped env still has finger contacts at the end -- the cup never "
           "separated (the grip kill is not reaching the drive)";
    {
        const EnvSnap sd = SnapBatched(world, kDroppedEnv);
        EXPECT_LT(sd.cup.position.z, sc.cup0.position.z - 0.10f)
            << "the dropped env's cup never fell";
    }
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        if (e == kDroppedEnv || e == kAirborneEnv) continue;
        EXPECT_GE(foot_steps[e], (kRun * 7u) / 10u)
            << "env " << e << " (scale " << scales[e] << ") feet unloaded";
        // The VIGOROUS per-env stance scales (the independence point: 0.0 straightens
        // the legs, 1.45 over-bends) shake the marginal force-closure transiently at
        // some scales -- MEASURED finger persistence across the both-classes envs:
        // 80/68/80/80/53/80 of 80 (scales 0.0/0.6/1.0/1.15/1.3/1.45). The honest
        // non-vacuity guard is a SUSTAINED MAJORITY (>50%) of the run, mirroring
        // G1b's hop-and-reland allowance for the hard-driven feet; the byte-exact
        // vs-own-N=1 check below is the REAL gate and is unconditional.
        EXPECT_GE(finger_steps[e], kRun / 2u)
            << "env " << e << " (scale " << scales[e] << ") lost the cup";
    }

    for (uint32_t e = 0u; e < kEnvs; ++e) {
        coresident::BatchedUnifiedWorld solo(context, tmpl, 1u, kGravityZ, kDt);
        if (e == kAirborneEnv) {
            solo.SetGripperBasePose(0u, high_pose);
            solo.BodyMut(0u, 0u).position.z += 0.8f;
        }
        CopCtl solo_cop(context, sc, 1u);
        const std::vector<float> solo_scale = {scales[e]};
        const std::vector<uint8_t> solo_grip = {grip_on[e]};
        for (uint32_t s = 0u; s < kRun; ++s) {
            DrivePdBatched(solo, sc, drive, solo_scale, 0.0f, &solo_cop, &solo_grip);
            solo.Step();
        }
        EXPECT_TRUE(SnapByteEqual(SnapBatched(world, e), SnapBatched(solo, 0u)))
            << "env " << e << " (N=8, scale " << scales[e]
            << (e == kAirborneEnv ? ", AIRBORNE" : "")
            << (e == kDroppedEnv ? ", CUP-DROPPED" : "")
            << ") NOT byte-exact vs its own N=1 run -- env-major cross-contamination "
               "through the COMPOSED union rows / per-class tile gaps";
    }
    for (uint32_t e = 1u; e < kEnvs; ++e) {
        EXPECT_FALSE(SnapByteEqual(SnapBatched(world, e), SnapBatched(world, e - 1u)))
            << "env " << e << " collapsed onto its neighbor";
    }
    std::printf("[G1c GATE-B] N=%u MIXED (env %u CUP-DROPPED, env %u AIRBORNE/finger-"
                "only, 6 both-classes): every env byte-exact vs its own N=1 run; "
                "foot_steps/finger_steps per env:", kEnvs, kDroppedEnv, kAirborneEnv);
    for (uint32_t e = 0u; e < kEnvs; ++e)
        std::printf(" %u/%u", foot_steps[e], finger_steps[e]);
    std::printf(" (of %u)\n", kRun);

    // ---- (c) D1 two-run byte-identity of the FULL N=8 trajectory ----
    auto run = [&]() {
        coresident::BatchedUnifiedWorld w(context, tmpl, kEnvs, kGravityZ, kDt);
        w.SetGripperBasePose(kAirborneEnv, high_pose);
        w.BodyMut(kAirborneEnv, 0u).position.z += 0.8f;
        CopCtl run_cop(context, sc, kEnvs);
        for (uint32_t s = 0u; s < kRun; ++s) {
            DrivePdBatched(w, sc, drive, scales, 0.0f, &run_cop, &grip_on);
            w.Step();
        }
        std::vector<EnvSnap> out(kEnvs);
        for (uint32_t e = 0u; e < kEnvs; ++e) out[e] = SnapBatched(w, e);
        return out;
    };
    const auto a = run();
    const auto b = run();
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        EXPECT_TRUE(SnapByteEqual(a[e], b[e]))
            << "D1: env " << e << " differed between two identical composed-union runs";
    }
    std::printf("[G1c GATE-C D1] N=%u: two identical composed-union rollouts (incl. "
                "dropped-cup + airborne envs) byte-exact\n", kEnvs);
}

// ===========================================================================
// G1c GATE (d) -- HONESTY/BITE (spec §3, the STANDING eval): kill the GRIP mid-
// run while the stand drive continues -> the cup FALLS AWAY from the still-
// standing robot. Phase 1 (120 steps): full drive, feet loaded + cup held.
// Phase 2 (240 steps): grip torque -> 0 (the stance-hold + CoP law keep running)
// -> the cup must drop far below the (still-up) hand, finger contacts -> 0,
// while the base stays at stance height. Friction-only hold, no parenting/
// scripting: a parented/scripted cup CANNOT fall while the hand stays up.
// ===========================================================================
TEST(BatchedH1UnionWorld, G1c_GraspStandUnion_MidRunGripKill_CupFallsRobotStands_BITE) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildGraspStandScene(context);
    ASSERT_TRUE(sc.place.found);
    const auto tmpl = MakeUnionTemplate(sc);
    const auto drive = BuildGraspStanceDriveSet(sc);
    for (const DriveLink& d : drive) ASSERT_NE(d.link, kInvalidLink);

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGravityZ, kDt);
    CopCtl cop(context, sc, 1u);
    const std::vector<float> one_scale = {1.0f};

    // ----- phase 1: FULL DRIVE -- standing AND holding. ---------------------------
    constexpr uint32_t kHeld = 120u;
    for (uint32_t s = 0u; s < kHeld; ++s) {
        DrivePdBatched(world, sc, drive, one_scale, 0.0f, &cop);
        world.Step();
        ASSERT_GT(world.GraspReports()[0].foot_normal_rows, 0u)
            << "feet unloaded during the held phase at step " << s;
        ASSERT_GT(world.GraspReports()[0].finger_contacts, 0u)
            << "the cup left the hand during the held phase at step " << s;
    }
    const EnvSnap held = SnapBatched(world, 0u);
    const float cup_z_held = held.cup.position.z;
    const float base_z_held = held.base_pose.position.z;

    // ----- phase 2: GRIP KILLED, stand drive continues. ---------------------------
    constexpr uint32_t kDead = 240u;
    const std::vector<uint8_t> grip_off = {0u};
    uint32_t late_finger_contacts = 0u;
    for (uint32_t s = 0u; s < kDead; ++s) {
        DrivePdBatched(world, sc, drive, one_scale, 0.0f, &cop, &grip_off);
        world.Step();
        if (s >= kDead - 50u)
            late_finger_contacts += world.GraspReports()[0].finger_contacts;
    }
    const EnvSnap dead = SnapBatched(world, 0u);

    // The hand is still UP: read the live fingertip world z's (ExportObsState) and
    // require the cup far below the LOWEST fingertip -- "falls away from the hand",
    // not "hand and cup fell together".
    coresident::ObsStateBatch obs;
    world.ExportObsState(obs);
    float min_tip_z = std::numeric_limits<float>::infinity();
    for (size_t f = 0u; f * 3u + 2u < obs.fingertip_world_pos.size(); ++f)
        min_tip_z = std::min(min_tip_z, obs.fingertip_world_pos[f * 3u + 2u]);

    std::printf("[G1c GATE-D BITE] held(%u): cup_z=%.4f base_z=%.4f -> grip killed"
                "(%u): cup_z=%.4f (drop %.3f m) base_z=%.4f min_fingertip_z=%.4f "
                "late finger contacts=%u\n",
                kHeld, cup_z_held, base_z_held, kDead, dead.cup.position.z,
                cup_z_held - dead.cup.position.z, dead.base_pose.position.z, min_tip_z,
                late_finger_contacts);
    EXPECT_LT(dead.cup.position.z, cup_z_held - 0.5f)
        << "the cup did NOT fall when the grip was killed -- the hold is not "
           "friction-only (parenting/scripting/contact fakery: a FINDING)";
    EXPECT_LT(dead.cup.position.z, min_tip_z - 0.4f)
        << "the cup did not fall AWAY from the hand";
    EXPECT_GT(dead.base_pose.position.z, sc.seat_pose.position.z - 0.2f)
        << "the robot fell with the cup -- the stand did not survive the grip kill";
    EXPECT_EQ(late_finger_contacts, 0u)
        << "finger contacts persist long after the grip kill -- the cup is stuck "
           "to the hand";
}

// ===========================================================================
// ★ G1d GATE (a) -- N=1 COMPLETE union parity (feet x ground + finger x cup +
// cup-proxy x TABLE in ONE solve) vs the FULL StandGraspConfig oracle
// (has_table=true), 300 steps, the LIFT CHOREOGRAPHY:
//   WINDOW (steps 0..9):    fingers HOLD the curl (offset 0) -- the quasi-
//                           static seeded regime the FP-floor parity window is
//                           measured in (all THREE classes live: 30 finger
//                           contacts + 4 feet + 4..2 table rows, both sides);
//   REST  (steps 10..99):   fingers BACK OFF the curl (offset -0.10); the
//                           wedge releases, the cup re-lands and RESTS -- the
//                           TABLE alone carries it (late-rest window asserted);
//   CLOSE (steps 100..179): the H1.1 grip PD (close_offset=0.18) sweeps the
//                           fingers back in + squeezes the wrap into the hold;
//   LIFT  (steps 180..299): SetTableEnabled(false) on BOTH sides at step 180
//                           -- the table pair vanishes; the cup is held by
//                           finger friction force-closure ALONE.
// TWO MEASURED G1d scene findings shaped this schedule (BOTH sides identically
// affected -- scene physics, not batching defects):
//   (1) the CURL WEDGE: holding the bare curl pre-pose, the <=2mm-shallow wrap
//       contacts squeeze the resting cup UP-and-OUT (finger vertical impulse
//       NEGATIVE ~-1..-9 m*g*dt, table over-reacting, cup climbing ~0.6 mm/
//       step; table rows 4->3->2->1->0 by ~step 27 -- a watermelon-seed
//       squeeze, not a rest). The fingers must BACK OFF for an honest rest --
//       and the wedge still launches the cup BEFORE the slow grip PD (Kp=4)
//       clears, so the cup pops a few mm off the table, falls back once the
//       wrap opens, and settles -- the mid-REST table contact is therefore
//       TRANSIENT (reported), and the table-liveness HARD asserts live in the
//       seeded WINDOW + the settled late-REST window.
//   (2) the moving-finger back-off SLIDES the wrap contacts through the parity
//       window and amplifies the GPU-vs-host NP seam ~10x faster (|dvel| 7e-6
//       by step 5) -- so the WINDOW phase holds the curl (offset 0), the exact
//       quasi-static regime G1c's FP-floor bar was measured in (|dvel|<=6e-7,
//       |dqdot|<=2.9e-6 there), and the back-off starts AFTER the window.
// ---------------------------------------------------------------------------
// THE PARITY BAR == G1c's (the H1.2 reformulated bar; spec §G1(a) written
// justification, carried verbatim): the finger GPU-vs-host SphereHull NP seam
// is LIVE from step 0 (the resting cup sits IN the wrap) and the 30-contact
// condim=3 grasp on 12 shared links is CHAOTIC at the FP floor (H1.2/G1c
// measured + chaos-controlled: a 1e-7 m IC seed in IDENTICAL code amplifies
// past 1e-4) -- full-run tol-0 is impossible for ANY two code paths. The TABLE
// rows themselves add NO new seam (host C3b BoxPlane on BOTH sides), so the
// strongest honest bar is UNCHANGED: HARD FP-floor window (cup pos/vel/base
// <=1e-6, gripper qdot <=5e-6) + contact-COUNT parity through the window (now
// including the TABLE row count) + the chaos control re-run through the SAME
// choreography + full-run deltas REPORTED. If tol-0 held it would be asserted
// instead -- it does not (the finger NP seam, measured in G1c).
// THE IMPULSE CHOREOGRAPHY (HARD -- the gate's physics, oracle §1.2 fields):
//   * resting (late-rest window): table_vertical_impulse ~= m_cup*g*dt (the
//     table carries the cup);
//   * after the toggle: table rows + table impulse EXACTLY 0 (no ghost rows);
//   * the triangle closes: Σλ_finger*j_z rises to ~= m_cup*g*dt (late window);
//   * the feet carry robot+cup throughout (the G1c balance, late window).
// All THREE union row classes -- (ChainJ,StaticNull) feet, (ChainJ,RigidInvMass)
// fingers, (RigidInvMass,StaticNull) cup x table -- live in ONE run.
// ===========================================================================
TEST(BatchedH1UnionWorld, G1d_TableLiftUnion_N1_FpFloorWindowAndImpulseTriangleVsOracle) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildGraspStandTableScene(context);
    ASSERT_EQ(sc.base_dof, 6u) << "the cook is NOT floating-base";
    ASSERT_EQ(sc.dof_stride, 51u);
    ASSERT_EQ(sc.sg.feet.size(), 4u);
    ASSERT_GE(sc.config.fingertips.size(), 30u);
    ASSERT_TRUE(sc.place.found);
    ASSERT_TRUE(sc.sg.has_table);
    const double mg_dt_feet = (sc.total_mass + kCupMass) * 9.81 * kDt;
    const double mg_dt_cup = kCupMass * 9.81 * kDt;
    std::printf("[G1d GATE-A] table-lift union: table_z=%.4f (cup0_z=%.4f proxy_half_z="
                "%.4f) m_cup*g*dt=%.5f (m_r+m_c)*g*dt=%.4f feet=%zu fingertips=%zu\n",
                sc.sg.table_height, sc.cup0.position.z, sc.sg.cup_table_proxy_half.z,
                mg_dt_cup, mg_dt_feet, sc.sg.feet.size(), sc.config.fingertips.size());

    const auto tmpl = MakeUnionTemplate(sc);
    const auto hold_drive = BuildGraspStanceDriveSet(sc, /*close_offset=*/0.0f);
    const auto rest_drive = BuildGraspStanceDriveSet(sc, kRestBackOffset);
    const auto close_drive = BuildGraspStanceDriveSet(sc);
    for (const DriveLink& d : hold_drive) ASSERT_NE(d.link, kInvalidLink);
    for (const DriveLink& d : rest_drive) ASSERT_NE(d.link, kInvalidLink);
    for (const DriveLink& d : close_drive) ASSERT_NE(d.link, kInvalidLink);

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGravityZ, kDt);
    coresident::UnifiedCoResidentStepper oracle(context, sc.host, sc.sg, kGravityZ, kDt);
    ASSERT_TRUE(world.TableEnabled(0u));  // starts at has_table, the oracle's init.
    ASSERT_TRUE(oracle.TableEnabled());
    CopCtl cop_o(context, sc, 1u);
    CopCtl cop_b(context, sc, 1u);

    constexpr uint32_t kFloorSteps = 10u;  // the H1.2/G1c FP-floor window bar; the
                                           // hold-curl WINDOW phase spans exactly it.
    constexpr uint32_t kRest = 100u;   // back-off from kFloorSteps; cup re-lands+rests.
    constexpr uint32_t kClose = 80u;   // grip PD sweeps in + squeezes into the hold.
    constexpr uint32_t kLiftAt = kRest + kClose;  // table removed HERE (both sides).
    constexpr uint32_t kRun = 300u;
    constexpr uint32_t kRestWin = 50u;        // late-REST settled window length.
    constexpr double kFloorTol = 1.0e-6;
    constexpr double kFloorTolQdot = 5.0e-6;
    const std::vector<float> one_scale = {1.0f};

    double floor_max_dpos = 0.0, floor_max_dvel = 0.0, floor_max_dqdot = 0.0,
           floor_max_dbase = 0.0;
    double full_max_dpos = 0.0, full_max_dvel = 0.0, full_max_dqdot = 0.0;
    uint32_t count_match_onset = kRun;
    bool count_open = true;
    uint32_t rest_finger_steps = 0u, all3_steps = 0u, close_table_steps = 0u;
    uint32_t first_engage = kRun;  // first CLOSE-phase step with finger contact.
    uint32_t lift_table_rows_total = 0u;
    double lift_table_vimp_abs = 0.0;
    double rest_table_ratio_sum = 0.0, rest_tri_ratio_sum = 0.0,
           rest_feet_ratio_sum = 0.0;
    uint32_t rest_ratio_n = 0u;
    double late_cup_ratio_sum = 0.0, late_feet_ratio_sum = 0.0;
    uint32_t late_ratio_n = 0u;
    float cup_z_at_toggle = 0.0f;
    for (uint32_t s = 0u; s < kRun; ++s) {
        if (s == kLiftAt) {
            oracle.SetTableEnabled(false);  // the SAME step on BOTH sides.
            world.SetTableEnabled(false);
            cup_z_at_toggle = SnapBatched(world, 0u).cup.position.z;
        }
        const auto& drv = (s < kFloorSteps) ? hold_drive
                          : (s < kRest)     ? rest_drive
                                            : close_drive;
        DrivePdOracle(oracle, sc, drv, 1.0f, &cop_o);
        DrivePdBatched(world, sc, drv, one_scale, 0.0f, &cop_b);
        const auto rep = oracle.Step();
        world.Step();
        const auto& brep = world.GraspReports()[0];

        // ----- (i) ROW-CLASS liveness: the COMPLETE union, phase-aware. -----------
        ASSERT_GT(rep.foot_normal_rows, 0u) << "oracle lost foot contact at step " << s;
        ASSERT_GT(brep.foot_normal_rows, 0u)
            << "batched world lost foot contact at step " << s;
        if (brep.finger_contacts > 0u && brep.table_row_count > 0u)
            ++all3_steps;  // feet asserted above -> all three classes in ONE solve.
        if (s < kRest) {
            // The table must carry rows on BOTH sides through the seeded WINDOW
            // (measured 4..2 rows, all three classes live) and through the SETTLED
            // late-REST window (the cup re-landed and rests). The mid-REST contact
            // is TRANSIENT (the curl wedge pops the cup a few mm up before the
            // back-off clears -- scene finding (1) in the header): REPORTED only.
            if (s < kFloorSteps || s >= kRest - kRestWin) {
                ASSERT_GT(rep.table_row_count, 0u)
                    << "oracle table rows vanished while RESTING at step " << s;
                ASSERT_GT(brep.table_row_count, 0u)
                    << "batched table rows vanished while RESTING at step " << s
                    << " (table emission missing? has_table support absent)";
            }
            if (brep.finger_contacts > 0u) ++rest_finger_steps;
        } else if (s < kLiftAt) {
            // CLOSING: the fingers sweep in from the back-off -- record the first
            // engagement (asserted < kLiftAt below); table support is REPORTED (the
            // squeeze may progressively unload it).
            if (first_engage == kRun && brep.finger_contacts > 0u &&
                rep.finger_contacts > 0u)
                first_engage = s;
            if (brep.table_row_count > 0u) ++close_table_steps;
        } else {
            // LIFTED: the table pair is GONE (exactly zero rows, both sides) and the
            // hold is friction-only.
            ASSERT_EQ(rep.table_row_count, 0u)
                << "oracle still emits table rows after SetTableEnabled(false), step "
                << s;
            lift_table_rows_total += brep.table_row_count;
            lift_table_vimp_abs += std::fabs(brep.table_vertical_impulse);
            ASSERT_GT(brep.finger_contacts, 0u)
                << "batched world lost the cup after the table was removed, step " << s;
            ASSERT_GT(rep.finger_contacts, 0u)
                << "oracle lost the cup after the table was removed, step " << s;
        }

        // ----- (ii) contact-set COUNT parity (fingers + feet + TABLE rows). -------
        if (count_open && (rep.finger_contacts != brep.finger_contacts ||
                           rep.foot_normal_rows != brep.foot_normal_rows ||
                           rep.table_row_count != brep.table_row_count)) {
            count_open = false;
            count_match_onset = s;
        }

        // ----- (iii) state deltas: HARD in-window, REPORTED full-run. -------------
        const EnvSnap so = SnapOracle(oracle);
        const EnvSnap sb = SnapBatched(world, 0u);
        const SnapDiff d = SnapMaxDiff(so, sb);
        const double dvel = (so.cup.linear_velocity - sb.cup.linear_velocity).Length();
        full_max_dpos = std::max(full_max_dpos, d.dcup);
        full_max_dvel = std::max(full_max_dvel, dvel);
        full_max_dqdot = std::max(full_max_dqdot, d.dqdot);
        if (s < kFloorSteps) {
            floor_max_dpos = std::max(floor_max_dpos, d.dcup);
            floor_max_dvel = std::max(floor_max_dvel, dvel);
            floor_max_dqdot = std::max(floor_max_dqdot, d.dqdot);
            floor_max_dbase = std::max(floor_max_dbase, d.dbase);
        }

        // ----- (iv) the impulse-choreography windows (batched side). --------------
        if (s >= kRest - kRestWin && s < kRest) {  // late-REST: settled on the table.
            rest_table_ratio_sum += brep.table_vertical_impulse / mg_dt_cup;
            rest_tri_ratio_sum +=
                (brep.table_vertical_impulse + brep.cup_vertical_impulse) / mg_dt_cup;
            rest_feet_ratio_sum += brep.foot_normal_impulse_sum / mg_dt_feet;
            ++rest_ratio_n;
        }
        if (s >= kRun - 50u) {  // late-LIFT: friction-only hold.
            late_cup_ratio_sum += brep.cup_vertical_impulse / mg_dt_cup;
            late_feet_ratio_sum += brep.foot_normal_impulse_sum / mg_dt_feet;
            ++late_ratio_n;
        }
        if (s <= 11u || (s % 50u) == 49u || s == kLiftAt || s == kRun - 1u) {
            std::printf("[G1d GATE-A] step %3u%s: |dcup|=%.3e |dvel|=%.3e |dqdot|=%.3e "
                        "contacts(o=%u b=%u) feet(o=%u b=%u) table(o=%u b=%u) "
                        "Tλ/mgdt=%.3f Fλ/mgdt=%.3f feetλ/mgdt=%.3f cup_z=%.4f\n",
                        s, (s < kRest ? " REST" : s < kLiftAt ? " CLOS" : " LIFT"),
                        d.dcup, dvel, d.dqdot, rep.finger_contacts, brep.finger_contacts,
                        rep.foot_normal_rows, brep.foot_normal_rows, rep.table_row_count,
                        brep.table_row_count, brep.table_vertical_impulse / mg_dt_cup,
                        brep.cup_vertical_impulse / mg_dt_cup,
                        brep.foot_normal_impulse_sum / mg_dt_feet, sb.cup.position.z);
        }
    }

    const double rest_table_ratio = rest_table_ratio_sum / std::max(1u, rest_ratio_n);
    const double rest_tri_ratio = rest_tri_ratio_sum / std::max(1u, rest_ratio_n);
    const double rest_feet_ratio = rest_feet_ratio_sum / std::max(1u, rest_ratio_n);
    const double late_cup_ratio = late_cup_ratio_sum / std::max(1u, late_ratio_n);
    const double late_feet_ratio = late_feet_ratio_sum / std::max(1u, late_ratio_n);
    const EnvSnap send = SnapBatched(world, 0u);
    std::printf("[G1d GATE-A RESULT] FP-FLOOR WINDOW (steps 0-%u): max|dcup|=%.3e "
                "max|dvel|=%.3e max|dqdot|=%.3e max|dbase|=%.3e. FULL RUN: "
                "max|dcup|=%.3e max|dvel|=%.3e max|dqdot|=%.3e (REPORTED, chaotic). "
                "count fork @ %u. IMPULSE TRIANGLE: rest Tλ=%.4f tri=%.4f feet=%.4f | "
                "lift table rows=%u |Tλ|=%.3e | late Fλ=%.4f feet=%.4f. rest finger-"
                "steps=%u/%u first-engage @ %u all3-steps=%u close table-steps=%u/%u. "
                "cup_z %.4f ->(toggle) %.4f -> %.4f (sink %.3f) base_z=%.4f\n",
                kFloorSteps - 1u, floor_max_dpos, floor_max_dvel, floor_max_dqdot,
                floor_max_dbase, full_max_dpos, full_max_dvel, full_max_dqdot,
                count_match_onset, rest_table_ratio, rest_tri_ratio, rest_feet_ratio,
                lift_table_rows_total, lift_table_vimp_abs, late_cup_ratio,
                late_feet_ratio, rest_finger_steps, kRest, first_engage, all3_steps,
                close_table_steps, kClose, sc.cup0.position.z, cup_z_at_toggle,
                send.cup.position.z, cup_z_at_toggle - send.cup.position.z,
                send.base_pose.position.z);

    // ★ THE HEADLINE (HARD): the FP-floor window + in-window count parity.
    EXPECT_LE(floor_max_dpos, kFloorTol)
        << "batched cup POSITION diverged in the FP-floor window -- a table-row "
           "composition defect (emission order / body key / friction stamp), not the "
           "finger NP seam";
    EXPECT_LE(floor_max_dvel, kFloorTol)
        << "batched cup VELOCITY diverged in the FP-floor window";
    EXPECT_LE(floor_max_dqdot, kFloorTolQdot)
        << "batched gripper qdot diverged in the FP-floor window";
    EXPECT_LE(floor_max_dbase, kFloorTol)
        << "batched BASE position diverged in the FP-floor window";
    EXPECT_GE(count_match_onset, kFloorSteps)
        << "the contact-set COUNT (incl. table rows) forked INSIDE the FP-floor "
           "window -- a structural composition defect";

    // ★ THE IMPULSE CHOREOGRAPHY (HARD): rest -> toggle -> triangle closes.
    EXPECT_GT(rest_table_ratio, 0.7)
        << "the TABLE is not carrying the resting cup (~m*g*dt) -- no honest rest";
    EXPECT_LT(rest_table_ratio, 1.3) << "table impulse far above the cup weight";
    EXPECT_EQ(lift_table_rows_total, 0u)
        << "table rows persist after SetTableEnabled(false) -- GHOST table support";
    EXPECT_EQ(lift_table_vimp_abs, 0.0)
        << "nonzero table impulse after the toggle -- ghost support";
    EXPECT_GT(late_cup_ratio, 0.7)
        << "fingers NOT carrying the cup weight after the table removal -- the "
           "triangle never closed (no honest friction hold)";
    EXPECT_LT(late_cup_ratio, 1.3) << "finger vertical impulse far above the cup weight";
    EXPECT_GT(late_feet_ratio, 0.7) << "feet NOT carrying robot+cup";
    EXPECT_LT(late_feet_ratio, 1.3) << "foot impulse far above the robot+cup weight";

    // The CLOSE phase genuinely captured the cup BEFORE the table was removed.
    EXPECT_LT(first_engage, kLiftAt)
        << "the fingers never re-engaged the resting cup during the CLOSE phase -- "
           "the lift choreography has no hold to hand the cup to";

    // All three classes were SIMULTANEOUSLY live in one solve (the complete union:
    // feet + fingers + table -- the capture window, fingers closed on the still-
    // table-supported cup).
    EXPECT_GT(all3_steps, 0u)
        << "feet+fingers+table never coexisted in one solve -- the union is not "
           "genuinely composed";

    // Non-vacuous: still standing, still holding (a dropped cup falls ~7.7m/300).
    EXPECT_GT(send.base_pose.position.z, sc.seat_pose.position.z - 0.2f)
        << "the base sank -- the feet did not support the robot";
    EXPECT_GT(send.cup.position.z, sc.cup0.position.z - 0.25f)
        << "the cup fell after the table removal -- the close never built a hold";

    // ------------------------------------------------------------------------------
    // THE CHAOS CONTROL through the SAME choreography (proves the full-run divergence
    // is the SCENE, not the batched path): two SAME-CODE oracles, cup IC nudged
    // 1e-7 m, identical phase schedule incl. the table toggle at kLiftAt.
    // ------------------------------------------------------------------------------
    coresident::StandGraspConfig sg_a = sc.sg;
    coresident::StandGraspConfig sg_b = sc.sg;
    sg_b.cup_state.position.x += 1.0e-7f;
    coresident::UnifiedCoResidentStepper o_a(context, sc.host, sg_a, kGravityZ, kDt);
    coresident::UnifiedCoResidentStepper o_b(context, sc.host, sg_b, kGravityZ, kDt);
    CopCtl cop_a2(context, sc, 1u);
    CopCtl cop_b2(context, sc, 1u);
    double self_max_dpos = 0.0, self_max_dvel = 0.0;
    for (uint32_t s = 0u; s < kRun; ++s) {
        if (s == kLiftAt) {
            o_a.SetTableEnabled(false);
            o_b.SetTableEnabled(false);
        }
        const auto& drv = (s < kFloorSteps) ? hold_drive
                          : (s < kRest)     ? rest_drive
                                            : close_drive;
        DrivePdOracle(o_a, sc, drv, 1.0f, &cop_a2);
        DrivePdOracle(o_b, sc, drv, 1.0f, &cop_b2);
        o_a.Step();
        o_b.Step();
        const double dpos = (o_a.Cup().position - o_b.Cup().position).Length();
        const double dvel =
            (o_a.Cup().linear_velocity - o_b.Cup().linear_velocity).Length();
        self_max_dpos = std::max(self_max_dpos, dpos);
        self_max_dvel = std::max(self_max_dvel, dvel);
    }
    std::printf("[G1d GATE-A CHAOS CONTROL] two SAME-code oracles through the SAME "
                "lift choreography, cup IC nudged 1e-7 m: self max|dpos|=%.3e "
                "max|dvel|=%.3e (vs batched-vs-oracle full-run max|dpos|=%.3e)\n",
                self_max_dpos, self_max_dvel, full_max_dpos);
    EXPECT_GT(self_max_dpos, 1.0e-4)
        << "a 1e-7 m IC seed did NOT amplify -- the scene is NOT chaotic, so the "
           "batched-vs-oracle full-run divergence is unexplained (investigate)";
    EXPECT_LT(full_max_dpos, 10.0 * self_max_dpos)
        << "the batched-vs-oracle delta is >10x a 1e-7 m IC nudge's amplification -- "
           "larger than scene chaos explains (investigate the batched table path)";
}

// ===========================================================================
// G1d GATE (b)+(c) -- N=8 MIXED independence + D1, the table LIVE, with ALL the
// per-class gap kinds: (i) env 2 NEVER-GRASPED (grip 0 + the cup parked 0.35 m
// to +X via BodyMut: it rests on the table plane OUT of the hand's reach the
// whole run -- TABLE rows persist, FINGER rows absent; the only robot<->cup
// collidables are the fingertips, so 0.35 m guarantees separation); (ii) env 6
// FOOT-AIRBORNE (base + cup seated 0.8 m high -- NO foot rows AND no table rows
// [the cup free-falls 0.55 m < 0.8 m in 80 steps, never reaching the plane]
// while the FINGER class is live: the dual-class tile gap). The remaining envs
// carry ALL THREE classes (feet + fingers closing + cup resting on the table)
// at per-env distinct stance scales. Each env must be BYTE-EXACT vs its OWN N=1
// run (same IC + grip seams); adjacent envs differ; D1 two-run byte-identity.
// ===========================================================================
TEST(BatchedH1UnionWorld, G1d_TableLiftUnion_N8_MixedNeverGraspedAirborneIndependenceAndD1) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildGraspStandTableScene(context);
    ASSERT_EQ(sc.base_dof, 6u);
    ASSERT_TRUE(sc.place.found);
    ASSERT_TRUE(sc.sg.has_table);
    const auto tmpl = MakeUnionTemplate(sc);
    const auto drive = BuildGraspStanceDriveSet(sc);  // close PD from step 0.
    for (const DriveLink& d : drive) ASSERT_NE(d.link, kInvalidLink);

    constexpr uint32_t kEnvs = 8u;
    constexpr uint32_t kRun = 80u;
    constexpr uint32_t kParkedEnv = 2u;    // never grasped: cup on the table, far.
    constexpr uint32_t kAirborneEnv = 6u;  // base+cup 0.8m high: fingers only.
    constexpr float kParkX = 0.35f;
    const std::vector<float> scales = {0.0f, 0.6f, 0.8f, 1.0f, 1.15f, 1.3f, 1.0f, 1.45f};
    std::vector<uint8_t> grip_on(kEnvs, 1u);
    grip_on[kParkedEnv] = 0u;
    Transform high_pose = sc.seat_pose;
    high_pose.position.z += 0.8f;

    // ---- (b) PER-ENV INDEPENDENCE ----
    coresident::BatchedUnifiedWorld world(context, tmpl, kEnvs, kGravityZ, kDt);
    world.SetGripperBasePose(kAirborneEnv, high_pose);
    world.BodyMut(kAirborneEnv, 0u).position.z += 0.8f;  // cup rides up with the hand.
    world.BodyMut(kParkedEnv, 0u).position.x += kParkX;  // cup parked on the table.
    CopCtl cop(context, sc, kEnvs);
    std::vector<uint32_t> foot_steps(kEnvs, 0u), finger_steps(kEnvs, 0u),
        table_steps(kEnvs, 0u);
    uint32_t parked_finger_total = 0u, airborne_foot_total = 0u,
             airborne_table_total = 0u;
    for (uint32_t s = 0u; s < kRun; ++s) {
        DrivePdBatched(world, sc, drive, scales, 0.0f, &cop, &grip_on);
        world.Step();
        for (uint32_t e = 0u; e < kEnvs; ++e) {
            const auto& r = world.GraspReports()[e];
            if (r.foot_normal_rows > 0u) ++foot_steps[e];
            if (r.finger_contacts > 0u) ++finger_steps[e];
            if (r.table_row_count > 0u) ++table_steps[e];
        }
        parked_finger_total += world.GraspReports()[kParkedEnv].finger_contacts;
        airborne_foot_total += world.GraspReports()[kAirborneEnv].foot_normal_rows;
        airborne_table_total += world.GraspReports()[kAirborneEnv].table_row_count;
    }
    // The MIXED shape is REAL -- every per-class gap present:
    EXPECT_EQ(parked_finger_total, 0u)
        << "the parked cup was touched by a finger -- the never-grasped env is vacuous";
    EXPECT_EQ(table_steps[kParkedEnv], kRun)
        << "the parked cup left the table -- it must REST the whole run";
    EXPECT_GE(foot_steps[kParkedEnv], (kRun * 7u) / 10u)
        << "the never-grasped env's feet unloaded";
    EXPECT_EQ(airborne_foot_total, 0u)
        << "the 'airborne' env touched the ground -- the foot-class gap is vacuous";
    EXPECT_EQ(airborne_table_total, 0u)
        << "the airborne env's cup reached the table plane -- raise it or shorten "
           "the run";
    EXPECT_GE(finger_steps[kAirborneEnv], (kRun * 7u) / 10u)
        << "the airborne env's hand lost the cup -- the finger-only shape is vacuous";
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        if (e == kParkedEnv || e == kAirborneEnv) continue;
        EXPECT_GE(foot_steps[e], (kRun * 7u) / 10u)
            << "env " << e << " (scale " << scales[e] << ") feet unloaded";
        EXPECT_GE(finger_steps[e], kRun / 2u)
            << "env " << e << " (scale " << scales[e] << ") never engaged the cup";
        // The close-squeeze on a table-seated cup can progressively unload the
        // table (the gate-(a) wedge finding: the wrap squeezes the cup up); the
        // table-class PERSISTENCE env is the designed parked env (==kRun above).
        // Here require the seeded table contact was genuinely live (the all-three-
        // classes union ran in this env) and REPORT the persistence count.
        EXPECT_GE(table_steps[e], 1u)
            << "env " << e << " (scale " << scales[e]
            << ") never had a table row -- the seeded rest contact is missing";
    }

    for (uint32_t e = 0u; e < kEnvs; ++e) {
        coresident::BatchedUnifiedWorld solo(context, tmpl, 1u, kGravityZ, kDt);
        if (e == kAirborneEnv) {
            solo.SetGripperBasePose(0u, high_pose);
            solo.BodyMut(0u, 0u).position.z += 0.8f;
        }
        if (e == kParkedEnv) solo.BodyMut(0u, 0u).position.x += kParkX;
        CopCtl solo_cop(context, sc, 1u);
        const std::vector<float> solo_scale = {scales[e]};
        const std::vector<uint8_t> solo_grip = {grip_on[e]};
        for (uint32_t s = 0u; s < kRun; ++s) {
            DrivePdBatched(solo, sc, drive, solo_scale, 0.0f, &solo_cop, &solo_grip);
            solo.Step();
        }
        EXPECT_TRUE(SnapByteEqual(SnapBatched(world, e), SnapBatched(solo, 0u)))
            << "env " << e << " (N=8, scale " << scales[e]
            << (e == kAirborneEnv ? ", AIRBORNE" : "")
            << (e == kParkedEnv ? ", NEVER-GRASPED" : "")
            << ") NOT byte-exact vs its own N=1 run -- env-major cross-contamination "
               "through the table rows / per-class tile gaps";
    }
    for (uint32_t e = 1u; e < kEnvs; ++e) {
        EXPECT_FALSE(SnapByteEqual(SnapBatched(world, e), SnapBatched(world, e - 1u)))
            << "env " << e << " collapsed onto its neighbor";
    }
    std::printf("[G1d GATE-B] N=%u MIXED (env %u NEVER-GRASPED/table-only, env %u "
                "AIRBORNE/finger-only, 6 all-three-classes): every env byte-exact vs "
                "its own N=1 run; foot/finger/table steps per env:",
                kEnvs, kParkedEnv, kAirborneEnv);
    for (uint32_t e = 0u; e < kEnvs; ++e)
        std::printf(" %u/%u/%u", foot_steps[e], finger_steps[e], table_steps[e]);
    std::printf(" (of %u)\n", kRun);

    // ---- (c) D1 two-run byte-identity of the FULL N=8 trajectory ----
    auto run = [&]() {
        coresident::BatchedUnifiedWorld w(context, tmpl, kEnvs, kGravityZ, kDt);
        w.SetGripperBasePose(kAirborneEnv, high_pose);
        w.BodyMut(kAirborneEnv, 0u).position.z += 0.8f;
        w.BodyMut(kParkedEnv, 0u).position.x += kParkX;
        CopCtl run_cop(context, sc, kEnvs);
        for (uint32_t s = 0u; s < kRun; ++s) {
            DrivePdBatched(w, sc, drive, scales, 0.0f, &run_cop, &grip_on);
            w.Step();
        }
        std::vector<EnvSnap> out(kEnvs);
        for (uint32_t e = 0u; e < kEnvs; ++e) out[e] = SnapBatched(w, e);
        return out;
    };
    const auto a = run();
    const auto b = run();
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        EXPECT_TRUE(SnapByteEqual(a[e], b[e]))
            << "D1: env " << e << " differed between two identical table-union runs";
    }
    std::printf("[G1d GATE-C D1] N=%u: two identical table-union rollouts (incl. "
                "never-grasped + airborne envs) byte-exact\n", kEnvs);
}

// ===========================================================================
// G1d GATE (d) -- HONESTY/BITE (spec §3, the STANDING eval): after the table is
// REMOVED and the cup is friction-held, kill the GRIP -> the cup must fall PAST
// the table height it used to rest at, while the robot keeps standing. This is
// the toggle's anti-ghost proof: phantom table rows would catch the falling cup
// AT the plane; falling far below it proves SetTableEnabled(false) genuinely
// removed the row class (and the report's table_row_count==0 is not cosmetic).
// Choreography: rest(60, hold-curl, table carries) -> close(60, grip PD) ->
// SetTableEnabled(false) -> held-free(60, friction-only hold proven) -> grip
// killed (240, stance drive continues) -> the cup falls away.
// NOTE: the cup pairs ONLY with the TABLE plane (the oracle's pair set -- there
// is no cup x ground pair), so once the table is off NOTHING stops the fall.
// ===========================================================================
TEST(BatchedH1UnionWorld, G1d_TableLiftUnion_TableRemovedGripKill_CupFallsPastTable_BITE) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    UnionScene sc = BuildGraspStandTableScene(context);
    ASSERT_TRUE(sc.place.found);
    ASSERT_TRUE(sc.sg.has_table);
    const auto tmpl = MakeUnionTemplate(sc);
    const auto rest_drive = BuildGraspStanceDriveSet(sc, kRestBackOffset);
    const auto close_drive = BuildGraspStanceDriveSet(sc);
    for (const DriveLink& d : close_drive) ASSERT_NE(d.link, kInvalidLink);

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGravityZ, kDt);
    CopCtl cop(context, sc, 1u);
    const std::vector<float> one_scale = {1.0f};

    // ----- phase 1: REST (fingers backed off -- the table carries the cup; the
    // baseline table support is real). The gate-(a) wedge finding applies: the
    // curl wedge pops the cup a few mm up before the back-off clears, then it
    // re-lands -- so the support baseline is asserted on the SETTLED tail. --------
    constexpr uint32_t kRest = 80u;
    constexpr uint32_t kRestTail = 20u;
    uint32_t rest_table_steps = 0u, rest_tail_table_steps = 0u;
    for (uint32_t s = 0u; s < kRest; ++s) {
        DrivePdBatched(world, sc, rest_drive, one_scale, 0.0f, &cop);
        world.Step();
        if (world.GraspReports()[0].table_row_count > 0u) {
            ++rest_table_steps;
            if (s >= kRest - kRestTail) ++rest_tail_table_steps;
        }
    }
    EXPECT_EQ(rest_tail_table_steps, kRestTail)
        << "the cup is not RESTING on the table at the end of the rest phase ("
        << rest_table_steps << "/" << kRest
        << " supported overall) -- the BITE has no support baseline";

    // ----- phase 2: CLOSE (the grip PD sweeps in + builds the hold while the table
    // still supports). The fingers START backed off, so engagement is asserted by
    // the END of the phase, not per step. -------------------------------------------
    constexpr uint32_t kClose = 80u;
    uint32_t close_engaged_steps = 0u;
    for (uint32_t s = 0u; s < kClose; ++s) {
        DrivePdBatched(world, sc, close_drive, one_scale, 0.0f, &cop);
        world.Step();
        if (world.GraspReports()[0].finger_contacts > 0u) ++close_engaged_steps;
    }
    ASSERT_GT(close_engaged_steps, 0u)
        << "fingers never engaged during the close phase -- no hold to hand off";
    ASSERT_GT(world.GraspReports()[0].finger_contacts, 0u)
        << "fingers not engaged at the END of the close phase";

    // ----- phase 3: TABLE REMOVED -- friction-only hold (the lift). ---------------
    world.SetTableEnabled(false);
    ASSERT_FALSE(world.TableEnabled(0u));
    constexpr uint32_t kHeldFree = 60u;
    uint32_t free_table_rows = 0u;
    for (uint32_t s = 0u; s < kHeldFree; ++s) {
        DrivePdBatched(world, sc, close_drive, one_scale, 0.0f, &cop);
        world.Step();
        free_table_rows += world.GraspReports()[0].table_row_count;
        ASSERT_GT(world.GraspReports()[0].finger_contacts, 0u)
            << "the cup left the hand right after the table removal at step " << s
            << " -- no friction hold to BITE";
    }
    EXPECT_EQ(free_table_rows, 0u)
        << "table rows persist after SetTableEnabled(false) -- ghost support";
    const EnvSnap held = SnapBatched(world, 0u);
    EXPECT_GT(held.cup.position.z, sc.cup0.position.z - 0.25f)
        << "the cup fell during the friction-held phase -- no hold to BITE";

    // ----- phase 4: GRIP KILLED (stance drive continues) -> the cup falls PAST the
    // table plane it used to rest on. -----------------------------------------------
    constexpr uint32_t kDead = 240u;
    const std::vector<uint8_t> grip_off = {0u};
    uint32_t late_finger_contacts = 0u, dead_table_rows = 0u;
    for (uint32_t s = 0u; s < kDead; ++s) {
        DrivePdBatched(world, sc, close_drive, one_scale, 0.0f, &cop, &grip_off);
        world.Step();
        dead_table_rows += world.GraspReports()[0].table_row_count;
        if (s >= kDead - 50u)
            late_finger_contacts += world.GraspReports()[0].finger_contacts;
    }
    const EnvSnap dead = SnapBatched(world, 0u);
    std::printf("[G1d GATE-D BITE] table_z=%.4f: rest(%u, table rows %u/%u) -> close"
                "(%u) -> table OFF + held(%u, cup_z=%.4f, 0 table rows) -> grip killed"
                "(%u): cup_z=%.4f (%.3f m BELOW the table plane) base_z=%.4f late "
                "finger contacts=%u dead-phase table rows=%u\n",
                sc.sg.table_height, kRest, rest_table_steps, kRest, kClose, kHeldFree,
                held.cup.position.z, kDead, dead.cup.position.z,
                sc.sg.table_height - dead.cup.position.z, dead.base_pose.position.z,
                late_finger_contacts, dead_table_rows);
    EXPECT_LT(dead.cup.position.z, sc.sg.table_height - 0.5f)
        << "the cup did NOT fall past the table height it rested at -- GHOST table "
           "support survives the toggle (or the hold is not friction-only): a FINDING";
    EXPECT_EQ(dead_table_rows, 0u)
        << "table rows re-appeared while the cup fell through the plane -- the "
           "toggle did not remove the pair";
    EXPECT_GT(dead.base_pose.position.z, sc.seat_pose.position.z - 0.2f)
        << "the robot fell with the cup -- the stand did not survive the grip kill";
    EXPECT_EQ(late_finger_contacts, 0u)
        << "finger contacts persist long after the grip kill -- the cup is stuck "
           "to the hand";
}
