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
//
// ADDITIVE: a NEW parallel TU. ZERO production edits -- the no-contact grasp
// template (has_grasp=true, cup far away, never touched) is the brief's fallback
// and needs no new BatchedUnifiedWorld branch.
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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace coresident = nuka::runtime::coresident;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kGravityZ = -9.81f;
constexpr float kDt = 1.0f / 240.0f;

// The H1.2 / H1.1b LOCKED constants (cup ~10.9 cm, mu, mass) -- reused verbatim so
// the G1c/G1d extensions of this TU share ONE authoring path with the proven scene.
constexpr float kCupMass = 0.2f;
constexpr float kMu = 0.8f;
constexpr float kSxy = 1.8f;
constexpr float kSz = 1.8f;

// G1a: the cup is parked THIS far (m) to the robot's +X. Both the cup and the
// floating robot free-fall at the same rate, so the gap is invariant over the run
// -- the fingertip<->cup narrowphase runs every step and must emit ZERO contacts.
constexpr float kCupFarOffsetX = 5.0f;

const std::string kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
const std::string kCupUsda =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";

bool AssetsAvailable() {
    return std::filesystem::exists(kH1Mjcf) && std::filesystem::exists(kCupUsda);
}

// ===========================================================================
// The G1a scene: a floating-base whole-body H1 + the (FAR) 10.9cm cup, wrapped
// into BOTH a GraspConfig (the oracle input) and a BatchedSceneTemplate.
// ===========================================================================
struct UnionScene {
    nuka::demo::Cooked h1;
    articulation::ArticulationHostState host;  // the floating proto (zeroed vels).
    coresident::GraspConfig config;            // oracle input (cup FAR -> no contact).
    BodyState cup0;
    uint32_t root_link = 0u;
    uint32_t dof_stride = 0u;
    uint32_t base_dof = 0u;
    uint32_t link_count = 0u;
};

UnionScene BuildFloatingNoContactScene() {
    UnionScene sc;
    sc.h1 = nuka::demo::LoadFloating(kH1Mjcf);
    sc.host = sc.h1.host;
    // Quiescent IC (== the cup-sequence demo's floating seed): zero velocities,
    // identity base rotation; q stays at the cook rest pose.
    for (auto& v : sc.host.link_velocity)
        for (float& c : v.v) c = 0.0f;
    for (float& qd : sc.host.qdot) qd = 0.0f;
    sc.host.base_pose[0].rotation = Quat::Identity();

    sc.root_link = sc.host.articulation_link_offset[0];
    sc.dof_stride = articulation::ArticulationDofCount(sc.host, 0u);
    sc.base_dof =
        articulation::ArticulationJointDofCount(sc.host.joint_type[sc.root_link]);
    sc.link_count = sc.host.TotalLinkCount();

    // The 30-sphere finger-only wrap (the H1.1 GO set) resolved BY NAME on the
    // floating cook -- handles 9000+ (the canonical disjoint handle map).
    const auto spheres = nuka::demo::WrapSpheres(false, Vec3{});
    uint32_t handle = 9000u;
    for (const auto& s : spheres) {
        const uint32_t l = nuka::demo::LinkByName(sc.h1, s.body);
        if (l == kInvalidLink) continue;
        coresident::CoResidentFingertip ft;
        ft.link = l;
        ft.broadphase_handle = handle++;
        ft.local_offset = s.local_offset;
        ft.radius = s.radius;
        sc.config.fingertips.push_back(ft);
    }

    // The 10.9cm cup hull (COM-centered mesh-local verts), parked FAR to +X so the
    // narrowphase genuinely runs and genuinely finds nothing, the whole run.
    const nuka::demo::CupHull base = nuka::demo::LoadCupHull(kCupUsda);
    const nuka::demo::CupHull hull = nuka::demo::ScaleCupHull(base, kSxy, kSz);
    const Vec3 hull_center = (hull.hi + hull.lo) * 0.5f;
    const Vec3 half = (hull.hi - hull.lo) * 0.5f;
    sc.config.cup.hull_verts = hull.verts;
    for (uint32_t i = 0u; i < hull.vcount; ++i) {
        sc.config.cup.hull_verts[i * 3u + 0u] -= hull_center.x;
        sc.config.cup.hull_verts[i * 3u + 1u] -= hull_center.y;
        sc.config.cup.hull_verts[i * 3u + 2u] -= hull_center.z;
    }
    sc.config.cup.broadphase_body_id = 7000u;

    BodyState cup;
    cup.inv_mass = 1.0f / kCupMass;
    const float ix = kCupMass * (half.y * half.y + half.z * half.z) / 3.0f;
    const float iy = kCupMass * (half.x * half.x + half.z * half.z) / 3.0f;
    const float iz = kCupMass * (half.x * half.x + half.y * half.y) / 3.0f;
    cup.inv_inertia = Vec3{1.0f / ix, 1.0f / iy, 1.0f / iz};
    cup.position = sc.host.base_pose[0].position + Vec3{kCupFarOffsetX, 0.0f, 0.0f};
    cup.orientation = Quat::Identity();
    sc.config.cup_state = cup;
    sc.cup0 = cup;

    sc.config.grip_torque.assign(sc.link_count, 0.0f);
    sc.config.drive_force_limits.assign(sc.link_count, 0.0f);
    sc.config.friction_mu = kMu;
    sc.config.condim = 3u;
    // NO table (has_table defaults false): G1a is the NO-row increment.
    return sc;
}

coresident::BatchedSceneTemplate MakeUnionTemplate(const UnionScene& sc) {
    coresident::BatchedSceneTemplate tmpl;
    tmpl.bodies_per_env = {sc.config.cup_state};
    tmpl.has_grasp = true;
    tmpl.gripper_proto = sc.host;  // the FLOATING-base whole-body H1.
    tmpl.fingertips = sc.config.fingertips;
    tmpl.cup = sc.config.cup;
    tmpl.cup_local_index = 0u;
    tmpl.grip_torque = sc.config.grip_torque;  // all-zero; drive is PD via SetActions.
    tmpl.drive_force_limits = sc.config.drive_force_limits;
    tmpl.friction_mu = sc.config.friction_mu;
    tmpl.condim = sc.config.condim;
    return tmpl;
}

// ===========================================================================
// The NON-TRIVIAL leg/arm PD drive (the discriminative motion source). Legs swing
// from the rest cook pose to the BENT STANCE (kStanceHipPitch/-Knee/-Ankle) under
// the bridge-spike hold gains; shoulder-pitch + elbow reach a small flex under
// gentler gains. Both seams compute tau from their OWN downloaded state each step
// (the H1.2 DrivePdOracle / DrivePdBatched pattern).
// ===========================================================================
struct DriveLink {
    uint32_t link = kInvalidLink;
    float target = 0.0f;
    float kp = 0.0f;
    float kd = 0.0f;
};

std::vector<DriveLink> BuildDriveSet(const nuka::demo::Cooked& h1) {
    std::vector<DriveLink> out;
    const auto legs = nuka::demo::ResolveLegLinks(h1);
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t j = s % 5u;
        float tgt = 0.0f;
        if (j == 2u) tgt = nuka::demo::kStanceHipPitch;
        else if (j == 3u) tgt = nuka::demo::kStanceKnee;
        else if (j == 4u) tgt = nuka::demo::kStanceAnkle;
        out.push_back({legs[s], tgt, nuka::demo::kKpHold, nuka::demo::kKdHold});
    }
    for (const char* nm : {"left_shoulder_pitch_link", "right_shoulder_pitch_link"})
        out.push_back({nuka::demo::LinkByName(h1, nm), 0.35f, 18.0f, 1.5f});
    for (const char* nm : {"left_elbow_link", "right_elbow_link"})
        out.push_back({nuka::demo::LinkByName(h1, nm), 0.60f, 18.0f, 1.5f});
    return out;
}

// The articulation-local prefix-sum DOF column of a device link (== the batched
// world's DofIndexOf). For the FLOATING root this INCLUDES the 6 base DOF, so a
// driven revolute link's action column is base_dof + its joint prefix-sum.
uint32_t DofIndexOf(const articulation::ArticulationHostState& host,
                    uint32_t root_link, uint32_t link) {
    uint32_t idx = 0u;
    for (uint32_t l = root_link; l < link; ++l)
        idx += articulation::ArticulationJointDofCount(host.joint_type[l]);
    return idx;
}

// Oracle seam: per-LINK tau from the oracle's own downloaded state -> SetGripTorque.
void DrivePdOracle(coresident::UnifiedCoResidentStepper& stepper, const UnionScene& sc,
                   const std::vector<DriveLink>& drive, float target_scale) {
    articulation::ArticulationHostState st;
    stepper.Download(&st);
    std::vector<float> tau(sc.link_count, 0.0f);
    for (const DriveLink& d : drive) {
        tau[d.link] =
            d.kp * (target_scale * d.target - st.q[d.link]) - d.kd * st.qdot[d.link];
    }
    stepper.SetGripTorque(tau);
}

// Batched seam: env-major DOF-indexed actions from EACH env's own downloaded state
// -> SetActions. `base_col_value` != 0 stuffs ALL SIX base columns (0..5) with that
// value -- gate (d)'s dead-column probe; 0 leaves them at the parity default.
void DrivePdBatched(coresident::BatchedUnifiedWorld& world, const UnionScene& sc,
                    const std::vector<DriveLink>& drive,
                    const std::vector<float>& target_scales,
                    float base_col_value = 0.0f) {
    const uint32_t n = world.EnvCount();
    const uint32_t dof_stride = sc.dof_stride;
    std::vector<float> actions(static_cast<size_t>(n) * dof_stride, 0.0f);
    for (uint32_t e = 0u; e < n; ++e) {
        articulation::ArticulationHostState st;
        world.DownloadGripper(e, &st);
        const size_t aoff = static_cast<size_t>(e) * dof_stride;
        if (base_col_value != 0.0f) {
            for (uint32_t d = 0u; d < sc.base_dof && d < dof_stride; ++d)
                actions[aoff + d] = base_col_value;
        }
        const float scale = target_scales[e];
        for (const DriveLink& dl : drive) {
            const uint32_t d = DofIndexOf(sc.host, sc.root_link, dl.link);
            if (d < dof_stride) {
                actions[aoff + d] = dl.kp * (scale * dl.target - st.q[dl.link]) -
                                    dl.kd * st.qdot[dl.link];
            }
        }
    }
    world.SetActions(actions.data(), actions.size());
}

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
