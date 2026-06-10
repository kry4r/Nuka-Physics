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
    // ----- G1b: the feet x ground union scene (filled by BuildFeetGroundScene) -----
    coresident::StandGraspConfig sg;  // the StepStandGrasp oracle input (feet+ground+far cup).
    bool has_feet = false;            // gates the template's has_feet fields.
    double total_mass = 0.0;          // Σ link masses (the m*g*dt force-balance reference).
    Transform seat_pose{};            // the seated base pose (foot bottoms 2mm into ground).
    float poly_cx = 0.0f;             // pinned world foot-polygon center x (the CoP ref).
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
    // ----- G1b: the feet x ground union fields (inert when the scene has none) -----
    tmpl.has_feet = sc.has_feet;
    if (sc.has_feet) {
        tmpl.feet = sc.sg.feet;
        tmpl.ground = sc.sg.ground;
        tmpl.foot_mu = sc.sg.foot_mu;
    }
    return tmpl;
}

// ===========================================================================
// G1b scene: the G1a floating no-contact scene + the BENT STANCE + 4 authored
// ankle foot spheres + the static ground, base SEATED so the lowest foot bottom
// rests 2mm INTO the plane (active contact from step 0 -- the cup-sequence demo
// authoring, brief §4.1). The far cup + the 30 fingertips are KEPT on both sides
// so the oracle emits the SAME pair set the batched world resolves: every finger
// pair finds NOTHING (cup 5m away) -> EXACTLY feet x ground rows exist.
// ===========================================================================
UnionScene BuildFeetGroundScene(const nuka::phi::DeviceContext& context) {
    UnionScene sc = BuildFloatingNoContactScene();

    // Bent stance on both legs (hip_pitch/knee/ankle; yaw/roll = 0).
    const auto legs = nuka::demo::ResolveLegLinks(sc.h1);
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t j = s % 5u;
        float v = 0.0f;
        if (j == 2u) v = nuka::demo::kStanceHipPitch;
        else if (j == 3u) v = nuka::demo::kStanceKnee;
        else if (j == 4u) v = nuka::demo::kStanceAnkle;
        if (legs[s] != kInvalidLink) sc.host.q[legs[s]] = v;
    }

    // Feet: toe/heel spheres per ankle, handles 12000+ (the canonical disjoint
    // handle map: cup 7000 / ground 8000 / fingers 9000+ / feet 12000+).
    uint32_t foot_handle = 12000u;
    for (uint32_t a : {legs[4], legs[9]}) {  // left_ankle, right_ankle.
        for (float x : {nuka::demo::kFootToeX, nuka::demo::kFootHeelX}) {
            coresident::CoResidentFootSphere fs;
            fs.link = a;
            fs.broadphase_handle = foot_handle++;
            fs.local_offset = Vec3{x, 0.0f, nuka::demo::kFootBottomZ};
            fs.radius = nuka::demo::kFootSphereR;
            sc.sg.feet.push_back(fs);
        }
    }

    // Seat the base so the LOWEST foot bottom rests 2mm into the ground (an ACTIVE
    // rest contact at step 0 -- no free-fall-then-catch transient).
    const auto poses = nuka::demo::ForwardKinematics(context, sc.host);
    float min_bottom = std::numeric_limits<float>::infinity();
    for (const auto& fs : sc.sg.feet) {
        const Vec3 c =
            poses[fs.link].position + poses[fs.link].rotation.Rotate(fs.local_offset);
        min_bottom = std::min(min_bottom, c.z - fs.radius);
    }
    constexpr float kGroundZ = 0.0f;
    constexpr float kRestPen = 0.002f;
    sc.host.base_pose[0].position.z -= (min_bottom - (kGroundZ - kRestPen));
    sc.seat_pose = sc.host.base_pose[0];
    sc.sg.ground.height = kGroundZ;
    sc.sg.ground.broadphase_id = 8000u;
    sc.sg.foot_mu = 0.8f;
    // Pin the world foot-polygon center x at the seat (the feet stay planted there;
    // the z-only seat shift left the FK x's valid) -- the CoP law's reference.
    float cx_sum = 0.0f;
    for (const auto& fs : sc.sg.feet) {
        const Vec3 c =
            poses[fs.link].position + poses[fs.link].rotation.Rotate(fs.local_offset);
        cx_sum += c.x;
    }
    sc.poly_cx = cx_sum / static_cast<float>(sc.sg.feet.size());

    // Mirror the grasp side (the far cup + 30 fingertips) into the StandGraspConfig
    // so the ORACLE emits the SAME pair set the batched world resolves.
    sc.sg.fingertips = sc.config.fingertips;
    sc.sg.cup = sc.config.cup;
    sc.sg.cup_state = sc.config.cup_state;
    sc.sg.finger_mu = sc.config.friction_mu;
    sc.sg.condim = sc.config.condim;
    sc.sg.has_table = false;  // G1b: NO cup/table contact -- feet x ground ONLY.
    sc.sg.drive_torque.assign(sc.link_count, 0.0f);
    sc.sg.drive_force_limits.assign(sc.link_count, 0.0f);

    sc.has_feet = true;
    sc.total_mass = nuka::demo::BuildCoMModel(sc.host).total;
    return sc;
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
    // G1b: the PHYSICAL torque clamp (N*m; 0 -> unclamped). The G1a sets brace-init
    // four members -> tlim stays 0 -> the G1a drive math is bit-unchanged. Applied
    // IDENTICALLY in DrivePdOracle and DrivePdBatched (parity-preserving).
    float tlim = 0.0f;
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

// G1b: the FULL-BODY stance-hold drive (the costand posture-PD law, per-link): legs
// PD toward the bent stance at the costand-proven gains/limits (hip_yaw 150/8,
// hip_roll 200/12, hip_pitch 200/14 @ ±200; knee 300/18 @ ±300; ankles = the CoP
// law below, NOT posture PD -- exactly the costand STAGE3 controller split), every
// OTHER actuated joint held at its seated rest q with the kKpHold/kKdHold gentle
// hold clamped to its REAL per-joint limit (HoldLimitFor). Both seams compute tau
// from their OWN downloaded state -- the drive is the DISCRIMINATIVE load source
// AND the thing gate (d) kills.
std::vector<DriveLink> BuildStanceHoldDriveSet(const UnionScene& sc) {
    std::vector<DriveLink> out;
    const auto legs = nuka::demo::ResolveLegLinks(sc.h1);
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t j = s % 5u;
        if (j == 4u) continue;  // ankles are CoP-driven (CopCtl), not posture PD.
        DriveLink d;
        d.link = legs[s];
        if (j == 0u) {        // hip_yaw.
            d.target = 0.0f; d.kp = 150.0f; d.kd = 8.0f; d.tlim = 200.0f;
        } else if (j == 1u) { // hip_roll.
            d.target = 0.0f; d.kp = 200.0f; d.kd = 12.0f; d.tlim = 200.0f;
        } else if (j == 2u) { // hip_pitch.
            d.target = nuka::demo::kStanceHipPitch; d.kp = 200.0f; d.kd = 14.0f; d.tlim = 200.0f;
        } else {              // knee.
            d.target = nuka::demo::kStanceKnee; d.kp = 300.0f; d.kd = 18.0f; d.tlim = 300.0f;
        }
        out.push_back(d);
    }
    auto is_leg = [&](uint32_t l) {
        for (uint32_t s = 0u; s < 10u; ++s)
            if (legs[s] == l) return true;
        return false;
    };
    for (uint32_t l = 0u; l < sc.link_count; ++l) {
        if (l == sc.root_link || is_leg(l)) continue;
        if (articulation::ArticulationJointDofCount(sc.host.joint_type[l]) == 0u) continue;
        DriveLink d;
        d.link = l;
        d.target = sc.host.q[l];  // the seated rest q (cooked 0 for torso/arms/fingers).
        d.kp = nuka::demo::kKpHold;
        d.kd = nuka::demo::kKdHold;
        d.tlim = nuka::demo::HoldLimitFor(nuka::demo::LinkName(sc.h1, l));
        out.push_back(d);
    }
    return out;
}

// ===========================================================================
// G1b: the costand-PROVEN ankle CoP balance law (test_h1_costand_transfer.cpp
// STAGE3, headline gains Kp_cop=320 / Kd_cop=50, ankle damping 1.5, ±40 N*m
// physical clamp, 15-step settle): tau_ankle = Kp*(com_x - poly_cx) + Kd*com_vx
// - kAnkleKd*qdot_ankle. The whole-body CoM comes from an FK + mass-model probe
// over THIS seam's OWN downloaded state, so both seams produce bit-identical tau
// from bit-identical states (parity preserved). One CopCtl per WORLD INSTANCE
// (per-env com_prev / call counters); the D1 runs construct fresh ones.
// ===========================================================================
struct CopCtl {
    const nuka::phi::DeviceContext* context = nullptr;
    nuka::demo::CoMModel com;
    std::array<uint32_t, 2> ankle_links{};
    float poly_cx = 0.0f;
    float kp = 320.0f, kd = 50.0f;   // the costand STAGE3 headline gains.
    float ankle_kd = 1.5f;           // small ankle joint damping (kAnkleKd).
    float tlim = 40.0f;              // the REAL ankle torque limit.
    uint32_t settle = 15u;           // engage after this many drive calls.
    std::vector<Vec3> com_prev;      // per-env previous whole-body CoM.
    std::vector<uint32_t> calls;     // per-env drive-call counter.

    CopCtl(const nuka::phi::DeviceContext& ctx, const UnionScene& sc, uint32_t envs)
        : context(&ctx),
          com(nuka::demo::BuildCoMModel(sc.host)),
          poly_cx(sc.poly_cx),
          com_prev(envs, Vec3{}),
          calls(envs, 0u) {
        const auto legs = nuka::demo::ResolveLegLinks(sc.h1);
        ankle_links = {legs[4], legs[9]};
    }

    // The CoP ankle torque for env `e` from ITS downloaded state. Mirrors the
    // costand RunRollout: com_v by finite difference of the FK CoM, engage after
    // the settle window (0 N*m feedforward before that).
    float AnkleTorque(const articulation::ArticulationHostState& st, uint32_t e) {
        const auto poses = nuka::demo::ForwardKinematics(*context, st);
        const Vec3 c = nuka::demo::WholeBodyCoM(com, poses);
        const uint32_t k = calls[e]++;
        const float com_vx = (k == 0u) ? 0.0f : (c.x - com_prev[e].x) / kDt;
        com_prev[e] = c;
        if (k < settle) return 0.0f;
        return kp * (c.x - poly_cx) + kd * com_vx;
    }
};

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
// `cop` (G1b): the optional ankle CoP law; nullptr (the G1a sets) leaves the seam
// byte-identical to before.
void DrivePdOracle(coresident::UnifiedCoResidentStepper& stepper, const UnionScene& sc,
                   const std::vector<DriveLink>& drive, float target_scale,
                   CopCtl* cop = nullptr) {
    articulation::ArticulationHostState st;
    stepper.Download(&st);
    std::vector<float> tau(sc.link_count, 0.0f);
    for (const DriveLink& d : drive) {
        float u =
            d.kp * (target_scale * d.target - st.q[d.link]) - d.kd * st.qdot[d.link];
        if (d.tlim > 0.0f) u = std::max(-d.tlim, std::min(d.tlim, u));  // G1b clamp.
        tau[d.link] = u;
    }
    if (cop != nullptr) {
        const float tau_cop = cop->AnkleTorque(st, 0u);
        for (uint32_t l : cop->ankle_links) {
            float u = tau_cop - cop->ankle_kd * st.qdot[l];
            u = std::max(-cop->tlim, std::min(cop->tlim, u));
            tau[l] = u;
        }
    }
    stepper.SetGripTorque(tau);
}

// Batched seam: env-major DOF-indexed actions from EACH env's own downloaded state
// -> SetActions. `base_col_value` != 0 stuffs ALL SIX base columns (0..5) with that
// value -- gate (d)'s dead-column probe; 0 leaves them at the parity default.
void DrivePdBatched(coresident::BatchedUnifiedWorld& world, const UnionScene& sc,
                    const std::vector<DriveLink>& drive,
                    const std::vector<float>& target_scales,
                    float base_col_value = 0.0f, CopCtl* cop = nullptr) {
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
                float u = dl.kp * (scale * dl.target - st.q[dl.link]) -
                          dl.kd * st.qdot[dl.link];
                if (dl.tlim > 0.0f)
                    u = std::max(-dl.tlim, std::min(dl.tlim, u));  // G1b clamp.
                actions[aoff + d] = u;
            }
        }
        if (cop != nullptr) {  // G1b: the ankle CoP law (same math as the oracle seam).
            const float tau_cop = cop->AnkleTorque(st, e);
            for (uint32_t l : cop->ankle_links) {
                const uint32_t d = DofIndexOf(sc.host, sc.root_link, l);
                if (d < dof_stride) {
                    float u = tau_cop - cop->ankle_kd * st.qdot[l];
                    u = std::max(-cop->tlim, std::min(cop->tlim, u));
                    actions[aoff + d] = u;
                }
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
