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

// ----- G1c: the H1.1/H1.2 PROVEN grip-close PD (Kp=4/Kd=0.4/offset 0.18, the
// validated 10.9cm close drive) applied to the 12 wrap-driven right-hand links.
constexpr float kGripKp = 4.0f;
constexpr float kGripKd = 0.4f;
constexpr float kCloseOffset = 0.18f;
// The H1.1/H1.2 driven-finger joint properties (LoadH1Fixed's overrides -- the
// cook the 10.9cm grasp GO was PROVEN on). The G1c grasp scenes pin the 12
// driven joints to these explicitly (see LoadFloatingGraspCook for the measured
// no-op finding: the importer's default-class-only damping parse means the plain
// cook already carries them). G1a/G1b scenes keep the plain cook.
constexpr float kRealArmature = 0.1f;
constexpr float kRealDamping = 1.0f;

// The 12 grip-DRIVEN right-hand links (== H1.2 kWrapDriven): every link carrying a
// wrap sphere is in this set, so every contacted phalanx actively squeezes.
const char* const kWrapDriven[] = {
    "R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal",
    "R_index_proximal",  "R_index_intermediate",
    "R_middle_proximal", "R_middle_intermediate",
    "R_ring_proximal",   "R_ring_intermediate",
    "R_pinky_proximal",  "R_pinky_intermediate",
};

const std::string kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
const std::string kCupUsda =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";

bool AssetsAvailable() {
    return std::filesystem::exists(kH1Mjcf) && std::filesystem::exists(kCupUsda);
}

// ===========================================================================
// G1c: the FLOATING GRASP cook -- nuka::demo::LoadFloating + the H1.1/H1.2
// driven-finger joint overrides (LoadH1Fixed's damping/armature seam, here on
// the FLOATING whole-body cook: every joint keeps its natural type, only the 12
// wrap-driven finger joints get kRealDamping/kRealArmature). This reproduces,
// on the floating cook, exactly the finger dynamics the 10.9cm grasp GO (H1.1)
// and the H1.2 parity were proven on. Used by the G1c grasp scenes ONLY.
//   MEASURED no-op FINDING (kept for explicitness + as a guard against a future
// importer fix changing the cook under this scene): the MJCF importer reads
// joint damping/armature from the DEFAULT CLASS only (mjcf_importer.cpp:624,
// per-joint damping="0.5" attributes are silently ignored), and h1_with_hand's
// default class is damping=1/armature=0.1 -- so the plain floating cook ALREADY
// carries the H1.1 values on every joint (a run with this override is bit-
// identical to one without). Pre-existing importer behavior, NOT a G1c bug;
// surfaced as named debt.
// ===========================================================================
nuka::demo::Cooked LoadFloatingGraspCook(const std::string& mjcf) {
    nuka::demo::Cooked out;
    out.scene = nuka::import::LoadMjcf(mjcf);
    // Drop mesh geometry (== LoadFloating: dodge V-HACD; inertia + topology only).
    for (size_t i = 0u; i < out.scene.ShapeCount(); ++i) {
        auto& s = out.scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        s.mesh_vertices.clear();
        s.mesh_indices.clear();
    }
    const auto blob = nuka::scene::CookScene(out.scene);
    auto topos = nuka::runtime::articulation::CookArticulations(blob);
    // Resolve the 12 wrap-driven finger BODIES by name (== LoadH1Fixed's seam).
    std::vector<uint32_t> driven_bodies;
    for (const char* nm : kWrapDriven) {
        for (uint32_t b = 0u; b < out.scene.RigidBodyCount(); ++b) {
            if (out.scene.GetBody(b).name == nm) {
                driven_bodies.push_back(b);
                break;
            }
        }
    }
    auto is_driven = [&](uint32_t body) {
        return std::find(driven_bodies.begin(), driven_bodies.end(), body) !=
               driven_bodies.end();
    };
    for (auto& topo : topos) {
        for (size_t i = 0u; i < topo.link_bodies.size(); ++i) {
            if (!is_driven(topo.link_bodies[i])) continue;
            topo.joint_dampings[i] = kRealDamping;
            topo.joint_armatures[i] = kRealArmature;
        }
    }
    out.host = nuka::runtime::articulation::BuildArticulationHostState(topos, blob.bodies);
    return out;
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
    // ----- G1c: the grasp-while-standing scene (filled by BuildGraspStandScene) ----
    nuka::demo::Place place{};        // the cup placement in the curled hand (world).
    std::vector<uint32_t> grip_links; // the 12 wrap-driven right-hand device links.
};

// `grasp_cook` (G1c): cook the 12 wrap-driven finger joints at the H1.1-proven
// kRealDamping/kRealArmature (LoadFloatingGraspCook). Default false -> the plain
// LoadFloating cook, byte-identical for every G1a/G1b scene.
UnionScene BuildFloatingNoContactScene(bool grasp_cook = false) {
    UnionScene sc;
    sc.h1 = grasp_cook ? LoadFloatingGraspCook(kH1Mjcf)
                       : nuka::demo::LoadFloating(kH1Mjcf);
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
    // ----- G1d: the cup(-proxy) x table union fields (inert when the scene has
    // none -- every G1a/G1b/G1c scene authors sg.has_table=false). Copied verbatim
    // from the StandGraspConfig so BOTH sides resolve the identical table. -----
    tmpl.has_table = sc.sg.has_table;
    if (sc.sg.has_table) {
        tmpl.table_height = sc.sg.table_height;
        tmpl.table_mu = sc.sg.table_mu;
        tmpl.table_broadphase_id = sc.sg.table_broadphase_id;
        tmpl.cup_table_proxy_half = sc.sg.cup_table_proxy_half;
        tmpl.cup_table_proxy_offset = sc.sg.cup_table_proxy_offset;
        tmpl.cup_table_proxy_id = sc.sg.cup_table_proxy_id;
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
UnionScene BuildFeetGroundScene(const nuka::phi::DeviceContext& context,
                                bool grasp_cook = false) {
    UnionScene sc = BuildFloatingNoContactScene(grasp_cook);

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
    // G1c: a GRIP-close link. Its target is NOT scaled by the per-env stance scale
    // (the close target q_curl+0.18 is absolute) and it honors the per-env grip
    // kill switch (gate b's cup-dropped env / gate d's BITE). false everywhere in
    // the G1a/G1b sets -> their drive math stays bit-identical (the non-grip path
    // computes the SAME scale*target expression as before).
    bool grip = false;
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

// G1c: the GRASP-while-standing drive = the G1b stance-hold set with the 12 wrap-
// driven right-hand links' gentle rest-hold REPLACED by the H1.1 grip-close PD
// (Kp=4/Kd=0.4/target = q + close_offset, UNCLAMPED -- exactly the proven close
// the H1.1 probe / H1.2 parity scene drive at close_offset=kCloseOffset; the
// scene builder's SETTLE pre-roll uses close_offset=0 so the fingers HOLD the
// curled pre-pose while the stance settles). Built from the CURRENT sc.host q
// (curled; settled after the pre-roll), so the non-grip links' hold targets are
// their equilibrium q. grip=true marks the close links for the per-env grip kill
// switch + the unscaled-target rule in the drive seams.
std::vector<DriveLink> BuildGraspStanceDriveSet(const UnionScene& sc,
                                                float close_offset = kCloseOffset) {
    std::vector<DriveLink> out = BuildStanceHoldDriveSet(sc);
    auto is_grip = [&](uint32_t l) {
        for (uint32_t g : sc.grip_links)
            if (g == l) return true;
        return false;
    };
    const uint32_t wrist = nuka::demo::LinkByName(sc.h1, "right_hand_link");
    for (DriveLink& d : out) {
        if (d.link == wrist) {
            // The WRIST is the one right-hand joint NOT in the wrap-driven set. The
            // H1.1/H1.2 cooks WELDED it (LoadH1Fixed); on the floating whole-body
            // cook it is live and carries the cup load + the close-reaction torque,
            // so it gets its REAL MJCF motor limit (right_hand_joint ctrlrange
            // +/-6 N*m, h1_with_hand.xml:408) instead of HoldLimitFor's 1 N*m finger
            // default -- a DRIVE choice within the real actuator, applied identically
            // to both seams (parity-preserving), standing in for the weld the proven
            // H1.1 scene had. NOT contact fakery: the cup is held by finger friction
            // alone. (Honesty note: the pre-settle cup escape that motivated probing
            // this was MEASURED insensitive to 1-vs-6 N*m -- its root cause was the
            // unsettled scene authoring, fixed by the settle pre-roll above; the real
            // limit is kept because it is the honest actuator capability.)
            d.tlim = 6.0f;
            continue;
        }
        if (!is_grip(d.link)) continue;
        d.target = sc.host.q[d.link] + close_offset;  // close past the curled pre-pose.
        d.kp = kGripKp;
        d.kd = kGripKd;
        d.tlim = 0.0f;  // unclamped (the H1.1/H1.2 close drive has no clamp).
        d.grip = true;
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
        // G1c: a grip link's close target is ABSOLUTE (never stance-scaled). For the
        // G1a/G1b sets (grip=false everywhere) tgt == target_scale*d.target -- the
        // identical float expression as before, bit-unchanged.
        const float tgt = d.grip ? d.target : target_scale * d.target;
        float u = d.kp * (tgt - st.q[d.link]) - d.kd * st.qdot[d.link];
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
// `grip_on` (G1c): an optional per-env grip kill switch -- 0 leaves the grip links'
// action columns at 0 (zero close torque from that step on: the cup-dropped env /
// the BITE), 1 (or nullptr) drives the H1.1 close PD. Non-grip links are unaffected.
void DrivePdBatched(coresident::BatchedUnifiedWorld& world, const UnionScene& sc,
                    const std::vector<DriveLink>& drive,
                    const std::vector<float>& target_scales,
                    float base_col_value = 0.0f, CopCtl* cop = nullptr,
                    const std::vector<uint8_t>* grip_on = nullptr) {
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
                if (dl.grip && grip_on != nullptr && (*grip_on)[e] == 0u)
                    continue;  // G1c: grip killed for this env -> zero close torque.
                // G1c: a grip link's close target is ABSOLUTE (never stance-scaled);
                // non-grip links compute the identical scale*target as before.
                const float tgt = dl.grip ? dl.target : scale * dl.target;
                float u = dl.kp * (tgt - st.q[dl.link]) -
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
// G1c scene: the G1b feet/ground/stance scene + the H1.1/H1.2 GRASP POSE -- the
// cup moved from 5m away INTO the curled right hand's wrap, AT THE SETTLED
// STANDING STATE. Authoring sequence (all deterministic, both sides share the
// result):
//   1. curl the right hand to the proven 10.9cm open pre-pose (CurlForScale),
//   2. SETTLE pre-roll: run the ORACLE on the cup-far config (feet-only rows,
//      exactly the G1b regime) for kGraspSettleSteps with the full stance drive
//      + CoP law + fingers HOLDING the curl (close_offset=0) -> the base sinks
//      to its compliant-contact equilibrium and the hold-PD arm sags to ITS
//      equilibrium; download that settled state as the scene proto,
//   3. search the cup placement on the SETTLED FK (the same validated force-
//      closure caging search the H1.1 GO used) and seed the cup there.
// WHY the settle is REQUIRED (measured): authored at the raw seat pose, the
// stance/arm settle transient moves the hand away from the searched placement
// within the first steps (base -7mm by step 14) and the marginally-caged cup
// walks out of the wrap by step ~58 -- on the ORACLE and the batched world
// IDENTICALLY (so it was never a parity defect; it was a scene-authoring
// defect). The H1.1 GO held its cup on a FIXED cook (zero robot transient);
// "the cup starts held in the closed hand" therefore means: at the standing
// EQUILIBRIUM. NO table either way: the cup free-hangs in the wrap, held by
// finger friction force-closure alone.
// ===========================================================================
constexpr uint32_t kGraspSettleSteps = 200u;  // ~0.83s: stance+arm quasi-static.

UnionScene BuildGraspStandScene(const nuka::phi::DeviceContext& context) {
    // grasp_cook=true: the 12 driven finger joints pinned to the H1.1-proven
    // damping/armature (a measured no-op today; see LoadFloatingGraspCook).
    UnionScene sc = BuildFeetGroundScene(context, /*grasp_cook=*/true);

    // Curl the right hand to the H1.1 curled-open pre-pose. ApplyCurl touches
    // EXACTLY the 12 wrap-driven links, so the G1b stance/seat (ankle FK) is
    // unchanged; the proto BOTH sides step from now carries the curled q.
    nuka::demo::ApplyCurl(sc.h1, &sc.host, nuka::demo::CurlForScale(kSxy));

    // The 12 grip-driven links, resolved by name on the floating cook.
    for (const char* nm : kWrapDriven) {
        const uint32_t l = nuka::demo::LinkByName(sc.h1, nm);
        if (l != kInvalidLink) sc.grip_links.push_back(l);
    }
    const uint32_t hand = nuka::demo::LinkByName(sc.h1, "right_hand_link");
    if (hand == kInvalidLink) return sc;  // place.found stays false -> loud test fail.

    // ----- The cup placement, searched on the SEAT + curled FK -- the H1.1-GO
    // search geometry (arc 218deg / 11 genuine / 1.9mm shallow, measured). The
    // settled hand TILTS slightly, which pushes every grid candidate off the
    // marginal <=2mm-shallow / >200deg-arc criteria (measured: found=false on the
    // settled FK), so the search runs HERE and the result is carried through the
    // settle by the HAND-FRAME delta below -- the cup IC rides the hand exactly,
    // preserving the searched wrap geometry. -----
    const auto seat_poses = nuka::demo::ForwardKinematics(context, sc.host);
    const Transform hand_seat = seat_poses[hand];
    {
        const auto spheres = nuka::demo::WrapSpheres(false, Vec3{});
        const nuka::demo::CupHull base = nuka::demo::LoadCupHull(kCupUsda);
        const nuka::demo::CupHull hull = nuka::demo::ScaleCupHull(base, kSxy, kSz);
        sc.place =
            nuka::demo::BestPlacementShallowNoPalmCaging(sc.h1, seat_poses, spheres, hull);
    }

    // ----- SETTLE pre-roll (step 2 above; the cup is STILL parked 5m away, so
    // this is exactly the G1b feet-only regime). Deterministic -> the scene is
    // identical across every world/oracle a test builds from it. -----
    {
        coresident::UnifiedCoResidentStepper settle(context, sc.host, sc.sg,
                                                    kGravityZ, kDt);
        CopCtl cop(context, sc, 1u);
        const auto hold = BuildGraspStanceDriveSet(sc, /*close_offset=*/0.0f);
        for (uint32_t s = 0u; s < kGraspSettleSteps; ++s) {
            DrivePdOracle(settle, sc, hold, 1.0f, &cop);
            settle.Step();
        }
        articulation::ArticulationHostState settled;
        settle.Download(&settled);
        sc.host.q = settled.q;
        sc.host.qdot = settled.qdot;
        sc.host.link_velocity = settled.link_velocity;
        sc.host.base_pose[0] = settled.base_pose[0];
    }

    // ----- Carry the searched placement through the settle by the HAND-FRAME
    // delta: the cup is rigidly attached to the hand frame across the settle
    // (cup-in-hand offset preserved; orientation gets the same small rotation),
    // exactly how a held cup would ride. -----
    if (sc.place.found) {
        const auto settled_poses = nuka::demo::ForwardKinematics(context, sc.host);
        const Transform hand_settled = settled_poses[hand];
        const Quat dq = hand_settled.rotation * hand_seat.rotation.Conjugate();
        const Vec3 local = hand_seat.rotation.Conjugate().Rotate(
            sc.place.center - hand_seat.position);
        sc.place.center = hand_settled.position + hand_settled.rotation.Rotate(local);
        sc.config.cup_state.position = sc.place.center;
        sc.config.cup_state.orientation = dq;  // near-identity settle tilt.
    }

    // Move the cup INTO the hand on BOTH sides (oracle config + template seed).
    sc.sg.cup_state = sc.config.cup_state;
    sc.cup0 = sc.config.cup_state;
    return sc;
}

// ===========================================================================
// G1d scene: the G1c grasp-while-standing scene + the STATIC TABLE under the
// settled carried cup (the lift choreography's support). The cup<->table pair
// uses the FLAT-BOTTOM PROXY box (id 7001 -- the cup-sequence demo's advisor-
// confirmed authoring, test_h1_cup_sequence_demo.cpp:368-370: the raw hull's
// coplanar bottom rim ROCKS on a plane, named engine debt; a real mug has a
// flat base, so the bounding box centered on the cup origin -- proxy bottom ==
// hull bottom -- is the FAITHFUL resting representation, not a cheat). TABLE
// HEIGHT (the G1c scene finding "derive from the SETTLED cup bottom"): cup0 IS
// the settled carried placement (the searched seat-pose placement carried
// through the 200-step stance settle by the hand-frame delta), so its proxy
// bottom -half.z is the settled in-wrap cup bottom at t=0; the plane is seated
// kTableRestPen INTO it (the foot/cup-demo 2mm active-rest authoring) so the
// cup RESTS from step 0, no fall-then-catch transient. The FURTHER ~6cm sink
// G1c measured happens only once the support is REMOVED -- exactly gate (a)'s
// post-SetTableEnabled(false) phase (reported there).
// ===========================================================================
constexpr float kTableRestPen = 0.002f;

// G1d: the REST-phase grip target offset (rad, NEGATIVE = backed off the curl).
// MEASURED scene finding (gate a header): holding the bare curl (offset 0) WEDGES
// the resting cup -- the shallow wrap contacts squeeze it down-and-out and it
// CLIMBS off the table (~0.6 mm/step, table rows gone by ~step 27, BOTH sides
// identically). A -0.10 back-off was MEASURED INSUFFICIENT: the cup stays jammed
// hovering ~6 mm up (wrap contacts persist 100/100 rest steps, table over-
// carrying ~4.5 m*g*dt against fingers pressing -3.7). Backing the 12 driven
// joints 0.25 rad off the curl genuinely opens the wrap so the cup drops back
// and the table ALONE carries it; the close phase then sweeps back in
// (-0.25 -> +0.18 target) and captures -- the honest lift choreography.
constexpr float kRestBackOffset = -0.25f;

UnionScene BuildGraspStandTableScene(const nuka::phi::DeviceContext& context) {
    UnionScene sc = BuildGraspStandScene(context);
    if (!sc.place.found) return sc;  // upstream ASSERT(place.found) fails loudly.
    const nuka::demo::CupHull base = nuka::demo::LoadCupHull(kCupUsda);
    const nuka::demo::CupHull hull = nuka::demo::ScaleCupHull(base, kSxy, kSz);
    const Vec3 half = (hull.hi - hull.lo) * 0.5f;
    sc.sg.has_table = true;
    sc.sg.cup_table_proxy_half = half;  // bbox proxy, centered (offset 0): bottom
    sc.sg.cup_table_proxy_offset = Vec3{0.0f, 0.0f, 0.0f};  // == hull bottom.
    sc.sg.cup_table_proxy_id = 7001u;   // canonical disjoint handle map (risk #7):
    sc.sg.table_broadphase_id = 8500u;  // cup 7000 / proxy 7001 / ground 8000 /
                                        // table 8500 / fingers 9000+ / feet 12000+.
    sc.sg.table_mu = 0.6f;
    // The settled carried-cup bottom: the COM-centered hull's bottom is -half.z in
    // the cup body frame (the settle tilt dq is near-identity); seat the plane 2mm
    // ABOVE it (cup bottom 2mm below the plane -> active rest contact at step 0,
    // the same arithmetic the cup-sequence demo + the G1b foot seat use).
    sc.sg.table_height = sc.cup0.position.z - half.z + kTableRestPen;
    return sc;
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
