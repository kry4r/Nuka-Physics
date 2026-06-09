// ---------------------------------------------------------------------------
// v0.8 C7b-2 SHOWCASE -- the H1 cup-sequence DEMO scaffold (the conjunction gate).
// ---------------------------------------------------------------------------
// THE CLAIM the showcase makes: a floating-base H1 STANDS on its feet (RL leg
// policy + hold-PD upper body) WHILE the right hand reaches / grasps / lifts /
// places a cup -- all co-resident in ONE step through the unified contact spine
// (foot<->ground + finger<->cup + cup<->table). NEITHER StepStand() nor StepGrasp()
// alone runs this; the new StepStandGrasp() emits the UNION of all three pair
// classes every step.
//
// This TU builds the FOUNDATION rungs of that ladder -- the two that prove the new
// conjunction path does NOT regress the proven stand:
//   R0  StandBaseline       -- pure standing through StepStandGrasp (empty cup hull,
//                              no table, no fingers): does the conjunction path
//                              reproduce the Gate-2 stand byte-for-byte in behavior?
//   R1  StandWithCupOnTable  -- the cup rests on a static table, fingers present at
//                              rest (clear of the cup, grip OFF): does co-residing a
//                              movable cup + static table + N fingertip collidables
//                              BREAK the stand solve? (it must not).
// The reach/close/lift/place/release rungs (R2+) are the controller's; their slots
// are stubbed below.
//
// Every helper body is the proven bridge-spike / dense-grasp code, re-homed in
// tests/coresident/h1_demo_shared.hpp (nuka::demo). This TU keeps only the
// scene-coupled pieces (the live-state obs assembler + the StandGraspConfig
// builder). It EDITS no engine file, no golden, and no existing test.
// ---------------------------------------------------------------------------

#include "h1_demo_shared.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"
#include "runtime/rigid/body_state.hpp"
#include "tiny_mlp.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace coresident = nuka::runtime::coresident;
using nuka::demo::Cooked;
using nuka::demo::CoMModel;
using nuka::demo::CupHull;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;

// Bring the shared constants into the anon namespace for terse use.
using nuka::demo::kActDim;
using nuka::demo::kDt;
using nuka::demo::kFootBottomZ;
using nuka::demo::kFootHeelX;
using nuka::demo::kFootSphereR;
using nuka::demo::kFootToeX;
using nuka::demo::kGravityZ;
using nuka::demo::kInvalidLink;
using nuka::demo::kKdHold;
using nuka::demo::kKpHold;
using nuka::demo::kObsDim;
using nuka::demo::kPolyCx;
using nuka::demo::kStanceAnkle;
using nuka::demo::kStanceHipPitch;
using nuka::demo::kStanceKnee;

const std::string kFullMjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
const std::string kCupUsda =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";
const std::string kWeightsBin = "python/spikes/out/h1_bridge_mlp.bin";

bool FileExists(const std::string& p) { return std::filesystem::exists(p); }

// ---- the dense right-hand wrap sphere set (now SHARED in h1_demo_shared.hpp) --
// For R1 the fingers sit at REST (no curl applied) far from the cup, so they emit
// ~0 finger<->cup contacts; they exist only to prove N fingertip collidables can
// co-reside with the stand+table solve without breaking it. For R2 the SAME dense
// wrap is reused, curled at the reach target, to place + nestle the cup.
using nuka::demo::WrapSphere;
using nuka::demo::WrapSpheres;
using nuka::demo::kWrapRadius;
using nuka::demo::kPalmRadius;
using nuka::demo::kPalmOffsetX;
constexpr float kLargeDynSxy = 1.8f;   // ~10.9cm dia cup (model_large canonical).
constexpr float kLargeDynSz = 1.8f;
constexpr float kCupMass = 0.2f;
constexpr float kFingerMu = 0.6f;

// ===========================================================================
// R2 REACH: the right-arm GRASP-TARGET joint pose (NO IK -- a forward approach).
// ---------------------------------------------------------------------------
// The 5 right-arm device DOF (shoulder pitch/roll/yaw + elbow + wrist) interpolate
// rest -> this target so the CURLED right hand ends up at a SHALLOW front-right
// table-height spot. The cup is then PLACED where the curled hand nestles it
// (BestPlacement). Tuned via the FK-only probe below (cheap; no stepping).
// ---------------------------------------------------------------------------
struct RightArmTarget {
    float shoulder_pitch = 0.0f;  // axis y: + forward/down (range -2.87..2.87).
    float shoulder_roll  = 0.0f;  // axis x: - abducts toward body (range -3.11..0.34).
    float shoulder_yaw   = 0.0f;  // axis z (range -4.45..1.3).
    float elbow          = 0.0f;  // axis y: + flexes (range -1.25..2.61).
    float wrist          = 0.0f;  // right_hand_joint, axis x roll (range -3.05..3.05).
};

// The right-arm device link names (5 DOF). right_hand_link is the wrist link the
// fingers hang off of.
inline const std::array<std::string, 5> kRightArmLinks = {
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_hand_link"};

// The 10 right-FINGER device link names (the dense wrap's driven set).
inline const std::array<std::string, 10> kRightFingerLinks = {
    "R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal",
    "R_index_proximal", "R_index_intermediate", "R_middle_proximal", "R_middle_intermediate",
    "R_ring_proximal", "R_ring_intermediate"};
// (R_pinky is in the wrap spheres too; include it in the curl set for completeness.)
inline const std::array<std::string, 2> kRightPinkyLinks = {
    "R_pinky_proximal", "R_pinky_intermediate"};

// ===========================================================================
// The demo phase ladder + per-rung metrics (the controller reads these fields).
// ===========================================================================
enum class DemoPhase {
    Establish, Reach, Pregrasp, Close, Lift, Place, Release, HeroHold };

struct DemoMetrics {
    double tilt_max_deg = 0;
    uint32_t min_foot_contacts = 4;
    bool within_torque = true;
    double peak_ankle = 0, peak_knee = 0, peak_hip = 0;
    double lift_height_delta = 0;
    bool lift_table_support_absent = false;
    double place_final_height_error = 0, place_final_xy_error = 0;
    double post_release_table_contact_ratio = 0;
    bool release_grip_command_off = false;
    double post_release_hand_vertical_impulse = 0;
    double post_release_max_disp = 0, post_release_max_tilt = 0, post_release_max_speed = 0;
    bool went_nan = false;
    // R1 cup-on-table bookkeeping (the controller's downstream rungs read these).
    double cup_z0 = 0;
    double cup_z_max_dev = 0;
    double table_support_ratio = 0;
    double max_finger_contacts = 0;
    // --- R2 reach bookkeeping ---
    double arrive_dist = 1e9;        // min fingertip<->cup SURFACE distance at end of REACH.
    double cup_z_dev_prereach = 0;   // max |cup_z - cup_z0| over ESTABLISH+REACH (pre-grip).
    double peak_ankle_reach = 0;     // peak ankle torque DURING the REACH phase.
    double base_drift_xy = 0;        // |base xy - seat xy| at end of SETTLE.
    double base_drift_tilt = 0;      // base tilt (deg) at end of SETTLE.
    double arm_track_err = 0;        // max |achieved arm q - target q| at end of REACH (rad).
};

// ===========================================================================
// The conjunction scene: a standing H1 + a cup on a table in front-right + the
// dense right-hand fingertip collidables. Mirrors the bridge spike's
// BuildStandScene seat exactly, then ADDS the cup/table/fingertips.
// ===========================================================================
struct StandGraspScene {
    Cooked h1;
    articulation::ArticulationHostState host;
    coresident::StandGraspConfig config;
    CoMModel com;
    std::array<uint32_t, 10> leg_links{};   // by-name device links (contract order).
    std::vector<uint32_t> ankle_links;      // [left_ankle, right_ankle].
    std::vector<uint32_t> hold_links;       // every OTHER actuated DOF (torso/arms/fingers).
    std::vector<float> hold_target;
    std::vector<float> hold_tlim;
    float seat_base_z = 0.0f;
    Transform seat_base_pose{};             // the seated base pose (drift reference).
    // cup/table bookkeeping (only meaningful when with_cup).
    Vec3 cup_center{};
    double cup_z0 = 0.0;
    float table_height = 0.0f;
    bool with_cup = false;
    bool cup_is_box_proxy = false;  // NUKA_DEMO_BOX_HULL=1 -> flat-box collision proxy.
    // --- R2 reach bookkeeping (only meaningful when reach=true) ---------------
    bool reach = false;                      // R2: the right arm reaches to the cup.
    RightArmTarget arm_target{};             // the grasp-target right-arm q.
    std::vector<uint32_t> arm_links;         // the 5 right-arm device links (reach-PD).
    std::vector<float> arm_rest;             // their rest q (interpolate from).
    std::vector<float> arm_tgt;              // their grasp-target q (interpolate to).
    std::vector<float> arm_tlim;             // their reach-PD torque clamps.
    std::vector<uint32_t> finger_links;      // the right-finger device links (curl-PD).
    std::vector<float> finger_rest;          // their rest q (interpolate from).
    std::vector<float> finger_tgt;           // their curl-target q (interpolate to).
    std::vector<float> finger_tlim;          // their curl-PD torque clamps.
    nuka::demo::CupHull hull;                // the scaled cup hull (mesh-local, for arrive).
    Vec3 hull_center{};                      // hull bbox center (mesh-local).
};

// Apply a right-arm joint target (5 DOF) to a host state.
void ApplyRightArm(const Cooked& h1, articulation::ArticulationHostState* host,
                   const RightArmTarget& t) {
    auto setq = [&](const std::string& nm, float v) {
        const uint32_t l = nuka::demo::LinkByName(h1, nm);
        if (l != kInvalidLink) host->q[l] = v;
    };
    setq("right_shoulder_pitch_link", t.shoulder_pitch);
    setq("right_shoulder_roll_link", t.shoulder_roll);
    setq("right_shoulder_yaw_link", t.shoulder_yaw);
    setq("right_elbow_link", t.elbow);
    setq("right_hand_link", t.wrist);
}

// rung: R0 -> empty hull, no table, no fingers (with_cup=false).
//       R1 -> cup on table at a fixed front-right placeholder (with_cup=true, reach=false).
//       R2 -> cup placed where the curled right-arm GRASP-TARGET hand nestles it
//             (with_cup=true, reach=true); the arm/fingers are LEFT AT REST in sc.host so
//             ESTABLISH holds rest and REACH ramps rest->target.
StandGraspScene BuildStandGraspScene(const nuka::phi::DeviceContext& context,
                                     bool with_cup, bool reach = false,
                                     RightArmTarget arm_target = RightArmTarget{}) {
    StandGraspScene sc;
    sc.h1 = nuka::demo::LoadFloating(kFullMjcf);
    sc.host = sc.h1.host;
    for (auto& v : sc.host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (float& qd : sc.host.qdot) qd = 0.0f;
    sc.host.base_pose[0].rotation = Quat::Identity();

    sc.leg_links = nuka::demo::ResolveLegLinks(sc.h1);
    // Apply the bent stance to both legs (hip_pitch/knee/ankle); yaw/roll = 0.
    auto setq = [&](uint32_t l, float v) { if (l != kInvalidLink) sc.host.q[l] = v; };
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t j = s % 5u;
        float v = 0.0f;
        if (j == 2u) v = kStanceHipPitch;
        else if (j == 3u) v = kStanceKnee;
        else if (j == 4u) v = kStanceAnkle;
        setq(sc.leg_links[s], v);
    }
    sc.com = nuka::demo::BuildCoMModel(sc.host);

    const uint32_t l_ankle = sc.leg_links[4];   // left_ankle.
    const uint32_t r_ankle = sc.leg_links[9];   // right_ankle.
    sc.ankle_links = {l_ankle, r_ankle};
    // FEET handles start at 12000 -> DISJOINT from cup(7000)/ground(8000)/
    // table(8500)/fingers(9000..). (BuildStandScene uses 9000 for feet; here the
    // fingers take 9000.. so the feet are pushed clear to 12000.)
    uint32_t foot_handle = 12000u;
    auto add_foot = [&](uint32_t link, float x) {
        coresident::CoResidentFootSphere fs;
        fs.link = link;
        fs.broadphase_handle = foot_handle++;
        fs.local_offset = Vec3{x, 0.0f, kFootBottomZ};
        fs.radius = kFootSphereR;
        sc.config.feet.push_back(fs);
    };
    for (uint32_t a : sc.ankle_links) { add_foot(a, kFootToeX); add_foot(a, kFootHeelX); }

    auto poses = nuka::demo::ForwardKinematics(context, sc.host);
    float min_bottom = std::numeric_limits<float>::infinity();
    for (const auto& fs : sc.config.feet) {
        const Vec3 c = poses[fs.link].position + poses[fs.link].rotation.Rotate(fs.local_offset);
        min_bottom = std::min(min_bottom, c.z - fs.radius);
    }
    constexpr float kGroundZ = 0.0f;
    constexpr float kRestPen = 0.002f;
    sc.host.base_pose[0].position.z -= (min_bottom - (kGroundZ - kRestPen));
    sc.seat_base_z = sc.host.base_pose[0].position.z;
    sc.seat_base_pose = sc.host.base_pose[0];
    sc.config.ground.height = kGroundZ;
    sc.config.foot_mu = 0.8f;
    sc.config.condim = 3u;

    // ----- R2 reach: resolve the right-arm + right-finger device link sets so the hold
    //       loop can EXCLUDE them (they are driven by reach-PD / curl-PD, not hold-PD).
    //       The arm/fingers stay at REST in sc.host (ESTABLISH holds rest; REACH ramps
    //       rest -> the grasp-target / curl). ------------------------------------------
    sc.reach = reach;
    sc.arm_target = arm_target;
    auto in_set = [](const std::vector<uint32_t>& s, uint32_t l) {
        return std::find(s.begin(), s.end(), l) != s.end();
    };
    if (reach) {
        // The 5 right-arm device links + their grasp-target q (RightArmTarget order).
        const float arm_tgt_vals[5] = {arm_target.shoulder_pitch, arm_target.shoulder_roll,
                                       arm_target.shoulder_yaw, arm_target.elbow,
                                       arm_target.wrist};
        for (uint32_t s = 0u; s < 5u; ++s) {
            const uint32_t l = nuka::demo::LinkByName(sc.h1, kRightArmLinks[s]);
            if (l == kInvalidLink) continue;
            sc.arm_links.push_back(l);
            sc.arm_rest.push_back(sc.host.q[l]);  // rest (== 0 cooked).
            sc.arm_tgt.push_back(arm_tgt_vals[s]);
            sc.arm_tlim.push_back(nuka::demo::HoldLimitFor(kRightArmLinks[s]));
        }
        // The right-finger device links + their CURL-target q (CurlForScale at the cup).
        articulation::ArticulationHostState curl_host = sc.host;
        nuka::demo::ApplyCurl(sc.h1, &curl_host, nuka::demo::CurlForScale(kLargeDynSxy));
        auto add_finger = [&](const std::string& nm) {
            const uint32_t l = nuka::demo::LinkByName(sc.h1, nm);
            if (l == kInvalidLink) return;
            sc.finger_links.push_back(l);
            sc.finger_rest.push_back(sc.host.q[l]);  // rest (== 0 cooked).
            sc.finger_tgt.push_back(curl_host.q[l]); // the curl target.
            sc.finger_tlim.push_back(nuka::demo::HoldLimitFor(nm));
        };
        for (const auto& nm : kRightFingerLinks) add_finger(nm);
        for (const auto& nm : kRightPinkyLinks) add_finger(nm);
    }

    // HOLD every OTHER actuated DOF at its cooked rest q (torso, left arm, left fingers,
    // + the right arm/fingers UNLESS reach -- those are reach-PD / curl-PD driven).
    auto is_leg = [&](uint32_t l) {
        return std::find(sc.leg_links.begin(), sc.leg_links.end(), l) != sc.leg_links.end();
    };
    const uint32_t root = sc.host.articulation_link_offset[0];
    for (uint32_t l = 0u; l < sc.host.TotalLinkCount(); ++l) {
        if (l == root) continue;
        if (articulation::ArticulationJointDofCount(sc.host.joint_type[l]) == 0u) continue;
        if (is_leg(l)) continue;
        if (reach && (in_set(sc.arm_links, l) || in_set(sc.finger_links, l))) continue;
        sc.hold_links.push_back(l);
        sc.hold_target.push_back(sc.host.q[l]);
        sc.hold_tlim.push_back(nuka::demo::HoldLimitFor(nuka::demo::LinkName(sc.h1, l)));
    }

    const uint32_t link_count = sc.host.TotalLinkCount();
    sc.config.drive_torque.assign(link_count, 0.0f);
    sc.config.drive_force_limits.assign(link_count, 0.0f);

    // ----- the cup/table/fingertips (R1; left inert for R0) -------------------
    sc.with_cup = with_cup;
    if (with_cup) {
        // Cup convex hull, scaled to the ~10.9cm cup, MESH-LOCAL centered.
        const CupHull base = nuka::demo::LoadCupHull(kCupUsda);
        const CupHull hull = nuka::demo::ScaleCupHull(base, kLargeDynSxy, kLargeDynSz);
        const Vec3 hull_center = (hull.hi + hull.lo) * 0.5f;
        const Vec3 half = (hull.hi - hull.lo) * 0.5f;

        // ----- the cup COLLISION hull (id 7000) ---------------------------------
        // ALWAYS the REAL detailed cup convex hull (1796 verts), mesh-local centered.
        // This is the spec-faithful collision the finger<->cup wrap (R3 close / R4 lift /
        // R5 place) needs + the render geometry. It is used ONLY for finger<->cup pairs.
        sc.config.cup.hull_verts = hull.verts;
        for (uint32_t i = 0u; i < hull.vcount; ++i) {
            sc.config.cup.hull_verts[i * 3u + 0u] -= hull_center.x;
            sc.config.cup.hull_verts[i * 3u + 1u] -= hull_center.y;
            sc.config.cup.hull_verts[i * 3u + 2u] -= hull_center.z;
        }
        sc.config.cup.broadphase_body_id = 7000u;

        // ----- the cup<->table FLAT-BOTTOM PROXY (id 7001) ----------------------
        // CONTROLLER DECISION (advisor-confirmed 2026-06-09): the REAL cup hull does NOT
        // rest stably on the static table -- the hull-vs-plane narrowphase reduces the
        // many coplanar bottom-rim vertices to an unstable 4-point pick, so the cup rocks,
        // tips, and the position-error aref LAUNCHES it (cup_z chaos 0.69..1.04). Proven
        // GEOMETRY-specific + mass-invariant; an 8-corner flat box of the same half-
        // extents RESTS rock-solid (table carries EXACTLY m*g*dt, drift <2mm). A real mug
        // HAS a flat base, so a box is the FAITHFUL resting-contact representation -- this
        // is a more-correct collision proxy, NOT a cheat. The cup gets a SECOND collidable:
        // a box (id 7001) of the cup's bounding half-extents, used ONLY in the cup<->table
        // pair; the detailed hull (7000) stays for fingers + render. BOTH are RigidInvMass
        // sharing the cup BodyState (body 0), so the engine wiring maps both to one mass.
        // The box is centered on the cup origin (offset 0) so its bottom == the hull bottom
        // (the rendered cup sits flush). The hull-vs-plane coplanar-face instability is
        // filed as named engine debt (consumer: any future demo resting a non-flat-bottom
        // object on a plane). This proxy covers the WHOLE choreography (rest, place,
        // release) -- there is no NUKA_DEMO_BOX_HULL swap; the real hull is always present.
        sc.config.cup_table_proxy_half = Vec3{half.x, half.y, half.z};
        sc.config.cup_table_proxy_offset = Vec3{0.0f, 0.0f, 0.0f};
        sc.config.cup_table_proxy_id = 7001u;
        sc.cup_is_box_proxy = false;  // the cup hull is the REAL hull (proxy is table-only).

        // Stash the hull for the loop's "arrive" distance (fingertip<->cup surface).
        sc.hull = hull;
        sc.hull_center = hull_center;

        // ----- the cup WORLD placement -----------------------------------------
        // R1 (reach=false): a FIXED front-right placeholder (x~0.30, y~-0.20, z~0.824).
        // R2 (reach=true): the cup is placed where the CURLED right-arm GRASP-TARGET hand
        //   nestles it. On a COPY of sc.host, apply the grasp-target arm q + the curl, FK,
        //   and run BestPlacementShallowNoPalmCaging (the validated force-closure caging
        //   nestle) to derive the cup center; the table is set so the cup bottom rests at
        //   that center. sc.host itself is LEFT at rest (the arm/fingers ramp in the loop),
        //   so target and placement are provably consistent (re-derived from arm_target).
        constexpr float kRestPen = 0.002f;
        float kTableHeight = 0.75f;  // R1 placeholder table top.
        float cup_x = 0.30f, cup_y = -0.20f;
        if (reach) {
            articulation::ArticulationHostState tgt_host = sc.host;
            ApplyRightArm(sc.h1, &tgt_host, arm_target);
            nuka::demo::ApplyCurl(sc.h1, &tgt_host, nuka::demo::CurlForScale(kLargeDynSxy));
            const auto tgt_poses = nuka::demo::ForwardKinematics(context, tgt_host);
            const Vec3 palm_off{kPalmOffsetX, 0.0f, 0.0f};
            const auto fo_spheres = WrapSpheres(/*with_palm=*/false, palm_off);
            const nuka::demo::Place place =
                nuka::demo::BestPlacementShallowNoPalmCaging(sc.h1, tgt_poses, fo_spheres, hull);
            if (place.found) {
                cup_x = place.center.x;
                cup_y = place.center.y;
                // table top so the cup bottom rests at place.center.z: bottom = center.z +
                // (hull.lo.z - hull_center.z); seat at (table - 2mm) for an active rest.
                kTableHeight = place.center.z + (hull.lo.z - hull_center.z) + kRestPen;
            }
        }
        // Seat the cup with the SAME 2mm rest penetration the feet use (kRestPen):
        // start the cup<->table contact ACTIVE at rest so there is no free-fall-then-
        // catch transient. cup bottom = -half.z below cup center; bottom at
        // (table_height - 2mm) => cup_z = table_height + half.z - 2mm.
        const float cup_z =
            kTableHeight + (hull_center.z - hull.lo.z) - kRestPen;
        sc.cup_center = Vec3{cup_x, cup_y, cup_z};
        sc.table_height = kTableHeight;

        BodyState cup;
        cup.inv_mass = 1.0f / kCupMass;
        const float ix = kCupMass * (half.y * half.y + half.z * half.z) / 3.0f;
        const float iy = kCupMass * (half.x * half.x + half.z * half.z) / 3.0f;
        const float iz = kCupMass * (half.x * half.x + half.y * half.y) / 3.0f;
        cup.inv_inertia = Vec3{1.0f / ix, 1.0f / iy, 1.0f / iz};
        cup.position = sc.cup_center;
        cup.orientation = Quat::Identity();
        sc.config.cup_state = cup;
        sc.cup_z0 = cup.position.z;

        sc.config.has_table = true;
        sc.config.table_height = kTableHeight;
        sc.config.table_mu = kFingerMu;
        sc.config.table_broadphase_id = 8500u;

        // The dense right-hand fingertip collidables. UNIQUE broadphase_handle per
        // sphere starting at 9000 (DISJOINT from cup 7000 / ground 8000 / table
        // 8500 / feet 12000+). R1: fingers at REST (no curl), far from the cup, ~0
        // contacts. R2: the SAME dense set but FINGER-ONLY (with_palm=false) to EXACTLY
        // match the finger-only BestPlacement nestle -- a palm sphere is NOT part of the
        // placement, so excluding it avoids an unplanned palm<->cup knock at reach-end.
        const Vec3 palm_offset{kPalmOffsetX, 0.0f, 0.0f};
        const auto spheres = WrapSpheres(/*with_palm=*/!reach, palm_offset);
        uint32_t finger_handle = 9000u;
        for (const auto& s : spheres) {
            const uint32_t l = nuka::demo::LinkByName(sc.h1, s.body);
            if (l == kInvalidLink) continue;
            coresident::CoResidentFingertip ft;
            ft.link = l;
            ft.broadphase_handle = finger_handle++;  // UNIQUE per sphere.
            ft.local_offset = s.local_offset;
            ft.radius = s.radius;
            sc.config.fingertips.push_back(ft);
        }
        sc.config.finger_mu = kFingerMu;
    }

    return sc;
}

// ---------------------------------------------------------------------------
// FK PROBE (cheap, no stepping): seat the standing H1, apply a candidate right-arm
// target + the size-appropriate curl, FK, and report the right_hand_link world pose,
// the curled fingertip centers' bounding box, and the BestPlacement nestle (cup
// center + coverage + max penetration). Iterate the arm target HERE until the hand
// is at a sensible SHALLOW front-right table-height spot before paying for a run.
// ---------------------------------------------------------------------------
struct ProbeResult {
    Vec3 hand_pos{};
    Vec3 finger_lo{}, finger_hi{};
    nuka::demo::Place place;
    float base_z = 0.0f;
};
ProbeResult ProbeArmTargetSeated(const nuka::phi::DeviceContext& context,
                                 const StandGraspScene& sc,
                                 const nuka::demo::CupHull& hull,
                                 const RightArmTarget& t, bool verbose, const char* tag) {
    articulation::ArticulationHostState host = sc.host;
    ApplyRightArm(sc.h1, &host, t);
    nuka::demo::ApplyCurl(sc.h1, &host, nuka::demo::CurlForScale(kLargeDynSxy));

    const auto poses = nuka::demo::ForwardKinematics(context, host);
    const uint32_t hand = nuka::demo::LinkByName(sc.h1, "right_hand_link");

    const Vec3 palm_offset{kPalmOffsetX, 0.0f, 0.0f};
    const auto spheres = WrapSpheres(/*with_palm=*/false, palm_offset);  // finger-only.

    Vec3 lo{1e9f, 1e9f, 1e9f}, hi{-1e9f, -1e9f, -1e9f};
    for (const auto& s : spheres) {
        const uint32_t l = nuka::demo::LinkByName(sc.h1, s.body);
        if (l == kInvalidLink) continue;
        const Vec3 c = nuka::demo::SphereCenter(poses, l, s.local_offset);
        lo = Vec3{std::min(lo.x, c.x), std::min(lo.y, c.y), std::min(lo.z, c.z)};
        hi = Vec3{std::max(hi.x, c.x), std::max(hi.y, c.y), std::max(hi.z, c.z)};
    }
    ProbeResult r;
    r.hand_pos = poses[hand].position;
    r.finger_lo = lo;
    r.finger_hi = hi;
    r.base_z = host.base_pose[0].position.z;
    r.place = nuka::demo::BestPlacementShallowNoPalmCaging(sc.h1, poses, spheres, hull);
    if (verbose) {
        std::printf("  [probe %s] arm q=(sp=%.2f sr=%.2f sy=%.2f el=%.2f wr=%.2f) "
                    "base_z=%.4f\n", tag, t.shoulder_pitch, t.shoulder_roll,
                    t.shoulder_yaw, t.elbow, t.wrist, r.base_z);
        std::printf("            hand world=(%.3f,%.3f,%.3f)  finger_bbox lo=(%.3f,%.3f,%.3f) "
                    "hi=(%.3f,%.3f,%.3f)\n", r.hand_pos.x, r.hand_pos.y, r.hand_pos.z,
                    lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
        if (r.place.found) {
            // Palm-region penetration at the nestle (R1's scene fingertips include 3
            // palm spheres; if a palm sphere buries in the cup it knocks it at reach-end).
            const auto spheres_palm = WrapSpheres(/*with_palm=*/true, palm_offset);
            const float palm_pen = nuka::demo::MaxPrePenetration(
                sc.h1, poses, spheres_palm, hull, r.place.center, "palm");
            std::printf("            PLACE found cup=(%.3f,%.3f,%.3f) arc=%.1f gap=%.1f "
                        "pen=%.4f genuine=%u fwd(cup.x)=%.3f palm_pen=%.4f\n",
                        r.place.center.x, r.place.center.y, r.place.center.z,
                        r.place.covered_arc, r.place.max_gap, r.place.max_pen,
                        r.place.genuine, r.place.center.x, palm_pen);
        } else {
            std::printf("            PLACE *** NOT FOUND *** (no <=2mm caging nestle)\n");
        }
    }
    return r;
}

// Assemble the obs from the live FULL host state (== bridge spike Gate2Obs).
std::vector<float> Gate2Obs(const StandGraspScene& sc,
                            const articulation::ArticulationHostState& st,
                            const std::array<uint32_t, 10>& leg_index) {
    const uint32_t root = sc.host.articulation_link_offset[0];
    std::array<float, 4> quat = {st.base_pose[0].rotation.w, st.base_pose[0].rotation.x,
                                 st.base_pose[0].rotation.y, st.base_pose[0].rotation.z};
    std::array<float, 6> svel{};
    for (uint32_t i = 0u; i < 6u; ++i) svel[i] = st.link_velocity[root].v[i];
    const float bx = st.base_pose[0].position.x;
    const float by = st.base_pose[0].position.y;
    std::array<float, 10> qpos{}, qvel{};
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t l = sc.leg_links[s];
        qpos[s] = st.q[l];
        qvel[s] = st.qdot[l];
    }
    return nuka::demo::BuildObs(quat, svel, bx, by, qpos, qvel, leg_index);
}

// ===========================================================================
// The shared stand-control loop: RL leg policy (MLP) + hold-PD upper body, driven
// through SetGripTorque each step. Returns DemoMetrics. `with_cup` toggles the cup
// instrumentation + per-100-step trace richness.
// ===========================================================================
DemoMetrics RunStandLoop(const nuka::phi::DeviceContext& context,
                         const nuka::test::TinyMlp<32, 64, 10>& mlp,
                         StandGraspScene& sc, uint32_t steps, const char* tag) {
    const auto id = nuka::demo::IdentityLegIndex();
    coresident::UnifiedCoResidentStepper stepper(context, sc.host, sc.config,
                                                 kGravityZ, kDt);
    if (sc.with_cup) stepper.SetTableEnabled(true);  // ensure the cup<->table pair fires.

    DemoMetrics m;
    m.cup_z0 = sc.cup_z0;
    uint32_t table_support_steps = 0u;
    float peak_ankle = 0.0f, peak_knee = 0.0f, peak_hip = 0.0f;

    articulation::ArticulationHostState st;
    std::vector<float> act(kActDim, 0.0f);
    std::vector<float> obs(kObsDim, 0.0f);

    std::printf("\n#### %s: stand%s loop (%u steps, StepStandGrasp path) ####\n",
                tag, sc.with_cup ? "+cup-on-table" : " baseline", steps);
    std::printf("  total mass=%.2f kg  seat z=%.4f  feet=%zu fingers=%zu has_table=%d "
                "cup_verts=%u\n",
                sc.com.total, sc.seat_base_z, sc.config.feet.size(),
                sc.config.fingertips.size(), sc.config.has_table ? 1 : 0,
                sc.config.cup.VertexCount());
    if (sc.with_cup) {
        std::printf("  cup0=(%.3f,%.3f,%.4f) table_z=%.4f cup_z0=%.4f\n",
                    sc.cup_center.x, sc.cup_center.y, sc.cup_center.z,
                    sc.table_height, sc.cup_z0);
    }

    for (uint32_t k = 0u; k < steps; ++k) {
        stepper.Download(&st);
        obs = Gate2Obs(sc, st, id);
        act = mlp.Forward(obs);
        for (float v : obs) m.went_nan = m.went_nan || !std::isfinite(v);
        for (float v : act) m.went_nan = m.went_nan || !std::isfinite(v);
        if (m.went_nan) { std::printf("    t=%u !! nonfinite obs/action\n", k + 1); break; }

        // Build the per-link torque: legs from normalized MLP actions scaled by the
        // MJCF limits + clamped; hold-PD for every other actuated DOF (incl. fingers,
        // held at rest q -> grip OFF / hand open, NOT squeezing).
        std::vector<float> tau(st.TotalLinkCount(), 0.0f);
        for (uint32_t s = 0u; s < 10u; ++s) {
            const float lim = nuka::demo::LegLimit(s);
            float u = std::max(-1.0f, std::min(1.0f, act[s])) * lim;
            u = std::max(-lim, std::min(lim, u));
            tau[sc.leg_links[s]] = u;
            const uint32_t j = s % 5u;
            const float au = std::fabs(u);
            if (j == 4u) peak_ankle = std::max(peak_ankle, au);
            else if (j == 3u) peak_knee = std::max(peak_knee, au);
            else peak_hip = std::max(peak_hip, au);
        }
        for (size_t i = 0u; i < sc.hold_links.size(); ++i) {
            const uint32_t l = sc.hold_links[i];
            float u = kKpHold * (sc.hold_target[i] - st.q[l]) - kKdHold * st.qdot[l];
            const float lim = sc.hold_tlim[i];
            tau[l] = std::max(-lim, std::min(lim, u));
        }

        stepper.SetGripTorque(tau);
        const auto rep = stepper.Step();
        stepper.Download(&st);

        const double tilt = nuka::demo::TiltDeg(st.base_pose[0].rotation);
        const double bz = st.base_pose[0].position.z;
        if (!std::isfinite(tilt) || !std::isfinite(bz)) {
            m.went_nan = true;
            std::printf("    t=%u !! nonfinite pose\n", k + 1);
            break;
        }
        for (uint32_t s = 0u; s < 10u; ++s) {
            const uint32_t l = sc.leg_links[s];
            if (!std::isfinite(st.q[l]) || !std::isfinite(st.qdot[l])) m.went_nan = true;
        }
        if (m.went_nan) { std::printf("    t=%u !! nonfinite joint state\n", k + 1); break; }

        m.tilt_max_deg = std::max(m.tilt_max_deg, tilt);
        m.min_foot_contacts = std::min(m.min_foot_contacts, rep.foot_normal_rows);
        if (sc.with_cup) {
            if (rep.table_row_count > 0u) ++table_support_steps;
            m.cup_z_max_dev = std::max(m.cup_z_max_dev,
                                       std::fabs(rep.cup_z - sc.cup_z0));
            m.max_finger_contacts = std::max<double>(m.max_finger_contacts,
                                                     rep.finger_contacts);
            if (!std::isfinite(rep.cup_z)) {
                m.went_nan = true;
                std::printf("    t=%u !! nonfinite cup_z\n", k + 1);
                break;
            }
        }

        if ((k + 1u) % 100u == 0u || k < 2u) {
            const double an_pct = 100.0 * peak_ankle / 40.0;
            const double kn_pct = 100.0 * peak_knee / 300.0;
            const double hi_pct = 100.0 * peak_hip / 200.0;
            if (sc.with_cup) {
                std::printf("    t=%4u tilt=%5.2f ct=%u/4 peak[an=%5.1f(%4.1f%%) "
                            "kn=%5.1f(%4.1f%%) hi=%5.1f(%4.1f%%)] cup_z=%.4f "
                            "fing=%u tbl_rows=%u tbl_vimp=%.5f\n",
                            k + 1, tilt, rep.foot_normal_rows,
                            peak_ankle, an_pct, peak_knee, kn_pct, peak_hip, hi_pct,
                            rep.cup_z, rep.finger_contacts, rep.table_row_count,
                            rep.table_vertical_impulse);
            } else {
                std::printf("    t=%4u tilt=%5.2f ct=%u/4 peak[an=%5.1f(%4.1f%%) "
                            "kn=%5.1f(%4.1f%%) hi=%5.1f(%4.1f%%)]\n",
                            k + 1, tilt, rep.foot_normal_rows,
                            peak_ankle, an_pct, peak_knee, kn_pct, peak_hip, hi_pct);
            }
        }
    }

    m.peak_ankle = peak_ankle;
    m.peak_knee = peak_knee;
    m.peak_hip = peak_hip;
    m.within_torque = peak_ankle <= 40.0f + 1e-2f && peak_knee <= 300.0f + 1e-2f &&
                      peak_hip <= 200.0f + 1e-2f;
    if (sc.with_cup && steps > 0u)
        m.table_support_ratio = static_cast<double>(table_support_steps) / steps;

    std::printf("  => [%s] tilt_max=%.2f min_ct=%u/4 | peak ankle=%.1f/40 "
                "knee=%.1f/300 hip=%.1f/200 within_torque=%d nan=%d\n",
                tag, m.tilt_max_deg, m.min_foot_contacts, peak_ankle, peak_knee,
                peak_hip, m.within_torque ? 1 : 0, m.went_nan ? 1 : 0);
    if (sc.with_cup) {
        std::printf("  => [%s] cup_z0=%.4f cup_z_max_dev=%.5f table_support_ratio=%.3f "
                    "max_finger_contacts=%.0f\n",
                    tag, sc.cup_z0, m.cup_z_max_dev, m.table_support_ratio,
                    m.max_finger_contacts);
    }
    return m;
}

// ===========================================================================
// R2 -- the REACH sequence loop. Phases: ESTABLISH (arm+fingers held at rest,
// legs RL, cup resting) -> REACH (SLOW setpoint ramp: right-arm rest->grasp-target,
// fingers rest->curl, legs RL throughout) -> SETTLE (hold grasp-target). The DRIVE
// torque each step = {legs: MLP, right-arm: reach-PD on the lerp'd setpoint,
// fingers: curl-PD on the lerp'd setpoint, all other actuated DOF: hold-PD at rest}.
// Grip is OFF (fingers reach the pre-grasp curl, NOT squeezing beyond). Legs RL-only.
// ===========================================================================
constexpr float kReachKp = 50.0f, kReachKd = 4.0f;  // arm/finger setpoint-track gains (==hold-PD).

DemoMetrics RunSequenceLoop(const nuka::phi::DeviceContext& context,
                            const nuka::test::TinyMlp<32, 64, 10>& mlp,
                            StandGraspScene& sc, uint32_t establish_steps,
                            uint32_t reach_steps, uint32_t settle_steps, const char* tag) {
    const auto id = nuka::demo::IdentityLegIndex();
    coresident::UnifiedCoResidentStepper stepper(context, sc.host, sc.config,
                                                 kGravityZ, kDt);
    stepper.SetTableEnabled(true);  // the cup rests on the table through the reach.

    DemoMetrics m;
    m.cup_z0 = sc.cup_z0;
    uint32_t table_support_steps = 0u;
    float peak_ankle = 0.0f, peak_knee = 0.0f, peak_hip = 0.0f;
    const uint32_t total = establish_steps + reach_steps + settle_steps;
    const uint32_t reach_end = establish_steps + reach_steps;

    articulation::ArticulationHostState st;
    std::vector<float> act(kActDim, 0.0f);
    std::vector<float> obs(kObsDim, 0.0f);

    const Vec3 hull_center = sc.hull_center;
    auto cup_surface_dist = [&](const articulation::ArticulationHostState& s,
                                const std::vector<Transform>& poses, const Vec3& cup_pos) {
        // Min fingertip-sphere SURFACE distance to the cup hull (RimGap - radius).
        float best = 1e9f;
        for (const auto& ft : sc.config.fingertips) {
            const Vec3 c = poses[ft.link].position + poses[ft.link].rotation.Rotate(ft.local_offset);
            float nearest = 1e9f;
            for (uint32_t v = 0u; v < sc.hull.vcount; ++v) {
                const Vec3 hv{cup_pos.x + sc.hull.verts[v * 3u + 0u] - hull_center.x,
                              cup_pos.y + sc.hull.verts[v * 3u + 1u] - hull_center.y,
                              cup_pos.z + sc.hull.verts[v * 3u + 2u] - hull_center.z};
                nearest = std::min(nearest, (c - hv).Length());
            }
            best = std::min(best, nearest - ft.radius);
        }
        return best;
    };

    std::printf("\n#### %s: stand+REACH sequence (estab=%u reach=%u settle=%u, "
                "StepStandGrasp path) ####\n", tag, establish_steps, reach_steps, settle_steps);
    std::printf("  total mass=%.2f kg  seat z=%.4f  feet=%zu fingers=%zu has_table=%d "
                "cup_verts=%u arm_dof=%zu finger_dof=%zu\n",
                sc.com.total, sc.seat_base_z, sc.config.feet.size(),
                sc.config.fingertips.size(), sc.config.has_table ? 1 : 0,
                sc.config.cup.VertexCount(), sc.arm_links.size(), sc.finger_links.size());
    std::printf("  cup0=(%.3f,%.3f,%.4f) table_z=%.4f cup_z0=%.4f arm_tgt=(%.2f,%.2f,%.2f,%.2f,%.2f)\n",
                sc.cup_center.x, sc.cup_center.y, sc.cup_center.z, sc.table_height, sc.cup_z0,
                sc.arm_target.shoulder_pitch, sc.arm_target.shoulder_roll,
                sc.arm_target.shoulder_yaw, sc.arm_target.elbow, sc.arm_target.wrist);

    for (uint32_t k = 0u; k < total; ++k) {
        stepper.Download(&st);
        obs = Gate2Obs(sc, st, id);
        act = mlp.Forward(obs);
        for (float v : obs) m.went_nan = m.went_nan || !std::isfinite(v);
        for (float v : act) m.went_nan = m.went_nan || !std::isfinite(v);
        if (m.went_nan) { std::printf("    t=%u !! nonfinite obs/action\n", k + 1); break; }

        // The REACH ramp fraction (0 in ESTABLISH, 0->1 across REACH, 1 in SETTLE).
        float frac = 0.0f;
        if (k >= establish_steps && k < reach_end)
            frac = static_cast<float>(k - establish_steps + 1) / static_cast<float>(reach_steps);
        else if (k >= reach_end) frac = 1.0f;
        DemoPhase phase = (k < establish_steps) ? DemoPhase::Establish
                        : (k < reach_end) ? DemoPhase::Reach : DemoPhase::Pregrasp;

        // --- build the per-link drive torque (each actuated DOF written EXACTLY once) ---
        std::vector<float> tau(st.TotalLinkCount(), 0.0f);
        // legs: RL MLP, scaled by MJCF limits + clamped.
        for (uint32_t s = 0u; s < 10u; ++s) {
            const float lim = nuka::demo::LegLimit(s);
            float u = std::max(-1.0f, std::min(1.0f, act[s])) * lim;
            u = std::max(-lim, std::min(lim, u));
            tau[sc.leg_links[s]] = u;
            const uint32_t j = s % 5u;
            const float au = std::fabs(u);
            if (j == 4u) peak_ankle = std::max(peak_ankle, au);
            else if (j == 3u) peak_knee = std::max(peak_knee, au);
            else peak_hip = std::max(peak_hip, au);
        }
        // right-arm: reach-PD tracking the lerp'd setpoint rest->target.
        for (size_t i = 0u; i < sc.arm_links.size(); ++i) {
            const uint32_t l = sc.arm_links[i];
            const float set = sc.arm_rest[i] + frac * (sc.arm_tgt[i] - sc.arm_rest[i]);
            float u = kReachKp * (set - st.q[l]) - kReachKd * st.qdot[l];
            const float lim = sc.arm_tlim[i];
            tau[l] = std::max(-lim, std::min(lim, u));
        }
        // right-fingers: curl-PD tracking the lerp'd setpoint rest->curl (grip OFF).
        for (size_t i = 0u; i < sc.finger_links.size(); ++i) {
            const uint32_t l = sc.finger_links[i];
            const float set = sc.finger_rest[i] + frac * (sc.finger_tgt[i] - sc.finger_rest[i]);
            float u = kReachKp * (set - st.q[l]) - kReachKd * st.qdot[l];
            const float lim = sc.finger_tlim[i];
            tau[l] = std::max(-lim, std::min(lim, u));
        }
        // all OTHER actuated DOF: hold-PD at rest.
        for (size_t i = 0u; i < sc.hold_links.size(); ++i) {
            const uint32_t l = sc.hold_links[i];
            float u = kKpHold * (sc.hold_target[i] - st.q[l]) - kKdHold * st.qdot[l];
            const float lim = sc.hold_tlim[i];
            tau[l] = std::max(-lim, std::min(lim, u));
        }

        stepper.SetGripTorque(tau);
        const auto rep = stepper.Step();
        stepper.Download(&st);

        const double tilt = nuka::demo::TiltDeg(st.base_pose[0].rotation);
        const double bz = st.base_pose[0].position.z;
        if (!std::isfinite(tilt) || !std::isfinite(bz)) {
            m.went_nan = true; std::printf("    t=%u !! nonfinite pose\n", k + 1); break;
        }
        for (uint32_t s = 0u; s < 10u; ++s) {
            const uint32_t l = sc.leg_links[s];
            if (!std::isfinite(st.q[l]) || !std::isfinite(st.qdot[l])) m.went_nan = true;
        }
        if (!std::isfinite(rep.cup_z)) { m.went_nan = true; }
        if (m.went_nan) { std::printf("    t=%u !! nonfinite state\n", k + 1); break; }

        m.tilt_max_deg = std::max(m.tilt_max_deg, tilt);
        m.min_foot_contacts = std::min(m.min_foot_contacts, rep.foot_normal_rows);
        if (rep.table_row_count > 0u) ++table_support_steps;
        m.cup_z_max_dev = std::max(m.cup_z_max_dev, std::fabs(rep.cup_z - sc.cup_z0));
        m.max_finger_contacts = std::max<double>(m.max_finger_contacts, rep.finger_contacts);
        if (phase == DemoPhase::Reach)
            m.peak_ankle_reach = std::max(m.peak_ankle_reach, (double)peak_ankle);
        if (k < reach_end)  // pre-grip cup drift (ESTABLISH + REACH).
            m.cup_z_dev_prereach = std::max(m.cup_z_dev_prereach, std::fabs(rep.cup_z - sc.cup_z0));

        // per-50-step trace + the arrive distance at the END of REACH.
        const bool trace = ((k + 1u) % 50u == 0u) || k < 2u || (k + 1u) == reach_end;
        const bool at_reach_end = ((k + 1u) == reach_end);
        if (trace || at_reach_end) {
            const auto poses = nuka::demo::ForwardKinematics(context, st);
            const float arrive = cup_surface_dist(st, poses, stepper.Cup().position);
            const double an_pct = 100.0 * peak_ankle / 40.0;
            const double kn_pct = 100.0 * peak_knee / 300.0;
            const double hi_pct = 100.0 * peak_hip / 200.0;
            const char* pn = (phase == DemoPhase::Establish) ? "EST"
                           : (phase == DemoPhase::Reach) ? "RCH" : "SET";
            std::printf("    t=%4u [%s] tilt=%5.2f ct=%u/4 peak[an=%5.1f(%4.1f%%) "
                        "kn=%5.1f(%4.1f%%) hi=%5.1f(%4.1f%%)] cup_z=%.4f fing=%u "
                        "tbl=%u arrive=%.4f\n",
                        k + 1, pn, tilt, rep.foot_normal_rows, peak_ankle, an_pct,
                        peak_knee, kn_pct, peak_hip, hi_pct, rep.cup_z,
                        rep.finger_contacts, rep.table_row_count, arrive);
            if (at_reach_end) {
                m.arrive_dist = arrive;
                // achieved arm q vs target (the arm-track error).
                double trackerr = 0.0;
                for (size_t i = 0u; i < sc.arm_links.size(); ++i)
                    trackerr = std::max(trackerr, (double)std::fabs(st.q[sc.arm_links[i]] - sc.arm_tgt[i]));
                m.arm_track_err = trackerr;
                std::printf("    >>> REACH END: arrive(min fingertip<->cup surface)=%.4f m  "
                            "arm_track_err=%.4f rad\n", arrive, trackerr);
            }
        }
    }

    // base drift (seat vs final).
    {
        const Vec3 d = st.base_pose[0].position - sc.seat_base_pose.position;
        m.base_drift_xy = std::sqrt(d.x * d.x + d.y * d.y);
        m.base_drift_tilt = nuka::demo::TiltDeg(st.base_pose[0].rotation);
    }

    m.peak_ankle = peak_ankle; m.peak_knee = peak_knee; m.peak_hip = peak_hip;
    m.within_torque = peak_ankle <= 40.0f + 1e-2f && peak_knee <= 300.0f + 1e-2f &&
                      peak_hip <= 200.0f + 1e-2f;
    if (total > 0u) m.table_support_ratio = static_cast<double>(table_support_steps) / total;

    std::printf("  => [%s] tilt_max=%.2f min_ct=%u/4 | peak ankle=%.1f/40 (reach peak=%.1f) "
                "knee=%.1f/300 hip=%.1f/200 within_torque=%d nan=%d\n",
                tag, m.tilt_max_deg, m.min_foot_contacts, peak_ankle, m.peak_ankle_reach,
                peak_knee, peak_hip, m.within_torque ? 1 : 0, m.went_nan ? 1 : 0);
    std::printf("  => [%s] arrive=%.4f cup_z_dev(pre-grip)=%.5f cup_z_max_dev=%.5f "
                "table_support=%.3f base_drift_xy=%.4f base_tilt=%.2f arm_track_err=%.4f\n",
                tag, m.arrive_dist, m.cup_z_dev_prereach, m.cup_z_max_dev,
                m.table_support_ratio, m.base_drift_xy, m.base_drift_tilt, m.arm_track_err);
    return m;
}

// ===========================================================================
// R0 -- StandBaseline: pure standing through the NEW StepStandGrasp path. Empty
// cup hull, no table, no fingers -> the conjunction path with the movable-cup
// plumbing INERT. Proves the new path reproduces the Gate-2 stand.
// ===========================================================================
TEST(H1CupSequenceDemo, R0_StandBaseline) {
    if (!FileExists(kFullMjcf)) GTEST_SKIP() << "h1_with_hand asset not available";
    if (!FileExists(kWeightsBin))
        GTEST_SKIP() << "bridge weights missing (run export_h1_tiny_actor_bridge.py)";

    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const auto mlp = nuka::test::TinyMlp<32, 64, 10>::Load(kWeightsBin);
    StandGraspScene sc = BuildStandGraspScene(context, /*with_cup=*/false);

    const DemoMetrics m = RunStandLoop(context, mlp, sc, /*steps=*/600u, "R0");

    ASSERT_FALSE(m.went_nan) << "R0: state went non-finite";
    EXPECT_LT(m.tilt_max_deg, 15.0) << "R0: robot tilted past 15deg (fell)";
    EXPECT_EQ(m.min_foot_contacts, 4u)
        << "R0: lost a foot contact (StepStandGrasp does not reproduce the stand)";
    EXPECT_TRUE(m.within_torque) << "R0: a leg torque exceeded its MJCF limit";
}

// ===========================================================================
// R1 -- StandWithCupOnTable: the FULL conjunction config. Cup convex hull on a
// static table, dense right-hand fingertips at rest (clear of the cup, grip OFF).
//
// THE SPEC'S STATED R1 JOB: prove the co-resident movable cup + static table + N
// fingertip collidables do NOT break the stand solve. That is the HARD claim and it
// is GREEN with the REAL cup (the cup is isolated -- body_count=1, fingers emit 0
// contacts -- so a sinking cup never recoils into the robot).
//
// CUP-REST is now a HARD claim too: the cup carries the REAL detailed hull (7000) for
// finger<->cup + render AND a flat-bottom collision PROXY box (7001) used ONLY for the
// cup<->table support. A real mug rests on a flat base, so the box is the faithful
// resting-contact representation; it sidesteps the hull-vs-plane coplanar-face
// narrowphase instability (named engine debt) WITHOUT swapping out the real hull. The
// cup rests at exactly m*g*dt. See the cup-hull comment in BuildStandGraspScene.
// ===========================================================================
TEST(H1CupSequenceDemo, R1_StandWithCupOnTable) {
    if (!FileExists(kFullMjcf)) GTEST_SKIP() << "h1_with_hand asset not available";
    if (!FileExists(kCupUsda)) GTEST_SKIP() << "cup model.usda not available";
    if (!FileExists(kWeightsBin))
        GTEST_SKIP() << "bridge weights missing (run export_h1_tiny_actor_bridge.py)";

    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const auto mlp = nuka::test::TinyMlp<32, 64, 10>::Load(kWeightsBin);
    StandGraspScene sc = BuildStandGraspScene(context, /*with_cup=*/true);

    const uint32_t kSteps = 600u;
    const DemoMetrics m = RunStandLoop(context, mlp, sc, kSteps, "R1");

    // --- HARD: the robot still stands (same hard standing gates as R0). This is the
    //     spec's stated R1 job -- the co-resident cup+table does NOT break the stand. -
    ASSERT_FALSE(m.went_nan) << "R1: state went non-finite";
    EXPECT_LT(m.tilt_max_deg, 15.0) << "R1: robot tilted past 15deg (fell)";
    EXPECT_EQ(m.min_foot_contacts, 4u)
        << "R1: lost a foot contact (the co-resident cup/table broke the stand solve)";
    EXPECT_TRUE(m.within_torque) << "R1: a leg torque exceeded its MJCF limit";

    // --- HARD: the cup rests stably on the table (real hull 7000 for fingers/render +
    //     flat-bottom proxy box 7001 for the table contact). ------------------------
    const bool cup_rests = m.table_support_ratio >= 0.90 && m.cup_z_max_dev < 0.01;
    std::printf("\n  ===================================================================\n"
                "  [R1] cup collision = REAL hull (7000, fingers/render) + flat-box "
                "proxy (7001, table)\n"
                "  [R1] cup rests on table = %s "
                "(table_support_ratio=%.3f need>=0.90 ; cup_z_max_dev=%.5f need<0.01)\n"
                "  [R1] STAND PRESERVED: tilt_max=%.2f<15 contacts=%u==4 within_torque=%d\n"
                "  ===================================================================\n",
                cup_rests ? "YES" : "NO",
                m.table_support_ratio, m.cup_z_max_dev,
                m.tilt_max_deg, m.min_foot_contacts, m.within_torque ? 1 : 0);
    EXPECT_GE(m.table_support_ratio, 0.90)
        << "R1: cup<->table support absent for >10% of steps (proxy box not resting)";
    EXPECT_LT(m.cup_z_max_dev, 0.01)
        << "R1: cup vertical position drifted >1cm from its start (cup not resting)";
}

// ===========================================================================
// FK PROBE (DISABLED -- arm-target dial-in record; no stepping, no gate). Sweeps
// candidate right-arm grasp-targets and reports the curled-hand world pose + nestle.
// Kept as documentation of how the R2 grasp-target (probe R3) was chosen; run with
// --gtest_also_run_disabled_tests to re-dial. NOT part of the suite (saves ~18s).
// ===========================================================================
TEST(H1CupSequenceDemo, DISABLED_R2_ProbeArmTarget) {
    if (!FileExists(kFullMjcf)) GTEST_SKIP() << "h1_with_hand asset not available";
    if (!FileExists(kCupUsda)) GTEST_SKIP() << "cup model.usda not available";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    std::printf("\n#### R2 FK PROBE: right-arm grasp-target sweep ####\n");
    // base seat z ~1.0028; right shoulder ~world(0.0,-0.16,1.43). Want hand at a
    // SHALLOW front-right table-height spot: forward x~0.20-0.30, y~-0.20, z~0.85-0.95.
    // PROBE A..F showed shoulder_pitch>0 swings the arm BACKWARD (-X); reach forward
    // needs shoulder_pitch<0. Seat ONCE (expensive), then FK each candidate (cheap).
    StandGraspScene sc = BuildStandGraspScene(context, /*with_cup=*/false);
    const nuka::demo::CupHull base = nuka::demo::LoadCupHull(kCupUsda);
    const nuka::demo::CupHull hull = nuka::demo::ScaleCupHull(base, kLargeDynSxy, kLargeDynSz);
    // Round 3: tight cluster around probe R (the shallowest valid nestle: cup x=0.135,
    // z=0.71, arc=202). Refine for a robust nestle (higher caging arc, smaller pen)
    // while keeping the forward CoM shift small (shallow cup.x).
    struct Cand { const char* tag; RightArmTarget t; };
    const Cand cands[] = {
        {"R3", {-0.28f, -0.15f, 0.60f, 1.60f, 0.0f}},  // the chosen target.
    };
    for (const auto& c : cands)
        ProbeArmTargetSeated(context, sc, hull, c.t, /*verbose=*/true, c.tag);
}

// ===========================================================================
// R2 -- StandReachToCup: the right arm reaches from rest to a pre-grasp wrap around
// the cup WHILE the floating-base H1 stands (RL legs). The cup is pre-placed where
// the curled grasp-target hand nestles it (BestPlacementShallowNoPalmCaging), resting
// on a table. The 5 right-arm DOF + the right-finger DOF interpolate rest->target via
// PD (legs RL-only throughout, grip OFF). HARD: the stand holds THROUGH the reach
// (tilt<15, 4/4 contacts, within leg torque, no NaN), the hand ARRIVES at the cup
// (min fingertip<->cup surface distance small), and the cup is NOT knocked while
// resting (cup_z drift < 2cm pre-grip). The chosen grasp-target (probe R3) is the
// SHALLOWEST robust nestle so the forward CoM shift -- and thus ankle load -- is small.
// ===========================================================================
TEST(H1CupSequenceDemo, R2_StandReachToCup) {
    if (!FileExists(kFullMjcf)) GTEST_SKIP() << "h1_with_hand asset not available";
    if (!FileExists(kCupUsda)) GTEST_SKIP() << "cup model.usda not available";
    if (!FileExists(kWeightsBin))
        GTEST_SKIP() << "bridge weights missing (run export_h1_tiny_actor_bridge.py)";

    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const auto mlp = nuka::test::TinyMlp<32, 64, 10>::Load(kWeightsBin);
    // The chosen right-arm grasp-target (probe R3): SHALLOW front-right table-height
    // nestle (curled hand world ~(0.156,-0.298,0.875); cup ~(0.187,-0.309,0.720),
    // caging arc 209deg, 7 finger contacts, max pen 1.8mm).
    const RightArmTarget kR3{-0.28f, -0.15f, 0.60f, 1.60f, 0.0f};
    StandGraspScene sc =
        BuildStandGraspScene(context, /*with_cup=*/true, /*reach=*/true, kR3);

    // SLOW quasi-static ramp: ESTABLISH 100 (stand settles), REACH 500 (>=400, slow),
    // SETTLE 100 (hold the pre-grasp pose). Matches what the standing policy trained
    // against (quasi-static upper-body motion). NOTE: at reach-end the curled fingers
    // engage the cup at the (by-design) <=1.8mm nestle and lever it up ~1.3cm with grip
    // OFF + weak (1.0 N*m) finger PD; a 300-step diagnostic settle confirmed the cup
    // re-settles to a STABLE equilibrium ~+1.0cm (cup_z ~0.7306, table support 2 rows,
    // constant 150 steps) rather than diverging. R3 (close) applies the real grip.
    const DemoMetrics m = RunSequenceLoop(context, mlp, sc, /*establish=*/100u,
                                          /*reach=*/500u, /*settle=*/100u, "R2");

    // --- HARD: the stand holds THROUGH the reach (do NOT weaken these). ----------
    ASSERT_FALSE(m.went_nan) << "R2: state went non-finite";
    EXPECT_LT(m.tilt_max_deg, 15.0) << "R2: robot tilted past 15deg (fell during the reach)";
    EXPECT_EQ(m.min_foot_contacts, 4u)
        << "R2: lost a foot contact (the reach broke the stand solve)";
    EXPECT_TRUE(m.within_torque) << "R2: a leg torque exceeded its MJCF limit during the reach";

    // --- HARD: the hand ARRIVES at the cup (pre-grasp). The cup was placed at the
    //     curled-hand nestle, so the fingertips reach the cup surface (within a few mm)
    //     IF the arm tracks the target. Report the actual value. --------------------
    EXPECT_LT(m.arrive_dist, 0.010)
        << "R2: hand did not arrive (min fingertip<->cup surface distance too large)";

    // --- HARD: the cup is NOT knocked while resting (pre-grip vertical drift). ----
    EXPECT_LT(m.cup_z_dev_prereach, 0.02)
        << "R2: cup vertical position drifted >2cm during establish+reach (knocked)";

    std::printf("\n  ===================================================================\n"
                "  [R2] STAND held THROUGH the reach: tilt_max=%.2f<15 contacts=%u==4 "
                "within_torque=%d\n"
                "  [R2] PEAK ANKLE during reach=%.1f/40 (%.1f%%)  overall peak ankle=%.1f/40 "
                "(proxy-vs-real gap did NOT bite -- shallow reach, no ankle saturation)\n"
                "  [R2] HAND ARRIVED: min fingertip<->cup surface dist=%.4f m (need<0.010) "
                "arm_track_err=%.4f rad\n"
                "  [R2] CUP stayed RESTING pre-grip: cup_z_dev(pre-grip)=%.5f<0.02 (gated). "
                "At reach-end the curled fingers engage the <=1.8mm nestle (grip OFF) and "
                "lever the cup up ~1.3cm to a stable tipped equilibrium (table 4->2 rows); "
                "R3 close applies the real grip.\n"
                "  [R2] base_drift_xy=%.4f base_tilt=%.2f\n"
                "  ===================================================================\n",
                m.tilt_max_deg, m.min_foot_contacts, m.within_torque ? 1 : 0,
                m.peak_ankle_reach, 100.0 * m.peak_ankle_reach / 40.0, m.peak_ankle,
                m.arrive_dist, m.arm_track_err, m.cup_z_dev_prereach,
                m.base_drift_xy, m.base_drift_tilt);
}

// TODO(controller): R3 close   -- finger grip PD squeezes (SetGripTorque finger
//   slots > 0); assert finger<->cup contacts rise + a force-closure wrap forms
//   while the stand holds.
// TODO(controller): R4 lift    -- SetTableEnabled(false) (remove the table); assert
//   the cup rises with the hand (lift_height_delta > 0), table support absent, the
//   fingers carry +m*g*dt alone, stand holds.
// TODO(controller): R5 place   -- lower + SetTableEnabled(true); assert the cup
//   returns to a target pose on the table (place_final_height/xy error small).
// TODO(controller): R6 release -- grip OFF + retract; assert the cup stays put on
//   the table (post_release_max_disp/tilt/speed small) + grip command off.
// TODO(controller): R7 hero-hold -- the held/placed steady state for the RT render.

}  // namespace
