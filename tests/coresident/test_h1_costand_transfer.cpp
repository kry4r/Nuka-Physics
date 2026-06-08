// ---------------------------------------------------------------------------
// v0.8 S3 -- the TRANSFER-SMOKE GATE: does the SAME balance controller that stood
// the reduced-DOF H1 in the BATCHED world (python/spikes/h1_balance_gate.py, verified
// Kp_cop=320 stand) ALSO stand the FULL-DOF FLOATING-BASE h1_with_hand in the
// CO-RESIDENT DEPLOY world (the compliant unified spine), through the NEW StepStand()
// path (N foot spheres <-> static ground, condim=3 friction, art<->static rows)?
//
// THE GATE THIS DECIDES: funding a multi-week PPO standing campaign. The batched world
// trains with a PGS-Baumgarte contact + a LUMPED-inertia reduced robot; the deploy world
// runs a compliant velocity-level spine + the FULL articulated H1 (free torso/arms/
// fingers). If the same controller stands in BOTH, the contact-model gap AND the
// lumped->articulated inertia gap are benign and training is safe to fund. If it does
// NOT stand, this is the gate's RED result -- we report the precise failure mechanism
// (contact-stiffness sink / unheld torso / sign / gain) BEFORE spending the training.
//
// STAGED (so a FAIL is attributable, exactly as the Python spike stages it):
//   SEAT  -- structural sanity: a single tau=0 solve at the seated identity-base pose.
//            Asserts the FULL-DOF H1 actually seats on the ground (4 foot<->ground
//            normal rows form, carry positive normal impulse, penetration stays bounded
//            < sphere r) and reports Sigma(lambda_n) vs m*g*dt. (The foot chain-J
//            BASE-COLUMN structure -- col5==1, col0==+lever_y, col1==-lever_x -- is proven
//            to 0.000e+00 on the SAME production floating-base chain-J in
//            test_foot_ground_subsume.cpp; those 6 base columns are robot-agnostic of the
//            leg/joint count, so the H1 prefix-sum/J is validated there + end-to-end here
//            (the seat solve drives the full 51-DOF J) + implicitly by STAGE3 standing.)
//   STAGE1 -- contact-model probe: posture PD on EVERY actuated joint (legs + torso +
//            arms + fingers held at the cooked rest), ankle=0, NO CoP. Logs base-z
//            DRIFT + Sigma(lambda_n)/(m*g*dt) over the window. A slow steady sink here is
//            the compliant-vs-Baumgarte finding (velocity-only solve, baumgarte=0); a
//            stable z + ratio~1 means the contact carries the weight.
//   STAGE2 -- ankle SIGN calibration (+/-5 N*m fixed ankle) in THIS world (the spike
//            flags the sign as empirical; the cooked joint-axis convention may flip it).
//   STAGE3 -- CoP ankle strategy engaged after a settle, swept over Kp_cop (the batched
//            window was narrow ~320). The PASS gate: tilt<15deg, CoM over the polygon,
//            base height stable, feet in contact, all commanded torques within the MJCF
//            physical limits (ankle+/-40, knee+/-300, hip+/-200), no NaN.
//
// ADDITIVE: a NEW translation unit driving the NEW StepStand() path. The W1a Step() +
// the grasp gates are byte-for-byte unchanged.
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kGravityZ = -9.81f;
constexpr float kDt = 0.005f;        // the spike's dt (NOT 1/240).

const std::string kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";

bool AssetAvailable() { return std::filesystem::exists(kH1Mjcf); }

// --- physical MJCF actuator torque limits (N*m, gear=1 -> ctrlrange == torque) ----
struct TorqueLimit { float hip, knee, ankle; };
constexpr TorqueLimit kLimit{200.0f, 300.0f, 40.0f};

// --- the stance the batched gate stood at (hip_pitch, knee, ankle), + CoP gains -----
constexpr float kStanceHipPitch = -0.40f;
constexpr float kStanceKnee = 0.70f;
constexpr float kStanceAnkle = -0.30f;
// posture PD gains (knee/hip + everything-held), torque-mode explicit Kd.
constexpr float kKpHipYaw = 150.0f, kKpHipRoll = 200.0f, kKpHipPitch = 200.0f, kKpKnee = 300.0f;
constexpr float kKdHipYaw = 8.0f,  kKdHipRoll = 12.0f,  kKdHipPitch = 14.0f,  kKdKnee = 18.0f;
constexpr float kKpHold = 50.0f, kKdHold = 4.0f;   // torso/arms/fingers posture hold.
constexpr float kAnkleKd = 1.5f;

// foot polygon: 2 spheres per ankle (toe/heel), the SAME 4-corner polygon the batched
// gate validated. Foot-sphere geometry from the H1 MJCF leg kinematics (the spike's
// _foot_z_under_pose constants): sphere center z = -0.055 below the ankle joint.
constexpr float kFootSphereR = 0.025f;
constexpr float kFootBottomZ = -0.055f;
constexpr float kFootToeX = 0.10f;    // symmetric about the ankle (polygon-centered).
constexpr float kFootHeelX = -0.10f;

// ---------------------------------------------------------------------------
// Cooked FULL-DOF FLOATING-BASE H1 (skip mesh decomposition; do NOT force-fix base).
// ---------------------------------------------------------------------------
struct CookedH1 {
    articulation::ArticulationHostState host;
    nuka::scene::SceneIR scene;
};

CookedH1 LoadH1Floating() {
    CookedH1 out;
    out.scene = nuka::import::LoadMjcf(kH1Mjcf);
    // DODGE the >16-min V-HACD wall: drop mesh geometry (we only need inertia + topo;
    // contact is the analytical foot-sphere we add, not the cooked meshes).
    for (size_t i = 0u; i < out.scene.ShapeCount(); ++i) {
        auto& s = out.scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        s.mesh_vertices.clear();
        s.mesh_indices.clear();
    }
    const auto blob = nuka::scene::CookScene(out.scene);
    auto topos = articulation::CookArticulations(blob);
    // The pelvis has a <freejoint> -> the cooker makes a FloatingBase root. We do NOT
    // force-fix it (the dense-grasp test force-fixes; standing needs the floating base).
    out.host = articulation::BuildArticulationHostState(topos, blob.bodies);
    return out;
}

std::string LinkName(const CookedH1& h1, uint32_t link) {
    if (h1.host.link_body.size() <= link) return "?";
    const uint32_t body = h1.host.link_body[link];
    if (body < h1.scene.Bodies().size()) return h1.scene.GetBody(body).name;
    return "?";
}
uint32_t LinkByName(const CookedH1& h1, const std::string& name) {
    for (uint32_t l = 0u; l < h1.host.TotalLinkCount(); ++l)
        if (LinkName(h1, l) == name) return l;
    return kInvalidLink;
}

std::vector<Transform> ForwardKinematics(const nuka::phi::DeviceContext& context,
                                         const articulation::ArticulationHostState& host) {
    const uint32_t link_count = host.TotalLinkCount();
    auto device = articulation::UploadArticulationState(context, host);
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                               nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, device.View(),
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> poses(link_count);
    pose_buf.CopyToHost(poses.data(), poses.size() * sizeof(Transform));
    return poses;
}

// --- whole-body CoM (world) from per-link FK poses + each link's inertial offset +
// mass (the Energy() pattern: mass = I[21], offset = link_inertial_frame.position).
// Strictly better than the spike's lumped CoM -- the FULL articulated mass. -----------
struct CoMModel {
    std::vector<float> mass;       // per link.
    std::vector<Vec3> com_local;   // per link inertial-frame origin (link frame).
    double total = 0.0;
};
CoMModel BuildCoMModel(const articulation::ArticulationHostState& host) {
    CoMModel m;
    const uint32_t n = host.TotalLinkCount();
    m.mass.resize(n);
    m.com_local.resize(n);
    for (uint32_t l = 0u; l < n; ++l) {
        m.mass[l] = host.link_inertia[l].I[3u * 6u + 3u];
        m.com_local[l] = host.link_inertial_frame[l].position;
        m.total += m.mass[l];
    }
    return m;
}
Vec3 WholeBodyCoM(const CoMModel& m, const std::vector<Transform>& poses) {
    Vec3 acc{0, 0, 0};
    for (uint32_t l = 0u; l < poses.size(); ++l) {
        const Vec3 cw = poses[l].position + poses[l].rotation.Rotate(m.com_local[l]);
        acc += cw * m.mass[l];
    }
    return acc * static_cast<float>(1.0 / m.total);
}

// Base tilt (deg): the angle of the base +Z from world +Z.
double TiltDeg(const Quat& q) {
    const double gz = 1.0 - 2.0 * (static_cast<double>(q.x) * q.x + static_cast<double>(q.y) * q.y);
    return std::acos(std::max(-1.0, std::min(1.0, gz))) * 180.0 / M_PI;
}

// ---------------------------------------------------------------------------
// The standing scene: cooked floating H1, seated at the bent stance, 4 foot spheres,
// ground under the foot bottoms. Returns the StandConfig + the driven-joint sets.
// ---------------------------------------------------------------------------
struct StandScene {
    CookedH1 h1;
    articulation::ArticulationHostState host;     // the SEATED host (base + q set).
    coresident::StandConfig config;
    CoMModel com;
    // driven sets (device link indices).
    std::vector<uint32_t> ankle_links;            // [left_ankle, right_ankle].
    std::vector<uint32_t> posture_links;          // knee/hip (posture PD) device links.
    std::vector<float> posture_kp, posture_kd, posture_target;  // aligned to posture_links.
    std::vector<float> posture_tlim;              // physical torque limit per posture link.
    std::vector<uint32_t> hold_links;             // ALL other actuated joints (held at rest).
    std::vector<float> hold_target;               // cooked q for the held joints.
    std::vector<float> hold_tlim;                 // REAL MJCF torque limit per held joint.
    float poly_cx = 0.0f;                          // foot-polygon centre world x (seated).
    float seat_base_z = 0.0f;
};

// Apply the bent stance to both legs' q (hip_pitch/knee/ankle); hip_yaw/roll = 0.
void ApplyStance(const CookedH1& h1, articulation::ArticulationHostState* host) {
    auto setq = [&](const std::string& nm, float v) {
        const uint32_t l = LinkByName(h1, nm);
        if (l != kInvalidLink) host->q[l] = v;
    };
    for (const char* side : {"left", "right"}) {
        setq(std::string(side) + "_hip_yaw_link", 0.0f);
        setq(std::string(side) + "_hip_roll_link", 0.0f);
        setq(std::string(side) + "_hip_pitch_link", kStanceHipPitch);
        setq(std::string(side) + "_knee_link", kStanceKnee);
        setq(std::string(side) + "_ankle_link", kStanceAnkle);
    }
}

StandScene BuildStandScene(const nuka::phi::DeviceContext& context) {
    StandScene sc;
    sc.h1 = LoadH1Floating();
    sc.host = sc.h1.host;
    // Rest velocities (subsume seats at rest).
    for (auto& v : sc.host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (float& qd : sc.host.qdot) qd = 0.0f;
    sc.host.base_pose[0].rotation = Quat::Identity();

    ApplyStance(sc.h1, &sc.host);
    sc.com = BuildCoMModel(sc.host);

    // Foot spheres: 2 per ankle (toe/heel) on the ankle link.
    const uint32_t l_ankle = LinkByName(sc.h1, "left_ankle_link");
    const uint32_t r_ankle = LinkByName(sc.h1, "right_ankle_link");
    sc.ankle_links = {l_ankle, r_ankle};
    uint32_t handle = 9000u;
    auto add_foot = [&](uint32_t link, float x) {
        coresident::CoResidentFootSphere fs;
        fs.link = link;
        fs.broadphase_handle = handle++;
        fs.local_offset = Vec3{x, 0.0f, kFootBottomZ};
        fs.radius = kFootSphereR;
        sc.config.feet.push_back(fs);
    };
    for (uint32_t a : sc.ankle_links) { add_foot(a, kFootToeX); add_foot(a, kFootHeelX); }

    // Seat the base z so the foot bottoms sit just above the ground (q=0-style consistent
    // seat): place the ground at the lowest foot bottom - r + small rest penetration, and
    // lower the base so the bent feet land there. We compute FK at the current base z,
    // find the lowest foot bottom, then drop the base so it sits on the ground.
    auto poses = ForwardKinematics(context, sc.host);
    float min_bottom = std::numeric_limits<float>::infinity();
    for (const auto& fs : sc.config.feet) {
        const Vec3 c = poses[fs.link].position + poses[fs.link].rotation.Rotate(fs.local_offset);
        min_bottom = std::min(min_bottom, c.z - fs.radius);
    }
    // Target: foot bottoms at z=0 ground with a small rest penetration. Lower the base.
    constexpr float kGroundZ = 0.0f;
    constexpr float kRestPen = 0.002f;
    sc.host.base_pose[0].position.z -= (min_bottom - (kGroundZ - kRestPen));
    sc.seat_base_z = sc.host.base_pose[0].position.z;
    sc.config.ground.height = kGroundZ;
    sc.config.friction_mu = 0.8f;
    sc.config.condim = 3u;

    // The foot-polygon centre world-x (seated): the toe/heel are symmetric about each
    // ankle, so poly_cx = mean ankle world-x + (toe+heel)/2.
    poses = ForwardKinematics(context, sc.host);
    sc.poly_cx = 0.5f * (poses[l_ankle].position.x + poses[r_ankle].position.x)
                 + 0.5f * (kFootToeX + kFootHeelX);

    // --- driven sets ---
    auto kp_kd_for = [&](const std::string& joint, float* kp, float* kd, float* tlim) {
        if (joint == "hip_yaw")   { *kp = kKpHipYaw;   *kd = kKdHipYaw;   *tlim = kLimit.hip; }
        else if (joint == "hip_roll")  { *kp = kKpHipRoll;  *kd = kKdHipRoll;  *tlim = kLimit.hip; }
        else if (joint == "hip_pitch") { *kp = kKpHipPitch; *kd = kKdHipPitch; *tlim = kLimit.hip; }
        else if (joint == "knee")      { *kp = kKpKnee;     *kd = kKdKnee;     *tlim = kLimit.knee; }
    };
    // posture PD legs (knee + hip yaw/roll/pitch -- NOT ankle, the CoP drives that).
    for (const char* side : {"left", "right"}) {
        for (const char* j : {"hip_yaw", "hip_roll", "hip_pitch", "knee"}) {
            const uint32_t l = LinkByName(sc.h1, std::string(side) + "_" + j + "_link");
            if (l == kInvalidLink) continue;
            float kp = 0, kd = 0, tlim = 0;
            kp_kd_for(j, &kp, &kd, &tlim);
            sc.posture_links.push_back(l);
            sc.posture_kp.push_back(kp);
            sc.posture_kd.push_back(kd);
            sc.posture_target.push_back(sc.host.q[l]);  // hold at the stance setpoint.
            sc.posture_tlim.push_back(tlim);
        }
    }
    // HOLD every OTHER actuated DOF (torso + shoulders + elbows + fingers + ankles-are-
    // separate) at its cooked rest q. This is the lumped->articulated gap the gate tests:
    // the full H1's torso/arms are FREE (unlike the batched lumped robot) -> let them hang
    // = a disturbance, so posture-PD-hold them. The root + the foot ankle links + the
    // posture legs are excluded.
    auto is_posture = [&](uint32_t l) {
        return std::find(sc.posture_links.begin(), sc.posture_links.end(), l)
               != sc.posture_links.end();
    };
    auto is_ankle = [&](uint32_t l) {
        return std::find(sc.ankle_links.begin(), sc.ankle_links.end(), l)
               != sc.ankle_links.end();
    };
    // REAL MJCF actuator limits for the held (non-leg) joints, by name substring:
    // torso +/-200, shoulder pitch/roll +/-40, shoulder yaw +/-18, elbow +/-18,
    // fingers (proximal/intermediate/thumb/etc.) +/-1. The within-torque verdict for
    // the HELD set uses THESE (not the +/-200 envelope) -- holding the arms/torso/
    // fingers rigid is how the lumped->articulated gap is neutralized, so it must be
    // done with PHYSICAL torque or the gap is clamped away, not shown benign.
    auto hold_limit_for = [&](const std::string& nm) -> float {
        if (nm.find("torso") != std::string::npos) return 200.0f;
        if (nm.find("elbow") != std::string::npos) return 18.0f;
        if (nm.find("shoulder_yaw") != std::string::npos) return 18.0f;
        if (nm.find("shoulder") != std::string::npos) return 40.0f;   // pitch/roll.
        // Hand joints (index/middle/ring/pinky/thumb proximal/intermediate/distal/yaw/pitch).
        return 1.0f;
    };
    const uint32_t root = sc.host.articulation_link_offset[0];
    for (uint32_t l = 0u; l < sc.host.TotalLinkCount(); ++l) {
        if (l == root) continue;
        // Only joints that actually have a DOF (skip welded/fixed links).
        if (articulation::ArticulationJointDofCount(sc.host.joint_type[l]) == 0u) continue;
        if (is_posture(l) || is_ankle(l)) continue;
        sc.hold_links.push_back(l);
        sc.hold_target.push_back(sc.host.q[l]);
        sc.hold_tlim.push_back(hold_limit_for(LinkName(sc.h1, l)));
    }

    const uint32_t link_count = sc.host.TotalLinkCount();
    sc.config.drive_torque.assign(link_count, 0.0f);
    sc.config.drive_force_limits.assign(link_count, 0.0f);
    return sc;
}

// ---------------------------------------------------------------------------
// The PORTED balance controller: posture PD (legs + hold-set) + ankle CoP strategy,
// all clamped to the physical limits. Returns the per-link clamped torque + the peak
// UNCLAMPED ankle/knee/hip magnitudes (the within-torque verdict reads the unclamped).
// ---------------------------------------------------------------------------
struct ControlOut {
    std::vector<float> tau;          // per device link, clamped.
    float ankle_uncl = 0.0f, knee_uncl = 0.0f, hip_uncl = 0.0f;  // peak |unclamped|.
    float com_ex = 0.0f;             // CoM x - poly_cx.
    bool clamp_ankle = false;
    // HELD set (torso/arms/fingers): the worst over-limit margin (uncl - real_lim) and
    // whether ANY held joint commanded above its REAL physical limit this step.
    float hold_worst_over = 0.0f;    // max(|u| - real_lim) over held joints (<=0 == ok).
    bool hold_exceeds = false;
};
ControlOut ComputeControl(const StandScene& sc, const articulation::ArticulationHostState& st,
                          const std::vector<Transform>& poses, const Vec3& com,
                          const Vec3& com_v, float Kp_cop, float Kd_cop, float ankle_ff,
                          bool cop_on) {
    ControlOut out;
    out.tau.assign(st.TotalLinkCount(), 0.0f);

    auto clamp = [](float v, float lim) { return std::max(-lim, std::min(lim, v)); };

    // posture PD legs.
    for (size_t i = 0u; i < sc.posture_links.size(); ++i) {
        const uint32_t l = sc.posture_links[i];
        const float u = sc.posture_kp[i] * (sc.posture_target[i] - st.q[l])
                        - sc.posture_kd[i] * st.qdot[l];
        out.tau[l] = clamp(u, sc.posture_tlim[i]);
        const float au = std::fabs(u);
        if (sc.posture_tlim[i] == kLimit.knee) out.knee_uncl = std::max(out.knee_uncl, au);
        else out.hip_uncl = std::max(out.hip_uncl, au);
    }
    // HOLD set (torso/arms/fingers) -- posture PD at a gentle gain, clamped to each
    // joint's REAL MJCF limit. Holding them is the lumped->articulated disturbance
    // rejection (the gap this gate exists to prove benign), so it must be done with
    // PHYSICAL torque: we track whether any held command exceeds its real limit (if so,
    // the rigid hold is non-physical -> the gap was clamped away, not shown benign).
    for (size_t i = 0u; i < sc.hold_links.size(); ++i) {
        const uint32_t l = sc.hold_links[i];
        const float u = kKpHold * (sc.hold_target[i] - st.q[l]) - kKdHold * st.qdot[l];
        const float lim = sc.hold_tlim[i];
        out.tau[l] = clamp(u, lim);
        const float over = std::fabs(u) - lim;
        if (over > out.hold_worst_over) out.hold_worst_over = over;
        if (over > 1e-3f) out.hold_exceeds = true;
    }
    // ANKLE: CoP-under-CoM strategy (after settle) else the ff bias.
    out.com_ex = com.x - sc.poly_cx;
    const float evx = com_v.x;
    float tau_ankle = ankle_ff;
    if (cop_on) tau_ankle = Kp_cop * out.com_ex + Kd_cop * evx + ankle_ff;
    for (uint32_t l : sc.ankle_links) {
        const float u = tau_ankle - kAnkleKd * st.qdot[l];  // CoP + small ankle damping.
        out.tau[l] = clamp(u, kLimit.ankle);
        const float au = std::fabs(u);
        out.ankle_uncl = std::max(out.ankle_uncl, au);
        if (au > kLimit.ankle + 1e-3f) out.clamp_ankle = true;
    }
    return out;
}

// ---------------------------------------------------------------------------
// SEAT structural sanity: one tau=0 solve at the seated identity-base pose. Asserts the
// full-DOF H1 seats (4 foot rows, positive normal impulse, bounded depth) and reports
// Sigma(lambda_n) vs m*g*dt. (The chain-J base-column structure is proven on the same
// production floating-base path in test_foot_ground_subsume.cpp; here the seat solve
// exercises the full 51-DOF J end-to-end via the report.)
// ---------------------------------------------------------------------------
TEST(H1CoStandTransfer, SeatStructuralSanityAndContactSupport) {
    if (!AssetAvailable()) GTEST_SKIP() << "h1_with_hand asset not available";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    StandScene sc = BuildStandScene(context);

    ASSERT_EQ(sc.host.joint_type[sc.host.articulation_link_offset[0]],
              articulation::ArticulationJointType::FloatingBase)
        << "the H1 pelvis must cook a FloatingBase root (do NOT force-fix the base)";
    ASSERT_EQ(sc.config.feet.size(), 4u) << "expected 4 foot spheres (toe/heel x 2 ankles)";

    std::printf("[SEAT] total mass=%.2f kg  seat base z=%.4f  poly_cx=%.4f  "
                "dof_stride=%u  link_count=%u  posture=%zu hold=%zu\n",
                sc.com.total, sc.seat_base_z, sc.poly_cx,
                articulation::ArticulationDofCount(sc.host, 0u), sc.host.TotalLinkCount(),
                sc.posture_links.size(), sc.hold_links.size());

    coresident::UnifiedCoResidentStepper stepper(context, sc.host, sc.config, kGravityZ, kDt);
    // tau=0: pure contact support probe (the controller is off).
    const auto rep = stepper.Step();

    const double mg_dt = sc.com.total * static_cast<double>(-kGravityZ) * kDt;
    std::printf("[SEAT] contact: pair_found=%d foot_points=%u normal_rows=%u "
                "max_lambda=%.6f Sigma(lambda_n)=%.6f  m*g*dt=%.6f  ratio=%.4f  depth=%.5f\n",
                rep.pair_found ? 1 : 0, rep.finger_contacts, rep.foot_normal_rows,
                rep.lambda, rep.foot_normal_impulse_sum, mg_dt,
                rep.foot_normal_impulse_sum / std::max(mg_dt, 1e-12), rep.contact_depth);

    // The feet must make contact (all 4) -- a vacuous seat would silently fall.
    EXPECT_TRUE(rep.pair_found) << "no foot<->ground contact at the seat (feet not on ground)";
    EXPECT_EQ(rep.foot_normal_rows, 4u) << "expected 4 foot normal rows (one per sphere)";
    EXPECT_GT(rep.foot_normal_impulse_sum, 0.0) << "feet carried no normal impulse (vacuous)";
    // The contact penetration stays bounded (not exploding).
    EXPECT_LT(rep.contact_depth, kFootSphereR) << "foot penetration exceeded the sphere radius";
}

// ---------------------------------------------------------------------------
// The shared rollout: run the staged balance controller for n_steps and gather metrics.
// ---------------------------------------------------------------------------
struct Rollout {
    bool nan = false;
    double tilt_max = 0.0, tilt_final = 0.0;
    double base_z0 = 0.0, base_z_min = 0.0, base_z_max = 0.0, base_z_final = 0.0;
    double com_ex_max = 0.0;
    double min_edge = 1e9;
    uint32_t min_contacts = 4u;
    uint32_t low_contact_steps = 0u;
    float peak_ankle_uncl = 0.0f, peak_knee_uncl = 0.0f, peak_hip_uncl = 0.0f;
    uint32_t ankle_clamp_steps = 0u;
    double sum_lambda_ratio_last = 0.0;  // Sigma(lambda_n)/(m*g*dt) at the final step.
    double base_z_sink = 0.0;            // base_z0 - base_z_min.
    float hold_worst_over = 0.0f;        // worst held-joint over-limit margin (N*m).
    uint32_t hold_exceed_steps = 0u;     // steps where a held joint exceeded its REAL limit.
};

Rollout RunRollout(const nuka::phi::DeviceContext& context, StandScene& sc,
                   float Kp_cop, float Kd_cop, float ankle_ff, bool cop_on,
                   uint32_t settle_steps, uint32_t n_steps, const char* label,
                   bool verbose) {
    coresident::UnifiedCoResidentStepper stepper(context, sc.host, sc.config, kGravityZ, kDt);
    Rollout r;
    const float half_len = 0.5f * (kFootToeX - kFootHeelX);
    const double mg_dt = sc.com.total * static_cast<double>(-kGravityZ) * kDt;

    articulation::ArticulationHostState st;
    stepper.Download(&st);
    r.base_z0 = st.base_pose[0].position.z;
    r.base_z_min = r.base_z0;
    r.base_z_max = r.base_z0;

    auto poses = ForwardKinematics(context, st);
    Vec3 com_prev = WholeBodyCoM(sc.com, poses);
    Vec3 com_v{0, 0, 0};

    for (uint32_t k = 0u; k < n_steps; ++k) {
        stepper.Download(&st);
        poses = ForwardKinematics(context, st);
        const Vec3 com = WholeBodyCoM(sc.com, poses);
        com_v = (com - com_prev) * (1.0f / kDt);
        com_prev = com;

        const bool engage = cop_on && k >= settle_steps;
        const auto ctl = ComputeControl(sc, st, poses, com, com_v, Kp_cop, Kd_cop,
                                        ankle_ff, engage);
        stepper.SetGripTorque(ctl.tau);
        const auto rep = stepper.Step();

        // metrics (post-step).
        stepper.Download(&st);
        const double tilt = TiltDeg(st.base_pose[0].rotation);
        const double bz = st.base_pose[0].position.z;
        const double exv = ctl.com_ex;
        const double edge = half_len - std::fabs(exv);
        const bool nan_now = !std::isfinite(tilt) || !std::isfinite(bz);
        if (nan_now) { r.nan = true; if (verbose) std::printf("    t=%u !! NaN\n", k + 1); break; }

        r.tilt_max = std::max(r.tilt_max, tilt);
        r.tilt_final = tilt;
        r.base_z_min = std::min(r.base_z_min, bz);
        r.base_z_max = std::max(r.base_z_max, bz);
        r.base_z_final = bz;
        r.com_ex_max = std::max(r.com_ex_max, std::fabs(exv));
        r.min_edge = std::min(r.min_edge, edge);
        r.min_contacts = std::min(r.min_contacts, rep.foot_normal_rows);
        if (rep.foot_normal_rows < 4u) ++r.low_contact_steps;
        r.peak_ankle_uncl = std::max(r.peak_ankle_uncl, ctl.ankle_uncl);
        r.peak_knee_uncl = std::max(r.peak_knee_uncl, ctl.knee_uncl);
        r.peak_hip_uncl = std::max(r.peak_hip_uncl, ctl.hip_uncl);
        if (ctl.clamp_ankle) ++r.ankle_clamp_steps;
        r.hold_worst_over = std::max(r.hold_worst_over, ctl.hold_worst_over);
        if (ctl.hold_exceeds) ++r.hold_exceed_steps;
        r.sum_lambda_ratio_last = rep.foot_normal_impulse_sum / std::max(mg_dt, 1e-12);

        if (verbose && ((k + 1) % 100u == 0u || k < 6u)) {
            std::printf("    t=%4u tilt=%5.2f z=%+.4f com_ex=%+.4f edge=%+.4f "
                        "contacts=%u/4 |tau_an|=%5.1f(uncl) Sl/mgdt=%.3f\n",
                        k + 1, tilt, bz, exv, edge, rep.foot_normal_rows,
                        ctl.ankle_uncl, r.sum_lambda_ratio_last);
        }
    }
    r.base_z_sink = r.base_z0 - r.base_z_min;
    std::printf("  => [%s] tilt_max=%.2f tf=%.2f z %.4f->%.4f (min %.4f, sink %.4f) "
                "com_ex_max=%.4f min_edge=%+.4f min_ct=%u/4 low_ct_steps=%u | "
                "peak uncl ankle=%.1f/40 knee=%.1f/300 hip=%.1f/200 ankle_clamp=%u | "
                "held_over=%.2f exceed_steps=%u nan=%d\n",
                label, r.tilt_max, r.tilt_final, r.base_z0, r.base_z_final, r.base_z_min,
                r.base_z_sink, r.com_ex_max, r.min_edge, r.min_contacts, r.low_contact_steps,
                r.peak_ankle_uncl, r.peak_knee_uncl, r.peak_hip_uncl, r.ankle_clamp_steps,
                r.hold_worst_over, r.hold_exceed_steps, r.nan ? 1 : 0);
    return r;
}

// ---------------------------------------------------------------------------
// STAGE1: the CONTACT-MODEL probe. Posture PD everything, ankle=0, NO CoP. Measure
// base-z drift + Sigma(lambda_n)/(m*g*dt). This isolates the compliant-vs-Baumgarte
// support question from the controller.
// ---------------------------------------------------------------------------
TEST(H1CoStandTransfer, Stage1ContactModelProbe) {
    if (!AssetAvailable()) GTEST_SKIP() << "h1_with_hand asset not available";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    StandScene sc = BuildStandScene(context);
    std::printf("\n#### STAGE1: contact-model probe (posture PD all, ankle=0, no CoP) ####\n");
    const auto r = RunRollout(context, sc, 0.0f, 0.0f, 0.0f, /*cop_on=*/false,
                              /*settle=*/0u, /*n_steps=*/200u, "STAGE1", /*verbose=*/true);
    std::printf("[STAGE1] final Sigma(lambda_n)/(m*g*dt)=%.4f  base-z sink=%.4f m\n",
                r.sum_lambda_ratio_last, r.base_z_sink);
    // The seat must not NaN and must keep the feet in contact for the probe to be valid.
    EXPECT_FALSE(r.nan) << "STAGE1 went NaN -- the seat/contact is unstable";
    // REPORT (not a PASS gate): the sink rate + lambda ratio ARE the contact-model finding.
    SUCCEED();
}

// ---------------------------------------------------------------------------
// STAGE2: ankle SIGN calibration (+/-5 N*m fixed ankle). Which sign drives com_ex
// toward the polygon centre? (The spike's sign is empirical; recalibrate in THIS world.)
// ---------------------------------------------------------------------------
TEST(H1CoStandTransfer, Stage2AnkleSignCalibration) {
    if (!AssetAvailable()) GTEST_SKIP() << "h1_with_hand asset not available";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    std::printf("\n#### STAGE2: ankle sign calibration (+/-5 N*m fixed ankle) ####\n");
    for (float ff : {+5.0f, -5.0f}) {
        StandScene sc = BuildStandScene(context);
        char lbl[32]; std::snprintf(lbl, sizeof(lbl), "STAGE2 ff=%+.0f", ff);
        RunRollout(context, sc, 0.0f, 0.0f, ff, /*cop_on=*/false, /*settle=*/0u,
                   /*n_steps=*/120u, lbl, /*verbose=*/false);
    }
    SUCCEED();
}

// ---------------------------------------------------------------------------
// STAGE3: the GATE. CoP ankle strategy engaged after a settle, swept over Kp_cop. The
// PASS criterion: tilt<15deg over the whole rollout, CoM over the polygon (edge>0),
// base height stable, feet in contact, all commanded torques within physical limits,
// no NaN. A within-torque clean stand at ANY swept gain == the transfer-smoke PASS.
// ---------------------------------------------------------------------------
TEST(H1CoStandTransfer, Stage3BalanceGate) {
    if (!AssetAvailable()) GTEST_SKIP() << "h1_with_hand asset not available";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    std::printf("\n#### STAGE3: CoP ankle strategy (settle 15, sweep Kp_cop) ####\n");

    struct Gain { float kpc, kdc; };
    const Gain sweep[] = {{200.f, 30.f}, {255.f, 40.f}, {320.f, 50.f}, {350.f, 55.f},
                          {450.f, 60.f}, {600.f, 70.f}};
    const uint32_t kSteps = 1000u;  // 5 s at dt=0.005.

    bool any_clean_stand = false;
    bool any_stand = false;        // upright (maybe over-torque).
    float best_gain = 0.0f;
    Rollout best{};
    for (const auto& g : sweep) {
        StandScene sc = BuildStandScene(context);
        char lbl[48]; std::snprintf(lbl, sizeof(lbl), "STAGE3 Kcop%.0f", g.kpc);
        const auto r = RunRollout(context, sc, g.kpc, g.kdc, 0.0f, /*cop_on=*/true,
                                  /*settle=*/15u, kSteps, lbl, /*verbose=*/(g.kpc == 320.f));

        const bool upright = !r.nan && r.tilt_max < 15.0 && r.tilt_final < 15.0;
        const bool height_stable = std::fabs(r.base_z_final - r.base_z0) < 0.12
                                   && r.base_z_sink < 0.12;
        const bool com_in_poly = r.min_edge > 0.0;
        const bool contacts_held = r.min_contacts >= 2u;  // transient dip allowed.
        const bool within_torque = r.peak_ankle_uncl <= kLimit.ankle + 1e-2f
                                   && r.peak_knee_uncl <= kLimit.knee + 1e-2f
                                   && r.peak_hip_uncl <= kLimit.hip + 1e-2f
                                   && r.hold_exceed_steps == 0u;  // held arms/torso/fingers
                                                                   // held with PHYSICAL torque.
        const bool stood = upright && height_stable && com_in_poly && contacts_held;
        std::printf("     [verdict Kcop%.0f] stood=%d within_torque=%d (upright=%d "
                    "height_stable=%d com_in_poly=%d contacts=%d)\n",
                    g.kpc, stood ? 1 : 0, within_torque ? 1 : 0, upright ? 1 : 0,
                    height_stable ? 1 : 0, com_in_poly ? 1 : 0, contacts_held ? 1 : 0);
        if (stood) { any_stand = true; }
        if (stood && within_torque) {
            any_clean_stand = true;
            // Headline the batched gain (Kp_cop=320) when it cleanly stands -- the
            // apples-to-apples transfer comparison; else the first clean gain.
            if (best_gain == 0.0f || g.kpc == 320.0f) { best = r; best_gain = g.kpc; }
        }
    }

    if (any_clean_stand) {
        std::printf("\n[GATE VERDICT] PASS -- the SAME balance controller STANDS the FULL-DOF "
                    "floating-base H1 in the co-resident DEPLOY world. Headline gain Kp_cop=%.0f: "
                    "tilt_max=%.2f deg, base-z sink=%.4f m, min_edge=%+.4f m, min_contacts=%u/4, "
                    "peak ankle=%.1f/40, knee=%.1f/300, hip=%.1f/200; HELD arms/torso/fingers "
                    "within REAL limits (exceed_steps=%u, worst_over=%.2f N*m).\n",
                    best_gain, best.tilt_max, best.base_z_sink, best.min_edge,
                    best.min_contacts, best.peak_ankle_uncl, best.peak_knee_uncl,
                    best.peak_hip_uncl, best.hold_exceed_steps, best.hold_worst_over);
    } else if (any_stand) {
        std::printf("\n[GATE VERDICT] STANDS-but-EXCEEDS-TORQUE -- upright but no within-torque "
                    "gain found in the sweep. Report the gain delta vs batched Kp_cop=320.\n");
    } else {
        std::printf("\n[GATE VERDICT] FAIL -- the controller does NOT stand the full-DOF H1 in "
                    "the deploy world. See STAGE1 (contact-model sink), the tilt/CoM/contact "
                    "numbers above, and the peak torques to attribute the mechanism.\n");
    }

    // The GATE assertion: a within-torque clean stand exists at SOME swept gain. An honest
    // FAIL here (with the staged diagnostics above) is the gate's RED result.
    EXPECT_TRUE(any_clean_stand)
        << "no within-torque clean stand found across the Kp_cop sweep -- the transfer-smoke "
           "gate is RED. Read STAGE1 (compliant-vs-Baumgarte sink), STAGE2 (ankle sign), and "
           "the per-gain verdicts above for the failure mechanism.";
}

}  // namespace
