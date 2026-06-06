// ---------------------------------------------------------------------------
// v0.8 C7b-1 -- the HOLD-AGAINST-GRAVITY grasp SPIKE. VALIDATED-NOT-WIRED.
// ---------------------------------------------------------------------------
// THE ONE THING THIS PROVES. A grasp "hold" gate is GREEN-WASHED if the object
// still rests on a table -- the table holds it, the fingers do nothing, every gate
// passes while the grasp is untested (the W1a floor-clamp lesson, 2nd verse). So
// this spike REMOVES THE TABLE and proves the cup is HELD AGAINST GRAVITY BY FINGER
// FRICTION ALONE: a 2-finger gripper pre-posed in contact with the cup's outer
// walls, a constant grip force squeezing inward, the cup under gravity (-z), NO
// table/support of any kind.
//
// THE SETUP (minimal -- isolate the physics):
//   * a SYNTHETIC fixed-base 2-finger gripper (a fixed base link + 2 opposing
//     PRISMATIC finger links along +/-X, each with a fingertip SPHERE), built the
//     cheapest correct way through ArticulationCookedTopology -> BuildArticulation-
//     HostState (the SAME path the W1a single-link floating base uses). NOT the real
//     h1_with_hand (that + finger-collision-enablement is the showcase step C7b-2).
//   * the CUP = a RigidBody whose collision shape is the CONVEX HULL of the C7a cup
//     (loaded via the C7a usdc reader -> SceneIR -> cooked to a single ConvexHull;
//     the cup's outer wall is what the fingers pinch). Narrowphase = fingertip
//     (sphere) x cup (convex hull) -> the C3c GJK/EPA Convex tier.
//   * PRE-POSE the fingers already in contact with the cup walls (no approach
//     phase). Apply a constant grip force (fingers squeeze inward). Gravity on the
//     cup (-z). NO TABLE.
//
// THE DRIVE PATH (Step Zero -- proven LIVE by GraspDriveProbe below). The fingers
// are driven by a constant grip FORCE through the production control->tau path:
// UnifiedCoResidentStepper::StepGrasp re-applies the grip torque to state.tau via
// LaunchApplyTorqueDriveKernels BEFORE FeatherstoneAba::ComputeAccelerations (which
// reads tau in its bias-force pass). W1a ran PASSIVE (tau held at 0); a grasp needs
// ACTIVE grip torque through the articulation drive path -- this is it.
//
// THE DISCRIMINATING GATE (GraspHoldsCupAgainstGravity):
//   * HELD: cup vertical position stays ~stable (NOT free-falling), no slip out.
//   * Sigma(finger contact vertical impulse) ~= m_cup*g*dt each step (the friction/
//     contact the fingers deliver balances the cup's weight), asserted two ways
//     that must agree (row-impulse sum vs cup-velocity-change*mass).
//   * NO table contact exists (NO StaticNull row, NO table body at all).
//   * D1 (2-run byte-exact on the full rollout) + V2 energy non-increasing.
//
// THE BITE PROOF (GraspGripOffCupFalls -- mandatory anti-green-wash): zero the grip
// torque -> friction collapses -> the cup FALLS (~free-fall) -> the hold FAILS. A
// hold that survives grip=0 is fake (something else holds the cup).
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"
#include "runtime/rigid/body_state.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

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

// The cup mass (a fixed small mass -- sane for a few-cm cup). The cup's outer wall
// is what the fingers pinch; the cavity is irrelevant to a convex hull.
constexpr float kCupMass = 0.2f;  // 0.2 kg.
constexpr float kCupInvMass = 1.0f / kCupMass;

// The newton-assets cup (a fetch-per-env asset; .nuka-assets is gitignored).
const std::string kModelUsda =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";

bool CupAvailable() { return std::filesystem::exists(kModelUsda); }

// Load the C7a cup -> a single cooked ConvexHull -> its mesh-local hull vertices +
// bbox. The fingers pinch the cup's outer wall; the hull is the collidable.
struct CupHull {
    std::vector<float> verts;  // flat x,y,z triples, mesh-local.
    Vec3 lo{}, hi{};           // mesh-local bbox.
    uint32_t vcount = 0u;
};
CupHull LoadCupHull() {
    auto scene = nuka::import::LoadUsd(kModelUsda);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& s = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        if (!s.mesh_vertices.empty())
            s.decompose_mode = nuka::scene::DecomposeMode::Skip;  // single hull.
    }
    const auto blob = nuka::scene::CookScene(scene);
    CupHull out;
    const auto& cg = blob.convex_geometry;
    if (cg.vertex_counts.empty()) return out;
    const uint32_t voff = cg.vertex_offsets[0];
    const uint32_t vcnt = cg.vertex_counts[0];
    out.vcount = vcnt;
    out.verts.assign(cg.vertices.begin() + voff * 3u,
                     cg.vertices.begin() + (voff + vcnt) * 3u);
    out.lo = Vec3{out.verts[0], out.verts[1], out.verts[2]};
    out.hi = out.lo;
    for (uint32_t i = 0u; i < vcnt; ++i) {
        const Vec3 v{out.verts[i * 3u], out.verts[i * 3u + 1u], out.verts[i * 3u + 2u]};
        out.lo = Vec3{std::min(out.lo.x, v.x), std::min(out.lo.y, v.y), std::min(out.lo.z, v.z)};
        out.hi = Vec3{std::max(out.hi.x, v.x), std::max(out.hi.y, v.y), std::max(out.hi.z, v.z)};
    }
    return out;
}

// ---------------------------------------------------------------------------
// The synthetic FIXED-base 2-finger gripper.
// ---------------------------------------------------------------------------
// Topology (3 links, local order = device order; parent-local-index < child):
//   link 0: the FIXED base, anchored at world `base_pos` (root, no DOF).
//   link 1: the +X finger -- a PRISMATIC joint along +X (q1 slides it toward -X to
//           pinch when grip force is NEGATIVE), parented to the base.
//   link 2: the -X finger -- a PRISMATIC joint along -X (q2 slides it toward +X to
//           pinch when grip force is NEGATIVE), parented to the base.
// Each finger has a fingertip SPHERE at a local offset; the spheres face the cup.
// dof_stride = 2 (two prismatic DOFs; the fixed base contributes 0).
//
// GRIP: a constant inward force on each prismatic DOF. The prismatic joint axis is
// +X for finger 1 and -X for finger 2; a NEGATIVE tau on finger 1 (push -X = inward
// toward the cup) and a NEGATIVE tau on finger 2 (push -(-X)=+X... -> see signs).
// We instead choose both axes pointing TOWARD the cup center so a POSITIVE tau =
// squeeze. Finger 1 axis = -X (so +q slides toward the cup from the +X side); finger
// 2 axis = +X (so +q slides toward the cup from the -X side). A POSITIVE grip tau on
// both pushes both fingers inward.
struct Gripper {
    articulation::ArticulationHostState host;
    uint32_t finger_link[2];      // device link indices of the two fingers.
    Vec3     fingertip_local[2];  // sphere center in the finger link frame.
    float    fingertip_radius = 0.008f;
};

// A diagonal spatial inertia (mass at the origin, isotropic small inertia).
articulation::LinkSpatialInertia MakeInertia(float mass, float i) {
    return articulation::MakeDiagonalSpatialInertia(mass, Vec3{i, i, i});
}

Gripper BuildGripper(const Vec3& base_pos, float cup_half_x, float fingertip_radius) {
    articulation::ArticulationCookedTopology topo;
    topo.root_body = 0u;
    topo.link_bodies = {0u, 1u, 2u};
    topo.parent_links = {kInvalidLink, 0u, 0u};  // base root; both fingers parent=base.
    topo.joint_types = {articulation::ArticulationJointType::Fixed,
                        articulation::ArticulationJointType::Prismatic,
                        articulation::ArticulationJointType::Prismatic};
    // Finger 1 (from +X side): axis -X -> +q slides toward the cup (inward).
    // Finger 2 (from -X side): axis +X -> +q slides toward the cup (inward).
    topo.joint_axes = {Vec3::UnitX(), Vec3{-1.0f, 0.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}};
    topo.parent_frames = {Transform::Identity(), Transform::Identity(),
                          Transform::Identity()};
    topo.child_frames = {Transform::Identity(), Transform::Identity(),
                         Transform::Identity()};
    // local_poses: the base sits at base_pos; the fingers start at the base origin
    // (their prismatic slide + fingertip offset position the sphere at the cup wall).
    Transform base_lp = Transform::Identity();
    base_lp.position = base_pos;
    topo.local_poses = {base_lp, Transform::Identity(), Transform::Identity()};
    topo.inertial_frames = {Transform::Identity(), Transform::Identity(),
                            Transform::Identity()};
    topo.initial_positions = {0.0f, 0.0f, 0.0f};  // fingers pre-posed via fingertip offset.
    topo.joint_dampings = {0.0f, 0.0f, 0.0f};
    topo.joint_armatures = {0.0f, 0.0f, 0.0f};
    topo.masses = {1.0f, 0.05f, 0.05f};
    topo.inertias = {Vec3{1e-2f, 1e-2f, 1e-2f}, Vec3{1e-4f, 1e-4f, 1e-4f},
                     Vec3{1e-4f, 1e-4f, 1e-4f}};

    nuka::scene::CookedBodyTable bodies;
    Transform base_pose = Transform::Identity();
    base_pose.position = base_pos;
    bodies.poses = {base_pose, Transform::Identity(), Transform::Identity()};
    bodies.local_poses = {base_lp, Transform::Identity(), Transform::Identity()};
    bodies.inertial_frames = {Transform::Identity(), Transform::Identity(),
                              Transform::Identity()};
    bodies.masses = {1.0f, 0.05f, 0.05f};
    bodies.inertias = {Vec3{1e-2f, 1e-2f, 1e-2f}, Vec3{1e-4f, 1e-4f, 1e-4f},
                       Vec3{1e-4f, 1e-4f, 1e-4f}};
    bodies.is_static = {0u, 0u, 0u};

    Gripper g;
    g.host = articulation::BuildArticulationHostState({topo}, bodies);
    g.finger_link[0] = 1u;
    g.finger_link[1] = 2u;
    g.fingertip_radius = fingertip_radius;
    // The fingertip spheres sit at the cup outer wall in +/-X. The finger link frame
    // is at the base origin (q=0); the fingertip offset reaches the cup wall minus a
    // pre-pose penetration so the contact is LIVE from step 0 (no approach phase). The
    // cup is centered under the base; cup wall at +/- cup_half_x.
    //
    // SHALLOW PRE-POSE (C7b-1): a realistic 1.5 mm penetration so the contact is LIVE
    // from step 0 (no approach phase). This holds STABLY because sphere x ConvexHull
    // narrowphase is robust at ALL depths via the C3 SphereHull closest-point special-
    // case (cvx::SphereHull, src/collision/convex_narrowphase.hpp + narrowphase_dispatch
    // .hpp), which bypasses EPA. Before that fix the general GJK/EPA path had a shallow-
    // penetration DETECTION HOLE on the irregular cup hull (non-monotonic in depth:
    // detect ~0.4mm, DROP across a ~1.6-2.8mm band, re-detect ~2.87mm); a contact
    // drifting into that band flickered 1<->2 and limit-cycled the impulse 8.17e-3<->
    // 1.23e-2 (balance err 0.52). The fix makes detection monotone, so this shallow
    // pre-pose -- the experiment we mean to run -- gives a stable symmetric 2-contact
    // grasp (balance err ~0). Locked by GjkEpaConvex.SphereHullShallowPenetrationIsMono-
    // tone (the EPA dead-band regression that BITES with the fix reverted).
    const float kPenetration = 0.0015f;  // 1.5 mm shallow pre-pose -> contact live @ step 0.
    const float reach = cup_half_x + fingertip_radius - kPenetration;
    g.fingertip_local[0] = Vec3{reach, 0.0f, 0.0f};   // +X finger reaches -toward 0.
    g.fingertip_local[1] = Vec3{-reach, 0.0f, 0.0f};  // -X finger reaches +toward 0.
    return g;
}

// ---------------------------------------------------------------------------
// Assemble the grasp config: gripper + cup hull + constant grip force + friction.
// The cup is centered at `cup_center` (under the base, at the fingertip height); the
// fingertips pinch its +/-X walls.
// ---------------------------------------------------------------------------
struct GraspScene {
    Gripper gripper;
    coresident::GraspConfig config;
};
GraspScene BuildGraspScene(const CupHull& hull, float grip_force, float mu) {
    // The cup mesh-local bbox -> world half extents (cup at identity orientation).
    const Vec3 half = (hull.hi - hull.lo) * 0.5f;
    const Vec3 cup_local_center = (hull.hi + hull.lo) * 0.5f;
    // Place the cup so its CENTER is at the fingertip height (the base just above).
    // Base at z=0.30; cup center at z=0.20 (the fingers reach down to the cup walls).
    const Vec3 cup_center{0.0f, 0.0f, 0.20f};
    const Vec3 base_pos{0.0f, 0.0f, 0.30f};

    GraspScene gs;
    gs.gripper = BuildGripper(base_pos, half.x, 0.008f);
    // The finger links are at the base origin (q=0); the fingertip world center is
    // base_pos + finger_local_pose + fingertip_offset. Both fingers' z must reach the
    // cup center height -> set the fingertip local offset z to (cup_center.z - base.z).
    const float drop = cup_center.z - base_pos.z;  // negative (cup below the base).
    gs.gripper.fingertip_local[0].z += drop;
    gs.gripper.fingertip_local[1].z += drop;

    // Build the fingertip descriptors.
    coresident::CoResidentFingertip ft0;
    ft0.link = gs.gripper.finger_link[0];
    ft0.broadphase_handle = gs.gripper.finger_link[0];
    ft0.local_offset = gs.gripper.fingertip_local[0];
    ft0.radius = gs.gripper.fingertip_radius;
    coresident::CoResidentFingertip ft1;
    ft1.link = gs.gripper.finger_link[1];
    ft1.broadphase_handle = gs.gripper.finger_link[1];
    ft1.local_offset = gs.gripper.fingertip_local[1];
    ft1.radius = gs.gripper.fingertip_radius;
    gs.config.fingertips = {ft0, ft1};

    // The cup: a movable rigid body whose collision shape is the convex hull. We
    // center the hull verts about the cup COM by shifting them by -cup_local_center
    // so the cup BodyState.position IS the geometric center (the hull view's frame
    // translation == cup_center). Diagonal box-like inertia from the cup half extents.
    gs.config.cup.hull_verts = hull.verts;
    for (uint32_t i = 0u; i < hull.vcount; ++i) {
        gs.config.cup.hull_verts[i * 3u + 0u] -= cup_local_center.x;
        gs.config.cup.hull_verts[i * 3u + 1u] -= cup_local_center.y;
        gs.config.cup.hull_verts[i * 3u + 2u] -= cup_local_center.z;
    }
    gs.config.cup.broadphase_body_id = 7000u;

    BodyState cup;
    cup.inv_mass = kCupInvMass;
    // Solid-box-equivalent diagonal inertia from the cup extents (a sane value; the
    // cavity is irrelevant to the hold-against-gravity proof). I_axis = m*(a^2+b^2)/3
    // for half-extents a,b about the third axis -> inv.
    const float ix = kCupMass * (half.y * half.y + half.z * half.z) / 3.0f;
    const float iy = kCupMass * (half.x * half.x + half.z * half.z) / 3.0f;
    const float iz = kCupMass * (half.x * half.x + half.y * half.y) / 3.0f;
    cup.inv_inertia = Vec3{1.0f / ix, 1.0f / iy, 1.0f / iz};
    cup.position = cup_center;
    cup.orientation = Quat::Identity();
    cup.linear_velocity = Vec3::Zero();
    cup.angular_velocity = Vec3::Zero();
    gs.config.cup_state = cup;

    // The constant grip force: a POSITIVE tau on both prismatic finger DOFs squeezes
    // inward (both finger axes point toward the cup center). The device link layout:
    // link 0 = fixed base (tau ignored), link 1/2 = the fingers. grip_torque is
    // indexed by DEVICE LINK.
    const uint32_t link_count = gs.gripper.host.TotalLinkCount();
    gs.config.grip_torque.assign(link_count, 0.0f);
    gs.config.grip_torque[gs.gripper.finger_link[0]] = grip_force;
    gs.config.grip_torque[gs.gripper.finger_link[1]] = grip_force;
    gs.config.drive_force_limits.assign(link_count, 0.0f);  // no clamp.
    gs.config.friction_mu = mu;
    gs.config.condim = 3u;  // normal + 4-edge friction pyramid (friction holds it).
    return gs;
}

}  // namespace

// ===========================================================================
// STEP ZERO -- prove the DRIVE PATH is live (the hard blocker the spike needs).
// ===========================================================================
// Grip force ON, NO cup. The constant grip tau must drive the fingers INWARD through
// the production control->tau->ABA path (LaunchApplyTorqueDriveKernels -> tau ->
// ComputeAccelerations). After a few steps the finger prismatic q must have moved in
// the driven (inward, +q) direction and the finger qdot must be nonzero. A passive
// (tau=0) stepper -- the W1a default -- would leave the fingers at rest.
TEST(GraspDriveProbe, ConstantGripTorqueDrivesFingersInward) {
    if (!CupAvailable()) GTEST_SKIP() << "newton-assets cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);

    // Build the scene but EMPTY the cup geometry so there is no contact -- the
    // fingers are free to accelerate under the grip force alone.
    GraspScene gs = BuildGraspScene(hull, /*grip_force=*/2.0f, /*mu=*/0.6f);
    gs.config.cup.hull_verts.clear();  // no cup geometry -> no contact.

    coresident::UnifiedCoResidentStepper stepper(context, gs.gripper.host, gs.config,
                                                 kGravityZ, kDt);
    articulation::ArticulationHostState before;
    stepper.Download(&before);
    const float q0_f0 = before.q[gs.gripper.finger_link[0]];
    const float q0_f1 = before.q[gs.gripper.finger_link[1]];

    for (uint32_t s = 0u; s < 10u; ++s) stepper.Step();

    articulation::ArticulationHostState after;
    stepper.Download(&after);
    const float dq_f0 = after.q[gs.gripper.finger_link[0]] - q0_f0;
    const float dq_f1 = after.q[gs.gripper.finger_link[1]] - q0_f1;
    const float qd_f0 = after.qdot[gs.gripper.finger_link[0]];
    const float qd_f1 = after.qdot[gs.gripper.finger_link[1]];
    std::printf("[STEP0] grip drives fingers: dq=(%.5f, %.5f) m  qdot=(%.5f, %.5f) m/s "
                "(both must be > 0 = driven inward)\n", dq_f0, dq_f1, qd_f0, qd_f1);

    // A POSITIVE grip tau on each prismatic finger (axis toward the cup) drives +q
    // (inward). The drive path is LIVE iff q moved in the +q direction AND qdot is
    // nonzero -- a passive (tau=0) stepper leaves them at rest.
    EXPECT_GT(dq_f0, 1.0e-5f) << "finger 0 did not move inward -- the drive path (tau) is dead";
    EXPECT_GT(dq_f1, 1.0e-5f) << "finger 1 did not move inward -- the drive path (tau) is dead";
    EXPECT_GT(qd_f0, 1.0e-4f) << "finger 0 qdot zero -- ComputeAccelerations did not see tau";
    EXPECT_GT(qd_f1, 1.0e-4f) << "finger 1 qdot zero -- ComputeAccelerations did not see tau";
}

// ===========================================================================
// THE DISCRIMINATING GATE: the cup is held against gravity by finger friction
// ALONE, NO table.
// ===========================================================================
TEST(GraspHoldSpike, GraspHoldsCupAgainstGravity) {
    if (!CupAvailable()) GTEST_SKIP() << "newton-assets cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);

    // Size the grip so mu*lambda_n clears the cup weight by a comfortable margin (the
    // advisor's feasibility check). A large grip force -> a strong pinch -> a deep
    // friction reserve.
    constexpr float kGripForce = 8.0f;  // N (squeeze force per finger).
    constexpr float kMu = 0.8f;
    GraspScene gs = BuildGraspScene(hull, kGripForce, kMu);
    const BodyState cup0 = gs.config.cup_state;

    coresident::UnifiedCoResidentStepper stepper(context, gs.gripper.host, gs.config,
                                                 kGravityZ, kDt);

    const uint32_t kSteps = 120u;  // 0.5 s.
    const double weight_kick = static_cast<double>(kCupMass) * (-kGravityZ) * kDt;
    const double cup_z0 = cup0.position.z;

    // The hold is a STEADY-STATE force balance: at startup the cup is at rest and the
    // contact establishes over a few steps (a settling transient where the impulse
    // briefly over/undershoots). We assess the balance over the STEADY-STATE window
    // (after the first kSettle steps), the way a real hold gate measures a sustained
    // hold rather than the startup transient. The crosscheck (row impulse vs cup
    // velocity change) must hold EVERY step (it is pure bookkeeping, not dynamics).
    constexpr uint32_t kSettle = 30u;  // 0.125 s settling window.
    uint32_t held_steps = 0u, contact_steps = 0u, steady_held = 0u;
    double max_drift = 0.0, max_balance_err = 0.0, max_crosscheck_err = 0.0;
    bool any_static = false;
    double last_vert_impulse = 0.0;
    for (uint32_t s = 0u; s < kSteps; ++s) {
        const auto rep = stepper.Step();
        if (rep.finger_contacts > 0u) ++contact_steps;
        if (rep.any_static_row) any_static = true;
        const double drift = std::fabs(rep.cup_z - cup_z0);
        max_drift = std::max(max_drift, drift);
        // The cup is FINITE + not free-falling (its speed stays bounded).
        ASSERT_TRUE(std::isfinite(rep.cup_z) && std::isfinite(rep.cup_vz))
            << "cup state non-finite at step " << s;
        if (rep.finger_contacts > 0u && rep.cup_vertical_impulse > 0.0) {
            ++held_steps;
            last_vert_impulse = rep.cup_vertical_impulse;
            const double bal = std::fabs(rep.cup_vertical_impulse - weight_kick) /
                               std::max(weight_kick, 1e-9);
            // The two measures (row-impulse sum vs cup-velocity-change) must AGREE
            // every step -- this is bookkeeping, true in transient + steady state.
            const double cross =
                std::fabs(rep.cup_vertical_impulse - rep.cup_dvz_impulse) /
                std::max(std::fabs(rep.cup_vertical_impulse), 1e-9);
            max_crosscheck_err = std::max(max_crosscheck_err, cross);
            // The force balance (impulse ~= m*g*dt) is the STEADY-STATE hold gate.
            if (s >= kSettle) {
                ++steady_held;
                if (bal > max_balance_err) {
                    max_balance_err = bal;
                    std::printf("[bal-max] new worst steady-state balance at step %u: "
                                "bal=%.4f impulse=%.5e contacts=%u\n",
                                s, bal, rep.cup_vertical_impulse, rep.finger_contacts);
                }
            }
        }
        if (s < 6u || s == kSteps - 1u) {
            std::printf("[traj] step %3u: cup_z=%.5f vz=%.5f  vert_impulse=%.5e "
                        "(mgdt=%.5e, bal=%.3f)  contacts=%u\n",
                        s, rep.cup_z, rep.cup_vz, rep.cup_vertical_impulse, weight_kick,
                        std::fabs(rep.cup_vertical_impulse - weight_kick) /
                            std::max(weight_kick, 1e-9),
                        rep.finger_contacts);
        }
    }

    const BodyState cupF = stepper.Cup();
    const double total_drift = std::fabs(cupF.position.z - cup_z0);
    const double free_fall =
        0.5 * static_cast<double>(-kGravityZ) *
        (kSteps * static_cast<double>(kDt)) * (kSteps * static_cast<double>(kDt));
    std::printf("[GATE] held %u/%u steps, contact %u/%u; cup drift=%.5f m (free-fall "
                "would be %.4f m)\n",
                held_steps, kSteps, contact_steps, kSteps, total_drift, free_fall);
    std::printf("[GATE] Sigma(finger vert impulse)=%.6e  m*g*dt=%.6e  balance err=%.3f  "
                "crosscheck err=%.3e\n",
                last_vert_impulse, weight_kick, max_balance_err, max_crosscheck_err);
    std::printf("[GATE] cup final z=%.5f (z0=%.5f); any_static_row=%d\n",
                cupF.position.z, cup_z0, any_static ? 1 : 0);

    // --- NO TABLE: no row ever carried a StaticNull side (the table lambda is 0
    // because there IS no table -- not because it solved small). ------------------
    EXPECT_FALSE(any_static)
        << "a StaticNull (table/ground) row appeared -- the cup must be held by "
           "FINGER FRICTION ALONE, no support of any kind";

    // --- HELD: the cup is in contact + carrying load for the whole rollout, and its
    // vertical drift is tiny vs free fall (it did NOT free-fall out of the pinch). --
    EXPECT_GE(contact_steps, kSteps - 2u)
        << "the cup left the pinch (broadphase/narrowphase lost the contact)";
    EXPECT_GE(held_steps, kSteps - 2u)
        << "the fingers never delivered an upward contact impulse (vacuous hold)";
    EXPECT_GE(steady_held, kSteps - kSettle - 2u)
        << "the steady-state hold did not persist after settling";
    EXPECT_LT(total_drift, 0.1 * free_fall)
        << "the cup drifted like a free-fall -- it was NOT held";

    // --- Sigma(finger vert impulse) ~= m_cup*g*dt (the force-balance discriminator,
    // the W1a 'base sinks at free-fall = honest' analog), measured two ways that
    // AGREE. The STEADY-STATE hold balances the weight kick to a tight tolerance. --
    EXPECT_LT(max_balance_err, 0.05)
        << "the steady-state finger vertical impulse did NOT balance the cup weight "
           "(m*g*dt) -- the hold is not in force balance";
    EXPECT_LT(max_crosscheck_err, 1.0e-3)
        << "the row-impulse sum and the cup-velocity-change disagree -- the impulse "
           "bookkeeping is wrong";
}

// ===========================================================================
// V2 energy: total mechanical energy is non-increasing across the hold. The fingers
// are PRE-POSED in contact (Delta-theta ~ 0) so the constant grip does ~no work and
// the compliant contact only dissipates -- energy must not be INJECTED.
// ===========================================================================
TEST(GraspHoldSpike, EnergyNonIncreasing) {
    if (!CupAvailable()) GTEST_SKIP() << "newton-assets cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);
    GraspScene gs = BuildGraspScene(hull, /*grip_force=*/8.0f, /*mu=*/0.8f);

    coresident::UnifiedCoResidentStepper stepper(context, gs.gripper.host, gs.config,
                                                 kGravityZ, kDt);
    const uint32_t kSteps = 120u;
    const double e0 = stepper.Energy().Total();
    double e_prev = e0, max_increase = 0.0;
    for (uint32_t s = 0u; s < kSteps; ++s) {
        stepper.Step();
        const double et = stepper.Energy().Total();
        max_increase = std::max(max_increase, et - e_prev);
        e_prev = et;
    }
    const double e_final = stepper.Energy().Total();
    std::printf("[V2] E0=%.6f E_final=%.6f total dE=%.3e max per-step increase=%.3e\n",
                e0, e_final, e_final - e0, max_increase);
    const double e_scale = std::max(std::fabs(e0), 1.0);
    EXPECT_LE(e_final - e0, 1.0e-3 * e_scale)
        << "total mechanical energy INCREASED over the hold (energy injected)";
    EXPECT_LE(max_increase, 5.0e-3 * e_scale)
        << "a single step injected energy beyond the symplectic-Euler ripple";
}

// ===========================================================================
// THE BITE PROOF (mandatory anti-green-wash): grip=0 -> friction collapses -> the
// cup FALLS (~free-fall) -> the hold gate FAILS. A hold that survives grip=0 is fake.
// ===========================================================================
TEST(GraspHoldSpike, GraspGripOffCupFalls) {
    if (!CupAvailable()) GTEST_SKIP() << "newton-assets cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);

    // SAME scene as the hold gate but GRIP FORCE = 0. With no inward squeeze, the
    // pre-posed penetration relaxes (nothing holds the fingers in), the normal force
    // -> 0, the friction cap (mu*lambda_n) -> 0, and the cup falls.
    GraspScene gs = BuildGraspScene(hull, /*grip_force=*/0.0f, /*mu=*/0.8f);
    const BodyState cup0 = gs.config.cup_state;

    coresident::UnifiedCoResidentStepper stepper(context, gs.gripper.host, gs.config,
                                                 kGravityZ, kDt);
    const uint32_t kSteps = 120u;
    const double cup_z0 = cup0.position.z;
    for (uint32_t s = 0u; s < kSteps; ++s) stepper.Step();
    const BodyState cupF = stepper.Cup();

    const double drop = cup_z0 - cupF.position.z;
    const double free_fall =
        0.5 * static_cast<double>(-kGravityZ) *
        (kSteps * static_cast<double>(kDt)) * (kSteps * static_cast<double>(kDt));
    std::printf("[BITE] grip=0: cup dropped %.5f m over %u steps (free-fall = %.4f m); "
                "final vz=%.4f m/s\n", drop, kSteps, free_fall, cupF.linear_velocity.z);

    // The cup FELL: it dropped a large fraction of free-fall (the hold COLLAPSED).
    // This is the discriminator -- if the cup did NOT fall with grip=0, something
    // else (geometry/wedging) is holding it -> the hold gate is green-washed.
    EXPECT_GT(drop, 0.5 * free_fall)
        << "grip=0 did NOT drop the cup -- the hold survives without the grip, so it "
           "is FAKE (geometry/wedging holds the cup, not finger friction)";
    EXPECT_LT(cupF.linear_velocity.z, -1.0f)
        << "grip=0 cup is not accelerating downward -- it is not in free fall";
}

// ===========================================================================
// D1: the full hold rollout run TWICE -> byte-exact final cup state + articulation.
// ===========================================================================
TEST(GraspHoldSpike, DeterministicTwoRun) {
    if (!CupAvailable()) GTEST_SKIP() << "newton-assets cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const CupHull hull = LoadCupHull();
    ASSERT_GT(hull.vcount, 0u);

    const uint32_t kSteps = 100u;
    auto rollout = [&](BodyState* cup_out, articulation::ArticulationHostState* art_out) {
        GraspScene gs = BuildGraspScene(hull, /*grip_force=*/8.0f, /*mu=*/0.8f);
        coresident::UnifiedCoResidentStepper stepper(context, gs.gripper.host,
                                                     gs.config, kGravityZ, kDt);
        for (uint32_t s = 0u; s < kSteps; ++s) stepper.Step();
        *cup_out = stepper.Cup();
        stepper.Download(art_out);
    };
    BodyState cup_a, cup_b;
    articulation::ArticulationHostState art_a, art_b;
    rollout(&cup_a, &art_a);
    rollout(&cup_b, &art_b);

    EXPECT_EQ(std::memcmp(&cup_a, &cup_b, sizeof(BodyState)), 0)
        << "cup BodyState diverged across two rollouts (nondeterministic)";
    ASSERT_EQ(art_a.q.size(), art_b.q.size());
    EXPECT_EQ(std::memcmp(art_a.q.data(), art_b.q.data(), art_a.q.size() * sizeof(float)), 0)
        << "q diverged across two rollouts";
    ASSERT_EQ(art_a.qdot.size(), art_b.qdot.size());
    EXPECT_EQ(std::memcmp(art_a.qdot.data(), art_b.qdot.data(),
                          art_a.qdot.size() * sizeof(float)), 0)
        << "qdot diverged across two rollouts";
    std::printf("[D1] cup BodyState + q + qdot byte-identical across 2 full %u-step "
                "rollouts\n", kSteps);
}
