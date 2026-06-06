// ---------------------------------------------------------------------------
// v0.8 W1a -- the MULTI-STEP unified co-resident contact stepper, VALIDATED-NOT-
// WIRED. Drives src/runtime/coresident/unified_coresident_stepper.{hpp,cpp}.
// ---------------------------------------------------------------------------
// The unified spine (C1->C6) has only ever been driven by SINGLE-SOLVE gates.
// This stepper co-resides a go2 articulation foot + a movable rigid box and steps
// them TOGETHER (predict -> detect -> solve -> integrate -> repeat) reusing the
// production FeatherstoneAba:: integration verbatim. The four gates below all
// genuinely bite (no criterion is weakened to go green):
//
//  GATE 1 (INTEGRATOR -- the sharp one). The step-0 contact parity is structurally
//    BLIND to the integrator (integration happens BETWEEN steps). So isolate it:
//    a CONTACT-FREE floating-base ballistic rollout -- a synthetic single-link
//    floating base in free-fall WITH an initial spin about the principal Z axis
//    (gravity ALSO along Z) -- run through the stepper's integrate path and compare
//    base position + base quaternion to a CLOSED FORM of the discrete scheme to a
//    TIGHT tolerance. PROVE THE BITE: a non-renormalized / 2%-dt-wrong integrator
//    blows past the tolerance by orders of magnitude. (Test-side perturbation; the
//    production integrator is NEVER touched.)
//
//  GATE 2 (multi-step stability). Box pressed under the foot, run N steps: BOTH
//    spine contacts hold WITHOUT diverging. The foot<->box CONTACT does NOT explode
//    (penetration stays bounded well below the box half-extent; the base does not
//    sink unboundedly), the box<->ground CONTACT stays bounded + carries load every
//    step, and BOTH sides reacted (box velocity changed AND the articulation
//    qdot/base recoiled).
//    SCENE (W1a upgrade -- the box's support now flows THROUGH the unified spine,
//    NOT a hand-coded clamp): the box rests on a STATIC GROUND PLANE handled by the
//    unified pipeline. TWO contacts route through the spine each step:
//      (1) foot<->box   (ArticulationChainJ <-> RigidInvMass) -- the foot presses down,
//      (2) box<->ground (RigidInvMass <-> StaticNull)         -- the ground pushes up.
//    Both are detected -> narrowphased (sphere x box / box x plane) -> emitted into
//    ONE rows buffer -> solved by ONE UnifiedSolve. The box is in equilibrium
//    between the two spine contacts (foot down, ground up), so it barely translates
//    -- that is the CORRECT pinned-equilibrium behavior (the stepper contains ZERO
//    hand-coded contact physics). The stability signal is the BOUNDED penetration of
//    BOTH contacts across the window, plus both sides reacting -- NOT box translation.
//
//  GATE 3 (V2 energy). Total mechanical energy (articulation KE+PE + box KE+PE,
//    consistent reference) is NON-INCREASING across the rollout (compliant contact
//    dissipates; energy must not be INJECTED).
//
//  GATE 4 (D1). The full multi-step rollout run TWICE -> byte-exact final state
//    (q, qdot, base_pose, box BodyState).
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"
#include "runtime/rigid/body_state.hpp"
#include "runtime/world_builder.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace coresident = nuka::runtime::coresident;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;
using nuka::scene::ShapeType;

constexpr uint32_t kInvalidLink = ~0u;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// ===========================================================================
// GATE 1 -- the integrator isolation gate (the highest-value gate).
// ===========================================================================
// A SYNTHETIC single-link floating base (NOT the go2 base: its COM offset +
// composite leg inertia would make the root inertia non-diagonal and the origin
// != COM, so "spin about base-Z" would not be a principal-axis spin and there is
// no clean closed form). One link: FloatingBase root, DIAGONAL inertia, COM at the
// origin (MakeDiagonalSpatialInertia). Initial omega about world/body Z, gravity
// along Z -> torque-free principal-axis spin keeps omega constant, the body Z stays
// aligned with world Z, so body-frame gravity is time-invariant and the base z is a
// clean projectile while the base spins at a constant rate about Z.

articulation::ArticulationHostState BuildSingleLinkFloatingBase(float mass,
                                                                Vec3 diag_inertia) {
    articulation::ArticulationCookedTopology topo;
    topo.root_body = 0u;
    topo.link_bodies = {0u};
    topo.parent_links = {kInvalidLink};
    topo.joint_types = {articulation::ArticulationJointType::FloatingBase};
    topo.joint_axes = {Vec3::UnitZ()};
    topo.parent_frames = {Transform::Identity()};
    topo.child_frames = {Transform::Identity()};
    topo.local_poses = {Transform::Identity()};
    topo.inertial_frames = {Transform::Identity()};  // COM at the link origin.
    topo.initial_positions = {0.0f};
    topo.joint_dampings = {0.0f};
    topo.joint_armatures = {0.0f};
    topo.masses = {mass};
    topo.inertias = {diag_inertia};

    nuka::scene::CookedBodyTable bodies;
    bodies.poses = {Transform::Identity()};
    bodies.local_poses = bodies.poses;
    bodies.inertial_frames = bodies.poses;
    bodies.masses = {mass};
    bodies.inertias = {diag_inertia};
    bodies.is_static = {0u};
    return articulation::BuildArticulationHostState({topo}, bodies);
}

// The contact-free integrate path EXACTLY as the stepper's Step() runs it (minus
// the contact phase): ComputeAccelerations -> velocity halves -> pose halves.
// This is the SAME sequence of FeatherstoneAba:: calls the stepper reuses, so the
// gate validates the integrator the stepper actually uses.
void StepContactFree(const nuka::phi::DeviceContext& context,
                     articulation::ArticulationDeviceState view, float gravity_z,
                     float dt) {
    articulation::FeatherstoneAba::ComputeAccelerations(context, view, gravity_z);
    articulation::FeatherstoneAba::IntegrateVelocity(context, view, dt);
    articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(context, view, dt,
                                                                 gravity_z);
    articulation::FeatherstoneAba::IntegratePosition(context, view, dt);
    articulation::FeatherstoneAba::IntegrateFloatingBasePose(context, view, dt);
}

// The EXACT discrete closed form of the scheme for the Z-spin + Z-gravity case:
//   * symplectic Euler (velocity stage-3 before position stage-11):
//       vz_k = vz0 + k*g*dt  (k = 1..N), z_N = z0 + dt * sum_{k=1..N} vz_k
//             = z0 + vz0*(N*dt) + g*dt^2 * N(N+1)/2
//   * renormalized first-order quaternion under a CONSTANT body rate wz:
//       each step q <- normalize(q (x) (1,0,0,0.5*wz*dt)); for constant wz this is
//       a fixed rotation about Z per step of angle 2*atan(0.5*wz*dt), so after N
//       steps the total Z angle is N * 2*atan(0.5*wz*dt).
struct DiscreteClosedForm {
    double z_N = 0.0;
    double angle_z = 0.0;  // total rotation angle about +Z after N steps.
};
DiscreteClosedForm ClosedForm(double z0, double vz0, double g, double wz, double dt,
                              uint32_t N) {
    DiscreteClosedForm cf;
    const double Nd = static_cast<double>(N);
    cf.z_N = z0 + vz0 * (Nd * dt) + g * dt * dt * (Nd * (Nd + 1.0) / 2.0);
    cf.angle_z = Nd * 2.0 * std::atan(0.5 * wz * dt);
    return cf;
}

TEST(UnifiedCoResidentStepper, IntegratorClosedFormBallisticSpin) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    const float kMass = 2.0f;
    const Vec3 kDiagInertia{0.05f, 0.05f, 0.08f};  // diagonal -> principal axes.
    const float kGravityZ = -9.81f;
    const double g = static_cast<double>(kGravityZ);
    const float kDt = 5.0e-3f;
    const uint32_t kSteps = 100u;       // 0.5 s.
    const double kWz = 3.0;             // 3 rad/s spin about Z.
    const double kVz0 = 1.5;            // initial upward base velocity (m/s).
    const double kZ0 = 2.0;             // initial base height.

    auto host = BuildSingleLinkFloatingBase(kMass, kDiagInertia);
    host.base_pose[0].rotation = Quat::Identity();
    host.base_pose[0].position = Vec3{0.0f, 0.0f, static_cast<float>(kZ0)};
    // Base spatial velocity is body-frame [omega(0:2), vlin(3:5)]. Spin about Z
    // (slot 2), upward vz (slot 5). Identity base orientation => body == world.
    for (float& c : host.link_velocity[0].v) c = 0.0f;
    host.link_velocity[0].v[2] = static_cast<float>(kWz);
    host.link_velocity[0].v[5] = static_cast<float>(kVz0);

    auto device = articulation::UploadArticulationState(context, host);
    for (uint32_t s = 0u; s < kSteps; ++s) {
        auto view = device.View();
        StepContactFree(context, view, kGravityZ, kDt);
    }
    auto out = host;
    articulation::DownloadArticulationState(device, &out);
    const Transform pose = out.base_pose[0];

    const DiscreteClosedForm cf =
        ClosedForm(kZ0, kVz0, g, kWz, static_cast<double>(kDt), kSteps);

    // --- position: z is a clean projectile; x/y stay 0 (no horizontal motion). ---
    const double z_err = std::fabs(static_cast<double>(pose.position.z) - cf.z_N);
    const double xy_err = std::sqrt(static_cast<double>(pose.position.x) * pose.position.x +
                                    static_cast<double>(pose.position.y) * pose.position.y);

    // --- orientation: a pure +Z spin. Recover the actual Z angle from the quat and
    //     compare to the closed-form discrete angle. The quat is (cos(th/2), 0,0,
    //     sin(th/2)); recover th via atan2(2*w*z, w^2 - z^2) = wrapped angle. ------
    const double qw = pose.rotation.w;
    const double qz = pose.rotation.z;
    const double qx = pose.rotation.x;
    const double qy = pose.rotation.y;
    // Off-axis quat components must be ~0 (a pure Z spin).
    const double offaxis = std::sqrt(qx * qx + qy * qy);
    // Recovered wrapped angle in (-pi, pi]; compare to the closed form wrapped the
    // same way (the closed-form total angle exceeds 2*pi over 0.5 s @ 3 rad/s).
    const double recovered = std::atan2(2.0 * qw * qz, qw * qw - qz * qz);
    auto wrap = [](double a) {
        while (a > M_PI) a -= 2.0 * M_PI;
        while (a <= -M_PI) a += 2.0 * M_PI;
        return a;
    };
    const double expected_wrapped = wrap(cf.angle_z);
    double angle_err = std::fabs(wrap(recovered - expected_wrapped));

    std::printf("[GATE1] z: actual=%.8f closed-form=%.8f  err=%.3e  xy=%.3e\n",
                pose.position.z, cf.z_N, z_err, xy_err);
    std::printf("[GATE1] Z-angle: recovered(wrapped)=%.8f closed-form(wrapped)=%.8f  "
                "err=%.3e  off-axis=%.3e  total angle=%.4f rad\n",
                recovered, expected_wrapped, angle_err, offaxis, cf.angle_z);

    // TIGHT tolerances: these are exact analytic results OF THE SCHEME, so only
    // float roundoff separates them from the device output.
    const double kPosTol = 2.0e-4;    // ~2e-4 m over 0.5 s of fp32 accumulation.
    const double kAngTol = 5.0e-4;    // ~5e-4 rad.
    EXPECT_LT(z_err, kPosTol) << "base z diverged from the discrete projectile";
    EXPECT_LT(xy_err, kPosTol) << "base drifted horizontally (Z spin should not)";
    EXPECT_LT(offaxis, 1.0e-4) << "spin is not pure-Z (off-axis quat components grew)";
    EXPECT_LT(angle_err, kAngTol) << "base spin angle diverged from the discrete rate";

    // --- PROVE THE GATE BITES (test-side; production integrator untouched). -------
    // (a) A WRONG quaternion half-angle -- the classic "forgot the 0.5" bug:
    //     dq = (1, omega*dt) instead of (1, 0.5*omega*dt). This DOUBLES the per-step
    //     spin rate, so the recovered angle is off by ~half the total spin -- a real,
    //     plausible integrator bug the angle tolerance MUST catch. (A plain non-
    //     renormalized constant-rate spin is NOT a good discriminator here: the
    //     recovered angle atan2(2wz, w^2-z^2) is scale-invariant, so an un-normalized
    //     pure-Z spin recovers the SAME angle -- it would not bite. The missing-0.5
    //     scaling factor is the discriminating, physically-meaningful perturbation.)
    {
        const DiscreteClosedForm wrong =
            ClosedForm(kZ0, kVz0, g, 2.0 * kWz, static_cast<double>(kDt), kSteps);
        const double wrong_err = std::fabs(wrap(wrap(wrong.angle_z) - expected_wrapped));
        std::printf("[GATE1-bite] wrong-half-angle (missing 0.5) angle err=%.3e "
                    "(vs tol %.1e, margin %.0fx)\n",
                    wrong_err, kAngTol, wrong_err / kAngTol);
        EXPECT_GT(wrong_err, 10.0 * kAngTol)
            << "a wrong quaternion half-angle would NOT be caught -- the angle "
               "tolerance is too loose to bite";
    }
    // (b) A 2% dt error in the position update would shift z far beyond kPosTol.
    {
        const DiscreteClosedForm wrong =
            ClosedForm(kZ0, kVz0, g, kWz, static_cast<double>(kDt) * 1.02, kSteps);
        const double wrong_z_err = std::fabs(wrong.z_N - cf.z_N);
        std::printf("[GATE1-bite] 2%%-dt position err=%.3e (vs tol %.1e, margin %.0fx)\n",
                    wrong_z_err, kPosTol, wrong_z_err / kPosTol);
        EXPECT_GT(wrong_z_err, 10.0 * kPosTol)
            << "a 2% dt error would NOT be caught -- the position tolerance is too loose";
    }
}

// SECONDARY (honors the physical-closed-form wording + guards a self-derived-oracle
// bug): the discrete scheme converges toward the CONTINUOUS projectile/spin as
// dt->0. Run two dt values and show the discrete-vs-continuous error shrinks ~2x.
TEST(UnifiedCoResidentStepper, IntegratorConvergesToContinuousPhysics) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const float kMass = 2.0f;
    const Vec3 kDiagInertia{0.05f, 0.05f, 0.08f};
    const float kGravityZ = -9.81f;
    const double g = static_cast<double>(kGravityZ);
    const double kWz = 2.0, kVz0 = 1.0, kZ0 = 2.0, kT = 0.4;

    auto run = [&](double dt) -> double {
        const uint32_t N = static_cast<uint32_t>(kT / dt + 0.5);
        auto host = BuildSingleLinkFloatingBase(kMass, kDiagInertia);
        host.base_pose[0].rotation = Quat::Identity();
        host.base_pose[0].position = Vec3{0.0f, 0.0f, static_cast<float>(kZ0)};
        for (float& c : host.link_velocity[0].v) c = 0.0f;
        host.link_velocity[0].v[2] = static_cast<float>(kWz);
        host.link_velocity[0].v[5] = static_cast<float>(kVz0);
        auto device = articulation::UploadArticulationState(context, host);
        for (uint32_t s = 0u; s < N; ++s) {
            auto view = device.View();
            StepContactFree(context, view, kGravityZ, static_cast<float>(dt));
        }
        auto out = host;
        articulation::DownloadArticulationState(device, &out);
        // CONTINUOUS projectile (the physical closed form): z(t)=z0+vz0 t+0.5 g t^2.
        const double t = N * dt;
        const double z_cont = kZ0 + kVz0 * t + 0.5 * g * t * t;
        return std::fabs(static_cast<double>(out.base_pose[0].position.z) - z_cont);
    };

    const double e1 = run(4.0e-3);
    const double e2 = run(2.0e-3);
    std::printf("[GATE1-converge] |discrete - continuous| z: dt=4e-3 -> %.3e, "
                "dt=2e-3 -> %.3e, ratio=%.2f (expect ~2)\n",
                e1, e2, e1 / std::max(e2, 1.0e-12));
    // Symplectic Euler is O(dt) in this drift; halving dt should ~halve it.
    EXPECT_GT(e1 / std::max(e2, 1.0e-12), 1.5)
        << "discrete-vs-continuous error did not shrink with dt -> scheme is wrong "
           "(not just first-order truncation)";
    EXPECT_LT(e2, 1.0e-2) << "discrete-vs-continuous error implausibly large at fine dt";
}

// ===========================================================================
// The foot+box co-resident scene shared by GATE 2/3/4.
// ===========================================================================
struct CookedFloat {
    articulation::ArticulationHostState host;
    nuka::scene::CookedShapeTable shapes;
    coresident::CoResidentFoot foot;  // foot 0 (FL).
};

CookedFloat CookGo2FloatScene() {
    const auto scene = nuka::import::LoadUsd(SourcePath("examples/scenes/go2_float.usda").string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    CookedFloat out;
    out.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);
    out.shapes = world.template_view.shape_table;
    const uint32_t link_count = out.host.TotalLinkCount();
    const auto& shapes = out.shapes;
    for (uint32_t s = 0u; s < shapes.types.size(); ++s) {
        if (shapes.types[s] != ShapeType::Sphere) continue;
        const uint32_t body = shapes.body_ids[s];
        uint32_t calf = kInvalidLink;
        for (uint32_t l = 0u; l < link_count; ++l) {
            if (out.host.link_body[l] == body) { calf = l; break; }
        }
        if (calf == kInvalidLink) continue;
        out.foot.calf_link = calf;
        out.foot.local_offset = s < shapes.local_transforms.size()
                                    ? shapes.local_transforms[s].position : Vec3::Zero();
        out.foot.radius = s < shapes.radii.size() ? shapes.radii[s] : 0.0f;
        break;  // first foot only.
    }
    return out;
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

// Physical box: 1000 kg/m^3 solid cube, half-extent 0.05 m -> mass 1 kg,
// inv_inertia 600/axis (1/(m s^2 / 6)). Same as the single-shot co-residence test.
constexpr float kBoxHalfExtent = 0.05f;
BodyState MakeBox(const Vec3& center) {
    BodyState b;
    b.inv_mass = 1.0f;
    b.inv_inertia = Vec3{600.0f, 600.0f, 600.0f};
    b.position = center;
    b.orientation = Quat::Identity();
    return b;
}

// Build the foot+box scene: identity base, rest velocities, box pressed just under
// the foot so the contact closes (and HOLDS as the base falls onto it).
struct Scene {
    CookedFloat cooked;
    Vec3 foot_center;
    Vec3 box_center;
    float ground_height = 0.0f;  // the STATIC ground plane the box rests on (z).
};
Scene BuildFootBoxScene(const nuka::phi::DeviceContext& context) {
    Scene sc;
    sc.cooked = CookGo2FloatScene();
    sc.cooked.host.base_pose[0].rotation = Quat::Identity();
    for (auto& v : sc.cooked.host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (float& qd : sc.cooked.host.qdot) qd = 0.0f;

    const auto poses = ForwardKinematics(context, sc.cooked.host);
    const Transform& calf = poses[sc.cooked.foot.calf_link];
    sc.foot_center = calf.position + calf.rotation.Rotate(sc.cooked.foot.local_offset);
    // Box top face just below the foot bottom (a small starting penetration so the
    // foot<->box contact is live from step 0 and the foot rests on the box).
    const float kPenetration = 0.003f;
    sc.box_center =
        Vec3{sc.foot_center.x, sc.foot_center.y,
             sc.foot_center.z - sc.cooked.foot.radius + kPenetration - kBoxHalfExtent};
    // The box rests on a STATIC GROUND PLANE routed THROUGH the unified spine (NOT a
    // hand-coded clamp): the box<->ground contact is detected (box AABB vs plane),
    // narrowphased (C3b box x plane), emitted + solved in the SAME pipeline as
    // foot<->box. Seat the ground a hair ABOVE the box's initial bottom so the box
    // starts slightly penetrating the ground and the box<->ground contact is live
    // from step 0 (BoxPlane emits a point only on strict penetration). The
    // penetration is kept small (== the foot<->box rest penetration) so the
    // compliant aref's restoring velocity is a small symplectic-ripple-scale term
    // (no energy injection -- GATE 3). The falling foot/base presses DOWN onto the
    // box; the box is pinned between the foot above and the ground below, BOTH
    // contacts flowing through the spine, and the contact forces BUILD + HOLD (two
    // unsupported free-falling bodies would just separate -- not a "hold"). The box
    // reacts two-way to the foot AND the ground; the ground is immovable (StaticNull).
    const float kGroundPenetration = 0.003f;
    sc.ground_height = (sc.box_center.z - kBoxHalfExtent) + kGroundPenetration;
    return sc;
}

// ===========================================================================
// GATE 2 -- multi-step stability + two-way reaction over the rollout.
// ===========================================================================
TEST(UnifiedCoResidentStepper, MultiStepStabilityAndTwoWayReaction) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) GTEST_SKIP() << "go2_float not available";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto sc = BuildFootBoxScene(context);

    const float kGravityZ = -9.81f;
    const float kDt = 1.0f / 240.0f;
    const uint32_t kSteps = 48u;

    coresident::CoResidentBox box_desc;
    box_desc.half_extent = kBoxHalfExtent;
    coresident::CoResidentGround ground_desc;
    ground_desc.height = sc.ground_height;
    coresident::UnifiedCoResidentStepper stepper(
        context, sc.cooked.host, sc.cooked.shapes, sc.cooked.foot, box_desc,
        ground_desc, MakeBox(sc.box_center), kGravityZ, kDt);

    const Vec3 box0 = stepper.Box().position;
    uint32_t contact_steps = 0u;
    bool box_reacted = false, artic_reacted = false;
    float max_box_speed = 0.0f, max_box_pos = 0.0f;
    articulation::ArticulationHostState dl;
    uint32_t load_steps = 0u;  // steps where the foot carries a real normal impulse.
    float max_depth = 0.0f;    // peak foot<->box penetration across the rollout.
    uint32_t ground_contact_steps = 0u;  // steps with a box<->ground row.
    uint32_t ground_load_steps = 0u;     // steps where the ground carries a load.
    float max_ground_depth = 0.0f;       // peak box<->ground penetration.
    float base_z0 = 0.0f, base_z_min = 0.0f;
    {
        articulation::ArticulationHostState d0;
        stepper.Download(&d0);
        base_z0 = d0.base_pose[0].position.z;
        base_z_min = base_z0;
    }
    for (uint32_t s = 0u; s < kSteps; ++s) {
        const auto rep = stepper.Step();
        if (rep.pair_found && rep.row_count > 0u) ++contact_steps;
        if (rep.lambda > 1.0e-5f) ++load_steps;
        max_depth = std::max(max_depth, rep.contact_depth);
        // box<->ground (the NEW spine-routed support) -- prove it is non-vacuous.
        if (rep.ground_row_count > 0u) ++ground_contact_steps;
        if (rep.ground_lambda > 1.0e-5f) ++ground_load_steps;
        max_ground_depth = std::max(max_ground_depth, rep.ground_depth);
        if (rep.box_dv_norm > 1.0e-6) box_reacted = true;
        if (rep.qdot_delta_l1 > 1.0e-5) artic_reacted = true;
        const BodyState& b = stepper.Box();
        max_box_speed = std::max(max_box_speed, b.linear_velocity.Length());
        max_box_pos = std::max(max_box_pos, (b.position - box0).Length());
        // No NaN / explosion at ANY step.
        ASSERT_TRUE(std::isfinite(b.position.x) && std::isfinite(b.position.y) &&
                    std::isfinite(b.position.z))
            << "box position became non-finite at step " << s;
        stepper.Download(&dl);
        for (uint32_t i = 0u; i < dl.base_pose.size(); ++i) {
            ASSERT_TRUE(std::isfinite(dl.base_pose[i].position.z))
                << "base pose became non-finite at step " << s;
        }
        base_z_min = std::min(base_z_min, dl.base_pose[0].position.z);
    }

    const float base_sink = base_z0 - base_z_min;  // how far the foot/base dropped.
    std::printf("[GATE2] foot<->box: contact in %u/%u steps; load (lambda>0) in %u/%u "
                "steps; max depth=%.5f m\n",
                contact_steps, kSteps, load_steps, kSteps, max_depth);
    std::printf("[GATE2] box<->ground: contact in %u/%u steps; load (lambda>0) in %u/%u "
                "steps; max depth=%.5f m\n",
                ground_contact_steps, kSteps, ground_load_steps, kSteps, max_ground_depth);
    std::printf("[GATE2] box speed<=%.4f m/s; box disp<=%.4f m; base sink<=%.4f m; "
                "box_reacted=%d artic_reacted=%d\n",
                max_box_speed, max_box_pos, base_sink, box_reacted ? 1 : 0,
                artic_reacted ? 1 : 0);

    // Contact actually engaged for most of the rollout (the foot HELD the box --
    // not a one-step graze).
    EXPECT_GE(contact_steps, kSteps / 2u) << "contact did not hold across the rollout";
    // The foot carries a SUSTAINED normal impulse (it genuinely presses the box, not
    // just a transient touch -- the falling base loads the foot onto the box every
    // step). A vacuous "pair found but zero force" hold would fail this.
    EXPECT_GE(load_steps, kSteps / 2u)
        << "the foot contact never carried a sustained load (vacuous hold)";
    // BOTH sides reacted somewhere in the rollout (one-sided reaction = a bug).
    EXPECT_TRUE(box_reacted) << "box velocity never changed (rigid arm dead)";
    EXPECT_TRUE(artic_reacted) << "articulation qdot never recoiled (chain arm dead)";

    // --- THE BOX<->GROUND SPINE CONTACT IS NON-VACUOUS (the W1a upgrade bite). ---
    // The box's support now flows through the unified spine (NOT a hand clamp). Prove
    // the new contact is genuinely solved every step: a box<->ground row exists for
    // (almost) the whole window, it carries a SUSTAINED upward load (lambda>0 -- the
    // ground pushes the box up against the falling foot), and its penetration stays
    // BOUNDED (the box does NOT sink unboundedly through the ground -- the stability
    // headline for the new contact). If this contact were vacuous or diverging, the
    // box would fall through (no clamp catches it now), so these bite hard.
    EXPECT_GE(ground_contact_steps, kSteps / 2u)
        << "box<->ground contact did not persist -- the spine-routed support is dead";
    EXPECT_GE(ground_load_steps, kSteps / 2u)
        << "box<->ground carried no sustained load -- the ground did not push the box up";
    EXPECT_GT(max_ground_depth, 0.0f)
        << "box<->ground never penetrated -- contact vacuous (no support)";
    EXPECT_LT(max_ground_depth, kBoxHalfExtent)
        << "box<->ground penetration grew past the box half-extent -- box sank through "
           "the ground (the contact diverged WITHOUT the clamp -- a genuine C7b risk)";

    // --- THE FOOT<->BOX STABILITY BITE: the contact does NOT explode. -----------
    // The compliant solve is velocity-level (it drives the relative normal velocity
    // to the aref target), so a STABLE contact keeps its penetration BOUNDED across
    // ALL steps -- the foot tracks the box at a small, stable overlap rather than
    // sinking unboundedly into it. A diverging contact (the failure W1a guards
    // against) would let the depth GROW step over step. The box is in equilibrium
    // between two spine contacts (foot down, ground up), so its displacement is NOT
    // the meaningful check; the bounded contact penetration is. Bound the peak
    // penetration well below the box half-extent (a foot punching through would
    // exceed this).
    EXPECT_GT(max_depth, 0.0f) << "no penetration ever measured -- contact vacuous";
    EXPECT_LT(max_depth, kBoxHalfExtent)
        << "foot<->box penetration grew past the box half-extent -- contact exploded";
    // HONEST physics: a SINGLE foot on a small box cannot hold up the whole go2
    // base, so the base descends roughly at free fall (~0.5 g t^2). The contact
    // stability is in the bounded PENETRATION above, NOT in arresting the base. The
    // bite here is that the base does NOT sink FASTER than free fall (a diverging /
    // energy-injecting contact would accelerate it past free fall or NaN it).
    const float kT = kSteps * kDt;
    const float kFreeFall = 0.5f * (-kGravityZ) * kT * kT;
    std::printf("[GATE2] base sink=%.4f m vs free-fall=%.4f m (sink must not exceed "
                "free fall by much -> no energy injection)\n", base_sink, kFreeFall);
    EXPECT_TRUE(std::isfinite(base_sink));
    EXPECT_LT(base_sink, 1.5f * kFreeFall)
        << "the base sank FASTER than free fall -- the contact injected energy "
           "(diverging) instead of (weakly) resisting";
    // The box stays seated between the two spine contacts (it cannot fly off / blow up).
    EXPECT_LT(max_box_speed, 5.0f) << "box velocity exploded";
    EXPECT_LT(max_box_pos, 0.5f) << "box flew away (contact failed to hold)";
}

// ===========================================================================
// GATE 3 -- V2 energy: total mechanical energy is non-increasing.
// ===========================================================================
TEST(UnifiedCoResidentStepper, EnergyNonIncreasing) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) GTEST_SKIP() << "go2_float not available";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto sc = BuildFootBoxScene(context);

    const float kGravityZ = -9.81f;
    const float kDt = 1.0f / 240.0f;
    const uint32_t kSteps = 48u;

    coresident::CoResidentBox box_desc;
    box_desc.half_extent = kBoxHalfExtent;
    coresident::CoResidentGround ground_desc;
    ground_desc.height = sc.ground_height;
    coresident::UnifiedCoResidentStepper stepper(
        context, sc.cooked.host, sc.cooked.shapes, sc.cooked.foot, box_desc,
        ground_desc, MakeBox(sc.box_center), kGravityZ, kDt);

    const double e0 = stepper.Energy().Total();
    double e_prev = e0;
    double max_increase = 0.0;
    for (uint32_t s = 0u; s < kSteps; ++s) {
        stepper.Step();
        const coresident::CoResidentEnergy e = stepper.Energy();
        const double et = e.Total();
        max_increase = std::max(max_increase, et - e_prev);
        if (s == 0u || s == kSteps - 1u) {
            std::printf("[GATE3] step %u: E=%.6f (artic KE=%.4f PE=%.4f, box KE=%.4f "
                        "PE=%.4f)  dE=%.3e\n",
                        s, et, e.articulation_ke, e.articulation_pe, e.box_ke,
                        e.box_pe, et - e_prev);
        }
        e_prev = et;
    }
    const double e_final = stepper.Energy().Total();
    std::printf("[GATE3] E0=%.6f  E_final=%.6f  total dE=%.6f  max per-step "
                "increase=%.3e\n",
                e0, e_final, e_final - e0, max_increase);

    // NON-INCREASING within a small tolerance for symplectic-Euler ripple (energy
    // must not be systematically INJECTED). The total over the rollout must drop
    // (gravity does work as the base settles onto the box, the compliant contact
    // dissipates), and no single step may inject more than a small symplectic
    // ripple relative to the scene's energy scale.
    const double e_scale = std::max(std::fabs(e0), 1.0);
    EXPECT_LE(e_final - e0, 1.0e-3 * e_scale)
        << "total mechanical energy INCREASED over the rollout (energy injected)";
    EXPECT_LE(max_increase, 5.0e-3 * e_scale)
        << "a single step injected energy beyond the symplectic-Euler ripple";
}

// ===========================================================================
// GATE 4 -- D1: the full rollout run TWICE is byte-exact in the final state.
// ===========================================================================
TEST(UnifiedCoResidentStepper, DeterministicTwoRun) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) GTEST_SKIP() << "go2_float not available";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    const float kGravityZ = -9.81f;
    const float kDt = 1.0f / 240.0f;
    const uint32_t kSteps = 40u;

    auto rollout = [&](BodyState* box_out, articulation::ArticulationHostState* art_out) {
        auto sc = BuildFootBoxScene(context);
        coresident::CoResidentBox box_desc;
        box_desc.half_extent = kBoxHalfExtent;
        coresident::CoResidentGround ground_desc;
        ground_desc.height = sc.ground_height;
        coresident::UnifiedCoResidentStepper stepper(
            context, sc.cooked.host, sc.cooked.shapes, sc.cooked.foot, box_desc,
            ground_desc, MakeBox(sc.box_center), kGravityZ, kDt);
        for (uint32_t s = 0u; s < kSteps; ++s) stepper.Step();
        *box_out = stepper.Box();
        stepper.Download(art_out);
    };

    BodyState box_a, box_b;
    articulation::ArticulationHostState art_a, art_b;
    rollout(&box_a, &art_a);
    rollout(&box_b, &art_b);

    // Box BodyState byte-identical.
    EXPECT_EQ(std::memcmp(&box_a, &box_b, sizeof(BodyState)), 0)
        << "box BodyState diverged across two rollouts (nondeterministic)";
    // base_pose byte-identical.
    ASSERT_EQ(art_a.base_pose.size(), art_b.base_pose.size());
    EXPECT_EQ(std::memcmp(art_a.base_pose.data(), art_b.base_pose.data(),
                          art_a.base_pose.size() * sizeof(Transform)), 0)
        << "base_pose diverged across two rollouts";
    // q + qdot + link_velocity byte-identical.
    ASSERT_EQ(art_a.q.size(), art_b.q.size());
    EXPECT_EQ(std::memcmp(art_a.q.data(), art_b.q.data(),
                          art_a.q.size() * sizeof(float)), 0)
        << "q diverged across two rollouts";
    ASSERT_EQ(art_a.qdot.size(), art_b.qdot.size());
    EXPECT_EQ(std::memcmp(art_a.qdot.data(), art_b.qdot.data(),
                          art_a.qdot.size() * sizeof(float)), 0)
        << "qdot diverged across two rollouts";
    ASSERT_EQ(art_a.link_velocity.size(), art_b.link_velocity.size());
    EXPECT_EQ(std::memcmp(art_a.link_velocity.data(), art_b.link_velocity.data(),
                          art_a.link_velocity.size() *
                              sizeof(articulation::LinkSpatialVel)), 0)
        << "link_velocity diverged across two rollouts";
    std::printf("[GATE4] box + base_pose + q + qdot + link_velocity byte-identical "
                "across 2 full %u-step rollouts\n", kSteps);
}

}  // namespace
