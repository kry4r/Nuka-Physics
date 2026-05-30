// ---------------------------------------------------------------------------
// [sim-val.PD] Go2 PD standing/tracking validation -- the convention-free
// physics ground truth.
//
// PD-drives a FREE-FLOATING Go2 (go2_float.usda: real 6.921 kg trunk, root cooks
// to ArticulationJointType::FloatingBase) to its cooked standing crouch
// (hip 0, thigh +0.8, calf -1.5 rad) and asserts the physics is right WITHOUT
// any Python / torch / ML / new C ABI. It uses ONLY the engine's existing per-
// link position drive on gpu::BatchedArticulatedWorld:
//   ApplyPositionDrives -> tau   (tau[link] = Kp*(target-q) - Kd*qdot, clamp)
//   then the validated 3-pass ABA / integrate / contact / integrate-q pipeline.
//
// This is the GROUND TRUTH a later learned-policy task builds on: if a policy
// later "flails," it is provably a convention issue, not the dynamics, because
// this test holds the standing pose on the same stepper.
//
// Construction (cook, ground-seating, summed-cooked-mass Mg, the replicated
// drive upload, the determinism harness) MIRRORS the proven T8b test
// tests/runtime/test_floating_base_contact.cpp. The ONE substantive change is
// the drive: T8b reuses the SOFT cooked gain (0.1) and lets the legs SAG into a
// low sprawl (it explicitly settles to a sprawl and still carries Mg). This test
// instead drives the 12 leg joints with a STIFF Go2-typical PD (Kp=60, Kd=4 --
// below 2*sqrt(Kp), see the explicit-damping note at the gain constants, finite
// force limit) so the robot HOLDS the crouch. That is why the
// discriminating gate here is JOINT TRACKING -- Sigma(lambda_n)~Mg passes even
// for a collapsed/sprawled robot (the feet still push up), so it is the WEAK
// gate; q-near-target is what certifies the robot is actually STANDING.
//
// GEOMETRY: the go2_float foot collision spheres now carry an explicit calf-local
// xformOp:translate = (0,0,-0.213), so each foot sits at the calf TIP (the real
// Go2 foot, FL_foot_joint origin in the project URDF), NOT at the calf link origin
// (the knee). The calf joint therefore genuinely repositions the foot, so the full
// Go2 crouch (hip 0, thigh +0.8, calf -1.5) is the real, statically-balanced stance
// and is what this test drives -- no single-link / calf-inert workaround.
//
// The settled-tail gates, asserted SIMULTANEOUSLY (so no single tautology can
// false-pass):
//   (A) Joint tracking      -- each of the 12 actuated q within a BOUNDED steady-
//                              state error of its crouch target (a gravity sag
//                              ~tau_grav/Kp is physical; bounded, not zero). THE
//                              PRIMARY DISCRIMINATOR -- excludes the sprawl.
//   (B) Standing height     -- trunk settles to and HOLDS a height consistent
//                              with the crouch geometry (does not fall through
//                              the ground nor launch); feet stay seated; drift
//                              over the tail bounded.
//   (C) Weight support      -- Sigma(lambda_n)/dt ~ Mg (exact for the free base,
//                              Mg from SUMMED COOKED MASSES) AND per-foot normal
//                              ~ Mg/4 within a LOOSE tol (Go2 COM isn't centred;
//                              a lifted foot drops a slot to 0 -> fails -> also a
//                              tipping catch). The flipped-trap note: per-foot
//                              ~Mg/4 IS valid here because the base is a real
//                              free trunk.
//   (D) No tipping          -- trunk tilt ANGLE (body +z vs world +z) AND base
//                              angular velocity both stay small over the tail.
//   (E) Determinism 0/0     -- two full runs bit-exact AND cross-replica bit-
//                              exact across replicas (identical init -> identical
//                              per-env trajectory; no float atomics, fixed order).
//   (F) 4096-env runnable    -- the full 4096-env run completes with NO NaN/Inf
//                              in q/qdot/base-pose and no OOM.
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
#include "runtime/gpu/batched_articulated_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace gpu = nuka::runtime::gpu;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;

// PD that HOLDS the standing pose under the free 6.9 kg trunk, in the engine's
// LITERAL Nm/rad units (NOT a normalized RL action space). A MODERATE Go2-typical
// stiffness is correct here: the corrected stance is stabilized primarily by
// GEOMETRY (feet under the hips at the real crouch, below), so the PD only carries
// the static gravity torque, not an unstable pose. Kp=60 holds the load with a
// small, physical gravity sag (~tau_grav/Kp): the measured worst joint tracking is
// 0.10 rad (calf carrying ~Mg/4 at the ~0.137 m folded-foot lever -> peak tau ~6 Nm
// -> 6/60 ~ 0.10 rad). Force limit 24 Nm >> 6 Nm, so it never clamps the stance.
//
// DAMPING NOTE (important): the joint drive is applied EXPLICITLY -- the kernel sets
// tau = Kp*(target-q) - Kd*qdot and then the integrator does qdot += qddot*dt
// (featherstone_aba.cu ApplyPositionDriveKernel). Explicit damping is only stable up
// to a dt/inertia-dependent ceiling; for these small-inertia legs at dt=1/240 that
// ceiling is MEASURED at ~Kd in (8,12) (Kd=8 holds cleanly, Kd=12 diverges to qdot
// blow-up + tipping). The textbook critically-damped value 2*sqrt(Kp) ~= 15.5 is
// ABOVE that ceiling and DIVERGES here (verified). So Kd is deliberately set BELOW
// 2*sqrt(Kp): Kd=4 sits ~2.5x under the stable ceiling, fully damps the crouch ring-
// down (no overshoot off the unilateral contact), and settles to ω~0 / v_z~0.
constexpr float kKp = 60.0f;
constexpr float kKd = 4.0f;
constexpr float kForceLimit = 24.0f;  // the scene's cooked actuator force limit
                                      // (the crouch needs ~6 Nm peak; 24 is ample
                                      // -- raising it lets the legs SLAM/bounce).
constexpr float kGravityZ = -9.81f;
constexpr float kDt = 1.0f / 240.0f;

// STANDING POSE -- the REAL Go2 cooked crouch. With the foot sphere now at the calf
// TIP (calf-local translate (0,0,-0.213)), the calf joint genuinely repositions the
// foot, so the full crouch is the correct, statically-balanced stance: hip splay
// +-0.1 (roll margin), thigh +0.8, calf -1.5. The calf is NO LONGER geometrically
// inert -- folding it -1.5 rad swings the foot ~0.137 m FORWARD of the knee, which
// (combined with thigh +0.8 carrying the knee back) lands the feet ESSENTIALLY
// UNDER the hips. The COM_x then sits inside the foot x-span (asserted by the in-
// test static-balance diagnostic), so the joint-PD -> trunk-pitch feedback is
// RESTORING -- the old "feet ~0.15 m behind the hips -> positive-feedback fold"
// pathology was an artifact of the foot-at-knee geometry bug and is gone. The
// INITIAL q is set to the SAME pose so the PD does not snap the legs at t=0 (a snap
// itself kicks a pitch).
constexpr float kPoseHip = 0.1f;
constexpr float kPoseThigh = 0.8f;
constexpr float kPoseCalf = -1.5f;
constexpr std::array<float, 4> kHipSign = {1.0f, -1.0f, 1.0f, -1.0f};  // fl,fr,rl,rr

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// Cooked go2_float + base-relative foot shapes + the STANDING PD drives. The
// drive TARGET is the cooked rest pose (host.q == the go2_stand golden crouch:
// hip 0, thigh +0.8, calf -1.5), which is provably holdable. The gains override
// the soft cooked gain with the stiff PD above on the 12 actuated leg links; the
// FloatingBase root gets ZERO drive (it is unactuated/free).
struct CookedFloat {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;
    std::vector<float> drive_targets;
    std::vector<float> drive_stiffness;
    std::vector<float> drive_damping;
    std::vector<float> drive_force_limits;
    std::vector<float> link_mass;          // per global link (cooked).
    std::vector<uint32_t> actuated_links;  // the 12 leg links (drive convention).
};

CookedFloat CookGo2Float() {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);

    CookedFloat result;
    result.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);
    const uint32_t link_count = result.host.TotalLinkCount();

    // Cooked per-link mass = spatial inertia [3][3] (m on the linear-linear diag).
    result.link_mass.assign(link_count, 0.0f);
    for (uint32_t link = 0u; link < link_count; ++link) {
        result.link_mass[link] = result.host.link_inertia[link].I[3u * 6u + 3u];
    }

    // Foot spheres -> their owning calf link (same as T8b).
    const auto& shapes = world.template_view.shape_table;
    for (uint32_t shape = 0u; shape < shapes.types.size(); ++shape) {
        if (shapes.types[shape] != nuka::scene::ShapeType::Sphere) {
            continue;
        }
        const uint32_t body = shapes.body_ids[shape];
        uint32_t calf_link = kInvalidLink;
        for (uint32_t link = 0u; link < link_count; ++link) {
            if (result.host.link_body[link] == body) {
                calf_link = link;
                break;
            }
        }
        if (calf_link == kInvalidLink) {
            continue;
        }
        articulation::FootShape foot;
        foot.calf_local_link = calf_link;
        foot.local_offset = shape < shapes.local_transforms.size()
                                ? shapes.local_transforms[shape].position
                                : Vec3::Zero();
        foot.radius = shape < shapes.radii.size() ? shapes.radii[shape] : 0.0f;
        result.feet.push_back(foot);
    }

    // STANDING PD drives. Map each POSITION actuator's joint -> its child link,
    // exactly as ApplyPositionDrives reads (one drive descriptor per global link).
    // The 12 mapped leg links get the moderate PD; everything else (incl. the
    // FloatingBase root) stays zero. The drive TARGET (and the INITIAL q) are set
    // to the balanced standing pose below, after the mapping is known.
    result.drive_targets = result.host.q;
    result.drive_stiffness.assign(link_count, 0.0f);
    result.drive_damping.assign(link_count, 0.0f);
    result.drive_force_limits.assign(link_count, 0.0f);
    const auto& actuators = world.template_view.actuator_table;
    const auto& joints = world.template_view.joint_table;
    for (uint32_t actuator = 0u;
         actuator < world.template_view.actuator_count; ++actuator) {
        if (actuator >= actuators.joint_ids.size() ||
            actuator >= actuators.types.size() ||
            actuators.types[actuator] != nuka::scene::ActuatorType::Position) {
            continue;
        }
        const auto joint = actuators.joint_ids[actuator];
        if (joint >= joints.child_bodies.size()) {
            continue;
        }
        const auto child_body = joints.child_bodies[joint];
        uint32_t link = kInvalidLink;
        for (uint32_t l = 0u; l < link_count; ++l) {
            if (result.host.link_body[l] == child_body) {
                link = l;
                break;
            }
        }
        if (link == kInvalidLink ||
            result.host.joint_type[link] ==
                articulation::ArticulationJointType::Fixed ||
            result.host.joint_type[link] ==
                articulation::ArticulationJointType::FloatingBase) {
            continue;
        }
        result.drive_stiffness[link] = kKp;
        result.drive_damping[link] = kKd;
        result.drive_force_limits[link] = kForceLimit;
        result.actuated_links.push_back(link);
    }

    // Set BOTH the standing-pose drive targets AND the initial q to the SAME pose,
    // so the PD does not snap the legs at t=0 (a snap kicks a pitch). actuated_links
    // is leg-major in actuator order: fl,fr,rl,rr x (hip,thigh,calf).
    for (uint32_t leg = 0u; leg < 4u && leg * 3u + 2u < result.actuated_links.size();
         ++leg) {
        const uint32_t hip_link = result.actuated_links[leg * 3u + 0u];
        const uint32_t thigh_link = result.actuated_links[leg * 3u + 1u];
        const uint32_t calf_link = result.actuated_links[leg * 3u + 2u];
        const float hip_target = kPoseHip * kHipSign[leg];
        result.drive_targets[hip_link] = hip_target;
        result.drive_targets[thigh_link] = kPoseThigh;
        result.drive_targets[calf_link] = kPoseCalf;
        result.host.q[hip_link] = hip_target;
        result.host.q[thigh_link] = kPoseThigh;
        result.host.q[calf_link] = kPoseCalf;
    }
    return result;
}

// Replicates a link-major host drive vector to `env_count` device replicas.
nuka::phi::Buffer UploadReplicatedDrive(const std::vector<float>& base,
                                        uint32_t env_count) {
    std::vector<float> all(base.size() * env_count);
    for (uint32_t e = 0u; e < env_count; ++e) {
        for (size_t i = 0u; i < base.size(); ++i) {
            all[e * base.size() + i] = base[i];
        }
    }
    nuka::phi::Buffer b(all.size() * sizeof(float), nuka::phi::MemoryKind::Device);
    b.CopyFromHost(all.data(), all.size() * sizeof(float));
    return b;
}

// Seats the ground a hair above the rest foot bottoms (mirrors T8b / the c_abi
// DeriveGroundHeight): run FK at the rest crouch, find the lowest foot bottom,
// seat the ground kRestPenetration above it so the feet contact and the solve
// pushes the base to equilibrium.
float SeatGround(const nuka::phi::DeviceContext& context, CookedFloat& cooked) {
    auto& host = cooked.host;
    const uint32_t link_count = host.TotalLinkCount();
    host.base_pose[0].rotation = Quat::Identity();
    auto probe = articulation::UploadArticulationState(context, host);
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                               nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, probe.View(),
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> rest_pose(link_count);
    pose_buf.CopyToHost(rest_pose.data(), rest_pose.size() * sizeof(Transform));
    float min_bottom = std::numeric_limits<float>::infinity();
    for (const auto& foot : cooked.feet) {
        const Transform calf = rest_pose[foot.calf_local_link];
        const Vec3 center = calf.position + calf.rotation.Rotate(foot.local_offset);
        min_bottom = std::min(min_bottom, center.z - foot.radius);
    }
    constexpr float kRestPenetration = 0.002f;
    return min_bottom + kRestPenetration;
}

// Evaluates a candidate standing pose STATICALLY (FK only, no stepping): sets the
// 12 leg q to (hip, thigh, calf) per leg, runs forward kinematics at the level
// identity base, and reports the foot x relative to its OWN hip x (the pitch-
// feedback sign criterion: feet UNDER the hips => restoring; feet BEHIND => the
// positive-pitch-feedback fold) plus COM and the foot-x polygon span. Returns the
// resulting per-leg foot world position so the caller can also seat the ground.
struct PoseEval {
    double com_x = 0.0, com_y = 0.0;
    float foot_x_front = 0.0f, foot_x_rear = 0.0f;  // averaged front/rear foot x.
    float min_foot_bottom = 0.0f;
    float max_foot_minus_hip = 0.0f;  // worst |foot_x - hip_x| over the 4 legs.
};

// Leg link layout (confirmed by name/targets): links 1..12 are
// (hip,thigh,calf) x (fl,fr,rl,rr). Hip is X-axis, thigh+calf are Y-axis.
PoseEval EvaluatePose(const nuka::phi::DeviceContext& context,
                      const CookedFloat& cooked, float hip, float thigh, float calf,
                      const std::array<float, 4>& hip_sign) {
    articulation::ArticulationHostState h = cooked.host;
    h.base_pose[0].rotation = Quat::Identity();
    h.base_pose[0].position = Vec3{0.0f, 0.0f, 0.445f};
    const uint32_t link_count = h.TotalLinkCount();
    // actuated_links order: fl(hip,thigh,calf), fr, rl, rr.
    for (uint32_t leg = 0u; leg < 4u; ++leg) {
        h.q[cooked.actuated_links[leg * 3u + 0u]] = hip * hip_sign[leg];
        h.q[cooked.actuated_links[leg * 3u + 1u]] = thigh;
        h.q[cooked.actuated_links[leg * 3u + 2u]] = calf;
    }
    auto probe = articulation::UploadArticulationState(context, h);
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                               nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, probe.View(),
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> rest(link_count);
    pose_buf.CopyToHost(rest.data(), rest.size() * sizeof(Transform));

    PoseEval e;
    double mx = 0.0, my = 0.0, msum = 0.0;
    for (uint32_t l = 0u; l < link_count; ++l) {
        const Transform t = rest[l];
        const Vec3 com_w =
            t.position + t.rotation.Rotate(h.link_inertial_frame[l].position);
        const double m = cooked.link_mass[l];
        mx += m * com_w.x;
        my += m * com_w.y;
        msum += m;
    }
    e.com_x = mx / msum;
    e.com_y = my / msum;
    e.min_foot_bottom = std::numeric_limits<float>::infinity();
    double front_sum = 0.0, rear_sum = 0.0;
    for (uint32_t fi = 0u; fi < cooked.feet.size(); ++fi) {
        const auto& f = cooked.feet[fi];
        const Transform calf_t = rest[f.calf_local_link];
        const Vec3 c = calf_t.position + calf_t.rotation.Rotate(f.local_offset);
        e.min_foot_bottom = std::min(e.min_foot_bottom, c.z - f.radius);
        // The hip link for this leg is actuated_links[fi*3 + 0]; its world x is the
        // hip joint x (the reference the foot should sit under).
        const uint32_t hip_link = cooked.actuated_links[fi * 3u + 0u];
        const float hip_x = rest[hip_link].position.x;
        e.max_foot_minus_hip =
            std::max(e.max_foot_minus_hip, std::fabs(c.x - hip_x));
        if (fi < 2u) front_sum += c.x; else rear_sum += c.x;
    }
    e.foot_x_front = static_cast<float>(front_sum / 2.0);
    e.foot_x_rear = static_cast<float>(rear_sum / 2.0);
    return e;
}

gpu::BatchedArticulatedStepParams MakeStepParams(const nuka::phi::Buffer& targets,
                                                 const nuka::phi::Buffer& stiffness,
                                                 const nuka::phi::Buffer& damping,
                                                 const nuka::phi::Buffer& limits) {
    gpu::BatchedArticulatedStepParams params;
    params.drive_targets = static_cast<const float*>(targets.Data());
    params.drive_stiffness = static_cast<const float*>(stiffness.Data());
    params.drive_damping = static_cast<const float*>(damping.Data());
    params.drive_force_limits = static_cast<const float*>(limits.Data());
    params.gravity_z = kGravityZ;
    params.dt = kDt;
    params.baumgarte_max_velocity = 3.0f;  // production value (firm contact).
    return params;
}

// Tilt angle (radians) between the body +z axis and world +z.
double TiltAngle(const Quat& rotation) {
    const Vec3 body_z = rotation.Rotate(Vec3{0.0f, 0.0f, 1.0f});
    const double c = std::max(-1.0, std::min(1.0, static_cast<double>(body_z.z)));
    return std::acos(c);
}

}  // namespace

// ---------------------------------------------------------------------------
// DIAGNOSTIC (manual): static pose sweep -- find a thigh/calf that lands the feet
// UNDER the hips (foot_x ~ hip_x), so the joint-PD -> trunk-pitch feedback is
// restoring, not the feet-behind-hips positive-feedback fold. FK only, no
// stepping. Run with --gtest_also_run_disabled_tests.
// ---------------------------------------------------------------------------
TEST(Go2PdStanding, DISABLED_StaticPoseSweep) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    const std::array<float, 4> splay = {0.1f, -0.1f, 0.1f, -0.1f};
    std::printf("hip thigh calf | COM_x foot_x_front foot_x_rear "
                "max|foot-hip| min_bottom\n");
    for (float thigh = 0.2f; thigh <= 0.95f; thigh += 0.1f) {
        for (float calf = -2.0f; calf <= -0.9f; calf += 0.2f) {
            const PoseEval e = EvaluatePose(context, cooked, 0.1f, thigh, calf, splay);
            std::printf("0.1 %.2f %.2f | %+.4f %+.4f %+.4f %.4f %.4f\n",
                        thigh, calf, e.com_x, e.foot_x_front, e.foot_x_rear,
                        e.max_foot_minus_hip, e.min_foot_bottom);
        }
    }
}

// ---------------------------------------------------------------------------
// (A)+(B)+(C)+(D) The standing certificate: PD-drive go2_float to the crouch on
// a small env count, settle, then assert tracking + height + weight + no-tip ALL
// hold together on the settled tail. Tracking is the discriminator; the others
// rule out a sum-to-Mg sprawl false-pass.
// ---------------------------------------------------------------------------
TEST(Go2PdStanding, HoldsCrouchTrackingHeightWeightNoTip) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    ASSERT_EQ(host.ArticulationCount(), 1u);
    const uint32_t root = host.articulation_link_offset[0];
    ASSERT_EQ(host.joint_type[root],
              articulation::ArticulationJointType::FloatingBase);
    const uint32_t link_count = host.TotalLinkCount();
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);
    ASSERT_EQ(max_dof, 18u) << "go2_float base-inclusive DOF must be 18 (6 base + 12)";

    // Exactly 12 actuated leg joints mapped (4 legs x hip/thigh/calf).
    ASSERT_EQ(cooked.actuated_links.size(), 12u)
        << "expected 12 actuated leg joints mapped to links";
    ASSERT_EQ(cooked.feet.size(), 4u) << "expected 4 feet";

    // Mg from SUMMED COOKED LINK MASSES (independent of the operator under test).
    double total_mass = 0.0;
    for (uint32_t l = 0u; l < link_count; ++l) total_mass += cooked.link_mass[l];
    const double mg = total_mass * static_cast<double>(-kGravityZ);
    const double mg_per_foot = mg / 4.0;
    std::printf("[diag] summed cooked mass = %.5f kg -> Mg = %.4f N (Mg/4=%.4f)\n",
                total_mass, mg, mg_per_foot);
    EXPECT_NEAR(cooked.link_mass[root], 6.921f, 1.0e-3f)
        << "cooked base mass disagrees with the go2_float trunk mass";

    // COOK-TIME GEOMETRY DIAGNOSTIC: prove the foot collision sphere now sits at the
    // calf TIP (calf-local (0,0,-0.213), the real Go2 FL_foot_joint origin), NOT at
    // the calf link origin (the knee). If this is (0,0,0) the USDA translate did not
    // take -- the whole crouch stance premise is invalid, so fail loud and early.
    std::printf("[diag] foot local_offset (calf-local):");
    for (const auto& f : cooked.feet) {
        std::printf(" (%.4f,%.4f,%.4f)", f.local_offset.x, f.local_offset.y,
                    f.local_offset.z);
    }
    std::printf("\n");
    for (const auto& f : cooked.feet) {
        EXPECT_NEAR(f.local_offset.z, -0.213f, 1.0e-4f)
            << "foot collision sphere is NOT at the calf tip -- the go2_float "
               "xformOp:translate=(0,0,-0.213) did not cook through; the crouch "
               "stance is geometrically invalid";
        EXPECT_NEAR(f.local_offset.x, 0.0f, 1.0e-4f);
        EXPECT_NEAR(f.local_offset.y, 0.0f, 1.0e-4f);
    }

    const float ground_height = SeatGround(context, cooked);
    std::printf("[diag] ground height = %.5f\n", ground_height);

    // STATIC-BALANCE DIAGNOSTIC: at the INITIAL level pose (z=0.445, identity), is
    // the mass-weighted COM_x inside the foot-polygon x-span? COM inside => the pose
    // is statically holdable (any tip is a stiffness threshold, fix by stiffening);
    // COM outside => the pose itself is unbalanced (fix by retargeting thigh/calf).
    {
        host.base_pose[0].rotation = Quat::Identity();
        auto probe = articulation::UploadArticulationState(context, host);
        nuka::phi::Buffer pose_buf(
            static_cast<size_t>(link_count) * sizeof(Transform),
            nuka::phi::MemoryKind::Device);
        articulation::UpdateWorldLinkPoses(context, probe.View(),
                                           static_cast<Transform*>(pose_buf.Data()));
        context.stream.Synchronize();
        std::vector<Transform> rest(link_count);
        pose_buf.CopyToHost(rest.data(), rest.size() * sizeof(Transform));
        double mx = 0.0, my = 0.0, msum = 0.0;
        for (uint32_t l = 0u; l < link_count; ++l) {
            const Transform t = rest[l];
            const Vec3 com_w =
                t.position + t.rotation.Rotate(host.link_inertial_frame[l].position);
            const double m = cooked.link_mass[l];
            mx += m * com_w.x;
            my += m * com_w.y;
            msum += m;
        }
        const double com_x = mx / msum, com_y = my / msum;
        float fx_min = std::numeric_limits<float>::infinity();
        float fx_max = -std::numeric_limits<float>::infinity();
        float max_foot_below_knee = 0.0f;  // worst foot-center-z below its calf origin.
        for (const auto& f : cooked.feet) {
            const Transform calf = rest[f.calf_local_link];
            const Vec3 c = calf.position + calf.rotation.Rotate(f.local_offset);
            fx_min = std::min(fx_min, c.x);
            fx_max = std::max(fx_max, c.x);
            // The calf link ORIGIN is the knee; the foot center is now below it (the
            // geometry-fix signature). At the crouch the calf is folded so the drop
            // is < 0.213, but it must be clearly negative (foot below the knee).
            max_foot_below_knee =
                std::max(max_foot_below_knee, calf.position.z - c.z);
        }
        std::printf("[diag] static balance: COM=(%.4f,%.4f)  foot x-span=[%.4f,%.4f] "
                    "(inside=%s)\n", com_x, com_y, fx_min, fx_max,
                    (com_x > fx_min && com_x < fx_max) ? "yes" : "NO");
        std::printf("[diag] geometry: foot center dropped %.4f m below the knee "
                    "(calf origin) at the crouch -- foot is at the calf tip\n",
                    max_foot_below_knee);
        EXPECT_GT(max_foot_below_knee, 0.05f)
            << "foot is not below the knee -- the calf-tip translate did not move "
               "the contact point";
    }

    // Per-link crouch targets (read back so the report shows the actual pose).
    std::printf("[diag] crouch targets (link:target):");
    for (uint32_t link : cooked.actuated_links) {
        std::printf(" %u:%.3f", link, cooked.drive_targets[link]);
    }
    std::printf("\n");

    constexpr uint32_t kEnvCount = 64u;  // small N for the cheap invariant asserts.
    auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
    gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                    ground_height);
    auto targets = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
    auto stiffness = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
    auto damping = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
    auto limits = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);
    auto params = MakeStepParams(targets, stiffness, damping, limits);

    // Settle. Kp=60/Kd=4 is well-damped within the explicit-damping stable region
    // (see the gain constants); step long enough that the crouch ring-down has fully
    // decayed before we measure the tail.
    const uint32_t kSettleSteps = 1500u;
    for (uint32_t s = 0u; s < kSettleSteps; ++s) {
        bw.Step(params);
        if (s % 50u == 0u) {
            context.stream.Synchronize();
            articulation::ArticulationHostState dbg = batched;
            bw.Download(&dbg);
            const Quat r = dbg.base_pose[0].rotation;
            const Vec3 bx = r.Rotate(Vec3{1.0f, 0.0f, 0.0f});
            const Vec3 by = r.Rotate(Vec3{0.0f, 1.0f, 0.0f});
            const auto lam = bw.DownloadLambda();
            double sn = 0.0;
            for (uint32_t sl = 0u; sl < articulation::kMaxFootContactsPerEnv; ++sl)
                sn += lam[static_cast<size_t>(sl) * 3u + 0u];
            // roll = tilt of body-y out of world-z=0 plane (~y.z); pitch ~x.z.
            std::printf("[traj] s=%u z=%.4f tilt=%.4f pitch(x.z)=%.4f roll(y.z)=%.4f "
                        "SumLam/dt=%.1f\n",
                        s, dbg.base_pose[0].position.z,
                        TiltAngle(r), bx.z, by.z, sn / static_cast<double>(params.dt));
        }
    }
    context.stream.Synchronize();

    // Average the settled tail to suppress per-step PGS ripple, and re-run FK each
    // tail step for the per-foot seating / standing-height check.
    const uint32_t kTailSteps = 120u;
    nuka::phi::Buffer tail_pose(static_cast<size_t>(link_count) * sizeof(Transform),
                                nuka::phi::MemoryKind::Device);
    double sum_normal_force = 0.0;
    std::array<double, articulation::kMaxFootContactsPerEnv> sum_per_foot{};
    double sum_base_omega = 0.0;
    double sum_base_vz = 0.0;
    double sum_tilt = 0.0;
    double max_tilt = 0.0;
    std::vector<double> max_track_err(cooked.actuated_links.size(), 0.0);
    std::vector<double> last_q(cooked.actuated_links.size(), 0.0);
    float first_tail_z = 0.0f;
    float last_tail_z = 0.0f;
    float max_foot_seat_error = 0.0f;
    float min_trunk_above_feet = std::numeric_limits<float>::infinity();

    for (uint32_t s = 0u; s < kTailSteps; ++s) {
        bw.Step(params);
        context.stream.Synchronize();

        const auto lambda = bw.DownloadLambda();  // env-major, [slot*3]=(n,t1,t2).
        double step_normal = 0.0;
        for (uint32_t slot = 0u; slot < articulation::kMaxFootContactsPerEnv; ++slot) {
            const double n = lambda[static_cast<size_t>(slot) * 3u + 0u];
            step_normal += n;
            sum_per_foot[slot] += n / static_cast<double>(params.dt);
        }
        sum_normal_force += step_normal / static_cast<double>(params.dt);

        articulation::ArticulationHostState out = batched;
        bw.Download(&out);

        // (D) tipping: base angular velocity + tilt angle (env 0).
        const float* vroot = out.link_velocity[root].v;
        const double omega =
            std::sqrt(static_cast<double>(vroot[0]) * vroot[0] +
                      vroot[1] * vroot[1] + vroot[2] * vroot[2]);
        sum_base_omega += omega;
        sum_base_vz += std::fabs(vroot[5]);
        const double tilt = TiltAngle(out.base_pose[0].rotation);
        sum_tilt += tilt;
        max_tilt = std::max(max_tilt, tilt);

        // (A) tracking: each actuated q vs its target (env 0, global link index).
        for (size_t i = 0u; i < cooked.actuated_links.size(); ++i) {
            const uint32_t link = cooked.actuated_links[i];
            const double q = out.q[link];
            last_q[i] = q;
            const double err = std::fabs(q - cooked.drive_targets[link]);
            max_track_err[i] = std::max(max_track_err[i], err);
        }

        // (B) height + per-foot seating via FK.
        const float trunk_z = out.base_pose[0].position.z;
        if (s == 0u) first_tail_z = trunk_z;
        last_tail_z = trunk_z;
        auto devp = articulation::UploadArticulationState(context, out);
        articulation::UpdateWorldLinkPoses(context, devp.View(),
                                           static_cast<Transform*>(tail_pose.Data()));
        context.stream.Synchronize();
        std::vector<Transform> wp(link_count);
        tail_pose.CopyToHost(wp.data(), wp.size() * sizeof(Transform));
        float lowest_foot = std::numeric_limits<float>::infinity();
        for (const auto& f : cooked.feet) {
            const Transform calf = wp[f.calf_local_link];
            const Vec3 c = calf.position + calf.rotation.Rotate(f.local_offset);
            lowest_foot = std::min(lowest_foot, c.z - f.radius);
        }
        max_foot_seat_error =
            std::max(max_foot_seat_error, std::fabs(lowest_foot - ground_height));
        min_trunk_above_feet = std::min(min_trunk_above_feet, trunk_z - lowest_foot);
    }

    const double avg_normal_force = sum_normal_force / kTailSteps;
    const double avg_base_omega = sum_base_omega / kTailSteps;
    const double avg_base_vz = sum_base_vz / kTailSteps;
    const double avg_tilt = sum_tilt / kTailSteps;
    const float height_drift = last_tail_z - first_tail_z;

    double worst_track_err = 0.0;
    std::printf("[diag] (A) tracking (link:err:final_q):");
    for (size_t i = 0u; i < cooked.actuated_links.size(); ++i) {
        worst_track_err = std::max(worst_track_err, max_track_err[i]);
        std::printf(" %u:%.4f:%.4f", cooked.actuated_links[i], max_track_err[i],
                    last_q[i]);
    }
    std::printf("\n[diag] (A) worst joint tracking error = %.4f rad\n", worst_track_err);
    std::printf("[diag] (B) trunk z = %.4f m  drift = %.3e m  "
                "min trunk-above-feet = %.4f m  max foot-seat err = %.3e m\n",
                last_tail_z, height_drift, min_trunk_above_feet, max_foot_seat_error);
    std::printf("[diag] (C) Sigma(lambda_n)/dt = %.4f N  Mg = %.4f N  ratio = %.4f\n",
                avg_normal_force, mg, avg_normal_force / mg);
    std::printf("[diag] (C) per-foot avg N:");
    for (uint32_t slot = 0u; slot < articulation::kMaxFootContactsPerEnv; ++slot) {
        std::printf(" %.4f", sum_per_foot[slot] / kTailSteps);
    }
    std::printf("  (Mg/4 = %.4f)\n", mg_per_foot);
    std::printf("[diag] (D) avg tilt = %.4f rad  max tilt = %.4f rad  "
                "avg base |omega| = %.5f rad/s  avg base |v_z| = %.5f m/s\n",
                avg_tilt, max_tilt, avg_base_omega, avg_base_vz);

    // (A) PRIMARY DISCRIMINATOR: every actuated joint holds near its crouch target.
    // A real gravity sag (~tau_grav/Kp) is bounded; the bound must EXCLUDE a sprawl
    // (where thigh/calf would be off by ~1 rad), so 0.35 rad is loose enough for the
    // physical sag yet tight enough to fail a collapse.
    EXPECT_LT(worst_track_err, 0.35)
        << "a joint is far from its crouch target -- the robot is NOT standing "
           "(collapsed/sprawled), so Sigma~Mg alone would be a false pass";

    // (B) Standing height: trunk held well above the feet, drift bounded, feet
    // seated at the ground (not sunk through / launched).
    EXPECT_GT(min_trunk_above_feet, 0.20)
        << "trunk not held high above the feet -- crouch collapsed";
    EXPECT_LT(std::fabs(height_drift), 5.0e-3)
        << "trunk height drifting over the tail -- not settled";
    EXPECT_LT(max_foot_seat_error, 2.0e-2)
        << "feet did not stay seated at the ground (fall-through or lift-off)";
    EXPECT_TRUE(std::isfinite(last_tail_z))
        << "trunk diverged (non-finite height)";

    // (C) Weight support: total normal force ~ Mg (allow 10% for soft Baumgarte /
    // finite PGS), and each foot carries ~Mg/4 within a LOOSE tol (COM not centred).
    EXPECT_NEAR(avg_normal_force, mg, 0.10 * mg)
        << "Sigma(lambda_n)/dt does not balance the summed weight";
    for (uint32_t slot = 0u; slot < articulation::kMaxFootContactsPerEnv; ++slot) {
        const double per_foot = sum_per_foot[slot] / kTailSteps;
        EXPECT_NEAR(per_foot, mg_per_foot, 0.60 * mg_per_foot)
            << "foot slot " << slot << " normal force far from Mg/4 -- a foot is "
               "lifted (tipping) or the load is badly distributed";
    }

    // (D) No tipping: trunk stays upright in BOTH angle and rate.
    EXPECT_LT(max_tilt, 0.20)
        << "trunk tilt angle too large -- the robot is tipping";
    EXPECT_LT(avg_base_omega, 0.10)
        << "base angular velocity not ~0 -- load distribution wrong";
    EXPECT_LT(avg_base_vz, 0.05)
        << "base vertical velocity not ~0 -- not at vertical equilibrium";
}

// ---------------------------------------------------------------------------
// (E) DETERMINISM 0/0 on the PD-driven float path: two-run bit-identical AND
// N>=32 cross-replica bit-identical (q, qdot, base velocity, lambda).
// ---------------------------------------------------------------------------
TEST(Go2PdStanding, PdPathDeterministic) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    const uint32_t root = host.articulation_link_offset[0];
    const uint32_t link_count = host.TotalLinkCount();
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);

    const float ground_height = SeatGround(context, cooked);

    constexpr uint32_t kEnvCount = 32u;  // N>=32 cross-replica.
    const uint32_t kSteps = 200u;

    auto run = [&]() {
        auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
        gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                        ground_height);
        auto targets = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
        auto stiffness = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
        auto damping = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
        auto limits = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);
        auto params = MakeStepParams(targets, stiffness, damping, limits);
        for (uint32_t s = 0u; s < kSteps; ++s) bw.Step(params);
        context.stream.Synchronize();
        articulation::ArticulationHostState out = batched;
        bw.Download(&out);
        return std::make_pair(out, bw.DownloadLambda());
    };

    const auto [a_state, a_lambda] = run();
    const auto [b_state, b_lambda] = run();

    // Two-run bit-identical.
    size_t q_mis = 0u, qdot_mis = 0u, vroot_mis = 0u, lambda_mis = 0u;
    for (size_t i = 0u; i < a_state.q.size(); ++i) {
        if (a_state.q[i] != b_state.q[i]) ++q_mis;
        if (a_state.qdot[i] != b_state.qdot[i]) ++qdot_mis;
    }
    for (uint32_t env = 0u; env < kEnvCount; ++env) {
        const uint32_t r = env * link_count;
        for (uint32_t c = 0u; c < 6u; ++c) {
            if (a_state.link_velocity[r].v[c] != b_state.link_velocity[r].v[c]) {
                ++vroot_mis;
            }
        }
    }
    for (size_t i = 0u; i < a_lambda.size(); ++i) {
        if (a_lambda[i] != b_lambda[i]) ++lambda_mis;
    }
    std::printf("[diag] (E) two-run mismatches: q=%zu qdot=%zu v_root=%zu lambda=%zu\n",
                q_mis, qdot_mis, vroot_mis, lambda_mis);
    EXPECT_EQ(q_mis, 0u);
    EXPECT_EQ(qdot_mis, 0u);
    EXPECT_EQ(vroot_mis, 0u);
    EXPECT_EQ(lambda_mis, 0u);

    // Cross-replica bit-identical (replica 0 vs every other replica).
    const uint32_t per_env_slots = articulation::kMaxFootContactsPerEnv * 3u;
    size_t xq = 0u, xqdot = 0u, xvroot = 0u, xlambda = 0u;
    for (uint32_t env = 1u; env < kEnvCount; ++env) {
        const uint32_t base_off = env * link_count;
        for (uint32_t l = 0u; l < link_count; ++l) {
            if (a_state.q[base_off + l] != a_state.q[l]) ++xq;
            if (a_state.qdot[base_off + l] != a_state.qdot[l]) ++xqdot;
        }
        const uint32_t r = env * link_count;
        for (uint32_t c = 0u; c < 6u; ++c) {
            if (a_state.link_velocity[r].v[c] != a_state.link_velocity[root].v[c]) {
                ++xvroot;
            }
        }
        for (uint32_t i = 0u; i < per_env_slots; ++i) {
            if (a_lambda[env * per_env_slots + i] != a_lambda[i]) ++xlambda;
        }
    }
    std::printf("[diag] (E) cross-replica mismatches: q=%zu qdot=%zu v_root=%zu "
                "lambda=%zu\n", xq, xqdot, xvroot, xlambda);
    EXPECT_EQ(xq, 0u);
    EXPECT_EQ(xqdot, 0u);
    EXPECT_EQ(xvroot, 0u);
    EXPECT_EQ(xlambda, 0u);
}

// ---------------------------------------------------------------------------
// (F) 4096-env RUNNABILITY gate: the full production env count steps the same PD
// drive to completion with NO NaN/Inf in q / qdot / base-pose and no OOM, AND
// stays cross-replica + two-run bit-exact at 4096 (the runnability gate). One
// end-of-run readback (no per-tail FK loop -- too heavy at 4096).
// ---------------------------------------------------------------------------
TEST(Go2PdStanding, FourThousandEnvRunsCleanAndDeterministic) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    const uint32_t root = host.articulation_link_offset[0];
    const uint32_t link_count = host.TotalLinkCount();
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);

    const float ground_height = SeatGround(context, cooked);

    constexpr uint32_t kEnvCount = 4096u;
    const uint32_t kSteps = 600u;

    auto run = [&]() {
        auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
        gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                        ground_height);
        auto targets = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
        auto stiffness = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
        auto damping = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
        auto limits = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);
        auto params = MakeStepParams(targets, stiffness, damping, limits);
        for (uint32_t s = 0u; s < kSteps; ++s) bw.Step(params);
        context.stream.Synchronize();
        articulation::ArticulationHostState out = batched;
        bw.Download(&out);
        return std::make_pair(out, bw.DownloadLambda());
    };

    const auto [a_state, a_lambda] = run();
    const auto [b_state, b_lambda] = run();

    // No NaN/Inf across ALL 4096 envs: q, qdot, base pose (position + rotation).
    size_t nonfinite = 0u;
    for (size_t i = 0u; i < a_state.q.size(); ++i) {
        if (!std::isfinite(a_state.q[i])) ++nonfinite;
        if (!std::isfinite(a_state.qdot[i])) ++nonfinite;
    }
    for (uint32_t env = 0u; env < kEnvCount; ++env) {
        const auto& bp = a_state.base_pose[env];
        if (!std::isfinite(bp.position.x) || !std::isfinite(bp.position.y) ||
            !std::isfinite(bp.position.z)) {
            ++nonfinite;
        }
        const auto& r = bp.rotation;
        if (!std::isfinite(r.w) || !std::isfinite(r.x) || !std::isfinite(r.y) ||
            !std::isfinite(r.z)) {
            ++nonfinite;
        }
        const float* v = a_state.link_velocity[env * link_count].v;
        for (uint32_t c = 0u; c < 6u; ++c) {
            if (!std::isfinite(v[c])) ++nonfinite;
        }
    }
    std::printf("[diag] (F) 4096-env non-finite count = %zu\n", nonfinite);
    EXPECT_EQ(nonfinite, 0u) << "NaN/Inf in the 4096-env state";

    // Spot-check the settled standing pose at 4096 (env 0): trunk above ground.
    std::printf("[diag] (F) env0 trunk z = %.4f  base rot w = %.5f\n",
                a_state.base_pose[0].position.z, a_state.base_pose[0].rotation.w);
    EXPECT_GT(a_state.base_pose[0].position.z, ground_height)
        << "env0 trunk sank below the ground at 4096 envs";

    // Two-run + cross-replica bit-exact at 4096 (the runnability/determinism gate).
    size_t q_mis = 0u, qdot_mis = 0u, lambda_mis = 0u;
    for (size_t i = 0u; i < a_state.q.size(); ++i) {
        if (a_state.q[i] != b_state.q[i]) ++q_mis;
        if (a_state.qdot[i] != b_state.qdot[i]) ++qdot_mis;
    }
    for (size_t i = 0u; i < a_lambda.size(); ++i) {
        if (a_lambda[i] != b_lambda[i]) ++lambda_mis;
    }
    std::printf("[diag] (F) 4096 two-run mismatches: q=%zu qdot=%zu lambda=%zu\n",
                q_mis, qdot_mis, lambda_mis);
    EXPECT_EQ(q_mis, 0u);
    EXPECT_EQ(qdot_mis, 0u);
    EXPECT_EQ(lambda_mis, 0u);

    size_t xq = 0u, xqdot = 0u, xvroot = 0u;
    for (uint32_t env = 1u; env < kEnvCount; ++env) {
        const uint32_t base_off = env * link_count;
        for (uint32_t l = 0u; l < link_count; ++l) {
            if (a_state.q[base_off + l] != a_state.q[l]) ++xq;
            if (a_state.qdot[base_off + l] != a_state.qdot[l]) ++xqdot;
        }
        const uint32_t r = env * link_count;
        for (uint32_t c = 0u; c < 6u; ++c) {
            if (a_state.link_velocity[r].v[c] != a_state.link_velocity[root].v[c]) {
                ++xvroot;
            }
        }
    }
    std::printf("[diag] (F) 4096 cross-replica mismatches: q=%zu qdot=%zu v_root=%zu\n",
                xq, xqdot, xvroot);
    EXPECT_EQ(xq, 0u);
    EXPECT_EQ(xqdot, 0u);
    EXPECT_EQ(xvroot, 0u);
}
