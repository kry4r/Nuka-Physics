// ---------------------------------------------------------------------------
// p01-F T8a -- floating-base (free 6-DOF root) Featherstone ABA
//
// Validates the floating-base ABA path added to featherstone_aba.cu /
// articulation_contacts.cu / articulation_cooker.cpp / articulation_state.cpp,
// all gated on ArticulationJointType::FloatingBase so the fixed-base path stays
// byte-for-byte identical (proven by the separate Go2Stand golden + batched
// equivalence gates).
//
// Correctness strategy (advisor-guided):
//
//  (1) PRIMARY -- MOMENTUM CONSERVATION. Build a SMALL floating articulation
//      programmatically (base + two revolute links) for tight control. Start
//      from FULL REST (base velocity 0, joint velocities 0) under ZERO gravity,
//      then drive the joints with applied torques (tau). With no external force
//      and no gravity, total LINEAR momentum P and total ANGULAR momentum L_cm
//      about the CoM are exactly conserved at their initial values P(0)=0,
//      L(0)=0 -- internal joint torques cannot change them (Newton's third law).
//      So both must stay ~0 over many steps WHILE THE BODY VISIBLY MOVES (the
//      base recoils as the joints accelerate). A wrong base/joint inertia
//      coupling breaks the recoil cancellation and P/L drift. We also assert the
//      motion is non-trivial (base velocity and qdot become clearly nonzero) so
//      we are not passing the all-zeros solution.
//
//      Momentum is computed INDEPENDENTLY of the operator under test: from link
//      masses + the Pass-1 spatial velocities, transformed from each link frame
//      to world frame using the live world poses. h_i = I_i v_i (link frame);
//      P_i = R_i * h_i.linear; L_i = R_i * h_i.angular + o_i x P_i; sum, then
//      L_cm = L - r_cm x P.
//
//  (2) SECONDARY -- FREE FALL. Under gravity with no contacts, the base spatial
//      acceleration ~ (0,0,-g) at t=0 (identity orientation) and the CoM falls
//      like -1/2 g t^2. Confirms gravity enters EXACTLY ONCE (not twice/zero).
//
//  (3) LIVE-POSE FK. Translate/rotate base_pose, run UpdateWorldLinkPoses, and
//      assert the resulting world poses reflect the moved base (FK seeded from
//      the live base_pose, not the static cook pose).
//
//  (4) DETERMINISM (D1). Floating dynamics are bit-identical across two runs and
//      across replicated envs.
//
//  (5) REAL-PATH SMOKE. Cook examples/scenes/go2_float.usda, confirm the root
//      cooks FloatingBase, create the batched world, step it, and confirm the
//      base actually MOVES (finite, non-frozen).
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/gpu/batched_articulated_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// ---------------------------------------------------------------------------
// Programmatic small floating articulation: a FloatingBase trunk + two revolute
// links in series. All per-link physical data lives in the topology vectors so
// the validated BuildArticulationHostState fills the flat device arrays for us
// (this reuses the cooked-field-filling path instead of hand-poking 25 arrays).
// ---------------------------------------------------------------------------
struct SmallFloatingModel {
    articulation::ArticulationHostState host;
    std::vector<float> link_mass;          // per global link
    std::vector<Vec3> link_inertia_diag;   // per global link (about COM)
    uint32_t link_count = 0u;
};

SmallFloatingModel BuildSmallFloatingModel() {
    articulation::ArticulationCookedTopology topo;

    // 3 links: 0 = floating base, 1 = link about X at +0.30 along base x,
    // 2 = link about Y at +0.25 further out. Masses/inertias chosen so the base
    // is heavier (a real recoil is visible but not degenerate).
    const float kBaseMass = 4.0f;
    const float kLink1Mass = 1.0f;
    const float kLink2Mass = 0.6f;
    const Vec3 kBaseInertia{0.08f, 0.07f, 0.05f};
    const Vec3 kLink1Inertia{0.010f, 0.012f, 0.006f};
    const Vec3 kLink2Inertia{0.004f, 0.005f, 0.0025f};

    topo.root_body = 0u;
    topo.link_bodies = {0u, 1u, 2u};
    topo.parent_links = {kInvalidLink, 0u, 1u};
    topo.joint_types = {articulation::ArticulationJointType::FloatingBase,
                        articulation::ArticulationJointType::Revolute,
                        articulation::ArticulationJointType::Revolute};
    topo.joint_axes = {Vec3::UnitZ(), Vec3::UnitX(), Vec3::UnitY()};

    // parent_frames -> parent_offset (translation of child frame in parent).
    topo.parent_frames = {Transform::Identity(),
                          Transform{Vec3{0.30f, 0.0f, 0.0f}, Quat::Identity()},
                          Transform{Vec3{0.25f, 0.0f, 0.0f}, Quat::Identity()}};
    topo.child_frames = {Transform::Identity(), Transform::Identity(),
                         Transform::Identity()};
    // local_poses: each link's rest frame relative to its parent (identity here,
    // the joint offset lives in parent_frames). The root's local_pose is the base
    // body frame at the origin.
    topo.local_poses = {Transform::Identity(), Transform::Identity(),
                        Transform::Identity()};
    // inertial_frames: COM at the link origin, principal axes aligned (identity).
    topo.inertial_frames = {Transform::Identity(), Transform::Identity(),
                            Transform::Identity()};
    topo.initial_positions = {0.0f, 0.0f, 0.0f};
    topo.joint_dampings = {0.0f, 0.0f, 0.0f};
    topo.joint_armatures = {0.0f, 0.0f, 0.0f};
    topo.masses = {kBaseMass, kLink1Mass, kLink2Mass};
    topo.inertias = {kBaseInertia, kLink1Inertia, kLink2Inertia};

    // Minimal body table -- the topology carries the physical data, so the body
    // table only needs to be large enough to be indexed by link_bodies. Poses
    // identity (base starts at origin); BuildArticulationHostState seeds base_pose
    // from PoseForBody(root) = identity.
    nuka::scene::CookedBodyTable bodies;
    bodies.poses = {Transform::Identity(), Transform::Identity(), Transform::Identity()};
    bodies.local_poses = bodies.poses;
    bodies.inertial_frames = bodies.poses;
    bodies.masses = {kBaseMass, kLink1Mass, kLink2Mass};
    bodies.inertias = {kBaseInertia, kLink1Inertia, kLink2Inertia};
    bodies.is_static = {0u, 0u, 0u};

    SmallFloatingModel model;
    model.host = articulation::BuildArticulationHostState({topo}, bodies);
    model.link_count = model.host.TotalLinkCount();
    model.link_mass = {kBaseMass, kLink1Mass, kLink2Mass};
    model.link_inertia_diag = {kBaseInertia, kLink1Inertia, kLink2Inertia};
    return model;
}

// Runs ABA (Pass 1-3) for the current state, then one floating-base velocity +
// joint velocity + position integrate step pair, mirroring the batched step's
// split order: floating-velocity & joint-velocity together (the "pre-contact"
// half here has no contacts), then floating-pose & joint-position.
void StepFloating(const nuka::phi::DeviceContext& context,
                  articulation::ArticulationDeviceState view,
                  float gravity_z,
                  float dt) {
    articulation::FeatherstoneAba::ComputeAccelerations(context, view, gravity_z);
    articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(context, view, dt,
                                                                 gravity_z);
    articulation::FeatherstoneAba::IntegrateVelocity(context, view, dt);
    articulation::FeatherstoneAba::IntegrateFloatingBasePose(context, view, dt);
    articulation::FeatherstoneAba::IntegratePosition(context, view, dt);
}

struct Momentum {
    Vec3 linear{0.0f, 0.0f, 0.0f};
    Vec3 angular_cm{0.0f, 0.0f, 0.0f};  // about the system CoM
    Vec3 com{0.0f, 0.0f, 0.0f};
};

// World-frame momentum from link masses + Pass-1 spatial velocities + live world
// poses. Spatial convention is [angular(0:2), linear(3:5)] in LINK frame; spatial
// inertia I_i maps it to spatial momentum h_i = I_i v_i in link frame. Then:
//   P_i = R_i * h_i.linear
//   L_i(about world origin) = R_i * h_i.angular + o_i x P_i
// where R_i, o_i are link i's world rotation/origin. L_cm = sum L_i - r_cm x P.
// This shares no code with the ABA solve under test (independent cross-check).
Momentum ComputeWorldMomentum(const articulation::ArticulationHostState& host,
                              const std::vector<Transform>& world_pose,
                              const std::vector<float>& link_mass) {
    const uint32_t n = host.TotalLinkCount();
    Vec3 total_p{0.0f, 0.0f, 0.0f};
    Vec3 total_l_origin{0.0f, 0.0f, 0.0f};
    Vec3 mass_weighted_pos{0.0f, 0.0f, 0.0f};
    float total_mass = 0.0f;

    for (uint32_t link = 0u; link < n; ++link) {
        const float* I = host.link_inertia[link].I;
        const float* v = host.link_velocity[link].v;
        // h = I v (6x6 * 6), link frame.
        double h[6];
        for (uint32_t r = 0u; r < 6u; ++r) {
            double s = 0.0;
            for (uint32_t c = 0u; c < 6u; ++c) {
                s += static_cast<double>(I[r * 6u + c]) * static_cast<double>(v[c]);
            }
            h[r] = s;
        }
        const Vec3 h_ang{static_cast<float>(h[0]), static_cast<float>(h[1]),
                         static_cast<float>(h[2])};
        const Vec3 h_lin{static_cast<float>(h[3]), static_cast<float>(h[4]),
                         static_cast<float>(h[5])};

        const Transform& pose = world_pose[link];
        const Vec3 p_world = pose.rotation.Rotate(h_lin);
        const Vec3 l_world = pose.rotation.Rotate(h_ang) + pose.position.Cross(p_world);

        total_p = total_p + p_world;
        total_l_origin = total_l_origin + l_world;

        const float m = link_mass[link];
        total_mass += m;
        mass_weighted_pos = mass_weighted_pos + pose.position * m;
    }

    Momentum out;
    out.linear = total_p;
    out.com = total_mass > 0.0f ? (mass_weighted_pos * (1.0f / total_mass))
                                : Vec3{0.0f, 0.0f, 0.0f};
    // L about CoM = L about origin - r_cm x P.
    out.angular_cm = total_l_origin - out.com.Cross(total_p);
    return out;
}

// Downloads link_velocity + base_pose and computes the live world poses via the
// device FK (UpdateWorldLinkPoses), which seeds the root from base_pose for a
// floating root. Returns (host snapshot, world poses).
struct StateSnapshot {
    articulation::ArticulationHostState host;
    std::vector<Transform> world_pose;
};

StateSnapshot SnapshotWithPoses(const nuka::phi::DeviceContext& context,
                                articulation::ArticulationDeviceBuffers& device,
                                const articulation::ArticulationHostState& proto) {
    auto view = device.View();
    nuka::phi::Buffer pose_buf(
        static_cast<size_t>(proto.TotalLinkCount()) * sizeof(Transform),
        nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, view,
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();

    StateSnapshot snap;
    snap.host = proto;
    articulation::DownloadArticulationState(device, &snap.host);
    snap.world_pose.resize(proto.TotalLinkCount());
    pose_buf.CopyToHost(snap.world_pose.data(),
                        snap.world_pose.size() * sizeof(Transform));
    return snap;
}

} // namespace

// ---------------------------------------------------------------------------
// (1a) PRIMARY DISCRIMINATOR: instantaneous CoM linear acceleration sum.
//
// Zero gravity, base at identity, full rest, joint torques applied. Run ABA
// ONCE (NO integration). With v=0 and each link's COM at its body origin
// (inertial_frame = identity in this model), the world-frame CoM linear accel of
// link i is exactly R_i * a_i[3:6] (no centripetal/Coriolis at v=0). Internal
// joint torques CANNOT accelerate the system CoM, so:
//     sum_i m_i * (R_i * a_i.linear)  ==  0   to MACHINE PRECISION.
// This isolates the base/joint inertia coupling (Pass-2 Ia accumulation +
// Pass-3 root 6x6 solve) from ALL integration/frame-transport error: a wrong
// coupling leaves a residual at the scale of the individual terms; truncation
// is absent (no integration). This is the non-fakeable proof of correct
// coupling -- it cannot be softened by changing the trajectory.
// ---------------------------------------------------------------------------
TEST(FloatingBaseAba, InstantaneousComAccelerationSumIsZero) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto model = BuildSmallFloatingModel();
    for (auto& a : model.host.link_velocity) {
        for (float& c : a.v) c = 0.0f;
    }
    for (auto& q : model.host.qdot) q = 0.0f;
    model.host.tau = {0.0f, 1.2f, -0.7f};  // nonzero joint torques.
    // Base at a non-identity orientation so the R_i rotations are exercised (the
    // body-frame accel must be transported to world correctly for the sum to
    // cancel). This stresses the frame handling, not just the zero solution.
    model.host.base_pose[0].rotation =
        Quat::FromAxisAngle(Vec3{0.3f, 0.5f, 0.2f}.Normalized(), 0.6f).Normalized();

    auto device = articulation::UploadArticulationState(context, model.host);
    auto view = device.View();
    // ABA only (no integrate). gravity = 0: pure internal-torque dynamics.
    articulation::FeatherstoneAba::ComputeAccelerations(context, view, 0.0f);
    // World poses seeded from the (rotated) live base_pose.
    nuka::phi::Buffer pose_buf(
        static_cast<size_t>(model.link_count) * sizeof(Transform),
        nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, view,
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();

    auto host = model.host;
    articulation::DownloadArticulationState(device, &host);
    std::vector<Transform> world(model.link_count);
    pose_buf.CopyToHost(world.data(), world.size() * sizeof(Transform));

    // sum_i m_i R_i a_i.linear, and the term scale for a relative tolerance.
    Vec3 com_accel{0.0f, 0.0f, 0.0f};
    double term_scale = 0.0;
    for (uint32_t link = 0u; link < model.link_count; ++link) {
        const float* a = host.link_acceleration[link].a;
        const Vec3 a_lin_body{a[3], a[4], a[5]};
        const Vec3 a_lin_world = world[link].rotation.Rotate(a_lin_body);
        const float m = model.link_mass[link];
        com_accel = com_accel + a_lin_world * m;
        term_scale += static_cast<double>(m) * a_lin_world.Length();
    }
    const double residual = com_accel.Length();
    const double rel = residual / std::max(term_scale, 1.0e-12);
    std::printf("[diag] (1a) instantaneous |sum m_i R_i a_lin| = %.3e  "
                "term_scale=%.3e  rel=%.3e\n",
                residual, term_scale, rel);

    // Term scale must be non-trivial (real internal accelerations, not all zero).
    EXPECT_GT(term_scale, 1.0e-2) << "no real internal dynamics (torques did nothing)";
    // Machine-precision cancellation: internal torques do not move the CoM. A
    // few-ulp relative residual is expected from the single-precision device
    // solve; 1e-5 relative is a tight, coupling-sensitive bound.
    EXPECT_LT(rel, 1.0e-5)
        << "system CoM accelerated under internal torques -- base/joint inertia "
           "coupling is WRONG (not an integration artifact)";

    // The base must really participate (its spatial accel is nonzero: the base
    // recoils against the joint torques). a_0 is the apparent accel; with the
    // joints torqued it is nonzero even at rest.
    const float* a0 = host.link_acceleration[0].a;
    float a0_norm = 0.0f;
    for (uint32_t i = 0u; i < 6u; ++i) a0_norm += a0[i] * a0[i];
    EXPECT_GT(std::sqrt(a0_norm), 1.0e-3f) << "base did not recoil -- root frozen";
}

// ---------------------------------------------------------------------------
// (1b) PRIMARY: trajectory momentum conservation + dt-scaling discriminator.
//
// From full rest under a brief joint-torque pulse then coast (zero gravity), in
// a GENTLE regime (omega*dt << 1). P(0)=0, L(0)=0 are conserved exactly by the
// continuous dynamics; the discrete integrator leaks at O(dt). We run the SAME
// physical duration at dt, dt/2, dt/4 and assert the peak |P| and |L_cm - L0|
// SHRINK ~linearly with dt. A coupling bug leaves a dt-independent drift; only
// truncation vanishes as dt->0. This sets the tolerance honestly (by the
// observed truncation rate) instead of by trial, and proves the integrator +
// frame transport (which 1a skips) are correct too.
// ---------------------------------------------------------------------------
TEST(FloatingBaseAba, TrajectoryMomentumConservedAndDriftScalesWithDt) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    // One physical run: torque pulse for `pulse_time`, then coast to `total_time`.
    // Samples momentum EVERY step; returns peak |P|, peak |L_cm - L0|, and the
    // peak base speed / qdot (to confirm the gentle regime + non-trivial motion).
    struct RunResult {
        float max_lin = 0.0f;
        float max_ang_dev = 0.0f;
        float max_base_omega_dt = 0.0f;
        float max_base_speed = 0.0f;
        float max_qdot = 0.0f;
        float com_drift = 0.0f;
    };
    auto run = [&](float dt) -> RunResult {
        auto model = BuildSmallFloatingModel();
        for (auto& a : model.host.link_velocity) {
            for (float& c : a.v) c = 0.0f;
        }
        for (auto& q : model.host.qdot) q = 0.0f;
        const float kPulseTau1 = 0.05f;   // small torques -> gentle velocities.
        const float kPulseTau2 = -0.03f;
        const float kPulseTime = 0.05f;   // 50 ms pulse, then coast.
        const float kTotalTime = 0.5f;
        const uint32_t steps = static_cast<uint32_t>(kTotalTime / dt + 0.5f);
        const uint32_t pulse_steps = static_cast<uint32_t>(kPulseTime / dt + 0.5f);

        auto device = articulation::UploadArticulationState(context, model.host);
        auto snap0 = SnapshotWithPoses(context, device, model.host);
        const Momentum m0 = ComputeWorldMomentum(snap0.host, snap0.world_pose,
                                                 model.link_mass);
        RunResult r;
        float max_base_omega = 0.0f;
        Vec3 com_first = m0.com;
        Vec3 com_last = m0.com;
        for (uint32_t step = 0u; step < steps; ++step) {
            // Apply the torque pulse only during the pulse window, then coast.
            articulation::ArticulationHostState patch = model.host;
            const float t1 = step < pulse_steps ? kPulseTau1 : 0.0f;
            const float t2 = step < pulse_steps ? kPulseTau2 : 0.0f;
            std::vector<float> tau = {0.0f, t1, t2};
            // Upload only tau (cheap path: write into the device tau buffer).
            device.tau.CopyFromHost(tau.data(), tau.size() * sizeof(float));

            auto view = device.View();
            StepFloating(context, view, 0.0f, dt);

            auto snap = SnapshotWithPoses(context, device, model.host);
            const Momentum m = ComputeWorldMomentum(snap.host, snap.world_pose,
                                                    model.link_mass);
            r.max_lin = std::max(r.max_lin, m.linear.Length());
            r.max_ang_dev = std::max(r.max_ang_dev,
                                     (m.angular_cm - m0.angular_cm).Length());
            const float* bv = snap.host.link_velocity[0].v;
            const float base_omega = std::sqrt(bv[0] * bv[0] + bv[1] * bv[1] +
                                               bv[2] * bv[2]);
            r.max_base_omega_dt = std::max(r.max_base_omega_dt, base_omega * dt);
            max_base_omega = std::max(max_base_omega, base_omega);
            r.max_base_speed = std::max(r.max_base_speed,
                                        std::sqrt(bv[3] * bv[3] + bv[4] * bv[4] +
                                                  bv[5] * bv[5]));
            for (float qd : snap.host.qdot) {
                r.max_qdot = std::max(r.max_qdot, std::fabs(qd));
            }
            com_last = m.com;
        }
        r.com_drift = (com_last - com_first).Length();
        // Fold the base angular recoil into the recorded base-speed proxy: the
        // base recoils against the joint torques predominantly in rotation here,
        // so the recoil magnitude is captured by max(linear speed, angular rate).
        r.max_base_speed = std::max(r.max_base_speed, max_base_omega);
        return r;
    };

    const float kDt = 1.0e-3f;
    const RunResult r1 = run(kDt);
    const RunResult r2 = run(kDt * 0.5f);
    const RunResult r4 = run(kDt * 0.25f);

    std::printf("[diag] (1b) dt=%.1e: max|P|=%.3e max|L_cm-L0|=%.3e "
                "base_omega*dt=%.3e base_speed=%.4f max|qdot|=%.4f com_drift=%.3e\n",
                kDt, r1.max_lin, r1.max_ang_dev, r1.max_base_omega_dt,
                r1.max_base_speed, r1.max_qdot, r1.com_drift);
    std::printf("[diag] (1b) dt=%.1e: max|P|=%.3e max|L_cm-L0|=%.3e\n",
                kDt * 0.5f, r2.max_lin, r2.max_ang_dev);
    std::printf("[diag] (1b) dt=%.1e: max|P|=%.3e max|L_cm-L0|=%.3e\n",
                kDt * 0.25f, r4.max_lin, r4.max_ang_dev);
    std::printf("[diag] (1b) drift ratios P: dt/(dt/2)=%.2f (dt/2)/(dt/4)=%.2f "
                "(expect ~2 for O(dt) truncation)\n",
                r1.max_lin / std::max(r2.max_lin, 1.0e-12f),
                r2.max_lin / std::max(r4.max_lin, 1.0e-12f));

    // Gentle regime: per-step base rotation is small (linearization valid).
    EXPECT_LT(r1.max_base_omega_dt, 5.0e-2f) << "regime not gentle (omega*dt too large)";
    // Non-trivial motion: base recoils (linear or angular), joints move. The base
    // recoil is what a wrong coupling would suppress -- proves we are not passing
    // the all-zeros solution.
    EXPECT_GT(r1.max_base_speed, 1.0e-3f) << "base never recoiled (no linear or angular motion)";
    EXPECT_GT(r1.max_qdot, 1.0e-2f) << "joints never moved";

    // dt-SCALING DISCRIMINATOR: halving dt must (roughly) halve the drift. A
    // coupling bug would leave a floor that does NOT shrink with dt. We require
    // each halving to cut the drift by at least 1.6x (allowing for the coast
    // phase and single-precision noise; pure O(dt) gives 2.0x). This is the
    // honest proof that the residual is integration truncation, not a wrong
    // operator.
    EXPECT_GT(r1.max_lin / std::max(r2.max_lin, 1.0e-12f), 1.6f)
        << "linear-momentum drift did not shrink with dt -> coupling bug, not truncation";
    EXPECT_GT(r2.max_lin / std::max(r4.max_lin, 1.0e-12f), 1.6f)
        << "linear-momentum drift did not shrink with dt -> coupling bug, not truncation";
    EXPECT_GT(r1.max_ang_dev / std::max(r2.max_ang_dev, 1.0e-12f), 1.6f)
        << "angular-momentum drift did not shrink with dt -> coupling bug, not truncation";

    // With the drift confirmed as O(dt) truncation, the absolute level at the
    // finest dt is small. (Loose absolute bound; the SCALING above is the real
    // check.)
    EXPECT_LT(r4.max_lin, 5.0e-3f) << "linear-momentum drift implausibly large at fine dt";
}

// ---------------------------------------------------------------------------
// (2) SECONDARY: free fall -- gravity enters exactly once.
// ---------------------------------------------------------------------------
TEST(FloatingBaseAba, FreeFallBaseAccelerationAndComDrop) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto model = BuildSmallFloatingModel();
    for (auto& a : model.host.link_velocity) {
        for (float& c : a.v) c = 0.0f;
    }
    for (auto& q : model.host.qdot) q = 0.0f;
    for (auto& t : model.host.tau) t = 0.0f;  // no joint torques: pure free fall.

    // Production convention: gravity_z is the SIGNED z-component of gravity
    // acceleration (g_z = -9.81). GravityAcceleration sets the base seed accel
    // out[5] = -gravity_z. For a free-falling base the physical z-acceleration
    // must be g_z = -9.81 (downward), so the CoM drops by 1/2 |g| t^2.
    const float kGravityZ = -9.81f;
    const float kGmag = 9.81f;
    const float kDt = 1.0e-3f;
    const uint32_t kSteps = 500u; // 0.5 s.

    auto device = articulation::UploadArticulationState(context, model.host);

    // t=0 checks. link_acceleration[root] holds the APPARENT (gravity-frame)
    // spatial accel a_0 = (Ia)^-1(-p); at rest p=0 so a_0 ~ 0 (a free-falling
    // base has zero apparent accel). The REAL base accel is a_0 - a_grav, which
    // the velocity integrator applies, so after one velocity step the base z
    // velocity must be ~ g_z*dt = -|g|*dt (the base really falls).
    {
        auto view = device.View();
        articulation::FeatherstoneAba::ComputeAccelerations(context, view, kGravityZ);
        articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(context, view,
                                                                     kDt, kGravityZ);
        context.stream.Synchronize();
        auto host = model.host;
        articulation::DownloadArticulationState(device, &host);
        const float* a = host.link_acceleration[0].a;
        const float* v = host.link_velocity[0].v;
        std::printf("[diag] (2) apparent base accel a_0 t=0 = "
                    "(%.4f,%.4f,%.4f | %.4f,%.4f,%.4f)\n",
                    a[0], a[1], a[2], a[3], a[4], a[5]);
        std::printf("[diag] (2) base vel after 1 step = "
                    "(%.4f,%.4f,%.4f | %.6f,%.6f,%.6f)  expect vz~%.6f\n",
                    v[0], v[1], v[2], v[3], v[4], v[5], -kGmag * kDt);
        // Apparent accel ~ 0 at rest (no double-counted gravity in a_0).
        for (uint32_t i = 0u; i < 6u; ++i) {
            EXPECT_NEAR(a[i], 0.0f, 1.0e-2f)
                << "apparent base accel a_0[" << i << "] != 0 at rest";
        }
        // Real base z velocity after one step ~ g_z*dt (downward) => correct sign.
        EXPECT_NEAR(v[5], -kGmag * kDt, 1.0e-3f * kGmag)
            << "base did not start falling at g_z (gravity sign/magnitude wrong)";
        EXPECT_NEAR(v[0], 0.0f, 1.0e-4f);
        EXPECT_NEAR(v[3], 0.0f, 1.0e-4f);
        EXPECT_NEAR(v[4], 0.0f, 1.0e-4f);
    }

    // Re-upload (the partial step above mutated state) and integrate fully.
    device = articulation::UploadArticulationState(context, model.host);
    const float z0 = model.host.base_pose[0].position.z;
    for (uint32_t step = 0u; step < kSteps; ++step) {
        auto view = device.View();
        StepFloating(context, view, kGravityZ, kDt);
    }
    auto host = model.host;
    articulation::DownloadArticulationState(device, &host);
    const float z_end = host.base_pose[0].position.z;
    const float t = kSteps * kDt;
    const float drop_expected = 0.5f * kGmag * t * t;  // semi-implicit Euler ~ analytic.
    const float drop_actual = z0 - z_end;
    std::printf("[diag] (2) free-fall drop after %.3fs: actual=%.5f expected=%.5f\n",
                t, drop_actual, drop_expected);
    // Semi-implicit Euler overshoots the analytic 1/2 g t^2 by ~ (1 + dt/t); with
    // dt=1ms, t=0.5s the bias is ~0.1%. A 5% band is comfortably tight and would
    // catch gravity being doubled (2x) or zeroed.
    EXPECT_NEAR(drop_actual, drop_expected, 0.05f * drop_expected)
        << "CoM did not fall like 1/2 g t^2 (gravity double-counted or cancelled)";
    EXPECT_TRUE(std::isfinite(z_end));
}

// ---------------------------------------------------------------------------
// (2b) TILTED FREE FALL: gravity is world-vertical at ANY base orientation.
//
// The base velocity integrator rotates the world gravity pseudo-accel into the
// body frame: g_body = RotateByQuatInverse(base_rot, g_world). At identity
// orientation (test 2) that rotation is a no-op, so it is unchecked there. Here
// we tilt the base ~30 deg off vertical with NO joint torque and integrate under
// gravity. Gravity acts world-down regardless of orientation, so the base must
// fall VERTICALLY: world x/y stay ~0 while z drops ~1/2 g t^2. If the rotation
// used R instead of R^T (a conjugate-direction slip), the fall would tilt and
// x/y would drift at O(g t^2) -- caught immediately. This is the decisive check
// of the body-frame gravity rotation, in the corner where the first sign bug hid.
// ---------------------------------------------------------------------------
TEST(FloatingBaseAba, TiltedBaseFreeFallStaysVertical) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto model = BuildSmallFloatingModel();
    for (auto& a : model.host.link_velocity) {
        for (float& c : a.v) c = 0.0f;
    }
    for (auto& q : model.host.qdot) q = 0.0f;
    for (auto& t : model.host.tau) t = 0.0f;  // pure free fall, no joint drive.

    // Tilt the base ~30 deg about a non-axis-aligned direction.
    const Quat kTilt =
        Quat::FromAxisAngle(Vec3{1.0f, 0.6f, -0.3f}.Normalized(), 0.5236f).Normalized();
    model.host.base_pose[0].rotation = kTilt;
    model.host.base_pose[0].position = Vec3{0.0f, 0.0f, 1.0f};

    const float kGravityZ = -9.81f;
    const float kGmag = 9.81f;
    const float kDt = 1.0e-3f;
    const uint32_t kSteps = 400u;  // 0.4 s.

    auto device = articulation::UploadArticulationState(context, model.host);
    const Vec3 p0 = model.host.base_pose[0].position;
    for (uint32_t step = 0u; step < kSteps; ++step) {
        auto view = device.View();
        StepFloating(context, view, kGravityZ, kDt);
    }
    auto host = model.host;
    articulation::DownloadArticulationState(device, &host);
    const Vec3 p_end = host.base_pose[0].position;

    const float t = kSteps * kDt;
    const float drop_expected = 0.5f * kGmag * t * t;
    const float drop_actual = p0.z - p_end.z;
    const float lateral = std::sqrt((p_end.x - p0.x) * (p_end.x - p0.x) +
                                    (p_end.y - p0.y) * (p_end.y - p0.y));
    std::printf("[diag] (2b) tilted free-fall: pos (%.5f,%.5f,%.5f)->(%.5f,%.5f,%.5f) "
                "drop=%.5f expected=%.5f lateral=%.3e\n",
                p0.x, p0.y, p0.z, p_end.x, p_end.y, p_end.z, drop_actual,
                drop_expected, lateral);

    // Falls vertically: world x/y essentially unchanged (gravity is world-down,
    // not body-down -- proves the R^T rotation maps world gravity correctly).
    EXPECT_LT(lateral, 1.0e-3f)
        << "tilted base drifted laterally -- body-frame gravity rotation is wrong "
           "(R vs R^T slip)";
    // And still drops ~1/2 g t^2.
    EXPECT_NEAR(drop_actual, drop_expected, 0.05f * drop_expected)
        << "tilted base did not fall at g";
    EXPECT_TRUE(std::isfinite(p_end.x) && std::isfinite(p_end.y) &&
                std::isfinite(p_end.z));
}

// ---------------------------------------------------------------------------
// (3) LIVE-POSE FK: UpdateWorldLinkPoses seeds the root from the live base_pose.
// ---------------------------------------------------------------------------
TEST(FloatingBaseAba, LivePoseFkSeedsFromBasePose) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto model = BuildSmallFloatingModel();

    // Move + rotate the base by a known amount in the LIVE base_pose only (the
    // static link_pose stays at the cook pose).
    const Vec3 kShift{1.5f, -2.0f, 0.75f};
    const Quat kRot = Quat::FromAxisAngle(Vec3::UnitZ(), 0.7f).Normalized();
    model.host.base_pose[0].position = kShift;
    model.host.base_pose[0].rotation = kRot;

    auto device = articulation::UploadArticulationState(context, model.host);
    auto view = device.View();
    nuka::phi::Buffer pose_buf(
        static_cast<size_t>(model.link_count) * sizeof(Transform),
        nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, view,
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> world(model.link_count);
    pose_buf.CopyToHost(world.data(), world.size() * sizeof(Transform));

    std::printf("[diag] (3) root world pos=(%.4f,%.4f,%.4f) rot=(%.4f,%.4f,%.4f,%.4f)\n",
                world[0].position.x, world[0].position.y, world[0].position.z,
                world[0].rotation.w, world[0].rotation.x, world[0].rotation.y,
                world[0].rotation.z);

    // Root world pose == the live base_pose (FK seeded from base_pose).
    EXPECT_NEAR(world[0].position.x, kShift.x, 1.0e-5f);
    EXPECT_NEAR(world[0].position.y, kShift.y, 1.0e-5f);
    EXPECT_NEAR(world[0].position.z, kShift.z, 1.0e-5f);
    EXPECT_NEAR(world[0].rotation.w, kRot.w, 1.0e-5f);
    EXPECT_NEAR(world[0].rotation.z, kRot.z, 1.0e-5f);

    // Child link 1 sits at base + R(base) * (parent_offset 0.30 along x). With the
    // 0.7 rad z-rotation, that offset rotates into the world xy-plane, so the
    // child's world position must reflect BOTH the base shift and the rotation
    // (proves children compose against the moved base, not the cook pose).
    const Vec3 offset_local{0.30f, 0.0f, 0.0f};
    const Vec3 expected_child = kShift + kRot.Rotate(offset_local);
    std::printf("[diag] (3) child world pos=(%.4f,%.4f,%.4f) expected=(%.4f,%.4f,%.4f)\n",
                world[1].position.x, world[1].position.y, world[1].position.z,
                expected_child.x, expected_child.y, expected_child.z);
    EXPECT_NEAR(world[1].position.x, expected_child.x, 1.0e-4f);
    EXPECT_NEAR(world[1].position.y, expected_child.y, 1.0e-4f);
    EXPECT_NEAR(world[1].position.z, expected_child.z, 1.0e-4f);
}

// ---------------------------------------------------------------------------
// (4) DETERMINISM (D1): bit-identical across two runs and across replicas.
// ---------------------------------------------------------------------------
TEST(FloatingBaseAba, FloatingDynamicsDeterministic) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const float kGravity = -9.81f;  // signed g_z, production convention.
    const float kDt = 1.0e-3f;
    const uint32_t kSteps = 300u;

    auto run = [&](const articulation::ArticulationHostState& start) {
        auto host = start;
        auto device = articulation::UploadArticulationState(context, host);
        for (uint32_t s = 0u; s < kSteps; ++s) {
            auto view = device.View();
            StepFloating(context, view, kGravity, kDt);
        }
        articulation::DownloadArticulationState(device, &host);
        return host;
    };

    auto model = BuildSmallFloatingModel();
    model.host.tau = {0.0f, 0.8f, -0.5f};

    const auto a = run(model.host);
    const auto b = run(model.host);

    // Bit-identical base pose, velocities, and q across two runs.
    uint32_t mismatch = 0u;
    for (uint32_t i = 0u; i < a.base_pose.size(); ++i) {
        mismatch += (a.base_pose[i].position.x != b.base_pose[i].position.x);
        mismatch += (a.base_pose[i].position.y != b.base_pose[i].position.y);
        mismatch += (a.base_pose[i].position.z != b.base_pose[i].position.z);
        mismatch += (a.base_pose[i].rotation.w != b.base_pose[i].rotation.w);
    }
    for (uint32_t link = 0u; link < a.TotalLinkCount(); ++link) {
        for (uint32_t c = 0u; c < 6u; ++c) {
            mismatch += (a.link_velocity[link].v[c] != b.link_velocity[link].v[c]);
        }
    }
    for (uint32_t i = 0u; i < a.q.size(); ++i) {
        mismatch += (a.q[i] != b.q[i]);
    }
    std::printf("[diag] (4) two-run mismatches=%u\n", mismatch);
    EXPECT_EQ(mismatch, 0u) << "floating dynamics not bit-identical across runs";

    // Replicate to N envs; every replica's base pose must match env 0 bit-for-bit.
    constexpr uint32_t kEnvCount = 32u;
    auto batched = articulation::ReplicateArticulationHostState(model.host, kEnvCount);
    const auto batched_out = run(batched);
    ASSERT_EQ(batched_out.base_pose.size(), kEnvCount);
    uint32_t replica_mismatch = 0u;
    for (uint32_t env = 1u; env < kEnvCount; ++env) {
        replica_mismatch += (batched_out.base_pose[env].position.x !=
                             batched_out.base_pose[0].position.x);
        replica_mismatch += (batched_out.base_pose[env].position.z !=
                             batched_out.base_pose[0].position.z);
        replica_mismatch += (batched_out.base_pose[env].rotation.w !=
                             batched_out.base_pose[0].rotation.w);
    }
    const uint32_t links_per_env = model.host.TotalLinkCount();
    for (uint32_t env = 1u; env < kEnvCount; ++env) {
        for (uint32_t l = 0u; l < links_per_env; ++l) {
            for (uint32_t c = 0u; c < 6u; ++c) {
                replica_mismatch += (batched_out.link_velocity[env * links_per_env + l].v[c] !=
                                     batched_out.link_velocity[l].v[c]);
            }
        }
    }
    std::printf("[diag] (4) cross-replica mismatches (N=%u)=%u\n", kEnvCount,
                replica_mismatch);
    EXPECT_EQ(replica_mismatch, 0u) << "replicas diverged -- determinism broken";
}

// ---------------------------------------------------------------------------
// (5) REAL-PATH SMOKE: cook go2_float, confirm FloatingBase root, step, base moves.
// ---------------------------------------------------------------------------
TEST(FloatingBaseAba, Go2FloatCooksFloatingRootAndBaseMoves) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    auto host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);

    ASSERT_EQ(host.ArticulationCount(), 1u);
    ASSERT_GT(host.TotalLinkCount(), 1u);
    const uint32_t root = host.articulation_link_offset[0];
    ASSERT_EQ(host.parent_link[root], kInvalidLink);
    EXPECT_EQ(host.joint_type[root], articulation::ArticulationJointType::FloatingBase)
        << "go2_float root did not cook to FloatingBase";

    // Confirm the fixed-base oracle scene is UNCHANGED: go2_stand still cooks Fixed.
    {
        const auto stand_path = SourcePath("examples/scenes/go2_stand.usda");
        if (std::filesystem::exists(stand_path)) {
            const auto sscene = nuka::import::LoadUsd(stand_path.string());
            const auto sblob = nuka::scene::CookScene(sscene);
            const auto sworld = nuka::runtime::BuildWorld(sblob);
            auto shost = articulation::BuildArticulationHostState(
                sworld.template_view.articulations, sworld.template_view.body_table);
            const uint32_t sroot = shost.articulation_link_offset[0];
            EXPECT_EQ(shost.joint_type[sroot],
                      articulation::ArticulationJointType::Fixed)
                << "go2_stand root must stay Fixed (oracle scene must not change)";
        }
    }

    // Step the real batched path a few steps under gravity; confirm the base MOVES
    // (the "few steps finite" smoke would pass even with a frozen base, so we
    // assert real displacement, not just finiteness).
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    constexpr uint32_t kEnvCount = 4u;

    // Collect foot spheres (needed by the batched world ctor); ground is placed
    // far below so contacts never activate -- T8a only adds the FREE base path,
    // so the base should fall freely under gravity (T8b adds base contact).
    std::vector<articulation::FootShape> feet;
    {
        const auto& shapes = world.template_view.shape_table;
        for (uint32_t shape = 0u; shape < shapes.types.size(); ++shape) {
            if (shapes.types[shape] != nuka::scene::ShapeType::Sphere) {
                continue;
            }
            const uint32_t body = shapes.body_ids[shape];
            uint32_t calf_link = kInvalidLink;
            for (uint32_t link = 0u; link < host.TotalLinkCount(); ++link) {
                if (host.link_body[link] == body) {
                    calf_link = link;
                    break;
                }
            }
            if (calf_link == kInvalidLink) continue;
            articulation::FootShape foot;
            foot.calf_local_link = calf_link;
            foot.local_offset = shape < shapes.local_transforms.size()
                                    ? shapes.local_transforms[shape].position
                                    : Vec3::Zero();
            foot.radius = shape < shapes.radii.size() ? shapes.radii[shape] : 0.0f;
            feet.push_back(foot);
        }
    }
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);
    const float kGroundFarAway = -1000.0f;  // contacts never active.

    auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
    nuka::runtime::gpu::BatchedArticulatedWorld bw(context, batched, feet, max_dof,
                                                   kGroundFarAway);
    const Vec3 base0 = batched.base_pose[0].position;

    nuka::runtime::gpu::BatchedArticulatedStepParams params;
    params.dt = 1.0e-3f;
    params.gravity_z = -9.81f;  // signed g_z, matching the production step path.
    for (uint32_t s = 0u; s < 50u; ++s) {
        bw.Step(params);
    }
    context.stream.Synchronize();

    articulation::ArticulationHostState out = batched;
    bw.Download(&out);
    const Vec3 base_end = out.base_pose[0].position;
    const float moved = (base_end - base0).Length();
    std::printf("[diag] (5) go2_float base moved %.5f m in 50 steps (z: %.4f -> %.4f)\n",
                moved, base0.z, base_end.z);

    EXPECT_TRUE(std::isfinite(base_end.x) && std::isfinite(base_end.y) &&
                std::isfinite(base_end.z));
    EXPECT_GT(moved, 1.0e-4f) << "go2_float base did not move (floating root frozen)";
    // Under gravity with no ground contact yet (T8b), the trunk should fall (z<).
    EXPECT_LT(base_end.z, base0.z) << "base did not fall under gravity";
}
