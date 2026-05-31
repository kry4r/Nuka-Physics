// ---------------------------------------------------------------------------
// v0.5 C-fwd slice 1 -- Torque & Velocity control modes (engine forward)
// ---------------------------------------------------------------------------
//
// Validates the new stage-1 control-mode dispatch on BatchedArticulatedWorld:
//   * Torque mode    -- tau = clamp(torque_input). Zero torque + gravity-off =>
//                       NO motion; a nonzero torque on a single joint moves THAT
//                       joint, and the velocity sign matches the torque sign
//                       (qddot = M^-1 tau, and the actuated diagonal of M^-1 is
//                       positive, so qdot tracks the torque sign).
//   * Velocity mode  -- tau = Kp_v*(velocity_target - qdot). With a positive
//                       target and Kp_v>0 the joint velocity tracks TOWARD the
//                       target (rises from 0, sign matches the target).
//   * D1 two-run byte-exact -- for EACH mode, two worlds in that mode fed the
//                       same inputs and stepped N steps are bit-identical (q,qdot).
//   * Default-PD-unchanged -- a control_mode=PDPosition world is byte-identical
//                       to one constructed with the default ctor arg (the local
//                       proxy for the go2_stand golden gate).
//
// The worlds are driven DIRECTLY (no C ABI) by mirroring the proven CookGo2 setup
// in test_batched_articulated_world.cpp. Gravity is turned OFF and the ground is
// placed far away so the ONLY joint forcing is the control law under test, which
// isolates the new modes' behavior. Note the implicit joint damping (#43, driven
// by drive_damping) still runs for every mode -- the torque/velocity directional
// checks set drive_damping = 0 so it does not muddy the sign of the response.
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/control_mode.hpp"
#include "runtime/gpu/batched_articulated_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace gpu = nuka::runtime::gpu;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// Cooked single-env Go2 host state + base-relative foot shapes (mirrors the
// CookGo2 helper in test_batched_articulated_world.cpp -- we do NOT need the
// hold-drive gains here, the control modes supply their own inputs).
struct CookedGo2 {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;
};

CookedGo2 CookGo2() {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);

    CookedGo2 result;
    result.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);

    const auto& shapes = world.template_view.shape_table;
    const uint32_t link_count = result.host.TotalLinkCount();
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
    return result;
}

// Index of the first ACTUATED (non-fixed) joint link. Slot 0 is the root; the
// actuated leg joints follow. Used to inject a single-joint control input.
uint32_t FirstActuatedLink(const articulation::ArticulationHostState& host) {
    const uint32_t link_count = host.TotalLinkCount();
    for (uint32_t link = 0u; link < link_count; ++link) {
        if (host.joint_type[link] != articulation::ArticulationJointType::Fixed) {
            return link;
        }
    }
    return kInvalidLink;
}

// Upload a host float vector to a device buffer.
nuka::phi::Buffer UploadFloats(const std::vector<float>& v) {
    nuka::phi::Buffer b(v.size() * sizeof(float), nuka::phi::MemoryKind::Device);
    if (!v.empty()) {
        b.CopyFromHost(v.data(), v.size() * sizeof(float));
    }
    return b;
}

bool AllFinite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

// A common single-env, gravity-off, ground-far-away setup: the only joint
// forcing is the control law under test.
constexpr float kGravityOff = 0.0f;
constexpr float kDt = 1.0f / 240.0f;
constexpr float kGroundFarAway = -1000.0f;

}  // namespace

// ---------------------------------------------------------------------------
// Torque mode: zero torque + gravity-off => no motion.
// ---------------------------------------------------------------------------
TEST(ControlModes, TorqueZeroInputGravityOffNoMotion) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    auto& base = cooked.host;
    const uint32_t max_dof = articulation::ArticulationDofCount(base, 0u);
    ASSERT_GT(max_dof, 0u);
    const uint32_t base_link_count = base.TotalLinkCount();

    gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                    kGroundFarAway,
                                    gpu::DeterminismLevel::Strong,
                                    articulation::ControlMode::Torque);
    EXPECT_EQ(bw.ControlMode(), articulation::ControlMode::Torque);

    // torque_input is the world's own buffer, zero-initialized by the ctor; do
    // NOT write it -> all-zero torque. No force limits (null) => no clamp needed.
    gpu::BatchedArticulatedStepParams params;
    params.torque_input =
        static_cast<const float*>(bw.TorqueInputBuffer().Data());
    params.gravity_z = kGravityOff;
    params.dt = kDt;

    articulation::ArticulationHostState start = base;
    for (uint32_t step = 0u; step < 30u; ++step) {
        bw.Step(params);
    }
    context.stream.Synchronize();
    articulation::ArticulationHostState dl = base;
    bw.Download(&dl);
    ASSERT_TRUE(AllFinite(dl.q));
    ASSERT_TRUE(AllFinite(dl.qdot));
    // No torque + no gravity => the state is unchanged from the cooked start.
    for (uint32_t l = 0u; l < base_link_count; ++l) {
        EXPECT_EQ(dl.q[l], start.q[l]) << "q moved with zero torque at link " << l;
        EXPECT_EQ(dl.qdot[l], 0.0f) << "qdot nonzero with zero torque at link " << l;
    }
}

// ---------------------------------------------------------------------------
// Torque mode: a nonzero torque on one joint moves it the expected way (qdot
// sign matches torque sign). qddot = M^-1 tau and the actuated diagonal of
// M^-1 is positive, so a positive torque drives a positive joint velocity.
// ---------------------------------------------------------------------------
TEST(ControlModes, TorqueNonzeroDrivesJointInTorqueDirection) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    auto& base = cooked.host;
    const uint32_t max_dof = articulation::ArticulationDofCount(base, 0u);
    ASSERT_GT(max_dof, 0u);
    const uint32_t base_link_count = base.TotalLinkCount();
    const uint32_t joint = FirstActuatedLink(base);
    ASSERT_NE(joint, kInvalidLink);

    for (const float sign : {+1.0f, -1.0f}) {
        gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                        kGroundFarAway,
                                        gpu::DeterminismLevel::Strong,
                                        articulation::ControlMode::Torque);
        // Inject a constant torque on the single actuated joint via an external
        // device buffer (the step param is just a pointer; the world reads
        // whatever buffer it is given). No force limit (null) so the directional
        // response is unclamped.
        std::vector<float> torque(base_link_count, 0.0f);
        const float kTorque = sign * 5.0f;  // Nm
        torque[joint] = kTorque;
        nuka::phi::Buffer torque_dev = UploadFloats(torque);

        gpu::BatchedArticulatedStepParams params;
        params.torque_input = static_cast<const float*>(torque_dev.Data());
        params.gravity_z = kGravityOff;
        params.dt = kDt;

        for (uint32_t step = 0u; step < 20u; ++step) {
            bw.Step(params);
        }
        context.stream.Synchronize();
        articulation::ArticulationHostState dl = base;
        bw.Download(&dl);
        ASSERT_TRUE(AllFinite(dl.qdot));
        // The driven joint velocity must be nonzero and share the torque sign.
        EXPECT_GT(dl.qdot[joint] * sign, 0.0f)
            << "joint qdot sign mismatch for torque sign " << sign
            << " (qdot=" << dl.qdot[joint] << ")";
        EXPECT_NE(dl.qdot[joint], 0.0f);
        // q must have advanced in the same direction.
        EXPECT_GT((dl.q[joint] - base.q[joint]) * sign, 0.0f)
            << "joint q did not advance in the torque direction for sign " << sign;
    }
}

// ---------------------------------------------------------------------------
// Velocity mode: with a positive velocity_target and Kp_v>0 (drive_stiffness),
// the joint velocity tracks TOWARD the target -- from rest it takes the target's
// sign, never overshoots (so the explicit servo is stable), and converges to an
// appreciable fraction of the target. drive_damping is left at zero so the
// implicit joint damping does not skew the sign of the tracking response.
//
// The law tau = Kp_v*(velocity_target - qdot) integrated EXPLICITLY is only
// conditionally stable (the per-step map is qdot_{n+1} = qdot_n*(1 -
// dt*Kp_v/m_eff) + dt*(Kp_v/m_eff)*target, which diverges for dt*Kp_v/m_eff > 2).
// Kp_v == 1.0 keeps the loop gain ~0.19 (decay ~0.81) for this scene -- a clean
// first-order servo that converges over 40 steps with no oscillation. A larger
// Kp_v overshoots and diverges (that is physics, not a kernel defect).
// ---------------------------------------------------------------------------
TEST(ControlModes, VelocityTargetTracksTowardTarget) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    auto& base = cooked.host;
    const uint32_t max_dof = articulation::ArticulationDofCount(base, 0u);
    ASSERT_GT(max_dof, 0u);
    const uint32_t base_link_count = base.TotalLinkCount();
    const uint32_t joint = FirstActuatedLink(base);
    ASSERT_NE(joint, kInvalidLink);

    for (const float target_vel : {+2.0f, -2.0f}) {
        gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                        kGroundFarAway,
                                        gpu::DeterminismLevel::Strong,
                                        articulation::ControlMode::Velocity);
        EXPECT_EQ(bw.ControlMode(), articulation::ControlMode::Velocity);

        // Velocity gain Kp_v reuses the drive_stiffness buffer. Zero drive_damping
        // so the orthogonal implicit joint damping is inert for this directional
        // probe. No force limit (null) -> unclamped.
        std::vector<float> stiffness(base_link_count, 0.0f);
        std::vector<float> damping(base_link_count, 0.0f);
        std::vector<float> vel_target(base_link_count, 0.0f);
        stiffness[joint] = 1.0f;      // Kp_v -- stable explicit servo gain.
        vel_target[joint] = target_vel;
        nuka::phi::Buffer stiff_dev = UploadFloats(stiffness);
        nuka::phi::Buffer damp_dev = UploadFloats(damping);
        nuka::phi::Buffer vel_dev = UploadFloats(vel_target);

        gpu::BatchedArticulatedStepParams params;
        params.velocity_target = static_cast<const float*>(vel_dev.Data());
        params.drive_stiffness = static_cast<const float*>(stiff_dev.Data());
        params.drive_damping = static_cast<const float*>(damp_dev.Data());
        params.gravity_z = kGravityOff;
        params.dt = kDt;

        for (uint32_t step = 0u; step < 40u; ++step) {
            bw.Step(params);
            context.stream.Synchronize();
            articulation::ArticulationHostState dl = base;
            bw.Download(&dl);
            ASSERT_TRUE(AllFinite(dl.qdot)) << "qdot non-finite at step " << step;
            // After the first step the driven joint already has the target sign
            // (from rest tau = Kp_v*target). And a STABLE servo never overshoots:
            // |qdot| stays below the target magnitude every step (a small tol for
            // coupling reaction torques from the other links).
            if (step >= 1u) {
                EXPECT_GT(dl.qdot[joint] * target_vel, 0.0f)
                    << "qdot sign mismatch at step " << step
                    << " (target=" << target_vel << ", qdot=" << dl.qdot[joint]
                    << ")";
            }
            EXPECT_LE(std::fabs(dl.qdot[joint]), std::fabs(target_vel) + 1e-2f)
                << "qdot overshot target at step " << step
                << " (qdot=" << dl.qdot[joint] << ")";
        }
        // After many steps the joint velocity has the target's sign and has
        // converged to an appreciable fraction of it (a one-pole servo at this
        // gain reaches well over half the target in 40 steps).
        articulation::ArticulationHostState dl = base;
        bw.Download(&dl);
        EXPECT_GT(dl.qdot[joint] * target_vel, 0.0f)
            << "joint velocity sign mismatch (target=" << target_vel
            << ", qdot=" << dl.qdot[joint] << ")";
        EXPECT_GT(std::fabs(dl.qdot[joint]), 0.5f * std::fabs(target_vel))
            << "joint velocity did not converge to half the target (qdot="
            << dl.qdot[joint] << ", target=" << target_vel << ")";
    }
}

// ---------------------------------------------------------------------------
// D1 two-run byte-exact for EACH new mode: two worlds in the same mode, same
// inputs, N steps -> bit-identical q/qdot.
// ---------------------------------------------------------------------------
namespace {

// Runs a single-env world in `mode` for `steps`, with the supplied per-link
// control input written into the matching control buffer, and returns the final
// (q, qdot). Used to prove run-to-run determinism (call twice, EXPECT_EQ).
std::pair<std::vector<float>, std::vector<float>> RunMode(
    const nuka::phi::DeviceContext& context, const CookedGo2& cooked,
    articulation::ControlMode mode, uint32_t joint, uint32_t steps) {
    auto base = cooked.host;
    const uint32_t max_dof = articulation::ArticulationDofCount(base, 0u);
    const uint32_t base_link_count = base.TotalLinkCount();

    gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                    kGroundFarAway,
                                    gpu::DeterminismLevel::Strong, mode);
    gpu::BatchedArticulatedStepParams params;
    params.gravity_z = kGravityOff;
    params.dt = kDt;

    std::vector<float> stiffness(base_link_count, 0.0f);
    std::vector<float> damping(base_link_count, 0.0f);
    // Buffers must outlive the stepping loop -> declared in this scope.
    nuka::phi::Buffer torque_dev;
    nuka::phi::Buffer vel_dev;
    nuka::phi::Buffer stiff_dev;
    nuka::phi::Buffer damp_dev;

    if (mode == articulation::ControlMode::Torque) {
        std::vector<float> torque(base_link_count, 0.0f);
        torque[joint] = 3.0f;
        torque_dev = UploadFloats(torque);
        params.torque_input = static_cast<const float*>(torque_dev.Data());
    } else if (mode == articulation::ControlMode::Velocity) {
        std::vector<float> vel_target(base_link_count, 0.0f);
        stiffness[joint] = 1.0f;  // stable explicit velocity-servo gain.
        vel_target[joint] = 1.5f;
        stiff_dev = UploadFloats(stiffness);
        damp_dev = UploadFloats(damping);
        vel_dev = UploadFloats(vel_target);
        params.velocity_target = static_cast<const float*>(vel_dev.Data());
        params.drive_stiffness = static_cast<const float*>(stiff_dev.Data());
        params.drive_damping = static_cast<const float*>(damp_dev.Data());
    }

    for (uint32_t step = 0u; step < steps; ++step) {
        bw.Step(params);
    }
    context.stream.Synchronize();
    articulation::ArticulationHostState dl = base;
    bw.Download(&dl);
    return {dl.q, dl.qdot};
}

}  // namespace

TEST(ControlModes, TorqueModeTwoRunByteExact) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    const uint32_t joint = FirstActuatedLink(cooked.host);
    ASSERT_NE(joint, kInvalidLink);
    const auto a = RunMode(context, cooked, articulation::ControlMode::Torque, joint, 50u);
    const auto b = RunMode(context, cooked, articulation::ControlMode::Torque, joint, 50u);
    ASSERT_EQ(a.first.size(), b.first.size());
    for (size_t i = 0u; i < a.first.size(); ++i) {
        EXPECT_EQ(a.first[i], b.first[i]) << "q differs run-to-run at " << i;
        EXPECT_EQ(a.second[i], b.second[i]) << "qdot differs run-to-run at " << i;
    }
}

TEST(ControlModes, VelocityModeTwoRunByteExact) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    const uint32_t joint = FirstActuatedLink(cooked.host);
    ASSERT_NE(joint, kInvalidLink);
    const auto a = RunMode(context, cooked, articulation::ControlMode::Velocity, joint, 50u);
    const auto b = RunMode(context, cooked, articulation::ControlMode::Velocity, joint, 50u);
    ASSERT_EQ(a.first.size(), b.first.size());
    for (size_t i = 0u; i < a.first.size(); ++i) {
        EXPECT_EQ(a.first[i], b.first[i]) << "q differs run-to-run at " << i;
        EXPECT_EQ(a.second[i], b.second[i]) << "qdot differs run-to-run at " << i;
    }
}

// ---------------------------------------------------------------------------
// Default-PD-unchanged: a world built with control_mode=PDPosition is byte-
// identical to one built with the DEFAULT ctor arg (no mode specified). Local
// proxy for the go2_stand golden gate.
// ---------------------------------------------------------------------------
TEST(ControlModes, DefaultPdModeMatchesUnspecified) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    auto& base = cooked.host;
    const uint32_t max_dof = articulation::ArticulationDofCount(base, 0u);
    const uint32_t base_link_count = base.TotalLinkCount();
    const float kGravityZ = -9.81f;
    const uint32_t kSteps = 60u;

    // Scene-derived hold drives (mirror the C-ABI BuildHoldDriveTargets used by
    // the golden). Pull per-actuator gain + force limit; damping = 2*sqrt(gain),
    // target = the cooked stance q.
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    std::vector<float> targets = base.q;
    std::vector<float> stiffness(base_link_count, 0.0f);
    std::vector<float> damping(base_link_count, 0.0f);
    std::vector<float> limits(base_link_count, 0.0f);
    const auto& actuators = world.template_view.actuator_table;
    const auto& joints = world.template_view.joint_table;
    for (uint32_t a = 0u; a < world.template_view.actuator_count; ++a) {
        if (a >= actuators.joint_ids.size() || a >= actuators.types.size() ||
            actuators.types[a] != nuka::scene::ActuatorType::Position) {
            continue;
        }
        const auto j = actuators.joint_ids[a];
        if (j >= joints.child_bodies.size()) {
            continue;
        }
        const auto child_body = joints.child_bodies[j];
        uint32_t link = kInvalidLink;
        for (uint32_t l = 0u; l < base_link_count; ++l) {
            if (base.link_body[l] == child_body) {
                link = l;
                break;
            }
        }
        if (link == kInvalidLink ||
            base.joint_type[link] == articulation::ArticulationJointType::Fixed) {
            continue;
        }
        const float gain =
            a < actuators.gains.size() ? std::fmax(actuators.gains[a], 0.0f) : 0.0f;
        const float force_limit = a < actuators.force_limits.size()
                                      ? std::fmax(actuators.force_limits[a], 0.0f)
                                      : 0.0f;
        if (gain > 0.0f) {
            stiffness[link] = gain;
            damping[link] = 2.0f * std::sqrt(gain);
        }
        if (force_limit > 0.0f) {
            limits[link] = force_limit;
        }
    }

    auto run = [&](articulation::ControlMode mode, bool specify)
        -> std::pair<std::vector<float>, std::vector<float>> {
        auto host = cooked.host;
        gpu::BatchedArticulatedWorld bw =
            specify ? gpu::BatchedArticulatedWorld(context, host, cooked.feet,
                                                   max_dof, kGroundFarAway,
                                                   gpu::DeterminismLevel::Strong,
                                                   mode)
                    : gpu::BatchedArticulatedWorld(context, host, cooked.feet,
                                                   max_dof, kGroundFarAway,
                                                   gpu::DeterminismLevel::Strong);
        nuka::phi::Buffer t = UploadFloats(targets);
        nuka::phi::Buffer s = UploadFloats(stiffness);
        nuka::phi::Buffer d = UploadFloats(damping);
        nuka::phi::Buffer fl = UploadFloats(limits);
        gpu::BatchedArticulatedStepParams params;
        params.drive_targets = static_cast<const float*>(t.Data());
        params.drive_stiffness = static_cast<const float*>(s.Data());
        params.drive_damping = static_cast<const float*>(d.Data());
        params.drive_force_limits = static_cast<const float*>(fl.Data());
        params.gravity_z = kGravityZ;
        params.dt = kDt;
        for (uint32_t step = 0u; step < kSteps; ++step) {
            bw.Step(params);
        }
        context.stream.Synchronize();
        articulation::ArticulationHostState dl = host;
        bw.Download(&dl);
        return {dl.q, dl.qdot};
    };

    const auto specified =
        run(articulation::ControlMode::PDPosition, /*specify=*/true);
    const auto unspecified =
        run(articulation::ControlMode::PDPosition, /*specify=*/false);
    ASSERT_EQ(specified.first.size(), unspecified.first.size());
    for (size_t i = 0u; i < specified.first.size(); ++i) {
        EXPECT_EQ(specified.first[i], unspecified.first[i])
            << "PD q differs (specified vs default) at " << i;
        EXPECT_EQ(specified.second[i], unspecified.second[i])
            << "PD qdot differs (specified vs default) at " << i;
    }
}

// ---------------------------------------------------------------------------
// A reserved/unimplemented control mode is rejected at construction (never
// silently mis-actuated).
// ---------------------------------------------------------------------------
TEST(ControlModes, ReservedModeRejectedAtConstruction) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    const uint32_t max_dof = articulation::ArticulationDofCount(cooked.host, 0u);
    EXPECT_THROW(
        gpu::BatchedArticulatedWorld(context, cooked.host, cooked.feet, max_dof,
                                     kGroundFarAway,
                                     gpu::DeterminismLevel::Strong,
                                     articulation::ControlMode::ComputedTorque),
        std::invalid_argument);
}
