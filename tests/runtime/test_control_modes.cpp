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
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/control_mode.hpp"
#include "runtime/gpu/batched_articulated_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

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
// An out-of-range control mode is rejected at construction (never silently mis-
// actuated). All six enumerators {0..5} (PDPosition..Actuator) are now
// IMPLEMENTED -- Osc (4) was the last hole and is filled by p03 R2 -- so only a
// value ABOVE Actuator (here a synthetic id 6) must be rejected.
// ---------------------------------------------------------------------------
TEST(ControlModes, OutOfRangeModeRejectedAtConstruction) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    const uint32_t max_dof = articulation::ArticulationDofCount(cooked.host, 0u);
    // id 6 is above Actuator (5) -- no such control law exists, so it is rejected.
    EXPECT_THROW(
        gpu::BatchedArticulatedWorld(context, cooked.host, cooked.feet, max_dof,
                                     kGroundFarAway,
                                     gpu::DeterminismLevel::Strong,
                                     static_cast<articulation::ControlMode>(6u)),
        std::invalid_argument);
}

// ===========================================================================
// Slice 2: ComputedTorque (inverse-dynamics) + Actuator (DC-motor) control.
// ===========================================================================
namespace {

// Reads the device qddot (one float per link) for the single-env world.
std::vector<float> DownloadQddot(const nuka::phi::DeviceContext& context,
                                 gpu::BatchedArticulatedWorld& bw,
                                 uint32_t link_count) {
    const articulation::ArticulationDeviceState state = bw.View();
    std::vector<float> qddot(link_count, 0.0f);
    context.stream.Synchronize();
    cudaMemcpy(qddot.data(), state.qddot,
               static_cast<size_t>(link_count) * sizeof(float),
               cudaMemcpyDeviceToHost);
    return qddot;
}

// Scene-derived hold gains + force limits (mirrors DefaultPdModeMatchesUnspecified
// / the c_abi BuildHoldDriveTargets): per actuated link, Kp = scene gain, the
// force limit = scene force limit. Used to drive ComputedTorque (Kp/Kd reuse the
// stiffness/damping buffers; the q_target is the cooked stance).
struct SceneGains {
    std::vector<float> stiffness;
    std::vector<float> damping;
    std::vector<float> limits;
};

SceneGains DeriveSceneGains(const articulation::ArticulationHostState& base) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    const uint32_t base_link_count = base.TotalLinkCount();
    SceneGains g;
    g.stiffness.assign(base_link_count, 0.0f);
    g.damping.assign(base_link_count, 0.0f);
    g.limits.assign(base_link_count, 0.0f);
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
            g.stiffness[link] = gain;
            g.damping[link] = 2.0f * std::sqrt(gain);
        }
        if (force_limit > 0.0f) {
            g.limits[link] = force_limit;
        }
    }
    return g;
}

constexpr float kGravityOn = -9.81f;
// Penetrates the cooked stance feet (~0.031 m) so the base is SUPPORTED -- the
// ground reaction lets gravity-comp HOLD vs zero-torque BUCKLE. Mirrors the
// ContactsOnScaleStabilityDeterminismTiming ground in test_batched_articulated.
constexpr float kGroundStand = 0.31f;

}  // namespace

// ---------------------------------------------------------------------------
// ComputedTorque gravity compensation: with Kp=Kd=0 (pure bias compensation),
// the realized joint qddot is ~0 (tau == bias cancels gravity/Coriolis), so the
// (fixed-trunk) robot HOLDS its joint angles, while at zero torque the legs sag
// under gravity. Ground is FAR away (no contacts) so the ONLY joint forcing is
// gravity vs its compensation -- a clean discriminator (the trunk is fixed-base
// for this scene, so there is no base free-fall to mask the difference). Two
// assertions:
//   (1) qddot ~ a_des == 0 right after a step (the defining ID correctness check);
//   (2) joint drift over N steps << the zero-torque (Torque mode, zero input)
//       baseline drift.
// ---------------------------------------------------------------------------
TEST(ControlModes, ComputedTorqueGravityCompensationHolds) {
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
    const uint32_t kSteps = 60u;

    // q_target = the cooked stance; Kp = Kd = 0 (pure gravity/bias comp).
    std::vector<float> targets = base.q;
    std::vector<float> zeros(base_link_count, 0.0f);
    nuka::phi::Buffer tgt_dev = UploadFloats(targets);
    nuka::phi::Buffer kp_dev = UploadFloats(zeros);   // Kp = 0.
    nuka::phi::Buffer kd_dev = UploadFloats(zeros);   // Kd = 0.

    // (1) qddot ~ a_des == 0 after one step (the robust ID correctness check).
    // NO force limits so the clamp cannot leave residue -> the joint qddot must be
    // FP-close to a_des == 0 (CRBA-M vs ABA-implicit-M differ only in last bits).
    {
        gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                        kGroundFarAway,
                                        gpu::DeterminismLevel::Strong,
                                        articulation::ControlMode::ComputedTorque);
        gpu::BatchedArticulatedStepParams params;
        params.drive_targets = static_cast<const float*>(tgt_dev.Data());
        params.drive_stiffness = static_cast<const float*>(kp_dev.Data());
        params.drive_damping = static_cast<const float*>(kd_dev.Data());
        // drive_force_limits intentionally null -> unclamped exact comp torque.
        params.gravity_z = kGravityOn;
        params.dt = kDt;
        bw.Step(params);
        const auto qddot = DownloadQddot(context, bw, base_link_count);
        ASSERT_TRUE(AllFinite(qddot));
        for (uint32_t l = 0u; l < base_link_count; ++l) {
            if (base.joint_type[l] == articulation::ArticulationJointType::Fixed) {
                continue;
            }
            EXPECT_LT(std::fabs(qddot[l]), 1.0e-2f)
                << "ComputedTorque (Kp=Kd=0) joint qddot not ~0 at link " << l
                << " (qddot=" << qddot[l] << ")";
        }
    }

    // (2) Joint drift vs the zero-torque baseline over N steps (ground far away,
    // so the only forcing is gravity vs its compensation; no contacts).
    auto stance_drift = [&](articulation::ControlMode mode,
                            bool comp) -> float {
        gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                        kGroundFarAway,
                                        gpu::DeterminismLevel::Strong, mode);
        gpu::BatchedArticulatedStepParams params;
        params.gravity_z = kGravityOn;
        params.dt = kDt;
        if (comp) {
            params.drive_targets = static_cast<const float*>(tgt_dev.Data());
            params.drive_stiffness = static_cast<const float*>(kp_dev.Data());
            params.drive_damping = static_cast<const float*>(kd_dev.Data());
            // No force limits -> the exact comp torque is applied.
        } else {
            // Torque mode, zero input == zero torque baseline (legs sag).
            params.torque_input =
                static_cast<const float*>(bw.TorqueInputBuffer().Data());
        }
        for (uint32_t step = 0u; step < kSteps; ++step) {
            bw.Step(params);
        }
        context.stream.Synchronize();
        articulation::ArticulationHostState dl = base;
        bw.Download(&dl);
        EXPECT_TRUE(AllFinite(dl.q));
        float drift = 0.0f;
        for (uint32_t l = 0u; l < base_link_count; ++l) {
            if (base.joint_type[l] == articulation::ArticulationJointType::Fixed) {
                continue;
            }
            drift += std::fabs(dl.q[l] - base.q[l]);
        }
        return drift;
    };

    const float comp_drift =
        stance_drift(articulation::ControlMode::ComputedTorque, /*comp=*/true);
    const float zero_drift =
        stance_drift(articulation::ControlMode::Torque, /*comp=*/false);
    // Gravity comp holds the joints MUCH better than letting the legs sag.
    EXPECT_GT(zero_drift, 0.05f) << "zero-torque baseline did not sag";
    EXPECT_LT(comp_drift, 0.1f * zero_drift)
        << "ComputedTorque gravity comp did not hold better than zero torque "
        << "(comp_drift=" << comp_drift << ", zero_drift=" << zero_drift << ")";
}

// ---------------------------------------------------------------------------
// ComputedTorque tracking: with Kp>0, Kd=0 (drive_damping null -> NO control Kd
// AND no orthogonal implicit joint damping) and gravity off, the joint is a pure
// undamped oscillator qddot = Kp*e about q_target. The position error therefore
// (a) DECREASES early (the joint accelerates toward the target -- the task's
// "error decreases / tracks target" requirement) and (b) reaches ~0 at the
// quarter period (~38 steps for Kp=100, dt=1/240). We assert the MIN error over
// the run is a small fraction of the initial -- robust against the post-quarter-
// period overshoot of an undamped law. The defining ID property qddot == a_des =
// Kp*e is checked exactly in ComputedTorqueGravityCompensationHolds.
// ---------------------------------------------------------------------------
TEST(ControlModes, ComputedTorqueTracksTarget) {
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

    // Target = stance + an offset on ONE joint. Kp moderate; Kd / drive_damping
    // LEFT NULL so the law is the undamped oscillator qddot = Kp*e (drive_damping
    // would double as the implicit joint damping that crushes qdot every step).
    std::vector<float> targets = base.q;
    const float kOffset = 0.3f;  // rad.
    targets[joint] = base.q[joint] + kOffset;
    std::vector<float> kp(base_link_count, 0.0f);
    kp[joint] = 100.0f;          // rad/s^2 per rad of error.
    nuka::phi::Buffer tgt_dev = UploadFloats(targets);
    nuka::phi::Buffer kp_dev = UploadFloats(kp);

    gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                    kGroundFarAway,
                                    gpu::DeterminismLevel::Strong,
                                    articulation::ControlMode::ComputedTorque);
    gpu::BatchedArticulatedStepParams params;
    params.drive_targets = static_cast<const float*>(tgt_dev.Data());
    params.drive_stiffness = static_cast<const float*>(kp_dev.Data());
    // params.drive_damping intentionally null (no control Kd, no implicit damping).
    params.gravity_z = kGravityOff;
    params.dt = kDt;

    const float err0 = std::fabs(base.q[joint] - targets[joint]);
    float min_err = err0;
    const uint32_t kSteps = 50u;
    const uint32_t kEarly = 15u;  // before the quarter-period overshoot.
    for (uint32_t step = 0u; step < kSteps; ++step) {
        bw.Step(params);
        context.stream.Synchronize();
        articulation::ArticulationHostState dl = base;
        bw.Download(&dl);
        ASSERT_TRUE(AllFinite(dl.q)) << "q non-finite at step " << step;
        const float err = std::fabs(dl.q[joint] - targets[joint]);
        min_err = std::fmin(min_err, err);
        if (step < kEarly) {
            // Early phase: the joint moves monotonically TOWARD the target.
            EXPECT_GT((dl.q[joint] - base.q[joint]) * kOffset, -1e-3f)
                << "joint moved AWAY from target at early step " << step;
            EXPECT_LT(err, err0 + 1e-3f)
                << "error grew above the initial in the early phase at step "
                << step;
        }
    }
    // The undamped oscillator reaches the target (min error ~0) at the quarter
    // period within the run -- the joint demonstrably tracked q_target.
    EXPECT_LT(min_err, 0.15f * err0)
        << "ComputedTorque did not track the target (err0=" << err0
        << ", min_err=" << min_err << ")";
}

// ---------------------------------------------------------------------------
// Actuator envelope: the deliverable torque is clamped by the DC-motor
// torque-speed envelope tau_max(qdot) = clamp(tau_stall*(1-|qdot|/qdot_noload),
// 0, tau_stall). Probe by commanding a large torque and reading qddot:
//   * at qdot=0 (first step from rest) the FULL tau_stall is delivered;
//   * at qdot > qdot_noload the deliverable torque (and so qddot from that joint)
//     is clamped to ~0 -- the envelope ENGAGES. We compare the first-step joint
//     acceleration (full torque) to a setup whose no-load speed is tiny so the
//     envelope clamps from the very first non-rest step.
// ---------------------------------------------------------------------------
TEST(ControlModes, ActuatorEnvelopeLimitsTorqueAtSpeed) {
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

    const float kStall = 10.0f;     // tau_stall (Nm), via drive_force_limits.
    const float kCmd = 100.0f;      // command FAR above stall so the clamp binds.

    std::vector<float> torque(base_link_count, 0.0f);
    torque[joint] = kCmd;
    std::vector<float> limits(base_link_count, 0.0f);
    limits[joint] = kStall;
    nuka::phi::Buffer torque_dev = UploadFloats(torque);
    nuka::phi::Buffer lim_dev = UploadFloats(limits);

    // Run A: no-load speed LARGE (envelope effectively off at the speeds reached)
    // => full tau_stall every step => qdot keeps rising.
    // Run B: no-load speed TINY => after the first step |qdot|>qdot_noload so
    // tau_max collapses to ~0 => qdot stalls near its first-step value.
    auto run = [&](float qdot_noload, uint32_t steps) -> std::vector<float> {
        gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                        kGroundFarAway,
                                        gpu::DeterminismLevel::Strong,
                                        articulation::ControlMode::Actuator);
        std::vector<float> noload(base_link_count, 0.0f);
        noload[joint] = qdot_noload;
        nuka::phi::Buffer noload_dev = UploadFloats(noload);
        gpu::BatchedArticulatedStepParams params;
        params.torque_input = static_cast<const float*>(torque_dev.Data());
        params.drive_force_limits = static_cast<const float*>(lim_dev.Data());
        params.actuator_noload_speed =
            static_cast<const float*>(noload_dev.Data());
        params.gravity_z = kGravityOff;
        params.dt = kDt;
        std::vector<float> qdot_hist;
        for (uint32_t step = 0u; step < steps; ++step) {
            bw.Step(params);
            context.stream.Synchronize();
            articulation::ArticulationHostState dl = base;
            bw.Download(&dl);
            qdot_hist.push_back(dl.qdot[joint]);
        }
        return qdot_hist;
    };

    const uint32_t kSteps = 40u;
    const auto fast = run(1000.0f, kSteps);   // envelope effectively off.
    const auto slow = run(0.05f, kSteps);     // envelope clamps after step 0.
    ASSERT_EQ(fast.size(), kSteps);
    ASSERT_EQ(slow.size(), kSteps);

    // First step from rest (qdot=0): both deliver the SAME (full tau_stall) torque,
    // so the first-step qdot is identical (envelope inactive at qdot=0).
    EXPECT_NEAR(fast[0], slow[0], 1e-4f * std::fabs(fast[0]) + 1e-6f)
        << "first-step (qdot=0) torque differed -- envelope must be inactive at "
           "standstill (fast=" << fast[0] << ", slow=" << slow[0] << ")";
    EXPECT_NE(fast[0], 0.0f);

    // By the final step the LARGE-no-load run has accelerated much further (full
    // torque every step) than the TINY-no-load run (whose torque collapsed to ~0
    // once |qdot| passed qdot_noload). The envelope demonstrably limited delivery.
    EXPECT_GT(std::fabs(fast.back()), 2.0f * std::fabs(slow.back()))
        << "DC-motor envelope did not limit torque at speed (fast qdot="
        << fast.back() << ", slow qdot=" << slow.back() << ")";
    // And the clamped run's velocity stays bounded near the no-load speed region
    // (it cannot keep accelerating once the envelope shuts the torque off).
    EXPECT_LT(std::fabs(slow.back()), std::fabs(fast.back()));
}

// ---------------------------------------------------------------------------
// D1 two-run byte-exact for the slice-2 modes: same mode, same inputs, N steps
// -> bit-identical q/qdot (mirrors the slice-1 mode-determinism tests).
// ---------------------------------------------------------------------------
namespace {

std::pair<std::vector<float>, std::vector<float>> RunSlice2Mode(
    const nuka::phi::DeviceContext& context, const CookedGo2& cooked,
    articulation::ControlMode mode, uint32_t joint, uint32_t steps) {
    auto base = cooked.host;
    const uint32_t max_dof = articulation::ArticulationDofCount(base, 0u);
    const uint32_t base_link_count = base.TotalLinkCount();

    gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                    kGroundStand,
                                    gpu::DeterminismLevel::Strong, mode);
    gpu::BatchedArticulatedStepParams params;
    params.gravity_z = kGravityOn;
    params.dt = kDt;

    // Buffers must outlive the loop.
    nuka::phi::Buffer tgt_dev, kp_dev, kd_dev, lim_dev, torque_dev, noload_dev;
    if (mode == articulation::ControlMode::ComputedTorque) {
        const SceneGains g = DeriveSceneGains(base);
        std::vector<float> targets = base.q;
        tgt_dev = UploadFloats(targets);
        kp_dev = UploadFloats(g.stiffness);
        kd_dev = UploadFloats(g.damping);
        lim_dev = UploadFloats(g.limits);
        params.drive_targets = static_cast<const float*>(tgt_dev.Data());
        params.drive_stiffness = static_cast<const float*>(kp_dev.Data());
        params.drive_damping = static_cast<const float*>(kd_dev.Data());
        params.drive_force_limits = static_cast<const float*>(lim_dev.Data());
    } else {  // Actuator.
        std::vector<float> torque(base_link_count, 0.0f);
        std::vector<float> limits(base_link_count, 0.0f);
        std::vector<float> noload(base_link_count, 0.0f);
        torque[joint] = 8.0f;
        limits[joint] = 10.0f;
        noload[joint] = 2.0f;
        torque_dev = UploadFloats(torque);
        lim_dev = UploadFloats(limits);
        noload_dev = UploadFloats(noload);
        params.torque_input = static_cast<const float*>(torque_dev.Data());
        params.drive_force_limits = static_cast<const float*>(lim_dev.Data());
        params.actuator_noload_speed =
            static_cast<const float*>(noload_dev.Data());
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

TEST(ControlModes, ComputedTorqueModeTwoRunByteExact) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    const uint32_t joint = FirstActuatedLink(cooked.host);
    ASSERT_NE(joint, kInvalidLink);
    const auto a = RunSlice2Mode(context, cooked,
                                 articulation::ControlMode::ComputedTorque, joint, 50u);
    const auto b = RunSlice2Mode(context, cooked,
                                 articulation::ControlMode::ComputedTorque, joint, 50u);
    ASSERT_EQ(a.first.size(), b.first.size());
    for (size_t i = 0u; i < a.first.size(); ++i) {
        EXPECT_EQ(a.first[i], b.first[i]) << "q differs run-to-run at " << i;
        EXPECT_EQ(a.second[i], b.second[i]) << "qdot differs run-to-run at " << i;
    }
}

TEST(ControlModes, ActuatorModeTwoRunByteExact) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    const uint32_t joint = FirstActuatedLink(cooked.host);
    ASSERT_NE(joint, kInvalidLink);
    const auto a = RunSlice2Mode(context, cooked,
                                 articulation::ControlMode::Actuator, joint, 50u);
    const auto b = RunSlice2Mode(context, cooked,
                                 articulation::ControlMode::Actuator, joint, 50u);
    ASSERT_EQ(a.first.size(), b.first.size());
    for (size_t i = 0u; i < a.first.size(); ++i) {
        EXPECT_EQ(a.first[i], b.first[i]) << "q differs run-to-run at " << i;
        EXPECT_EQ(a.second[i], b.second[i]) << "qdot differs run-to-run at " << i;
    }
}

// ---------------------------------------------------------------------------
// FLOATING-BASE ComputedTorque: exercises the base_dof=6 SCHUR-COMPLEMENT branch
// (the fixed-base go2_stand scene above only hits base_dof=0, where the joint
// admittance D is the full M^-1 block). On a free-floating root the effective
// joint admittance is the Schur complement = the JOINT sub-block of the full-tile
// physics-M inverse (offset 6), and the kernel solves D*tau_j = a_des - qddot_free
// against THAT block. The defining ID check holds regardless of base type: with
// Kp=Kd=0 (pure gravity/bias comp) the realized JOINT qddot ~ a_des == 0 (the base
// itself still free-falls -- it is unactuated -- but the JOINTS hold). Ground is
// far away so the only forcing is gravity vs its compensation.
// ---------------------------------------------------------------------------
namespace {

CookedGo2 CookGo2Float() {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);

    CookedGo2 result;
    result.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);
    const uint32_t link_count = result.host.TotalLinkCount();
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
    return result;
}

}  // namespace

TEST(ControlModes, ComputedTorqueFloatingBaseGravityComp) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& base = cooked.host;
    const uint32_t max_dof = articulation::ArticulationDofCount(base, 0u);
    ASSERT_GT(max_dof, 0u);
    const uint32_t base_link_count = base.TotalLinkCount();
    // Confirm this scene IS floating base (the root link is a FloatingBase joint),
    // so the base_dof=6 Schur path is genuinely exercised.
    bool has_floating_root = false;
    for (uint32_t l = 0u; l < base_link_count; ++l) {
        if (base.joint_type[l] == articulation::ArticulationJointType::FloatingBase) {
            has_floating_root = true;
            break;
        }
    }
    ASSERT_TRUE(has_floating_root)
        << "go2_float must have a FloatingBase root for the Schur-path test";

    std::vector<float> targets = base.q;
    std::vector<float> zeros(base_link_count, 0.0f);
    nuka::phi::Buffer tgt_dev = UploadFloats(targets);
    nuka::phi::Buffer kp_dev = UploadFloats(zeros);  // Kp = 0 (pure bias comp).
    nuka::phi::Buffer kd_dev = UploadFloats(zeros);  // Kd = 0.

    gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                    kGroundFarAway,
                                    gpu::DeterminismLevel::Strong,
                                    articulation::ControlMode::ComputedTorque);
    gpu::BatchedArticulatedStepParams params;
    params.drive_targets = static_cast<const float*>(tgt_dev.Data());
    params.drive_stiffness = static_cast<const float*>(kp_dev.Data());
    params.drive_damping = static_cast<const float*>(kd_dev.Data());
    // No force limits -> the exact (Schur-solved) comp torque is applied.
    params.gravity_z = kGravityOn;
    params.dt = kDt;
    bw.Step(params);

    const auto qddot = DownloadQddot(context, bw, base_link_count);
    ASSERT_TRUE(AllFinite(qddot));
    // The JOINT (revolute) accelerations must be ~0 (a_des == 0): the Schur-block
    // solve makes stage-2 ABA realize qddot_j = a_des_j exactly. The FloatingBase
    // root DOFs are NOT actuated -- skip them (the base free-falls under gravity).
    for (uint32_t l = 0u; l < base_link_count; ++l) {
        if (base.joint_type[l] != articulation::ArticulationJointType::Revolute &&
            base.joint_type[l] != articulation::ArticulationJointType::Prismatic) {
            continue;
        }
        EXPECT_LT(std::fabs(qddot[l]), 1.0e-2f)
            << "floating-base ComputedTorque (Kp=Kd=0) JOINT qddot not ~0 at link "
            << l << " (qddot=" << qddot[l] << ")";
    }

    // Two-run byte-exactness on the floating-base Schur path (D1).
    gpu::BatchedArticulatedWorld bw2(context, base, cooked.feet, max_dof,
                                     kGroundFarAway,
                                     gpu::DeterminismLevel::Strong,
                                     articulation::ControlMode::ComputedTorque);
    bw2.Step(params);
    const auto qddot2 = DownloadQddot(context, bw2, base_link_count);
    ASSERT_EQ(qddot.size(), qddot2.size());
    for (size_t i = 0u; i < qddot.size(); ++i) {
        EXPECT_EQ(qddot[i], qddot2[i]) << "floating-base qddot differs run-to-run "
                                       << "at " << i;
    }
}

// ===========================================================================
// Slice 3 (p03 R2): Osc -- operational-space control, FORWARD position task.
// ===========================================================================
//
// The Osc forward law drives the WORLD POSITION of a chosen task link toward a
// per-env 3-vector target:
//   tau = J^T*Lambda*(Kp*e_x + Kd*edot_x),  Lambda = (J M^-1 J^T)^-1,
//   e_x = x_target - x_current,  edot_x = -(J*qdot).
// With gravity OFF, a fixed-base trunk (base_dof=0, so the joint admittance D is
// the full M^-1 and J M^-1 J^T * Lambda == I exactly) and Kd / drive_damping = 0,
// the realized task acceleration is xddot = Kp*e_x -- the controller pushes the
// task link STRAIGHT toward the target. We verify the cleanest robust invariant:
// the task-link world position MOVES toward the target (its displacement over the
// run has a positive projection on e_x, and the task-space error DECREASES).
// drive_damping is held at 0 so it does not double as the orthogonal implicit
// joint damping (#43) -- a Kp-only undamped task spring, like ComputedTorqueTracksTarget.
namespace {

// The Osc task link: a foot/calf link (a real end-effector with a rich task
// Jacobian, NOT the near-root FirstActuatedLink). Returns its articulation-local
// index (== global for a single-env world).
uint32_t OscTaskLink(const CookedGo2& cooked) {
    return cooked.feet.empty() ? 0u : cooked.feet.front().calf_local_link;
}

// Downloads one link's world position from the device link_pose buffer. After a
// Step() link_pose holds the FK pose for that step (the Osc kernel refreshed it
// at stage 1 from the current q; stage 4 refreshed it again to the same bytes).
Vec3 DownloadLinkPosition(const nuka::phi::DeviceContext& context,
                          gpu::BatchedArticulatedWorld& bw, uint32_t link) {
    const articulation::ArticulationDeviceState state = bw.View();
    context.stream.Synchronize();
    nuka::math::Transform t{};
    cudaMemcpy(&t, state.link_pose + link, sizeof(nuka::math::Transform),
               cudaMemcpyDeviceToHost);
    return t.position;
}

float Dot3(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

}  // namespace

// ---------------------------------------------------------------------------
// Osc drives the task link toward the target: the task-link world position moves
// toward x_target (positive projection on the initial error) and the task-space
// error |x - x_target| DECREASES over the run.
// ---------------------------------------------------------------------------
TEST(ControlModes, OscDrivesTaskTowardTarget) {
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
    const uint32_t task_link = OscTaskLink(cooked);
    ASSERT_NE(task_link, 0u) << "no foot/calf task link found";

    // Kp-only task gain on the task link (drive_stiffness reused). drive_damping
    // = 0 (no control Kd AND no implicit joint damping muddying the response).
    std::vector<float> stiffness(base_link_count, 0.0f);
    std::vector<float> damping(base_link_count, 0.0f);
    const float kKp = 200.0f;  // rad/s^2-ish task gain.
    stiffness[task_link] = kKp;
    nuka::phi::Buffer stiff_dev = UploadFloats(stiffness);
    nuka::phi::Buffer damp_dev = UploadFloats(damping);

    gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                    kGroundFarAway,
                                    gpu::DeterminismLevel::Strong,
                                    articulation::ControlMode::Osc, task_link);
    EXPECT_EQ(bw.ControlMode(), articulation::ControlMode::Osc);
    EXPECT_EQ(bw.OscTaskLink(), task_link);

    // Step once with ZERO stiffness first so the kernel refreshes link_pose to the
    // live FK pose WITHOUT injecting velocity (Kp=0 => tau=0 => foot stays at rest,
    // qdot still 0). Reading x_current after this is the cooked rest task position
    // -- the clean start the directional invariant needs. (Stepping with the REAL
    // Kp here, while task_target still points at the zeroed origin buffer, would
    // dump a large origin-ward velocity into qdot before we anchor the target.)
    std::vector<float> zero_stiff(base_link_count, 0.0f);
    nuka::phi::Buffer zero_stiff_dev = UploadFloats(zero_stiff);
    gpu::BatchedArticulatedStepParams warm;
    warm.gravity_z = kGravityOff;
    warm.dt = kDt;
    warm.drive_stiffness = static_cast<const float*>(zero_stiff_dev.Data());
    warm.drive_damping = static_cast<const float*>(damp_dev.Data());
    warm.task_target = static_cast<const float*>(bw.TaskTargetBuffer().Data());
    bw.Step(warm);
    const Vec3 x0 = DownloadLinkPosition(context, bw, task_link);

    // Target = current task position + a small +X offset. +X is a REACHABLE task
    // direction for the calf-link origin (the hip/thigh joints sweep it fore/aft).
    // NOTE the task point is the calf's LINK ORIGIN, so the calf's OWN joint
    // contributes a zero Jacobian column (cross(axis, x - origin) with x == origin);
    // J is therefore rank-2 and vertical (+Z) motion of this point lies largely in
    // the UNREACHABLE left-null space -- the controller correctly tracks only the
    // reachable subspace (+X/+Y), which is the documented bounded scope. A small +X
    // offset sits squarely in range(J), so the task tracks cleanly.
    const Vec3 target{x0.x + 0.05f, x0.y, x0.z};
    std::vector<float> target_host = {target.x, target.y, target.z};
    nuka::phi::Buffer target_dev = UploadFloats(target_host);

    gpu::BatchedArticulatedStepParams params;
    params.gravity_z = kGravityOff;
    params.dt = kDt;
    params.drive_stiffness = static_cast<const float*>(stiff_dev.Data());
    params.drive_damping = static_cast<const float*>(damp_dev.Data());
    params.task_target = static_cast<const float*>(target_dev.Data());

    const Vec3 e0{target.x - x0.x, target.y - x0.y, target.z - x0.z};
    const float err0 =
        std::sqrt(e0.x * e0.x + e0.y * e0.y + e0.z * e0.z);
    ASSERT_GT(err0, 1e-4f);

    float min_err = err0;
    Vec3 x_last = x0;
    const uint32_t kSteps = 25u;  // short horizon: pre-quarter-period (undamped).
    for (uint32_t step = 0u; step < kSteps; ++step) {
        bw.Step(params);
        const Vec3 x = DownloadLinkPosition(context, bw, task_link);
        ASSERT_TRUE(std::isfinite(x.x) && std::isfinite(x.y) &&
                    std::isfinite(x.z))
            << "task position non-finite at step " << step;
        const Vec3 e{target.x - x.x, target.y - x.y, target.z - x.z};
        const float err = std::sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
        min_err = std::fmin(min_err, err);
        x_last = x;
    }

    // (1) The task link moved TOWARD the target: net displacement projected on the
    // initial error direction is positive.
    const Vec3 disp{x_last.x - x0.x, x_last.y - x0.y, x_last.z - x0.z};
    EXPECT_GT(Dot3(disp, e0), 0.0f)
        << "task link did not move toward the target (disp.e0="
        << Dot3(disp, e0) << ")";
    // (2) The task-space error decreased appreciably (the controller is doing work
    // toward the target, not just jittering).
    EXPECT_LT(min_err, 0.9f * err0)
        << "Osc task error did not decrease (err0=" << err0
        << ", min_err=" << min_err << ")";
}

// ---------------------------------------------------------------------------
// Osc D1 two-run byte-exact: two Osc worlds, same task link / target / gains,
// N steps -> bit-identical q/qdot (no float atomics, fixed order => D1).
// ---------------------------------------------------------------------------
namespace {

std::pair<std::vector<float>, std::vector<float>> RunOscMode(
    const nuka::phi::DeviceContext& context, const CookedGo2& cooked,
    uint32_t steps) {
    auto base = cooked.host;
    const uint32_t max_dof = articulation::ArticulationDofCount(base, 0u);
    const uint32_t base_link_count = base.TotalLinkCount();
    const uint32_t task_link =
        cooked.feet.empty() ? 0u : cooked.feet.front().calf_local_link;

    gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof,
                                    kGroundFarAway,
                                    gpu::DeterminismLevel::Strong,
                                    articulation::ControlMode::Osc, task_link);
    std::vector<float> stiffness(base_link_count, 0.0f);
    std::vector<float> damping(base_link_count, 0.0f);
    stiffness[task_link] = 200.0f;
    nuka::phi::Buffer stiff_dev = UploadFloats(stiffness);
    nuka::phi::Buffer damp_dev = UploadFloats(damping);
    // A fixed nonzero target (env 0; single env so 3 floats).
    std::vector<float> target_host = {0.2f, 0.0f, 0.1f};
    nuka::phi::Buffer target_dev = UploadFloats(target_host);

    gpu::BatchedArticulatedStepParams params;
    params.gravity_z = kGravityOff;
    params.dt = kDt;
    params.drive_stiffness = static_cast<const float*>(stiff_dev.Data());
    params.drive_damping = static_cast<const float*>(damp_dev.Data());
    params.task_target = static_cast<const float*>(target_dev.Data());

    for (uint32_t step = 0u; step < steps; ++step) {
        bw.Step(params);
    }
    context.stream.Synchronize();
    articulation::ArticulationHostState dl = base;
    bw.Download(&dl);
    return {dl.q, dl.qdot};
}

}  // namespace

TEST(ControlModes, OscModeTwoRunByteExact) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    const auto a = RunOscMode(context, cooked, 50u);
    const auto b = RunOscMode(context, cooked, 50u);
    ASSERT_EQ(a.first.size(), b.first.size());
    for (size_t i = 0u; i < a.first.size(); ++i) {
        EXPECT_EQ(a.first[i], b.first[i]) << "q differs run-to-run at " << i;
        EXPECT_EQ(a.second[i], b.second[i]) << "qdot differs run-to-run at " << i;
    }
}
