// ---------------------------------------------------------------------------
// p01-F T6 -- StepBatchedArticulatedWorld production step path
//
// Validates the batched 4096-env Go2 articulated-with-contacts step path
// (src/runtime/gpu/batched_articulated_world.{cu,hpp}) which orchestrates the
// validated T1..T5 kernels into one deterministic per-step pipeline.
//
// There is NO contacts-on oracle, and "4096 envs identical + two-run
// determinism" only proves replication+determinism -- a uniform ORCHESTRATION
// bug (mis-ordered stage, wrong integration split, scratch aliasing, broken
// lambda carry) passes those. So the gates measure what the wiring cannot fake:
//
//  (1) PRIMARY -- contacts-INACTIVE batched == single-env Featherstone engine
//      path, BIT-FOR-BIT, over K steps. Ground placed far away (contacts never
//      active => the solve is a genuine no-op, but ALL stages still execute),
//      diffed against the trusted single-env engine sequence
//      (ApplyPositionDrives -> ComputeAccelerations -> Integrate) used by
//      nuka::c_abi::StepWorldGpu. ASSERT_EQ on raw floats at every step.
//  (2) Multi-step contacts-ON at 4096-env scale: no NaN/Inf; feet carry load
//      (lambda_n > 0); penetration bounded.
//  (3) Determinism (D1): all 4096 envs bit-identical to each other AND
//      bit-identical across two independent runs (q, qdot, lambda).
//  (4) Clamp binds: with a large injected penetration, the normal-row bias is
//      capped at baumgarte_max_velocity and the one-step joint velocities are
//      bounded.
//  (5) Single-env path untouched: reported separately (nuka_regression_test).
//  (6) Timing: every stage wrapped with the canonical perf tags; DumpJson is
//      exercised and a perf-JSON sample for the 4096-env run is PRINTED.
// ---------------------------------------------------------------------------

#include "core/perf/perf_recorder.hpp"
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
#include "scene/cooked_blob.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace gpu = nuka::runtime::gpu;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// Hold-drive buffers (target = the cooked stance q, scene-derived PD gains and
// force limits) for ONE replica's links. Built by mirroring the C-ABI engine's
// BuildHoldDriveTargets (src/c_abi/world.cpp): pull per-actuator gain + force
// limit from the cooked actuator table, damping = 2*sqrt(gain), target = the
// cooked stance q. This is the SAME proven-stable drive config the trusted Go2
// golden runs (1200 steps at dt=1/240 bounded), so the gate-1 reference stays
// finite and the bit-exact batched-vs-engine diff is meaningful. Both paths are
// fed these identical values so the diff isolates orchestration, not drives.
struct HoldDrives {
    std::vector<float> targets;
    std::vector<float> stiffness;
    std::vector<float> damping;
    std::vector<float> force_limits;
};

// Cooked single-env Go2 plus its base-relative foot shapes and the scene-derived
// hold drives (mirrors the T5/T4 tests' CookGo2 + the engine's drive setup).
struct CookedGo2 {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;
    HoldDrives drives;
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

    // Scene-derived hold drives, mirroring c_abi/world.cpp BuildHoldDriveTargets.
    HoldDrives& d = result.drives;
    d.targets = result.host.q;  // hold the cooked stance.
    d.stiffness.assign(link_count, 0.0f);
    d.damping.assign(link_count, 0.0f);
    d.force_limits.assign(link_count, 0.0f);
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
                articulation::ArticulationJointType::Fixed) {
            continue;
        }
        const float gain = actuator < actuators.gains.size()
                               ? std::fmax(actuators.gains[actuator], 0.0f)
                               : 0.0f;
        const float force_limit =
            actuator < actuators.force_limits.size()
                ? std::fmax(actuators.force_limits[actuator], 0.0f)
                : 0.0f;
        if (gain > 0.0f) {
            d.stiffness[link] = gain;
            d.damping[link] = 2.0f * std::sqrt(gain);  // DefaultDriveDamping.
        }
        if (force_limit > 0.0f) {
            d.force_limits[link] = force_limit;  // ClampDriveLimit (>0 passthrough).
        }
    }
    return result;
}

// Device-resident copy of a HoldDrives for `env_count` replicas (the drive
// buffers are link-major, same convention ApplyPositionDrives uses).
struct DeviceDrives {
    nuka::phi::Buffer targets;
    nuka::phi::Buffer stiffness;
    nuka::phi::Buffer damping;
    nuka::phi::Buffer force_limits;
};

DeviceDrives UploadDrives(const HoldDrives& base, uint32_t env_count) {
    const size_t per = base.targets.size();
    std::vector<float> targets(per * env_count);
    std::vector<float> stiffness(per * env_count);
    std::vector<float> damping(per * env_count);
    std::vector<float> limits(per * env_count);
    for (uint32_t e = 0u; e < env_count; ++e) {
        for (size_t i = 0u; i < per; ++i) {
            targets[e * per + i] = base.targets[i];
            stiffness[e * per + i] = base.stiffness[i];
            damping[e * per + i] = base.damping[i];
            limits[e * per + i] = base.force_limits[i];
        }
    }
    DeviceDrives d;
    auto up = [](const std::vector<float>& v) {
        nuka::phi::Buffer b(v.size() * sizeof(float), nuka::phi::MemoryKind::Device);
        b.CopyFromHost(v.data(), v.size() * sizeof(float));
        return b;
    };
    d.targets = up(targets);
    d.stiffness = up(stiffness);
    d.damping = up(damping);
    d.force_limits = up(limits);
    return d;
}

bool AllFinite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Gate 1 (PRIMARY) -- contacts-inactive batched == single-env engine path,
// bit-for-bit, over K steps.
// ---------------------------------------------------------------------------
TEST(BatchedArticulatedWorld, ContactsInactiveMatchesSingleEnvEngineBitExact) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    auto& base = cooked.host;
    ASSERT_EQ(base.ArticulationCount(), 1u);
    const uint32_t dof = articulation::ArticulationDofCount(base, 0u);
    ASSERT_GT(dof, 0u);
    const uint32_t max_dof = dof;
    const uint32_t base_link_count = base.TotalLinkCount();

    const float kGravityZ = -9.81f;       // matches StepWorldGpu.
    const float kDt = 1.0f / 240.0f;      // matches the single-env golden.
    const uint32_t kSteps = 120u;         // exercises lambda carry + scratch reuse.
    const float kGroundFarAway = -1000.0f;  // contacts never active.

    const HoldDrives& drives = cooked.drives;  // scene-derived, golden-stable.

    // -- Reference: the trusted single-env engine sequence (the same calls
    //    StepWorldGpu makes): ApplyPositionDrives -> ComputeAccelerations ->
    //    Integrate(combined). NO contacts, NO link_pose refresh.
    auto ref_buffers = articulation::UploadArticulationState(context, base);
    DeviceDrives ref_drives = UploadDrives(drives, 1u);
    std::vector<std::vector<float>> ref_q(kSteps), ref_qdot(kSteps);
    {
        const auto state = ref_buffers.View();
        for (uint32_t step = 0u; step < kSteps; ++step) {
            articulation::FeatherstoneAba::ApplyPositionDrives(
                context, state,
                static_cast<const float*>(ref_drives.targets.Data()),
                static_cast<const float*>(ref_drives.stiffness.Data()),
                static_cast<const float*>(ref_drives.damping.Data()),
                static_cast<const float*>(ref_drives.force_limits.Data()));
            articulation::FeatherstoneAba::ComputeAccelerations(context, state, kGravityZ);
            articulation::FeatherstoneAba::Integrate(context, state, kDt);
            context.stream.Synchronize();
            articulation::ArticulationHostState dl = base;
            articulation::DownloadArticulationState(ref_buffers, &dl);
            ref_q[step] = dl.q;
            ref_qdot[step] = dl.qdot;
            // The trusted reference must stay finite for the bit-exact diff to be
            // meaningful (the scene-derived drives keep the golden bounded over
            // 1200 steps, so 120 here must not blow up).
            ASSERT_TRUE(AllFinite(dl.q)) << "reference q non-finite at step " << step;
            ASSERT_TRUE(AllFinite(dl.qdot)) << "reference qdot non-finite at step " << step;
        }
    }

    // -- Batched path, single env, ground far away (contacts inactive). ------
    gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof, kGroundFarAway);
    DeviceDrives bw_drives = UploadDrives(drives, 1u);
    gpu::BatchedArticulatedStepParams params;
    params.drive_targets = static_cast<const float*>(bw_drives.targets.Data());
    params.drive_stiffness = static_cast<const float*>(bw_drives.stiffness.Data());
    params.drive_damping = static_cast<const float*>(bw_drives.damping.Data());
    params.drive_force_limits = static_cast<const float*>(bw_drives.force_limits.Data());
    params.gravity_z = kGravityZ;
    params.dt = kDt;
    params.baumgarte_max_velocity = std::numeric_limits<float>::infinity();  // non-binding.

    size_t total_q_mismatch = 0u;
    size_t total_qdot_mismatch = 0u;
    uint32_t first_mismatch_step = kSteps;
    for (uint32_t step = 0u; step < kSteps; ++step) {
        bw.Step(params);
        context.stream.Synchronize();
        articulation::ArticulationHostState dl = base;
        bw.Download(&dl);
        // Confirm no contacts fired this step (the solve must be a no-op).
        const auto links = bw.DownloadContactLink();
        for (uint32_t s = 0u; s < links.size(); ++s) {
            ASSERT_EQ(links[s], kInvalidLink)
                << "contact unexpectedly active at step " << step << " slot " << s;
        }
        for (uint32_t l = 0u; l < base_link_count; ++l) {
            if (dl.q[l] != ref_q[step][l]) {
                ++total_q_mismatch;
                first_mismatch_step = std::min(first_mismatch_step, step);
            }
            if (dl.qdot[l] != ref_qdot[step][l]) {
                ++total_qdot_mismatch;
                first_mismatch_step = std::min(first_mismatch_step, step);
            }
        }
        // Hard bit-exact assertion every step.
        for (uint32_t l = 0u; l < base_link_count; ++l) {
            ASSERT_EQ(dl.q[l], ref_q[step][l])
                << "q diverged at step " << step << " link " << l;
            ASSERT_EQ(dl.qdot[l], ref_qdot[step][l])
                << "qdot diverged at step " << step << " link " << l;
        }
    }
    std::printf("[diag] (1) PRIMARY equivalence over %u steps: q mismatches=%zu "
                "qdot mismatches=%zu (first at step %u; %u==no-mismatch)\n",
                kSteps, total_q_mismatch, total_qdot_mismatch,
                (first_mismatch_step == kSteps ? kSteps : first_mismatch_step),
                kSteps);
    EXPECT_EQ(total_q_mismatch, 0u);
    EXPECT_EQ(total_qdot_mismatch, 0u);
}

// ---------------------------------------------------------------------------
// Gates 2,3,6 -- contacts-ON at 4096-env scale: stability, determinism, timing.
// ---------------------------------------------------------------------------
TEST(BatchedArticulatedWorld, ContactsOnScaleStabilityDeterminismTiming) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    auto& base = cooked.host;
    ASSERT_EQ(base.ArticulationCount(), 1u);
    const uint32_t dof = articulation::ArticulationDofCount(base, 0u);
    const uint32_t max_dof = dof;
    const uint32_t base_link_count = base.TotalLinkCount();

    const float kGravityZ = -9.81f;
    // dt = 1/240, the Go2 golden's proven-stable step. (A larger step like 0.005
    // pushes the stiff scene PD gains past their explicit-integration stability
    // edge -- a contacts-INACTIVE probe at dt=0.005 already shows ~47 rad/s purely
    // from the drive integration, swamping the contact response; at 1/240 the
    // contacts-inactive baseline is a calm ~1.9 rad/s, so this dt isolates the
    // contact behavior in the validated-stable integration regime.)
    const float kDt = 1.0f / 240.0f;
    const float kGround = 0.31f;  // penetrates the cooked stance feet (~0.031 m).
    const float kBaumgarteMaxVel = 3.0f;  // tuned finite cap (m/s); documented choice.
    const uint32_t kEnvCount = 4096u;
    const uint32_t kSteps = 60u;

    const HoldDrives& drives = cooked.drives;  // scene-derived, golden-stable.

    // Replicate to 4096 envs.
    auto batched = articulation::ReplicateArticulationHostState(base, kEnvCount);
    ASSERT_EQ(batched.ArticulationCount(), kEnvCount);
    DeviceDrives dd = UploadDrives(drives, kEnvCount);

    gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof, kGround);
    gpu::BatchedArticulatedStepParams params;
    params.drive_targets = static_cast<const float*>(dd.targets.Data());
    params.drive_stiffness = static_cast<const float*>(dd.stiffness.Data());
    params.drive_damping = static_cast<const float*>(dd.damping.Data());
    params.drive_force_limits = static_cast<const float*>(dd.force_limits.Data());
    params.gravity_z = kGravityZ;
    params.dt = kDt;
    params.baumgarte_max_velocity = kBaumgarteMaxVel;

    uint32_t loaded_feet_seen = 0u;
    float max_total_depth = 0.0f;
    float first_total_depth = -1.0f;
    float last_total_depth = 0.0f;
    // Energy-bounded proxy: the joint-velocity magnitude must not diverge over the
    // run. A slow blow-up can stay finite and keep feet loaded, so finiteness +
    // penetration alone do not prove energy boundedness. The cooked stance starts
    // ~0.124 m penetrated, so the first contact steps inject a large one-step
    // correction transient (the chain Jacobian amplifies the foot-space Baumgarte
    // cap into joint space). "Bounded over the run" is therefore proven by DECAY:
    // the early-transient peak is finite, and the steady-state tail is much smaller
    // than that peak (energy is being removed, not injected). A monotonically
    // growing instability would instead make the tail >= the early peak.
    const uint32_t kTransientSteps = 5u;  // initial penetration-correction window.
    float max_qdot_transient = 0.0f;  // peak over the early correction steps.
    float max_qdot_tail = 0.0f;       // peak over the settled tail.
    // "Energy bounded" must be measured on ACTUATED DOFs: the integrate kernel
    // advances qdot for every link including Fixed joints (qddot for a Fixed slot
    // is unconstrained scratch), and those inert slots never move a body, never
    // appear in the golden (JOINT_POSITION reads actuated joints only), and stay
    // bit-exact across both paths in gate 1 -- so they must not drive the gate.
    float max_actuated_transient = 0.0f;
    float max_actuated_tail = 0.0f;
    uint32_t argmax_link = ~0u;  // diagnostic: which link carries the peak.
    for (uint32_t step = 0u; step < kSteps; ++step) {
        bw.Step(params);
        context.stream.Synchronize();
        articulation::ArticulationHostState dl = batched;
        bw.Download(&dl);
        ASSERT_TRUE(AllFinite(dl.q)) << "non-finite q at step " << step;
        ASSERT_TRUE(AllFinite(dl.qdot)) << "non-finite qdot at step " << step;
        float step_max_qdot = 0.0f;
        float step_max_actuated = 0.0f;
        for (uint32_t l = 0u; l < dl.qdot.size(); ++l) {
            const float a = std::fabs(dl.qdot[l]);
            step_max_qdot = std::fmax(step_max_qdot, a);
            const uint32_t base_l = l % base_link_count;
            if (base.joint_type[base_l] != articulation::ArticulationJointType::Fixed) {
                if (a > step_max_actuated) {
                    step_max_actuated = a;
                    argmax_link = base_l;
                }
            }
        }
        if (step < kTransientSteps) {
            max_qdot_transient = std::fmax(max_qdot_transient, step_max_qdot);
            max_actuated_transient = std::fmax(max_actuated_transient, step_max_actuated);
        } else {
            max_qdot_tail = std::fmax(max_qdot_tail, step_max_qdot);
            max_actuated_tail = std::fmax(max_actuated_tail, step_max_actuated);
        }

        // env-0 penetration + load checks.
        const auto depth = bw.DownloadContactDepth();
        const auto lambda = bw.DownloadLambda();
        const auto rows = bw.DownloadRows();
        float total_depth = 0.0f;
        for (uint32_t s = 0u; s < articulation::kMaxFootContactsPerEnv; ++s) {
            total_depth += std::fmax(depth[s], 0.0f);
            if (rows[s].row_type == articulation::ContactRowType::kRowNormal &&
                lambda[s * 3u] > 0.0f) {
                ++loaded_feet_seen;
            }
        }
        if (first_total_depth < 0.0f) {
            first_total_depth = total_depth;
        }
        last_total_depth = total_depth;
        max_total_depth = std::fmax(max_total_depth, total_depth);
    }
    const char* argmax_type =
        (argmax_link != ~0u &&
         base.joint_type[argmax_link] == articulation::ArticulationJointType::Revolute)
            ? "Revolute"
            : "Prismatic";
    std::printf("[diag] (2) 4096-env contacts-on over %u steps: loaded-feet-samples=%u "
                "env0 total depth first=%.6f last=%.6f max=%.6f | "
                "max|qdot| all: transient(<%u)=%.4f tail=%.4f | actuated: transient=%.4f "
                "tail=%.4f rad/s (argmax link=%u type=%s)\n",
                kSteps, loaded_feet_seen,
                static_cast<double>(first_total_depth),
                static_cast<double>(last_total_depth),
                static_cast<double>(max_total_depth),
                kTransientSteps,
                static_cast<double>(max_qdot_transient),
                static_cast<double>(max_qdot_tail),
                static_cast<double>(max_actuated_transient),
                static_cast<double>(max_actuated_tail),
                argmax_link, argmax_type);
    EXPECT_GT(loaded_feet_seen, 0u) << "feet must carry load (lambda_n>0) at some step";
    EXPECT_LT(max_total_depth, 0.5f) << "penetration grew unbounded";
    // Penetration must drive toward zero: the feet start ~0.124 m through the
    // ground and the contact solve resolves it. The settled depth is ~7e-5 m, so
    // 0.01 is a generous bound that still gates genuine convergence (not just
    // boundedness).
    EXPECT_LT(last_total_depth, 0.01f)
        << "penetration did not converge toward zero (feet not resolving to the surface)";
    // Energy bounded over the run (measured on actuated DOFs): the velocity is
    // finite and does NOT grow -- the settled tail does not exceed the early
    // penetration-correction transient (a small margin absorbs solver jitter). A
    // diverging system would instead have tail >> transient.
    EXPECT_TRUE(std::isfinite(max_actuated_transient));
    EXPECT_LT(max_actuated_tail, max_actuated_transient * 1.10f + 1.0f)
        << "actuated joint velocity (energy proxy) grew over the run -- not bounded";

    // -- (3) Determinism: all 4096 envs bit-identical to env 0. -------------
    {
        articulation::ArticulationHostState dl = batched;
        bw.Download(&dl);
        const auto lambda = bw.DownloadLambda();
        size_t q_mis = 0u, qdot_mis = 0u, lambda_mis = 0u;
        const uint32_t per_env_slots = articulation::kMaxFootContactsPerEnv * 3u;
        for (uint32_t env = 1u; env < kEnvCount; ++env) {
            for (uint32_t l = 0u; l < base_link_count; ++l) {
                if (dl.q[env * base_link_count + l] != dl.q[l]) ++q_mis;
                if (dl.qdot[env * base_link_count + l] != dl.qdot[l]) ++qdot_mis;
            }
            for (uint32_t i = 0u; i < per_env_slots; ++i) {
                if (lambda[env * per_env_slots + i] != lambda[i]) ++lambda_mis;
            }
        }
        std::printf("[diag] (3) cross-env mismatches: q=%zu qdot=%zu lambda=%zu\n",
                    q_mis, qdot_mis, lambda_mis);
        EXPECT_EQ(q_mis, 0u);
        EXPECT_EQ(qdot_mis, 0u);
        EXPECT_EQ(lambda_mis, 0u);
    }

    // -- (3) Determinism: bit-identical across two independent runs. ---------
    {
        const uint32_t kRunEnv = 256u;  // smaller for the second cold run.
        const uint32_t kRunSteps = 30u;
        auto rep = articulation::ReplicateArticulationHostState(base, kRunEnv);
        DeviceDrives ddr = UploadDrives(drives, kRunEnv);
        gpu::BatchedArticulatedStepParams rp = params;
        rp.drive_targets = static_cast<const float*>(ddr.targets.Data());
        rp.drive_stiffness = static_cast<const float*>(ddr.stiffness.Data());
        rp.drive_damping = static_cast<const float*>(ddr.damping.Data());
        rp.drive_force_limits = static_cast<const float*>(ddr.force_limits.Data());

        auto run = [&]() {
            gpu::BatchedArticulatedWorld w(context, rep, cooked.feet, max_dof, kGround);
            for (uint32_t s = 0u; s < kRunSteps; ++s) {
                w.Step(rp);
            }
            context.stream.Synchronize();
            articulation::ArticulationHostState dl = rep;
            w.Download(&dl);
            return std::make_pair(dl, w.DownloadLambda());
        };
        const auto [a_state, a_lambda] = run();
        const auto [b_state, b_lambda] = run();
        size_t q_mis = 0u, qdot_mis = 0u, lambda_mis = 0u;
        for (size_t i = 0u; i < a_state.q.size(); ++i) {
            if (a_state.q[i] != b_state.q[i]) ++q_mis;
            if (a_state.qdot[i] != b_state.qdot[i]) ++qdot_mis;
        }
        for (size_t i = 0u; i < a_lambda.size(); ++i) {
            if (a_lambda[i] != b_lambda[i]) ++lambda_mis;
        }
        std::printf("[diag] (3) two-run mismatches: q=%zu qdot=%zu lambda=%zu\n",
                    q_mis, qdot_mis, lambda_mis);
        EXPECT_EQ(q_mis, 0u);
        EXPECT_EQ(qdot_mis, 0u);
        EXPECT_EQ(lambda_mis, 0u);
    }

    // -- (6) Timing: exercise DumpJson and print the 4096-env perf sample. ---
    const std::string perf_json = bw.Perf().DumpJson();
    ASSERT_FALSE(perf_json.empty());
    std::printf("[diag] (6) 4096-env Go2 step perf sample (RTX 4000 Ada):\n%s\n",
                perf_json.c_str());
    bw.Perf().PrintTable(stdout);
    // Sanity: the canonical stage tags are present.
    for (const char* tag : {"featherstone_aba", "integrator", "contact_generation",
                            "row_builder", "row_solver", "buffer_mgmt", "step_total"}) {
        EXPECT_NE(perf_json.find(tag), std::string::npos)
            << "perf JSON missing canonical tag " << tag;
    }
}

// ---------------------------------------------------------------------------
// Gate 4 -- the Baumgarte clamp binds and bounds one-step joint velocities.
// ---------------------------------------------------------------------------
TEST(BatchedArticulatedWorld, BaumgarteClampBindsAtLargePenetration) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    auto& base = cooked.host;
    const uint32_t dof = articulation::ArticulationDofCount(base, 0u);
    const uint32_t max_dof = dof;
    const uint32_t base_link_count = base.TotalLinkCount();

    const float kGravityZ = -9.81f;
    const float kDt = 0.005f;
    // Ground well above the feet -> large penetration -> uncapped bias would be
    // beta/dt*depth = 0.2/0.005*depth, ~tens of m/s at depth ~ 0.1 m.
    const float kGroundDeep = 0.40f;
    const float kClampVel = 3.0f;

    // Single env so we can read env-0 directly.
    auto run = [&](float baumgarte_max_velocity) {
        gpu::BatchedArticulatedWorld bw(context, base, cooked.feet, max_dof, kGroundDeep);
        gpu::BatchedArticulatedStepParams params;
        params.gravity_z = kGravityZ;
        params.dt = kDt;
        params.baumgarte_max_velocity = baumgarte_max_velocity;
        // No drives (tau=0): isolate the contact response.
        bw.Step(params);
        context.stream.Synchronize();
        articulation::ArticulationHostState dl = base;
        bw.Download(&dl);
        const auto depth = bw.DownloadContactDepth();
        float max_depth = 0.0f, max_qdot = 0.0f;
        for (uint32_t s = 0u; s < articulation::kMaxFootContactsPerEnv; ++s) {
            max_depth = std::fmax(max_depth, depth[s]);
        }
        for (uint32_t l = 0u; l < base_link_count; ++l) {
            max_qdot = std::fmax(max_qdot, std::fabs(dl.qdot[l]));
        }
        return std::make_tuple(max_depth, max_qdot, dl.qdot);
    };

    const auto [depth_unc, qdot_unc, qd_unc] = run(std::numeric_limits<float>::infinity());
    const auto [depth_cl, qdot_cl, qd_cl] = run(kClampVel);
    (void)qd_unc;
    (void)qd_cl;

    const float uncapped_bias = articulation::kBaumgarteBeta / kDt *
                                std::fmax(depth_unc - articulation::kPenetrationSlop, 0.0f);
    std::printf("[diag] (4) max penetration=%.4f uncapped_bias=%.3f m/s | "
                "max|qdot|: uncapped=%.3f capped(@%.1f)=%.3f\n",
                static_cast<double>(depth_unc), static_cast<double>(uncapped_bias),
                static_cast<double>(qdot_unc), static_cast<double>(kClampVel),
                static_cast<double>(qdot_cl));
    // Pre-condition: the penetration is large enough that the uncapped bias
    // exceeds the clamp (otherwise the clamp is trivially inactive).
    ASSERT_GT(uncapped_bias, kClampVel)
        << "test setup: penetration too small to exercise the clamp";
    // The clamp must measurably reduce the one-step joint velocity transient.
    EXPECT_LT(qdot_cl, qdot_unc)
        << "clamp did not reduce the one-step velocity transient";
    // And bound it: the clamped foot velocity cannot exceed the cap by more than
    // the Jacobian scale; assert a generous but finite bound proving it is not
    // the tens-of-m/s uncapped transient.
    EXPECT_LT(qdot_cl, qdot_unc * 0.9f)
        << "clamp effect too weak to be meaningfully binding";
    EXPECT_TRUE(std::isfinite(qdot_cl));
}
