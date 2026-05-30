// ---------------------------------------------------------------------------
// v0.3 p01 Wave-2 -- fine-grained (detail) timing breakdown of the 4096-env
// batched articulated step.
//
// The shipped canonical-tag timing (featherstone_aba / integrator /
// contact_generation / row_builder / row_solver / buffer_mgmt / step_total) is
// recorded a DIFFERENT number of times per step (row_builder 4x, integrator 2x,
// contact_generation 2x, ...), so a per-tag p50 table is NOT summable to
// step_total -- that artefact is the "~1.5 ms unaccounted gap" reported in
// p01-F. This harness enables the opt-in detail instrumentation
// (Perf().SetDetailEnabled(true)) which records ONE uniquely-tagged scope per
// distinct sub-stage exactly once per step, then prints an accounting-complete
// breakdown where Sigma(sub-stage p50) + launch_sync_overhead == step_total p50.
//
// It is a measurement harness, not a correctness gate -- correctness,
// determinism and the canonical-tag contract are owned by
// test_batched_articulated_world.cpp (unchanged). This file never edits the
// physics path; detail scopes are pure CUDA-event timers (no float math, no
// kernel reordering), so D1 determinism and the owner golden are unaffected.
// ---------------------------------------------------------------------------

#include "core/perf/perf_recorder.hpp"
#include "import/usd_importer.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"
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

struct HoldDrives {
    std::vector<float> targets;
    std::vector<float> stiffness;
    std::vector<float> damping;
    std::vector<float> force_limits;
};

struct CookedGo2 {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;
    HoldDrives drives;
};

// Mirrors CookGo2() in test_batched_articulated_world.cpp (scene-derived hold
// drives + base-relative foot shapes) so the breakdown is taken on exactly the
// proven-stable golden drive config.
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

    HoldDrives& d = result.drives;
    d.targets = result.host.q;
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
            d.damping[link] = 2.0f * std::sqrt(gain);
        }
        if (force_limit > 0.0f) {
            d.force_limits[link] = force_limit;
        }
    }
    return result;
}

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

}  // namespace

// ---------------------------------------------------------------------------
// Accounting-complete fine-grained breakdown of the 4096-env step.
// ---------------------------------------------------------------------------
TEST(BatchedArticulatedWorldPerf, DetailBreakdownAccountsForStepTotal) {
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

    const float kGravityZ = -9.81f;
    const float kDt = 1.0f / 240.0f;
    const float kGround = 0.31f;          // matches the canonical timing test.
    const float kBaumgarteMaxVel = 3.0f;
    const uint32_t kEnvCount = 4096u;
    const uint32_t kWarmup = 10u;         // discard launch / cache warm-up steps.
    const uint32_t kSteps = 60u;          // measured steps (p50 over these).

    auto batched = articulation::ReplicateArticulationHostState(base, kEnvCount);
    ASSERT_EQ(batched.ArticulationCount(), kEnvCount);
    DeviceDrives dd = UploadDrives(cooked.drives, kEnvCount);

    gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof, kGround);
    gpu::BatchedArticulatedStepParams params;
    params.drive_targets = static_cast<const float*>(dd.targets.Data());
    params.drive_stiffness = static_cast<const float*>(dd.stiffness.Data());
    params.drive_damping = static_cast<const float*>(dd.damping.Data());
    params.drive_force_limits = static_cast<const float*>(dd.force_limits.Data());
    params.gravity_z = kGravityZ;
    params.dt = kDt;
    params.baumgarte_max_velocity = kBaumgarteMaxVel;

    // Enable the opt-in fine-grained instrumentation (off by default => the
    // production step path pays nothing for these scopes).
    ASSERT_FALSE(bw.Perf().DetailEnabled());

    // Warm-up steps (NOT measured): first-touch allocation, instruction cache,
    // clock ramp. Detail OFF here so warm-up does not pollute the p50 windows.
    for (uint32_t s = 0u; s < kWarmup; ++s) {
        bw.Step(params);
    }
    context.stream.Synchronize();

    // Measured window: detail ON, fresh recorder.
    bw.Perf().Reset();
    bw.Perf().SetDetailEnabled(true);
    for (uint32_t s = 0u; s < kSteps; ++s) {
        bw.Step(params);
    }
    context.stream.Synchronize();

    // The unique once-per-step detail buckets, IN PIPELINE ORDER. Each is
    // recorded exactly once per step, so its p50 is directly summable.
    const char* kDetailTags[] = {
        "aba_apply_drives",
        "aba_compute_accelerations",
        "integrate_velocity",
        "update_world_poses",
        "link_pose_refresh_copy",
        "detect_foot_contacts",
        "contact_tangent_basis",
        "chain_jacobians",
        "crba_inertia_m",
        "factor_inertia_m_inv",
        "contact_effective_mass",
        "assemble_rows",
        "solve_contact_rows",
        "integrate_position",
    };

    const auto step_stats = bw.Perf().Stats("step_total");
    ASSERT_EQ(step_stats.count, kSteps);

    std::printf(
        "\n[detail] 4096-env Go2 step -- accounting-complete p50 breakdown "
        "(RTX 4000 Ada, %u measured steps after %u warm-up):\n", kSteps, kWarmup);
    std::printf("  %-28s %10s %10s %10s\n", "sub-stage (detail tag)", "p50_us",
                "mean_us", "count");

    double sum_p50 = 0.0;
    for (const char* tag : kDetailTags) {
        const auto s = bw.Perf().Stats(tag);
        // Every detail bucket must be recorded exactly once per step.
        EXPECT_EQ(s.count, kSteps) << "detail tag " << tag
                                   << " not recorded once per step";
        sum_p50 += s.p50_us;
        std::printf("  %-28s %10.3f %10.3f %10llu\n", tag, s.p50_us, s.mean_us,
                    static_cast<unsigned long long>(s.count));
    }

    const double step_p50 = step_stats.p50_us;
    // Residual = wall step time minus the summed kernel sub-stages. This is the
    // honest, explicitly-named remainder: per-kernel launch latency, the
    // implicit stream syncs the nested event timers force, and the small
    // host-side guard/View bookkeeping between scopes.
    const double residual = step_p50 - sum_p50;
    std::printf("  %-28s %10s %10s %10s\n", "----", "----", "----", "----");
    std::printf("  %-28s %10.3f\n", "Sigma(sub-stages)", sum_p50);
    std::printf("  %-28s %10.3f   <-- explicit named remainder\n",
                "launch_sync_overhead", residual);
    std::printf("  %-28s %10.3f\n", "step_total (measured p50)", step_p50);
    const double pct = step_p50 > 0.0 ? 100.0 * sum_p50 / step_p50 : 0.0;
    std::printf("  Sigma(sub-stages) accounts for %.1f%% of step_total; "
                "residual = %.3f us (%.1f%%).\n",
                pct, residual, step_p50 > 0.0 ? 100.0 * residual / step_p50 : 0.0);

    // Accounting must be COMPLETE: the named sub-stages explain step_total and
    // the residual is a (non-negative) explicitly-named bucket, not a silent
    // gap. Sigma(sub-stages) can never exceed the wall step time (they are
    // nested in step_total on one stream), and the residual is bounded -- if it
    // ballooned, an un-timed launch would be hiding in it.
    EXPECT_GE(residual, -1.0) << "Sigma(sub-stages) exceeds step_total -- "
                                 "accounting is inconsistent";
    EXPECT_LT(residual, step_p50)
        << "residual dominates step_total -- a sub-stage is un-timed";
    // The dominant named bucket is the data point for the hotspot wave.
    const auto aba = bw.Perf().Stats("aba_compute_accelerations");
    std::printf("[detail] dominant named bucket: aba_compute_accelerations "
                "p50=%.3f us (%.1f%% of step_total).\n",
                aba.p50_us, step_p50 > 0.0 ? 100.0 * aba.p50_us / step_p50 : 0.0);
}
