// ---------------------------------------------------------------------------
// v0.3 p01 Task 3.1.6 -- 4096-env Go2 per-env-step PERF-GATE regression test.
//
// The master-plan §7 gate is "< 1 ms step time PER ENV-STEP at 4096 envs" on
// the validation GPU (RTX 4090; the owner also validates a supplementary
// < 1.3 ms@5080 target). This test is CI-SAFE on non-validation hardware: it
// ALWAYS records the measured per-env-step µs, but only ASSERTS the gate when
// the running GPU's name contains NUKA_PERF_VALIDATION_GPU. Otherwise it
// GTEST_SKIPs with the recorded number in the message -- so on this dev box
// (RTX 4000 Ada) it records-and-skips (green, no false CI failure), and the
// absolute < 1 ms@4090 / < 1.3 ms@5080 sign-off is the owner's validation run.
//
// Knobs (env vars, with defaults):
//   NUKA_PERF_GATE_US        default 1000.0  (§7 4090 bar; owner sets 1300 @5080)
//   NUKA_PERF_VALIDATION_GPU substring of the validation GPU name, e.g. "RTX 5080"
//
// The world construction / drive config / stepping MIRROR the proven-stable
// 4096-env timing gate (BatchedArticulatedWorld.ContactsOnScaleStability-
// DeterminismTiming in tests/runtime/test_batched_articulated_world.cpp): the
// owner golden go2_stand.usda scene, scene-derived hold drives, base-relative
// foot shapes, dt = 1/240, the same ground seat / Baumgarte cap. So the timed
// step is the genuine production step path (active contact solve), not a
// synthetic micro-benchmark. D1 (Strong) determinism by default.
//
// The whole-batch step is wall-clock timed (a host steady_clock around N
// Step() calls + ONE final device sync, so the GPU pipeline stays saturated
// across steps -- this is throughput, the metric the per-env-step gate is
// about). Reported: per_step_us = total_wall_us / N (whole 4096-env step), and
// per_env_step_us = per_step_us / 4096 (the gated figure).
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/vec3.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/device.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/gpu/batched_articulated_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

// --- M3b nk::World perf twin (same scene/constants, StepPlanned graph path) --
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "scene/cook/cook_to_model.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// --- env-var helpers --------------------------------------------------------
double EnvDouble(const char* name, double fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const double v = std::strtod(raw, &end);
    if (end == raw) {
        return fallback;  // unparseable -> keep the protected default.
    }
    return v;
}

// --- scene cook (mirrors CookGo2() in test_batched_articulated_world.cpp) ----
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
// THE PERF GATE (records-and-skips off the validation GPU; asserts on it).
// ---------------------------------------------------------------------------
TEST(Go2_4096env_StepTime, MeetsGatePerEnv) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }

    const nuka::phi::DeviceInfo gpu_info = nuka::phi::GetDeviceInfo(0);
    const std::string gpu_name = gpu_info.name;

    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2();
    auto& base = cooked.host;
    ASSERT_EQ(base.ArticulationCount(), 1u);
    const uint32_t dof = articulation::ArticulationDofCount(base, 0u);
    const uint32_t max_dof = dof;

    // Constants match the canonical 4096-env timing gate -- proven-stable golden.
    const float kGravityZ = -9.81f;
    const float kDt = 1.0f / 240.0f;            // golden-stable step.
    const float kGround = 0.31f;                // penetrates the cooked stance feet.
    const float kBaumgarteMaxVel = 3.0f;
    constexpr uint32_t kEnvCount = 4096u;
    constexpr uint32_t kWarmup = 100u;          // first-touch / clock-ramp (not timed).
    constexpr uint32_t kSteps = 1000u;          // timed window (master-plan N=1000).

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
    params.baumgarte_max_velocity = kBaumgarteMaxVel;  // D1 (Strong) by default.

    // Warmup -- NOT timed (allocations, I-cache, GPU clock ramp). One sync to
    // drain the pipeline before the measured window starts.
    for (uint32_t s = 0u; s < kWarmup; ++s) {
        bw.Step(params);
    }
    context.stream.Synchronize();

    // Timed window: wall-clock around N back-to-back Step() calls + ONE final
    // device sync. No per-step sync -> the GPU stays saturated, so this is the
    // throughput-relevant whole-batch step cost the per-env-step gate measures.
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0u; s < kSteps; ++s) {
        bw.Step(params);
    }
    context.stream.Synchronize();
    const auto t1 = std::chrono::steady_clock::now();

    const double total_wall_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double per_step_us = total_wall_us / static_cast<double>(kSteps);
    const double per_env_step_us = per_step_us / static_cast<double>(kEnvCount);

    // ALWAYS record/print the measured numbers, regardless of hardware (CI logs
    // / baseline diffing). This line is the machine-greppable record.
    std::printf(
        "[perf-gate] gpu='%s' envs=%u steps=%u | per_step=%.3f us | "
        "per_env_step=%.4f us (total_wall=%.1f us)\n",
        gpu_name.c_str(), kEnvCount, kSteps, per_step_us, per_env_step_us,
        total_wall_us);
    std::fflush(stdout);
    RecordProperty("per_step_us", std::to_string(per_step_us));
    RecordProperty("per_env_step_us", std::to_string(per_env_step_us));
    RecordProperty("gpu", gpu_name);

    const double gate_us = EnvDouble("NUKA_PERF_GATE_US", 1000.0);  // §7 4090 bar.
    const char* validation_gpu = std::getenv("NUKA_PERF_VALIDATION_GPU");
    const bool on_validation_hw =
        validation_gpu != nullptr && validation_gpu[0] != '\0' &&
        gpu_name.find(validation_gpu) != std::string::npos;

    if (on_validation_hw) {
        EXPECT_LT(per_env_step_us, gate_us)
            << "Per-env-step " << per_env_step_us << " µs exceeds " << gate_us
            << " µs gate on validation GPU '" << gpu_name << "'.";
    } else {
        GTEST_SKIP() << "Recorded per_env_step=" << per_env_step_us
                     << " µs (per_step=" << per_step_us << " µs) on '" << gpu_name
                     << "' -- not the validation GPU (set NUKA_PERF_VALIDATION_GPU"
                     << " to assert; gate=" << gate_us << " µs). Records-and-skips"
                     << " on the dev box; the §7 gate is the owner's validation run.";
    }
}

// ---------------------------------------------------------------------------
// M3b: the SAME 4096-env Go2 production step through nk::World::StepPlanned
// (the CUDA-graph plan path) — the kernel-port perf regression gate. Identical
// scene / constants / drives / contact seat as the legacy gate above; prints
// the same machine-greppable record line so legacy-vs-nk numbers are directly
// comparable in one run. Records-and-skips off the validation GPU (same
// policy); the M3 plan bar is "<= ~1 µs/env-step, no regression vs legacy".
// ---------------------------------------------------------------------------
TEST(Go2_4096env_StepTime, NkWorldStepPlannedMeetsGatePerEnv) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }

    const nuka::phi::DeviceInfo gpu_info = nuka::phi::GetDeviceInfo(0);
    const std::string gpu_name = gpu_info.name;

    constexpr float kGravityZ = -9.81f;
    constexpr float kDt = 1.0f / 240.0f;
    constexpr float kGround = 0.31f;
    constexpr float kBaumgarteMaxVel = 3.0f;
    constexpr uint32_t kEnvCount = 4096u;
    constexpr uint32_t kWarmup = 100u;
    constexpr uint32_t kSteps = 1000u;

    nuka::phi::Device* dev = nuka::phi::InitBestDevice();
    ASSERT_NE(dev, nullptr);
    nuka::phi::Backend* backend = nuka::phi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    const auto scene = nuka::import::LoadUsd(scene_path.string());
    auto cooked = nuka::scene::cook::CookToModel(scene, 1);
    ASSERT_EQ(cooked.model.feet.size(), 4u);
    cooked.model.ground_height = kGround;
    cooked.model.baumgarte_max_velocity = kBaumgarteMaxVel;
    nuka::nk::Pipeline::SolverConfig cfg;
    cfg.dt = kDt;
    cfg.gravity[2] = kGravityZ;
    nuka::nk::World world(std::move(cooked.model), kEnvCount, dev, backend, cfg);
    ASSERT_TRUE(world.Ready());

    // Warmup (plan capture happens on the first StepPlanned) -- NOT timed.
    for (uint32_t s = 0u; s < kWarmup; ++s) {
        ASSERT_EQ(world.StepPlanned(), nuka::phi::Status::Ok);
    }
    nuka::phi::BackendSynchronize(backend);

    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0u; s < kSteps; ++s) {
        (void)world.StepPlanned();
    }
    nuka::phi::BackendSynchronize(backend);
    const auto t1 = std::chrono::steady_clock::now();

    const double total_wall_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double per_step_us = total_wall_us / static_cast<double>(kSteps);
    const double per_env_step_us = per_step_us / static_cast<double>(kEnvCount);

    std::printf(
        "[perf-gate-nk] gpu='%s' envs=%u steps=%u | per_step=%.3f us | "
        "per_env_step=%.4f us (total_wall=%.1f us)\n",
        gpu_name.c_str(), kEnvCount, kSteps, per_step_us, per_env_step_us,
        total_wall_us);
    std::fflush(stdout);
    RecordProperty("nk_per_step_us", std::to_string(per_step_us));
    RecordProperty("nk_per_env_step_us", std::to_string(per_env_step_us));
    RecordProperty("gpu", gpu_name);

    nuka::phi::BackendFree(backend);

    const double gate_us = EnvDouble("NUKA_PERF_GATE_US", 1000.0);
    const char* validation_gpu = std::getenv("NUKA_PERF_VALIDATION_GPU");
    const bool on_validation_hw =
        validation_gpu != nullptr && validation_gpu[0] != '\0' &&
        gpu_name.find(validation_gpu) != std::string::npos;
    if (on_validation_hw) {
        EXPECT_LT(per_env_step_us, gate_us)
            << "nk per-env-step " << per_env_step_us << " µs exceeds " << gate_us
            << " µs gate on validation GPU '" << gpu_name << "'.";
    } else {
        GTEST_SKIP() << "Recorded nk per_env_step=" << per_env_step_us
                     << " µs (per_step=" << per_step_us << " µs) on '" << gpu_name
                     << "' -- not the validation GPU; records-and-skips.";
    }
}
