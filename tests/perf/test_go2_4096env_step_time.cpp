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
// The world construction / drive config / stepping use the owner golden
// go2_stand.usda scene, scene-derived hold drives, base-relative foot shapes,
// dt = 1/240, the canonical ground seat / Baumgarte cap. So the timed step is
// the genuine production step path (active contact solve), not a synthetic
// micro-benchmark. D1 (Strong) determinism by default.
//
// The whole-batch step is wall-clock timed (a host steady_clock around N
// Step() calls + ONE final device sync, so the GPU pipeline stays saturated
// across steps -- this is throughput, the metric the per-env-step gate is
// about). Reported: per_step_us = total_wall_us / N (whole 4096-env step), and
// per_env_step_us = per_step_us / 4096 (the gated figure).
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "phi/device.hpp"

// --- nk::World perf gate (StepPlanned graph path) ---------------------------
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

}  // namespace

// ---------------------------------------------------------------------------
// The 4096-env Go2 production step through nk::World::StepPlanned (the CUDA-
// graph plan path) — the per-env-step perf gate. Owner go2_stand.usda scene,
// scene-derived hold drives, base-relative foot shapes, dt = 1/240, the same
// ground seat / Baumgarte cap as the production step (active contact solve).
// Prints a machine-greppable record line. Records-and-skips off the validation
// GPU; the §7 plan bar is "< 1 ms/env-step @4090".
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

    // M4 fix: do NOT BackendFree here — `world` (declared after `backend`) is
    // destroyed AFTER this point and its dtor frees the captured plan THROUGH
    // the backend vtable; the explicit free left it dangling (a latent M3b
    // teardown crash, surfaced once the allocator started reusing the block).
    // The backend is process-lifetime here, reclaimed at exit.

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
