// ---------------------------------------------------------------------------
// M4 — nk_union_n1: the plan §5 performance RED LINE.
//   union(H1 + cup + table), N = 1, FULL CONTACT (feet x ground + 30
//   fingertips x cup hull + cup-proxy x table, every step), 1000 steps through
//   nk::World::StepPlanned (the CUDA-graph plan: ONE capture, 1000 replays —
//   narrowphase + AssembleRows + SolveRowsBlockIsland all inside the graph,
//   ZERO per-step host participation):  **<= 5 ms/step HARD**.
//
// The scene holds the settled rest pose under the factory's HOLD drive (the
// curl held at the settled q, table ON) — the maximal steady contact set
// (feet + fingers + table rows live every step). The drive torque is constant
// (uploaded once): the timing loop is pure StepPlanned.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/scoped_device_guard.hpp"
#include "scene/cook/union_nk_model.hpp"
#include "scene/cook/union_cook.hpp"
#include "scene/cook/union_scene_constants.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace coresident = nuka::runtime::coresident;
namespace cook = nuka::scene::cook;

bool AssetsAvailable() {
    return std::filesystem::exists(cook::kH1UnionNksDefault) &&
           std::filesystem::exists(cook::kH1UnionMjcfDefault) &&
           std::filesystem::exists(cook::kH1UnionCupDefault);
}

// Cook the AUTHORED .nks union scene into the UnionSceneTemplate (replacing
// the deleted factory BuildH1UnionScene — proven equivalent to ~1e-8 by the
// h1_grasp_lift gate). Same product surface (tmpl + drive tables + grip dofs +
// scene scalars) the perf gate reads.
cook::CookedUnionScene CookUnionScene(int env_count) {
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(cook::kH1UnionNksDefault);
    return cook::CookSceneToUnionTemplate(scene, env_count);
}

}  // namespace

TEST(NkUnionN1, FullContactStepPlannedUnder5msHard) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    // NOTE: the backend deliberately outlives this test body (the World's
    // dtor frees its plan through the backend vtable) — freed at process exit.
    nphi::Device* dev = nphi::InitBestDevice();
    ASSERT_NE(dev, nullptr);
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    cook::CookedUnionScene sc = CookUnionScene(1);
    ASSERT_TRUE(sc.place_found);
    const float dt = cook::kH1UnionDt;

    nk::Model model = coresident::BuildNkUnionModel(sc.tmpl, 1u);
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[2] = cook::kH1UnionGravityZ;
    cfg.vel_iters = 64;
    cfg.pos_iters = 0;
    cfg.fold_drive_damping = 0;
    nk::World world(std::move(model), 1u, dev, backend, cfg);
    ASSERT_TRUE(world.Ready());

    // Closed-loop HOLD drive (the settled curl + the factory's ankle posture
    // stand-in): per step download q/qdot -> table PD -> upload torque ->
    // StepPlanned. The host PD round-trip is the SAME per-step cost any RL/
    // choreography consumer pays; the gate number is the FULL loop, and the
    // pure StepPlanned time is reported alongside.
    const uint32_t L = sc.link_count;
    std::vector<float> q(L), qd(L), torque(L, 0.0f);
    for (uint32_t l = 0; l < L && l < sc.tmpl.grip_torque.size(); ++l) {
        torque[l] = sc.tmpl.grip_torque[l];
    }
    auto drive_step = [&](double* planned_ms) {
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Q, q.data(),
                                                  L * sizeof(float)));
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Qdot, qd.data(),
                                                  L * sizeof(float)));
        for (const auto& d : sc.drive_hold) {
            if (d.link >= L) continue;
            float u = d.kp * (d.target - q[d.link]) - d.kd * qd[d.link];
            if (d.tlim > 0.0f) u = std::max(-d.tlim, std::min(d.tlim, u));
            torque[d.link] = u;
        }
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::DriveTarget,
                                                torque.data(),
                                                torque.size() * sizeof(float)));
        if (planned_ms != nullptr) {
            const auto p0 = std::chrono::steady_clock::now();
            ASSERT_EQ(world.StepPlanned(), nphi::Status::Ok);
            const auto p1 = std::chrono::steady_clock::now();
            *planned_ms +=
                std::chrono::duration<double, std::milli>(p1 - p0).count();
        } else {
            ASSERT_EQ(world.StepPlanned(), nphi::Status::Ok);
        }
    };

    // Warm-up: first StepPlanned captures + instantiates the graph; a few
    // replays settle clocks/caches. Then assert the contact set is FULL.
    constexpr uint32_t kWarm = 20u;
    for (uint32_t s = 0; s < kWarm; ++s) {
        drive_step(nullptr);
    }
    {
        const uint32_t slots = world.GetModel().capacities.max_contacts_per_env;
        std::vector<uint32_t> counts(slots);
        ASSERT_TRUE(world.GetData().DownloadField(
            nk::FieldId::UcontactCount, counts.data(),
            counts.size() * sizeof(uint32_t)));
        uint32_t row_count = 0;
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::RowCount,
                                                  &row_count, sizeof(uint32_t)));
        uint32_t feet = 0, fingers = 0, table = 0;
        const uint32_t n_feet = static_cast<uint32_t>(sc.tmpl.feet.size());
        const uint32_t n_fing = static_cast<uint32_t>(sc.tmpl.fingertips.size());
        for (uint32_t i = 0; i < slots; ++i) {
            if (i < n_feet) feet += counts[i];
            else if (i < n_feet + n_fing) fingers += counts[i];
            else table += counts[i];
        }
        std::printf("[NK-UNION-N1] live contact set: feet=%u fingers=%u "
                    "table=%u active_rows=%u (slots=%u rows_cap=%u islands=%u)\n",
                    feet, fingers, table, row_count, slots,
                    world.GetModel().capacities.max_rows_per_env,
                    world.GetModel().schedule_island_count);
        ASSERT_GT(feet, 0u) << "feet not in contact — not the full-contact scene";
        ASSERT_GT(fingers, 0u) << "fingers not on the cup — not full contact";
        ASSERT_GT(table, 0u) << "cup not on the table — not full contact";
    }

    // ---- THE GATE: 1000 full-contact steps, wall / 1000 <= 5 ms -------------
    constexpr uint32_t kSteps = 1000u;
    double planned_ms = 0.0;
    nphi::BackendSynchronize(backend);
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0; s < kSteps; ++s) {
        drive_step(&planned_ms);
    }
    nphi::BackendSynchronize(backend);
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_per_step = total_ms / kSteps;
    // NOTE: StepPlanned is an ASYNC graph launch — its wall time below is the
    // enqueue cost only; the GPU step itself is absorbed by the next PD
    // download's sync. The honest gate number is the FULL closed loop.
    std::printf("[NK-UNION-N1 RESULT] %u closed-loop steps in %.1f ms -> "
                "%.4f ms/step TOTAL=GATE (graph-launch enqueue %.4f ms/step; "
                "GPU step + PD round-trip %.4f ms/step) — gate <= 5 ms; "
                "margin %.2fx\n",
                kSteps, total_ms, ms_per_step, planned_ms / kSteps,
                ms_per_step - planned_ms / kSteps, 5.0 / ms_per_step);

    // The scene must still be ALIVE after the timed run (no NaN collapse).
    {
        nuka::math::Transform base{};
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BasePose, &base,
                                                  sizeof(base)));
        EXPECT_TRUE(std::isfinite(base.position.z));
        EXPECT_GT(base.position.z, 0.0f) << "the robot collapsed during the timed run";
    }

    EXPECT_LE(ms_per_step, 5.0) << "the plan §5 union N=1 red line is BROKEN";
}

// ===========================================================================
// M5 — NK GRASP THROUGHPUT (plan M5 RED LINE: "grasp 聚合 >= 11k env-steps/s").
//   The A1-era grasp scene (the SAME union(H1 + cup + table) full-contact
//   topology — feet x ground + fingertips x cup hull + cup-proxy x table) batched
//   at N=1024 through CookToModel-shaped BuildNkUnionModel -> nk::World ->
//   StepPlanned (ONE capture, replayed). The aggregate throughput (N * steps /
//   wall) must clear 11,000 env-steps/s (the baseline the legacy host-O(N)
//   path could not reach: tests/coresident/test_batched_unified_world_perf.cpp
//   measured ~10k eps WITH a host PD round-trip per step; the nk path keeps the
//   whole step device-resident in the graph). The full M5 collision spine
//   (BuildAabbs/Lbvh*/NarrowphaseSdf) is REGISTERED + in the pipeline; for the
//   union family it early-exits, so this is the production union path with the
//   M5 ops present (proving they do not regress the gate).
// ===========================================================================
TEST(NkGraspThroughput, BatchedUnionAtLeast11kEnvStepsPerSecond) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    nphi::Device* dev = nphi::InitBestDevice();
    ASSERT_NE(dev, nullptr);
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    cook::CookedUnionScene sc = CookUnionScene(static_cast<int>(1024u));
    ASSERT_TRUE(sc.place_found);
    const float dt = cook::kH1UnionDt;

    constexpr uint32_t kEnvs = 1024u;
    nk::Model model = coresident::BuildNkUnionModel(sc.tmpl, kEnvs);
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[2] = cook::kH1UnionGravityZ;
    cfg.vel_iters = 64;
    cfg.pos_iters = 0;
    cfg.fold_drive_damping = 0;
    nk::World world(std::move(model), kEnvs, dev, backend, cfg);
    ASSERT_TRUE(world.Ready());
    std::printf("[NK-GRASP-TP] batched union N=%u: links/env=%u slots=%u "
                "rows/env=%u islands=%u\n", kEnvs, sc.link_count,
                world.GetModel().capacities.max_contacts_per_env,
                world.GetModel().capacities.max_rows_per_env,
                world.GetModel().schedule_island_count);

    // The steady-state RL seam: a constant grip torque uploaded ONCE (the bulk
    // SetActions posture), then pure StepPlanned. Throughput = N*steps/wall;
    // no per-step host PD (the aggregate-throughput question, not closed-loop).
    const uint32_t L = sc.link_count;
    std::vector<float> torque(static_cast<size_t>(L) * kEnvs, 0.0f);
    for (uint32_t e = 0; e < kEnvs; ++e)
        for (uint32_t l = 0; l < L && l < sc.tmpl.grip_torque.size(); ++l)
            torque[static_cast<size_t>(e) * L + l] = sc.tmpl.grip_torque[l];
    ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::DriveTarget,
                                            torque.data(),
                                            torque.size() * sizeof(float)));

    constexpr uint32_t kWarm = 20u;
    for (uint32_t s = 0; s < kWarm; ++s) ASSERT_EQ(world.StepPlanned(), nphi::Status::Ok);
    nphi::BackendSynchronize(backend);

    constexpr uint32_t kSteps = 200u;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0; s < kSteps; ++s) ASSERT_EQ(world.StepPlanned(), nphi::Status::Ok);
    nphi::BackendSynchronize(backend);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_s =
        std::chrono::duration<double>(t1 - t0).count();
    const double eps = static_cast<double>(kEnvs) * kSteps / wall_s;
    std::printf("[NK-GRASP-TP RESULT] N=%u x %u steps in %.3f s -> "
                "%.1f env-steps/s (gate >= 11000; margin %.2fx)\n",
                kEnvs, kSteps, wall_s, eps, eps / 11000.0);

    // Liveness: the batch did not NaN-collapse.
    {
        std::vector<nuka::math::Transform> base(kEnvs);
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BasePose,
                                                  base.data(),
                                                  base.size() * sizeof(base[0])));
        EXPECT_TRUE(std::isfinite(base[0].position.z));
    }

    EXPECT_GE(eps, 11000.0) << "the plan M5 grasp throughput red line is BROKEN";
}
