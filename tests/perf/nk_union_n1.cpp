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
//
// Also prints the LEGACY BatchedUnifiedWorld wall on the SAME scene/drive for
// the before/after record (not a gate — the legacy path is the known
// pathology this milestone retires).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/device_context.hpp"
#include "runtime/coresident/batched_unified_world.hpp"
#include "runtime/coresident/h1_union_nk_model.hpp"
#include "runtime/coresident/h1_union_scene_factory.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace coresident = nuka::runtime::coresident;

bool AssetsAvailable() {
    return std::filesystem::exists(coresident::kH1UnionMjcfDefault) &&
           std::filesystem::exists(coresident::kH1UnionCupDefault);
}

}  // namespace

TEST(NkUnionN1, FullContactStepPlannedUnder5msHard) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    // NOTE: the backend deliberately outlives this test body (the World's
    // dtor frees its plan through the backend vtable) — freed at process exit.
    nphi::Device* dev = nphi::InitBestDevice();
    ASSERT_NE(dev, nullptr);
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    coresident::H1UnionScene sc = coresident::BuildH1UnionScene(context);
    ASSERT_TRUE(sc.place_found);
    const float dt = coresident::kH1UnionDt;

    nk::Model model = coresident::BuildNkUnionModel(sc.tmpl, 1u);
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[2] = coresident::kH1UnionGravityZ;
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

    // ---- the legacy baseline (printed, not gated) ---------------------------
    {
        coresident::BatchedUnifiedWorld legacy(context, sc.tmpl, 1u,
                                               coresident::kH1UnionGravityZ, dt);
        constexpr uint32_t kLegacySteps = 50u;
        for (uint32_t s = 0; s < 5u; ++s) legacy.Step();
        const auto l0 = std::chrono::steady_clock::now();
        for (uint32_t s = 0; s < kLegacySteps; ++s) legacy.Step();
        const auto l1 = std::chrono::steady_clock::now();
        const double legacy_ms =
            std::chrono::duration<double, std::milli>(l1 - l0).count() /
            kLegacySteps;
        std::printf("[NK-UNION-N1] legacy BatchedUnifiedWorld baseline: %.3f "
                    "ms/step (same scene; %.1fx vs nk)\n",
                    legacy_ms, legacy_ms / ms_per_step);
    }
}
