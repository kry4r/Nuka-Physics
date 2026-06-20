// ---------------------------------------------------------------------------
// Per-stage GPU profile of the 4096-env Go2 production step. Builds the same
// world as the perf gate, then times each OpCall (BackendDispatch) with CUDA
// events. Prints a ranked per-op ms + % breakdown. Eager dispatch (per-op
// granularity), median over repeated runs. Records-and-skips (never asserts).
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "import/usd_importer.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/op_schema.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

std::filesystem::path SourcePath(const char* rel) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / rel;
}

const char* OpName(nuka::phi::NkOp op) {
    using nuka::phi::NkOp;
    switch (op) {
        case NkOp::ApplyDrives: return "ApplyDrives";
        case NkOp::AbaForward: return "AbaForward";
        case NkOp::IntegrateVelocity: return "IntegrateVelocity";
        case NkOp::FkWorldPoses: return "FkWorldPoses";
        case NkOp::IntegratePosition: return "IntegratePosition";
        case NkOp::CrbaComputeM: return "CrbaComputeM";
        case NkOp::CrbaFactorM: return "CrbaFactorM";
        case NkOp::ApplyImplicitDamping: return "ApplyImplicitDamping";
        case NkOp::BuildAabbs: return "BuildAabbs";
        case NkOp::LbvhBuild: return "LbvhBuild";
        case NkOp::LbvhQueryPairs: return "LbvhQueryPairs";
        case NkOp::NarrowphasePrimitives: return "NarrowphasePrimitives";
        case NkOp::NarrowphaseSdf: return "NarrowphaseSdf";
        case NkOp::NarrowphaseHeightfield: return "NarrowphaseHeightfield";
        case NkOp::ContactTangentBasis: return "ContactTangentBasis";
        case NkOp::AssembleRows: return "AssembleRows";
        case NkOp::SolveRowsBlockIsland: return "SolveRowsBlockIsland";
        case NkOp::ReadoutContactWrench: return "ReadoutContactWrench";
        case NkOp::SyncLinkBodyPose: return "SyncLinkBodyPose";
        default: return "<other>";
    }
}

}  // namespace

TEST(Go2_4096env_Profile, PerStageBreakdown) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) GTEST_SKIP() << "no scene";

    constexpr float kDt = 1.0f / 240.0f;
    constexpr float kGround = 0.31f;
    constexpr uint32_t kEnvCount = 4096u;
    constexpr uint32_t kWarmup = 50u;
    constexpr uint32_t kSteps = 200u;

    nuka::phi::Device* dev = nuka::phi::InitBestDevice();
    ASSERT_NE(dev, nullptr);
    nuka::phi::Backend* backend = nuka::phi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    const auto scene = nuka::import::LoadUsd(scene_path.string());
    auto cooked = nuka::scene::cook::CookToModel(scene, 1);
    cooked.model.ground_height = kGround;
    cooked.model.baumgarte_max_velocity = 3.0f;
    nuka::nk::Pipeline::SolverConfig cfg;
    cfg.dt = kDt;
    cfg.gravity[2] = -9.81f;
    nuka::nk::World world(std::move(cooked.model), kEnvCount, dev, backend, cfg);
    ASSERT_TRUE(world.Ready());

    const auto& calls = world.GetPipeline().Calls();
    const auto& mv = world.ModelViewRef();
    auto& data = world.GetData();
    const auto& dv = world.DataViewRef();

    std::printf("[profile] envs=%u  ops=%zu  bodies/env=%u  links/env=%u  "
                "max_contacts/env=%u\n",
                kEnvCount, calls.size(),
                world.GetModel().capacities.bodies_per_env,
                world.GetModel().capacities.links_per_env,
                world.GetModel().capacities.max_contacts_per_env);

    for (uint32_t s = 0; s < kWarmup; ++s) (void)world.Step();
    nuka::phi::BackendSynchronize(backend);

    const size_t nops = calls.size();
    std::vector<cudaEvent_t> ev(nops + 1);
    for (auto& e : ev) cudaEventCreate(&e);
    std::vector<double> sum_ms(nops, 0.0);

    double total_wall_ms = 0.0;
    for (uint32_t s = 0; s < kSteps; ++s) {
        const auto t0 = std::chrono::steady_clock::now();
        cudaEventRecord(ev[0], nullptr);
        for (size_t i = 0; i < nops; ++i) {
            (void)nuka::phi::BackendDispatch(backend, mv, dv, calls[i]);
            cudaEventRecord(ev[i + 1], nullptr);
        }
        nuka::phi::BackendSynchronize(backend);
        const auto t1 = std::chrono::steady_clock::now();
        total_wall_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        for (size_t i = 0; i < nops; ++i) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev[i], ev[i + 1]);
            sum_ms[i] += ms;
        }
    }

    struct Row { std::string name; double ms; };
    std::vector<Row> rows;
    double total_gpu = 0.0;
    for (size_t i = 0; i < nops; ++i) {
        const double ms = sum_ms[i] / kSteps;
        total_gpu += ms;
        rows.push_back({OpName(calls[i].op), ms});
    }
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.ms > b.ms; });

    const double wall_per_step = total_wall_ms / kSteps;
    std::printf("[profile] wall/step=%.4f ms  sum(GPU events)/step=%.4f ms  "
                "eps=%.0f env-steps/s\n",
                wall_per_step, total_gpu, kEnvCount / (wall_per_step / 1000.0));
    std::printf("[profile] %-26s %12s %8s\n", "stage", "ms/step", "%%");
    for (const auto& r : rows) {
        std::printf("[profile] %-26s %12.5f %7.2f\n", r.name.c_str(), r.ms,
                    100.0 * r.ms / total_gpu);
    }
    std::fflush(stdout);

    for (auto& e : ev) cudaEventDestroy(e);
    (void)data;
    GTEST_SKIP() << "profile recorded";
}
