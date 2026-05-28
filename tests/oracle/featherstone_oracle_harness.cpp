#include "oracle/golden_trajectory.hpp"

#include "import/mjcf_importer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <string_view>
#include <vector>

namespace {

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

std::filesystem::path GoldenPath(const char* filename) {
    if (const char* golden_dir = std::getenv("NUKA_GOLDEN_DIR")) {
        if (golden_dir[0] != '\0') {
            return std::filesystem::path(golden_dir) / filename;
        }
    }
    return SourcePath("tests/oracle/golden") / filename;
}

std::vector<float> ComputeCudaAbaQddot(const std::filesystem::path& model_path,
                                       const float* record,
                                       const nuka::tests::oracle::GoldenTrajectory& golden) {
    const auto scene = nuka::import::LoadMjcf(model_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    auto host_state =
        nuka::runtime::articulation::BuildArticulationHostState(
            world.template_view.articulations,
            world.template_view.body_table);

    if (host_state.q.size() != golden.qpos_count ||
        host_state.qdot.size() != golden.qvel_count ||
        host_state.tau.size() != golden.qvel_count ||
        host_state.qddot.size() != golden.qacc_count) {
        throw std::runtime_error("golden shape does not match cooked articulation");
    }

    const float* qpos = record;
    const float* qvel = qpos + golden.qpos_count;
    const float* tau = qvel + golden.qvel_count;
    std::copy(qpos, qpos + golden.qpos_count, host_state.q.begin());
    std::copy(qvel, qvel + golden.qvel_count, host_state.qdot.begin());
    std::copy(tau, tau + golden.qvel_count, host_state.tau.begin());

    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto device_state =
        nuka::runtime::articulation::UploadArticulationState(context, host_state);
    nuka::runtime::articulation::FeatherstoneAba::ComputeAccelerations(
        context,
        device_state.View(),
        -9.81f);
    context.stream.Synchronize();
    nuka::runtime::articulation::DownloadArticulationState(device_state, &host_state);
    return host_state.qddot;
}

void ExpectRandomSampleGoldenMatchesCuda(const std::filesystem::path& model_path,
                                         const std::filesystem::path& golden_path) {
    const auto golden = nuka::tests::oracle::LoadGoldenTrajectory(golden_path);
    ASSERT_EQ(golden.kind, nuka::tests::oracle::GoldenKind::RandomQacc);
    ASSERT_EQ(golden.qvel_count, golden.qacc_count);
    constexpr uint32_t kMaxSamplesCheckedInTest = 32u;
    const uint32_t sample_count =
        std::min(golden.sample_count, kMaxSamplesCheckedInTest);
    ASSERT_GT(sample_count, 0u);

    float max_abs = 0.0f;
    for (uint32_t sample = 0u; sample < sample_count; ++sample) {
        const float* record = golden.Record(sample);
        const float* qacc = record + golden.qpos_count +
                            golden.qvel_count + golden.qvel_count;
        const auto qddot = ComputeCudaAbaQddot(model_path, record, golden);
        ASSERT_EQ(qddot.size(), golden.qacc_count);
        for (uint32_t dof = 0u; dof < golden.qacc_count; ++dof) {
            max_abs = std::max(max_abs, std::abs(qddot[dof] - qacc[dof]));
        }
    }

    EXPECT_LE(max_abs, 1.0e-3f)
        << "CUDA Featherstone ABA qddot differs from MJX golden";
}

} // namespace

TEST(FeatherstoneOracle, GoldenFilesAreOwnerProvided) {
    const auto go2 = GoldenPath("featherstone_go2_random_sample.bin");
    const auto h1 = GoldenPath("featherstone_h1_random_sample.bin");
    const auto stand = GoldenPath("go2_stand_5s.bin");
    if (!std::filesystem::exists(go2) ||
        !std::filesystem::exists(h1) ||
        !std::filesystem::exists(stand)) {
        GTEST_SKIP() << "v0.1 golden files are owner-protected and not present";
    }

    for (const auto& path : {go2, h1, stand}) {
        const auto golden = nuka::tests::oracle::LoadGoldenTrajectory(path);
        EXPECT_GT(golden.sample_count, 0u) << path;
        EXPECT_GT(golden.qpos_count, 0u) << path;
        if (golden.kind == nuka::tests::oracle::GoldenKind::RandomQacc) {
            EXPECT_GT(golden.qvel_count, 0u) << path;
            EXPECT_GT(golden.qacc_count, 0u) << path;
        }
        EXPECT_EQ(golden.payload.size(),
                  static_cast<size_t>(golden.sample_count) * golden.RecordFloatCount())
            << path;
    }
}

TEST(FeatherstoneOracle, RandomSampleGoldenShapeMatchesV1Contract) {
    const auto go2 = GoldenPath("featherstone_go2_random_sample.bin");
    const auto h1 = GoldenPath("featherstone_h1_random_sample.bin");
    if (!std::filesystem::exists(go2) || !std::filesystem::exists(h1)) {
        GTEST_SKIP() << "v0.1 random-sample golden files are owner-protected and not present";
    }

    for (const auto& path : {go2, h1}) {
        const auto golden = nuka::tests::oracle::LoadGoldenTrajectory(path);
        EXPECT_EQ(golden.kind, nuka::tests::oracle::GoldenKind::RandomQacc) << path;
        EXPECT_EQ(golden.sample_count, 1000u) << path;
        EXPECT_EQ(golden.qvel_count, golden.qacc_count) << path;
    }
}

TEST(FeatherstoneOracle, RandomSampleGoldensMatchCudaAba) {
    const auto go2_model =
        SourcePath(".nuka-assets/mujoco_menagerie/unitree_go2/go2_mjx.xml");
    const auto h1_model =
        SourcePath(".nuka-assets/mujoco_menagerie/unitree_h1/h1.xml");
    const auto go2_golden =
        GoldenPath("featherstone_go2_random_sample.bin");
    const auto h1_golden =
        GoldenPath("featherstone_h1_random_sample.bin");

    if (!std::filesystem::exists(go2_model) || !std::filesystem::exists(h1_model)) {
        GTEST_SKIP() << "MuJoCo Menagerie Go2/H1 assets are not available";
    }
    if (!std::filesystem::exists(go2_golden) || !std::filesystem::exists(h1_golden)) {
        GTEST_SKIP() << "v0.1 random-sample golden files are owner-protected and not present";
    }

    ExpectRandomSampleGoldenMatchesCuda(go2_model, go2_golden);
    ExpectRandomSampleGoldenMatchesCuda(h1_model, h1_golden);
}
