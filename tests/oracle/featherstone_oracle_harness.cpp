#include "oracle/golden_trajectory.hpp"

#include <gtest/gtest.h>

#include <filesystem>

namespace {

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

} // namespace

TEST(FeatherstoneOracle, GoldenFilesAreOwnerProvided) {
    const auto go2 = SourcePath("tests/oracle/golden/featherstone_go2_random_sample.bin");
    const auto h1 = SourcePath("tests/oracle/golden/featherstone_h1_random_sample.bin");
    const auto stand = SourcePath("tests/oracle/golden/go2_stand_5s.bin");
    if (!std::filesystem::exists(go2) ||
        !std::filesystem::exists(h1) ||
        !std::filesystem::exists(stand)) {
        GTEST_SKIP() << "v0.1 golden files are owner-protected and not present";
    }

    for (const auto& path : {go2, h1, stand}) {
        const auto golden = nuka::tests::oracle::LoadGoldenTrajectory(path);
        EXPECT_GT(golden.sample_count, 0u) << path;
        EXPECT_GT(golden.qpos_count, 0u) << path;
        EXPECT_GT(golden.qvel_count, 0u) << path;
        EXPECT_GT(golden.qacc_count, 0u) << path;
        EXPECT_EQ(golden.payload.size(),
                  static_cast<size_t>(golden.sample_count) * golden.RecordFloatCount())
            << path;
    }
}

TEST(FeatherstoneOracle, RandomSampleGoldenShapeMatchesV1Contract) {
    const auto go2 = SourcePath("tests/oracle/golden/featherstone_go2_random_sample.bin");
    const auto h1 = SourcePath("tests/oracle/golden/featherstone_h1_random_sample.bin");
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
