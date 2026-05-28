#include "nuka/nuka.h"
#include "oracle/golden_trajectory.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
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

std::vector<float> DownloadJointPositions(nuka_world_handle world) {
    nuka_buffer_view_t view{};
    EXPECT_EQ(nuka_world_get_buffer_view(world, NUKA_FIELD_JOINT_POSITION, &view),
              NUKA_RESULT_OK);
    EXPECT_NE(view.device_ptr, nullptr);
    std::vector<float> values(view.element_count);
    if (!values.empty()) {
        EXPECT_EQ(cudaMemcpy(values.data(),
                             view.device_ptr,
                             values.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
    return values;
}

std::vector<float> RunGo2StandTrajectory(uint32_t step_count) {
    const auto scene = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene)) {
        throw std::runtime_error("Go2 stand scene is not available");
    }

    nuka_device_desc_t device_desc{};
    device_desc.gpu_index = 0u;
    device_desc.cuda_stream = nullptr;
    device_desc.backend_selection_layer_enabled = 1u;
    nuka_device_handle device = nullptr;
    if (nuka_device_create(&device_desc, &device) != NUKA_RESULT_OK) {
        throw std::runtime_error("nuka_device_create failed");
    }

    nuka_world_desc_t world_desc{};
    const std::string scene_string = scene.string();
    world_desc.scene_path = scene_string.c_str();
    world_desc.env_count = 1u;
    world_desc.fixed_dt = 1.0f / 240.0f;

    nuka_world_handle world = nullptr;
    if (nuka_world_create_from_scene(device, &world_desc, &world) != NUKA_RESULT_OK) {
        nuka_device_destroy(device);
        throw std::runtime_error("nuka_world_create_from_scene failed");
    }

    std::vector<float> trajectory;
    for (uint32_t step = 0; step < step_count; ++step) {
        if (nuka_world_step(world) != NUKA_RESULT_OK) {
            nuka_world_destroy(world);
            nuka_device_destroy(device);
            throw std::runtime_error("nuka_world_step failed");
        }
        auto q = DownloadJointPositions(world);
        if (trajectory.empty()) {
            trajectory.reserve(static_cast<size_t>(step_count) * q.size());
        }
        trajectory.insert(trajectory.end(), q.begin(), q.end());
    }

    nuka_world_destroy(world);
    nuka_device_destroy(device);
    return trajectory;
}

} // namespace

TEST(Go2Stand, CAbiStepsFiveSecondsAndIsDeterministic) {
    const auto scene = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }

    nuka_device_desc_t device_desc{};
    device_desc.gpu_index = 0u;
    device_desc.cuda_stream = nullptr;
    device_desc.backend_selection_layer_enabled = 1u;
    nuka_device_handle device = nullptr;
    ASSERT_EQ(nuka_device_create(&device_desc, &device), NUKA_RESULT_OK);

    nuka_world_desc_t world_desc{};
    const std::string scene_string = scene.string();
    world_desc.scene_path = scene_string.c_str();
    world_desc.env_count = 1u;
    world_desc.fixed_dt = 1.0f / 240.0f;

    nuka_world_handle first_world = nullptr;
    nuka_world_handle second_world = nullptr;
    ASSERT_EQ(nuka_world_create_from_scene(device, &world_desc, &first_world),
              NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_create_from_scene(device, &world_desc, &second_world),
              NUKA_RESULT_OK);

    constexpr uint32_t kFiveSecondsAt240Hz = 1200u;
    ASSERT_EQ(nuka_world_step_n(first_world, kFiveSecondsAt240Hz), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step_n(second_world, kFiveSecondsAt240Hz), NUKA_RESULT_OK);

    const std::vector<float> first = DownloadJointPositions(first_world);
    const std::vector<float> second = DownloadJointPositions(second_world);
    ASSERT_EQ(first.size(), second.size());
    ASSERT_GT(first.size(), 12u);
    for (size_t index = 0; index < first.size(); ++index) {
        EXPECT_EQ(first[index], second[index]) << "joint " << index;
    }

    nuka_world_destroy(second_world);
    nuka_world_destroy(first_world);
    nuka_device_destroy(device);
}

TEST(Go2Stand, OwnerGoldenTrajectoryMatchesWithinTolerance) {
    const auto golden_path = GoldenPath("go2_stand_5s.bin");
    if (!std::filesystem::exists(golden_path)) {
        GTEST_SKIP() << "v0.1 Go2 stand golden trajectory is owner-protected and not present";
    }

    const auto golden = nuka::tests::oracle::LoadGoldenTrajectory(golden_path);
    ASSERT_EQ(golden.kind, nuka::tests::oracle::GoldenKind::JointTrajectory);
    ASSERT_EQ(golden.sample_count, 1200u);
    ASSERT_GT(golden.qpos_count, 12u);

    const std::vector<float> q = RunGo2StandTrajectory(golden.sample_count);
    ASSERT_EQ(q.size(), static_cast<size_t>(golden.sample_count) * golden.qpos_count);

    float max_abs = 0.0f;
    for (uint32_t step = 0; step < golden.sample_count; ++step) {
        const float* record = golden.Record(step);
        const size_t offset = static_cast<size_t>(step) * golden.qpos_count;
        for (uint32_t joint = 0; joint < golden.qpos_count; ++joint) {
            max_abs = std::max(max_abs, std::abs(q[offset + joint] - record[joint]));
        }
    }
    EXPECT_LE(max_abs, 1.0e-4f)
        << "Go2 stand joint trajectory differs from MJX golden";
}
