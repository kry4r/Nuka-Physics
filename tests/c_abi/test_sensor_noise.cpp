#include "nuka/nuka.h"
#include "nuka/nuka_noise.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

std::string ScenePath() {
    return SourcePath("examples/scenes/go2_stand.usda").string();
}

bool SceneAvailable() {
    return std::filesystem::exists(SourcePath("examples/scenes/go2_stand.usda"));
}

struct DeviceGuard {
    nuka_device_handle handle = nullptr;
    DeviceGuard() {
        nuka_device_desc_t desc{};
        desc.gpu_index = 0u;
        desc.cuda_stream = nullptr;
        desc.backend_selection_layer_enabled = 1u;
        EXPECT_EQ(nuka_device_create(&desc, &handle), NUKA_RESULT_OK);
    }
    ~DeviceGuard() {
        if (handle != nullptr) nuka_device_destroy(handle);
    }
};

struct WorldGuard {
    nuka_world_handle handle = nullptr;
    ~WorldGuard() {
        if (handle != nullptr) nuka_world_destroy(handle);
    }
};

nuka_result_t CreateWorld(nuka_device_handle device, uint32_t env_count,
                          nuka_world_handle* out) {
    const std::string scene = ScenePath();
    nuka_world_desc_t desc{};
    desc.scene_path = scene.c_str();
    desc.env_count = env_count;
    desc.fixed_dt = 1.0f / 240.0f;
    return nuka_world_create_from_scene(device, &desc, out);
}

std::vector<float> DownloadFloatField(nuka_world_handle world,
                                      nuka_state_field_t field, bool observation = false) {
    nuka_buffer_view_t view{};
    EXPECT_EQ(observation ? nuka_world_get_observation_view(world, field, &view) :
              nuka_world_get_buffer_view(world, field, &view), NUKA_RESULT_OK);
    std::vector<float> out;
    if (view.device_ptr == nullptr || view.element_count == 0u) return out;
    const size_t floats_per_element = view.element_stride_bytes / sizeof(float);
    out.resize(view.element_count * floats_per_element);
    EXPECT_EQ(cudaMemcpy(out.data(), view.device_ptr, out.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return out;
}

constexpr nuka_state_field_t kField = NUKA_FIELD_JOINT_VELOCITY;

}  // namespace

// 1. NONE no-op: apply on an unregistered field leaves the buffer byte-unchanged.
TEST(SensorNoiseCAbi, NoneIsByteNoOp) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand.usda missing";
    DeviceGuard dev;
    WorldGuard w;
    ASSERT_EQ(CreateWorld(dev.handle, 4u, &w.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step(w.handle), NUKA_RESULT_OK);

    const std::vector<float> before = DownloadFloatField(w.handle, kField);
    ASSERT_FALSE(before.empty());

    // No noise registered -> apply is OK and a byte no-op.
    EXPECT_EQ(nuka_world_apply_sensor_noise(w.handle, kField), NUKA_RESULT_OK);

    const std::vector<float> after = DownloadFloatField(w.handle, kField, true);
    EXPECT_EQ(before, DownloadFloatField(w.handle, kField));
    ASSERT_EQ(before.size(), after.size());
    EXPECT_EQ(std::memcmp(before.data(), after.data(),
                          before.size() * sizeof(float)),
              0)
        << "NONE apply must leave the buffer byte-identical";
}

// 2. Apply changes the buffer when Gaussian noise is registered.
TEST(SensorNoiseCAbi, GaussianApplyChangesBuffer) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand.usda missing";
    DeviceGuard dev;
    WorldGuard w;
    ASSERT_EQ(CreateWorld(dev.handle, 4u, &w.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step(w.handle), NUKA_RESULT_OK);

    const std::vector<float> before = DownloadFloatField(w.handle, kField);
    ASSERT_FALSE(before.empty());

    nuka_sensor_noise_desc_t desc{};
    desc.kind = NUKA_NOISE_GAUSSIAN;
    desc.param1 = 0.0f;     // mean
    desc.param2 = 0.05f;    // stddev
    desc.seed = 12345u;
    EXPECT_EQ(nuka_world_set_sensor_noise(w.handle, kField, &desc),
              NUKA_RESULT_OK);
    EXPECT_EQ(nuka_world_apply_sensor_noise(w.handle, kField), NUKA_RESULT_OK);

    const std::vector<float> after = DownloadFloatField(w.handle, kField, true);
    EXPECT_EQ(before, DownloadFloatField(w.handle, kField));
    ASSERT_EQ(before.size(), after.size());
    EXPECT_NE(std::memcmp(before.data(), after.data(),
                          before.size() * sizeof(float)),
              0)
        << "Gaussian apply must perturb the buffer";
    // Perturbation magnitude is ~stddev -- not wild.
    double max_abs = 0.0;
    for (size_t i = 0; i < before.size(); ++i) {
        const double d = std::abs(static_cast<double>(after[i]) -
                                  static_cast<double>(before[i]));
        if (d > max_abs) max_abs = d;
    }
    EXPECT_GT(max_abs, 0.0);
    EXPECT_LT(max_abs, 1.0) << "noise should be bounded ~ a few stddev";
}

// 3. D1: same registered (seed, fresh seq) on two fresh worlds -> identical delta.
TEST(SensorNoiseCAbi, DeterminismTwoWorldsBitExact) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand.usda missing";
    DeviceGuard dev;

    auto run = [&]() -> std::vector<float> {
        WorldGuard w;
        EXPECT_EQ(CreateWorld(dev.handle, 4u, &w.handle), NUKA_RESULT_OK);
        EXPECT_EQ(nuka_world_step(w.handle), NUKA_RESULT_OK);
        const std::vector<float> before = DownloadFloatField(w.handle, kField);
        nuka_sensor_noise_desc_t desc{};
        desc.kind = NUKA_NOISE_GAUSSIAN;
        desc.param1 = 0.0f;
        desc.param2 = 0.05f;
        desc.seed = 0xabcdef01u;
        EXPECT_EQ(nuka_world_set_sensor_noise(w.handle, kField, &desc),
                  NUKA_RESULT_OK);
        EXPECT_EQ(nuka_world_apply_sensor_noise(w.handle, kField),
                  NUKA_RESULT_OK);
        const std::vector<float> after = DownloadFloatField(w.handle, kField, true);
        EXPECT_EQ(before, DownloadFloatField(w.handle, kField));
        std::vector<float> delta(after.size());
        for (size_t i = 0; i < after.size(); ++i) delta[i] = after[i] - before[i];
        return delta;
    };

    const std::vector<float> a = run();
    const std::vector<float> b = run();
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0)
        << "noise delta must be two-run byte-exact through the C-ABI (D1)";
}

// 4. Sequence advances: two successive applies give different increments.
TEST(SensorNoiseCAbi, SequenceAdvancesAcrossApplies) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand.usda missing";
    DeviceGuard dev;
    WorldGuard w;
    ASSERT_EQ(CreateWorld(dev.handle, 4u, &w.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step(w.handle), NUKA_RESULT_OK);

    nuka_sensor_noise_desc_t desc{};
    desc.kind = NUKA_NOISE_GAUSSIAN;
    desc.param1 = 0.0f;
    desc.param2 = 0.05f;
    desc.seed = 777u;
    EXPECT_EQ(nuka_world_set_sensor_noise(w.handle, kField, &desc),
              NUKA_RESULT_OK);

    const std::vector<float> s0 = DownloadFloatField(w.handle, kField);
    EXPECT_EQ(nuka_world_apply_sensor_noise(w.handle, kField), NUKA_RESULT_OK);
    const std::vector<float> s1 = DownloadFloatField(w.handle, kField, true);
    EXPECT_EQ(nuka_world_apply_sensor_noise(w.handle, kField), NUKA_RESULT_OK);
    const std::vector<float> s2 = DownloadFloatField(w.handle, kField, true);

    ASSERT_EQ(s0.size(), s1.size());
    ASSERT_EQ(s1.size(), s2.size());
    std::vector<float> d1(s0.size()), d2(s0.size());
    for (size_t i = 0; i < s0.size(); ++i) {
        d1[i] = s1[i] - s0[i];  // seq 0 noise
        d2[i] = s2[i] - s0[i];  // seq 1 noise
    }
    EXPECT_NE(std::memcmp(d1.data(), d2.data(), d1.size() * sizeof(float)), 0)
        << "successive applies must advance the sequence (independent noise)";
    nuka_checkpoint_handle checkpoint = nullptr;
    ASSERT_EQ(nuka_world_checkpoint_capture(w.handle, &checkpoint), NUKA_RESULT_OK);
    nuka_buffer_view_t view{};
    ASSERT_EQ(nuka_world_get_observation_view(w.handle, kField, &view), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_apply_sensor_noise(w.handle, kField), NUKA_RESULT_OK);
    const auto next = DownloadFloatField(w.handle, kField, true);
    ASSERT_EQ(nuka_world_checkpoint_restore(w.handle, checkpoint), NUKA_RESULT_OK);
    EXPECT_EQ(DownloadFloatField(w.handle, kField, true), s2);
    ASSERT_EQ(nuka_world_apply_sensor_noise(w.handle, kField), NUKA_RESULT_OK);
    EXPECT_EQ(DownloadFloatField(w.handle, kField, true), next);
    nuka_checkpoint_destroy(checkpoint);
    const uint32_t selected[] = {1u, 1u};
    ASSERT_EQ(nuka_world_reset_envs(w.handle, selected, 2u), NUKA_RESULT_OK);
    const auto reset = DownloadFloatField(w.handle, kField, true);
    const size_t width = reset.size() / 4u;
    for (size_t i = 0u; i < reset.size(); ++i)
        EXPECT_EQ(reset[i], i / width == 1u ? 0.0f : next[i]);
    nuka_observation_stamp_t stamp{};
    ASSERT_EQ(nuka_world_get_observation_stamp(w.handle, kField, 1u, &stamp), NUKA_RESULT_OK);
    EXPECT_EQ(stamp.sequence, 0u);
    EXPECT_EQ(stamp.valid, 0u);
    ASSERT_EQ(nuka_world_get_observation_stamp(w.handle, kField, 0u, &stamp), NUKA_RESULT_OK);
    EXPECT_EQ(stamp.sequence, 3u);
    EXPECT_EQ(stamp.valid, 1u);
    nuka_buffer_view_t reset_view{};
    ASSERT_EQ(nuka_world_get_observation_view(w.handle, kField, &reset_view), NUKA_RESULT_OK);
    EXPECT_EQ(view.device_ptr, reset_view.device_ptr);

}

// 5. Clearing restores the no-op.
TEST(SensorNoiseCAbi, ClearRestoresNoOp) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand.usda missing";
    DeviceGuard dev;
    WorldGuard w;
    ASSERT_EQ(CreateWorld(dev.handle, 4u, &w.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step(w.handle), NUKA_RESULT_OK);

    nuka_sensor_noise_desc_t desc{};
    desc.kind = NUKA_NOISE_GAUSSIAN;
    desc.param2 = 0.05f;
    desc.seed = 1u;
    EXPECT_EQ(nuka_world_set_sensor_noise(w.handle, kField, &desc),
              NUKA_RESULT_OK);
    // Clear via NULL desc.
    EXPECT_EQ(nuka_world_set_sensor_noise(w.handle, kField, nullptr),
              NUKA_RESULT_OK);

    const std::vector<float> before = DownloadFloatField(w.handle, kField);
    EXPECT_EQ(nuka_world_apply_sensor_noise(w.handle, kField), NUKA_RESULT_OK);
    const std::vector<float> after = DownloadFloatField(w.handle, kField, true);
    EXPECT_EQ(before, DownloadFloatField(w.handle, kField));
    ASSERT_EQ(before.size(), after.size());
    EXPECT_EQ(std::memcmp(before.data(), after.data(),
                          before.size() * sizeof(float)),
              0)
        << "cleared field must be a byte no-op again";
}

// 6. Guards: out-of-range field, null handle, unknown kind.
TEST(SensorNoiseCAbi, ArgumentGuards) {
    DeviceGuard dev;
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand.usda missing";
    WorldGuard w;
    ASSERT_EQ(CreateWorld(dev.handle, 4u, &w.handle), NUKA_RESULT_OK);

    nuka_sensor_noise_desc_t desc{};
    desc.kind = NUKA_NOISE_GAUSSIAN;
    desc.param2 = 0.01f;

    // Unknown field IDs are rejected before allocating observation storage.
    const nuka_state_field_t bad_field = static_cast<nuka_state_field_t>(~0u);
    EXPECT_EQ(nuka_world_set_sensor_noise(w.handle, bad_field, &desc),
              NUKA_RESULT_INVALID_ARG);
    EXPECT_EQ(nuka_world_apply_sensor_noise(w.handle, bad_field),
              NUKA_RESULT_INVALID_ARG);

    // Null world handle.
    EXPECT_EQ(nuka_world_set_sensor_noise(nullptr, kField, &desc),
              NUKA_RESULT_NULL_HANDLE);
    EXPECT_EQ(nuka_world_apply_sensor_noise(nullptr, kField),
              NUKA_RESULT_NULL_HANDLE);

    for (float stddev : {-0.1f, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        desc.param2 = stddev;
        EXPECT_EQ(nuka_world_set_sensor_noise(w.handle, kField, &desc), NUKA_RESULT_INVALID_ARG);
    }
    EXPECT_EQ(nuka_world_sample_observation(w.handle, kField, 0.0, 25.0f), NUKA_RESULT_INVALID_ARG);

    // Unknown kind.
    nuka_sensor_noise_desc_t bad_kind{};
    bad_kind.kind = static_cast<nuka_noise_kind_t>(99);
    EXPECT_EQ(nuka_world_set_sensor_noise(w.handle, kField, &bad_kind),
              NUKA_RESULT_INVALID_ARG);
}

// 7. Non-float-stride field (link pose, 7-float Transform) -> NOT_SUPPORTED.
TEST(SensorNoiseCAbi, NonFloatStrideFieldRejected) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand.usda missing";
    DeviceGuard dev;
    WorldGuard w;
    ASSERT_EQ(CreateWorld(dev.handle, 4u, &w.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step(w.handle), NUKA_RESULT_OK);

    nuka_sensor_noise_desc_t desc{};
    desc.kind = NUKA_NOISE_GAUSSIAN;
    desc.param2 = 0.01f;
    desc.seed = 5u;
    for (auto field : {NUKA_FIELD_CONTACT_LINK, NUKA_FIELD_ENV_STATUS}) {
        EXPECT_EQ(nuka_world_set_sensor_noise(w.handle, field, &desc), NUKA_RESULT_NOT_SUPPORTED);
        EXPECT_EQ(nuka_world_apply_sensor_noise(w.handle, field), NUKA_RESULT_NOT_SUPPORTED);
    }

    EXPECT_EQ(nuka_world_set_sensor_noise(w.handle,
                                          NUKA_FIELD_ARTICULATION_LINK_POSE,
                                          &desc),
              NUKA_RESULT_NOT_SUPPORTED);
    EXPECT_EQ(nuka_world_apply_sensor_noise(w.handle,
                                            NUKA_FIELD_ARTICULATION_LINK_POSE),
              NUKA_RESULT_NOT_SUPPORTED);
}

TEST(SensorNoiseCAbi, CalibratedMeasurementAndResponseUsePhysicalUnits) {
    DeviceGuard dev;
    WorldGuard w;
    ASSERT_EQ(CreateWorld(dev.handle, 4u, &w.handle), NUKA_RESULT_OK);
    auto truth = DownloadFloatField(w.handle, kField);
    std::fill(truth.begin(), truth.end(), 1.25f);
    ASSERT_EQ(nuka_world_upload_field(w.handle, kField, truth.data(), truth.size() * sizeof(float), 0u),
              NUKA_RESULT_OK);
    nuka_sensor_error_desc_t error{};
    error.struct_size = sizeof(error);
    error.bias = 0.05f;
    error.scale_error = 0.1f;
    error.quantization = 0.1f;
    error.temperature_coefficient = 0.01f;
    error.reference_temperature = 25.0f;
    error.saturation_enabled = 1u;
    error.minimum = -1.4f;
    error.maximum = 1.4f;
    ASSERT_EQ(nuka_world_set_sensor_error(w.handle, kField, &error), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_sample_observation(w.handle, kField, 0.02, 35.0f), NUKA_RESULT_OK);
    const auto measured = DownloadFloatField(w.handle, kField, true);
    for (float value : measured) EXPECT_FLOAT_EQ(value, 1.4f);
    EXPECT_EQ(DownloadFloatField(w.handle, kField), truth);
    error = {};
    error.struct_size = sizeof(error);
    error.response_time = 0.1f;
    ASSERT_EQ(nuka_world_set_sensor_error(w.handle, kField, &error), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_sample_observation(w.handle, kField, 0.02, 25.0f), NUKA_RESULT_OK);
    std::fill(truth.begin(), truth.end(), 2.25f);
    ASSERT_EQ(nuka_world_upload_field(w.handle, kField, truth.data(), truth.size() * sizeof(float), 0u),
              NUKA_RESULT_OK);
    for (uint32_t sample = 1u; sample <= 5u; ++sample) {
        ASSERT_EQ(nuka_world_sample_observation(w.handle, kField, 0.02, 25.0f), NUKA_RESULT_OK);
        const auto response = DownloadFloatField(w.handle, kField, true);
        for (float value : response)
            EXPECT_NEAR(value, 2.25f - std::exp(-0.2f * static_cast<float>(sample)), 2.0e-6f);
    }
    error.correlated_bias_stddev = 1.0f;
    EXPECT_EQ(nuka_world_set_sensor_error(w.handle, kField, &error), NUKA_RESULT_INVALID_ARG);
    error.correlation_time = 1.0f;
    error.quantization = -0.1f;
    EXPECT_EQ(nuka_world_set_sensor_error(w.handle, kField, &error), NUKA_RESULT_INVALID_ARG);
}
