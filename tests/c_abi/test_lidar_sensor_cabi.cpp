// ---------------------------------------------------------------------------
// Device-resident batched LIDAR sensor over the C ABI: attach a lidar to an N-env
// world, step, render every env's (az,el) fan into ONE device RANGE tensor, read it
// back zero-copy via nuka_world_get_sensor_view(RANGE). The lidar rides the SAME RT
// TLAS the cameras use; this is the shipped public path the RL stack would consume
// with torch.from_dlpack (no host download).
//
// Gates (through the public C ABI, no internal headers):
//  (1) attach + render + RANGE view: a DEVICE pointer with the (E*S*az*el) float
//      layout, finite, and the robot in front of the fan (some ray < max_range).
//  (2) dims: nuka_world_get_lidar_dims returns (E,S,az,el) matching the attach.
//  (3) determinism: two renders of the same stepped world are byte-identical.
//  (4) per-env tiles byte-identical (same robot + mount across the N TLASes).
//  (5) lifecycle: RANGE before render -> NOT_SUPPORTED; a zero fan -> INVALID_ARG.
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::filesystem::path SourcePath(const char* rel) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / rel;
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

// A Base-mounted lidar 1.5 m above the env origin, tilted to look DOWN-and-around so
// its fan sweeps the robot below. local frame: az about +Z, el about +X, az=el=0 =>
// +X; a -90deg pitch about Y points local +X at world -Z (straight down). We instead
// keep identity offset and use a wide el band that dips below horizon to catch the
// robot sitting at/under the lidar origin's z.
nuka_result_t AttachLidar(nuka_world_handle world, uint32_t az, uint32_t el) {
    // Lidar at origin (identity offset); a full az ring + an el band from -1.2 rad
    // (down) to +0.2 rad so the downward beams sweep the standing robot.
    const float off[7] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    return nuka_world_attach_lidar_sensor(world, NUKA_SENSOR_MOUNT_BASE, 0u, off, az,
                                          el, -3.14159265f, 3.14159265f, -1.2f, 0.2f,
                                          0.05f, 30.0f);
}

std::vector<float> DownloadFloat(const nuka_buffer_view_t& v) {
    std::vector<float> out(v.element_count);
    if (!out.empty()) {
        EXPECT_EQ(cudaMemcpy(out.data(), v.device_ptr, out.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
    return out;
}

bool IsDevicePointer(const void* ptr) {
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return attr.type == cudaMemoryTypeDevice;
}

}  // namespace

// ---------------------------------------------------------------------------
// Gate 1+2 -- attach + render + RANGE view + dims.
// ---------------------------------------------------------------------------
TEST(LidarSensorCabi, AttachRenderRangeViewAndDims) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand scene unavailable";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnv = 8u, kAz = 64u, kEl = 8u;
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnv, &world.handle), NUKA_RESULT_OK);
    ASSERT_NE(world.handle, nullptr);

    ASSERT_EQ(nuka_world_step_n(world.handle, 10u), NUKA_RESULT_OK);
    ASSERT_EQ(AttachLidar(world.handle, kAz, kEl), NUKA_RESULT_OK);

    // Dims valid right after attach (no render needed).
    uint32_t e = 0u, s = 0u, az = 0u, el = 0u;
    ASSERT_EQ(nuka_world_get_lidar_dims(world.handle, &e, &s, &az, &el),
              NUKA_RESULT_OK);
    EXPECT_EQ(e, kEnv);
    EXPECT_EQ(s, 1u);
    EXPECT_EQ(az, kAz);
    EXPECT_EQ(el, kEl);

    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);

    nuka_buffer_view_t range{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_RANGE,
                                         &range),
              NUKA_RESULT_OK);
    ASSERT_NE(range.device_ptr, nullptr);
    EXPECT_EQ(range.dtype, 0u);
    EXPECT_EQ(range.element_stride_bytes, sizeof(float));
    const uint64_t cells = static_cast<uint64_t>(kEnv) * 1u * kAz * kEl;
    EXPECT_EQ(range.element_count, cells);
    EXPECT_TRUE(IsDevicePointer(range.device_ptr))
        << "range view is not a CUDA device pointer (zero-copy violated)";

    const std::vector<float> r = DownloadFloat(range);
    bool finite = true;
    uint64_t hits = 0u;
    for (float v : r) {
        if (!std::isfinite(v)) finite = false;
        if (v < 30.0f - 1.0e-3f) ++hits;  // < max_range => a real return
    }
    EXPECT_TRUE(finite) << "range tensor has non-finite values";
    EXPECT_GT(hits, 0u) << "all rays returned max_range -- the robot was not in the fan";
    std::printf("[diag] (1) lidar N=%u %ux%u cells=%llu | returns(<max)=%llu\n", kEnv,
                kAz, kEl, (unsigned long long)cells, (unsigned long long)hits);
}

// ---------------------------------------------------------------------------
// Gate 3 -- determinism.
// ---------------------------------------------------------------------------
TEST(LidarSensorCabi, RenderIsDeterministic) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand scene unavailable";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnv = 4u, kAz = 32u, kEl = 4u;
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnv, &world.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step_n(world.handle, 12u), NUKA_RESULT_OK);
    ASSERT_EQ(AttachLidar(world.handle, kAz, kEl), NUKA_RESULT_OK);

    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);
    nuka_buffer_view_t v1{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_RANGE, &v1),
              NUKA_RESULT_OK);
    const std::vector<float> a = DownloadFloat(v1);

    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);
    nuka_buffer_view_t v2{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_RANGE, &v2),
              NUKA_RESULT_OK);
    const std::vector<float> b = DownloadFloat(v2);

    ASSERT_EQ(a.size(), b.size());
    size_t mismatches = 0u;
    for (size_t i = 0u; i < a.size(); ++i) {
        if (a[i] != b[i]) ++mismatches;
    }
    std::printf("[diag] (3) two-render mismatches = %zu / %zu\n", mismatches, a.size());
    EXPECT_EQ(mismatches, 0u);
}

// ---------------------------------------------------------------------------
// Gate 4 -- per-env tiles byte-identical (same robot + Base-relative mount).
// ---------------------------------------------------------------------------
TEST(LidarSensorCabi, PerEnvTilesByteIdentical) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand scene unavailable";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnv = 6u, kAz = 32u, kEl = 4u;
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnv, &world.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step_n(world.handle, 12u), NUKA_RESULT_OK);
    ASSERT_EQ(AttachLidar(world.handle, kAz, kEl), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);

    nuka_buffer_view_t range{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_RANGE,
                                         &range),
              NUKA_RESULT_OK);
    const std::vector<float> r = DownloadFloat(range);
    const size_t per = static_cast<size_t>(kAz) * kEl;
    ASSERT_EQ(r.size(), per * kEnv);

    size_t mismatches = 0u;
    for (uint32_t e = 1u; e < kEnv; ++e) {
        for (size_t i = 0u; i < per; ++i) {
            if (r[static_cast<size_t>(e) * per + i] != r[i]) ++mismatches;
        }
    }
    std::printf("[diag] (4) cross-env tile mismatches = %zu / %zu\n", mismatches,
                per * (kEnv - 1u));
    EXPECT_EQ(mismatches, 0u)
        << "the N per-env TLAS range tiles diverge for identical state";
}

// ---------------------------------------------------------------------------
// Gate 5 -- lifecycle + arg validation.
// ---------------------------------------------------------------------------
TEST(LidarSensorCabi, LifecycleAndArgs) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand scene unavailable";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnv = 4u;
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnv, &world.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step_n(world.handle, 6u), NUKA_RESULT_OK);

    // A zero fan / bad range is rejected before any backend work.
    const float off[7] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_EQ(nuka_world_attach_lidar_sensor(world.handle, NUKA_SENSOR_MOUNT_BASE, 0u,
                                             off, 0u, 4u, -1.0f, 1.0f, 0.0f, 0.0f,
                                             0.0f, 10.0f),
              NUKA_RESULT_INVALID_ARG);
    EXPECT_EQ(nuka_world_attach_lidar_sensor(world.handle, NUKA_SENSOR_MOUNT_BASE, 0u,
                                             off, 8u, 4u, -1.0f, 1.0f, 0.0f, 0.0f,
                                             5.0f, 1.0f),  // max<=min
              NUKA_RESULT_INVALID_ARG);

    // Lidar dims before any attach -> NOT_SUPPORTED.
    EXPECT_EQ(nuka_world_get_lidar_dims(world.handle, nullptr, nullptr, nullptr,
                                        nullptr),
              NUKA_RESULT_NOT_SUPPORTED);

    ASSERT_EQ(AttachLidar(world.handle, 16u, 4u), NUKA_RESULT_OK);

    // RANGE view before any render -> NOT_SUPPORTED (no device tensor yet).
    nuka_buffer_view_t pre{};
    EXPECT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_RANGE, &pre),
              NUKA_RESULT_NOT_SUPPORTED);

    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);
    nuka_buffer_view_t v{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_RANGE, &v),
              NUKA_RESULT_OK);
    EXPECT_NE(v.device_ptr, nullptr);
    EXPECT_EQ(v.element_count, static_cast<uint64_t>(kEnv) * 16u * 4u);
    std::printf("[diag] (5) lifecycle + arg validation OK\n");
}
