// ---------------------------------------------------------------------------
// Multiple cameras per env over the C ABI: attach TWO cameras with different mount
// rows to an N-env world, render every (env, sensor) into ONE device AOV tensor
// (E, S, H, W, ch), and prove the multi-camera path is the single-camera path fanned
// out -- the SAME batched RT kernel, just one camera index per (env, sensor) instead
// of one per env.
//
// Gates (all through the public C ABI, no internal headers):
//  (1) dims: after attaching two cameras, get_sensor_dims reports (E, 2, H, W) and
//      the color view is E*2*H*W*3 floats, device-resident.
//  (2) tile == single-camera render: the (env, sensor=s) tile is byte-identical to a
//      standalone world rendering ONLY camera s's mount. Proves the per-camera
//      ray-gen renders each mount exactly as the one-camera path would.
//  (3) cross-env byte-identical: every env's (sensor=s) tile matches env 0's, for
//      both sensors (the scene is env-shared, only the per-env camera fans out).
//
// A co-resident multi-robot scene is not reachable from nuka_world_create_from_scene
// here, so the two cameras ride two different mount rows of ONE robot (a Base mount
// and a base-Link mount, different offsets). The trace kernel is identical for two
// robots' cameras vs two cameras on one robot -- it indexes cameras env-major and the
// env-shared TLAS by env -- so this proves the S>1 fan-out.
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

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

// Camera 0: Base mount, 3 m straight above each env, identity rotation (looks down).
nuka_result_t AttachCam0(nuka_world_handle world, uint32_t w, uint32_t h) {
    const float off[7] = {0.0f, 0.0f, 3.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    return nuka_world_attach_camera_sensor(world, NUKA_SENSOR_MOUNT_BASE, 0u, off,
                                           60.0f, w, h);
}

// Camera 1: base-Link mount (a DIFFERENT mount row -- LinkPose, not BasePose), 4 m
// above with a wider FOV. A genuinely different camera that exercises the second
// mount row through the same kernel.
nuka_result_t AttachCam1(nuka_world_handle world, uint32_t w, uint32_t h) {
    const float off[7] = {0.0f, 0.0f, 4.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    return nuka_world_attach_camera_sensor(world, NUKA_SENSOR_MOUNT_LINK, 0u, off,
                                           75.0f, w, h);
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

// The flat color view of a freshly-stepped world carrying ONLY `which` camera (0 or
// 1): the single-camera reference each two-camera tile must match byte-for-byte.
std::vector<float> SingleCameraColor(nuka_device_handle device, uint32_t env,
                                     uint32_t steps, uint32_t w, uint32_t h,
                                     int which) {
    WorldGuard world;
    EXPECT_EQ(CreateWorld(device, env, &world.handle), NUKA_RESULT_OK);
    EXPECT_EQ(nuka_world_step_n(world.handle, steps), NUKA_RESULT_OK);
    EXPECT_EQ(which == 0 ? AttachCam0(world.handle, w, h)
                         : AttachCam1(world.handle, w, h),
              NUKA_RESULT_OK);
    EXPECT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);
    nuka_buffer_view_t color{};
    EXPECT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_COLOR,
                                         &color),
              NUKA_RESULT_OK);
    return DownloadFloat(color);
}

}  // namespace

// ---------------------------------------------------------------------------
// Gate 1 -- two cameras: dims report (E, 2, H, W); the color tensor is E*2*H*W*3
// floats, device-resident.
// ---------------------------------------------------------------------------
TEST(MultiCameraSensor, TwoCamerasReportSensorDims) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand scene unavailable";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnv = 8u, kW = 48u, kH = 48u;
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnv, &world.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step_n(world.handle, 10u), NUKA_RESULT_OK);
    ASSERT_EQ(AttachCam0(world.handle, kW, kH), NUKA_RESULT_OK);
    ASSERT_EQ(AttachCam1(world.handle, kW, kH), NUKA_RESULT_OK);

    uint32_t e = 0u, s = 0u, hh = 0u, ww = 0u, ch = 0u;
    ASSERT_EQ(nuka_world_get_sensor_dims(world.handle, &e, &s, &hh, &ww, &ch),
              NUKA_RESULT_OK);
    EXPECT_EQ(e, kEnv);
    EXPECT_EQ(s, 2u) << "two attaches at one size must accumulate to S=2";
    EXPECT_EQ(hh, kH);
    EXPECT_EQ(ww, kW);
    EXPECT_EQ(ch, 3u);

    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);
    nuka_buffer_view_t color{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_COLOR,
                                         &color),
              NUKA_RESULT_OK);
    const uint64_t pixels = static_cast<uint64_t>(kEnv) * 2u * kH * kW;
    EXPECT_EQ(color.element_count, pixels * 3u);
    EXPECT_TRUE(IsDevicePointer(color.device_ptr));
    std::printf("[diag] (1) dims (E=%u, S=%u, %ux%u, ch=%u) color=%llu floats\n", e,
                s, ww, hh, ch, (unsigned long long)color.element_count);
}

// ---------------------------------------------------------------------------
// Gate 2 -- each (env, sensor) tile is byte-identical to a standalone single-camera
// render of that sensor's mount. The S>1 fan-out renders each camera exactly as the
// S==1 path does.
// ---------------------------------------------------------------------------
TEST(MultiCameraSensor, PerSensorTileMatchesSingleCamera) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand scene unavailable";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnv = 6u, kW = 40u, kH = 40u, kSteps = 12u;
    const size_t per = static_cast<size_t>(kW) * kH * 3u;  // floats per tile

    // The two-camera world (S=2), color tensor laid out (E, S, H, W, 3) env-major.
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnv, &world.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step_n(world.handle, kSteps), NUKA_RESULT_OK);
    ASSERT_EQ(AttachCam0(world.handle, kW, kH), NUKA_RESULT_OK);
    ASSERT_EQ(AttachCam1(world.handle, kW, kH), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);
    nuka_buffer_view_t color{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_COLOR,
                                         &color),
              NUKA_RESULT_OK);
    const std::vector<float> rgb = DownloadFloat(color);
    ASSERT_EQ(rgb.size(), per * kEnv * 2u);

    // Single-camera references (same scene, same steps, same size).
    const std::vector<float> ref0 =
        SingleCameraColor(device.handle, kEnv, kSteps, kW, kH, 0);
    const std::vector<float> ref1 =
        SingleCameraColor(device.handle, kEnv, kSteps, kW, kH, 1);
    ASSERT_EQ(ref0.size(), per * kEnv);
    ASSERT_EQ(ref1.size(), per * kEnv);

    // The (env, sensor=s) tile sits at ((env*S + s) * per). Compare to ref_s[env].
    size_t mism0 = 0u, mism1 = 0u;
    for (uint32_t e = 0u; e < kEnv; ++e) {
        const size_t t0 = (static_cast<size_t>(e) * 2u + 0u) * per;
        const size_t t1 = (static_cast<size_t>(e) * 2u + 1u) * per;
        const size_t r = static_cast<size_t>(e) * per;
        for (size_t i = 0u; i < per; ++i) {
            if (rgb[t0 + i] != ref0[r + i]) ++mism0;
            if (rgb[t1 + i] != ref1[r + i]) ++mism1;
        }
    }
    std::printf("[diag] (2) sensor0 vs single-cam mismatches=%zu | sensor1=%zu / %zu\n",
                mism0, mism1, per * kEnv);
    EXPECT_EQ(mism0, 0u) << "sensor 0 tile differs from its single-camera render";
    EXPECT_EQ(mism1, 0u) << "sensor 1 tile differs from its single-camera render";

    // The two cameras genuinely differ (distinct mounts) -> the two reference images
    // are not the same (guards against a trivially-passing identical-camera test).
    size_t cam_diff = 0u;
    for (size_t i = 0u; i < per; ++i)
        if (ref0[i] != ref1[i]) ++cam_diff;
    EXPECT_GT(cam_diff, 0u) << "the two mounts produced identical images (not a real "
                               "two-camera test)";
}

// ---------------------------------------------------------------------------
// Gate 3 -- cross-env byte-identical for each sensor: env e's (sensor=s) tile equals
// env 0's (the scene is env-shared, the cameras are env-replicated).
// ---------------------------------------------------------------------------
TEST(MultiCameraSensor, CrossEnvTilesByteIdenticalPerSensor) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand scene unavailable";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnv = 8u, kW = 36u, kH = 36u;
    const size_t per = static_cast<size_t>(kW) * kH * 3u;

    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnv, &world.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step_n(world.handle, 9u), NUKA_RESULT_OK);
    ASSERT_EQ(AttachCam0(world.handle, kW, kH), NUKA_RESULT_OK);
    ASSERT_EQ(AttachCam1(world.handle, kW, kH), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);

    nuka_buffer_view_t color{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_COLOR,
                                         &color),
              NUKA_RESULT_OK);
    const std::vector<float> rgb = DownloadFloat(color);
    ASSERT_EQ(rgb.size(), per * kEnv * 2u);

    size_t mism = 0u;
    for (uint32_t s = 0u; s < 2u; ++s) {
        const size_t base = (0u * 2u + s) * per;  // env 0, sensor s
        for (uint32_t e = 1u; e < kEnv; ++e) {
            const size_t t = (static_cast<size_t>(e) * 2u + s) * per;
            for (size_t i = 0u; i < per; ++i)
                if (rgb[t + i] != rgb[base + i]) ++mism;
        }
    }
    std::printf("[diag] (3) cross-env per-sensor mismatches = %zu / %zu\n", mism,
                per * 2u * (kEnv - 1u));
    EXPECT_EQ(mism, 0u) << "per-env tiles diverge for identical state -- batched bug";
}
