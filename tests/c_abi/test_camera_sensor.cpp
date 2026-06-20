// ---------------------------------------------------------------------------
// Device-resident batched camera sensor over the C ABI: attach a camera to an
// N-env world, step, render every env into ONE device AOV tensor, read it back
// zero-copy via nuka_world_get_sensor_view. The in-the-loop obs surface a
// training stack consumes with torch.from_dlpack (no host download).
//
// Gates (all through the public C ABI, no internal headers):
//  (1) attach + render + view round-trip: the color view is a DEVICE pointer with
//      the (env_count*height*width*3) float layout the caller reshapes to
//      (N,H,W,3); finite; non-empty geometry (the robot is visible -- PRIM has
//      hits, not an all-background image).
//  (2) batched correctness: every env renders the SAME articulation at the SAME
//      mount, so the per-env tiles are BYTE-IDENTICAL across the N per-env TLASes
//      (the batched single-launch render is consistent env-to-env).
//  (3) determinism: two renders of the same stepped world are byte-identical.
//  (4) lifecycle: get_sensor_view before any render is NOT_SUPPORTED; attach is
//      re-callable; the view dtypes match (color/depth/normal/albedo float32,
//      prim uint32).
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

// A Base-mounted camera 3 m above each env origin, identity rotation: local -Z
// looks straight DOWN at the robot -> guaranteed geometry per env.
nuka_result_t AttachTopDownCamera(nuka_world_handle world, uint32_t w, uint32_t h) {
    const float local_offset[7] = {0.0f, 0.0f, 3.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    return nuka_world_attach_camera_sensor(world, NUKA_SENSOR_MOUNT_BASE, 0u,
                                           local_offset, 60.0f, w, h);
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
std::vector<uint32_t> DownloadU32(const nuka_buffer_view_t& v) {
    std::vector<uint32_t> out(v.element_count);
    if (!out.empty()) {
        EXPECT_EQ(cudaMemcpy(out.data(), v.device_ptr, out.size() * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
    return out;
}

// True iff `ptr` is a CUDA DEVICE allocation (proves zero-copy: the view aliases
// device memory, no host staging copy).
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
// Gate 1 -- attach + render + view: device-resident, right shape, finite, robot
// visible (non-empty geometry).
// ---------------------------------------------------------------------------
TEST(CameraSensor, AttachRenderViewDeviceResidentNonEmpty) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand scene unavailable";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnv = 16u, kW = 64u, kH = 64u;
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnv, &world.handle), NUKA_RESULT_OK);
    ASSERT_NE(world.handle, nullptr);

    ASSERT_EQ(nuka_world_step_n(world.handle, 10u), NUKA_RESULT_OK);
    ASSERT_EQ(AttachTopDownCamera(world.handle, kW, kH), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);

    // Color view: (N*H*W*3) float32, device-resident.
    nuka_buffer_view_t color{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_COLOR,
                                         &color),
              NUKA_RESULT_OK);
    ASSERT_NE(color.device_ptr, nullptr);
    EXPECT_EQ(color.dtype, 0u);
    EXPECT_EQ(color.element_stride_bytes, sizeof(float));
    const uint64_t pixels = static_cast<uint64_t>(kEnv) * kH * kW;
    EXPECT_EQ(color.element_count, pixels * 3u);
    EXPECT_TRUE(IsDevicePointer(color.device_ptr))
        << "color view is not a CUDA device pointer (zero-copy violated)";

    const std::vector<float> rgb = DownloadFloat(color);
    double sum = 0.0;
    bool finite = true;
    for (float c : rgb) {
        sum += c;
        if (!std::isfinite(c)) finite = false;
    }
    EXPECT_TRUE(finite) << "color tensor has non-finite values";
    EXPECT_GT(sum, 0.0) << "color tensor is all-zero (nothing rendered)";

    // PRIM view: a non-background image means the robot is in frame.
    nuka_buffer_view_t prim{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_PRIM,
                                         &prim),
              NUKA_RESULT_OK);
    EXPECT_EQ(prim.dtype, 1u);
    EXPECT_EQ(prim.element_stride_bytes, sizeof(uint32_t));
    EXPECT_EQ(prim.element_count, pixels);
    EXPECT_TRUE(IsDevicePointer(prim.device_ptr));
    const std::vector<uint32_t> ids = DownloadU32(prim);
    uint64_t hits = 0u;
    for (uint32_t v : ids) {
        if (v != 0xFFFFFFFFu) ++hits;
    }
    EXPECT_GT(hits, 0u) << "all-miss image -- the robot was not visible";
    std::printf("[diag] (1) N=%u %ux%u color sum=%.3f | PRIM hits=%llu / %llu\n",
                kEnv, kW, kH, sum, (unsigned long long)hits,
                (unsigned long long)ids.size());
}

// ---------------------------------------------------------------------------
// Gate 2 -- batched correctness: every env's tile is byte-identical (same
// articulation, same Base-relative mount across the N per-env TLASes).
// ---------------------------------------------------------------------------
TEST(CameraSensor, PerEnvTilesByteIdentical) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand scene unavailable";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnv = 8u, kW = 48u, kH = 48u;
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnv, &world.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step_n(world.handle, 12u), NUKA_RESULT_OK);
    ASSERT_EQ(AttachTopDownCamera(world.handle, kW, kH), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);

    nuka_buffer_view_t color{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_COLOR,
                                         &color),
              NUKA_RESULT_OK);
    const std::vector<float> rgb = DownloadFloat(color);
    const size_t per = static_cast<size_t>(kW) * kH * 3u;
    ASSERT_EQ(rgb.size(), per * kEnv);

    size_t mismatches = 0u;
    for (uint32_t e = 1u; e < kEnv; ++e) {
        for (size_t i = 0u; i < per; ++i) {
            if (rgb[static_cast<size_t>(e) * per + i] != rgb[i]) ++mismatches;
        }
    }
    std::printf("[diag] (2) cross-env tile mismatches = %zu / %zu\n", mismatches,
                per * (kEnv - 1u));
    EXPECT_EQ(mismatches, 0u)
        << "the N per-env TLAS tiles diverge for identical state -- batched render bug";
}

// ---------------------------------------------------------------------------
// Gate 3 -- determinism: two renders of the same stepped world match byte-for-byte.
// ---------------------------------------------------------------------------
TEST(CameraSensor, RenderIsDeterministic) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand scene unavailable";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnv = 8u, kW = 48u, kH = 48u;
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnv, &world.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step_n(world.handle, 12u), NUKA_RESULT_OK);
    ASSERT_EQ(AttachTopDownCamera(world.handle, kW, kH), NUKA_RESULT_OK);

    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);
    nuka_buffer_view_t v1{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_COLOR, &v1),
              NUKA_RESULT_OK);
    const std::vector<float> a = DownloadFloat(v1);

    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);
    nuka_buffer_view_t v2{};
    ASSERT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_COLOR, &v2),
              NUKA_RESULT_OK);
    const std::vector<float> b = DownloadFloat(v2);

    ASSERT_EQ(a.size(), b.size());
    size_t mismatches = 0u;
    for (size_t i = 0u; i < a.size(); ++i) {
        if (a[i] != b[i]) ++mismatches;
    }
    std::printf("[diag] (3) two-render mismatches = %zu / %zu\n", mismatches,
                a.size());
    EXPECT_EQ(mismatches, 0u);
}

// ---------------------------------------------------------------------------
// Gate 4 -- lifecycle: view-before-render rejects; attach is re-callable; every
// channel resolves with the right dtype after a render.
// ---------------------------------------------------------------------------
TEST(CameraSensor, LifecycleAndChannels) {
    if (!SceneAvailable()) GTEST_SKIP() << "go2_stand scene unavailable";
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnv = 4u, kW = 32u, kH = 32u;
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnv, &world.handle), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_step_n(world.handle, 6u), NUKA_RESULT_OK);

    // Render before attach -> NOT_SUPPORTED (no sensor).
    EXPECT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_NOT_SUPPORTED);

    ASSERT_EQ(AttachTopDownCamera(world.handle, kW, kH), NUKA_RESULT_OK);
    // View before any render -> NOT_SUPPORTED (no device tensor yet).
    nuka_buffer_view_t pre{};
    EXPECT_EQ(nuka_world_get_sensor_view(world.handle, NUKA_SENSOR_CHANNEL_COLOR, &pre),
              NUKA_RESULT_NOT_SUPPORTED);

    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);

    // Re-attach at a different resolution (re-callable) then render again.
    ASSERT_EQ(AttachTopDownCamera(world.handle, 40u, 24u), NUKA_RESULT_OK);
    ASSERT_EQ(nuka_world_render_sensors(world.handle), NUKA_RESULT_OK);
    const uint64_t pix2 = static_cast<uint64_t>(kEnv) * 24u * 40u;

    struct ChanExpect {
        nuka_sensor_channel_t ch;
        uint64_t count;
        uint8_t dtype;
        uint32_t stride;
    };
    const ChanExpect chans[5] = {
        {NUKA_SENSOR_CHANNEL_COLOR, pix2 * 3u, 0u, sizeof(float)},
        {NUKA_SENSOR_CHANNEL_DEPTH, pix2, 0u, sizeof(float)},
        {NUKA_SENSOR_CHANNEL_NORMAL, pix2 * 3u, 0u, sizeof(float)},
        {NUKA_SENSOR_CHANNEL_ALBEDO, pix2 * 3u, 0u, sizeof(float)},
        {NUKA_SENSOR_CHANNEL_PRIM, pix2, 1u, sizeof(uint32_t)},
    };
    for (const ChanExpect& c : chans) {
        nuka_buffer_view_t v{};
        ASSERT_EQ(nuka_world_get_sensor_view(world.handle, c.ch, &v), NUKA_RESULT_OK)
            << "channel " << static_cast<int>(c.ch);
        EXPECT_NE(v.device_ptr, nullptr);
        EXPECT_EQ(v.element_count, c.count) << "channel " << static_cast<int>(c.ch);
        EXPECT_EQ(v.dtype, c.dtype) << "channel " << static_cast<int>(c.ch);
        EXPECT_EQ(v.element_stride_bytes, c.stride)
            << "channel " << static_cast<int>(c.ch);
        EXPECT_TRUE(IsDevicePointer(v.device_ptr));
    }
    std::printf("[diag] (4) lifecycle + all 5 channels resolve (re-attach OK)\n");
}
