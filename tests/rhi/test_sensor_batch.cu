// ---------------------------------------------------------------------------
// Offline RHI sensor-batch gate: the host render layer (the SensorRenderer over
// the PHI-unified batched sensor render) must add ZERO numeric change vs driving
// the rt-layer batched entry (rt::RenderSensorsMounted) directly. Builds ONE
// N-env replicated sensor scene BOTH ways with a Sensor profile and asserts the
// device AOV tensors are BYTE-IDENTICAL across all channels (download for the
// memcmp only) -- proving the RHI layer is pure orchestration over the same kernel.
//
// Mirrors tests/rhi/test_offline_replay.cpp (the beauty byte-exact gate) for the
// sensor path. A .cu (the reference drives the CUDA-token rt entry + ScatterFk);
// --fmad=false + host -ffp-contract=off so it stays consistent with the rt kernel.
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/backend_cuda/rt/batched_sensor_render.hpp"
#include "phi/interop_scatter.hpp"
#include "render/sensor_backend.hpp"
#include "rhi/offline/render_profile.hpp"
#include "rhi/offline/sensor_renderer.hpp"
#include "rt/material.hpp"
#include "rt/two_level_render.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

using namespace nuka;

namespace {

constexpr uint32_t kInstancesPerEnv = 12u;
constexpr float kEnvPitch = 6.0f;
constexpr float kLinkSpread = 0.35f;
constexpr float kLinkHalf = 0.12f;
constexpr uint32_t kRes = 48u;
constexpr uint32_t kEnv = 32u;

#define NK_CUDA_OK(call)                                       \
    do {                                                       \
        const cudaError_t e_ = (call);                        \
        ASSERT_EQ(e_, cudaSuccess) << cudaGetErrorString(e_); \
    } while (0)

// One UNIT box mesh (12 triangles) in LOCAL space, shared by every instance.
rt::BlasMesh BuildBoxMesh(float half) {
    const math::Vec3 c[8] = {
        {-half, -half, -half}, {half, -half, -half}, {half, half, -half}, {-half, half, -half},
        {-half, -half, half},  {half, -half, half},  {half, half, half},  {-half, half, half}};
    const int faces[12][3] = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7},
                              {0, 1, 5}, {0, 5, 4}, {2, 3, 7}, {2, 7, 6},
                              {1, 2, 6}, {1, 6, 5}, {0, 4, 7}, {0, 7, 3}};
    rt::BlasMesh m;
    for (auto& f : faces) m.triangles.push_back({c[f[0]], c[f[1]], c[f[2]], 0u});
    return m;
}

uint32_t GridSide(uint32_t n) {
    uint32_t s = 1u;
    while (s * s < n) ++s;
    return s;
}

// The env-invariant local layout of instance k (a Base-mounted cluster).
math::Transform LinkLocal(uint32_t k) {
    const uint32_t side = GridSide(kInstancesPerEnv);
    const uint32_t lx = k % side;
    const uint32_t ly = k / side;
    const float ox = (static_cast<float>(lx) - 0.5f * static_cast<float>(side - 1u)) * kLinkSpread;
    const float oy = (static_cast<float>(ly) - 0.5f * static_cast<float>(side - 1u)) * kLinkSpread;
    const float oz = kLinkHalf + 0.02f * static_cast<float>(k % 5u);
    math::Transform t;
    t.position = {ox, oy, oz};
    t.rotation = math::Quat::FromAxisAngle(math::Vec3{0.2f, 1.0f, 0.3f},
                                           0.1f + 0.05f * static_cast<float>(k));
    return t;
}

// The world placement (Base pose) of env e: its tile origin + a per-env yaw.
math::Transform EnvBasePose(uint32_t e, uint32_t n) {
    const uint32_t side = GridSide(n);
    const uint32_t ix = e % side;
    const uint32_t iy = e / side;
    const float cx = (static_cast<float>(ix) - 0.5f * static_cast<float>(side - 1u)) * kEnvPitch;
    const float cy = (static_cast<float>(iy) - 0.5f * static_cast<float>(side - 1u)) * kEnvPitch;
    math::Transform t;
    t.position = {cx, cy, 0.0f};
    t.rotation = math::Quat::FromAxisAngle(math::Vec3{0.0f, 0.0f, 1.0f},
                                           0.07f * static_cast<float>(e));
    return t;
}

rt::Material LinkMaterial() {
    rt::Material mat;
    mat.albedo = {0.65f, 0.55f, 0.4f};
    mat.metallic = 0.0f;
    mat.roughness = 0.6f;
    return mat;
}

phi::InstanceScatterRow BaseRow(const math::Transform& cvl) {
    phi::InstanceScatterRow r;
    r.kind = 3u;  // Base-mounted
    r.row = 0u;
    r.cached_visual_local[0] = cvl.position.x;
    r.cached_visual_local[1] = cvl.position.y;
    r.cached_visual_local[2] = cvl.position.z;
    r.cached_visual_local[3] = cvl.rotation.w;
    r.cached_visual_local[4] = cvl.rotation.x;
    r.cached_visual_local[5] = cvl.rotation.y;
    r.cached_visual_local[6] = cvl.rotation.z;
    return r;
}

rt::TwoLevelScene MakeScene() {
    rt::TwoLevelScene s;
    s.meshes.push_back(BuildBoxMesh(kLinkHalf));
    s.materials = {LinkMaterial()};
    s.light.directional = true;
    s.light.direction = {0.3f, 0.4f, -0.85f};
    s.ambient.color = {0.05f, 0.05f, 0.05f};
    return s;
}

// One overhead Base-mounted depth/RGB camera 3 m above each env origin (identity
// rotation -> local -Z looks straight down at the cluster). Authored as the
// scene::SensorDesc the SensorRenderer + the direct path both lower identically.
scene::SensorDesc OverheadSensor() {
    scene::SensorDesc s;
    s.name = "overhead";
    s.mount = scene::MountFrame::Base;
    s.mount_index = 0u;
    s.local_offset.position = {0.0f, 0.0f, 3.0f};
    s.local_offset.rotation = math::Quat::Identity();
    s.cam.width = static_cast<uint16_t>(kRes);
    s.cam.height = static_cast<uint16_t>(kRes);
    s.cam.vfov_degrees = 0.6f * 180.0f;  // 0.6*pi rad expressed in degrees
    return s;
}

// Common scene-desc both paths consume (the env-shared binding + the mount table).
render::SensorSceneDesc MakeSensorSceneDesc() {
    render::SensorSceneDesc d;
    d.scene = MakeScene();
    for (uint32_t k = 0; k < kInstancesPerEnv; ++k) {
        d.rows.push_back(BaseRow(LinkLocal(k)));
        d.blas_id.push_back(0u);
        d.material_id.push_back(0u);
    }
    d.sensors = {OverheadSensor()};
    return d;
}

// Device base-pose buffer (one Transform per env) + the ScatterFkSource over it.
struct DeviceBasePoses {
    math::Transform* d_base = nullptr;
    phi::ScatterFkSource fk;
    void Upload(const std::vector<math::Transform>& base) {
        const size_t bytes = base.size() * sizeof(math::Transform);
        cudaMalloc(&d_base, bytes);
        cudaMemcpy(d_base, base.data(), bytes, cudaMemcpyHostToDevice);
        fk.base_pose = d_base;
        fk.link_pose = nullptr;
        fk.body_pose = nullptr;
        fk.links_per_env = 0u;
        fk.bodies_per_env = 0u;
    }
    void Free() { if (d_base != nullptr) cudaFree(d_base); }
};

}  // namespace

// The SensorRenderer AOV tensor equals the direct rt::RenderSensorsMounted tensor,
// byte-identical across all channels: the RHI layer adds zero numeric change.
TEST(SensorBatch, SensorRendererMatchesDirectBatchedByteExact) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend available";

    std::vector<math::Transform> base(kEnv);
    for (uint32_t e = 0; e < kEnv; ++e) base[e] = EnvBasePose(e, kEnv);

    const render::SensorSceneDesc desc = MakeSensorSceneDesc();
    const size_t pix = static_cast<size_t>(kRes) * kRes;
    const uint64_t rays = static_cast<uint64_t>(kEnv) * pix;

    // --- Path A: drive the rt-layer batched entry directly. ---------------------
    std::vector<float> a_color(rays * 3u), a_normal(rays * 3u), a_albedo(rays * 3u), a_depth(rays);
    std::vector<uint32_t> a_prim(rays);
    {
        rt::BatchedSensorSceneDesc cuda_desc;
        cuda_desc.scene = desc.scene;
        cuda_desc.rows = desc.rows;
        cuda_desc.blas_id = desc.blas_id;
        cuda_desc.material_id = desc.material_id;
        rt::BatchedSensorSceneDevice scene = rt::BuildBatchedSensorScene(cuda_desc);
        rt::SetSensorMounts(scene, desc.sensors);

        DeviceBasePoses dev;
        dev.Upload(base);
        rt::RenderSensorsMounted(scene, dev.fk, kEnv, kRes, kRes);
        NK_CUDA_OK(cudaDeviceSynchronize());

        NK_CUDA_OK(cudaMemcpy(a_color.data(), rt::SensorColorDevice(scene), a_color.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(a_depth.data(), rt::SensorDepthDevice(scene), a_depth.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(a_normal.data(), rt::SensorNormalDevice(scene), a_normal.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(a_albedo.data(), rt::SensorAlbedoDevice(scene), a_albedo.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(a_prim.data(), rt::SensorPrimDevice(scene), a_prim.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        dev.Free();
    }

    bool any_hit = false;
    for (uint32_t v : a_prim) if (v != 0xFFFFFFFFu) { any_hit = true; break; }
    EXPECT_TRUE(any_hit) << "direct batched render produced an all-miss image (scene/camera bug)";

    // --- Path B: the same scene + poses through the offline-RHI SensorRenderer. --
    std::vector<float> b_color(rays * 3u), b_normal(rays * 3u), b_albedo(rays * 3u), b_depth(rays);
    std::vector<uint32_t> b_prim(rays);
    {
        rhi::offline::SensorRenderer renderer;
        ASSERT_TRUE(renderer.usable()) << "SensorRenderer obtained no sensor backend";
        const rhi::offline::RenderProfile profile = rhi::offline::RenderProfile::Sensor(kRes, kRes);
        ASSERT_TRUE(renderer.Build(desc, profile)) << "SensorRenderer::Build failed";

        DeviceBasePoses dev;
        dev.Upload(base);
        renderer.Render(dev.fk, kEnv);
        NK_CUDA_OK(cudaDeviceSynchronize());

        const render::SensorAovShape shape = renderer.AovShape();
        EXPECT_EQ(shape.env_count, kEnv);
        EXPECT_EQ(shape.sensors_per_env, 1u);
        EXPECT_EQ(shape.height, kRes);
        EXPECT_EQ(shape.width, kRes);

        NK_CUDA_OK(cudaMemcpy(b_color.data(), renderer.ColorDevice(), b_color.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(b_depth.data(), renderer.DepthDevice(), b_depth.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(b_normal.data(), renderer.NormalDevice(), b_normal.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(b_albedo.data(), renderer.AlbedoDevice(), b_albedo.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(b_prim.data(), renderer.PrimDevice(), b_prim.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        dev.Free();
    }

    EXPECT_EQ(std::memcmp(a_color.data(), b_color.data(), a_color.size() * sizeof(float)), 0)
        << "color tensor differs (RHI added numeric change)";
    EXPECT_EQ(std::memcmp(a_depth.data(), b_depth.data(), a_depth.size() * sizeof(float)), 0)
        << "depth tensor differs (RHI added numeric change)";
    EXPECT_EQ(std::memcmp(a_normal.data(), b_normal.data(), a_normal.size() * sizeof(float)), 0)
        << "normal tensor differs (RHI added numeric change)";
    EXPECT_EQ(std::memcmp(a_albedo.data(), b_albedo.data(), a_albedo.size() * sizeof(float)), 0)
        << "albedo tensor differs (RHI added numeric change)";
    EXPECT_EQ(std::memcmp(a_prim.data(), b_prim.data(), a_prim.size() * sizeof(uint32_t)), 0)
        << "prim tensor differs (RHI added numeric change)";
}
