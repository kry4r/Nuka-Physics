// ---------------------------------------------------------------------------
// Per-env render domain-randomization gate. The batched render ALWAYS reads
// material/light/ambient BY ENV; whether the per-env tables are base replicas or
// randomized is the ONLY difference. This proves the three contract points:
//   (1) DR OFF -> cross-env tiles byte-identical (the existing invariant: tables
//       are exact base replicas, so every env's color is the same bytes).
//   (2) DR ON  -> env tiles DIFFER (per-env appearance variety -- the whole point).
//   (3) DR ON  -> deterministic: the SAME seed yields byte-identical bytes across
//       two independent renders (Philox(seed,env,axis), no mutable RNG state).
//
// Each env is a Base-mounted replicated cluster (world = base_pose[env] * cvl) with
// IDENTICAL geometry + pose across envs, so any cross-env color difference comes
// PURELY from the per-env appearance tables, not from geometry/camera.
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/backend_cuda/rt/batched_sensor_render.hpp"
#include "phi/interop_scatter.hpp"
#include "rt/camera.hpp"
#include "rt/material.hpp"
#include "rt/render_dr.hpp"
#include "rt/two_level_render.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

using namespace nuka;

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kInstancesPerEnv = 12u;
constexpr float kLinkSpread = 0.35f;
constexpr float kLinkHalf = 0.12f;
constexpr uint32_t kRes = 48u;

#define NK_CUDA_OK(call)                                          \
    do {                                                          \
        const cudaError_t e_ = (call);                           \
        ASSERT_EQ(e_, cudaSuccess) << cudaGetErrorString(e_);    \
    } while (0)

rt::BlasMesh BuildBoxMesh(float half) {
    const math::Vec3 c[8] = {
        {-half, -half, -half}, {half, -half, -half}, {half, half, -half}, {-half, half, -half},
        {-half, -half, half},  {half, -half, half},  {half, half, half},  {-half, half, half}};
    const int faces[12][3] = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7},
                              {0, 1, 5}, {0, 5, 4}, {2, 3, 7}, {2, 7, 6},
                              {1, 2, 6}, {1, 6, 5}, {0, 4, 7}, {0, 7, 3}};
    rt::BlasMesh m;
    for (auto& f : faces) {
        m.triangles.push_back({c[f[0]], c[f[1]], c[f[2]], 0u});
    }
    return m;
}

uint32_t GridSide(uint32_t n) {
    uint32_t s = 1u;
    while (s * s < n) ++s;
    return s;
}

math::Transform LinkLocal(uint32_t k) {
    const uint32_t side = GridSide(kInstancesPerEnv);
    const uint32_t lx = k % side;
    const uint32_t ly = k / side;
    const float ox = (static_cast<float>(lx) - 0.5f * static_cast<float>(side - 1u)) * kLinkSpread;
    const float oy = (static_cast<float>(ly) - 0.5f * static_cast<float>(side - 1u)) * kLinkSpread;
    const float oz = kLinkHalf + 0.02f * static_cast<float>(k % 5u);
    const float ang = 0.1f + 0.05f * static_cast<float>(k);
    math::Transform t;
    t.position = {ox, oy, oz};
    t.rotation = math::Quat::FromAxisAngle(math::Vec3{0.2f, 1.0f, 0.3f}, ang);
    return t;
}

// IDENTICAL base pose for every env (no per-env yaw): geometry is env-invariant so
// any cross-env color difference is appearance-only. Camera placed once.
math::Transform EnvBasePose() {
    math::Transform t;
    t.position = {0.0f, 0.0f, 0.0f};
    t.rotation = math::Quat::FromAxisAngle(math::Vec3{0.0f, 0.0f, 1.0f}, 0.0f);
    return t;
}

rt::PinholeCamera SharedCamera() {
    const math::Vec3 eye{1.6f, -1.6f, 1.4f};
    const math::Vec3 target{0.0f, 0.0f, kLinkHalf};
    return rt::BuildPinhole(eye, target, {0.0f, 0.0f, 1.0f}, 0.6f * kPi, kRes, kRes);
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

rt::BatchedSensorSceneDesc MakeSceneDesc() {
    rt::BatchedSensorSceneDesc d;
    d.scene.meshes.push_back(BuildBoxMesh(kLinkHalf));
    d.scene.materials = {LinkMaterial()};
    d.scene.light.directional = true;
    d.scene.light.direction = {0.3f, 0.4f, -0.85f};
    d.scene.ambient.color = {0.05f, 0.05f, 0.05f};
    for (uint32_t k = 0; k < kInstancesPerEnv; ++k) {
        d.rows.push_back(BaseRow(LinkLocal(k)));
        d.blas_id.push_back(0u);
        d.material_id.push_back(0u);
    }
    return d;
}

struct DeviceBasePoses {
    math::Transform* d_base = nullptr;
    phi::ScatterFkSource fk;
    void Upload(const std::vector<math::Transform>& base) {
        cudaMalloc(&d_base, base.size() * sizeof(math::Transform));
        cudaMemcpy(d_base, base.data(), base.size() * sizeof(math::Transform),
                   cudaMemcpyHostToDevice);
        fk.base_pose = d_base;
        fk.link_pose = nullptr;
        fk.body_pose = nullptr;
        fk.links_per_env = 0u;
        fk.bodies_per_env = 0u;
    }
    void Free() { if (d_base) cudaFree(d_base); }
};

// A strong appearance DR config (visible per-env variety).
rt::RenderDrConfig DrOn(uint64_t seed) {
    rt::RenderDrConfig cfg;
    cfg.enabled = true;
    cfg.seed = seed;
    cfg.color_jitter = 0.3f;
    cfg.roughness_jitter = 0.3f;
    cfg.metallic_jitter = 0.2f;
    cfg.light_dir_jitter = 0.2f;
    cfg.light_intensity_jitter = 0.4f;
    cfg.light_color_jitter = 0.3f;
    cfg.ambient_intensity_jitter = 0.5f;
    return cfg;
}

// Render kEnv identical-geometry envs with the given DR config, return the color
// tensor (E*pix*3 floats).
std::vector<float> RenderColors(const rt::RenderDrConfig& cfg, uint32_t env_count) {
    const size_t pix = static_cast<size_t>(kRes) * kRes;
    rt::BatchedSensorSceneDevice scene = rt::BuildBatchedSensorScene(MakeSceneDesc());
    rt::SetRenderDr(scene, cfg, env_count);

    std::vector<math::Transform> base(env_count, EnvBasePose());
    std::vector<rt::PinholeCamera> cams(env_count, SharedCamera());
    DeviceBasePoses dev;
    dev.Upload(base);
    rt::PinholeCamera* d_cams = nullptr;
    cudaMalloc(&d_cams, env_count * sizeof(rt::PinholeCamera));
    cudaMemcpy(d_cams, cams.data(), env_count * sizeof(rt::PinholeCamera),
               cudaMemcpyHostToDevice);

    rt::RenderSensorsBatched(scene, dev.fk, d_cams, env_count, 1u, kRes, kRes);
    cudaDeviceSynchronize();

    const uint64_t rays = static_cast<uint64_t>(env_count) * pix;
    std::vector<float> color(rays * 3u);
    cudaMemcpy(color.data(), rt::SensorColorDevice(scene), color.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    dev.Free();
    cudaFree(d_cams);
    return color;
}

}  // namespace

// (1) DR OFF -> cross-env tiles byte-identical (tables are base replicas).
TEST(RenderDr, OffCrossEnvByteIdentical) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";
    const uint32_t kEnv = 16u;
    const size_t pix3 = static_cast<size_t>(kRes) * kRes * 3u;

    rt::RenderDrConfig off;  // disabled by default
    const std::vector<float> color = RenderColors(off, kEnv);

    size_t mism = 0u;
    for (uint32_t e = 1u; e < kEnv; ++e) {
        const size_t off_e = static_cast<size_t>(e) * pix3;
        if (std::memcmp(color.data(), color.data() + off_e, pix3 * sizeof(float)) != 0) {
            ++mism;
        }
    }
    EXPECT_EQ(mism, 0u) << "DR off must keep every env's tile byte-identical (replicas)";
}

// (2) DR ON -> env tiles DIFFER (per-env appearance variety).
TEST(RenderDr, OnEnvTilesDiffer) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";
    const uint32_t kEnv = 16u;
    const size_t pix3 = static_cast<size_t>(kRes) * kRes * 3u;

    const std::vector<float> color = RenderColors(DrOn(1234u), kEnv);

    // Every env beyond 0 must differ from env 0 (its own appearance set).
    uint32_t differing = 0u;
    for (uint32_t e = 1u; e < kEnv; ++e) {
        const size_t off_e = static_cast<size_t>(e) * pix3;
        if (std::memcmp(color.data(), color.data() + off_e, pix3 * sizeof(float)) != 0) {
            ++differing;
        }
    }
    EXPECT_EQ(differing, kEnv - 1u)
        << "DR on must give each env a distinct appearance (only " << differing
        << "/" << (kEnv - 1u) << " envs differ from env 0)";

    // And the variety is genuine: count distinct env-0-vs-env-e pixel deltas.
    size_t changed_px = 0u;
    for (size_t i = 0u; i < pix3; ++i) {
        if (color[pix3 + i] != color[i]) ++changed_px;
    }
    EXPECT_GT(changed_px, 0u) << "env 1 identical to env 0 under DR on";
}

// (3) DR ON -> deterministic: same seed, two renders byte-identical.
TEST(RenderDr, OnSameSeedByteIdentical) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";
    const uint32_t kEnv = 16u;

    const std::vector<float> a = RenderColors(DrOn(777u), kEnv);
    const std::vector<float> b = RenderColors(DrOn(777u), kEnv);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0)
        << "same seed must produce byte-identical bytes (RL-reproducible)";

    // A different seed must produce a different appearance distribution.
    const std::vector<float> c = RenderColors(DrOn(888u), kEnv);
    EXPECT_NE(std::memcmp(a.data(), c.data(), a.size() * sizeof(float)), 0)
        << "a different seed must change the per-env appearance";
}
