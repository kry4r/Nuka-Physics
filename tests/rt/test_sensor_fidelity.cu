// ---------------------------------------------------------------------------
// Opt-in shading-fidelity gate for the batched sensor render. The batched trace
// shades flat (1 spp, 1 hard shadow); SetSensorFidelity lifts it to the beauty
// look (spp MSAA + soft shadow + AO + one-bounce GI + tonemap) in the SAME kernel,
// gated on the config. The DEFAULT config takes the exact cheap arithmetic.
//
// Gates:
//  (1) DEFAULT == byte-no-op: a render after SetSensorFidelity(default) is
//      byte-identical to a render with no fidelity ever set.
//  (2) fidelity-ON deterministic: two renders at the same seed are byte-identical
//      (Philox-seeded per (ray,sample), no mutable RNG state).
//  (3) fidelity-ON visibly different + finite: the color differs from the cheap
//      shade on a real fraction of pixels and every value is finite.
//  (4) AOVs unchanged: depth/normal/prim are byte-identical cheap vs fidelity
//      (the center ray fills them; only color is replaced).
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/backend_cuda/rt/batched_sensor_render.hpp"
#include "phi/interop_scatter.hpp"
#include "rt/camera.hpp"
#include "rt/material.hpp"
#include "rt/sensor_fidelity.hpp"
#include "rt/two_level_render.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

using namespace nuka;

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kInstancesPerEnv = 12u;
constexpr float kEnvPitch = 6.0f;
constexpr float kLinkSpread = 0.35f;
constexpr float kLinkHalf = 0.12f;
constexpr uint32_t kRes = 48u;
constexpr uint32_t kEnv = 16u;

#define NK_CUDA_OK(call)                                       \
    do {                                                       \
        const cudaError_t e_ = (call);                        \
        ASSERT_EQ(e_, cudaSuccess) << cudaGetErrorString(e_); \
    } while (0)

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

rt::PinholeCamera CameraForEnv(uint32_t e, uint32_t n) {
    const math::Transform base = EnvBasePose(e, n);
    const math::Vec3 eye{base.position.x + 1.6f, base.position.y - 1.6f, 1.4f};
    const math::Vec3 target{base.position.x, base.position.y, kLinkHalf};
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
        NK_CUDA_OK(cudaMalloc(&d_base, base.size() * sizeof(math::Transform)));
        NK_CUDA_OK(cudaMemcpy(d_base, base.data(), base.size() * sizeof(math::Transform),
                              cudaMemcpyHostToDevice));
        fk.base_pose = d_base;
        fk.link_pose = nullptr;
        fk.body_pose = nullptr;
        fk.links_per_env = 0u;
        fk.bodies_per_env = 0u;
    }
    void Free() { if (d_base) cudaFree(d_base); }
};

// Render the scene once and return the flat color (+ optional depth/normal/prim).
struct Aov {
    std::vector<float> color, normal, depth;
    std::vector<uint32_t> prim;
};

Aov RenderOnce(const rt::SensorFidelityConfig* fid, bool want_extra) {
    const size_t pix = static_cast<size_t>(kRes) * kRes;
    const uint64_t rays = static_cast<uint64_t>(kEnv) * pix;
    rt::BatchedSensorSceneDevice scene = rt::BuildBatchedSensorScene(MakeSceneDesc());
    if (fid != nullptr) rt::SetSensorFidelity(scene, *fid);

    std::vector<math::Transform> base(kEnv);
    std::vector<rt::PinholeCamera> cams(kEnv);
    for (uint32_t e = 0; e < kEnv; ++e) {
        base[e] = EnvBasePose(e, kEnv);
        cams[e] = CameraForEnv(e, kEnv);
    }
    DeviceBasePoses dev;
    dev.Upload(base);
    rt::PinholeCamera* d_cams = nullptr;
    EXPECT_EQ(cudaMalloc(&d_cams, kEnv * sizeof(rt::PinholeCamera)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_cams, cams.data(), kEnv * sizeof(rt::PinholeCamera),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    rt::RenderSensorsBatched(scene, dev.fk, d_cams, kEnv, 1u, kRes, kRes);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    Aov out;
    out.color.resize(rays * 3u);
    EXPECT_EQ(cudaMemcpy(out.color.data(), rt::SensorColorDevice(scene),
                         out.color.size() * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    if (want_extra) {
        out.normal.resize(rays * 3u);
        out.depth.resize(rays);
        out.prim.resize(rays);
        EXPECT_EQ(cudaMemcpy(out.normal.data(), rt::SensorNormalDevice(scene),
                             out.normal.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(cudaMemcpy(out.depth.data(), rt::SensorDepthDevice(scene),
                             out.depth.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(cudaMemcpy(out.prim.data(), rt::SensorPrimDevice(scene),
                             out.prim.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
    dev.Free();
    cudaFree(d_cams);
    return out;
}

// A representative beauty profile: MSAA + soft shadow + AO + GI + tonemap.
rt::SensorFidelityConfig BeautyProfile() {
    rt::SensorFidelityConfig f;
    f.spp = 4u;
    f.shadow_samples = 4u;
    f.ao_enabled = true;
    f.ao_samples = 3u;
    f.gi_enabled = true;
    f.tonemap_enabled = true;
    f.seed = 1234567u;
    return f;
}

}  // namespace

// Gate 1 -- the default config is a byte no-op (cheap-shade arithmetic).
TEST(SensorFidelity, DefaultConfigIsByteNoOp) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";
    const Aov never = RenderOnce(nullptr, false);
    const rt::SensorFidelityConfig def;  // default == cheap shade
    EXPECT_FALSE(def.Enabled());
    const Aov set_default = RenderOnce(&def, false);
    ASSERT_EQ(never.color.size(), set_default.color.size());
    EXPECT_EQ(std::memcmp(never.color.data(), set_default.color.data(),
                          never.color.size() * sizeof(float)),
              0)
        << "default fidelity config moved bytes -- not a byte no-op";
}

// Gate 2 -- fidelity ON is deterministic (same seed -> two renders byte-identical).
TEST(SensorFidelity, FidelityOnDeterministic) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";
    const rt::SensorFidelityConfig f = BeautyProfile();
    ASSERT_TRUE(f.Enabled());
    const Aov a = RenderOnce(&f, false);
    const Aov b = RenderOnce(&f, false);
    ASSERT_EQ(a.color.size(), b.color.size());
    EXPECT_EQ(std::memcmp(a.color.data(), b.color.data(), a.color.size() * sizeof(float)),
              0)
        << "fidelity render is not byte-exact two-run (RNG not stateless-seeded)";
}

// Gate 3 -- fidelity ON differs from the cheap shade on a real fraction of pixels
// AND every value is finite.
TEST(SensorFidelity, FidelityOnVisiblyDifferentAndFinite) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";
    const Aov cheap = RenderOnce(nullptr, false);
    const rt::SensorFidelityConfig f = BeautyProfile();
    const Aov fancy = RenderOnce(&f, false);
    ASSERT_EQ(cheap.color.size(), fancy.color.size());

    size_t diff = 0u, nonfinite = 0u;
    for (size_t i = 0; i < fancy.color.size(); ++i) {
        if (!std::isfinite(fancy.color[i])) ++nonfinite;
        if (fancy.color[i] != cheap.color[i]) ++diff;
    }
    const double frac = static_cast<double>(diff) / fancy.color.size();
    std::printf("[diag] fidelity vs cheap: %zu/%zu channels differ (%.1f%%), nonfinite=%zu\n",
                diff, fancy.color.size(), 100.0 * frac, nonfinite);
    EXPECT_EQ(nonfinite, 0u) << "fidelity produced non-finite color";
    EXPECT_GT(frac, 0.05) << "fidelity barely changed the image (shade not applied)";
}

// Gate 4 -- depth/normal/prim are byte-identical cheap vs fidelity (only color
// changes; the center ray fills the AOVs in both modes).
TEST(SensorFidelity, AovsUnchangedUnderFidelity) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";
    const Aov cheap = RenderOnce(nullptr, true);
    const rt::SensorFidelityConfig f = BeautyProfile();
    const Aov fancy = RenderOnce(&f, true);
    ASSERT_EQ(cheap.depth.size(), fancy.depth.size());
    EXPECT_EQ(std::memcmp(cheap.depth.data(), fancy.depth.data(),
                          cheap.depth.size() * sizeof(float)),
              0)
        << "depth changed under fidelity (must come from the center ray)";
    EXPECT_EQ(std::memcmp(cheap.normal.data(), fancy.normal.data(),
                          cheap.normal.size() * sizeof(float)),
              0)
        << "normal changed under fidelity";
    EXPECT_EQ(std::memcmp(cheap.prim.data(), fancy.prim.data(),
                          cheap.prim.size() * sizeof(uint32_t)),
              0)
        << "prim changed under fidelity";
}

// Gate 5 -- spp over the cap is rejected LOUDLY (no silent clamp / hang).
TEST(SensorFidelity, SppOverCapThrows) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";
    rt::BatchedSensorSceneDevice scene = rt::BuildBatchedSensorScene(MakeSceneDesc());
    rt::SensorFidelityConfig bad;
    bad.spp = 100000u;  // far over kMaxFidelitySpp
    EXPECT_THROW(rt::SetSensorFidelity(scene, bad), std::runtime_error);
}
