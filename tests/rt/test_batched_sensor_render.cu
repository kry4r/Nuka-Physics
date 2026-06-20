// ---------------------------------------------------------------------------
// Batched sensor render gate: a per-env tile of the ONE batched (N,H,W,ch) AOV
// tensor is BYTE-IDENTICAL to a standalone single-env render of that env's scene
// + camera. Same FP32 trace (ClosestHit/ReconstructHit/ShadeDirect, Real=float)
// + same per-env TLAS + same camera -> same FP32 bytes (one writer per pixel, no
// atomics -> deterministic).
//
// Each env is a Base-mounted replicated cluster: world = base_pose[env] * cvl
// (the env-invariant link layout). The reference for env e drives the SAME
// batched path with E=1 and base_pose[0]=base_pose[e], so global==env-local ids
// and every channel (color/depth/normal/albedo/prim) is byte-comparable. Tested
// at envs 0 / mid / last over a multi-instance box scene.
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/backend_cuda/rt/batched_sensor_render.hpp"
#include "phi/interop_scatter.hpp"
#include "rt/camera.hpp"
#include "rt/material.hpp"
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

#define NK_CUDA_OK(call)                                          \
    do {                                                          \
        const cudaError_t e_ = (call);                           \
        ASSERT_EQ(e_, cudaSuccess) << cudaGetErrorString(e_);    \
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

// The env-invariant local offset of instance k (the replicated "robot" layout),
// expressed as a cached_visual_local Transform (Base-mounted -> world=base*cvl).
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

// The world placement (Base pose) of env e: its tile origin + a per-env yaw so
// envs genuinely differ (distinct TLAS + distinct image).
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

// A camera looking at env e's cluster from a fixed offset above its tile origin.
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

// cached_visual_local row from a Transform (pos3 + quat w,x,y,z).
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

// Device Base-pose buffer (one Transform per env) + the ScatterFkSource over it.
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

}  // namespace

// A per-env tile of the batched render == an E=1 batched render of that env.
TEST(BatchedSensorRender, PerEnvTileMatchesSingleEnv) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";

    const uint32_t kEnv = 64u;
    const size_t pix = static_cast<size_t>(kRes) * kRes;

    // --- Full batched render over kEnv envs. -------------------------------
    rt::BatchedSensorSceneDevice scene = rt::BuildBatchedSensorScene(MakeSceneDesc());

    std::vector<math::Transform> base(kEnv);
    std::vector<rt::PinholeCamera> cams(kEnv);
    for (uint32_t e = 0; e < kEnv; ++e) {
        base[e] = EnvBasePose(e, kEnv);
        cams[e] = CameraForEnv(e, kEnv);
    }
    DeviceBasePoses dev;
    dev.Upload(base);
    rt::PinholeCamera* d_cams = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_cams, kEnv * sizeof(rt::PinholeCamera)));
    NK_CUDA_OK(cudaMemcpy(d_cams, cams.data(), kEnv * sizeof(rt::PinholeCamera),
                          cudaMemcpyHostToDevice));

    rt::RenderSensorsBatched(scene, dev.fk, d_cams, kEnv, kRes, kRes);
    NK_CUDA_OK(cudaDeviceSynchronize());

    // Download the whole tensor (color/depth/normal/albedo/prim).
    const uint64_t rays = static_cast<uint64_t>(kEnv) * pix;
    std::vector<float> color(rays * 3u), normal(rays * 3u), albedo(rays * 3u), depth(rays);
    std::vector<uint32_t> prim(rays);
    NK_CUDA_OK(cudaMemcpy(color.data(), rt::SensorColorDevice(scene), color.size() * sizeof(float), cudaMemcpyDeviceToHost));
    NK_CUDA_OK(cudaMemcpy(depth.data(), rt::SensorDepthDevice(scene), depth.size() * sizeof(float), cudaMemcpyDeviceToHost));
    NK_CUDA_OK(cudaMemcpy(normal.data(), rt::SensorNormalDevice(scene), normal.size() * sizeof(float), cudaMemcpyDeviceToHost));
    NK_CUDA_OK(cudaMemcpy(albedo.data(), rt::SensorAlbedoDevice(scene), albedo.size() * sizeof(float), cudaMemcpyDeviceToHost));
    NK_CUDA_OK(cudaMemcpy(prim.data(), rt::SensorPrimDevice(scene), prim.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    bool any_hit = false;
    for (uint32_t v : prim) if (v != 0xFFFFFFFFu) { any_hit = true; break; }
    EXPECT_TRUE(any_hit) << "batched render produced an all-miss image (scene/camera bug)";

    // --- For envs 0 / mid / last: an E=1 batched render must match the tile. -
    const uint32_t probes[3] = {0u, kEnv / 2u, kEnv - 1u};
    for (uint32_t e : probes) {
        rt::BatchedSensorSceneDevice solo = rt::BuildBatchedSensorScene(MakeSceneDesc());
        std::vector<math::Transform> sb = {base[e]};
        DeviceBasePoses sdev;
        sdev.Upload(sb);
        rt::PinholeCamera one = cams[e];
        rt::PinholeCamera* d_one = nullptr;
        NK_CUDA_OK(cudaMalloc(&d_one, sizeof(rt::PinholeCamera)));
        NK_CUDA_OK(cudaMemcpy(d_one, &one, sizeof(rt::PinholeCamera), cudaMemcpyHostToDevice));

        rt::RenderSensorsBatched(solo, sdev.fk, d_one, 1u, kRes, kRes);
        NK_CUDA_OK(cudaDeviceSynchronize());

        std::vector<float> rc(pix * 3u), rn(pix * 3u), ra(pix * 3u), rd(pix);
        std::vector<uint32_t> rp(pix);
        NK_CUDA_OK(cudaMemcpy(rc.data(), rt::SensorColorDevice(solo), rc.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(rd.data(), rt::SensorDepthDevice(solo), rd.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(rn.data(), rt::SensorNormalDevice(solo), rn.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(ra.data(), rt::SensorAlbedoDevice(solo), ra.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(rp.data(), rt::SensorPrimDevice(solo), rp.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        const size_t off3 = static_cast<size_t>(e) * pix * 3u;
        const size_t off1 = static_cast<size_t>(e) * pix;
        EXPECT_EQ(std::memcmp(rc.data(), color.data() + off3, rc.size() * sizeof(float)), 0)
            << "env " << e << " color tile != standalone";
        EXPECT_EQ(std::memcmp(rd.data(), depth.data() + off1, rd.size() * sizeof(float)), 0)
            << "env " << e << " depth tile != standalone";
        EXPECT_EQ(std::memcmp(rn.data(), normal.data() + off3, rn.size() * sizeof(float)), 0)
            << "env " << e << " normal tile != standalone";
        EXPECT_EQ(std::memcmp(ra.data(), albedo.data() + off3, ra.size() * sizeof(float)), 0)
            << "env " << e << " albedo tile != standalone";
        EXPECT_EQ(std::memcmp(rp.data(), prim.data() + off1, rp.size() * sizeof(uint32_t)), 0)
            << "env " << e << " prim tile != standalone";

        sdev.Free();
        cudaFree(d_one);
    }

    dev.Free();
    cudaFree(d_cams);
}
