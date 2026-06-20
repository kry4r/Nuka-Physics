// ---------------------------------------------------------------------------
// Batched sensor render PERF deliverable: the near-real-time verdict. Measures
// the device-resident BATCHED path (ONE batched LBVH build + ONE flat trace over
// [E*H*W] rays into a (N,H,W,ch) tensor, NO host download) vs the naive per-env
// loop (N separate BuildTwoLevelScene + RenderFrameToAovs) at N=256/1024/4096
// envs x 64^2 x 1spp, and prints ms/step + fps + the speedup.
//
// Both paths render the SAME replicated box scene (Base-mounted clusters). The
// batched path is the structural lever the feasibility bench called for: it fills
// the GPU with every env's rays and removes the N-serial launches.
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/backend_cuda/rt/batched_sensor_render.hpp"
#include "phi/interop_scatter.hpp"
#include "rt/camera.hpp"
#include "rt/material.hpp"
#include "rt/two_level_render.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

using namespace nuka;

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kInstancesPerEnv = 36u;
constexpr float kEnvPitch = 6.0f;
constexpr float kLinkSpread = 0.35f;
constexpr float kLinkHalf = 0.12f;

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

uint32_t GridSide(uint32_t n) { uint32_t s = 1u; while (s * s < n) ++s; return s; }

math::Transform LinkLocal(uint32_t k) {
    const uint32_t side = GridSide(kInstancesPerEnv);
    const uint32_t lx = k % side, ly = k / side;
    const float ox = (static_cast<float>(lx) - 0.5f * static_cast<float>(side - 1u)) * kLinkSpread;
    const float oy = (static_cast<float>(ly) - 0.5f * static_cast<float>(side - 1u)) * kLinkSpread;
    const float oz = kLinkHalf + 0.02f * static_cast<float>(k % 5u);
    math::Transform t;
    t.position = {ox, oy, oz};
    t.rotation = math::Quat::FromAxisAngle(math::Vec3{0.2f, 1.0f, 0.3f},
                                           0.1f + 0.05f * static_cast<float>(k));
    return t;
}

math::Transform EnvBasePose(uint32_t e, uint32_t n, float nudge) {
    const uint32_t side = GridSide(n);
    const uint32_t ix = e % side, iy = e / side;
    const float cx = (static_cast<float>(ix) - 0.5f * static_cast<float>(side - 1u)) * kEnvPitch;
    const float cy = (static_cast<float>(iy) - 0.5f * static_cast<float>(side - 1u)) * kEnvPitch;
    math::Transform t;
    t.position = {cx, cy, 0.0f};
    t.rotation = math::Quat::FromAxisAngle(math::Vec3{0.0f, 0.0f, 1.0f},
                                           0.07f * static_cast<float>(e) + nudge);
    return t;
}

rt::PinholeCamera CameraForEnv(uint32_t e, uint32_t n, uint32_t res) {
    const math::Transform b = EnvBasePose(e, n, 0.0f);
    const math::Vec3 eye{b.position.x + 1.6f, b.position.y - 1.6f, 1.4f};
    const math::Vec3 target{b.position.x, b.position.y, kLinkHalf};
    return rt::BuildPinhole(eye, target, {0.0f, 0.0f, 1.0f}, 0.6f * kPi, res, res);
}

rt::Material LinkMaterial() {
    rt::Material mat;
    mat.albedo = {0.65f, 0.55f, 0.4f};
    mat.roughness = 0.6f;
    return mat;
}

phi::InstanceScatterRow BaseRow(const math::Transform& cvl) {
    phi::InstanceScatterRow r;
    r.kind = 3u;
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

// Per-env scene (one cluster) at env e's tile, for the naive per-env loop. The
// instances bake world = base * local (the same placement the batched path makes).
rt::TwoLevelScene MakeEnvScene(uint32_t e, uint32_t n) {
    rt::TwoLevelScene s;
    s.meshes.push_back(BuildBoxMesh(kLinkHalf));
    s.materials = {LinkMaterial()};
    s.light.directional = true;
    s.light.direction = {0.3f, 0.4f, -0.85f};
    s.ambient.color = {0.05f, 0.05f, 0.05f};
    const math::Transform base = EnvBasePose(e, n, 0.0f);
    for (uint32_t k = 0; k < kInstancesPerEnv; ++k) {
        rt::Instance inst;
        inst.blas_id = 0u;
        inst.material_id = 0u;
        inst.transform = base * LinkLocal(k);
        s.instances.push_back(inst);
    }
    return s;
}

double NowSeconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct EnvAovSlice {
    phi::Buffer* color = nullptr; phi::Buffer* depth = nullptr; phi::Buffer* normal = nullptr;
    phi::Buffer* albedo = nullptr; phi::Buffer* uv = nullptr; phi::Buffer* prim = nullptr;
    rt::RtDeviceAovs View() const {
        rt::RtDeviceAovs a;
        a.color = color; a.depth = depth; a.normal = normal;
        a.albedo = albedo; a.uv = uv; a.prim = prim;
        return a;
    }
    void Alloc(phi::BufferType* bt, size_t px) {
        color = phi::BufferAlloc(bt, px * 3u * sizeof(float));
        depth = phi::BufferAlloc(bt, px * sizeof(float));
        normal = phi::BufferAlloc(bt, px * 3u * sizeof(float));
        albedo = phi::BufferAlloc(bt, px * 3u * sizeof(float));
        uv = phi::BufferAlloc(bt, px * 2u * sizeof(float));
        prim = phi::BufferAlloc(bt, px * sizeof(uint32_t));
    }
    void Free() {
        phi::BufferFree(color); phi::BufferFree(depth); phi::BufferFree(normal);
        phi::BufferFree(albedo); phi::BufferFree(uv); phi::BufferFree(prim);
    }
};

}  // namespace

int main() {
    if (phi::ActiveBackend() == nullptr) {
        std::printf("[perf] no CUDA backend available -- cannot run.\n");
        return 1;
    }
    phi::BufferType* bt = phi::BackendDeviceBufferType(phi::ActiveBackend());

    const std::vector<uint32_t> kEnvCounts = {256u, 1024u, 4096u};
    const uint32_t kRes = 64u;
    const size_t pix = static_cast<size_t>(kRes) * kRes;
    const uint32_t kIters = 5u;  // steady-state steps averaged per config

    std::printf("[perf] BATCHED vs NAIVE device-resident sensor render "
                "(%u inst/env, box BLAS=12 tris, %ux%u, 1spp, NO host download)\n",
                kInstancesPerEnv, kRes, kRes);
    std::printf("%-6s | %-12s %-9s | %-12s %-9s | %-8s\n",
                "N", "naive ms", "naive fps", "batched ms", "batch fps", "speedup");
    std::printf("------------------------------------------------------------------------\n");

    for (uint32_t n : kEnvCounts) {
        // ---- NAIVE per-env loop: N devices + N AOV slices; refit steady state. --
        std::vector<rt::TwoLevelScene> scenes(n);
        std::vector<rt::PinholeCamera> cams(n);
        std::vector<EnvAovSlice> aovs(n);
        std::vector<rt::TwoLevelSceneDevice> devs;
        devs.reserve(n);
        for (uint32_t e = 0; e < n; ++e) {
            scenes[e] = MakeEnvScene(e, n);
            cams[e] = CameraForEnv(e, n, kRes);
            devs.push_back(rt::BuildTwoLevelScene(scenes[e]));
            aovs[e].Alloc(bt, pix);
        }
        for (uint32_t e = 0; e < n; ++e)  // warm-up (first-touch builds)
            rt::RenderFrameToAovs(devs[e], scenes[e], cams[e], aovs[e].View());
        cudaDeviceSynchronize();

        double naive_ms = 0.0;
        for (uint32_t it = 0; it < kIters; ++it) {
            const double t0 = NowSeconds();
            for (uint32_t e = 0; e < n; ++e)
                rt::RenderFrameToAovs(devs[e], scenes[e], cams[e], aovs[e].View());
            cudaDeviceSynchronize();
            naive_ms += (NowSeconds() - t0) * 1.0e3;
        }
        naive_ms /= kIters;
        for (uint32_t e = 0; e < n; ++e) aovs[e].Free();
        devs.clear();

        // ---- BATCHED path: ONE scene + device cameras + base poses; one launch. -
        rt::BatchedSensorSceneDevice scene = rt::BuildBatchedSensorScene(MakeSceneDesc());
        std::vector<math::Transform> base(n);
        for (uint32_t e = 0; e < n; ++e) base[e] = EnvBasePose(e, n, 0.0f);
        math::Transform* d_base = nullptr;
        cudaMalloc(&d_base, n * sizeof(math::Transform));
        cudaMemcpy(d_base, base.data(), n * sizeof(math::Transform), cudaMemcpyHostToDevice);
        rt::PinholeCamera* d_cams = nullptr;
        cudaMalloc(&d_cams, n * sizeof(rt::PinholeCamera));
        cudaMemcpy(d_cams, cams.data(), n * sizeof(rt::PinholeCamera), cudaMemcpyHostToDevice);

        phi::ScatterFkSource fk;
        fk.base_pose = d_base;

        rt::RenderSensorsBatched(scene, fk, d_cams, n, 1u, kRes, kRes);  // warm-up (build)
        cudaDeviceSynchronize();

        double batched_ms = 0.0;
        for (uint32_t it = 0; it < kIters; ++it) {
            // Nudge the base yaw so the steady-state refit genuinely runs.
            for (uint32_t e = 0; e < n; ++e)
                base[e] = EnvBasePose(e, n, 0.01f * static_cast<float>(it + 1u));
            cudaMemcpy(d_base, base.data(), n * sizeof(math::Transform), cudaMemcpyHostToDevice);
            const double t0 = NowSeconds();
            rt::RenderSensorsBatched(scene, fk, d_cams, n, 1u, kRes, kRes);
            cudaDeviceSynchronize();
            batched_ms += (NowSeconds() - t0) * 1.0e3;
        }
        batched_ms /= kIters;

        cudaFree(d_base);
        cudaFree(d_cams);

        const double naive_fps = naive_ms > 0.0 ? 1.0e3 / naive_ms : 0.0;
        const double batched_fps = batched_ms > 0.0 ? 1.0e3 / batched_ms : 0.0;
        const double speedup = batched_ms > 0.0 ? naive_ms / batched_ms : 0.0;
        std::printf("%-6u | %-12.2f %-9.1f | %-12.2f %-9.1f | %.1fx\n",
                    n, naive_ms, naive_fps, batched_ms, batched_fps, speedup);
    }

    std::printf("------------------------------------------------------------------------\n");
    std::printf("Both device-resident (no D2H). naive = N x (BuildTwoLevelScene refit + "
                "RenderFrameToAovs).\nbatched = ONE batched LBVH build/refit + ONE flat "
                "trace over [E*H*W] rays.\n");
    return 0;
}
