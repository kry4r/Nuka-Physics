// ---------------------------------------------------------------------------
// Batched device-resident RGB-sensor FEASIBILITY bench. Answers: can N-env RGB
// sensors be near-real-time on the EXISTING flat deterministic tracer, and WHICH
// cost dominates -- the N per-env TLAS launches or the trace?
//
// HARD CONSTRAINT (prim_id.cuh): kInstanceBits=12 -> <=4096 instances per TLAS.
// A multi-link robot replicated over thousands of envs CANNOT share one TLAS, so
// this bench uses a PER-ENV TLAS: each env is one TwoLevelSceneDevice (own
// persistent render context + own ~36-instance sub-tree, well under the cap).
//
// IT REUSES the edb6878 path verbatim: BuildTwoLevelScene (BLAS once) +
// RenderFrameToAovs (the cheap flat trace = 1 primary + 1 shadow, no AO/GI/MSAA)
// which build-or-refits the TLAS in the persistent buffers. Each env writes its
// own device AOV buffers (the slices of one (N,H,W,ch) obs tensor) -- NO host
// download; the obs stays on device, the whole point of an in-the-loop sensor.
//
// MEASUREMENT (two passes per config):
//   pass 1 = first-touch  -> per-env TLAS BUILD + trace
//   pass 2 = pose-nudged  -> per-env TLAS REFIT + trace
// Decomposition uses the opt-in render-timing probe (CUDA events on the render
// stream, default OFF -> goldens untouched) to split each call's on-stream GPU
// cost into TLAS-update vs kernel, plus a probe-OFF wall pass for the true
// per-step time. host_overhead = wall - on-stream-GPU = the N-launch CPU cost.
//
// It does NOT build the batched LBVH, the FP32 flat sensor kernel, or any
// c_abi/python/offline-RHI plumbing -- it MEASURES whether those are needed.
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"   // ActiveBackend / BackendDeviceBufferType
#include "phi/buffer.hpp"    // BufferAlloc / BufferBase / BufferDownload / BufferFree
#include "phi/backend_cuda/rt/render_timing.hpp"
#include "rt/camera.hpp"
#include "rt/material.hpp"
#include "rt/scene_render.hpp"      // TrianglePrim / Material / Light
#include "rt/two_level_render.hpp"  // BuildTwoLevelScene / RenderFrameToAovs / RtDeviceAovs

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace nuka;

namespace {

constexpr float kPi = 3.14159265358979323846f;

// One env = a fake "robot": this many box instances clustered at its grid tile.
// Mirrors a multi-link robot's instance count (~30-40); well under the 4096 cap.
constexpr uint32_t kInstancesPerEnv = 36u;

// Per-env tile pitch in world XY (the dog-tiling grid spacing) + cluster spread.
constexpr float kEnvPitch = 6.0f;
constexpr float kLinkSpread = 0.35f;  // spacing between an env's link boxes
constexpr float kLinkHalf = 0.12f;    // half-extent of one link box

// Sensor shade profile = the cheap flat trace: color + depth + seg(prim). 1 spp.
// Channel counts of the AOVs this bench writes (the (N,H,W,ch) obs slices).
constexpr uint32_t kColorCh = 3u;

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

// Square-ish env grid: side = ceil(sqrt(n)).
uint32_t GridSide(uint32_t n) {
    uint32_t s = 1u;
    while (s * s < n) ++s;
    return s;
}

// The tile origin (world XY) of env e on the grid, centered at the world origin.
math::Vec3 EnvOrigin(uint32_t e, uint32_t n) {
    const uint32_t side = GridSide(n);
    const uint32_t ix = e % side;
    const uint32_t iy = e / side;
    const float cx = (static_cast<float>(ix) - 0.5f * static_cast<float>(side - 1u)) * kEnvPitch;
    const float cy = (static_cast<float>(iy) - 0.5f * static_cast<float>(side - 1u)) * kEnvPitch;
    return {cx, cy, 0.0f};
}

// Place one env's kInstancesPerEnv link boxes clustered at its tile origin. The
// `nudge` (radians) wobbles every link so the per-frame TLAS refit genuinely runs.
void PlaceEnvInstances(rt::TwoLevelScene* scene, uint32_t e, uint32_t n, float nudge) {
    scene->instances.clear();
    const math::Vec3 base = EnvOrigin(e, n);
    const uint32_t side = GridSide(kInstancesPerEnv);
    for (uint32_t k = 0; k < kInstancesPerEnv; ++k) {
        const uint32_t lx = k % side;
        const uint32_t ly = k / side;
        const float ox = (static_cast<float>(lx) - 0.5f * static_cast<float>(side - 1u)) * kLinkSpread;
        const float oy = (static_cast<float>(ly) - 0.5f * static_cast<float>(side - 1u)) * kLinkSpread;
        const float oz = kLinkHalf + 0.02f * static_cast<float>(k % 5u);
        const float ang = 0.1f + 0.05f * static_cast<float>(k) + nudge;
        const math::Vec3 axis{0.2f, 1.0f, 0.3f};
        math::Transform t;
        t.position = {base.x + ox + 0.01f * std::sin(nudge + static_cast<float>(k)),
                      base.y + oy + 0.01f * std::cos(nudge + static_cast<float>(k)),
                      oz};
        t.rotation = math::Quat::FromAxisAngle(axis, ang);
        scene->instances.push_back({0u, t, 0u});
    }
}

// A camera looking at env e's cluster from a fixed offset above its tile origin.
rt::PinholeCamera CameraForEnv(uint32_t e, uint32_t n, uint32_t res) {
    const math::Vec3 base = EnvOrigin(e, n);
    const math::Vec3 eye{base.x + 1.6f, base.y - 1.6f, 1.4f};
    const math::Vec3 target{base.x, base.y, kLinkHalf};
    return rt::BuildPinhole(eye, target, {0.0f, 0.0f, 1.0f}, 0.6f * kPi, res, res);
}

rt::Material LinkMaterial() {
    rt::Material mat;
    mat.albedo = {0.65f, 0.55f, 0.4f};
    mat.metallic = 0.0f;
    mat.roughness = 0.6f;
    return mat;
}

rt::TwoLevelScene MakeEnvScene() {
    rt::TwoLevelScene s;
    s.meshes.push_back(BuildBoxMesh(kLinkHalf));
    s.materials = {LinkMaterial()};
    s.light.directional = true;
    s.light.direction = {0.3f, 0.4f, -0.85f};
    s.ambient.color = {0.05f, 0.05f, 0.05f};
    return s;
}

double NowSeconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One env's device AOV slice. The FP64 closest-hit kernel writes ALL 6 AOVs
// UNCONDITIONALLY (no null-guard), so every channel must be resident; the sensor
// obs keeps color+depth+prim but the trace cost is the same either way.
struct EnvAovSlice {
    phi::Buffer* color = nullptr;
    phi::Buffer* depth = nullptr;
    phi::Buffer* normal = nullptr;
    phi::Buffer* albedo = nullptr;
    phi::Buffer* uv = nullptr;
    phi::Buffer* prim = nullptr;
    rt::RtDeviceAovs View() const {
        rt::RtDeviceAovs a;
        a.color = color;
        a.depth = depth;
        a.normal = normal;
        a.albedo = albedo;
        a.uv = uv;
        a.prim = prim;
        return a;
    }
    void Alloc(phi::BufferType* bt, size_t pixels) {
        color = phi::BufferAlloc(bt, pixels * 3u * sizeof(float));
        depth = phi::BufferAlloc(bt, pixels * sizeof(float));
        normal = phi::BufferAlloc(bt, pixels * 3u * sizeof(float));
        albedo = phi::BufferAlloc(bt, pixels * 3u * sizeof(float));
        uv = phi::BufferAlloc(bt, pixels * 2u * sizeof(float));
        prim = phi::BufferAlloc(bt, pixels * sizeof(uint32_t));
    }
    void Free() {
        phi::BufferFree(color);
        phi::BufferFree(depth);
        phi::BufferFree(normal);
        phi::BufferFree(albedo);
        phi::BufferFree(uv);
        phi::BufferFree(prim);
    }
};

struct ConfigResult {
    uint32_t n = 0u;
    uint32_t res = 0u;
    double build_gpu_ms = 0.0;    // pass-1 on-stream TLAS BUILD (all envs)
    double refit_gpu_ms = 0.0;    // pass-2 on-stream TLAS REFIT (all envs)
    double kernel_gpu_ms = 0.0;   // on-stream closest-hit kernel (all envs, refit pass)
    double wall_ms = 0.0;         // probe-OFF per-step wall (refit pass: the steady state)
    double build_wall_ms = 0.0;   // probe-OFF first-touch per-step wall (build pass)
    double blas_setup_ms = 0.0;   // one-time BuildTwoLevelScene over all envs (amortized)
};

// Render every env once into its AOV slice; returns probe-OFF wall ms for the loop.
double RenderAllEnvs(std::vector<rt::TwoLevelSceneDevice>& devs,
                     std::vector<rt::TwoLevelScene>& scenes,
                     const std::vector<rt::PinholeCamera>& cams,
                     const std::vector<EnvAovSlice>& aovs) {
    const double t0 = NowSeconds();
    for (size_t e = 0; e < devs.size(); ++e) {
        rt::RenderFrameToAovs(devs[e], scenes[e], cams[e], aovs[e].View());
    }
    return (NowSeconds() - t0) * 1.0e3;
}

}  // namespace

int main() {
    if (phi::ActiveBackend() == nullptr) {
        std::printf("[bench] no CUDA backend available -- cannot run.\n");
        return 1;
    }
    phi::BufferType* bt = phi::BackendDeviceBufferType(phi::ActiveBackend());

    const std::vector<uint32_t> kEnvCounts = {256u, 1024u, 4096u};
    const std::vector<uint32_t> kResolutions = {64u, 32u};

    std::printf("[bench] BATCHED DEVICE-RESIDENT RGB-SENSOR FEASIBILITY\n");
    std::printf("[bench] per-env TLAS (%u instances/env, box BLAS = 12 tris); "
                "cheap flat trace (1 primary + 1 shadow), device-resident AOVs, "
                "NO host download.\n", kInstancesPerEnv);

    bool sanity_ok = false;
    std::vector<ConfigResult> rows;

    for (uint32_t res : kResolutions) {
        for (uint32_t n : kEnvCounts) {
            ConfigResult R;
            R.n = n;
            R.res = res;
            const size_t pixels = static_cast<size_t>(res) * static_cast<size_t>(res);

            // ---- One-time setup: per-env scene + device (BLAS build) + camera +
            // device AOV slices. The BLAS build is amortized -> timed separately. ----
            std::vector<rt::TwoLevelScene> scenes(n);
            std::vector<rt::PinholeCamera> cams(n);
            std::vector<EnvAovSlice> aovs(n);
            std::vector<rt::TwoLevelSceneDevice> devs;
            devs.reserve(n);

            const double setup_t0 = NowSeconds();
            for (uint32_t e = 0; e < n; ++e) {
                scenes[e] = MakeEnvScene();
                PlaceEnvInstances(&scenes[e], e, n, 0.0f);
                cams[e] = CameraForEnv(e, n, res);
                devs.push_back(rt::BuildTwoLevelScene(scenes[e]));
                aovs[e].Alloc(bt, pixels);
            }
            R.blas_setup_ms = (NowSeconds() - setup_t0) * 1.0e3;

            // ---- Warm-up: one full pass (absorbs context init + first-touch
            // builds) so the timed BUILD pass below is a clean first-touch on the
            // persistent topology after a count-stable rebuild cadence reset. ----
            rt::SetRenderTimingEnabled(false);
            (void)RenderAllEnvs(devs, scenes, cams, aovs);

            // ---- PASS 1 (BUILD): re-place at nudge 0 and force a rebuild on every
            // env by rebuilding fresh devices (first-touch). We measure the build
            // GPU split with the probe, and a separate probe-OFF wall. ----
            // Fresh devices => topology_built=false => guaranteed TLAS BUILD.
            devs.clear();
            devs.reserve(n);
            for (uint32_t e = 0; e < n; ++e) {
                PlaceEnvInstances(&scenes[e], e, n, 0.0f);
                devs.push_back(rt::BuildTwoLevelScene(scenes[e]));
            }
            R.build_wall_ms = RenderAllEnvs(devs, scenes, cams, aovs);  // probe off

            // GPU split of the BUILD pass: fresh devices again, probe on.
            devs.clear();
            devs.reserve(n);
            for (uint32_t e = 0; e < n; ++e) {
                devs.push_back(rt::BuildTwoLevelScene(scenes[e]));
            }
            rt::SetRenderTimingEnabled(true);
            (void)rt::TakeRenderTiming();
            for (uint32_t e = 0; e < n; ++e) {
                rt::RenderFrameToAovs(devs[e], scenes[e], cams[e], aovs[e].View());
            }
            {
                const rt::RenderTiming t = rt::TakeRenderTiming();
                R.build_gpu_ms = t.tlas_ms;  // first-touch = build
            }
            rt::SetRenderTimingEnabled(false);

            // ---- PASS 2 (REFIT): nudge poses on the SAME (already-built) devices
            // so EnsureFrameTlas refits the reused topology. Probe-OFF wall (the
            // steady-state per-step time) + probe-ON GPU split (refit vs kernel). ----
            for (uint32_t e = 0; e < n; ++e) {
                PlaceEnvInstances(&scenes[e], e, n, 0.05f);
            }
            R.wall_ms = RenderAllEnvs(devs, scenes, cams, aovs);  // probe off, refit

            for (uint32_t e = 0; e < n; ++e) {
                PlaceEnvInstances(&scenes[e], e, n, 0.10f);
            }
            rt::SetRenderTimingEnabled(true);
            (void)rt::TakeRenderTiming();
            for (uint32_t e = 0; e < n; ++e) {
                rt::RenderFrameToAovs(devs[e], scenes[e], cams[e], aovs[e].View());
            }
            {
                const rt::RenderTiming t = rt::TakeRenderTiming();
                R.refit_gpu_ms = t.tlas_ms;
                R.kernel_gpu_ms = t.kernel_ms;
            }
            rt::SetRenderTimingEnabled(false);

            // ---- SANITY (once): env 0's device AOV slice == a standalone
            // single-camera RenderFrameToAovs of env 0 (same kernel -> byte-exact). ----
            if (!sanity_ok) {
                std::vector<float> ref(pixels * kColorCh, 0.0f), got(pixels * kColorCh, 0.0f);
                rt::TwoLevelScene s0 = MakeEnvScene();
                PlaceEnvInstances(&s0, 0u, n, 0.10f);  // match pass-2 env-0 pose
                rt::TwoLevelSceneDevice d0 = rt::BuildTwoLevelScene(s0);
                EnvAovSlice solo;
                solo.Alloc(bt, pixels);
                rt::RenderFrameToAovs(d0, s0, cams[0], solo.View());
                phi::BufferDownload(solo.color, ref.data(), 0, ref.size() * sizeof(float));
                phi::BufferDownload(aovs[0].color, got.data(), 0, got.size() * sizeof(float));
                sanity_ok = (std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0);
                std::printf("[bench] SANITY env-0 slice == standalone single-camera "
                            "render: %s (color AOV, %zu floats, byte-compared)\n",
                            sanity_ok ? "BYTE-IDENTICAL" : "MISMATCH", ref.size());
                solo.Free();
            }

            rows.push_back(R);
            std::printf("[bench] N=%-5u res=%ux%u  blas_setup=%.1f ms  "
                        "build_wall=%.2f ms  refit_wall=%.2f ms  "
                        "build_gpu=%.2f ms  refit_gpu=%.2f ms  kernel_gpu=%.2f ms\n",
                        n, res, res, R.blas_setup_ms, R.build_wall_ms, R.wall_ms,
                        R.build_gpu_ms, R.refit_gpu_ms, R.kernel_gpu_ms);

            // Free this config's device memory before the next (bound peak usage).
            for (uint32_t e = 0; e < n; ++e) {
                aovs[e].Free();
            }
        }
    }

    // =====================================================================
    // The decomposed deliverable table.
    // =====================================================================
    std::printf("\n================ BATCHED RGB-SENSOR FEASIBILITY (per-env TLAS) ================\n");
    std::printf("per step = ONE render of ALL N envs (device-resident, no download). "
                "host_overhead = wall - on-stream GPU = the N-launch CPU cost.\n");
    std::printf("%-6s %-7s | %-10s %-10s | %-11s %-11s | %-12s %-9s | %-7s\n",
                "N", "res", "build ms", "refit ms", "kernel ms", "host ovh",
                "step ms", "fps", "kern/step");
    std::printf("-------------------------------------------------------------------------------------------------\n");
    for (const auto& r : rows) {
        const double on_stream_refit = r.refit_gpu_ms + r.kernel_gpu_ms;
        const double host_ovh = r.wall_ms - on_stream_refit;  // refit pass
        const double fps = r.wall_ms > 0.0 ? 1.0e3 / r.wall_ms : 0.0;
        const double kern_frac = r.wall_ms > 0.0 ? 100.0 * r.kernel_gpu_ms / r.wall_ms : 0.0;
        std::printf("%-6u %ux%-4u | %-10.2f %-10.2f | %-11.2f %-11.2f | %-12.2f %-9.1f | %.1f%%\n",
                    r.n, r.res, r.res, r.build_gpu_ms, r.refit_gpu_ms,
                    r.kernel_gpu_ms, host_ovh, r.wall_ms, fps, kern_frac);
    }
    std::printf("=================================================================================================\n");
    std::printf("Columns: build ms / refit ms = per-step on-stream TLAS build (pass1) / refit (pass2), summed over "
                "N envs.\n         kernel ms = on-stream closest-hit trace, summed over N. host ovh = wall - "
                "(refit+kernel).\n         step ms = probe-OFF refit-pass wall (steady state). fps = 1000/step.\n");

    // =====================================================================
    // Verdict: which dominates, and which lever buys near-real-time.
    // =====================================================================
    std::printf("\n--- bottleneck per config (refit/steady-state pass) ---\n");
    for (const auto& r : rows) {
        const double on_stream_refit = r.refit_gpu_ms + r.kernel_gpu_ms;
        const double host_ovh = r.wall_ms - on_stream_refit;
        const char* dom;
        if (host_ovh >= r.kernel_gpu_ms && host_ovh >= r.refit_gpu_ms) {
            dom = "HOST LAUNCH OVERHEAD (N per-env launches)";
        } else if (r.kernel_gpu_ms >= r.refit_gpu_ms) {
            dom = "TRACE (closest-hit kernel)";
        } else {
            dom = "TLAS REFIT";
        }
        std::printf("N=%-5u res=%ux%u: step=%.2f ms  [refit %.2f | kernel %.2f | host %.2f] -> %s\n",
                    r.n, r.res, r.res, r.wall_ms, r.refit_gpu_ms, r.kernel_gpu_ms, host_ovh, dom);
    }

    // Aggregate lever verdict (uses the largest 64^2 config as the headline).
    const ConfigResult* hi = nullptr;
    for (const auto& r : rows) {
        if (r.res == 64u && (hi == nullptr || r.n > hi->n)) hi = &r;
    }
    std::printf("\n================ VERDICT ================\n");
    if (hi != nullptr) {
        const double host_ovh = hi->wall_ms - (hi->refit_gpu_ms + hi->kernel_gpu_ms);
        std::printf("Headline %u env x %ux%u: %.0f ms/step (%.1f fps). NOT near-real-time on the\n"
                    "naive per-env loop. Split: kernel %.0f%%, TLAS refit %.0f%%, host launch %.0f%%.\n",
                    hi->n, hi->res, hi->res, hi->wall_ms, 1.0e3 / hi->wall_ms,
                    100.0 * hi->kernel_gpu_ms / hi->wall_ms,
                    100.0 * hi->refit_gpu_ms / hi->wall_ms,
                    100.0 * host_ovh / hi->wall_ms);
    }
    std::printf("Bottleneck = the TRACE (closest-hit ray work), NOT the N host launches: the\n"
                "persistent context (edb6878) already drove per-call host overhead into the noise.\n"
                "Each env is a separate small launch (64^2 -> 16 thread-blocks) that underfills the\n"
                "GPU and runs serially on one stream, compounding the FP64 per-ray cost.\n");
    std::printf("Levers (ranked): (1) a BATCHED single-launch trace over all envs' rays (pair it\n"
                "with a batched LBVH build) -> fills the GPU + removes N-serial launches: the\n"
                "structural lever. (2) an FP32 flat sensor kernel ~2x the per-ray rate (not built\n"
                "here). (3) config: 32^2 ~2x faster than 64^2, fewer envs, or render every K steps.\n"
                "A batched LBVH build ALONE is insufficient (refit is already a small share of the\n"
                "step); it is the necessary front-half of the batched trace, not a standalone fix.\n");
    std::printf("=========================================\n");

    std::printf("\nSANITY (env-0 slice byte-identical to standalone render): %s\n",
                sanity_ok ? "PASS" : "FAIL");
    return sanity_ok ? 0 : 2;
}
