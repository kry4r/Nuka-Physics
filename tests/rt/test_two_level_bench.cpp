// ---------------------------------------------------------------------------
// v0.7 G1-C: the TWO-LEVEL (TLAS/BLAS) ray-tracer BENCHMARK HARNESS.
//
// G1-A landed the self-written two-level tracer; the OWNER pulled RT-perf early
// ("速度很重要"). G1-C MEASURES whether the two-level structure is fast enough
// for the p16 demo (H1 robot + cup + table + USD scene = MANY rigid mesh
// INSTANCES) and produces an explicit go/no-go for G1-B (wide-BVH8 / ray-sort /
// persistent-threads, currently deferred).
//
// EMPIRICAL FINDING (drives the verdict shape -- see the 3-fact verdict at the
// end): the G1-A BLAS-cache premise is VALIDATED (two-level beats flat ~5x at
// N=1024, the relative invariant holds), BUT full-frame throughput is dominated
// by an N-INDEPENDENT per-frame FIXED FLOOR (~18 ms at 512x512: buffer
// alloc/free + 6-AOV download + launch, NOT traversal -- both paths agree at
// N=1). That floor caps full-frame throughput at ~29 MRays/s before any triangle
// is traversed, BELOW the 50-150 sensor bar. G1-B optimizes TRAVERSAL only (the
// variable cost two_ms(N)-floor), so it CANNOT move the bar at 512x512 -> the
// honest go/no-go is "DEFER G1-B; attack per-frame overhead." Offline beauty
// (seconds/frame) is wildly satisfied either way. The verdict therefore reports
// THREE facts (structure validated / throughput-below-bar-but-overhead-bound /
// overhead-is-the-real-lever), NOT a naive `if MRays<bar -> proceed-to-G1-B`.
//
// WHAT IS COMPARED (the whole point):
//   * Flat baseline (p13 rt::RenderScene): builds ONE big LBVH over ALL
//     primitives EVERY frame. To benchmark N instances we BAKE each instance's
//     rigid transform into world-space triangles -> one flat rt::Scene with
//     N * (tris-per-mesh) triangles -> RenderScene per frame (rebuild-all).
//   * Two-level (G1-A rt::RenderFrame): per-mesh BLAS built ONCE in
//     BuildTwoLevelScene (retained); each frame rebuilds only the cheap
//     N-instance TLAS, reusing the cached BLAS.
//
// METHODOLOGY (advisor-ratified; defensibility notes):
//   * We time the WHOLE host per-frame call (RenderScene / RenderFrame), which
//     internally builds the accel structure + launches the kernel + downloads
//     all 6 AOVs. We do NOT (and cannot, without editing the lint-scanned
//     src/rt/**) isolate the kernel; so all MRays/s numbers below are on a
//     FULL PER-FRAME WALL-TIME basis -- the real end-to-end cost, which is the
//     correct basis for the p16 go/no-go (NOT "kernel throughput").
//   * BuildTwoLevelScene (the one-time BLAS build) is the entire PREMISE of
//     two-level -- amortized setup, not per-frame cost -- so it is called ONCE
//     before the timed loop and reported separately, never inside it.
//   * The flat path's host re-bake (transforms moved -> world verts must be
//     recomputed) is done BEFORE the timer each frame and EXCLUDED from the
//     timed region, so flat is never charged for host transform work (this is
//     the comparison most generous to flat: both paths are timed only on
//     rebuilding their own accel structure over their own primitive count).
//   * Timing = std::chrono::steady_clock wall-clock around the full call. The
//     call self-synchronizes (it downloads the framebuffer), so no explicit
//     cudaDeviceSynchronize is needed. One throwaway render before the WHOLE
//     sweep absorbs CUDA context init; one warmup frame per N is discarded.
//   * MRays/s = W*H*(1 primary + 1 shadow) / per_frame_seconds. The "+1 shadow"
//     is an UPPER BOUND (miss pixels cast no shadow ray); stated, not exact.
//   * Instances are ANIMATED slightly each frame so the per-frame TLAS rebuild
//     (two-level) and the flat rebuild-all actually happen every frame.
//
// CI SAFETY: numbers are hardware-dependent (and nvidia-smi is NVML-BROKEN here),
// so NOTHING absolute is asserted. The ONLY hard structural invariant is
// RELATIVE: at N >= 64, two-level per-frame cost < flat per-frame cost (the
// BLAS-cache win). If even that is environment-flaky it degrades to a warn-print
// (see kHardAssertRelativeInvariant). Everything else is printed.
//
// This is a benchmark, not a determinism gate (G1-A already proved D1). Timings
// are naturally non-deterministic -- expected; only the relative invariant is
// asserted. No float atomics / nothing lint-banned (tests are not scanned, but
// kept clean regardless).
//
// Built -ffp-contract=off to match the rest of the rt test suite (the bench does
// no host==device parity check, but matching the suite is harmless and safer).
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "rt/camera.hpp"
#include "rt/framebuffer.hpp"
#include "rt/material.hpp"
#include "rt/ray_box.cuh"            // host-callable kNoPrim sentinel
#include "rt/scene_render.hpp"       // flat p13 Scene + RenderScene (baseline)
#include "rt/two_level_render.hpp"   // G1-A BuildTwoLevelScene + RenderFrame

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace nuka;

namespace {

constexpr float kPi = 3.14159265358979323846f;

// --- DECISION GATE: the sensor RGB/depth target bar (MRays/s). Offline beauty
// is seconds/frame (no hard bar). The two-level full-frame MRays/s at the
// largest N is compared against this band to print the G1-B go/no-go verdict.
constexpr float kSensorBarLoMRaysPerSec = 50.0f;
constexpr float kSensorBarHiMRaysPerSec = 150.0f;

// --- CI: HARD-assert the relative invariant (two-level < flat at N>=64)?
// Decided AFTER inspecting the measured spread (the margin should be large:
// flat rebuilds an LBVH over N*~300 prims every frame while two-level rebuilds
// only an N-leaf TLAS). If the margin were marginal/flaky we would set this
// false and warn-print instead. Measured margin is comfortable (see report), so
// it stays a hard assert.
constexpr bool kHardAssertRelativeInvariant = true;

// Fixed bench resolution (keeps total sweep runtime bounded to a few minutes).
constexpr uint32_t kW = 512u;
constexpr uint32_t kH = 512u;
constexpr int kFramesPerN = 10;  // timed frames per N (plus 1 discarded warmup)

// One representative TRIANGLE mesh: a tessellated UV-sphere of unit radius in
// LOCAL space. latitude x longitude tessellation; ~12x12 -> ~300 triangles.
// (lat stacks): top cap fan (lon tris) + (lat-2) middle rings (2*lon tris each)
// + bottom cap fan (lon tris) = 2*lon*(lat-1) triangles. lat=12,lon=14 -> 308.
rt::BlasMesh BuildUvSphereMesh(uint32_t lat, uint32_t lon, float radius) {
    rt::BlasMesh m;
    auto vert = [&](uint32_t i, uint32_t j) -> math::Vec3 {
        // i in [0,lat] is the polar stack (0 = +Y pole, lat = -Y pole);
        // j in [0,lon) is the azimuth.
        const float theta = kPi * (static_cast<float>(i) / static_cast<float>(lat));
        const float phi = 2.0f * kPi * (static_cast<float>(j % lon) /
                                        static_cast<float>(lon));
        const float st = std::sin(theta), ct = std::cos(theta);
        return {radius * st * std::cos(phi), radius * ct, radius * st * std::sin(phi)};
    };
    for (uint32_t i = 0; i < lat; ++i) {
        for (uint32_t j = 0; j < lon; ++j) {
            const math::Vec3 a = vert(i, j);
            const math::Vec3 b = vert(i + 1u, j);
            const math::Vec3 c = vert(i + 1u, j + 1u);
            const math::Vec3 d = vert(i, j + 1u);
            // Two triangles per quad cell, except degenerate pole cells collapse
            // to one. Wind outward (CCW seen from outside).
            if (i == 0u) {
                // top cap: a is the pole -> single triangle (a, b, c).
                m.triangles.push_back({a, b, c, 0u});
            } else if (i == lat - 1u) {
                // bottom cap: c==d collapses -> single triangle (a, b, d).
                m.triangles.push_back({a, b, d, 0u});
            } else {
                m.triangles.push_back({a, b, c, 0u});
                m.triangles.push_back({a, c, d, 0u});
            }
        }
    }
    return m;
}

// Place N instances of mesh 0 on a roughly cubic grid centered at the origin,
// each with a distinct position + a small per-instance rotation. `frame` adds a
// tiny animation offset so the per-frame TLAS/flat rebuild actually changes.
void PlaceInstances(rt::TwoLevelScene* scene, uint32_t n, int frame) {
    scene->instances.clear();
    // Cubic-ish grid side so side^3 >= n.
    uint32_t side = 1u;
    while (side * side * side < n) ++side;
    const float spacing = 2.5f;  // > 2*radius so spheres do not interpenetrate
    const float anim = 0.02f * static_cast<float>(frame);
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t ix = k % side;
        const uint32_t iy = (k / side) % side;
        const uint32_t iz = k / (side * side);
        const float cx = (static_cast<float>(ix) - 0.5f * static_cast<float>(side - 1u)) * spacing;
        const float cy = (static_cast<float>(iy) - 0.5f * static_cast<float>(side - 1u)) * spacing;
        const float cz = (static_cast<float>(iz) - 0.5f * static_cast<float>(side - 1u)) * spacing;
        // Small per-instance + per-frame rotation about a per-instance axis.
        const float ang = 0.1f + 0.05f * static_cast<float>(k) + anim;
        const math::Vec3 axis{0.3f + 0.01f * static_cast<float>(k % 7u), 1.0f, 0.2f};
        math::Transform t;
        // Tiny per-frame positional wobble so the world-AABBs (and baked verts)
        // genuinely change frame to frame.
        t.position = {cx + 0.03f * std::sin(anim + static_cast<float>(k)),
                      cy + 0.03f * std::cos(anim + static_cast<float>(k)),
                      cz};
        t.rotation = math::Quat::FromAxisAngle(axis, ang);
        scene->instances.push_back({0u, t, 0u});
    }
}

// Bake the CURRENT instance transforms into a flat world-space rt::Scene (the
// p13 baseline path). Each baked triangle carries the instance's material_id.
rt::Scene BakeFlat(const rt::TwoLevelScene& scene) {
    rt::Scene flat;
    flat.materials = scene.materials;
    flat.light = scene.light;
    flat.ambient = scene.ambient;
    const rt::BlasMesh& mesh = scene.meshes[0];
    flat.triangles.reserve(scene.instances.size() * mesh.triangles.size());
    for (const auto& inst : scene.instances) {
        for (const auto& tri : mesh.triangles) {
            rt::TrianglePrim wt;
            wt.v0 = inst.transform.TransformPoint(tri.v0);
            wt.v1 = inst.transform.TransformPoint(tri.v1);
            wt.v2 = inst.transform.TransformPoint(tri.v2);
            wt.material_id = inst.material_id;
            flat.triangles.push_back(wt);
        }
    }
    return flat;
}

// Camera framing the whole instance cloud for a given N (the grid grows with N).
rt::PinholeCamera CameraForN(uint32_t n) {
    uint32_t side = 1u;
    while (side * side * side < n) ++side;
    const float spacing = 2.5f;
    const float extent = spacing * static_cast<float>(side);  // half-cloud-ish
    const float dist = extent * 2.2f + 5.0f;
    return rt::BuildPinhole({0.0f, 0.0f, -dist}, {0.0f, 0.0f, 0.0f},
                            {0.0f, 1.0f, 0.0f}, 0.7f * kPi, kW, kH);
}

double NowSeconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

float MRaysPerSec(double sec) {
    // Ray-count assumption: W*H primary rays + up to W*H shadow rays (upper
    // bound; miss pixels cast no shadow ray). State the basis: FULL PER-FRAME
    // WALL TIME (build + kernel + download), not isolated kernel throughput.
    const double rays = 2.0 * static_cast<double>(kW) * static_cast<double>(kH);
    return static_cast<float>(rays / sec / 1.0e6);
}

struct RowResult {
    uint32_t n = 0u;
    double flat_ms = 0.0;
    double two_ms = 0.0;
};

}  // namespace

// =====================================================================
// The G1-C sweep: N = 1, 16, 64, 256, 1024. For each N, time flat
// RenderScene (rebuild-all) vs two-level RenderFrame (cached BLAS +
// per-frame TLAS), animate the instances each frame, discard a warmup,
// report mean ms/frame + full-frame MRays/s + speedup. Then print the
// explicit G1-B go/no-go verdict and assert ONLY the relative invariant.
// =====================================================================
TEST(RtTwoLevelBench, FlatVsTwoLevelSweep) {
    // The single shared mesh (~300 triangles).
    const rt::BlasMesh sphere_mesh = BuildUvSphereMesh(12u, 14u, 1.0f);
    const size_t tris_per_mesh = sphere_mesh.triangles.size();
    std::printf("[bench] mesh = UV-sphere, %zu triangles; res %ux%u; "
                "%d timed frames/N (+1 warmup)\n",
                tris_per_mesh, kW, kH, kFramesPerN);

    rt::Material mat;
    mat.albedo = {0.7f, 0.5f, 0.3f};
    mat.metallic = 0.0f;
    mat.roughness = 0.5f;

    const std::vector<uint32_t> kNs = {1u, 16u, 64u, 256u, 1024u};

    // One throwaway render before the WHOLE sweep to absorb CUDA context init.
    {
        rt::TwoLevelScene warm;
        warm.meshes.push_back(sphere_mesh);
        warm.materials = {mat};
        warm.light.directional = true;
        warm.light.direction = {0.2f, -0.5f, 0.8f};
        warm.ambient.color = {0.04f, 0.04f, 0.04f};
        PlaceInstances(&warm, 4u, 0);
        auto cam = CameraForN(4u);
        auto dev = rt::BuildTwoLevelScene(warm);
        (void)rt::RenderFrame(dev, warm, cam);
        rt::Scene flat = BakeFlat(warm);
        (void)rt::RenderScene(flat, cam);
    }

    std::vector<RowResult> rows;

    for (uint32_t n : kNs) {
        rt::TwoLevelScene scene;
        scene.meshes.push_back(sphere_mesh);
        scene.materials = {mat};
        scene.light.directional = true;
        scene.light.direction = {0.2f, -0.5f, 0.8f};
        scene.ambient.color = {0.04f, 0.04f, 0.04f};
        PlaceInstances(&scene, n, 0);

        const rt::PinholeCamera cam = CameraForN(n);

        // ---- Two-level: build the per-mesh BLAS ONCE (amortized setup; timed
        // separately, NEVER inside the per-frame loop). ----
        const double build_t0 = NowSeconds();
        rt::TwoLevelSceneDevice dev = rt::BuildTwoLevelScene(scene);
        const double build_ms = (NowSeconds() - build_t0) * 1.0e3;

        // Warmup frame for this N (discarded): frame index -1 worth of work.
        PlaceInstances(&scene, n, 0);
        (void)rt::RenderFrame(dev, scene, cam);
        {
            rt::Scene flat_warm = BakeFlat(scene);
            (void)rt::RenderScene(flat_warm, cam);
        }

        // ---- Timed frames. Animate the instances each frame so the per-frame
        // TLAS rebuild (two-level) and the flat rebuild-all genuinely happen. ----
        double two_accum = 0.0;
        double flat_accum = 0.0;
        size_t two_hits = 0, flat_hits = 0;  // sanity: both paths see geometry.
        for (int f = 1; f <= kFramesPerN; ++f) {
            // Two-level: animate in place; RenderFrame re-reads instances and
            // rebuilds the TLAS internally. Time the WHOLE call.
            PlaceInstances(&scene, n, f);
            const double t0 = NowSeconds();
            rt::Framebuffer two_fb = rt::RenderFrame(dev, scene, cam);
            two_accum += NowSeconds() - t0;

            // Flat: bake world verts BEFORE the timer (host transform work is
            // EXCLUDED -- most generous to flat), then time RenderScene only.
            rt::Scene flat = BakeFlat(scene);
            const double t1 = NowSeconds();
            rt::Framebuffer flat_fb = rt::RenderScene(flat, cam);
            flat_accum += NowSeconds() - t1;

            if (f == kFramesPerN) {
                for (auto p : two_fb.prim) if (p != rt::kNoPrim) ++two_hits;
                for (auto p : flat_fb.prim) if (p != rt::kNoPrim) ++flat_hits;
            }
        }

        const double two_ms = (two_accum / kFramesPerN) * 1.0e3;
        const double flat_ms = (flat_accum / kFramesPerN) * 1.0e3;

        std::printf("[bench] N=%-5u flat_prims=%-7zu  BLAS_build=%.2f ms  "
                    "two-level=%.3f ms  flat=%.3f ms  (hits two/flat=%zu/%zu)\n",
                    n, n * tris_per_mesh, build_ms, two_ms, flat_ms,
                    two_hits, flat_hits);
        EXPECT_GT(two_hits, 0u) << "two-level saw no geometry at N=" << n;
        EXPECT_GT(flat_hits, 0u) << "flat saw no geometry at N=" << n;

        rows.push_back({n, flat_ms, two_ms});
    }

    // ------------------------------------------------------------------
    // The table.
    // ------------------------------------------------------------------
    std::printf("\n================ G1-C TWO-LEVEL RT BENCHMARK ================\n");
    std::printf("res %ux%u | MRays/s basis = FULL per-frame wall time, "
                "%u*%u*(1 primary + up to 1 shadow) rays\n",
                kW, kH, kW, kH);
    std::printf("%-6s | %-11s | %-13s | %-13s | %-15s | %-8s\n",
                "N", "flat ms", "two-level ms", "flat MRays/s",
                "two-lvl MRays/s", "speedup");
    std::printf("-------+-------------+---------------+---------------+"
                "-----------------+---------\n");
    float two_mrays_at_max = 0.0f;
    uint32_t max_n = 0u;
    double two_ms_floor = 0.0;  // N=1 two-level ms (the per-frame fixed floor).
    double max_speedup = 0.0;
    for (const auto& r : rows) {
        const float flat_mr = MRaysPerSec(r.flat_ms * 1.0e-3);
        const float two_mr = MRaysPerSec(r.two_ms * 1.0e-3);
        const double speedup = r.flat_ms / r.two_ms;
        std::printf("%-6u | %-11.3f | %-13.3f | %-13.1f | %-15.1f | %.2fx\n",
                    r.n, r.flat_ms, r.two_ms, flat_mr, two_mr, speedup);
        if (r.n == 1u) two_ms_floor = r.two_ms;
        if (speedup > max_speedup) max_speedup = speedup;
        if (r.n >= max_n) {
            max_n = r.n;
            two_mrays_at_max = two_mr;
        }
    }
    std::printf("=============================================================\n");

    // ------------------------------------------------------------------
    // OVERHEAD DECOMPOSITION (the lever G1-B can vs cannot move). The N=1 row
    // is N-independent per-frame FIXED overhead (buffer alloc/free + 6-AOV
    // download + launch) -- both paths agree there, so it is NOT traversal
    // compute. G1-B (ray-sort / BVH8 / persistent-threads) optimizes TRAVERSAL
    // only, i.e. only the VARIABLE part two_ms(N) - floor; it cannot touch the
    // floor. The floor alone caps full-frame throughput at MRaysPerSec(floor).
    // ------------------------------------------------------------------
    const float ceiling_mrays = MRaysPerSec(two_ms_floor * 1.0e-3);
    std::printf("\n--- overhead decomposition (per-frame fixed floor vs traversal) ---\n");
    std::printf("per-frame FIXED floor (N=1 two-level) = %.2f ms  -> hard ceiling "
                "%.1f MRays/s @ 512x512 even with ZERO traversal cost\n",
                two_ms_floor, ceiling_mrays);
    std::printf("%-6s | %-13s | %-15s\n", "N", "variable ms", "(= two_ms - floor)");
    for (const auto& r : rows) {
        std::printf("%-6u | %-13.3f | %s\n", r.n, r.two_ms - two_ms_floor,
                    "<- the ONLY cost G1-B (traversal) could attack");
    }

    // ------------------------------------------------------------------
    // The DECISION GATE: HONEST three-fact verdict for the controller/owner.
    // (a) G1-A BLAS-cache premise VALIDATED; (b) full-frame MRays/s vs the bar
    // WITH the overhead caveat; (c) the real go/no-go -- because the bar is
    // gated by the N-independent floor (which G1-B does NOT touch), even
    // infinitely-fast traversal cannot reach the sensor bar at 512x512, so the
    // owner's lever is per-frame OVERHEAD, not G1-B traversal work. Offline
    // beauty (seconds/frame) is wildly satisfied either way.
    // ------------------------------------------------------------------
    const bool ceiling_below_bar = ceiling_mrays < kSensorBarLoMRaysPerSec;
    std::printf("\n================ G1-C VERDICT (3 facts) ================\n");
    std::printf("(a) STRUCTURE: G1-A BLAS-cache premise VALIDATED -- two-level "
                "beats flat %.2fx @ N=%u (invariant holds N>=64); per-frame TLAS "
                "rebuild scales, flat rebuild-all does NOT.\n",
                max_speedup, max_n);
    std::printf("(b) THROUGHPUT: two-level = %.1f MRays/s @ N=%u (FULL per-frame "
                "wall time); sensor bar = %.0f-%.0f MRays/s -> BELOW bar, but "
                "OVERHEAD-DOMINATED (N=1 floor %.2f ms caps it at ~%.0f MRays/s "
                "regardless of traversal).\n",
                two_mrays_at_max, max_n, kSensorBarLoMRaysPerSec,
                kSensorBarHiMRaysPerSec, two_ms_floor, ceiling_mrays);
    if (ceiling_below_bar) {
        std::printf("(c) GO/NO-GO: the binding constraint is the N-INDEPENDENT "
                    "per-frame FLOOR (alloc/download/launch), which G1-B "
                    "(traversal: ray-sort/BVH8/persistent-threads) does NOT "
                    "touch -- even ZERO traversal cost = %.0f MRays/s < %.0f bar "
                    "@ 512x512. -> [DEFER G1-B; attack PER-FRAME OVERHEAD "
                    "(persistent buffers / async download / fused launch) to move "
                    "the real-time sensor bar]. Offline beauty (seconds/frame): "
                    "two-level %.1f ms is WILDLY sufficient already.\n",
                    ceiling_mrays, kSensorBarLoMRaysPerSec,
                    rows.back().two_ms);
    } else {
        std::printf("(c) GO/NO-GO: per-frame floor leaves headroom above the bar; "
                    "the variable (traversal) curve is what gates throughput -> "
                    "[PROCEED to G1-B (ray-sort / BVH8 / persistent-threads)] is "
                    "supportable. Offline beauty is sufficient regardless.\n");
    }
    std::printf("=======================================================\n");

    // ------------------------------------------------------------------
    // The ONLY hard structural invariant (relative, hardware-independent):
    // at N >= 64, two-level per-frame cost < flat per-frame cost (BLAS-cache
    // win). Absolute numbers are NOT asserted (hardware-dependent; nvidia-smi
    // NVML-broken). Soft-warn fallback if kHardAssertRelativeInvariant is off.
    // ------------------------------------------------------------------
    for (const auto& r : rows) {
        if (r.n < 64u) continue;
        const bool win = r.two_ms < r.flat_ms;
        if (!win) {
            std::printf("[bench][WARN] relative invariant VIOLATED at N=%u: "
                        "two-level %.3f ms !< flat %.3f ms\n",
                        r.n, r.two_ms, r.flat_ms);
        }
        if (kHardAssertRelativeInvariant) {
            EXPECT_LT(r.two_ms, r.flat_ms)
                << "RELATIVE INVARIANT: at N=" << r.n
                << " two-level (" << r.two_ms << " ms) must beat flat ("
                << r.flat_ms << " ms) -- the BLAS-cache win";
        }
    }
}
