// ---------------------------------------------------------------------------
// lreconb_cook_profile -- profile the GENERAL (PairDriven) CookToModel cook of
// the whole-body H1 union scene (examples/scenes/h1_cup_table.nks), splitting the
// >178s cost across the two HEAVY per-shape stages (V-HACD convex decomposition,
// sparse-SDF bake) to confirm which dominates and quantify the L-RECON-B fix.
//
// The general cvx narrowphase consumes NEITHER product (OpNarrowphaseSdf is a
// no-op; the cvx support function wants one hull per mesh). So the general cook can
// skip both. This probe times four configs against the SAME scene:
//   (a) baseline   : V-HACD + SDF (current pre-fix behavior).
//   (b) sdf-off    : V-HACD only.
//   (c) hull-only  : single-hull-per-mesh + SDF.
//   (d) both       : single-hull-per-mesh + no SDF  == the L-RECON-B PairDriven cook.
//
// Each config aborts/logs if it exceeds a generous wall-clock cap (default 60s) so
// we do NOT have to wait the full pathological >178s baseline -- crossing the cap
// is itself the finding. Reports per-config wall-clock + (for the completed cooks)
// a cooked-model sanity check (shape rows / convex hull pieces+verts / SDF grids).
//
// IMPORTANT (cache): the V-HACD + SDF stages are content-addressed-cached in
// .nuka_cache; a WARM cache makes the baseline look fast. Pass --cold to delete the
// cache dir before each timed cook so the numbers reflect a cold (first) cook --
// the case the in-engine cook hits. Default is whatever the cache currently holds.
//
// Usage:  lreconb_cook_profile [--cap-s 60] [--cold] [--scene <path>]
// ---------------------------------------------------------------------------
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include "nk/model/model.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/cooker.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

namespace {
namespace cook = nuka::scene::cook;

constexpr const char* kDefaultScene = "examples/scenes/h1_cup_table.nks";
constexpr const char* kCacheDir = ".nuka_cache";

double NowSec() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void MaybeClearCache(bool cold) {
    if (!cold) return;
    std::error_code ec;
    std::filesystem::remove_all(kCacheDir, ec);  // regenerable content-addressed store
}

// Cook the scene with the given cook-stage gates on a worker thread, polling the
// wall-clock cap. Returns true if it COMPLETED within the cap (filling *blob), or
// false if it exceeded the cap (the worker thread is detached and keeps running,
// but we stop blocking on it -- crossing the cap is the finding).
struct TimedResult {
    bool completed = false;
    double seconds = 0.0;
    // sanity (valid only if completed):
    uint32_t shape_rows = 0;
    uint32_t convex_pieces = 0;
    uint32_t convex_verts = 0;
    uint32_t sdf_grids = 0;
};

TimedResult TimedCook(const nuka::scene::SceneIR& scene,
                      const nuka::scene::CookSceneOptions& opts, double cap_s,
                      bool cold) {
    MaybeClearCache(cold);
    TimedResult r;
    auto blob = std::make_shared<nuka::scene::CookedBlob>();
    auto done = std::make_shared<std::atomic<bool>>(false);
    const double t0 = NowSec();
    std::thread worker([scene_ptr = &scene, opts, blob, done]() {
        *blob = nuka::scene::CookScene(*scene_ptr, opts);
        done->store(true, std::memory_order_release);
    });
    while (!done->load(std::memory_order_acquire)) {
        if (NowSec() - t0 > cap_s) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    r.seconds = NowSec() - t0;
    r.completed = done->load(std::memory_order_acquire);
    if (r.completed) {
        worker.join();
        r.shape_rows = blob->shape_count;
        r.convex_pieces = blob->convex_geometry.Count();
        r.convex_verts =
            static_cast<uint32_t>(blob->convex_geometry.vertices.size() / 3u);
        r.sdf_grids = blob->sdfs.Count();
    } else {
        worker.detach();  // it will keep running; we no longer block on it.
    }
    return r;
}

void PrintRow(const char* label, const TimedResult& r, double cap_s) {
    if (r.completed) {
        std::printf("  %-34s %9.2f s  | shapes=%u hull_pieces=%u hull_verts=%u "
                    "sdf_grids=%u\n",
                    label, r.seconds, r.shape_rows, r.convex_pieces,
                    r.convex_verts, r.sdf_grids);
    } else {
        std::printf("  %-34s  >%6.1f s  | DID NOT COMPLETE within cap (pathological)\n",
                    label, cap_s);
    }
    std::fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
    double cap_s = 60.0;
    bool cold = false;
    std::string scene_path = kDefaultScene;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--cap-s" && i + 1 < argc) cap_s = std::atof(argv[++i]);
        else if (s == "--cold") cold = true;
        else if (s == "--scene" && i + 1 < argc) scene_path = argv[++i];
    }

    if (!std::filesystem::exists(scene_path)) {
        std::fprintf(stderr, "[lreconb_cook_profile] scene not found: %s\n",
                     scene_path.c_str());
        return 2;
    }

    std::printf("[lreconb_cook_profile] scene=%s cap=%.1fs cache=%s\n",
                scene_path.c_str(), cap_s, cold ? "COLD (cleared each cook)" : "as-is");

    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(scene_path);
    std::printf("[lreconb_cook_profile] loaded: bodies=%zu joints=%zu shapes=%zu\n\n",
                scene.Bodies().size(), scene.Joints().size(),
                scene.Shapes().size());

    nuka::scene::CookSceneOptions baseline;       // V-HACD + SDF (pre-fix)
    nuka::scene::CookSceneOptions sdf_off;
    sdf_off.bake_sdf = false;                      // V-HACD only
    nuka::scene::CookSceneOptions hull_only;
    hull_only.general_single_hull = true;          // single-hull + SDF
    nuka::scene::CookSceneOptions both;
    both.bake_sdf = false;
    both.general_single_hull = true;               // == L-RECON-B PairDriven cook

    std::printf("config                                wall-clock | cooked-model sanity\n");
    PrintRow("(a) baseline  V-HACD + SDF",        TimedCook(scene, baseline, cap_s, cold), cap_s);
    PrintRow("(b) sdf-off   V-HACD only",         TimedCook(scene, sdf_off, cap_s, cold), cap_s);
    PrintRow("(c) hull-only single-hull + SDF",   TimedCook(scene, hull_only, cap_s, cold), cap_s);
    PrintRow("(d) both      single-hull, no SDF", TimedCook(scene, both, cap_s, cold), cap_s);

    // Also cook the END-TO-END general model via the real CookToModel(PairDriven)
    // entry the engine uses (covers the post-CookScene transcription cost too).
    std::printf("\n[lreconb_cook_profile] end-to-end CookToModel(PairDriven):\n");
    MaybeClearCache(cold);
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    const double t0 = NowSec();
    cook::CookToModelResult res = cook::CookToModel(scene, 1, opt);
    const double dt = NowSec() - t0;
    const nuka::nk::Model& m = res.model;
    std::printf("  CookToModel(PairDriven)            %9.2f s  | links=%u dofs=%u "
                "artic=%u bodies=%u shape_rows=%zu hull_verts=%zu family=%d\n",
                dt, m.capacities.links_per_env, m.capacities.dofs_per_env,
                m.capacities.articulations_per_env, m.capacities.bodies_per_env,
                m.shape_table_rows.size(), m.hull_verts.size(),
                (int)m.contact_family);

    // Convex-collision sanity: how many shape rows carry a hull/mesh kind with a
    // FINITE non-zero bound radius (the cvx broadphase AABB extent). Proves the
    // general cook produced valid convex collision, not empty / NaN rows.
    uint32_t hull_rows = 0, nonzero_radius = 0;
    bool all_finite = true;
    for (const auto& r : m.shape_table_rows) {
        const bool is_hull =
            r.kind == static_cast<uint32_t>(nuka::scene::ShapeType::ConvexHull) ||
            r.kind == static_cast<uint32_t>(nuka::scene::ShapeType::TriMesh);
        if (!is_hull) continue;
        ++hull_rows;
        if (r.params[0] > 0.0f) ++nonzero_radius;
        if (!std::isfinite(r.params[0])) all_finite = false;
    }
    std::printf("  convex-collision sanity            shape_rows=%zu hull/mesh_rows=%u "
                "nonzero_bound_radius=%u all_finite=%s\n",
                m.shape_table_rows.size(), hull_rows, nonzero_radius,
                all_finite ? "yes" : "NO");
    return 0;
}
