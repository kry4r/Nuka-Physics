// ---------------------------------------------------------------------------
// vproof_go2_throughput -- measure the GENERAL-path (PairDriven + flat
// heightfield) go2 world env-steps/sec at N in {1,64,256,1024}.
//
// The one-general-solver-spec (§4 "Throughput gate") requires the general contact
// path (LBVH broadphase + cvx narrowphase over ALL bodies every step) to stay fast
// enough for RL training: >= ~10000 env-steps/s @N=1024 (<= ~3 h / 100M-step
// stage). The pre-collapse union-solver path measured 16,367 env-steps/s
// (nuka_union_throughput); this bench re-measures on the GENERAL go2 world (whose
// per-step LBVH+cvx is the new cost the collapse adds vs the cheap analytic foot
// path). Mirrors examples/demo/union_throughput.cpp's structure: cook once, build
// the batched nk::World per N, warm + time StepPlanned, report per-step ms +
// env-steps/s, and a finite-q check.
//
// Cook: CookToModel(go2_stand, N, {PairDriven}) + CookHeightfieldGrid(kTerrainFlat)
// at the perf-gate ground (0.31) + capacity grow (the heightfield static collidable
// enters the LBVH; the foot spheres collide against its prisms via the SAME cvx
// narrowphase -> general mixed-island solve). Pure engine step cost (no action
// injection / no render).
//
// Usage:  vproof_go2_throughput [--steps 200] [--warm 12] [--ns 1,64,256,1024]
//
// Prints PASS/BELOW vs the 10000 env-steps/s reference; does NOT gate-fail the
// build (a measurement, not a hard CI assert — spec §"throughput bench").
// ---------------------------------------------------------------------------
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "import/usd_importer.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/terrain/heightfield.hpp"
#include "scene/terrain/heightfield_loaders.hpp"

namespace {
namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace terrain = nuka::terrain;
using nuka::math::Vec3;

constexpr const char* kScenePath = "examples/scenes/go2_stand.usda";
constexpr float kGround = 0.31f;            // perf-gate seat (feet load).
constexpr double kRefEnvStepsPerSec = 10000.0;  // spec §4 throughput floor.

double NowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Cook the GENERAL (PairDriven) go2 + a FLAT heightfield collidable at kGround for
// N envs (same recipe as tests/scenario/vproof_go2_ground.cpp's general side).
nk::Model CookGeneral(const nuka::scene::SceneIR& scene, uint32_t N) {
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    nk::Model m = cook::CookToModel(scene, static_cast<int>(N), opt).model;
    const uint32_t orig_bodies = m.capacities.bodies_per_env;
    m.body_init.resize(orig_bodies);  // heightfield seeds at index orig_bodies.

    terrain::TerrainGenConfig cfg;  // FLAT: every feature amplitude 0.
    cfg.nrow = 21u;
    cfg.ncol = 21u;
    cfg.cell_x = 0.25f;
    cfg.cell_y = 0.25f;
    cfg.origin = Vec3{-0.5f * 20.0f * 0.25f, -0.5f * 20.0f * 0.25f, 0.0f};
    cfg.base_z = kGround;
    terrain::HeightField hf;
    terrain::GenerateHeightField(cfg, hf);
    const uint32_t hf_row = cook::CookHeightfieldGrid(m, hf);
    if (hf_row != orig_bodies) {
        std::fprintf(stderr, "[vproof_go2_throughput] WARN hf_row=%u != %u\n",
                     hf_row, orig_bodies);
    }
    nk::ModelCapacities& cap = m.capacities;
    cap.bodies_per_env = static_cast<uint32_t>(m.body_init.size());
    cap.max_bodies_total = static_cast<uint32_t>(m.shape_table_rows.size());
    cap.max_contacts_per_env = 32u;
    cap.max_rows_per_env = 32u * nk::kPairDrivenRowsPerSlot;
    return m;
}
}  // namespace

int main(int argc, char** argv) {
    uint32_t steps = 200, warm = 12;
    std::vector<uint32_t> ns = {1, 64, 256, 1024};
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--steps" && i + 1 < argc) steps = (uint32_t)std::atoi(argv[++i]);
        else if (s == "--warm" && i + 1 < argc) warm = (uint32_t)std::atoi(argv[++i]);
        else if (s == "--ns" && i + 1 < argc) {
            ns.clear();
            char* tok = std::strtok(argv[++i], ",");
            while (tok) { ns.push_back((uint32_t)std::atoi(tok)); tok = std::strtok(nullptr, ","); }
        }
    }

    const nuka::scene::SceneIR scene = nuka::import::LoadUsd(kScenePath);

    nphi::Device* dev = nphi::InitBestDevice();
    if (!dev) { std::fprintf(stderr, "[vproof_go2_throughput] no phi device\n"); return 4; }
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    if (!backend) { std::fprintf(stderr, "[vproof_go2_throughput] no backend\n"); return 4; }

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;

    std::printf("[vproof_go2_throughput] GENERAL (PairDriven + flat heightfield) go2 "
                "| dt=%.5f steps=%u warm=%u | bar: >= %.0f env-steps/s\n",
                cfg.dt, steps, warm, kRefEnvStepsPerSec);
    std::printf("%6s %12s %14s %16s %10s %8s\n", "N", "ms/step", "us/env-step",
                "env-steps/s", "q_finite", "verdict");

    for (uint32_t N : ns) {
        nk::Model model = CookGeneral(scene, N);
        nk::World w(std::move(model), N, dev, backend, cfg);
        if (!w.Ready()) { std::fprintf(stderr, "[vproof_go2_throughput] N=%u world !ready\n", N); continue; }

        bool planned = true;
        for (uint32_t i = 0; i < warm; ++i) {
            if (w.StepPlanned() != nphi::Status::Ok) { planned = false; break; }
        }
        if (!planned) { for (uint32_t i = 0; i < warm; ++i) w.Step(); }

        nphi::BackendSynchronize(backend);
        const double t0 = NowMs();
        for (uint32_t i = 0; i < steps; ++i) { if (planned) w.StepPlanned(); else w.Step(); }
        nphi::BackendSynchronize(backend);
        const double t1 = NowMs();

        std::vector<float> q(w.GetModel().capacities.ElementCount(nk::FieldId::Q), 0.0f);
        w.GetData().DownloadField(nk::FieldId::Q, q.data(), q.size() * sizeof(float), 0);
        bool finite = true;
        for (float v : q) { if (!(v == v) || v > 1e30f || v < -1e30f) { finite = false; break; } }

        const double ms_step = (t1 - t0) / steps;
        const double us_env = ms_step * 1000.0 / N;
        const double eps = (double)N * steps / ((t1 - t0) / 1000.0);
        const char* verdict = (eps >= kRefEnvStepsPerSec) ? "PASS" : "BELOW";
        std::printf("%6u %12.3f %14.3f %16.0f %10s %8s%s\n", N, ms_step, us_env, eps,
                    finite ? "yes" : "NO-NAN", verdict, planned ? "" : " (eager)");
        std::fflush(stdout);
    }

    nphi::BackendFree(backend);
    return 0;
}
