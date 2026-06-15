// ---------------------------------------------------------------------------
// union_throughput -- measure batched H1+cup+table UNION world env-steps/sec.
//
// THE M10 RL feasibility gate (SG-spec G1d): one PPO stage ~100M env-steps must
// fit < 24 h on this GPU (=> need ~1160 env-steps/s sustained). The pre-refactor
// concern was ~180 env-steps/s (infeasible). This harness cooks the union once
// (CookSceneToUnionTemplate) and, for each N, BuildNkUnionModel(tmpl,N) -> nk::World
// -> warm + time StepPlanned, reporting per-step ms and env-steps/sec. Also checks
// the batched union STEPS at N>1 (built-but-unexercised) with finite q (no NaN).
//
// Pure engine step cost (no action injection / no render). Built behind
// NK_BUILD_VULKAN_VALIDATION purely to share the demo link set; it uses NO Vulkan.
//
// Usage:  union_throughput [--steps 200] [--warm 12] [--ns 1,64,256,1024]
// ---------------------------------------------------------------------------
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "scene/cook/union_cook.hpp"
#include "scene/cook/union_nk_model.hpp"
#include "scene/cook/union_scene_constants.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

namespace {
namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace coresident = nuka::runtime::coresident;

constexpr const char* kNksPath = "examples/scenes/h1_cup_table.nks";

double NowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
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

    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    cook::CookedUnionScene cooked = cook::CookSceneToUnionTemplate(scene, 1);
    const coresident::UnionSceneTemplate& tmpl = cooked.tmpl;
    std::printf("[union_throughput] cooked: dof_stride=%u links=%u feet=%zu fingertips=%zu cup_mass=%.3f\n",
                cooked.dof_stride, cooked.link_count, tmpl.feet.size(),
                tmpl.fingertips.size(), cooked.cup_mass);

    nphi::Device* dev = nphi::InitBestDevice();
    if (!dev) { std::fprintf(stderr, "[union_throughput] no phi device\n"); return 4; }
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    if (!backend) { std::fprintf(stderr, "[union_throughput] no backend\n"); return 4; }

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = cook::kH1UnionDt;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = cook::kH1UnionGravityZ;
    cfg.vel_iters = 64; cfg.pos_iters = 0; cfg.fold_drive_damping = 0;

    std::printf("[union_throughput] dt=%.5f steps=%u warm=%u | bar: >=1160 env-steps/s => 100M < 24h\n",
                cfg.dt, steps, warm);
    std::printf("%6s %12s %14s %16s %10s\n", "N", "ms/step", "us/env-step", "env-steps/s", "q_finite");

    for (uint32_t N : ns) {
        nk::Model model = coresident::BuildNkUnionModel(tmpl, N);
        nk::World w(std::move(model), N, dev, backend, cfg);
        if (!w.Ready()) { std::fprintf(stderr, "[union_throughput] N=%u world !ready\n", N); continue; }

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

        // finite check on q
        std::vector<float> q(w.GetModel().capacities.ElementCount(nk::FieldId::Q), 0.0f);
        w.GetData().DownloadField(nk::FieldId::Q, q.data(), q.size() * sizeof(float), 0);
        bool finite = true;
        for (float v : q) { if (!(v == v) || v > 1e30f || v < -1e30f) { finite = false; break; } }

        const double ms_step = (t1 - t0) / steps;
        const double us_env = ms_step * 1000.0 / N;
        const double eps = (double)N * steps / ((t1 - t0) / 1000.0);
        std::printf("%6u %12.3f %14.3f %16.0f %10s%s\n", N, ms_step, us_env, eps,
                    finite ? "yes" : "NO-NAN", planned ? "" : " (eager)");
        std::fflush(stdout);
    }

    nphi::BackendFree(backend);
    return 0;
}
