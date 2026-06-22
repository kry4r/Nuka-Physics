// ---------------------------------------------------------------------------
// PBF CUDA-graph capture parity: a fluid world (which runs ParticleGridBuild,
// the only op that used a mid-capture allocator) must (1) capture into a graph
// so StepPlanned() returns Ok with NO fallback to eager Step(), and (2) the
// graph replay must produce BYTE-IDENTICAL particle output to eager stepping
// (the graph runs the same physics). Both prove the pre-allocated grid sort
// scratch removed the only graph-capture blocker for coupled/particle scenes.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Vec3;

struct NkCtx { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
NkCtx GetNkCtx() {
    static NkCtx c = [] {
        NkCtx r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return c;
}

// A small fluid block + grid domain (mirrors the V2-invariant lattice setup).
nk::Model MakePbfModel(uint32_t nx, uint32_t ny, uint32_t nz, float dx) {
    std::vector<Vec3> pos;
    for (uint32_t iz = 0; iz < nz; ++iz)
        for (uint32_t iy = 0; iy < ny; ++iy)
            for (uint32_t ix = 0; ix < nx; ++ix)
                pos.push_back(Vec3{static_cast<float>(ix) * dx,
                                   static_cast<float>(iy) * dx,
                                   0.15f + static_cast<float>(iz) * dx});
    const uint32_t count = static_cast<uint32_t>(pos.size());
    const float h = 1.6f * dx;

    nk::Model model;
    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Pbf;
    mp.initial_pos = pos;
    mp.initial_vel.assign(count, Vec3::Zero());
    mp.inv_mass.assign(count, 1.0f);
    mp.pbf_rest_density = 1000.0f;
    mp.pbf_support_radius = h;
    mp.pbf_particle_mass = 1.0f;
    mp.pbf_iters = 4u;
    mp.pbf_clamp_overdensity = true;
    mp.pbf_xsph_viscosity = 0.1f;
    mp.pbf_surface_tension = 5.0f;
    mp.cell_size = h;
    mp.query_radius = h;
    mp.boundary_enabled = true;
    mp.floor_z = 0.0f;
    Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (const Vec3& p : pos) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    mp.grid_min = Vec3{lo.x - h, lo.y - h, lo.z - h};
    auto dim = [&](float span) {
        return static_cast<uint32_t>(std::floor(span / h)) + 4u;
    };
    mp.grid_dims[0] = dim(hi.x - lo.x);
    mp.grid_dims[1] = dim(hi.y - lo.y);
    mp.grid_dims[2] = dim(hi.z - lo.z);
    model.capacities.env_count = 1;
    model.capacities.particles_per_env = count;
    model.capacities.max_grid_cells =
        mp.grid_dims[0] * mp.grid_dims[1] * mp.grid_dims[2];
    return model;
}

nk::Pipeline::SolverConfig PbfCfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    return cfg;
}

} // namespace

// Gate 1: a fluid world running ParticleGridBuild captures into a CUDA graph ->
// StepPlanned() returns Ok (no Unsupported fallback to Step()).
TEST(PbfGraphCapture, StepPlannedSucceedsForFluidWorld) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    NkCtx c = GetNkCtx();
    nk::World world(MakePbfModel(8u, 8u, 8u, 0.05f), 1u, c.dev, c.backend, PbfCfg());
    ASSERT_TRUE(world.Ready());
    // First call builds + instantiates the graph; Unsupported here would mean the
    // grid op still blocks capture (the bug this change closes).
    const nphi::Status s0 = world.StepPlanned();
    EXPECT_EQ(s0, nphi::Status::Ok)
        << "StepPlanned must capture the PBF grid op, not fall back to eager";
    // Replays must also stay Ok.
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(world.StepPlanned(), nphi::Status::Ok) << "replay " << i;
    }
}

// Gate 2: graph replay == eager Step, BYTE-for-BYTE on the particle output.
TEST(PbfGraphCapture, GraphReplayByteIdenticalToEager) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    NkCtx c = GetNkCtx();
    const uint32_t kSteps = 120u;

    nk::World eager(MakePbfModel(8u, 8u, 8u, 0.05f), 1u, c.dev, c.backend, PbfCfg());
    nk::World graph(MakePbfModel(8u, 8u, 8u, 0.05f), 1u, c.dev, c.backend, PbfCfg());
    ASSERT_TRUE(eager.Ready());
    ASSERT_TRUE(graph.Ready());
    const uint32_t n = eager.GetModel().capacities.particles_per_env;
    ASSERT_GT(n, 0u);

    for (uint32_t s = 0; s < kSteps; ++s) {
        eager.Step();
        ASSERT_EQ(graph.StepPlanned(), nphi::Status::Ok)
            << "graph world fell back to eager at step " << s;
    }
    nphi::BackendSynchronize(c.backend);

    std::vector<Vec3> pe(n), pg(n), ve(n), vg(n);
    eager.GetData().DownloadField(nk::FieldId::ParticlePos, pe.data(), n * sizeof(Vec3));
    graph.GetData().DownloadField(nk::FieldId::ParticlePos, pg.data(), n * sizeof(Vec3));
    eager.GetData().DownloadField(nk::FieldId::ParticleVel, ve.data(), n * sizeof(Vec3));
    graph.GetData().DownloadField(nk::FieldId::ParticleVel, vg.data(), n * sizeof(Vec3));

    EXPECT_EQ(std::memcmp(pe.data(), pg.data(), n * sizeof(Vec3)), 0)
        << "particle POSITION drift between eager Step and graph StepPlanned";
    EXPECT_EQ(std::memcmp(ve.data(), vg.data(), n * sizeof(Vec3)), 0)
        << "particle VELOCITY drift between eager Step and graph StepPlanned";
}

// Measurement (not a hard gate): eager vs graph ms/step for a fluid world WITH
// the particle grid — the "every coupled scene gets the graph speedup" proof.
// A bigger lattice + many envs so the per-step kernel-launch overhead the graph
// collapses is visible. Prints both timings; only asserts the graph runs.
TEST(PbfGraphCapture, MeasureEagerVsGraphMsPerStep) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    NkCtx c = GetNkCtx();
    const uint32_t envs = 256u;
    const uint32_t kSteps = 300u;

    auto time_path = [&](bool planned, const char* label) -> double {
        nk::World w(MakePbfModel(8u, 8u, 8u, 0.05f), envs, c.dev, c.backend, PbfCfg());
        if (!w.Ready()) return -1.0;
        for (uint32_t s = 0; s < 20u; ++s) {
            if (planned) { if (w.StepPlanned() != nphi::Status::Ok) return -2.0; }
            else w.Step();
        }
        nphi::BackendSynchronize(c.backend);
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t s = 0; s < kSteps; ++s) {
            if (planned) w.StepPlanned(); else w.Step();
        }
        nphi::BackendSynchronize(c.backend);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() /
                          static_cast<double>(kSteps);
        std::printf("[grid_world %s] envs=%u %.4f ms/step\n", label, envs, ms);
        return ms;
    };

    const double eager_ms = time_path(false, "eager");
    const double graph_ms = time_path(true, "graph");
    ASSERT_GT(eager_ms, 0.0);
    ASSERT_GT(graph_ms, 0.0) << "graph world fell back to eager (capture failed)";
    std::printf("[grid_world] eager %.4f ms/step | graph %.4f ms/step | %.2fx\n",
                eager_ms, graph_ms, eager_ms / graph_ms);
}
