// ---------------------------------------------------------------------------
// XPBD constraint-solve micro-benchmark. Times full Step() of a soft-only world
// (no floor / no contacts) so the cost is dominated by the XpbdProject op; the
// same source built against the serial vs colored solver gives the speedup.
// Scenes: a soft-ball tet cage (~14^3) + a 64x64 cloth, at the demo iter counts.
// ---------------------------------------------------------------------------

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "math/vec3.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "runtime/soft/cloth_topology.hpp"
#include "runtime/soft/tetmesh_topology.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace soft = nuka::runtime::soft;
namespace cook = nuka::scene::cook;
using nuka::math::Vec3;

cook::XpbdCookInput BuildBall(uint32_t cells, uint16_t iters) {
    const float radius = 0.10f;
    const float cell_len = 2.0f * radius / static_cast<float>(cells);
    const soft::TetLattice lat =
        soft::BuildSphereTetLattice(Vec3{0, 0, 0}, radius, cells, cell_len);
    soft::TetMeshTopologyOptions opts;
    opts.distance_compliance_alpha = 1.0e-6f;
    opts.volume_compliance_alpha = 1.0e-6f;
    opts.emit_distance_constraints = true;
    soft::XpbdConstraintSet cs;
    soft::BuildTetMeshConstraints(lat.rest, lat.tets, opts, cs);
    cook::XpbdCookInput in;
    in.positions = lat.rest;
    in.velocities.assign(lat.rest.size(), Vec3::Zero());
    in.inv_mass.assign(lat.rest.size(), 1.0f);
    for (const auto& d : cs.distance)
        in.distance.push_back({d.particle_a, d.particle_b, d.rest_length,
                               d.compliance_alpha});
    for (const auto& v : cs.volume) {
        cook::CookVolumeCon vc;
        for (int j = 0; j < 4; ++j) vc.p[j] = v.particle[j];
        vc.rest_volume_times6 = v.rest_volume_times6;
        vc.compliance_alpha = v.compliance_alpha;
        in.volume.push_back(vc);
    }
    in.solver_iterations = iters;
    return in;
}

cook::XpbdCookInput BuildCloth(uint32_t n, uint16_t iters) {
    std::vector<Vec3> rest;
    rest.reserve(static_cast<size_t>(n) * n);
    const float h = 1.0f / static_cast<float>(n - 1u);
    for (uint32_t y = 0; y < n; ++y)
        for (uint32_t x = 0; x < n; ++x)
            rest.push_back(Vec3{static_cast<float>(x) * h,
                                static_cast<float>(y) * h, 0.0f});
    std::vector<soft::ClothTriangle> tris;
    auto gid = [n](uint32_t x, uint32_t y) { return y * n + x; };
    for (uint32_t y = 0; y + 1 < n; ++y)
        for (uint32_t x = 0; x + 1 < n; ++x) {
            tris.push_back(soft::ClothTriangle{{gid(x, y), gid(x + 1, y), gid(x + 1, y + 1)}});
            tris.push_back(soft::ClothTriangle{{gid(x, y), gid(x + 1, y + 1), gid(x, y + 1)}});
        }
    soft::ClothTopologyOptions opts;
    opts.distance_compliance_alpha = 1.0e-6f;
    opts.bend_compliance_alpha = 1.0e-4f;
    opts.emit_distance_constraints = true;
    opts.emit_bend_constraints = true;
    soft::XpbdConstraintSet cs;
    soft::BuildClothConstraints(rest, tris, opts, cs);
    cook::XpbdCookInput in;
    in.positions = rest;
    in.velocities.assign(rest.size(), Vec3::Zero());
    in.inv_mass.assign(rest.size(), 1.0f);
    for (const auto& d : cs.distance)
        in.distance.push_back({d.particle_a, d.particle_b, d.rest_length,
                               d.compliance_alpha});
    for (const auto& b : cs.bend) {
        cook::CookBendCon bc;
        for (int j = 0; j < 4; ++j) { bc.p[j] = b.particle[j]; bc.k[j] = b.k[j]; }
        bc.compliance_alpha = b.compliance_alpha;
        in.bend.push_back(bc);
    }
    in.solver_iterations = iters;
    return in;
}

double TimeWorld(const cook::XpbdCookInput& in, nphi::Device* dev,
                 nphi::Backend* backend, uint32_t steps, const char* label) {
    nk::Model model;
    cook::CookXpbdParticles(model, 1u, in);
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    nk::World world(std::move(model), 1u, dev, backend, cfg);
    if (!world.Ready()) { std::fprintf(stderr, "world not ready (%s)\n", label); return -1.0; }
    for (uint32_t s = 0; s < 20u; ++s) world.Step();  // warmup.
    nphi::BackendSynchronize(backend);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t s = 0; s < steps; ++s) world.Step();
    nphi::BackendSynchronize(backend);  // wait for GPU before stopping the clock.
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double eager_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const auto& cap = world.GetModel().capacities;
    // Graph-replay path: one cudaGraphLaunch replaces the per-color launch storm
    // (captureable here -- a soft-only world carries no thrust-sort op).
    double graph_ms = -1.0;
    if (world.StepPlanned() == nphi::Status::Ok) {
        for (uint32_t s = 0; s < 20u; ++s) world.StepPlanned();
        nphi::BackendSynchronize(backend);
        const auto g0 = std::chrono::high_resolution_clock::now();
        for (uint32_t s = 0; s < steps; ++s) world.StepPlanned();
        nphi::BackendSynchronize(backend);  // wait for GPU before stopping the clock.
        const auto g1 = std::chrono::high_resolution_clock::now();
        graph_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
    }
    std::printf("[%s] particles=%u dist=%u bend=%u vol=%u | eager %.4f ms/step",
                label, cap.particles_per_env, cap.dist_cons_per_env,
                cap.bend_cons_per_env, cap.vol_cons_per_env,
                eager_ms / static_cast<double>(steps));
    if (graph_ms > 0.0)
        std::printf(" | graph %.4f ms/step (%.1fx eager)\n",
                    graph_ms / static_cast<double>(steps), eager_ms / graph_ms);
    else
        std::printf(" | graph unsupported\n");
    return eager_ms / static_cast<double>(steps);
}

// Step a batched (multi-env) soft world twice and report finiteness + run-to-run
// byte-identity (the colored kernel's env-major (env, in-color) thread mapping).
bool CheckBatched(const cook::XpbdCookInput& in, uint32_t envs, nphi::Device* dev,
                  nphi::Backend* backend, uint32_t steps) {
    auto run = [&](std::vector<Vec3>& out) {
        nk::Model model;
        cook::CookXpbdParticles(model, envs, in);
        nk::Pipeline::SolverConfig cfg;
        cfg.dt = 1.0f / 240.0f;
        cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
        nk::World world(std::move(model), envs, dev, backend, cfg);
        if (!world.Ready()) return false;
        for (uint32_t s = 0; s < steps; ++s) world.Step();
        const uint32_t P = world.GetModel().capacities.particles_per_env * envs;
        out.assign(P, Vec3::Zero());
        world.GetData().DownloadField(nk::FieldId::ParticlePos, out.data(),
                                      P * sizeof(Vec3));
        return true;
    };
    std::vector<Vec3> a, b;
    if (!run(a) || !run(b)) { std::fprintf(stderr, "batched world not ready\n"); return false; }
    bool finite = true;
    for (const Vec3& p : a)
        finite &= std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    const bool d1 = a.size() == b.size() &&
                    std::memcmp(a.data(), b.data(), a.size() * sizeof(Vec3)) == 0;
    std::printf("[batched_%uenv] particles=%zu finite=%d two_run_byte_identical=%d\n",
                envs, a.size(), finite ? 1 : 0, d1 ? 1 : 0);
    return finite && d1;
}

// Time the eager step of a batched (multi-env) soft world; the env-major colored
// kernels amortize the per-step launch cost across all envs (the RL regime).
void TimeBatched(const cook::XpbdCookInput& in, uint32_t envs, nphi::Device* dev,
                 nphi::Backend* backend, uint32_t steps, const char* label) {
    nk::Model model;
    cook::CookXpbdParticles(model, envs, in);
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    nk::World world(std::move(model), envs, dev, backend, cfg);
    if (!world.Ready()) { std::fprintf(stderr, "batched world not ready (%s)\n", label); return; }
    for (uint32_t s = 0; s < 10u; ++s) world.Step();
    nphi::BackendSynchronize(backend);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t s = 0; s < steps; ++s) world.Step();
    nphi::BackendSynchronize(backend);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double per_step = std::chrono::duration<double, std::milli>(t1 - t0).count() /
                            static_cast<double>(steps);
    std::printf("[%s] envs=%u | %.4f ms/step = %.4f ms/env-step (%.0f env-steps/s)\n",
                label, envs, per_step, per_step / static_cast<double>(envs),
                static_cast<double>(envs) * 1000.0 / per_step);
}

}  // namespace

int main() {
    nphi::Device* dev = nphi::InitBestDevice();
    nphi::Backend* backend = dev ? nphi::DeviceInitBackend(dev, nullptr) : nullptr;
    if (backend == nullptr) { std::fprintf(stderr, "no CUDA backend\n"); return 2; }
    const uint32_t steps = 500u;
    TimeWorld(BuildBall(14u, 40u), dev, backend, steps, "soft_ball_14cage_40it");
    TimeWorld(BuildCloth(64u, 24u), dev, backend, steps, "cloth_64x64_24it");
    CheckBatched(BuildBall(8u, 20u), 8u, dev, backend, 200u);
    CheckBatched(BuildCloth(16u, 16u), 8u, dev, backend, 200u);
    TimeBatched(BuildBall(14u, 40u), 1024u, dev, backend, 100u, "soft_ball_1024env");
    TimeBatched(BuildCloth(64u, 24u), 1024u, dev, backend, 100u, "cloth_1024env");
    return 0;
}
