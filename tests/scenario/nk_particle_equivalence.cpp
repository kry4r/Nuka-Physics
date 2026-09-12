// Particle pipeline checks for physical invariants and deterministic replay.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/scoped_device_guard.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Vec3;
constexpr float kPi = 3.14159265358979323846f;

struct Ctx { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
Ctx GetCtx() {
    static Ctx c = [] {
        Ctx r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return c;
}

float TetVolume6(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3) {
    const Vec3 e1 = p1 - p0, e2 = p2 - p0, e3 = p3 - p0;
    return e1.Dot(e2.Cross(e3));
}

}  // namespace

TEST(NkParticleEquivalence, SmallStepFreeMotionRetainsMomentumAtTranslatedOrigins) {
    const auto backend = GetCtx();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (float origin : {0.0f, 100.0f, 10000.0f}) {
        nk::Model model;
        model.particles.mode = nk::Model::ParticleMode::Xpbd;
        model.particles.initial_pos = {{origin, origin, origin}};
        model.particles.initial_vel = {{0.125f, -0.25f, 0.5f}};
        model.particles.inv_mass = {0.001f};
        model.capacities.particles_per_env = 1u;
        nk::Pipeline::SolverConfig config;
        config.dt = 1.0e-5f;
        config.gravity[0] = 0.1f;
        config.gravity[1] = 0.2f;
        config.gravity[2] = -0.3f;
        nk::World world(std::move(model), 1u, backend.dev, backend.backend, config);
        ASSERT_TRUE(world.Ready()) << world.CreationError();
        for (uint32_t step = 0u; step < 50u; ++step) ASSERT_TRUE(world.Step().AllOk());
        Vec3 actual{};
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::ParticleVel, &actual, sizeof(actual)));
        const Vec3 expected = Vec3{0.125f, -0.25f, 0.5f} + Vec3{0.1f, 0.2f, -0.3f} * (50.0f * config.dt);
        EXPECT_LT((actual - expected).Length(), 2.0e-6f) << "origin=" << origin;
    }
}

TEST(NkParticleEquivalence, SmallStepMaterialImpulseRetainsMomentumAtTranslatedOrigins) {
    const auto backend = GetCtx();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    constexpr float rest = 0.9f, compliance = 0.001f, dt = 1.0e-5f;
    const float expected_speed = (1.0f - rest) / (2.0f + compliance / (dt * dt)) / dt;
    for (float origin : {0.0f, 10000.0f}) {
        nk::Model model;
        auto& particles = model.particles;
        particles.mode = nk::Model::ParticleMode::Xpbd;
        particles.initial_pos = {{origin, 0.0f, 0.0f}, {origin + 1.0f, 0.0f, 0.0f}};
        particles.initial_vel = {Vec3::Zero(), Vec3::Zero()};
        particles.inv_mass = {1.0f, 1.0f};
        particles.dist_a = {0u};
        particles.dist_b = {1u};
        particles.dist_rest = {rest};
        particles.dist_alpha = {compliance};
        particles.xpbd_iters = 1u;
        model.capacities.particles_per_env = 2u;
        model.capacities.dist_cons_per_env = 1u;
        nk::Pipeline::SolverConfig config;
        config.dt = dt;
        config.gravity[0] = config.gravity[1] = config.gravity[2] = 0.0f;
        nk::World world(std::move(model), 1u, backend.dev, backend.backend, config);
        ASSERT_TRUE(world.Ready()) << world.CreationError();
        ASSERT_TRUE(world.Step().AllOk());
        Vec3 velocity[2]{};
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::ParticleVel, velocity, sizeof(velocity)));
        EXPECT_LT((velocity[0] - Vec3{expected_speed, 0.0f, 0.0f}).Length(), 1.0e-8f)
            << "origin=" << origin;
        EXPECT_LT((velocity[1] + Vec3{expected_speed, 0.0f, 0.0f}).Length(), 1.0e-8f)
            << "origin=" << origin;
        EXPECT_LT((velocity[0] + velocity[1]).Length(), 1.0e-9f) << "origin=" << origin;
    }
}

TEST(NkParticleEquivalence, XpbdDistanceOscillationPeriodMatchesAnalytic) {
    if (GetCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const float rest = 1.0f, stretch = 0.1f, inv_mass = 1.0f, alpha = 0.02f;
    const float dt = 1.0f / 240.0f;

    nk::Model model;
    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Xpbd;
    mp.initial_pos = {Vec3{0, 0, 0}, Vec3{rest + stretch, 0, 0}};
    mp.initial_vel = {Vec3::Zero(), Vec3::Zero()};
    mp.inv_mass = {0.0f, inv_mass};  // particle 0 pinned.
    mp.dist_a = {0u}; mp.dist_b = {1u};
    mp.dist_rest = {rest}; mp.dist_alpha = {alpha};
    mp.xpbd_iters = 1;
    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = 1; cap.particles_per_env = 2; cap.dist_cons_per_env = 1;

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[0] = cfg.gravity[1] = cfg.gravity[2] = 0.0f;  // pure SHM.
    Ctx c = GetCtx();
    nk::World world(std::move(model), 1u, c.dev, c.backend, cfg);
    ASSERT_TRUE(world.Ready());

    const float w_sum = inv_mass;
    const float gamma = alpha / (alpha + w_sum * dt * dt);
    const float t_discrete = 2.0f * kPi * dt / std::acos(std::sqrt(gamma));

    const uint32_t kSteps = 6000u;
    std::vector<float> x; x.reserve(kSteps);
    std::vector<Vec3> p(2);
    for (uint32_t i = 0; i < kSteps; ++i) {
        world.Step();
        world.GetData().DownloadField(nk::FieldId::ParticlePos, p.data(), 2 * sizeof(Vec3));
        const float d = (p[1] - p[0]).Length();
        x.push_back(d - rest);
    }
    int first = -1, last = -1, crossings = 0;
    for (uint32_t i = 1; i < kSteps; ++i) {
        const bool up = x[i - 1] <= 0.0f && x[i] > 0.0f;
        const bool down = x[i - 1] >= 0.0f && x[i] < 0.0f;
        if (up || down) { if (first < 0) first = (int)i; last = (int)i; ++crossings; }
    }
    ASSERT_GE(crossings, 3);
    const float half_period_steps = (float)(last - first) / (float)(crossings - 1);
    const float t_obs = 2.0f * half_period_steps * dt;
    const float rel_err = std::fabs(t_obs - t_discrete) / t_discrete;
    std::printf("[nk-xpbd] T_discrete=%.5f T_obs=%.5f rel-err=%.4f crossings=%d\n",
                t_discrete, t_obs, rel_err, crossings);
    EXPECT_LT(rel_err, 0.02f) << "nk XPBD distance period diverges from the analytic oracle";
}

// XPBD tet volume conservation (the test_xpbd_volume_tet invariant, nk path) + D1.
TEST(NkParticleEquivalence, XpbdTetVolumeConservedAndD1) {
    if (GetCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    auto build = [] {
        nk::Model model;
        nk::Model::ModelParticles& mp = model.particles;
        mp.mode = nk::Model::ParticleMode::Xpbd;
        std::vector<Vec3> pos = {Vec3{0, 0, 0.04f}, Vec3{0.03f, 0, 0},
                                 Vec3{-0.015f, 0.026f, 0}, Vec3{-0.015f, -0.026f, 0}};
        mp.initial_pos = pos;
        mp.initial_vel.assign(4, Vec3::Zero());
        mp.inv_mass.assign(4, 1.0f / 0.02f);
        const uint32_t edges[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
        for (auto& e : edges) {
            mp.dist_a.push_back(e[0]); mp.dist_b.push_back(e[1]);
            mp.dist_rest.push_back((pos[e[0]] - pos[e[1]]).Length());
            mp.dist_alpha.push_back(1.0e-7f);
        }
        for (uint32_t j = 0; j < 4; ++j) mp.vol_particles.push_back(j);
        mp.vol_rest6.push_back(TetVolume6(pos[0], pos[1], pos[2], pos[3]));
        mp.vol_alpha.push_back(1.0e-7f);
        mp.xpbd_iters = 12;
        nk::ModelCapacities& cap = model.capacities;
        cap.env_count = 1; cap.particles_per_env = 4;
        cap.dist_cons_per_env = 6; cap.vol_cons_per_env = 1;
        return model;
    };
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f; cfg.gravity[2] = -9.81f;
    Ctx c = GetCtx();
    auto run = [&](std::vector<Vec3>* out) {
        nk::Model m = build();
        const float rest6 = m.particles.vol_rest6.front();
        nk::World world(std::move(m), 1u, c.dev, c.backend, cfg);
        EXPECT_TRUE(world.Ready());
        for (uint32_t i = 0; i < 100u; ++i) world.Step();
        std::vector<Vec3> p(4);
        world.GetData().DownloadField(nk::FieldId::ParticlePos, p.data(), 4 * sizeof(Vec3));
        *out = p;
        return rest6;
    };
    std::vector<Vec3> a, b;
    const float rest6 = run(&a);
    run(&b);
    const float v6 = TetVolume6(a[0], a[1], a[2], a[3]);
    const float rel = std::fabs(v6 - rest6) / std::fabs(rest6);
    std::printf("[nk-xpbd-vol] rest6=%.4e final6=%.4e rel-err=%.4f\n", rest6, v6, rel);
    EXPECT_LT(rel, 0.10f) << "nk XPBD tet volume drifted (vs the legacy invariant)";
    EXPECT_EQ(std::memcmp(a.data(), b.data(), 4 * sizeof(Vec3)), 0)
        << "nk XPBD forward is not D1 (two runs differ)";
}

// PBF density conservation (the legacy PBF density / volume oracle, nk path) + D1.
TEST(NkParticleEquivalence, PbfRestLatticeDensityNearRho0AndD1) {
    if (GetCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const float spacing = 0.03f, h = spacing * 1.5f, rho0 = 1000.0f;
    const float mass = rho0 * spacing * spacing * spacing;
    auto build = [&] {
        nk::Model model;
        nk::Model::ModelParticles& mp = model.particles;
        mp.mode = nk::Model::ParticleMode::Pbf;
        const uint32_t n = 4;
        std::vector<Vec3> pos;
        const float c0 = -0.5f * (n - 1) * spacing;
        for (uint32_t iz = 0; iz < n; ++iz)
            for (uint32_t iy = 0; iy < n; ++iy)
                for (uint32_t ix = 0; ix < n; ++ix)
                    pos.push_back(Vec3{c0 + ix * spacing, c0 + iy * spacing, c0 + iz * spacing});
        const uint32_t P = (uint32_t)pos.size();
        mp.initial_pos = pos;
        mp.initial_vel.assign(P, Vec3::Zero());
        mp.inv_mass.assign(P, 1.0f / mass);
        mp.pbf_rest_density = rho0; mp.pbf_support_radius = h; mp.pbf_particle_mass = mass;
        mp.pbf_iters = 4; mp.pbf_clamp_overdensity = true;
        mp.cell_size = h; mp.query_radius = h;
        mp.grid_min = Vec3{c0 - h, c0 - h, c0 - h};
        mp.grid_dims[0] = 4; mp.grid_dims[1] = 4; mp.grid_dims[2] = 4;
        model.capacities.env_count = 1; model.capacities.particles_per_env = P;
        // Cell capacity sizes grid_cell_start/end (cells x env_count).
        model.capacities.max_grid_cells =
            mp.grid_dims[0] * mp.grid_dims[1] * mp.grid_dims[2];
        return model;
    };
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = cfg.gravity[1] = cfg.gravity[2] = 0.0f;  // rest lattice (no fall).
    Ctx c = GetCtx();
    auto run = [&](std::vector<Vec3>* out) {
        nk::World world(build(), 1u, c.dev, c.backend, cfg);
        EXPECT_TRUE(world.Ready());
        const uint32_t P = world.GetModel().capacities.particles_per_env;
        for (uint32_t i = 0; i < 30u; ++i) world.Step();
        std::vector<Vec3> p(P);
        world.GetData().DownloadField(nk::FieldId::ParticlePos, p.data(), P * sizeof(Vec3));
        *out = p;
    };
    std::vector<Vec3> a, b;
    run(&a); run(&b);
    // Interior density (host Poly6 mirror); a rest lattice's central particle is
    // fully surrounded -> near rho0 (the calibration the legacy density test asserts).
    const float h2 = h * h, h9 = std::pow(h, 9.0f);
    const float poly6 = 315.0f / (64.0f * kPi * h9);
    auto W = [&](float r2) { return r2 >= h2 ? 0.0f : poly6 * (h2 - r2) * (h2 - r2) * (h2 - r2); };
    float best = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float rho = mass * W(0.0f);
        for (size_t j = 0; j < a.size(); ++j)
            if (j != i) rho += mass * W((a[i] - a[j]).Dot(a[i] - a[j]));
        best = std::max(best, rho);
    }
    const float rel = std::fabs(best - rho0) / rho0;
    std::printf("[nk-pbf] rho0=%.1f max-interior rho=%.1f rel-err=%.4f\n", rho0, best, rel);
    EXPECT_LT(rel, 0.20f) << "nk PBF rest-lattice density diverges from rho0 (vs the legacy oracle)";
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(Vec3)), 0)
        << "nk PBF forward is not D1 (two runs differ)";
}
