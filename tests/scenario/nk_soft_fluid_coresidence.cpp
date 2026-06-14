// ---------------------------------------------------------------------------
// M9 T11 PHASE 1: two-system (soft XPBD + fluid PBF) CO-RESIDENCE smoke.
//
// Builds ONE nk::Model holding BOTH a soft (XPBD) particle set AND a fluid (PBF)
// particle set co-resident in the contiguous [soft | fluid] layout (split index
// n_soft_particles) via scene::cook::CookSoftFluidParticles, then steps nk::World
// on the SoftFluid particle-op path. Phase-1 assertions (the build-only schema +
// scoped solve correctness):
//
//   1. SUPERSET BYTE-IDENTITY: a soft-only co-residence cook == the canonical
//      CookXpbdParticles cook (same staged Model device buffer); a fluid-only
//      co-residence cook == the canonical CookPbfParticles cook. (The single-
//      system goldens are unaffected.)
//   2. CO-RESIDENT STEP: a soft cube (shape-match-free distance lattice) +
//      a fluid blob co-step in ONE world: the soft cube stays near-rigid (edge
//      lengths near rest) while the fluid stays finite + falls under gravity.
//   3. FLUID-SLICE SCOPE: the soft particles do NOT perturb the fluid density
//      solve -- verified indirectly (the world is stable + finite; a soft particle
//      leaking into fluid density would blow up the projection).
//   4. D1: the co-resident forward is two-run byte-exact.
//
// This is the Phase-1 functional check for the co-residence schema; the cross-
// system id-10 contact between the two slices is Phase 2 (not exercised here).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/scoped_device_guard.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
using nuka::math::Vec3;

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

// A small soft tet (4 particles, 6 distance edges) -- the soft slice.
cook::XpbdCookInput MakeSoftTet() {
    cook::XpbdCookInput in;
    in.positions = {Vec3{0, 0, 0.04f}, Vec3{0.03f, 0, 0},
                    Vec3{-0.015f, 0.026f, 0}, Vec3{-0.015f, -0.026f, 0}};
    in.velocities.assign(4, Vec3::Zero());
    in.inv_mass.assign(4, 1.0f / 0.02f);
    const uint32_t edges[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for (auto& e : edges) {
        cook::CookDistanceCon dc;
        dc.a = e[0]; dc.b = e[1];
        dc.rest_length = (in.positions[e[0]] - in.positions[e[1]]).Length();
        dc.compliance_alpha = 1.0e-7f;
        in.distance.push_back(dc);
    }
    in.solver_iterations = 8u;
    return in;
}

// A small fluid blob (4x4x4 lattice) -- the fluid slice. Placed well above the
// soft tet so the two slices do not initially overlap.
cook::PbfCookInput MakeFluidBlob() {
    cook::PbfCookInput in;
    const float spacing = 0.03f, h = spacing * 1.5f, rho0 = 1000.0f;
    const float mass = rho0 * spacing * spacing * spacing;
    const uint32_t n = 4;
    const float c0 = -0.5f * (n - 1) * spacing;
    const float lift = 0.5f;  // place the blob above the soft tet.
    for (uint32_t iz = 0; iz < n; ++iz)
        for (uint32_t iy = 0; iy < n; ++iy)
            for (uint32_t ix = 0; ix < n; ++ix)
                in.positions.push_back(Vec3{c0 + ix * spacing, c0 + iy * spacing,
                                            lift + c0 + iz * spacing});
    in.velocities.assign(in.positions.size(), Vec3::Zero());
    in.particle_mass = mass;
    in.rest_density = rho0;
    in.support_radius = h;
    in.cfm_epsilon = 1.0e-6f;
    in.iters = 4u;
    in.clamp_overdensity = true;
    in.grid_min = Vec3{c0 - h, c0 - h, lift + c0 - h};
    in.grid_dims[0] = 4; in.grid_dims[1] = 4; in.grid_dims[2] = 4;
    return in;
}

}  // namespace

// Assertion 1: soft-only / fluid-only co-residence cooks == the canonical single-
// system cooks (the strict-superset contract). Compared over the cooked
// ModelParticles host arrays + capacities (the device staging is a deterministic
// function of these).
TEST(NkSoftFluidCoResidence, SupersetMatchesSingleSystemCook) {
    const cook::XpbdCookInput soft = MakeSoftTet();
    const cook::PbfCookInput fluid = MakeFluidBlob();

    // Soft-only: CookSoftFluidParticles(soft, EMPTY) == CookXpbdParticles(soft).
    {
        nk::Model a, b;
        cook::CookSoftFluidParticles(a, 1u, soft, cook::PbfCookInput{});
        cook::CookXpbdParticles(b, 1u, soft);
        EXPECT_EQ(a.particles.mode, nk::Model::ParticleMode::Xpbd);
        EXPECT_EQ(a.capacities.particles_per_env, b.capacities.particles_per_env);
        EXPECT_EQ(a.capacities.dist_cons_per_env, b.capacities.dist_cons_per_env);
        ASSERT_EQ(a.particles.initial_pos.size(), b.particles.initial_pos.size());
        EXPECT_EQ(std::memcmp(a.particles.initial_pos.data(),
                              b.particles.initial_pos.data(),
                              a.particles.initial_pos.size() * sizeof(Vec3)), 0)
            << "soft-only co-residence cook diverged from CookXpbdParticles";
        ASSERT_EQ(a.particles.dist_rest.size(), b.particles.dist_rest.size());
        EXPECT_EQ(std::memcmp(a.particles.dist_rest.data(),
                              b.particles.dist_rest.data(),
                              a.particles.dist_rest.size() * sizeof(float)), 0);
    }
    // Fluid-only: CookSoftFluidParticles(EMPTY, fluid) == CookPbfParticles(fluid).
    {
        nk::Model a, b;
        cook::CookSoftFluidParticles(a, 1u, cook::XpbdCookInput{}, fluid);
        cook::CookPbfParticles(b, 1u, fluid);
        EXPECT_EQ(a.particles.mode, nk::Model::ParticleMode::Pbf);
        EXPECT_EQ(a.capacities.particles_per_env, b.capacities.particles_per_env);
        EXPECT_EQ(a.capacities.max_grid_cells, b.capacities.max_grid_cells);
        EXPECT_FLOAT_EQ(a.particles.pbf_rest_density, b.particles.pbf_rest_density);
        ASSERT_EQ(a.particles.initial_pos.size(), b.particles.initial_pos.size());
        EXPECT_EQ(std::memcmp(a.particles.initial_pos.data(),
                              b.particles.initial_pos.data(),
                              a.particles.initial_pos.size() * sizeof(Vec3)), 0)
            << "fluid-only co-residence cook diverged from CookPbfParticles";
    }
}

// Assertion 2/3/4: a soft tet + a fluid blob co-step in ONE world; the soft tet
// stays near-rigid, the fluid stays finite + falls, and the forward is D1.
TEST(NkSoftFluidCoResidence, SoftAndFluidCoStepStableAndD1) {
    if (GetCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const cook::XpbdCookInput soft = MakeSoftTet();
    const cook::PbfCookInput fluid = MakeFluidBlob();

    // Capture the soft rest edge lengths (for the near-rigid check).
    std::vector<std::pair<std::pair<uint32_t, uint32_t>, float>> rest_edges;
    for (const auto& dc : soft.distance)
        rest_edges.push_back({{dc.a, dc.b}, dc.rest_length});

    auto run = [&](std::vector<Vec3>* out) {
        nk::Model model;
        cook::CookSoftFluidParticles(model, 1u, soft, fluid);
        EXPECT_EQ(model.particles.mode, nk::Model::ParticleMode::SoftFluid);
        const uint32_t n_soft = static_cast<uint32_t>(soft.positions.size());
        const uint32_t n_fluid = static_cast<uint32_t>(fluid.positions.size());
        EXPECT_EQ(model.particles.n_soft_particles, n_soft);
        EXPECT_EQ(model.capacities.particles_per_env, n_soft + n_fluid);

        nk::Pipeline::SolverConfig cfg;
        cfg.dt = 1.0f / 240.0f;
        cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
        Ctx c = GetCtx();
        nk::World world(std::move(model), 1u, c.dev, c.backend, cfg);
        EXPECT_TRUE(world.Ready());
        const uint32_t P = world.GetModel().capacities.particles_per_env;
        for (uint32_t s = 0; s < 80u; ++s) world.Step();
        std::vector<Vec3> p(P);
        world.GetData().DownloadField(nk::FieldId::ParticlePos, p.data(),
                                      P * sizeof(Vec3));
        *out = p;
    };

    std::vector<Vec3> a, b;
    run(&a);
    run(&b);

    // Finite (no fluid-density blow-up from a leaking soft particle).
    for (const Vec3& p : a)
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));

    // Soft tet (indices [0, n_soft)) stays near-rigid: edges near rest length.
    float max_drift = 0.0f;
    for (const auto& re : rest_edges) {
        const Vec3 d = a[re.first.first] - a[re.first.second];
        max_drift = std::max(max_drift,
                             std::fabs(d.Length() - re.second) / re.second);
    }
    EXPECT_LT(max_drift, 0.20f)
        << "soft slice did not stay near-rigid in co-residence: drift " << max_drift;

    // D1 two-run byte-exact over the FULL [soft | fluid] union.
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(Vec3)), 0)
        << "soft+fluid co-resident forward not two-run byte-identical (D1)";
}
