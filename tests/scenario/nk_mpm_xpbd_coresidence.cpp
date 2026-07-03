// ---------------------------------------------------------------------------
// MLS-MPM + XPBD co-residence in ONE nk::Model ([mpm | xpbd] layout). Builds a
// granular column (MPM, slice [0, n_mpm)) and a perimeter-pinned cloth membrane
// (XPBD, slice [n_mpm, P)) co-resident via scene::cook::CookMpmXpbd, then co-steps
// nk::World on the MpmXpbd particle-op path. Asserts:
//
//   1. SUPERSET BYTE-IDENTITY: an MPM-only co-residence cook == CookMpmParticles;
//      an XPBD-only co-residence cook == CookXpbdParticles (single-medium goldens
//      unaffected).
//   2. CO-RESIDENT LAYOUT: mode == MpmXpbd, n_mpm_particles == n_mpm, the XPBD
//      constraint indices land in [n_mpm, P), the slices do not overlap.
//   3. CO-STEP, NO CROSS-CORRUPTION: the granular slice settles (finite, no grid
//      escape) while the cloth slice's pinned corners stay put and its interior
//      sags under gravity -- the MPM transfer never overwrites a cloth particle and
//      XpbdProject never zeroes an MPM velocity. Run with env_count = 2 to catch
//      per-env sub-slice striding.
//   4. D1: the co-resident forward is two-run byte-exact.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/op_schema.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace ns = nuka::scene;
using nuka::math::Vec3;

struct Backend { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    return cfg;
}

constexpr float kDx = 0.02f;
constexpr float kColHalf = 0.03f;    // granular column half footprint.
constexpr float kColTop  = 0.06f;    // granular column height.
constexpr float kClothSpacing = 0.03f;
constexpr float kClothZ = 0.30f;     // cloth membrane placed well above the column.

// A compact dry-sand column (MLS-MPM, model_kind 4) resting on the static floor.
cook::MpmCookInput BuildGranularColumn() {
    cook::MpmCookInput in;
    const float pdx = kDx * 0.5f;
    for (float x = -kColHalf; x <= kColHalf + 1e-5f; x += pdx)
        for (float y = -kColHalf; y <= kColHalf + 1e-5f; y += pdx)
            for (float z = pdx; z <= kColTop + 1e-5f; z += pdx)
                in.positions.push_back(Vec3{x, y, z});
    const size_t n = in.positions.size();
    const float density = 1600.0f, vol0 = pdx * pdx * pdx;
    in.velocities.assign(n, Vec3::Zero());
    in.inv_mass.assign(n, 1.0f / (density * vol0));
    in.vol0.assign(n, vol0);
    in.material.youngs = 3.0e5f; in.material.poisson = 0.3f;
    in.material.density = density; in.material.model_kind = 4.0f;
    in.material.dp_friction = 35.0f;
    in.grid_origin = Vec3{-kColHalf - 4.0f * kDx, -kColHalf - 4.0f * kDx, -3.0f * kDx};
    in.grid_dims[0] = static_cast<uint32_t>((2.0f * (kColHalf + 4.0f * kDx)) / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>((kColTop + 12.0f * kDx) / kDx) + 4u;
    in.dx = kDx; in.substeps = 12u;
    in.floor_normal = Vec3{0, 0, 1}; in.floor_d = 0.0f; in.floor_friction = 0.7f;
    return in;
}

// A perimeter-pinned cloth membrane (XPBD), so the corners are anchored and the
// interior sags -- a pinned particle would move only if the MPM transfer wrongly
// overwrote it, so the anchors double as the no-cross-corruption probe.
cook::XpbdCookInput BuildPinnedMembrane(uint32_t n) {
    ns::MediaRecord m;
    m.kind = ns::MediaRecord::Kind::Cloth;
    m.method = ns::MediaRecord::Method::Xpbd;
    m.cloth_grid.nx = n; m.cloth_grid.ny = n;
    m.cloth_grid.spacing = kClothSpacing;
    m.cloth_grid.origin = Vec3{0.0f, 0.0f, kClothZ};
    m.cloth_grid.free = false;    // pin the perimeter (interior sags).
    m.xpbd.particle_mass = 0.01f;
    m.xpbd.iters = 8u;
    m.xpbd.distance_alpha = 1.0e-7f;
    m.xpbd.bend_alpha = 1.0e-4f;
    return cook::BuildClothXpbdInput(m);
}

}  // namespace

// Assertion 1: the one-medium co-residence cooks == the single-medium cooks.
TEST(NkMpmXpbdCoResidence, SupersetMatchesSingleSystemCook) {
    const cook::MpmCookInput mpm = BuildGranularColumn();
    const cook::XpbdCookInput cloth = BuildPinnedMembrane(5u);

    // MPM-only: CookMpmXpbd(mpm, EMPTY) == CookMpmParticles(mpm).
    {
        nk::Model a, b;
        cook::CookMpmXpbd(a, 1u, mpm, cook::XpbdCookInput{});
        cook::CookMpmParticles(b, 1u, mpm);
        EXPECT_EQ(a.particles.mode, nk::Model::ParticleMode::Mpm);
        EXPECT_EQ(a.capacities.particles_per_env, b.capacities.particles_per_env);
        EXPECT_EQ(a.capacities.mpm_grid_nodes_per_env,
                  b.capacities.mpm_grid_nodes_per_env);
        ASSERT_EQ(a.particles.initial_pos.size(), b.particles.initial_pos.size());
        EXPECT_EQ(std::memcmp(a.particles.initial_pos.data(),
                              b.particles.initial_pos.data(),
                              a.particles.initial_pos.size() * sizeof(Vec3)), 0);
        ASSERT_EQ(a.particles.initial_F.size(), b.particles.initial_F.size());
        EXPECT_EQ(std::memcmp(a.particles.initial_F.data(),
                              b.particles.initial_F.data(),
                              a.particles.initial_F.size() * sizeof(float)), 0);
    }
    // XPBD-only: CookMpmXpbd(EMPTY, cloth) == CookXpbdParticles(cloth).
    {
        nk::Model a, b;
        cook::CookMpmXpbd(a, 1u, cook::MpmCookInput{}, cloth);
        cook::CookXpbdParticles(b, 1u, cloth);
        EXPECT_EQ(a.particles.mode, nk::Model::ParticleMode::Xpbd);
        EXPECT_EQ(a.capacities.particles_per_env, b.capacities.particles_per_env);
        EXPECT_EQ(a.capacities.dist_cons_per_env, b.capacities.dist_cons_per_env);
        ASSERT_EQ(a.particles.dist_a.size(), b.particles.dist_a.size());
        EXPECT_EQ(std::memcmp(a.particles.dist_a.data(), b.particles.dist_a.data(),
                              a.particles.dist_a.size() * sizeof(uint32_t)), 0);
    }
}

// Assertion 2: the co-resident cook lays out [mpm | xpbd] with the split + rebased
// constraint indices.
TEST(NkMpmXpbdCoResidence, CoResidentCookLayout) {
    const cook::MpmCookInput mpm = BuildGranularColumn();
    const cook::XpbdCookInput cloth = BuildPinnedMembrane(5u);
    const uint32_t n_mpm = static_cast<uint32_t>(mpm.positions.size());
    const uint32_t n_xpbd = static_cast<uint32_t>(cloth.positions.size());

    nk::Model m;
    cook::CookMpmXpbd(m, 1u, mpm, cloth);
    const nk::Model::ModelParticles& mp = m.particles;
    EXPECT_EQ(mp.mode, nk::Model::ParticleMode::MpmXpbd);
    EXPECT_EQ(mp.n_mpm_particles, n_mpm);
    EXPECT_EQ(m.capacities.particles_per_env, n_mpm + n_xpbd);
    // The MPM slice keeps the sand positions in [0, n_mpm); the cloth positions
    // land in [n_mpm, P).
    ASSERT_EQ(mp.initial_pos.size(), n_mpm + n_xpbd);
    for (uint32_t i = 0; i < n_mpm; ++i)
        EXPECT_EQ(std::memcmp(&mp.initial_pos[i], &mpm.positions[i], sizeof(Vec3)), 0);
    for (uint32_t i = 0; i < n_xpbd; ++i)
        EXPECT_EQ(std::memcmp(&mp.initial_pos[n_mpm + i], &cloth.positions[i],
                              sizeof(Vec3)), 0);
    // Every XPBD distance constraint references the XPBD slice [n_mpm, P).
    ASSERT_FALSE(mp.dist_a.empty());
    for (uint32_t v : mp.dist_a) { EXPECT_GE(v, n_mpm); EXPECT_LT(v, n_mpm + n_xpbd); }
    for (uint32_t v : mp.dist_b) { EXPECT_GE(v, n_mpm); EXPECT_LT(v, n_mpm + n_xpbd); }
    // The MPM continuum fields stay sized to the MPM slice.
    EXPECT_EQ(mp.initial_vol0.size(), n_mpm);
    EXPECT_EQ(mp.initial_material_id.size(), n_mpm);
}

// Assertion 3/4: the two media co-step in ONE world (env_count = 2) without
// corrupting each other, and the forward is D1.
TEST(NkMpmXpbdCoResidence, CoStepNoCrossCorruptionAndD1) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    const cook::MpmCookInput mpm = BuildGranularColumn();
    const cook::XpbdCookInput cloth = BuildPinnedMembrane(5u);
    const uint32_t n_mpm = static_cast<uint32_t>(mpm.positions.size());
    const uint32_t n_xpbd = static_cast<uint32_t>(cloth.positions.size());
    constexpr uint32_t E = 2u;   // catches per-env sub-slice striding bugs.

    // The pinned cloth corners: perimeter particles (BuildClothXpbdInput pins the
    // whole perimeter of the n x n lattice, laid out row-major).
    const uint32_t nx = 5u;

    auto run = [&](std::vector<Vec3>* out) {
        nk::Model m;
        m.capacities.env_count = E;
        cook::CookMpmXpbd(m, E, mpm, cloth);
        EXPECT_EQ(m.particles.mode, nk::Model::ParticleMode::MpmXpbd);
        nk::World w(std::move(m), E, b.dev, b.backend, Cfg());
        EXPECT_TRUE(w.Ready());
        const uint32_t P = w.GetModel().capacities.particles_per_env;
        for (uint32_t s = 0; s < 240u; ++s) w.Step();
        std::vector<Vec3> pos(static_cast<size_t>(P) * E);
        w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(),
                                  pos.size() * sizeof(Vec3));
        *out = pos;
        return P;
    };

    std::vector<Vec3> a, bpos;
    const uint32_t P = run(&a);
    run(&bpos);
    ASSERT_EQ(P, n_mpm + n_xpbd);

    // Everything finite; no grid escape.
    for (const Vec3& p : a)
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));

    for (uint32_t e = 0; e < E; ++e) {
        const Vec3* env = a.data() + static_cast<size_t>(e) * P;
        // Granular slice [0, n_mpm): settled on the floor (all above the floor plane,
        // top not risen -- the transfer ran, gravity pulled it down / compacted).
        float g_top0 = -1e30f, g_top = -1e30f, g_minz = 1e30f;
        for (uint32_t i = 0; i < n_mpm; ++i) {
            g_top0 = std::max(g_top0, mpm.positions[i].z);
            g_top = std::max(g_top, env[i].z);
            g_minz = std::min(g_minz, env[i].z);
        }
        EXPECT_GT(g_minz, -kDx) << "granular fell through the floor (env " << e << ")";
        EXPECT_LT(g_top, g_top0 + 3.0f * kDx)
            << "granular column blew up instead of settling (env " << e << ")";
        // Cloth slice [n_mpm, P): the pinned corners held at the initial z (the MPM
        // transfer did NOT overwrite them), while the interior sagged under gravity.
        auto corner = [&](uint32_t ix, uint32_t iy) {
            return n_mpm + iy * nx + ix;
        };
        const uint32_t corners[4] = {corner(0, 0), corner(nx - 1, 0),
                                     corner(0, nx - 1), corner(nx - 1, nx - 1)};
        for (uint32_t c : corners) {
            EXPECT_NEAR(env[c].z, kClothZ, 1.0e-3f)
                << "a pinned cloth corner moved (cross-corruption) env " << e;
        }
        // The interior sags strictly below the pinned corners under gravity (a fully
        // inert cloth would stay exactly at kClothZ). The stiff (alpha 1e-7) small
        // membrane bows only microns, so the margin is small but deterministic (D1).
        const uint32_t center = n_mpm + (nx / 2u) * nx + (nx / 2u);
        EXPECT_LT(env[center].z, kClothZ - 1.0e-5f)
            << "the cloth interior did not sag -- XPBD inert? env " << e
            << " center_z " << env[center].z;
        EXPECT_GT(env[center].z, kClothZ - 0.10f)
            << "the cloth interior collapsed -- overwritten by G2P? env " << e;
    }

    // D1: two-run byte-exact over the FULL [mpm | xpbd] union across both envs.
    ASSERT_EQ(a.size(), bpos.size());
    EXPECT_EQ(std::memcmp(a.data(), bpos.data(), a.size() * sizeof(Vec3)), 0)
        << "MpmXpbd co-resident forward not two-run byte-identical (D1)";

    std::fprintf(stderr, "[mpm-xpbd costep] n_mpm=%u n_xpbd=%u P=%u E=%u OK\n",
                 n_mpm, n_xpbd, P, E);
}
