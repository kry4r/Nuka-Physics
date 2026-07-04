// ---------------------------------------------------------------------------
// Config-time per-medium solver selector + the media-list validator (host cook
// assertions, no GPU).
//
// A bulk-soft body cooks as XPBD by DEFAULT (byte-identical to today) and as MPM
// when the solver selection is Mpm. The media-list validator (ValidateMedia)
// enforces the legal (kind x method) set -- cloth must be XPBD, a fluid PBF or
// MLS-MPM; MPM may co-reside with XPBD but not with PBF -- all LOUDLY (never a
// silent fallback).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <vector>

#include "import/cooker/fluid_cooker.hpp"  // CookFluidBox (predicts the MPM fluid count)
#include "math/vec3.hpp"
#include "nk/model/model.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace cook = nuka::scene::cook;
using nuka::math::Vec3;

cook::XpbdCookInput SoftInput() {
    cook::XpbdCookInput in;
    in.positions = {Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.1f, 0.0f, 0.0f},
                    Vec3{0.0f, 0.1f, 0.0f}, Vec3{0.0f, 0.0f, 0.1f}};
    in.velocities.assign(in.positions.size(), Vec3::Zero());
    in.inv_mass.assign(in.positions.size(), 1.0f);
    return in;
}

cook::MpmCookInput MpmInput() {
    cook::MpmCookInput in;
    in.positions = {Vec3{0.25f, 0.25f, 0.25f}, Vec3{0.30f, 0.25f, 0.25f}};
    in.velocities.assign(in.positions.size(), Vec3::Zero());
    in.inv_mass.assign(in.positions.size(), 1.0f);
    in.material.youngs = 1.0e4f; in.material.poisson = 0.3f;
    in.material.density = 1000.0f;
    in.grid_origin = Vec3{0.0f, 0.0f, 0.0f};
    in.grid_dims[0] = 8u; in.grid_dims[1] = 8u; in.grid_dims[2] = 8u;
    in.dx = 0.1f;
    return in;
}

}  // namespace

// Unspecified solver => XPBD (the default), byte-identical to a direct XPBD cook.
TEST(MpmSimMethodSelector, DefaultSolverCooksXpbd) {
    nk::Model via_dispatch;
    cook::XpbdCookInput in = SoftInput();
    EXPECT_EQ(in.solver, nk::Model::ParticleMode::Xpbd) << "default must be Xpbd";
    cook::CookSoftBodyParticles(via_dispatch, 1u, in, MpmInput());
    EXPECT_EQ(via_dispatch.particles.mode, nk::Model::ParticleMode::Xpbd);
    EXPECT_EQ(via_dispatch.capacities.mpm_grid_nodes_per_env, 0u);
    EXPECT_TRUE(via_dispatch.mpm_materials.empty());

    // Same body via the direct XPBD cook: same cooked mode + particle count.
    nk::Model direct;
    cook::CookXpbdParticles(direct, 1u, SoftInput());
    EXPECT_EQ(direct.particles.mode, via_dispatch.particles.mode);
    EXPECT_EQ(direct.capacities.particles_per_env,
              via_dispatch.capacities.particles_per_env);
}

// solver == Mpm => ParticleMode::Mpm + grid/material caps + F seeded identity.
TEST(MpmSimMethodSelector, MlsMpmSolverCooksMpm) {
    nk::Model m;
    cook::XpbdCookInput in = SoftInput();
    in.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(m, 1u, in, MpmInput());
    EXPECT_EQ(m.particles.mode, nk::Model::ParticleMode::Mpm);
    EXPECT_EQ(m.capacities.particles_per_env, 2u);
    EXPECT_GT(m.capacities.mpm_grid_nodes_per_env, 0u);
    EXPECT_EQ(m.capacities.mpm_material_count, 1u);
    ASSERT_EQ(m.particles.initial_F.size(), m.particles.initial_pos.size() * 9u);
    // F seeded to identity per particle.
    for (size_t p = 0; p < m.particles.initial_pos.size(); ++p) {
        EXPECT_FLOAT_EQ(m.particles.initial_F[p * 9u + 0u], 1.0f);
        EXPECT_FLOAT_EQ(m.particles.initial_F[p * 9u + 4u], 1.0f);
        EXPECT_FLOAT_EQ(m.particles.initial_F[p * 9u + 8u], 1.0f);
    }
}

// The media-list validator: cloth must be XPBD, a fluid PBF or MLS-MPM; an MLS-MPM
// medium may co-reside with an XPBD medium but not with a PBF one. Illegal pairs /
// mixes throw LOUDLY; legal cases (cloth XPBD, fluid PBF, mpm+xpbd) and empty pass.
TEST(MpmSimMethodSelector, MediaValidatorRejectsIllegalLoudly) {
    using MediaRecord = nuka::scene::MediaRecord;
    auto medium = [](MediaRecord::Kind k, MediaRecord::Method m) {
        MediaRecord r; r.kind = k; r.method = m; return r;
    };
    const MediaRecord cloth_xpbd = medium(MediaRecord::Kind::Cloth,
                                          MediaRecord::Method::Xpbd);
    const MediaRecord fluid_pbf  = medium(MediaRecord::Kind::Fluid,
                                          MediaRecord::Method::Pbf);

    EXPECT_NO_THROW(cook::ValidateMedia({}))
        << "an empty media list is a no-op (byte-identical cook)";
    EXPECT_NO_THROW(cook::ValidateMedia({cloth_xpbd, fluid_pbf}))
        << "the wired legal pair (cloth XPBD + fluid PBF) must pass";
    EXPECT_THROW(cook::ValidateMedia(
                     {medium(MediaRecord::Kind::Cloth, MediaRecord::Method::MlsMpm)}),
                 std::runtime_error)
        << "cloth -> MLS-MPM must throw (cloth stays XPBD)";
    EXPECT_THROW(cook::ValidateMedia(
                     {medium(MediaRecord::Kind::Fluid, MediaRecord::Method::Xpbd)}),
                 std::runtime_error)
        << "fluid -> XPBD must throw";
    EXPECT_NO_THROW(cook::ValidateMedia(
        {medium(MediaRecord::Kind::SoftTet, MediaRecord::Method::MlsMpm), cloth_xpbd}))
        << "an MLS-MPM medium co-resident with an XPBD medium is legal";
    EXPECT_THROW(cook::ValidateMedia(
                     {medium(MediaRecord::Kind::Granular, MediaRecord::Method::MlsMpm),
                      fluid_pbf}),
                 std::runtime_error)
        << "an MLS-MPM medium co-resident with a PBF medium must throw";
    EXPECT_NO_THROW(cook::ValidateMedia(
        {medium(MediaRecord::Kind::Fluid, MediaRecord::Method::MlsMpm)}))
        << "a lone fluid-as-MLS-MPM medium is a legal pair";
}

// A lone fluid MLS-MPM medium routes through CookSceneMedia onto ParticleMode::Mpm with
// the box lattice particle count + one cooked material + a non-empty env-private grid.
TEST(MpmSimMethodSelector, FluidMlsMpmMediaRoutesToMpmCook) {
    using MediaRecord = nuka::scene::MediaRecord;
    MediaRecord m;
    m.kind = MediaRecord::Kind::Fluid;
    m.method = MediaRecord::Method::MlsMpm;
    m.fluid_box.min = {-0.05f, -0.05f, 0.0f};
    m.fluid_box.max = {0.05f, 0.05f, 0.10f};
    m.fluid_box.spacing = 0.02f;
    m.mpm.density = 1000.0f;
    m.mpm.model_kind = 3.0f;          // weakly-compressible fluid.
    m.mpm.bulk_modulus = 2.0e5f;
    m.mpm.tait_gamma = 7.0f;
    m.mpm.dx = 0.02f;
    m.mpm.substeps = 20u;

    nuka::import::cooker::FluidBoxSpec spec;
    spec.min_corner = m.fluid_box.min; spec.max_corner = m.fluid_box.max;
    spec.spacing = 0.02f; spec.rest_density = 1000.0f;
    const uint32_t expect = static_cast<uint32_t>(
        nuka::import::cooker::CookFluidBox(spec).positions.size());
    ASSERT_GT(expect, 0u);

    nk::Model model;
    cook::CookSceneMedia(model, 1u, {m});
    EXPECT_EQ(model.particles.mode, nk::Model::ParticleMode::Mpm);
    EXPECT_EQ(model.capacities.particles_per_env, expect);
    EXPECT_EQ(model.capacities.mpm_material_count, 1u);
    EXPECT_GT(model.capacities.mpm_grid_nodes_per_env, 0u);
    ASSERT_FALSE(model.mpm_materials.empty());
    EXPECT_FLOAT_EQ(model.mpm_materials[0].model_kind, 3.0f);
    EXPECT_FLOAT_EQ(model.mpm_materials[0].bulk_modulus, 2.0e5f);
}

// A lone tet-soft MLS-MPM medium routes onto ParticleMode::Mpm from the sphere lattice.
TEST(MpmSimMethodSelector, SoftTetMlsMpmMediaRoutesToMpmCook) {
    using MediaRecord = nuka::scene::MediaRecord;
    MediaRecord m;
    m.kind = MediaRecord::Kind::SoftTet;
    m.method = MediaRecord::Method::MlsMpm;
    m.tet_sphere.center = {0.0f, 0.0f, 0.20f};
    m.tet_sphere.radius = 0.10f;
    m.tet_sphere.cells = 8u;
    m.tet_sphere.cell_len = 2.0f * 0.10f / 8.0f;
    m.mpm.density = 1000.0f;
    m.mpm.youngs = 3.0e4f; m.mpm.poisson = 0.3f; m.mpm.model_kind = 0.0f;
    m.mpm.dx = 0.025f; m.mpm.substeps = 20u;

    nk::Model model;
    cook::CookSceneMedia(model, 1u, {m});
    EXPECT_EQ(model.particles.mode, nk::Model::ParticleMode::Mpm);
    EXPECT_GT(model.capacities.particles_per_env, 0u);
    EXPECT_EQ(model.capacities.mpm_material_count, 1u);
}
