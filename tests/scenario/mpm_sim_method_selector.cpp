// ---------------------------------------------------------------------------
// Config-time per-medium solver selector + the media-list validator (host cook
// assertions, no GPU).
//
// A bulk-soft body cooks as XPBD by DEFAULT (byte-identical to today) and as MPM
// when the solver selection is Mpm. The media-list validator (ValidateMedia)
// enforces the legal (kind x method) set -- cloth must be XPBD, a fluid PBF or
// MLS-MPM -- and rejects an MLS-MPM medium co-resident with an XPBD/PBF medium,
// all LOUDLY (never a silent fallback).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <vector>

#include "math/vec3.hpp"
#include "nk/model/model.hpp"
#include "scene/cook/cook_to_model.hpp"

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
// medium may not co-reside with an XPBD/PBF medium. Illegal pairs / mixes throw
// LOUDLY; the wired legal cases (cloth XPBD, fluid PBF) and the empty list pass.
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
    EXPECT_THROW(cook::ValidateMedia(
                     {medium(MediaRecord::Kind::SoftTet, MediaRecord::Method::MlsMpm),
                      cloth_xpbd}),
                 std::runtime_error)
        << "an MLS-MPM medium co-resident with an XPBD/PBF medium must throw";
    EXPECT_NO_THROW(cook::ValidateMedia(
        {medium(MediaRecord::Kind::Fluid, MediaRecord::Method::MlsMpm)}))
        << "a lone fluid-as-MLS-MPM medium is a legal pair";
}
