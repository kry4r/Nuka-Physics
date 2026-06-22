// ---------------------------------------------------------------------------
// FR7 config-time per-medium solver selector (host cook assertions).
//
// A bulk-soft body cooks as XPBD by DEFAULT (byte-identical to today) and as MPM
// when the solver selection is Mpm; cloth -> mlsmpm and fluid -> mlsmpm are
// rejected LOUDLY at cook (never a silent fallback). No GPU needed.
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

// cloth -> mlsmpm and fluid -> mlsmpm are rejected LOUDLY (cloth stays XPBD,
// fluid stays PBF); the default selector is a no-op (no throw).
TEST(MpmSimMethodSelector, ClothFluidMpmRejectedLoudly) {
    EXPECT_THROW(cook::RejectUnsupportedClothFluidMpm(
                     nk::Model::ParticleMode::Mpm, nk::Model::ParticleMode::Xpbd),
                 std::runtime_error)
        << "cloth -> mlsmpm must throw (cloth stays XPBD permanently)";
    EXPECT_THROW(cook::RejectUnsupportedClothFluidMpm(
                     nk::Model::ParticleMode::Xpbd, nk::Model::ParticleMode::Mpm),
                 std::runtime_error)
        << "fluid -> mlsmpm must throw (fluid stays PBF in the first batch)";
    EXPECT_NO_THROW(cook::RejectUnsupportedClothFluidMpm(
        nk::Model::ParticleMode::Xpbd, nk::Model::ParticleMode::Xpbd))
        << "the default (Xpbd) selector must be a no-op (byte-identical cook)";
}
