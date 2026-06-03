// ---------------------------------------------------------------------------
// v0.7 p10-B GATE 3: PBF fluid COOKER tests.
//
// The fluid cooker is a THIN cook-time CPU layer: a programmatic FluidBoxSpec ->
// runtime::fluid::PbfParticleSet, by sampling a cell-centered uniform lattice
// filling the AABB and computing the uniform mass = rest_density * spacing^3.
// These tests assert, on the RETURNED particle set (pure CPU -- no device upload):
//
//   1. COUNT: a box of known size + spacing cooks to the hand-computed lattice
//      count (n_axis = floor(L/s); total = nx*ny*nz). The cell-centered convention
//      makes the count unambiguous (no inclusive-endpoint off-by-one).
//   2. MASS: the uniform particle_mass == rest_density * spacing^3 (the mass of
//      fluid filling one cubic cell).
//   3. PLACEMENT: every particle sits at a cell CENTER inside the box; positions
//      are on the expected lattice (corner + (k+0.5)*s).
//   4. DETERMINISM: cooking the same spec twice yields a BYTE-IDENTICAL particle
//      set (positions + velocities + mass).
//   5. EDGE CASES: bad spacing / empty box / bad density -> empty set (no throw).
//   6. SPHERE CARVE: the optional inscribed-sphere fill keeps a strict subset of
//      the box lattice (all within the sphere; fewer than the box).
// ---------------------------------------------------------------------------

#include "import/cooker/fluid_cooker.hpp"

#include "math/vec3.hpp"
#include "runtime/fluid/pbf_world.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

using nuka::import::cooker::CookFluidBox;
using nuka::import::cooker::CookFluidSphere;
using nuka::import::cooker::FluidBoxLatticeCounts;
using nuka::import::cooker::FluidBoxSpec;
using nuka::import::cooker::FluidLatticeCounts;
using nuka::import::cooker::FluidParticleMass;
using nuka::math::Vec3;
using nuka::runtime::fluid::PbfParticleSet;

} // namespace

// Gate 3(1) + 3(2): a known box cooks to the hand-computed count + cell mass.
TEST(FluidCooker, BoxCooksToExpectedCountAndMass) {
    FluidBoxSpec spec;
    spec.min_corner = Vec3{0.0f, 0.0f, 0.0f};
    spec.max_corner = Vec3{1.0f, 0.5f, 0.2f};  // L = (1.0, 0.5, 0.2)
    spec.spacing = 0.1f;
    spec.rest_density = 1000.0f;

    // n = floor(L/s): floor(1.0/0.1)=10, floor(0.5/0.1)=5, floor(0.2/0.1)=2.
    // (Note: floor of exactly 10/5/2 is robust here -- the test below also accepts
    //  the float-floor result and asserts the cooked set matches the count fn.)
    const FluidLatticeCounts counts = FluidBoxLatticeCounts(spec);
    EXPECT_EQ(counts.nx, 10u);
    EXPECT_EQ(counts.ny, 5u);
    EXPECT_EQ(counts.nz, 2u);
    EXPECT_EQ(counts.total, 10u * 5u * 2u);

    const PbfParticleSet ps = CookFluidBox(spec);
    EXPECT_EQ(static_cast<uint32_t>(ps.positions.size()), counts.total);
    EXPECT_EQ(ps.velocities.size(), ps.positions.size());

    // mass = rest_density * spacing^3 = 1000 * 0.001 = 1.0.
    const float expected_mass = 1000.0f * 0.1f * 0.1f * 0.1f;
    EXPECT_FLOAT_EQ(ps.particle_mass, expected_mass);
    EXPECT_FLOAT_EQ(ps.particle_mass, FluidParticleMass(spec));

    // Velocities seeded to zero.
    for (const Vec3& v : ps.velocities) {
        EXPECT_FLOAT_EQ(v.x, 0.0f);
        EXPECT_FLOAT_EQ(v.y, 0.0f);
        EXPECT_FLOAT_EQ(v.z, 0.0f);
    }
}

// Gate 3(3): every particle sits at a cell center on the lattice, inside the box.
TEST(FluidCooker, ParticlesAreCellCenteredInsideBox) {
    FluidBoxSpec spec;
    spec.min_corner = Vec3{-0.2f, 1.0f, 0.5f};
    spec.max_corner = Vec3{0.3f, 1.4f, 0.8f};
    spec.spacing = 0.05f;
    spec.rest_density = 1.0f;

    const FluidLatticeCounts counts = FluidBoxLatticeCounts(spec);
    const PbfParticleSet ps = CookFluidBox(spec);
    ASSERT_EQ(static_cast<uint32_t>(ps.positions.size()), counts.total);

    const float s = spec.spacing;
    const float half = 0.5f * s;
    // Reconstruct the expected x-fastest cell-centered lattice and compare.
    std::size_t idx = 0;
    for (uint32_t iz = 0; iz < counts.nz; ++iz) {
        for (uint32_t iy = 0; iy < counts.ny; ++iy) {
            for (uint32_t ix = 0; ix < counts.nx; ++ix) {
                const Vec3 expected{
                    spec.min_corner.x + static_cast<float>(ix) * s + half,
                    spec.min_corner.y + static_cast<float>(iy) * s + half,
                    spec.min_corner.z + static_cast<float>(iz) * s + half};
                const Vec3 got = ps.positions[idx];
                EXPECT_FLOAT_EQ(got.x, expected.x) << "at idx " << idx;
                EXPECT_FLOAT_EQ(got.y, expected.y) << "at idx " << idx;
                EXPECT_FLOAT_EQ(got.z, expected.z) << "at idx " << idx;
                // Inside the box (cell centers are strictly interior).
                EXPECT_GE(got.x, spec.min_corner.x);
                EXPECT_LE(got.x, spec.max_corner.x);
                EXPECT_GE(got.y, spec.min_corner.y);
                EXPECT_LE(got.y, spec.max_corner.y);
                EXPECT_GE(got.z, spec.min_corner.z);
                EXPECT_LE(got.z, spec.max_corner.z);
                ++idx;
            }
        }
    }
}

// Gate 3(4): two cooks of the same spec are byte-identical.
TEST(FluidCooker, CookIsDeterministic) {
    FluidBoxSpec spec;
    spec.min_corner = Vec3{0.0f, 0.0f, 0.0f};
    spec.max_corner = Vec3{0.7f, 0.7f, 0.7f};
    spec.spacing = 0.05f;
    spec.rest_density = 998.0f;

    const PbfParticleSet a = CookFluidBox(spec);
    const PbfParticleSet b = CookFluidBox(spec);

    ASSERT_EQ(a.positions.size(), b.positions.size());
    ASSERT_EQ(a.velocities.size(), b.velocities.size());
    EXPECT_FLOAT_EQ(a.particle_mass, b.particle_mass);
    EXPECT_EQ(std::memcmp(a.positions.data(), b.positions.data(),
                          a.positions.size() * sizeof(Vec3)), 0)
        << "two cooks produced different position sets (non-deterministic)";
    EXPECT_EQ(std::memcmp(a.velocities.data(), b.velocities.data(),
                          a.velocities.size() * sizeof(Vec3)), 0)
        << "two cooks produced different velocity sets (non-deterministic)";
}

// Gate 3(5): degenerate specs cook to an empty set without throwing.
TEST(FluidCooker, BadSpecsCookToEmpty) {
    FluidBoxSpec base;
    base.min_corner = Vec3{0.0f, 0.0f, 0.0f};
    base.max_corner = Vec3{1.0f, 1.0f, 1.0f};
    base.spacing = 0.1f;
    base.rest_density = 1.0f;

    {  // spacing <= 0
        FluidBoxSpec s = base; s.spacing = 0.0f;
        EXPECT_TRUE(CookFluidBox(s).positions.empty());
        EXPECT_EQ(FluidBoxLatticeCounts(s).total, 0u);
    }
    {  // rest_density <= 0
        FluidBoxSpec s = base; s.rest_density = 0.0f;
        EXPECT_TRUE(CookFluidBox(s).positions.empty());
        EXPECT_EQ(FluidBoxLatticeCounts(s).total, 0u);
    }
    {  // inverted / empty box (max < min)
        FluidBoxSpec s = base; s.max_corner = Vec3{-1.0f, -1.0f, -1.0f};
        EXPECT_TRUE(CookFluidBox(s).positions.empty());
        EXPECT_EQ(FluidBoxLatticeCounts(s).total, 0u);
    }
    {  // spacing larger than every extent -> floor(L/s) == 0 on each axis.
        FluidBoxSpec s = base; s.spacing = 2.0f;
        EXPECT_TRUE(CookFluidBox(s).positions.empty());
        EXPECT_EQ(FluidBoxLatticeCounts(s).total, 0u);
    }
}

// Gate 3(6): the optional inscribed-sphere carve keeps a strict in-sphere subset.
TEST(FluidCooker, SphereCarveKeepsInSphereSubset) {
    FluidBoxSpec spec;
    spec.min_corner = Vec3{0.0f, 0.0f, 0.0f};
    spec.max_corner = Vec3{1.0f, 1.0f, 1.0f};
    spec.spacing = 0.1f;
    spec.rest_density = 1.0f;

    const PbfParticleSet box = CookFluidBox(spec);
    const PbfParticleSet sph = CookFluidSphere(spec);

    EXPECT_FLOAT_EQ(sph.particle_mass, box.particle_mass);
    EXPECT_EQ(sph.velocities.size(), sph.positions.size());
    // The sphere is a strict subset of the box lattice.
    EXPECT_GT(sph.positions.size(), 0u);
    EXPECT_LT(sph.positions.size(), box.positions.size());

    // Every kept particle lies within the inscribed sphere (center = box center,
    // radius = half the smallest extent = 0.5 here).
    const Vec3 center{0.5f, 0.5f, 0.5f};
    const float radius = 0.5f;
    const float r2 = radius * radius;
    for (const Vec3& p : sph.positions) {
        const float dx = p.x - center.x;
        const float dy = p.y - center.y;
        const float dz = p.z - center.z;
        EXPECT_LE(dx * dx + dy * dy + dz * dz, r2 + 1.0e-6f);
    }
}
