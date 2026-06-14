#pragma once
// ---------------------------------------------------------------------------
// nuka::import::cooker -- PBF FLUID COOKER (v0.7 p10-B)
// ---------------------------------------------------------------------------
//
// A THIN, deterministic, cook-time CPU layer that turns a programmatic fluid
// VOLUME spec into the runtime::fluid::PbfParticleSet the nk PBF world cooks. It
// owns NO simulation math: it SAMPLES particle initial positions filling the
// authored volume on a uniform lattice and computes the per-particle mass from
// the rest density and the lattice cell volume. (Same posture as the p09-D
// xpbd_cooker: a pure spec -> runtime-state mapping; the per-step sim is GPU.)
//
// SAMPLING CONVENTION (cell-centered): for an AABB volume of size L along an axis
// and a particle spacing s, the cooker places
//     n = floor(L / s)   particles along that axis,
// each at the CENTER of its lattice cell:
//     x_k = lo + (k + 0.5) * s ,   k = 0 .. n-1.
// Cell-centered sampling makes mass = rest_density * s^3 EXACT (one particle owns
// exactly one cell of volume s^3) and the count unambiguous (no inclusive-endpoint
// off-by-one). The fill order is x fastest, then y, then z (a fixed nested loop)
// so two cooks of the same spec yield a BYTE-IDENTICAL particle set.
//
// MASS: PbfParticleSet carries a SINGLE uniform particle_mass (p10-A/B), so the
// cooker emits one mass for the whole volume:
//     particle_mass = rest_density * spacing^3
// (the mass of fluid filling one cubic cell at the rest density). This matches the
// SPH convention that, at rest spacing, sum_j m * Poly6 reproduces rest_density.
//
// VOLUME SCOPE (p10-B): the cooker fills an axis-aligned BOX (AABB). An OPTIONAL
// coarse SPHERE inside-test (CookFluidSphere) is provided to demonstrate a
// non-box volume by carving particles outside the inscribed sphere. A general
// closed-MESH point-in-mesh fill is DEFERRED to the p16 asset pipeline (same
// posture as the xpbd_cooker's deferred USD/MJCF file parsing): the importer will
// populate a richer spec and call into a mesh-aware fill there. This file does NOT
// parse files (no USD nuka:fluid_volume reader -- that layers on at p16, #19/#26).
// ---------------------------------------------------------------------------

#include "import/cooker/fluid_cooker_types.hpp"  // PbfParticleSet (CUDA-free)
#include "math/vec3.hpp"

#include <cstdint>

namespace nuka::import::cooker {

// Programmatic description of a BOX (AABB) fluid volume to cook. The future USD
// importer (p16) populates this from an authored nuka:fluid_volume; today the
// tests + any in-tree caller build it directly.
struct FluidBoxSpec {
    // The AABB to fill, as [min_corner, max_corner]. max must be > min on every
    // axis (an empty / inverted box cooks to zero particles).
    math::Vec3 min_corner = {0.0f, 0.0f, 0.0f};
    math::Vec3 max_corner = {1.0f, 1.0f, 1.0f};

    // Uniform particle spacing s (lattice cell edge). Must be > 0.
    float spacing = 0.1f;

    // Rest density rho0 of the fluid. Used ONLY to set the per-particle mass
    // (mass = rest_density * spacing^3); it does NOT affect the sampled positions.
    // Must be > 0.
    float rest_density = 1000.0f;
};

// Number of lattice particles the box spec cooks to PER AXIS (n = floor(L/s)) and
// in total. Exposed so the cook test can assert a hand-computed count without
// running the full cook. Zero on any axis (or bad spacing) => zero total.
struct FluidLatticeCounts {
    uint32_t nx = 0u;
    uint32_t ny = 0u;
    uint32_t nz = 0u;
    uint32_t total = 0u;
};
FluidLatticeCounts FluidBoxLatticeCounts(const FluidBoxSpec& spec);

// The uniform per-particle mass a box spec cooks to: rest_density * spacing^3.
// Exposed so the cook test asserts the mass mapping directly.
float FluidParticleMass(const FluidBoxSpec& spec);

// Cook a BOX fluid volume into a PbfParticleSet. PURE + deterministic: samples the
// cell-centered uniform lattice (x fastest), zero initial velocities, and the
// uniform mass = rest_density * spacing^3. The result feeds the nk PBF world cook
// (nk::Model::ModelParticles, mode == Pbf). Invalid spec (spacing <= 0,
// rest_density <= 0, or an empty box) cooks to an EMPTY set (no throw -- mirrors
// the xpbd_cooker's empty-input tolerance).
runtime::fluid::PbfParticleSet CookFluidBox(const FluidBoxSpec& spec);

// OPTIONAL non-box demo: cook the box lattice, then KEEP only particles whose
// cell center lies within the sphere inscribed in the box (center = box center,
// radius = half the SMALLEST box extent). A coarse inside-test that demonstrates
// a carved volume; a general mesh fill is deferred to p16. Deterministic (the
// surviving particles keep their x-fastest order).
runtime::fluid::PbfParticleSet CookFluidSphere(const FluidBoxSpec& spec);

} // namespace nuka::import::cooker
