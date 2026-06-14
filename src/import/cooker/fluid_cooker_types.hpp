#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::fluid -- CUDA-FREE PBF param structs + a pure-host density helper
// ---------------------------------------------------------------------------
//
// M9 T11 soft/fluid Phase 2b: the PBF stepper (runtime/fluid/pbf_world.{cu,hpp})
// is being DELETED -- the forward sim is op-ified onto the unified nk core
// (phi/backend_cuda/ops/particles.cu, run by nk::World). The plain data structs
// the FLUID COOKER and the PBF tests still need -- PbfParticleSet / PbfParams --
// are relocated here, into a header with NO GPU/World dependency (no buffer, no
// device context, no .cu). The host density helper (ComputePbfDensities) that the
// rho0-calibration sites used to take from the GPU stepper is reproduced here as a
// CUDA-free brute-force Poly6 over the particle set.
//
// POLY6 MATCH: the density math is BYTE-faithful to the GPU path -- the same
// textbook normalization constant (315/(64*pi*h^9), the MakePbfKernelCoeffs recipe
// from pbf_kernels.cuh) and the same accumulation shape (self term first, r2 = 0,
// then every other particle's m*Poly6(|r|^2)). The GPU walks a 32-cap uniform-grid
// neighbor list in ascending-index order; the calibration scenes are sized so the
// grid NEVER truncates (every PBF test asserts max_truncated_particle_count == 0),
// so the grid neighbor SET == "all particles within h" == the host brute-force set.
// The only difference is the float reduction ORDER (grid-CSR vs particle-index),
// which perturbs rho_i at the ~1e-6 relative level -- far inside the calibration /
// drift tolerances the sites assert (|C| < 0.10, drift < 5%, over-density < 15%).
// This is the SAME host Poly6 mirror nk_particle_equivalence.cpp already uses for
// its PBF density check.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nuka::runtime::fluid {

// Host-side description of a PBF particle set. Mass is UNIFORM across the fluid
// (p10-A); per-particle mass is a future extension.
struct PbfParticleSet {
    std::vector<math::Vec3> positions;
    std::vector<math::Vec3> velocities;
    float particle_mass = 1.0f;  // m (same for every particle).
};

// PBF simulation parameters. rho0 is the rest density the incompressibility
// constraint pulls toward; it is typically CALIBRATED numerically from this
// engine's own Poly6 kernel over a rest lattice (so a kernel-constant slip is
// absorbed). See ComputePbfDensities below + the tests for the calibration recipe.
struct PbfParams {
    float support_radius_h = 0.1f;    // SPH support radius h (also the grid query
                                      // radius == cell_size; >= ~1.5*dx, < the cap).
    float rest_density_rho0 = 1.0f;   // rho0 (calibrate per-scene).
    float cfm_epsilon = 1.0e-6f;      // constraint-force-mixing relaxation in the
                                      // lambda denominator (M&M eq.11; > 0).
    uint32_t solver_iterations = 4u;  // N_pbf density-projection iterations (3-5).
    bool clamp_to_overdensity = true; // only correct C_i > 0 (rho_i > rho0): kills
                                      // spurious free-surface cohesion + improves
                                      // stability (a standard PBF choice).

    // --- p10-B polish passes (DEFAULT 0 => inert) ---------------------------
    // XSPH viscosity (M&M 2013 eq.17): a post-finalize velocity smoothing.
    float xsph_viscosity_c = 0.0f;
    // Surface tension -- BASIC Akinci et al. 2013 COHESION term only.
    float surface_tension_gamma = 0.0f;
};

// Compute the rest density at every particle by a single Poly6 density pass over
// the current positions (no integration / no constraint projection). Pure host,
// brute-force O(N^2); the calibration scenes are small (a few thousand particles).
// Used to numerically CALIBRATE rho0 from a rest lattice: build the lattice, call
// this, take a deep-interior particle's density. Returns the per-particle rho_i.
//
// The float recipe is identical to the GPU ComputeDensityKernel: the running sum
// starts from the self term (r2 = 0 -> m * poly6 * h^6), then adds every other
// particle's m * poly6 * (h^2 - r^2)^3 for r < h. poly6 == 315/(64*pi*h^9), the
// MakePbfKernelCoeffs value.
inline std::vector<float> ComputePbfDensities(const PbfParticleSet& particles,
                                              const PbfParams& params) {
    const std::size_t n = particles.positions.size();
    if (n == 0u) {
        return {};
    }
    const float h = params.support_radius_h;
    // GPU contract: support_radius_h > 0. Mirror it (a zero/negative h is a misuse;
    // the kernel coefficient would be non-finite). Return zeros rather than throw
    // is wrong -- match the GPU which throws; but keep this header exception-light:
    // an h <= 0 yields a non-finite coeff that the caller's assert will catch. The
    // calibration sites always pass h > 0.
    constexpr float kPi = 3.14159265358979323846f;
    const float h2 = h * h;
    const float h3 = h * h * h;
    const float h6 = h3 * h3;
    const float h9 = h6 * h3;
    const float poly6 = 315.0f / (64.0f * kPi * h9);
    const float m = particles.particle_mass;

    // Poly6 scalar value at squared distance r2 -- the EXACT GPU Poly6FromR2
    // expression (poly6 * diff * diff * diff, clamped to 0 outside the support).
    auto Poly6FromR2 = [&](float r2) -> float {
        if (r2 >= h2) {
            return 0.0f;
        }
        const float diff = h2 - r2;  // (h^2 - r^2) >= 0 here.
        return poly6 * diff * diff * diff;
    };

    std::vector<float> rho(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const math::Vec3 pi = particles.positions[i];
        // Self term first (r2 = 0): a fixed leading addend, identical for all i --
        // grouped m * Poly6FromR2(0) exactly as the GPU ComputeDensityKernel.
        float r = m * Poly6FromR2(0.0f);
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const math::Vec3 d = pi - particles.positions[j];
            const float r2 = d.Dot(d);
            r += m * Poly6FromR2(r2);  // adds 0 outside h (matches the grid skip).
        }
        rho[i] = r;
    }
    return rho;
}

} // namespace nuka::runtime::fluid
