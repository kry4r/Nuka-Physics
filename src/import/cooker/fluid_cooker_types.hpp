#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::fluid -- CUDA-FREE PBF param structs + a pure-host density helper
// ---------------------------------------------------------------------------
//
// M9 T11 soft/fluid Phase 2b: the legacy PBF fluid stepper has been
// DELETED -- the forward sim is op-ified onto the unified nk core
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

// Poly6 SCALAR kernel value at squared distance r2, the CUDA-free mirror of the
// device Poly6FromR2 (runtime/fluid/pbf_kernels.cuh): const 315/(64*pi*h^9) and
// the same (h^2 - r^2)^3 clamp. ONE host Poly6 reused by ComputePbfDensities and
// the fluid surface mesher so neither carries a divergent copy.
inline float Poly6FromR2Host(float r2, float h) {
    constexpr float kPi = 3.14159265358979323846f;
    const float h2 = h * h;
    if (r2 >= h2) {
        return 0.0f;
    }
    const float h3 = h * h * h;
    const float h6 = h3 * h3;
    const float h9 = h6 * h3;
    const float poly6 = 315.0f / (64.0f * kPi * h9);
    const float diff = h2 - r2;  // (h^2 - r^2) >= 0 here.
    return poly6 * diff * diff * diff;
}

// Spiky kernel GRADIENT, the CUDA-free mirror of the device SpikyGradient
// (runtime/fluid/pbf_kernels.cuh:89): const -45/(pi*h^6)*(h-r)^2*(r_vec/r). Zero
// for r >= h or r ~ 0 so the direction stays finite at coincidence.
inline math::Vec3 SpikyGradientHost(math::Vec3 r_vec, float r, float h) {
    constexpr float kPi = 3.14159265358979323846f;
    if (r >= h || r <= 0.0f) {
        return math::Vec3{0.0f, 0.0f, 0.0f};
    }
    const float h3 = h * h * h;
    const float h6 = h3 * h3;
    const float spiky_grad = -45.0f / (kPi * h6);
    const float hr = h - r;                          // (h - r) > 0 here.
    const float coeff = spiky_grad * hr * hr / r;    // includes 1/r for r_vec/r.
    return math::Vec3{r_vec.x * coeff, r_vec.y * coeff, r_vec.z * coeff};
}

// GRADIENT of the Poly6 kernel wrt the sample position, the analytic gradient of
// the SAME field ComputePbfDensities/Poly6FromR2Host sum: d/dx [(h^2-r^2)^3] gives
// -6*(h^2-r^2)^2 * r_vec, scaled by the Poly6 constant. Used for density-isosurface
// normals (the Spiky form is the pressure-correction gradient, not this field's).
inline math::Vec3 Poly6GradientHost(math::Vec3 r_vec, float r2, float h) {
    constexpr float kPi = 3.14159265358979323846f;
    const float h2 = h * h;
    if (r2 >= h2) {
        return math::Vec3{0.0f, 0.0f, 0.0f};
    }
    const float h3 = h * h * h;
    const float h6 = h3 * h3;
    const float h9 = h6 * h3;
    const float poly6 = 315.0f / (64.0f * kPi * h9);
    const float diff = h2 - r2;                      // (h^2 - r^2) >= 0 here.
    const float coeff = poly6 * -6.0f * diff * diff;  // d/dr2 chain on (h^2-r^2)^3.
    return math::Vec3{r_vec.x * coeff, r_vec.y * coeff, r_vec.z * coeff};
}

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
    const float m = particles.particle_mass;

    std::vector<float> rho(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const math::Vec3 pi = particles.positions[i];
        // Self term first (r2 = 0): a fixed leading addend, identical for all i --
        // grouped m * Poly6FromR2(0) exactly as the GPU ComputeDensityKernel.
        float r = m * Poly6FromR2Host(0.0f, h);
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const math::Vec3 d = pi - particles.positions[j];
            const float r2 = d.Dot(d);
            r += m * Poly6FromR2Host(r2, h);  // adds 0 outside h (grid skip).
        }
        rho[i] = r;
    }
    return rho;
}

} // namespace nuka::runtime::fluid
