#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::fluid::PbfWorld -- GPU-resident Position-Based Fluids world
// (v0.7 p10-A; Macklin & Mueller 2013, "Position Based Fluids").
// ---------------------------------------------------------------------------
//
// p10-A: the CORE forward density-projection fluid -- the foundational vertical
// slice of PBF. This module owns the GPU particle state (position, velocity,
// predicted position, density, lambda; uniform per-particle mass) and runs the
// PBF substep ENTIRELY on the GPU (master plan SS5.6 GPU-only). The loop is the
// standard M&M 2013 scheme:
//
//     predict :  v_i += dt*gravity ; p_pred_i = p_i + dt*v_i          (per particle)
//     (build the uniform-grid neighbor list ONCE on the predicted positions)
//     iterate N_pbf times:
//        density   : rho_i  = sum_j m * Poly6(|p_pred_i - p_pred_j|, h)  (incl self)
//        lambda    : C_i = rho_i/rho0 - 1 ;
//                    lambda_i = -C_i / (sum_k |grad_k C_i|^2 + epsilon)
//        correct   : dp_i = (1/rho0) sum_j (lambda_i + lambda_j) * SpikyGrad ;
//                    p_pred_i += dp_i
//     finalize:  v_i = (p_pred_i - p_i)/dt ;  p_i = p_pred_i           (per particle)
//
// FORWARD-ONLY + INTERNAL (master plan SS3 Round 7): the PBF density constraint
// does NOT emit into the universal row scheduler, is NOT a codegen row class --
// there is NO adjoint, NO V3 FD, NO .yaml, NO codegen. It is a self-contained GPU
// subsystem (unlike p09 XPBD, which IS row classes).
//
// D1 STRONG DETERMINISM: every device buffer is allocated once at upload (the
// per-step neighbor-grid build is the one exception -- a known perf deferral with
// a named consumer = the 1M-particle perf pass; src/runtime/fluid/** is NOT in the
// lint hot_path scope). The three density kernels are genuinely PARALLEL: each
// particle reads its neighbors (in the grid's ascending-by-index sorted order =
// fixed-order reduction) and writes ONLY its own density/lambda/position output --
// no intra-pass write conflict, NO float atomicAdd. Two runs of Step() with
// identical inputs produce byte-identical position + velocity buffers.
//
// DEFERRED to p10-B (named consumers): XSPH viscosity, surface tension, the fluid
// COOKER, the V2 particle-count invariant harness, and the s_corr tensile term
// (a documented hook only -- see pbf_world.cu). The Flex ball-into-water oracle
// is deferred to p15 (NVIDIA Flex not reproducible here); p10-A ships the analytic
// invariants instead (rest density, relaxation, stable pool, density-drift volume
// conservation, D1).
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nuka::runtime::fluid {

// Host-side description of a PBF particle set. Mass is UNIFORM across the fluid
// (p10-A); per-particle mass is a p10-B/future extension.
struct PbfParticleSet {
    std::vector<math::Vec3> positions;
    std::vector<math::Vec3> velocities;
    float particle_mass = 1.0f;  // m (same for every particle).
};

// PBF simulation parameters. rho0 is the rest density the incompressibility
// constraint pulls toward; it is typically CALIBRATED numerically from this
// engine's own Poly6 kernel over a rest lattice (so a kernel-constant slip is
// absorbed). See pbf_world.cu and the tests for the calibration recipe.
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
};

// Floor / boundary plane (p10-A keeps a single horizontal floor; full SDF / box
// boundaries are a later coupling concern). A predicted position below floor_y is
// clamped UP to floor_y after each correction pass (velocity self-corrects in the
// finalize). Set enabled=false for an unbounded (free-fall) scene.
struct PbfBoundary {
    bool enabled = false;
    float floor_y = 0.0f;
};

struct PbfStepOptions {
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    float dt = 1.0f / 60.0f;
    uint32_t step_count = 1u;
    PbfBoundary boundary{};
};

// Host snapshot downloaded from the device (state read-back for tests / oracles).
struct PbfWorldState {
    std::vector<math::Vec3> positions;
    std::vector<math::Vec3> velocities;
    std::vector<float> densities;  // last-iteration rho_i (diagnostics: C_i, drift).
};

struct PbfStepReport {
    uint32_t particle_count = 0u;
    uint32_t simulated_step_count = 0u;
    uint32_t solver_iterations = 0u;
    uint32_t kernel_launch_count = 0u;     // PBF predict/density/lambda/correct/finalize.
    uint32_t grid_build_count = 0u;        // neighbor-grid builds (1 per step).
    uint32_t max_truncated_particle_count = 0u;  // worst-step neighbor-cap
                                                 // truncations (should be 0 for a
                                                 // well-sized h; surfaced so a
                                                 // mis-sized scene is caught).
};

// GPU-resident PBF world. Particle-state buffers are allocated once on upload; the
// per-step neighbor grid is (re)built each step (perf deferral, see header note).
class PbfWorld {
public:
    PbfWorld() = default;
    PbfWorld(uint32_t particle_count,
             float particle_mass,
             phi::Buffer positions,
             phi::Buffer velocities,
             phi::Buffer predicted_positions,
             phi::Buffer position_deltas,
             phi::Buffer densities,
             phi::Buffer lambdas);

    PbfWorld(const PbfWorld&) = delete;
    PbfWorld& operator=(const PbfWorld&) = delete;
    PbfWorld(PbfWorld&&) noexcept = default;
    PbfWorld& operator=(PbfWorld&&) noexcept = default;

    uint32_t ParticleCount() const { return particle_count_; }
    float ParticleMass() const { return particle_mass_; }
    std::size_t DeviceBytes() const;
    bool HasUploadedState() const;
    PbfWorldState DownloadState() const;

    math::Vec3* DevicePositions();
    const math::Vec3* DevicePositions() const;
    math::Vec3* DeviceVelocities();
    const math::Vec3* DeviceVelocities() const;
    math::Vec3* DevicePredictedPositions();
    const math::Vec3* DevicePredictedPositions() const;
    math::Vec3* DevicePositionDeltas();
    const math::Vec3* DevicePositionDeltas() const;
    float* DeviceDensities();
    const float* DeviceDensities() const;
    float* DeviceLambdas();
    const float* DeviceLambdas() const;

private:
    uint32_t particle_count_ = 0u;
    float particle_mass_ = 1.0f;
    phi::Buffer positions_;
    phi::Buffer velocities_;
    phi::Buffer predicted_positions_;
    // Per-particle accumulated PBF position correction dp_i for the CURRENT
    // iteration. PBF correction is JACOBI: every dp_i is computed from the SAME
    // pre-correction predicted positions, then applied in a separate own-index
    // pass. This double-buffering is REQUIRED for D1 (and correctness): a single
    // fused correction kernel would have thread k read predicted[i] while thread
    // i writes it -- a global read-write race whose result depends on warp
    // scheduling. Allocated once at upload.
    phi::Buffer position_deltas_;
    phi::Buffer densities_;
    phi::Buffer lambdas_;
};

PbfWorld UploadPbfWorld(const phi::DeviceContext& context,
                        const PbfParticleSet& particles);
PbfWorld UploadPbfWorld(const PbfParticleSet& particles);

// Run `options.step_count` PBF substeps (predict -> grid build -> N density
// iterations -> finalize) on the GPU. `params` are the fixed fluid parameters
// (h, rho0, iterations, ...). Empty world (ParticleCount()==0) is a no-op.
PbfStepReport StepPbfWorld(const phi::DeviceContext& context,
                           PbfWorld& world,
                           const PbfParams& params,
                           const PbfStepOptions& options = {});
PbfStepReport StepPbfWorld(PbfWorld& world,
                           const PbfParams& params,
                           const PbfStepOptions& options = {});

// Compute the rest density at every particle by running ONE density pass over the
// current positions (no integration / no constraint projection). Used to
// numerically CALIBRATE rho0 from a rest lattice: build the lattice, call this,
// take a deep-interior particle's density. Returns the per-particle rho_i.
std::vector<float> ComputePbfDensities(const phi::DeviceContext& context,
                                       const PbfWorld& world,
                                       const PbfParams& params);
std::vector<float> ComputePbfDensities(const PbfWorld& world,
                                       const PbfParams& params);

} // namespace nuka::runtime::fluid
