// ---------------------------------------------------------------------------
// nuka::runtime::fluid::PbfWorld implementation (v0.7 p10-A)
// ---------------------------------------------------------------------------
//
// GPU Position-Based Fluids substep (Macklin & Mueller 2013): predict -> build
// the uniform-grid neighbor list on the predicted positions (ONCE per step) ->
// N_pbf density-projection iterations (density, lambda, correct -- three separate
// parallel kernel launches) -> finalize.
//
// The three density kernels are genuinely PARALLEL (one thread per particle):
// each reads its neighbors in the grid's ASCENDING-by-index sorted order (a
// fixed-order reduction) and writes ONLY its own output. No intra-pass write
// conflict, NO float atomicAdd. The launch boundary between the lambda pass and
// the correction pass is the required barrier: apply_correction reads neighbors'
// lambda_j, so all lambdas must be finalized first. D1 strong determinism: same
// float instruction stream for every particle, fixed neighbor order, no atomics.
// (Same posture as src/runtime/soft/xpbd_world.cu; deliberately allowlist-free.)
//
// The density / lambda / correct kernels fold into THIS TU (the spec permits a
// single-file layout); the SPH kernels live in pbf_kernels.cuh.
// ---------------------------------------------------------------------------

#include "runtime/fluid/pbf_world.hpp"

#include "collision/particle_uniform_grid.hpp"
#include "math/cuda_vec_ops.cuh"
#include "phi/buffer_transfer.hpp"
#include "runtime/fluid/pbf_kernels.cuh"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace nuka::runtime::fluid {

namespace {

namespace mg = ::nuka::math::gpu;
using mg::Add;
using mg::Dot;
using mg::Sub;

namespace cg = ::nuka::collision::gpu;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// --- predict ----------------------------------------------------------------
// One thread per particle. Integrate gravity into velocity, then predict the
// position. The stored velocity IS updated here (M&M apply external forces to v
// before the predict); the final per-step velocity is rebuilt from the corrected
// predicted position in Finalize. positions[] is left untouched (it is the t0
// position the finalize differences against).
//     v_i      += dt * gravity
//     p_pred_i  = p_i + dt * v_i
__global__ void PredictKernel(uint32_t particle_count,
                              const math::Vec3* __restrict__ positions,
                              math::Vec3* __restrict__ velocities,
                              math::Vec3* __restrict__ predicted_positions,
                              math::Vec3 gravity,
                              float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    math::Vec3 v = velocities[i];
    v.x += dt * gravity.x;
    v.y += dt * gravity.y;
    v.z += dt * gravity.z;
    velocities[i] = v;
    const math::Vec3 p = positions[i];
    predicted_positions[i] =
        math::Vec3{p.x + dt * v.x, p.y + dt * v.y, p.z + dt * v.z};
}

// --- density (pass 1) -------------------------------------------------------
// One thread per particle. SPH density estimate via the Poly6 kernel. INCLUDES
// the self term (r2 == 0 -> Poly6(0) = poly6 * h^6) per SPH convention -- so a
// fully-surrounded interior particle reaches the calibrated rho0. The neighbor
// reduction walks the CSR slice in the grid's ascending-by-index order (fixed
// order = D1); the running sum starts from the self term, then adds neighbors.
//     rho_i = m * Poly6(0) + sum_{j in N(i)} m * Poly6(|p_i - p_j|^2)
__global__ void ComputeDensityKernel(uint32_t particle_count,
                                     const math::Vec3* __restrict__ predicted,
                                     float particle_mass,
                                     PbfKernelCoeffs coeffs,
                                     const uint32_t* __restrict__ neighbor_offsets,
                                     const uint32_t* __restrict__ neighbor_counts,
                                     const uint32_t* __restrict__ neighbor_indices,
                                     float* __restrict__ out_density) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 pi = predicted[i];
    // Self term first (r2 = 0): a fixed leading addend, identical for all threads.
    float rho = particle_mass * Poly6FromR2(0.0f, coeffs);
    const uint32_t base = neighbor_offsets[i];
    const uint32_t n = neighbor_counts[i];
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        const math::Vec3 r = Sub(pi, predicted[j]);
        const float r2 = Dot(r, r);
        rho += particle_mass * Poly6FromR2(r2, coeffs);
    }
    out_density[i] = rho;
}

// --- lambda (pass 2) --------------------------------------------------------
// One thread per particle. The PBF per-particle Lagrange multiplier (M&M eq.9-11):
//   C_i      = rho_i / rho0 - 1
//   grad_{p_k} C_i = (1/rho0) * SpikyGrad(p_i - p_k)         for a neighbor k
//   grad_{p_i} C_i = (1/rho0) * sum_{j in N(i)} SpikyGrad(p_i - p_j)
//   sum_k |grad_k C_i|^2 = |grad_i C_i|^2 + sum_j |grad_j C_i|^2
//   lambda_i = -C_i / (sum_k |grad_k C_i|^2 + epsilon)
// epsilon is the CFM relaxation (keeps lambda finite where the gradient sum -> 0,
// i.e. at the free surface). If clamp_to_overdensity, C_i is clamped to >= 0 so
// only compression is corrected (no spurious surface cohesion). Fixed neighbor
// order; the accumulator (grad_i and the |grad_j|^2 sum) is ascending = D1.
__global__ void ComputeLambdaKernel(uint32_t particle_count,
                                    const math::Vec3* __restrict__ predicted,
                                    PbfKernelCoeffs coeffs,
                                    float inv_rho0,
                                    float rest_density,
                                    float cfm_epsilon,
                                    bool clamp_to_overdensity,
                                    const float* __restrict__ density,
                                    const uint32_t* __restrict__ neighbor_offsets,
                                    const uint32_t* __restrict__ neighbor_counts,
                                    const uint32_t* __restrict__ neighbor_indices,
                                    float* __restrict__ out_lambda) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    float c_i = density[i] / rest_density - 1.0f;
    if (clamp_to_overdensity && c_i < 0.0f) {
        out_lambda[i] = 0.0f;  // under-dense (free surface): do not pull inward.
        return;
    }

    const math::Vec3 pi = predicted[i];
    const uint32_t base = neighbor_offsets[i];
    const uint32_t n = neighbor_counts[i];

    // grad_i C_i = inv_rho0 * sum_j SpikyGrad(p_i - p_j); accumulate the gradient
    // wrt p_i AND the sum of squared gradients wrt each neighbor in one pass.
    math::Vec3 grad_i{0.0f, 0.0f, 0.0f};
    float sum_grad_sq = 0.0f;
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        const math::Vec3 r = Sub(pi, predicted[j]);
        const float dist = sqrtf(Dot(r, r));
        const math::Vec3 sg = SpikyGradient(r, dist, coeffs);
        // grad wrt neighbor j is -inv_rho0 * sg; its squared norm is the same.
        const math::Vec3 grad_j{sg.x * inv_rho0, sg.y * inv_rho0, sg.z * inv_rho0};
        sum_grad_sq += Dot(grad_j, grad_j);
        grad_i.x += grad_j.x;
        grad_i.y += grad_j.y;
        grad_i.z += grad_j.z;
    }
    sum_grad_sq += Dot(grad_i, grad_i);  // the |grad_i C_i|^2 term.

    out_lambda[i] = -c_i / (sum_grad_sq + cfm_epsilon);
}

// --- correct, pass 3a: compute delta ----------------------------------------
// One thread per particle. The PBF position correction (M&M eq.12) as a JACOBI
// update -- it reads neighbors' predicted positions + lambda but writes ONLY its
// own delta[i], leaving `predicted` untouched. A FUSED kernel that did
// `predicted[i] += dp` in place would have thread k read predicted[i] (i a
// neighbor of k) while thread i writes it in the SAME launch -- a global
// read-write race whose result depends on warp scheduling (a D1 violation). By
// computing every dp_i from the SAME pre-correction predicted positions and
// applying them in a separate own-index pass, the update is race-free, D1, AND
// the textbook Jacobi PBF semantics.
//   dp_i = (1/rho0) * sum_{j in N(i)} (lambda_i + lambda_j) * SpikyGrad(p_i - p_j)
__global__ void ComputeCorrectionKernel(
    uint32_t particle_count,
    const math::Vec3* __restrict__ predicted,
    PbfKernelCoeffs coeffs,
    float inv_rho0,
    const float* __restrict__ lambda,
    const uint32_t* __restrict__ neighbor_offsets,
    const uint32_t* __restrict__ neighbor_counts,
    const uint32_t* __restrict__ neighbor_indices,
    math::Vec3* __restrict__ out_delta) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 pi = predicted[i];
    const float lam_i = lambda[i];
    const uint32_t base = neighbor_offsets[i];
    const uint32_t n = neighbor_counts[i];

    math::Vec3 dp{0.0f, 0.0f, 0.0f};
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        const math::Vec3 r = Sub(pi, predicted[j]);
        const float dist = sqrtf(Dot(r, r));
        const math::Vec3 sg = SpikyGradient(r, dist, coeffs);
        const float scale = (lam_i + lambda[j]) * inv_rho0;
        dp.x += sg.x * scale;
        dp.y += sg.y * scale;
        dp.z += sg.z * scale;
    }
    out_delta[i] = dp;
}

// --- correct, pass 3b: apply delta ------------------------------------------
// One thread per particle. p_pred_i += dp_i, then the optional floor clamp. Pure
// own-index write -> no race, no atomics. Separated from the compute pass so the
// Jacobi update sees the consistent pre-correction positions (see above).
__global__ void ApplyCorrectionKernel(uint32_t particle_count,
                                      math::Vec3* __restrict__ predicted,
                                      const math::Vec3* __restrict__ delta,
                                      bool boundary_enabled,
                                      float floor_y) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 pi = predicted[i];
    const math::Vec3 dp = delta[i];
    math::Vec3 out{pi.x + dp.x, pi.y + dp.y, pi.z + dp.z};
    if (boundary_enabled && out.y < floor_y) {
        out.y = floor_y;  // clamp up to the floor; velocity self-corrects later.
    }
    predicted[i] = out;
}

// --- finalize ---------------------------------------------------------------
// One thread per particle. Rebuild velocity from the corrected predicted position
// and commit it as the new position.
//     v_i = (p_pred_i - p_i) / dt ;  p_i = p_pred_i
// (XSPH viscosity / vorticity confinement are p10-B; finalize is the hook point.)
__global__ void FinalizeKernel(uint32_t particle_count,
                               math::Vec3* __restrict__ positions,
                               const math::Vec3* __restrict__ predicted,
                               math::Vec3* __restrict__ velocities,
                               float inv_dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 p0 = positions[i];
    const math::Vec3 pp = predicted[i];
    velocities[i] = math::Vec3{(pp.x - p0.x) * inv_dt, (pp.y - p0.y) * inv_dt,
                               (pp.z - p0.z) * inv_dt};
    positions[i] = pp;
}

constexpr uint32_t kBlockSize = 128u;

// Build the per-step neighbor grid over a device position array. cell_size == h
// (so the grid's cell_size >= query_radius precondition holds with query_radius
// == h). The world AABB is derived from the device positions on the host (a small
// download); for the small p10-A test scenes this is cheap and keeps the grid
// tightly sized. The grid build is a known perf deferral (named consumer = the
// 1M-particle perf pass); src/runtime/fluid/** is not in the lint hot_path scope.
cg::ParticleGridResult BuildStepGrid(const phi::DeviceContext& context,
                                     const math::Vec3* device_positions,
                                     uint32_t particle_count,
                                     float h) {
    std::vector<math::Vec3> host(particle_count);
    CheckCuda(cudaMemcpy(host.data(), device_positions,
                         static_cast<std::size_t>(particle_count) * sizeof(math::Vec3),
                         cudaMemcpyDeviceToHost),
              "download predicted positions for grid AABB");
    math::Vec3 lo = host[0];
    math::Vec3 hi = host[0];
    for (const math::Vec3& p : host) {
        lo.x = p.x < lo.x ? p.x : lo.x;
        lo.y = p.y < lo.y ? p.y : lo.y;
        lo.z = p.z < lo.z ? p.z : lo.z;
        hi.x = p.x > hi.x ? p.x : hi.x;
        hi.y = p.y > hi.y ? p.y : hi.y;
        hi.z = p.z > hi.z ? p.z : hi.z;
    }
    const cg::ParticleGridConfig cfg = cg::MakeParticleGridConfig(lo, hi, h);
    return cg::BuildParticleUniformGrid(context, device_positions, particle_count,
                                        cfg, h);
}

} // namespace

// =============================================================================
// PbfWorld
// =============================================================================

PbfWorld::PbfWorld(uint32_t particle_count,
                   float particle_mass,
                   phi::Buffer positions,
                   phi::Buffer velocities,
                   phi::Buffer predicted_positions,
                   phi::Buffer position_deltas,
                   phi::Buffer densities,
                   phi::Buffer lambdas)
    : particle_count_(particle_count)
    , particle_mass_(particle_mass)
    , positions_(std::move(positions))
    , velocities_(std::move(velocities))
    , predicted_positions_(std::move(predicted_positions))
    , position_deltas_(std::move(position_deltas))
    , densities_(std::move(densities))
    , lambdas_(std::move(lambdas)) {}

std::size_t PbfWorld::DeviceBytes() const {
    return positions_.Size() + velocities_.Size() + predicted_positions_.Size() +
           position_deltas_.Size() + densities_.Size() + lambdas_.Size();
}

bool PbfWorld::HasUploadedState() const {
    if (particle_count_ == 0u) {
        return true;
    }
    return positions_.Data() != nullptr && velocities_.Data() != nullptr &&
           predicted_positions_.Data() != nullptr &&
           position_deltas_.Data() != nullptr &&
           densities_.Data() != nullptr && lambdas_.Data() != nullptr;
}

PbfWorldState PbfWorld::DownloadState() const {
    PbfWorldState state;
    state.positions = phi::DownloadVector<math::Vec3>(positions_, particle_count_);
    state.velocities = phi::DownloadVector<math::Vec3>(velocities_, particle_count_);
    state.densities = phi::DownloadVector<float>(densities_, particle_count_);
    return state;
}

math::Vec3* PbfWorld::DevicePositions() {
    return static_cast<math::Vec3*>(positions_.Data());
}
const math::Vec3* PbfWorld::DevicePositions() const {
    return static_cast<const math::Vec3*>(positions_.Data());
}
math::Vec3* PbfWorld::DeviceVelocities() {
    return static_cast<math::Vec3*>(velocities_.Data());
}
const math::Vec3* PbfWorld::DeviceVelocities() const {
    return static_cast<const math::Vec3*>(velocities_.Data());
}
math::Vec3* PbfWorld::DevicePredictedPositions() {
    return static_cast<math::Vec3*>(predicted_positions_.Data());
}
const math::Vec3* PbfWorld::DevicePredictedPositions() const {
    return static_cast<const math::Vec3*>(predicted_positions_.Data());
}
math::Vec3* PbfWorld::DevicePositionDeltas() {
    return static_cast<math::Vec3*>(position_deltas_.Data());
}
const math::Vec3* PbfWorld::DevicePositionDeltas() const {
    return static_cast<const math::Vec3*>(position_deltas_.Data());
}
float* PbfWorld::DeviceDensities() {
    return static_cast<float*>(densities_.Data());
}
const float* PbfWorld::DeviceDensities() const {
    return static_cast<const float*>(densities_.Data());
}
float* PbfWorld::DeviceLambdas() {
    return static_cast<float*>(lambdas_.Data());
}
const float* PbfWorld::DeviceLambdas() const {
    return static_cast<const float*>(lambdas_.Data());
}

// =============================================================================
// Upload
// =============================================================================

PbfWorld UploadPbfWorld(const phi::DeviceContext& context,
                        const PbfParticleSet& particles) {
    phi::ScopedDeviceGuard guard(context.device_id);

    const auto count = particles.positions.size();
    if (particles.velocities.size() != count) {
        throw std::runtime_error(
            "UploadPbfWorld requires positions and velocities to have matching lengths");
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("UploadPbfWorld particle count exceeds uint32_t range");
    }
    const uint32_t particle_count = static_cast<uint32_t>(count);

    std::vector<float> zeros(count, 0.0f);
    std::vector<math::Vec3> zero_vecs(count, math::Vec3{0.0f, 0.0f, 0.0f});
    return PbfWorld(particle_count,
                    particles.particle_mass,
                    phi::UploadVector(particles.positions),
                    phi::UploadVector(particles.velocities),
                    phi::UploadVector(particles.positions),  // predicted seeded = p
                    phi::UploadVector(zero_vecs),             // position deltas
                    phi::UploadVector(zeros),                 // densities
                    phi::UploadVector(zeros));                // lambdas
}

PbfWorld UploadPbfWorld(const PbfParticleSet& particles) {
    auto context = phi::MakeDefaultDeviceContext();
    return UploadPbfWorld(context, particles);
}

// =============================================================================
// Step: predict -> grid -> [density, lambda, correct] * N -> finalize
// =============================================================================

PbfStepReport StepPbfWorld(const phi::DeviceContext& context,
                           PbfWorld& world,
                           const PbfParams& params,
                           const PbfStepOptions& options) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    PbfStepReport report;
    report.particle_count = world.ParticleCount();

    if (options.dt <= 0.0f || options.step_count == 0u ||
        world.ParticleCount() == 0u) {
        return report;
    }
    if (!world.HasUploadedState()) {
        throw std::runtime_error(
            "StepPbfWorld requires UploadPbfWorld before stepping a non-empty world");
    }
    if (params.support_radius_h <= 0.0f || params.rest_density_rho0 <= 0.0f) {
        throw std::runtime_error(
            "StepPbfWorld requires support_radius_h > 0 and rest_density_rho0 > 0");
    }

    const uint32_t solver_iterations =
        params.solver_iterations == 0u ? 1u : params.solver_iterations;
    report.solver_iterations = solver_iterations;

    const PbfKernelCoeffs coeffs = MakePbfKernelCoeffs(params.support_radius_h);
    const float inv_rho0 = 1.0f / params.rest_density_rho0;
    const float inv_dt = 1.0f / options.dt;
    const uint32_t particle_count = world.ParticleCount();
    const uint32_t particle_blocks =
        (particle_count + kBlockSize - 1u) / kBlockSize;

    for (uint32_t step = 0u; step < options.step_count; ++step) {
        // --- predict --------------------------------------------------------
        PredictKernel<<<particle_blocks, kBlockSize, 0, stream>>>(
            particle_count,
            world.DevicePositions(),
            world.DeviceVelocities(),
            world.DevicePredictedPositions(),
            options.gravity,
            options.dt);
        CheckCuda(cudaGetLastError(), "PredictKernel launch");
        ++report.kernel_launch_count;

        // --- build the neighbor grid ON the predicted positions (once) ------
        // M&M 2013: one neighbor list per step, reused across the iterations.
        context.stream.Synchronize();  // predict must complete before the AABB read.
        cg::ParticleGridResult grid =
            BuildStepGrid(context, world.DevicePredictedPositions(),
                          particle_count, params.support_radius_h);
        ++report.grid_build_count;
        if (grid.TruncatedParticleCount() > report.max_truncated_particle_count) {
            report.max_truncated_particle_count = grid.TruncatedParticleCount();
        }

        const uint32_t* d_offsets = grid.DeviceNeighborOffsets();
        const uint32_t* d_counts = grid.DeviceNeighborCounts();
        const uint32_t* d_indices = grid.DeviceNeighborIndices();

        // --- N density-projection iterations --------------------------------
        for (uint32_t iter = 0u; iter < solver_iterations; ++iter) {
            ComputeDensityKernel<<<particle_blocks, kBlockSize, 0, stream>>>(
                particle_count,
                world.DevicePredictedPositions(),
                world.ParticleMass(),
                coeffs,
                d_offsets, d_counts, d_indices,
                world.DeviceDensities());
            CheckCuda(cudaGetLastError(), "ComputeDensityKernel launch");
            ++report.kernel_launch_count;

            ComputeLambdaKernel<<<particle_blocks, kBlockSize, 0, stream>>>(
                particle_count,
                world.DevicePredictedPositions(),
                coeffs,
                inv_rho0,
                params.rest_density_rho0,
                params.cfm_epsilon,
                params.clamp_to_overdensity,
                world.DeviceDensities(),
                d_offsets, d_counts, d_indices,
                world.DeviceLambdas());
            CheckCuda(cudaGetLastError(), "ComputeLambdaKernel launch");
            ++report.kernel_launch_count;

            // JACOBI correction: compute every dp_i from the SAME pre-correction
            // predicted positions (pass 3a, read-only on predicted), THEN apply
            // them own-index (pass 3b). The launch boundary between 3a and 3b is
            // the barrier that makes this race-free + D1 (a fused in-place update
            // would race predicted[] across threads). See the kernel comments.
            ComputeCorrectionKernel<<<particle_blocks, kBlockSize, 0, stream>>>(
                particle_count,
                world.DevicePredictedPositions(),
                coeffs,
                inv_rho0,
                world.DeviceLambdas(),
                d_offsets, d_counts, d_indices,
                world.DevicePositionDeltas());
            CheckCuda(cudaGetLastError(), "ComputeCorrectionKernel launch");
            ++report.kernel_launch_count;

            ApplyCorrectionKernel<<<particle_blocks, kBlockSize, 0, stream>>>(
                particle_count,
                world.DevicePredictedPositions(),
                world.DevicePositionDeltas(),
                options.boundary.enabled,
                options.boundary.floor_y);
            CheckCuda(cudaGetLastError(), "ApplyCorrectionKernel launch");
            ++report.kernel_launch_count;
        }

        // --- finalize -------------------------------------------------------
        FinalizeKernel<<<particle_blocks, kBlockSize, 0, stream>>>(
            particle_count,
            world.DevicePositions(),
            world.DevicePredictedPositions(),
            world.DeviceVelocities(),
            inv_dt);
        CheckCuda(cudaGetLastError(), "FinalizeKernel launch");
        ++report.kernel_launch_count;

        context.stream.Synchronize();  // grid frees at scope end; sync first.
    }

    report.simulated_step_count = options.step_count;
    return report;
}

PbfStepReport StepPbfWorld(PbfWorld& world,
                           const PbfParams& params,
                           const PbfStepOptions& options) {
    auto context = phi::MakeDefaultDeviceContext();
    return StepPbfWorld(context, world, params, options);
}

// =============================================================================
// Density-only pass (rho0 calibration helper)
// =============================================================================

std::vector<float> ComputePbfDensities(const phi::DeviceContext& context,
                                        const PbfWorld& world,
                                        const PbfParams& params) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    const uint32_t particle_count = world.ParticleCount();
    if (particle_count == 0u) {
        return {};
    }
    if (!world.HasUploadedState()) {
        throw std::runtime_error(
            "ComputePbfDensities requires UploadPbfWorld before calling");
    }
    if (params.support_radius_h <= 0.0f) {
        throw std::runtime_error("ComputePbfDensities requires support_radius_h > 0");
    }

    const PbfKernelCoeffs coeffs = MakePbfKernelCoeffs(params.support_radius_h);
    const uint32_t particle_blocks =
        (particle_count + kBlockSize - 1u) / kBlockSize;

    // Build the grid on the CURRENT positions (the rest configuration).
    cg::ParticleGridResult grid =
        BuildStepGrid(context, world.DevicePositions(), particle_count,
                      params.support_radius_h);

    phi::Buffer d_density(static_cast<std::size_t>(particle_count) * sizeof(float),
                          phi::MemoryKind::Device);
    ComputeDensityKernel<<<particle_blocks, kBlockSize, 0, stream>>>(
        particle_count,
        world.DevicePositions(),
        world.ParticleMass(),
        coeffs,
        grid.DeviceNeighborOffsets(),
        grid.DeviceNeighborCounts(),
        grid.DeviceNeighborIndices(),
        static_cast<float*>(d_density.Data()));
    CheckCuda(cudaGetLastError(), "ComputeDensityKernel (density-only) launch");
    context.stream.Synchronize();

    return phi::DownloadVector<float>(d_density, particle_count);
}

std::vector<float> ComputePbfDensities(const PbfWorld& world,
                                        const PbfParams& params) {
    auto context = phi::MakeDefaultDeviceContext();
    return ComputePbfDensities(context, world, params);
}

} // namespace nuka::runtime::fluid
