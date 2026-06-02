// ---------------------------------------------------------------------------
// nuka::runtime::soft::XpbdWorld implementation (v0.7 p09-A)
// ---------------------------------------------------------------------------
//
// GPU XPBD distance-constraint integration loop: predict -> solve -> correct.
// All device buffers are allocated once on upload; Step launches only kernels
// over them (no hot-path cudaMalloc). The constraint solve is a single-thread
// FIXED-ORDER Gauss-Seidel sweep -- D1 strong determinism, no float atomics, no
// thread-order dependence. Parallel/colored GS is deferred to p09-B.
// ---------------------------------------------------------------------------

#include "runtime/soft/xpbd_world.hpp"

#include "math/cuda_vec_ops.cuh"
#include "phi/buffer_transfer.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace nuka::runtime::soft {

namespace {

namespace mg = ::nuka::math::gpu;
using mg::Add;
using mg::Dot;
using mg::MakeVec3;
using mg::Scale;
using mg::Sub;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// --- predict ------------------------------------------------------------------
// One thread per particle. Symplectic position predict that folds gravity into
// the position update WITHOUT mutating the stored velocity (velocity is rebuilt
// from the corrected position in CorrectVelocities). prev_positions snapshots p
// BEFORE the predict so CorrectVelocities can recover the per-step velocity.
//     prev = p
//     p   += v*dt + gravity*dt^2          (pinned particles -- w==0 -- stay put)
__global__ void PredictPositionsKernel(uint32_t particle_count,
                                       math::Vec3* __restrict__ positions,
                                       math::Vec3* __restrict__ prev_positions,
                                       const math::Vec3* __restrict__ velocities,
                                       const float* __restrict__ inv_masses,
                                       math::Vec3 gravity,
                                       float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 p = positions[i];
    prev_positions[i] = p;
    if (inv_masses[i] <= 0.0f) {
        return;  // pinned / kinematic particle: position held fixed.
    }
    const math::Vec3 step =
        Add(Scale(velocities[i], dt), Scale(gravity, dt * dt));
    positions[i] = Add(p, step);
}

// --- solve --------------------------------------------------------------------
// SINGLE-THREAD fixed-order Gauss-Seidel sweep over all distance constraints.
// The single thread visits constraints in ascending index order and applies each
// projection in place, so there are no inter-thread races and no float atomics --
// the read-modify-write into positions is strictly sequential (D1). The XPBD
// lambda is reset to 0 at the START of the step (iteration 0) per Macklin 2016.
//
//   C   = |p_a - p_b| - rest_length
//   n   = (p_a - p_b)/|p_a - p_b|
//   a~  = compliance_alpha / dt^2
//   dl  = (-C - a~*lambda) / (w_a + w_b + a~)
//   p_a += w_a*n*dl ; p_b -= w_b*n*dl ; lambda += dl
__global__ void SolveDistanceConstraintsKernel(
    uint32_t distance_constraint_count,
    uint32_t solver_iterations,
    math::Vec3* __restrict__ positions,
    const float* __restrict__ inv_masses,
    const uint32_t* __restrict__ particle_a,
    const uint32_t* __restrict__ particle_b,
    const float* __restrict__ rest_length,
    const float* __restrict__ compliance_alpha,
    float* __restrict__ lambda,
    float dt) {
    // One thread total drives the entire serial sweep (fixed order == D1).
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }

    const float inv_dt2 = 1.0f / (dt * dt);

    for (uint32_t c = 0u; c < distance_constraint_count; ++c) {
        lambda[c] = 0.0f;  // XPBD lambda reset at step start.
    }

    for (uint32_t iter = 0u; iter < solver_iterations; ++iter) {
        for (uint32_t c = 0u; c < distance_constraint_count; ++c) {
            const uint32_t ia = particle_a[c];
            const uint32_t ib = particle_b[c];
            const float wa = inv_masses[ia];
            const float wb = inv_masses[ib];
            const float w_sum = wa + wb;
            if (w_sum <= 0.0f) {
                continue;  // both endpoints pinned: nothing to project.
            }

            const math::Vec3 pa = positions[ia];
            const math::Vec3 pb = positions[ib];
            const math::Vec3 r = Sub(pa, pb);
            const float dist = sqrtf(Dot(r, r));
            if (dist <= 0.0f) {
                continue;  // coincident endpoints: gradient direction undefined.
            }

            const math::Vec3 n = Scale(r, 1.0f / dist);  // grad C wrt p_a (|n|=1)
            const float constraint = dist - rest_length[c];
            const float alpha_tilde = compliance_alpha[c] * inv_dt2;
            const float lam = lambda[c];
            const float delta_lambda =
                (-constraint - alpha_tilde * lam) / (w_sum + alpha_tilde);

            // p_a += w_a*n*dl ; p_b -= w_b*n*dl  (sequential RMW -- no atomics).
            positions[ia] = Add(pa, Scale(n, wa * delta_lambda));
            positions[ib] = Sub(pb, Scale(n, wb * delta_lambda));
            lambda[c] = lam + delta_lambda;
        }
    }
}

// --- correct ------------------------------------------------------------------
// One thread per particle. Rebuilds the velocity from the corrected position.
//     v = (p - prev) / dt        (pinned particles keep their stored velocity)
__global__ void CorrectVelocitiesKernel(uint32_t particle_count,
                                        const math::Vec3* __restrict__ positions,
                                        const math::Vec3* __restrict__ prev_positions,
                                        math::Vec3* __restrict__ velocities,
                                        const float* __restrict__ inv_masses,
                                        float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    if (inv_masses[i] <= 0.0f) {
        return;  // pinned particle: velocity untouched.
    }
    const math::Vec3 delta = Sub(positions[i], prev_positions[i]);
    velocities[i] = Scale(delta, 1.0f / dt);
}

} // namespace

// =============================================================================
// XpbdWorld
// =============================================================================

XpbdWorld::XpbdWorld(uint32_t particle_count,
                     uint32_t distance_constraint_count,
                     phi::Buffer positions,
                     phi::Buffer prev_positions,
                     phi::Buffer velocities,
                     phi::Buffer inv_masses,
                     phi::Buffer distance_particle_a,
                     phi::Buffer distance_particle_b,
                     phi::Buffer distance_rest_length,
                     phi::Buffer distance_compliance_alpha,
                     phi::Buffer distance_lambda)
    : particle_count_(particle_count)
    , distance_constraint_count_(distance_constraint_count)
    , positions_(std::move(positions))
    , prev_positions_(std::move(prev_positions))
    , velocities_(std::move(velocities))
    , inv_masses_(std::move(inv_masses))
    , distance_particle_a_(std::move(distance_particle_a))
    , distance_particle_b_(std::move(distance_particle_b))
    , distance_rest_length_(std::move(distance_rest_length))
    , distance_compliance_alpha_(std::move(distance_compliance_alpha))
    , distance_lambda_(std::move(distance_lambda)) {}

std::size_t XpbdWorld::DeviceBytes() const {
    return positions_.Size() + prev_positions_.Size() + velocities_.Size() +
           inv_masses_.Size() + distance_particle_a_.Size() +
           distance_particle_b_.Size() + distance_rest_length_.Size() +
           distance_compliance_alpha_.Size() + distance_lambda_.Size();
}

bool XpbdWorld::HasUploadedState() const {
    if (particle_count_ == 0u) {
        return true;
    }
    const bool particles_ok =
        positions_.Data() != nullptr && prev_positions_.Data() != nullptr &&
        velocities_.Data() != nullptr && inv_masses_.Data() != nullptr;
    if (distance_constraint_count_ == 0u) {
        return particles_ok;
    }
    return particles_ok && distance_particle_a_.Data() != nullptr &&
           distance_particle_b_.Data() != nullptr &&
           distance_rest_length_.Data() != nullptr &&
           distance_compliance_alpha_.Data() != nullptr &&
           distance_lambda_.Data() != nullptr;
}

XpbdWorldState XpbdWorld::DownloadState() const {
    XpbdWorldState state;
    state.positions = phi::DownloadVector<math::Vec3>(positions_, particle_count_);
    state.velocities = phi::DownloadVector<math::Vec3>(velocities_, particle_count_);
    return state;
}

math::Vec3* XpbdWorld::DevicePositions() {
    return static_cast<math::Vec3*>(positions_.Data());
}
const math::Vec3* XpbdWorld::DevicePositions() const {
    return static_cast<const math::Vec3*>(positions_.Data());
}
math::Vec3* XpbdWorld::DevicePrevPositions() {
    return static_cast<math::Vec3*>(prev_positions_.Data());
}
math::Vec3* XpbdWorld::DeviceVelocities() {
    return static_cast<math::Vec3*>(velocities_.Data());
}
const math::Vec3* XpbdWorld::DeviceVelocities() const {
    return static_cast<const math::Vec3*>(velocities_.Data());
}
const float* XpbdWorld::DeviceInvMasses() const {
    return static_cast<const float*>(inv_masses_.Data());
}
const uint32_t* XpbdWorld::DeviceDistanceParticleA() const {
    return static_cast<const uint32_t*>(distance_particle_a_.Data());
}
const uint32_t* XpbdWorld::DeviceDistanceParticleB() const {
    return static_cast<const uint32_t*>(distance_particle_b_.Data());
}
const float* XpbdWorld::DeviceDistanceRestLength() const {
    return static_cast<const float*>(distance_rest_length_.Data());
}
const float* XpbdWorld::DeviceDistanceComplianceAlpha() const {
    return static_cast<const float*>(distance_compliance_alpha_.Data());
}
float* XpbdWorld::DeviceDistanceLambda() {
    return static_cast<float*>(distance_lambda_.Data());
}

// =============================================================================
// Upload
// =============================================================================

XpbdWorld UploadXpbdWorld(const phi::DeviceContext& context,
                          const XpbdParticleSet& particles,
                          const XpbdConstraintSet& constraints) {
    phi::ScopedDeviceGuard guard(context.device_id);

    const auto count = particles.positions.size();
    if (particles.velocities.size() != count ||
        particles.inv_masses.size() != count) {
        throw std::runtime_error(
            "UploadXpbdWorld requires positions, velocities, and inv_masses to "
            "have matching lengths");
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("UploadXpbdWorld particle count exceeds uint32_t range");
    }
    if (constraints.distance.size() >
        static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("UploadXpbdWorld distance constraint count exceeds uint32_t range");
    }

    const uint32_t particle_count = static_cast<uint32_t>(count);
    const uint32_t constraint_count =
        static_cast<uint32_t>(constraints.distance.size());

    // De-interleave the host constraint AoS into per-field device SoA. Validate
    // particle indices so a malformed constraint surfaces at upload, not as a
    // device OOB read inside the solver.
    std::vector<uint32_t> particle_a(constraint_count);
    std::vector<uint32_t> particle_b(constraint_count);
    std::vector<float> rest_length(constraint_count);
    std::vector<float> compliance_alpha(constraint_count);
    std::vector<float> lambda(constraint_count, 0.0f);
    for (uint32_t c = 0u; c < constraint_count; ++c) {
        const XpbdDistanceConstraint& dc = constraints.distance[c];
        if (dc.particle_a >= particle_count || dc.particle_b >= particle_count) {
            throw std::runtime_error(
                "UploadXpbdWorld distance constraint references out-of-range particle index");
        }
        particle_a[c] = dc.particle_a;
        particle_b[c] = dc.particle_b;
        rest_length[c] = dc.rest_length;
        compliance_alpha[c] = dc.compliance_alpha;
    }

    return XpbdWorld(particle_count,
                     constraint_count,
                     phi::UploadVector(particles.positions),
                     phi::UploadVector(particles.positions),  // prev seeded = p
                     phi::UploadVector(particles.velocities),
                     phi::UploadVector(particles.inv_masses),
                     phi::UploadVector(particle_a),
                     phi::UploadVector(particle_b),
                     phi::UploadVector(rest_length),
                     phi::UploadVector(compliance_alpha),
                     phi::UploadVector(lambda));
}

XpbdWorld UploadXpbdWorld(const XpbdParticleSet& particles,
                          const XpbdConstraintSet& constraints) {
    auto context = phi::MakeDefaultDeviceContext();
    return UploadXpbdWorld(context, particles, constraints);
}

// =============================================================================
// Step: predict -> solve -> correct
// =============================================================================

XpbdStepReport StepXpbdWorld(const phi::DeviceContext& context,
                             XpbdWorld& world,
                             const XpbdStepOptions& options) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    XpbdStepReport report;
    report.particle_count = world.ParticleCount();
    report.distance_constraint_count = world.DistanceConstraintCount();

    if (options.dt <= 0.0f || options.step_count == 0u ||
        world.ParticleCount() == 0u) {
        return report;
    }
    if (!world.HasUploadedState()) {
        throw std::runtime_error(
            "StepXpbdWorld requires UploadXpbdWorld before stepping a non-empty world");
    }

    const uint32_t solver_iterations =
        options.solver_iterations == 0u ? 1u : options.solver_iterations;

    constexpr uint32_t kBlockSize = 128u;
    const uint32_t particle_blocks =
        (world.ParticleCount() + kBlockSize - 1u) / kBlockSize;

    for (uint32_t step = 0u; step < options.step_count; ++step) {
        PredictPositionsKernel<<<particle_blocks, kBlockSize, 0, stream>>>(
            world.ParticleCount(),
            world.DevicePositions(),
            world.DevicePrevPositions(),
            world.DeviceVelocities(),
            world.DeviceInvMasses(),
            options.gravity,
            options.dt);
        CheckCuda(cudaGetLastError(), "PredictPositionsKernel launch");
        ++report.kernel_launch_count;

        if (world.DistanceConstraintCount() > 0u) {
            // Single-thread fixed-order serial sweep (D1). One block, one thread.
            SolveDistanceConstraintsKernel<<<1u, 1u, 0, stream>>>(
                world.DistanceConstraintCount(),
                solver_iterations,
                world.DevicePositions(),
                world.DeviceInvMasses(),
                world.DeviceDistanceParticleA(),
                world.DeviceDistanceParticleB(),
                world.DeviceDistanceRestLength(),
                world.DeviceDistanceComplianceAlpha(),
                world.DeviceDistanceLambda(),
                options.dt);
            CheckCuda(cudaGetLastError(), "SolveDistanceConstraintsKernel launch");
            ++report.kernel_launch_count;
        }

        CorrectVelocitiesKernel<<<particle_blocks, kBlockSize, 0, stream>>>(
            world.ParticleCount(),
            world.DevicePositions(),
            world.DevicePrevPositions(),
            world.DeviceVelocities(),
            world.DeviceInvMasses(),
            options.dt);
        CheckCuda(cudaGetLastError(), "CorrectVelocitiesKernel launch");
        ++report.kernel_launch_count;
    }

    context.stream.Synchronize();
    report.simulated_step_count = options.step_count;
    return report;
}

XpbdStepReport StepXpbdWorld(XpbdWorld& world, const XpbdStepOptions& options) {
    auto context = phi::MakeDefaultDeviceContext();
    return StepXpbdWorld(context, world, options);
}

} // namespace nuka::runtime::soft
