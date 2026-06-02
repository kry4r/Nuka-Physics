// ---------------------------------------------------------------------------
// nuka::runtime::soft::XpbdWorld implementation (v0.7 p09-A/B)
// ---------------------------------------------------------------------------
//
// GPU XPBD integration loop: predict -> solve -> correct. p09-A shipped the
// DISTANCE family; p09-B adds the BEND (id 7) and VOLUME (id 8) families. All
// device buffers are allocated once on upload; Step launches only kernels over
// them (no hot-path cudaMalloc). Each constraint solve is a single-thread
// FIXED-ORDER Gauss-Seidel sweep -- D1 strong determinism, no float atomics, no
// thread-order dependence. Parallel/colored GS is deferred (task #25).
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
using mg::Cross;
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

// --- solve: distance ----------------------------------------------------------
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

// --- solve: bend (Bergou isometric) -------------------------------------------
// SINGLE-THREAD fixed-order Gauss-Seidel sweep over all bend constraints. Each
// constraint is over four particles with CONSTANT per-particle gradient vectors
// K_i (= k_i*n_hat_rest, built at cook time). The constraint is LINEAR in the
// positions:
//   C    = sum_i K_i . p_i          (== 0 at the flat rest state)
//   a~   = compliance_alpha / dt^2
//   denom= sum_i w_i*|K_i|^2 + a~    (generalized XPBD; |grad| != 1 here)
//   dl   = (-C - a~*lambda) / denom
//   p_i += w_i*K_i*dl ;  lambda += dl
// sum_i K_i = n_hat * sum_i k_i = 0, so a rigid translation leaves C unchanged
// and the projection conserves linear momentum. D1: serial RMW, no atomics.
__global__ void SolveBendConstraintsKernel(
    uint32_t bend_constraint_count,
    uint32_t solver_iterations,
    math::Vec3* __restrict__ positions,
    const float* __restrict__ inv_masses,
    const uint32_t* __restrict__ particles,    // 4 * count
    const math::Vec3* __restrict__ gradients,  // 4 * count (the K_i)
    const float* __restrict__ compliance_alpha,
    float* __restrict__ lambda,
    float dt) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }

    const float inv_dt2 = 1.0f / (dt * dt);

    for (uint32_t c = 0u; c < bend_constraint_count; ++c) {
        lambda[c] = 0.0f;
    }

    for (uint32_t iter = 0u; iter < solver_iterations; ++iter) {
        for (uint32_t c = 0u; c < bend_constraint_count; ++c) {
            const uint32_t base = c * 4u;
            uint32_t idx[4];
            math::Vec3 grad[4];
            float w[4];
            float denom = 0.0f;
            float constraint = 0.0f;
            for (uint32_t j = 0u; j < 4u; ++j) {
                idx[j] = particles[base + j];
                grad[j] = gradients[base + j];
                w[j] = inv_masses[idx[j]];
                constraint += Dot(grad[j], positions[idx[j]]);
                denom += w[j] * Dot(grad[j], grad[j]);
            }
            const float alpha_tilde = compliance_alpha[c] * inv_dt2;
            denom += alpha_tilde;
            if (denom <= 0.0f) {
                continue;  // all particles pinned and rigid: nothing to project.
            }

            const float lam = lambda[c];
            const float delta_lambda = (-constraint - alpha_tilde * lam) / denom;

            for (uint32_t j = 0u; j < 4u; ++j) {
                if (w[j] > 0.0f) {
                    positions[idx[j]] =
                        Add(positions[idx[j]], Scale(grad[j], w[j] * delta_lambda));
                }
            }
            lambda[c] = lam + delta_lambda;
        }
    }
}

// --- solve: volume (tet) ------------------------------------------------------
// SINGLE-THREAD fixed-order Gauss-Seidel sweep over all tet volume constraints.
//   e1 = p2-p1 ; e2 = p3-p1 ; e3 = p4-p1
//   C    = e1 . (e2 x e3) - 6*V_rest        (det == 6*signed-volume)
//   g2 = e2 x e3 ; g3 = e3 x e1 ; g4 = e1 x e2 ; g1 = -(g2+g3+g4)
//   a~   = compliance_alpha / dt^2
//   denom= sum_i w_i*|g_i|^2 + a~
//   dl   = (-C - a~*lambda) / denom
//   p_i += w_i*g_i*dl ; lambda += dl
// V_rest is supplied as 6*V_rest computed at cook time with the IDENTICAL triple
// product, so C == 0 at rest regardless of winding. sum_i g_i = 0 (momentum).
// D1: serial RMW, no atomics.
__global__ void SolveVolumeConstraintsKernel(
    uint32_t volume_constraint_count,
    uint32_t solver_iterations,
    math::Vec3* __restrict__ positions,
    const float* __restrict__ inv_masses,
    const uint32_t* __restrict__ particles,  // 4 * count
    const float* __restrict__ rest_times6,
    const float* __restrict__ compliance_alpha,
    float* __restrict__ lambda,
    float dt) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }

    const float inv_dt2 = 1.0f / (dt * dt);

    for (uint32_t c = 0u; c < volume_constraint_count; ++c) {
        lambda[c] = 0.0f;
    }

    for (uint32_t iter = 0u; iter < solver_iterations; ++iter) {
        for (uint32_t c = 0u; c < volume_constraint_count; ++c) {
            const uint32_t base = c * 4u;
            const uint32_t i0 = particles[base + 0u];
            const uint32_t i1 = particles[base + 1u];
            const uint32_t i2 = particles[base + 2u];
            const uint32_t i3 = particles[base + 3u];

            const math::Vec3 p0 = positions[i0];
            const math::Vec3 p1 = positions[i1];
            const math::Vec3 p2 = positions[i2];
            const math::Vec3 p3 = positions[i3];

            const math::Vec3 e1 = Sub(p1, p0);
            const math::Vec3 e2 = Sub(p2, p0);
            const math::Vec3 e3 = Sub(p3, p0);

            const math::Vec3 g1 = Cross(e2, e3);  // grad wrt p1 (= det / e1)
            const math::Vec3 g2 = Cross(e3, e1);  // grad wrt p2
            const math::Vec3 g3 = Cross(e1, e2);  // grad wrt p3
            const math::Vec3 g0 =                 // grad wrt p0 = -(g1+g2+g3)
                Scale(Add(Add(g1, g2), g3), -1.0f);

            const float det = Dot(e1, g1);  // e1 . (e2 x e3) == 6*signed-volume
            const float constraint = det - rest_times6[c];

            const float w0 = inv_masses[i0];
            const float w1 = inv_masses[i1];
            const float w2 = inv_masses[i2];
            const float w3 = inv_masses[i3];
            const float alpha_tilde = compliance_alpha[c] * inv_dt2;
            const float denom = w0 * Dot(g0, g0) + w1 * Dot(g1, g1) +
                                w2 * Dot(g2, g2) + w3 * Dot(g3, g3) + alpha_tilde;
            if (denom <= 0.0f) {
                continue;  // degenerate (collapsed tet, all pinned): skip.
            }

            const float lam = lambda[c];
            const float delta_lambda = (-constraint - alpha_tilde * lam) / denom;

            if (w0 > 0.0f) positions[i0] = Add(p0, Scale(g0, w0 * delta_lambda));
            if (w1 > 0.0f) positions[i1] = Add(p1, Scale(g1, w1 * delta_lambda));
            if (w2 > 0.0f) positions[i2] = Add(p2, Scale(g2, w2 * delta_lambda));
            if (w3 > 0.0f) positions[i3] = Add(p3, Scale(g3, w3 * delta_lambda));
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
                     uint32_t bend_constraint_count,
                     uint32_t volume_constraint_count,
                     phi::Buffer positions,
                     phi::Buffer prev_positions,
                     phi::Buffer velocities,
                     phi::Buffer inv_masses,
                     phi::Buffer distance_particle_a,
                     phi::Buffer distance_particle_b,
                     phi::Buffer distance_rest_length,
                     phi::Buffer distance_compliance_alpha,
                     phi::Buffer distance_lambda,
                     phi::Buffer bend_particles,
                     phi::Buffer bend_gradients,
                     phi::Buffer bend_compliance_alpha,
                     phi::Buffer bend_lambda,
                     phi::Buffer volume_particles,
                     phi::Buffer volume_rest_times6,
                     phi::Buffer volume_compliance_alpha,
                     phi::Buffer volume_lambda)
    : particle_count_(particle_count)
    , distance_constraint_count_(distance_constraint_count)
    , bend_constraint_count_(bend_constraint_count)
    , volume_constraint_count_(volume_constraint_count)
    , positions_(std::move(positions))
    , prev_positions_(std::move(prev_positions))
    , velocities_(std::move(velocities))
    , inv_masses_(std::move(inv_masses))
    , distance_particle_a_(std::move(distance_particle_a))
    , distance_particle_b_(std::move(distance_particle_b))
    , distance_rest_length_(std::move(distance_rest_length))
    , distance_compliance_alpha_(std::move(distance_compliance_alpha))
    , distance_lambda_(std::move(distance_lambda))
    , bend_particles_(std::move(bend_particles))
    , bend_gradients_(std::move(bend_gradients))
    , bend_compliance_alpha_(std::move(bend_compliance_alpha))
    , bend_lambda_(std::move(bend_lambda))
    , volume_particles_(std::move(volume_particles))
    , volume_rest_times6_(std::move(volume_rest_times6))
    , volume_compliance_alpha_(std::move(volume_compliance_alpha))
    , volume_lambda_(std::move(volume_lambda)) {}

std::size_t XpbdWorld::DeviceBytes() const {
    return positions_.Size() + prev_positions_.Size() + velocities_.Size() +
           inv_masses_.Size() + distance_particle_a_.Size() +
           distance_particle_b_.Size() + distance_rest_length_.Size() +
           distance_compliance_alpha_.Size() + distance_lambda_.Size() +
           bend_particles_.Size() + bend_gradients_.Size() +
           bend_compliance_alpha_.Size() + bend_lambda_.Size() +
           volume_particles_.Size() + volume_rest_times6_.Size() +
           volume_compliance_alpha_.Size() + volume_lambda_.Size();
}

bool XpbdWorld::HasUploadedState() const {
    if (particle_count_ == 0u) {
        return true;
    }
    const bool particles_ok =
        positions_.Data() != nullptr && prev_positions_.Data() != nullptr &&
        velocities_.Data() != nullptr && inv_masses_.Data() != nullptr;
    if (!particles_ok) {
        return false;
    }
    if (distance_constraint_count_ > 0u) {
        if (distance_particle_a_.Data() == nullptr ||
            distance_particle_b_.Data() == nullptr ||
            distance_rest_length_.Data() == nullptr ||
            distance_compliance_alpha_.Data() == nullptr ||
            distance_lambda_.Data() == nullptr) {
            return false;
        }
    }
    if (bend_constraint_count_ > 0u) {
        if (bend_particles_.Data() == nullptr ||
            bend_gradients_.Data() == nullptr ||
            bend_compliance_alpha_.Data() == nullptr ||
            bend_lambda_.Data() == nullptr) {
            return false;
        }
    }
    if (volume_constraint_count_ > 0u) {
        if (volume_particles_.Data() == nullptr ||
            volume_rest_times6_.Data() == nullptr ||
            volume_compliance_alpha_.Data() == nullptr ||
            volume_lambda_.Data() == nullptr) {
            return false;
        }
    }
    return true;
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
const uint32_t* XpbdWorld::DeviceBendParticles() const {
    return static_cast<const uint32_t*>(bend_particles_.Data());
}
const math::Vec3* XpbdWorld::DeviceBendGradients() const {
    return static_cast<const math::Vec3*>(bend_gradients_.Data());
}
const float* XpbdWorld::DeviceBendComplianceAlpha() const {
    return static_cast<const float*>(bend_compliance_alpha_.Data());
}
float* XpbdWorld::DeviceBendLambda() {
    return static_cast<float*>(bend_lambda_.Data());
}
const uint32_t* XpbdWorld::DeviceVolumeParticles() const {
    return static_cast<const uint32_t*>(volume_particles_.Data());
}
const float* XpbdWorld::DeviceVolumeRestTimes6() const {
    return static_cast<const float*>(volume_rest_times6_.Data());
}
const float* XpbdWorld::DeviceVolumeComplianceAlpha() const {
    return static_cast<const float*>(volume_compliance_alpha_.Data());
}
float* XpbdWorld::DeviceVolumeLambda() {
    return static_cast<float*>(volume_lambda_.Data());
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
            static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
        constraints.bend.size() >
            static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
        constraints.volume.size() >
            static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("UploadXpbdWorld constraint count exceeds uint32_t range");
    }

    const uint32_t particle_count = static_cast<uint32_t>(count);
    const uint32_t distance_count =
        static_cast<uint32_t>(constraints.distance.size());
    const uint32_t bend_count = static_cast<uint32_t>(constraints.bend.size());
    const uint32_t volume_count =
        static_cast<uint32_t>(constraints.volume.size());

    // De-interleave the host constraint AoS into per-field device SoA. Validate
    // particle indices so a malformed constraint surfaces at upload, not as a
    // device OOB read inside the solver.
    std::vector<uint32_t> distance_a(distance_count);
    std::vector<uint32_t> distance_b(distance_count);
    std::vector<float> distance_rest(distance_count);
    std::vector<float> distance_alpha(distance_count);
    std::vector<float> distance_lambda(distance_count, 0.0f);
    for (uint32_t c = 0u; c < distance_count; ++c) {
        const XpbdDistanceConstraint& dc = constraints.distance[c];
        if (dc.particle_a >= particle_count || dc.particle_b >= particle_count) {
            throw std::runtime_error(
                "UploadXpbdWorld distance constraint references out-of-range particle index");
        }
        distance_a[c] = dc.particle_a;
        distance_b[c] = dc.particle_b;
        distance_rest[c] = dc.rest_length;
        distance_alpha[c] = dc.compliance_alpha;
    }

    std::vector<uint32_t> bend_particles(static_cast<std::size_t>(bend_count) * 4u);
    std::vector<math::Vec3> bend_gradients(static_cast<std::size_t>(bend_count) * 4u);
    std::vector<float> bend_alpha(bend_count);
    std::vector<float> bend_lambda(bend_count, 0.0f);
    for (uint32_t c = 0u; c < bend_count; ++c) {
        const XpbdBendConstraint& bc = constraints.bend[c];
        for (uint32_t j = 0u; j < 4u; ++j) {
            if (bc.particle[j] >= particle_count) {
                throw std::runtime_error(
                    "UploadXpbdWorld bend constraint references out-of-range particle index");
            }
            bend_particles[static_cast<std::size_t>(c) * 4u + j] = bc.particle[j];
            bend_gradients[static_cast<std::size_t>(c) * 4u + j] = bc.k[j];
        }
        bend_alpha[c] = bc.compliance_alpha;
    }

    std::vector<uint32_t> volume_particles(static_cast<std::size_t>(volume_count) * 4u);
    std::vector<float> volume_rest(volume_count);
    std::vector<float> volume_alpha(volume_count);
    std::vector<float> volume_lambda(volume_count, 0.0f);
    for (uint32_t c = 0u; c < volume_count; ++c) {
        const XpbdVolumeConstraint& vc = constraints.volume[c];
        for (uint32_t j = 0u; j < 4u; ++j) {
            if (vc.particle[j] >= particle_count) {
                throw std::runtime_error(
                    "UploadXpbdWorld volume constraint references out-of-range particle index");
            }
            volume_particles[static_cast<std::size_t>(c) * 4u + j] = vc.particle[j];
        }
        volume_rest[c] = vc.rest_volume_times6;
        volume_alpha[c] = vc.compliance_alpha;
    }

    return XpbdWorld(particle_count,
                     distance_count,
                     bend_count,
                     volume_count,
                     phi::UploadVector(particles.positions),
                     phi::UploadVector(particles.positions),  // prev seeded = p
                     phi::UploadVector(particles.velocities),
                     phi::UploadVector(particles.inv_masses),
                     phi::UploadVector(distance_a),
                     phi::UploadVector(distance_b),
                     phi::UploadVector(distance_rest),
                     phi::UploadVector(distance_alpha),
                     phi::UploadVector(distance_lambda),
                     phi::UploadVector(bend_particles),
                     phi::UploadVector(bend_gradients),
                     phi::UploadVector(bend_alpha),
                     phi::UploadVector(bend_lambda),
                     phi::UploadVector(volume_particles),
                     phi::UploadVector(volume_rest),
                     phi::UploadVector(volume_alpha),
                     phi::UploadVector(volume_lambda));
}

XpbdWorld UploadXpbdWorld(const XpbdParticleSet& particles,
                          const XpbdConstraintSet& constraints) {
    auto context = phi::MakeDefaultDeviceContext();
    return UploadXpbdWorld(context, particles, constraints);
}

// =============================================================================
// Step: predict -> solve (distance, bend, volume) -> correct
// =============================================================================

XpbdStepReport StepXpbdWorld(const phi::DeviceContext& context,
                             XpbdWorld& world,
                             const XpbdStepOptions& options) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    XpbdStepReport report;
    report.particle_count = world.ParticleCount();
    report.distance_constraint_count = world.DistanceConstraintCount();
    report.bend_constraint_count = world.BendConstraintCount();
    report.volume_constraint_count = world.VolumeConstraintCount();

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

        // Solve order is FIXED (distance, then bend, then volume), each a
        // single-thread serial sweep -- D1 by construction.
        if (world.DistanceConstraintCount() > 0u) {
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

        if (world.BendConstraintCount() > 0u) {
            SolveBendConstraintsKernel<<<1u, 1u, 0, stream>>>(
                world.BendConstraintCount(),
                solver_iterations,
                world.DevicePositions(),
                world.DeviceInvMasses(),
                world.DeviceBendParticles(),
                world.DeviceBendGradients(),
                world.DeviceBendComplianceAlpha(),
                world.DeviceBendLambda(),
                options.dt);
            CheckCuda(cudaGetLastError(), "SolveBendConstraintsKernel launch");
            ++report.kernel_launch_count;
        }

        if (world.VolumeConstraintCount() > 0u) {
            SolveVolumeConstraintsKernel<<<1u, 1u, 0, stream>>>(
                world.VolumeConstraintCount(),
                solver_iterations,
                world.DevicePositions(),
                world.DeviceInvMasses(),
                world.DeviceVolumeParticles(),
                world.DeviceVolumeRestTimes6(),
                world.DeviceVolumeComplianceAlpha(),
                world.DeviceVolumeLambda(),
                options.dt);
            CheckCuda(cudaGetLastError(), "SolveVolumeConstraintsKernel launch");
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
