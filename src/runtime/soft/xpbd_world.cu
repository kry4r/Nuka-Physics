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

// --- solve: shape-match (Mueller et al. 2005) ---------------------------------
// A 3x3 matrix stored ROW-MAJOR as nine floats m[0..8] (m[3*r+c]). Tiny dense
// helpers used ONLY by the shape-match polar decomposition; kept local so they do
// not leak into the general math headers. All operate in the single-thread
// fixed-order solve path (no atomics, deterministic).
struct Mat3 {
    float m[9];
};

__device__ __forceinline__ Mat3 Mat3Zero() {
    Mat3 a;
    for (int i = 0; i < 9; ++i) {
        a.m[i] = 0.0f;
    }
    return a;
}

__device__ __forceinline__ Mat3 Mat3Identity() {
    Mat3 a = Mat3Zero();
    a.m[0] = 1.0f;
    a.m[4] = 1.0f;
    a.m[8] = 1.0f;
    return a;
}

// out = a * b (row-major 3x3 multiply).
__device__ __forceinline__ Mat3 Mat3Mul(const Mat3& a, const Mat3& b) {
    Mat3 c;
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k) {
                s += a.m[3 * r + k] * b.m[3 * k + col];
            }
            c.m[3 * r + col] = s;
        }
    }
    return c;
}

__device__ __forceinline__ float Mat3Det(const Mat3& a) {
    return a.m[0] * (a.m[4] * a.m[8] - a.m[5] * a.m[7]) -
           a.m[1] * (a.m[3] * a.m[8] - a.m[5] * a.m[6]) +
           a.m[2] * (a.m[3] * a.m[7] - a.m[4] * a.m[6]);
}

// Inverse-transpose of a: (a^-1)^T = cofactor(a) / det(a). Used by the Higham
// Newton polar iteration R <- 0.5 (R + R^-T). Returns identity if |det| is below
// eps (degenerate / rank-deficient cluster -- the iteration then stalls at R and
// the caller's det-correction guard handles the sign).
__device__ __forceinline__ Mat3 Mat3InvTranspose(const Mat3& a, float eps) {
    const float det = Mat3Det(a);
    if (fabsf(det) < eps) {
        return Mat3Identity();
    }
    const float inv_det = 1.0f / det;
    // Cofactor matrix C (C[r][c] = cofactor of a[r][c]); (a^-1)^T = C / det.
    Mat3 c;
    c.m[0] = (a.m[4] * a.m[8] - a.m[5] * a.m[7]) * inv_det;
    c.m[1] = -(a.m[3] * a.m[8] - a.m[5] * a.m[6]) * inv_det;
    c.m[2] = (a.m[3] * a.m[7] - a.m[4] * a.m[6]) * inv_det;
    c.m[3] = -(a.m[1] * a.m[8] - a.m[2] * a.m[7]) * inv_det;
    c.m[4] = (a.m[0] * a.m[8] - a.m[2] * a.m[6]) * inv_det;
    c.m[5] = -(a.m[0] * a.m[7] - a.m[1] * a.m[6]) * inv_det;
    c.m[6] = (a.m[1] * a.m[5] - a.m[2] * a.m[4]) * inv_det;
    c.m[7] = -(a.m[0] * a.m[5] - a.m[2] * a.m[3]) * inv_det;
    c.m[8] = (a.m[0] * a.m[4] - a.m[1] * a.m[3]) * inv_det;
    return c;
}

// Orthogonal (rotation) factor R of the polar decomposition A = R S via the
// Higham Newton iteration R_{k+1} = 0.5 (R_k + R_k^-T), R_0 = A. A FIXED iteration
// count (kPolarIters) keeps the result byte-stable across runs (D1). The
// iteration converges to the orthogonal factor carrying the sign of det(A); a
// non-inverted cluster (det A > 0) yields a proper rotation directly. The
// det-correction guard flips the last column if a numerical / inverted case
// produced det(R) < 0, so R is always a proper rotation (det +1).
__device__ __forceinline__ Mat3 PolarRotation(const Mat3& A) {
    constexpr int kPolarIters = 24;
    constexpr float kDetEps = 1.0e-12f;
    Mat3 R = A;
    for (int it = 0; it < kPolarIters; ++it) {
        const Mat3 RinvT = Mat3InvTranspose(R, kDetEps);
        for (int i = 0; i < 9; ++i) {
            R.m[i] = 0.5f * (R.m[i] + RinvT.m[i]);
        }
    }
    // Det-correction: ensure a proper rotation (det +1). If the cluster is
    // inverted (det A < 0) the Newton factor is improper (det -1); flip the
    // third column to restore det +1 (Kabsch / Mueller det-correct convention).
    if (Mat3Det(R) < 0.0f) {
        R.m[2] = -R.m[2];
        R.m[5] = -R.m[5];
        R.m[8] = -R.m[8];
    }
    return R;
}

// SINGLE-THREAD fixed-order Gauss-Seidel sweep over all shape-match clusters. For
// each cluster (Mueller et al. 2005):
//   c   = (sum_i m_i p_i) / sum_i m_i                 (current centroid)
//   A   = sum_i m_i (p_i - c) q_i^T,  q_i = x_i^0 - c0 (3x3 covariance; cooked q_i)
//   R   = polar(A)  (det-corrected proper rotation)
//   g_i = c + R q_i                                   (shape-matched goal)
//   p_i += w_i_active * stiffness * (g_i - p_i)        (goal pull; pinned skip)
// The centroid + covariance sums are SINGLE-THREAD ASCENDING accumulations (no
// float atomicAdd) -- D1 fixed order. The shape-match WEIGHT m_i (cluster mass)
// is independent of the dynamics inverse mass w_i; a pinned particle (inv_mass 0)
// still anchors the cluster shape but its position is held fixed.
__global__ void SolveShapeMatchConstraintsKernel(
    uint32_t cluster_count,
    uint32_t solver_iterations,
    math::Vec3* __restrict__ positions,
    const float* __restrict__ inv_masses,
    const uint32_t* __restrict__ cluster_offset,
    const uint32_t* __restrict__ cluster_size,
    const float* __restrict__ stiffness,
    const math::Vec3* __restrict__ rest_centroid,  // c0 (unused; q_i pre-offset)
    const uint32_t* __restrict__ particles,        // sum(n_c)
    const math::Vec3* __restrict__ rest_q,         // q_i = x_i^0 - c0
    const float* __restrict__ mass) {              // m_i
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    (void)rest_centroid;  // c0 folded into the cooked q_i; kept for completeness.

    for (uint32_t iter = 0u; iter < solver_iterations; ++iter) {
        for (uint32_t cc = 0u; cc < cluster_count; ++cc) {
            const uint32_t base = cluster_offset[cc];
            const uint32_t n = cluster_size[cc];
            if (n == 0u) {
                continue;
            }
            const float s = stiffness[cc];

            // Current mass-weighted centroid c (fixed-order ascending sum).
            float mass_sum = 0.0f;
            math::Vec3 c_acc = MakeVec3(0.0f, 0.0f, 0.0f);
            for (uint32_t j = 0u; j < n; ++j) {
                const uint32_t idx = particles[base + j];
                const float mi = mass[base + j];
                mass_sum += mi;
                c_acc = Add(c_acc, Scale(positions[idx], mi));
            }
            if (mass_sum <= 0.0f) {
                continue;  // degenerate cluster weights.
            }
            const math::Vec3 c = Scale(c_acc, 1.0f / mass_sum);

            // Covariance A = sum_i m_i (p_i - c) q_i^T (row-major; fixed order).
            Mat3 A = Mat3Zero();
            for (uint32_t j = 0u; j < n; ++j) {
                const uint32_t idx = particles[base + j];
                const float mi = mass[base + j];
                const math::Vec3 d = Sub(positions[idx], c);  // p_i - c
                const math::Vec3 q = rest_q[base + j];        // q_i = x_i^0 - c0
                A.m[0] += mi * d.x * q.x;
                A.m[1] += mi * d.x * q.y;
                A.m[2] += mi * d.x * q.z;
                A.m[3] += mi * d.y * q.x;
                A.m[4] += mi * d.y * q.y;
                A.m[5] += mi * d.y * q.z;
                A.m[6] += mi * d.z * q.x;
                A.m[7] += mi * d.z * q.y;
                A.m[8] += mi * d.z * q.z;
            }

            const Mat3 R = PolarRotation(A);

            // Goal pull: g_i = c + R q_i ; p_i += w_active * s * (g_i - p_i).
            for (uint32_t j = 0u; j < n; ++j) {
                const uint32_t idx = particles[base + j];
                if (inv_masses[idx] <= 0.0f) {
                    continue;  // pinned particle: position held fixed.
                }
                const math::Vec3 q = rest_q[base + j];
                const math::Vec3 rq = MakeVec3(
                    R.m[0] * q.x + R.m[1] * q.y + R.m[2] * q.z,
                    R.m[3] * q.x + R.m[4] * q.y + R.m[5] * q.z,
                    R.m[6] * q.x + R.m[7] * q.y + R.m[8] * q.z);
                const math::Vec3 goal = Add(c, rq);
                const math::Vec3 p = positions[idx];
                positions[idx] = Add(p, Scale(Sub(goal, p), s));
            }
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
                     uint32_t shape_match_cluster_count,
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
                     phi::Buffer volume_lambda,
                     phi::Buffer shape_match_cluster_offset,
                     phi::Buffer shape_match_cluster_count_buf,
                     phi::Buffer shape_match_stiffness,
                     phi::Buffer shape_match_rest_centroid,
                     phi::Buffer shape_match_particles,
                     phi::Buffer shape_match_rest_q,
                     phi::Buffer shape_match_mass)
    : particle_count_(particle_count)
    , distance_constraint_count_(distance_constraint_count)
    , bend_constraint_count_(bend_constraint_count)
    , volume_constraint_count_(volume_constraint_count)
    , shape_match_cluster_count_(shape_match_cluster_count)
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
    , volume_lambda_(std::move(volume_lambda))
    , shape_match_cluster_offset_(std::move(shape_match_cluster_offset))
    , shape_match_cluster_count_buf_(std::move(shape_match_cluster_count_buf))
    , shape_match_stiffness_(std::move(shape_match_stiffness))
    , shape_match_rest_centroid_(std::move(shape_match_rest_centroid))
    , shape_match_particles_(std::move(shape_match_particles))
    , shape_match_rest_q_(std::move(shape_match_rest_q))
    , shape_match_mass_(std::move(shape_match_mass)) {}

std::size_t XpbdWorld::DeviceBytes() const {
    return positions_.Size() + prev_positions_.Size() + velocities_.Size() +
           inv_masses_.Size() + distance_particle_a_.Size() +
           distance_particle_b_.Size() + distance_rest_length_.Size() +
           distance_compliance_alpha_.Size() + distance_lambda_.Size() +
           bend_particles_.Size() + bend_gradients_.Size() +
           bend_compliance_alpha_.Size() + bend_lambda_.Size() +
           volume_particles_.Size() + volume_rest_times6_.Size() +
           volume_compliance_alpha_.Size() + volume_lambda_.Size() +
           shape_match_cluster_offset_.Size() +
           shape_match_cluster_count_buf_.Size() +
           shape_match_stiffness_.Size() + shape_match_rest_centroid_.Size() +
           shape_match_particles_.Size() + shape_match_rest_q_.Size() +
           shape_match_mass_.Size();
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
    if (shape_match_cluster_count_ > 0u) {
        if (shape_match_cluster_offset_.Data() == nullptr ||
            shape_match_cluster_count_buf_.Data() == nullptr ||
            shape_match_stiffness_.Data() == nullptr ||
            shape_match_rest_centroid_.Data() == nullptr ||
            shape_match_particles_.Data() == nullptr ||
            shape_match_rest_q_.Data() == nullptr ||
            shape_match_mass_.Data() == nullptr) {
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
const uint32_t* XpbdWorld::DeviceShapeMatchClusterOffset() const {
    return static_cast<const uint32_t*>(shape_match_cluster_offset_.Data());
}
const uint32_t* XpbdWorld::DeviceShapeMatchClusterCount() const {
    return static_cast<const uint32_t*>(shape_match_cluster_count_buf_.Data());
}
const float* XpbdWorld::DeviceShapeMatchStiffness() const {
    return static_cast<const float*>(shape_match_stiffness_.Data());
}
const math::Vec3* XpbdWorld::DeviceShapeMatchRestCentroid() const {
    return static_cast<const math::Vec3*>(shape_match_rest_centroid_.Data());
}
const uint32_t* XpbdWorld::DeviceShapeMatchParticles() const {
    return static_cast<const uint32_t*>(shape_match_particles_.Data());
}
const math::Vec3* XpbdWorld::DeviceShapeMatchRestQ() const {
    return static_cast<const math::Vec3*>(shape_match_rest_q_.Data());
}
const float* XpbdWorld::DeviceShapeMatchMass() const {
    return static_cast<const float*>(shape_match_mass_.Data());
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
            static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
        constraints.shape_match.size() >
            static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("UploadXpbdWorld constraint count exceeds uint32_t range");
    }

    const uint32_t particle_count = static_cast<uint32_t>(count);
    const uint32_t distance_count =
        static_cast<uint32_t>(constraints.distance.size());
    const uint32_t bend_count = static_cast<uint32_t>(constraints.bend.size());
    const uint32_t volume_count =
        static_cast<uint32_t>(constraints.volume.size());
    const uint32_t shape_match_count =
        static_cast<uint32_t>(constraints.shape_match.size());

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

    // Shape-match clusters: flatten the variable-size clusters into a CSR-style
    // (offset, count) layout over flat particle / rest-offset / weight arrays. The
    // rest centroid c0 is cooked once per cluster and the per-particle rest
    // OFFSET q_i = x_i^0 - c0 is precomputed so the solve kernel only forms the
    // covariance A = sum_i m_i (p_i - c) q_i^T (no per-step rest-centroid recompute).
    std::vector<uint32_t> sm_offset(shape_match_count);
    std::vector<uint32_t> sm_size(shape_match_count);
    std::vector<float> sm_stiffness(shape_match_count);
    std::vector<math::Vec3> sm_rest_centroid(shape_match_count);
    std::vector<uint32_t> sm_particles;
    std::vector<math::Vec3> sm_rest_q;
    std::vector<float> sm_mass;
    {
        std::size_t total = 0u;
        for (const XpbdShapeMatchCluster& cl : constraints.shape_match) {
            total += cl.particle.size();
        }
        sm_particles.reserve(total);
        sm_rest_q.reserve(total);
        sm_mass.reserve(total);
    }
    for (uint32_t cl = 0u; cl < shape_match_count; ++cl) {
        const XpbdShapeMatchCluster& smc = constraints.shape_match[cl];
        const std::size_t n = smc.particle.size();
        if (n == 0u) {
            throw std::runtime_error(
                "UploadXpbdWorld shape-match cluster must have at least one particle");
        }
        if (smc.rest_positions.size() != n || smc.cluster_mass.size() != n) {
            throw std::runtime_error(
                "UploadXpbdWorld shape-match cluster particle/rest/mass lengths must match");
        }
        // Cook the rest centroid c0 = (sum_i m_i x_i^0) / sum_i m_i (fixed order).
        float mass_sum = 0.0f;
        math::Vec3 c0{0.0f, 0.0f, 0.0f};
        for (std::size_t j = 0u; j < n; ++j) {
            if (smc.particle[j] >= particle_count) {
                throw std::runtime_error(
                    "UploadXpbdWorld shape-match cluster references out-of-range particle index");
            }
            const float mi = smc.cluster_mass[j];
            mass_sum += mi;
            c0 = c0 + smc.rest_positions[j] * mi;
        }
        if (mass_sum <= 0.0f) {
            throw std::runtime_error(
                "UploadXpbdWorld shape-match cluster total weight must be positive");
        }
        c0 = c0 * (1.0f / mass_sum);

        sm_offset[cl] = static_cast<uint32_t>(sm_particles.size());
        sm_size[cl] = static_cast<uint32_t>(n);
        sm_stiffness[cl] = smc.stiffness;
        sm_rest_centroid[cl] = c0;
        for (std::size_t j = 0u; j < n; ++j) {
            sm_particles.push_back(smc.particle[j]);
            sm_rest_q.push_back(smc.rest_positions[j] - c0);  // q_i = x_i^0 - c0
            sm_mass.push_back(smc.cluster_mass[j]);
        }
    }

    return XpbdWorld(particle_count,
                     distance_count,
                     bend_count,
                     volume_count,
                     shape_match_count,
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
                     phi::UploadVector(volume_lambda),
                     phi::UploadVector(sm_offset),
                     phi::UploadVector(sm_size),
                     phi::UploadVector(sm_stiffness),
                     phi::UploadVector(sm_rest_centroid),
                     phi::UploadVector(sm_particles),
                     phi::UploadVector(sm_rest_q),
                     phi::UploadVector(sm_mass));
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
    report.shape_match_cluster_count = world.ShapeMatchClusterCount();

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

        // Shape-match is solved LAST (after the local constraints) so it pulls the
        // already-projected configuration toward the rigid goal. Single-thread
        // serial sweep -- D1 fixed order, no atomics (the kernel does NOT reset a
        // lambda: shape matching has no Lagrange multiplier, it is a direct goal
        // projection re-evaluated from the current positions each iteration).
        if (world.ShapeMatchClusterCount() > 0u) {
            SolveShapeMatchConstraintsKernel<<<1u, 1u, 0, stream>>>(
                world.ShapeMatchClusterCount(),
                solver_iterations,
                world.DevicePositions(),
                world.DeviceInvMasses(),
                world.DeviceShapeMatchClusterOffset(),
                world.DeviceShapeMatchClusterCount(),
                world.DeviceShapeMatchStiffness(),
                world.DeviceShapeMatchRestCentroid(),
                world.DeviceShapeMatchParticles(),
                world.DeviceShapeMatchRestQ(),
                world.DeviceShapeMatchMass());
            CheckCuda(cudaGetLastError(), "SolveShapeMatchConstraintsKernel launch");
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
