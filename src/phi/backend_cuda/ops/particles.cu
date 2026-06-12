// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M6 particle ops (XPBD soft + PBF fluid as backend ops).
//
// LINE-BY-LINE ports of the kernel bodies from
//   src/runtime/soft/xpbd_world.cu  (predict / distance / bend / volume / correct)
//   src/runtime/fluid/pbf_world.cu  (predict / density / lambda / delta / apply /
//                                    finalize / xsph viscosity / cohesion)
// onto the nk arena fields (DataView/ModelView typed device pointers). The op
// order, the reduction order, and every float expression are preserved verbatim
// so the existing XPBD/PBF oracle tolerances hold UNCHANGED and the D1 byte-exact
// contract is kept (single-thread fixed-order GS for XPBD; one-thread-per-particle
// fixed-neighbor-order reductions for PBF, no float atomics).
//
// Two semantics live behind the ONE ParticlePredict / ParticleFinalize op (the
// XPBD vs PBF predict/finalize diverge): the param's `mode` selects the kernel
// body. The PBF density-projection iteration loop lives inside PbfDensityLambda /
// PbfApplyDelta (the pipeline is a flat OpCall list, so the N-iteration coupling
// is owned by the op TU, exactly as the legacy StepPbfWorld owned its inner loop).
//
// NEIGHBOR LAYOUT (M5 arena CSR): the ParticleGridBuild op (broadphase.cu) writes
// each particle's neighbor list into a PRIVATE slice grid_neighbor_idx[i*32 .. ]
// (stride kParticleGridMaxNeighbors == 32, ascending in-cell order) with the live
// count in grid_neighbor_count[i]. The legacy PBF kernels read a FLAT CSR
// (neighbor_indices[neighbor_offsets[i] + k]); the port adapts the indexing to the
// private-slice layout (base = i*32) — the SAME ascending k order, so the
// fixed-order reduction (D1) is unchanged. Self is excluded by the grid query
// (QueryParticleNeighbors skips index i), matching the legacy neighbor list.
//
// NO host allocation, no per-call device allocation (lint hot_path scope). All
// scratch is arena-resident (pbf_predicted_pos / pbf_position_delta / pbf_density
// / pbf_lambda + grid_neighbor_* from M5).
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "math/cuda_vec_ops.cuh"
#include "nk/model/generated/views.hpp"  // ModelView / DataView (complete types)
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"
#include "runtime/fluid/pbf_kernels.cuh"  // Poly6FromR2 / SpikyGradient / coeffs
#include "runtime/fluid/pbf_polish.cuh"   // CohesionSpline / coeffs

namespace nuka::phi {

namespace {

namespace mg = ::nuka::math::gpu;
using mg::Add;
using mg::Cross;
using mg::Dot;
using mg::Scale;
using mg::Sub;
namespace fl = ::nuka::runtime::fluid;

constexpr uint32_t kBlockSize = 128u;
// Mirror collision/particle_uniform_grid.hpp kParticleGridMaxNeighbors — the
// per-particle private CSR slice stride the M5 GridFillKernel wrote.
constexpr uint32_t kNeighborStride = 32u;

// =============================================================================
// XPBD kernels — VERBATIM from runtime/soft/xpbd_world.cu (predict / distance /
// bend / volume / correct). Indices/expressions unchanged; only the buffer
// arguments are the arena fields.
// =============================================================================

// predict: prev = p; p += v*dt + g*dt^2 (velocity NOT mutated; pinned w==0 stay).
__global__ void XpbdPredictKernel(uint32_t particle_count,
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
        return;
    }
    const math::Vec3 step =
        Add(Scale(velocities[i], dt), Scale(gravity, dt * dt));
    positions[i] = Add(p, step);
}

// distance: SINGLE-THREAD fixed-order Gauss-Seidel sweep (D1). Lambda reset at
// step start (iteration 0) — verbatim Macklin 2016.
__global__ void XpbdDistanceKernel(uint32_t distance_constraint_count,
                                   uint32_t solver_iterations,
                                   math::Vec3* __restrict__ positions,
                                   const float* __restrict__ inv_masses,
                                   const uint32_t* __restrict__ particle_a,
                                   const uint32_t* __restrict__ particle_b,
                                   const float* __restrict__ rest_length,
                                   const float* __restrict__ compliance_alpha,
                                   float* __restrict__ lambda,
                                   float dt) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    const float inv_dt2 = 1.0f / (dt * dt);
    for (uint32_t c = 0u; c < distance_constraint_count; ++c) {
        lambda[c] = 0.0f;
    }
    for (uint32_t iter = 0u; iter < solver_iterations; ++iter) {
        for (uint32_t c = 0u; c < distance_constraint_count; ++c) {
            const uint32_t ia = particle_a[c];
            const uint32_t ib = particle_b[c];
            const float wa = inv_masses[ia];
            const float wb = inv_masses[ib];
            const float w_sum = wa + wb;
            if (w_sum <= 0.0f) {
                continue;
            }
            const math::Vec3 pa = positions[ia];
            const math::Vec3 pb = positions[ib];
            const math::Vec3 r = Sub(pa, pb);
            const float dist = sqrtf(Dot(r, r));
            if (dist <= 0.0f) {
                continue;
            }
            const math::Vec3 n = Scale(r, 1.0f / dist);
            const float constraint = dist - rest_length[c];
            const float alpha_tilde = compliance_alpha[c] * inv_dt2;
            const float lam = lambda[c];
            const float delta_lambda =
                (-constraint - alpha_tilde * lam) / (w_sum + alpha_tilde);
            positions[ia] = Add(pa, Scale(n, wa * delta_lambda));
            positions[ib] = Sub(pb, Scale(n, wb * delta_lambda));
            lambda[c] = lam + delta_lambda;
        }
    }
}

// bend (Bergou isometric): SINGLE-THREAD fixed-order Gauss-Seidel sweep (D1).
__global__ void XpbdBendKernel(uint32_t bend_constraint_count,
                               uint32_t solver_iterations,
                               math::Vec3* __restrict__ positions,
                               const float* __restrict__ inv_masses,
                               const uint32_t* __restrict__ particles,
                               const math::Vec3* __restrict__ gradients,
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
                continue;
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

// volume (tet): SINGLE-THREAD fixed-order Gauss-Seidel sweep (D1).
__global__ void XpbdVolumeKernel(uint32_t volume_constraint_count,
                                 uint32_t solver_iterations,
                                 math::Vec3* __restrict__ positions,
                                 const float* __restrict__ inv_masses,
                                 const uint32_t* __restrict__ particles,
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
            const math::Vec3 g1 = Cross(e2, e3);
            const math::Vec3 g2 = Cross(e3, e1);
            const math::Vec3 g3 = Cross(e1, e2);
            const math::Vec3 g0 = Scale(Add(Add(g1, g2), g3), -1.0f);
            const float det = Dot(e1, g1);
            const float constraint = det - rest_times6[c];
            const float w0 = inv_masses[i0];
            const float w1 = inv_masses[i1];
            const float w2 = inv_masses[i2];
            const float w3 = inv_masses[i3];
            const float alpha_tilde = compliance_alpha[c] * inv_dt2;
            const float denom = w0 * Dot(g0, g0) + w1 * Dot(g1, g1) +
                                w2 * Dot(g2, g2) + w3 * Dot(g3, g3) + alpha_tilde;
            if (denom <= 0.0f) {
                continue;
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

// correct: v = (p - prev)/dt (pinned keep their stored velocity). Verbatim.
__global__ void XpbdCorrectKernel(uint32_t particle_count,
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
        return;
    }
    const math::Vec3 delta = Sub(positions[i], prev_positions[i]);
    velocities[i] = Scale(delta, 1.0f / dt);
}

// =============================================================================
// PBF kernels — VERBATIM from runtime/fluid/pbf_world.cu. The ONLY change is the
// neighbor read: the M5 arena grid writes a PRIVATE per-particle slice
// (neighbor_idx[i*32 + k], count[i]), so `base = i * kNeighborStride` here in
// place of the legacy flat `neighbor_offsets[i]`. The k-loop order is identical.
// =============================================================================

// predict: v += dt*g ; pbf_predicted = p + dt*v (positions left at t0). Verbatim.
__global__ void PbfPredictKernel(uint32_t particle_count,
                                 const math::Vec3* __restrict__ positions,
                                 math::Vec3* __restrict__ velocities,
                                 math::Vec3* __restrict__ predicted_positions,
                                 const float* __restrict__ inv_masses,
                                 math::Vec3 gravity,
                                 float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    math::Vec3 v = velocities[i];
    // A pinned PBF particle (inv_mass == 0) holds its velocity + position (the
    // boundary/static-particle convention; the legacy fluid-only world had no
    // pinned particles so this guard is inert there, byte-exact, and lets a
    // coupled scene anchor boundary particles).
    if (inv_masses[i] > 0.0f) {
        v.x += dt * gravity.x;
        v.y += dt * gravity.y;
        v.z += dt * gravity.z;
        velocities[i] = v;
    }
    const math::Vec3 p = positions[i];
    predicted_positions[i] =
        math::Vec3{p.x + dt * v.x, p.y + dt * v.y, p.z + dt * v.z};
}

// density (Poly6, self term first, ascending neighbor sum). Verbatim.
__global__ void PbfDensityKernel(uint32_t particle_count,
                                 const math::Vec3* __restrict__ predicted,
                                 float particle_mass,
                                 fl::PbfKernelCoeffs coeffs,
                                 const uint32_t* __restrict__ neighbor_counts,
                                 const uint32_t* __restrict__ neighbor_indices,
                                 float* __restrict__ out_density) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 pi = predicted[i];
    float rho = particle_mass * fl::Poly6FromR2(0.0f, coeffs);
    const uint32_t base = i * kNeighborStride;
    const uint32_t n = neighbor_counts[i];
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        const math::Vec3 r = Sub(pi, predicted[j]);
        const float r2 = Dot(r, r);
        rho += particle_mass * fl::Poly6FromR2(r2, coeffs);
    }
    out_density[i] = rho;
}

// lambda (M&M eq.9-11). Verbatim.
__global__ void PbfLambdaKernel(uint32_t particle_count,
                                const math::Vec3* __restrict__ predicted,
                                fl::PbfKernelCoeffs coeffs,
                                float inv_rho0,
                                float rest_density,
                                float cfm_epsilon,
                                bool clamp_to_overdensity,
                                const float* __restrict__ density,
                                const uint32_t* __restrict__ neighbor_counts,
                                const uint32_t* __restrict__ neighbor_indices,
                                float* __restrict__ out_lambda) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    float c_i = density[i] / rest_density - 1.0f;
    if (clamp_to_overdensity && c_i < 0.0f) {
        out_lambda[i] = 0.0f;
        return;
    }
    const math::Vec3 pi = predicted[i];
    const uint32_t base = i * kNeighborStride;
    const uint32_t n = neighbor_counts[i];
    math::Vec3 grad_i{0.0f, 0.0f, 0.0f};
    float sum_grad_sq = 0.0f;
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        const math::Vec3 r = Sub(pi, predicted[j]);
        const float dist = sqrtf(Dot(r, r));
        const math::Vec3 sg = fl::SpikyGradient(r, dist, coeffs);
        const math::Vec3 grad_j{sg.x * inv_rho0, sg.y * inv_rho0, sg.z * inv_rho0};
        sum_grad_sq += Dot(grad_j, grad_j);
        grad_i.x += grad_j.x;
        grad_i.y += grad_j.y;
        grad_i.z += grad_j.z;
    }
    sum_grad_sq += Dot(grad_i, grad_i);
    out_lambda[i] = -c_i / (sum_grad_sq + cfm_epsilon);
}

// correction compute (Jacobi, read-only on predicted). Verbatim.
__global__ void PbfComputeCorrectionKernel(
    uint32_t particle_count,
    const math::Vec3* __restrict__ predicted,
    fl::PbfKernelCoeffs coeffs,
    float inv_rho0,
    const float* __restrict__ lambda,
    const uint32_t* __restrict__ neighbor_counts,
    const uint32_t* __restrict__ neighbor_indices,
    math::Vec3* __restrict__ out_delta) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 pi = predicted[i];
    const float lam_i = lambda[i];
    const uint32_t base = i * kNeighborStride;
    const uint32_t n = neighbor_counts[i];
    math::Vec3 dp{0.0f, 0.0f, 0.0f};
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        const math::Vec3 r = Sub(pi, predicted[j]);
        const float dist = sqrtf(Dot(r, r));
        const math::Vec3 sg = fl::SpikyGradient(r, dist, coeffs);
        const float scale = (lam_i + lambda[j]) * inv_rho0;
        dp.x += sg.x * scale;
        dp.y += sg.y * scale;
        dp.z += sg.z * scale;
    }
    out_delta[i] = dp;
}

// correction apply (own-index write + optional floor clamp). The M5 grid is
// z-up; the boundary clamps the predicted z (the legacy clamped y — same shape).
__global__ void PbfApplyCorrectionKernel(uint32_t particle_count,
                                         math::Vec3* __restrict__ predicted,
                                         const math::Vec3* __restrict__ delta,
                                         bool boundary_enabled,
                                         float floor_z) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 pi = predicted[i];
    const math::Vec3 dp = delta[i];
    math::Vec3 out{pi.x + dp.x, pi.y + dp.y, pi.z + dp.z};
    if (boundary_enabled && out.z < floor_z) {
        out.z = floor_z;
    }
    predicted[i] = out;
}

// finalize: v = (predicted - position)/dt ; position = predicted. Verbatim
// (pinned particles, inv_mass 0, keep their velocity + held position — the
// boundary-anchor extension; inert for the fluid-only world).
__global__ void PbfFinalizeKernel(uint32_t particle_count,
                                  math::Vec3* __restrict__ positions,
                                  const math::Vec3* __restrict__ predicted,
                                  math::Vec3* __restrict__ velocities,
                                  const float* __restrict__ inv_masses,
                                  float inv_dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    if (inv_masses[i] <= 0.0f) {
        return;
    }
    const math::Vec3 p0 = positions[i];
    const math::Vec3 pp = predicted[i];
    velocities[i] = math::Vec3{(pp.x - p0.x) * inv_dt, (pp.y - p0.y) * inv_dt,
                               (pp.z - p0.z) * inv_dt};
    positions[i] = pp;
}

// XSPH viscosity compute (read-only into the delta scratch). Verbatim.
__global__ void PbfXsphDeltaKernel(uint32_t particle_count,
                                   const math::Vec3* __restrict__ positions,
                                   const math::Vec3* __restrict__ velocities,
                                   const float* __restrict__ density,
                                   float particle_mass,
                                   float c,
                                   fl::PbfKernelCoeffs coeffs,
                                   const uint32_t* __restrict__ neighbor_counts,
                                   const uint32_t* __restrict__ neighbor_indices,
                                   math::Vec3* __restrict__ out_dv) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 pi = positions[i];
    const math::Vec3 vi = velocities[i];
    const uint32_t base = i * kNeighborStride;
    const uint32_t n = neighbor_counts[i];
    math::Vec3 acc{0.0f, 0.0f, 0.0f};
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        const math::Vec3 r = Sub(pi, positions[j]);
        const float r2 = Dot(r, r);
        const float w = fl::Poly6FromR2(r2, coeffs);
        const float rho_j = density[j];
        if (rho_j <= 0.0f) {
            continue;
        }
        const float scale = (particle_mass / rho_j) * w;
        const math::Vec3 dvj = Sub(velocities[j], vi);
        acc.x += dvj.x * scale;
        acc.y += dvj.y * scale;
        acc.z += dvj.z * scale;
    }
    out_dv[i] = math::Vec3{acc.x * c, acc.y * c, acc.z * c};
}

// XSPH viscosity apply (own-index). Verbatim.
__global__ void PbfApplyVelocityDeltaKernel(uint32_t particle_count,
                                            math::Vec3* __restrict__ velocities,
                                            const math::Vec3* __restrict__ dv) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 v = velocities[i];
    const math::Vec3 d = dv[i];
    velocities[i] = math::Vec3{v.x + d.x, v.y + d.y, v.z + d.z};
}

// cohesion velocity nudge (Akinci cohesion only). Verbatim.
__global__ void PbfCohesionKernel(uint32_t particle_count,
                                  const math::Vec3* __restrict__ positions,
                                  math::Vec3* __restrict__ velocities,
                                  float particle_mass,
                                  float gamma,
                                  float dt,
                                  fl::PbfCohesionCoeffs coeffs,
                                  const uint32_t* __restrict__ neighbor_counts,
                                  const uint32_t* __restrict__ neighbor_indices) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 pi = positions[i];
    const uint32_t base = i * kNeighborStride;
    const uint32_t n = neighbor_counts[i];
    math::Vec3 force{0.0f, 0.0f, 0.0f};
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        const math::Vec3 r = Sub(pi, positions[j]);
        const float dist = sqrtf(Dot(r, r));
        if (dist <= 0.0f) {
            continue;
        }
        const float cval = fl::CohesionSpline(dist, coeffs);
        const float scale = -gamma * particle_mass * cval / dist;
        force.x += r.x * scale;
        force.y += r.y * scale;
        force.z += r.z * scale;
    }
    math::Vec3 v = velocities[i];
    v.x += dt * force.x;
    v.y += dt * force.y;
    v.z += dt * force.z;
    velocities[i] = v;
}

// =============================================================================
// COUPLED kernels (M6): particles co-step against rigid/artic bodies through the
// unified row solve. The contact solve corrects the particle velocity BETWEEN
// these two kernels (the legacy unified_costep pre/couple/post ordering inside
// the fixed pipeline). pbf_predicted_pos stores v_pre so finalize composes the
// XPBD soft-constraint (PBD) velocity with the contact velocity delta.
// =============================================================================

// predict: prev = pos; if free: v += g*dt; save v_pre := v; pos += v*dt.
// The PREDICTED position is written into particle_pos so the narrowphase detects
// the coupling contacts on it (the legacy couple-on-predicted-positions). For a
// PBF-internal coupled body, pbf_predicted_pos is seeded == the predicted position
// so the density-projection ops (which own pbf_predicted_pos) run on it; the
// finalize then reads pbf_predicted_pos as the density-projected position. The
// pre-contact velocity is saved into the dedicated particle_v_pre scratch.
__global__ void CoupledPredictKernel(uint32_t particle_count,
                                     math::Vec3* __restrict__ positions,
                                     math::Vec3* __restrict__ prev_positions,
                                     math::Vec3* __restrict__ velocities,
                                     math::Vec3* __restrict__ v_pre,
                                     math::Vec3* __restrict__ pbf_predicted,
                                     uint32_t internal,
                                     const float* __restrict__ inv_masses,
                                     math::Vec3 gravity,
                                     float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 p = positions[i];
    prev_positions[i] = p;
    math::Vec3 v = velocities[i];
    if (inv_masses[i] > 0.0f) {
        v.x += dt * gravity.x;
        v.y += dt * gravity.y;
        v.z += dt * gravity.z;
        velocities[i] = v;
    }
    v_pre[i] = v;  // pre-contact (predicted) velocity.
    math::Vec3 pred = p;
    if (inv_masses[i] > 0.0f) {
        pred = math::Vec3{p.x + v.x * dt, p.y + v.y * dt, p.z + v.z * dt};
        positions[i] = pred;
    }
    // PBF-internal coupled body: seed the density-projection working buffer.
    if (internal == kCoupledInternalPbf) {
        pbf_predicted[i] = pred;
    }
}

// finalize: compose the PBD (soft-constraint / density) velocity with the contact
// delta.
//   contact_delta = v_now - v_pre              (the row solve's velocity correction)
//   pbd_vel       = (pos_projected - prev)/dt   (the internal-projection velocity)
//   v_final       = pbd_vel + contact_delta
//   pos_final     = prev + v_final*dt
// pos_projected is pbf_predicted_pos for a PBF-internal body (the density-projected
// position) and particle_pos for XPBD / free-point bodies. A pinned particle
// (inv_mass 0) holds position + velocity (boundary anchor).
__global__ void CoupledFinalizeKernel(uint32_t particle_count,
                                      math::Vec3* __restrict__ positions,
                                      const math::Vec3* __restrict__ prev_positions,
                                      math::Vec3* __restrict__ velocities,
                                      const math::Vec3* __restrict__ v_pre,
                                      const math::Vec3* __restrict__ pbf_predicted,
                                      uint32_t internal,
                                      const float* __restrict__ inv_masses,
                                      float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    if (inv_masses[i] <= 0.0f) {
        return;
    }
    const math::Vec3 prev = prev_positions[i];
    const math::Vec3 pos_proj = (internal == kCoupledInternalPbf)
                                    ? pbf_predicted[i]   // density-projected pos.
                                    : positions[i];      // XPBD / free predicted pos.
    const math::Vec3 v_now = velocities[i];        // contact-corrected velocity.
    const math::Vec3 vp = v_pre[i];                // pre-contact velocity.
    const float inv_dt = 1.0f / dt;
    const math::Vec3 pbd_v{(pos_proj.x - prev.x) * inv_dt,
                           (pos_proj.y - prev.y) * inv_dt,
                           (pos_proj.z - prev.z) * inv_dt};
    const math::Vec3 v_final{pbd_v.x + (v_now.x - vp.x), pbd_v.y + (v_now.y - vp.y),
                             pbd_v.z + (v_now.z - vp.z)};
    velocities[i] = v_final;
    positions[i] = math::Vec3{prev.x + v_final.x * dt, prev.y + v_final.y * dt,
                              prev.z + v_final.z * dt};
}

// =============================================================================
// op entry points
// =============================================================================

Status OpParticlePredict(const ModelView& /*model*/, const DataView& data,
                         const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ParticlePredictParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->mode == kParticleModeNone || p->particle_count == 0u) {
        return Status::Ok;  // no particles: inert (the union/foot graph never adds it).
    }
    const uint32_t blocks = (p->particle_count + kBlockSize - 1u) / kBlockSize;
    const math::Vec3 g{p->gravity[0], p->gravity[1], p->gravity[2]};
    if (p->mode == kParticleModeXpbd) {
        LaunchCuda(XpbdPredictKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   p->particle_count, data.particle_pos, data.particle_prev_pos,
                   data.particle_vel, data.particle_inv_mass, g, p->dt);
    } else if (p->mode == kParticleModeCoupled) {
        LaunchCuda(CoupledPredictKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   p->particle_count, data.particle_pos, data.particle_prev_pos,
                   data.particle_vel, data.particle_v_pre, data.pbf_predicted_pos,
                   p->coupled_internal, data.particle_inv_mass, g, p->dt);
    } else {  // kParticleModePbf
        LaunchCuda(PbfPredictKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   p->particle_count, data.particle_pos, data.particle_vel,
                   data.pbf_predicted_pos, data.particle_inv_mass, g, p->dt);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpXpbdProject(const ModelView& model, const DataView& data,
                     const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const XpbdProjectParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->dist_con_count == 0u && p->bend_con_count == 0u &&
        p->vol_con_count == 0u) {
        return Status::Ok;  // PBF-only (or no XPBD) scene: inert.
    }
    const uint32_t iters = p->iters == 0u ? 1u : p->iters;
    // Solve order is FIXED (distance, then bend, then volume), each a
    // single-thread serial sweep — D1 by construction (legacy StepXpbdWorld).
    if (p->dist_con_count > 0u) {
        LaunchCuda(XpbdDistanceKernel, dim3(1u), dim3(1u), 0u, stream,
                   p->dist_con_count, iters, data.particle_pos,
                   data.particle_inv_mass, model.dist_particle_a,
                   model.dist_particle_b, model.dist_rest_length,
                   model.dist_compliance, data.dist_lambda, p->dt);
    }
    if (p->bend_con_count > 0u) {
        LaunchCuda(XpbdBendKernel, dim3(1u), dim3(1u), 0u, stream,
                   p->bend_con_count, iters, data.particle_pos,
                   data.particle_inv_mass, model.bend_particles,
                   model.bend_gradients, model.bend_compliance, data.bend_lambda,
                   p->dt);
    }
    if (p->vol_con_count > 0u) {
        LaunchCuda(XpbdVolumeKernel, dim3(1u), dim3(1u), 0u, stream,
                   p->vol_con_count, iters, data.particle_pos,
                   data.particle_inv_mass, model.vol_particles,
                   model.vol_rest_times6, model.vol_compliance, data.vol_lambda,
                   p->dt);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

// PbfDensityLambda owns the FULL iterated density-projection loop EXCEPT the
// final correction-apply pass (PbfApplyDelta runs that). For iters 0..N-2 the
// apply is in-loop here; the LAST iteration's compute leaves `pbf_position_delta`
// staged for PbfApplyDelta. This keeps both ops non-vacuous + byte-exact to the
// legacy [density,lambda,delta,apply]xN inner loop (the pipeline is a flat list,
// so the iteration coupling is owned by the op TU, exactly as StepPbfWorld did).
Status OpPbfDensityLambda(const ModelView& /*model*/, const DataView& data,
                          const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const PbfDensityLambdaParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->particle_count == 0u || p->support_radius <= 0.0f ||
        p->rest_density <= 0.0f) {
        return Status::Ok;  // XPBD-only scene: inert.
    }
    const fl::PbfKernelCoeffs coeffs = fl::MakePbfKernelCoeffs(p->support_radius);
    const float inv_rho0 = 1.0f / p->rest_density;
    const uint32_t iters = p->iters == 0u ? 1u : p->iters;
    const uint32_t N = p->particle_count;
    const uint32_t blocks = (N + kBlockSize - 1u) / kBlockSize;
    for (uint32_t it = 0u; it < iters; ++it) {
        LaunchCuda(PbfDensityKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   N, data.pbf_predicted_pos, p->particle_mass, coeffs,
                   data.grid_neighbor_count, data.grid_neighbor_idx,
                   data.pbf_density);
        LaunchCuda(PbfLambdaKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   N, data.pbf_predicted_pos, coeffs, inv_rho0, p->rest_density,
                   p->relaxation, p->clamp_overdensity != 0u, data.pbf_density,
                   data.grid_neighbor_count, data.grid_neighbor_idx,
                   data.pbf_lambda);
        LaunchCuda(PbfComputeCorrectionKernel, dim3(blocks), dim3(kBlockSize), 0u,
                   stream, N, data.pbf_predicted_pos, coeffs, inv_rho0,
                   data.pbf_lambda, data.grid_neighbor_count,
                   data.grid_neighbor_idx, data.pbf_position_delta);
        // Apply in-loop for every iteration EXCEPT the last (PbfApplyDelta runs
        // the last apply, so the two ops together == the legacy NxN loop).
        if (it + 1u < iters) {
            LaunchCuda(PbfApplyCorrectionKernel, dim3(blocks), dim3(kBlockSize), 0u,
                       stream, N, data.pbf_predicted_pos, data.pbf_position_delta,
                       p->boundary_enabled != 0u, p->floor_z);
        }
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

// PbfApplyDelta: the FINAL correction-apply pass + boundary floor clamp (the last
// iteration's apply of the legacy density-projection loop).
Status OpPbfApplyDelta(const ModelView& /*model*/, const DataView& data,
                       const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const PbfApplyDeltaParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->particle_count == 0u || p->support_radius <= 0.0f) {
        return Status::Ok;  // XPBD-only scene: inert.
    }
    const uint32_t N = p->particle_count;
    const uint32_t blocks = (N + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(PbfApplyCorrectionKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               N, data.pbf_predicted_pos, data.pbf_position_delta,
               p->boundary_enabled != 0u, p->floor_z);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpParticleFinalize(const ModelView& /*model*/, const DataView& data,
                          const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ParticleFinalizeParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->mode == kParticleModeNone || p->particle_count == 0u) {
        return Status::Ok;
    }
    const uint32_t N = p->particle_count;
    const uint32_t blocks = (N + kBlockSize - 1u) / kBlockSize;
    if (p->mode == kParticleModeXpbd) {
        LaunchCuda(XpbdCorrectKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   N, data.particle_pos, data.particle_prev_pos, data.particle_vel,
                   data.particle_inv_mass, p->dt);
        return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
    }
    if (p->mode == kParticleModeCoupled) {
        LaunchCuda(CoupledFinalizeKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   N, data.particle_pos, data.particle_prev_pos, data.particle_vel,
                   data.particle_v_pre, data.pbf_predicted_pos, p->coupled_internal,
                   data.particle_inv_mass, p->dt);
        return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
    }
    // PBF: finalize v from the corrected predicted positions + commit, then the
    // gated post-finalize polish (XSPH viscosity / Akinci cohesion) — exactly the
    // legacy StepPbfWorld order. The polish reuses THIS step's neighbor grid (the
    // M5 arena CSR, still live) and the position-delta scratch (free post-finalize).
    LaunchCuda(PbfFinalizeKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               N, data.particle_pos, data.pbf_predicted_pos, data.particle_vel,
               data.particle_inv_mass, 1.0f / p->dt);
    if (p->xsph_viscosity_c > 0.0f && p->support_radius > 0.0f) {
        const fl::PbfKernelCoeffs coeffs = fl::MakePbfKernelCoeffs(p->support_radius);
        LaunchCuda(PbfXsphDeltaKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   N, data.particle_pos, data.particle_vel, data.pbf_density,
                   p->particle_mass, p->xsph_viscosity_c, coeffs,
                   data.grid_neighbor_count, data.grid_neighbor_idx,
                   data.pbf_position_delta);
        LaunchCuda(PbfApplyVelocityDeltaKernel, dim3(blocks), dim3(kBlockSize), 0u,
                   stream, N, data.particle_vel, data.pbf_position_delta);
    }
    if (p->surface_tension_gamma > 0.0f && p->support_radius > 0.0f) {
        const fl::PbfCohesionCoeffs ccoeffs =
            fl::MakePbfCohesionCoeffs(p->support_radius);
        LaunchCuda(PbfCohesionKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   N, data.particle_pos, data.particle_vel, p->particle_mass,
                   p->surface_tension_gamma, p->dt, ccoeffs,
                   data.grid_neighbor_count, data.grid_neighbor_idx);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkParticleOps() {
    SetCudaOp(NkOp::ParticlePredict, &OpParticlePredict);
    SetCudaOp(NkOp::XpbdProject, &OpXpbdProject);
    SetCudaOp(NkOp::PbfDensityLambda, &OpPbfDensityLambda);
    SetCudaOp(NkOp::PbfApplyDelta, &OpPbfApplyDelta);
    SetCudaOp(NkOp::ParticleFinalize, &OpParticleFinalize);
}

}  // namespace nuka::phi
