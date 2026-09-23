#include "constraint/dihedral_bend.hpp"

// Particle integration and projection use stable CSR neighbor lists and arena storage.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>

#include "collision/mesh_surface.hpp"
#include "math/cuda_vec_ops.cuh"
#include "nk/model/generated/views.hpp"  // ModelView / DataView (complete types)
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/launch_grid.cuh"
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
using mg::Length;
using mg::Scale;
using mg::Sub;
namespace fl = ::nuka::runtime::fluid;

constexpr uint32_t kBlockSize = 128u;

// Contact geometry and impulse arms share the positions that own momentum.
__global__ void RefitParticleSurfacesKernel(ParticleSurfacesParams p, ModelView model, DataView data) {
    const uint32_t task = blockIdx.x;
    if (task >= p.env_count * p.surfaces_per_env) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t env = task / p.surfaces_per_env;
    const auto info = model.particle_surface_info[task % p.surfaces_per_env];
    const collision::MeshSurfaceView view{
        reinterpret_cast<const float*>(data.particle_pos + size_t{env} * p.particles_per_env),
        model.particle_surface_triangles, model.particle_surface_tree,
        {p.particles_per_env, p.triangles_per_env, p.nodes_per_env}};
    auto* nodes = data.particle_surface_nodes + size_t{env} * p.nodes_per_env + info.node_offset;
    __shared__ uint32_t ready[kBlockSize];
    __shared__ uint32_t valid;
    __shared__ float warp_speed[kBlockSize / 32u];
    if (lane == 0u) valid = 1u;
    float speed_squared = 0.0f;
    for (uint32_t i = lane; i < info.vertex_count; i += blockDim.x) {
        const uint32_t particle = env * p.particles_per_env + info.vertex_offset + i;
        speed_squared = fmaxf(speed_squared, data.particle_vel[particle].LengthSq());
    }
    for (uint32_t offset = warpSize / 2u; offset > 0u; offset /= 2u)
        speed_squared = fmaxf(speed_squared, __shfl_down_sync(0xffffffffu, speed_squared, offset));
    if (lane % warpSize == 0u) warp_speed[lane / warpSize] = speed_squared;
    __syncthreads();
    if (lane == 0u) {
        for (uint32_t i = 0u; i < kBlockSize / 32u; ++i)
            speed_squared = fmaxf(speed_squared, warp_speed[i]);
        data.particle_surface_max_speed[task] = sqrtf(speed_squared);
    }
    // Reverse tiles keep finished child subtrees available while each tile resolves its dependencies.
    for (uint32_t end = info.node_count; end > 0u;) {
        const uint32_t count = min(end, blockDim.x);
        const uint32_t begin = end - count;
        const uint32_t local = begin + lane;
        collision::MeshBvhNode node;
        if (lane < count) node = model.particle_surface_tree[info.node_offset + local];
        const bool leaf = lane < count && node.triangle != ~0u;
        if (leaf) {
            math::Vec3 a, b, c;
            if (!collision::MeshSurfaceTriangle(view, info, node.triangle, a, b, c) ||
                !isfinite(a.LengthSq()) || !isfinite(b.LengthSq()) || !isfinite(c.LengthSq())) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                atomicExch(&valid, 0u);
            } else {
                node.lower = {fminf(a.x, fminf(b.x, c.x)), fminf(a.y, fminf(b.y, c.y)), fminf(a.z, fminf(b.z, c.z))};
                node.upper = {fmaxf(a.x, fmaxf(b.x, c.x)), fmaxf(a.y, fmaxf(b.y, c.y)), fmaxf(a.z, fmaxf(b.z, c.z))};
            }
            nodes[local] = node;
        }
        ready[lane] = leaf || lane >= count;
        uint32_t pending = __syncthreads_count(ready[lane] == 0u);
        if (valid == 0u) break;
        while (pending > 0u) {
            bool completed = false;
            if (ready[lane] == 0u) {
                const uint32_t left_index = local + 1u;
                const uint32_t right_index = model.particle_surface_tree[info.node_offset + left_index].escape;
                if ((left_index >= end || ready[left_index - begin] != 0u) &&
                    (right_index >= end || ready[right_index - begin] != 0u)) {
                    const auto left = nodes[left_index];
                    const auto right = nodes[right_index];
                    node.lower = {fminf(left.lower.x, right.lower.x), fminf(left.lower.y, right.lower.y),
                                  fminf(left.lower.z, right.lower.z)};
                    node.upper = {fmaxf(left.upper.x, right.upper.x), fmaxf(left.upper.y, right.upper.y),
                                  fmaxf(left.upper.z, right.upper.z)};
                    nodes[local] = node;
                    completed = true;
                }
            }
            __syncthreads();
            if (completed) ready[lane] = 1u;
            pending = __syncthreads_count(ready[lane] == 0u);
        }
        end = begin;
    }
    if (valid == 0u && lane == 0u) nodes[0].escape = 0u;
}

Status OpRefitParticleSurfaces(const ModelView& model, const DataView& data,
                              const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ParticleSurfacesParams*>(params);
    if (p == nullptr) return Status::InvalidArgument;
    if (p->surfaces_per_env == 0u || p->env_count == 0u) return Status::Ok;
    if (!model.particle_surface_info || !model.particle_surface_tree ||
        !model.particle_surface_triangles || !data.particle_surface_nodes || !data.particle_pos ||
        !data.particle_vel || !data.particle_surface_max_speed)
        return Status::InvalidArgument;
    const uint32_t blocks = p->env_count * p->surfaces_per_env;
    LaunchCuda(RefitParticleSurfacesKernel, dim3(blocks), dim3(kBlockSize), 0u, stream, *p, model, data);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

// Particles are environment-major; the initial per-environment slice contains soft material.
__device__ __forceinline__ bool SfIsSoft(uint32_t i, uint32_t n_soft,
                                         uint32_t per_env) {
    const uint32_t local = per_env > 0u ? (i % per_env) : i;
    return local < n_soft;
}

// Faces read one velocity time layer; particles gather impulses in stable triangle order.
__global__ void ClothAeroImpulseKernel(uint32_t tri_count,
                                     const uint32_t* __restrict__ tri_verts,
                                     const float* __restrict__ tri_area,
                                     const math::Vec3* __restrict__ positions,
                                     const math::Vec3* __restrict__ velocities,
                                     const float* __restrict__ inv_masses,
                                     const uint32_t* __restrict__ incident_counts,
                                     math::Vec3* __restrict__ impulses,
                                     float drag_normal, float drag_tangent,
                                     float max_dv, float dt) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tri_count) {
        return;
    }
    impulses[t] = {};
    const uint32_t ia = tri_verts[t * 3u + 0u];
    const uint32_t ib = tri_verts[t * 3u + 1u];
    const uint32_t ic = tri_verts[t * 3u + 2u];
    const math::Vec3 pa = positions[ia], pb = positions[ib], pc = positions[ic];
    const math::Vec3 nrm = Cross(Sub(pb, pa), Sub(pc, pa));  // 2*area*n̂
    const float nlen = Length(nrm);
    if (nlen <= 1.0e-12f) {
        return;  // degenerate sliver: no defined normal.
    }
    const math::Vec3 nhat = Scale(nrm, 1.0f / nlen);
    const uint32_t idx[3] = {ia, ib, ic};
    math::Vec3 velocity_sum{};
    float weighted_degree = 0.0f, max_weighted_degree = 0.0f;
    for (uint32_t p : idx) {
        const float weight = inv_masses[p];
        if (weight <= 0.0f) continue;
        velocity_sum = Add(velocity_sum, velocities[p]);
        const float degree = weight * static_cast<float>(incident_counts[p]);
        weighted_degree += degree;
        max_weighted_degree = fmaxf(max_weighted_degree, degree);
    }
    if (weighted_degree == 0.0f) return;
    const math::Vec3 v = Scale(velocity_sum, 1.0f / 3.0f);
    const float vn = Dot(v, nhat);                 // signed normal speed
    const math::Vec3 v_n = Scale(nhat, vn);
    const math::Vec3 v_t = Sub(v, v_n);
    const float vt = Length(v_t);
    const float area = tri_area[t];
    // F = -(Cn|v_n|v_n + Ct|v_t|v_t) A, opposing the motion in each component.
    const math::Vec3 f = Scale(
        Add(Scale(v_n, -drag_normal * fabsf(vn)),
            Scale(v_t, -drag_tangent * vt)), area);
    const math::Vec3 impulse = Scale(f, dt / 3.0f);
    const float magnitude_sq = Dot(impulse, impulse);
    if (magnitude_sq <= 0.0f) return;
    // The incidence bound controls the sum over shared vertices, including its kinetic work.
    const float work = fminf(Dot(velocity_sum, impulse), 0.0f);
    float alpha = fminf(1.0f, -work / (weighted_degree * magnitude_sq));
    if (max_dv > 0.0f)
        alpha = fminf(alpha, max_dv / (sqrtf(magnitude_sq) * max_weighted_degree));
    impulses[t] = Scale(impulse, alpha);
}

__global__ void ClothAeroGatherKernel(uint32_t particle_count,
                                    const uint32_t* __restrict__ offsets,
                                    const uint32_t* __restrict__ counts,
                                    const uint32_t* __restrict__ incident_tri,
                                    const math::Vec3* __restrict__ impulses,
                                    const float* __restrict__ inv_masses,
                                    math::Vec3* __restrict__ velocities) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count || inv_masses[p] <= 0.0f || counts[p] == 0u) return;
    math::Vec3 impulse{};
    const uint32_t offset = offsets[p];
    for (uint32_t i = 0; i < counts[p]; ++i)
        impulse = Add(impulse, impulses[incident_tri[offset + i]]);
    velocities[p] = Add(velocities[p], Scale(impulse, inv_masses[p]));
}

// Prediction saves one step-start position and publishes one common working position.
__global__ void ParticlePredictKernel(
    uint32_t count, uint32_t per_env, uint32_t active_begin,
    const math::Vec3* __restrict__ positions, math::Vec3* __restrict__ previous,
    math::Vec3* __restrict__ predicted, math::Vec3* __restrict__ velocities,
    math::Vec3* __restrict__ contact_reference, math::Vec3* __restrict__ projection_delta,
    const float* __restrict__ inv_mass,
    math::Vec3 gravity, float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || i % per_env < active_begin) return;
    const math::Vec3 start = positions[i];
    previous[i] = start;
    math::Vec3 velocity = velocities[i];
    if (inv_mass[i] > 0.0f) velocity = Add(velocity, Scale(gravity, dt));
    velocities[i] = velocity;
    contact_reference[i] = velocity;
    predicted[i] = inv_mass[i] > 0.0f ? Add(start, Scale(velocity, dt)) : start;
    projection_delta[i] = {};
}

// XPBD multipliers reset to 0 at step start (Macklin 2016). One thread per
// env-major constraint; race-free own-index write (D1, no atomics).
__global__ void XpbdLambdaResetKernel(uint32_t count, float* __restrict__ lambda) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= count) {
        return;
    }
    lambda[c] = 0.0f;
}

__global__ void ParticleProjectionVelocityKernel(
    uint32_t count, uint32_t per_env, uint32_t active_begin,
    const math::Vec3* __restrict__ previous, math::Vec3* __restrict__ projected,
    math::Vec3* __restrict__ projection_delta,
    const float* __restrict__ inv_mass, math::Vec3* __restrict__ velocities,
    math::Vec3* __restrict__ contact_reference, float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || i % per_env < active_begin || inv_mass[i] <= 0.0f) return;
    const math::Vec3 velocity = Add(velocities[i], Scale(projection_delta[i], 1.0f / dt));
    velocities[i] = velocity;
    contact_reference[i] = velocity;
    projection_delta[i] = {};
    projected[i] = Add(previous[i], Scale(velocity, dt));
}

// Reconstruct working positions from the interval start to avoid accumulating position-rounding error.
__global__ void ParticleContactDeltaKernel(
    uint32_t count, uint32_t per_env, uint32_t active_begin,
    const math::Vec3* __restrict__ previous, math::Vec3* __restrict__ projected,
    const math::Vec3* __restrict__ velocities,
    math::Vec3* __restrict__ contact_reference,
    const float* __restrict__ inv_mass, float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || i % per_env < active_begin || inv_mass[i] <= 0.0f) return;
    projected[i] = Add(previous[i], Scale(velocities[i], dt));
    contact_reference[i] = velocities[i];
}

// Pseudo displacement changes the committed position without entering physical velocity.
__global__ void ParticleFinalizeKernel(
    uint32_t count, uint32_t per_env, uint32_t active_begin,
    math::Vec3* __restrict__ positions,
    const math::Vec3* __restrict__ projected, const math::Vec3* __restrict__ velocities,
    const math::Vec3* __restrict__ contact_reference, const math::Vec3* __restrict__ pseudo,
    const float* __restrict__ inv_mass, float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || i % per_env < active_begin || inv_mass[i] <= 0.0f) return;
    const math::Vec3 delta = Sub(velocities[i], contact_reference[i]);
    const math::Vec3 correction = pseudo != nullptr ? Add(delta, pseudo[i]) : delta;
    positions[i] = Add(projected[i], Scale(correction, dt));
}

// Accumulate corrections before absolute-position rounding can erase their velocity impulses.
__device__ __forceinline__ void ApplyProjectionDelta(math::Vec3* positions,
    math::Vec3* accumulated, uint32_t index, math::Vec3 delta) {
    positions[index] = Add(positions[index], delta);
    accumulated[index] = Add(accumulated[index], delta);
}

// Distance projection updates only the two particles owned by its constraint.
struct XpbdDistanceProjector {
    math::Vec3* __restrict__ positions;
    math::Vec3* __restrict__ projection_delta;
    const float* __restrict__ inv_masses;
    const uint32_t* __restrict__ particle_a;
    const uint32_t* __restrict__ particle_b;
    const float* __restrict__ rest_length;
    const float* __restrict__ compliance_alpha;
    float* __restrict__ lambda;
    float dt;

    __device__ __forceinline__ void operator()(uint32_t c) const {
        const float inv_dt2 = 1.0f / (dt * dt);
        const uint32_t ia = particle_a[c];
        const uint32_t ib = particle_b[c];
        const float wa = inv_masses[ia];
        const float wb = inv_masses[ib];
        const float w_sum = wa + wb;
        if (w_sum <= 0.0f) {
            return;
        }
        const math::Vec3 pa = positions[ia];
        const math::Vec3 pb = positions[ib];
        const math::Vec3 r = Sub(pa, pb);
        const float dist = sqrtf(Dot(r, r));
        if (dist <= 0.0f) {
            return;
        }
        const math::Vec3 n = Scale(r, 1.0f / dist);
        const float constraint = dist - rest_length[c];
        const float alpha_tilde = compliance_alpha[c] * inv_dt2;
        const float lam = lambda[c];
        const float delta_lambda =
            (-constraint - alpha_tilde * lam) / (w_sum + alpha_tilde);
        ApplyProjectionDelta(positions, projection_delta, ia, Scale(n, wa * delta_lambda));
        ApplyProjectionDelta(positions, projection_delta, ib, Scale(n, -wb * delta_lambda));
        lambda[c] = lam + delta_lambda;
    }
};

// Signed dihedral bending recomputes gradients from the current geometry.
struct XpbdBendProjector {
    math::Vec3* __restrict__ positions;
    math::Vec3* __restrict__ projection_delta;
    const float* __restrict__ inv_masses;
    const uint32_t* __restrict__ particles;
    const float* __restrict__ rest_angles;
    const float* __restrict__ compliance_alpha;
    float* __restrict__ lambda;
    float dt;

    __device__ __forceinline__ void operator()(uint32_t c) const {
        const float inv_dt2 = 1.0f / (dt * dt);
        const size_t base = static_cast<size_t>(c) * 4u;
        uint32_t idx[4];
        math::Vec3 grad[4];
        float w[4];
        float denom = 0.0f;
        for (uint32_t j = 0u; j < 4u; ++j) idx[j] = particles[base + j];
        const auto geometry = constraint::EvaluateDihedralBend(
            positions[idx[0]], positions[idx[1]], positions[idx[2]], positions[idx[3]]);
        if (!geometry.valid) {
            lambda[c] = 0.0f;
            return;
        }
        const float constraint = nuka::constraint::DihedralBendError(geometry.angle, rest_angles[c]);
        for (uint32_t j = 0u; j < 4u; ++j) {
            grad[j] = geometry.gradients[j];
            w[j] = inv_masses[idx[j]];
            denom += w[j] * Dot(grad[j], grad[j]);
        }
        const float alpha_tilde = compliance_alpha[c] * inv_dt2;
        denom += alpha_tilde;
        if (denom <= 0.0f) {
            return;
        }
        const float lam = lambda[c];
        const float delta_lambda = (-constraint - alpha_tilde * lam) / denom;
        for (uint32_t j = 0u; j < 4u; ++j) {
            if (w[j] > 0.0f) {
                ApplyProjectionDelta(positions, projection_delta, idx[j], Scale(grad[j], w[j] * delta_lambda));
            }
        }
        lambda[c] = lam + delta_lambda;
    }
};

// Tetrahedral volume projection uses the signed rest determinant.
struct XpbdVolumeProjector {
    math::Vec3* __restrict__ positions;
    math::Vec3* __restrict__ projection_delta;
    const float* __restrict__ inv_masses;
    const uint32_t* __restrict__ particles;
    const float* __restrict__ rest_times6;
    const float* __restrict__ compliance_alpha;
    float* __restrict__ lambda;
    float dt;

    __device__ __forceinline__ void operator()(uint32_t c) const {
        const float inv_dt2 = 1.0f / (dt * dt);
        const size_t base = static_cast<size_t>(c) * 4u;
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
            return;
        }
        const float lam = lambda[c];
        const float delta_lambda = (-constraint - alpha_tilde * lam) / denom;
        if (w0 > 0.0f) ApplyProjectionDelta(positions, projection_delta, i0, Scale(g0, w0 * delta_lambda));
        if (w1 > 0.0f) ApplyProjectionDelta(positions, projection_delta, i1, Scale(g1, w1 * delta_lambda));
        if (w2 > 0.0f) ApplyProjectionDelta(positions, projection_delta, i2, Scale(g2, w2 * delta_lambda));
        if (w3 > 0.0f) ApplyProjectionDelta(positions, projection_delta, i3, Scale(g3, w3 * delta_lambda));
        lambda[c] = lam + delta_lambda;
    }
};

// Shape matching uses row-major 3x3 matrices and a fixed-order polar iteration.
struct SmMat3 {
    float m[9];
};
__device__ __forceinline__ SmMat3 SmMat3Zero() {
    SmMat3 a;
    for (int i = 0; i < 9; ++i) {
        a.m[i] = 0.0f;
    }
    return a;
}
__device__ __forceinline__ SmMat3 SmMat3Identity() {
    SmMat3 a = SmMat3Zero();
    a.m[0] = 1.0f;
    a.m[4] = 1.0f;
    a.m[8] = 1.0f;
    return a;
}
__device__ __forceinline__ float SmMat3Det(const SmMat3& a) {
    return a.m[0] * (a.m[4] * a.m[8] - a.m[5] * a.m[7]) -
           a.m[1] * (a.m[3] * a.m[8] - a.m[5] * a.m[6]) +
           a.m[2] * (a.m[3] * a.m[7] - a.m[4] * a.m[6]);
}
// Inverse-transpose (a^-1)^T = cofactor(a)/det(a). Identity if |det| < eps.
__device__ __forceinline__ SmMat3 SmMat3InvTranspose(const SmMat3& a, float eps) {
    const float det = SmMat3Det(a);
    if (fabsf(det) < eps) {
        return SmMat3Identity();
    }
    const float inv_det = 1.0f / det;
    SmMat3 c;
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
// Higham Newton polar rotation R = polar(A), det-corrected proper rotation.
__device__ __forceinline__ SmMat3 SmPolarRotation(const SmMat3& A) {
    constexpr int kPolarIters = 24;
    constexpr float kDetEps = 1.0e-12f;
    SmMat3 R = A;
    for (int it = 0; it < kPolarIters; ++it) {
        const SmMat3 RinvT = SmMat3InvTranspose(R, kDetEps);
        for (int i = 0; i < 9; ++i) {
            R.m[i] = 0.5f * (R.m[i] + RinvT.m[i]);
        }
    }
    if (SmMat3Det(R) < 0.0f) {
        R.m[2] = -R.m[2];
        R.m[5] = -R.m[5];
        R.m[8] = -R.m[8];
    }
    return R;
}
// Shape matching keeps centroid, covariance, and goal updates in member order.
struct XpbdShapeMatchProjector {
    math::Vec3* __restrict__ positions;
    math::Vec3* __restrict__ projection_delta;
    const float* __restrict__ inv_masses;
    const uint32_t* __restrict__ cluster_offset;
    const uint32_t* __restrict__ cluster_size;
    const float* __restrict__ stiffness;
    const math::Vec3* __restrict__ rest_centroid;
    const uint32_t* __restrict__ particles;
    const math::Vec3* __restrict__ rest_q;
    const float* __restrict__ mass;

    __device__ __forceinline__ void operator()(uint32_t cc) const {
        (void)rest_centroid;  // c0 folded into the cooked q_i; kept for completeness.
        using mg::MakeVec3;
        const uint32_t base = cluster_offset[cc];
        const uint32_t n = cluster_size[cc];
        if (n == 0u) {
            return;
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
            return;  // degenerate cluster weights.
        }
        const math::Vec3 c = Scale(c_acc, 1.0f / mass_sum);
        // Covariance A = sum_i m_i (p_i - c) q_i^T (row-major; fixed order).
        SmMat3 A = SmMat3Zero();
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
        const SmMat3 R = SmPolarRotation(A);
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
            ApplyProjectionDelta(positions, projection_delta, idx, Scale(Sub(goal, p), s));
        }
    }
};

template <typename Projector>
__global__ void XpbdColorSweepKernel(Projector project,
                                     const uint32_t* __restrict__ color_segments,
                                     uint32_t colors, uint32_t constraints_per_env,
                                     uint32_t env_count, uint32_t iters,
                                     uint32_t iteration_start) {
    const auto grid = cooperative_groups::this_grid();
    const uint64_t first = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t{gridDim.x} * blockDim.x;
    for (uint32_t iter = 0u; iter < iters; ++iter) {
        for (uint32_t ci = 0u; ci < colors; ++ci) {
            const uint32_t col = ((iteration_start + iter) & 1u) ? colors - 1u - ci : ci;
            const uint32_t offset = color_segments[static_cast<size_t>(col) * 2u];
            const uint32_t count = color_segments[static_cast<size_t>(col) * 2u + 1u];
            const uint64_t work = uint64_t{count} * env_count;
            for (uint64_t item = first; item < work; item += stride) {
                const uint32_t t = static_cast<uint32_t>(item);
                const uint32_t env = t / count;
                project(env * constraints_per_env + offset + t % count);
            }
            // Every thread participates, including trailing threads and empty colors.
            grid.sync();
        }
    }
}

template <typename Projector>
Status PrepareXpbdSweep(uint32_t constraints, uint32_t constraints_per_env,
                        uint32_t env_count, uint32_t colors,
                        const uint32_t* host_segments, const uint32_t* device_segments,
                        uint32_t* blocks) {
    *blocks = 0u;
    if (constraints == 0u) return Status::Ok;
    if (colors == 0u || host_segments == nullptr || device_segments == nullptr ||
        uint64_t{constraints_per_env} * env_count != constraints)
        return Status::InvalidArgument;
    uint64_t end = 0u;
    uint32_t widest_color = 0u;
    for (uint32_t color = 0u; color < colors; ++color) {
        const uint32_t offset = host_segments[static_cast<size_t>(color) * 2u];
        const uint32_t count = host_segments[static_cast<size_t>(color) * 2u + 1u];
        if (offset != end || end + count > constraints_per_env)
            return Status::InvalidArgument;
        end += count;
        widest_color = std::max(widest_color, count);
    }
    if (end != constraints_per_env || widest_color == 0u) return Status::InvalidArgument;
    const uint64_t work = uint64_t{widest_color} * env_count;
    const uint64_t bound = (work - 1u) / kBlockSize + 1u;
    if (work > std::numeric_limits<uint32_t>::max() ||
        bound > std::numeric_limits<uint32_t>::max()) return Status::InvalidArgument;
    return ResidentGridSize(XpbdColorSweepKernel<Projector>, kBlockSize, 0u,
                            static_cast<uint32_t>(bound), blocks) == cudaSuccess
        ? Status::Ok : Status::Failed;
}

template <typename Projector>
Status ProjectXpbdFamily(Projector project, uint32_t blocks, uint32_t constraints,
                         uint32_t constraints_per_env, uint32_t env_count,
                         uint32_t colors, const uint32_t* color_segments,
                         uint32_t iters, uint32_t iteration_start,
                         float* lambda, cudaStream_t stream) {
    if (constraints == 0u) return Status::Ok;
    if (lambda != nullptr && iteration_start == 0u) {
        const uint32_t reset_blocks = (constraints - 1u) / kBlockSize + 1u;
        LaunchCuda(XpbdLambdaResetKernel, dim3(reset_blocks), dim3(kBlockSize), 0u,
                   stream, constraints, lambda);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
    }
    const auto result = LaunchCooperativeCuda(
        XpbdColorSweepKernel<Projector>, dim3(blocks), dim3(kBlockSize), 0u, stream,
        project, color_segments, colors, constraints_per_env, env_count, iters,
        iteration_start);
    if (result != cudaSuccess) return Status::Failed;
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

// Poly6 density sums the self term and fluid neighbors in stable order.
__global__ void PbfDensityKernel(uint32_t particle_count,
                                 uint32_t n_soft,
                                 uint32_t per_env,
                                 const math::Vec3* __restrict__ predicted,
                                 float particle_mass,
                                 fl::PbfKernelCoeffs coeffs,
                                 const uint32_t* __restrict__ neighbor_counts,
                                 const uint32_t* __restrict__ neighbor_offsets,
                                 const uint32_t* __restrict__ neighbor_indices,
                                 float* __restrict__ out_density) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    if (n_soft > 0u && SfIsSoft(i, n_soft, per_env)) {
        return;  // SoftFluid: soft owner is not a fluid particle.
    }
    const math::Vec3 pi = predicted[i];
    float rho = particle_mass * fl::Poly6FromR2(0.0f, coeffs);
    const uint32_t base = neighbor_offsets[i];
    const uint32_t n = neighbor_counts[i];
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        if (n_soft > 0u && SfIsSoft(j, n_soft, per_env)) {
            continue;  // SoftFluid: skip soft neighbors in the fluid density sum.
        }
        const math::Vec3 r = Sub(pi, predicted[j]);
        const float r2 = Dot(r, r);
        rho += particle_mass * fl::Poly6FromR2(r2, coeffs);
    }
    out_density[i] = rho;
}

// lambda (M&M eq.9-11). Verbatim, plus the SoftFluid fluid-slice scope.
__global__ void PbfLambdaKernel(uint32_t particle_count,
                                uint32_t n_soft,
                                uint32_t per_env,
                                const math::Vec3* __restrict__ predicted,
                                fl::PbfKernelCoeffs coeffs,
                                float inv_rho0,
                                float rest_density,
                                float cfm_epsilon,
                                bool clamp_to_overdensity,
                                const float* __restrict__ density,
                                const uint32_t* __restrict__ neighbor_counts,
                                const uint32_t* __restrict__ neighbor_offsets,
                                const uint32_t* __restrict__ neighbor_indices,
                                float* __restrict__ out_lambda) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    if (n_soft > 0u && SfIsSoft(i, n_soft, per_env)) {
        out_lambda[i] = 0.0f;  // SoftFluid: soft owner contributes no fluid lambda.
        return;
    }
    float c_i = density[i] / rest_density - 1.0f;
    if (clamp_to_overdensity && c_i < 0.0f) {
        out_lambda[i] = 0.0f;
        return;
    }
    const math::Vec3 pi = predicted[i];
    const uint32_t base = neighbor_offsets[i];
    const uint32_t n = neighbor_counts[i];
    math::Vec3 grad_i{0.0f, 0.0f, 0.0f};
    float sum_grad_sq = 0.0f;
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        if (n_soft > 0u && SfIsSoft(j, n_soft, per_env)) {
            continue;  // SoftFluid: skip soft neighbors in the fluid gradient sum.
        }
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

// correction compute (Jacobi, read-only on predicted). Verbatim, plus the 
// SoftFluid fluid-slice scope (soft owner gets no delta; soft neighbors skipped).
__global__ void PbfComputeCorrectionKernel(
    uint32_t particle_count,
    uint32_t n_soft,
    uint32_t per_env,
    const math::Vec3* __restrict__ predicted,
    fl::PbfKernelCoeffs coeffs,
    float inv_rho0,
    const float* __restrict__ lambda,
    const uint32_t* __restrict__ neighbor_counts,
    const uint32_t* __restrict__ neighbor_offsets,
    const uint32_t* __restrict__ neighbor_indices,
    math::Vec3* __restrict__ out_delta) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    if (n_soft > 0u && SfIsSoft(i, n_soft, per_env)) {
        out_delta[i] = math::Vec3{0.0f, 0.0f, 0.0f};  // SoftFluid: no fluid delta.
        return;
    }
    const math::Vec3 pi = predicted[i];
    const float lam_i = lambda[i];
    const uint32_t base = neighbor_offsets[i];
    const uint32_t n = neighbor_counts[i];
    math::Vec3 dp{0.0f, 0.0f, 0.0f};
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        if (n_soft > 0u && SfIsSoft(j, n_soft, per_env)) {
            continue;  // SoftFluid: skip soft neighbors in the fluid correction.
        }
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

// correction apply (own-index write + optional floor clamp). The grid is
// z-up; the boundary clamps the predicted z (the legacy clamped y — same shape).
__global__ void PbfApplyCorrectionKernel(uint32_t particle_count,
                                         math::Vec3* __restrict__ predicted,
                                         math::Vec3* __restrict__ projection_delta,
                                         const math::Vec3* __restrict__ delta,
                                         const float* __restrict__ inv_mass,
                                         uint32_t n_soft, uint32_t per_env,
                                         bool boundary_enabled,
                                         float floor_z) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count || inv_mass[i] <= 0.0f || SfIsSoft(i, n_soft, per_env)) {
        return;
    }
    const math::Vec3 pi = predicted[i];
    math::Vec3 dp = delta[i];
    math::Vec3 out{pi.x + dp.x, pi.y + dp.y, pi.z + dp.z};
    if (boundary_enabled && out.z < floor_z) {
        out.z = floor_z;
        dp.z = floor_z - pi.z;
    }
    predicted[i] = out;
    projection_delta[i] = Add(projection_delta[i], dp);
}

// XSPH viscosity reads fluid velocities and stages per-particle corrections.
__global__ void PbfXsphDeltaKernel(uint32_t particle_count,
                                   uint32_t n_soft,
                                   uint32_t per_env,
                                   const math::Vec3* __restrict__ positions,
                                   const math::Vec3* __restrict__ velocities,
                                   const float* __restrict__ density,
                                   float particle_mass,
                                   float c,
                                   fl::PbfKernelCoeffs coeffs,
                                   const uint32_t* __restrict__ neighbor_counts,
    const uint32_t* __restrict__ neighbor_offsets,
                                   const uint32_t* __restrict__ neighbor_indices,
                                   math::Vec3* __restrict__ out_dv) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    if (n_soft > 0u && SfIsSoft(i, n_soft, per_env)) {
        out_dv[i] = math::Vec3{0.0f, 0.0f, 0.0f};  // SoftFluid: no fluid XSPH delta.
        return;
    }
    const math::Vec3 pi = positions[i];
    const math::Vec3 vi = velocities[i];
    const uint32_t base = neighbor_offsets[i];
    const uint32_t n = neighbor_counts[i];
    math::Vec3 acc{0.0f, 0.0f, 0.0f};
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        if (n_soft > 0u && SfIsSoft(j, n_soft, per_env)) {
            continue;  // SoftFluid: skip soft neighbors in the fluid XSPH sum.
        }
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

// XSPH viscosity apply (own-index). Verbatim. A SoftFluid soft owner had its
// out_dv zeroed in the compute pass, so the own-index add is a no-op for it.
__global__ void PbfApplyVelocityDeltaKernel(uint32_t particle_count,
                                            math::Vec3* __restrict__ velocities,
                                            const float* __restrict__ inv_mass,
                                            const math::Vec3* __restrict__ dv) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count || inv_mass[i] <= 0.0f) {
        return;
    }
    const math::Vec3 v = velocities[i];
    const math::Vec3 d = dv[i];
    velocities[i] = math::Vec3{v.x + d.x, v.y + d.y, v.z + d.z};
}

// Akinci cohesion gathers fluid-neighbor velocity corrections in stable order.
__global__ void PbfCohesionKernel(uint32_t particle_count,
                                  uint32_t n_soft,
                                  uint32_t per_env,
                                  const math::Vec3* __restrict__ positions,
                                  math::Vec3* __restrict__ velocities,
                                  const float* __restrict__ inv_mass,
                                  float particle_mass,
                                  float gamma,
                                  float dt,
                                  fl::PbfCohesionCoeffs coeffs,
                                  const uint32_t* __restrict__ neighbor_counts,
    const uint32_t* __restrict__ neighbor_offsets,
                                  const uint32_t* __restrict__ neighbor_indices) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count || inv_mass[i] <= 0.0f) {
        return;
    }
    if (n_soft > 0u && SfIsSoft(i, n_soft, per_env)) {
        return;  // SoftFluid: a soft owner is not nudged by fluid cohesion.
    }
    const math::Vec3 pi = positions[i];
    const uint32_t base = neighbor_offsets[i];
    const uint32_t n = neighbor_counts[i];
    math::Vec3 force{0.0f, 0.0f, 0.0f};
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        if (n_soft > 0u && SfIsSoft(j, n_soft, per_env)) {
            continue;  // SoftFluid: skip soft neighbors in the fluid cohesion sum.
        }
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

// Structural neighbors may overlap in the rest shape; distant folded members still collide.
__device__ __forceinline__ bool RestNeighbors(
    uint32_t i, uint32_t j, uint32_t per_env, float distance,
    const uint32_t* offsets, const uint32_t* elements, const math::Vec3* rest) {
    if (offsets == nullptr) return false;
    const uint32_t a = i % per_env, b = j % per_env;
    const math::Vec3 separation = Sub(rest[a], rest[b]);
    if (Dot(separation, separation) >= distance * distance) return false;
    uint32_t ai = offsets[a], bi = offsets[b];
    const uint32_t ae = offsets[a + 1u], be = offsets[b + 1u];
    while (ai < ae && bi < be) {
        const uint32_t ag = elements[ai], bg = elements[bi];
        if (ag == bg) return true;
        if (ag < bg) ++ai;
        else ++bi;
    }
    return false;
}

// Each particle gathers its mass-weighted contact correction from one geometry time layer.
__global__ void PpContactHalfCorrectionKernel(
    uint32_t union_count,
    uint32_t per_env,
    const math::Vec3* __restrict__ positions,
    const float* __restrict__ inv_mass,
    const uint32_t* __restrict__ topology_offsets,
    const uint32_t* __restrict__ topology_elements,
    const math::Vec3* __restrict__ rest_positions,
    float d_min,
    float alpha_tilde,
    const uint32_t* __restrict__ neighbor_counts,
    const uint32_t* __restrict__ neighbor_offsets,
    const uint32_t* __restrict__ neighbor_indices,
    math::Vec3* __restrict__ out_delta) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= union_count) {
        return;
    }
    const math::Vec3 pi = positions[i];
    const float wi = inv_mass[i];
    const uint32_t base = neighbor_offsets[i];
    const uint32_t n = neighbor_counts[i];

    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        const math::Vec3 r = Sub(pi, positions[j]);  // p_i - p_j
        const float dist = sqrtf(Dot(r, r));
        if (dist <= 0.0f) {
            continue;  // coincident: no contact normal (degenerate; skip).
        }
        const float c = dist - d_min;
        if (c >= 0.0f) {
            continue;  // separating: unilateral contact is inactive.
        }
        if (RestNeighbors(i, j, per_env, d_min, topology_offsets, topology_elements, rest_positions))
            continue;
        const float wj = inv_mass[j];
        const float wsum = wi + wj + alpha_tilde;
        if (wsum <= 0.0f) {
            continue;  // both pinned (w_i = w_j = 0, a~ = 0): no correction.
        }
        const float dl = (-c) / wsum;          // XPBD multiplier (lambda warm 0).
        const float inv_dist = 1.0f / dist;    // n = r/dist; fold into the scale.
        const float scale = wi * dl * inv_dist;  // i's half: + w_i * n * dl.
        dx = __fadd_rn(dx, r.x * scale);
        dy = __fadd_rn(dy, r.y * scale);
        dz = __fadd_rn(dz, r.z * scale);
    }
    out_delta[i] = math::Vec3{dx, dy, dz};
}

// Each particle applies its gathered contact correction without shared writes.
__global__ void PpContactApplyKernel(uint32_t union_count,
                                     math::Vec3* __restrict__ positions,
                                     math::Vec3* __restrict__ projection_delta,
                                     const math::Vec3* __restrict__ delta) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= union_count) {
        return;
    }
    const math::Vec3 d = delta[i];
    if (d.x == 0.0f && d.y == 0.0f && d.z == 0.0f) {
        return;  // no correction: leave the position bit-untouched (inert path).
    }
    ApplyProjectionDelta(positions, projection_delta, i, d);
}

// Particle op entry points validate buffers and launch on the supplied stream.

Status OpParticleAeroDrag(const ModelView& model, const DataView& data,
                          const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const AeroDragParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->tri_count == 0u ||
        (p->drag_normal == 0.0f && p->drag_tangent == 0.0f)) {
        return Status::Ok;  // no aero triangles / drag off: inert.
    }
    const uint32_t blocks = (p->tri_count + kBlockSize - 1u) / kBlockSize;
    if (p->particle_count == 0u) return Status::InvalidArgument;
    LaunchCuda(ClothAeroImpulseKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               p->tri_count, model.aero_tri_verts, model.aero_tri_area,
               data.particle_pos, data.particle_vel, data.particle_inv_mass,
               model.aero_particle_count, data.aero_tri_impulse,
               p->drag_normal, p->drag_tangent, p->max_dv, p->dt);
    if (cudaGetLastError() != cudaSuccess) return Status::Failed;
    LaunchCuda(ClothAeroGatherKernel,
               dim3((p->particle_count + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, p->particle_count,
               model.aero_particle_offset, model.aero_particle_count, model.aero_incident_tri,
               data.aero_tri_impulse, data.particle_inv_mass, data.particle_vel);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpParticlePredict(const ModelView&, const DataView& data,
                         const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ParticlePredictParams*>(params);
    if (p == nullptr) return Status::InvalidArgument;
    if (p->mode == kParticleModeNone || p->mode == kParticleModeMpm || p->particle_count == 0u)
        return Status::Ok;
    const uint32_t per_env = p->particles_per_env != 0u ? p->particles_per_env : p->particle_count;
    const uint32_t active_begin = p->mode == kParticleModeMpmXpbd ? p->n_mpm_particles : 0u;
    if (!(p->dt > 0.0f) || !std::isfinite(p->dt) || p->particle_count % per_env != 0u ||
        active_begin > per_env || data.particle_pos == nullptr || data.particle_prev_pos == nullptr ||
        data.pbf_predicted_pos == nullptr || data.particle_vel == nullptr ||
        data.particle_v_pre == nullptr || data.particle_projection_delta == nullptr ||
        data.particle_inv_mass == nullptr)
        return Status::InvalidArgument;
    const uint32_t blocks = (p->particle_count - 1u) / kBlockSize + 1u;
    LaunchCuda(ParticlePredictKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               p->particle_count, per_env, active_begin, data.particle_pos, data.particle_prev_pos,
               data.pbf_predicted_pos, data.particle_vel, data.particle_v_pre,
               data.particle_projection_delta, data.particle_inv_mass,
               math::Vec3{p->gravity[0], p->gravity[1], p->gravity[2]}, p->dt);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpParticleProjectionVelocity(const ModelView&, const DataView& data,
                                    const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ParticleProjectionVelocityParams*>(params);
    if (p == nullptr) return Status::InvalidArgument;
    if (p->particle_count == 0u) return Status::Ok;
    if (!(p->dt > 0.0f) || !std::isfinite(p->dt) || p->particles_per_env == 0u ||
        p->particle_count % p->particles_per_env != 0u ||
        p->active_begin_per_env > p->particles_per_env ||
        data.particle_vel == nullptr || data.particle_v_pre == nullptr ||
        data.particle_inv_mass == nullptr || data.particle_projection_delta == nullptr ||
        data.pbf_predicted_pos == nullptr || data.particle_prev_pos == nullptr)
        return Status::InvalidArgument;
    if (p->active_begin_per_env == p->particles_per_env) return Status::Ok;
    const uint32_t blocks = (p->particle_count - 1u) / kBlockSize + 1u;
    LaunchCuda(ParticleProjectionVelocityKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               p->particle_count, p->particles_per_env, p->active_begin_per_env,
               data.particle_prev_pos, data.pbf_predicted_pos,
               data.particle_projection_delta, data.particle_inv_mass,
               data.particle_vel, data.particle_v_pre, p->dt);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpParticleContactDelta(const ModelView&, const DataView& data,
                              const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ParticleContactDeltaParams*>(params);
    if (p == nullptr) return Status::InvalidArgument;
    if (p->particle_count == 0u) return Status::Ok;
    if (!(p->dt > 0.0f) || !std::isfinite(p->dt) || p->particles_per_env == 0u ||
        p->particle_count % p->particles_per_env != 0u ||
        p->active_begin_per_env > p->particles_per_env || data.pbf_predicted_pos == nullptr ||
        data.particle_vel == nullptr || data.particle_v_pre == nullptr ||
        data.particle_inv_mass == nullptr || data.particle_prev_pos == nullptr)
        return Status::InvalidArgument;
    if (p->active_begin_per_env == p->particles_per_env) return Status::Ok;
    const uint32_t blocks = (p->particle_count - 1u) / kBlockSize + 1u;
    LaunchCuda(ParticleContactDeltaKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               p->particle_count, p->particles_per_env, p->active_begin_per_env,
               data.particle_prev_pos, data.pbf_predicted_pos, data.particle_vel, data.particle_v_pre,
               data.particle_inv_mass, p->dt);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpXpbdProject(const ModelView& model, const DataView& data,
                     const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const XpbdProjectParams*>(params);
    if (p == nullptr) return Status::InvalidArgument;
    if (p->dist_con_count == 0u && p->bend_con_count == 0u &&
        p->vol_con_count == 0u && p->shape_match_cluster_count == 0u) return Status::Ok;
    const uint32_t iters = p->iters == 0u ? 1u : p->iters;
    const uint32_t env_count = p->env_count == 0u ? 1u : p->env_count;
    if (!(p->dt > 0.0f) || !std::isfinite(p->dt) ||
        data.pbf_predicted_pos == nullptr || data.particle_inv_mass == nullptr ||
        data.particle_projection_delta == nullptr)
        return Status::InvalidArgument;
    if ((p->dist_con_count > 0u &&
         (model.dist_particle_a == nullptr || model.dist_particle_b == nullptr ||
          model.dist_rest_length == nullptr || model.dist_compliance == nullptr ||
          data.dist_lambda == nullptr)) ||
        (p->bend_con_count > 0u &&
         (model.bend_particles == nullptr || model.bend_rest_angle == nullptr ||
          model.bend_compliance == nullptr || data.bend_lambda == nullptr)) ||
        (p->vol_con_count > 0u &&
         (model.vol_particles == nullptr || model.vol_rest_times6 == nullptr ||
          model.vol_compliance == nullptr || data.vol_lambda == nullptr)) ||
        (p->shape_match_cluster_count > 0u &&
         (model.sm_cluster_offset == nullptr || model.sm_cluster_size == nullptr ||
          model.sm_stiffness == nullptr || model.sm_rest_centroid == nullptr ||
          model.sm_particles == nullptr || model.sm_rest_q == nullptr || model.sm_mass == nullptr)))
        return Status::InvalidArgument;
    const auto cooperative = RequireCooperativeLaunch();
    if (cooperative == cudaErrorNotSupported) return Status::Unsupported;
    if (cooperative != cudaSuccess) return Status::Failed;

    // Validate every family and its launch resources before modifying particle state.
    uint32_t dist_blocks = 0u, bend_blocks = 0u, vol_blocks = 0u, sm_blocks = 0u;
    auto status = PrepareXpbdSweep<XpbdDistanceProjector>(
        p->dist_con_count, p->dist_cons_per_env, env_count, p->dist_colors,
        p->dist_color_segments, model.dist_color_segments, &dist_blocks);
    if (status != Status::Ok) return status;
    status = PrepareXpbdSweep<XpbdBendProjector>(
        p->bend_con_count, p->bend_cons_per_env, env_count, p->bend_colors,
        p->bend_color_segments, model.bend_color_segments, &bend_blocks);
    if (status != Status::Ok) return status;
    status = PrepareXpbdSweep<XpbdVolumeProjector>(
        p->vol_con_count, p->vol_cons_per_env, env_count, p->vol_colors,
        p->vol_color_segments, model.vol_color_segments, &vol_blocks);
    if (status != Status::Ok) return status;
    status = PrepareXpbdSweep<XpbdShapeMatchProjector>(
        p->shape_match_cluster_count, p->sm_clusters_per_env, env_count, p->sm_colors,
        p->sm_color_segments, model.sm_color_segments, &sm_blocks);
    if (status != Status::Ok) return status;

    // Each call retains the material-family order and the step's accumulated lambdas.
    status = ProjectXpbdFamily(
        XpbdDistanceProjector{data.pbf_predicted_pos, data.particle_projection_delta, data.particle_inv_mass,
                             model.dist_particle_a, model.dist_particle_b,
                             model.dist_rest_length, model.dist_compliance, data.dist_lambda, p->dt},
        dist_blocks, p->dist_con_count, p->dist_cons_per_env, env_count, p->dist_colors,
        model.dist_color_segments, iters, p->iteration_start, data.dist_lambda, stream);
    if (status != Status::Ok) return status;
    status = ProjectXpbdFamily(
        XpbdBendProjector{data.pbf_predicted_pos, data.particle_projection_delta, data.particle_inv_mass,
                         model.bend_particles, model.bend_rest_angle,
                         model.bend_compliance, data.bend_lambda, p->dt},
        bend_blocks, p->bend_con_count, p->bend_cons_per_env, env_count, p->bend_colors,
        model.bend_color_segments, iters, p->iteration_start, data.bend_lambda, stream);
    if (status != Status::Ok) return status;
    status = ProjectXpbdFamily(
        XpbdVolumeProjector{data.pbf_predicted_pos, data.particle_projection_delta, data.particle_inv_mass,
                           model.vol_particles, model.vol_rest_times6,
                           model.vol_compliance, data.vol_lambda, p->dt},
        vol_blocks, p->vol_con_count, p->vol_cons_per_env, env_count, p->vol_colors,
        model.vol_color_segments, iters, p->iteration_start, data.vol_lambda, stream);
    if (status != Status::Ok) return status;
    return ProjectXpbdFamily(
        XpbdShapeMatchProjector{data.pbf_predicted_pos, data.particle_projection_delta, data.particle_inv_mass,
                               model.sm_cluster_offset, model.sm_cluster_size,
                               model.sm_stiffness, model.sm_rest_centroid,
                               model.sm_particles, model.sm_rest_q, model.sm_mass},
        sm_blocks, p->shape_match_cluster_count, p->sm_clusters_per_env, env_count, p->sm_colors,
        model.sm_color_segments, iters, p->iteration_start, nullptr, stream);
}

// Density projection applies all but the last iteration here.
// PbfApplyDelta consumes the final staged correction.
Status OpPbfDensityLambda(const ModelView& /*model*/, const DataView& data,
                          const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const PbfDensityLambdaParams*>(params);
    if (p == nullptr) return Status::InvalidArgument;
    if (p->particle_count == 0u || p->support_radius <= 0.0f ||
        p->rest_density <= 0.0f) {
        return Status::Ok;  // XPBD-only scene: inert.
    }
    const uint32_t per_env = p->particles_per_env != 0u ? p->particles_per_env : p->particle_count;
    if (!std::isfinite(p->support_radius) || !std::isfinite(p->rest_density) ||
        !(p->particle_mass > 0.0f) || !std::isfinite(p->particle_mass) ||
        p->relaxation < 0.0f || !std::isfinite(p->relaxation) ||
        p->particle_count % per_env != 0u || p->n_soft_particles > per_env ||
        (p->boundary_enabled != 0u && !std::isfinite(p->floor_z)) ||
        data.pbf_predicted_pos == nullptr || data.particle_inv_mass == nullptr ||
        data.pbf_density == nullptr || data.pbf_lambda == nullptr || data.pbf_position_delta == nullptr ||
        data.particle_projection_delta == nullptr ||
        data.grid_neighbor_count == nullptr || data.grid_neighbor_offset == nullptr)
        return Status::InvalidArgument;
    const fl::PbfKernelCoeffs coeffs = fl::MakePbfKernelCoeffs(p->support_radius);
    const float inv_rho0 = 1.0f / p->rest_density;
    const uint32_t iters = p->iters == 0u ? 1u : p->iters;
    const uint32_t N = p->particle_count;
    const uint32_t blocks = (N + kBlockSize - 1u) / kBlockSize;
    const uint32_t n_soft = p->n_soft_particles;
    for (uint32_t it = 0u; it < iters; ++it) {
        LaunchCuda(PbfDensityKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   N, n_soft, per_env, data.pbf_predicted_pos, p->particle_mass,
                   coeffs, data.grid_neighbor_count, data.grid_neighbor_offset, data.grid_neighbor_idx,
                   data.pbf_density);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
        LaunchCuda(PbfLambdaKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   N, n_soft, per_env, data.pbf_predicted_pos, coeffs, inv_rho0,
                   p->rest_density, p->relaxation, p->clamp_overdensity != 0u,
                   data.pbf_density, data.grid_neighbor_count, data.grid_neighbor_offset,
                   data.grid_neighbor_idx, data.pbf_lambda);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
        LaunchCuda(PbfComputeCorrectionKernel, dim3(blocks), dim3(kBlockSize), 0u,
                   stream, N, n_soft, per_env, data.pbf_predicted_pos, coeffs,
                   inv_rho0, data.pbf_lambda, data.grid_neighbor_count, data.grid_neighbor_offset,
                   data.grid_neighbor_idx, data.pbf_position_delta);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
        // Apply in-loop for every iteration EXCEPT the last (PbfApplyDelta runs
        // the last apply, so the two ops together == the legacy NxN loop).
        if (it + 1u < iters) {
            LaunchCuda(PbfApplyCorrectionKernel, dim3(blocks), dim3(kBlockSize), 0u,
                       stream, N, data.pbf_predicted_pos, data.particle_projection_delta, data.pbf_position_delta,
                       data.particle_inv_mass, n_soft, per_env,
                       p->boundary_enabled != 0u, p->floor_z);
            if (cudaGetLastError() != cudaSuccess) return Status::Failed;
        }
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

// PbfApplyDelta: the FINAL correction-apply pass + boundary floor clamp (the last
// iteration's apply of the legacy density-projection loop).
Status OpPbfApplyDelta(const ModelView& /*model*/, const DataView& data,
                       const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const PbfApplyDeltaParams*>(params);
    if (p == nullptr) return Status::InvalidArgument;
    if (p->particle_count == 0u || p->support_radius <= 0.0f) {
        return Status::Ok;  // XPBD-only scene: inert.
    }
    const uint32_t N = p->particle_count;
    const uint32_t per_env = p->particles_per_env != 0u ? p->particles_per_env : N;
    if (!std::isfinite(p->support_radius) || N % per_env != 0u || p->n_soft_particles > per_env ||
        (p->boundary_enabled != 0u && !std::isfinite(p->floor_z)) ||
        data.pbf_predicted_pos == nullptr || data.pbf_position_delta == nullptr || data.particle_inv_mass == nullptr ||
        data.particle_projection_delta == nullptr)
        return Status::InvalidArgument;
    const uint32_t blocks = (N + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(PbfApplyCorrectionKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               N, data.pbf_predicted_pos, data.particle_projection_delta, data.pbf_position_delta,
               data.particle_inv_mass, p->n_soft_particles, per_env,
               p->boundary_enabled != 0u, p->floor_z);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpParticleFinalize(const ModelView&, const DataView& data,
                          const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ParticleFinalizeParams*>(params);
    if (p == nullptr) return Status::InvalidArgument;
    if (p->mode == kParticleModeNone || p->mode == kParticleModeMpm || p->particle_count == 0u)
        return Status::Ok;
    const uint32_t per_env = p->particles_per_env != 0u ? p->particles_per_env : p->particle_count;
    const uint32_t active_begin = p->mode == kParticleModeMpmXpbd ? p->n_mpm_particles : 0u;
    if (!(p->dt > 0.0f) || !std::isfinite(p->dt) || p->particle_count % per_env != 0u ||
        active_begin > per_env || data.particle_pos == nullptr || data.particle_prev_pos == nullptr ||
        data.pbf_predicted_pos == nullptr || data.particle_vel == nullptr ||
        data.particle_v_pre == nullptr || data.particle_inv_mass == nullptr ||
        (p->pos_pass != 0u && data.particle_pseudo_vel == nullptr))
        return Status::InvalidArgument;
    const uint32_t count = p->particle_count;
    const uint32_t blocks = (count - 1u) / kBlockSize + 1u;
    LaunchCuda(ParticleFinalizeKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               count, per_env, active_begin, data.particle_pos,
               data.pbf_predicted_pos, data.particle_vel, data.particle_v_pre,
               p->pos_pass != 0u ? data.particle_pseudo_vel : nullptr,
               data.particle_inv_mass, p->dt);
    if (cudaGetLastError() != cudaSuccess) return Status::Failed;
    const bool has_fluid = p->mode == kParticleModePbf || p->mode == kParticleModeSoftFluid ||
        (p->mode == kParticleModeCoupled && p->coupled_internal == kCoupledInternalPbf);
    if (!has_fluid || !(p->support_radius > 0.0f)) return Status::Ok;
    const uint32_t fluid_begin = p->mode == kParticleModeSoftFluid ? p->n_soft_particles : 0u;
    if (p->xsph_viscosity_c > 0.0f) {
        const fl::PbfKernelCoeffs coeffs = fl::MakePbfKernelCoeffs(p->support_radius);
        LaunchCuda(PbfXsphDeltaKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   count, fluid_begin, per_env, data.particle_pos, data.particle_vel,
                   data.pbf_density, p->particle_mass, p->xsph_viscosity_c, coeffs,
                   data.grid_neighbor_count, data.grid_neighbor_offset,
                   data.grid_neighbor_idx, data.pbf_position_delta);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
        LaunchCuda(PbfApplyVelocityDeltaKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   count, data.particle_vel, data.particle_inv_mass, data.pbf_position_delta);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
    }
    if (p->surface_tension_gamma > 0.0f) {
        const fl::PbfCohesionCoeffs coeffs = fl::MakePbfCohesionCoeffs(p->support_radius);
        LaunchCuda(PbfCohesionKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   count, fluid_begin, per_env, data.particle_pos, data.particle_vel,
                   data.particle_inv_mass, p->particle_mass, p->surface_tension_gamma, p->dt, coeffs,
                   data.grid_neighbor_count, data.grid_neighbor_offset, data.grid_neighbor_idx);
    }
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

// The contact sweep consumes projected positions and the shared neighbor CSR.
// Each Jacobi iteration gathers corrections before applying them at separate indices.
Status OpParticleParticleContact(const ModelView& model, const DataView& data,
                                 const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ParticleParticleContactParams*>(params);
    if (p == nullptr) return Status::Failed;
    // Emitted only for SoftFluid; inert for a zero/negative d_min or no particles.
    if (p->mode != kParticleModeSoftFluid || p->particle_count == 0u ||
        p->contact_distance_d_min <= 0.0f) {
        return Status::Ok;
    }
    const uint32_t N = p->particle_count;
    if (p->particles_per_env == 0u || N % p->particles_per_env != 0u ||
        data.particle_projection_delta == nullptr ||
        (model.particle_topology_offsets != nullptr &&
         (model.particle_topology_elements == nullptr || model.particle_contact_rest_pos == nullptr)))
        return Status::InvalidArgument;
    const uint32_t blocks = (N + kBlockSize - 1u) / kBlockSize;
    const uint32_t iters = p->solver_iterations == 0u ? 1u : p->solver_iterations;
    const float alpha_tilde = p->compliance_alpha;  // a~ at dt=1 (position-based).
    for (uint32_t it = 0u; it < iters; ++it) {
        // Density projection has consumed pbf_position_delta before contact gathers reuse it.
        LaunchCuda(PpContactHalfCorrectionKernel, dim3(blocks), dim3(kBlockSize), 0u,
                   stream, N, p->particles_per_env, data.pbf_predicted_pos, data.particle_inv_mass,
                   model.particle_topology_offsets, model.particle_topology_elements,
                   model.particle_contact_rest_pos,
                   p->contact_distance_d_min, alpha_tilde, data.grid_neighbor_count, data.grid_neighbor_offset,
                   data.grid_neighbor_idx, data.pbf_position_delta);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
        LaunchCuda(PpContactApplyKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   N, data.pbf_predicted_pos, data.particle_projection_delta, data.pbf_position_delta);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkParticleOps() {
    SetCudaOp(NkOp::RefitParticleSurfaces, &OpRefitParticleSurfaces);
    SetCudaOp(NkOp::ParticleAeroDrag, &OpParticleAeroDrag);
    SetCudaOp(NkOp::ParticlePredict, &OpParticlePredict);
    SetCudaOp(NkOp::ParticleProjectionVelocity, &OpParticleProjectionVelocity);
    SetCudaOp(NkOp::ParticleContactDelta, &OpParticleContactDelta);
    SetCudaOp(NkOp::XpbdProject, &OpXpbdProject);
    SetCudaOp(NkOp::PbfDensityLambda, &OpPbfDensityLambda);
    SetCudaOp(NkOp::PbfApplyDelta, &OpPbfApplyDelta);
    SetCudaOp(NkOp::ParticleFinalize, &OpParticleFinalize);
    SetCudaOp(NkOp::ParticleParticleContact, &OpParticleParticleContact);
}

}  // namespace nuka::phi
