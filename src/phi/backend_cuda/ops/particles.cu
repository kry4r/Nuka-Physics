
// Particle integration, aero drag and surface refits share arena storage.

#include <cooperative_groups.h>
#include <cuda/atomic>
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

namespace nuka::phi {

namespace {

namespace mg = ::nuka::math::gpu;
using mg::Add;
using mg::Cross;
using mg::Dot;
using mg::Length;
using mg::Scale;
using mg::Sub;

constexpr uint32_t kBlockSize = 128u;

// Internal nodes hold this triangle id until both children are final; escape 0 marks invalid geometry.
constexpr uint32_t kRefitPending = ~1u;

__device__ inline bool SurfaceOfNode(const ModelView& model, const ParticleSurfacesParams& p,
                                     uint32_t node, collision::MeshSurfaceInfo* info) {
    for (uint32_t surface = 0u; surface < p.surfaces_per_env; ++surface) {
        *info = model.particle_surface_info[surface];
        if (node >= info->node_offset && node - info->node_offset < info->node_count) return true;
    }
    return false;
}

__device__ inline bool SurfaceOfEdgeNode(const ModelView& model, const ParticleSurfacesParams& p,
                                        uint32_t node, collision::MeshEdgeInfo* info) {
    for (uint32_t surface = 0u; surface < p.surfaces_per_env; ++surface) {
        *info = model.particle_surface_edge_info[surface];
        if (node >= info->node_offset && node - info->node_offset < info->node_count) return true;
    }
    return false;
}

// Contact geometry and impulse arms share the positions that own momentum.
// Leaves refit grid-wide; each internal node then waits only for its two preorder children.
__global__ void RefitParticleSurfacesKernel(ParticleSurfacesParams p, ModelView model, DataView data) {
    const auto grid = cooperative_groups::this_grid();
    const uint32_t lane = threadIdx.x;
    __shared__ float warp_speed[kBlockSize / 32u];
    for (uint32_t task = blockIdx.x;
         task < p.env_count * p.surfaces_per_env; task += gridDim.x) {
        const uint32_t env = task / p.surfaces_per_env;
        const auto info = model.particle_surface_info[task % p.surfaces_per_env];
        float speed_squared = 0.0f;
        for (uint32_t i = lane; i < info.vertex_count; i += blockDim.x) {
            const uint32_t particle = env * p.particles_per_env + info.vertex_offset + i;
            const math::Vec3 motion = p.position_source == 0u ? data.particle_vel[particle]
                : data.particle_pos[particle] - data.particle_prev_pos[particle];
            speed_squared = fmaxf(speed_squared, motion.LengthSq());
            if (!isfinite(motion.x) || !isfinite(motion.y) || !isfinite(motion.z))
                speed_squared = INFINITY;
        }
        for (uint32_t offset = warpSize / 2u; offset > 0u; offset /= 2u)
            speed_squared = fmaxf(speed_squared, __shfl_down_sync(0xffffffffu, speed_squared, offset));
        if (lane % warpSize == 0u) warp_speed[lane / warpSize] = speed_squared;
        __syncthreads();
        if (lane == 0u) {
            for (uint32_t i = 0u; i < kBlockSize / 32u; ++i)
                speed_squared = fmaxf(speed_squared, warp_speed[i]);
            if (p.position_source == 0u)
                data.particle_surface_max_speed[task] = sqrtf(speed_squared);
            else
                data.dat_surface_motion[task] = sqrtf(speed_squared);
        }
        __syncthreads();
    }
    const uint32_t total = p.env_count * p.nodes_per_env;
    const uint32_t edge_total = p.env_count * p.edge_nodes_per_env;
    const uint32_t items = total + edge_total;
    const uint32_t stride = gridDim.x * blockDim.x;
    const uint32_t first = blockIdx.x * blockDim.x + lane;
    const auto triangle_leaf = [&](uint32_t item) {
        const uint32_t env = item / p.nodes_per_env;
        collision::MeshSurfaceInfo info;
        if (!SurfaceOfNode(model, p, item % p.nodes_per_env, &info)) return;
        auto node = model.particle_surface_tree[item % p.nodes_per_env];
        if (node.triangle == ~0u) {
            node.triangle = kRefitPending;
        } else {
            const math::Vec3* positions = p.position_source == 0u
                ? data.particle_pos : data.particle_prev_pos;
            const collision::MeshSurfaceView view{
                reinterpret_cast<const float*>(positions + size_t{env} * p.particles_per_env),
                model.particle_surface_triangles, model.particle_surface_tree,
                {p.particles_per_env, p.triangles_per_env, p.nodes_per_env}};
            math::Vec3 a, b, c;
            if (!collision::MeshSurfaceTriangle(view, info, node.triangle, a, b, c) ||
                !isfinite(a.LengthSq()) || !isfinite(b.LengthSq()) || !isfinite(c.LengthSq())) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                node.escape = 0u;
            } else {
                node.lower = {fminf(a.x, fminf(b.x, c.x)), fminf(a.y, fminf(b.y, c.y)), fminf(a.z, fminf(b.z, c.z))};
                node.upper = {fmaxf(a.x, fmaxf(b.x, c.x)), fmaxf(a.y, fmaxf(b.y, c.y)), fmaxf(a.z, fmaxf(b.z, c.z))};
            }
        }
        data.particle_surface_nodes[item] = node;
    };
    const auto triangle_internal = [&](uint32_t item) {
        const uint32_t env = item / p.nodes_per_env;
        collision::MeshSurfaceInfo info;
        if (!SurfaceOfNode(model, p, item % p.nodes_per_env, &info)) return;
        auto node = model.particle_surface_tree[item % p.nodes_per_env];
        if (node.triangle != ~0u) return;
        auto* nodes = data.particle_surface_nodes + size_t{env} * p.nodes_per_env + info.node_offset;
        const uint32_t local = item % p.nodes_per_env - info.node_offset;
        const uint32_t left_index = local + 1u;
        const uint32_t right_index = left_index < info.node_count
            ? model.particle_surface_tree[info.node_offset + left_index].escape : 0u;
        if (right_index <= left_index || right_index >= info.node_count) {
            node.escape = 0u;
        } else {
            const uint32_t children[2] = {left_index, right_index};
            for (const uint32_t child : children) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> state(nodes[child].triangle);
                while (state.load(cuda::memory_order_acquire) == kRefitPending) {}
            }
            const auto left = nodes[left_index];
            const auto right = nodes[right_index];
            node.lower = {fminf(left.lower.x, right.lower.x), fminf(left.lower.y, right.lower.y),
                          fminf(left.lower.z, right.lower.z)};
            node.upper = {fmaxf(left.upper.x, right.upper.x), fmaxf(left.upper.y, right.upper.y),
                          fmaxf(left.upper.z, right.upper.z)};
            if (left.escape == 0u || right.escape == 0u) node.escape = 0u;
        }
        nodes[local].lower = node.lower;
        nodes[local].upper = node.upper;
        nodes[local].escape = node.escape;
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(nodes[local].triangle)
            .store(~0u, cuda::memory_order_release);
    };
    const auto edge_leaf = [&](uint32_t item) {
        const uint32_t env = item / p.edge_nodes_per_env;
        collision::MeshEdgeInfo info;
        if (!SurfaceOfEdgeNode(model, p, item % p.edge_nodes_per_env, &info)) return;
        auto node = model.particle_surface_edge_tree[item % p.edge_nodes_per_env];
        if (node.triangle == ~0u) {
            node.triangle = kRefitPending;
        } else {
            const auto edge = model.particle_surface_edges[info.edge_offset + node.triangle];
            const math::Vec3* positions = p.position_source == 0u
                ? data.particle_pos : data.particle_prev_pos;
            const math::Vec3 a = positions[size_t{env} * p.particles_per_env +
                                                   info.vertex_offset + edge.vertex0];
            const math::Vec3 b = positions[size_t{env} * p.particles_per_env +
                                                   info.vertex_offset + edge.vertex1];
            if (!isfinite(a.LengthSq()) || !isfinite(b.LengthSq())) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                node.escape = 0u;
            } else {
                node.lower = {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
                node.upper = {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
            }
        }
        data.particle_surface_edge_nodes[item] = node;
    };
    const auto edge_internal = [&](uint32_t item) {
        const uint32_t env = item / p.edge_nodes_per_env;
        collision::MeshEdgeInfo info;
        if (!SurfaceOfEdgeNode(model, p, item % p.edge_nodes_per_env, &info)) return;
        auto node = model.particle_surface_edge_tree[item % p.edge_nodes_per_env];
        if (node.triangle != ~0u) return;
        auto* nodes = data.particle_surface_edge_nodes +
            size_t{env} * p.edge_nodes_per_env + info.node_offset;
        const uint32_t local = item % p.edge_nodes_per_env - info.node_offset;
        const uint32_t left_index = local + 1u;
        const uint32_t right_index = left_index < info.node_count
            ? model.particle_surface_edge_tree[info.node_offset + left_index].escape : 0u;
        if (right_index <= left_index || right_index >= info.node_count) {
            node.escape = 0u;
        } else {
            const uint32_t children[2] = {left_index, right_index};
            for (const uint32_t child : children) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> state(nodes[child].triangle);
                while (state.load(cuda::memory_order_acquire) == kRefitPending) {}
            }
            const auto left = nodes[left_index], right = nodes[right_index];
            node.lower = {fminf(left.lower.x, right.lower.x), fminf(left.lower.y, right.lower.y),
                          fminf(left.lower.z, right.lower.z)};
            node.upper = {fmaxf(left.upper.x, right.upper.x), fmaxf(left.upper.y, right.upper.y),
                          fmaxf(left.upper.z, right.upper.z)};
            if (left.escape == 0u || right.escape == 0u) node.escape = 0u;
        }
        nodes[local].lower = node.lower;
        nodes[local].upper = node.upper;
        nodes[local].escape = node.escape;
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(nodes[local].triangle)
            .store(~0u, cuda::memory_order_release);
    };
    for (uint32_t item = first; item < items; item += stride) {
        if (item < total) triangle_leaf(item);
        else edge_leaf(item - total);
    }
    grid.sync();
    // Descending order always leaves the highest pending node with final children, so waits progress;
    // the trees share one pass because a node's children follow it within its own tree.
    for (uint32_t item = items - 1u - first; item < items; item -= stride) {
        if (item < total) triangle_internal(item);
        else edge_internal(item - total);
    }
}

Status OpRefitParticleSurfaces(const ModelView& model, const DataView& data,
                              const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ParticleSurfacesParams*>(params);
    if (p == nullptr) return Status::InvalidArgument;
    if (p->surfaces_per_env == 0u || p->env_count == 0u) return Status::Ok;
    if (!model.particle_surface_info ||
        (p->nodes_per_env > 0u && (!model.particle_surface_tree ||
                                   !model.particle_surface_triangles ||
                                   !data.particle_surface_nodes)) ||
        !(p->position_source == 0u ? data.particle_pos : data.particle_prev_pos) ||
        !data.particle_vel || !data.particle_surface_max_speed ||
        (p->position_source != 0u && (!data.particle_pos || !data.dat_surface_motion)) ||
        !model.particle_surface_edge_info ||
        (p->edge_nodes_per_env > 0u && (!model.particle_surface_edges ||
            !model.particle_surface_edge_tree || !data.particle_surface_edge_nodes)))
        return Status::InvalidArgument;
    const uint64_t nodes = uint64_t{p->env_count} * p->nodes_per_env;
    const uint64_t edge_nodes = uint64_t{p->env_count} * p->edge_nodes_per_env;
    if (nodes + edge_nodes > std::numeric_limits<uint32_t>::max()) return Status::InvalidArgument;
    const uint32_t bound = std::max(p->env_count * p->surfaces_per_env,
        static_cast<uint32_t>((nodes + edge_nodes + kBlockSize - 1u) / kBlockSize));
    uint32_t blocks = 0u;
    if (RequireCooperativeLaunch() != cudaSuccess ||
        ResidentGridSize(RefitParticleSurfacesKernel, kBlockSize, 0u, bound, &blocks) != cudaSuccess)
        return Status::Failed;
    return LaunchCooperativeCuda(RefitParticleSurfacesKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                                 *p, model, data) == cudaSuccess ? Status::Ok : Status::Failed;
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
// Vertex blocks [block_begin, block_end) integrate at second order in ClothPredict.
__global__ void ParticlePredictKernel(
    uint32_t count, uint32_t per_env, uint32_t active_begin, uint32_t block_begin,
    uint32_t block_end,
    math::Vec3* __restrict__ positions, math::Vec3* __restrict__ previous,
    math::Vec3* __restrict__ predicted, math::Vec3* __restrict__ velocities,
    math::Vec3* __restrict__ contact_reference, math::Vec3* __restrict__ projection_delta,
    const float* __restrict__ inv_mass,
    const math::Vec3* __restrict__ kinematic_target,
    math::Vec3 gravity, float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || i % per_env < active_begin) return;
    if (i % per_env >= block_begin && i % per_env < block_end) return;
    const math::Vec3 start = positions[i];
    previous[i] = start;
    if (!(inv_mass[i] > 0.0f)) {
        const math::Vec3 target = kinematic_target[i];
        const math::Vec3 velocity = (target - start) / dt;
        predicted[i] = target;
        velocities[i] = contact_reference[i] = velocity;
        projection_delta[i] = {};
        return;
    }
    math::Vec3 velocity = velocities[i];
    velocity = Add(velocity, Scale(gravity, dt));
    velocities[i] = velocity;
    contact_reference[i] = velocity;
    predicted[i] = Add(start, Scale(velocity, dt));
    projection_delta[i] = {};
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
    const float* __restrict__ inv_mass,
    const math::Vec3* __restrict__ kinematic_target, float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || i % per_env < active_begin) return;
    if (!(inv_mass[i] > 0.0f)) {
        positions[i] = kinematic_target[i];
        return;
    }
    const math::Vec3 delta = Sub(velocities[i], contact_reference[i]);
    const math::Vec3 correction = pseudo != nullptr ? Add(delta, pseudo[i]) : delta;
    positions[i] = Add(projected[i], Scale(correction, dt));
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
        data.particle_inv_mass == nullptr || data.particle_kinematic_target == nullptr)
        return Status::InvalidArgument;
    const uint32_t blocks = (p->particle_count - 1u) / kBlockSize + 1u;
    const uint32_t block_begin = p->vertex_blocks.begin;
    const uint32_t block_end = block_begin + p->vertex_blocks.vertices;
    if (block_end > per_env) return Status::InvalidArgument;
    LaunchCuda(ParticlePredictKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               p->particle_count, per_env, active_begin, block_begin, block_end,
               data.particle_pos, data.particle_prev_pos,
               data.pbf_predicted_pos, data.particle_vel, data.particle_v_pre,
               data.particle_projection_delta, data.particle_inv_mass,
               data.particle_kinematic_target,
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
        data.particle_kinematic_target == nullptr ||
        (p->pos_pass != 0u && data.particle_pseudo_vel == nullptr))
        return Status::InvalidArgument;
    const uint32_t count = p->particle_count;
    const uint32_t blocks = (count - 1u) / kBlockSize + 1u;
    LaunchCuda(ParticleFinalizeKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               count, per_env, active_begin, data.particle_pos,
               data.pbf_predicted_pos, data.particle_vel, data.particle_v_pre,
               p->pos_pass != 0u ? data.particle_pseudo_vel : nullptr,
               data.particle_inv_mass, data.particle_kinematic_target, p->dt);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkParticleOps() {
    SetCudaOp(NkOp::RefitParticleSurfaces, &OpRefitParticleSurfaces);
    SetCudaOp(NkOp::ParticleAeroDrag, &OpParticleAeroDrag);
    SetCudaOp(NkOp::ParticlePredict, &OpParticlePredict);
    SetCudaOp(NkOp::ParticleProjectionVelocity, &OpParticleProjectionVelocity);
    SetCudaOp(NkOp::ParticleContactDelta, &OpParticleContactDelta);
    SetCudaOp(NkOp::ParticleFinalize, &OpParticleFinalize);
}

}  // namespace nuka::phi
