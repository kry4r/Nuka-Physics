#pragma once

#include "collision/lbvh_batched.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/rt/prim_id.cuh"
#include "phi/backend_cuda/rt/rt_device_context.cuh"
#include "phi/backend_cuda/rt/sensor_scatter.hpp"
#include "rt/particle_surface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nuka::rt::particle_surface_detail {

inline constexpr uint32_t kBlock = 128u;
inline constexpr uint32_t kRebuildPeriod = 32u;

struct MeshView {
    const uint32_t* particles;
    const uint32_t* triangles;
    const uint32_t* neighbor_offsets;
    const uint32_t* neighbors;
    const uint32_t* incident_offsets;
    const uint32_t* incident_triangles;
    uint32_t vertices;
    uint32_t faces;
};

static __global__ void LoadParticleSurfaceKernel(ParticlePositionSource source, MeshView mesh,
                                                 uint32_t envs, math::Vec3* positions) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= envs * mesh.vertices) return;
    positions[i] = source.positions[uint64_t{i / mesh.vertices} * source.particles_per_env +
                                   mesh.particles[i % mesh.vertices]];
}

static __global__ void SmoothParticleSurfaceKernel(MeshView mesh, uint32_t envs,
    const math::Vec3* positions, float lambda, math::Vec3* next) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= envs * mesh.vertices) return;
    const uint32_t vertex = i % mesh.vertices, base = i - vertex;
    const uint32_t begin = mesh.neighbor_offsets[vertex], end = mesh.neighbor_offsets[vertex + 1u];
    math::Vec3 sum{};
    for (uint32_t n = begin; n < end; ++n) sum += positions[base + mesh.neighbors[n]];
    next[i] = end > begin ? positions[i] +
        (sum * (1.0f / float(end - begin)) - positions[i]) * lambda : positions[i];
}

static __global__ void NormalParticleSurfaceKernel(MeshView mesh, uint32_t envs,
    const math::Vec3* positions, math::Vec3* normals) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= envs * mesh.vertices) return;
    const uint32_t vertex = i % mesh.vertices, base = i - vertex;
    math::Vec3 sum{};
    for (uint32_t n = mesh.incident_offsets[vertex]; n < mesh.incident_offsets[vertex + 1u]; ++n) {
        const auto* t = mesh.triangles + 3u * mesh.incident_triangles[n];
        const auto a = positions[base + t[0]];
        sum += (positions[base + t[1]] - a).Cross(positions[base + t[2]] - a);
    }
    const float length = sqrtf(sum.LengthSq());
    normals[i] = length > 1.0e-12f ? sum / length : math::Vec3{0.0f, 0.0f, 1.0f};
}

static __global__ void GatherParticleSurfaceKernel(MeshView mesh, uint32_t envs,
    const math::Vec3* positions, const math::Vec3* normals, float normal_offset,
    math::Vec3* geometry, collision::AABB* bounds) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = envs * mesh.faces;
    if (i >= total) return;
    const uint32_t face = i % mesh.faces, vertex_base = (i / mesh.faces) * mesh.vertices;
    collision::AABB bound;
    for (uint32_t corner = 0u; corner < 3u; ++corner) {
        const uint32_t vertex = vertex_base + mesh.triangles[3u * face + corner];
        const auto normal = normals[vertex];
        const auto point = positions[vertex] + normal * normal_offset;
        geometry[corner * total + i] = point;
        geometry[(corner + 3u) * total + i] = normal;
        bound.min.x = fminf(bound.min.x, point.x); bound.max.x = fmaxf(bound.max.x, point.x);
        bound.min.y = fminf(bound.min.y, point.y); bound.max.y = fmaxf(bound.max.y, point.y);
        bound.min.z = fminf(bound.min.z, point.z); bound.max.z = fmaxf(bound.max.z, point.z);
    }
    bounds[i] = bound;
}

static __global__ void PublishParticleSurfaceKernel(MeshView mesh, uint32_t envs,
    uint32_t mesh_id, uint32_t mesh_count, const DevPrim* primitives,
    const math::Vec3* geometry,
    const collision::gpu::LbvhNode* nodes, SensorBlasRef* refs) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= envs) return;
    const uint32_t total = envs * mesh.faces, base = env * mesh.faces;
    SensorBlasRef ref;
    ref.blas_nodes = nodes + env * (2u * mesh.faces - 1u);
    ref.blas_leaf_count = mesh.faces;
    ref.local_bound = ref.blas_nodes[0].aabb;
    ref.blas.prims = primitives;
    ref.blas.prim_count = mesh.faces;
    ref.blas.tri_v0 = geometry + base;
    ref.blas.tri_v1 = geometry + total + base;
    ref.blas.tri_v2 = geometry + 2u * total + base;
    ref.blas.tri_n0 = geometry + 3u * total + base;
    ref.blas.tri_n1 = geometry + 4u * total + base;
    ref.blas.tri_n2 = geometry + 5u * total + base;
    refs[env * mesh_count + mesh_id] = ref;
}

inline void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// Immutable adjacency preserves triangle and neighbor accumulation order without atomic sums.
class SurfaceCache {
public:
    SurfaceCache(const ParticleSurfaceBinding& binding, uint32_t particles_per_env,
                 const RtContext& ctx) : binding_(binding) {
        const auto& input = binding.triangle_particles;
        if (input.empty() || input.size() % 3u || input.size() / 3u > kMaxBlasPrims ||
            !std::isfinite(binding.normal_offset) || !std::isfinite(binding.smooth_lambda) ||
            binding.smooth_lambda < 0.0f || binding.smooth_lambda > 1.0f)
            throw std::invalid_argument("invalid particle surface topology or skin");
        auto particles = input;
        std::sort(particles.begin(), particles.end());
        particles.erase(std::unique(particles.begin(), particles.end()), particles.end());
        if (particles.back() >= particles_per_env)
            throw std::invalid_argument("particle surface index exceeds the environment particle field");
        std::vector<uint32_t> triangles;
        triangles.reserve(input.size());
        for (uint32_t particle : input) triangles.push_back(static_cast<uint32_t>(
            std::lower_bound(particles.begin(), particles.end(), particle) - particles.begin()));
        vertices_ = static_cast<uint32_t>(particles.size());
        faces_ = static_cast<uint32_t>(triangles.size() / 3u);
        std::vector<std::vector<uint32_t>> neighbors(vertices_), incident(vertices_);
        for (uint32_t face = 0u; face < faces_; ++face) {
            const auto* t = triangles.data() + 3u * face;
            for (uint32_t corner = 0u; corner < 3u; ++corner) {
                incident[t[corner]].push_back(face);
                const uint32_t a = t[corner], b = t[(corner + 1u) % 3u];
                neighbors[a].push_back(b);
                neighbors[b].push_back(a);
            }
        }
        auto upload_adjacency = [&](const std::vector<std::vector<uint32_t>>& lists,
                                    OwnedBuffer& offsets_buffer, OwnedBuffer& entries_buffer) {
            std::vector<uint32_t> offsets(1u, 0u), entries;
            for (const auto& list : lists) {
                entries.insert(entries.end(), list.begin(), list.end());
                offsets.push_back(static_cast<uint32_t>(entries.size()));
            }
            offsets_buffer = UploadOwned(ctx.device_bt, offsets);
            entries_buffer = UploadOwned(ctx.device_bt, entries);
        };
        particles_ = UploadOwned(ctx.device_bt, particles);
        triangles_ = UploadOwned(ctx.device_bt, triangles);
        upload_adjacency(neighbors, neighbor_offsets_, neighbors_);
        upload_adjacency(incident, incident_offsets_, incident_);
        std::vector<DevPrim> primitives(faces_);
        for (uint32_t i = 0u; i < faces_; ++i)
            primitives[i] = {static_cast<uint32_t>(PrimKind::Triangle), i};
        primitives_ = UploadOwned(ctx.device_bt, primitives);
    }

    void Update(const RtContext& ctx, ParticlePositionSource source, uint32_t envs,
                SensorBlasRef* refs, uint32_t mesh_count) {
        if (!source.positions || envs > source.env_count)
            throw std::invalid_argument("particle surface source is missing environments");
        Allocate(ctx, envs);
        MeshView mesh{static_cast<const uint32_t*>(particles_.Data()),
            static_cast<const uint32_t*>(triangles_.Data()),
            static_cast<const uint32_t*>(neighbor_offsets_.Data()),
            static_cast<const uint32_t*>(neighbors_.Data()),
            static_cast<const uint32_t*>(incident_offsets_.Data()),
            static_cast<const uint32_t*>(incident_.Data()), vertices_, faces_};
        const dim3 vertex_grid((envs * vertices_ + kBlock - 1u) / kBlock);
        auto* positions = static_cast<math::Vec3*>(positions_.Data());
        auto* next = static_cast<math::Vec3*>(next_.Data());
        auto* normals = static_cast<math::Vec3*>(normals_.Data());
        auto* geometry = static_cast<math::Vec3*>(geometry_.Data());
        auto* bounds = static_cast<collision::AABB*>(bounds_.Data());
        auto* nodes = static_cast<collision::gpu::LbvhNode*>(nodes_.Data());
        phi::LaunchCuda(LoadParticleSurfaceKernel, vertex_grid, dim3(kBlock), 0u,
                        ctx.stream, source, mesh, envs, positions);
        if (binding_.smooth_lambda > 0.0f) for (uint32_t pass = 0u; pass < binding_.smooth_iters; ++pass) {
            phi::LaunchCuda(SmoothParticleSurfaceKernel, vertex_grid, dim3(kBlock), 0u,
                            ctx.stream, mesh, envs, positions, binding_.smooth_lambda, next);
            std::swap(positions, next);
        }
        phi::LaunchCuda(NormalParticleSurfaceKernel, vertex_grid, dim3(kBlock), 0u,
                        ctx.stream, mesh, envs, positions, normals);
        phi::LaunchCuda(GatherParticleSurfaceKernel, dim3((envs * faces_ + kBlock - 1u) / kBlock),
            dim3(kBlock), 0u, ctx.stream, mesh, envs, positions, normals,
            binding_.normal_offset, geometry, bounds);
        const bool rebuild = active_envs_ != envs || updates_ >= kRebuildPeriod;
        if (rebuild) {
            Check(collision::gpu::BuildLbvhBatchedNodes(ctx.stream, ctx.device_id, bounds,
                envs, faces_, nodes, static_cast<uint32_t*>(morton_.Data()),
                static_cast<uint32_t*>(indices_.Data()), static_cast<uint64_t*>(keys_.Data()),
                static_cast<uint32_t*>(visits_.Data()), workspace_.Data(), workspace_bytes_));
        } else {
            collision::gpu::RefitLbvhBatched(ctx.stream, ctx.device_id, nodes, bounds,
                envs, faces_, static_cast<uint32_t*>(visits_.Data()));
        }
        active_envs_ = envs;
        updates_ = rebuild ? 0u : updates_ + 1u;
        phi::LaunchCuda(PublishParticleSurfaceKernel, dim3((envs + kBlock - 1u) / kBlock),
            dim3(kBlock), 0u, ctx.stream, mesh, envs, binding_.mesh_id, mesh_count,
            static_cast<const DevPrim*>(primitives_.Data()), geometry, nodes, refs);
        Check(cudaPeekAtLastError());
    }

private:
    void Allocate(const RtContext& ctx, uint32_t envs) {
        if (envs <= capacity_envs_) return;
        const uint64_t vertices = uint64_t{envs} * vertices_, faces = uint64_t{envs} * faces_;
        if (vertices > std::numeric_limits<uint32_t>::max() - kBlock ||
            faces > std::numeric_limits<uint32_t>::max() / 6u)
            throw std::invalid_argument("particle surface exceeds device index capacity");
        Check(collision::gpu::QueryLbvhWorkspaceBytes(envs, faces_, &workspace_bytes_));
        positions_ = OwnedBuffer(ctx.device_bt, vertices * sizeof(math::Vec3));
        next_ = OwnedBuffer(ctx.device_bt, vertices * sizeof(math::Vec3));
        normals_ = OwnedBuffer(ctx.device_bt, vertices * sizeof(math::Vec3));
        geometry_ = OwnedBuffer(ctx.device_bt, faces * 6u * sizeof(math::Vec3));
        bounds_ = OwnedBuffer(ctx.device_bt, faces * sizeof(collision::AABB));
        nodes_ = OwnedBuffer(ctx.device_bt, uint64_t{envs} * (2u * faces_ - 1u) * sizeof(collision::gpu::LbvhNode));
        morton_ = OwnedBuffer(ctx.device_bt, faces * sizeof(uint32_t));
        indices_ = OwnedBuffer(ctx.device_bt, faces * sizeof(uint32_t));
        keys_ = OwnedBuffer(ctx.device_bt, faces * sizeof(uint64_t));
        visits_ = OwnedBuffer(ctx.device_bt, faces * sizeof(uint32_t));
        if (workspace_bytes_) workspace_ = OwnedBuffer(ctx.device_bt, workspace_bytes_);
        capacity_envs_ = envs;
        active_envs_ = 0u;
    }

    ParticleSurfaceBinding binding_;
    uint32_t vertices_ = 0u, faces_ = 0u, capacity_envs_ = 0u, active_envs_ = 0u, updates_ = 0u;
    size_t workspace_bytes_ = 0u;
    OwnedBuffer particles_, triangles_, primitives_, neighbor_offsets_, neighbors_, incident_offsets_, incident_;
    OwnedBuffer positions_, next_, normals_, geometry_, bounds_, nodes_, morton_, indices_, keys_, visits_, workspace_;
};

}  // namespace nuka::rt::particle_surface_detail
