#pragma once

#include "phi/backend_cuda/rt/particle_surface.cuh"

namespace nuka::rt::particle_surface_detail {

static __global__ void PublishReconstructedSurfaceKernel(uint32_t envs, uint32_t slots,
    uint32_t mesh_id, uint32_t mesh_count, bool spheres, bool normals, bool colors,
    const uint32_t* counts, const DevPrim* primitives, const math::Vec3* geometry,
    const float* radii, const collision::gpu::LbvhNode* nodes, SensorBlasRef* refs) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= envs) return;
    SensorBlasRef ref;
    const uint32_t count = counts[env];
    if (count > 0u) {
        const uint32_t base = env * slots, total = envs * slots;
        ref.blas_nodes = nodes + env * (2u * slots - 1u);
        ref.blas_leaf_count = slots;
        ref.local_bound = ref.blas_nodes[0].aabb;
        ref.blas.prims = primitives;
        ref.blas.prim_count = count;
        if (spheres) {
            ref.blas.sph_center = geometry + base;
            ref.blas.sph_radius = radii + base;
            if (colors) ref.blas.sph_color = geometry + total + base;
        } else {
            ref.blas.tri_v0 = geometry + base;
            ref.blas.tri_v1 = geometry + total + base;
            ref.blas.tri_v2 = geometry + 2u * total + base;
            if (normals) {
                ref.blas.tri_n0 = geometry + 3u * total + base;
                ref.blas.tri_n1 = geometry + 4u * total + base;
                ref.blas.tri_n2 = geometry + 5u * total + base;
            }
        }
    }
    refs[env * mesh_count + mesh_id] = ref;
}

// Reconstruction uses the shared host mesher; uploads and tracing use the selected device stream.
class ReconstructedSurfaceCache {
public:
    ReconstructedSurfaceCache(const ParticleSurfaceBinding& binding, uint32_t particles_per_env)
        : binding_(binding), particles_per_env_(particles_per_env) {
        using Kind = ParticleSurfaceBinding::Kind;
        if (binding.kind != Kind::Density && binding.kind != Kind::Grains)
            throw std::invalid_argument("invalid reconstructed particle surface kind");
        if (!binding.triangle_particles.empty() || binding.particle_first > particles_per_env)
            throw std::invalid_argument("invalid reconstructed particle range");
        const uint32_t available = particles_per_env - binding.particle_first;
        count_ = binding.particle_count ? binding.particle_count : available;
        if (count_ > available)
            throw std::invalid_argument("reconstructed particle range exceeds the environment field");
        if (binding.kind == Kind::Density) {
            const auto& p = binding.density;
            if (!(p.h > 0.0f) || !std::isfinite(p.h) || !(p.CellSize() > 0.0f) ||
                !std::isfinite(p.CellSize()) || !(p.particle_mass > 0.0f) ||
                !std::isfinite(p.particle_mass) || !(p.rest_density_rho0 > 0.0f) ||
                !std::isfinite(p.rest_density_rho0) || !(p.iso_fraction > 0.0f) ||
                !std::isfinite(p.iso_fraction))
                throw std::invalid_argument("invalid density surface parameters");
        } else {
            const auto& p = binding.grains;
            if (!(p.radius > 0.0f) || !std::isfinite(p.radius) ||
                !std::isfinite(p.radius_jitter) || p.radius_jitter < 0.0f ||
                !std::isfinite(p.tint_jitter) || p.tint_jitter < 0.0f)
                throw std::invalid_argument("invalid grain surface parameters");
        }
    }

    void Update(const RtContext& ctx, ParticlePositionSource source, uint32_t envs,
                SensorBlasRef* refs, uint32_t mesh_count,
                const std::vector<std::vector<runtime::fluid::FluidSurfaceBoundary>>& boundaries) {
        if (!source.positions || envs == 0u || envs > source.env_count ||
            source.particles_per_env != particles_per_env_)
            throw std::invalid_argument("reconstructed particle source is missing environments");
        const size_t pitch = size_t{count_} * sizeof(math::Vec3);
        positions_.resize(size_t{envs} * count_);
        if (count_ > 0u) {
            Check(cudaMemcpy2DAsync(positions_.data(), pitch, source.positions + binding_.particle_first,
                size_t{source.particles_per_env} * sizeof(math::Vec3), pitch, envs,
                cudaMemcpyDeviceToHost, ctx.stream));
            Check(cudaStreamSynchronize(ctx.stream));
        }
        for (const auto& position : positions_) {
            if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z))
                throw std::invalid_argument("non-finite particle position in reconstructed surface");
        }

        const bool density = binding_.kind == ParticleSurfaceBinding::Kind::Density;
        const bool spheres = !density && binding_.grains.round;
        meshes_.resize(envs);
        counts_.resize(envs);
        uint32_t slots = 1u;
        for (uint32_t env = 0u; env < envs; ++env) {
            const auto begin = positions_.begin() + size_t{env} * count_;
            samples_.assign(begin, begin + count_);
            if (density) {
                meshes_[env] = runtime::fluid::MarchFluidSurface(samples_, binding_.density,
                    boundaries.empty() ? std::vector<runtime::fluid::FluidSurfaceBoundary>{} : boundaries[env]);
            } else {
                const auto& p = binding_.grains;
                meshes_[env] = runtime::BakeParticleSpheres(samples_, 0u, count_, p.radius,
                    p.round, p.radius_jitter, p.tint_jitter, binding_.particle_first);
            }
            const auto& mesh = meshes_[env];
            const size_t count = spheres ? mesh.sphere_radii.size() : mesh.indices.size() / 3u;
            if (count > kMaxBlasPrims)
                throw std::length_error("reconstructed surface exceeds the BLAS primitive capacity");
            counts_[env] = static_cast<uint32_t>(count);
            slots = std::max(slots, counts_[env]);
        }
        const uint64_t total = uint64_t{envs} * slots;
        if (total > std::numeric_limits<uint32_t>::max() / 6u)
            throw std::length_error("reconstructed surface exceeds device index capacity");
        geometry_.resize(size_t{total} * (spheres ? 2u : 6u));
        bounds_.resize(total);
        if (spheres) radii_.resize(total);
        for (uint32_t env = 0u; env < envs; ++env) Pack(env, slots, spheres);
        d_geometry_.Upload(ctx, geometry_.data(), geometry_.size() * sizeof(math::Vec3));
        d_bounds_.Upload(ctx, bounds_.data(), bounds_.size() * sizeof(collision::AABB));
        d_counts_.Upload(ctx, counts_.data(), counts_.size() * sizeof(uint32_t));
        if (spheres) d_radii_.Upload(ctx, radii_.data(), radii_.size() * sizeof(float));
        if (slots > primitive_capacity_) {
            std::vector<DevPrim> primitives(slots);
            const auto kind = spheres ? PrimKind::Sphere : PrimKind::Triangle;
            for (uint32_t i = 0u; i < slots; ++i) primitives[i] = {static_cast<uint32_t>(kind), i};
            d_primitives_.Upload(ctx, primitives.data(), primitives.size() * sizeof(DevPrim));
            primitive_capacity_ = slots;
        }
        bvh_.Update(ctx, static_cast<const collision::AABB*>(d_bounds_.Data()), envs, slots, density);
        phi::LaunchCuda(PublishReconstructedSurfaceKernel, dim3((envs + kBlock - 1u) / kBlock),
            dim3(kBlock), 0u, ctx.stream, envs, slots, binding_.mesh_id, mesh_count, spheres,
            density, binding_.grains.tint_jitter > 0.0f, static_cast<const uint32_t*>(d_counts_.Data()),
            static_cast<const DevPrim*>(d_primitives_.Data()), static_cast<const math::Vec3*>(d_geometry_.Data()),
            static_cast<const float*>(d_radii_.Data()), bvh_.Nodes(), refs);
        Check(cudaPeekAtLastError());
    }

private:
    static math::Vec3 VectorAt(const std::vector<float>& values, uint32_t index) {
        const size_t at = size_t{index} * 3u;
        return {values[at], values[at + 1u], values[at + 2u]};
    }

    void Pack(uint32_t env, uint32_t slots, bool spheres) {
        const auto& mesh = meshes_[env];
        const size_t total = meshes_.size() * slots, base = size_t{env} * slots;
        const auto& points = spheres ? mesh.sphere_centers : mesh.positions;
        const math::Vec3 anchor = points.empty() ? math::Vec3{} : VectorAt(points, 0u);
        const bool normals = !mesh.normals.empty(), colors = !mesh.sphere_colors.empty();
        for (uint32_t local = 0u; local < slots; ++local) {
            const size_t i = base + local;
            collision::AABB bound;
            if (local >= counts_[env]) {
                bound.min = anchor;
                bound.max = anchor;
            } else if (spheres) {
                const auto center = VectorAt(mesh.sphere_centers, local);
                const float radius = mesh.sphere_radii[local];
                geometry_[i] = center;
                geometry_[total + i] = colors ? VectorAt(mesh.sphere_colors, local) : math::Vec3{1, 1, 1};
                radii_[i] = radius;
                const math::Vec3 extent{radius, radius, radius};
                bound.min = center - extent;
                bound.max = center + extent;
            } else {
                for (uint32_t corner = 0u; corner < 3u; ++corner) {
                    const uint32_t vertex = mesh.indices[3u * local + corner];
                    const auto point = VectorAt(mesh.positions, vertex);
                    geometry_[corner * total + i] = point;
                    geometry_[(corner + 3u) * total + i] = normals ? VectorAt(mesh.normals, vertex) : math::Vec3{};
                    bound.min.x = std::min(bound.min.x, point.x); bound.max.x = std::max(bound.max.x, point.x);
                    bound.min.y = std::min(bound.min.y, point.y); bound.max.y = std::max(bound.max.y, point.y);
                    bound.min.z = std::min(bound.min.z, point.z); bound.max.z = std::max(bound.max.z, point.z);
                }
            }
            bounds_[i] = bound;
        }
    }

    ParticleSurfaceBinding binding_;
    uint32_t particles_per_env_ = 0u, count_ = 0u, primitive_capacity_ = 0u;
    std::vector<math::Vec3> positions_, samples_, geometry_;
    std::vector<render::MeshGeometry> meshes_;
    std::vector<collision::AABB> bounds_;
    std::vector<uint32_t> counts_;
    std::vector<float> radii_;
    SurfaceBuffer d_geometry_, d_radii_, d_bounds_, d_counts_, d_primitives_;
    SurfaceBvh bvh_;
};

}  // namespace nuka::rt::particle_surface_detail
