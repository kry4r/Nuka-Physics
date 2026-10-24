#pragma once

#include <cstdint>

#include "collision/ogc_geometry.hpp"
#include "nk/model/generated/views.hpp"

namespace nuka::phi {

// A particle surface's corners and edges in one env, which follow the runtime vertex splits of a fracturing cloth
// when it carries them, else the cooked topology.
__device__ inline const uint32_t* ParticleSurfaceTriangles(const ModelView& model, const DataView& data, uint32_t env,
                                                           uint32_t triangles_per_env) {
    return data.particle_surface_corners != nullptr
        ? data.particle_surface_corners + size_t{env} * triangles_per_env * 3u : model.particle_surface_triangles;
}

__device__ inline const collision::MeshEdge* ParticleSurfaceEdges(const ModelView& model, const DataView& data,
                                                                  uint32_t env, uint32_t edges_per_env) {
    return data.particle_surface_live_edges != nullptr
        ? data.particle_surface_live_edges + size_t{env} * edges_per_env : model.particle_surface_edges;
}

// The particle each env particle was split from, itself unless split; nullptr without fracture.
__device__ inline const uint32_t* ParticleRoots(const DataView& data, uint32_t env, uint32_t particles_per_env) {
    return data.particle_root != nullptr ? data.particle_root + size_t{env} * particles_per_env : nullptr;
}

__device__ inline collision::MeshVertexIncidence ParticleTriangleIncidence(const ModelView& model,
                                                                          const DataView& data, uint32_t env,
                                                                          uint32_t particles_per_env) {
    return {model.particle_surface_vertex_triangle_offsets, model.particle_surface_vertex_triangles,
            ParticleRoots(data, env, particles_per_env)};
}

__device__ inline collision::MeshVertexIncidence ParticleEdgeIncidence(const ModelView& model, const DataView& data,
                                                                      uint32_t env, uint32_t particles_per_env) {
    return {model.particle_surface_vertex_edge_offsets, model.particle_surface_vertex_edges,
            ParticleRoots(data, env, particles_per_env)};
}

// A split particle keeps the rest position of the particle it was split from.
__device__ inline math::Vec3 ParticleRest(const ModelView& model, const DataView& data, uint32_t env,
                                          uint32_t particles_per_env, uint32_t particle) {
    return model.particle_rest_pos[data.particle_root != nullptr
        ? data.particle_root[size_t{env} * particles_per_env + particle] : particle];
}

}  // namespace nuka::phi
