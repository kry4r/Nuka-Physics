#pragma once

#include "math/cuda_vec_ops.cuh"
#include "nk/model/generated/views.hpp"

namespace nuka::phi {
namespace {

constexpr uint32_t kContactSourceSphereCenter = 1u << 4u;

struct ContactPointRef {
    uint32_t owner = 0u;
    uint32_t vertex = 0u;
    bool particle = false;
    bool center = false;
};

__device__ math::Vec3 ContactFeaturePoint(const ModelView& model, const DataView& data,
    uint32_t env, uint32_t bodies_per_env, ContactPointRef ref) {
    if (ref.particle) return data.particle_pos[ref.owner];
    if (ref.center) return data.body_pose[env * bodies_per_env + ref.owner].position;
    const auto info = model.mesh_surface_info[ref.owner];
    const size_t at = (size_t{info.vertex_offset} + ref.vertex) * 3u;
    const math::Vec3 local{model.hull_verts[at], model.hull_verts[at + 1u], model.hull_verts[at + 2u]};
    const auto pose = data.body_pose[env * bodies_per_env + ref.owner];
    return pose.position + math::gpu::RotateByQuatNormalized(pose.rotation, local);
}

template <class Layout>
__device__ bool ContactFeatureRefs(const Layout& p, ModelView model, DataView data,
                              uint32_t env, uint32_t slot, ContactPointRef* a,
                              uint32_t* count_a, ContactPointRef* b, uint32_t* count_b) {
    const uint32_t kind = data.dat_pair_kind[slot];
    const bool particle_a = (kind & 4u) != 0u;
    const bool particle_b = (kind & 8u) != 0u;
    const uint32_t owner_a = data.dat_pair_owner_a[slot];
    const uint32_t owner_b = data.dat_pair_owner_b[slot];
    const uint32_t feature_a = data.dat_pair_feature_a[slot];
    const uint32_t feature_b = data.dat_pair_feature_b[slot];
    if ((kind & 3u) == 1u) {
        *count_a = 1u;
        *count_b = 3u;
        if (particle_a) {
            if (owner_a < env * p.particles_per_env ||
                owner_a >= (env + 1u) * p.particles_per_env) return false;
            a[0] = {owner_a, 0u, true};
        } else {
            const bool center = (kind & kContactSourceSphereCenter) != 0u;
            if (owner_a >= p.bodies_per_env || (!center &&
                feature_a >= model.mesh_surface_info[owner_a].vertex_count)) return false;
            a[0] = {owner_a, feature_a, false, center};
        }
        if (particle_b) {
            if (owner_b >= p.surfaces_per_env) return false;
            const auto info = model.particle_surface_info[owner_b];
            if (feature_b >= info.triangle_count) return false;
            const size_t at = (size_t{info.triangle_offset} + feature_b) * 3u;
            for (uint32_t j = 0u; j < 3u; ++j) {
                const uint32_t vertex = model.particle_surface_triangles[at + j];
                if (vertex >= info.vertex_count) return false;
                b[j] = {env * p.particles_per_env + info.vertex_offset + vertex, 0u, true};
            }
        } else {
            if (owner_b >= p.bodies_per_env) return false;
            const auto info = model.mesh_surface_info[owner_b];
            if (feature_b >= info.triangle_count) return false;
            const size_t at = (size_t{info.triangle_offset} + feature_b) * 3u;
            for (uint32_t j = 0u; j < 3u; ++j) {
                const uint32_t vertex = model.mesh_triangles[at + j];
                if (vertex >= info.vertex_count) return false;
                b[j] = {owner_b, vertex, false};
            }
        }
        return true;
    }
    if ((kind & 3u) != 2u || particle_b) return false;
    *count_a = *count_b = 2u;
    if (particle_a) {
        if (owner_a >= p.surfaces_per_env) return false;
        const auto info = model.particle_surface_info[owner_a];
        const auto edges = model.particle_surface_edge_info[owner_a];
        if (feature_a >= edges.edge_count) return false;
        const auto edge = model.particle_surface_edges[edges.edge_offset + feature_a];
        if (edge.vertex0 >= info.vertex_count || edge.vertex1 >= info.vertex_count) return false;
        a[0] = {env * p.particles_per_env + info.vertex_offset + edge.vertex0, 0u, true};
        a[1] = {env * p.particles_per_env + info.vertex_offset + edge.vertex1, 0u, true};
    } else {
        if (owner_a >= p.bodies_per_env) return false;
        const auto info = model.mesh_surface_info[owner_a];
        const auto edges = model.mesh_edge_info[owner_a];
        if (feature_a >= edges.edge_count) return false;
        const auto edge = model.mesh_edges[edges.edge_offset + feature_a];
        if (edge.vertex0 >= info.vertex_count || edge.vertex1 >= info.vertex_count) return false;
        a[0] = {owner_a, edge.vertex0, false};
        a[1] = {owner_a, edge.vertex1, false};
    }
    if (owner_b >= p.bodies_per_env) return false;
    const auto info = model.mesh_surface_info[owner_b];
    const auto edges = model.mesh_edge_info[owner_b];
    if (feature_b >= edges.edge_count) return false;
    const auto edge = model.mesh_edges[edges.edge_offset + feature_b];
    if (edge.vertex0 >= info.vertex_count || edge.vertex1 >= info.vertex_count) return false;
    b[0] = {owner_b, edge.vertex0, false};
    b[1] = {owner_b, edge.vertex1, false};
    return true;
}

}
}  // namespace nuka::phi
