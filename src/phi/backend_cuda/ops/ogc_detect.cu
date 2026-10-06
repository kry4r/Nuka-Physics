#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "collision/dat_geometry.hpp"
#include "nk/contact/contact_identity.hpp"
#include "nk/model/generated/views.hpp"
#include "nk/solve/point_endpoint.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"
#include "phi/backend_cuda/ops/ogc_mesh.cuh"
#include "phi/backend_cuda/ops/bvh_work_share.cuh"
#include "phi/backend_cuda/ops/ogc_order.cuh"

namespace nuka::phi {
namespace {

constexpr uint32_t kBlockSize = 128u;

__device__ bool VertexInTriangle(const collision::MeshSurfaceView& view,
                                 const collision::MeshSurfaceInfo& info,
                                 uint32_t vertex, uint32_t triangle) {
    if (triangle >= info.triangle_count) return false;
    const size_t at = (size_t{info.triangle_offset} + triangle) * 3u;
    return view.triangles[at] == vertex || view.triangles[at + 1u] == vertex ||
           view.triangles[at + 2u] == vertex;
}

__global__ void ClearOgcSlotsKernel(OgcDetectParams p, DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = p.env_count * p.slot_capacity;
    if (item >= total) return;
    const uint32_t env = item / p.slot_capacity;
    const uint32_t ordinal = item % p.slot_capacity;
    const size_t slot = size_t{env} * p.slot_stride + p.slot_base + ordinal;
    data.ucontact_count[slot] = 0u;
    data.dat_pair_kind[slot] = 0u;
    if (ordinal == 0u) {
        data.ogc_contact_count[env] = 0u;
        if (p.slot_base == 0u) data.contact_count[env] = 0u;
    }
}

// A source primitive and one target leaf its traversal reached, tested by any free lane of the warp.
struct OgcCandidate {
    uint32_t item;
    uint32_t source_surface;
    uint32_t target_surface;
    uint32_t target;
    float query;
};

__device__ float OgcParticleQuery(const OgcDetectParams& p, const ModelView& model,
                                  const DataView& data, uint32_t env, uint32_t source_surface,
                                  uint32_t target_surface) {
    const float radius = model.particle_surface_thickness[source_surface] +
                         model.particle_surface_thickness[target_surface];
    const float source_speed = data.particle_surface_max_speed[
        size_t{env} * p.surfaces_per_env + source_surface];
    const float target_speed = data.particle_surface_max_speed[
        size_t{env} * p.surfaces_per_env + target_surface];
    return collision::DatQueryRadius(radius + p.margin + p.dt * (source_speed + target_speed));
}

__device__ void OgcEmitParticleFace(const OgcDetectParams& p, const ModelView& model,
                                    const DataView& data, const OgcCandidate& c) {
    const uint32_t item = c.item;
    const uint32_t env = item / p.particles_per_env;
    const uint32_t source_surface = c.source_surface, target_surface = c.target_surface;
    const auto view = OgcParticleView(p, model, data, env);
    const auto target_info = model.particle_surface_info[target_surface];
    const float radius = model.particle_surface_thickness[source_surface] +
                         model.particle_surface_thickness[target_surface];
    const math::Vec3 source = data.particle_pos[item];
    const float query = c.query;
    const uint32_t triangle = c.target;
    const auto feature = collision::OgcFacetFeature(view, target_info,
        model.particle_surface_edges, model.particle_surface_triangle_edges,
        model.particle_surface_edge_info[target_surface],
        {model.particle_surface_vertex_triangle_offsets,
         model.particle_surface_vertex_triangles},
        triangle, source, query);
    if (!(feature.feasible && feature.owner_triangle == triangle &&
          feature.distance <= query && feature.distance > 0.0f)) return;
    const uint32_t ordinal = OgcClaimSlot(p, data, env);
    if (ordinal >= p.slot_capacity) {
        atomicOr(data.env_status + env, kEnvStatusPairOverflow);
        return;
    }
    const uint32_t slot = env * p.slot_stride + p.slot_base + ordinal;
    const size_t at = size_t{slot} * nk::kPairDrivenPtsPerSlot;
    const uint32_t endpoint = env * p.point_endpoints_per_env + p.point_endpoint_first +
                              ordinal * 2u + 1u;
    const uint32_t first = env * p.point_endpoint_terms_per_env + p.point_endpoint_term_first +
                           ordinal * 6u + 3u;
    const size_t tri_at = (size_t{target_info.triangle_offset} + triangle) * 3u;
    const float weights[3] = {feature.barycentric.x, feature.barycentric.y,
                              feature.barycentric.z};
    for (uint32_t j = 0u; j < 3u; ++j) {
        const uint32_t local = view.triangles[tri_at + j];
        const uint32_t particle = env * p.particles_per_env +
            target_info.vertex_offset + local;
        data.point_endpoint_terms[first + j] =
            nk::WeightedPointEndpointTerm(nk::kNkSideParticle, particle, weights[j]);
    }
    const uint32_t terms = nk::CanonicalizePointEndpointTerms(
        data.point_endpoint_terms + first, 3u);
    data.point_endpoint_ranges[endpoint] = {first, terms};
    data.ucontact_point[at] = (source + feature.point) * 0.5f;
    data.ucontact_witness_a[at] = source;
    data.ucontact_witness_b[at] = feature.point;
    data.ucontact_normal[at] = feature.normal;
    data.ucontact_depth[at] = radius - feature.distance;
    data.ucontact_a[at] = item;
    data.ucontact_b[at] = endpoint;
    data.ucontact_a_kind[at] = nk::kUContactSideParticle;
    data.ucontact_b_kind[at] = nk::kUContactSidePointEndpoint;
    data.ucontact_gen[at] = 1u;
    data.ucontact_law[slot] = nk::kContactLawSpeculative;
    data.ucontact_friction[slot] = fmaxf(
        model.particle_surface_friction[source_surface],
        model.particle_surface_friction[target_surface]);
    nk::CanonicalContactDescriptor descriptor;
    descriptor.a.type = constraint::CollidableType::Particle;
    descriptor.a.handle = item;
    descriptor.b.type = constraint::CollidableType::ParticleSurface;
    descriptor.b.handle = env * p.surfaces_per_env + target_surface;
    descriptor.feature_a = 0u;
    descriptor.feature_b =
        ((target_info.triangle_offset + triangle) << 2u) | feature.kind;
    descriptor.contact_kind = nk::kContactKindOgcVertexFace;
    descriptor.normal = feature.normal;
    const nk::ContactId id = nk::MakeContactId(descriptor);
    data.ucontact_id_pair[at] = id.pair;
    data.ucontact_id_feature[at] = id.feature;
    data.ucontact_count[slot] = 1u;
    atomicAdd(data.contact_count + env, 1u);
}

__device__ void OgcEmitParticleEdge(const OgcDetectParams& p, const ModelView& model,
                                    const DataView& data, const OgcCandidate& c) {
    const uint32_t env = c.item / p.edges_per_env;
    const uint32_t local_edge = c.item % p.edges_per_env;
    const uint32_t source_surface = c.source_surface, target_surface = c.target_surface;
    const auto view = OgcParticleView(p, model, data, env);
    const auto source_info = model.particle_surface_info[source_surface];
    const auto source_edges = model.particle_surface_edge_info[source_surface];
    const auto target_info = model.particle_surface_info[target_surface];
    const auto target_edges = model.particle_surface_edge_info[target_surface];
    const uint32_t source_edge = local_edge - source_edges.edge_offset;
    const uint32_t target_edge = c.target;
    const auto edge_a = model.particle_surface_edges[local_edge];
    const auto edge_b = model.particle_surface_edges[target_edges.edge_offset + target_edge];
    const float radius = model.particle_surface_thickness[source_surface] +
                         model.particle_surface_thickness[target_surface];
    const float query = c.query;
    const collision::MeshVertexIncidence incidence{
        model.particle_surface_vertex_edge_offsets, model.particle_surface_vertex_edges};
    const auto pair = collision::OgcEdgeContact(
        view, source_info, model.particle_surface_edges, incidence,
        source_edges, source_edge, view, target_info,
        model.particle_surface_edges, incidence, target_edges, target_edge, query);
    if (!(pair.feasible && pair.owner_edge_a == source_edge &&
          pair.owner_edge_b == target_edge &&
          pair.distance <= query && pair.distance > 0.0f)) return;
    const uint32_t ordinal = OgcClaimSlot(p, data, env);
    if (ordinal >= p.slot_capacity) {
        atomicOr(data.env_status + env, kEnvStatusPairOverflow);
        return;
    }
    const uint32_t slot = env * p.slot_stride + p.slot_base + ordinal;
    const size_t at = size_t{slot} * nk::kPairDrivenPtsPerSlot;
    const uint32_t endpoint_a = env * p.point_endpoints_per_env + p.point_endpoint_first +
                                ordinal * 2u;
    const uint32_t endpoint_b = endpoint_a + 1u;
    const uint32_t first_a = env * p.point_endpoint_terms_per_env + p.point_endpoint_term_first +
                             ordinal * 6u;
    const uint32_t first_b = first_a + 3u;
    const uint32_t vertices_a[2] = {edge_a.vertex0, edge_a.vertex1};
    const uint32_t vertices_b[2] = {edge_b.vertex0, edge_b.vertex1};
    for (uint32_t j = 0u; j < 2u; ++j) {
        const uint32_t particle_a = env * p.particles_per_env +
            source_info.vertex_offset + vertices_a[j];
        const uint32_t particle_b = env * p.particles_per_env +
            target_info.vertex_offset + vertices_b[j];
        data.point_endpoint_terms[first_a + j] =
            nk::WeightedPointEndpointTerm(nk::kNkSideParticle,
                particle_a, j == 0u ? 1.0f - pair.weight_a : pair.weight_a);
        data.point_endpoint_terms[first_b + j] =
            nk::WeightedPointEndpointTerm(nk::kNkSideParticle,
                particle_b, j == 0u ? 1.0f - pair.weight_b : pair.weight_b);
    }
    const uint32_t count_a = nk::CanonicalizePointEndpointTerms(
        data.point_endpoint_terms + first_a, 2u);
    const uint32_t count_b = nk::CanonicalizePointEndpointTerms(
        data.point_endpoint_terms + first_b, 2u);
    data.point_endpoint_ranges[endpoint_a] = {first_a, count_a};
    data.point_endpoint_ranges[endpoint_b] = {first_b, count_b};
    data.ucontact_point[at] = (pair.point_a + pair.point_b) * 0.5f;
    data.ucontact_witness_a[at] = pair.point_a;
    data.ucontact_witness_b[at] = pair.point_b;
    data.ucontact_normal[at] = pair.normal;
    data.ucontact_depth[at] = radius - pair.distance;
    data.ucontact_a[at] = endpoint_a;
    data.ucontact_b[at] = endpoint_b;
    data.ucontact_a_kind[at] = nk::kUContactSidePointEndpoint;
    data.ucontact_b_kind[at] = nk::kUContactSidePointEndpoint;
    data.ucontact_gen[at] = 1u;
    data.ucontact_law[slot] = nk::kContactLawSpeculative;
    data.ucontact_friction[slot] = fmaxf(
        model.particle_surface_friction[source_surface],
        model.particle_surface_friction[target_surface]);
    nk::CanonicalContactDescriptor descriptor;
    descriptor.a.type = constraint::CollidableType::ParticleSurface;
    descriptor.a.handle = env * p.surfaces_per_env + source_surface;
    descriptor.b.type = constraint::CollidableType::ParticleSurface;
    descriptor.b.handle = env * p.surfaces_per_env + target_surface;
    descriptor.feature_a = source_edges.edge_offset + source_edge;
    descriptor.feature_b = target_edges.edge_offset + target_edge;
    descriptor.contact_kind = nk::kContactKindOgcEdgeEdge;
    descriptor.normal = pair.normal;
    const nk::ContactId id = nk::MakeContactId(descriptor);
    data.ucontact_id_pair[at] = id.pair;
    data.ucontact_id_feature[at] = id.feature;
    data.ucontact_count[slot] = 1u;
    atomicAdd(data.contact_count + env, 1u);
}

__global__ void OgcParticleVertexFaceKernel(OgcDetectParams p, ModelView model,
                                             DataView data) {
    __shared__ OgcCandidate queue[kBlockSize / kWarpLanes][2u * kWarpLanes];
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t item = thread / kTreeShares, share = thread % kTreeShares;
    bool active = item < p.env_count * p.particles_per_env;
    const uint32_t env = item / p.particles_per_env;
    const uint32_t local_particle = item % p.particles_per_env;
    const auto view = OgcParticleView(p, model, data, env);
    // Surfaces partition the vertices, so at most one source surface holds this one.
    uint32_t source_surface = 0u;
    for (; active && source_surface < p.surfaces_per_env; ++source_surface) {
        const auto info = model.particle_surface_info[source_surface];
        if (local_particle >= info.vertex_offset &&
            local_particle - info.vertex_offset < info.vertex_count) break;
    }
    active = active && source_surface < p.surfaces_per_env;
    const auto source_info = active ? model.particle_surface_info[source_surface]
                                    : collision::MeshSurfaceInfo{};
    const uint32_t source_vertex = local_particle - source_info.vertex_offset;
    const math::Vec3 source = active ? data.particle_pos[item] : math::Vec3{};
    uint32_t target_surface = 0u, cursor = 0u, end = 0u;
    collision::MeshSurfaceInfo target_info{};
    float query = 0.0f;
    const auto enter = [&]() {
        cursor = end = 0u;
        if (target_surface >= p.surfaces_per_env) return;
        target_info = model.particle_surface_info[target_surface];
        query = OgcParticleQuery(p, model, data, env, source_surface, target_surface);
        if (!(query > 0.0f) || !collision::MeshSurfaceRangeValid(view, target_info)) return;
        if (!TreeShare(view.nodes + target_info.node_offset, target_info.node_count, share,
                       &cursor, &end))
            atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
    };
    if (active) enter();
    const auto next = [&](OgcCandidate* candidate) {
        while (target_surface < p.surfaces_per_env) {
            if (cursor >= end) {
                ++target_surface;
                enter();
                continue;
            }
            const auto& node = view.nodes[target_info.node_offset + cursor];
            if (node.escape <= cursor || node.escape > target_info.node_count) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                cursor = end;
                continue;
            }
            if (collision::MeshBoundsDistanceSquared(source, node) > query * query) {
                cursor = node.escape;
                continue;
            }
            ++cursor;
            if (node.triangle == ~0u) continue;
            if (source_surface == target_surface &&
                VertexInTriangle(view, source_info, source_vertex, node.triangle)) continue;
            *candidate = {item, source_surface, target_surface, node.triangle, query};
            return true;
        }
        return false;
    };
    TestInWarpBatches(queue[threadIdx.x / kWarpLanes], active, next,
        [&](const OgcCandidate& c) { OgcEmitParticleFace(p, model, data, c); });
}

__global__ void OgcParticleEdgeEdgeKernel(OgcDetectParams p, ModelView model,
                                           DataView data) {
    __shared__ OgcCandidate queue[kBlockSize / kWarpLanes][2u * kWarpLanes];
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t item = thread / kTreeShares, share = thread % kTreeShares;
    bool active = item < p.env_count * p.edges_per_env;
    const uint32_t env = item / p.edges_per_env;
    const uint32_t local_edge = item % p.edges_per_env;
    const auto view = OgcParticleView(p, model, data, env);
    const auto* edge_nodes = data.particle_surface_edge_nodes +
        size_t{env} * p.edge_nodes_per_env;
    // Surfaces partition the edges, so at most one source surface holds this one.
    uint32_t source_surface = 0u;
    for (; active && source_surface < p.surfaces_per_env; ++source_surface) {
        const auto edges = model.particle_surface_edge_info[source_surface];
        if (local_edge >= edges.edge_offset && local_edge - edges.edge_offset < edges.edge_count)
            break;
    }
    active = active && source_surface < p.surfaces_per_env;
    const auto source_edges = active ? model.particle_surface_edge_info[source_surface]
                                     : collision::MeshEdgeInfo{};
    const uint32_t source_edge = local_edge - source_edges.edge_offset;
    const auto edge_a = active ? model.particle_surface_edges[local_edge] : collision::MeshEdge{};
    math::Vec3 midpoint{};
    float half_length = 0.0f;
    if (active) {
        const auto source_info = model.particle_surface_info[source_surface];
        const math::Vec3 a0 = collision::MeshSurfaceVertex(view, source_info, edge_a.vertex0);
        const math::Vec3 a1 = collision::MeshSurfaceVertex(view, source_info, edge_a.vertex1);
        midpoint = (a0 + a1) * 0.5f;
        half_length = sqrtf((a1 - a0).LengthSq()) * 0.5f;
    }
    uint32_t target_surface = source_surface, cursor = 0u, end = 0u;
    collision::MeshEdgeInfo target_edges{};
    float query = 0.0f, broad_radius = 0.0f;
    const auto enter = [&]() {
        cursor = end = 0u;
        if (target_surface >= p.surfaces_per_env) return;
        target_edges = model.particle_surface_edge_info[target_surface];
        query = OgcParticleQuery(p, model, data, env, source_surface, target_surface);
        if (!(query > 0.0f)) return;
        broad_radius = query + half_length;
        if (!TreeShare(edge_nodes + target_edges.node_offset, target_edges.node_count, share,
                       &cursor, &end))
            atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
    };
    if (active) enter();
    const auto next = [&](OgcCandidate* candidate) {
        while (target_surface < p.surfaces_per_env) {
            if (cursor >= end) {
                ++target_surface;
                enter();
                continue;
            }
            const auto& node = edge_nodes[target_edges.node_offset + cursor];
            if (node.escape <= cursor || node.escape > target_edges.node_count) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                cursor = end;
                continue;
            }
            if (collision::MeshBoundsDistanceSquared(midpoint, node) >
                broad_radius * broad_radius) {
                cursor = node.escape;
                continue;
            }
            ++cursor;
            if (node.triangle == ~0u) continue;
            const uint32_t target_edge = node.triangle;
            if (target_edge >= target_edges.edge_count) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                cursor = end;
                continue;
            }
            if (source_surface == target_surface && target_edge <= source_edge) continue;
            const auto edge_b = model.particle_surface_edges[
                target_edges.edge_offset + target_edge];
            if (source_surface == target_surface &&
                (edge_a.vertex0 == edge_b.vertex0 || edge_a.vertex0 == edge_b.vertex1 ||
                 edge_a.vertex1 == edge_b.vertex0 || edge_a.vertex1 == edge_b.vertex1)) continue;
            *candidate = {item, source_surface, target_surface, target_edge, query};
            return true;
        }
        return false;
    };
    TestInWarpBatches(queue[threadIdx.x / kWarpLanes], active, next,
        [&](const OgcCandidate& c) { OgcEmitParticleEdge(p, model, data, c); });
}

__global__ void FinalizeOgcCountKernel(OgcDetectParams p, DataView data) {
    const uint64_t env = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (env >= p.env_count) return;
    const uint32_t count = data.ogc_contact_count[env];
    data.ogc_contact_count[env] = min(count, p.slot_capacity);
    if (count > p.slot_capacity) atomicOr(data.env_status + env, kEnvStatusPairOverflow);
}

Status OpOgcDetect(const ModelView& model, const DataView& data,
                   const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const OgcDetectParams*>(params);
    if (!p || !(p->dt > 0.0f) || !std::isfinite(p->dt) ||
        !(p->relaxation > 0.0f && p->relaxation < 1.0f) ||
        !(p->margin >= 0.0f) || !std::isfinite(p->margin)) return Status::InvalidArgument;
    const bool mixed = p->mesh_vertex_sources > 0u || p->mesh_edge_sources > 0u;
    if (p->slot_capacity == 0u ||
        (p->surfaces_per_env == 0u && !mixed) || p->env_count == 0u)
        return Status::Ok;
    if (p->slot_base > p->slot_stride ||
        p->slot_capacity > p->slot_stride - p->slot_base ||
        p->point_endpoint_first + uint64_t{p->slot_capacity} * 2u > p->point_endpoints_per_env ||
        p->point_endpoint_term_first + uint64_t{p->slot_capacity} * 6u >
            p->point_endpoint_terms_per_env ||
        (p->surfaces_per_env > 0u &&
            (!model.particle_surface_info ||
             (p->triangles_per_env > 0u && (!model.particle_surface_triangles ||
                                           !data.particle_surface_nodes ||
                                           !model.particle_surface_triangle_edges)) ||
             (p->edges_per_env > 0u && (!model.particle_surface_edges ||
                                       !data.particle_surface_edge_nodes)) ||
             !model.particle_surface_edge_info || !model.particle_surface_thickness ||
             !model.particle_surface_friction || !data.particle_pos ||
             !data.particle_surface_max_speed)) ||
        (mixed && (!model.mesh_surface_info || !model.mesh_edge_info ||
                   !model.mesh_triangles || !model.mesh_bvh_nodes ||
                   !model.mesh_edges || !model.mesh_edge_nodes ||
                   !model.mesh_triangle_edges || !model.hull_verts ||
                   (p->mesh_vertex_sources > 0u && !model.mesh_vertex_sources) ||
                   (p->mesh_edge_sources > 0u && !model.mesh_edge_sources) ||
                   !model.shape_table || !data.body_pose ||
                   !data.body_aabb_lo || !data.body_aabb_hi ||
                   !data.body_linear_velocity || !data.body_angular_velocity ||
                   !data.dat_body_speed || !data.dat_body_query_radius ||
                   !data.dat_body_motion || !data.mat_buckets ||
                   (p->excluded_pairs > 0u && !model.excluded_pairs) ||
                   (p->articulations_per_env > 0u &&
                    (!model.link_to_articulation || !model.articulation_link_offset ||
                     !model.articulation_link_count || !model.parent_link ||
                     !model.joint_type || !model.link_local_pose || !model.parent_offset ||
                     !model.link_geom_kind || !model.link_geom_local ||
                     !model.mesh_vertex_reach || !data.q || !data.qdot ||
                     !data.dat_joint_motion || !data.dat_joint_rate)))) ||
        !data.ogc_contact_count || !data.contact_count ||
        !data.point_endpoint_ranges || !data.point_endpoint_terms ||
        !data.ucontact_count || !data.ucontact_point || !data.ucontact_normal ||
        !data.ucontact_witness_a || !data.ucontact_witness_b ||
        !data.ucontact_depth || !data.ucontact_a || !data.ucontact_b ||
        !data.ucontact_a_kind || !data.ucontact_b_kind || !data.ucontact_gen ||
        !data.ucontact_law || !data.ucontact_friction ||
        !data.ucontact_id_pair || !data.ucontact_id_feature || !data.env_status ||
        !data.dat_pair_kind || !data.dat_pair_owner_a || !data.dat_pair_owner_b ||
        !data.dat_pair_feature_a || !data.dat_pair_feature_b)
        return Status::InvalidArgument;
    const uint64_t slots = uint64_t{p->env_count} * p->slot_capacity;
    const uint64_t particles = uint64_t{p->env_count} * p->particles_per_env;
    const uint64_t edges = uint64_t{p->env_count} * p->edges_per_env;
    const uint64_t vertices = uint64_t{p->env_count} *
        (p->particles_per_env + p->mesh_vertex_sources);
    const uint64_t mixed_edges = uint64_t{p->env_count} *
        (p->edges_per_env + p->mesh_edge_sources);
    const uint64_t sources = uint64_t{p->env_count} *
        (p->particles_per_env + p->edges_per_env +
         (mixed ? p->particles_per_env + p->edges_per_env +
                  p->mesh_vertex_sources + p->mesh_edge_sources : 0u));
    const uint64_t links = uint64_t{p->env_count} * p->links_per_env;
    if (slots > UINT32_MAX || particles + edges > UINT32_MAX ||
        particles > UINT32_MAX / kTreeShares || edges > UINT32_MAX / kTreeShares ||
        vertices > UINT32_MAX || mixed_edges > UINT32_MAX || sources > UINT32_MAX ||
        links > UINT32_MAX)
        return Status::InvalidArgument;
    if (slots > static_cast<uint64_t>(std::numeric_limits<int>::max()) || !data.contact_cache_scratch ||
        p->workspace_bytes <= ogc_order::Layout(static_cast<uint32_t>(slots)).temp_offset)
        return Status::InvalidArgument;
    LaunchCuda(ClearOgcSlotsKernel,
               dim3((static_cast<uint32_t>(slots) + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, *p, data);
    if (mixed && p->articulations_per_env > 0u && links > 0u)
        LaunchCuda(OgcJointRateKernel,
                   dim3((static_cast<uint32_t>(links) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, data);
    if (mixed && p->bodies_per_env > 0u)
        LaunchCuda(OgcBodyQueryKernel,
                   dim3((p->env_count * p->bodies_per_env + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (particles > 0u)
        LaunchCuda(OgcParticleVertexFaceKernel,
                   dim3((static_cast<uint32_t>(particles) * kTreeShares + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (edges > 0u)
        LaunchCuda(OgcParticleEdgeEdgeKernel,
                   dim3((static_cast<uint32_t>(edges) * kTreeShares + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (mixed && vertices > 0u)
        LaunchCuda(OgcMixedVertexFaceKernel,
                   dim3((static_cast<uint32_t>(vertices) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (mixed && mixed_edges > 0u)
        LaunchCuda(OgcMixedEdgeEdgeKernel,
                   dim3((static_cast<uint32_t>(mixed_edges) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    LaunchCuda(FinalizeOgcCountKernel,
               dim3((p->env_count - 1u) / kBlockSize + 1u),
               dim3(kBlockSize), 0u, stream, *p, data);
    // Slot order follows the contact records, not the order in which threads claimed slots.
    const ogc_order::Workspace workspace(data.contact_cache_scratch, p->workspace_bytes,
                                         static_cast<uint32_t>(slots));
    if (ogc_order::Canonicalize(*p, data, workspace, stream) != cudaSuccess) return Status::Failed;
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

}  // namespace

uint64_t OgcOrderScratchBytes(uint32_t slot_capacity, uint32_t env_count) {
    return ogc_order::ScratchBytes(slot_capacity, env_count);
}

void RegisterNkOgcDetectOps() {
    SetCudaOp(NkOp::OgcDetect, &OpOgcDetect);
}

}  // namespace nuka::phi
