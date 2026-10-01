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

__global__ void OgcParticleVertexFaceKernel(OgcDetectParams p, ModelView model,
                                             DataView data, bool emit) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = p.env_count * p.particles_per_env;
    if (item >= total) return;
    const uint32_t env = item / p.particles_per_env;
    const uint32_t local_particle = item % p.particles_per_env;
    const size_t source_index = size_t{env} * (p.particles_per_env + p.edges_per_env) + local_particle;
    uint64_t next_slot = emit ? data.ogc_source_offsets[source_index] : 0u;
    const collision::MeshSurfaceView view{
        reinterpret_cast<const float*>(data.particle_pos + size_t{env} * p.particles_per_env),
        model.particle_surface_triangles,
        data.particle_surface_nodes + size_t{env} * p.nodes_per_env,
        {p.particles_per_env, p.triangles_per_env, p.nodes_per_env}};
    for (uint32_t source_surface = 0u; source_surface < p.surfaces_per_env; ++source_surface) {
        const auto source_info = model.particle_surface_info[source_surface];
        if (local_particle < source_info.vertex_offset ||
            local_particle - source_info.vertex_offset >= source_info.vertex_count) continue;
        const uint32_t source_vertex = local_particle - source_info.vertex_offset;
        const math::Vec3 source = data.particle_pos[item];
        for (uint32_t target_surface = 0u; target_surface < p.surfaces_per_env; ++target_surface) {
            const auto target_info = model.particle_surface_info[target_surface];
            const float radius = model.particle_surface_thickness[source_surface] +
                                 model.particle_surface_thickness[target_surface];
            const float source_speed = data.particle_surface_max_speed[
                size_t{env} * p.surfaces_per_env + source_surface];
            const float target_speed = data.particle_surface_max_speed[
                size_t{env} * p.surfaces_per_env + target_surface];
            const float query = collision::DatQueryRadius(
                radius + p.margin + p.dt * (source_speed + target_speed));
            if (!(query > 0.0f) || !collision::MeshSurfaceRangeValid(view, target_info)) continue;
            const float query_sq = query * query;
            uint32_t cursor = 0u;
            while (cursor < target_info.node_count) {
                const auto& node = view.nodes[target_info.node_offset + cursor];
                if (node.escape <= cursor || node.escape > target_info.node_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    break;
                }
                if (collision::MeshBoundsDistanceSquared(source, node) > query_sq) {
                    cursor = node.escape;
                    continue;
                }
                if (node.triangle != ~0u) {
                    if (source_surface == target_surface &&
                        VertexInTriangle(view, source_info, source_vertex, node.triangle)) {
                        ++cursor;
                        continue;
                    }
                    const auto feature = collision::OgcFacetFeature(view, target_info,
                        model.particle_surface_edges, model.particle_surface_triangle_edges,
                        model.particle_surface_edge_info[target_surface], node.triangle,
                        source, false);
                    if (feature.feasible && feature.owner_triangle == node.triangle &&
                        feature.distance <= query && feature.distance > 0.0f) {
                        const uint64_t ordinal = next_slot++;
                        if (!emit) {
                            ++cursor;
                            continue;
                        }
                        if (ordinal >= p.slot_capacity) {
                            atomicOr(data.env_status + env, kEnvStatusPairOverflow);
                        } else {
                            const uint32_t slot = env * p.slot_stride + p.slot_base + ordinal;
                            const size_t at = size_t{slot} * nk::kPairDrivenPtsPerSlot;
                            const uint32_t endpoint = env * p.point_endpoints_per_env + ordinal * 2u + 1u;
                            const uint32_t first = env * p.point_endpoint_terms_per_env + ordinal * 6u + 3u;
                            const size_t tri_at = (size_t{target_info.triangle_offset} + node.triangle) * 3u;
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
                                ((target_info.triangle_offset + node.triangle) << 2u) | feature.kind;
                            descriptor.contact_kind = nk::kContactKindOgcVertexFace;
                            descriptor.normal = feature.normal;
                            const nk::ContactId id = nk::MakeContactId(descriptor);
                            data.ucontact_id_pair[at] = id.pair;
                            data.ucontact_id_feature[at] = id.feature;
                            __threadfence();
                            data.ucontact_count[slot] = 1u;
                            atomicAdd(data.contact_count + env, 1u);
                        }
                    }
                }
                ++cursor;
            }
        }
    }
    if (!emit) data.ogc_source_offsets[source_index] = next_slot;
}

__global__ void OgcParticleEdgeEdgeKernel(OgcDetectParams p, ModelView model,
                                           DataView data, bool emit) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = p.env_count * p.edges_per_env;
    if (item >= total) return;
    const uint32_t env = item / p.edges_per_env;
    const uint32_t local_edge = item % p.edges_per_env;
    const size_t source_index = size_t{env} * (p.particles_per_env + p.edges_per_env) +
                                p.particles_per_env + local_edge;
    uint64_t next_slot = emit ? data.ogc_source_offsets[source_index] : 0u;
    const collision::MeshSurfaceView view{
        reinterpret_cast<const float*>(data.particle_pos + size_t{env} * p.particles_per_env),
        model.particle_surface_triangles,
        data.particle_surface_nodes + size_t{env} * p.nodes_per_env,
        {p.particles_per_env, p.triangles_per_env, p.nodes_per_env}};
    const auto* edge_nodes = data.particle_surface_edge_nodes +
        size_t{env} * p.edge_nodes_per_env;
    for (uint32_t source_surface = 0u; source_surface < p.surfaces_per_env; ++source_surface) {
        const auto source_info = model.particle_surface_info[source_surface];
        const auto source_edges = model.particle_surface_edge_info[source_surface];
        if (local_edge < source_edges.edge_offset ||
            local_edge - source_edges.edge_offset >= source_edges.edge_count) continue;
        const uint32_t source_edge = local_edge - source_edges.edge_offset;
        const auto edge_a = model.particle_surface_edges[local_edge];
        const math::Vec3 a0 = collision::MeshSurfaceVertex(view, source_info, edge_a.vertex0);
        const math::Vec3 a1 = collision::MeshSurfaceVertex(view, source_info, edge_a.vertex1);
        const math::Vec3 midpoint = (a0 + a1) * 0.5f;
        const float half_length = sqrtf((a1 - a0).LengthSq()) * 0.5f;
        for (uint32_t target_surface = source_surface;
             target_surface < p.surfaces_per_env; ++target_surface) {
            const auto target_info = model.particle_surface_info[target_surface];
            const auto target_edges = model.particle_surface_edge_info[target_surface];
            const float radius = model.particle_surface_thickness[source_surface] +
                                 model.particle_surface_thickness[target_surface];
            const float source_speed = data.particle_surface_max_speed[
                size_t{env} * p.surfaces_per_env + source_surface];
            const float target_speed = data.particle_surface_max_speed[
                size_t{env} * p.surfaces_per_env + target_surface];
            const float query = collision::DatQueryRadius(
                radius + p.margin + p.dt * (source_speed + target_speed));
            if (!(query > 0.0f)) continue;
            const float broad_radius = query + half_length;
            uint32_t cursor = 0u;
            while (cursor < target_edges.node_count) {
                const auto& node = edge_nodes[target_edges.node_offset + cursor];
                if (node.escape <= cursor || node.escape > target_edges.node_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    break;
                }
                if (collision::MeshBoundsDistanceSquared(midpoint, node) >
                    broad_radius * broad_radius) {
                    cursor = node.escape;
                    continue;
                }
                if (node.triangle != ~0u) {
                    const uint32_t target_edge = node.triangle;
                    if (target_edge >= target_edges.edge_count) {
                        atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                        break;
                    }
                    if (source_surface == target_surface && target_edge <= source_edge) {
                        ++cursor;
                        continue;
                    }
                    const auto edge_b = model.particle_surface_edges[
                        target_edges.edge_offset + target_edge];
                    if (source_surface == target_surface &&
                        (edge_a.vertex0 == edge_b.vertex0 ||
                         edge_a.vertex0 == edge_b.vertex1 ||
                         edge_a.vertex1 == edge_b.vertex0 ||
                         edge_a.vertex1 == edge_b.vertex1)) {
                        ++cursor;
                        continue;
                    }
                    const auto pair = collision::OgcEdgeContact(
                        view, source_info, model.particle_surface_edges, edge_nodes,
                        source_edges, source_edge, view, target_info,
                        model.particle_surface_edges, edge_nodes, target_edges, target_edge,
                        false, false);
                    if (pair.feasible && pair.owner_edge_a == source_edge &&
                        pair.owner_edge_b == target_edge &&
                        pair.distance <= query && pair.distance > 0.0f) {
                        const uint64_t ordinal = next_slot++;
                        if (!emit) {
                            ++cursor;
                            continue;
                        }
                        if (ordinal >= p.slot_capacity) {
                            atomicOr(data.env_status + env, kEnvStatusPairOverflow);
                        } else {
                            const uint32_t slot = env * p.slot_stride + p.slot_base + ordinal;
                            const size_t at = size_t{slot} * nk::kPairDrivenPtsPerSlot;
                            const uint32_t endpoint_a = env * p.point_endpoints_per_env + ordinal * 2u;
                            const uint32_t endpoint_b = endpoint_a + 1u;
                            const uint32_t first_a = env * p.point_endpoint_terms_per_env + ordinal * 6u;
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
                            __threadfence();
                            data.ucontact_count[slot] = 1u;
                            atomicAdd(data.contact_count + env, 1u);
                        }
                    }
                }
                ++cursor;
            }
        }
    }
    if (!emit) data.ogc_source_offsets[source_index] = next_slot;
}

__global__ void ScanOgcSourcesKernel(OgcDetectParams p, DataView data) {
    const uint32_t env = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const bool mixed = p.mesh_vertex_sources > 0u || p.mesh_edge_sources > 0u;
    const uint32_t count = p.particles_per_env + p.edges_per_env +
        (mixed ? p.particles_per_env + p.edges_per_env +
                 p.mesh_vertex_sources + p.mesh_edge_sources : 0u);
    auto* offsets = data.ogc_source_offsets + size_t{env} * count;
    __shared__ uint64_t scan[kBlockSize];
    __shared__ uint64_t carried;
    if (lane == 0u) carried = 0u;
    __syncthreads();
    for (uint32_t base = 0u; base < count; base += kBlockSize) {
        const uint64_t value = base + lane < count ? offsets[base + lane] : 0u;
        scan[lane] = value;
        __syncthreads();
        for (uint32_t distance = 1u; distance < kBlockSize; distance *= 2u) {
            const uint64_t previous = lane >= distance ? scan[lane - distance] : 0u;
            __syncthreads();
            scan[lane] += previous;
            __syncthreads();
        }
        if (base + lane < count) offsets[base + lane] = carried + scan[lane] - value;
        __syncthreads();
        if (lane == 0u) carried += scan[kBlockSize - 1u];
        __syncthreads();
    }
    if (lane == 0u) {
        data.ogc_contact_count[env] = carried > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(carried);
        if (carried > p.slot_capacity) atomicOr(data.env_status + env, kEnvStatusPairOverflow);
    }
}

Status OpOgcDetect(const ModelView& model, const DataView& data,
                   const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const OgcDetectParams*>(params);
    if (!p || !(p->dt > 0.0f) || !std::isfinite(p->dt) ||
        !(p->margin >= 0.0f) || !std::isfinite(p->margin)) return Status::InvalidArgument;
    const bool mixed = p->mesh_vertex_sources > 0u || p->mesh_edge_sources > 0u;
    if (p->slot_capacity == 0u ||
        (p->surfaces_per_env == 0u && !mixed) || p->env_count == 0u)
        return Status::Ok;
    if (p->slot_base > p->slot_stride ||
        p->slot_capacity > p->slot_stride - p->slot_base ||
        uint64_t{p->slot_capacity} * 2u > p->point_endpoints_per_env ||
        uint64_t{p->slot_capacity} * 6u > p->point_endpoint_terms_per_env ||
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
                   !data.mat_buckets ||
                   (p->excluded_pairs > 0u && !model.excluded_pairs))) ||
        !data.ogc_contact_count || !data.ogc_source_offsets || !data.contact_count ||
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
    if (slots > UINT32_MAX || particles + edges > UINT32_MAX ||
        vertices > UINT32_MAX || mixed_edges > UINT32_MAX || sources > UINT32_MAX)
        return Status::InvalidArgument;
    LaunchCuda(ClearOgcSlotsKernel,
               dim3((static_cast<uint32_t>(slots) + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, *p, data);
    if (mixed && p->bodies_per_env > 0u)
        LaunchCuda(OgcBodyQueryKernel,
                   dim3((p->env_count * p->bodies_per_env + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (particles > 0u)
        LaunchCuda(OgcParticleVertexFaceKernel,
                   dim3((static_cast<uint32_t>(particles) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data, false);
    if (edges > 0u)
        LaunchCuda(OgcParticleEdgeEdgeKernel,
                   dim3((static_cast<uint32_t>(edges) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data, false);
    if (mixed && vertices > 0u)
        LaunchCuda(OgcMixedVertexFaceKernel,
                   dim3((static_cast<uint32_t>(vertices) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data, false);
    if (mixed && mixed_edges > 0u)
        LaunchCuda(OgcMixedEdgeEdgeKernel,
                   dim3((static_cast<uint32_t>(mixed_edges) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data, false);
    LaunchCuda(ScanOgcSourcesKernel, dim3(p->env_count), dim3(kBlockSize),
               0u, stream, *p, data);
    if (particles > 0u)
        LaunchCuda(OgcParticleVertexFaceKernel,
                   dim3((static_cast<uint32_t>(particles) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data, true);
    if (edges > 0u)
        LaunchCuda(OgcParticleEdgeEdgeKernel,
                   dim3((static_cast<uint32_t>(edges) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data, true);
    if (mixed && vertices > 0u)
        LaunchCuda(OgcMixedVertexFaceKernel,
                   dim3((static_cast<uint32_t>(vertices) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data, true);
    if (mixed && mixed_edges > 0u)
        LaunchCuda(OgcMixedEdgeEdgeKernel,
                   dim3((static_cast<uint32_t>(mixed_edges) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data, true);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkOgcDetectOps() {
    SetCudaOp(NkOp::OgcDetect, &OpOgcDetect);
}

}  // namespace nuka::phi
