#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "collision/dat_geometry.hpp"
#include "math/cuda_vec_ops.cuh"
#include "nk/model/generated/arena_layout.hpp"
#include "nk/model/generated/views.hpp"
#include "nk/solve/collidable_owner.hpp"
#include "nk/solve/vertex_block.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/dat_motion.cuh"
#include "phi/backend_cuda/ops/rigid_types.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"

namespace nuka::phi {
namespace {

constexpr uint32_t kBlockSize = 128u;
static_assert(nk::LayoutOf(nk::FieldId::DatFailureWitness).elem_size ==
              collision::kDatWitnessWords * sizeof(uint64_t));

__device__ uint32_t DatBodyWitness(const DatTruncateParams& p, const ModelView& model,
                                   uint32_t env, uint32_t body) {
    if (body >= p.bodies_per_env) return collision::kDatWitnessNone;
    const auto owner = nk::ResolveCollidableOwner(
        nkops::LoadPrimShape(model.shape_table, body).body_id, env, body, p.bodies_per_env,
        p.links_per_env, p.articulations_per_env, model.body_to_link,
        model.body_to_articulation, model.body_collidable_body);
    if (owner.kind == nk::kNkSideArtic)
        return collision::DatWitnessOwner(collision::kDatWitnessLink,
                                          owner.link - env * p.links_per_env);
    if (owner.kind == nk::kNkSideRigid)
        return collision::DatWitnessOwner(collision::kDatWitnessBody,
                                          owner.body - env * p.bodies_per_env);
    if (owner.kind == nk::kNkSideStatic)
        return collision::DatWitnessOwner(collision::kDatWitnessStatic, body);
    return collision::kDatWitnessNone;
}

__device__ uint32_t DatSurfaceWitness(const DatTruncateParams& p, const ModelView& model,
                                      uint32_t particle) {
    const uint32_t local = p.particles_per_env > 0u ? particle % p.particles_per_env : 0u;
    for (uint32_t surface = 0u; surface < p.surfaces_per_env; ++surface) {
        const auto info = model.particle_surface_info[surface];
        if (local >= info.vertex_offset && local - info.vertex_offset < info.vertex_count)
            return collision::DatWitnessOwner(collision::kDatWitnessSurface, surface);
    }
    return collision::kDatWitnessNone;
}

__device__ uint32_t DatRefWitness(const DatTruncateParams& p, const ModelView& model,
                                  uint32_t env, const DatPointRef& ref) {
    return ref.particle ? DatSurfaceWitness(p, model, ref.owner)
                        : DatBodyWitness(p, model, env, ref.owner);
}

// Counts a failure and its deepest overlap per reason and unordered owner pair.
__device__ void RecordDatFailure(const DataView& data, uint32_t env, uint32_t reason,
                                 uint32_t owner_a, uint32_t owner_b, float depth) {
    auto* words = reinterpret_cast<unsigned long long*>(
        data.dat_failure_witness + size_t{env} * collision::kDatWitnessWords);
    atomicAdd(words + reason - 1u, 1ull);
    const unsigned long long key = (static_cast<unsigned long long>(reason) << 60u) |
        (static_cast<unsigned long long>(min(owner_a, owner_b)) << 30u) | max(owner_a, owner_b);
    auto* slots = words + collision::kDatFailureReasons + 1u;
    uint32_t slot = static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ull) >> 32u) %
        collision::kDatWitnessSlots;
    for (uint32_t probe = 0u; probe < collision::kDatWitnessSlots; ++probe) {
        const unsigned long long seen = atomicCAS(slots + 3u * slot, 0ull, key);
        if (seen == 0ull || seen == key) {
            atomicAdd(slots + 3u * slot + 1u, 1ull);
            if (depth > 0.0f)
                atomicMax(slots + 3u * slot + 2u,
                          static_cast<unsigned long long>(__float_as_uint(depth)));
            return;
        }
        slot = (slot + 1u) % collision::kDatWitnessSlots;
    }
    atomicAdd(words + collision::kDatFailureReasons, 1ull);
}

__device__ float DatOverlapDepth(const collision::DatPrimitive& a,
                                 const collision::DatPrimitive& b) {
    return fmaxf(0.0f, -collision::DatFindSeparator(a, b, -FLT_MAX).gap);
}

__device__ float RequestedRadius(const DatTruncateParams& p, const ModelView& model,
                                 const DataView& data, uint32_t env,
                                 uint32_t source, uint32_t target) {
    const size_t offset = size_t{env} * p.surfaces_per_env;
    return model.particle_surface_thickness[source] +
           model.particle_surface_thickness[target] + p.margin +
           data.dat_surface_motion[offset + source] +
           data.dat_surface_motion[offset + target];
}

__device__ float QueryRadius(const DatTruncateParams& p, const ModelView& model,
                             const DataView& data, uint32_t env,
                             uint32_t source, uint32_t target) {
    return collision::DatQueryRadius(RequestedRadius(p, model, data, env, source, target));
}

__global__ void ClearDatCountersKernel(DatTruncateParams p, ModelView model, DataView data) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= p.env_count) return;
    data.dat_failure_count[env] = 0u;
    uint32_t capped = 0u;
    for (uint32_t source = 0u; source < p.surfaces_per_env; ++source)
        for (uint32_t target = source; target < p.surfaces_per_env; ++target) {
            const float radius = RequestedRadius(p, model, data, env, source, target);
            const float base = model.particle_surface_thickness[source] +
                               model.particle_surface_thickness[target] + p.margin;
            if (!(radius >= 0.0f && radius <= FLT_MAX) ||
                !(base <= collision::kDatQueryRadiusMax)) {
                atomicOr(data.env_status + env, kEnvStatusDatFailure);
                data.dat_failure_count[env] = 1u;
                RecordDatFailure(data, env, base <= collision::kDatQueryRadiusMax
                    ? collision::kDatFailureMotion : collision::kDatFailureRadius,
                    collision::DatWitnessOwner(collision::kDatWitnessSurface, source),
                    collision::DatWitnessOwner(collision::kDatWitnessSurface, target), 0.0f);
            }
            capped |= radius > collision::kDatQueryRadiusMax;
        }
    for (uint32_t source = 0u; source < p.surfaces_per_env; ++source)
        for (uint32_t body = 0u; body < p.bodies_per_env; ++body) {
            if (model.mesh_surface_info[body].vertex_count == 0u) continue;
            const float base = model.particle_surface_thickness[source] + p.margin;
            const float radius = base + p.dt * (
                data.particle_surface_max_speed[size_t{env} * p.surfaces_per_env + source] +
                data.dat_body_speed[size_t{env} * p.bodies_per_env + body]);
            if (!(radius >= 0.0f && radius <= FLT_MAX) ||
                !(base <= collision::kDatQueryRadiusMax)) {
                atomicOr(data.env_status + env, kEnvStatusDatFailure);
                data.dat_failure_count[env] += 1u;
                RecordDatFailure(data, env, base <= collision::kDatQueryRadiusMax
                    ? collision::kDatFailureMotion : collision::kDatFailureRadius,
                    collision::DatWitnessOwner(collision::kDatWitnessSurface, source),
                    DatBodyWitness(p, model, env, body), 0.0f);
            }
            capped |= radius > collision::kDatQueryRadiusMax;
        }
    for (uint32_t source = 0u; source < p.bodies_per_env; ++source)
        for (uint32_t target = source + 1u; target < p.bodies_per_env; ++target) {
            if (model.mesh_surface_info[source].vertex_count == 0u ||
                model.mesh_surface_info[target].vertex_count == 0u) continue;
            const auto sa = nkops::LoadPrimShape(model.shape_table, source);
            const auto sb = nkops::LoadPrimShape(model.shape_table, target);
            if (nkops::RouteMeshContact(sa.kind, sb.kind,
                    model.mesh_contact_mode[source], model.mesh_contact_mode[target]) !=
                nkops::MeshContactRoute::Ogc) continue;
            const float radius = 0.0005f + p.margin + p.dt * (
                data.dat_body_speed[size_t{env} * p.bodies_per_env + source] +
                data.dat_body_speed[size_t{env} * p.bodies_per_env + target]);
            if (!(radius >= 0.0f && radius <= FLT_MAX)) {
                atomicOr(data.env_status + env, kEnvStatusDatFailure);
                data.dat_failure_count[env] += 1u;
                RecordDatFailure(data, env, collision::kDatFailureMotion,
                                 DatBodyWitness(p, model, env, source),
                                 DatBodyWitness(p, model, env, target), 0.0f);
            }
            capped |= radius > collision::kDatQueryRadiusMax;
        }
    data.dat_query_limit_count[env] = capped;
}

__device__ void FailPrimitivePair(const DatTruncateParams& p, const ModelView& model,
                                  const DataView& data, uint32_t env, uint32_t reason, float depth,
                                  const uint32_t* ids_a, uint32_t count_a,
                                  const uint32_t* ids_b, uint32_t count_b) {
    atomicOr(data.env_status + env, kEnvStatusDatFailure);
    atomicAdd(data.dat_failure_count + env, 1u);
    RecordDatFailure(data, env, reason, DatSurfaceWitness(p, model, ids_a[0]), count_b > 0u
        ? DatSurfaceWitness(p, model, ids_b[0]) : collision::kDatWitnessNone, depth);
    for (uint32_t i = 0u; i < count_a; ++i)
        atomicMin(reinterpret_cast<uint32_t*>(data.dat_particle_beta + ids_a[i]), 0u);
    for (uint32_t i = 0u; i < count_b; ++i)
        atomicMin(reinterpret_cast<uint32_t*>(data.dat_particle_beta + ids_b[i]), 0u);
}

__device__ collision::MeshSurfaceView ReferenceView(const DatTruncateParams& p,
                                                    const ModelView& model,
                                                    const DataView& data,
                                                    uint32_t env) {
    return {reinterpret_cast<const float*>(data.particle_prev_pos +
                size_t{env} * p.particles_per_env), model.particle_surface_triangles,
            data.particle_surface_nodes + size_t{env} * p.nodes_per_env,
            {p.particles_per_env, p.triangles_per_env, p.nodes_per_env}};
}

__device__ void ClipPrimitivePair(const DatTruncateParams& p, const ModelView& model,
                                  const DataView& data, uint32_t env, const uint32_t* ids_a,
                                  uint32_t count_a, const uint32_t* ids_b, uint32_t count_b) {
    collision::DatPrimitive a, b;
    a.count = count_a;
    b.count = count_b;
    for (uint32_t i = 0u; i < count_a; ++i) a.vertex[i] = data.particle_prev_pos[ids_a[i]];
    for (uint32_t i = 0u; i < count_b; ++i) b.vertex[i] = data.particle_prev_pos[ids_b[i]];
    const auto separator = collision::DatFindSeparator(a, b);
    if (!(separator.gap > 0.0f)) {
        const bool valid = collision::DatPrimitiveValid(a) && collision::DatPrimitiveValid(b);
        FailPrimitivePair(p, model, data, env, valid ? collision::kDatFailureOverlap
            : collision::kDatFailureDegenerate, valid ? DatOverlapDepth(a, b) : 0.0f,
            ids_a, count_a, ids_b, count_b);
        return;
    }
    float approach_a = 0.0f, approach_b = 0.0f;
    for (uint32_t i = 0u; i < count_a; ++i)
        approach_a = fmaxf(approach_a,
            -(data.particle_pos[ids_a[i]] - a.vertex[i]).Dot(separator.normal));
    for (uint32_t i = 0u; i < count_b; ++i)
        approach_b = fmaxf(approach_b,
            (data.particle_pos[ids_b[i]] - b.vertex[i]).Dot(separator.normal));
    const float total = approach_a + approach_b;
    collision::DatSweptPrimitive swept_a, swept_b;
    swept_a.count = 2u * count_a;
    swept_b.count = 2u * count_b;
    for (uint32_t i = 0u; i < count_a; ++i) {
        swept_a.vertex[i] = a.vertex[i];
        swept_a.vertex[count_a + i] = data.particle_pos[ids_a[i]];
    }
    for (uint32_t i = 0u; i < count_b; ++i) {
        swept_b.vertex[i] = b.vertex[i];
        swept_b.vertex[count_b + i] = data.particle_pos[ids_b[i]];
    }
    const float minimum_gap = (1.0f - p.relaxation) * separator.gap;
    if (collision::DatSweptSeparated(swept_a, swept_b, separator.normal, minimum_gap)) return;
    const float fraction = total > 0.0f ? approach_b / total : 0.5f;
    const float plane_offset = separator.gap * fraction;
    for (uint32_t side = 0u; side < 2u; ++side) {
        const uint32_t* ids = side == 0u ? ids_a : ids_b;
        const uint32_t count = side == 0u ? count_a : count_b;
        const math::Vec3 normal = separator.normal * (side == 0u ? 1.0f : -1.0f);
        for (uint32_t i = 0u; i < count; ++i) {
            const math::Vec3 start = data.particle_prev_pos[ids[i]];
            const float projection = (start - separator.negative_support).Dot(separator.normal);
            const float distance = side == 0u ? projection - plane_offset : plane_offset - projection;
            const float closing = -(data.particle_pos[ids[i]] - start).Dot(normal);
            if (!(closing > 0.0f) || closing <= distance) continue;
            const float beta = fmaxf(0.0f, p.relaxation * distance / closing);
            atomicMin(reinterpret_cast<uint32_t*>(data.dat_particle_beta + ids[i]), __float_as_uint(beta));
        }
    }
}

__global__ void ClearDatKernel(DatTruncateParams p, ModelView model, DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.env_count * p.particles_per_env) return;
    const uint32_t env = item / p.particles_per_env;
    const uint32_t local = item % p.particles_per_env;
    float bound = FLT_MAX;
    for (uint32_t source = 0u; source < p.surfaces_per_env; ++source) {
        const auto info = model.particle_surface_info[source];
        if (local < info.vertex_offset || local - info.vertex_offset >= info.vertex_count)
            continue;
        for (uint32_t target = 0u; target < p.surfaces_per_env; ++target) {
            // Uncapped queries cover both primitives' complete linear trajectories.
            if (RequestedRadius(p, model, data, env, source, target) >
                collision::kDatQueryRadiusMax)
                bound = fminf(bound, 0.5f * p.relaxation * collision::kDatQueryRadiusMax);
        }
        for (uint32_t body = 0u; body < p.bodies_per_env; ++body) {
            if (model.mesh_surface_info[body].vertex_count == 0u) continue;
            const float radius = model.particle_surface_thickness[source] + p.margin +
                p.dt * (data.particle_surface_max_speed[
                    size_t{env} * p.surfaces_per_env + source] +
                    data.dat_body_speed[size_t{env} * p.bodies_per_env + body]);
            bound = fminf(bound, 0.5f * p.relaxation *
                collision::DatQueryRadius(radius));
        }
    }
    const float motion = sqrtf((data.particle_pos[item] - data.particle_prev_pos[item]).LengthSq());
    const float beta = (data.env_status[env] & kEnvStatusDatFailure) != 0u
        ? 0.0f : motion > bound ? bound / motion : 1.0f;
    data.dat_particle_beta[item] = beta;
}

__global__ void DatVertexFaceKernel(DatTruncateParams p, ModelView model, DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.env_count * p.particles_per_env) return;
    const uint32_t env = item / p.particles_per_env;
    const uint32_t local = item % p.particles_per_env;
    const auto view = ReferenceView(p, model, data, env);
    const math::Vec3 vertex = data.particle_prev_pos[item];
    for (uint32_t source = 0u; source < p.surfaces_per_env; ++source) {
        const auto source_info = model.particle_surface_info[source];
        if (local < source_info.vertex_offset ||
            local - source_info.vertex_offset >= source_info.vertex_count) continue;
        for (uint32_t target = 0u; target < p.surfaces_per_env; ++target) {
            const auto info = model.particle_surface_info[target];
            const float query = QueryRadius(p, model, data, env, source, target);
            uint32_t cursor = 0u;
            while (cursor < info.node_count) {
                const auto& node = view.nodes[info.node_offset + cursor];
                if (node.escape <= cursor || node.escape > info.node_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    break;
                }
                if (collision::MeshBoundsDistanceSquared(vertex, node) > query * query) {
                    cursor = node.escape;
                    continue;
                }
                if (node.triangle != ~0u) {
                    if (node.triangle >= info.triangle_count) {
                        atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                        break;
                    }
                    const size_t at = (size_t{info.triangle_offset} + node.triangle) * 3u;
                    uint32_t ids[3];
                    bool incident = false;
                    for (uint32_t j = 0u; j < 3u; ++j) {
                        ids[j] = env * p.particles_per_env + info.vertex_offset + view.triangles[at + j];
                        incident |= ids[j] == item;
                    }
                    if (!incident) {
                        const auto closest = collision::ClosestTrianglePoint(vertex,
                            data.particle_prev_pos[ids[0]], data.particle_prev_pos[ids[1]],
                            data.particle_prev_pos[ids[2]]);
                        if ((closest.point - vertex).LengthSq() <= query * query)
                            ClipPrimitivePair(p, model, data, env, &item, 1u, ids, 3u);
                    }
                }
                ++cursor;
            }
        }
    }
}

__global__ void DatEdgeEdgeKernel(DatTruncateParams p, ModelView model, DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.env_count * p.edges_per_env) return;
    const uint32_t env = item / p.edges_per_env;
    const uint32_t local = item % p.edges_per_env;
    const auto* nodes = data.particle_surface_edge_nodes + size_t{env} * p.edge_nodes_per_env;
    for (uint32_t source = 0u; source < p.surfaces_per_env; ++source) {
        const auto info_a = model.particle_surface_info[source];
        const auto edges_a = model.particle_surface_edge_info[source];
        if (local < edges_a.edge_offset || local - edges_a.edge_offset >= edges_a.edge_count)
            continue;
        const auto edge_a = model.particle_surface_edges[local];
        const uint32_t base_a = env * p.particles_per_env + info_a.vertex_offset;
        const uint32_t ids_a[2] = {base_a + edge_a.vertex0, base_a + edge_a.vertex1};
        const math::Vec3 a0 = data.particle_prev_pos[ids_a[0]];
        const math::Vec3 a1 = data.particle_prev_pos[ids_a[1]];
        const math::Vec3 midpoint = (a0 + a1) * 0.5f;
        const float half_length = sqrtf((a1 - a0).LengthSq()) * 0.5f;
        for (uint32_t target = source; target < p.surfaces_per_env; ++target) {
            const auto info_b = model.particle_surface_info[target];
            const auto edges_b = model.particle_surface_edge_info[target];
            const float query = QueryRadius(p, model, data, env, source, target);
            const float broad_radius = query + half_length;
            uint32_t cursor = 0u;
            while (cursor < edges_b.node_count) {
                const auto& node = nodes[edges_b.node_offset + cursor];
                if (node.escape <= cursor || node.escape > edges_b.node_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    break;
                }
                if (collision::MeshBoundsDistanceSquared(midpoint, node) > broad_radius * broad_radius) {
                    cursor = node.escape;
                    continue;
                }
                if (node.triangle != ~0u) {
                    if (node.triangle >= edges_b.edge_count) {
                        atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                        break;
                    }
                    const uint32_t target_edge = edges_b.edge_offset + node.triangle;
                    if (target_edge <= local) {
                        ++cursor;
                        continue;
                    }
                    const auto edge_b = model.particle_surface_edges[target_edge];
                    const uint32_t base_b = env * p.particles_per_env + info_b.vertex_offset;
                    const uint32_t ids_b[2] = {base_b + edge_b.vertex0, base_b + edge_b.vertex1};
                    const bool incident = ids_a[0] == ids_b[0] || ids_a[0] == ids_b[1] ||
                                          ids_a[1] == ids_b[0] || ids_a[1] == ids_b[1];
                    if (!incident) {
                        const auto closest = collision::OgcClosestSegments(a0, a1,
                            data.particle_prev_pos[ids_b[0]], data.particle_prev_pos[ids_b[1]]);
                        if ((closest.a - closest.b).LengthSq() <= query * query)
                            ClipPrimitivePair(p, model, data, env, ids_a, 2u, ids_b, 2u);
                    }
                }
                ++cursor;
            }
        }
    }
}

__global__ void DatTriangleValidityKernel(DatTruncateParams p, ModelView model,
                                          DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.env_count * p.triangles_per_env) return;
    const uint32_t env = item / p.triangles_per_env;
    const uint32_t local = item % p.triangles_per_env;
    for (uint32_t surface = 0u; surface < p.surfaces_per_env; ++surface) {
        const auto info = model.particle_surface_info[surface];
        if (local < info.triangle_offset ||
            local - info.triangle_offset >= info.triangle_count) continue;
        const uint32_t base = env * p.particles_per_env + info.vertex_offset;
        const size_t at = size_t{local} * 3u;
        const uint32_t ids[3] = {base + model.particle_surface_triangles[at],
                                 base + model.particle_surface_triangles[at + 1u],
                                 base + model.particle_surface_triangles[at + 2u]};
        collision::DatPrimitive triangle;
        triangle.count = 3u;
        for (uint32_t i = 0u; i < 3u; ++i)
            triangle.vertex[i] = data.particle_prev_pos[ids[i]];
        if (!collision::DatPrimitiveValid(triangle))
            FailPrimitivePair(p, model, data, env, collision::kDatFailureDegenerate, 0.0f,
                              ids, 3u, ids, 0u);
        break;
    }
}

__global__ void DatInitialEdgeFaceKernel(DatTruncateParams p, ModelView model,
                                         DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.env_count * p.edges_per_env) return;
    const uint32_t env = item / p.edges_per_env;
    const uint32_t local = item % p.edges_per_env;
    const auto view = ReferenceView(p, model, data, env);
    for (uint32_t source = 0u; source < p.surfaces_per_env; ++source) {
        const auto info_a = model.particle_surface_info[source];
        const auto edges_a = model.particle_surface_edge_info[source];
        if (local < edges_a.edge_offset || local - edges_a.edge_offset >= edges_a.edge_count)
            continue;
        const auto edge = model.particle_surface_edges[local];
        const uint32_t base_a = env * p.particles_per_env + info_a.vertex_offset;
        const uint32_t ids_a[2] = {base_a + edge.vertex0, base_a + edge.vertex1};
        collision::DatPrimitive segment;
        segment.count = 2u;
        segment.vertex[0] = data.particle_prev_pos[ids_a[0]];
        segment.vertex[1] = data.particle_prev_pos[ids_a[1]];
        if (!collision::DatPrimitiveValid(segment)) {
            FailPrimitivePair(p, model, data, env, collision::kDatFailureDegenerate, 0.0f,
                              ids_a, 2u, ids_a, 0u);
            return;
        }
        const math::Vec3 midpoint = (segment.vertex[0] + segment.vertex[1]) * 0.5f;
        const float half_length = sqrtf((segment.vertex[1] - segment.vertex[0]).LengthSq()) * 0.5f;
        for (uint32_t target = 0u; target < p.surfaces_per_env; ++target) {
            const auto info_b = model.particle_surface_info[target];
            uint32_t cursor = 0u;
            while (cursor < info_b.node_count) {
                const auto& node = view.nodes[info_b.node_offset + cursor];
                if (node.escape <= cursor || node.escape > info_b.node_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    break;
                }
                if (collision::MeshBoundsDistanceSquared(midpoint, node) >
                    half_length * half_length) {
                    cursor = node.escape;
                    continue;
                }
                if (node.triangle != ~0u) {
                    if (node.triangle >= info_b.triangle_count) {
                        atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                        break;
                    }
                    const uint32_t base_b = env * p.particles_per_env + info_b.vertex_offset;
                    const size_t at = (size_t{info_b.triangle_offset} + node.triangle) * 3u;
                    uint32_t ids_b[3];
                    bool incident = false;
                    for (uint32_t i = 0u; i < 3u; ++i) {
                        ids_b[i] = base_b + view.triangles[at + i];
                        incident |= ids_b[i] == ids_a[0] || ids_b[i] == ids_a[1];
                    }
                    if (!incident) {
                        collision::DatPrimitive triangle;
                        triangle.count = 3u;
                        for (uint32_t i = 0u; i < 3u; ++i)
                            triangle.vertex[i] = data.particle_prev_pos[ids_b[i]];
                        const uint32_t reason = collision::DatPrimitiveValid(triangle)
                            ? collision::kDatFailureOverlap : collision::kDatFailureDegenerate;
                        if (reason == collision::kDatFailureDegenerate ||
                            !(collision::DatFindSeparator(segment, triangle).gap > 0.0f))
                            FailPrimitivePair(p, model, data, env, reason,
                                reason == collision::kDatFailureOverlap
                                    ? DatOverlapDepth(segment, triangle) : 0.0f,
                                ids_a, 2u, ids_b, 3u);
                    }
                }
                ++cursor;
            }
        }
        return;
    }
}

__global__ void ClearDatOwnersKernel(DatTruncateParams p, DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t bodies = p.env_count * p.bodies_per_env;
    const uint32_t articulations = p.env_count * p.articulations_per_env;
    if (item < bodies) data.dat_body_beta[item] = __float_as_uint(1.0f);
    if (item < articulations) data.dat_artic_beta[item] = __float_as_uint(1.0f);
    if (item < p.env_count) {
        data.dat_joint_truncation_count[item] = 0u;
        data.dat_joint_truncation_energy[item] = 0.0f;
        data.dat_body_truncation_count[item] = 0u;
        data.dat_body_truncation_energy[item] = 0.0f;
    }
}

__global__ void DatBodyMotionBoundKernel(DatTruncateParams p, ModelView model,
                                          DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.env_count * p.mesh_vertex_sources) return;
    const uint32_t env = item / p.mesh_vertex_sources;
    const uint64_t packed = model.mesh_vertex_sources[item % p.mesh_vertex_sources];
    const uint32_t body = static_cast<uint32_t>(packed >> 32u);
    const uint32_t vertex = static_cast<uint32_t>(packed);
    if (body >= p.bodies_per_env || vertex >= model.mesh_surface_info[body].vertex_count) {
        atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
        return;
    }
    const DatPointRef ref{body, vertex, false};
    const float speed = DatPointLipschitz(p, model, data, env, ref);
    const float query = data.dat_body_query_radius[size_t{env} * p.bodies_per_env + body];
    if (!(speed >= 0.0f && speed <= FLT_MAX) ||
        !(query > 0.0f && query <= collision::kDatQueryRadiusMax)) {
        atomicOr(data.env_status + env, kEnvStatusDatFailure);
        atomicAdd(data.dat_failure_count + env, 1u);
        RecordDatFailure(data, env, speed >= 0.0f && speed <= FLT_MAX
            ? collision::kDatFailureRadius : collision::kDatFailureMotion,
            DatBodyWitness(p, model, env, body), collision::kDatWitnessNone, 0.0f);
        DatMinOwnerBeta(p, model, data, env, ref, 0.0f);
        return;
    }
    const float bound = 0.5f * p.relaxation * query;
    if (speed > bound)
        DatMinOwnerBeta(p, model, data, env, ref, bound / speed);
}

__device__ bool DatMixedRefs(DatTruncateParams p, ModelView model, DataView data,
                              uint32_t env, uint32_t slot, DatPointRef* a,
                              uint32_t* count_a, DatPointRef* b, uint32_t* count_b) {
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
            if (owner_a >= p.bodies_per_env ||
                feature_a >= model.mesh_surface_info[owner_a].vertex_count) return false;
            a[0] = {owner_a, feature_a, false};
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

__device__ void FailMixedPair(DatTruncateParams p, ModelView model, DataView data,
                               uint32_t env, uint32_t reason, float depth, const DatPointRef* a,
                               uint32_t count_a, const DatPointRef* b, uint32_t count_b) {
    atomicOr(data.env_status + env, kEnvStatusDatFailure);
    atomicAdd(data.dat_failure_count + env, 1u);
    RecordDatFailure(data, env, reason, DatRefWitness(p, model, env, a[0]), count_b > 0u
        ? DatRefWitness(p, model, env, b[0]) : collision::kDatWitnessNone, depth);
    for (uint32_t i = 0u; i < count_a; ++i)
        DatMinOwnerBeta(p, model, data, env, a[i], 0.0f);
    for (uint32_t i = 0u; i < count_b; ++i)
        DatMinOwnerBeta(p, model, data, env, b[i], 0.0f);
}

__device__ bool DatBodyPairAllowed(DatTruncateParams p, ModelView model,
                                   uint32_t a, uint32_t b) {
    if (a == b) return false;
    const auto sa = nkops::LoadPrimShape(model.shape_table, a);
    const auto sb = nkops::LoadPrimShape(model.shape_table, b);
    if (nkops::RouteMeshContact(sa.kind, sb.kind, model.mesh_contact_mode[a],
                                model.mesh_contact_mode[b]) != nkops::MeshContactRoute::Ogc)
        return false;
    if (((sa.contype & sb.conaffinity) | (sb.contype & sa.conaffinity)) == 0u)
        return false;
    const int32_t ga = static_cast<int32_t>(sa.group);
    const int32_t gb = static_cast<int32_t>(sb.group);
    if (ga != 0 && gb != 0 &&
        (ga > 0 ? ga != gb && gb > 0 : ga == gb)) return false;
    const uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
    const uint64_t key = (uint64_t{lo} << 32u) | hi;
    uint32_t first = 0u, last = p.excluded_pairs;
    while (first < last) {
        const uint32_t mid = first + (last - first) / 2u;
        const uint64_t value = model.excluded_pairs[mid];
        if (value < key) first = mid + 1u;
        else if (value > key) last = mid;
        else return false;
    }
    return true;
}

__device__ collision::MeshSurfaceView DatOldBodyView(
    DatTruncateParams p, ModelView model, DataView data,
    uint32_t env, uint32_t body) {
    const auto pose = data.dat_prev_body_pose[size_t{env} * p.bodies_per_env + body];
    return {model.hull_verts, model.mesh_triangles, model.mesh_bvh_nodes,
            {p.mesh_vertices, p.mesh_triangles, p.mesh_nodes},
            pose.position, pose.rotation, true};
}

__global__ void DatInitialMixedEdgeFaceKernel(DatTruncateParams p,
                                               ModelView model, DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t per_env = p.edges_per_env + p.mesh_edge_sources;
    if (item >= p.env_count * per_env) return;
    const uint32_t env = item / per_env;
    const uint32_t local = item % per_env;
    const bool source_particle = local < p.edges_per_env;
    uint32_t source_body = ~0u;
    DatPointRef source[2];
    if (source_particle) {
        uint32_t surface = ~0u;
        for (uint32_t s = 0u; s < p.surfaces_per_env; ++s) {
            const auto edges = model.particle_surface_edge_info[s];
            if (local >= edges.edge_offset && local - edges.edge_offset < edges.edge_count) {
                surface = s;
                break;
            }
        }
        if (surface == ~0u) return;
        const auto info = model.particle_surface_info[surface];
        const auto edge = model.particle_surface_edges[local];
        if (edge.vertex0 >= info.vertex_count || edge.vertex1 >= info.vertex_count) {
            atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
            return;
        }
        source[0] = {env * p.particles_per_env + info.vertex_offset + edge.vertex0, 0u, true};
        source[1] = {env * p.particles_per_env + info.vertex_offset + edge.vertex1, 0u, true};
    } else {
        const uint64_t packed = model.mesh_edge_sources[local - p.edges_per_env];
        source_body = static_cast<uint32_t>(packed >> 32u);
        const uint32_t edge_index = static_cast<uint32_t>(packed);
        if (source_body >= p.bodies_per_env ||
            edge_index >= model.mesh_edge_info[source_body].edge_count) {
            atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
            return;
        }
        const auto edge_info = model.mesh_edge_info[source_body];
        const auto info = model.mesh_surface_info[source_body];
        const auto edge = model.mesh_edges[edge_info.edge_offset + edge_index];
        if (edge.vertex0 >= info.vertex_count || edge.vertex1 >= info.vertex_count) {
            atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
            return;
        }
        source[0] = {source_body, edge.vertex0, false};
        source[1] = {source_body, edge.vertex1, false};
    }
    collision::DatPrimitive segment;
    segment.count = 2u;
    segment.vertex[0] = DatPointAt(p, model, data, env, source[0], 0.0f);
    segment.vertex[1] = DatPointAt(p, model, data, env, source[1], 0.0f);
    if (!collision::DatPrimitiveValid(segment)) {
        FailMixedPair(p, model, data, env, collision::kDatFailureDegenerate, 0.0f,
                      source, 2u, source, 0u);
        return;
    }
    const auto midpoint = (segment.vertex[0] + segment.vertex[1]) * 0.5f;
    const float half_length = 0.5f * sqrtf(
        (segment.vertex[1] - segment.vertex[0]).LengthSq()) + 1.0e-6f;
    for (uint32_t group = 0u; group < (source_particle ? 1u : 2u); ++group) {
        const bool target_particle = !source_particle && group == 0u;
        const uint32_t targets = target_particle ? p.surfaces_per_env : p.bodies_per_env;
        for (uint32_t target = 0u; target < targets; ++target) {
            if (!target_particle && !source_particle &&
                !DatBodyPairAllowed(p, model, source_body, target)) continue;
            const auto info = target_particle ? model.particle_surface_info[target]
                                              : model.mesh_surface_info[target];
            const auto view = target_particle ? ReferenceView(p, model, data, env)
                : DatOldBodyView(p, model, data, env, target);
            if (!collision::MeshSurfaceRangeValid(view, info)) continue;
            const auto query = collision::MeshSurfaceLocalPoint(view, midpoint);
            uint32_t cursor = 0u;
            while (cursor < info.node_count) {
                const auto node = view.nodes[info.node_offset + cursor];
                if (node.escape <= cursor || node.escape > info.node_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    break;
                }
                if (collision::MeshBoundsDistanceSquared(query, node) >
                    half_length * half_length) {
                    cursor = node.escape;
                    continue;
                }
                if (node.triangle != ~0u) {
                    if (node.triangle >= info.triangle_count) {
                        atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                        break;
                    }
                    DatPointRef face[3];
                    const size_t at = (size_t{info.triangle_offset} + node.triangle) * 3u;
                    bool incident = false, valid = true;
                    for (uint32_t j = 0u; j < 3u; ++j) {
                        const uint32_t vertex = view.triangles[at + j];
                        valid &= vertex < info.vertex_count;
                        face[j] = target_particle
                            ? DatPointRef{env * p.particles_per_env + info.vertex_offset + vertex,
                                          0u, true}
                            : DatPointRef{target, vertex, false};
                        incident |= face[j].particle && source[0].particle &&
                            (face[j].owner == source[0].owner || face[j].owner == source[1].owner);
                    }
                    if (!valid) {
                        atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                        break;
                    }
                    if (!incident) {
                        collision::DatPrimitive triangle;
                        triangle.count = 3u;
                        for (uint32_t j = 0u; j < 3u; ++j)
                            triangle.vertex[j] = DatPointAt(p, model, data, env, face[j], 0.0f);
                        const uint32_t reason = collision::DatPrimitiveValid(triangle)
                            ? collision::kDatFailureOverlap : collision::kDatFailureDegenerate;
                        if (reason == collision::kDatFailureDegenerate ||
                            !(collision::DatFindSeparator(segment, triangle).gap > 0.0f))
                            FailMixedPair(p, model, data, env, reason,
                                reason == collision::kDatFailureOverlap
                                    ? DatOverlapDepth(segment, triangle) : 0.0f,
                                source, 2u, face, 3u);
                    }
                }
                ++cursor;
            }
        }
    }
}

__global__ void DatMixedPairsKernel(DatTruncateParams p, ModelView model, DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.env_count * p.slot_capacity) return;
    const uint32_t env = item / p.slot_capacity;
    const uint32_t ordinal = item % p.slot_capacity;
    if (ordinal >= data.ogc_contact_count[env]) return;
    const uint32_t slot = env * p.slot_stride + p.slot_base + ordinal;
    if (data.dat_pair_kind[slot] == 0u) return;
    DatPointRef refs_a[3], refs_b[3];
    uint32_t count_a = 0u, count_b = 0u;
    if (!DatMixedRefs(p, model, data, env, slot, refs_a, &count_a,
                      refs_b, &count_b)) {
        atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
        return;
    }
    collision::DatPrimitive a, b;
    a.count = count_a;
    b.count = count_b;
    for (uint32_t i = 0u; i < count_a; ++i)
        a.vertex[i] = DatPointAt(p, model, data, env, refs_a[i], 0.0f);
    for (uint32_t i = 0u; i < count_b; ++i)
        b.vertex[i] = DatPointAt(p, model, data, env, refs_b[i], 0.0f);
    const auto separator = collision::DatFindSeparator(a, b);
    if (!(separator.gap > 0.0f)) {
        const bool valid = collision::DatPrimitiveValid(a) && collision::DatPrimitiveValid(b);
        FailMixedPair(p, model, data, env, valid ? collision::kDatFailureOverlap
            : collision::kDatFailureDegenerate, valid ? DatOverlapDepth(a, b) : 0.0f,
            refs_a, count_a, refs_b, count_b);
        return;
    }
    float speed_a[3], speed_b[3];
    float bound_a = 0.0f, bound_b = 0.0f;
    for (uint32_t i = 0u; i < count_a; ++i) {
        speed_a[i] = DatPointLipschitz(p, model, data, env, refs_a[i]);
        if (!(speed_a[i] >= 0.0f && speed_a[i] < FLT_MAX)) {
            FailMixedPair(p, model, data, env, collision::kDatFailureMotion, 0.0f,
                          refs_a, count_a, refs_b, count_b);
            return;
        }
        bound_a = fmaxf(bound_a, speed_a[i]);
    }
    for (uint32_t i = 0u; i < count_b; ++i) {
        speed_b[i] = DatPointLipschitz(p, model, data, env, refs_b[i]);
        if (!(speed_b[i] >= 0.0f && speed_b[i] < FLT_MAX)) {
            FailMixedPair(p, model, data, env, collision::kDatFailureMotion, 0.0f,
                          refs_a, count_a, refs_b, count_b);
            return;
        }
        bound_b = fmaxf(bound_b, speed_b[i]);
    }
    if (bound_a + bound_b < separator.gap) return;
    float approach_a = 0.0f, approach_b = 0.0f;
    for (uint32_t i = 0u; i < count_a; ++i)
        approach_a = fmaxf(approach_a, -(
            DatPointAt(p, model, data, env, refs_a[i], 1.0f) - a.vertex[i]).Dot(separator.normal));
    for (uint32_t i = 0u; i < count_b; ++i)
        approach_b = fmaxf(approach_b, (
            DatPointAt(p, model, data, env, refs_b[i], 1.0f) - b.vertex[i]).Dot(separator.normal));
    const float total = approach_a + approach_b;
    const float fraction = total > 0.0f ? approach_b / total : 0.5f;
    const float plane_offset = separator.gap * fraction;
    for (uint32_t side = 0u; side < 2u; ++side) {
        const DatPointRef* refs = side == 0u ? refs_a : refs_b;
        const uint32_t count = side == 0u ? count_a : count_b;
        const math::Vec3 normal = separator.normal * (side == 0u ? 1.0f : -1.0f);
        for (uint32_t i = 0u; i < count; ++i) {
            const float offset = side == 0u ? plane_offset : -plane_offset;
            const float speed = side == 0u ? speed_a[i] : speed_b[i];
            const float root = DatPlaneFraction(p, model, data, env, refs[i],
                separator.negative_support, normal, offset, speed);
            if (root < 1.0f)
                DatMinOwnerBeta(p, model, data, env, refs[i], p.relaxation * root);
        }
    }
}

__global__ void DatSnapshotKernel(DatTruncateParams p, DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t links = p.env_count * p.links_per_env;
    const uint32_t articulations = p.env_count * p.articulations_per_env;
    const uint32_t bodies = p.env_count * p.bodies_per_env;
    if (item < links) data.dat_prev_q[item] = data.q[item];
    if (item < articulations)
        data.dat_prev_base_pose[item] = data.base_pose[item];
    if (item < bodies) data.dat_prev_body_pose[item] = data.body_pose[item];
}

__device__ float DatBodyKineticEnergy(const DataView& data, uint32_t body) {
    const float inv_mass = data.body_inv_mass[body];
    if (!(inv_mass > 0.0f)) return 0.0f;
    const auto linear = data.body_linear_velocity[body];
    const auto pose = data.body_pose[body];
    const auto frame = data.body_inertial_frame[body];
    const auto principal = math::gpu::QuatNormalizeRsqrt(
        math::gpu::QuatMul(pose.rotation, frame.rotation), 1.0e-12f);
    const math::Quat inverse = math::gpu::MakeQuat(
        principal.w, -principal.x, -principal.y, -principal.z);
    const auto angular = math::gpu::RotateByQuatNormalized(
        inverse, data.body_angular_velocity[body]);
    const auto inverse_inertia = data.body_inv_inertia[body];
    if (!(inverse_inertia.x > 0.0f && inverse_inertia.y > 0.0f &&
          inverse_inertia.z > 0.0f)) return INFINITY;
    return 0.5f * (linear.LengthSq() / inv_mass +
        angular.x * angular.x / inverse_inertia.x +
        angular.y * angular.y / inverse_inertia.y +
        angular.z * angular.z / inverse_inertia.z);
}

__global__ void TallyDatOwnersKernel(DatTruncateParams p, ModelView model,
                                      DataView data) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= p.env_count) return;
    uint32_t joint_count = 0u, body_count = 0u;
    float joint_energy = 0.0f, body_energy = 0.0f;
    for (uint32_t local = 0u; local < p.articulations_per_env; ++local) {
        const uint32_t articulation = env * p.articulations_per_env + local;
        const float beta = __uint_as_float(data.dat_artic_beta[articulation]);
        if (!(beta < 1.0f)) continue;
        const size_t tile = size_t{articulation} * p.max_dof;
        const size_t matrix = tile * p.max_dof;
        float kinetic = 0.0f;
        for (uint32_t i = 0u; i < p.max_dof; ++i) {
            if (!(data.m[matrix + size_t{i} * p.max_dof + i] > 0.0f)) continue;
            ++joint_count;
            float momentum = 0.0f;
            for (uint32_t j = 0u; j < p.max_dof; ++j)
                momentum += data.m[matrix + size_t{i} * p.max_dof + j] *
                    data.qdot_flat[tile + j];
            kinetic += data.qdot_flat[tile + i] * momentum;
        }
        joint_energy += 0.5f * (1.0f - beta * beta) * kinetic;
    }
    for (uint32_t local = 0u; local < p.bodies_per_env; ++local) {
        const uint32_t body = env * p.bodies_per_env + local;
        const float beta = __uint_as_float(data.dat_body_beta[body]);
        if (!(beta < 1.0f) || !(data.body_inv_mass[body] > 0.0f) ||
            (model.body_collidable_body && model.body_collidable_body[body] != ~0u) ||
            (model.body_collidable_link && model.body_collidable_link[body] != ~0u) ||
            (model.body_to_link && model.body_to_link[local] != ~0u)) continue;
        ++body_count;
        body_energy += (1.0f - beta * beta) * DatBodyKineticEnergy(data, body);
    }
    if (!(joint_energy >= 0.0f && joint_energy <= FLT_MAX) ||
        !(body_energy >= 0.0f && body_energy <= FLT_MAX))
        atomicOr(data.env_status + env, kEnvStatusDatFailure);
    data.dat_joint_truncation_count[env] = joint_count;
    data.dat_joint_truncation_energy[env] = joint_energy;
    data.dat_body_truncation_count[env] = body_count;
    data.dat_body_truncation_energy[env] = body_energy;
    data.dat_truncation_count[env] += joint_count + body_count;
    data.dat_truncation_energy[env] += joint_energy + body_energy;
}

__global__ void ApplyDatArticulationKernel(DatTruncateParams p, ModelView model,
                                            DataView data) {
    const uint32_t articulation = blockIdx.x * blockDim.x + threadIdx.x;
    if (articulation >= p.env_count * p.articulations_per_env) return;
    const float beta = __uint_as_float(data.dat_artic_beta[articulation]);
    if (!(beta < 1.0f)) return;
    const uint32_t offset = model.articulation_link_offset[articulation];
    const uint32_t count = model.articulation_link_count[articulation];
    for (uint32_t i = 0u; i < count; ++i) {
        const uint32_t link = offset + i;
        data.q[link] = data.dat_prev_q[link] +
            (data.q[link] - data.dat_prev_q[link]) * beta;
        data.qdot[link] *= beta;
    }
    if (count > 0u && model.parent_link[offset] == ~0u &&
        static_cast<ArticulationJointType>(model.joint_type[offset]) ==
            ArticulationJointType::FloatingBase) {
        data.base_pose[articulation] = DatInterpolatePose(
            data.dat_prev_base_pose[articulation], data.base_pose[articulation], beta);
        auto& velocity = data.link_velocity[offset];
        for (uint32_t i = 0u; i < 6u; ++i) velocity.v[i] *= beta;
    }
}

__global__ void ApplyDatBodyKernel(DatTruncateParams p, ModelView model,
                                    DataView data) {
    const uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= p.env_count * p.bodies_per_env) return;
    const float beta = __uint_as_float(data.dat_body_beta[body]);
    if (!(beta < 1.0f) || !(data.body_inv_mass[body] > 0.0f) ||
        (model.body_collidable_body && model.body_collidable_body[body] != ~0u) ||
        (model.body_collidable_link && model.body_collidable_link[body] != ~0u) ||
        (model.body_to_link && model.body_to_link[body % p.bodies_per_env] != ~0u)) return;
    const auto pose = DatInterpolatePose(data.dat_prev_body_pose[body],
                                         data.body_pose[body], beta);
    data.body_pose[body] = pose;
    data.body_linear_velocity[body] = data.body_linear_velocity[body] * beta;
    data.body_angular_velocity[body] = data.body_angular_velocity[body] * beta;
    data.body_world_inv_inertia[body] = nkops::BodyWorldInverseInertia(
        pose, data.body_inertial_frame[body], data.body_inv_inertia[body]);
}

Status OpDatSnapshot(const ModelView&, const DataView& data,
                     const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const DatTruncateParams*>(params);
    if (!p || p->env_count == 0u) return Status::InvalidArgument;
    const uint32_t count = p->env_count *
        std::max(p->links_per_env, std::max(p->articulations_per_env, p->bodies_per_env));
    if (count == 0u) return Status::Ok;
    if ((p->links_per_env > 0u && (!data.q || !data.dat_prev_q)) ||
        (p->articulations_per_env > 0u && (!data.base_pose || !data.dat_prev_base_pose)) ||
        (p->bodies_per_env > 0u && (!data.body_pose || !data.dat_prev_body_pose)))
        return Status::InvalidArgument;
    LaunchCuda(DatSnapshotKernel, dim3((count + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, *p, data);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

// Counts truncations and signed physical kinetic-energy and momentum losses in fixed order.
__global__ void TallyDatKernel(DatTruncateParams p, DataView data) {
    const uint32_t env = blockIdx.x;
    __shared__ uint32_t counts[kBlockSize];
    __shared__ float energies[kBlockSize];
    __shared__ math::Vec3 momenta[kBlockSize];
    uint32_t count = 0u;
    float energy = 0.0f;
    math::Vec3 momentum{};
    for (uint32_t local = threadIdx.x; local < p.particles_per_env; local += kBlockSize) {
        const uint32_t item = env * p.particles_per_env + local;
        const float beta = data.dat_particle_beta[item];
        if (!(beta < 1.0f)) continue;
        ++count;
        const float inv_mass = data.particle_inv_mass[item];
        if (inv_mass > 0.0f) {
            math::Vec3 before = data.particle_vel[item], after = before * beta;
            if (local >= p.vbd_particle_begin &&
                local - p.vbd_particle_begin < p.vbd_vertices_per_env) {
                const uint32_t slot = env * p.vbd_vertices_per_env + local - p.vbd_particle_begin;
                const math::Vec3 offset = data.vbd_offset[slot];
                const float effective_step = data.vbd_step[slot];
                before = nk::vbd::PhysicalVelocity(before, offset, p.dt, effective_step);
                after = nk::vbd::PhysicalVelocity(after, offset, p.dt, effective_step);
            }
            const double loss =
                (double(before.x) - after.x) * (double(before.x) + after.x) +
                (double(before.y) - after.y) * (double(before.y) + after.y) +
                (double(before.z) - after.z) * (double(before.z) + after.z);
            energy += static_cast<float>(0.5 * loss / inv_mass);
            momentum += (before - after) / inv_mass;
        }
    }
    counts[threadIdx.x] = count;
    energies[threadIdx.x] = energy;
    momenta[threadIdx.x] = momentum;
    __syncthreads();
    for (uint32_t half = kBlockSize / 2u; half > 0u; half /= 2u) {
        if (threadIdx.x < half) {
            counts[threadIdx.x] += counts[threadIdx.x + half];
            energies[threadIdx.x] += energies[threadIdx.x + half];
            momenta[threadIdx.x] += momenta[threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        data.dat_particle_truncation_count[env] = counts[0];
        data.dat_particle_truncation_energy[env] = energies[0];
        data.dat_particle_truncation_momentum[env] = momenta[0];
        data.dat_truncation_count[env] = counts[0];
        data.dat_truncation_energy[env] = energies[0];
    }
}

// The accepted fraction scales the physical and pseudo displacement alike; pseudo stays out of velocity.
__global__ void ApplyDatKernel(DatTruncateParams p, DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.env_count * p.particles_per_env) return;
    const float beta = data.dat_particle_beta[item];
    if (!(beta < 1.0f)) return;
    const math::Vec3 start = data.particle_prev_pos[item];
    data.particle_pos[item] = start + (data.particle_pos[item] - start) * beta;
    data.particle_vel[item] = data.particle_vel[item] * beta;
}

Status OpDatTruncate(const ModelView& model, const DataView& data,
                     const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const DatTruncateParams*>(params);
    if (!p || !(p->dt > 0.0f) || !std::isfinite(p->dt) ||
        !(p->relaxation > 0.0f && p->relaxation < 1.0f) ||
        !(p->margin >= 0.0f) || !std::isfinite(p->margin)) return Status::InvalidArgument;
    if (p->env_count == 0u) return Status::Ok;
    if (uint64_t{p->vbd_particle_begin} + p->vbd_vertices_per_env > p->particles_per_env ||
        (p->vbd_vertices_per_env > 0u && (!data.vbd_offset || !data.vbd_step)))
        return Status::InvalidArgument;
    if ((p->surfaces_per_env > 0u &&
         (!model.particle_surface_info || !model.particle_surface_edge_info ||
          (p->triangles_per_env > 0u && (!model.particle_surface_triangles ||
                                        !data.particle_surface_nodes)) ||
          (p->edges_per_env > 0u && (!model.particle_surface_edges ||
                                    !data.particle_surface_edge_nodes)) ||
          !model.particle_surface_thickness || !data.particle_surface_max_speed ||
          !data.dat_surface_motion ||
          !data.dat_particle_beta || !data.particle_inv_mass ||
          !data.particle_prev_pos || !data.particle_pos || !data.particle_vel)) ||
        (p->bodies_per_env > 0u &&
         (!model.mesh_surface_info || !model.mesh_edge_info || !model.mesh_triangles ||
          !model.mesh_bvh_nodes || !model.shape_table ||
          (p->excluded_pairs > 0u && !model.excluded_pairs) ||
          !model.mesh_edges || !model.hull_verts || !model.mesh_vertex_sources ||
          !data.body_pose || !data.dat_prev_body_pose || !data.dat_body_beta ||
          !data.dat_body_speed || !data.dat_body_query_radius ||
          !data.body_inv_mass || !data.body_linear_velocity ||
          !data.body_angular_velocity || !data.body_inertial_frame ||
          !data.body_inv_inertia || !data.body_world_inv_inertia)) ||
        (p->articulations_per_env > 0u &&
         (!model.link_to_articulation || !model.articulation_link_offset ||
          !model.articulation_link_count || !model.parent_link ||
          !model.joint_type || !model.joint_axis || !model.link_local_pose ||
          !model.parent_offset || !model.link_geom_kind || !model.link_geom_local ||
          !data.q || !data.qdot || !data.dat_prev_q || !data.base_pose ||
          !data.dat_prev_base_pose || !data.link_velocity ||
          !data.dat_artic_beta || !data.m || !data.qdot_flat)) ||
        (p->slot_capacity > 0u &&
         (!data.ogc_contact_count || !data.dat_pair_kind ||
          !data.dat_pair_owner_a || !data.dat_pair_owner_b ||
          !data.dat_pair_feature_a || !data.dat_pair_feature_b)) ||
        !data.dat_truncation_count || !data.dat_truncation_energy ||
        !data.dat_particle_truncation_count || !data.dat_particle_truncation_energy ||
        !data.dat_particle_truncation_momentum ||
        !data.dat_joint_truncation_count || !data.dat_joint_truncation_energy ||
        !data.dat_body_truncation_count || !data.dat_body_truncation_energy ||
        !data.dat_failure_count || !data.dat_query_limit_count ||
        !data.dat_failure_witness || !data.env_status) {
        return Status::InvalidArgument;
    }
    const uint64_t particles = uint64_t{p->env_count} * p->particles_per_env;
    const uint64_t edges = uint64_t{p->env_count} * p->edges_per_env;
    const uint64_t triangles = uint64_t{p->env_count} * p->triangles_per_env;
    const uint64_t body_vertices = uint64_t{p->env_count} * p->mesh_vertex_sources;
    const uint64_t slots = uint64_t{p->env_count} * p->slot_capacity;
    if (particles > UINT32_MAX || edges > UINT32_MAX || triangles > UINT32_MAX ||
        body_vertices > UINT32_MAX || slots > UINT32_MAX ||
        uint64_t{p->env_count} * (p->edges_per_env + p->mesh_edge_sources) > UINT32_MAX ||
        p->slot_base > p->slot_stride ||
        p->slot_capacity > p->slot_stride - p->slot_base)
        return Status::InvalidArgument;
    const dim3 particle_grid((static_cast<uint32_t>(particles) + kBlockSize - 1u) / kBlockSize);
    if (cudaMemsetAsync(data.dat_failure_witness, 0, size_t{p->env_count} *
            collision::kDatWitnessWords * sizeof(uint64_t), stream) != cudaSuccess)
        return Status::Failed;
    LaunchCuda(ClearDatCountersKernel, dim3((p->env_count + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, *p, model, data);
    const uint32_t owners = p->env_count *
        std::max(p->bodies_per_env, std::max(p->articulations_per_env, 1u));
    LaunchCuda(ClearDatOwnersKernel, dim3((owners + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, *p, data);
    if (particles > 0u && p->surfaces_per_env > 0u)
        LaunchCuda(ClearDatKernel, particle_grid, dim3(kBlockSize), 0u, stream, *p, model, data);
    if (triangles > 0u)
        LaunchCuda(DatTriangleValidityKernel,
                   dim3((static_cast<uint32_t>(triangles) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (particles > 0u && p->surfaces_per_env > 0u)
        LaunchCuda(DatVertexFaceKernel, particle_grid, dim3(kBlockSize), 0u, stream, *p, model, data);
    if (edges > 0u)
        LaunchCuda(DatEdgeEdgeKernel,
                   dim3((static_cast<uint32_t>(edges) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (edges > 0u)
        LaunchCuda(DatInitialEdgeFaceKernel,
                   dim3((static_cast<uint32_t>(edges) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    const uint64_t mixed_edges = uint64_t{p->env_count} *
        (p->edges_per_env + p->mesh_edge_sources);
    if (p->bodies_per_env > 0u && mixed_edges > 0u)
        LaunchCuda(DatInitialMixedEdgeFaceKernel,
                   dim3((static_cast<uint32_t>(mixed_edges) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (body_vertices > 0u)
        LaunchCuda(DatBodyMotionBoundKernel,
                   dim3((static_cast<uint32_t>(body_vertices) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (slots > 0u)
        LaunchCuda(DatMixedPairsKernel,
                   dim3((static_cast<uint32_t>(slots) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    LaunchCuda(TallyDatKernel, dim3(p->env_count), dim3(kBlockSize), 0u, stream, *p, data);
    if (p->bodies_per_env > 0u || p->articulations_per_env > 0u)
        LaunchCuda(TallyDatOwnersKernel,
                   dim3((p->env_count + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (particles > 0u)
        LaunchCuda(ApplyDatKernel, particle_grid, dim3(kBlockSize), 0u, stream, *p, data);
    if (p->articulations_per_env > 0u) {
        const uint32_t count = p->env_count * p->articulations_per_env;
        LaunchCuda(ApplyDatArticulationKernel,
                   dim3((count + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    }
    if (p->bodies_per_env > 0u) {
        const uint32_t count = p->env_count * p->bodies_per_env;
        LaunchCuda(ApplyDatBodyKernel, dim3((count + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    }
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkDatTruncateOps() {
    SetCudaOp(NkOp::DatSnapshot, &OpDatSnapshot);
    SetCudaOp(NkOp::DatTruncate, &OpDatTruncate);
}

}  // namespace nuka::phi
