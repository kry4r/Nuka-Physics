#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "collision/dat_geometry.hpp"
#include "collision/convex_narrowphase.hpp"
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
#include "phi/backend_cuda/ops/bvh_work_share.cuh"
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

// DatFindSeparator(a, b).gap > 0. Each axis gap is independent of the others, so any order decides;
// the face normal of b and the edge pairs come first because they separate most nearby pairs.
__device__ bool DatSeparated(const collision::DatPrimitive& a, const collision::DatPrimitive& b) {
    if (!collision::DatPrimitiveValid(a) || !collision::DatPrimitiveValid(b)) return false;
    collision::DatSeparator best;
    const auto separates = [&](math::Vec3 axis) {
        collision::DatConsiderAxis(axis, a, b, &best);
        return best.gap > 0.0f;
    };
    if (b.count == collision::kDatPrimitiveVertices &&
        separates((b.vertex[1] - b.vertex[0]).Cross(b.vertex[2] - b.vertex[0]))) return true;
    const uint32_t edges_a = a.count == 1u ? 0u : a.count == 2u ? 1u : a.count;
    const uint32_t edges_b = b.count == 1u ? 0u : b.count == 2u ? 1u : b.count;
    for (uint32_t i = 0u; i < edges_a; ++i) {
        const math::Vec3 edge = a.vertex[(i + 1u) % a.count] - a.vertex[i];
        for (uint32_t j = 0u; j < edges_b; ++j)
            if (separates(edge.Cross(b.vertex[(j + 1u) % b.count] - b.vertex[j]))) return true;
    }
    for (uint32_t i = 0u; i < a.count; ++i)
        for (uint32_t j = 0u; j < b.count; ++j)
            if (separates(a.vertex[i] - b.vertex[j])) return true;
    for (uint32_t i = 0u; i < edges_a; ++i) {
        const math::Vec3 edge = a.vertex[(i + 1u) % a.count] - a.vertex[i];
        for (uint32_t j = 0u; j < b.count; ++j) {
            const math::Vec3 offset = a.vertex[i] - b.vertex[j];
            if (separates(edge.Cross(offset.Cross(edge)))) return true;
        }
    }
    for (uint32_t j = 0u; j < edges_b; ++j) {
        const math::Vec3 edge = b.vertex[(j + 1u) % b.count] - b.vertex[j];
        for (uint32_t i = 0u; i < a.count; ++i) {
            const math::Vec3 offset = a.vertex[i] - b.vertex[j];
            if (separates(edge.Cross(offset.Cross(edge)))) return true;
        }
    }
    return a.count == collision::kDatPrimitiveVertices &&
           separates((a.vertex[1] - a.vertex[0]).Cross(a.vertex[2] - a.vertex[0]));
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

__device__ bool DatBodyPairAllowed(DatTruncateParams p, ModelView model,
                                   uint32_t a, uint32_t b, bool include_convex = false) {
    if (a == b) return false;
    const auto sa = nkops::LoadPrimShape(model.shape_table, a);
    const auto sb = nkops::LoadPrimShape(model.shape_table, b);
    const auto route = nkops::RouteMeshContact(sa.kind, sb.kind, model.mesh_contact_mode[a],
                                              model.mesh_contact_mode[b]);
    if (route != nkops::MeshContactRoute::Ogc && !(include_convex &&
        route == nkops::MeshContactRoute::Analytic && collision::TakesConvexManifold(sa.kind, sb.kind)))
        return false;
    if (nkops::AuthoredPairProfile(model.contact_pair_keys, model.contact_pair_profiles, p.contact_pairs, a, b) != ~0u)
        return true;
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

__global__ void ClearDatCountersKernel(DatTruncateParams p, ModelView model, DataView data) {
    const uint32_t env = blockIdx.x;
    if (env == 0u && threadIdx.x < 2u && p.mesh_vertex_sources > 0u)
        reinterpret_cast<uint32_t*>(data.contact_cache_scratch)[threadIdx.x] = 0u;
    for (uint32_t word = threadIdx.x; word < collision::kDatWitnessWords; word += blockDim.x)
        data.dat_failure_witness[size_t{env} * collision::kDatWitnessWords + word] = 0u;
    __syncthreads();
    bool capped = false;
    if (threadIdx.x == 0u) {
        data.dat_failure_count[env] = 0u;
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
    }
    __syncthreads();
    const uint64_t mixed = uint64_t{p.surfaces_per_env} * p.bodies_per_env;
    for (uint64_t item = threadIdx.x; item < mixed; item += blockDim.x) {
        const uint32_t source = static_cast<uint32_t>(item / p.bodies_per_env);
        const uint32_t body = static_cast<uint32_t>(item % p.bodies_per_env);
        if (model.mesh_surface_info[body].vertex_count == 0u) continue;
        const float base = model.particle_surface_thickness[source] + p.margin;
        const float radius = collision::DatMotionRadius(base, p.dt,
            data.particle_surface_max_speed[size_t{env} * p.surfaces_per_env + source],
            data.dat_body_speed[size_t{env} * p.bodies_per_env + body], p.relaxation);
        if (!(radius >= 0.0f && radius <= FLT_MAX) ||
            !(base <= collision::kDatQueryRadiusMax)) {
            atomicOr(data.env_status + env, kEnvStatusDatFailure);
            atomicAdd(data.dat_failure_count + env, 1u);
            RecordDatFailure(data, env, base <= collision::kDatQueryRadiusMax
                ? collision::kDatFailureMotion : collision::kDatFailureRadius,
                collision::DatWitnessOwner(collision::kDatWitnessSurface, source),
                DatBodyWitness(p, model, env, body), 0.0f);
        }
        capped |= radius > collision::kDatQueryRadiusMax;
    }
    capped = __syncthreads_or(capped) != 0;
    if (threadIdx.x == 0u) data.dat_query_limit_count[env] = capped;
}

// Body pairs of the environments, a thread each after ClearDatCountersKernel; a capped pair raises its
// environment's query limit.
__global__ void DatBodyPairLimitKernel(DatTruncateParams p, ModelView model, DataView data) {
    const uint32_t env = blockIdx.y;
    const uint64_t item = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const uint32_t source = static_cast<uint32_t>(item / p.bodies_per_env);
    const uint32_t target = static_cast<uint32_t>(item % p.bodies_per_env);
    bool capped = false;
    if (item < uint64_t{p.bodies_per_env} * p.bodies_per_env && target > source &&
        model.mesh_surface_info[source].vertex_count != 0u &&
        model.mesh_surface_info[target].vertex_count != 0u &&
        DatBodyPairAllowed(p, model, source, target)) {
        const uint32_t frame = DatCommonFrame(p, model, env, source, target);
        const float radius = frame != ~0u
            ? DatArticMotionRadius(p, model, data, env, source, target, frame, data.dat_prev_q)
            : collision::DatMotionRadius(0.0005f + p.margin, p.dt,
                  data.dat_body_speed[size_t{env} * p.bodies_per_env + source],
                  data.dat_body_speed[size_t{env} * p.bodies_per_env + target], p.relaxation);
        if (!(radius >= 0.0f && radius <= FLT_MAX)) {
            atomicOr(data.env_status + env, kEnvStatusDatFailure);
            atomicAdd(data.dat_failure_count + env, 1u);
            RecordDatFailure(data, env, collision::kDatFailureMotion,
                             DatBodyWitness(p, model, env, source),
                             DatBodyWitness(p, model, env, target), 0.0f);
        }
        capped = radius > collision::kDatQueryRadiusMax;
    }
    if (__syncthreads_or(capped) != 0 && threadIdx.x == 0u) atomicOr(data.dat_query_limit_count + env, 1u);
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

// A primitive pair one lane's traversal reached, tested by whichever lane of its warp is free.
struct DatCandidate {
    uint32_t a[2];
    uint32_t b[3];
    uint32_t env;
    float query;
};

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
            const float radius = collision::DatMotionRadius(
                model.particle_surface_thickness[source] + p.margin, p.dt,
                data.particle_surface_max_speed[size_t{env} * p.surfaces_per_env + source],
                data.dat_body_speed[size_t{env} * p.bodies_per_env + body], p.relaxation);
            bound = fminf(bound, 0.5f * p.relaxation *
                collision::DatQueryRadius(radius));
        }
    }
    const float motion = sqrtf((data.particle_pos[item] - data.particle_prev_pos[item]).LengthSq());
    const float beta = (data.env_status[env] & kEnvStatusDatFailure) != 0u
        ? 0.0f : motion > bound ? bound / motion : 1.0f;
    data.dat_particle_beta[item] = beta;
    data.dat_particle_parent[item] = item;
}

__global__ void DatVertexFaceKernel(DatTruncateParams p, ModelView model, DataView data) {
    __shared__ DatCandidate queue[kBlockSize / kWarpLanes][2u * kWarpLanes];
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t item = thread / kTreeShares, share = thread % kTreeShares;
    bool active = item < p.env_count * p.particles_per_env;
    const uint32_t env = item / p.particles_per_env;
    const uint32_t local = item % p.particles_per_env;
    const auto view = ReferenceView(p, model, data, env);
    // Surfaces partition the vertices, so at most one source surface holds this one.
    uint32_t source = 0u;
    for (; active && source < p.surfaces_per_env; ++source) {
        const auto info = model.particle_surface_info[source];
        if (local >= info.vertex_offset && local - info.vertex_offset < info.vertex_count) break;
    }
    active = active && source < p.surfaces_per_env;
    const math::Vec3 vertex = active ? data.particle_prev_pos[item] : math::Vec3{};
    uint32_t target = 0u, cursor = 0u, end = 0u;
    collision::MeshSurfaceInfo info{};
    float query = 0.0f;
    const auto enter = [&]() {
        if (target >= p.surfaces_per_env) return;
        info = model.particle_surface_info[target];
        query = QueryRadius(p, model, data, env, source, target);
        if (!TreeShare(view.nodes + info.node_offset, info.node_count, share, &cursor, &end))
            atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
    };
    if (active) enter();
    const auto next = [&](DatCandidate* candidate) {
        while (target < p.surfaces_per_env) {
            if (cursor >= end) {
                ++target;
                enter();
                continue;
            }
            const auto& node = view.nodes[info.node_offset + cursor];
            if (node.escape <= cursor || node.escape > info.node_count) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                cursor = end;
                continue;
            }
            if (collision::MeshBoundsDistanceSquared(vertex, node) > query * query) {
                cursor = node.escape;
                continue;
            }
            ++cursor;
            if (node.triangle == ~0u) continue;
            if (node.triangle >= info.triangle_count) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                cursor = end;
                continue;
            }
            const size_t at = (size_t{info.triangle_offset} + node.triangle) * 3u;
            bool incident = false;
            for (uint32_t j = 0u; j < 3u; ++j) {
                candidate->b[j] = env * p.particles_per_env + info.vertex_offset +
                                  view.triangles[at + j];
                incident |= candidate->b[j] == item;
            }
            if (incident) continue;
            candidate->a[0] = item;
            candidate->env = env;
            candidate->query = query;
            return true;
        }
        return false;
    };
    TestInWarpBatches(queue[threadIdx.x / kWarpLanes], active, next, [&](const DatCandidate& c) {
        const math::Vec3 point = data.particle_prev_pos[c.a[0]];
        const auto closest = collision::ClosestTrianglePoint(point, data.particle_prev_pos[c.b[0]],
            data.particle_prev_pos[c.b[1]], data.particle_prev_pos[c.b[2]]);
        if ((closest.point - point).LengthSq() <= c.query * c.query)
            ClipPrimitivePair(p, model, data, c.env, c.a, 1u, c.b, 3u);
    });
}

__global__ void DatEdgeEdgeKernel(DatTruncateParams p, ModelView model, DataView data) {
    __shared__ DatCandidate queue[kBlockSize / kWarpLanes][2u * kWarpLanes];
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t item = thread / kTreeShares, share = thread % kTreeShares;
    bool active = item < p.env_count * p.edges_per_env;
    const uint32_t env = item / p.edges_per_env;
    const uint32_t local = item % p.edges_per_env;
    const auto* nodes = data.particle_surface_edge_nodes + size_t{env} * p.edge_nodes_per_env;
    // Surfaces partition the edges, so at most one source surface holds this one.
    uint32_t source = 0u;
    for (; active && source < p.surfaces_per_env; ++source) {
        const auto edges = model.particle_surface_edge_info[source];
        if (local >= edges.edge_offset && local - edges.edge_offset < edges.edge_count) break;
    }
    active = active && source < p.surfaces_per_env;
    uint32_t ids_a[2] = {};
    math::Vec3 midpoint{};
    float half_length = 0.0f;
    if (active) {
        const auto edge_a = model.particle_surface_edges[local];
        const uint32_t base_a = env * p.particles_per_env +
                                model.particle_surface_info[source].vertex_offset;
        ids_a[0] = base_a + edge_a.vertex0;
        ids_a[1] = base_a + edge_a.vertex1;
        const math::Vec3 a0 = data.particle_prev_pos[ids_a[0]];
        const math::Vec3 a1 = data.particle_prev_pos[ids_a[1]];
        midpoint = (a0 + a1) * 0.5f;
        half_length = sqrtf((a1 - a0).LengthSq()) * 0.5f;
    }
    uint32_t target = source, cursor = 0u, end = 0u, base_b = 0u;
    collision::MeshEdgeInfo edges_b{};
    float query = 0.0f, broad_radius = 0.0f;
    const auto enter = [&]() {
        if (target >= p.surfaces_per_env) return;
        edges_b = model.particle_surface_edge_info[target];
        base_b = env * p.particles_per_env + model.particle_surface_info[target].vertex_offset;
        query = QueryRadius(p, model, data, env, source, target);
        broad_radius = query + half_length;
        if (!TreeShare(nodes + edges_b.node_offset, edges_b.node_count, share, &cursor, &end))
            atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
    };
    if (active) enter();
    const auto next = [&](DatCandidate* candidate) {
        while (target < p.surfaces_per_env) {
            if (cursor >= end) {
                ++target;
                enter();
                continue;
            }
            const auto& node = nodes[edges_b.node_offset + cursor];
            if (node.escape <= cursor || node.escape > edges_b.node_count) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                cursor = end;
                continue;
            }
            if (collision::MeshBoundsDistanceSquared(midpoint, node) > broad_radius * broad_radius) {
                cursor = node.escape;
                continue;
            }
            ++cursor;
            if (node.triangle == ~0u) continue;
            if (node.triangle >= edges_b.edge_count) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                cursor = end;
                continue;
            }
            const uint32_t target_edge = edges_b.edge_offset + node.triangle;
            if (target_edge <= local) continue;
            const auto edge_b = model.particle_surface_edges[target_edge];
            const uint32_t ids_b[2] = {base_b + edge_b.vertex0, base_b + edge_b.vertex1};
            if (ids_a[0] == ids_b[0] || ids_a[0] == ids_b[1] ||
                ids_a[1] == ids_b[0] || ids_a[1] == ids_b[1]) continue;
            *candidate = {{ids_a[0], ids_a[1]}, {ids_b[0], ids_b[1], 0u}, env, query};
            return true;
        }
        return false;
    };
    TestInWarpBatches(queue[threadIdx.x / kWarpLanes], active, next, [&](const DatCandidate& c) {
        const auto closest = collision::OgcClosestSegments(
            data.particle_prev_pos[c.a[0]], data.particle_prev_pos[c.a[1]],
            data.particle_prev_pos[c.b[0]], data.particle_prev_pos[c.b[1]]);
        if ((closest.a - closest.b).LengthSq() <= c.query * c.query)
            ClipPrimitivePair(p, model, data, c.env, c.a, 2u, c.b, 2u);
    });
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
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t item = thread / kTreeShares, share = thread % kTreeShares;
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
            if (share == 0u)
                FailPrimitivePair(p, model, data, env, collision::kDatFailureDegenerate, 0.0f,
                                  ids_a, 2u, ids_a, 0u);
            return;
        }
        const math::Vec3 midpoint = (segment.vertex[0] + segment.vertex[1]) * 0.5f;
        const float half_length = sqrtf((segment.vertex[1] - segment.vertex[0]).LengthSq()) * 0.5f;
        for (uint32_t target = 0u; target < p.surfaces_per_env; ++target) {
            const auto info_b = model.particle_surface_info[target];
            uint32_t cursor = 0u, end = 0u;
            if (!TreeShare(view.nodes + info_b.node_offset, info_b.node_count, share, &cursor, &end)) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                continue;
            }
            while (cursor < end) {
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
                            !DatSeparated(segment, triangle))
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
    const uint32_t particles = p.env_count * p.particles_per_env;
    if (item < bodies) {
        data.dat_body_beta[item] = __float_as_uint(1.0f);
        data.dat_body_motion[item] = 0.0f;
        data.dat_body_parent[item] = particles + item;
    }
    if (item < articulations) {
        data.dat_artic_beta[item] = __float_as_uint(1.0f);
        data.dat_artic_parent[item] = particles + bodies + item;
        if (data.dat_artic_witness != nullptr)
            for (uint32_t k = 0u; k < kDatBoundKinds; ++k)
                data.dat_artic_witness[size_t{item} * kDatBoundKinds + k] = ~0ull;
    }
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
        DatMinOwnerBeta(p, model, data, env, ref, 0.0f, kDatBoundMotion, vertex);
        return;
    }
    // Non-negative float bits order like the values, so the maximum is order independent.
    atomicMax(reinterpret_cast<uint32_t*>(data.dat_body_motion) + size_t{env} * p.bodies_per_env +
              body, __float_as_uint(speed / p.dt));
    // A vertex on a triangle past this bound is certified with its triangles by the reach pass.
    const float bound = 0.5f * p.relaxation * query;
    const uint32_t at = model.mesh_surface_info[body].vertex_offset + vertex;
    if (speed > bound &&
        model.mesh_vertex_triangle_offsets[at + 1u] == model.mesh_vertex_triangle_offsets[at])
        DatMinOwnerBeta(p, model, data, env, ref, bound / speed, kDatBoundMotion, vertex);
}

__device__ void FailMixedPair(DatTruncateParams p, ModelView model, DataView data,
                               uint32_t env, uint32_t reason, float depth, const DatPointRef* a,
                               uint32_t count_a, const DatPointRef* b, uint32_t count_b) {
    atomicOr(data.env_status + env, kEnvStatusDatFailure);
    atomicAdd(data.dat_failure_count + env, 1u);
    RecordDatFailure(data, env, reason, DatRefWitness(p, model, env, a[0]), count_b > 0u
        ? DatRefWitness(p, model, env, b[0]) : collision::kDatWitnessNone, depth);
    const bool particle_a = a[0].particle, particle_b = count_b > 0u && b[0].particle;
    for (uint32_t i = 0u; i < count_a; ++i)
        DatMinOwnerBeta(p, model, data, env, a[i], 0.0f,
            particle_b ? kDatBoundParticlePair : kDatBoundBodyPair,
            count_b > 0u ? DatCounterpart(p, b[0]) : kDatNoCounterpart);
    for (uint32_t i = 0u; i < count_b; ++i)
        DatMinOwnerBeta(p, model, data, env, b[i], 0.0f,
            particle_a ? kDatBoundParticlePair : kDatBoundBodyPair, DatCounterpart(p, a[0]));
}

__device__ collision::MeshSurfaceView DatOldBodyView(
    DatTruncateParams p, ModelView model, DataView data,
    uint32_t env, uint32_t body) {
    const auto pose = data.dat_prev_body_pose[size_t{env} * p.bodies_per_env + body];
    return {model.hull_verts, model.mesh_triangles, model.mesh_bvh_nodes,
            {p.mesh_vertices, p.mesh_triangles, p.mesh_nodes},
            pose.position, pose.rotation, true};
}

// Whether segment ab meets the node box widened by the 1 um slack of the bounding sphere it refines.
__device__ bool DatSegmentNearBounds(math::Vec3 a, math::Vec3 b, const collision::MeshBvhNode& node) {
    return collision::SegmentWithinBox(a, b, node.lower, node.upper, 1.0e-6f);
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
    if (source_particle) {
        segment.vertex[0] = DatPointAt(p, model, data, env, source[0], 0.0f);
        segment.vertex[1] = DatPointAt(p, model, data, env, source[1], 0.0f);
    } else {
        const auto pose = DatBodyPoseAt(p, model, data, env, source_body, 0.0f);
        for (uint32_t j = 0u; j < 2u; ++j)
            segment.vertex[j] = pose.position + math::gpu::RotateByQuatNormalized(
                pose.rotation, DatBodyLocalPoint(model, source_body, source[j].vertex));
    }
    if (!collision::DatPrimitiveValid(segment)) {
        FailMixedPair(p, model, data, env, collision::kDatFailureDegenerate, 0.0f,
                      source, 2u, source, 0u);
        return;
    }
    for (uint32_t group = 0u; group < (source_particle ? 1u : 2u); ++group) {
        const bool target_particle = !source_particle && group == 0u;
        const uint32_t targets = target_particle ? p.surfaces_per_env : p.bodies_per_env;
        for (uint32_t target = 0u; target < targets; ++target) {
            const auto info = target_particle ? model.particle_surface_info[target]
                                              : model.mesh_surface_info[target];
            const auto view = target_particle ? ReferenceView(p, model, data, env)
                : DatOldBodyView(p, model, data, env, target);
            if (info.node_count == 0u || !collision::MeshSurfaceRangeValid(view, info)) continue;
            const auto local_a = collision::MeshSurfaceLocalPoint(view, segment.vertex[0]);
            const auto local_b = collision::MeshSurfaceLocalPoint(view, segment.vertex[1]);
            if (!DatSegmentNearBounds(local_a, local_b, view.nodes[info.node_offset])) continue;
            if (!target_particle && !source_particle &&
                !DatBodyPairAllowed(p, model, source_body, target)) continue;
            const auto pose = target_particle ? math::Transform{}
                : DatBodyPoseAt(p, model, data, env, target, 0.0f);
            uint32_t cursor = 0u;
            while (cursor < info.node_count) {
                const auto node = view.nodes[info.node_offset + cursor];
                if (node.escape <= cursor || node.escape > info.node_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    break;
                }
                if (!DatSegmentNearBounds(local_a, local_b, node)) {
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
                            triangle.vertex[j] = target_particle
                                ? DatPointAt(p, model, data, env, face[j], 0.0f)
                                : pose.position + math::gpu::RotateByQuatNormalized(
                                      pose.rotation, DatBodyLocalPoint(model, target, face[j].vertex));
                        const uint32_t reason = collision::DatPrimitiveValid(triangle)
                            ? collision::kDatFailureOverlap : collision::kDatFailureDegenerate;
                        if (reason == collision::kDatFailureDegenerate ||
                            !DatSeparated(segment, triangle))
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

// Owners are union-find nodes: particles, then bodies, then articulations. A join hooks the larger
// root under the smaller one, so parents only decrease and every find ends.
__device__ uint32_t* DatParentSlot(const DatTruncateParams& p, const DataView& data, uint32_t node) {
    const uint32_t particles = p.env_count * p.particles_per_env;
    const uint32_t bodies = p.env_count * p.bodies_per_env;
    if (node < particles) return data.dat_particle_parent + node;
    if (node < particles + bodies) return data.dat_body_parent + (node - particles);
    return data.dat_artic_parent + (node - particles - bodies);
}

__device__ uint32_t* DatBetaSlot(const DatTruncateParams& p, const DataView& data, uint32_t node) {
    const uint32_t particles = p.env_count * p.particles_per_env;
    const uint32_t bodies = p.env_count * p.bodies_per_env;
    if (node < particles) return reinterpret_cast<uint32_t*>(data.dat_particle_beta + node);
    if (node < particles + bodies) return data.dat_body_beta + (node - particles);
    return data.dat_artic_beta + (node - particles - bodies);
}

__device__ uint32_t DatFind(const DatTruncateParams& p, const DataView& data, uint32_t node) {
    while (true) {
        const uint32_t parent = *reinterpret_cast<volatile uint32_t*>(DatParentSlot(p, data, node));
        if (parent == node) return node;
        node = parent;
    }
}

__device__ void DatJoin(const DatTruncateParams& p, const DataView& data, uint32_t a, uint32_t b) {
    while (true) {
        a = DatFind(p, data, a);
        b = DatFind(p, data, b);
        if (a == b) return;
        const uint32_t child = a > b ? a : b, root = a > b ? b : a;
        if (atomicCAS(DatParentSlot(p, data, child), child, root) == child) return;
    }
}

// The node whose fraction moves a point, as DatMinOwnerBeta resolves it; false for a body that
// no fraction moves.
__device__ bool DatOwnerNode(const DatTruncateParams& p, const ModelView& model, const DataView& data,
                             uint32_t env, DatPointRef ref, uint32_t* node) {
    const uint32_t particles = p.env_count * p.particles_per_env;
    const uint32_t bodies = p.env_count * p.bodies_per_env;
    if (ref.particle) {
        *node = ref.owner;
        return true;
    }
    uint32_t link = ~0u;
    math::Transform local{};
    if (DatBodyLink(p, model, env, ref.owner, &link, &local)) {
        *node = particles + bodies + model.link_to_articulation[link];
        return true;
    }
    const uint32_t global = env * p.bodies_per_env + ref.owner;
    const uint32_t owner_local = model.body_collidable_body ? model.body_collidable_body[global] : ~0u;
    const uint32_t owner = owner_local == ~0u ? global : env * p.bodies_per_env + owner_local;
    *node = particles + owner;
    return data.body_inv_mass[owner] > 0.0f;
}

// Projection gap of the two sides on a fixed axis, less the margin, at step fraction s.
__device__ float DatAxisGapAt(const DatTruncateParams& p, const ModelView& model, const DataView& data,
                              uint32_t env, const DatPointRef* refs_a, uint32_t count_a,
                              const DatPointRef* refs_b, uint32_t count_b, math::Vec3 normal,
                              float margin, float s, uint32_t frame,
                              const math::Transform* pose_a = nullptr,
                              const math::Transform* pose_b = nullptr) {
    float lower = FLT_MAX, upper = -FLT_MAX;
    const auto a = DatPrimitiveAt(p, model, data, env, refs_a, count_a, s, frame, pose_a);
    const auto b = DatPrimitiveAt(p, model, data, env, refs_b, count_b, s, frame, pose_b);
    for (uint32_t i = 0u; i < count_a; ++i)
        lower = fminf(lower, a.vertex[i].Dot(normal));
    for (uint32_t i = 0u; i < count_b; ++i)
        upper = fmaxf(upper, b.vertex[i].Dot(normal));
    return lower - upper - margin;
}

// True when the projection gap stays above the margin for the whole step, by Lipschitz subdivision.
__device__ bool DatAxisHolds(const DatTruncateParams& p, const ModelView& model, const DataView& data,
                             uint32_t env, const DatPointRef* refs_a, uint32_t count_a,
                             const DatPointRef* refs_b, uint32_t count_b, math::Vec3 normal,
                             float margin, float speed, uint32_t frame, float curvature, float rounding,
                             const DatPairPoses* poses) {
    if (!(speed < FLT_MAX)) return false;
    float values[5];
    for (uint32_t i = 0u; i <= 4u; ++i) {
        values[i] = DatAxisGapAt(p, model, data, env, refs_a, count_a, refs_b, count_b, normal, margin,
            0.25f * i, frame,
            poses ? (i == 0u ? &poses->a_start : i == 4u ? &poses->a_end : &poses->a_quarters[i - 1u]) : nullptr,
            poses ? (i == 0u ? &poses->b_start : i == 4u ? &poses->b_end : &poses->b_quarters[i - 1u]) : nullptr);
        if (!(values[i] > 0.0f)) return false;
    }
    DatInterval stack[20];
    uint32_t top = 0u;
    for (uint32_t i = 4u; i > 0u; --i) stack[top++] = {0.25f * (i - 1u), 0.25f * i, values[i - 1u], values[i], 0u};
    while (top > 0u) {
        const auto interval = stack[--top];
        const float width = interval.end - interval.start;
        if (fminf(interval.value_start, interval.value_end) >
            fminf(0.5f * speed * width, curvature * width * width + rounding)) continue;
        if (interval.depth == 12u) return false;
        const float mid = 0.5f * (interval.start + interval.end);
        const float value_mid = DatAxisGapAt(p, model, data, env, refs_a, count_a, refs_b, count_b, normal,
                                             margin, mid, frame);
        if (!(value_mid > 0.0f)) return false;
        stack[top++] = {mid, interval.end, value_mid, interval.value_end, interval.depth + 1u};
        stack[top++] = {interval.start, mid, interval.value_start, value_mid, interval.depth + 1u};
    }
    return true;
}

// Joined owners share a step fraction, preserving separation certified along their paths.
// A stationary side needs no join; a moving side must have a truncatable owner.
__device__ bool DatJoinPair(const DatTruncateParams& p, const ModelView& model, const DataView& data,
                            uint32_t env, const DatPointRef* refs_a, uint32_t count_a,
                            const DatPointRef* refs_b, uint32_t count_b, float bound_a, float bound_b,
                            const collision::DatSeparator& separator, float minimum_gap, uint32_t frame,
                            float curvature, float rounding, const DatPairPoses* poses) {
    uint32_t nodes[6];
    uint32_t count = 0u;
    for (uint32_t side = 0u; side < 2u; ++side) {
        const DatPointRef* refs = side == 0u ? refs_a : refs_b;
        const uint32_t points = side == 0u ? count_a : count_b;
        if ((side == 0u ? bound_a : bound_b) == 0.0f) continue;
        for (uint32_t i = 0u; i < points; ++i)
            if (!DatOwnerNode(p, model, data, env, refs[i], &nodes[count++])) return false;
    }
    if (!DatAxisHolds(p, model, data, env, refs_a, count_a, refs_b, count_b, separator.normal,
                      minimum_gap, bound_a + bound_b, frame, curvature, rounding, poses)) return false;
    if (bound_a > 0.0f && bound_b > 0.0f)
        for (uint32_t i = 1u; i < count; ++i) DatJoin(p, data, nodes[0], nodes[i]);
    return true;
}

// Each primitive keeps its share of a separating gap with a reserve for FK rounding.
// Otherwise its owner is truncated, unless joined owners preserve the gap over the whole step.
__device__ void DatCertifyPair(const DatTruncateParams& p, const ModelView& model,
                               const DataView& data, uint32_t env,
                               const DatPointRef* refs_a, uint32_t count_a,
                               const DatPointRef* refs_b, uint32_t count_b, uint32_t frame,
                               const DatPairPoses* poses = nullptr) {
    const auto motion_a = poses ? poses->a_motion : refs_a[0].particle ? DatBodyMotion{}
        : DatBodyMotionBounds(p, model, data, env, refs_a[0].owner, frame);
    const auto motion_b = poses ? poses->b_motion : refs_b[0].particle ? DatBodyMotion{}
        : DatBodyMotionBounds(p, model, data, env, refs_b[0].owner, frame);
    const auto a = DatPrimitiveAt(p, model, data, env, refs_a, count_a, 0.0f, frame,
                                  poses ? &poses->a_start : nullptr);
    const auto b = DatPrimitiveAt(p, model, data, env, refs_b, count_b, 0.0f, frame,
                                  poses ? &poses->b_start : nullptr);
    if (!collision::DatPrimitiveValid(a) || !collision::DatPrimitiveValid(b)) {
        FailMixedPair(p, model, data, env, collision::kDatFailureDegenerate, 0.0f,
            refs_a, count_a, refs_b, count_b);
        return;
    }
    const auto end_a = DatPrimitiveAt(p, model, data, env, refs_a, count_a, 1.0f, frame,
                                      poses ? &poses->a_end : nullptr);
    const auto end_b = DatPrimitiveAt(p, model, data, env, refs_b, count_b, 1.0f, frame,
                                      poses ? &poses->b_end : nullptr);
    float speed_a[3], speed_b[3], curves_a[3], curves_b[3];
    float bound_a = 0.0f, bound_b = 0.0f, curve_a = 0.0f, curve_b = 0.0f;
    for (uint32_t i = 0u; i < count_a; ++i) {
        float curve = 0.0f;
        speed_a[i] = DatPointLipschitz(p, model, data, env, refs_a[i], frame, &curve,
                                      &a.vertex[i], &end_a.vertex[i], &motion_a);
        curves_a[i] = curve;
        if (!(speed_a[i] >= 0.0f && speed_a[i] < FLT_MAX)) {
            FailMixedPair(p, model, data, env, collision::kDatFailureMotion, 0.0f,
                          refs_a, count_a, refs_b, count_b);
            return;
        }
        bound_a = fmaxf(bound_a, speed_a[i]);
        curve_a = fmaxf(curve_a, curve);
    }
    for (uint32_t i = 0u; i < count_b; ++i) {
        float curve = 0.0f;
        speed_b[i] = DatPointLipschitz(p, model, data, env, refs_b[i], frame, &curve,
                                      &b.vertex[i], &end_b.vertex[i], &motion_b);
        curves_b[i] = curve;
        if (!(speed_b[i] >= 0.0f && speed_b[i] < FLT_MAX)) {
            FailMixedPair(p, model, data, env, collision::kDatFailureMotion, 0.0f,
                          refs_a, count_a, refs_b, count_b);
            return;
        }
        bound_b = fmaxf(bound_b, speed_b[i]);
        curve_b = fmaxf(curve_b, curve);
    }
    float scale_a = 0.0f, scale_b = 0.0f;
    for (uint32_t i = 0u; i < count_a; ++i) scale_a = fmaxf(scale_a, a.vertex[i].LengthSq());
    for (uint32_t i = 0u; i < count_b; ++i) scale_b = fmaxf(scale_b, b.vertex[i].LengthSq());
    const float roundoff = 64.0f * FLT_EPSILON *
        (sqrtf(scale_a) + sqrtf(scale_b) + bound_a + bound_b);
    const auto separator = collision::DatFindSeparator(a, b, 0.0f, bound_a + bound_b + roundoff);
    if (!(separator.gap > 0.0f)) {
        FailMixedPair(p, model, data, env, collision::kDatFailureOverlap, DatOverlapDepth(a, b),
                      refs_a, count_a, refs_b, count_b);
        return;
    }
    const float available_gap = fmaxf(separator.gap - roundoff, 0.0f);
    if (bound_a + bound_b == 0.0f || bound_a + bound_b < available_gap) return;
    collision::DatSweptPrimitive swept_a, swept_b;
    swept_a.count = 2u * count_a;
    swept_b.count = 2u * count_b;
    float approach_a = 0.0f, approach_b = 0.0f;
    for (uint32_t i = 0u; i < count_a; ++i) {
        swept_a.vertex[i] = a.vertex[i];
        swept_a.vertex[count_a + i] = end_a.vertex[i];
        approach_a = fmaxf(approach_a, -(swept_a.vertex[count_a + i] -
                                         a.vertex[i]).Dot(separator.normal));
    }
    for (uint32_t i = 0u; i < count_b; ++i) {
        swept_b.vertex[i] = b.vertex[i];
        swept_b.vertex[count_b + i] = end_b.vertex[i];
        approach_b = fmaxf(approach_b, (swept_b.vertex[count_b + i] -
                                        b.vertex[i]).Dot(separator.normal));
    }
    const float minimum_gap = fmaxf(roundoff, (1.0f - p.relaxation) * separator.gap);
    // The chord's interpolation error is at most one eighth of the second derivative bound.
    if (collision::DatSweptAxisSeparates(separator.normal, swept_a, swept_b,
                                         minimum_gap + curve_a + curve_b)) return;
    const float total = approach_a + approach_b;
    // A side that can move keeps half the relaxation slack of the gap, so a side whose endpoint
    // does not approach is never pinned to the plane.
    const float slack = 0.5f * (1.0f - p.relaxation);
    const float share = total > 0.0f ? approach_b / total : bound_b / (bound_a + bound_b);
    const float fraction = fminf(fmaxf(share, bound_b > 0.0f ? slack : 0.0f),
                                 bound_a > 0.0f ? 1.0f - slack : 1.0f);
    const float plane_offset = available_gap * fraction;
    float roots[2][3];
    bool truncated = false;
    for (uint32_t side = 0u; side < 2u; ++side) {
        const DatPointRef* refs = side == 0u ? refs_a : refs_b;
        const uint32_t count = side == 0u ? count_a : count_b;
        const math::Vec3 normal = separator.normal * (side == 0u ? 1.0f : -1.0f);
        for (uint32_t i = 0u; i < count; ++i) {
            const float offset = side == 0u ? plane_offset + roundoff : -plane_offset;
            const float speed = side == 0u ? speed_a[i] : speed_b[i];
            roots[side][i] = DatPlaneFraction(p, model, data, env, refs[i],
                separator.negative_support, normal, offset, speed, frame,
                side == 0u ? curves_a[i] : curves_b[i],
                side == 0u ? &a.vertex[i] : &b.vertex[i],
                side == 0u ? &end_a.vertex[i] : &end_b.vertex[i],
                poses ? (side == 0u ? poses->a_quarters : poses->b_quarters) : nullptr);
            truncated |= roots[side][i] < 1.0f;
        }
    }
    if (!truncated || DatJoinPair(p, model, data, env, refs_a, count_a, refs_b, count_b, bound_a, bound_b,
                                  separator, minimum_gap, frame, curve_a + curve_b, roundoff, poses)) return;
    for (uint32_t side = 0u; side < 2u; ++side) {
        const DatPointRef* refs = side == 0u ? refs_a : refs_b;
        const uint32_t count = side == 0u ? count_a : count_b;
        for (uint32_t i = 0u; i < count; ++i) {
            const DatPointRef& other = side == 0u ? refs_b[0] : refs_a[0];
            if (roots[side][i] < 1.0f)
                DatMinOwnerBeta(p, model, data, env, refs[i], p.relaxation * roots[side][i],
                    other.particle ? kDatBoundParticlePair : kDatBoundBodyPair,
                    DatCounterpart(p, other));
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
    // Links of one articulation are certified relative to their lowest common ancestor.
    const uint32_t frame = refs_a[0].particle || refs_b[0].particle ? ~0u
        : DatCommonFrame(p, model, env, refs_a[0].owner, refs_b[0].owner);
    DatCertifyPair(p, model, data, env, refs_a, count_a, refs_b, count_b, frame);
}

constexpr uint32_t kDatReachTriangles = 0u, kDatReachSegments = 1u, kDatReachBody = 2u;

// A tree a reaching body triangle searches, with its start bounds in the tree frame.
struct DatReachTree {
    const collision::MeshBvhNode* nodes = nullptr;
    uint32_t node_count = 0u, leaf_count = 0u, first = 0u, base = 0u;
    uint32_t vertex_count = 0u, owner = 0u, kind = kDatReachTriangles;
    uint32_t frame = ~0u;
    math::Vec3 lower{}, upper{};
    float reach = 0.0f;
};

// Particle surfaces are searched by triangle, or by segment when they have none.
// Every allowed body pair is searched, including links of the same articulation.
__device__ bool DatReachTarget(const DatTruncateParams& p, const ModelView& model,
                               const DataView& data, uint32_t env, uint32_t body,
                               uint32_t index, const math::Vec3* corner, float travel,
                               DatReachTree* tree) {
    math::Vec3 points[3] = {corner[0], corner[1], corner[2]};
    if (index < p.surfaces_per_env) {
        const auto info = model.particle_surface_info[index];
        if (info.vertex_count == 0u) return false;
        // Particles keep within half the relaxed radius their surface shares with this body.
        const float other = 0.5f * p.relaxation * collision::DatQueryRadius(
            collision::DatMotionRadius(model.particle_surface_thickness[index] + p.margin, p.dt,
                data.particle_surface_max_speed[size_t{env} * p.surfaces_per_env + index],
                data.dat_body_speed[size_t{env} * p.bodies_per_env + body], p.relaxation));
        tree->base = env * p.particles_per_env + info.vertex_offset;
        tree->vertex_count = info.vertex_count;
        tree->reach = (travel + other) / p.relaxation;
        if (info.triangle_count > 0u) {
            tree->nodes = data.particle_surface_nodes + size_t{env} * p.nodes_per_env +
                          info.node_offset;
            tree->node_count = info.node_count;
            tree->leaf_count = info.triangle_count;
            tree->first = info.triangle_offset;
            tree->kind = kDatReachTriangles;
        } else {
            const auto edges = model.particle_surface_edge_info[index];
            tree->nodes = data.particle_surface_edge_nodes + size_t{env} * p.edge_nodes_per_env +
                          edges.node_offset;
            tree->node_count = edges.node_count;
            tree->leaf_count = edges.edge_count;
            tree->first = edges.edge_offset;
            tree->kind = kDatReachSegments;
        }
        if (tree->node_count == 0u) return false;
    } else {
        const uint32_t other = index - p.surfaces_per_env;
        const auto info = model.mesh_surface_info[other];
        if (info.vertex_count == 0u || info.node_count == 0u) return false;
        const auto view = DatOldBodyView(p, model, data, env, other);
        tree->nodes = view.nodes + info.node_offset;
        tree->node_count = info.node_count;
        tree->leaf_count = info.triangle_count;
        tree->first = info.triangle_offset;
        tree->vertex_count = info.vertex_count;
        tree->owner = other;
        tree->kind = kDatReachBody;
        tree->frame = DatCommonFrame(p, model, env, body, other);
        for (uint32_t i = 0u; i < 3u; ++i)
            points[i] = collision::MeshSurfaceLocalPoint(view, corner[i]);
        // The other body's motion is its farthest vertex travel over the step, per unit time.
        tree->reach = (travel + data.dat_body_motion[size_t{env} * p.bodies_per_env + other] *
                                p.dt) / p.relaxation;
        if (tree->frame != ~0u) {
            const float relative = p.dt *
                (DatChainSpeed(p, model, data, env, body, tree->frame, nullptr, data.dat_prev_q) +
                 DatChainSpeed(p, model, data, env, other, tree->frame, nullptr, data.dat_prev_q));
            tree->reach = fminf(tree->reach, relative / p.relaxation);
        }
        tree->reach += 64.0f * FLT_EPSILON *
            (sqrtf(view.position.LengthSq()) + sqrtf(corner[0].LengthSq()) + tree->reach);
    }
    tree->lower = tree->upper = points[0];
    for (uint32_t i = 1u; i < 3u; ++i) {
        tree->lower = {fminf(tree->lower.x, points[i].x), fminf(tree->lower.y, points[i].y),
                       fminf(tree->lower.z, points[i].z)};
        tree->upper = {fmaxf(tree->upper.x, points[i].x), fmaxf(tree->upper.y, points[i].y),
                       fmaxf(tree->upper.z, points[i].z)};
    }
    return true;
}

__device__ float DatReachBoundsDistanceSquared(const DatReachTree& tree,
                                               const collision::MeshBvhNode& node) {
    const math::Vec3 gap{
        fmaxf(fmaxf(node.lower.x - tree.upper.x, tree.lower.x - node.upper.x), 0.0f),
        fmaxf(fmaxf(node.lower.y - tree.upper.y, tree.lower.y - node.upper.y), 0.0f),
        fmaxf(fmaxf(node.lower.z - tree.upper.z, tree.lower.z - node.upper.z), 0.0f)};
    return gap.LengthSq();
}

__device__ bool DatReachLeaf(const ModelView& model, const DatReachTree& tree, uint32_t leaf,
                             DatPointRef* refs, uint32_t* count) {
    if (leaf >= tree.leaf_count) return false;
    if (tree.kind == kDatReachSegments) {
        const auto edge = model.particle_surface_edges[tree.first + leaf];
        if (edge.vertex0 >= tree.vertex_count || edge.vertex1 >= tree.vertex_count) return false;
        refs[0] = {tree.base + edge.vertex0, 0u, true};
        refs[1] = {tree.base + edge.vertex1, 0u, true};
        *count = 2u;
        return true;
    }
    const size_t at = (size_t{tree.first} + leaf) * 3u;
    for (uint32_t k = 0u; k < 3u; ++k) {
        const uint32_t vertex = tree.kind == kDatReachBody ? model.mesh_triangles[at + k]
                                                           : model.particle_surface_triangles[at + k];
        if (vertex >= tree.vertex_count) return false;
        refs[k] = tree.kind == kDatReachBody ? DatPointRef{tree.owner, vertex, false}
                                             : DatPointRef{tree.base + vertex, 0u, true};
    }
    *count = 3u;
    return true;
}

struct DatReachCandidate {
    uint32_t triangle, leaf;
};

struct DatReachWork {
    DatReachTree tree;
    DatPairPoses poses;
    math::Transform world_start;
    math::Vec3 axis;
    float curvature;
    bool active;
};

// Contact normals are axis hints only; support projections independently certify every separation.
__device__ math::Vec3 DatContactAxis(const DatTruncateParams& p, const DataView& data,
    uint32_t env, uint32_t body, const DatReachWork& work) {
    if (work.tree.kind != kDatReachBody) return {};
    const uint32_t lane = threadIdx.x % kWarpLanes;
    const uint32_t count = min(data.pair_count[env], p.slot_stride);
    for (uint32_t first = 0u; first < count; first += kWarpLanes) {
        const size_t slot = size_t{env} * p.slot_stride + first + lane;
        uint32_t a = ~0u, b = ~0u;
        if (first + lane < count) {
            a = data.candidate_pairs[slot * 2u];
            b = data.candidate_pairs[slot * 2u + 1u];
        }
        const bool pair = (a == body && b == work.tree.owner) || (b == body && a == work.tree.owner);
        const uint32_t found = __ballot_sync(~0u, pair && data.ucontact_count[slot] > 0u);
        if (found == 0u) continue;
        const uint32_t index = __ffs(found) - 1u;
        const auto normal = data.ucontact_normal[(size_t{env} * p.slot_stride + first + index) * 4u];
        const uint32_t source = __shfl_sync(~0u, a, index);
        auto inverse = work.world_start.rotation;
        inverse.x = -inverse.x; inverse.y = -inverse.y; inverse.z = -inverse.z;
        const auto local = math::gpu::RotateByQuatNormalized(inverse, normal);
        const auto axis = math::gpu::RotateByQuatNormalized(work.poses.a_start.rotation, local);
        return axis * ((source == body ? -1.0f : 1.0f) / fmaxf(sqrtf(axis.LengthSq()), FLT_MIN));
    }
    return {};
}

__device__ math::Vec3 DatBoxAxis(const math::Transform& pose, uint32_t axis) {
    return math::gpu::RotateByQuatNormalized(pose.rotation,
        axis == 0u ? math::Vec3{1.0f, 0.0f, 0.0f} : axis == 1u ? math::Vec3{0.0f, 1.0f, 0.0f}
                                                              : math::Vec3{0.0f, 0.0f, 1.0f});
}

__device__ float2 DatBoxInterval(const collision::MeshBvhNode& box, const math::Transform& pose,
                                  math::Vec3 axis, math::Vec3 origin) {
    const auto center = (box.lower + box.upper) * 0.5f;
    const auto half = (box.upper - box.lower) * 0.5f;
    auto inverse = pose.rotation;
    inverse.x = -inverse.x;
    inverse.y = -inverse.y;
    inverse.z = -inverse.z;
    const auto local = math::gpu::RotateByQuatNormalized(inverse, axis);
    const float middle = (pose.position - origin).Dot(axis) + center.Dot(local);
    const float radius = half.x * fabsf(local.x) + half.y * fabsf(local.y) + half.z * fabsf(local.z);
    return {middle - radius, middle + radius};
}

// Oriented root boxes enclose every mesh; endpoint intervals plus curvature cover the full motion.
__device__ bool DatReachBoxesSeparated(const ModelView& model, uint32_t body, const DatReachWork& work,
                                        float relaxation) {
    if (work.tree.kind != kDatReachBody) return false;
    const uint32_t lane = threadIdx.x % kWarpLanes;
    const auto& poses = work.poses;
    math::Vec3 axis{};
    if (lane < 12u) {
        const auto pose = lane < 3u ? poses.a_start : lane < 6u ? poses.b_start
            : lane < 9u ? poses.a_end : poses.b_end;
        axis = DatBoxAxis(pose, lane % 3u);
    } else if (lane < 30u) {
        const uint32_t cross = (lane - 12u) % 9u;
        const auto a = lane < 21u ? poses.a_start : poses.a_end;
        const auto b = lane < 21u ? poses.b_start : poses.b_end;
        axis = DatBoxAxis(a, cross / 3u).Cross(DatBoxAxis(b, cross % 3u));
    } else if (lane == 30u) {
        axis = poses.a_start.position - poses.b_start.position;
    }
    const float length = sqrtf(axis.LengthSq());
    axis = axis / fmaxf(length, FLT_MIN);
    const auto a = model.mesh_bvh_nodes[model.mesh_surface_info[body].node_offset];
    const auto b = model.mesh_bvh_nodes[model.mesh_surface_info[work.tree.owner].node_offset];
    const auto origin = poses.a_start.position;
    const float2 a0 = DatBoxInterval(a, poses.a_start, axis, origin);
    const float2 a1 = DatBoxInterval(a, poses.a_end, axis, origin);
    const float2 b0 = DatBoxInterval(b, poses.b_start, axis, origin);
    const float2 b1 = DatBoxInterval(b, poses.b_end, axis, origin);
    const float scale = sqrtf(poses.a_start.position.LengthSq()) + sqrtf(poses.a_end.position.LengthSq()) +
        sqrtf(poses.b_start.position.LengthSq()) + sqrtf(poses.b_end.position.LengthSq()) +
        sqrtf(a.lower.LengthSq()) + sqrtf(a.upper.LengthSq()) +
        sqrtf(b.lower.LengthSq()) + sqrtf(b.upper.LengthSq()) + work.curvature;
    const float rounding = 128.0f * FLT_EPSILON * scale;
    const bool forward = fminf(a0.x, a1.x) - fmaxf(b0.y, b1.y) >
        work.curvature + fmaxf(rounding, (1.0f - relaxation) * (a0.x - b0.y));
    const bool reverse = fminf(b0.x, b1.x) - fmaxf(a0.y, a1.y) >
        work.curvature + fmaxf(rounding, (1.0f - relaxation) * (b0.x - a0.y));
    return __any_sync(~0u, length > FLT_MIN && (forward || reverse));
}

struct DatSupportPoint {
    math::Vec3 point;
    float start_projection;
    float scale;
};

// A warp scans both endpoints in local coordinates; ties retain the lowest vertex and endpoint index.
__device__ DatSupportPoint DatEndpointSupport(const DatTruncateParams& p, const ModelView& model,
    const DataView& data, uint32_t env, uint32_t owner, bool particle,
    const math::Transform& start, const math::Transform& finish, math::Vec3 origin, math::Vec3 direction) {
    const uint32_t lane = threadIdx.x % kWarpLanes;
    const auto info = particle ? model.particle_surface_info[owner] : model.mesh_surface_info[owner];
    auto inverse_start = start.rotation, inverse_finish = finish.rotation;
    inverse_start.x = -inverse_start.x; inverse_start.y = -inverse_start.y; inverse_start.z = -inverse_start.z;
    inverse_finish.x = -inverse_finish.x; inverse_finish.y = -inverse_finish.y; inverse_finish.z = -inverse_finish.z;
    const auto local_start = math::gpu::RotateByQuatNormalized(inverse_start, direction);
    const auto local_finish = math::gpu::RotateByQuatNormalized(inverse_finish, direction);
    const float offset_start = (start.position - origin).Dot(direction);
    const float offset_finish = (finish.position - origin).Dot(direction);
    float best = -FLT_MAX, first = -FLT_MAX, scale = 0.0f;
    uint32_t index = ~0u;
    for (uint32_t vertex = lane; vertex < info.vertex_count; vertex += kWarpLanes) {
        const uint32_t at = env * p.particles_per_env + info.vertex_offset + vertex;
        const auto local = particle ? math::Vec3{} : DatBodyLocalPoint(model, owner, vertex);
        const auto particle_start = particle ? data.particle_prev_pos[at] : math::Vec3{};
        const auto particle_finish = particle ? data.particle_pos[at] : math::Vec3{};
        if (particle) scale = fmaxf(scale, fmaxf(
            fabsf(particle_start.x) + fabsf(particle_start.y) + fabsf(particle_start.z),
            fabsf(particle_finish.x) + fabsf(particle_finish.y) + fabsf(particle_finish.z)));
        const float a = particle ? (particle_start - origin).Dot(direction)
                                 : local.Dot(local_start) + offset_start;
        const float b = particle ? (particle_finish - origin).Dot(direction)
                                 : local.Dot(local_finish) + offset_finish;
        first = fmaxf(first, a);
        const float value = fmaxf(a, b);
        const uint32_t candidate = 2u * vertex + (b > a ? 1u : 0u);
        if (value > best || (value == best && candidate < index)) { best = value; index = candidate; }
    }
    for (uint32_t offset = kWarpLanes / 2u; offset > 0u; offset /= 2u) {
        scale = fmaxf(scale, __shfl_xor_sync(~0u, scale, offset));
        first = fmaxf(first, __shfl_xor_sync(~0u, first, offset));
        const float other = __shfl_xor_sync(~0u, best, offset);
        const uint32_t candidate = __shfl_xor_sync(~0u, index, offset);
        if (other > best || (other == best && candidate < index)) { best = other; index = candidate; }
    }
    if (index == ~0u) return {{}, FLT_MAX, scale};
    math::Vec3 point;
    if (particle) {
        const uint32_t at = env * p.particles_per_env + info.vertex_offset + index / 2u;
        point = ((index & 1u) != 0u ? data.particle_pos[at] : data.particle_prev_pos[at]) - origin;
    } else {
        const auto pose = (index & 1u) != 0u ? finish : start;
        point = pose.position - origin + math::gpu::RotateByQuatNormalized(pose.rotation,
            DatBodyLocalPoint(model, owner, index / 2u));
    }
    return {point, first, scale};
}

// GJK only proposes axes; endpoint supports, curvature and rounding must independently certify separation.
__device__ bool DatReachSweptSeparated(const DatTruncateParams& p, const ModelView& model,
    const DataView& data, uint32_t env, uint32_t body, uint32_t target, const DatReachWork& work,
    bool start_only = false, math::Vec3* separating_axis = nullptr, float* separating_gap = nullptr) {
    const uint32_t lane = threadIdx.x % kWarpLanes;
    const bool particle = work.tree.kind != kDatReachBody;
    const uint32_t other = particle ? target : work.tree.owner;
    const auto& poses = work.poses;
    const auto origin = poses.a_start.position;
    math::Vec3 direction = work.axis.LengthSq() > 0.0f ? work.axis : particle
        ? (work.tree.nodes[0].lower + work.tree.nodes[0].upper) * 0.5f - origin
        : poses.b_start.position - origin;
    const float scale = sqrtf(poses.a_start.position.LengthSq()) + sqrtf(poses.a_end.position.LengthSq()) +
        model.mesh_vertex_reach[body] + (particle ? sqrtf(work.tree.nodes[0].lower.LengthSq()) +
            sqrtf(work.tree.nodes[0].upper.LengthSq()) + work.tree.reach :
            sqrtf(poses.b_start.position.LengthSq()) + sqrtf(poses.b_end.position.LengthSq()) +
            model.mesh_vertex_reach[other]) + work.curvature;
    collision::cvx::MinkPoint simplex[4];
    int count = 0;
    for (int iteration = 0; iteration < collision::cvx::kGjkMaxIter; ++iteration) {
        const float length = sqrtf(direction.LengthSq());
        if (!(length > FLT_MIN && length < FLT_MAX)) return false;
        direction = direction / length;
        const auto a = DatEndpointSupport(p, model, data, env, body, false,
            poses.a_start, start_only ? poses.a_start : poses.a_end, origin, direction);
        const auto b = DatEndpointSupport(p, model, data, env, other, particle,
            poses.b_start, start_only ? poses.b_start : poses.b_end, origin, -direction);
        const auto difference = a.point - b.point;
        const float rounding = 128.0f * FLT_EPSILON * (scale + a.scale + b.scale);
        const float start_gap = -a.start_projection - b.start_projection;
        const float gap = -difference.Dot(direction);
        if (gap > (start_only ? 0.0f : work.curvature +
            fmaxf(rounding, (1.0f - p.relaxation) * start_gap))) {
            if (separating_axis) *separating_axis = direction;
            if (separating_gap) *separating_gap = gap;
            return true;
        }
        bool finished = false;
        if (lane == 0u) {
            for (int i = 0; i < count; ++i)
                finished |= (simplex[i].v - difference).LengthSq() == 0.0f;
            if (!finished) {
                simplex[count++] = {difference, a.point, b.point};
                if (count == 1) direction = -difference;
                else finished = collision::cvx::GjkDoSimplex(simplex, count, direction);
            }
        }
        if (__shfl_sync(~0u, finished, 0u)) return false;
        direction.x = __shfl_sync(~0u, direction.x, 0u);
        direction.y = __shfl_sync(~0u, direction.y, 0u);
        direction.z = __shfl_sync(~0u, direction.z, 0u);
    }
    return false;
}

// Whole-convex projections certify a motion interval without visiting its boundary triangles.
__device__ float DatConvexGapAt(const DatTruncateParams& p, const ModelView& model,
    const DataView& data, uint32_t env, uint32_t body, const DatReachWork& work,
    math::Vec3 direction, float fraction, float reserve) {
    const auto a_pose = fraction == 0.0f ? work.poses.a_start : fraction == 1.0f ? work.poses.a_end :
        DatBodyPoseAt(p, model, data, env, body, fraction, work.tree.frame);
    const auto b_pose = fraction == 0.0f ? work.poses.b_start : fraction == 1.0f ? work.poses.b_end :
        DatBodyPoseAt(p, model, data, env, work.tree.owner, fraction, work.tree.frame);
    const auto origin = work.poses.a_start.position;
    const auto a = DatEndpointSupport(p, model, data, env, body, false, a_pose, a_pose, origin, direction);
    const auto b = DatEndpointSupport(p, model, data, env, work.tree.owner, false,
        b_pose, b_pose, origin, -direction);
    return -(a.point - b.point).Dot(direction) - reserve;
}

// Joined owners traverse the same certified prefix; subdivision bounds the actual joint-space path.
__device__ bool DatCertifyConvex(const DatTruncateParams& p, const ModelView& model,
    const DataView& data, uint32_t env, uint32_t body, uint32_t target, const DatReachWork& work) {
    math::Vec3 direction;
    float gap = 0.0f;
    if (!DatReachSweptSeparated(p, model, data, env, body, target, work, true, &direction, &gap))
        return false;
    const auto& poses = work.poses;
    const float speed_a = DatRadiusSpeed(poses.a_motion,
        sqrtf(poses.a_motion.local.position.LengthSq()) + model.mesh_vertex_reach[body]);
    const float speed_b = DatRadiusSpeed(poses.b_motion,
        sqrtf(poses.b_motion.local.position.LengthSq()) + model.mesh_vertex_reach[work.tree.owner]);
    const float speed = speed_a + speed_b;
    const float rounding = 128.0f * FLT_EPSILON * (sqrtf(poses.a_start.position.LengthSq()) +
        sqrtf(poses.b_start.position.LengthSq()) + model.mesh_vertex_reach[body] +
        model.mesh_vertex_reach[work.tree.owner] + speed);
    const float reserve = fmaxf(rounding, (1.0f - p.relaxation) * gap);
    // Near the roundoff envelope, local surface certificates resolve the tighter separating features.
    if (!(gap > reserve && speed < FLT_MAX)) return false;
    float fraction = 1.0f;
    {
        DatInterval stack[14];
        uint32_t top = 1u;
        stack[0] = {0.0f, 1.0f, gap - reserve,
            DatConvexGapAt(p, model, data, env, body, work, direction, 1.0f, reserve), 0u};
        while (top > 0u) {
            const auto interval = stack[--top];
            const float width = interval.end - interval.start;
            if (fminf(interval.value_start, interval.value_end) >
                fminf(0.5f * speed * width, work.curvature * width * width + rounding)) continue;
            if (interval.depth == 12u) {
                // The Lipschitz bound certifies a positive prefix without quantizing it to the subdivision grid.
                const float prefix = speed > 0.0f ? fmaxf(interval.value_start, 0.0f) / speed : 0.0f;
                fraction = interval.start + fminf(width, prefix);
                break;
            }
            const float mid = 0.5f * (interval.start + interval.end);
            const float value = DatConvexGapAt(p, model, data, env, body, work, direction, mid, reserve);
            stack[top++] = {mid, interval.end, value, interval.value_end, interval.depth + 1u};
            stack[top++] = {interval.start, mid, interval.value_start, value, interval.depth + 1u};
        }
    }
    if (threadIdx.x % kWarpLanes == 0u) {
        const DatPointRef a{body, 0u, false}, b{work.tree.owner, 0u, false};
        uint32_t node_a = 0u, node_b = 0u;
        const bool moving_a = DatOwnerNode(p, model, data, env, a, &node_a);
        const bool moving_b = DatOwnerNode(p, model, data, env, b, &node_b);
        if (moving_a && moving_b) DatJoin(p, data, node_a, node_b);
        if (fraction < 1.0f) {
            DatMinOwnerBeta(p, model, data, env, a, p.relaxation * fraction,
                kDatBoundBodyPair, DatCounterpart(p, b));
            DatMinOwnerBeta(p, model, data, env, b, p.relaxation * fraction,
                kDatBoundBodyPair, DatCounterpart(p, a));
        }
    }
    return true;
}

// Compact potential pairs before the register-intensive traversal and motion certificates.
__global__ void DatQueueReachPairsKernel(DatTruncateParams p, ModelView model, DataView data,
                                         uint32_t pair_count) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= pair_count) return;
    const uint32_t targets = p.surfaces_per_env + p.bodies_per_env;
    const uint32_t source = item / targets;
    const uint32_t env = source / p.bodies_per_env, body = source % p.bodies_per_env;
    const uint32_t target = item % targets;
    const auto info = model.mesh_surface_info[body];
    const size_t owner = size_t{env} * p.bodies_per_env + body;
    const float travel = data.dat_body_motion[owner] * p.dt;
    const float query = data.dat_body_query_radius[owner];
    if (info.triangle_count == 0u || info.node_count == 0u || !(travel > 0.0f) ||
        !(query > 0.0f && query <= collision::kDatQueryRadiusMax)) return;
    {
        bool allowed = true;
        if (target >= p.surfaces_per_env) {
            const uint32_t other = target - p.surfaces_per_env;
            const size_t other_owner = size_t{env} * p.bodies_per_env + other;
            allowed = other != body && model.mesh_surface_info[other].node_count > 0u &&
                !(other < body && data.dat_body_motion[other_owner] > 0.0f);
            if (allowed) {
                const auto a = data.dat_prev_body_pose[owner].position;
                const auto b = data.dat_prev_body_pose[other_owner].position;
                const float radii = model.mesh_vertex_reach[body] + model.mesh_vertex_reach[other];
                const float reach = (travel + p.dt * data.dat_body_motion[other_owner]) / p.relaxation;
                const float rounding = 64.0f * FLT_EPSILON *
                    (sqrtf(a.LengthSq()) + sqrtf(b.LengthSq()) + radii + reach);
                allowed = sqrtf((a - b).LengthSq()) <= radii + reach + rounding &&
                    DatBodyPairAllowed(p, model, body, other, true);
            }
        }
        if (!allowed) return;
    }
    auto* candidates = reinterpret_cast<uint32_t*>(data.contact_cache_scratch);
    const uint32_t active = __activemask(), lane = threadIdx.x % kWarpLanes;
    const uint32_t leader = __ffs(active) - 1u;
    uint32_t first = 0u;
    if (lane == leader) first = atomicAdd(candidates, __popc(active));
    first = __shfl_sync(active, first, leader);
    candidates[2u + first + __popc(active & ((1u << lane) - 1u))] = item;
}

// Resident blocks take compacted body pairs and share their motion samples across triangle certificates.
// Every pair retains the same traversal and continuous separation bounds.
__global__ void DatBodyReachKernel(DatTruncateParams p, ModelView model, DataView data) {
    __shared__ DatReachCandidate queue[kBlockSize / kWarpLanes][2u * kWarpLanes];
    __shared__ __align__(alignof(DatReachWork)) unsigned char storage[sizeof(DatReachWork)];
    __shared__ uint32_t ticket;
    auto* work = reinterpret_cast<DatReachWork*>(storage);
    auto* candidates = reinterpret_cast<uint32_t*>(data.contact_cache_scratch);
    const uint32_t warp = threadIdx.x / kWarpLanes, lane = threadIdx.x % kWarpLanes;
    const uint32_t targets = p.surfaces_per_env + p.bodies_per_env;
    for (;;) {
        if (threadIdx.x == 0u) ticket = atomicAdd(candidates + 1u, 1u);
        __syncthreads();
        if (ticket >= candidates[0]) return;
        const uint32_t item = candidates[2u + ticket];
        const uint32_t source = item / targets;
        const uint32_t env = source / p.bodies_per_env, body = source % p.bodies_per_env;
        const uint32_t target = item % targets;
        const auto info = model.mesh_surface_info[body];
        const size_t owner = size_t{env} * p.bodies_per_env + body;
        const float travel = data.dat_body_motion[owner] * p.dt;
        auto& shared = *work;
        if (threadIdx.x == 0u) {
            shared.active = false;
            shared.axis = {};
            shared.tree = DatReachTree{};
            shared.world_start = DatBodyPoseAt(p, model, data, env, body, 0.0f);
            const auto root = model.mesh_bvh_nodes[info.node_offset];
            math::Vec3 corners[8];
            for (uint32_t i = 0u; i < 8u; ++i) {
                const math::Vec3 local{i & 1u ? root.upper.x : root.lower.x,
                    i & 2u ? root.upper.y : root.lower.y, i & 4u ? root.upper.z : root.lower.z};
                corners[i] = shared.world_start.position + math::gpu::RotateByQuatNormalized(
                    shared.world_start.rotation, local);
            }
            if (DatReachTarget(p, model, data, env, body, target, corners, travel, &shared.tree)) {
                const auto view = target < p.surfaces_per_env ? collision::MeshSurfaceView{}
                    : DatOldBodyView(p, model, data, env, target - p.surfaces_per_env);
                auto& tree = shared.tree;
                tree.lower = {FLT_MAX, FLT_MAX, FLT_MAX};
                tree.upper = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                for (uint32_t i = 0u; i < 8u; ++i) {
                    const auto point = collision::MeshSurfaceLocalPoint(view, corners[i]);
                    tree.lower = {fminf(tree.lower.x, point.x), fminf(tree.lower.y, point.y),
                                  fminf(tree.lower.z, point.z)};
                    tree.upper = {fmaxf(tree.upper.x, point.x), fmaxf(tree.upper.y, point.y),
                                  fmaxf(tree.upper.z, point.z)};
                }
                if (DatReachBoundsDistanceSquared(tree, tree.nodes[0]) <= tree.reach * tree.reach) {
                    shared.active = true;
                    shared.poses.a_start = DatBodyPoseAt(p, model, data, env, body, 0.0f, tree.frame);
                    shared.poses.a_end = DatBodyPoseAt(p, model, data, env, body, 1.0f, tree.frame);
                    shared.poses.b_start = shared.poses.b_end = math::Transform{};
                    shared.poses.a_motion = DatBodyMotionBounds(p, model, data, env, body, tree.frame);
                    shared.poses.b_motion = DatBodyMotion{};
                    shared.curvature = DatBodyCurvature(model, body, shared.poses.a_motion);
                    if (tree.kind == kDatReachBody) {
                        shared.poses.b_start = DatBodyPoseAt(p, model, data, env, tree.owner, 0.0f, tree.frame);
                        shared.poses.b_end = DatBodyPoseAt(p, model, data, env, tree.owner, 1.0f, tree.frame);
                        shared.poses.b_motion = DatBodyMotionBounds(p, model, data, env, tree.owner, tree.frame);
                        shared.curvature += DatBodyCurvature(model, tree.owner, shared.poses.b_motion);
                    }
                }
            }
        }
        __syncthreads();
        if (warp == 0u && shared.active) {
            const auto axis = DatContactAxis(p, data, env, body, shared);
            if (lane == 0u) shared.axis = axis;
            __syncwarp();
            const bool separated = DatReachBoxesSeparated(model, body, shared, p.relaxation) ||
                DatReachSweptSeparated(p, model, data, env, body, target, shared);
            if (lane == 0u) shared.active = !separated;
        }
        __syncthreads();
        if (!shared.active) continue;
        if (warp == 0u && shared.tree.kind == kDatReachBody) {
            const auto a = nkops::LoadPrimShape(model.shape_table, body);
            const auto b = nkops::LoadPrimShape(model.shape_table, shared.tree.owner);
            if (collision::IsConvexShape(a.kind) && collision::IsConvexShape(b.kind) &&
                DatCertifyConvex(p, model, data, env, body, target, shared) && lane == 0u)
                shared.active = false;
        }
        __syncthreads();
        if (!shared.active) continue;
        if (threadIdx.x < 3u) shared.poses.a_quarters[threadIdx.x] =
            DatBodyPoseAt(p, model, data, env, body, 0.25f * (threadIdx.x + 1u), shared.tree.frame);
        if (threadIdx.x >= 3u && threadIdx.x < 6u && shared.tree.kind == kDatReachBody)
            shared.poses.b_quarters[threadIdx.x - 3u] = DatBodyPoseAt(p, model, data, env,
                shared.tree.owner, 0.25f * (threadIdx.x - 2u), shared.tree.frame);
        __syncthreads();
        DatReachTree tree = shared.tree;
        uint32_t triangle = threadIdx.x, cursor = tree.node_count;
        bool entered = false;
        const auto view = tree.kind == kDatReachBody ? DatOldBodyView(p, model, data, env, tree.owner)
                                                    : collision::MeshSurfaceView{};
        const auto next = [&](DatReachCandidate* candidate) {
            while (triangle < info.triangle_count) {
                if (cursor >= tree.node_count) {
                    if (entered) triangle += blockDim.x;
                    if (triangle >= info.triangle_count) break;
                    entered = true;
                    tree.lower = {FLT_MAX, FLT_MAX, FLT_MAX};
                    tree.upper = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                    bool valid = true;
                    for (uint32_t j = 0u; j < 3u; ++j) {
                        const uint32_t vertex = model.mesh_triangles[
                            (size_t{info.triangle_offset} + triangle) * 3u + j];
                        if (vertex >= info.vertex_count) { valid = false; break; }
                        const auto world = shared.world_start.position + math::gpu::RotateByQuatNormalized(
                            shared.world_start.rotation, DatBodyLocalPoint(model, body, vertex));
                        const auto point = collision::MeshSurfaceLocalPoint(view, world);
                        tree.lower = {fminf(tree.lower.x, point.x), fminf(tree.lower.y, point.y),
                                      fminf(tree.lower.z, point.z)};
                        tree.upper = {fmaxf(tree.upper.x, point.x), fmaxf(tree.upper.y, point.y),
                                      fmaxf(tree.upper.z, point.z)};
                    }
                    if (!valid) {
                        atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                        continue;
                    }
                    cursor = 0u;
                }
                const auto node = tree.nodes[cursor];
                if (node.escape <= cursor || node.escape > tree.node_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    cursor = tree.node_count;
                    continue;
                }
                if (DatReachBoundsDistanceSquared(tree, node) > tree.reach * tree.reach) {
                    cursor = node.escape;
                    continue;
                }
                ++cursor;
                if (node.triangle == ~0u) continue;
                *candidate = {triangle, node.triangle};
                return true;
            }
            return false;
        };
        TestInWarpBatches(queue[warp], triangle < info.triangle_count, next, [&](const DatReachCandidate& c) {
            DatPointRef face[3], refs[3];
            for (uint32_t j = 0u; j < 3u; ++j)
                face[j] = {body, model.mesh_triangles[(size_t{info.triangle_offset} + c.triangle) * 3u + j], false};
            uint32_t count = 0u;
            if (!DatReachLeaf(model, shared.tree, c.leaf, refs, &count)) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                return;
            }
            DatCertifyPair(p, model, data, env, face, 3u, refs, count, tree.frame, &shared.poses);
        });
    }
}

// Two links of one articulation travel, relative to their common ancestor, less than the relaxed
// share of their pair radius or of their start sphere gap, so a pair kept apart cannot meet.
__global__ void DatArticPairsKernel(DatTruncateParams p, ModelView model, DataView data) {
    const uint64_t pairs = uint64_t{p.bodies_per_env} * p.bodies_per_env;
    const uint64_t item = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (item >= pairs * p.env_count) return;
    const uint32_t env = static_cast<uint32_t>(item / pairs);
    const uint32_t source = static_cast<uint32_t>(item % pairs / p.bodies_per_env);
    const uint32_t target = static_cast<uint32_t>(item % p.bodies_per_env);
    if (target <= source || model.mesh_surface_info[source].vertex_count == 0u ||
        model.mesh_surface_info[target].vertex_count == 0u ||
        !DatBodyPairAllowed(p, model, source, target)) return;
    const uint32_t frame = DatCommonFrame(p, model, env, source, target);
    if (frame == ~0u) return;
    const float radius = collision::DatQueryRadius(
        DatArticMotionRadius(p, model, data, env, source, target, frame, data.dat_prev_q));
    const float travel = p.dt *
        (DatChainSpeed(p, model, data, env, source, frame, nullptr, data.dat_prev_q) +
         DatChainSpeed(p, model, data, env, target, frame, nullptr, data.dat_prev_q));
    if (travel <= p.relaxation * radius) return;
    // Spheres about the start origins through the farthest vertices bound the pair's start separation.
    const size_t first = size_t{env} * p.bodies_per_env;
    const float clear = fmaxf(radius, sqrtf((data.dat_prev_body_pose[first + source].position -
                                             data.dat_prev_body_pose[first + target].position).LengthSq()) -
                                          model.mesh_vertex_reach[source] - model.mesh_vertex_reach[target]);
    if (travel <= p.relaxation * clear) return;
    const DatPointRef ref{source, 0u, false};
    if (!(travel <= FLT_MAX)) {
        atomicOr(data.env_status + env, kEnvStatusDatFailure);
        atomicAdd(data.dat_failure_count + env, 1u);
        RecordDatFailure(data, env, collision::kDatFailureMotion,
                         DatBodyWitness(p, model, env, source),
                         DatBodyWitness(p, model, env, target), 0.0f);
        DatMinOwnerBeta(p, model, data, env, ref, 0.0f, kDatBoundChain, target);
        return;
    }
    DatMinOwnerBeta(p, model, data, env, ref, p.relaxation * clear / travel, kDatBoundChain,
                    target);
}

__global__ void DatSnapshotKernel(DatTruncateParams p, DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t links = p.env_count * p.links_per_env;
    const uint32_t articulations = p.env_count * p.articulations_per_env;
    const uint32_t bodies = p.env_count * p.bodies_per_env;
    if (item < links) {
        data.dat_prev_q[item] = data.q[item];
        data.dat_prev_joint_rotation[item] = data.joint_rotation[item];
    }
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
    const uint32_t offset = model.articulation_link_offset[articulation];
    const uint32_t count = model.articulation_link_count[articulation];
    for (uint32_t i = 0u; i < count; ++i)
        data.dat_joint_motion[offset + i] =
            (static_cast<ArticulationJointType>(model.joint_type[offset + i]) == ArticulationJointType::Spherical
                 ? DatQuatArc(data.dat_prev_joint_rotation[offset + i], data.joint_rotation[offset + i])
                 : fabsf(data.q[offset + i] - data.dat_prev_q[offset + i])) / p.dt;
    if (!(beta < 1.0f)) return;
    for (uint32_t i = 0u; i < count; ++i) {
        const uint32_t link = offset + i;
        data.q[link] = data.dat_prev_q[link] +
            (data.q[link] - data.dat_prev_q[link]) * beta;
        data.qdot[link] *= beta;
        data.joint_rotation[link] = DatSlerp(data.dat_prev_joint_rotation[link], data.joint_rotation[link], beta);
        data.joint_angular_velocity[link] = data.joint_angular_velocity[link] * beta;
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
                const float effective_step = data.vbd_step[slot];
                before = nk::vbd::PhysicalVelocity(before, data.vbd_free_rate[slot],
                    data.vbd_free_velocity[slot], p.dt, effective_step);
                after = nk::vbd::PhysicalVelocity(after, data.vbd_free_rate[slot],
                    data.vbd_free_velocity[slot], p.dt, effective_step);
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

// Joined owners take the smallest fraction among them, so every joined pair moves along its prefix.
__global__ void DatJoinMinKernel(DatTruncateParams p, DataView data, uint32_t first, uint32_t nodes) {
    const uint32_t node = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodes) return;
    const uint32_t root = DatFind(p, data, node);
    if (root != node) atomicMin(DatBetaSlot(p, data, root), *DatBetaSlot(p, data, node));
}

__global__ void DatJoinShareKernel(DatTruncateParams p, DataView data, uint32_t first, uint32_t nodes) {
    const uint32_t node = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodes) return;
    const uint32_t root = DatFind(p, data, node);
    if (root != node) *DatBetaSlot(p, data, node) = *DatBetaSlot(p, data, root);
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
        (p->vbd_vertices_per_env > 0u &&
         (!data.vbd_free_rate || !data.vbd_free_velocity || !data.vbd_step)))
        return Status::InvalidArgument;
    if ((p->surfaces_per_env > 0u &&
         (!model.particle_surface_info || !model.particle_surface_edge_info ||
          (p->triangles_per_env > 0u && (!model.particle_surface_triangles ||
                                        !data.particle_surface_nodes)) ||
          (p->edges_per_env > 0u && (!model.particle_surface_edges ||
                                    !data.particle_surface_edge_nodes)) ||
          !model.particle_surface_thickness || !data.particle_surface_max_speed ||
          !data.dat_surface_motion ||
          !data.dat_particle_beta || !data.dat_particle_parent || !data.particle_inv_mass ||
          !data.particle_prev_pos || !data.particle_pos || !data.particle_vel)) ||
        (p->bodies_per_env > 0u &&
         (!model.mesh_surface_info || !model.mesh_edge_info || !model.mesh_triangles ||
          !model.mesh_bvh_nodes || !model.shape_table || !model.mesh_vertex_reach ||
          (p->excluded_pairs > 0u && !model.excluded_pairs) ||
          !model.mesh_edges || !model.hull_verts || !model.mesh_vertex_sources ||
          !model.mesh_vertex_triangle_offsets || !model.mesh_vertex_triangles ||
          !data.body_pose || !data.dat_prev_body_pose || !data.dat_body_beta ||
          !data.dat_body_parent || !data.dat_body_speed || !data.dat_body_query_radius ||
          !data.dat_body_motion ||
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
          !data.dat_joint_motion || !data.dat_joint_rate ||
          !data.dat_artic_beta || !data.dat_artic_parent || !data.m || !data.qdot_flat)) ||
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
    const uint64_t body_pairs = uint64_t{p->env_count} * p->bodies_per_env * p->bodies_per_env;
    const uint64_t reach_pairs = uint64_t{p->env_count} * p->bodies_per_env *
        (uint64_t{p->surfaces_per_env} + p->bodies_per_env);
    const uint64_t reach_workspace = DatReachScratchBytes(*p);
    if (body_vertices > 0u && (reach_workspace == 0u || p->workspace_bytes < reach_workspace ||
                              !data.contact_cache_scratch)) return Status::InvalidArgument;
    const uint64_t nodes = particles + uint64_t{p->env_count} * (p->bodies_per_env + p->articulations_per_env);
    if (particles > UINT32_MAX / kTreeShares || edges > UINT32_MAX / kTreeShares ||
        triangles > UINT32_MAX || nodes > UINT32_MAX ||
        body_vertices > UINT32_MAX || slots > UINT32_MAX || body_pairs > UINT32_MAX ||
        reach_pairs > UINT32_MAX / kTreeShares ||
        uint64_t{p->env_count} * (p->edges_per_env + p->mesh_edge_sources) > UINT32_MAX ||
        p->slot_base > p->slot_stride ||
        p->slot_capacity > p->slot_stride - p->slot_base)
        return Status::InvalidArgument;
    const dim3 particle_grid((static_cast<uint32_t>(particles) + kBlockSize - 1u) / kBlockSize);
    LaunchCuda(ClearDatCountersKernel, dim3(p->env_count),
               dim3(kBlockSize), 0u, stream, *p, model, data);
    if (p->bodies_per_env > 1u && p->ogc_body_pairs != 0u)
        LaunchCuda(DatBodyPairLimitKernel,
                   dim3((p->bodies_per_env * p->bodies_per_env + kBlockSize - 1u) / kBlockSize, p->env_count),
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
        LaunchCuda(DatVertexFaceKernel,
                   dim3((static_cast<uint32_t>(particles) * kTreeShares + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (edges > 0u)
        LaunchCuda(DatEdgeEdgeKernel,
                   dim3((static_cast<uint32_t>(edges) * kTreeShares + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (edges > 0u)
        LaunchCuda(DatInitialEdgeFaceKernel,
                   dim3((static_cast<uint32_t>(edges) * kTreeShares + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    const uint64_t mixed_edges = uint64_t{p->env_count} *
        (p->edges_per_env + p->mesh_edge_sources);
    const bool mixed_contacts = p->surfaces_per_env > 0u || p->ogc_body_pairs != 0u;
    if (p->bodies_per_env > 0u && mixed_edges > 0u && mixed_contacts)
        LaunchCuda(DatInitialMixedEdgeFaceKernel,
                   dim3((static_cast<uint32_t>(mixed_edges) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (body_vertices > 0u)
        LaunchCuda(DatBodyMotionBoundKernel,
                   dim3((static_cast<uint32_t>(body_vertices) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (body_vertices > 0u) {
        int device = 0, multiprocessors = 0, resident_blocks = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks, DatBodyReachKernel,
                kBlockSize, 0u) != cudaSuccess) return Status::Failed;
        const uint32_t blocks = std::min(static_cast<uint32_t>(reach_pairs),
            static_cast<uint32_t>(std::max(multiprocessors * resident_blocks, 1)));
        LaunchCuda(DatQueueReachPairsKernel,
                   dim3((static_cast<uint32_t>(reach_pairs) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data, static_cast<uint32_t>(reach_pairs));
        LaunchCuda(DatBodyReachKernel,
                   dim3(blocks),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    }
    if (slots > 0u && mixed_contacts)
        LaunchCuda(DatMixedPairsKernel,
                   dim3((static_cast<uint32_t>(slots) + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    if (p->articulations_per_env > 0u && p->bodies_per_env > 1u && p->ogc_body_pairs != 0u)
        LaunchCuda(DatArticPairsKernel,
                   dim3(static_cast<uint32_t>((body_pairs + kBlockSize - 1u) / kBlockSize)),
                   dim3(kBlockSize), 0u, stream, *p, model, data);
    // Particle parents exist only where surfaces cleared them, and only surfaces join particles.
    const uint32_t first = p->surfaces_per_env > 0u ? 0u : static_cast<uint32_t>(particles);
    if (nodes > first) {
        const dim3 grid((static_cast<uint32_t>(nodes) - first + kBlockSize - 1u) / kBlockSize);
        LaunchCuda(DatJoinMinKernel, grid, dim3(kBlockSize), 0u, stream, *p, data, first,
                   static_cast<uint32_t>(nodes));
        LaunchCuda(DatJoinShareKernel, grid, dim3(kBlockSize), 0u, stream, *p, data, first,
                   static_cast<uint32_t>(nodes));
    }
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

uint64_t DatReachScratchBytes(const DatTruncateParams& p) {
    if (p.env_count == 0u || p.bodies_per_env == 0u || p.mesh_vertex_sources == 0u) return 0u;
    const uint64_t pairs_per_env = uint64_t{p.bodies_per_env} * (uint64_t{p.bodies_per_env} + p.surfaces_per_env);
    if (pairs_per_env > UINT32_MAX / kTreeShares ||
        p.env_count > UINT32_MAX / kTreeShares / pairs_per_env) return 0u;
    return (uint64_t{p.env_count} * pairs_per_env + 2u) * sizeof(uint32_t);
}

void RegisterNkDatTruncateOps() {
    SetCudaOp(NkOp::DatSnapshot, &OpDatSnapshot);
    SetCudaOp(NkOp::DatTruncate, &OpDatTruncate);
}

}  // namespace nuka::phi
