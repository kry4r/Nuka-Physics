#pragma once

#include "collision/dat_geometry.hpp"
#include "collision/ogc_geometry.hpp"
#include "nk/contact/contact_profile.hpp"
#include "scene/contact_filter.hpp"
#include "phi/backend_cuda/ops/dat_chain.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"

namespace nuka::phi {
namespace {

__device__ __forceinline__ uint32_t OgcClaimSlot(
    const OgcDetectParams& p, DataView data, uint32_t env) {
    const uint32_t active = __activemask();
    const uint32_t lane = threadIdx.x & 31u;
    uint32_t peers;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    peers = __match_any_sync(active, env);
#else
    peers = 0u;
    uint32_t pending = active;
    while (pending != 0u) {
        const uint32_t owner_env = __shfl_sync(active, env, __ffs(pending) - 1);
        const uint32_t matching = __ballot_sync(active, env == owner_env);
        if (env == owner_env) peers = matching;
        pending &= ~matching;
    }
#endif
    const uint32_t leader = static_cast<uint32_t>(__ffs(peers) - 1);
    const uint32_t requested = static_cast<uint32_t>(__popc(peers));
    uint32_t first = p.slot_capacity;
    uint32_t reserved = 0u;
    if (lane == leader) {
        // One fetch-add per group reserves the slots below capacity; finalize clamps the overshoot.
        first = atomicAdd(data.ogc_contact_count + env, requested);
        reserved = first < p.slot_capacity ? min(requested, p.slot_capacity - first) : 0u;
        if (reserved < requested) atomicOr(data.env_status + env, kEnvStatusPairOverflow);
    }
    first = __shfl_sync(peers, first, leader);
    reserved = __shfl_sync(peers, reserved, leader);
    const uint32_t rank = static_cast<uint32_t>(__popc(peers & ((1u << lane) - 1u)));
    return rank < reserved ? first + rank : p.slot_capacity;
}

__device__ collision::MeshSurfaceView OgcParticleView(
    const OgcDetectParams& p, const ModelView& model, const DataView& data,
    uint32_t env) {
    return {reinterpret_cast<const float*>(data.particle_pos +
                size_t{env} * p.particles_per_env),
            model.particle_surface_triangles,
            data.particle_surface_nodes + size_t{env} * p.nodes_per_env,
            {p.particles_per_env, p.triangles_per_env, p.nodes_per_env}};
}

__device__ collision::MeshSurfaceView OgcBodyView(
    const OgcDetectParams& p, const ModelView& model, const DataView& data,
    uint32_t env, uint32_t body) {
    const auto pose = data.body_pose[size_t{env} * p.bodies_per_env + body];
    return {model.hull_verts, model.mesh_triangles, model.mesh_bvh_nodes,
            {p.mesh_vertices, p.mesh_triangles, p.mesh_nodes},
            pose.position, pose.rotation, true};
}

__device__ bool OgcBodyPairAllowed(const OgcDetectParams& p,
                                   const ModelView& model,
                                   uint32_t a, uint32_t b) {
    if (a == b) return false;
    const auto sa = nkops::LoadPrimShape(model.shape_table, a);
    const auto sb = nkops::LoadPrimShape(model.shape_table, b);
    if (sa.body_id < 0 && sb.body_id < 0) return false;
    if (nkops::RouteMeshContact(sa.kind, sb.kind, model.mesh_contact_mode[a],
                                model.mesh_contact_mode[b]) != nkops::MeshContactRoute::Ogc)
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

// Predicted vertex speed bound: the start velocity, raised to the last untruncated DAT motion.
__device__ float OgcBodySpeed(const OgcDetectParams& p, const DataView& data,
                              uint32_t env, uint32_t body) {
    const size_t at = size_t{env} * p.bodies_per_env + body;
    const auto center = data.body_pose[at].position;
    const auto lo = data.body_aabb_lo[at], hi = data.body_aabb_hi[at];
    const math::Vec3 extents{
        fmaxf(fabsf(lo.x - center.x), fabsf(hi.x - center.x)),
        fmaxf(fabsf(lo.y - center.y), fabsf(hi.y - center.y)),
        fmaxf(fabsf(lo.z - center.z), fabsf(hi.z - center.z))};
    return fmaxf(sqrtf(data.body_linear_velocity[at].LengthSq()) +
                     sqrtf(data.body_angular_velocity[at].LengthSq()) *
                         sqrtf(extents.LengthSq()),
                 data.dat_body_motion[at]);
}

// Predicted joint speed bound: the start speed, raised to the last untruncated DAT motion.
__global__ void OgcJointRateKernel(OgcDetectParams p, DataView data) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= p.env_count * p.links_per_env) return;
    data.dat_joint_rate[link] = fmaxf(fabsf(data.qdot[link]), data.dat_joint_motion[link]);
}

constexpr uint32_t kOgcQueryLanes = 32u;

// A warp per body: its lanes take the surfaces and the other bodies, and the minimum over them is order-free.
__global__ void OgcBodyQueryKernel(OgcDetectParams p, ModelView model, DataView data) {
    const uint32_t item = (blockIdx.x * blockDim.x + threadIdx.x) / kOgcQueryLanes;
    const uint32_t lane = threadIdx.x % kOgcQueryLanes;
    if (item >= p.env_count * p.bodies_per_env) return;
    const uint32_t env = item / p.bodies_per_env;
    const uint32_t body = item % p.bodies_per_env;
    const float speed = OgcBodySpeed(p, data, env, body);
    if (lane == 0u) data.dat_body_speed[item] = speed;
    float radius = collision::kDatQueryRadiusMax;
    if (model.mesh_surface_info[body].vertex_count > 0u) {
        for (uint32_t s = lane; s < p.surfaces_per_env; s += kOgcQueryLanes) {
            if (model.particle_surface_info[s].vertex_count == 0u) continue;
            const float other = data.particle_surface_max_speed[
                size_t{env} * p.surfaces_per_env + s];
            radius = fminf(radius, collision::DatQueryRadius(collision::DatMotionRadius(
                model.particle_surface_thickness[s] + p.margin, p.dt, speed, other,
                p.relaxation)));
        }
        // Links of one articulation are bounded per pair, relative to their common ancestor.
        for (uint32_t other = lane; other < p.bodies_per_env; other += kOgcQueryLanes) {
            if (!OgcBodyPairAllowed(p, model, body, other) ||
                model.mesh_surface_info[other].vertex_count == 0u ||
                DatCommonFrame(p, model, env, body, other) != ~0u) continue;
            radius = fminf(radius, collision::DatQueryRadius(collision::DatMotionRadius(
                collision::kRigidContactSeparation + p.margin, p.dt, speed, OgcBodySpeed(p, data, env, other),
                p.relaxation)));
        }
    }
    for (uint32_t offset = kOgcQueryLanes / 2u; offset > 0u; offset /= 2u)
        radius = fminf(radius, __shfl_xor_sync(~0u, radius, offset));
    if (lane == 0u) data.dat_body_query_radius[item] = radius;
}

__device__ float OgcBodyFriction(const ModelView& model, const DataView& data,
                                 uint32_t body, uint32_t env, uint32_t profiles_per_env) {
    const auto shape = nkops::LoadPrimShape(model.shape_table, body);
    return data.mat_buckets[(size_t{env} * profiles_per_env + shape.contact_profile_index) *
                            nk::kContactProfileWordCount + nk::kContactProfileMu1];
}

// Two bodies under an authored pair slide with the pair's coefficient.
__device__ float OgcPairFriction(const OgcDetectParams& p, const ModelView& model, const DataView& data,
                                 uint32_t env, bool source_particle, bool target_particle,
                                 uint32_t source_owner, uint32_t target_owner, float source_mu, float target_mu) {
    if (source_particle != target_particle) return scene::ParticleBodyFriction(
        source_particle ? source_mu : target_mu, source_particle ? target_mu : source_mu);
    const uint32_t authored = source_particle ? ~0u : nkops::AuthoredPairProfile(
        model.contact_pair_keys, model.contact_pair_profiles, p.contact_pairs, source_owner, target_owner);
    if (authored != ~0u)
        return data.mat_buckets[(size_t{env} * p.num_material_buckets + authored) *
                                nk::kContactProfileWordCount + nk::kContactProfileMu1];
    return fmaxf(source_mu, target_mu);
}

struct OgcParticleMaterial {
    float half_thickness = 0.0f;
    float friction = 0.0f;
};

// Contact half thickness and friction of a particle face, interpolated at barycentric weights.
__device__ OgcParticleMaterial OgcParticleFaceMaterial(
    const ModelView& model, const collision::MeshSurfaceView& view,
    const collision::MeshSurfaceInfo& info, uint32_t triangle, math::Vec3 weights) {
    const size_t tri_at = (size_t{info.triangle_offset} + triangle) * 3u;
    const float w[3] = {weights.x, weights.y, weights.z};
    OgcParticleMaterial out;
    for (uint32_t j = 0u; j < 3u; ++j) {
        const uint32_t local = info.vertex_offset + view.triangles[tri_at + j];
        out.half_thickness += w[j] * model.particle_half_thickness[local];
        out.friction += w[j] * model.particle_friction[local];
    }
    return out;
}

__device__ constraint::CollidableRef OgcBodyReference(
    const OgcDetectParams& p, const ModelView& model,
    uint32_t env, uint32_t body) {
    constraint::CollidableRef ref;
    const uint32_t link = model.body_to_link ? model.body_to_link[body] : ~0u;
    if (link != ~0u) {
        ref.type = constraint::CollidableType::ArticulationLink;
        ref.handle = env * p.links_per_env + link;
    } else {
        ref.type = constraint::CollidableType::RigidBody;
        ref.handle = env * p.bodies_per_env + body;
    }
    return ref;
}

__device__ float OgcAabbDistanceSquared(math::Vec3 point,
                                        math::Vec3 lower, math::Vec3 upper) {
    const math::Vec3 delta{
        fmaxf(0.0f, fmaxf(lower.x - point.x, point.x - upper.x)),
        fmaxf(0.0f, fmaxf(lower.y - point.y, point.y - upper.y)),
        fmaxf(0.0f, fmaxf(lower.z - point.z, point.z - upper.z))};
    return delta.LengthSq();
}

__device__ void OgcEmitMixedFace(
    const OgcDetectParams& p, const ModelView& model, DataView data,
    uint32_t env, uint64_t ordinal, bool source_particle,
    uint32_t source_index, uint32_t source_body, uint32_t source_vertex,
    bool target_particle, uint32_t target_index, uint32_t target_body,
    const collision::MeshSurfaceView& target_view,
    const collision::MeshSurfaceInfo& target_info, uint32_t triangle,
    math::Vec3 source_point, float radius, float friction,
    const collision::OgcTriangleFeature& feature) {
    if (ordinal >= p.slot_capacity) {
        atomicOr(data.env_status + env, kEnvStatusPairOverflow);
        return;
    }
    const uint32_t slot = env * p.slot_stride + p.slot_base + ordinal;
    const size_t at = size_t{slot} * nk::kPairDrivenPtsPerSlot;
    data.dat_pair_kind[slot] = 1u | (source_particle ? 4u : 0u) |
        (target_particle ? 8u : 0u);
    data.dat_pair_owner_a[slot] = source_particle ? source_index : source_body;
    data.dat_pair_owner_b[slot] = target_body;
    data.dat_pair_feature_a[slot] = source_vertex;
    data.dat_pair_feature_b[slot] = triangle;
    data.ucontact_a[at] = source_particle ? source_index : source_body;
    data.ucontact_a_kind[at] = source_particle
        ? nk::kUContactSideParticle : nk::kUContactSideBody;
    if (target_particle) {
        const uint32_t endpoint = env * p.point_endpoints_per_env + p.point_endpoint_first +
                                  ordinal * 2u + 1u;
        const uint32_t first = env * p.point_endpoint_terms_per_env + p.point_endpoint_term_first +
                               ordinal * 6u + 3u;
        const size_t tri_at = (size_t{target_info.triangle_offset} + triangle) * 3u;
        const float weights[3] = {feature.barycentric.x, feature.barycentric.y,
                                  feature.barycentric.z};
        for (uint32_t j = 0u; j < 3u; ++j) {
            const uint32_t particle = env * p.particles_per_env +
                target_info.vertex_offset + target_view.triangles[tri_at + j];
            data.point_endpoint_terms[first + j] =
                nk::WeightedPointEndpointTerm(nk::kNkSideParticle, particle, weights[j]);
        }
        const uint32_t terms = nk::CanonicalizePointEndpointTerms(
            data.point_endpoint_terms + first, 3u);
        data.point_endpoint_ranges[endpoint] = {first, terms};
        data.ucontact_b[at] = endpoint;
        data.ucontact_b_kind[at] = nk::kUContactSidePointEndpoint;
    } else {
        data.ucontact_b[at] = target_body;
        data.ucontact_b_kind[at] = nk::kUContactSideBody;
    }
    data.ucontact_point[at] = (source_point + feature.point) * 0.5f;
    data.ucontact_witness_a[at] = source_point;
    data.ucontact_witness_b[at] = feature.point;
    data.ucontact_normal[at] = feature.normal;
    data.ucontact_depth[at] = radius - feature.distance;
    data.ucontact_gen[at] = 1u;
    data.ucontact_law[slot] = nk::kContactLawSpeculative;
    data.ucontact_friction[slot] = friction;
    nk::CanonicalContactDescriptor descriptor;
    descriptor.a = source_particle
        ? constraint::CollidableRef{constraint::CollidableType::Particle,
                                   constraint::ReactionProviderKind::ParticleInvMass,
                                   source_index}
        : OgcBodyReference(p, model, env, source_body);
    descriptor.b = target_particle
        ? constraint::CollidableRef{constraint::CollidableType::ParticleSurface,
                                   constraint::ReactionProviderKind::ParticleInvMass,
                                   env * p.surfaces_per_env + target_index}
        : OgcBodyReference(p, model, env, target_body);
    descriptor.feature_a = source_particle ? 0u :
        model.mesh_surface_info[source_body].vertex_offset + source_vertex;
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

struct OgcSurface {
    collision::MeshSurfaceInfo info;
    collision::MeshEdgeInfo edges;
    collision::MeshSurfaceView view;
    const collision::MeshEdge* edge_array;
    const uint32_t* triangle_edges;
    collision::MeshVertexIncidence vertex_faces, vertex_edges;
    uint32_t owner;
    bool particle;
    float speed, radius, friction;
};

__device__ OgcSurface OgcLoadSurface(const OgcDetectParams& p, const ModelView& model,
    const DataView& data, uint32_t env, uint32_t index) {
    OgcSurface out{};
    out.particle = index < p.surfaces_per_env;
    out.owner = out.particle ? index : index - p.surfaces_per_env;
    if (out.particle) {
        out.info = model.particle_surface_info[out.owner];
        out.edges = model.particle_surface_edge_info[out.owner];
        out.view = OgcParticleView(p, model, data, env);
        out.edge_array = model.particle_surface_edges;
        out.triangle_edges = model.particle_surface_triangle_edges;
        out.vertex_faces = {model.particle_surface_vertex_triangle_offsets, model.particle_surface_vertex_triangles};
        out.vertex_edges = {model.particle_surface_vertex_edge_offsets, model.particle_surface_vertex_edges};
        out.speed = data.particle_surface_max_speed[size_t{env} * p.surfaces_per_env + out.owner];
        out.radius = model.particle_surface_thickness[out.owner];
    } else {
        out.info = model.mesh_surface_info[out.owner];
        out.edges = model.mesh_edge_info[out.owner];
        out.view = OgcBodyView(p, model, data, env, out.owner);
        out.edge_array = model.mesh_edges;
        out.triangle_edges = model.mesh_triangle_edges;
        out.vertex_faces = {model.mesh_vertex_triangle_offsets, model.mesh_vertex_triangles};
        out.vertex_edges = {model.mesh_vertex_edge_offsets, model.mesh_vertex_edges};
        out.speed = data.dat_body_speed[size_t{env} * p.bodies_per_env + out.owner];
        out.friction = OgcBodyFriction(model, data, out.owner, env, p.num_material_buckets);
    }
    return out;
}

struct OgcSurfacePair {
    OgcSurface source, target;
    float body_query;
    bool active;
};

// A block compacts its nearby surface pairs before assigning complete warps to feature queries.
__device__ bool OgcPairPotential(const OgcDetectParams& p, const ModelView& model,
    const DataView& data, uint32_t env, uint32_t source, uint32_t target, bool edges) {
    const bool sp = source < p.surfaces_per_env, tp = target < p.surfaces_per_env;
    if ((sp && tp) || (edges && (tp || (!sp && target <= source)))) return false;
    const uint32_t a = source - p.surfaces_per_env, b = target - p.surfaces_per_env;
    if (!sp) {
        const auto shape = nkops::LoadPrimShape(model.shape_table, a);
        if (shape.contype == 0u || shape.conaffinity == 0u) return false;
    }
    if (!tp) {
        const auto shape = nkops::LoadPrimShape(model.shape_table, b);
        if (shape.contype == 0u && shape.conaffinity == 0u) return false;
    }
    if (!sp && !tp) {
        const size_t ai = size_t{env} * p.bodies_per_env + a, bi = size_t{env} * p.bodies_per_env + b;
        const auto al = data.body_aabb_lo[ai], ah = data.body_aabb_hi[ai];
        const auto bl = data.body_aabb_lo[bi], bh = data.body_aabb_hi[bi];
        const math::Vec3 gap{fmaxf(0.0f, fmaxf(al.x - bh.x, bl.x - ah.x)),
            fmaxf(0.0f, fmaxf(al.y - bh.y, bl.y - ah.y)), fmaxf(0.0f, fmaxf(al.z - bh.z, bl.z - ah.z))};
        const float reach = collision::kDatQueryRadiusMax + 1.0e-6f;
        if (gap.LengthSq() > reach * reach || !OgcBodyPairAllowed(p, model, a, b)) return false;
    }
    return true;
}

__device__ void OgcPreparePair(const OgcDetectParams& p, const ModelView& model,
    const DataView& data, uint32_t env, uint32_t source, uint32_t target, OgcSurfacePair* pair) {
    pair->active = false;
    const bool sp = source < p.surfaces_per_env, tp = target < p.surfaces_per_env;
    const uint32_t a = source - p.surfaces_per_env, b = target - p.surfaces_per_env;
    pair->source = OgcLoadSurface(p, model, data, env, source);
    pair->target = OgcLoadSurface(p, model, data, env, target);
    if (!collision::MeshSurfaceRangeValid(pair->source.view, pair->source.info) ||
        !collision::MeshSurfaceRangeValid(pair->target.view, pair->target.info)) return;
    pair->body_query = 0.0f;
    if (!sp && !tp) {
        const uint32_t frame = DatCommonFrame(p, model, env, a, b);
        pair->body_query = collision::DatQueryRadius(frame != ~0u
            ? DatArticMotionRadius(p, model, data, env, a, b, frame, data.q)
            : collision::DatMotionRadius(collision::kRigidContactSeparation + p.margin, p.dt,
                pair->source.speed, pair->target.speed, p.relaxation));
    }
    pair->active = true;
}

__device__ float OgcSurfaceQuery(const OgcDetectParams& p, const OgcSurfacePair& pair, float radius) {
    return !pair.source.particle && !pair.target.particle ? pair.body_query
        : collision::DatQueryRadius(collision::DatMotionRadius(radius + p.margin, p.dt,
            pair.source.speed, pair.target.speed, p.relaxation));
}

struct OgcMixedCandidate { uint32_t source, target; };

__global__ void OgcMixedVertexFaceKernel(OgcDetectParams p, ModelView model, DataView data) {
    constexpr uint32_t kWarps = 4u;
    __shared__ OgcMixedCandidate queue[kWarps][2u * kWarpLanes];
    __shared__ __align__(alignof(OgcSurfacePair)) unsigned char storage[sizeof(OgcSurfacePair)];
    auto* pairs = reinterpret_cast<OgcSurfacePair*>(storage);
    const uint32_t warp = threadIdx.x / kWarpLanes, lane = threadIdx.x % kWarpLanes;
    __shared__ uint32_t active_targets[kWarpLanes], active_count;
    const uint32_t surfaces = p.surfaces_per_env + p.bodies_per_env;
    const uint32_t groups = (surfaces + kWarpLanes - 1u) / kWarpLanes;
    const uint32_t item = blockIdx.x / groups;
    const uint32_t env = item / surfaces, source = item % surfaces;
    const uint32_t first_target = (blockIdx.x % groups) * kWarpLanes;
    const uint32_t share = warp;
    if (warp == 0u) {
        const uint32_t target = first_target + lane;
        const bool allowed = target < surfaces && OgcPairPotential(p, model, data, env, source, target, false);
        const uint32_t mask = __ballot_sync(~0u, allowed);
        if (allowed) active_targets[__popc(mask & ((1u << lane) - 1u))] = target;
        if (lane == 0u) active_count = __popc(mask);
    }
    __syncthreads();
    for (uint32_t candidate = 0u; candidate < active_count; ++candidate) {
        __syncthreads();
        const uint32_t target = active_targets[candidate];
        auto& pair = *pairs;
        if (threadIdx.x == 0u) OgcPreparePair(p, model, data, env, source, target, &pair);
        __syncthreads();
        if (!pair.active) continue;
        const auto& a = pair.source;
        const auto& b = pair.target;
        uint32_t vertex = share * kWarpLanes + lane, cursor = b.info.node_count;
        bool entered = false;
        math::Vec3 point{}, local{};
        float query = 0.0f;
        const auto next = [&](OgcMixedCandidate* candidate) {
            while (vertex < a.info.vertex_count) {
                if (cursor >= b.info.node_count) {
                    if (entered) vertex += blockDim.x;
                    if (vertex >= a.info.vertex_count) break;
                    entered = true;
                    point = collision::MeshSurfaceVertex(a.view, a.info, vertex);
                    const float radius = a.particle ? model.particle_half_thickness[a.info.vertex_offset + vertex]
                        : b.particle ? b.radius : collision::kRigidContactSeparation;
                    query = OgcSurfaceQuery(p, pair, radius);
                    if (!(query > 0.0f)) continue;
                    local = collision::MeshSurfaceLocalPoint(b.view, point);
                    cursor = 0u;
                }
                const auto node = b.view.nodes[b.info.node_offset + cursor];
                if (node.escape <= cursor || node.escape > b.info.node_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    cursor = b.info.node_count;
                    continue;
                }
                if (collision::MeshBoundsDistanceSquared(local, node) > query * query) {
                    cursor = node.escape;
                    continue;
                }
                ++cursor;
                if (node.triangle == ~0u) continue;
                *candidate = {vertex, node.triangle};
                return true;
            }
            return false;
        };
        TestInWarpBatches(queue[warp], vertex < a.info.vertex_count, next, [&](const OgcMixedCandidate& c) {
            const auto source = collision::MeshSurfaceVertex(a.view, a.info, c.source);
            const uint32_t particle = a.info.vertex_offset + c.source;
            const float source_radius = a.particle ? model.particle_half_thickness[particle] : 0.0f;
            const float radius = source_radius + b.radius + (!a.particle && !b.particle ? collision::kRigidContactSeparation : 0.0f);
            const float reach = OgcSurfaceQuery(p, pair, radius);
            const auto feature = collision::OgcFacetFeature(b.view, b.info, b.edge_array, b.triangle_edges,
                b.edges, b.vertex_faces, c.target, source, reach);
            if (!feature.feasible || feature.owner_triangle != c.target ||
                !(feature.distance >= 0.0f && feature.distance <= reach)) return;
            const uint32_t ordinal = OgcClaimSlot(p, data, env);
            if (ordinal >= p.slot_capacity) return;
            const auto face = b.particle ? OgcParticleFaceMaterial(model, b.view, b.info, c.target, feature.barycentric)
                                        : OgcParticleMaterial{};
            OgcEmitMixedFace(p, model, data, env, ordinal, a.particle,
                env * p.particles_per_env + particle, a.owner, c.source, b.particle, b.owner, b.owner,
                b.view, b.info, c.target, source, b.particle ? source_radius + face.half_thickness : radius,
                OgcPairFriction(p, model, data, env, a.particle, b.particle, a.owner, b.owner,
                    a.particle ? model.particle_friction[particle] : a.friction,
                    b.particle ? face.friction : b.friction), feature);
        });
    }
}

__device__ void OgcEmitMixedEdge(
    const OgcDetectParams& p, const ModelView& model, DataView data,
    uint32_t env, uint64_t ordinal, bool source_particle,
    uint32_t source_surface, uint32_t source_body, uint32_t source_edge,
    uint32_t target_body, uint32_t target_edge,
    const collision::MeshEdge& edge_a, float radius, float friction,
    const collision::OgcEdgePair& pair) {
    if (ordinal >= p.slot_capacity) {
        atomicOr(data.env_status + env, kEnvStatusPairOverflow);
        return;
    }
    const uint32_t slot = env * p.slot_stride + p.slot_base + ordinal;
    const size_t at = size_t{slot} * nk::kPairDrivenPtsPerSlot;
    data.dat_pair_kind[slot] = 2u | (source_particle ? 4u : 0u);
    data.dat_pair_owner_a[slot] = source_particle ? source_surface : source_body;
    data.dat_pair_owner_b[slot] = target_body;
    data.dat_pair_feature_a[slot] = source_edge;
    data.dat_pair_feature_b[slot] = target_edge;
    if (source_particle) {
        const uint32_t endpoint = env * p.point_endpoints_per_env + p.point_endpoint_first +
                                  ordinal * 2u;
        const uint32_t first = env * p.point_endpoint_terms_per_env + p.point_endpoint_term_first +
                               ordinal * 6u;
        const auto info = model.particle_surface_info[source_surface];
        const uint32_t vertices[2] = {edge_a.vertex0, edge_a.vertex1};
        for (uint32_t j = 0u; j < 2u; ++j) {
            const uint32_t particle = env * p.particles_per_env + info.vertex_offset + vertices[j];
            data.point_endpoint_terms[first + j] = nk::WeightedPointEndpointTerm(
                nk::kNkSideParticle, particle,
                j == 0u ? 1.0f - pair.weight_a : pair.weight_a);
        }
        const uint32_t terms = nk::CanonicalizePointEndpointTerms(
            data.point_endpoint_terms + first, 2u);
        data.point_endpoint_ranges[endpoint] = {first, terms};
        data.ucontact_a[at] = endpoint;
        data.ucontact_a_kind[at] = nk::kUContactSidePointEndpoint;
    } else {
        data.ucontact_a[at] = source_body;
        data.ucontact_a_kind[at] = nk::kUContactSideBody;
    }
    data.ucontact_b[at] = target_body;
    data.ucontact_b_kind[at] = nk::kUContactSideBody;
    data.ucontact_point[at] = (pair.point_a + pair.point_b) * 0.5f;
    data.ucontact_witness_a[at] = pair.point_a;
    data.ucontact_witness_b[at] = pair.point_b;
    data.ucontact_normal[at] = pair.normal;
    data.ucontact_depth[at] = radius - pair.distance;
    data.ucontact_gen[at] = 1u;
    data.ucontact_law[slot] = nk::kContactLawSpeculative;
    data.ucontact_friction[slot] = friction;
    nk::CanonicalContactDescriptor descriptor;
    descriptor.a = source_particle
        ? constraint::CollidableRef{constraint::CollidableType::ParticleSurface,
                                   constraint::ReactionProviderKind::ParticleInvMass,
                                   env * p.surfaces_per_env + source_surface}
        : OgcBodyReference(p, model, env, source_body);
    descriptor.b = OgcBodyReference(p, model, env, target_body);
    descriptor.feature_a = source_particle
        ? model.particle_surface_edge_info[source_surface].edge_offset + source_edge
        : model.mesh_edge_info[source_body].edge_offset + source_edge;
    descriptor.feature_b = model.mesh_edge_info[target_body].edge_offset + target_edge;
    descriptor.contact_kind = nk::kContactKindOgcEdgeEdge;
    descriptor.normal = pair.normal;
    const nk::ContactId id = nk::MakeContactId(descriptor);
    data.ucontact_id_pair[at] = id.pair;
    data.ucontact_id_feature[at] = id.feature;
    data.ucontact_count[slot] = 1u;
    atomicAdd(data.contact_count + env, 1u);
}

__global__ void OgcMixedEdgeEdgeKernel(OgcDetectParams p, ModelView model, DataView data) {
    constexpr uint32_t kWarps = 4u;
    __shared__ OgcMixedCandidate queue[kWarps][2u * kWarpLanes];
    __shared__ __align__(alignof(OgcSurfacePair)) unsigned char storage[sizeof(OgcSurfacePair)];
    auto* pairs = reinterpret_cast<OgcSurfacePair*>(storage);
    const uint32_t warp = threadIdx.x / kWarpLanes, lane = threadIdx.x % kWarpLanes;
    __shared__ uint32_t active_targets[kWarpLanes], active_count;
    const uint32_t surfaces = p.surfaces_per_env + p.bodies_per_env;
    const uint32_t groups = (surfaces + kWarpLanes - 1u) / kWarpLanes;
    const uint32_t item = blockIdx.x / groups;
    const uint32_t env = item / surfaces, source = item % surfaces;
    const uint32_t first_target = (blockIdx.x % groups) * kWarpLanes;
    const uint32_t share = warp;
    if (warp == 0u) {
        const uint32_t target = first_target + lane;
        const bool allowed = target < surfaces && OgcPairPotential(p, model, data, env, source, target, true);
        const uint32_t mask = __ballot_sync(~0u, allowed);
        if (allowed) active_targets[__popc(mask & ((1u << lane) - 1u))] = target;
        if (lane == 0u) active_count = __popc(mask);
    }
    __syncthreads();
    for (uint32_t candidate = 0u; candidate < active_count; ++candidate) {
        __syncthreads();
        const uint32_t target = active_targets[candidate];
        auto& pair = *pairs;
        if (threadIdx.x == 0u) OgcPreparePair(p, model, data, env, source, target, &pair);
        __syncthreads();
        if (!pair.active || pair.target.edges.node_count == 0u) continue;
        const auto& a = pair.source;
        const auto& b = pair.target;
        const float radius = a.radius + (a.particle ? 0.0f : collision::kRigidContactSeparation);
        const float query = OgcSurfaceQuery(p, pair, radius), within = query + 1.0e-6f;
        if (!(query > 0.0f)) continue;
        uint32_t edge = share * kWarpLanes + lane, cursor = b.edges.node_count;
        bool entered = false;
        math::Vec3 local_a0{}, local_a1{}, midpoint{};
        float broad = 0.0f;
        const auto next = [&](OgcMixedCandidate* candidate) {
            while (edge < a.edges.edge_count) {
                if (cursor >= b.edges.node_count) {
                    if (entered) edge += blockDim.x;
                    if (edge >= a.edges.edge_count) break;
                    entered = true;
                    const auto feature = a.edge_array[a.edges.edge_offset + edge];
                    const auto a0 = collision::MeshSurfaceVertex(a.view, a.info, feature.vertex0);
                    const auto a1 = collision::MeshSurfaceVertex(a.view, a.info, feature.vertex1);
                    local_a0 = collision::MeshSurfaceLocalPoint(b.view, a0);
                    local_a1 = collision::MeshSurfaceLocalPoint(b.view, a1);
                    midpoint = collision::MeshSurfaceLocalPoint(b.view, (a0 + a1) * 0.5f);
                    broad = query + sqrtf((a1 - a0).LengthSq()) * 0.5f;
                    cursor = 0u;
                }
                const auto node = model.mesh_edge_nodes[b.edges.node_offset + cursor];
                if (node.escape <= cursor || node.escape > b.edges.node_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    cursor = b.edges.node_count;
                    continue;
                }
                if (collision::MeshBoundsDistanceSquared(midpoint, node) > broad * broad ||
                    !collision::SegmentWithinBox(local_a0, local_a1, node.lower, node.upper, within)) {
                    cursor = node.escape;
                    continue;
                }
                ++cursor;
                if (node.triangle == ~0u) continue;
                if (node.triangle >= b.edges.edge_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    cursor = b.edges.node_count;
                    continue;
                }
                *candidate = {edge, node.triangle};
                return true;
            }
            return false;
        };
        TestInWarpBatches(queue[warp], edge < a.edges.edge_count, next, [&](const OgcMixedCandidate& c) {
            const auto contact = collision::OgcEdgeContact(a.view, a.info, a.edge_array, a.vertex_edges,
                a.edges, c.source, b.view, b.info, b.edge_array, b.vertex_edges, b.edges, c.target, query);
            if (!contact.feasible || contact.owner_edge_a != c.source || contact.owner_edge_b != c.target ||
                !(contact.distance >= 0.0f && contact.distance <= query)) return;
            const auto edge_a = a.edge_array[a.edges.edge_offset + c.source];
            float contact_radius = radius, mu = a.friction;
            if (a.particle) {
                const uint32_t v0 = a.info.vertex_offset + edge_a.vertex0, v1 = a.info.vertex_offset + edge_a.vertex1;
                contact_radius = (1.0f - contact.weight_a) * model.particle_half_thickness[v0] +
                    contact.weight_a * model.particle_half_thickness[v1];
                mu = (1.0f - contact.weight_a) * model.particle_friction[v0] +
                    contact.weight_a * model.particle_friction[v1];
            }
            const uint32_t ordinal = OgcClaimSlot(p, data, env);
            if (ordinal < p.slot_capacity) OgcEmitMixedEdge(p, model, data, env, ordinal, a.particle,
                a.owner, a.owner, c.source, b.owner, c.target, edge_a, contact_radius,
                OgcPairFriction(p, model, data, env, a.particle, false, a.owner, b.owner, mu, b.friction), contact);
        });
    }
}

}  // namespace
}  // namespace nuka::phi
