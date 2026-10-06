#pragma once

#include "collision/dat_geometry.hpp"
#include "collision/ogc_geometry.hpp"
#include "nk/contact/contact_profile.hpp"
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

__global__ void OgcBodyQueryKernel(OgcDetectParams p, ModelView model, DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.env_count * p.bodies_per_env) return;
    const uint32_t env = item / p.bodies_per_env;
    const uint32_t body = item % p.bodies_per_env;
    const float speed = OgcBodySpeed(p, data, env, body);
    data.dat_body_speed[item] = speed;
    float radius = collision::kDatQueryRadiusMax;
    if (model.mesh_surface_info[body].vertex_count > 0u) {
        for (uint32_t s = 0u; s < p.surfaces_per_env; ++s) {
            if (model.particle_surface_info[s].vertex_count == 0u) continue;
            const float other = data.particle_surface_max_speed[
                size_t{env} * p.surfaces_per_env + s];
            radius = fminf(radius, collision::DatQueryRadius(collision::DatMotionRadius(
                model.particle_surface_thickness[s] + p.margin, p.dt, speed, other,
                p.relaxation)));
        }
        // Links of one articulation are bounded per pair, relative to their common ancestor.
        for (uint32_t other = 0u; other < p.bodies_per_env; ++other) {
            if (!OgcBodyPairAllowed(p, model, body, other) ||
                model.mesh_surface_info[other].vertex_count == 0u ||
                DatCommonFrame(p, model, env, body, other) != ~0u) continue;
            radius = fminf(radius, collision::DatQueryRadius(collision::DatMotionRadius(
                0.0005f + p.margin, p.dt, speed, OgcBodySpeed(p, data, env, other),
                p.relaxation)));
        }
    }
    data.dat_body_query_radius[item] = radius;
}

__device__ float OgcBodyFriction(const ModelView& model, const DataView& data,
                                 uint32_t body) {
    const auto shape = nkops::LoadPrimShape(model.shape_table, body);
    return data.mat_buckets[size_t{shape.contact_profile_index} *
                            nk::kContactProfileWordCount + nk::kContactProfileMu1];
}

__device__ float OgcPairFriction(bool source_particle, bool target_particle,
                                 float source_mu, float target_mu) {
    if (source_particle && !target_particle) return target_mu;
    if (!source_particle && target_particle) return source_mu;
    return fmaxf(source_mu, target_mu);
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

__global__ void OgcMixedVertexFaceKernel(OgcDetectParams p, ModelView model,
                                         DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t per_env = p.particles_per_env + p.mesh_vertex_sources;
    const uint32_t total = p.env_count * per_env;
    if (item >= total) return;
    const uint32_t env = item / per_env, local = item % per_env;
    const bool source_particle = local < p.particles_per_env;
    const uint32_t source_index = source_particle
        ? env * p.particles_per_env + local : 0u;
    const uint64_t packed = source_particle ? 0u :
        model.mesh_vertex_sources[local - p.particles_per_env];
    const uint32_t source_body = static_cast<uint32_t>(packed >> 32u);
    const uint32_t source_vertex = static_cast<uint32_t>(packed);
    const auto particle_view = p.surfaces_per_env > 0u
        ? OgcParticleView(p, model, data, env) : collision::MeshSurfaceView{};
    uint32_t source_surface = ~0u;
    math::Vec3 source{};
    float source_radius = 0.0f, source_speed = 0.0f, source_mu = 0.0f;
    if (source_particle) {
        source = data.particle_pos[source_index];
        for (uint32_t s = 0u; s < p.surfaces_per_env; ++s) {
            const auto info = model.particle_surface_info[s];
            if (local >= info.vertex_offset && local - info.vertex_offset < info.vertex_count) {
                source_surface = s;
                source_radius = model.particle_surface_thickness[s];
                source_speed = data.particle_surface_max_speed[size_t{env} * p.surfaces_per_env + s];
                source_mu = model.particle_surface_friction[s];
                break;
            }
        }
        if (source_surface == ~0u) {
            return;
        }
    } else {
        if (source_body >= p.bodies_per_env) {
            atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
            return;
        }
        const auto info = model.mesh_surface_info[source_body];
        const auto view = OgcBodyView(p, model, data, env, source_body);
        if (!collision::MeshSurfaceRangeValid(view, info) || source_vertex >= info.vertex_count) {
            atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
            return;
        }
        source = collision::MeshSurfaceVertex(view, info, source_vertex);
        source_speed = OgcBodySpeed(p, data, env, source_body);
        source_mu = OgcBodyFriction(model, data, source_body);
    }
    const uint32_t target_groups = source_particle ? 1u : 2u;
    for (uint32_t group = 0u; group < target_groups; ++group) {
        const bool target_particle = !source_particle && group == 0u;
        const uint32_t target_count = target_particle ? p.surfaces_per_env : p.bodies_per_env;
        for (uint32_t target = 0u; target < target_count; ++target) {
            if (!target_particle) {
                // Every query radius is capped, so farther bodies fail the exact test below.
                const size_t reach_at = size_t{env} * p.bodies_per_env + target;
                if (OgcAabbDistanceSquared(source, data.body_aabb_lo[reach_at],
                                           data.body_aabb_hi[reach_at]) >
                    collision::kDatQueryRadiusMax * collision::kDatQueryRadiusMax) continue;
                const auto shape = nkops::LoadPrimShape(model.shape_table, target);
                if (shape.contype == 0u && shape.conaffinity == 0u) continue;
                if (!source_particle && !OgcBodyPairAllowed(p, model, source_body, target))
                    continue;
            }
            const auto info = target_particle ? model.particle_surface_info[target]
                                              : model.mesh_surface_info[target];
            const auto view = target_particle ? particle_view :
                OgcBodyView(p, model, data, env, target);
            if (!collision::MeshSurfaceRangeValid(view, info)) continue;
            const float target_radius = target_particle
                ? model.particle_surface_thickness[target] : 0.0f;
            const float radius = source_radius + target_radius +
                (!source_particle && !target_particle ? 0.0005f : 0.0f);
            const float target_speed = target_particle
                ? data.particle_surface_max_speed[size_t{env} * p.surfaces_per_env + target]
                : OgcBodySpeed(p, data, env, target);
            const uint32_t frame = source_particle || target_particle ? ~0u
                : DatCommonFrame(p, model, env, source_body, target);
            const float query = collision::DatQueryRadius(frame != ~0u
                ? DatArticMotionRadius(p, model, data, env, source_body, target, frame, data.q)
                : collision::DatMotionRadius(radius + p.margin, p.dt, source_speed,
                                             target_speed, p.relaxation));
            if (!(query > 0.0f)) continue;
            if (!target_particle) {
                const size_t at = size_t{env} * p.bodies_per_env + target;
                if (OgcAabbDistanceSquared(source, data.body_aabb_lo[at],
                                           data.body_aabb_hi[at]) > query * query) continue;
            }
            const math::Vec3 local_source = collision::MeshSurfaceLocalPoint(view, source);
            uint32_t cursor = 0u;
            while (cursor < info.node_count) {
                const auto node = view.nodes[info.node_offset + cursor];
                if (node.escape <= cursor || node.escape > info.node_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    break;
                }
                if (collision::MeshBoundsDistanceSquared(local_source, node) > query * query) {
                    cursor = node.escape;
                    continue;
                }
                if (node.triangle != ~0u) {
                    const auto feature = collision::OgcFacetFeature(
                        view, info,
                        target_particle ? model.particle_surface_edges : model.mesh_edges,
                        target_particle ? model.particle_surface_triangle_edges : model.mesh_triangle_edges,
                        target_particle ? model.particle_surface_edge_info[target] : model.mesh_edge_info[target],
                        target_particle
                            ? collision::MeshVertexIncidence{model.particle_surface_vertex_triangle_offsets,
                                                             model.particle_surface_vertex_triangles}
                            : collision::MeshVertexIncidence{model.mesh_vertex_triangle_offsets,
                                                             model.mesh_vertex_triangles},
                        node.triangle, source, query);
                    if (feature.feasible && feature.owner_triangle == node.triangle &&
                        feature.distance <= query && feature.distance >= 0.0f) {
                        const uint32_t ordinal = OgcClaimSlot(p, data, env);
                        if (ordinal < p.slot_capacity) {
                            const float target_mu = target_particle
                                ? model.particle_surface_friction[target]
                                : OgcBodyFriction(model, data, target);
                            OgcEmitMixedFace(p, model, data, env, ordinal,
                                source_particle, source_index, source_body, source_vertex,
                                target_particle, target, target, view, info,
                                node.triangle, source, radius,
                                OgcPairFriction(source_particle, target_particle,
                                                source_mu, target_mu), feature);
                        }
                    }
                }
                ++cursor;
            }
        }
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

__global__ void OgcMixedEdgeEdgeKernel(OgcDetectParams p, ModelView model,
                                       DataView data) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t per_env = p.edges_per_env + p.mesh_edge_sources;
    if (item >= p.env_count * per_env) return;
    const uint32_t env = item / per_env, local = item % per_env;
    const bool source_particle = local < p.edges_per_env;
    const uint64_t packed = source_particle ? 0u :
        model.mesh_edge_sources[local - p.edges_per_env];
    const uint32_t source_body = static_cast<uint32_t>(packed >> 32u);
    const uint32_t source_edge = static_cast<uint32_t>(packed);
    uint32_t source_surface = ~0u;
    collision::MeshSurfaceInfo source_info{};
    collision::MeshEdgeInfo source_edges{};
    collision::MeshSurfaceView source_view{};
    const collision::MeshEdge* source_array = nullptr;
    collision::MeshVertexIncidence source_incidence{};
    float source_radius = 0.0f, source_speed = 0.0f, source_mu = 0.0f;
    uint32_t local_edge = source_edge;
    if (source_particle) {
        for (uint32_t s = 0u; s < p.surfaces_per_env; ++s) {
            const auto info = model.particle_surface_edge_info[s];
            if (local >= info.edge_offset && local - info.edge_offset < info.edge_count) {
                source_surface = s;
                local_edge = local - info.edge_offset;
                source_info = model.particle_surface_info[s];
                source_edges = info;
                source_view = OgcParticleView(p, model, data, env);
                source_array = model.particle_surface_edges;
                source_incidence = {model.particle_surface_vertex_edge_offsets,
                                    model.particle_surface_vertex_edges};
                source_radius = model.particle_surface_thickness[s];
                source_speed = data.particle_surface_max_speed[size_t{env} * p.surfaces_per_env + s];
                source_mu = model.particle_surface_friction[s];
                break;
            }
        }
        if (source_surface == ~0u) {
            return;
        }
    } else {
        if (source_body >= p.bodies_per_env) {
            atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
            return;
        }
        source_info = model.mesh_surface_info[source_body];
        source_edges = model.mesh_edge_info[source_body];
        source_view = OgcBodyView(p, model, data, env, source_body);
        source_array = model.mesh_edges;
        source_incidence = {model.mesh_vertex_edge_offsets, model.mesh_vertex_edges};
        source_speed = OgcBodySpeed(p, data, env, source_body);
        source_mu = OgcBodyFriction(model, data, source_body);
        if (!collision::MeshSurfaceRangeValid(source_view, source_info) ||
            local_edge >= source_edges.edge_count) {
            atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
            return;
        }
    }
    const auto edge_a = source_array[source_edges.edge_offset + local_edge];
    const math::Vec3 a0 = collision::MeshSurfaceVertex(source_view, source_info, edge_a.vertex0);
    const math::Vec3 a1 = collision::MeshSurfaceVertex(source_view, source_info, edge_a.vertex1);
    const math::Vec3 midpoint = (a0 + a1) * 0.5f;
    const float half_length = sqrtf((a1 - a0).LengthSq()) * 0.5f;
    // Every query radius is capped, so farther bodies fail the exact test below; segment tests
    // add 1 um so frame rounding cannot drop a pair.
    const float reach = collision::kDatQueryRadiusMax + half_length;
    constexpr float kSlack = 1.0e-6f;
    for (uint32_t target = 0u; target < p.bodies_per_env; ++target) {
        if (!source_particle && target <= source_body) continue;
        const size_t body_at = size_t{env} * p.bodies_per_env + target;
        const math::Vec3 body_lo = data.body_aabb_lo[body_at], body_hi = data.body_aabb_hi[body_at];
        if (OgcAabbDistanceSquared(midpoint, body_lo, body_hi) > reach * reach ||
            !collision::SegmentWithinBox(a0, a1, body_lo, body_hi,
                                         collision::kDatQueryRadiusMax + kSlack)) continue;
        const auto shape = nkops::LoadPrimShape(model.shape_table, target);
        if (shape.contype == 0u && shape.conaffinity == 0u) continue;
        if (!source_particle && !OgcBodyPairAllowed(p, model, source_body, target)) continue;
        const auto target_info = model.mesh_surface_info[target];
        const auto target_edges = model.mesh_edge_info[target];
        const auto target_view = OgcBodyView(p, model, data, env, target);
        if (!collision::MeshSurfaceRangeValid(target_view, target_info) ||
            target_edges.node_count == 0u) continue;
        const float radius = source_radius + (source_particle ? 0.0f : 0.0005f);
        const uint32_t frame = source_particle ? ~0u
            : DatCommonFrame(p, model, env, source_body, target);
        const float query = collision::DatQueryRadius(frame != ~0u
            ? DatArticMotionRadius(p, model, data, env, source_body, target, frame, data.q)
            : collision::DatMotionRadius(radius + p.margin, p.dt, source_speed,
                                         OgcBodySpeed(p, data, env, target), p.relaxation));
        if (!(query > 0.0f)) continue;
        const float broad = query + half_length, within = query + kSlack;
        if (OgcAabbDistanceSquared(midpoint, body_lo, body_hi) > broad * broad ||
            !collision::SegmentWithinBox(a0, a1, body_lo, body_hi, within)) continue;
        const math::Vec3 local_midpoint = collision::MeshSurfaceLocalPoint(target_view, midpoint);
        const math::Vec3 local_a0 = collision::MeshSurfaceLocalPoint(target_view, a0);
        const math::Vec3 local_a1 = collision::MeshSurfaceLocalPoint(target_view, a1);
        uint32_t cursor = 0u;
        while (cursor < target_edges.node_count) {
            const auto node = model.mesh_edge_nodes[target_edges.node_offset + cursor];
            if (node.escape <= cursor || node.escape > target_edges.node_count) {
                atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                break;
            }
            if (collision::MeshBoundsDistanceSquared(local_midpoint, node) > broad * broad ||
                !collision::SegmentWithinBox(local_a0, local_a1, node.lower, node.upper, within)) {
                cursor = node.escape;
                continue;
            }
            if (node.triangle != ~0u) {
                if (node.triangle >= target_edges.edge_count) {
                    atomicOr(data.env_status + env, kEnvStatusContactGeometryUnavailable);
                    break;
                }
                const auto pair = collision::OgcEdgeContact(
                    source_view, source_info, source_array, source_incidence,
                    source_edges, local_edge, target_view, target_info, model.mesh_edges,
                    {model.mesh_vertex_edge_offsets, model.mesh_vertex_edges}, target_edges,
                    node.triangle, query);
                if (pair.feasible && pair.owner_edge_a == local_edge &&
                    pair.owner_edge_b == node.triangle &&
                    pair.distance >= 0.0f && pair.distance <= query) {
                    const uint32_t ordinal = OgcClaimSlot(p, data, env);
                    if (ordinal < p.slot_capacity) OgcEmitMixedEdge(p, model, data, env, ordinal,
                        source_particle, source_surface, source_body, local_edge,
                        target, node.triangle, edge_a, radius,
                        OgcPairFriction(source_particle, false, source_mu,
                                        OgcBodyFriction(model, data, target)),
                        pair);
                }
            }
            ++cursor;
        }
    }
}

}  // namespace
}  // namespace nuka::phi
