#pragma once

#include "collision/mesh_surface.hpp"
#include "nk/contact/contact_identity.hpp"
#include "nk/material/mpm_transfer.hpp"
#include "nk/model/generated/views.hpp"
#include "nk/solve/collidable_owner.hpp"
#include "phi/backend_cuda/ops/rigid_types.cuh"
#include "phi/backend_cuda/ops/surface_query.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi::mpm_contact {

template <class Visitor>
__device__ bool VisitStencil(const MpmParams& p, uint32_t env, math::Vec3 point, Visitor visit) {
    const float inverse_dx = 1.0f / p.dx;
    const auto x = nk::MpmQuadraticWeights((point.x - p.grid_origin[0]) * inverse_dx);
    const auto y = nk::MpmQuadraticWeights((point.y - p.grid_origin[1]) * inverse_dx);
    const auto z = nk::MpmQuadraticWeights((point.z - p.grid_origin[2]) * inverse_dx);
    for (uint32_t c = 0u; c < nk::kMpmStencilWidth; ++c) {
        for (uint32_t b = 0u; b < nk::kMpmStencilWidth; ++b) {
            for (uint32_t a = 0u; a < nk::kMpmStencilWidth; ++a) {
                const float weight = x.w[a] * y.w[b] * z.w[c];
                if (!isfinite(weight)) return false;
                if (!(weight > 0.0f)) continue;
                const int64_t node = nk::MpmNodeIndex(env, x.base + a, y.base + b, z.base + c,
                                                     p.grid_dims, p.nodes_per_env);
                if (node < 0) return false;
                visit(static_cast<uint32_t>(node), weight);
            }
        }
    }
    return true;
}

// Count and emit traverse the same material points and collidables in stable order.
template <bool emit>
__device__ void Visit(const MpmParams& p, const DataView& data, uint32_t sample,
    uint32_t mpm_per_env, uint64_t& count, uint32_t side_kind, uint32_t side_index,
    constraint::CollidableRef side, math::Vec3 point, math::Vec3 normal,
    float depth, float friction, uint32_t feature, uint32_t topology) {
    const uint64_t ordinal = count++;
    if constexpr (!emit) return;
    const uint32_t env = sample / mpm_per_env;
    const uint64_t local = data.grid_contact_offset[sample] -
        data.grid_contact_offset[env * mpm_per_env] + ordinal;
    if (local >= p.contact_capacity) return;
    const uint32_t slot = env * p.contact_slots_per_env + p.contact_slot_base +
                          static_cast<uint32_t>(local);
    const size_t address = static_cast<size_t>(slot) * nk::kPairDrivenPtsPerSlot;
    data.ucontact_count[slot] = 1u;
    data.ucontact_point[address] = point;
    data.ucontact_normal[address] = normal;
    data.ucontact_depth[address] = depth;
    data.ucontact_a[address] = env * p.point_endpoints_per_env + sample % mpm_per_env;
    data.ucontact_b[address] = side_index;
    data.ucontact_a_kind[address] = nk::kUContactSidePointEndpoint;
    data.ucontact_b_kind[address] = side_kind;
    data.ucontact_gen[address] = 1u;
    data.ucontact_law[slot] = nk::kContactLawSpeculative;
    data.ucontact_friction[slot] = friction;
    nk::CanonicalContactDescriptor descriptor;
    descriptor.a = {constraint::CollidableType::MaterialPoint,
        constraint::ReactionProviderKind::PointEndpoint, env * p.particles_per_env + sample % mpm_per_env};
    descriptor.b = side;
    descriptor.normal = normal;
    descriptor.feature_a = 0u;
    descriptor.feature_b = feature;
    descriptor.topology_version = topology;
    const nk::ContactId id = nk::MakeContactId(descriptor);
    data.ucontact_id_pair[address] = id.pair;
    data.ucontact_id_feature[address] = id.feature;
}

// Floor first, then the grid walls, in a fixed order.
template <bool emit>
__device__ void VisitBoundaries(const MpmParams& p, const DataView& data, uint32_t sample,
                                uint32_t mpm_per_env, uint64_t& count, math::Vec3 point,
                                float reach) {
    const uint32_t env = sample / mpm_per_env;
    const auto boundary = [&](uint32_t id, math::Vec3 normal, float depth, float mu) {
        if (depth < -reach) return;
        Visit<emit>(p, data, sample, mpm_per_env, count, nk::kUContactSideBoundary, id,
            {constraint::CollidableType::StaticBoundary,
             constraint::ReactionProviderKind::StaticNull, env * nk::kMpmBoundaryCount + id},
            point, normal, depth, mu, id, 0u);
    };
    const math::Vec3 floor_normal{p.plane_n[0], p.plane_n[1], p.plane_n[2]};
    boundary(0u, floor_normal, p.plane_d - point.Dot(floor_normal), p.plane_mu);
    boundary(1u, {1, 0, 0}, p.grid_origin[0] + p.dx - point.x, 0.0f);
    boundary(2u, {-1, 0, 0}, point.x - (p.grid_origin[0] + (p.grid_dims[0] - 2u) * p.dx), 0.0f);
    boundary(3u, {0, 1, 0}, p.grid_origin[1] + p.dx - point.y, 0.0f);
    boundary(4u, {0, -1, 0}, point.y - (p.grid_origin[1] + (p.grid_dims[1] - 2u) * p.dx), 0.0f);
    boundary(5u, {0, 0, 1}, p.grid_origin[2] + p.dx - point.z, 0.0f);
}

struct BodyHit {
    bool hit = false;
    constraint::CollidableRef endpoint;
    math::Vec3 normal{};
    float depth = 0.0f;
    uint32_t feature = 0u;
};

// A body is hit when its surface lies within the band the sample and body can close in a step.
__device__ inline BodyHit QueryBody(const MpmParams& p, const ModelView& model,
    const DataView& data, const nkops::SurfaceQueryView& surfaces, uint32_t env, uint32_t body,
    math::Vec3 point, float reach, uint32_t& status) {
    BodyHit result;
    const auto shape = nkops::LoadPrimShape(model.shape_table, body);
    if ((shape.contype | shape.conaffinity) == 0u) return result;
    const auto owner = nk::ResolveCollidableOwner(shape.body_id, env, body,
        p.bodies_per_env, p.base_link_count, p.artics_per_env, model.body_to_link,
        model.body_to_articulation, model.body_collidable_body);
    if (owner.kind == ~0u) {
        status |= kEnvStatusInvalidEndpoint;
        return result;
    }
    math::Vec3 linear{}, angular{}, origin{};
    if (owner.kind == nk::kNkSideArtic) {
        const auto pose = data.link_pose[owner.link];
        const auto v = data.link_velocity[owner.link];
        angular = nkops::PrimRotate(pose.rotation, {v.v[0], v.v[1], v.v[2]});
        linear = nkops::PrimRotate(pose.rotation, {v.v[3], v.v[4], v.v[5]});
        origin = pose.position;
    } else if (owner.kind == nk::kNkSideRigid) {
        linear = data.body_linear_velocity[owner.body];
        angular = data.body_angular_velocity[owner.body];
        origin = nkops::BodyCenterOfMass(data.body_pose[owner.body], data.body_inertial_frame[owner.body]);
    }
    const float speed = sqrtf(linear.LengthSq()) + sqrtf(angular.LengthSq() * (point - origin).LengthSq());
    const float query_distance = p.body_band + reach + p.dt * speed;
    const auto pose = data.body_pose[env * p.bodies_per_env + body];
    const auto surface = nkops::QueryCollidableSurface(surfaces, body, shape,
        nkops::PrimInverseTransformPoint(pose, point), query_distance);
    if (!surface.valid) {
        status |= kEnvStatusMpmOneWayBody | kEnvStatusContactGeometryUnavailable;
        return result;
    }
    if (!isfinite(surface.distance) || surface.distance > query_distance) return result;
    const math::Vec3 raw_normal = nkops::PrimRotate(pose.rotation, surface.normal);
    const float length = sqrtf(raw_normal.LengthSq());
    if (!isfinite(length) || length <= 1.0e-8f) return result;
    result.endpoint.handle = owner.body;
    result.endpoint.type = constraint::CollidableType::RigidBody;
    result.endpoint.react = constraint::ReactionProviderKind::RigidInvMass;
    if (owner.kind == nk::kNkSideArtic) {
        result.endpoint.handle = owner.link;
        result.endpoint.type = constraint::CollidableType::ArticulationLink;
        result.endpoint.react = constraint::ReactionProviderKind::ArticulationChainJ;
    } else if (owner.kind == nk::kNkSideStatic) {
        result.endpoint.type = constraint::CollidableType::StaticWorld;
        result.endpoint.react = constraint::ReactionProviderKind::StaticNull;
    }
    result.hit = true;
    result.normal = raw_normal * (1.0f / length);
    result.depth = -surface.distance;
    result.feature = surface.feature;
    return result;
}

// QueryBodies stores each warp's hits on a body contiguously so emission need not query them
// again; first is per (sample warp, body), or ~0u when the records were full.
struct BodyHitCache {
    BodyHit* records = nullptr;
    uint32_t* first = nullptr;
    uint32_t* count = nullptr;
    uint32_t capacity = 0u;
};

// Particle surfaces follow bodies; emission writes each hit's triangle endpoint terms.
template <bool emit>
__device__ void VisitParticleSurfaces(const MpmParams& p, const ModelView& model,
    const DataView& data, uint32_t sample, uint32_t mpm_per_env, uint64_t& count,
    math::Vec3 point, float reach, uint32_t* contact_hit_mask, size_t mask_base,
    uint32_t& status) {
    const uint32_t env = sample / mpm_per_env;
    const collision::MeshSurfaceView particle_surfaces{
        reinterpret_cast<const float*>(data.particle_pos + size_t{env} * p.particles_per_env),
        model.particle_surface_triangles,
        data.particle_surface_nodes != nullptr
            ? data.particle_surface_nodes + size_t{env} * p.particle_surface_nodes_per_env : nullptr,
        {p.particles_per_env, p.particle_surface_triangles, p.particle_surface_nodes_per_env}};
    for (uint32_t mesh = 0u; mesh < p.particle_surfaces_per_env; ++mesh) {
        const uint64_t bit = uint64_t{p.bodies_per_env} + mesh;
        if constexpr (emit) {
            if ((contact_hit_mask[mask_base + bit / 32u] & (1u << (bit % 32u))) == 0u)
                continue;
        }
        const auto info = model.particle_surface_info[mesh];
        const float thickness = model.particle_surface_thickness[mesh];
        const float query_distance = p.body_band + thickness + reach + p.dt *
            data.particle_surface_max_speed[env * p.particle_surfaces_per_env + mesh];
        const auto surface = collision::QueryMeshSurface(particle_surfaces, info, point, query_distance);
        if (!surface.valid) {
            status |= kEnvStatusContactGeometryUnavailable;
            continue;
        }
        if (surface.distance > query_distance || surface.triangle == ~0u) continue;
        const size_t at = size_t{info.triangle_offset + surface.triangle} * 3u;
        uint32_t indices[nk::kTriangleEndpointTerms];
        math::Vec3 vertices[nk::kTriangleEndpointTerms];
        for (uint32_t i = 0u; i < nk::kTriangleEndpointTerms; ++i) {
            indices[i] = env * p.particles_per_env + info.vertex_offset + model.particle_surface_triangles[at + i];
            vertices[i] = data.particle_pos[indices[i]];
        }
        nk::PointEndpointTerm terms[nk::kTriangleEndpointTerms];
        if (!nk::BuildTrianglePointEndpoint(indices, vertices, surface.barycentric, point, terms)) {
            status |= kEnvStatusContactGeometryUnavailable;
            continue;
        }
        const uint32_t term_count = nk::CanonicalizePointEndpointTerms(terms, nk::kTriangleEndpointTerms);
        uint32_t endpoint = 0u;
        if constexpr (emit) {
            const uint64_t contact = data.grid_contact_offset[sample] -
                data.grid_contact_offset[env * mpm_per_env] + count;
            if (contact < p.contact_capacity) {
                endpoint = env * p.point_endpoints_per_env + p.particles_per_env + static_cast<uint32_t>(contact);
                const uint32_t first = env * p.point_endpoint_terms_per_env +
                    p.particles_per_env * nk::kMpmStencilNodes + static_cast<uint32_t>(contact) * nk::kTriangleEndpointTerms;
                data.point_endpoint_ranges[endpoint] = {first, term_count};
                for (uint32_t i = 0u; i < term_count; ++i)
                    data.point_endpoint_terms[first + i] = terms[i];
            }
        }
        Visit<emit>(p, data, sample, mpm_per_env, count, nk::kUContactSidePointEndpoint, endpoint,
            {constraint::CollidableType::ParticleSurface, constraint::ReactionProviderKind::PointEndpoint,
             env * p.particle_surfaces_per_env + mesh}, point, surface.normal, thickness - surface.distance,
            fmaxf(p.body_mu, model.particle_surface_friction[mesh]), info.triangle_offset + surface.triangle, mesh);
        if constexpr (!emit)
            contact_hit_mask[mask_base + bit / 32u] |= 1u << (bit % 32u);
    }
}

__device__ inline size_t ContactMaskWords(const MpmParams& p) {
    return (uint64_t{p.bodies_per_env} + p.particle_surfaces_per_env + 31u) / 32u;
}

// Counting runs per sample, then per (sample, body), then per sample again; the hit mask carries
// body hits between passes. A negative reach marks samples that make no queries.
__global__ void PrepareSamples(MpmParams p, DataView data, uint32_t mpm_per_env,
                               uint32_t* contact_hit_mask, float* sample_reach,
                               uint32_t* body_hit_count) {
    const uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample == 0u) *body_hit_count = 0u;
    if (sample >= p.env_count * mpm_per_env) return;
    const uint32_t env = sample / mpm_per_env;
    const uint32_t particle = env * p.particles_per_env + sample % mpm_per_env;
    const size_t mask_words = ContactMaskWords(p);
    data.grid_contact_count[sample] = 0u;
    for (size_t word = 0u; word < mask_words; ++word)
        contact_hit_mask[size_t{sample} * mask_words + word] = 0u;
    sample_reach[sample] = -1.0f;
    if (!(data.particle_inv_mass[particle] > 0.0f)) return;
    const math::Vec3 point = data.particle_pos[particle];
    math::Vec3 velocity{};
    const bool valid = VisitStencil(p, env, point, [&](uint32_t node, float weight) {
        const auto value = data.grid_velocity[node];
        velocity.x = __fadd_rn(velocity.x, weight * value.x);
        velocity.y = __fadd_rn(velocity.y, weight * value.y);
        velocity.z = __fadd_rn(velocity.z, weight * value.z);
    });
    if (!valid) {
        atomicOr(&data.env_status[env], kEnvStatusMpmGridEscape | kEnvStatusInvalidEndpoint);
        return;
    }
    sample_reach[sample] = p.dx + p.dt * sqrtf(velocity.LengthSq());
}

// Warps share one body, so its shape, owner and pose loads are uniform across the warp.
__global__ void QueryBodies(MpmParams p, ModelView model, DataView data,
                            nkops::SurfaceQueryView surfaces, uint32_t mpm_per_env,
                            uint32_t* contact_hit_mask, const float* sample_reach,
                            BodyHitCache cache) {
    const uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = sample < p.env_count * mpm_per_env && sample_reach[sample] >= 0.0f;
    const uint32_t env = sample / mpm_per_env;
    const uint32_t lane = threadIdx.x % warpSize;
    const math::Vec3 point = active
        ? data.particle_pos[env * p.particles_per_env + sample % mpm_per_env] : math::Vec3{};
    const float reach = active ? sample_reach[sample] : 0.0f;
    const size_t mask_base = size_t{sample} * ContactMaskWords(p);
    uint32_t status = 0u;
    for (uint32_t body = blockIdx.y; body < p.bodies_per_env; body += gridDim.y) {
        BodyHit hit;
        if (active) hit = QueryBody(p, model, data, surfaces, env, body, point, reach, status);
        if (hit.hit) atomicOr(&contact_hit_mask[mask_base + body / 32u], 1u << (body % 32u));
        const uint32_t hits = __ballot_sync(~0u, hit.hit);
        if (hits == 0u) continue;
        const uint32_t leader = __ffs(hits) - 1u;
        uint32_t first = 0u;
        if (lane == leader) {
            first = atomicAdd(cache.count, static_cast<uint32_t>(__popc(hits)));
            if (uint64_t{first} + __popc(hits) > cache.capacity) first = ~0u;
            cache.first[size_t{sample / warpSize} * p.bodies_per_env + body] = first;
        }
        first = __shfl_sync(~0u, first, leader);
        if (hit.hit && first != ~0u)
            cache.records[first + __popc(hits & ((1u << lane) - 1u))] = hit;
    }
    if (status != 0u) atomicOr(&data.env_status[env], status);
}

__global__ void CountContacts(MpmParams p, ModelView model, DataView data, uint32_t mpm_per_env,
                              uint32_t* contact_hit_mask, const float* sample_reach) {
    const uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= p.env_count * mpm_per_env) return;
    const float reach = sample_reach[sample];
    if (reach < 0.0f) return;
    const uint32_t env = sample / mpm_per_env;
    const math::Vec3 point = data.particle_pos[env * p.particles_per_env + sample % mpm_per_env];
    const size_t mask_words = ContactMaskWords(p);
    const size_t mask_base = size_t{sample} * mask_words;
    uint64_t count = 0u;
    VisitBoundaries<false>(p, data, sample, mpm_per_env, count, point, reach);
    for (size_t word = 0u; word < mask_words; ++word)
        count += static_cast<uint32_t>(__popc(contact_hit_mask[mask_base + word]));
    uint32_t status = 0u;
    VisitParticleSurfaces<false>(p, model, data, sample, mpm_per_env, count, point, reach,
                                 contact_hit_mask, mask_base, status);
    data.grid_contact_count[sample] = count;
    if (status != 0u) atomicOr(&data.env_status[env], status);
}

// Body hits a sample's mask holds below limit, which precede that body's contact in order.
__device__ inline uint32_t CountBodyHits(const uint32_t* contact_hit_mask, size_t mask_base,
                                         uint32_t limit) {
    uint32_t hits = 0u;
    for (uint32_t word = 0u; word < limit / 32u; ++word)
        hits += static_cast<uint32_t>(__popc(contact_hit_mask[mask_base + word]));
    if (limit % 32u != 0u)
        hits += static_cast<uint32_t>(__popc(
            contact_hit_mask[mask_base + limit / 32u] & ((1u << (limit % 32u)) - 1u)));
    return hits;
}

// Emission revisits only the hits counting found, in the same boundary, body, surface order;
// EmitBodyContacts fills the body slots.
__global__ void EmitContacts(MpmParams p, ModelView model, DataView data, uint32_t mpm_per_env,
                             uint32_t* contact_hit_mask, const float* sample_reach) {
    const uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= p.env_count * mpm_per_env) return;
    if (data.grid_contact_count[sample] == 0u) return;
    const uint32_t env = sample / mpm_per_env;
    const math::Vec3 point = data.particle_pos[env * p.particles_per_env + sample % mpm_per_env];
    const float reach = sample_reach[sample];
    const size_t mask_base = size_t{sample} * ContactMaskWords(p);
    uint64_t count = 0u;
    VisitBoundaries<true>(p, data, sample, mpm_per_env, count, point, reach);
    count += CountBodyHits(contact_hit_mask, mask_base, p.bodies_per_env);
    uint32_t status = 0u;
    VisitParticleSurfaces<true>(p, model, data, sample, mpm_per_env, count, point, reach,
                                contact_hit_mask, mask_base, status);
    if (count > 0u) {
        const uint32_t endpoint = env * p.point_endpoints_per_env + sample % mpm_per_env;
        const uint32_t first = env * p.point_endpoint_terms_per_env + (sample % mpm_per_env) * nk::kMpmStencilNodes;
        uint32_t terms = 0u;
        VisitStencil(p, env, point, [&](uint32_t node, float weight) {
            data.point_endpoint_terms[first + terms++] = nk::WeightedPointEndpointTerm(nk::kNkSideGrid, node, weight);
        });
        data.point_endpoint_ranges[endpoint] = {first, terms};
    }
    if (status != 0u) atomicOr(&data.env_status[env], status);
}

// Each body hit is written at its ordinal after the sample's boundaries, from the record QueryBodies
// stored or, when the records were full, from the same query run again.
__global__ void EmitBodyContacts(MpmParams p, ModelView model, DataView data,
                                 nkops::SurfaceQueryView surfaces, uint32_t mpm_per_env,
                                 const uint32_t* contact_hit_mask, const float* sample_reach,
                                 BodyHitCache cache) {
    const uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = sample < p.env_count * mpm_per_env;
    const uint32_t env = sample / mpm_per_env;
    const uint32_t lane = threadIdx.x % warpSize;
    const size_t mask_base = size_t{sample} * ContactMaskWords(p);
    uint32_t status = 0u;
    for (uint32_t body = blockIdx.y; body < p.bodies_per_env; body += gridDim.y) {
        const bool hit = valid &&
            (contact_hit_mask[mask_base + body / 32u] & (1u << (body % 32u))) != 0u;
        const uint32_t hits = __ballot_sync(~0u, hit);
        if (!hit) continue;
        const math::Vec3 point = data.particle_pos[env * p.particles_per_env + sample % mpm_per_env];
        const float reach = sample_reach[sample];
        uint64_t count = 0u;
        VisitBoundaries<false>(p, data, sample, mpm_per_env, count, point, reach);
        count += CountBodyHits(contact_hit_mask, mask_base, body);
        const uint32_t first = cache.first[size_t{sample / warpSize} * p.bodies_per_env + body];
        const BodyHit found = first != ~0u
            ? cache.records[first + __popc(hits & ((1u << lane) - 1u))]
            : QueryBody(p, model, data, surfaces, env, body, point, reach, status);
        if (found.hit)
            Visit<true>(p, data, sample, mpm_per_env, count, nk::kUContactSideBody, body,
                found.endpoint, point, found.normal, found.depth, p.body_mu, found.feature, body);
    }
    if (status != 0u) atomicOr(&data.env_status[env], status);
}

__global__ void ClearSlots(MpmParams p, DataView data) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.env_count * p.contact_capacity) {
        const uint32_t slot = (i / p.contact_capacity) * p.contact_slots_per_env +
                              p.contact_slot_base + i % p.contact_capacity;
        data.ucontact_count[slot] = 0u;
    }
    if (i < p.env_count * p.point_endpoints_per_env) data.point_endpoint_ranges[i] = {};
}

__global__ void CountDiagnostics(MpmParams p, DataView data, uint32_t mpm_per_env) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= p.env_count) return;
    const uint32_t first = env * mpm_per_env;
    const uint32_t last = first + mpm_per_env - 1u;
    const uint64_t count = data.grid_contact_offset[last] + data.grid_contact_count[last] -
                           data.grid_contact_offset[first];
    const uint64_t retained = count < p.contact_capacity ? count : p.contact_capacity;
    data.grid_contact_attempted[env] = count;
    data.grid_contact_retained[env] = static_cast<uint32_t>(retained);
    data.grid_contact_overflow[env] = count - retained;
    if (count > data.grid_contact_peak[env]) data.grid_contact_peak[env] = count;
    if (count > retained) atomicOr(&data.env_status[env], kEnvStatusGridContactOverflow);
}

struct ReactionContribution {
    math::Vec3 impulse;
    math::Vec3 moment;
};

// Each retained contact contributes once to its resolved body or boundary target.
__global__ void ClassifyReactions(MpmParams p, ModelView model, DataView data, uint32_t* targets,
                                  ReactionContribution* contributions) {
    const uint32_t total = p.env_count * p.contact_capacity;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const uint32_t env = i / p.contact_capacity;
        const uint32_t c = i % p.contact_capacity;
        if (c >= data.grid_contact_retained[env]) continue;
        const size_t address = static_cast<size_t>(env * p.contact_slots_per_env +
            p.contact_slot_base + c) * nk::kPairDrivenPtsPerSlot;
        const uint32_t side = data.ucontact_b[address];
        const uint32_t kind = data.ucontact_b_kind[address];
        uint32_t target = ~0u;
        if (kind == nk::kUContactSideBody) {
            const auto shape = nkops::LoadPrimShape(model.shape_table, side);
            const auto owner = nk::ResolveCollidableOwner(shape.body_id, env, side,
                p.bodies_per_env, p.base_link_count, p.artics_per_env, model.body_to_link,
                model.body_to_articulation, model.body_collidable_body);
            const uint32_t local = owner.body - env * p.bodies_per_env;
            if (local < p.bodies_per_env) target = local;
        } else if (kind == nk::kUContactSideBoundary && side < nk::kMpmBoundaryCount) {
            target = p.bodies_per_env + side;
        }
        targets[i] = target;
        if (target == ~0u) continue;
        const uint32_t local_slot = p.contact_slot_base + c;
        const uint32_t base = env * p.rows_per_env +
            p.full_row_slot_count * nk::kPairDrivenRowsPerSlot +
            (local_slot - p.full_row_slot_count) * nk::kPairDrivenParticleRowsPerSlot;
        const auto* rows = reinterpret_cast<const nk::NkRow*>(data.urows);
        math::Vec3 impulse{};
        for (uint32_t axis = 0u; axis < nk::kPairDrivenParticleRowsPerSlot; ++axis)
            impulse += rows[base + axis].b.jlin * data.lambda[base + axis];
        contributions[i] = {impulse, data.ucontact_point[address].Cross(impulse)};
    }
}

// Each lane owns disjoint heads; descending insertion preserves the original increasing contact order.
template <uint32_t block_size>
__global__ void IndexReactions(MpmParams p, DataView data, const uint32_t* contact_targets,
                               uint32_t* heads, uint32_t* next) {
    const uint32_t env = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t targets = p.bodies_per_env + nk::kMpmBoundaryCount;
    uint32_t* env_heads = heads + size_t{env} * targets * block_size;
    uint32_t* env_next = next + size_t{env} * p.contact_capacity;
    const uint32_t* env_targets = contact_targets + size_t{env} * p.contact_capacity;
    for (uint32_t target = 0u; target < targets; ++target)
        env_heads[size_t{target} * block_size + lane] = ~0u;
    const uint32_t retained = data.grid_contact_retained[env];
    if (lane >= retained) return;
    uint32_t c = lane + ((retained - 1u - lane) / block_size) * block_size;
    for (;;) {
        const uint32_t target = env_targets[c];
        if (target < targets) {
            uint32_t& head = env_heads[size_t{target} * block_size + lane];
            env_next[c] = head;
            head = c;
        }
        if (c < block_size) break;
        c -= block_size;
    }
}

// Readout gathers solver impulses with the original lane assignment and reduction order.
template <uint32_t block_size>
__global__ void ReadReactions(MpmParams p, ModelView model, DataView data,
                              const uint32_t* heads, const uint32_t* next,
                              const ReactionContribution* contributions) {
    const uint32_t targets = p.bodies_per_env + nk::kMpmBoundaryCount;
    const uint32_t env = blockIdx.x / targets;
    const uint32_t target = blockIdx.x % targets;
    if (env >= p.env_count) return;
    const bool body_target = target < p.bodies_per_env;
    const uint32_t global_body = env * p.bodies_per_env + target;
    const uint32_t* env_next = next + size_t{env} * p.contact_capacity;
    const ReactionContribution* env_contributions = contributions + size_t{env} * p.contact_capacity;
    double sum[6] = {};
    for (uint32_t c = heads[size_t{blockIdx.x} * block_size + threadIdx.x];
         c != ~0u; c = env_next[c]) {
        const auto contribution = env_contributions[c];
        const math::Vec3 impulse = contribution.impulse;
        const math::Vec3 moment = contribution.moment;
        sum[0] += impulse.x; sum[1] += impulse.y; sum[2] += impulse.z;
        sum[3] += moment.x; sum[4] += moment.y; sum[5] += moment.z;
    }
    __shared__ double partial[6][block_size];
    for (uint32_t axis = 0u; axis < 6u; ++axis) partial[axis][threadIdx.x] = sum[axis];
    __syncthreads();
    for (uint32_t stride = block_size / 2u; stride > 0u; stride /= 2u) {
        if (threadIdx.x < stride)
            for (uint32_t axis = 0u; axis < 6u; ++axis)
                partial[axis][threadIdx.x] += partial[axis][threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x != 0u) return;
    const math::Vec3 impulse{float(partial[0][0]), float(partial[1][0]), float(partial[2][0])};
    const math::Vec3 moment{float(partial[3][0]), float(partial[4][0]), float(partial[5][0])};
    if (body_target) {
        const auto shape = nkops::LoadPrimShape(model.shape_table, target);
        const auto owner = nk::ResolveCollidableOwner(shape.body_id, env, target,
            p.bodies_per_env, p.base_link_count, p.artics_per_env, model.body_to_link,
            model.body_to_articulation, model.body_collidable_body);
        const math::Vec3 origin = owner.kind == nk::kNkSideArtic ? data.link_pose[owner.link].position :
            nkops::BodyCenterOfMass(data.body_pose[global_body], data.body_inertial_frame[global_body]);
        data.mpm_body_reaction[global_body] = impulse;
        data.mpm_body_ang_reaction[global_body] = moment - origin.Cross(impulse);
    } else {
        const uint32_t index = env * nk::kMpmBoundaryCount + target - p.bodies_per_env;
        data.mpm_boundary_impulse[index] = impulse;
        data.mpm_boundary_moment[index] = moment;
    }
}

}  // namespace nuka::phi::mpm_contact
