#pragma once

#include "nk/contact/contact_identity.hpp"
#include "nk/model/generated/views.hpp"
#include "nk/solve/collidable_owner.hpp"
#include "phi/backend_cuda/ops/rigid_types.cuh"
#include "phi/backend_cuda/ops/surface_query.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi::mpm_contact {

// Count and emit traverse the same node, boundary and collidable order.
template <bool emit>
__device__ void Visit(const MpmParams& p, const DataView& data, uint32_t node,
                      uint64_t& count, uint32_t side_kind, uint32_t side_index,
                      constraint::CollidableRef side, math::Vec3 point,
                      math::Vec3 normal, float depth, float friction,
                      uint32_t feature, uint32_t topology) {
    const uint64_t ordinal = count++;
    if constexpr (!emit) return;
    const uint32_t env = node / p.nodes_per_env;
    const uint64_t local = data.grid_contact_offset[node] -
        data.grid_contact_offset[env * p.nodes_per_env] + ordinal;
    if (local >= p.contact_capacity) return;
    const uint32_t slot = env * p.contact_slots_per_env + p.contact_slot_base +
                          static_cast<uint32_t>(local);
    const size_t address = static_cast<size_t>(slot) * nk::kPairDrivenPtsPerSlot;
    data.ucontact_count[slot] = 1u;
    data.ucontact_point[address] = point;
    data.ucontact_normal[address] = normal;
    data.ucontact_depth[address] = depth;
    data.ucontact_a[address] = node;
    data.ucontact_b[address] = side_index;
    data.ucontact_a_kind[address] = nk::kUContactSideGrid;
    data.ucontact_b_kind[address] = side_kind;
    data.ucontact_gen[address] = 1u;
    data.ucontact_law[slot] = nk::kContactLawVelocity;
    data.ucontact_friction[slot] = friction;
    nk::CanonicalContactDescriptor descriptor;
    descriptor.a = {constraint::CollidableType::GridNode,
                    constraint::ReactionProviderKind::GridInvMass, node};
    descriptor.b = side;
    descriptor.normal = normal;
    descriptor.feature_a = 0u;
    descriptor.feature_b = feature;
    descriptor.topology_version = topology;
    const nk::ContactId id = nk::MakeContactId(descriptor);
    data.ucontact_id_pair[address] = id.pair;
    data.ucontact_id_feature[address] = id.feature;
}

template <bool emit>
__global__ void Generate(MpmParams p, ModelView model, DataView data,
                         nkops::SurfaceQueryView surfaces,
                         const uint32_t* active_nodes, const uint32_t* active_count) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= p.env_count * p.nodes_per_env || slot >= *active_count) return;
    const uint32_t node = active_nodes[slot];
    if (!(data.grid_inv_mass[node] > 0.0f)) return;
    const uint32_t env = node / p.nodes_per_env;
    const uint32_t local = node % p.nodes_per_env;
    const uint32_t x = local % p.grid_dims[0];
    const uint32_t y = (local / p.grid_dims[0]) % p.grid_dims[1];
    const uint32_t z = local / (p.grid_dims[0] * p.grid_dims[1]);
    const math::Vec3 point{p.grid_origin[0] + x * p.dx,
                           p.grid_origin[1] + y * p.dx,
                           p.grid_origin[2] + z * p.dx};
    const math::Vec3 floor_normal{p.plane_n[0], p.plane_n[1], p.plane_n[2]};
    const float floor_distance = point.Dot(floor_normal) - p.plane_d;
    uint64_t count = 0u;
    const auto boundary = [&](uint32_t id, math::Vec3 normal, float depth, float mu) {
        Visit<emit>(p, data, node, count, nk::kUContactSideBoundary, id,
            {constraint::CollidableType::StaticBoundary,
             constraint::ReactionProviderKind::StaticNull, env * nk::kMpmBoundaryCount + id},
            point, normal, depth, mu, id, 0u);
    };
    if (floor_distance <= 0.0f) boundary(0u, floor_normal, -floor_distance, p.plane_mu);
    if (x == 0u) boundary(1u, {1.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
    if (x + 1u == p.grid_dims[0]) boundary(2u, {-1.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
    if (y == 0u) boundary(3u, {0.0f, 1.0f, 0.0f}, 0.0f, 0.0f);
    if (y + 1u == p.grid_dims[1]) boundary(4u, {0.0f, -1.0f, 0.0f}, 0.0f, 0.0f);
    uint32_t status = 0u;
    if (p.dynamic_body_bc != 0u && p.bite_disable_dynamic_bc == 0u) {
        for (uint32_t body = 0u; body < p.bodies_per_env; ++body) {
            const auto shape = nkops::LoadPrimShape(model.shape_table, body);
            if ((shape.contype | shape.conaffinity) == 0u) continue;
            const auto pose = data.body_pose[env * p.bodies_per_env + body];
            const auto surface = nkops::QueryCollidableSurface(surfaces, body, shape,
                nkops::PrimInverseTransformPoint(pose, point), p.body_band);
            if (!surface.valid) {
                status |= kEnvStatusMpmOneWayBody | kEnvStatusContactGeometryUnavailable;
                continue;
            }
            if (!isfinite(surface.distance) || surface.distance >= p.body_band) continue;
            const math::Vec3 raw_normal = nkops::PrimRotate(pose.rotation, surface.normal);
            const float length = sqrtf(raw_normal.LengthSq());
            if (!isfinite(length) || length <= 1.0e-8f) continue;
            const auto owner = nk::ResolveCollidableOwner(shape.body_id, env, body,
                p.bodies_per_env, p.base_link_count, p.artics_per_env, model.body_to_link,
                model.body_to_articulation, model.body_collidable_body);
            if (owner.kind == ~0u) {
                status |= kEnvStatusInvalidEndpoint;
                continue;
            }
            constraint::CollidableRef endpoint;
            endpoint.handle = owner.body;
            endpoint.type = constraint::CollidableType::RigidBody;
            endpoint.react = constraint::ReactionProviderKind::RigidInvMass;
            if (owner.kind == nk::kNkSideArtic) {
                endpoint.handle = owner.link;
                endpoint.type = constraint::CollidableType::ArticulationLink;
                endpoint.react = constraint::ReactionProviderKind::ArticulationChainJ;
            } else if (owner.kind == nk::kNkSideStatic) {
                endpoint.type = constraint::CollidableType::StaticWorld;
                endpoint.react = constraint::ReactionProviderKind::StaticNull;
            }
            Visit<emit>(p, data, node, count, nk::kUContactSideBody, body, endpoint,
                point, raw_normal * (1.0f / length), p.body_band - surface.distance,
                p.body_mu, surface.feature, body);
        }
    }
    if constexpr (!emit) data.grid_contact_count[node] = count;
    if (status != 0u) atomicOr(&data.env_status[env], status);
}

__global__ void ClearSlots(MpmParams p, DataView data) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.env_count * p.contact_capacity) return;
    const uint32_t slot = (i / p.contact_capacity) * p.contact_slots_per_env +
                          p.contact_slot_base + i % p.contact_capacity;
    data.ucontact_count[slot] = 0u;
}

__global__ void CountDiagnostics(MpmParams p, DataView data) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= p.env_count) return;
    const uint32_t first = env * p.nodes_per_env;
    const uint32_t last = first + p.nodes_per_env - 1u;
    const uint64_t count = data.grid_contact_offset[last] + data.grid_contact_count[last] -
                           data.grid_contact_offset[first];
    const uint64_t retained = count < p.contact_capacity ? count : p.contact_capacity;
    data.grid_contact_attempted[env] = count;
    data.grid_contact_retained[env] = static_cast<uint32_t>(retained);
    data.grid_contact_overflow[env] = count - retained;
    if (count > data.grid_contact_peak[env]) data.grid_contact_peak[env] = count;
    if (count > retained) atomicOr(&data.env_status[env], kEnvStatusGridContactOverflow);
}

// Readout gathers the solver's impulses; it never changes endpoint velocities.
template <uint32_t block_size>
__global__ void ReadReactions(MpmParams p, ModelView model, DataView data) {
    const uint32_t targets = p.bodies_per_env + nk::kMpmBoundaryCount;
    const uint32_t env = blockIdx.x / targets;
    const uint32_t target = blockIdx.x % targets;
    if (env >= p.env_count) return;
    const bool body_target = target < p.bodies_per_env;
    const uint32_t global_body = env * p.bodies_per_env + target;
    const auto* rows = reinterpret_cast<const nk::NkRow*>(data.urows);
    double sum[6] = {};
    for (uint32_t c = threadIdx.x; c < data.grid_contact_retained[env]; c += block_size) {
        const uint32_t local_slot = p.contact_slot_base + c;
        const size_t address = static_cast<size_t>(env * p.contact_slots_per_env + local_slot) *
                                nk::kPairDrivenPtsPerSlot;
        const uint32_t side = data.ucontact_b[address];
        const uint32_t kind = data.ucontact_b_kind[address];
        if (body_target) {
            if (kind != nk::kUContactSideBody) continue;
            const auto shape = nkops::LoadPrimShape(model.shape_table, side);
            const auto owner = nk::ResolveCollidableOwner(shape.body_id, env, side,
                p.bodies_per_env, p.base_link_count, p.artics_per_env, model.body_to_link,
                model.body_to_articulation, model.body_collidable_body);
            if (owner.body != global_body) continue;
        } else if (kind != nk::kUContactSideBoundary || side != target - p.bodies_per_env) {
            continue;
        }
        const uint32_t base = env * p.rows_per_env +
            p.full_row_slot_count * nk::kPairDrivenRowsPerSlot +
            (local_slot - p.full_row_slot_count) * nk::kPairDrivenParticleRowsPerSlot;
        math::Vec3 impulse{};
        for (uint32_t axis = 0u; axis < nk::kPairDrivenParticleRowsPerSlot; ++axis)
            impulse += rows[base + axis].b.jlin * data.lambda[base + axis];
        const math::Vec3 moment = data.ucontact_point[address].Cross(impulse);
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
