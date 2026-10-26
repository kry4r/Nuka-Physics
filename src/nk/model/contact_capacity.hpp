#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "collision/shape_kind.hpp"
#include "nk/model/model.hpp"
#include "nk/solve/collidable_owner.hpp"
#include "nk/solve/nk_row.hpp"

namespace nuka::nk {

inline uint32_t RigidManifoldPointCapacity(uint32_t a, uint32_t b) {
    using namespace ::nuka::collision;
    if (a > b) std::swap(a, b);
    if ((a == kShapeSphere && (b == kShapeSphere || b == kShapeCapsule || b == kShapeBox || b == kShapePlane)) ||
        (a == kShapeCapsule && b == kShapeCapsule) || (a == kShapeSphere && b == kShapeCylinder)) return 1u;
    if (a == kShapeCapsule && b == kShapePlane) return 2u;
    return kPairDrivenPtsPerSlot;
}

struct RigidContactCapacity {
    std::array<uint64_t, kPairDrivenPtsPerSlot + 1u> slots{};

    uint64_t CandidateCount() const {
        uint64_t count = 0u;
        for (uint64_t width : slots) count += width;
        return count;
    }

    uint64_t RowsForSlots(uint64_t capacity) const {
        uint64_t points = 0u;
        for (uint32_t width = kPairDrivenPtsPerSlot; width > 0u; --width) {
            const uint64_t count = std::min(capacity, slots[width]);
            points += count * width;
            capacity -= count;
        }
        return (points + capacity * kPairDrivenPtsPerSlot) * (1u + kPairDrivenTangentRowsPerPt);
    }
};

inline RigidContactCapacity ComputeRigidContactCapacity(const Model& model) {
    RigidContactCapacity capacity;
    const uint32_t count = static_cast<uint32_t>(model.shape_table_rows.size());
    for (uint32_t a = 0u; a < count; ++a) {
        const auto& sa = model.shape_table_rows[a];
        for (uint32_t b = a + 1u; b < count; ++b) {
            const auto& sb = model.shape_table_rows[b];
            if ((sa.body_id < 0 && sb.body_id < 0) ||
                !::nuka::collision::ShapeFilterAdmits(sa.contype, sa.conaffinity, sa.group,
                    sb.contype, sb.conaffinity, sb.group) ||
                std::binary_search(model.excluded_pairs.begin(), model.excluded_pairs.end(),
                                   (uint64_t{a} << 32u) | b)) continue;
            ++capacity.slots[RigidManifoldPointCapacity(sa.kind, sb.kind)];
        }
    }
    return capacity;
}

// Whether some filtered primitive pair has no closed-form manifold and takes the convex narrowphase.
inline bool ConvexContactPairs(const Model& model) {
    using namespace ::nuka::collision;
    const auto& rows = model.shape_table_rows;
    const uint32_t count = static_cast<uint32_t>(rows.size());
    for (uint32_t a = 0u; a < count; ++a) {
        for (uint32_t b = a + 1u; b < count; ++b) {
            const auto& sa = rows[a];
            const auto& sb = rows[b];
            if ((sa.body_id < 0 && sb.body_id < 0) ||
                !ShapeFilterAdmits(sa.contype, sa.conaffinity, sa.group, sb.contype, sb.conaffinity, sb.group) ||
                std::binary_search(model.excluded_pairs.begin(), model.excluded_pairs.end(),
                                   (uint64_t{a} << 32u) | b)) continue;
            if (RouteMeshContact(sa.kind, sb.kind, sa.mesh_contact_mode, sb.mesh_contact_mode) ==
                    MeshContactRoute::Analytic &&
                TakesConvexManifold(sa.kind, sb.kind))
                return true;
        }
    }
    return false;
}

inline bool HasMeshContactPairs(const Model& model, collision::MeshContactRoute route) {
    const auto& rows = model.shape_table_rows;
    for (uint32_t a = 0u; a < rows.size(); ++a) {
        const auto& sa = rows[a];
        for (uint32_t b = a + 1u; b < rows.size(); ++b) {
            const auto& sb = rows[b];
            if ((sa.body_id < 0 && sb.body_id < 0) ||
                !collision::ShapeFilterAdmits(sa.contype, sa.conaffinity, sa.group,
                    sb.contype, sb.conaffinity, sb.group) ||
                std::binary_search(model.excluded_pairs.begin(), model.excluded_pairs.end(),
                                   (uint64_t{a} << 32u) | b)) continue;
            if (route == collision::MeshContactRoute::Ogc &&
                (a >= model.mesh_surface_info.size() || b >= model.mesh_surface_info.size() ||
                 model.mesh_surface_info[a].vertex_count == 0u ||
                 model.mesh_surface_info[b].vertex_count == 0u)) continue;
            if (collision::RouteMeshContact(sa.kind, sb.kind, sa.mesh_contact_mode,
                                            sb.mesh_contact_mode) == route) return true;
        }
    }
    return false;
}

// Mesh features of bodies that some filtered partner meets through mesh contact; one slot each.
inline uint64_t PairedMeshFeatureCount(const Model& model) {
    const auto& rows = model.shape_table_rows;
    const uint32_t bodies = std::min({model.capacities.bodies_per_env, static_cast<uint32_t>(rows.size()),
                                      static_cast<uint32_t>(model.mesh_surface_info.size())});
    uint64_t features = 0u;
    for (uint32_t a = 0u; a < bodies; ++a) {
        const auto& sa = rows[a];
        if ((sa.contype | sa.conaffinity) == 0u || model.mesh_surface_info[a].vertex_count == 0u) continue;
        bool paired = false;
        for (uint32_t b = 0u; b < bodies && !paired; ++b) {
            const auto& sb = rows[b];
            paired = b != a && !(sa.body_id < 0 && sb.body_id < 0) &&
                model.mesh_surface_info[b].vertex_count > 0u &&
                ::nuka::collision::RouteMeshContact(sa.kind, sb.kind, sa.mesh_contact_mode,
                    sb.mesh_contact_mode) == ::nuka::collision::MeshContactRoute::Ogc &&
                ::nuka::collision::ShapeFilterAdmits(sa.contype, sa.conaffinity, sa.group,
                    sb.contype, sb.conaffinity, sb.group) &&
                !std::binary_search(model.excluded_pairs.begin(), model.excluded_pairs.end(),
                                    (uint64_t{std::min(a, b)} << 32u) | std::max(a, b));
        }
        if (paired) features += model.mesh_surface_info[a].vertex_count + model.mesh_edge_info[a].edge_count;
    }
    return features;
}

// Whether some filtered shape pair has articulated owners on both sides.
inline bool ArticulatedContactPairs(const Model& model) {
    const auto& rows = model.shape_table_rows;
    const ModelCapacities& cap = model.capacities;
    const auto staged = [&](const std::vector<uint32_t>& table) {
        std::vector<uint32_t> values(cap.bodies_per_env, ~0u);
        std::copy_n(table.begin(), std::min(table.size(), values.size()), values.begin());
        return values;
    };
    const auto link = staged(model.body_to_link);
    const auto articulation = staged(model.body_to_articulation);
    const auto proxy = staged(model.body_collidable_body);
    const auto articulated = [&](uint32_t shape) {
        return ResolveCollidableOwner(rows[shape].body_id, 0u, shape, cap.bodies_per_env, cap.links_per_env,
            cap.articulations_per_env, link.data(), articulation.data(), proxy.data()).kind == kNkSideArtic;
    };
    const uint32_t count = std::min(cap.bodies_per_env, static_cast<uint32_t>(rows.size()));
    for (uint32_t a = 0u; a < count; ++a) {
        if (!articulated(a)) continue;
        for (uint32_t b = a + 1u; b < count; ++b)
            if (articulated(b) &&
                ::nuka::collision::ShapeFilterAdmits(rows[a].contype, rows[a].conaffinity, rows[a].group,
                    rows[b].contype, rows[b].conaffinity, rows[b].group) &&
                !std::binary_search(model.excluded_pairs.begin(), model.excluded_pairs.end(),
                                    (uint64_t{a} << 32u) | b)) return true;
    }
    return false;
}

// Filtered shape pairs whose sides resolve to two different movable owners; only their rows touch two
// solver blocks.
inline RigidContactCapacity DistinctOwnerContactCapacity(const Model& model) {
    const auto& rows = model.shape_table_rows;
    const ModelCapacities& cap = model.capacities;
    const auto staged = [&](const std::vector<uint32_t>& table) {
        std::vector<uint32_t> values(cap.bodies_per_env, ~0u);
        std::copy_n(table.begin(), std::min(table.size(), values.size()), values.begin());
        return values;
    };
    const auto link = staged(model.body_to_link);
    const auto articulation = staged(model.body_to_articulation);
    const auto proxy = staged(model.body_collidable_body);
    const uint32_t count = static_cast<uint32_t>(rows.size());
    std::vector<uint64_t> owners(count, ~uint64_t{0u});
    for (uint32_t shape = 0u; shape < count; ++shape) {
        const CollidableOwner owner = ResolveCollidableOwner(rows[shape].body_id, 0u, shape, cap.bodies_per_env,
            cap.links_per_env, cap.articulations_per_env, link.data(), articulation.data(), proxy.data());
        if (owner.kind == kNkSideArtic) owners[shape] = uint64_t{1u} << 32u | owner.articulation;
        // A rigid body without inverse mass keeps no solver block.
        if (owner.kind == kNkSideRigid && owner.body < model.body_init.size() &&
            model.body_init[owner.body].inv_mass > 0.0f) owners[shape] = owner.body;
    }
    RigidContactCapacity capacity;
    for (uint32_t a = 0u; a < count; ++a) {
        for (uint32_t b = a + 1u; b < count; ++b) {
            if (owners[a] == ~uint64_t{0u} || owners[b] == ~uint64_t{0u} || owners[a] == owners[b] ||
                !::nuka::collision::ShapeFilterAdmits(rows[a].contype, rows[a].conaffinity, rows[a].group,
                    rows[b].contype, rows[b].conaffinity, rows[b].group) ||
                std::binary_search(model.excluded_pairs.begin(), model.excluded_pairs.end(),
                                   (uint64_t{a} << 32u) | b)) continue;
            ++capacity.slots[RigidManifoldPointCapacity(rows[a].kind, rows[b].kind)];
        }
    }
    return capacity;
}

} // namespace nuka::nk
