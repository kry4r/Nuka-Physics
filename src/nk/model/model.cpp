// ---------------------------------------------------------------------------
// nk::Model implementation (plan §3.3).
// ---------------------------------------------------------------------------

#include "nk/model/model.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace nuka::nk {

namespace {

constexpr uint64_t kAlign = 256;  // CUDA buffer-type alignment (BufferTypeAlignment).

uint64_t AlignUp(uint64_t v) { return (v + (kAlign - 1)) & ~(kAlign - 1); }

}  // namespace

uint32_t ModelCapacities::PerEnvCount(FieldPer per) const {
    switch (per) {
        case FieldPer::Env:            return 1u;
        case FieldPer::Dof:            return dofs_per_env;
        case FieldPer::Link:           return links_per_env;
        case FieldPer::Body:           return bodies_per_env;
        case FieldPer::ContactSlot:    return max_contacts_per_env;
        case FieldPer::RowSlot:        return max_rows_per_env;
        case FieldPer::SlotDof:        return max_contacts_per_env * dofs_per_env;
        case FieldPer::RowDof:         return max_rows_per_env * dofs_per_env;
        case FieldPer::Particle:       return particles_per_env;
        case FieldPer::DistCon:        return dist_cons_per_env;
        case FieldPer::BendCon:        return bend_cons_per_env;
        case FieldPer::VolCon:         return vol_cons_per_env;
        case FieldPer::ShapeMatchSlot:   return shape_match_slots_per_env;
        case FieldPer::ShapeMatchMember: return shape_match_members_per_env;
        case FieldPer::EnvDof2:        return dofs_per_env * dofs_per_env;
        // Multi-articulation co-residence: per-artic count == articulations_per_env;
        // the per-artic M-tile == articulations_per_env * max_dof^2. At
        // articulations_per_env == 1 these equal PerEnvCount(Env) / PerEnvCount(EnvDof2)
        // EXACTLY (same element count + ordering => the K==1 byte-identity invariant).
        case FieldPer::Articulation:     return articulations_per_env;
        case FieldPer::ArticulationDof2: return articulations_per_env * dofs_per_env *
                                                dofs_per_env;
        // Per-articulation flat-DOF tile (S3 qdot_flat). At articulations_per_env
        // == 1 this equals dofs_per_env == PerEnvCount(Dof) EXACTLY (same element
        // count + ordering => the K==1 byte-identity invariant for qdot_flat).
        case FieldPer::ArticulationDof:  return articulations_per_env * dofs_per_env;
        case FieldPer::Scalar:         return 0u;  // resolved by ElementCount
    }
    return 0u;
}

uint64_t ModelCapacities::ElementCount(FieldId id) const {
    const FieldLayout& lay = LayoutOf(id);
    if (lay.per == FieldPer::Scalar) {
        // The one symbolic per:scalar field in M3a is mat_buckets (num_buckets*8),
        // a GLOBAL table (not env-replicated). Any other scalar field defaults to
        // a single env-major row. Keyed by FieldId so the resolution is explicit.
        if (id == FieldId::MatBuckets) {
            return static_cast<uint64_t>(num_material_buckets) * 8ull;
        }
        if (id == FieldId::FootShape) {
            // GLOBAL foot table (shared by every env, base-relative indices):
            // 5 packed scalars per slot, max_contacts_per_env slots.
            return static_cast<uint64_t>(max_contacts_per_env) * 5ull;
        }
        if (id == FieldId::UnionSlots) {
            // GLOBAL union contact-pair template (shared by every env): 16
            // packed scalars per contact slot.
            return static_cast<uint64_t>(max_contacts_per_env) * 16ull;
        }
        if (id == FieldId::HullVerts) {
            // GLOBAL convex-hull vertex pool, xyz packed.
            return static_cast<uint64_t>(max_hull_verts) * 3ull;
        }
        // M5 pair-driven / SDF GLOBAL tables (per-env template, base-relative).
        // R1: the shape_table record GREW 8 -> 10 f32/row (appended body_id +
        // group); the unpack of lanes 0..7 stays byte-identical.
        if (id == FieldId::ShapeTable) {
            return static_cast<uint64_t>(max_bodies_total) * 10ull;
        }
        if (id == FieldId::ExcludedPairs) {
            return static_cast<uint64_t>(max_excluded_pairs);
        }
        if (id == FieldId::SampPoints) {
            return static_cast<uint64_t>(max_samp_points) * 3ull;
        }
        if (id == FieldId::SampRanges) {
            return static_cast<uint64_t>(max_bodies_total) * 2ull;
        }
        if (id == FieldId::SdfHeaders) {
            return static_cast<uint64_t>(max_sdf_grids) * 8ull;
        }
        if (id == FieldId::SdfCellCount) {
            return static_cast<uint64_t>(max_sdf_grids);
        }
        if (id == FieldId::SdfCellKeys || id == FieldId::SdfCellValues ||
            id == FieldId::SdfCellGradients) {
            return static_cast<uint64_t>(max_sdf_cells);
        }
        // M6 particle uniform-grid cell ranges: per-env cell count x env_count
        // (the keys are env-offset, so every env owns a private cell span).
        if (id == FieldId::GridCellStart || id == FieldId::GridCellEnd) {
            return static_cast<uint64_t>(max_grid_cells) *
                   static_cast<uint64_t>(env_count);
        }
        // H1 (general contact pipeline Phase 0): the GLOBAL heightfield grid pool
        // (shared by every env; the heightfield is a static collidable). 0 cells
        // for every current scene (no cook fills it yet) -> a zero-byte segment.
        if (id == FieldId::Heights) {
            return static_cast<uint64_t>(max_heightfield_cells);
        }
        // LBVH per-env Karras tree: (2N-1) nodes/env (N = bodies_per_env), 9 f32
        // lanes/node, env_count envs. The kernel strides by env*(2N-1) so the
        // array MUST be env_count*(2N-1) nodes (the prior per:body N-node sizing
        // undersized it by N-1 nodes/env). N<2 -> no tree -> 0 nodes (the build
        // early-exits for N<2). 9 f32/node packs the LbvhNode {3 i32 + 6 f32}.
        if (id == FieldId::LbvhNodes) {
            const uint32_t n = max_bodies_total;
            const uint64_t nodes_per_env = (n >= 2u) ? (2ull * n - 1ull) : 0ull;
            return nodes_per_env * 9ull * static_cast<uint64_t>(env_count);
        }
        return static_cast<uint64_t>(env_count);
    }
    const uint64_t per_env = PerEnvCount(lay.per);
    return per_env * static_cast<uint64_t>(env_count);
}

std::vector<Model::Segment> Model::ComputeModelSegments(uint64_t* total_bytes) const {
    std::vector<Segment> segs;
    uint64_t offset = 0;
    for (int i = 0; i < kFieldCount; ++i) {
        const FieldId id = static_cast<FieldId>(i);
        const FieldLayout& lay = LayoutOf(id);
        if (lay.owner != FieldOwner::Model) {
            continue;
        }
        const uint64_t count = capacities.ElementCount(id);
        const uint64_t bytes = count * static_cast<uint64_t>(lay.elem_size);
        const uint64_t aligned_off = AlignUp(offset);
        segs.push_back(Segment{id, aligned_off, bytes});
        offset = aligned_off + bytes;
    }
    if (total_bytes != nullptr) {
        *total_bytes = AlignUp(offset);
    }
    return segs;
}

namespace {

// memcpy one element repeatedly: dst[i] = src for i in [0, n). Used to stamp a
// per-link template value across replicas (env-major tiling, the EXACT layout
// ReplicateArticulationHostState's TileConcat produces).
template <typename T>
void StampPerLink(uint8_t* dst, const std::vector<T>& tpl, uint32_t links,
                  uint32_t envs, size_t elem_size) {
    for (uint32_t e = 0; e < envs; ++e) {
        for (uint32_t l = 0; l < links && l < tpl.size(); ++l) {
            std::memcpy(dst + (static_cast<size_t>(e) * links + l) * elem_size,
                        &tpl[l], elem_size);
        }
    }
}

}  // namespace

void Model::StageModelField(FieldId id, const Segment& seg,
                            std::vector<uint8_t>& host) const {
    // M3b: the REAL transcription of the cook-constant tables into the packed
    // section. Per-link tables tile env-major (replica e at [e*L, (e+1)*L)) —
    // value-identical to ReplicateArticulationHostState's TileConcat; the two
    // index tables (link_to_articulation, articulation_link_offset) get the
    // per-replica offset exactly as TileConcatOffset does. Sections this model
    // does not populate (solve-schedule M4, XPBD templates M6) stay zero.
    uint8_t* dst = host.data() + seg.offset;
    const ModelArticulation& a = articulation;
    const uint32_t L = capacities.links_per_env;
    const uint32_t E = capacities.env_count;
    if (seg.bytes == 0) {
        return;
    }
    switch (id) {
        case FieldId::LinkInertia: {
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t l = 0; l < L; ++l) {
                    if ((static_cast<size_t>(l) + 1) * 36 <= a.link_inertia_spatial.size()) {
                        std::memcpy(dst + (static_cast<size_t>(e) * L + l) * 144,
                                    a.link_inertia_spatial.data() + static_cast<size_t>(l) * 36,
                                    144);
                    }
                }
            }
            break;
        }
        case FieldId::LinkLocalPose:
            StampPerLink(dst, a.link_local_pose, L, E, sizeof(math::Transform));
            break;
        case FieldId::LinkInertialFrame:
            StampPerLink(dst, a.link_inertial_frame, L, E, sizeof(math::Transform));
            break;
        case FieldId::JointAxis:
            StampPerLink(dst, a.joint_axis, L, E, sizeof(math::Vec3));
            break;
        case FieldId::ParentOffset:
            StampPerLink(dst, a.parent_offset, L, E, sizeof(math::Vec3));
            break;
        case FieldId::JointType:
            StampPerLink(dst, a.joint_type, L, E, sizeof(uint8_t));
            break;
        case FieldId::ParentLink:
            // articulation-LOCAL parent indices (the ~0u root sentinel included)
            // tile unchanged — the kernels compute global = offset + parent.
            StampPerLink(dst, a.parent_link, L, E, sizeof(uint32_t));
            break;
        case FieldId::LinkBody:
            StampPerLink(dst, a.link_body, L, E, sizeof(uint32_t));
            break;
        case FieldId::LinkToArticulation: {
            // Multi-articulation co-residence: link l of replica e belongs to GLOBAL
            // articulation (e * K + local_artic_of_link[l]), where K ==
            // articulations_per_env. The template-local link->artic map is
            // a.link_to_articulation (built by BuildArticulationHostState over ALL
            // co-resident topologies). The +e*K term is the per-replica articulation
            // offset (the TileConcatOffset stride). At K==1 local_artic is always 0,
            // so this reduces EXACTLY to p[e*L+l] = e (the prior 1-artic byte layout).
            auto* p = reinterpret_cast<uint32_t*>(dst);
            const uint32_t K = capacities.articulations_per_env;
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t l = 0; l < L; ++l) {
                    const uint32_t local_artic =
                        l < a.link_to_articulation.size() ? a.link_to_articulation[l] : 0u;
                    p[static_cast<size_t>(e) * L + l] = e * K + local_artic;
                }
            }
            break;
        }
        case FieldId::JointDamping:
            StampPerLink(dst, a.joint_damping, L, E, sizeof(float));
            break;
        case FieldId::JointArmature:
            StampPerLink(dst, a.joint_armature, L, E, sizeof(float));
            break;
        case FieldId::ArticulationLinkCount: {
            // K entries per replica (one per co-resident articulation). The per-dog
            // link counts come from a.articulation_link_count; the per-replica term
            // is none (a count is replica-invariant). At K==1 a.articulation_link_count
            // == { L }, so this reduces to p[e] = L (the prior per:env byte layout).
            auto* p = reinterpret_cast<uint32_t*>(dst);
            const uint32_t K = capacities.articulations_per_env;
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t k = 0; k < K; ++k) {
                    p[static_cast<size_t>(e) * K + k] =
                        k < a.articulation_link_count.size() ? a.articulation_link_count[k]
                                                             : 0u;
                }
            }
            break;
        }
        case FieldId::ArticulationLinkOffset: {
            // K entries per replica. Each dog's flat link offset is its template-local
            // offset (a.articulation_link_offset[k]) PLUS the per-replica e*L shift
            // (the device link arrays are env-major at stride L == total links/env --
            // the documented TileConcatOffset pattern, stride = total link count). At
            // K==1 a.articulation_link_offset == { 0 }, so this reduces to p[e] = e*L
            // (the prior per:env byte layout).
            auto* p = reinterpret_cast<uint32_t*>(dst);
            const uint32_t K = capacities.articulations_per_env;
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t k = 0; k < K; ++k) {
                    const uint32_t local_off =
                        k < a.articulation_link_offset.size() ? a.articulation_link_offset[k]
                                                              : 0u;
                    p[static_cast<size_t>(e) * K + k] = e * L + local_off;
                }
            }
            break;
        }
        case FieldId::FootShape: {
            // GLOBAL table: 5 packed scalars per slot {link bits, off.xyz, radius}.
            auto* p = reinterpret_cast<float*>(dst);
            const uint32_t slots = capacities.max_contacts_per_env;
            for (uint32_t f = 0; f < feet.size() && f < slots; ++f) {
                uint32_t link_bits = feet[f].calf_local_link;
                std::memcpy(&p[static_cast<size_t>(f) * 5 + 0], &link_bits, 4);
                p[static_cast<size_t>(f) * 5 + 1] = feet[f].local_offset.x;
                p[static_cast<size_t>(f) * 5 + 2] = feet[f].local_offset.y;
                p[static_cast<size_t>(f) * 5 + 3] = feet[f].local_offset.z;
                p[static_cast<size_t>(f) * 5 + 4] = feet[f].radius;
            }
            break;
        }
        case FieldId::UnionSlots: {
            // GLOBAL union contact-pair template: 16 packed scalars per slot
            // {cls, link, body, condim (u32 bits), offset.xyz, radius,
            //  box_half.xyz, plane_height, mu, flags (u32 bits), 2 reserved}.
            auto* p = reinterpret_cast<float*>(dst);
            const uint32_t slots = capacities.max_contacts_per_env;
            auto put_u32 = [&](size_t at, uint32_t v) { std::memcpy(&p[at], &v, 4); };
            uint32_t row_base = 0;  // per-env row-slot prefix sum (lane 14).
            for (uint32_t s = 0; s < union_slots.size() && s < slots; ++s) {
                const UnionSlot& u = union_slots[s];
                const size_t b = static_cast<size_t>(s) * 16u;
                put_u32(b + 0, u.cls);
                put_u32(b + 1, u.link);
                put_u32(b + 2, u.body);
                put_u32(b + 3, u.condim);
                p[b + 4] = u.offset.x;  p[b + 5] = u.offset.y;  p[b + 6] = u.offset.z;
                p[b + 7] = u.radius;
                p[b + 8] = u.box_half.x; p[b + 9] = u.box_half.y; p[b + 10] = u.box_half.z;
                p[b + 11] = u.plane_height;
                p[b + 12] = u.mu;
                put_u32(b + 13, u.flags);
                put_u32(b + 14, row_base);
                row_base += u.MaxRows();
            }
            break;
        }
        case FieldId::HullVerts: {
            if (!hull_verts.empty()) {
                const size_t n = hull_verts.size() < static_cast<size_t>(
                                     capacities.max_hull_verts) * 3u
                                     ? hull_verts.size()
                                     : static_cast<size_t>(capacities.max_hull_verts) * 3u;
                std::memcpy(dst, hull_verts.data(), n * sizeof(float));
            }
            break;
        }
        case FieldId::ShapeTable: {
            // GLOBAL pair-driven shape table: R1 GREW it 8 -> 10 packed f32 per
            // body row {kind(u32 bits), p0..p3, contype(u32), conaffinity(u32),
            //  sdf_grid(u32), body_id(int32 bits; -1==static), group(u32 bits)}.
            // Lanes 0..7 are byte-identical to the prior layout; lanes 8/9 are the
            // appended R1 fields (INERT in Phase 0). Empty for the union family.
            auto* p = reinterpret_cast<float*>(dst);
            auto put_u32 = [&](size_t at, uint32_t v) { std::memcpy(&p[at], &v, 4); };
            auto put_i32 = [&](size_t at, int32_t v) { std::memcpy(&p[at], &v, 4); };
            const uint32_t rows = capacities.max_bodies_total;
            for (uint32_t s = 0; s < shape_table_rows.size() && s < rows; ++s) {
                const PairDrivenShape& sh = shape_table_rows[s];
                const size_t b = static_cast<size_t>(s) * 10u;
                put_u32(b + 0, sh.kind);
                p[b + 1] = sh.params[0]; p[b + 2] = sh.params[1];
                p[b + 3] = sh.params[2]; p[b + 4] = sh.params[3];
                put_u32(b + 5, sh.contype);
                put_u32(b + 6, sh.conaffinity);
                put_u32(b + 7, sh.sdf_grid);
                put_i32(b + 8, sh.body_id);
                put_u32(b + 9, sh.group);
            }
            break;
        }
        case FieldId::ExcludedPairs: {
            if (!excluded_pairs.empty()) {
                const size_t n = std::min(excluded_pairs.size(),
                                          static_cast<size_t>(capacities.max_excluded_pairs));
                std::memcpy(dst, excluded_pairs.data(), n * sizeof(uint64_t));
            }
            break;
        }
        case FieldId::SampPoints: {
            if (!samp_points.empty()) {
                const size_t n = std::min(samp_points.size(),
                                          static_cast<size_t>(capacities.max_samp_points) * 3u);
                std::memcpy(dst, samp_points.data(), n * sizeof(float));
            }
            break;
        }
        case FieldId::SampRanges: {
            if (!samp_ranges.empty()) {
                const size_t n = std::min(samp_ranges.size(),
                                          static_cast<size_t>(capacities.max_bodies_total) * 2u);
                std::memcpy(dst, samp_ranges.data(), n * sizeof(uint32_t));
            }
            break;
        }
        case FieldId::SdfHeaders: {
            // Per-grid record: origin.xyz, voxel_size, dims.xyz (u32 bits),
            // cell_offset (u32 bits) — 8 f32 each.
            auto* p = reinterpret_cast<float*>(dst);
            auto put_u32 = [&](size_t at, uint32_t v) { std::memcpy(&p[at], &v, 4); };
            const uint32_t grids = capacities.max_sdf_grids;
            for (uint32_t g = 0; g < sdf_grids.size() && g < grids; ++g) {
                const SdfGrid& s = sdf_grids[g];
                const size_t b = static_cast<size_t>(g) * 8u;
                p[b + 0] = s.origin.x; p[b + 1] = s.origin.y; p[b + 2] = s.origin.z;
                p[b + 3] = s.voxel_size;
                put_u32(b + 4, s.dims[0]); put_u32(b + 5, s.dims[1]);
                put_u32(b + 6, s.dims[2]); put_u32(b + 7, s.cell_offset);
            }
            break;
        }
        case FieldId::SdfCellCount: {
            auto* p = reinterpret_cast<uint32_t*>(dst);
            const uint32_t grids = capacities.max_sdf_grids;
            for (uint32_t g = 0; g < sdf_grids.size() && g < grids; ++g) {
                p[g] = sdf_grids[g].cell_count;
            }
            break;
        }
        case FieldId::SdfCellKeys: {
            if (!sdf_cell_keys.empty()) {
                const size_t n = std::min(sdf_cell_keys.size(),
                                          static_cast<size_t>(capacities.max_sdf_cells));
                std::memcpy(dst, sdf_cell_keys.data(), n * sizeof(uint64_t));
            }
            break;
        }
        case FieldId::SdfCellValues: {
            if (!sdf_cell_values.empty()) {
                const size_t n = std::min(sdf_cell_values.size(),
                                          static_cast<size_t>(capacities.max_sdf_cells));
                std::memcpy(dst, sdf_cell_values.data(), n * sizeof(float));
            }
            break;
        }
        case FieldId::SdfCellGradients: {
            if (!sdf_cell_gradients.empty()) {
                const size_t n = std::min(sdf_cell_gradients.size(),
                                          static_cast<size_t>(capacities.max_sdf_cells));
                std::memcpy(dst, sdf_cell_gradients.data(), n * sizeof(math::Vec3));
            }
            break;
        }
        case FieldId::DofToLink:
            StampPerLink(dst, dof_to_link, capacities.dofs_per_env, E, sizeof(uint32_t));
            break;
        case FieldId::DofToComponent:
            StampPerLink(dst, dof_to_component, capacities.dofs_per_env, E,
                         sizeof(uint32_t));
            break;
        case FieldId::RowOrder: {
            if (!schedule_row_order.empty()) {
                const size_t cap = static_cast<size_t>(capacities.max_rows_per_env) * E;
                const size_t n = schedule_row_order.size() < cap
                                     ? schedule_row_order.size() : cap;
                std::memcpy(dst, schedule_row_order.data(), n * sizeof(uint32_t));
            }
            break;
        }
        case FieldId::IslandColorSegments: {
            if (!schedule_color_segments.empty()) {
                const size_t cap = static_cast<size_t>(capacities.max_rows_per_env) * E * 2u;
                const size_t n = schedule_color_segments.size() < cap
                                     ? schedule_color_segments.size() : cap;
                std::memcpy(dst, schedule_color_segments.data(), n * sizeof(uint32_t));
            }
            break;
        }
        case FieldId::IslandRowOffsets: {
            if (!schedule_islands.empty()) {
                const size_t cap = static_cast<size_t>(capacities.max_rows_per_env) * E * 4u;
                const size_t n = schedule_islands.size() < cap
                                     ? schedule_islands.size() : cap;
                std::memcpy(dst, schedule_islands.data(), n * sizeof(uint32_t));
            }
            break;
        }
        // ---------------------------------------------------------------
        // M6 XPBD constraint templates (owner:model). The single-env template
        // is replicated env-major; env e's constraints get their particle
        // indices offset by e*particles_per_env (the device particle arrays
        // are env-major, so a constraint must point at its own env's particles).
        // This mirrors the legacy per-world soft-upload layout tiled E times.
        // ---------------------------------------------------------------
        case FieldId::DistParticleA:
        case FieldId::DistParticleB: {
            const auto& src = (id == FieldId::DistParticleA) ? particles.dist_a
                                                             : particles.dist_b;
            auto* p = reinterpret_cast<uint32_t*>(dst);
            const uint32_t dn = capacities.dist_cons_per_env;
            const uint32_t pn = capacities.particles_per_env;
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t c = 0; c < dn && c < src.size(); ++c) {
                    p[static_cast<size_t>(e) * dn + c] = src[c] + e * pn;
                }
            }
            break;
        }
        case FieldId::DistRestLength:
            StampPerLink(dst, particles.dist_rest, capacities.dist_cons_per_env, E,
                         sizeof(float));
            break;
        case FieldId::DistCompliance:
            StampPerLink(dst, particles.dist_alpha, capacities.dist_cons_per_env, E,
                         sizeof(float));
            break;
        case FieldId::BendParticles: {
            auto* p = reinterpret_cast<uint32_t*>(dst);
            const uint32_t bn = capacities.bend_cons_per_env;
            const uint32_t pn = capacities.particles_per_env;
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t c = 0; c < bn; ++c) {
                    for (uint32_t j = 0; j < 4u; ++j) {
                        const size_t si = static_cast<size_t>(c) * 4u + j;
                        if (si >= particles.bend_particles.size()) continue;
                        p[(static_cast<size_t>(e) * bn + c) * 4u + j] =
                            particles.bend_particles[si] + e * pn;
                    }
                }
            }
            break;
        }
        case FieldId::BendGradients: {
            auto* p = reinterpret_cast<math::Vec3*>(dst);
            const uint32_t bn = capacities.bend_cons_per_env;
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t c = 0; c < bn; ++c) {
                    for (uint32_t j = 0; j < 4u; ++j) {
                        const size_t si = static_cast<size_t>(c) * 4u + j;
                        if (si >= particles.bend_gradients.size()) continue;
                        p[(static_cast<size_t>(e) * bn + c) * 4u + j] =
                            particles.bend_gradients[si];
                    }
                }
            }
            break;
        }
        case FieldId::BendCompliance:
            StampPerLink(dst, particles.bend_alpha, capacities.bend_cons_per_env, E,
                         sizeof(float));
            break;
        case FieldId::VolParticles: {
            auto* p = reinterpret_cast<uint32_t*>(dst);
            const uint32_t vn = capacities.vol_cons_per_env;
            const uint32_t pn = capacities.particles_per_env;
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t c = 0; c < vn; ++c) {
                    for (uint32_t j = 0; j < 4u; ++j) {
                        const size_t si = static_cast<size_t>(c) * 4u + j;
                        if (si >= particles.vol_particles.size()) continue;
                        p[(static_cast<size_t>(e) * vn + c) * 4u + j] =
                            particles.vol_particles[si] + e * pn;
                    }
                }
            }
            break;
        }
        case FieldId::VolRestTimes6:
            StampPerLink(dst, particles.vol_rest6, capacities.vol_cons_per_env, E,
                         sizeof(float));
            break;
        case FieldId::VolCompliance:
            StampPerLink(dst, particles.vol_alpha, capacities.vol_cons_per_env, E,
                         sizeof(float));
            break;
        // ---------------------------------------------------------------
        // M9 T11 XPBD SHAPE-MATCH (id 9) cluster templates (owner:model).
        // The single-env CSR template is replicated env-major. The per-cluster
        // member OFFSET is shifted by e*members_per_env (the device member pool
        // is env-major) and the per-member PARTICLE index by e*particles_per_env
        // (the device particle arrays are env-major). The legacy single-world
        // CSR layout tiled E times -- byte-faithful to the legacy soft flatten.
        // ---------------------------------------------------------------
        case FieldId::SmClusterOffset: {
            auto* p = reinterpret_cast<uint32_t*>(dst);
            const uint32_t cn = capacities.shape_match_slots_per_env;
            const uint32_t mn = capacities.shape_match_members_per_env;
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t c = 0; c < cn && c < particles.sm_cluster_offset.size();
                     ++c) {
                    p[static_cast<size_t>(e) * cn + c] =
                        particles.sm_cluster_offset[c] + e * mn;
                }
            }
            break;
        }
        case FieldId::SmClusterSize:
            StampPerLink(dst, particles.sm_cluster_size,
                         capacities.shape_match_slots_per_env, E, sizeof(uint32_t));
            break;
        case FieldId::SmStiffness:
            StampPerLink(dst, particles.sm_stiffness,
                         capacities.shape_match_slots_per_env, E, sizeof(float));
            break;
        case FieldId::SmRestCentroid:
            StampPerLink(dst, particles.sm_rest_centroid,
                         capacities.shape_match_slots_per_env, E, sizeof(math::Vec3));
            break;
        case FieldId::SmParticles: {
            auto* p = reinterpret_cast<uint32_t*>(dst);
            const uint32_t mn = capacities.shape_match_members_per_env;
            const uint32_t pn = capacities.particles_per_env;
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t m = 0; m < mn && m < particles.sm_particles.size(); ++m) {
                    p[static_cast<size_t>(e) * mn + m] =
                        particles.sm_particles[m] + e * pn;
                }
            }
            break;
        }
        case FieldId::SmRestQ:
            StampPerLink(dst, particles.sm_rest_q,
                         capacities.shape_match_members_per_env, E, sizeof(math::Vec3));
            break;
        case FieldId::SmMass:
            StampPerLink(dst, particles.sm_mass,
                         capacities.shape_match_members_per_env, E, sizeof(float));
            break;
        case FieldId::LinkGeomKind:
            // WP5/WP6: per-link collision primitive kind. Empty for non-dog-dog
            // cooks -> StampPerLink leaves the section zero (inactive sentinel),
            // so K==1 stays byte-identical.
            StampPerLink(dst, a.link_geom_kind, L, E, sizeof(uint32_t));
            break;
        case FieldId::LinkGeomParams: {
            // 4 packed f32 per link, env-major tile (same pattern as StampPerLink
            // with a 4-wide element).
            auto* p = reinterpret_cast<float*>(dst);
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t l = 0; l < L; ++l) {
                    for (uint32_t c = 0; c < 4u; ++c) {
                        const size_t si = static_cast<size_t>(l) * 4u + c;
                        p[(static_cast<size_t>(e) * L + l) * 4u + c] =
                            si < a.link_geom_params.size() ? a.link_geom_params[si] : 0.0f;
                    }
                }
            }
            break;
        }
        case FieldId::LinkGeomLocal:
            StampPerLink(dst, a.link_geom_local, L, E, sizeof(math::Transform));
            break;
        case FieldId::BodyToLink:
        case FieldId::BodyToArticulation: {
            // R3 (general contact pipeline Phase 0): the shape->body inverse
            // tables. TEMPLATE-local per-body values tiled env-major (like
            // link_body). A body row not covered by the cooked table (free rigid /
            // static / an unpopulated cook) defaults to ~0u so the runtime resolves
            // it as a free-rigid/static side. INERT in Phase 0 (no op reads them).
            const std::vector<uint32_t>& tpl =
                (id == FieldId::BodyToLink) ? body_to_link : body_to_articulation;
            const uint32_t B = capacities.bodies_per_env;
            auto* p = reinterpret_cast<uint32_t*>(dst);
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t b = 0; b < B; ++b) {
                    p[static_cast<size_t>(e) * B + b] =
                        (b < tpl.size()) ? tpl[b] : ~uint32_t(0);
                }
            }
            break;
        }
        case FieldId::Heights: {
            // H1 (general contact pipeline Phase 0): the GLOBAL flat heightfield
            // grid pool. EMPTY for every current scene (the cook fill is H2, Phase
            // 2) -> seg.bytes == 0 already short-circuits above; the copy is a
            // forward-compatible no-op until H2 sizes max_heightfield_cells > 0.
            if (!heightfield_heights.empty()) {
                const size_t n = std::min(
                    heightfield_heights.size(),
                    static_cast<size_t>(capacities.max_heightfield_cells));
                std::memcpy(dst, heightfield_heights.data(), n * sizeof(float));
            }
            break;
        }
        default:
            // Unpopulated model sections: deterministic 0.
            break;
    }
}

namespace {

// Bind one segment's device pointer into the ModelView by FieldId. Centralized
// so the generated field list and the view members stay in lockstep (a missing
// case is a compile-time gap surfaced when a new model-owned field is added).
void BindModelPointer(phi::ModelView& v, FieldId id, void* p) {
    switch (id) {
        case FieldId::LinkInertia:           v.link_inertia = static_cast<Mat36*>(p); break;
        case FieldId::LinkLocalPose:         v.link_local_pose = static_cast<math::Transform*>(p); break;
        case FieldId::LinkInertialFrame:     v.link_inertial_frame = static_cast<math::Transform*>(p); break;
        case FieldId::JointAxis:             v.joint_axis = static_cast<math::Vec3*>(p); break;
        case FieldId::ParentOffset:          v.parent_offset = static_cast<math::Vec3*>(p); break;
        case FieldId::JointType:             v.joint_type = static_cast<uint8_t*>(p); break;
        case FieldId::ParentLink:            v.parent_link = static_cast<uint32_t*>(p); break;
        case FieldId::LinkBody:              v.link_body = static_cast<uint32_t*>(p); break;
        case FieldId::LinkToArticulation:    v.link_to_articulation = static_cast<uint32_t*>(p); break;
        case FieldId::LinkGeomKind:          v.link_geom_kind = static_cast<uint32_t*>(p); break;
        case FieldId::LinkGeomParams:        v.link_geom_params = static_cast<float*>(p); break;
        case FieldId::LinkGeomLocal:         v.link_geom_local = static_cast<math::Transform*>(p); break;
        // General contact pipeline Phase 0 (R3 inverse tables + H1 height grid).
        case FieldId::BodyToLink:            v.body_to_link = static_cast<uint32_t*>(p); break;
        case FieldId::BodyToArticulation:    v.body_to_articulation = static_cast<uint32_t*>(p); break;
        case FieldId::Heights:               v.heights = static_cast<float*>(p); break;
        case FieldId::JointDamping:          v.joint_damping = static_cast<float*>(p); break;
        case FieldId::JointArmature:         v.joint_armature = static_cast<float*>(p); break;
        case FieldId::ArticulationLinkCount: v.articulation_link_count = static_cast<uint32_t*>(p); break;
        case FieldId::ArticulationLinkOffset:v.articulation_link_offset = static_cast<uint32_t*>(p); break;
        case FieldId::FootShape:             v.foot_shape = static_cast<float*>(p); break;
        case FieldId::UnionSlots:            v.union_slots = static_cast<float*>(p); break;
        case FieldId::HullVerts:             v.hull_verts = static_cast<float*>(p); break;
        case FieldId::ShapeTable:            v.shape_table = static_cast<float*>(p); break;
        case FieldId::ExcludedPairs:         v.excluded_pairs = static_cast<uint64_t*>(p); break;
        case FieldId::SampPoints:            v.samp_points = static_cast<float*>(p); break;
        case FieldId::SampRanges:            v.samp_ranges = static_cast<uint32_t*>(p); break;
        case FieldId::SdfHeaders:            v.sdf_headers = static_cast<float*>(p); break;
        case FieldId::SdfCellCount:          v.sdf_cell_count = static_cast<uint32_t*>(p); break;
        case FieldId::SdfCellKeys:           v.sdf_cell_keys = static_cast<uint64_t*>(p); break;
        case FieldId::SdfCellValues:         v.sdf_cell_values = static_cast<float*>(p); break;
        case FieldId::SdfCellGradients:      v.sdf_cell_gradients = static_cast<math::Vec3*>(p); break;
        case FieldId::DofToLink:             v.dof_to_link = static_cast<uint32_t*>(p); break;
        case FieldId::DofToComponent:        v.dof_to_component = static_cast<uint32_t*>(p); break;
        case FieldId::IslandRowOffsets:      v.island_row_offsets = static_cast<uint32_t*>(p); break;
        case FieldId::IslandColorSegments:   v.island_color_segments = static_cast<uint32_t*>(p); break;
        case FieldId::RowOrder:              v.row_order = static_cast<uint32_t*>(p); break;
        case FieldId::DistParticleA:         v.dist_particle_a = static_cast<uint32_t*>(p); break;
        case FieldId::DistParticleB:         v.dist_particle_b = static_cast<uint32_t*>(p); break;
        case FieldId::DistRestLength:        v.dist_rest_length = static_cast<float*>(p); break;
        case FieldId::DistCompliance:        v.dist_compliance = static_cast<float*>(p); break;
        case FieldId::BendParticles:         v.bend_particles = static_cast<uint32_t*>(p); break;
        case FieldId::BendGradients:         v.bend_gradients = static_cast<math::Vec3*>(p); break;
        case FieldId::BendCompliance:        v.bend_compliance = static_cast<float*>(p); break;
        case FieldId::VolParticles:          v.vol_particles = static_cast<uint32_t*>(p); break;
        case FieldId::VolRestTimes6:         v.vol_rest_times6 = static_cast<float*>(p); break;
        case FieldId::VolCompliance:         v.vol_compliance = static_cast<float*>(p); break;
        case FieldId::SmClusterOffset:       v.sm_cluster_offset = static_cast<uint32_t*>(p); break;
        case FieldId::SmClusterSize:         v.sm_cluster_size = static_cast<uint32_t*>(p); break;
        case FieldId::SmStiffness:           v.sm_stiffness = static_cast<float*>(p); break;
        case FieldId::SmRestCentroid:        v.sm_rest_centroid = static_cast<math::Vec3*>(p); break;
        case FieldId::SmParticles:           v.sm_particles = static_cast<uint32_t*>(p); break;
        case FieldId::SmRestQ:               v.sm_rest_q = static_cast<math::Vec3*>(p); break;
        case FieldId::SmMass:                v.sm_mass = static_cast<float*>(p); break;
        default: break;  // a data-owned field id: not a ModelView member.
    }
}

}  // namespace

phi::Status Model::UploadTo(phi::BufferType* bt, phi::ModelView* out_view) {
    if (bt == nullptr || out_view == nullptr) {
        return phi::Status::Failed;
    }
    // Enforce the excluded-pairs contract AT THE STAGING BOUNDARY: the device
    // pair-query binary-searches this list, so it MUST be ascending + unique.
    // No cook fills it yet (M5 leaves it empty); normalizing here makes the
    // documented invariant structural instead of trusting future callers.
    // (Sorting an already-sorted list is the identity — zero behavior change.)
    if (!std::is_sorted(excluded_pairs.begin(), excluded_pairs.end())) {
        std::sort(excluded_pairs.begin(), excluded_pairs.end());
    }
    excluded_pairs.erase(
        std::unique(excluded_pairs.begin(), excluded_pairs.end()),
        excluded_pairs.end());
    uint64_t total = 0;
    const std::vector<Segment> segs = ComputeModelSegments(&total);

    phi::Buffer* buf = phi::BufferAlloc(bt, total == 0 ? kAlign : total);
    if (buf == nullptr) {
        return phi::Status::OutOfMemory;
    }
    // Zero the whole buffer first (deterministic; unimplemented sections stay 0).
    phi::BufferMemset(buf, 0, 0, total == 0 ? kAlign : total);

    // Stage the model-owned fields host-side, then ONE upload of the packed bytes.
    std::vector<uint8_t> host(total, 0u);
    for (const Segment& seg : segs) {
        StageModelField(seg.field, seg, host);
    }
    if (total > 0) {
        phi::BufferUpload(buf, host.data(), 0, total);
    }

    // Fill the view: each pointer = buffer base + section offset.
    auto* base = static_cast<uint8_t*>(phi::BufferBase(buf));
    *out_view = phi::ModelView{};
    for (const Segment& seg : segs) {
        BindModelPointer(*out_view, seg.field, base + seg.offset);
    }

    if (device_buffer_ != nullptr) {
        phi::BufferFree(device_buffer_);
    }
    device_buffer_ = buf;
    return phi::Status::Ok;
}

Model::~Model() {
    if (device_buffer_ != nullptr) {
        phi::BufferFree(device_buffer_);
        device_buffer_ = nullptr;
    }
}

namespace {

// Member-wise move of every host table + scalar (the union/M4 additions
// included). Centralized so the move ctor and move assignment cannot drift.
void MoveModelMembers(Model& dst, Model&& src) {
    dst.capacities = std::move(src.capacities);
    dst.articulation = std::move(src.articulation);
    dst.shapes = std::move(src.shapes);
    dst.material_buckets = std::move(src.material_buckets);
    dst.body_material_bucket = std::move(src.body_material_bucket);
    dst.feet = std::move(src.feet);
    dst.hold_drives = std::move(src.hold_drives);
    dst.ground_height = src.ground_height;
    dst.friction_coefficient = src.friction_coefficient;
    dst.baumgarte_max_velocity = src.baumgarte_max_velocity;
    dst.terrain = src.terrain;  // Go2-on-stairs Phase 1 procedural-terrain params.
    dst.filter_cross_env = src.filter_cross_env;
    dst.contact_family = src.contact_family;
    dst.drive_mode = src.drive_mode;
    dst.particles = std::move(src.particles);  // M6 particle cook product.
    dst.union_slots = std::move(src.union_slots);
    dst.hull_verts = std::move(src.hull_verts);
    for (int k = 0; k < 2; ++k) dst.union_solref[k] = src.union_solref[k];
    for (int k = 0; k < 5; ++k) dst.union_solimp[k] = src.union_solimp[k];
    dst.table_enabled_default = src.table_enabled_default;
    dst.body_init = std::move(src.body_init);
    dst.dof_to_link = std::move(src.dof_to_link);
    dst.dof_to_component = std::move(src.dof_to_component);
    dst.schedule_row_order = std::move(src.schedule_row_order);
    dst.schedule_color_segments = std::move(src.schedule_color_segments);
    dst.schedule_islands = std::move(src.schedule_islands);
    dst.schedule_island_count = src.schedule_island_count;
    dst.schedule_segment_count = src.schedule_segment_count;
    // M5 pair-driven / SDF tables (review fix: these were MISSING — a moved
    // Model silently dropped every pair-driven collision table, staging zeros).
    dst.excluded_pairs = std::move(src.excluded_pairs);
    dst.shape_table_rows = std::move(src.shape_table_rows);
    dst.samp_points = std::move(src.samp_points);
    dst.samp_ranges = std::move(src.samp_ranges);
    dst.sdf_grids = std::move(src.sdf_grids);
    dst.sdf_cell_keys = std::move(src.sdf_cell_keys);
    dst.sdf_cell_values = std::move(src.sdf_cell_values);
    dst.sdf_cell_gradients = std::move(src.sdf_cell_gradients);
    // General contact pipeline Phase 0/1B: the shape->body inverse tables (R3) +
    // the heightfield grid (H1). These were appended to the Model struct but were
    // MISSING from this manual move list -> they were silently DROPPED on every
    // move (return-by-value cook -> World), leaving body_to_link empty so the
    // PairDriven assembly resolved every collidable as free-rigid (no artic
    // reaction). Moved here so the PairDriven cook's registry survives into World.
    dst.body_to_link = std::move(src.body_to_link);
    dst.body_to_articulation = std::move(src.body_to_articulation);
    dst.heightfields = std::move(src.heightfields);
    dst.heightfield_heights = std::move(src.heightfield_heights);
}

}  // namespace

Model::Model(Model&& other) noexcept : device_buffer_(other.device_buffer_) {
    MoveModelMembers(*this, std::move(other));
    other.device_buffer_ = nullptr;
}

Model& Model::operator=(Model&& other) noexcept {
    if (this != &other) {
        if (device_buffer_ != nullptr) {
            phi::BufferFree(device_buffer_);
        }
        MoveModelMembers(*this, std::move(other));
        device_buffer_ = other.device_buffer_;
        other.device_buffer_ = nullptr;
    }
    return *this;
}

} // namespace nuka::nk
