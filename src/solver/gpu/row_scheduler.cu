// ---------------------------------------------------------------------------
// nuka::solver::gpu::row_scheduler implementation
// ---------------------------------------------------------------------------

#include "solver/gpu/row_scheduler.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace nuka::solver::gpu {

namespace {

// The DEDUPED set of valid (non-kInvalidBodyIndex) bodies a row touches. This is the
// exact set RowsConflict (the prior O(rows^2) reference) iterated over: an invalid body
// never participates in a conflict, and a row with body_a == body_b touches that body
// once. Two rows conflict iff their valid-body sets intersect -- identical semantics to
// the reference's nested-loop test.
struct RowValidBodies {
    uint32_t bodies[2];
    uint32_t count;
};

RowValidBodies ValidBodiesForRow(const constraint::RowBuffers& rows, uint32_t row) {
    const auto pair = rows.BodiesForRow(row);
    RowValidBodies out{{0u, 0u}, 0u};
    if (pair.body_a != constraint::kInvalidBodyIndex) {
        out.bodies[out.count++] = pair.body_a;
    }
    if (pair.body_b != constraint::kInvalidBodyIndex && pair.body_b != pair.body_a) {
        out.bodies[out.count++] = pair.body_b;
    }
    return out;
}

// ---------------------------------------------------------------------------
// G1(d) host-cost increment: a DENSE SLOT remap of the row body ids.
// ---------------------------------------------------------------------------
// The coloring/validation/component passes all key per-body state by the raw
// uint32 body id. The original implementation used unordered_map for that --
// MEASURED at 26.1 ms/step for Build+Validate at N=1024 x 10 rows/env (the G1e
// small-scene log), i.e. hash+allocation churn, rebuilt EVERY step. The remap
// below sorts the deduped valid body ids once (O(B log B)) and rewrites each
// row's bodies as dense [0, slot_count) slots, so every per-body lookup
// downstream is a plain array index. PURELY an internal indexing change: the
// row order, the conflict relation, and every emitted partition stay
// byte-identical (the equivalence suite pins Build against the quadratic
// greedy reference).
constexpr uint32_t kNoSlot = ~0u;

struct RowBodySlotIndex {
    std::vector<uint32_t> slots;  // [2 * row_count], kNoSlot-padded
    std::vector<uint8_t> counts;  // per row: 0..2 valid slots
    uint32_t slot_count = 0u;
};

RowBodySlotIndex BuildRowBodySlotIndex(const constraint::RowBuffers& rows) {
    const uint32_t row_count = rows.RowCount();
    RowBodySlotIndex out;
    out.slots.assign(static_cast<size_t>(2u) * row_count, kNoSlot);
    out.counts.assign(row_count, 0u);

    std::vector<uint32_t> bodies;
    bodies.reserve(static_cast<size_t>(2u) * row_count);
    for (uint32_t row = 0u; row < row_count; ++row) {
        const RowValidBodies vb = ValidBodiesForRow(rows, row);
        out.counts[row] = static_cast<uint8_t>(vb.count);
        for (uint32_t bi = 0u; bi < vb.count; ++bi) {
            out.slots[static_cast<size_t>(2u) * row + bi] = vb.bodies[bi];
            bodies.push_back(vb.bodies[bi]);
        }
    }
    std::sort(bodies.begin(), bodies.end());
    bodies.erase(std::unique(bodies.begin(), bodies.end()), bodies.end());
    out.slot_count = static_cast<uint32_t>(bodies.size());

    for (uint32_t row = 0u; row < row_count; ++row) {
        for (uint32_t bi = 0u; bi < out.counts[row]; ++bi) {
            uint32_t& slot = out.slots[static_cast<size_t>(2u) * row + bi];
            slot = static_cast<uint32_t>(
                std::lower_bound(bodies.begin(), bodies.end(), slot) -
                bodies.begin());
        }
    }
    return out;
}

} // namespace

RowColorPartitions BuildRowColorPartitions(const constraint::RowBuffers& rows) {
    RowColorPartitions partitions;
    const uint32_t row_count = rows.RowCount();
    if (row_count == 0u) {
        return partitions;
    }

    const RowBodySlotIndex slot_index = BuildRowBodySlotIndex(rows);

    std::vector<uint32_t> row_colors(row_count, 0u);
    uint32_t color_count = 0u;

    // slot -> the colors already assigned to prior rows touching that body (deduped,
    // kept sorted ascending). Per-row cost is O(K): the lowest available color is the
    // smallest non-negative integer not used by any conflicting prior row, which is
    // exactly the union of slot_colors over this row's slots. Output is byte-identical
    // to the plain greedy-lowest-available-by-row-index reference (and to the prior
    // unordered_map-keyed implementation -- only the per-body keying changed).
    std::vector<std::vector<uint32_t>> slot_colors(slot_index.slot_count);

    // Min-excludant scratch: stamp[color] == row+1 marks the color forbidden for the
    // CURRENT row (the per-row epoch makes clearing unnecessary).
    std::vector<uint32_t> stamp;

    for (uint32_t row = 0; row < row_count; ++row) {
        const uint32_t epoch = row + 1u;
        for (uint32_t bi = 0u; bi < slot_index.counts[row]; ++bi) {
            const uint32_t slot = slot_index.slots[static_cast<size_t>(2u) * row + bi];
            for (const uint32_t used_color : slot_colors[slot]) {
                stamp[used_color] = epoch;
            }
        }

        // The lowest color not stamped this row; color_count means every existing
        // color is forbidden -> open a new color.
        uint32_t color = 0u;
        while (color < color_count && stamp[color] == epoch) {
            ++color;
        }
        if (color == color_count) {
            ++color_count;
            stamp.push_back(0u);
        }
        row_colors[row] = color;

        // Record this color on each of the row's slots (deduped, sorted-insert).
        for (uint32_t bi = 0u; bi < slot_index.counts[row]; ++bi) {
            const uint32_t slot = slot_index.slots[static_cast<size_t>(2u) * row + bi];
            std::vector<uint32_t>& colors = slot_colors[slot];
            const auto pos = std::lower_bound(colors.begin(), colors.end(), color);
            if (pos == colors.end() || *pos != color) {
                colors.insert(pos, color);
            }
        }
    }

    // Counting sort by color (ascending row index within each color) -- exactly the
    // partition the former O(colors x rows) rescan emitted.
    partitions.color_ranges.resize(color_count);
    for (uint32_t row = 0; row < row_count; ++row) {
        ++partitions.color_ranges[row_colors[row]].row_count;
    }
    {
        uint32_t offset = 0u;
        for (uint32_t color = 0; color < color_count; ++color) {
            partitions.color_ranges[color].row_offset = offset;
            offset += partitions.color_ranges[color].row_count;
        }
    }
    partitions.row_indices.resize(row_count);
    std::vector<uint32_t> cursor(color_count, 0u);
    for (uint32_t row = 0; row < row_count; ++row) {
        const uint32_t color = row_colors[row];
        partitions.row_indices[partitions.color_ranges[color].row_offset +
                               cursor[color]++] = row;
    }

    return partitions;
}

namespace {

// Minimal deterministic union-find over row indices (path-halving find; union
// by smaller ROOT INDEX wins so component representatives -- and therefore the
// component numbering below -- depend only on the input, never on traversal
// order).
class RowUnionFind {
public:
    explicit RowUnionFind(uint32_t count) : parent_(count) {
        for (uint32_t i = 0u; i < count; ++i) {
            parent_[i] = i;
        }
    }

    uint32_t Find(uint32_t row) {
        while (parent_[row] != row) {
            parent_[row] = parent_[parent_[row]];
            row = parent_[row];
        }
        return row;
    }

    void Union(uint32_t a, uint32_t b) {
        const uint32_t root_a = Find(a);
        const uint32_t root_b = Find(b);
        if (root_a == root_b) {
            return;
        }
        if (root_a < root_b) {
            parent_[root_b] = root_a;
        } else {
            parent_[root_a] = root_b;
        }
    }

private:
    std::vector<uint32_t> parent_;
};

// Connect all rows that share a key, given the (key, row) incidence pairs.
// Sorting by (key, row) and unioning each run's rows into its first row yields
// the same connected components as any incremental keyed union: components are
// the transitive closure of the sharing relation, which is union-order
// independent (and the closure is what BuildRowComponentPartitions numbers
// deterministically afterwards).
void UnionBySortedKeyPairs(std::vector<std::pair<uint32_t, uint32_t>>& key_rows,
                           RowUnionFind& uf) {
    std::sort(key_rows.begin(), key_rows.end());
    for (size_t i = 1u; i < key_rows.size(); ++i) {
        if (key_rows[i].first == key_rows[i - 1u].first) {
            uf.Union(key_rows[i - 1u].second, key_rows[i].second);
        }
    }
}

} // namespace

RowComponentPartitions BuildRowComponentPartitions(
    const constraint::RowBuffers& rows,
    const RowColorPartitions& colors,
    const constraint::ContactRowSides* sides,
    uint32_t sides_count,
    const constraint::RowArticulationRefs* art_refs,
    uint32_t art_refs_count) {
    RowComponentPartitions out;
    const uint32_t row_count = rows.RowCount();
    if (row_count == 0u) {
        return out;
    }

    // ---- 1. union-find over the shared-mutable-state relation ----------------
    const RowBodySlotIndex slot_index = BuildRowBodySlotIndex(rows);
    RowUnionFind uf(row_count);
    // (1) shared valid bodies -- the coloring's own conflict relation, keyed by
    // the dense body slot: union every row touching a slot into the slot's
    // first row.
    std::vector<uint32_t> first_row_for_slot(slot_index.slot_count, ~0u);
    // (2)/(3): (key, row) incidence pairs for the qdot tiles / particle
    // velocities, connected by sort + adjacent-union below.
    std::vector<std::pair<uint32_t, uint32_t>> art_rows;
    std::vector<std::pair<uint32_t, uint32_t>> particle_rows;
    for (uint32_t row = 0u; row < row_count; ++row) {
        for (uint32_t bi = 0u; bi < slot_index.counts[row]; ++bi) {
            const uint32_t slot =
                slot_index.slots[static_cast<size_t>(2u) * row + bi];
            uint32_t& first = first_row_for_slot[slot];
            if (first == ~0u) {
                first = row;
            } else {
                uf.Union(first, row);
            }
        }
        // (2) shared articulation qdot tiles (both rows apply into the same
        // art_index slice). In every production emitter this is ALREADY implied
        // by the synthetic per-articulation body key (body_count + art_index),
        // but the kernel's shared state is keyed by art_index, so edge it
        // directly rather than by convention.
        if (art_refs != nullptr && row < art_refs_count) {
            const constraint::RowArticulationSide art_sides[2] = {
                art_refs[row].a, art_refs[row].b};
            for (const auto& side : art_sides) {
                if (side.art_index != constraint::kInvalidArtIndex) {
                    art_rows.emplace_back(side.art_index, row);
                }
            }
        }
        // (3) shared particle velocities (ParticleInvMass sides write
        // particle_velocities[handle] directly).
        if (sides != nullptr && row < sides_count) {
            const constraint::CollidableRef refs[2] = {sides[row].a,
                                                       sides[row].b};
            for (const auto& ref : refs) {
                if (ref.react == constraint::ReactionProviderKind::ParticleInvMass &&
                    ref.handle != ~0u) {
                    particle_rows.emplace_back(ref.handle, row);
                }
            }
        }
        // (4) contact lambda groups: a friction spoke's bound reads
        // TotalNormalLambda over [group_id, group_id + normal_row_count). Group
        // rows come from one manifold (same bodies), so this is normally
        // redundant -- kept so the component relation covers the kernel's reads
        // by CONSTRUCTION, not by emitter convention.
        const constraint::RowMaterial& material = rows.materials[row];
        if (material.kind == constraint::RowKind::Contact) {
            if (material.normal_row_count > 0u && material.group_id < row_count) {
                uf.Union(row, material.group_id);
            }
            if (material.friction_row_count > 0u &&
                material.first_friction_row < row_count) {
                uf.Union(row, material.first_friction_row);
            }
        }
    }
    UnionBySortedKeyPairs(art_rows, uf);
    UnionBySortedKeyPairs(particle_rows, uf);

    // ---- 2. deterministic component ids (first-appearance order by row) ------
    std::vector<uint32_t> component_of_row(row_count);
    std::vector<uint32_t> component_of_root(row_count, ~0u);
    uint32_t component_count = 0u;
    for (uint32_t row = 0u; row < row_count; ++row) {
        const uint32_t root = uf.Find(row);
        if (component_of_root[root] == ~0u) {
            component_of_root[root] = component_count++;
        }
        component_of_row[row] = component_of_root[root];
    }

    // ---- 3. per-component row spans (contiguous, sized up front) -------------
    out.components.resize(component_count);
    for (uint32_t row = 0u; row < row_count; ++row) {
        ++out.components[component_of_row[row]].row_count;
    }
    {
        uint32_t offset = 0u;
        for (auto& comp : out.components) {
            comp.row_offset = offset;
            offset += comp.row_count;
        }
    }

    // ---- 4. fill rows in ascending (color, intra-color position) order and
    // count the per-component color SEGMENTS (a new segment whenever a
    // component's color changes). Two passes over the color partition keep the
    // segment storage contiguous per component.
    out.row_indices.resize(row_count);
    std::vector<uint32_t> cursor(component_count, 0u);
    std::vector<uint32_t> last_color(component_count, ~0u);
    const uint32_t color_count = colors.ColorCount();
    for (uint32_t color = 0u; color < color_count; ++color) {
        const RowColorRange range = colors.color_ranges[color];
        const uint32_t end = range.row_offset + range.row_count;
        for (uint32_t index = range.row_offset; index < end; ++index) {
            const uint32_t row = colors.row_indices[index];
            const uint32_t comp = component_of_row[row];
            out.row_indices[out.components[comp].row_offset + cursor[comp]++] = row;
            if (last_color[comp] != color) {
                last_color[comp] = color;
                ++out.components[comp].segment_count;
            }
        }
    }

    uint32_t total_segments = 0u;
    for (auto& comp : out.components) {
        comp.segment_offset = total_segments;
        total_segments += comp.segment_count;
    }
    out.segments.resize(total_segments);

    std::fill(cursor.begin(), cursor.end(), 0u);
    std::fill(last_color.begin(), last_color.end(), ~0u);
    std::vector<uint32_t> segment_cursor(component_count, 0u);
    for (uint32_t color = 0u; color < color_count; ++color) {
        const RowColorRange range = colors.color_ranges[color];
        const uint32_t end = range.row_offset + range.row_count;
        for (uint32_t index = range.row_offset; index < end; ++index) {
            const uint32_t row = colors.row_indices[index];
            const uint32_t comp = component_of_row[row];
            const RowComponentRange& comp_range = out.components[comp];
            if (last_color[comp] != color) {
                last_color[comp] = color;
                RowColorRange& segment =
                    out.segments[comp_range.segment_offset + segment_cursor[comp]++];
                segment.row_offset = comp_range.row_offset + cursor[comp];
                segment.row_count = 0u;
            }
            RowColorRange& segment =
                out.segments[comp_range.segment_offset + segment_cursor[comp] - 1u];
            ++segment.row_count;
            ++cursor[comp];
        }
    }

    return out;
}

bool ValidateNoSharedBodiesPerColor(const constraint::RowBuffers& rows,
                                    const RowColorPartitions& partitions) {
    // For each color, walk its rows once with a per-color body->seen mark. A VALID body
    // seen twice within the same color means two rows in that color share a body -> the
    // reference's nested-pair RowsConflict would have returned true for that pair. Same
    // boolean result as the O(rows_per_color^2) reference, in O(rows) per color. The
    // seen set is the dense slot remap + a per-color epoch stamp (no clearing, no
    // hashing -- the former unordered_map churn was nearly half the measured
    // Build+Validate 26.1 ms/step at N=1024).
    const RowBodySlotIndex slot_index = BuildRowBodySlotIndex(rows);
    std::vector<uint32_t> seen_epoch(slot_index.slot_count, 0u);
    uint32_t epoch = 0u;
    for (const RowColorRange& range : partitions.color_ranges) {
        ++epoch;
        const uint32_t end = range.row_offset + range.row_count;
        for (uint32_t index = range.row_offset; index < end; ++index) {
            const uint32_t row = partitions.row_indices[index];
            for (uint32_t bi = 0u; bi < slot_index.counts[row]; ++bi) {
                const uint32_t slot =
                    slot_index.slots[static_cast<size_t>(2u) * row + bi];
                if (seen_epoch[slot] == epoch) {
                    return false;
                }
                seen_epoch[slot] = epoch;
            }
        }
    }
    return true;
}

} // namespace nuka::solver::gpu
