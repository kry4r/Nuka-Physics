// ---------------------------------------------------------------------------
// nk::SolveSchedule implementation (plan §3.4 / M4).
//
// The coloring + component algorithms below are the MIGRATED
// src/solver/gpu/row_scheduler.cu BuildRowColorPartitions /
// BuildRowComponentPartitions (the legacy per-step host scheduler), re-keyed
// from constraint::RowBuffers to ScheduleRow. Every algorithmic decision —
// the dense slot remap, greedy lowest-available color in row-index order,
// the counting-sort partition, the smaller-root-wins union-find, the
// first-appearance component numbering, the (color, intra-color) segment
// emission — is preserved 1:1 so the worst-case schedule's execution order
// matches what the legacy scheduler produced for the same emission order.
// ---------------------------------------------------------------------------

#include "nk/solve/schedule.hpp"

#include <algorithm>
#include <utility>

#include "nk/model/model.hpp"

namespace nuka::nk {

namespace {

constexpr uint32_t kNoKey = ~0u;
constexpr uint32_t kNoSlot = ~0u;

// Dense remap of the row keys (legacy RowBodySlotIndex 1:1: sort + unique the
// valid keys once, rewrite each row's keys as dense [0, slot_count) slots).
struct RowKeySlotIndex {
    std::vector<uint32_t> slots;   // [2 * row_count], kNoSlot-padded
    std::vector<uint8_t>  counts;  // per row: 0..2 valid slots
    uint32_t slot_count = 0;
};

RowKeySlotIndex BuildRowKeySlotIndex(const std::vector<ScheduleRow>& rows) {
    const uint32_t row_count = static_cast<uint32_t>(rows.size());
    RowKeySlotIndex out;
    out.slots.assign(static_cast<size_t>(2u) * row_count, kNoSlot);
    out.counts.assign(row_count, 0u);

    std::vector<uint32_t> keys;
    keys.reserve(static_cast<size_t>(2u) * row_count);
    for (uint32_t row = 0; row < row_count; ++row) {
        uint32_t vals[2];
        uint32_t n = 0;
        if (rows[row].key_a != kNoKey) vals[n++] = rows[row].key_a;
        if (rows[row].key_b != kNoKey && rows[row].key_b != rows[row].key_a) {
            vals[n++] = rows[row].key_b;
        }
        out.counts[row] = static_cast<uint8_t>(n);
        for (uint32_t k = 0; k < n; ++k) {
            out.slots[static_cast<size_t>(2u) * row + k] = vals[k];
            keys.push_back(vals[k]);
        }
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    out.slot_count = static_cast<uint32_t>(keys.size());
    for (uint32_t row = 0; row < row_count; ++row) {
        for (uint32_t k = 0; k < out.counts[row]; ++k) {
            uint32_t& slot = out.slots[static_cast<size_t>(2u) * row + k];
            slot = static_cast<uint32_t>(
                std::lower_bound(keys.begin(), keys.end(), slot) - keys.begin());
        }
    }
    return out;
}

// Legacy BuildRowColorPartitions 1:1 (greedy lowest-available color in row
// index order; per-slot sorted color lists; epoch-stamped min-excludant).
struct ColorPartitions {
    std::vector<uint32_t> row_indices;
    std::vector<std::pair<uint32_t, uint32_t>> color_ranges;  // {offset, count}
};

ColorPartitions BuildColors(const std::vector<ScheduleRow>& rows,
                            const RowKeySlotIndex& slot_index) {
    ColorPartitions out;
    const uint32_t row_count = static_cast<uint32_t>(rows.size());
    if (row_count == 0) return out;

    std::vector<uint32_t> row_colors(row_count, 0u);
    uint32_t color_count = 0;
    std::vector<std::vector<uint32_t>> slot_colors(slot_index.slot_count);
    std::vector<uint32_t> stamp;

    for (uint32_t row = 0; row < row_count; ++row) {
        const uint32_t epoch = row + 1u;
        for (uint32_t k = 0; k < slot_index.counts[row]; ++k) {
            const uint32_t slot = slot_index.slots[static_cast<size_t>(2u) * row + k];
            for (const uint32_t used : slot_colors[slot]) stamp[used] = epoch;
        }
        uint32_t color = 0;
        while (color < color_count && stamp[color] == epoch) ++color;
        if (color == color_count) {
            ++color_count;
            stamp.push_back(0u);
        }
        row_colors[row] = color;
        for (uint32_t k = 0; k < slot_index.counts[row]; ++k) {
            const uint32_t slot = slot_index.slots[static_cast<size_t>(2u) * row + k];
            std::vector<uint32_t>& colors = slot_colors[slot];
            const auto pos = std::lower_bound(colors.begin(), colors.end(), color);
            if (pos == colors.end() || *pos != color) colors.insert(pos, color);
        }
    }

    out.color_ranges.assign(color_count, {0u, 0u});
    for (uint32_t row = 0; row < row_count; ++row) {
        ++out.color_ranges[row_colors[row]].second;
    }
    {
        uint32_t offset = 0;
        for (uint32_t c = 0; c < color_count; ++c) {
            out.color_ranges[c].first = offset;
            offset += out.color_ranges[c].second;
        }
    }
    out.row_indices.resize(row_count);
    std::vector<uint32_t> cursor(color_count, 0u);
    for (uint32_t row = 0; row < row_count; ++row) {
        const uint32_t c = row_colors[row];
        out.row_indices[out.color_ranges[c].first + cursor[c]++] = row;
    }
    return out;
}

// Legacy RowUnionFind 1:1 (path-halving find; smaller-root-index wins).
class RowUnionFind {
public:
    explicit RowUnionFind(uint32_t count) : parent_(count) {
        for (uint32_t i = 0; i < count; ++i) parent_[i] = i;
    }
    uint32_t Find(uint32_t row) {
        while (parent_[row] != row) {
            parent_[row] = parent_[parent_[row]];
            row = parent_[row];
        }
        return row;
    }
    void Union(uint32_t a, uint32_t b) {
        const uint32_t ra = Find(a);
        const uint32_t rb = Find(b);
        if (ra == rb) return;
        if (ra < rb) parent_[rb] = ra; else parent_[ra] = rb;
    }

private:
    std::vector<uint32_t> parent_;
};

}  // namespace

SolveScheduleResult SolveSchedule::Partition(const std::vector<ScheduleRow>& rows,
                                             const std::vector<uint32_t>& row_env) {
    SolveScheduleResult out;
    const uint32_t row_count = static_cast<uint32_t>(rows.size());
    if (row_count == 0) return out;

    const RowKeySlotIndex slot_index = BuildRowKeySlotIndex(rows);
    const ColorPartitions colors = BuildColors(rows, slot_index);

    // ---- union-find over the shared-mutable-state relation (legacy pass 1:1):
    // (1) shared keys (bodies / artic tiles), (2) explicit artic-tile edges,
    // (3) lambda-group anchors.
    RowUnionFind uf(row_count);
    std::vector<uint32_t> first_row_for_slot(slot_index.slot_count, ~0u);
    std::vector<std::pair<uint32_t, uint32_t>> art_rows;
    for (uint32_t row = 0; row < row_count; ++row) {
        for (uint32_t k = 0; k < slot_index.counts[row]; ++k) {
            const uint32_t slot = slot_index.slots[static_cast<size_t>(2u) * row + k];
            uint32_t& first = first_row_for_slot[slot];
            if (first == ~0u) first = row; else uf.Union(first, row);
        }
        if (rows[row].artic != ~0u) art_rows.emplace_back(rows[row].artic, row);
        if (rows[row].group_first != ~0u && rows[row].group_first < row_count) {
            uf.Union(row, rows[row].group_first);
        }
    }
    std::sort(art_rows.begin(), art_rows.end());
    for (size_t i = 1; i < art_rows.size(); ++i) {
        if (art_rows[i].first == art_rows[i - 1].first) {
            uf.Union(art_rows[i - 1].second, art_rows[i].second);
        }
    }

    // ---- deterministic component ids (first-appearance by row index) -------
    std::vector<uint32_t> component_of_row(row_count);
    std::vector<uint32_t> component_of_root(row_count, ~0u);
    uint32_t component_count = 0;
    for (uint32_t row = 0; row < row_count; ++row) {
        const uint32_t root = uf.Find(row);
        if (component_of_root[root] == ~0u) component_of_root[root] = component_count++;
        component_of_row[row] = component_of_root[root];
    }

    // ---- per-component row spans + (color, intra-color) fill + segments ----
    struct CompRange { uint32_t seg_off = 0, seg_cnt = 0, row_off = 0, row_cnt = 0; };
    std::vector<CompRange> comps(component_count);
    for (uint32_t row = 0; row < row_count; ++row) ++comps[component_of_row[row]].row_cnt;
    {
        uint32_t offset = 0;
        for (auto& c : comps) { c.row_off = offset; offset += c.row_cnt; }
    }

    out.row_order.resize(row_count);
    std::vector<uint32_t> cursor(component_count, 0u);
    std::vector<uint32_t> last_color(component_count, ~0u);
    const uint32_t color_count = static_cast<uint32_t>(colors.color_ranges.size());
    for (uint32_t color = 0; color < color_count; ++color) {
        const auto range = colors.color_ranges[color];
        for (uint32_t i = range.first; i < range.first + range.second; ++i) {
            const uint32_t row = colors.row_indices[i];
            const uint32_t comp = component_of_row[row];
            out.row_order[comps[comp].row_off + cursor[comp]++] = row;
            if (last_color[comp] != color) {
                last_color[comp] = color;
                ++comps[comp].seg_cnt;
            }
        }
    }
    uint32_t total_segments = 0;
    for (auto& c : comps) { c.seg_off = total_segments; total_segments += c.seg_cnt; }

    out.color_segments.assign(static_cast<size_t>(total_segments) * 2u, 0u);
    std::fill(cursor.begin(), cursor.end(), 0u);
    std::fill(last_color.begin(), last_color.end(), ~0u);
    std::vector<uint32_t> seg_cursor(component_count, 0u);
    for (uint32_t color = 0; color < color_count; ++color) {
        const auto range = colors.color_ranges[color];
        for (uint32_t i = range.first; i < range.first + range.second; ++i) {
            const uint32_t row = colors.row_indices[i];
            const uint32_t comp = component_of_row[row];
            const CompRange& cr = comps[comp];
            if (last_color[comp] != color) {
                last_color[comp] = color;
                const uint32_t seg = cr.seg_off + seg_cursor[comp]++;
                out.color_segments[static_cast<size_t>(seg) * 2u + 0u] =
                    cr.row_off + cursor[comp];
                out.color_segments[static_cast<size_t>(seg) * 2u + 1u] = 0u;
            }
            const uint32_t seg = cr.seg_off + seg_cursor[comp] - 1u;
            ++out.color_segments[static_cast<size_t>(seg) * 2u + 1u];
            ++cursor[comp];
        }
    }

    // ---- island quads {seg_off, seg_cnt, flags, env} ------------------------
    out.islands.assign(static_cast<size_t>(component_count) * 4u, 0u);
    for (uint32_t comp = 0; comp < component_count; ++comp) {
        out.islands[static_cast<size_t>(comp) * 4u + 0u] = comps[comp].seg_off;
        out.islands[static_cast<size_t>(comp) * 4u + 1u] = comps[comp].seg_cnt;
    }
    for (uint32_t row = 0; row < row_count; ++row) {
        const uint32_t comp = component_of_row[row];
        if (rows[row].artic != ~0u) {
            out.islands[static_cast<size_t>(comp) * 4u + 2u] |= 1u;
        }
        out.islands[static_cast<size_t>(comp) * 4u + 3u] =
            row < row_env.size() ? row_env[row] : 0u;
    }
    out.island_count = component_count;
    out.segment_count = total_segments;
    return out;
}

std::vector<uint32_t> UnionSlotRowBases(const Model& model, uint32_t* rows_per_env) {
    std::vector<uint32_t> bases;
    bases.reserve(model.union_slots.size());
    uint32_t cursor = 0;
    for (const UnionSlot& s : model.union_slots) {
        bases.push_back(cursor);
        cursor += s.MaxRows();
    }
    if (rows_per_env != nullptr) *rows_per_env = cursor;
    return bases;
}

void SolveSchedule::Build(Model* model) {
    if (model == nullptr) return;
    if (model->contact_family == ContactFamily::UnionCsr &&
        model->union_slots.empty() && model->schedule_island_count > 0) {
        // Oracle-harness seam: a hand-built schedule (Partition output written
        // straight into the model by a test that uploads rows directly) is
        // kept as-is — there is no slot template to derive one from.
        return;
    }
    model->schedule_row_order.clear();
    model->schedule_color_segments.clear();
    model->schedule_islands.clear();
    model->schedule_island_count = 0;
    model->schedule_segment_count = 0;
    if (model->contact_family != ContactFamily::UnionCsr ||
        model->union_slots.empty()) {
        return;  // fused / contact-free family: no CSR schedule.
    }

    const ModelCapacities& cap = model->capacities;
    const uint32_t E = cap.env_count;
    uint32_t rows_per_env = 0;
    const std::vector<uint32_t> bases = UnionSlotRowBases(*model, &rows_per_env);

    // Worst-case conflict keys, EXACTLY the legacy emission's body-index keys:
    //   articulation side -> total_body_count + env   (the bijection-invariant
    //                        synthetic key paired with art_index == env)
    //   rigid side        -> env * bodies_per_env + body
    //   static side       -> no key (legacy kInvalidBodyIndex never conflicts)
    // Emission order: env-major, slots in template order, per slot all normal
    // rows then all spokes — the AssembleRows fixed slot layout.
    const uint32_t total_body_count = cap.bodies_per_env * E;
    std::vector<ScheduleRow> rows(static_cast<size_t>(rows_per_env) * E);
    std::vector<uint32_t> row_env(rows.size(), 0u);
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t env_base = e * rows_per_env;
        for (size_t s = 0; s < model->union_slots.size(); ++s) {
            const UnionSlot& u = model->union_slots[s];
            const uint32_t slot_base = env_base + bases[s];
            uint32_t key_a = kNoKey;  // ~0u
            uint32_t key_b = kNoKey;
            uint32_t artic = ~0u;
            switch (u.cls) {
                case UnionSlot::kFootSpherePlane:
                    key_a = total_body_count + e;
                    artic = e;
                    break;
                case UnionSlot::kFingerSphereHull:
                    key_a = total_body_count + e;
                    key_b = e * cap.bodies_per_env + u.body;
                    artic = e;
                    break;
                case UnionSlot::kBodyBoxPlane:
                    key_a = e * cap.bodies_per_env + u.body;
                    break;
                default:
                    break;
            }
            const uint32_t n_rows = u.MaxRows();
            for (uint32_t r = 0; r < n_rows; ++r) {
                ScheduleRow& sr = rows[slot_base + r];
                sr.key_a = key_a;
                sr.key_b = key_b;
                sr.artic = artic;
                sr.group_first = slot_base;  // the slot's first normal row
                row_env[slot_base + r] = e;
            }
        }
    }

    SolveScheduleResult result = Partition(rows, row_env);
    model->schedule_row_order = std::move(result.row_order);
    model->schedule_color_segments = std::move(result.color_segments);
    model->schedule_islands = std::move(result.islands);
    model->schedule_island_count = result.island_count;
    model->schedule_segment_count = result.segment_count;
}

}  // namespace nuka::nk
