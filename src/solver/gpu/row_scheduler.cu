// ---------------------------------------------------------------------------
// nuka::solver::gpu::row_scheduler implementation
// ---------------------------------------------------------------------------

#include "solver/gpu/row_scheduler.hpp"

#include <algorithm>
#include <unordered_map>
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

} // namespace

RowColorPartitions BuildRowColorPartitions(const constraint::RowBuffers& rows) {
    RowColorPartitions partitions;
    const uint32_t row_count = rows.RowCount();
    if (row_count == 0u) {
        return partitions;
    }

    std::vector<uint32_t> row_colors(row_count, 0u);
    uint32_t color_count = 0u;

    // body -> the colors already assigned to prior rows touching that body (deduped,
    // kept sorted ascending so the forbidden-set merge + min-excludant scan is cheap).
    // This makes per-row cost O(K) instead of O(prior rows): the lowest available color
    // is the smallest non-negative integer not used by any conflicting prior row, which
    // is exactly the union of body_colors over this row's valid bodies. Output is
    // byte-identical to the plain greedy-lowest-available-by-row-index reference.
    std::unordered_map<uint32_t, std::vector<uint32_t>> body_colors;

    // Scratch forbidden set, sorted ascending, rebuilt per row (cleared, not reallocated).
    std::vector<uint32_t> forbidden;

    for (uint32_t row = 0; row < row_count; ++row) {
        const RowValidBodies vb = ValidBodiesForRow(rows, row);

        forbidden.clear();
        for (uint32_t bi = 0u; bi < vb.count; ++bi) {
            const auto it = body_colors.find(vb.bodies[bi]);
            if (it == body_colors.end()) {
                continue;
            }
            forbidden.insert(forbidden.end(), it->second.begin(), it->second.end());
        }
        std::sort(forbidden.begin(), forbidden.end());

        // Min-excludant over the sorted forbidden colors: the lowest color not present.
        // Equivalent to the reference's "lowest index in used[] that is 0"; a value
        // equal to color_count means every existing color is forbidden -> new color.
        uint32_t color = 0u;
        for (const uint32_t used_color : forbidden) {
            if (used_color > color) {
                break;
            }
            if (used_color == color) {
                ++color;
            }
        }
        if (color == color_count) {
            ++color_count;
        }
        row_colors[row] = color;

        // Record this color on each of the row's valid bodies (deduped, sorted-insert).
        for (uint32_t bi = 0u; bi < vb.count; ++bi) {
            std::vector<uint32_t>& colors = body_colors[vb.bodies[bi]];
            const auto pos = std::lower_bound(colors.begin(), colors.end(), color);
            if (pos == colors.end() || *pos != color) {
                colors.insert(pos, color);
            }
        }
    }

    partitions.color_ranges.resize(color_count);
    for (uint32_t color = 0; color < color_count; ++color) {
        partitions.color_ranges[color].row_offset =
            static_cast<uint32_t>(partitions.row_indices.size());
        for (uint32_t row = 0; row < row_count; ++row) {
            if (row_colors[row] == color) {
                partitions.row_indices.push_back(row);
            }
        }
        partitions.color_ranges[color].row_count =
            static_cast<uint32_t>(partitions.row_indices.size()) -
            partitions.color_ranges[color].row_offset;
    }

    return partitions;
}

bool ValidateNoSharedBodiesPerColor(const constraint::RowBuffers& rows,
                                    const RowColorPartitions& partitions) {
    // For each color, walk its rows once with a per-color body->seen set. A VALID body
    // seen twice within the same color means two rows in that color share a body -> the
    // reference's nested-pair RowsConflict would have returned true for that pair. Same
    // boolean result as the O(rows_per_color^2) reference, in O(rows) per color.
    std::unordered_map<uint32_t, char> seen;
    for (const RowColorRange& range : partitions.color_ranges) {
        seen.clear();
        const uint32_t end = range.row_offset + range.row_count;
        for (uint32_t index = range.row_offset; index < end; ++index) {
            const RowValidBodies vb =
                ValidBodiesForRow(rows, partitions.row_indices[index]);
            for (uint32_t bi = 0u; bi < vb.count; ++bi) {
                if (!seen.emplace(vb.bodies[bi], 1).second) {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace nuka::solver::gpu
