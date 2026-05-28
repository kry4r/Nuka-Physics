// ---------------------------------------------------------------------------
// nuka::solver::gpu::row_scheduler implementation
// ---------------------------------------------------------------------------

#include "solver/gpu/row_scheduler.hpp"

#include <algorithm>
#include <vector>

namespace nuka::solver::gpu {

namespace {

bool RowsConflict(const constraint::RowBuffers& rows,
                  uint32_t lhs,
                  uint32_t rhs) {
    const auto a = rows.BodiesForRow(lhs);
    const auto b = rows.BodiesForRow(rhs);
    const uint32_t lhs_bodies[2] = {a.body_a, a.body_b};
    const uint32_t rhs_bodies[2] = {b.body_a, b.body_b};
    for (const uint32_t lhs_body : lhs_bodies) {
        if (lhs_body == constraint::kInvalidBodyIndex) {
            continue;
        }
        for (const uint32_t rhs_body : rhs_bodies) {
            if (lhs_body == rhs_body) {
                return true;
            }
        }
    }
    return false;
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

    for (uint32_t row = 0; row < row_count; ++row) {
        std::vector<uint8_t> used(color_count, 0u);
        for (uint32_t prior = 0; prior < row; ++prior) {
            if (RowsConflict(rows, row, prior)) {
                used[row_colors[prior]] = 1u;
            }
        }

        uint32_t color = 0u;
        while (color < color_count && used[color] != 0u) {
            ++color;
        }
        if (color == color_count) {
            ++color_count;
        }
        row_colors[row] = color;
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
    for (const RowColorRange& range : partitions.color_ranges) {
        const uint32_t end = range.row_offset + range.row_count;
        for (uint32_t lhs_index = range.row_offset; lhs_index < end; ++lhs_index) {
            for (uint32_t rhs_index = lhs_index + 1u; rhs_index < end; ++rhs_index) {
                if (RowsConflict(rows,
                                 partitions.row_indices[lhs_index],
                                 partitions.row_indices[rhs_index])) {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace nuka::solver::gpu
