#pragma once
// ---------------------------------------------------------------------------
// nuka::solver::gpu::row_scheduler -- deterministic CSR row coloring
// ---------------------------------------------------------------------------

#include "constraint/row_buffers.hpp"

#include <cstdint>
#include <vector>

namespace nuka::solver::gpu {

struct RowColorRange {
    uint32_t row_offset = 0u;
    uint32_t row_count = 0u;
};

struct RowColorPartitions {
    std::vector<uint32_t> row_indices;
    std::vector<RowColorRange> color_ranges;

    uint32_t ColorCount() const {
        return static_cast<uint32_t>(color_ranges.size());
    }
};

RowColorPartitions BuildRowColorPartitions(const constraint::RowBuffers& rows);
bool ValidateNoSharedBodiesPerColor(const constraint::RowBuffers& rows,
                                    const RowColorPartitions& partitions);

} // namespace nuka::solver::gpu
