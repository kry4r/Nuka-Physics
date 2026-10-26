#pragma once

#include <cstddef>
#include "nk/contact/row_layout.hpp"
#include "nk/contact/row_readout.hpp"

namespace nuka::phi::nkops {

__device__ inline nk::ContactRowRange ContactRowRangeFor(const uint32_t* ranges, uint32_t slot) {
    return {ranges[size_t{slot} * 2u], ranges[size_t{slot} * 2u + 1u]};
}

__device__ inline uint32_t ContactSlotForRow(const uint32_t* ranges,
    uint32_t first, uint32_t count, uint32_t row) {
    uint32_t low = first;
    uint32_t high = first + count;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2u;
        if (ranges[size_t{middle} * 2u] <= row) low = middle + 1u;
        else high = middle;
    }
    return low - 1u;
}

} // namespace nuka::phi::nkops
