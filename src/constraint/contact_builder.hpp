#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint -- contact row builder compatibility header
// ---------------------------------------------------------------------------

#include "constraint/contact_manifold.hpp"
#include "constraint/row_builder.hpp"

#include <vector>

namespace nuka::constraint {

inline uint32_t BuildContactRowsForPointCount(uint32_t contact_points,
                                              RowBuffers* out_rows) {
    return BuildContactRows(contact_points, out_rows);
}

} // namespace nuka::constraint
