#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "collision/shape_kind.hpp"
#include "nk/model/model.hpp"
#include "nk/solve/nk_row.hpp"

namespace nuka::nk {

inline uint32_t RigidManifoldPointCapacity(uint32_t a, uint32_t b) {
    using namespace ::nuka::collision;
    if (a > b) std::swap(a, b);
    if ((a == kShapeSphere && (b == kShapeSphere || b == kShapeCapsule || b == kShapeBox || b == kShapePlane)) ||
        (a == kShapeCapsule && b == kShapeCapsule)) return 1u;
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

} // namespace nuka::nk
