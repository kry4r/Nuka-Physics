// ---------------------------------------------------------------------------
// Contact constraint builder
// ---------------------------------------------------------------------------

#include "constraint/contact_builder.hpp"

#include <algorithm>
#include <cfloat>

namespace nuka::constraint {

ConstraintBlock BuildContactBlock(uint32_t contact_points) {
    ConstraintBlock block;
    block.type = ConstraintType::Contact;

    // contact_points normal rows + 1 friction row, clamped to kMaxRows.
    block.row_count = std::min(contact_points + 1,
                               ConstraintBlock::kMaxRows);

    // Initialise Jacobians to sensible defaults (unit-Y normal).
    for (uint32_t i = 0; i < contact_points && i < ConstraintBlock::kMaxRows; ++i) {
        block.jacobian_linear_a[i]  = {0.0f, 1.0f, 0.0f};
        block.jacobian_angular_a[i] = math::Vec3::Zero();
        block.jacobian_linear_b[i]  = {0.0f, -1.0f, 0.0f};
        block.jacobian_angular_b[i] = math::Vec3::Zero();

        block.lower_limit[i] = 0.0f;
        block.upper_limit[i] = FLT_MAX;
    }

    // Friction row (last row in the block).
    uint32_t f = block.row_count - 1;
    if (f < ConstraintBlock::kMaxRows) {
        block.jacobian_linear_a[f]  = {1.0f, 0.0f, 0.0f};
        block.jacobian_angular_a[f] = math::Vec3::Zero();
        block.jacobian_linear_b[f]  = {-1.0f, 0.0f, 0.0f};
        block.jacobian_angular_b[f] = math::Vec3::Zero();

        block.lower_limit[f] = -FLT_MAX;
        block.upper_limit[f] =  FLT_MAX;
    }

    return block;
}

std::vector<ConstraintBlock> BuildContactConstraints(
    const std::vector<ContactManifold>& manifolds) {
    std::vector<ConstraintBlock> blocks;
    blocks.reserve(manifolds.size());

    for (const auto& m : manifolds) {
        if (m.point_count == 0) continue;
        ConstraintBlock block = BuildContactBlock(m.point_count);
        block.body_a = m.body_a;
        block.body_b = m.body_b;
        blocks.push_back(block);
    }

    return blocks;
}

} // namespace nuka::constraint
