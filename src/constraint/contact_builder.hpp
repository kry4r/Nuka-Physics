#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint -- contact constraint builder
// ---------------------------------------------------------------------------

#include "constraint/constraint_block.hpp"
#include "constraint/contact_manifold.hpp"

#include <vector>

namespace nuka::constraint {

/// Build a single ConstraintBlock for `contact_points` normal contacts.
/// Rows = contact_points (normal) + 1 (friction), capped at kMaxRows.
ConstraintBlock BuildContactBlock(uint32_t contact_points);

/// Build ConstraintBlocks for an entire list of manifolds.
std::vector<ConstraintBlock> BuildContactConstraints(
    const std::vector<ContactManifold>& manifolds);

} // namespace nuka::constraint
