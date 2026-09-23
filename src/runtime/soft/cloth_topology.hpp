#pragma once

#include "import/cooker/xpbd_cooker_types.hpp"
#include "math/vec3.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::soft {

struct ClothTriangle {
    uint32_t v[3] = {0u, 0u, 0u};
};

struct ClothTopologyOptions {
    float distance_compliance_alpha = 0.0f;
    float bend_compliance_alpha = 0.0f;
    bool emit_distance_constraints = true;
    bool emit_bend_constraints = true;
};

// Deterministic stretch edges and signed dihedral hinges from a triangle mesh.
// Boundary and degenerate edges have no bending constraint.
void BuildClothConstraints(const std::vector<math::Vec3>& rest_positions,
                           const std::vector<ClothTriangle>& triangles,
                           const ClothTopologyOptions& options,
                           XpbdConstraintSet& out);

}  // namespace nuka::runtime::soft
