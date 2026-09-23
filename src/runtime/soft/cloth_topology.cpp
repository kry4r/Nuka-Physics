#include "runtime/soft/cloth_topology.hpp"

#include "constraint/dihedral_bend.hpp"

#include <map>
#include <stdexcept>
#include <utility>

namespace nuka::runtime::soft {

void BuildClothConstraints(const std::vector<math::Vec3>& rest_positions,
                           const std::vector<ClothTriangle>& triangles,
                           const ClothTopologyOptions& options,
                           XpbdConstraintSet& out) {
    using Edge = std::pair<uint32_t, uint32_t>;
    std::map<Edge, std::vector<uint32_t>> edge_to_apex;
    for (const ClothTriangle& tri : triangles) {
        for (int e = 0; e < 3; ++e) {
            const uint32_t a = tri.v[e];
            const uint32_t b = tri.v[(e + 1) % 3];
            if (a >= rest_positions.size() || b >= rest_positions.size())
                throw std::invalid_argument("Cloth triangle vertex is out of range");
            const Edge key = a < b ? Edge{a, b} : Edge{b, a};
            edge_to_apex[key].push_back(tri.v[(e + 2) % 3]);
        }
    }
    for (const auto& entry : edge_to_apex) {
        const uint32_t a = entry.first.first;
        const uint32_t b = entry.first.second;
        if (options.emit_distance_constraints) {
            XpbdDistanceConstraint distance;
            distance.particle_a = a;
            distance.particle_b = b;
            distance.rest_length = (rest_positions[a] - rest_positions[b]).Length();
            distance.compliance_alpha = options.distance_compliance_alpha;
            out.distance.push_back(distance);
        }
        if (!options.emit_bend_constraints || entry.second.size() != 2u) continue;
        const uint32_t c = entry.second[0];
        const uint32_t d = entry.second[1];
        const auto geometry = constraint::EvaluateDihedralBend(
            rest_positions[a], rest_positions[b], rest_positions[c], rest_positions[d]);
        if (!geometry.valid) continue;
        XpbdBendConstraint bend;
        bend.particle[0] = a;
        bend.particle[1] = b;
        bend.particle[2] = c;
        bend.particle[3] = d;
        bend.rest_angle = geometry.angle;
        bend.compliance_alpha = options.bend_compliance_alpha;
        out.bend.push_back(bend);
    }
}

}  // namespace nuka::runtime::soft
