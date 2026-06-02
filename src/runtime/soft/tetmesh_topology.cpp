// ---------------------------------------------------------------------------
// nuka::runtime::soft -- cook-time tet-mesh XPBD topology builder (v0.7 p09-B)
// ---------------------------------------------------------------------------
// HOST cook-time code (no device physics): builds the distance + volume
// constraint list once from a tet mesh. The runtime iterates it on the GPU.
// ---------------------------------------------------------------------------

#include "runtime/soft/tetmesh_topology.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>

namespace nuka::runtime::soft {

float TetSignedVolumeTimes6(const math::Vec3& p0, const math::Vec3& p1,
                            const math::Vec3& p2, const math::Vec3& p3) {
    // det([p1-p0, p2-p0, p3-p0]) = (p1-p0) . ((p2-p0) x (p3-p0)). This MUST match
    // the solver's e1 . (e2 x e3) byte-for-byte in operand order so the rest
    // constraint value is exactly the runtime value at the rest configuration.
    const math::Vec3 e1 = p1 - p0;
    const math::Vec3 e2 = p2 - p0;
    const math::Vec3 e3 = p3 - p0;
    return e1.Dot(e2.Cross(e3));
}

void BuildTetMeshConstraints(const std::vector<math::Vec3>& rest_positions,
                             const std::vector<TetMeshTet>& tets,
                             const TetMeshTopologyOptions& options,
                             XpbdConstraintSet& out) {
    // The six edges of a tet (vertex-index pairs into the tet's local 0..3).
    static constexpr int kEdges[6][2] = {{0, 1}, {0, 2}, {0, 3},
                                         {1, 2}, {1, 3}, {2, 3}};

    // De-duplicate edges shared between tets so each physical edge gets ONE
    // distance constraint. Ordered set => deterministic emission order.
    std::set<std::pair<uint32_t, uint32_t>> seen_edges;

    if (options.emit_distance_constraints) {
        for (const TetMeshTet& tet : tets) {
            for (const auto& e : kEdges) {
                uint32_t a = tet.v[e[0]];
                uint32_t b = tet.v[e[1]];
                if (a > b) {
                    std::swap(a, b);
                }
                if (!seen_edges.emplace(a, b).second) {
                    continue;  // edge already emitted by an adjacent tet.
                }
                XpbdDistanceConstraint dc;
                dc.particle_a = a;
                dc.particle_b = b;
                dc.rest_length = (rest_positions[a] - rest_positions[b]).Length();
                dc.compliance_alpha = options.distance_compliance_alpha;
                out.distance.push_back(dc);
            }
        }
    }

    // One volume constraint per tet (volume is per-tet, never shared).
    for (const TetMeshTet& tet : tets) {
        XpbdVolumeConstraint vc;
        for (int j = 0; j < 4; ++j) {
            vc.particle[j] = tet.v[j];
        }
        vc.rest_volume_times6 = TetSignedVolumeTimes6(
            rest_positions[tet.v[0]], rest_positions[tet.v[1]],
            rest_positions[tet.v[2]], rest_positions[tet.v[3]]);
        vc.compliance_alpha = options.volume_compliance_alpha;
        out.volume.push_back(vc);
    }
}

} // namespace nuka::runtime::soft
