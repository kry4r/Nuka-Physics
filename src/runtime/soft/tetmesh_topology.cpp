// ---------------------------------------------------------------------------
// nuka::runtime::soft -- cook-time tet-mesh XPBD topology builder (v0.7 p09-B)
// ---------------------------------------------------------------------------
// HOST cook-time code (no device physics): builds the distance + volume
// constraint list once from a tet mesh. The runtime iterates it on the GPU.
// ---------------------------------------------------------------------------

#include "runtime/soft/tetmesh_topology.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

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

std::vector<uint32_t> ExtractBoundaryTriangles(
    const std::vector<math::Vec3>& rest_positions,
    const std::vector<TetMeshTet>& tets) {
    // The four faces of a tet, each as (3 face verts, opposite vert) in local
    // 0..3 indices. The opposite vertex lets us orient the face outward.
    static constexpr int kFaces[4][4] = {{1, 2, 3, 0}, {0, 3, 2, 1},
                                         {0, 1, 3, 2}, {0, 2, 1, 3}};

    // Count incidence per UNDIRECTED face (sorted triple) and remember one
    // oriented instance + its opposite vertex; a face seen once is on the boundary.
    struct FaceRec { uint32_t a, b, c, opp; uint32_t count; uint32_t order; };
    std::map<std::array<uint32_t, 3>, FaceRec> faces;
    uint32_t order = 0u;
    for (const TetMeshTet& tet : tets) {
        for (const auto& f : kFaces) {
            const uint32_t a = tet.v[f[0]], b = tet.v[f[1]], c = tet.v[f[2]];
            std::array<uint32_t, 3> key{a, b, c};
            std::sort(key.begin(), key.end());
            auto it = faces.find(key);
            if (it == faces.end()) {
                faces.emplace(key, FaceRec{a, b, c, tet.v[f[3]], 1u, order++});
            } else {
                ++it->second.count;
            }
        }
    }

    // Emit boundary faces (count == 1) in first-seen order, wound so the face
    // normal points AWAY from the opposite vertex (outward).
    std::vector<std::pair<uint32_t, const FaceRec*>> boundary;
    for (const auto& kv : faces) {
        if (kv.second.count == 1u) boundary.emplace_back(kv.second.order, &kv.second);
    }
    std::sort(boundary.begin(), boundary.end(),
              [](const auto& x, const auto& y) { return x.first < y.first; });

    std::vector<uint32_t> tris;
    tris.reserve(boundary.size() * 3u);
    for (const auto& entry : boundary) {
        const FaceRec& fr = *entry.second;
        const math::Vec3& pa = rest_positions[fr.a];
        const math::Vec3& pb = rest_positions[fr.b];
        const math::Vec3& pc = rest_positions[fr.c];
        const math::Vec3 n = (pb - pa).Cross(pc - pa);
        const math::Vec3 to_opp = rest_positions[fr.opp] - pa;
        if (n.Dot(to_opp) > 0.0f) {
            tris.insert(tris.end(), {fr.a, fr.c, fr.b});  // flip to face outward
        } else {
            tris.insert(tris.end(), {fr.a, fr.b, fr.c});
        }
    }
    return tris;
}

} // namespace nuka::runtime::soft
