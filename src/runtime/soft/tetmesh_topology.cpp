// ---------------------------------------------------------------------------
// nuka::runtime::soft -- cook-time tet-mesh XPBD topology builder (v0.7 p09-B)
// ---------------------------------------------------------------------------
// HOST cook-time code (no device physics): builds the distance + volume
// constraint list once from a tet mesh. The runtime iterates it on the GPU.
// ---------------------------------------------------------------------------

#include "runtime/soft/tetmesh_topology.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
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

namespace {

// 5-tet split of one grid cell (8 corners in (x,y,z) order). The split alternates
// by cell parity so a shared face's diagonal matches between neighbours (no cracks).
void EmitCellTets(const uint32_t cn[8], bool even, std::vector<TetMeshTet>& out) {
    auto tet = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        out.push_back(TetMeshTet{{cn[a], cn[b], cn[c], cn[d]}});
    };
    if (even) {
        tet(0, 1, 2, 4); tet(1, 2, 3, 7); tet(1, 4, 5, 7);
        tet(2, 4, 6, 7); tet(1, 2, 4, 7);
    } else {
        tet(0, 1, 3, 5); tet(0, 2, 3, 6); tet(0, 4, 5, 6);
        tet(3, 5, 6, 7); tet(0, 3, 5, 6);
    }
}

}  // namespace

TetLattice BuildTetLattice(const math::Vec3& origin, uint32_t cells[3],
                           float cell_len,
                           const std::function<bool(const math::Vec3&)>& keep_cell) {
    const uint32_t nx = cells[0] + 1u, ny = cells[1] + 1u, nz = cells[2] + 1u;
    auto gid = [nx, ny](uint32_t i, uint32_t j, uint32_t k) {
        return (k * ny + j) * nx + i;
    };
    auto corner = [&](uint32_t i, uint32_t j, uint32_t k) {
        return math::Vec3{origin.x + static_cast<float>(i) * cell_len,
                          origin.y + static_cast<float>(j) * cell_len,
                          origin.z + static_cast<float>(k) * cell_len};
    };

    // Mark used lattice vertices over kept cells, then compact to a dense index.
    std::vector<uint32_t> remap(static_cast<size_t>(nx) * ny * nz, ~0u);
    TetLattice lat;
    std::vector<TetMeshTet> cell_tets;
    for (uint32_t k = 0; k + 1 < nz; ++k)
        for (uint32_t j = 0; j + 1 < ny; ++j)
            for (uint32_t i = 0; i + 1 < nx; ++i) {
                const math::Vec3 c = corner(i, j, k) +
                    math::Vec3{0.5f * cell_len, 0.5f * cell_len, 0.5f * cell_len};
                if (!keep_cell(c)) continue;
                const uint32_t g[8] = {
                    gid(i, j, k),         gid(i + 1, j, k),
                    gid(i, j + 1, k),     gid(i + 1, j + 1, k),
                    gid(i, j, k + 1),     gid(i + 1, j, k + 1),
                    gid(i, j + 1, k + 1), gid(i + 1, j + 1, k + 1)};
                uint32_t cn[8];
                const uint32_t base[3] = {i, j, k};
                const uint32_t off[8][3] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},
                                            {0,0,1},{1,0,1},{0,1,1},{1,1,1}};
                for (uint32_t v = 0; v < 8u; ++v) {
                    if (remap[g[v]] == ~0u) {
                        remap[g[v]] = static_cast<uint32_t>(lat.rest.size());
                        lat.rest.push_back(corner(base[0] + off[v][0],
                                                  base[1] + off[v][1],
                                                  base[2] + off[v][2]));
                    }
                    cn[v] = remap[g[v]];
                }
                EmitCellTets(cn, ((i + j + k) & 1u) == 0u, cell_tets);
            }
    lat.tets = std::move(cell_tets);
    return lat;
}

TetLattice BuildSphereTetLattice(const math::Vec3& center, float radius,
                                 uint32_t cells, float cell_len) {
    const float span = static_cast<float>(cells) * cell_len;
    const math::Vec3 origin{center.x - 0.5f * span, center.y - 0.5f * span,
                            center.z - 0.5f * span};
    // Keep a cell only when ALL 8 corners are inside the radius (the cell lies
    // fully within the sphere), so the kept solid has a watertight boundary.
    const float r2 = radius * radius;
    auto inside_cell = [&](const math::Vec3& cell_center) {
        const float h = 0.5f * cell_len;
        for (int sx = -1; sx <= 1; sx += 2)
            for (int sy = -1; sy <= 1; sy += 2)
                for (int sz = -1; sz <= 1; sz += 2) {
                    const float dx = cell_center.x + sx * h - center.x;
                    const float dy = cell_center.y + sy * h - center.y;
                    const float dz = cell_center.z + sz * h - center.z;
                    if (dx * dx + dy * dy + dz * dz > r2) return false;
                }
        return true;
    };
    uint32_t c3[3] = {cells, cells, cells};
    return BuildTetLattice(origin, c3, cell_len, inside_cell);
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

namespace {

// Barycentric coords of p in tet (p0,p1,p2,p3): columns M=[p1-p0,p2-p0,p3-p0],
// solve M*[u,v,w]=p-p0, bary=(1-u-v-w,u,v,w). False on a degenerate (flat) tet.
bool TetBarycentric(const math::Vec3& p, const math::Vec3& p0, const math::Vec3& p1,
                    const math::Vec3& p2, const math::Vec3& p3, float bary[4]) {
    const math::Vec3 e1 = p1 - p0, e2 = p2 - p0, e3 = p3 - p0;
    const float det = e1.Dot(e2.Cross(e3));
    if (std::fabs(det) < 1e-20f) return false;
    const math::Vec3 r = p - p0;
    const float inv = 1.0f / det;
    const float u = r.Dot(e2.Cross(e3)) * inv;
    const float v = e1.Dot(r.Cross(e3)) * inv;
    const float w = e1.Dot(e2.Cross(r)) * inv;
    bary[0] = 1.0f - u - v - w;
    bary[1] = u; bary[2] = v; bary[3] = w;
    return true;
}

}  // namespace

TriMesh BuildIcosphere(const math::Vec3& center, float radius, uint32_t subdivisions) {
    // 12 icosahedron vertices (golden-ratio rectangles), normalized to the sphere.
    const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
    std::vector<math::Vec3> verts = {
        {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
        {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    for (math::Vec3& v : verts) v = v.Normalized();
    std::vector<uint32_t> tris = {
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
        1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
        3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
        4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1};

    // Each pass splits every triangle into 4 by its edge midpoints (projected to
    // the sphere); a midpoint cache keeps shared edges welded.
    for (uint32_t s = 0; s < subdivisions; ++s) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoint;
        auto mid = [&](uint32_t a, uint32_t b) -> uint32_t {
            auto key = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
            auto it = midpoint.find(key);
            if (it != midpoint.end()) return it->second;
            const math::Vec3 m = ((verts[a] + verts[b]) * 0.5f).Normalized();
            const uint32_t idx = static_cast<uint32_t>(verts.size());
            verts.push_back(m);
            midpoint.emplace(key, idx);
            return idx;
        };
        std::vector<uint32_t> next;
        next.reserve(tris.size() * 4u);
        for (size_t i = 0; i + 2 < tris.size(); i += 3) {
            const uint32_t a = tris[i], b = tris[i + 1], c = tris[i + 2];
            const uint32_t ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
            next.insert(next.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
        }
        tris = std::move(next);
    }

    TriMesh m;
    m.positions.reserve(verts.size());
    for (const math::Vec3& v : verts) m.positions.push_back(center + v * radius);
    m.triangles = std::move(tris);
    return m;
}

EmbeddedSkin EmbedSurfaceInTetCage(const std::vector<math::Vec3>& skin_rest,
                                   const std::vector<uint32_t>& skin_tris,
                                   const std::vector<math::Vec3>& tet_rest,
                                   const std::vector<TetMeshTet>& tets,
                                   uint32_t* out_extrapolated) {
    // An empty cage cannot bind a skin vertex; fail loudly, never index tets[0].
    if (!skin_rest.empty() && (tets.empty() || tet_rest.empty())) {
        throw std::invalid_argument(
            "EmbedSurfaceInTetCage: empty tet cage for a non-empty skin");
    }
    EmbeddedSkin e;
    e.triangles = skin_tris;
    e.binds.resize(skin_rest.size());
    uint32_t extrapolated = 0u;
    constexpr float kContainEps = 1e-5f;

    for (size_t s = 0; s < skin_rest.size(); ++s) {
        const math::Vec3& p = skin_rest[s];
        bool contained = false;
        float best_outside = 1e30f, best_bary[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        uint32_t best_tet = 0u, bary_tet = 0u;
        for (size_t ti = 0; ti < tets.size(); ++ti) {
            const TetMeshTet& tet = tets[ti];
            float bary[4];
            if (!TetBarycentric(p, tet_rest[tet.v[0]], tet_rest[tet.v[1]],
                                tet_rest[tet.v[2]], tet_rest[tet.v[3]], bary))
                continue;
            const float outside = std::max(0.0f, -bary[0]) + std::max(0.0f, -bary[1]) +
                                  std::max(0.0f, -bary[2]) + std::max(0.0f, -bary[3]);
            if (bary[0] >= -kContainEps && bary[1] >= -kContainEps &&
                bary[2] >= -kContainEps && bary[3] >= -kContainEps) {
                contained = true;
                for (int j = 0; j < 4; ++j) best_bary[j] = bary[j];
                best_tet = static_cast<uint32_t>(ti);
                break;
            }
            if (outside < best_outside) {
                best_outside = outside;
                for (int j = 0; j < 4; ++j) best_bary[j] = bary[j];
                bary_tet = static_cast<uint32_t>(ti);
            }
        }
        const uint32_t pick = contained ? best_tet : bary_tet;
        if (!contained) ++extrapolated;
        SkinVertexBind& b = e.binds[s];
        for (int j = 0; j < 4; ++j) {
            b.vi[j] = tets[pick].v[j];
            b.w[j] = best_bary[j];
        }
    }
    if (out_extrapolated) *out_extrapolated = extrapolated;
    return e;
}

EmbeddedSkin EmbedSurfaceInParticleSet(const std::vector<math::Vec3>& skin_rest,
                                       const std::vector<uint32_t>& skin_tris,
                                       const std::vector<math::Vec3>& particle_rest) {
    EmbeddedSkin e;
    e.triangles = skin_tris;
    e.binds.resize(skin_rest.size());
    constexpr uint32_t K = 4u;
    for (size_t s = 0; s < skin_rest.size(); ++s) {
        const math::Vec3& p = skin_rest[s];
        // Smallest-K nearest particles by squared distance (insertion into a K slot).
        uint32_t idx[K] = {0u, 0u, 0u, 0u};
        float d2[K] = {1e30f, 1e30f, 1e30f, 1e30f};
        for (size_t q = 0; q < particle_rest.size(); ++q) {
            const float dq = (particle_rest[q] - p).LengthSq();
            int slot = -1;
            for (int k = 0; k < static_cast<int>(K); ++k)
                if (dq < d2[k]) { slot = k; break; }
            if (slot < 0) continue;
            for (int k = static_cast<int>(K) - 1; k > slot; --k) {
                d2[k] = d2[k - 1]; idx[k] = idx[k - 1];
            }
            d2[slot] = dq; idx[slot] = static_cast<uint32_t>(q);
        }
        // Inverse-distance weights normalized to Σ == 1 (a coincident particle pins
        // the vertex to it); the blend is affine over the nearest-K particle cloud.
        float w[K], sum = 0.0f;
        for (int k = 0; k < static_cast<int>(K); ++k) {
            const float dist = std::sqrt(std::max(d2[k], 0.0f));
            w[k] = 1.0f / (dist + 1e-4f);
            sum += w[k];
        }
        const float inv = sum > 1e-12f ? 1.0f / sum : 0.0f;
        SkinVertexBind& b = e.binds[s];
        for (int k = 0; k < static_cast<int>(K); ++k) {
            b.vi[k] = idx[k];
            b.w[k] = w[k] * inv;
        }
    }
    return e;
}

void DeformEmbeddedSkin(const EmbeddedSkin& e,
                        const std::vector<math::Vec3>& tet_live,
                        std::vector<math::Vec3>& out) {
    out.resize(e.binds.size());
    for (size_t v = 0; v < e.binds.size(); ++v) {
        const SkinVertexBind& b = e.binds[v];
        out[v] = tet_live[b.vi[0]] * b.w[0] + tet_live[b.vi[1]] * b.w[1] +
                 tet_live[b.vi[2]] * b.w[2] + tet_live[b.vi[3]] * b.w[3];
    }
}

void SmoothSurface(const std::vector<uint32_t>& triangles, uint32_t iterations,
                   float lambda, std::vector<math::Vec3>& positions) {
    if (iterations == 0u || lambda <= 0.0f || positions.empty()) return;
    // Unique undirected edge neighbours per vertex (built once from the triangles).
    std::vector<std::set<uint32_t>> adj(positions.size());
    auto link = [&](uint32_t a, uint32_t b) { adj[a].insert(b); adj[b].insert(a); };
    for (size_t t = 0; t + 2 < triangles.size(); t += 3) {
        link(triangles[t], triangles[t + 1]);
        link(triangles[t + 1], triangles[t + 2]);
        link(triangles[t + 2], triangles[t]);
    }
    std::vector<math::Vec3> next(positions.size());
    for (uint32_t it = 0; it < iterations; ++it) {
        for (size_t v = 0; v < positions.size(); ++v) {
            if (adj[v].empty()) { next[v] = positions[v]; continue; }
            math::Vec3 c{0.0f, 0.0f, 0.0f};
            for (uint32_t n : adj[v]) c += positions[n];
            c = c * (1.0f / static_cast<float>(adj[v].size()));
            next[v] = positions[v] + (c - positions[v]) * lambda;
        }
        positions.swap(next);
    }
}

} // namespace nuka::runtime::soft
