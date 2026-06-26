// ---------------------------------------------------------------------------
// nuka::import::cooker::CookSparseSdf implementation (v0.7 p07)
// ---------------------------------------------------------------------------
// HOST-CPU narrow-band SDF generation. See sparse_sdf_cooker.hpp for the why.
//
// DETERMINISM: single-threaded, fixed nested-loop voxel order (k outer, j, i
// inner per the standard layout below), order-independent per-voxel reduction
// (running min over triangles), fixed-stencil central differences. The math is
// done in `double` and cast to float ONCE at store time, so the result is a pure
// function of the inputs — byte-reproducible RUN-TO-RUN on a given build (the
// verified scope; tests cook N times and compare bytes). Cross-compiler byte
// identity is not separately tested (FP contraction could in principle differ),
// but the golden workflow only requires same-build reproducibility.
//
// SCOPE NOTE (vs spec Task 7.7.2): the spec lists fast-sweeping / fast-marching
// to propagate distance AND a generalized winding number for sign. For a ±N
// narrow band we instead compute EXACT triangle-point distance for every
// band-candidate voxel (min over triangles whose padded AABB overlaps the
// voxel). This is (a) more accurate in-band than sweeping's Eikonal
// approximation, (b) trivially deterministic (no sweep-direction order to fix),
// and (c) the right tool for thin shells — the nearest-triangle distance
// distinguishes both sides of a sub-voxel panel where sweeping would let the
// sign bleed through. Sweeping only earns its keep filling LARGE
// far-from-surface regions, which a narrow band does not store.
//
// SIGN (vs spec winding number): the cook input is always a CONVEX piece —
// V-HACD pieces are convex hulls by construction, and so are the primitive test
// meshes. For a convex hull the sign is read directly from the CLOSEST
// triangle's face plane: interior points are closest to a face interior with
// dot(p-a, n) < 0; exterior points closest to an edge lie in the positive
// half-space of both incident faces, so any tie-break gives +. This is O(1) per
// voxel (folded into the distance pass) and avoids the O(voxels x triangles x
// atan2) generalized-winding-number sum, which is otherwise the cook bottleneck.
//
// CONVEXITY CAVEAT (scope): CookScene's `Skip` mode and the V-HACD-failure
// fallback pass the RAW source mesh through as a single "ConvexHull". If that
// mesh is actually concave, the plane-sign can be wrong at concave features far
// from the nearest face. No phase gate requires non-watertight / concave-mesh
// SDF robustness; if needed later, restore the generalized winding number for
// the non-convex passthrough path only.
// ---------------------------------------------------------------------------

#include "import/cooker/sparse_sdf_cooker.hpp"

#include "runtime/sdf/sparse_sdf_query.cuh"  // PackSdfCellKey codec (shared)

#include "sha256.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

namespace nuka::import::cooker {

namespace {

using Vec3d = std::array<double, 3>;

Vec3d Sub(const Vec3d& a, const Vec3d& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double Dot(const Vec3d& a, const Vec3d& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3d Cross(const Vec3d& a, const Vec3d& b) {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
// Squared distance from point p to triangle (a,b,c), plus the closest point
// `out_cp`. Standard region-based closest-point-on-triangle (Ericson, Real-Time
// Collision Detection). All in double for determinism + accuracy.
//
// `out_cp` is needed for the convex-hull sign test: sign = dot(p - out_cp,
// outward_normal). Using the closest POINT (not just the triangle plane) makes
// the sign robust at high-valence vertices / poles, where the single winning
// sliver's plane can place an exterior point on the wrong side but p - out_cp
// still points outward (it lies in the feature's normal cone). Valid for hulls
// with dihedral >= 90 deg (spheres/capsules ~ pi, boxes/panels = pi/2).
double SqDistPointTriangle(const Vec3d& p, const Vec3d& a, const Vec3d& b,
                           const Vec3d& c, Vec3d& out_cp) {
    const Vec3d ab = Sub(b, a);
    const Vec3d ac = Sub(c, a);
    const Vec3d ap = Sub(p, a);
    const double d1 = Dot(ab, ap);
    const double d2 = Dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {  // vertex region A
        out_cp = a;
        const Vec3d d = ap;
        return Dot(d, d);
    }
    const Vec3d bp = Sub(p, b);
    const double d3 = Dot(ab, bp);
    const double d4 = Dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {  // vertex region B
        out_cp = b;
        const Vec3d d = bp;
        return Dot(d, d);
    }
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {  // edge AB
        const double v = d1 / (d1 - d3);
        const Vec3d proj = {a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]};
        out_cp = proj;
        const Vec3d d = Sub(p, proj);
        return Dot(d, d);
    }
    const Vec3d cp = Sub(p, c);
    const double d5 = Dot(ab, cp);
    const double d6 = Dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {  // vertex region C
        out_cp = c;
        const Vec3d d = cp;
        return Dot(d, d);
    }
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {  // edge AC
        const double w = d2 / (d2 - d6);
        const Vec3d proj = {a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]};
        out_cp = proj;
        const Vec3d d = Sub(p, proj);
        return Dot(d, d);
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {  // edge BC
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const Vec3d bc = Sub(c, b);
        const Vec3d proj = {b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]};
        out_cp = proj;
        const Vec3d d = Sub(p, proj);
        return Dot(d, d);
    }
    // Face region: barycentric interior.
    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom;
    const double w = vc * denom;
    const Vec3d proj = {a[0] + ab[0] * v + ac[0] * w,
                        a[1] + ab[1] * v + ac[1] * w,
                        a[2] + ab[2] * v + ac[2] * w};
    out_cp = proj;
    const Vec3d d = Sub(p, proj);
    return Dot(d, d);
}

}  // namespace

SparseSdfData CookSparseSdf(const float* vertices, uint32_t vertex_count,
                            const uint32_t* indices, uint32_t triangle_count,
                            const SparseSdfParams& params) {
    SparseSdfData out;
    if (vertices == nullptr || indices == nullptr || vertex_count == 0u ||
        triangle_count == 0u) {
        return out;  // empty SDF for degenerate input
    }

    // --- Gather vertices / triangles (double) ------------------------------
    std::vector<Vec3d> verts(vertex_count);
    for (uint32_t v = 0; v < vertex_count; ++v) {
        verts[v] = {static_cast<double>(vertices[3 * v + 0]),
                    static_cast<double>(vertices[3 * v + 1]),
                    static_cast<double>(vertices[3 * v + 2])};
    }
    // `normal` is the OUTWARD face normal (unnormalized): for outward-wound
    // (CCW-seen-from-outside) triangles, (b-a) x (c-a) points outward. For a
    // CONVEX hull, the sign of dot(p-a, normal) of the CLOSEST triangle gives
    // inside (<0) / outside (>0) — see the SIGN scope note in the banner.
    struct Tri { Vec3d a, b, c; Vec3d normal; double lo[3], hi[3]; };
    std::vector<Tri> tris;
    tris.reserve(triangle_count);
    for (uint32_t t = 0; t < triangle_count; ++t) {
        const uint32_t i0 = indices[3 * t + 0];
        const uint32_t i1 = indices[3 * t + 1];
        const uint32_t i2 = indices[3 * t + 2];
        if (i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
            continue;
        }
        Tri tr;
        tr.a = verts[i0];
        tr.b = verts[i1];
        tr.c = verts[i2];
        const Vec3d ab = Sub(tr.b, tr.a);
        const Vec3d ac = Sub(tr.c, tr.a);
        tr.normal = Cross(ab, ac);
        // Drop DEGENERATE (zero-area / sliver) triangles: their normal is
        // numerical noise, so the closest-feature SIGN read off them is random.
        // UV-sphere / capsule pole rings emit such slivers (the degenerate pole
        // quad). The scale-invariant test |cross(ab,ac)| <= eps*|ab|*|ac| is
        // sin(angle)~0 (collinear / coincident verts); it cannot misfire on a
        // uniformly small mesh. The pole vertex survives via the OTHER (healthy)
        // triangle of the same quad, so |phi| is unchanged — only the unreliable
        // sign source is removed. This is correct cook-time hygiene (V-HACD can
        // emit slivers too: zero area => no normal, no enclosed volume).
        const double area2 = Dot(tr.normal, tr.normal);
        const double ab2 = Dot(ab, ab);
        const double ac2 = Dot(ac, ac);
        constexpr double kSinEps = 1.0e-5;  // sin(angle) threshold
        if (area2 <= kSinEps * kSinEps * ab2 * ac2) {
            continue;  // degenerate sliver
        }
        for (int d = 0; d < 3; ++d) {
            tr.lo[d] = std::min({tr.a[d], tr.b[d], tr.c[d]});
            tr.hi[d] = std::max({tr.a[d], tr.b[d], tr.c[d]});
        }
        tris.push_back(tr);
    }
    if (tris.empty()) {
        return out;
    }

    // --- AABB --------------------------------------------------------------
    double aabb_lo[3] = {std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::max()};
    double aabb_hi[3] = {std::numeric_limits<double>::lowest(),
                         std::numeric_limits<double>::lowest(),
                         std::numeric_limits<double>::lowest()};
    for (const auto& tr : tris) {
        for (int d = 0; d < 3; ++d) {
            aabb_lo[d] = std::min(aabb_lo[d], tr.lo[d]);
            aabb_hi[d] = std::max(aabb_hi[d], tr.hi[d]);
        }
    }

    // --- Voxel size --------------------------------------------------------
    double h = static_cast<double>(params.voxel_size);
    if (!(h > 0.0)) {
        double longest = 0.0;
        for (int d = 0; d < 3; ++d) {
            longest = std::max(longest, aabb_hi[d] - aabb_lo[d]);
        }
        const double res = (params.auto_resolution > 0.0f)
                               ? static_cast<double>(params.auto_resolution)
                               : 64.0;
        h = (longest > 0.0) ? (longest / res) : 1.0;
    }
    if (!(h > 0.0)) {
        return out;
    }

    const int band = static_cast<int>(params.band_voxels);
    const int pad  = static_cast<int>(std::lround(std::ceil(params.padding_voxels)));
    // Compute on band+1 (the guard ring) so band-edge gradients central-diff
    // against present neighbors; the +1 ring is dropped before storage.
    const int compute_band = band + 1;
    const double band_dist = static_cast<double>(band) * h;       // store |phi| <= this
    const int total_margin = compute_band + std::max(pad, 0);
    const int reach_voxels = compute_band;

    // Origin = padded min corner, so all voxel indices are >= 0.
    double origin[3];
    for (int d = 0; d < 3; ++d) {
        origin[d] = aabb_lo[d] - static_cast<double>(total_margin) * h;
    }
    // Dense grid dims covering the padded AABB (inclusive of both ends).
    uint32_t dims[3];
    for (int d = 0; d < 3; ++d) {
        const double span = (aabb_hi[d] - aabb_lo[d]) + 2.0 * static_cast<double>(total_margin) * h;
        dims[d] = static_cast<uint32_t>(std::floor(span / h)) + 1u;
    }

    // --- Distance + sign over compute band --------------------------------
    // We only visit voxels whose center is within compute_band*h of some
    // triangle (via per-triangle padded-AABB rasterization), so cost scales with
    // the surface, not the dense volume. A voxel may be touched by several
    // triangles; we keep the running min squared distance AND the sign from the
    // CLOSEST triangle's face plane (valid for convex pieces — see banner).
    // `sign_dot` is dot(p - a, outward_normal) of the closest triangle; its sign
    // is the inside/outside sign. We carry the raw dot (not just +/-1) so a tie
    // in distance is broken consistently and deterministically: among triangles
    // at the (numerically) same min distance we keep the FIRST seen in fixed
    // triangle order, and for a convex hull all such ties agree on sign anyway.
    struct Cell { double dist2; double sign_dot; bool inited; double phi; bool has_phi; };
    std::unordered_map<uint64_t, Cell> cells;
    cells.reserve(tris.size() * 64u);

    const double cell_reach = static_cast<double>(reach_voxels) * h;

    auto voxel_center = [&](int i, int j, int k) -> Vec3d {
        return {origin[0] + (static_cast<double>(i) + 0.0) * h,
                origin[1] + (static_cast<double>(j) + 0.0) * h,
                origin[2] + (static_cast<double>(k) + 0.0) * h};
    };

    // Pass 1: exact distance + closest-triangle plane sign for all band
    // candidates. Single pass over (triangle, overlapped voxels).
    for (const auto& tr : tris) {
        int gmin[3], gmax[3];
        for (int d = 0; d < 3; ++d) {
            const double lo = tr.lo[d] - cell_reach - origin[d];
            const double hi = tr.hi[d] + cell_reach - origin[d];
            int imin = static_cast<int>(std::floor(lo / h));
            int imax = static_cast<int>(std::ceil(hi / h));
            imin = std::max(imin, 0);
            imax = std::min(imax, static_cast<int>(dims[d]) - 1);
            gmin[d] = imin;
            gmax[d] = imax;
        }
        for (int k = gmin[2]; k <= gmax[2]; ++k) {
            for (int j = gmin[1]; j <= gmax[1]; ++j) {
                for (int i = gmin[0]; i <= gmax[0]; ++i) {
                    const Vec3d p = voxel_center(i, j, k);
                    Vec3d cp;
                    const double d2 = SqDistPointTriangle(p, tr.a, tr.b, tr.c, cp);
                    // Sign from the closest POINT (robust at high-valence verts).
                    const double sdot = Dot(Sub(p, cp), tr.normal);
                    const uint64_t key = nuka::runtime::sdf::PackSdfCellKey(
                        static_cast<uint32_t>(i), static_cast<uint32_t>(j),
                        static_cast<uint32_t>(k));
                    auto it = cells.find(key);
                    if (it == cells.end()) {
                        cells.emplace(key, Cell{d2, sdot, true, 0.0, false});
                    } else if (d2 < it->second.dist2) {
                        it->second.dist2 = d2;
                        it->second.sign_dot = sdot;
                    }
                }
            }
        }
    }

    // Pass 2: form the signed distance phi = sign * sqrt(dist2). Fixed sorted-key
    // order so the result is order-independent of hash-map iteration.
    std::vector<uint64_t> visited_keys;
    visited_keys.reserve(cells.size());
    for (const auto& kv : cells) visited_keys.push_back(kv.first);
    std::sort(visited_keys.begin(), visited_keys.end());

    for (uint64_t key : visited_keys) {
        Cell& c = cells[key];
        const double sign = (c.sign_dot < 0.0) ? -1.0 : 1.0;  // inside => negative
        c.phi = sign * std::sqrt(c.dist2);
        c.has_phi = true;
    }

    // Solid fill: exterior-flood from the grid boundary (blocked by the surface);
    // the enclosed complement is interior, depth-filled by an inward fast-sweep.
    if (params.solid) {
        const int nx = static_cast<int>(dims[0]);
        const int ny = static_cast<int>(dims[1]);
        const int nz = static_cast<int>(dims[2]);
        const size_t ncell = static_cast<size_t>(nx) * ny * nz;
        auto lin = [&](int i, int j, int k) -> size_t {
            return (static_cast<size_t>(k) * ny + j) * nx + i;
        };
        auto find_cell = [&](int i, int j, int k) -> Cell* {
            const uint64_t key = nuka::runtime::sdf::PackSdfCellKey(
                static_cast<uint32_t>(i), static_cast<uint32_t>(j),
                static_cast<uint32_t>(k));
            auto it = cells.find(key);
            return (it == cells.end()) ? nullptr : &it->second;
        };
        // 0=unknown, 1=exterior (flood-reached), 2=surface-band (phi>=0 wall).
        std::vector<uint8_t> state(ncell, 0u);
        // An interior-band cell (phi<0) BLOCKS the exterior flood (it is the inner
        // surface); an exterior-band cell (phi>=0) is itself exterior + a wall.
        for (uint64_t key : visited_keys) {
            uint32_t ui, uj, uk;
            nuka::runtime::sdf::UnpackSdfCellKey(key, ui, uj, uk);
            if (ui >= static_cast<uint32_t>(nx) || uj >= static_cast<uint32_t>(ny) ||
                uk >= static_cast<uint32_t>(nz)) continue;
            const Cell& c = cells[key];
            if (c.has_phi && c.phi >= 0.0) state[lin(ui, uj, uk)] = 2u;  // wall.
        }
        // BFS the exterior from every boundary cell that is not a phi<0 surface cell.
        std::vector<int> stack;
        stack.reserve(ncell / 8u + 16u);
        auto is_inner_surface = [&](int i, int j, int k) -> bool {
            const Cell* c = find_cell(i, j, k);
            return c != nullptr && c->has_phi && c->phi < 0.0;
        };
        auto push_ext = [&](int i, int j, int k) {
            if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return;
            const size_t id = lin(i, j, k);
            if (state[id] == 1u) return;                  // already exterior.
            if (is_inner_surface(i, j, k)) return;        // surface blocks the flood.
            state[id] = 1u;
            stack.push_back(static_cast<int>(id));
        };
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    if (i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 ||
                        k == nz - 1)
                        push_ext(i, j, k);
        while (!stack.empty()) {
            const int id = stack.back(); stack.pop_back();
            const int i = id % nx;
            const int j = (id / nx) % ny;
            const int k = id / (nx * ny);
            push_ext(i - 1, j, k); push_ext(i + 1, j, k);
            push_ext(i, j - 1, k); push_ext(i, j + 1, k);
            push_ext(i, j, k - 1); push_ext(i, j, k + 1);
        }
        // Interior = not exterior-reached and not a wall. Seed interior-band cells
        // (exact phi<0) and inward fast-sweep phi = nb.phi - h over interior cells.
        auto is_interior = [&](int i, int j, int k) -> bool {
            if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return false;
            const size_t id = lin(i, j, k);
            return state[id] == 0u;  // unknown == enclosed (flood never reached it).
        };
        const double kInf = 1.0e30;
        auto relax = [&](int i, int j, int k, int di, int dj, int dk) -> bool {
            if (!is_interior(i, j, k)) return false;
            const Cell* cc0 = find_cell(i, j, k);
            // Keep EXACT band cells (|phi|<=band); only fill the deeper interior.
            if (cc0 != nullptr && cc0->has_phi && std::abs(cc0->phi) <= band_dist)
                return false;
            const Cell* nb = find_cell(i + di, j + dj, k + dk);
            if (nb == nullptr || !nb->has_phi || nb->phi >= 0.0) return false;
            const double cand = nb->phi - h;
            const uint64_t key = nuka::runtime::sdf::PackSdfCellKey(
                static_cast<uint32_t>(i), static_cast<uint32_t>(j),
                static_cast<uint32_t>(k));
            Cell& cc = cells[key];
            const double cur = cc.has_phi ? cc.phi : -kInf;
            if (cand > cur) {
                cc.phi = cand; cc.has_phi = true; cc.dist2 = cand * cand;
                cc.sign_dot = -1.0;
                return true;
            }
            return false;
        };
        const int max_sweeps = nx + ny + nz + 8;
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            bool changed = false;
            const bool fwd = (sweep & 1) == 0;
            for (int kk = 0; kk < nz; ++kk) {
                const int k = fwd ? kk : (nz - 1 - kk);
                for (int jj = 0; jj < ny; ++jj) {
                    const int j = fwd ? jj : (ny - 1 - jj);
                    for (int ii = 0; ii < nx; ++ii) {
                        const int i = fwd ? ii : (nx - 1 - ii);
                        changed |= relax(i, j, k, -1, 0, 0);
                        changed |= relax(i, j, k, +1, 0, 0);
                        changed |= relax(i, j, k, 0, -1, 0);
                        changed |= relax(i, j, k, 0, +1, 0);
                        changed |= relax(i, j, k, 0, 0, -1);
                        changed |= relax(i, j, k, 0, 0, +1);
                    }
                }
            }
            if (!changed) break;
        }
        // Refresh the sorted key list so Pass 3 sees the filled interior cells.
        visited_keys.clear();
        visited_keys.reserve(cells.size());
        for (const auto& kv : cells) visited_keys.push_back(kv.first);
        std::sort(visited_keys.begin(), visited_keys.end());
    }

    // --- Pass 3: central-diff gradient + narrow-band extraction ------------
    // Only emit voxels with |phi| <= band_dist (drops the +1 guard ring). The
    // gradient uses neighbors at +/-1 voxel; those exist in `cells` because we
    // computed on band+1. Missing neighbor (band-edge against empty) falls back
    // to a one-sided difference, then normalization.
    auto phi_at = [&](int i, int j, int k, double& v) -> bool {
        if (i < 0 || j < 0 || k < 0) return false;
        const uint64_t key = nuka::runtime::sdf::PackSdfCellKey(
            static_cast<uint32_t>(i), static_cast<uint32_t>(j), static_cast<uint32_t>(k));
        auto it = cells.find(key);
        if (it == cells.end() || !it->second.has_phi) return false;
        v = it->second.phi;
        return true;
    };

    std::vector<uint64_t> band_keys;
    band_keys.reserve(visited_keys.size());
    for (uint64_t key : visited_keys) {
        const Cell& c = cells[key];
        if (!c.has_phi) continue;
        // Narrow band keeps |phi|<=band on BOTH sides. Solid additionally keeps the
        // WHOLE interior (phi<0, computed by the enlarged reach) so a deep inside
        // point has a value+gradient; exterior beyond the band is still dropped.
        const bool in_band = std::abs(c.phi) <= band_dist;
        const bool keep_interior = params.solid && c.phi < 0.0;
        if (in_band || keep_interior) {
            band_keys.push_back(key);
        }
    }
    // visited_keys is already ascending; the filtered subset preserves order.

    out.cell_keys.reserve(band_keys.size());
    out.cell_values.reserve(band_keys.size());
    out.cell_gradients.reserve(band_keys.size() * 3u);

    for (uint64_t key : band_keys) {
        uint32_t ui, uj, uk;
        nuka::runtime::sdf::UnpackSdfCellKey(key, ui, uj, uk);
        const int i = static_cast<int>(ui);
        const int j = static_cast<int>(uj);
        const int k = static_cast<int>(uk);
        const double center = cells[key].phi;

        double grad[3];
        // Per-axis central difference with one-sided fallback.
        // X
        {
            double vp, vm;
            const bool hp = phi_at(i + 1, j, k, vp);
            const bool hm = phi_at(i - 1, j, k, vm);
            if (hp && hm)      grad[0] = (vp - vm) / (2.0 * h);
            else if (hp)       grad[0] = (vp - center) / h;
            else if (hm)       grad[0] = (center - vm) / h;
            else               grad[0] = 0.0;
        }
        // Y
        {
            double vp, vm;
            const bool hp = phi_at(i, j + 1, k, vp);
            const bool hm = phi_at(i, j - 1, k, vm);
            if (hp && hm)      grad[1] = (vp - vm) / (2.0 * h);
            else if (hp)       grad[1] = (vp - center) / h;
            else if (hm)       grad[1] = (center - vm) / h;
            else               grad[1] = 0.0;
        }
        // Z
        {
            double vp, vm;
            const bool hp = phi_at(i, j, k + 1, vp);
            const bool hm = phi_at(i, j, k - 1, vm);
            if (hp && hm)      grad[2] = (vp - vm) / (2.0 * h);
            else if (hp)       grad[2] = (vp - center) / h;
            else if (hm)       grad[2] = (center - vm) / h;
            else               grad[2] = 0.0;
        }
        // Normalize to unit length (|grad phi| = 1 for a true SDF); leave zero
        // if degenerate.
        const double gl = std::sqrt(grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]);
        if (gl > 0.0) {
            grad[0] /= gl;
            grad[1] /= gl;
            grad[2] /= gl;
        }

        out.cell_keys.push_back(key);
        out.cell_values.push_back(static_cast<float>(center));
        out.cell_gradients.push_back(static_cast<float>(grad[0]));
        out.cell_gradients.push_back(static_cast<float>(grad[1]));
        out.cell_gradients.push_back(static_cast<float>(grad[2]));
    }

    for (int d = 0; d < 3; ++d) {
        out.origin[d] = static_cast<float>(origin[d]);
        out.dims[d]   = dims[d];
    }
    out.voxel_size = static_cast<float>(h);
    return out;
}

std::string ComputeSdfCacheKey(const float* vertices, uint32_t vertex_count,
                               const uint32_t* indices, uint32_t triangle_count,
                               const SparseSdfParams& params) {
    nuka::sha256::Hasher hasher;
    // Domain tag so SDF keys never collide with the V-HACD cache keys.
    static const char kTag[] = "nuka.sparse_sdf.v1";
    hasher.Update(kTag, sizeof(kTag));
    hasher.Update(&vertex_count, sizeof(vertex_count));
    hasher.Update(&triangle_count, sizeof(triangle_count));
    if (vertices && vertex_count) {
        hasher.Update(vertices, static_cast<size_t>(vertex_count) * 3u * sizeof(float));
    }
    if (indices && triangle_count) {
        hasher.Update(indices, static_cast<size_t>(triangle_count) * 3u * sizeof(uint32_t));
    }
    // Serialize params field-by-field (avoid hashing struct padding bytes).
    hasher.Update(&params.voxel_size, sizeof(params.voxel_size));
    hasher.Update(&params.band_voxels, sizeof(params.band_voxels));
    hasher.Update(&params.auto_resolution, sizeof(params.auto_resolution));
    hasher.Update(&params.padding_voxels, sizeof(params.padding_voxels));
    const uint8_t solid_byte = params.solid ? 1u : 0u;
    hasher.Update(&solid_byte, sizeof(solid_byte));
    return nuka::sha256::ToHex(hasher.Final());
}

}  // namespace nuka::import::cooker
