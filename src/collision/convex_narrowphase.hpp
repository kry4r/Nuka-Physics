#pragma once
// ---------------------------------------------------------------------------
// nuka::collision -- convex narrowphase (GJK / EPA / face-clip), v0.8 C3c
// ---------------------------------------------------------------------------
// The Convex-tier geometry math for the C3a dispatch table: general
// convex x convex AND convex x primitive contact (Q2 hybrid narrowphase).
// Self-written GJK (Minkowski-difference support) -> EPA (penetration depth +
// normal) -> face-clip (supporting-vertex-set faces, Sutherland-Hodgman, <=4
// via the SHARED C3b reducer). NO external collision lib, NO MPR (deterministic;
// MPR's origin-ray pick is data-dependent).
//
//   support(hull, dir)  = the hull vertex maximizing dot(v, dir); FIXED-ORDER
//                         linear scan; tie-break = LOWEST vertex index (D1).
//   support_minkowski(d)= support_A(d) - support_B(-d).
//   GJK                 = boolean intersect + closest simplex (point/line/tri/
//                         tetra), FIXED sub-simplex eval order, fixed iter cap.
//   EPA                 = expand the GJK tetrahedron; closest-face selection is a
//                         FIXED-ORDER ARRAY SCAN with an integer tie-break
//                         (closest distance, ties -> LOWEST face index). NO
//                         priority_queue / float-keyed heap (the named D1 trap).
//                         Edge-toggle horizon (no adjacency table), fixed-order
//                         new-face insertion, fixed iter cap.
//   face-clip           = supporting-vertex-set reference + incident faces,
//                         clipped via amf::ClipAgainstPlane, reduced <=4 via
//                         amf::ReduceAndEmitSpread (FPS deepest+spread).
//
// NORMAL CONVENTION (matches ContactPoint + row_builder.cpp:88 + C3b): the
// emitted point.normal is the SEPARATION DIRECTION FOR SIDE A of the manifold
// (the direction A must move to resolve the overlap). The table registers BOTH
// orderings of each asymmetric pair; the swapped wrapper FLIPS the normal (the
// C3b posture -- keeps the dispatch table a pure index->fn lookup).
//
// ---------------------------------------------------------------------------
// HD-CLEAN MATH (device-safe pattern, sdf_contact.hpp / analytical_manifold.hpp):
//   Authored __host__ __device__ (NUKA_CVX_HD) so the SAME geometry compiles for
//   the C5 device dispatch. Uses ONLY Vec3 HD-constexpr ops + sqrtf/atan2f and a
//   baked PrimFrame (never Vec3::Length / Vec3::Normalized / Quat::Rotate, which
//   are plain `inline` -- NOT HD -- on the production include path). All working
//   storage is FIXED-CAPACITY stack arrays (no std::vector) so the body is a
//   no-op to wrap in a __global__ kernel (C5). On capacity overflow we terminate
//   with the best-so-far result (deterministic, never UB).
// ---------------------------------------------------------------------------
// D1 (every tie integer-keyed, no data-dependent FP reordering):
//   - support:        fixed vertex scan, strict `>` improvement -> lowest index.
//   - GJK simplex:    fixed sub-simplex evaluation order; the simplex array is
//                     rebuilt in a FIXED order each iteration (no data-dependent
//                     permutation); fixed iteration cap.
//   - EPA closest face: FIXED-ORDER array scan; key = (distance, face index); on
//                     a distance tie the LOWEST face index wins (strict `<`
//                     improvement only). NEVER a float-keyed heap.
//   - EPA horizon:    edge-toggle in a fixed-capacity edge buffer scanned in face
//                     index order, each face's 3 edges in fixed (0,1,2) order;
//                     new faces appended in that fixed scan order.
//   - face-clip:      supporting-vertex set gathered by a fixed vertex scan;
//                     ordered CCW by angle-around-normal (a GEOMETRIC ordering of
//                     distinct convex-polygon verts, tie-break collinear by
//                     vertex index); clip + reduce reuse the C3b D1-safe utils.
//   - feature ids:    clip-output index (the C3b D1-safe unique survivor key).
// ---------------------------------------------------------------------------

#include "collision/analytical_manifold.hpp"  // amf:: clip/reduce/Len/Norm/Comp + PrimParams/PrimFrame
#include "constraint/contact_manifold.hpp"
#include "math/vec3.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_CVX_HD __host__ __device__
#else
#define NUKA_CVX_HD
#endif

namespace nuka::collision::cvx {

using constraint::ContactManifold;
using constraint::ContactPoint;
using math::Vec3;

// ---------------------------------------------------------------------------
// ConvexHullView -- the geom_a/geom_b SEAM payload for a hull side (C3a reserved
// the void* for exactly this). Trivially-copyable POD (no std::vector): C5
// uploads the flat cooked arrays (CookedConvexGeometry slice) and points this at
// them; the host test points it at backing arrays that outlive the call.
//   verts : flat x,y,z triples in WORLD space (the caller bakes the world
//           transform into the vertices, OR supplies a frame -- see `frame`).
//   To keep the support function a pure dot-scan with no per-vertex transform,
//   the caller may bake vertices to world; but cooked hull vertices are
//   MESH-LOCAL, so we carry a baked PrimFrame and transform on read (one
//   LocalToWorld per support call -- fixed work, HD-clean). `local` selects.
// ---------------------------------------------------------------------------
struct ConvexHullView {
    const float* verts = nullptr;   // flat x,y,z triples (count*3 floats)
    uint32_t     vcount = 0u;       // vertex count
    amf::PrimFrame frame;           // mesh-local -> world (baked, HD-clean)

    NUKA_CVX_HD Vec3 Vertex(uint32_t i) const {
        const Vec3 l{verts[i * 3u + 0u], verts[i * 3u + 1u], verts[i * 3u + 2u]};
        return frame.LocalToWorld(l);
    }
};

// ---------------------------------------------------------------------------
// SUPPORT FUNCTIONS (the Minkowski-difference building blocks).
// ---------------------------------------------------------------------------
// A support point carries its world position AND the source-vertex index on each
// body (for the witness / feature id). For analytic primitives the index is a
// synthetic feature code (we never face-clip a sphere; capsule packs endpoint).

// Hull support: FIXED-ORDER scan, strict `>` -> LOWEST index ties (D1).
NUKA_CVX_HD inline Vec3 SupportHull(const ConvexHullView& h, Vec3 dir,
                                    uint32_t* out_idx) {
    uint32_t best = 0u;
    Vec3 best_v = h.Vertex(0u);
    float best_d = best_v.Dot(dir);
    for (uint32_t i = 1u; i < h.vcount; ++i) {
        const Vec3 v = h.Vertex(i);
        const float d = v.Dot(dir);
        if (d > best_d) { best_d = d; best_v = v; best = i; }  // strict -> lowest idx
    }
    if (out_idx) *out_idx = best;
    return best_v;
}

// Box (as a primitive) support: center + sum_k sign(dot(axis_k,dir))*he_k*axis_k.
NUKA_CVX_HD inline Vec3 SupportBox(const amf::PrimParams& b, Vec3 dir) {
    Vec3 p = b.frame.t;
    for (int k = 0; k < 3; ++k) {
        const Vec3 axis = b.frame.Axis(k);
        const float he = amf::Comp(b.half_extents, k);
        p += axis * ((axis.Dot(dir) >= 0.0f ? he : -he));
    }
    return p;
}

// Sphere support: center + r * normalize(dir).
NUKA_CVX_HD inline Vec3 SupportSphere(const amf::PrimParams& s, Vec3 dir) {
    return s.frame.t + amf::Norm(dir, Vec3::UnitX()) * s.radius;
}

// Capsule support (axis = local Y): segment endpoint by sign(dir.axis) + r*norm(dir).
NUKA_CVX_HD inline Vec3 SupportCapsule(const amf::PrimParams& c, Vec3 dir) {
    const Vec3 axis = c.frame.cy;
    const Vec3 seg = c.frame.t + axis * ((axis.Dot(dir) >= 0.0f) ? c.half_height
                                                                 : -c.half_height);
    return seg + amf::Norm(dir, Vec3::UnitX()) * c.radius;
}

// ---------------------------------------------------------------------------
// SupportProxy -- a tagged union so ONE GJK/EPA path handles hull-vs-hull AND
// hull-vs-primitive. The handler fills two of these (one per side). No virtuals
// (device-clean) -- a kind tag + the two possible payloads (a view ptr / a prim
// ptr). The support() dispatch is a small fixed switch.
// ---------------------------------------------------------------------------
enum class SupportKind : uint8_t { Hull = 0, Box = 1, Sphere = 2, Capsule = 3 };

struct SupportProxy {
    SupportKind kind = SupportKind::Hull;
    const ConvexHullView*   hull = nullptr;   // kind == Hull
    const amf::PrimParams*  prim = nullptr;   // kind != Hull

    NUKA_CVX_HD Vec3 Support(Vec3 dir, uint32_t* out_idx) const {
        switch (kind) {
            case SupportKind::Hull:   return SupportHull(*hull, dir, out_idx);
            case SupportKind::Box:    if (out_idx) *out_idx = 0u; return SupportBox(*prim, dir);
            case SupportKind::Sphere: if (out_idx) *out_idx = 0u; return SupportSphere(*prim, dir);
            case SupportKind::Capsule:if (out_idx) *out_idx = 0u; return SupportCapsule(*prim, dir);
        }
        if (out_idx) *out_idx = 0u;
        return Vec3::Zero();
    }
    // Center-ish interior point (a deterministic seed for the first GJK dir).
    NUKA_CVX_HD Vec3 Center() const {
        if (kind == SupportKind::Hull) return hull->frame.t;
        return prim->frame.t;
    }
    // Does this side have a flat polygonal supporting face worth clipping?
    NUKA_CVX_HD bool HasFace() const {
        return kind == SupportKind::Hull || kind == SupportKind::Box;
    }
};

// Minkowski-difference support: support_A(d) - support_B(-d). Records the world
// support points on each body for the witness reconstruction.
struct MinkPoint {
    Vec3 v;       // A_support - B_support (the CSO vertex)
    Vec3 a;       // world support point on A
    Vec3 b;       // world support point on B
};
NUKA_CVX_HD inline MinkPoint SupportMink(const SupportProxy& A,
                                         const SupportProxy& B, Vec3 dir) {
    uint32_t ia = 0u, ib = 0u;
    const Vec3 pa = A.Support(dir, &ia);
    const Vec3 pb = B.Support(-dir, &ib);
    return {pa - pb, pa, pb};
}

// ---------------------------------------------------------------------------
// GJK -- boolean intersection (we only need the overlap case for contact; the
// separated case returns "no contact"). Simplex evaluated point->line->triangle
// ->tetrahedron in FIXED order. Returns the 4 CSO vertices of an enclosing
// simplex (for EPA) on intersection.
// ---------------------------------------------------------------------------
inline constexpr int   kGjkMaxIter = 32;
inline constexpr float kGjkEps     = 1.0e-9f;

struct GjkResult {
    bool      intersect = false;
    MinkPoint simplex[4];
    int       count = 0;
};

// Triple product (a x b) x c, HD.
NUKA_CVX_HD inline Vec3 TripleCross(Vec3 a, Vec3 b, Vec3 c) {
    return b * a.Dot(c) - a * b.Dot(c);  // (a x b) x c = b(a.c) - a(b.c)
}

// Evolve the simplex toward the origin. Updates `s`/`n` in place and writes the
// next search direction into `dir`. Returns true if the simplex (a tetrahedron)
// now ENCLOSES the origin. FIXED sub-simplex order (Casey Muratori's GJK, but
// with no data-dependent vertex shuffling beyond the fixed regions below).
NUKA_CVX_HD inline bool GjkDoSimplex(MinkPoint* s, int& n, Vec3& dir) {
    // a = the most-recently-added point (always s[n-1]).
    const Vec3 a = s[n - 1].v;
    const Vec3 ao = -a;  // toward origin from a
    if (n == 2) {
        const Vec3 b = s[0].v;
        const Vec3 ab = b - a;
        if (ab.Dot(ao) > 0.0f) {
            dir = TripleCross(ab, ao, ab);                 // perp to ab toward O
            if (dir.Dot(dir) < kGjkEps) {                  // O on the line: pick any perp
                dir = ab.Cross(Vec3::UnitX());
                if (dir.Dot(dir) < kGjkEps) dir = ab.Cross(Vec3::UnitY());
            }
        } else {
            s[0] = s[1]; n = 1; dir = ao;                  // keep a only
        }
        return false;
    }
    if (n == 3) {
        const Vec3 b = s[1].v;
        const Vec3 c = s[0].v;
        const Vec3 ab = b - a;
        const Vec3 ac = c - a;
        const Vec3 abc = ab.Cross(ac);                     // triangle normal
        // edge ab region?
        const Vec3 ab_perp = TripleCross(ac, ab, ab);      // perp to ab, away from c
        const Vec3 ac_perp = TripleCross(ab, ac, ac);      // perp to ac, away from b
        if (ab_perp.Dot(ao) > 0.0f) {
            // origin outside ab -> {a,b}
            s[0] = s[1]; s[1] = s[2]; n = 2;
            dir = TripleCross(ab, ao, ab);
            return false;
        }
        if (ac_perp.Dot(ao) > 0.0f) {
            // origin outside ac -> {a,c}
            s[1] = s[2]; n = 2;                            // s[0]=c stays, s[1]=a
            dir = TripleCross(ac, ao, ac);
            return false;
        }
        // origin above/below the triangle face
        const float side = abc.Dot(ao);
        if (side > 0.0f) {
            dir = abc;                                     // above
        } else {
            // below: flip winding (swap b,c) so the tetra test stays consistent
            MinkPoint tmp = s[0]; s[0] = s[1]; s[1] = tmp;
            dir = -abc;
        }
        return false;
    }
    // n == 4: tetrahedron a=s[3], b=s[2], c=s[1], d=s[0]
    const Vec3 b = s[2].v, c = s[1].v, d = s[0].v;
    const Vec3 ab = b - a, ac = c - a, ad = d - a;
    const Vec3 abc = ab.Cross(ac);
    const Vec3 acd = ac.Cross(ad);
    const Vec3 adb = ad.Cross(ab);
    // Each face normal oriented OUTWARD (away from the 4th vertex d/b/c resp.).
    // Fixed-order face tests; first outside face wins (deterministic).
    if (abc.Dot(ao) > 0.0f) {        // outside face abc -> drop d, simplex {a,b,c}
        s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; n = 3;  // {c,b,a}
        dir = abc;
        return false;
    }
    if (acd.Dot(ao) > 0.0f) {        // outside face acd -> drop b, simplex {a,c,d}
        s[2] = s[3]; n = 3;          // {d,c,a}  (s[0]=d, s[1]=c, s[2]=a)
        dir = acd;
        return false;
    }
    if (adb.Dot(ao) > 0.0f) {        // outside face adb -> drop c, simplex {a,d,b}
        s[1] = s[2]; s[2] = s[3]; n = 3;  // {d,b,a}
        dir = adb;
        return false;
    }
    return true;  // origin enclosed by the tetrahedron
}

NUKA_CVX_HD inline GjkResult GjkIntersect(const SupportProxy& A,
                                          const SupportProxy& B) {
    GjkResult res;
    // Initial direction: from B center to A center (deterministic, non-zero
    // fallback). The first simplex point is the support along it.
    Vec3 dir = A.Center() - B.Center();
    if (dir.Dot(dir) < kGjkEps) dir = Vec3::UnitX();
    MinkPoint s[4];
    s[0] = SupportMink(A, B, dir);
    int n = 1;
    dir = -s[0].v;  // toward origin
    for (int it = 0; it < kGjkMaxIter; ++it) {
        if (dir.Dot(dir) < kGjkEps) { dir = Vec3::UnitY(); }
        const MinkPoint p = SupportMink(A, B, dir);
        if (p.v.Dot(dir) < 0.0f) {
            return res;  // no support past origin -> no overlap (intersect=false)
        }
        // append p as the new 'a' (last slot)
        s[n] = p; ++n;
        if (GjkDoSimplex(s, n, dir)) {
            res.intersect = true;
            res.count = n;
            for (int i = 0; i < n; ++i) res.simplex[i] = s[i];
            return res;
        }
    }
    // Iteration cap hit without a clean enclosure: treat as touching (best
    // effort -> EPA promotion will robustify). Deterministic (fixed cap).
    res.intersect = (n == 4);
    res.count = n;
    for (int i = 0; i < n; ++i) res.simplex[i] = s[i];
    return res;
}

// ---------------------------------------------------------------------------
// EPA -- expanding polytope. Penetration depth + normal from the GJK tetra.
// ---------------------------------------------------------------------------
inline constexpr int   kEpaMaxIter   = 64;
inline constexpr int   kEpaMaxFaces  = 128;
inline constexpr int   kEpaMaxVerts  = 64;
inline constexpr int   kEpaMaxEdges  = 64;
inline constexpr float kEpaTol       = 1.0e-4f;

struct EpaFace {
    int  a, b, c;       // vertex indices into the vert pool
    Vec3 normal;        // OUTWARD unit normal
    float dist;         // distance from origin to the face plane (>=0)
    bool alive;
};

struct EpaResult {
    bool  ok = false;
    Vec3  normal{0.0f, 1.0f, 0.0f};  // separation normal (CSO surface -> outward)
    float depth = 0.0f;              // penetration depth (>=0)
    // witness reconstruction (barycentric of the closest face's support points):
    Vec3  witness_a{0.0f, 0.0f, 0.0f};
    Vec3  witness_b{0.0f, 0.0f, 0.0f};
};

// Build an outward-normal face from 3 CSO verts (origin must be on the inside).
NUKA_CVX_HD inline EpaFace MakeFace(const MinkPoint* verts, int a, int b, int c) {
    EpaFace f;
    f.a = a; f.b = b; f.c = c; f.alive = true;
    const Vec3 va = verts[a].v, vb = verts[b].v, vc = verts[c].v;
    Vec3 nrm = (vb - va).Cross(vc - va);
    const float nlen = amf::Len(nrm);
    if (nlen < 1.0e-12f) { f.normal = Vec3::UnitY(); f.dist = 0.0f; return f; }
    nrm = nrm / nlen;
    float d = nrm.Dot(va);          // signed distance origin->plane
    if (d < 0.0f) { nrm = -nrm; d = -d; }   // force OUTWARD (origin inside polytope)
    f.normal = nrm; f.dist = d;
    return f;
}

NUKA_CVX_HD inline EpaResult EpaExpand(const SupportProxy& A, const SupportProxy& B,
                                       const GjkResult& gjk) {
    EpaResult res;
    // --- 1) Promote the GJK simplex to a non-degenerate tetrahedron (D1). ---
    MinkPoint verts[kEpaMaxVerts];
    int nverts = 0;
    for (int i = 0; i < gjk.count && i < 4; ++i) verts[nverts++] = gjk.simplex[i];
    // Blow up to 4 along FIXED basis directions in fixed order if needed.
    const Vec3 basis[6] = {Vec3::UnitX(), {-1,0,0}, Vec3::UnitY(),
                           {0,-1,0}, Vec3::UnitZ(), {0,0,-1}};
    int bi = 0;
    while (nverts < 4 && bi < 6) {
        const MinkPoint p = SupportMink(A, B, basis[bi]); ++bi;
        // accept only if it adds non-degenerate volume / new direction
        bool dup = false;
        for (int i = 0; i < nverts; ++i) {
            if (amf::Len(p.v - verts[i].v) < 1.0e-6f) { dup = true; break; }
        }
        if (!dup) verts[nverts++] = p;
    }
    if (nverts < 4) { res.ok = false; return res; }

    // --- 2) Seed the polytope: 4 faces of the tetra, each forced OUTWARD. ---
    EpaFace faces[kEpaMaxFaces];
    int nfaces = 0;
    faces[nfaces++] = MakeFace(verts, 0, 1, 2);
    faces[nfaces++] = MakeFace(verts, 0, 2, 3);
    faces[nfaces++] = MakeFace(verts, 0, 3, 1);
    faces[nfaces++] = MakeFace(verts, 1, 3, 2);

    Vec3  best_n = Vec3::UnitY();
    float best_d = 0.0f;
    int   best_face = -1;

    for (int it = 0; it < kEpaMaxIter; ++it) {
        // --- 3) FIXED-ORDER ARRAY SCAN for the closest ALIVE face. ---
        // Key = (dist ASC, face index ASC). Strict `<` improvement only, so on a
        // distance tie the LOWEST face index wins. NO float-keyed heap (D1 trap).
        int   close = -1;
        float close_d = 3.4e38f;
        for (int i = 0; i < nfaces; ++i) {
            if (!faces[i].alive) continue;
            if (faces[i].dist < close_d) { close_d = faces[i].dist; close = i; }
        }
        if (close < 0) break;
        best_face = close; best_n = faces[close].normal; best_d = faces[close].dist;

        // --- 4) New support along the closest face's outward normal. ---
        const MinkPoint p = SupportMink(A, B, faces[close].normal);
        const float pd = p.v.Dot(faces[close].normal);
        if (pd - close_d < kEpaTol) {
            break;  // converged: support no further than the face -> face is the hull
        }
        if (nverts >= kEpaMaxVerts) break;  // overflow -> best-so-far (deterministic)
        const int new_vi = nverts;
        verts[nverts++] = p;

        // --- 5) Remove all faces VISIBLE from p; build the horizon by edge-toggle.
        // Edge buffer scanned in face index order, each face's 3 edges in fixed
        // (a-b, b-c, c-a) order; add if absent, cancel if present (shared interior
        // edges cancel -> only the silhouette survives). Deterministic.
        int edge_a[kEpaMaxEdges];
        int edge_b[kEpaMaxEdges];
        int nedges = 0;
        bool overflow = false;
        for (int i = 0; i < nfaces && !overflow; ++i) {
            if (!faces[i].alive) continue;
            // visible if p is on the OUTWARD side of the face plane.
            if (faces[i].normal.Dot(p.v) - faces[i].dist <= 0.0f) continue;
            faces[i].alive = false;  // this face is consumed
            const int e[3][2] = {{faces[i].a, faces[i].b},
                                 {faces[i].b, faces[i].c},
                                 {faces[i].c, faces[i].a}};
            for (int k = 0; k < 3; ++k) {
                const int u = e[k][0], v = e[k][1];
                // look for the REVERSE edge (v,u) already in the buffer -> cancel.
                int found = -1;
                for (int q = 0; q < nedges; ++q) {
                    if (edge_a[q] == v && edge_b[q] == u) { found = q; break; }
                }
                if (found >= 0) {
                    // cancel: remove by shifting tail down (preserve order -> D1).
                    for (int q = found; q < nedges - 1; ++q) {
                        edge_a[q] = edge_a[q + 1]; edge_b[q] = edge_b[q + 1];
                    }
                    --nedges;
                } else {
                    if (nedges >= kEpaMaxEdges) { overflow = true; break; }
                    edge_a[nedges] = u; edge_b[nedges] = v; ++nedges;
                }
            }
        }
        if (overflow) break;  // best-so-far

        // --- 6) Append new faces from the horizon edges, FIXED scan order. ---
        bool face_of = false;
        for (int q = 0; q < nedges; ++q) {
            if (nfaces >= kEpaMaxFaces) { face_of = true; break; }
            faces[nfaces++] = MakeFace(verts, edge_a[q], edge_b[q], new_vi);
        }
        if (face_of) break;  // best-so-far
    }

    if (best_face < 0) { res.ok = false; return res; }

    // --- 7) Witness: barycentric projection of the origin onto the closest face,
    // then interpolate the per-body support points (the standard EPA witness). ---
    const EpaFace& f = faces[best_face];
    const Vec3 va = verts[f.a].v, vb = verts[f.b].v, vc = verts[f.c].v;
    const Vec3 proj = best_n * best_d;  // origin projected onto the face plane
    // barycentric (u,v,w) of proj in triangle (va,vb,vc).
    const Vec3 v0 = vb - va, v1 = vc - va, v2 = proj - va;
    const float d00 = v0.Dot(v0), d01 = v0.Dot(v1), d11 = v1.Dot(v1);
    const float d20 = v2.Dot(v0), d21 = v2.Dot(v1);
    const float denom = d00 * d11 - d01 * d01;
    float bv = 0.0f, bw = 0.0f;
    if (denom > 1.0e-12f) { bv = (d11 * d20 - d01 * d21) / denom;
                            bw = (d00 * d21 - d01 * d20) / denom; }
    const float bu = 1.0f - bv - bw;
    res.witness_a = verts[f.a].a * bu + verts[f.b].a * bv + verts[f.c].a * bw;
    res.witness_b = verts[f.a].b * bu + verts[f.b].b * bv + verts[f.c].b * bw;
    res.ok = true;
    res.normal = best_n;   // CSO outward normal (origin -> closest face); the MTV
                           // that separates A is -res.normal (see BuildConvexManifold)
    res.depth = best_d;
    return res;
}

// ---------------------------------------------------------------------------
// FACE MODEL: supporting-vertex-set face for one proxy along a world direction.
// ---------------------------------------------------------------------------
// Gather every vertex whose support value is within tol of the max along `dir`
// (the contact face). FIXED vertex scan. Then order the set CCW around `dir`
// (a geometric ordering of distinct convex-polygon verts; atan2f is a geometric
// ordering here, NOT a tie-break key -- distinct verts have distinct angles;
// collinear/coincident ties break by vertex index). Returns count (<= cap).
inline constexpr int kFaceMax = 16;  // supporting-set cap (hulls in v0.8 are small)

struct FacePoly {
    Vec3     v[kFaceMax];
    uint32_t idx[kFaceMax];  // source vertex index (feature)
    int      count = 0;
};

// Gather the supporting-vertex set of a HULL along world dir.
NUKA_CVX_HD inline FacePoly GatherHullFace(const ConvexHullView& h, Vec3 dir,
                                           float tol) {
    FacePoly fp;
    // max support value
    float best = -3.4e38f;
    for (uint32_t i = 0; i < h.vcount; ++i) {
        const float d = h.Vertex(i).Dot(dir);
        if (d > best) best = d;
    }
    // gather within tol (fixed scan, ascending index -> deterministic membership)
    for (uint32_t i = 0; i < h.vcount && fp.count < kFaceMax; ++i) {
        const Vec3 v = h.Vertex(i);
        if (v.Dot(dir) >= best - tol) { fp.v[fp.count] = v; fp.idx[fp.count] = i; ++fp.count; }
    }
    return fp;
}

// Gather a BOX's supporting face (4 corners) along world dir. The face is the
// box face whose outward axis is most aligned with dir; emit its 4 corners in a
// FIXED winding so the feature ids are stable.
NUKA_CVX_HD inline FacePoly GatherBoxFace(const amf::PrimParams& box, Vec3 dir) {
    FacePoly fp;
    // pick face axis = argmax |axis_k . dir| (fixed order, strict -> lowest k).
    int axis = 0; float best = -1.0f; float sgn = 1.0f;
    for (int k = 0; k < 3; ++k) {
        const float d = box.frame.Axis(k).Dot(dir);
        const float ad = d < 0.0f ? -d : d;
        if (ad > best) { best = ad; axis = k; sgn = d >= 0.0f ? 1.0f : -1.0f; }
    }
    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
    const Vec3 au = box.frame.Axis(u), av = box.frame.Axis(v);
    const Vec3 an = box.frame.Axis(axis);
    const Vec3 center = box.frame.t + an * (sgn * amf::Comp(box.half_extents, axis));
    const float ue = amf::Comp(box.half_extents, u), ve = amf::Comp(box.half_extents, v);
    const float su[4] = {-1.0f, +1.0f, +1.0f, -1.0f};
    const float sv[4] = {-1.0f, -1.0f, +1.0f, +1.0f};
    for (int k = 0; k < 4; ++k) {
        fp.v[fp.count] = center + au * (su[k] * ue) + av * (sv[k] * ve);
        fp.idx[fp.count] = static_cast<uint32_t>(k);
        ++fp.count;
    }
    return fp;
}

// Order a FacePoly's verts CCW around `normal` (geometric; in-place). Reference
// = vertex 0; angle measured in the plane basis (e0, e1) with e0 along (v[0]-c).
NUKA_CVX_HD inline void OrderFaceCCW(FacePoly& fp, Vec3 normal) {
    if (fp.count <= 2) return;
    // centroid
    Vec3 c{0.0f, 0.0f, 0.0f};
    for (int i = 0; i < fp.count; ++i) c += fp.v[i];
    c = c / static_cast<float>(fp.count);
    // plane basis
    Vec3 e0 = amf::Norm(fp.v[0] - c, Vec3::UnitX());
    Vec3 e1 = normal.Cross(e0);
    // compute angle per vertex, then insertion sort by (angle ASC, idx ASC).
    float ang[kFaceMax];
    for (int i = 0; i < fp.count; ++i) {
        const Vec3 r = fp.v[i] - c;
        ang[i] = atan2f(r.Dot(e1), r.Dot(e0));
    }
    for (int i = 1; i < fp.count; ++i) {
        const Vec3 kv = fp.v[i]; const uint32_t ki = fp.idx[i]; const float ka = ang[i];
        int j = i - 1;
        while (j >= 0 && (ang[j] > ka || (ang[j] == ka && fp.idx[j] > ki))) {
            fp.v[j + 1] = fp.v[j]; fp.idx[j + 1] = fp.idx[j]; ang[j + 1] = ang[j]; --j;
        }
        fp.v[j + 1] = kv; fp.idx[j + 1] = ki; ang[j + 1] = ka;
    }
}

// ---------------------------------------------------------------------------
// MANIFOLD BUILD from the EPA result. `normalA` = separation dir for side A.
// ---------------------------------------------------------------------------
// If EITHER side lacks a flat face (sphere/capsule), emit the single EPA witness
// (sphere) -- 1 point. Otherwise build the reference/incident faces and clip.
//   normal_cso = res.normal (CSO outward, points A away from B == sep dir for A).
//   reference face on A along +normal_cso ; incident face on B along -normal_cso.
NUKA_CVX_HD inline void BuildConvexManifold(const SupportProxy& A,
                                            const SupportProxy& B,
                                            const EpaResult& epa,
                                            ContactManifold* out) {
    out->Clear();
    if (!epa.ok || epa.depth <= 0.0f) return;
    // nA = epa.normal = the CSO outward normal at the closest face (points from the
    // origin toward that face). It drives ALL the geometry below: reference face on
    // A along +nA, incident face on B along -nA, depth = (ref_c - p).Dot(nA).
    // BUT the EMITTED contact normal must be "separation dir for side A" = the MTV
    // that moves A out of B = -depth*epa.normal direction = -nA. So we keep nA for
    // the geometry and write sep_dir_A == -nA into every emitted point. (Do NOT
    // negate nA globally -- that would gather A's FAR face and clip the wrong side.)
    const Vec3 nA = epa.normal;
    const Vec3 sep_dir_A = -nA;  // separation dir for side A (the emitted normal)

    if (!A.HasFace() || !B.HasFace()) {
        // 1-point witness (sphere/capsule side present). Contact point = the
        // surface midpoint between the witness points.
        ContactPoint pt;
        pt.position = (epa.witness_a + epa.witness_b) * 0.5f;
        pt.normal = sep_dir_A;
        pt.penetration = epa.depth;
        pt.stable_key = 0ull;
        out->AddPoint(pt);
        return;
    }

    // --- Reference (A) face along +nA ; incident (B) face along -nA. ---
    const float kFaceTol = 1.0e-3f;
    FacePoly ref = (A.kind == SupportKind::Box) ? GatherBoxFace(*A.prim, nA)
                                                : GatherHullFace(*A.hull, nA, kFaceTol);
    FacePoly inc = (B.kind == SupportKind::Box) ? GatherBoxFace(*B.prim, -nA)
                                                : GatherHullFace(*B.hull, -nA, kFaceTol);
    OrderFaceCCW(ref, nA);
    OrderFaceCCW(inc, -nA);

    if (ref.count < 3 || inc.count < 3) {
        // not enough for a polygon clip (edge/vertex feature) -> 1-pt witness.
        ContactPoint pt;
        pt.position = (epa.witness_a + epa.witness_b) * 0.5f;
        pt.normal = sep_dir_A; pt.penetration = epa.depth; pt.stable_key = 0ull;
        out->AddPoint(pt);
        return;
    }

    // Reference face plane: point = ref centroid, outward normal = nA.
    Vec3 ref_c{0.0f, 0.0f, 0.0f};
    for (int i = 0; i < ref.count; ++i) ref_c += ref.v[i];
    ref_c = ref_c / static_cast<float>(ref.count);

    // Load the incident polygon into a ClipPoly (feature = incident vertex idx).
    amf::ClipPoly poly;
    for (int i = 0; i < inc.count && i < amf::kClipMax; ++i) poly.Add(inc.v[i], inc.idx[i]);

    // Clip against each reference-face SIDE plane (edge i -> i+1), outward normal
    // = edge_dir x nA, point = edge start. FIXED edge order -> D1.
    for (int i = 0; i < ref.count; ++i) {
        const int j = (i + 1) % ref.count;
        const Vec3 edge = ref.v[j] - ref.v[i];
        Vec3 side_n = edge.Cross(nA);       // points OUT of the convex ref polygon
        // ensure it points away from the polygon interior (centroid behind it)
        if ((ref_c - ref.v[i]).Dot(side_n) > 0.0f) side_n = -side_n;
        poly = amf::ClipAgainstPlane(poly, ref.v[i], side_n);
    }

    // Keep survivors BELOW the reference face (penetrating); project onto the ref
    // plane for the contact position; penetration = depth below the ref face.
    amf::ManifoldPointCand cand[amf::kClipMax];
    int ncand = 0;
    for (int k = 0; k < poly.count; ++k) {
        const float depth = (ref_c - poly.v[k]).Dot(nA);  // >0 == below ref face
        if (depth >= 0.0f) {
            const Vec3 onref = poly.v[k] + nA * depth;
            cand[ncand].position    = (poly.v[k] + onref) * 0.5f;
            cand[ncand].penetration = depth;
            cand[ncand].feature     = static_cast<uint32_t>(k);  // clip-output index (D1)
            ++ncand;
        }
    }
    if (ncand == 0) {
        // no penetrating survivor -> 1-pt witness fallback (deterministic).
        ContactPoint pt;
        pt.position = (epa.witness_a + epa.witness_b) * 0.5f;
        pt.normal = sep_dir_A; pt.penetration = epa.depth; pt.stable_key = 0ull;
        out->AddPoint(pt);
        return;
    }
    amf::ReduceAndEmitSpread(cand, ncand, sep_dir_A, out);  // emit sep dir for A
}

// ---------------------------------------------------------------------------
// CONVEX x PLANE (special-cased -- NOT through GJK).
// ---------------------------------------------------------------------------
// A MuJoCo plane is an INFINITE half-space: its support is UNBOUNDED along
// +normal, so the Minkowski difference is unbounded and GJK never deterministically
// encloses the origin. We special-case it: signed-distance every hull vertex to
// the plane, keep the ones below (penetrating), normal = plane normal, depth =
// -signed_dist, feature = vertex index. This is C3b's BoxPlane generalized to an
// arbitrary hull (the SAME ReduceAndEmitSpread). `normal_hull_to_plane` is the
// separation dir for the HULL side (== plane normal). The caller orders args
// (hull, plane) and re-signs for the manifold's actual side A.
NUKA_CVX_HD inline void HullPlane(const ConvexHullView& hull,
                                  const amf::PrimParams& plane,
                                  Vec3 normal_for_hull, ContactManifold* out) {
    out->Clear();
    const Vec3 n = amf::Norm(plane.frame.cy, Vec3::UnitY());  // plane normal (world)
    amf::ManifoldPointCand cand[amf::kClipMax];
    int ncand = 0;
    // Fixed-order vertex scan; first kClipMax penetrating verts (deepest kept by
    // the FPS reducer). Feature = vertex index (deterministic, warm-start id).
    for (uint32_t i = 0; i < hull.vcount; ++i) {
        const Vec3 v = hull.Vertex(i);
        const float signed_dist = (v - plane.frame.t).Dot(n);
        if (signed_dist < 0.0f && ncand < amf::kClipMax) {
            cand[ncand].position    = v - n * (signed_dist * 0.5f);  // midpoint to surface
            cand[ncand].penetration = -signed_dist;
            cand[ncand].feature     = i;
            ++ncand;
        }
    }
    amf::ReduceAndEmitSpread(cand, ncand, normal_for_hull, out);
}

// ---------------------------------------------------------------------------
// TOP-LEVEL: convex narrowphase for two proxies. normal = sep dir for A.
// ---------------------------------------------------------------------------
NUKA_CVX_HD inline void ConvexNarrowphase(const SupportProxy& A,
                                          const SupportProxy& B,
                                          ContactManifold* out) {
    out->Clear();
    const GjkResult gjk = GjkIntersect(A, B);
    if (!gjk.intersect) return;  // separated -> no contact
    const EpaResult epa = EpaExpand(A, B, gjk);
    BuildConvexManifold(A, B, epa, out);
}

} // namespace nuka::collision::cvx
