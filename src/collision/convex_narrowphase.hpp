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
    // G2 (general-heightfield TRIANGLE_PRISM): the solid -Z extrusion depth, in
    // the hull's LOCAL frame (mirrors Newton support_function.py:167-174 which
    // extrudes the triangle along local -Z to make a SOLID prism so GJK/EPA
    // resolves shapes resting on the top face without tunnelling through the
    // back). Used ONLY by SupportTrianglePrism (SupportKind::TrianglePrism); the
    // first 3 `verts` are the triangle, the support adds local (0,0,-extrude)
    // when the query direction points -Z. extrude == 0 (default) makes the prism
    // a flat triangle, so a non-prism ConvexHullView is byte-unchanged.
    float        extrude = 0.0f;
    // M4 DEVICE-ONLY opt-in: when true, SupportHull runs its vertex scan
    // WARP-COOPERATIVELY (lanes split the pool, exact lowest-index argmax
    // reduce — bit-identical result). REQUIREMENT: every caller up the GJK
    // stack must then execute in FULL-WARP LOCKSTEP (all 32 lanes, converged,
    // identical inputs) — the nk union narrowphase does exactly that. Host
    // code and the legacy thread-per-slot device callers leave it false and
    // take the serial scan, byte-unchanged.
    bool warp_lockstep = false;

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

// Hull support: LOWEST-index argmax of v.Dot(dir) (D1). M4 NOTE — the scan is
// now a 4-WAY INDEPENDENT-CHAIN unroll with an explicit (value, lowest-index)
// tie-break at the combine. This is EXACTLY the former serial strict-`>`
// first-win scan's result: every per-vertex dot is computed identically, the
// max over a set is order-independent, and "lowest index among equal maxima"
// is precisely what strict-> first-win kept. The split exists for the DEVICE
// dependency chain (the serial compare-select chain over the 1796-vert cup
// hull was the measured M4 narrowphase bottleneck); host results are
// bit-identical.
NUKA_CVX_HD inline Vec3 SupportHull(const ConvexHullView& h, Vec3 dir,
                                    uint32_t* out_idx) {
#if defined(__CUDA_ARCH__)
    if (h.warp_lockstep) {
        // WARP-COOPERATIVE scan (see ConvexHullView::warp_lockstep): lanes
        // split the pool by stride-32, each keeping its subset's lowest-index
        // strict-> max; the shuffle reduce prefers (higher value, then lower
        // index) — exactly the serial first-win argmax, bit-identical.
        const uint32_t lane = threadIdx.x & 31u;
        float bd = -3.402823466e+38f;
        uint32_t bi = 0u;
        for (uint32_t i = lane; i < h.vcount; i += 32u) {
            const float d = h.Vertex(i).Dot(dir);
            if (d > bd) { bd = d; bi = i; }
        }
        for (uint32_t off = 16u; off > 0u; off >>= 1u) {
            const float od = __shfl_down_sync(0xffffffffu, bd, off);
            const uint32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (od > bd || (od == bd && oi < bi)) { bd = od; bi = oi; }
        }
        bi = __shfl_sync(0xffffffffu, bi, 0);
        if (out_idx) *out_idx = bi;
        return h.Vertex(bi);
    }
#endif
    // Four EXPLICIT scalar chains (no array indexing — a dynamically indexed
    // chain array spills to local memory on device, defeating the unroll).
    const float d0 = h.Vertex(0u).Dot(dir);
    float cd0 = d0, cd1 = d0, cd2 = d0, cd3 = d0;
    uint32_t ci0 = 0u, ci1 = 0u, ci2 = 0u, ci3 = 0u;
    uint32_t i = 0u;
    for (; i + 4u <= h.vcount; i += 4u) {
        const float a = h.Vertex(i + 0u).Dot(dir);
        const float b = h.Vertex(i + 1u).Dot(dir);
        const float c = h.Vertex(i + 2u).Dot(dir);
        const float d = h.Vertex(i + 3u).Dot(dir);
        if (a > cd0) { cd0 = a; ci0 = i + 0u; }  // strict -> lowest idx / chain
        if (b > cd1) { cd1 = b; ci1 = i + 1u; }
        if (c > cd2) { cd2 = c; ci2 = i + 2u; }
        if (d > cd3) { cd3 = d; ci3 = i + 3u; }
    }
    for (; i < h.vcount; ++i) {
        const float d = h.Vertex(i).Dot(dir);
        if (d > cd0) { cd0 = d; ci0 = i; }
    }
    // Combine with the (value, LOWEST index) tie-break == the serial result.
    uint32_t best = ci0;
    float best_d = cd0;
    if (cd1 > best_d || (cd1 == best_d && ci1 < best)) { best_d = cd1; best = ci1; }
    if (cd2 > best_d || (cd2 == best_d && ci2 < best)) { best_d = cd2; best = ci2; }
    if (cd3 > best_d || (cd3 == best_d && ci3 < best)) { best_d = cd3; best = ci3; }
    if (out_idx) *out_idx = best;
    return h.Vertex(best);
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

// Triangle-prism support (G2, the general-heightfield per-cell collider). The
// first 3 `verts` of the ConvexHullView are the triangle (mesh-local); the prism
// is the triangle extruded `extrude` along the hull's LOCAL -Z to make it SOLID.
// Mirrors Newton support_function.py:148-174: 3-vert max-dot (LOWEST-index tie-
// break -> D1, like SupportHull), then if the query direction points along the
// triangle's local -Z, add the extruded offset. The 3-vert scan is in WORLD
// space (Vertex() bakes the frame), and the extrude offset is the frame's local
// -Z column scaled by `extrude` -- so the whole shape rotates/translates with the
// frame exactly like every other support. Rides the SAME GJK/EPA/BuildConvex
// Manifold (a prism HAS a flat face, so HasFace() face-clip applies).
NUKA_CVX_HD inline Vec3 SupportTrianglePrism(const ConvexHullView& h, Vec3 dir,
                                             uint32_t* out_idx) {
    // 3-vert max-dot with a fixed-order, strict-`>` (lowest-index) tie-break.
    const Vec3 v0 = h.Vertex(0u);
    const Vec3 v1 = h.Vertex(1u);
    const Vec3 v2 = h.Vertex(2u);
    const float d0 = v0.Dot(dir);
    const float d1 = v1.Dot(dir);
    const float d2 = v2.Dot(dir);
    uint32_t best = 0u;
    float best_d = d0;
    Vec3 best_v = v0;
    if (d1 > best_d) { best_d = d1; best = 1u; best_v = v1; }
    if (d2 > best_d) { best_d = d2; best = 2u; best_v = v2; }
    // Solid -Z extrusion: when the support direction points along the triangle's
    // LOCAL -Z, the deepest point is on the bottom face. h.frame.cz is the local
    // +Z axis in world; the local -Z direction is -cz. Test dir against local -Z.
    if (h.extrude > 0.0f && dir.Dot(h.frame.cz) < 0.0f) {
        best_v = best_v - h.frame.cz * h.extrude;
    }
    if (out_idx) *out_idx = best;
    return best_v;
}

// ---------------------------------------------------------------------------
// SupportProxy -- a tagged union so ONE GJK/EPA path handles hull-vs-hull AND
// hull-vs-primitive. The handler fills two of these (one per side). No virtuals
// (device-clean) -- a kind tag + the two possible payloads (a view ptr / a prim
// ptr). The support() dispatch is a small fixed switch.
// ---------------------------------------------------------------------------
// TrianglePrism (G2) reuses the `hull` payload (its first 3 verts + frame +
// extrude); it is a hull-like side (HasFace() == true) for the face-clip.
enum class SupportKind : uint8_t {
    Hull = 0, Box = 1, Sphere = 2, Capsule = 3, TrianglePrism = 4
};

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
            case SupportKind::TrianglePrism: return SupportTrianglePrism(*hull, dir, out_idx);
        }
        if (out_idx) *out_idx = 0u;
        return Vec3::Zero();
    }
    // Center-ish interior point (a deterministic seed for the first GJK dir).
    NUKA_CVX_HD Vec3 Center() const {
        if (kind == SupportKind::Hull || kind == SupportKind::TrianglePrism)
            return hull->frame.t;
        return prim->frame.t;
    }
    // Does this side have a flat polygonal supporting face worth clipping?
    NUKA_CVX_HD bool HasFace() const {
        return kind == SupportKind::Hull || kind == SupportKind::Box ||
               kind == SupportKind::TrianglePrism;
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
inline constexpr int   kEpaMaxIter     = 64;
inline constexpr int   kEpaMaxFaces    = 128;
inline constexpr int   kEpaMaxVerts    = 64;
inline constexpr int   kEpaMaxEdges    = 64;
inline constexpr float kEpaTol         = 1.0e-4f;
// Enclosure pre-loop: a seed face whose SIGNED origin-distance is <= this is
// treated as "origin on/outside it" -> the polytope does not strictly contain the
// origin and must be expanded before the metric loop. Fixed retry budget so the
// pre-loop is bounded + D1 (origin-on-edge needs 2 dirs, on-vertex 3; 12 is ample).
inline constexpr float kEpaEnclEps     = 1.0e-6f;
inline constexpr int   kEpaMaxEnclIter = 12;

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

// Build an outward-normal face from 3 CSO verts. Orientation is by the POLYTOPE
// CENTROID (`interior`), NOT the origin-sign: GJK's output simplex has no
// guaranteed consistent winding, and for an origin-on-boundary seed a `dist==0`
// face's origin-sign flip never triggers, leaving the normal pointing INWARD --
// the precise root cause of the origin-on-boundary false-negative (the support
// query along an inward normal can never advance past the origin, so the polytope
// churns coplanar). The centroid of a non-degenerate seed tetra is a guaranteed
// strictly-interior point, so `nrm . (va - interior) > 0` reliably picks OUTWARD
// regardless of where the origin sits. `dist` is then the SIGNED origin->plane
// distance (>0 origin inside this face's halfspace, <=0 origin on/outside it).
// For a strictly-interior origin centroid-sign == origin-sign, so every config
// that already enclosed the origin gets a BIT-IDENTICAL face (D1 preserved).
NUKA_CVX_HD inline EpaFace MakeFace(const MinkPoint* verts, int a, int b, int c,
                                   Vec3 interior) {
    EpaFace f;
    f.a = a; f.b = b; f.c = c; f.alive = true;
    const Vec3 va = verts[a].v, vb = verts[b].v, vc = verts[c].v;
    Vec3 nrm = (vb - va).Cross(vc - va);
    const float nlen = amf::Len(nrm);
    if (nlen < 1.0e-12f) { f.normal = Vec3::UnitY(); f.dist = 0.0f; return f; }
    nrm = nrm / nlen;
    // Force OUTWARD relative to the interior point (centroid), not the origin.
    if (nrm.Dot(va - interior) < 0.0f) nrm = -nrm;
    f.normal = nrm;
    f.dist = nrm.Dot(va);           // SIGNED distance origin->plane (may be <=0)
    return f;
}

// Centroid of the first `nverts` CSO vertices (a strictly-interior point of their
// convex hull -- the polytope). FIXED-ORDER sum (D1). Used to orient faces OUTWARD
// independent of where the origin sits (origin-on-boundary robustness).
NUKA_CVX_HD inline Vec3 PolytopeCentroid(const MinkPoint* verts, int nverts) {
    Vec3 c{0.0f, 0.0f, 0.0f};
    for (int i = 0; i < nverts; ++i) c += verts[i].v;
    return c / static_cast<float>(nverts > 0 ? nverts : 1);
}

// Outcome of inserting a support vertex + rebuilding the horizon.
enum class EpaInsert : uint8_t { Inserted = 0, NoVisible = 1, Overflow = 2 };

// Insert support point `p` into the polytope: remove every face VISIBLE from p
// (p strictly on its OUTWARD side), build the silhouette horizon by edge-toggle,
// and append a fan of new faces (each horizon edge + p), oriented OUTWARD by the
// updated centroid. SHARED by the enclosure pre-loop AND the metric loop so the
// topology update is byte-for-byte identical on both paths (D1).
//   Edge buffer scanned in face index order, each face's 3 edges in fixed
//   (a-b, b-c, c-a) order; add if absent, cancel if reverse present -> only the
//   silhouette survives. Deterministic. The new vertex is appended at `verts`,
//   then faces are reoriented against the NEW centroid (which includes p) so the
//   fan normals point outward even when the origin is on the polytope boundary.
NUKA_CVX_HD inline EpaInsert EpaInsertSupport(MinkPoint* verts, int& nverts,
                                              EpaFace* faces, int& nfaces,
                                              const MinkPoint& p) {
    if (nverts >= kEpaMaxVerts) return EpaInsert::Overflow;
    // --- find + kill visible faces, toggle horizon edges (fixed order). ---
    int edge_a[kEpaMaxEdges];
    int edge_b[kEpaMaxEdges];
    int nedges = 0;
    bool any_visible = false;
    for (int i = 0; i < nfaces; ++i) {
        if (!faces[i].alive) continue;
        // visible if p is strictly on the OUTWARD side of the face plane.
        if (faces[i].normal.Dot(p.v) - faces[i].dist <= 0.0f) continue;
        any_visible = true;
        faces[i].alive = false;  // consumed
        const int e[3][2] = {{faces[i].a, faces[i].b},
                             {faces[i].b, faces[i].c},
                             {faces[i].c, faces[i].a}};
        for (int k = 0; k < 3; ++k) {
            const int u = e[k][0], v = e[k][1];
            int found = -1;
            for (int q = 0; q < nedges; ++q) {
                if (edge_a[q] == v && edge_b[q] == u) { found = q; break; }
            }
            if (found >= 0) {
                for (int q = found; q < nedges - 1; ++q) {
                    edge_a[q] = edge_a[q + 1]; edge_b[q] = edge_b[q + 1];
                }
                --nedges;
            } else {
                if (nedges >= kEpaMaxEdges) return EpaInsert::Overflow;
                edge_a[nedges] = u; edge_b[nedges] = v; ++nedges;
            }
        }
    }
    if (!any_visible) return EpaInsert::NoVisible;  // p inside polytope (no progress)

    // Append p, then build the fan, orienting against the NEW centroid (incl. p)
    // so the new faces' normals are OUTWARD even on an origin-on-boundary seed.
    const int new_vi = nverts;
    verts[nverts++] = p;
    const Vec3 c = PolytopeCentroid(verts, nverts);
    for (int q = 0; q < nedges; ++q) {
        if (nfaces >= kEpaMaxFaces) return EpaInsert::Overflow;
        faces[nfaces++] = MakeFace(verts, edge_a[q], edge_b[q], new_vi, c);
    }
    return EpaInsert::Inserted;
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

    // --- 2) Seed the polytope: 4 faces of the tetra, each oriented OUTWARD by the
    // tetra CENTROID (a guaranteed strictly-interior point of the non-degenerate
    // seed). This is the load-bearing fix: orienting by the origin-sign (the old
    // posture) leaves a `dist==0` face's normal pointing INWARD on an
    // origin-on-boundary seed, so the subsequent support query churns coplanar and
    // never finds the real penetrating face (the confirmed false-negative). With
    // centroid-orientation `face.dist` is the SIGNED origin->plane distance: <=0
    // means the origin is on/outside that face -> the seed does NOT strictly
    // contain the origin and must be expanded (step 2b) before the metric loop. ---
    EpaFace faces[kEpaMaxFaces];
    int nfaces = 0;
    {
        const Vec3 c0 = PolytopeCentroid(verts, nverts);
        faces[nfaces++] = MakeFace(verts, 0, 1, 2, c0);
        faces[nfaces++] = MakeFace(verts, 0, 2, 3, c0);
        faces[nfaces++] = MakeFace(verts, 0, 3, 1, c0);
        faces[nfaces++] = MakeFace(verts, 1, 3, 2, c0);
    }

    // --- 2b) ENCLOSURE PRE-LOOP: make the origin STRICTLY interior. ---
    // While any alive face has SIGNED dist <= kEpaEnclEps (origin on/outside it),
    // query the support along that face's OUTWARD normal:
    //   - if it advances strictly past the origin (pd - dist > eps), the CSO
    //     extends beyond the origin in that direction -> insert it (horizon
    //     rebuild) so the origin moves strictly inside; it cannot revert (the
    //     polytope only grows). Lowest-index offending face first (D1).
    //   - if it does NOT advance (pd <= eps), the CSO SURFACE passes through the
    //     origin along that normal: a GENUINE touching contact (true depth 0).
    //     Stop expanding -> the metric loop yields depth 0 -> BuildConvexManifold
    //     drops it (correct "no penetration"), NOT a false positive.
    // Fixed retry budget (kEpaMaxEnclIter) so the pre-loop is bounded + D1.
    for (int encl = 0; encl < kEpaMaxEnclIter; ++encl) {
        int   bad = -1;
        for (int i = 0; i < nfaces; ++i) {
            if (!faces[i].alive) continue;
            if (faces[i].dist <= kEpaEnclEps) { bad = i; break; }  // lowest index
        }
        if (bad < 0) break;  // origin strictly inside every face -> enclosed
        const MinkPoint p = SupportMink(A, B, faces[bad].normal);
        const float pd = p.v.Dot(faces[bad].normal);
        if (pd - faces[bad].dist <= kEpaEnclEps) {
            // The CSO does NOT extend past the origin along this deficient normal:
            // the origin sits ON the CSO SURFACE along `faces[bad].normal`, i.e. a
            // GENUINE TOUCHING contact -- the true penetration depth is 0 in this
            // direction, and that direction IS the contact normal. (The polytope may
            // still be DEEP on the OTHER sides of a flat CSO -- e.g. two boxes flush
            // face-to-face: the CSO is a box with the origin on its +Z face but ~2
            // deep on -Z/sides. Letting the metric loop run would pick one of those
            // FAR positive faces and report a bogus deep contact with the wrong
            // normal.) So a genuine touch must terminate as depth-0 / NO contact:
            // bail with ok=false -> empty manifold (correct; distinct from a real
            // overlap, which DOES strictly enclose the origin and never reaches here).
            res.ok = false;
            return res;
        }
        const EpaInsert st = EpaInsertSupport(verts, nverts, faces, nfaces, p);
        if (st != EpaInsert::Inserted) break;  // overflow / no-progress -> best-so-far
    }

    Vec3  best_n = Vec3::UnitY();
    float best_d = 0.0f;
    int   best_face = -1;
    bool  progressed = false;  // did the metric loop converge on a positive-depth face?

    for (int it = 0; it < kEpaMaxIter; ++it) {
        // --- 3) FIXED-ORDER ARRAY SCAN for the closest ALIVE face. ---
        // Key = (dist ASC, face index ASC). Strict `<` improvement only, so on a
        // distance tie the LOWEST face index wins. NO float-keyed heap (D1 trap).
        //
        // CRITICAL: SKIP degenerate "on-origin" faces (signed dist <= kEpaEnclEps).
        // The box-box (and any flat-faced) CSO carries a WALL of vertices coplanar
        // with the origin (e.g. the x=0 / y=0 cross-section vertices); incremental
        // expansion keeps fabricating SLIVER triangles whose 3 verts are coplanar-
        // through-origin -> dist==0 with a spurious axis normal. Such a sliver is an
        // INTERIOR artifact, never the true closest CSO face, yet its support
        // advances (pd>>0) so selecting it churns the polytope NON-MONOTONICALLY
        // (the confirmed 0->0.057->0 trace) until face overflow -> false negative.
        // After the enclosure pre-loop the origin is STRICTLY interior (a genuine
        // touch already bailed with ok=false above), so a strictly-POSITIVE-distance
        // face (the real MTV face, z=0.1 here) always exists; selecting among only
        // those gives classic monotone EPA convergence. If by FP edge effects no
        // positive face is found, close stays -1 -> break -> ok=false -> empty (a
        // safe no-contact). Fixed eps + fixed scan order keep the skip deterministic (D1).
        int   close = -1;
        float close_d = 3.4e38f;
        for (int i = 0; i < nfaces; ++i) {
            if (!faces[i].alive) continue;
            if (faces[i].dist <= kEpaEnclEps) continue;  // skip on-origin sliver
            if (faces[i].dist < close_d) { close_d = faces[i].dist; close = i; }
        }
        if (close < 0) break;
        best_face = close; best_n = faces[close].normal; best_d = faces[close].dist;

        // --- 4) New support along the closest face's outward normal. ---
        const MinkPoint p = SupportMink(A, B, faces[close].normal);
        const float pd = p.v.Dot(faces[close].normal);
        if (pd - close_d < kEpaTol) {
            progressed = true;  // support no further than the face -> CONVERGED
            break;
        }
        // --- 5) Insert + rebuild horizon (shared helper; centroid-oriented fan). ---
        const EpaInsert st = EpaInsertSupport(verts, nverts, faces, nfaces, p);
        if (st != EpaInsert::Inserted) break;  // overflow / no-progress -> best-so-far
    }

    if (best_face < 0) { res.ok = false; return res; }

    // --- ok semantics: only report success when the metric loop CONVERGED. An
    // iteration/capacity cap hit WITHOUT convergence ("progressed" stays false)
    // while the support proves a deeper face means the polytope is non-manifold /
    // the seed failed -> res.ok=false so the caller does NOT emit a depth-0
    // manifold for a real overlap (safety net; the enclosure step is what FINDS
    // the true face). A genuine touch (depth 0) DOES converge (pd<=close_d) -> ok,
    // and is dropped by the epa.depth<=0 guard, not by ok. ---

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
    res.ok = progressed;   // only a CONVERGED metric loop is a trustworthy result;
                           // an un-converged cap-hit with a deeper support proven is
                           // a seed/polytope failure -> ok=false -> empty (safety net).
    res.normal = best_n;   // CSO outward normal (origin -> closest face); the MTV
                           // that separates A is -res.normal (see BuildConvexManifold)
    res.depth = best_d;    // SIGNED closest-face distance; <=0 == genuine touch ->
                           // BuildConvexManifold's depth<=0 guard drops it (no contact)
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
// SPHERE x CONVEX-HULL (special-cased -- NOT through GJK/EPA).
// ---------------------------------------------------------------------------
// A sphere is the one analytic-support shape whose CSO curvature breaks EPA's
// shallow-penetration robustness: when the sphere CENTER is OUTSIDE the hull and
// only the RADIUS penetrates, the Minkowski difference's surface near the origin
// is a curved cap, and the incremental polytope expansion is NON-MONOTONE in
// depth (a confirmed detect->drop->detect dead band at pen ~1.0-1.4mm for a
// fingertip-on-cup-wall config). The robust formulation is the EXACT point-to-
// convex distance query: the closest point on the hull to the sphere CENTER. When
// the center is outside the hull this is a pure (monotone) distance problem; the
// penetration is radius - distance and the contact normal is the separation
// direction along (center - closest). This BYPASSES EPA entirely, mirroring the
// HullPlane special case.
//
// CLOSEST POINT ON A CONVEX HULL TO A POINT via GJK-DISTANCE. The ConvexHullView
// carries verts + a frame but NO face/edge connectivity, so a Voronoi-feature
// face walk is not available; the GJK distance sub-algorithm needs only the
// SUPPORT function (the SAME SupportHull the boolean GJK uses), so it is the
// connectivity-free robust choice. We run GJK on the Minkowski difference of the
// HULL and the single point `q`: each CSO vertex is (hull_vertex - q). The closest
// point on that CSO to the ORIGIN, mapped back, is the closest point on the hull
// to `q`. If the origin ends up INSIDE the CSO the point is inside the hull
// (overlap case). FIXED support scan + FIXED sub-simplex evaluation order + fixed
// iteration cap == D1 (no atomics, no float-keyed reordering).
// ---------------------------------------------------------------------------
inline constexpr int   kGjkDistMaxIter = 48;
inline constexpr float kGjkDistEps     = 1.0e-12f;
// Convergence is on a LENGTH gap along the UNIT search direction (scale-invariant):
// an absolute floor (sub-micron) + a relative term so both grasp-scale (mm) hulls
// and metre-scale hulls converge tightly without over-iterating.
inline constexpr float kGjkDistTol     = 1.0e-7f;   // absolute length gap (metres)
inline constexpr float kGjkDistRelTol  = 1.0e-5f;   // relative to |closest|

// A sub-simplex reduction result: the closest point to the origin AND the set of
// simplex vertices (by their index 0..n-1) that SUPPORT it (its Voronoi feature).
// Returning the support set is what lets the GJK distance loop shrink the simplex
// to the supporting feature each step (the textbook Johnson reduction) -- FIXED
// region-test order == D1.
struct SubSimplex {
    Vec3 closest{0.0f, 0.0f, 0.0f};
    int  keep[4] = {0, 0, 0, 0};   // indices (into the passed array) to retain
    int  nkeep = 0;
};

// Closest point to the origin on a SEGMENT (verts[i0], verts[i1]); keep set is the
// supporting vertex (endpoint) or both (interior). FIXED order.
NUKA_CVX_HD inline SubSimplex ReduceSegment(const Vec3* v, int i0, int i1) {
    SubSimplex r;
    const Vec3 a = v[i0], b = v[i1];
    const Vec3 ab = b - a;
    const float t = ab.Dot(ab);
    if (t < kGjkDistEps) { r.closest = a; r.keep[0] = i0; r.nkeep = 1; return r; }
    float u = (-a).Dot(ab) / t;
    if (u <= 0.0f)      { r.closest = a; r.keep[0] = i0; r.nkeep = 1; return r; }
    if (u >= 1.0f)      { r.closest = b; r.keep[0] = i1; r.nkeep = 1; return r; }
    r.closest = a + ab * u; r.keep[0] = i0; r.keep[1] = i1; r.nkeep = 2; return r;
}

// Closest point to the origin on a TRIANGLE (Ericson RTCD region test); the keep
// set is the supporting vertex / edge pair / all three. FIXED region-test order.
NUKA_CVX_HD inline SubSimplex ReduceTriangle(const Vec3* v, int i0, int i1, int i2) {
    SubSimplex r;
    const Vec3 a = v[i0], b = v[i1], c = v[i2];
    const Vec3 ab = b - a, ac = c - a, ap = -a;
    const float d1 = ab.Dot(ap), d2 = ac.Dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { r.closest = a; r.keep[0] = i0; r.nkeep = 1; return r; }
    const Vec3 bp = -b;
    const float d3 = ab.Dot(bp), d4 = ac.Dot(bp);
    if (d3 >= 0.0f && d4 <= d3) { r.closest = b; r.keep[0] = i1; r.nkeep = 1; return r; }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {       // edge AB
        const float t = d1 / (d1 - d3);
        r.closest = a + ab * t; r.keep[0] = i0; r.keep[1] = i1; r.nkeep = 2; return r;
    }
    const Vec3 cp = -c;
    const float d5 = ab.Dot(cp), d6 = ac.Dot(cp);
    if (d6 >= 0.0f && d5 <= d6) { r.closest = c; r.keep[0] = i2; r.nkeep = 1; return r; }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {       // edge AC
        const float t = d2 / (d2 - d6);
        r.closest = a + ac * t; r.keep[0] = i0; r.keep[1] = i2; r.nkeep = 2; return r;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {  // edge BC
        const float t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        r.closest = b + (c - b) * t; r.keep[0] = i1; r.keep[1] = i2; r.nkeep = 2; return r;
    }
    // face interior: keep all three (barycentric projection onto the plane).
    const float denom = 1.0f / (va + vb + vc);
    const float tv = vb * denom, tw = vc * denom;
    r.closest = a + ab * tv + ac * tw;
    r.keep[0] = i0; r.keep[1] = i1; r.keep[2] = i2; r.nkeep = 3; return r;
}

// Result of the point-to-hull GJK distance query.
struct HullPointDist {
    Vec3  closest{0.0f, 0.0f, 0.0f};  // closest point ON THE HULL (world) to q
    float dist = 0.0f;                // |q - closest| (>=0); ~0 when q is on/inside
    // `inside` is a BEST-EFFORT corroboration only: the distance loop terminates on
    // a near-zero LENGTH gap, so for a deeply-interior q it converges to a surface
    // point with tiny `dist` and reports inside=false. The AUTHORITATIVE inside test
    // is the boolean GJK in SphereHull (GjkIntersect on a zero-radius point), which
    // is scale-invariant. A direct caller that needs a reliable inside flag must use
    // GjkIntersect, NOT this field.
    bool  inside = false;
};

// Closest point on `h` to the world point `q` (GJK distance; SUPPORT-only, D1).
// We map the closest CSO point back to a hull point exactly: closest_cso ==
// closest_hull - q, so closest_hull == closest_cso + q. (The witness barycentric
// is the SAME on the hull-world simplex, but the identity makes it unnecessary.)
NUKA_CVX_HD inline HullPointDist ClosestHullToPoint(const ConvexHullView& h, Vec3 q) {
    HullPointDist res;
    // CSO of (hull - {q}): a simplex of <=4 CSO verts. cso[i] = hull_world_i - q.
    Vec3 cso[4];
    int  n = 0;
    // Seed: support along (q - hull_center) so the first vertex faces q (D1 dir).
    Vec3 dir = q - h.frame.t;
    if (dir.Dot(dir) < kGjkDistEps) dir = Vec3::UnitX();
    {
        uint32_t idx = 0u;
        cso[0] = SupportHull(h, dir, &idx) - q; n = 1;
    }
    Vec3 closest = cso[0];
    for (int it = 0; it < kGjkDistMaxIter; ++it) {
        // 1) closest point on the current simplex + its supporting feature.
        SubSimplex sub;
        if (n == 1) {
            sub.closest = cso[0]; sub.keep[0] = 0; sub.nkeep = 1;
        } else if (n == 2) {
            sub = ReduceSegment(cso, 0, 1);
        } else if (n == 3) {
            sub = ReduceTriangle(cso, 0, 1, 2);
        } else {  // n == 4: closest over the 4 triangular faces (fixed order).
            float best = 3.4e38f; SubSimplex bs;
            const int tri[4][3] = {{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
            for (int f = 0; f < 4; ++f) {
                const SubSimplex t = ReduceTriangle(cso, tri[f][0], tri[f][1], tri[f][2]);
                const float d2 = t.closest.Dot(t.closest);
                if (d2 < best) { best = d2; bs = t; }  // strict -> lowest face index
            }
            sub = bs;
        }
        closest = sub.closest;
        // 2) shrink the simplex to the supporting feature (FIXED order copy).
        Vec3 nc[4];
        for (int i = 0; i < sub.nkeep; ++i) nc[i] = cso[sub.keep[i]];
        for (int i = 0; i < sub.nkeep; ++i) cso[i] = nc[i];
        n = sub.nkeep;

        const float cd2 = closest.Dot(closest);
        if (cd2 < kGjkDistEps) {           // origin in the simplex -> q inside hull
            res.inside = true;
            res.dist = 0.0f;
            res.closest = q;               // zero distance: closest == q
            return res;
        }
        // 3) search toward the origin (toward q): support along the NORMALIZED
        //    -closest. Normalizing is load-bearing: the progress gap must have units
        //    of LENGTH so the convergence tolerance is SCALE-INVARIANT. An
        //    unnormalized -closest makes `improve` a length^2, which at grasp scale
        //    (|closest| ~ 3.6mm -> ~1.3e-5 m^2) trips an absolute length tol after
        //    1-2 iterations, returning a CRUDE closest point that jitters the contact
        //    point/penetration/normal as the cup pose shifts (the ~5% impulse beat).
        const float cd = sqrtf(cd2);
        const Vec3 sdir = -closest / cd;              // unit, points toward the origin
        uint32_t idx = 0u;
        const Vec3 newc = SupportHull(h, sdir, &idx) - q;
        // progress (in LENGTH): the support along the unit search dir must extend the
        // CSO past the current closest face by more than a small absolute gap (with a
        // relative floor so huge hulls still converge). Otherwise the surface is
        // reached -> converged. kGjkDistTol is an absolute LENGTH (metres).
        const float improve = newc.Dot(sdir) - closest.Dot(sdir);
        if (improve <= kGjkDistTol + kGjkDistRelTol * cd) break;  // converged
        // duplicate guard (D1: never re-append an existing simplex vertex).
        bool dup = false;
        for (int i = 0; i < n; ++i) {
            if (amf::Len(newc - cso[i]) < 1.0e-9f) { dup = true; break; }
        }
        if (dup) break;
        if (n >= 4) break;                  // full tetra w/ progress: numerically converged
        cso[n] = newc; ++n;
    }
    res.closest = closest + q;            // closest point on the hull (world)
    res.dist = amf::Len(closest);
    res.inside = false;
    return res;
}

// Sphere x convex-hull -> 1-pt manifold. `sphere_is_a` selects the manifold side
// convention: point.normal is the SEPARATION DIR FOR SIDE A. When the sphere is
// side A the separation dir is (center - closest) (push the sphere off the hull);
// when the hull is side A it is the negation. `out` is cleared first; an empty
// manifold means no contact (separated, or a center-inside fallback that produced
// no positive depth -- never reached by the fingertip-on-wall grasp).
NUKA_CVX_HD inline void SphereHull(const amf::PrimParams& sphere,
                                   const ConvexHullView& hull, bool sphere_is_a,
                                   ContactManifold* out) {
    out->Clear();
    const Vec3 c = sphere.frame.t;        // sphere center (world)
    const float r = sphere.radius;

    // Containment test: is the sphere CENTER inside the hull? We test it with the
    // boolean GJK on a zero-radius sphere (== the point `c`) vs the hull -- robust
    // and SCALE-INVARIANT (the boolean GJK encloses the origin or it does not; no
    // distance tolerance involved, unlike the closest-point loop's near-zero
    // termination). This is the load-bearing inside/outside decision; the
    // closest-point query's own `inside` flag is only a fast-path corroboration.
    amf::PrimParams point = sphere;
    point.radius = 0.0f;                  // a point at the sphere center
    SupportProxy pp; pp.kind = SupportKind::Sphere; pp.prim = &point;
    SupportProxy hp_proxy; hp_proxy.kind = SupportKind::Hull; hp_proxy.hull = &hull;
    const bool center_inside = GjkIntersect(pp, hp_proxy).intersect;

    if (!center_inside) {
        // Center OUTSIDE the hull: contact iff distance < radius. The closest point
        // on the hull to the center is a MONOTONE function of the center position
        // along the approach, so the penetration r - dist never drops out.
        const HullPointDist hp = ClosestHullToPoint(hull, c);
        const float pen = r - hp.dist;
        if (pen <= 0.0f) return;          // separated -> no contact
        // Separation dir for the SPHERE = (center - closest) (push sphere away).
        const Vec3 sep_sphere = amf::Norm(c - hp.closest, Vec3::UnitY());
        const Vec3 sep_dir_A = sphere_is_a ? sep_sphere : -sep_sphere;
        ContactPoint pt;
        pt.position = hp.closest;         // on the hull surface (the contact point)
        pt.normal = sep_dir_A;
        pt.penetration = pen;
        pt.stable_key = 0ull;
        out->AddPoint(pt);
        return;
    }
    // Center INSIDE the hull (deep penetration -- the grasp never hits this). We
    // have no face connectivity to read the min face-plane depth, so recover the
    // shallowest exit by a FIXED-ORDER scan over (a) the 6 axis directions and (b)
    // each hull VERTEX direction from the center (each vertex normal is a candidate
    // outward face direction): the smallest support gap (center->surface distance)
    // is the shallowest face-plane depth, and penetration = radius + that depth.
    // Deterministic (fixed scan, integer tie-break -> lowest index), never garbage;
    // correctness-bounded for the (unused) deep-overlap case, not the shallow case.
    Vec3 best_n = Vec3::UnitY();
    float best_inside = 3.4e38f;          // shallowest distance from center to a face
    const Vec3 axis_probe[6] = {Vec3::UnitX(), {-1,0,0}, Vec3::UnitY(),
                                {0,-1,0}, Vec3::UnitZ(), {0,0,-1}};
    for (int p = 0; p < 6; ++p) {
        uint32_t idx = 0u;
        const Vec3 sv = SupportHull(hull, axis_probe[p], &idx);
        const float gap = (sv - c).Dot(axis_probe[p]);
        if (gap < best_inside) { best_inside = gap; best_n = axis_probe[p]; }
    }
    for (uint32_t i = 0u; i < hull.vcount; ++i) {
        const Vec3 dir = amf::Norm(hull.Vertex(i) - c, Vec3::UnitY());
        uint32_t idx = 0u;
        const Vec3 sv = SupportHull(hull, dir, &idx);
        const float gap = (sv - c).Dot(dir);
        if (gap < best_inside) { best_inside = gap; best_n = dir; }
    }
    if (best_inside < 0.0f) best_inside = 0.0f;
    const Vec3 sep_sphere = best_n;       // shallowest outward direction for the sphere
    const Vec3 sep_dir_A = sphere_is_a ? sep_sphere : -sep_sphere;
    ContactPoint pt;
    pt.position = c - best_n * best_inside;          // on the hull surface
    pt.normal = sep_dir_A;
    pt.penetration = r + best_inside;     // radius + inside depth
    pt.stable_key = 0ull;
    out->AddPoint(pt);
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
