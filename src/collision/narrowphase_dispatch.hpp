#pragma once
// ---------------------------------------------------------------------------
// nuka::collision -- the unified narrowphase dispatch table (v0.8 C3a)
// ---------------------------------------------------------------------------
// The static `(ShapeType a × ShapeType b × NarrowphaseTier) → handler` routing
// table, modelled on MuJoCo's `MJ_COLLISION_TABLE` / `mjCOLLISIONFUNC` pattern
// (research-newton-mujocowarp.md:42,102 / mujoco_warp collision_driver.py:45).
// A CandidatePair (C2a broadphase output) carries two type-tagged collidables;
// the cooked geometry says which ShapeType each side is and whether an SDF is
// available; SelectTier picks the precision tier; the table maps
// (typeA,typeB,tier) to the ONE handler that fills the ContactManifold.
//
// WHY a constexpr table (D1): the route is decided at COMPILE TIME by integer
// indexing -- there is no runtime virtual dispatch, no registration order, no
// hash. Same (typeA,typeB,tier) always resolves to the same function pointer,
// run-to-run and replica-to-replica. O(1) lookup. Adding a new
// (type,type,tier) handler is a LOCALIZED edit: write the handler, then set one
// table slot in MakeNarrowphaseTable() (the extensibility seam, §3 of the
// v0.8 roadmap / the C3-checklist "new pairs = new table rows").
//
// SCOPE OF C3a (this file): the TABLE + SelectTier + the lookup are REAL and
// tested now. The geometry MATH is deferred: C3a ships identifiable STUB
// handlers (each writes a zero/empty manifold tagged with a marker so the
// routing test can prove "(typeA,typeB,tier) reached the tier-T handler").
//   - C3b fills the Analytical primitive×primitive handlers (multi-point/face-face).
//   - C3c fills the Convex (GJK/EPA/SAT + face-clip) handlers.
//   - C3d fills the Sdf (high-precision) handler (wraps find_sdf_contact_kernel).
// Each subtask REPLACES its stub by editing the handler body (and, where it needs
// real geometry inputs, GROWS ShapeProxyView -- see the seam note below) and,
// if it adds new (type,type) coverage, sets new slots in MakeNarrowphaseTable().
//
// VALIDATED-NOT-FORCED: like C2's broadphase, this dispatch is NOT wired into
// world_stepper.cpp. The v0.7 ad-hoc GenerateContact path (IsPlane/IsSphere/
// IsBoxLike, world_stepper.cpp:307) STAYS the production path until C5
// re-baselines. So goldens do not move from C3a.
// ---------------------------------------------------------------------------

#include "collision/analytical_manifold.hpp" // C3b: PrimParams + amf:: handlers
#include "collision/candidate_pair.hpp"     // CandidatePair, CollidableRef
#include "collision/convex_narrowphase.hpp" // C3c: cvx:: GJK/EPA/face-clip + ConvexHullView
#include "constraint/contact_manifold.hpp"  // ContactManifold (extended, C3a)
#include "scene/canonical_types.hpp"        // scene::ShapeType

#include <cstdint>

namespace nuka::collision {

using constraint::ContactManifold;
using scene::ShapeType;

// ---------------------------------------------------------------------------
// Precision tiers (Q2 hybrid narrowphase). Order is the lookup index -- keep
// dense from 0 so it indexes the table's tier dimension directly.
// ---------------------------------------------------------------------------
enum class NarrowphaseTier : uint8_t {
    Analytical = 0,   // primitive × primitive: closed-form multi-point/face-face (C3b)
    Convex     = 1,   // mesh/convex: GJK/EPA/SAT + face-clip (C3c)
    Sdf        = 2,   // SDF-equipped: high-precision witness query (C3d)
    // v0.8 CCD-hook seam (C3-checklist): a future NarrowphaseTier::SweptCCD slot
    // feeds TOI-aware contacts to the SAME contact rows (no new row class). NOT
    // added now -- reserved so adding it is one enum value + one table dimension
    // bump. v0.9 garment/cable are the named consumers.
};
inline constexpr uint32_t kNarrowphaseTierCount = 3u;

// ShapeType is a dense uint8 enum [Sphere..HeightField]; the table is keyed by
// its underlying value. Keep this count in lockstep with scene::ShapeType.
inline constexpr uint32_t kShapeTypeCount =
    static_cast<uint32_t>(ShapeType::HeightField) + 1u;  // 7 in v0.8

// ---------------------------------------------------------------------------
// ShapeProxyView -- the per-shape geometry view a narrowphase handler reads.
// ---------------------------------------------------------------------------
// LEAN BY DESIGN (C3a): a handler needs, per side, the cooked ShapeType and an
// OPAQUE pointer to that side's cooked geometry payload (primitive params, the
// convex-hull view, or the CookedSdfTable row). C3a stubs read NONE of it, so it
// is intentionally minimal now and GROWS as C3b/c/d wire real inputs:
//   - C3b adds the primitive params (half-extents/radius/half-height + world xform).
//   - C3c adds the convex-hull vertex/face spans.
//   - C3d adds the CookedSdfTable handle + initial-guess seed.
// Keeping it one struct (rather than a per-tier variant) preserves a SINGLE
// handler signature so the table stays a plain function-pointer array. The
// `geom_a`/`geom_b` void* are the seam those subtasks fill; `type_a`/`type_b`
// duplicate the CandidatePair's cooked shape types for the handler's convenience.
//
// C3b GROWTH (additive, INLINE primitive params): the Analytical handlers need,
// per side, the primitive params (box half-extents / sphere radius / capsule
// radius+half-height) + the world transform (baked into amf::PrimFrame columns).
// We add these INLINE (amf::PrimParams prim_a/prim_b) rather than as a void*+cast
// because the cooked store is SoA (CookedShapeTable parallel arrays: types[],
// half_extents[], radii[], half_heights[], + a Transform): there is NO per-shape
// CONTIGUOUS geometry struct to aim a void* at, so the caller would have to
// allocate a temporary just to point at it. Inline params let the caller fill
// from the parallel arrays with zero indirection, keep the view trivially
// copyable (device-upload friendly), and keep the handler math branch-free of a
// cast. The void* geom_a/geom_b SEAM STAYS for C3c (convex-hull vertex/face
// spans) + C3d (CookedSdfTable handle) -- those tiers DO have contiguous backing
// stores, so the void* is the right representation THERE. (C3a defaults preserved
// so the C3a routing test still constructs ShapeProxyView with no args.)
struct ShapeProxyView {
    ShapeType       type_a = ShapeType::Sphere;
    ShapeType       type_b = ShapeType::Sphere;
    const void*     geom_a = nullptr;   // SEAM (C3c/d): cooked geometry payload, side A
    const void*     geom_b = nullptr;   // SEAM (C3c/d): cooked geometry payload, side B
    amf::PrimParams prim_a;             // C3b: inline primitive params + world frame, side A
    amf::PrimParams prim_b;             // C3b: inline primitive params + world frame, side B
};

// The one handler signature. Out-param manifold; the handler fills it (or leaves
// it empty for "no contact"). Host function pointer -> constexpr-table-friendly.
using NarrowphaseFn = void (*)(const CandidatePair& pair,
                               const ShapeProxyView& geom,
                               ContactManifold* out);

// ---------------------------------------------------------------------------
// SelectTier -- pick the precision tier from the cooked shapes (Q2).
//   primitive × primitive            -> Analytical
//   any mesh/convex (no SDF)         -> Convex
//   any SDF-equipped piece           -> Sdf  (overrides, highest precision)
// ---------------------------------------------------------------------------
constexpr bool IsPrimitiveShape(ShapeType t) {
    // Closed-form analytical primitives in v0.8.
    return t == ShapeType::Sphere || t == ShapeType::Capsule ||
           t == ShapeType::Box || t == ShapeType::Plane;
}

constexpr NarrowphaseTier SelectTier(ShapeType a, ShapeType b, bool has_sdf) {
    if (has_sdf) {
        return NarrowphaseTier::Sdf;  // high-precision tier wins when available
    }
    if (IsPrimitiveShape(a) && IsPrimitiveShape(b)) {
        return NarrowphaseTier::Analytical;
    }
    return NarrowphaseTier::Convex;   // mesh / convex-hull / heightfield
}

// ---------------------------------------------------------------------------
// C3a STUB HANDLERS -- one per tier. TODO(C3b/c/d): replace each body with the
// real geometry math. Each writes a WELL-DEFINED EMPTY manifold (no contact
// points) AND stamps an identifiable marker into manifold_key so the routing
// test can assert which tier handled the pair WITHOUT relying on fn-ptr identity
// alone. Defined as `inline` BEFORE the table so taking their address is a valid
// constant expression and the address is the SAME across every TU that includes
// this header (ODR-merged). The stub PRESERVES the pair's collidable sides so a
// downstream consumer still sees correct a/b.
// ---------------------------------------------------------------------------

// Marker stamped into ContactManifold.manifold_key by the C3a stubs so a routing
// test can read back which tier ran. Real handlers (C3b/c/d) compute the REAL
// D1 manifold_key instead; these markers exist ONLY while the handlers are stubs.
inline constexpr uint64_t kStubMarkerBase      = 0xC3A5'7000'0000'0000ull;
inline constexpr uint64_t StubMarkerForTier(NarrowphaseTier tier) {
    return kStubMarkerBase | static_cast<uint64_t>(tier);
}

inline void StubFillEmpty(const CandidatePair& pair, ContactManifold* out,
                          NarrowphaseTier tier) {
    if (out == nullptr) {
        return;
    }
    out->Clear();
    out->a = pair.a;   // preserve type-tagged sides (rigid case: a.handle)
    out->b = pair.b;
    out->manifold_key = StubMarkerForTier(tier);  // routing marker (stub-only)
}

// TODO(C3b): analytical primitive×primitive multi-point / face-face manifolds.
inline void NarrowphaseStubAnalytical(const CandidatePair& pair,
                                      const ShapeProxyView& /*geom*/,
                                      ContactManifold* out) {
    StubFillEmpty(pair, out, NarrowphaseTier::Analytical);
}

// TODO(C3c): convex GJK/EPA/SAT + face-clip multi-point manifolds.
inline void NarrowphaseStubConvex(const CandidatePair& pair,
                                  const ShapeProxyView& /*geom*/,
                                  ContactManifold* out) {
    StubFillEmpty(pair, out, NarrowphaseTier::Convex);
}

// TODO(C3d): SDF high-precision single-witness manifold (wraps find_sdf_contact).
inline void NarrowphaseStubSdf(const CandidatePair& pair,
                               const ShapeProxyView& /*geom*/,
                               ContactManifold* out) {
    StubFillEmpty(pair, out, NarrowphaseTier::Sdf);
}

// ---------------------------------------------------------------------------
// C3b REAL Analytical handlers (primitive x primitive multi-point/face-face).
// ---------------------------------------------------------------------------
// Each handler adapts the ShapeProxyView's inline prim params to the amf:: math
// (analytical_manifold.hpp), then stamps the manifold's a/b sides from the pair.
// The amf:: helpers fill point.normal as "separation dir for the helper's first
// argument". A handler whose helper-first-arg == the pair's side A leaves the
// normal as-is; a handler that must call the helper with the pair's side B first
// (because the helper is written (sphere,box) but the pair is (box,sphere))
// FLIPS the normal so it always means "separation dir for the pair's actual A".
//
// Defined `inline` BEFORE MakeNarrowphaseTable so &Handler is a constant
// expression with one ODR-merged address across TUs (mirrors the stub posture).

// Set the manifold sides from the pair (a/b carry type-tagged CollidableRefs).
inline void StampSides(const CandidatePair& pair, ContactManifold* out) {
    out->a = pair.a;
    out->b = pair.b;
}
// Flip every point's normal in place (used by swapped-order wrappers).
inline void FlipNormals(ContactManifold* out) {
    for (uint32_t i = 0; i < out->point_count; ++i) {
        out->points[i].normal = -out->points[i].normal;
    }
}

// --- sphere x sphere (symmetric: normal flips with order) ---
inline void NarrowphaseSphereSphere(const CandidatePair& pair,
                                    const ShapeProxyView& g, ContactManifold* out) {
    amf::SphereSphere(g.prim_a, g.prim_b, out);  // normal = sep dir for prim_a == A
    StampSides(pair, out);
}

// --- sphere(A) x box(B): helper is (sphere,box); normal = sep dir for sphere == A ---
inline void NarrowphaseSphereBox(const CandidatePair& pair,
                                 const ShapeProxyView& g, ContactManifold* out) {
    amf::SphereBox(g.prim_a, g.prim_b, out);
    StampSides(pair, out);
}
// --- box(A) x sphere(B): call helper (sphere=B, box=A); normal = sep dir for
//     sphere(B); flip so it is sep dir for A(box). ---
inline void NarrowphaseBoxSphere(const CandidatePair& pair,
                                 const ShapeProxyView& g, ContactManifold* out) {
    amf::SphereBox(g.prim_b, g.prim_a, out);
    FlipNormals(out);
    StampSides(pair, out);
}

// --- sphere(A) x plane(B): helper (sphere,plane); normal = sep dir for sphere == A ---
inline void NarrowphaseSpherePlane(const CandidatePair& pair,
                                   const ShapeProxyView& g, ContactManifold* out) {
    amf::SpherePlane(g.prim_a, g.prim_b, out);
    StampSides(pair, out);
}
// --- plane(A) x sphere(B): helper (sphere=B,plane=A); normal = sep dir for
//     sphere(B); flip -> sep dir for A(plane). ---
inline void NarrowphasePlaneSphere(const CandidatePair& pair,
                                   const ShapeProxyView& g, ContactManifold* out) {
    amf::SpherePlane(g.prim_b, g.prim_a, out);
    FlipNormals(out);
    StampSides(pair, out);
}

// --- box(A) x plane(B): helper (box,plane); normal = sep dir for box == A ---
inline void NarrowphaseBoxPlane(const CandidatePair& pair,
                                const ShapeProxyView& g, ContactManifold* out) {
    amf::BoxPlane(g.prim_a, g.prim_b, out);
    StampSides(pair, out);
}
// --- plane(A) x box(B): helper (box=B,plane=A); normal = sep dir for box(B);
//     flip -> sep dir for A(plane). ---
inline void NarrowphasePlaneBox(const CandidatePair& pair,
                                const ShapeProxyView& g, ContactManifold* out) {
    amf::BoxPlane(g.prim_b, g.prim_a, out);
    FlipNormals(out);
    StampSides(pair, out);
}

// --- box x box: helper computes normal = sep dir for prim_a == A. Symmetric. ---
inline void NarrowphaseBoxBox(const CandidatePair& pair,
                              const ShapeProxyView& g, ContactManifold* out) {
    amf::BoxBox(g.prim_a, g.prim_b, out);
    StampSides(pair, out);
}

// --- capsule(A) x plane(B): helper (capsule,plane); normal = sep dir for capsule == A ---
inline void NarrowphaseCapsulePlane(const CandidatePair& pair,
                                    const ShapeProxyView& g, ContactManifold* out) {
    amf::CapsulePlane(g.prim_a, g.prim_b, out);
    StampSides(pair, out);
}
// --- plane(A) x capsule(B): helper (capsule=B,plane=A); flip -> sep dir for A(plane). ---
inline void NarrowphasePlaneCapsule(const CandidatePair& pair,
                                    const ShapeProxyView& g, ContactManifold* out) {
    amf::CapsulePlane(g.prim_b, g.prim_a, out);
    FlipNormals(out);
    StampSides(pair, out);
}

// --- capsule(A) x sphere(B): helper (capsule,sphere); normal = sep dir for capsule == A ---
inline void NarrowphaseCapsuleSphere(const CandidatePair& pair,
                                     const ShapeProxyView& g, ContactManifold* out) {
    amf::CapsuleSphere(g.prim_a, g.prim_b, out);
    StampSides(pair, out);
}
// --- sphere(A) x capsule(B): helper (capsule=B,sphere=A); flip -> sep dir for A(sphere). ---
inline void NarrowphaseSphereCapsule(const CandidatePair& pair,
                                     const ShapeProxyView& g, ContactManifold* out) {
    amf::CapsuleSphere(g.prim_b, g.prim_a, out);
    FlipNormals(out);
    StampSides(pair, out);
}

// ---------------------------------------------------------------------------
// C3c REAL Convex handlers (general convex x convex / convex x primitive).
// ---------------------------------------------------------------------------
// ONE generic handler serves every Convex-tier slot that involves a ConvexHull.
// It maps the pair's ACTUAL side A -> SupportProxy A and side B -> SupportProxy B
// (no order canonicalization), so cvx::ConvexNarrowphase already emits
// point.normal = "separation dir for side A" -- NO flip wrapper is needed for the
// GJK/EPA path (unlike C3b's analytical asymmetric wrappers). The handler reads
// type_a/type_b to pick each side's SupportKind:
//   ShapeType::ConvexHull -> kind=Hull, hull = (const cvx::ConvexHullView*)geom_X
//                            (the C3a void* seam, populated by the caller/C5).
//   ShapeType::Box/Sphere/Capsule -> kind=Box/Sphere/Capsule, prim = &g.prim_X
//                            (the inline C3b prim params -- convex x primitive is
//                            in C3c scope; the primitive's analytical support
//                            feeds the SAME Minkowski GJK/EPA path).
//
// PLANE is the one special case: a MuJoCo plane is an INFINITE half-space, so the
// Minkowski difference is unbounded and GJK never deterministically encloses the
// origin (convex_narrowphase.hpp HullPlane header note). We special-case it:
// call cvx::HullPlane(hull, plane, normal_for_hull). The plane normal is the
// separation dir for the HULL side; we re-sign it to "separation dir for the
// pair's ACTUAL side A":
//   hull is side A (plane is B): normal_for_hull = +plane_normal  (A==hull pushes off plane)
//   plane is side A (hull is B): A==plane; sep dir for A = -plane_normal (plane pushes
//                                opposite the hull); HullPlane fills "sep dir for hull"
//                                so we pass -plane_normal and DON'T flip (HullPlane
//                                applies it verbatim to every point).
// Both orderings are registered + tested (the sign bug hides in the swapped slot).

inline cvx::SupportProxy MakeConvexProxy(ShapeType t, const void* geom,
                                         const amf::PrimParams& prim) {
    cvx::SupportProxy p;
    switch (t) {
        case ShapeType::ConvexHull:
        case ShapeType::TriMesh:        // (Convex tier; TriMesh deferred -- see table)
        case ShapeType::HeightField:
            p.kind = cvx::SupportKind::Hull;
            p.hull = static_cast<const cvx::ConvexHullView*>(geom);
            break;
        case ShapeType::Box:
            p.kind = cvx::SupportKind::Box;    p.prim = &prim; break;
        case ShapeType::Sphere:
            p.kind = cvx::SupportKind::Sphere; p.prim = &prim; break;
        case ShapeType::Capsule:
            p.kind = cvx::SupportKind::Capsule; p.prim = &prim; break;
        case ShapeType::Plane:
            // Plane never goes through GJK; this branch is unreached (the handler
            // detects Plane first). Map to Box as a benign default.
            p.kind = cvx::SupportKind::Box;    p.prim = &prim; break;
    }
    return p;
}

// The single generic Convex-tier handler (registered in every ConvexHull slot).
inline void NarrowphaseConvex(const CandidatePair& pair, const ShapeProxyView& g,
                              ContactManifold* out) {
    // --- Plane special case (either side) -- NOT through GJK (unbounded CSO). ---
    if (g.type_a == ShapeType::Plane || g.type_b == ShapeType::Plane) {
        const bool plane_is_a = (g.type_a == ShapeType::Plane);
        // The non-plane side must be a hull (Convex tier => one side is ConvexHull;
        // plane x primitive routes Analytical, plane x plane is the stub).
        const ShapeType hull_t = plane_is_a ? g.type_b : g.type_a;
        const void* hull_geom  = plane_is_a ? g.geom_b : g.geom_a;
        if (hull_t != ShapeType::ConvexHull && hull_t != ShapeType::TriMesh &&
            hull_t != ShapeType::HeightField) {
            // Not a hull-vs-plane (e.g. (Plane,Plane)) -> empty manifold.
            out->Clear(); StampSides(pair, out); return;
        }
        const cvx::ConvexHullView* hull =
            static_cast<const cvx::ConvexHullView*>(hull_geom);
        const amf::PrimParams& plane = plane_is_a ? g.prim_a : g.prim_b;
        if (hull == nullptr) { out->Clear(); StampSides(pair, out); return; }
        // plane_normal = separation dir for the HULL side. Re-sign to side A:
        //   hull == A (plane is B): +plane_normal ; plane == A: -plane_normal.
        const math::Vec3 plane_n = amf::Norm(plane.frame.cy, math::Vec3::UnitY());
        const math::Vec3 normal_for_hull = plane_is_a ? -plane_n : plane_n;
        cvx::HullPlane(*hull, plane, normal_for_hull, out);
        StampSides(pair, out);
        return;
    }
    // --- General GJK/EPA/face-clip path. Side A == pair side A (no flip). ---
    cvx::SupportProxy A = MakeConvexProxy(g.type_a, g.geom_a, g.prim_a);
    cvx::SupportProxy B = MakeConvexProxy(g.type_b, g.geom_b, g.prim_b);
    // A null hull view (caller didn't populate the seam) -> empty manifold.
    if ((A.kind == cvx::SupportKind::Hull && A.hull == nullptr) ||
        (B.kind == cvx::SupportKind::Hull && B.hull == nullptr)) {
        out->Clear(); StampSides(pair, out); return;
    }
    cvx::ConvexNarrowphase(A, B, out);
    StampSides(pair, out);
}

// ---------------------------------------------------------------------------
// The dispatch table: table[typeA][typeB][tier] -> handler.
// ---------------------------------------------------------------------------
// Representation: a dense 3-D constexpr array of function pointers. O(1) lookup
// by integer index. Every slot defaults to its TIER's stub (so an
// un-specialized (type,type) pair still routes to a well-defined tier handler);
// C3b/c/d OVERRIDE specific (typeA,typeB,tier) slots with their real handlers --
// that override is the localized "new table row" edit. A struct wrapper keeps the
// array a single constexpr object with a documented Lookup().
struct NarrowphaseTable {
    NarrowphaseFn fns[kShapeTypeCount][kShapeTypeCount][kNarrowphaseTierCount];

    constexpr NarrowphaseFn Lookup(ShapeType a, ShapeType b, NarrowphaseTier tier) const {
        return fns[static_cast<uint32_t>(a)]
                  [static_cast<uint32_t>(b)]
                  [static_cast<uint32_t>(tier)];
    }
};

// Build the table at compile time. Default every slot to its tier stub; C3b/c/d
// add real-handler overrides HERE (e.g. `t.fns[(int)ShapeType::Sphere]
// [(int)ShapeType::Sphere][(int)NarrowphaseTier::Analytical] = &SphereSphere;`).
constexpr NarrowphaseTable MakeNarrowphaseTable() {
    NarrowphaseTable t{};
    for (uint32_t ia = 0; ia < kShapeTypeCount; ++ia) {
        for (uint32_t ib = 0; ib < kShapeTypeCount; ++ib) {
            t.fns[ia][ib][static_cast<uint32_t>(NarrowphaseTier::Analytical)] =
                &NarrowphaseStubAnalytical;
            t.fns[ia][ib][static_cast<uint32_t>(NarrowphaseTier::Convex)] =
                &NarrowphaseStubConvex;
            t.fns[ia][ib][static_cast<uint32_t>(NarrowphaseTier::Sdf)] =
                &NarrowphaseStubSdf;
        }
    }
    // --- C3b/C3c/C3d real-handler registration goes here (localized edits) ---
    // C3b: Analytical-tier primitive x primitive handlers. BOTH orderings of each
    // asymmetric pair are registered to a distinct wrapper that keeps the normal =
    // "separation dir for the pair's ACTUAL side A" (swapped wrappers flip it). We
    // register both slots (rather than canonicalize inside one handler) so the
    // table stays a pure index->fn lookup with no in-handler order branch.
    const uint32_t kAna = static_cast<uint32_t>(NarrowphaseTier::Analytical);
    const uint32_t kSph = static_cast<uint32_t>(ShapeType::Sphere);
    const uint32_t kCap = static_cast<uint32_t>(ShapeType::Capsule);
    const uint32_t kBox = static_cast<uint32_t>(ShapeType::Box);
    const uint32_t kPln = static_cast<uint32_t>(ShapeType::Plane);

    t.fns[kSph][kSph][kAna] = &NarrowphaseSphereSphere;
    t.fns[kSph][kBox][kAna] = &NarrowphaseSphereBox;
    t.fns[kBox][kSph][kAna] = &NarrowphaseBoxSphere;
    t.fns[kSph][kPln][kAna] = &NarrowphaseSpherePlane;
    t.fns[kPln][kSph][kAna] = &NarrowphasePlaneSphere;
    t.fns[kBox][kPln][kAna] = &NarrowphaseBoxPlane;
    t.fns[kPln][kBox][kAna] = &NarrowphasePlaneBox;
    t.fns[kBox][kBox][kAna] = &NarrowphaseBoxBox;
    t.fns[kCap][kPln][kAna] = &NarrowphaseCapsulePlane;
    t.fns[kPln][kCap][kAna] = &NarrowphasePlaneCapsule;
    t.fns[kCap][kSph][kAna] = &NarrowphaseCapsuleSphere;
    t.fns[kSph][kCap][kAna] = &NarrowphaseSphereCapsule;

    // DEFERRED to the C3a stub (NarrowphaseStubAnalytical) WITH a named consumer:
    //   (Capsule,Box)/(Box,Capsule)  -> C3c convex tier (capsule treated as a
    //                                    swept-segment vs OBB; the GJK/face-clip
    //                                    util generalises this cleanly).
    //   (Capsule,Capsule)            -> C3c convex tier (segment-segment closest
    //                                    point + 2-pt clip; shares the C3c clip).
    //   (Plane,Plane)                -> physically degenerate (two half-spaces);
    //                                    no finite manifold -> stub by design.
    // These slots intentionally KEEP the default Analytical stub so they route to
    // a well-defined empty manifold until C3c lands the segment-based handlers.
    //
    // C3c NOTE / HONEST ORPHAN (capsule x box, capsule x capsule): the C3b comment
    // above names "C3c convex tier" as the consumer for these. BUT SelectTier
    // routes BOTH capsule x box and capsule x capsule to the ANALYTICAL tier
    // (capsule + box + capsule are all IsPrimitiveShape == true). So a Convex-tier
    // registration for them would be UNREACHABLE dead code -- ResolveNarrowphase
    // never asks the Convex tier for two primitives. C3c therefore does NOT close
    // this with a phantom Convex-tier slot. Closing it correctly is either
    // (a) Analytical-tier wrappers calling cvx::ConvexNarrowphase (1-pt witness,
    // since capsule HasFace()==false -> weaker than C3b's envisioned swept-segment
    // vs OBB face contact), or (b) a true analytical capsule-OBB handler. Both are
    // OUT OF C3c SCOPE (C3c is the general convex path); the gap is surfaced to the
    // controller and remains the C3b deferral's open item -- NOT silently shipped.

    // --- C3c: Convex-tier handlers (general convex x convex / convex x primitive).
    // The SAME generic NarrowphaseConvex serves every ConvexHull-involving slot
    // (it reads type_a/type_b to pick each side's support kind; the plane case is
    // special-cased inside). Registered for BOTH orderings of each asymmetric pair
    // -- but no flip wrapper: side A == pair side A, so the normal is already the
    // separation dir for side A (the cleaner C3c posture vs C3b's flip wrappers).
    const uint32_t kCvx = static_cast<uint32_t>(NarrowphaseTier::Convex);
    const uint32_t kHull = static_cast<uint32_t>(ShapeType::ConvexHull);

    t.fns[kHull][kHull][kCvx] = &NarrowphaseConvex;   // convex x convex
    t.fns[kHull][kBox][kCvx]  = &NarrowphaseConvex;   // convex x box (both orders)
    t.fns[kBox][kHull][kCvx]  = &NarrowphaseConvex;
    t.fns[kHull][kSph][kCvx]  = &NarrowphaseConvex;   // convex x sphere
    t.fns[kSph][kHull][kCvx]  = &NarrowphaseConvex;
    t.fns[kHull][kCap][kCvx]  = &NarrowphaseConvex;   // convex x capsule
    t.fns[kCap][kHull][kCvx]  = &NarrowphaseConvex;
    t.fns[kHull][kPln][kCvx]  = &NarrowphaseConvex;   // convex x plane (special-cased)
    t.fns[kPln][kHull][kCvx]  = &NarrowphaseConvex;

    // DEFERRED in the Convex tier WITH a NAMED CONSUMER:
    //   TriMesh x * / HeightField x * -> KEEP the Convex stub. v0.8 grasp consumes
    //     convex HULL pieces from V-HACD (CookedConvexGeometry), NOT raw trimesh
    //     narrowphase; a raw-trimesh general narrowphase is a v0.9 concern (BVH-of-
    //     triangles + per-triangle convex). MakeConvexProxy maps TriMesh/HeightField
    //     to a Hull view so wiring them later is a one-line table edit once a cooked
    //     triangle-soup view exists. Until then these slots route to the well-defined
    //     Convex stub (empty manifold), NOT a phantom handler.
    return t;
}

// The single constexpr instance. `inline` so it is one object across all TUs.
inline constexpr NarrowphaseTable kNarrowphaseTable = MakeNarrowphaseTable();

// Convenience: resolve the handler for a pair given its cooked shape types + the
// SDF-availability flag. Pure -> deterministic.
constexpr NarrowphaseFn ResolveNarrowphase(ShapeType a, ShapeType b, bool has_sdf) {
    return kNarrowphaseTable.Lookup(a, b, SelectTier(a, b, has_sdf));
}

} // namespace nuka::collision
