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
#include "collision/sdf_contact.hpp"        // C3d: find_sdf_contact_newton + SdfBodyFrame
#include "constraint/contact_manifold.hpp"  // ContactManifold (extended, C3a)
#include "runtime/sdf/sparse_sdf_query.cuh" // C3d: SparseSdfDevice (the SDF view)
#include "scene/canonical_types.hpp"        // scene::ShapeType

#include <cstdint>
#include <cstdio>

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
//
// C3d GROWTH (Sdf tier, REUSES the void* seam): the Sdf-tier handler needs, per
// side, the cooked SDF's device view (SparseSdfDevice) + the body world<->local
// frame (SdfBodyFrame). Both have a contiguous backing struct, so -- exactly as
// the void* seam was reserved for "tiers that DO have a contiguous backing store"
// -- we point geom_a/geom_b at an SdfProxyView{sdf, frame} rather than growing the
// view with two more inline structs. The Sdf handler reads geom_X as a
// `const SdfProxyView*` (valid ONLY in the Sdf tier, which is the ONLY tier whose
// handler this struct feeds -- has_sdf overrides tier, so any SDF-equipped pair
// routes to the Sdf plane and nowhere else reads geom_X as an SDF view). A null
// geom (caller didn't populate the seam) -> empty manifold (mirrors C3c's hull
// null-guard). This keeps ONE handler signature: the void* is reinterpreted
// per-tier, never widening the shared struct.
struct SdfProxyView {
    runtime::sdf::SparseSdfDevice sdf{};   // the cooked SDF device/host view, this side
    collision::SdfBodyFrame       frame{}; // this body's world<->local frame
};

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
//   primitive × primitive            -> Analytical  (closed-form)
//   capsule×box / capsule×capsule    -> Convex      (no closed-form; GJK/EPA, C3c)
//   any mesh/convex (no SDF)         -> Convex
//   has_sdf == true                  -> Sdf  (overrides, highest precision)
//
// `has_sdf` CONTRACT (C5 stream driver derives + passes it): TRUE iff BOTH sides
// have a cooked SDF (CookedSdfTable, kNoSdf sentinel). The Sdf tier's Newton solve
// needs both fields (sdf_a AND sdf_b); a pair with only ONE SDF side must NOT set
// has_sdf (it falls through to Convex/Analytical). The NarrowphaseSdf handler's
// null-SDF-view guard (-> empty manifold) is a safety net, NOT the routing rule.
// ---------------------------------------------------------------------------
constexpr bool IsPrimitiveShape(ShapeType t) {
    // Closed-form analytical primitives in v0.8.
    return t == ShapeType::Sphere || t == ShapeType::Capsule ||
           t == ShapeType::Box || t == ShapeType::Plane;
}

// Capsule×Box and Capsule×Capsule have NO closed-form analytical contact (a
// segment-vs-OBB / segment-vs-segment + radius is exactly what GJK/EPA does well),
// so they route to the Convex tier (C3c handles capsules via SupportCapsule). All
// OTHER capsule pairs DO have closed-form handlers: capsule×plane, capsule×sphere
// (C3b) -> Analytical.
constexpr bool IsConvexOnlyPrimitivePair(ShapeType a, ShapeType b) {
    const bool a_cap = (a == ShapeType::Capsule);
    const bool b_cap = (b == ShapeType::Capsule);
    const bool capsule_box =
        (a_cap && b == ShapeType::Box) || (a == ShapeType::Box && b_cap);
    return (a_cap && b_cap) || capsule_box;
}

constexpr NarrowphaseTier SelectTier(ShapeType a, ShapeType b, bool has_sdf) {
    if (has_sdf) {
        return NarrowphaseTier::Sdf;  // high-precision tier wins when available
    }
    if (IsConvexOnlyPrimitivePair(a, b)) {
        return NarrowphaseTier::Convex;  // no closed-form -> GJK/EPA (C3c)
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

// DEFERRED-PAIR SENTINEL (NOT a real contact key). The only pairs that still
// route to a stub are PRINCIPLED DEFERRALS with NAMED CONSUMERS: (Plane,Plane)
// (two half-spaces -> no finite manifold, degenerate by design) and TriMesh /
// HeightField narrowphase (deferred to v0.9 raw-trimesh / C7b-2; see the table's
// "DEFERRED ... WITH A NAMED CONSUMER" note). A stub writes this distinguishable
// sentinel into manifold_key so a consumer (and the routing test) can tell a
// deferred-pair empty manifold from a real handler's no-contact (real handlers
// stamp MakeStableKey via StampSides). The high bits make it visually unmistakable
// vs a real (min,max handle) key, and StubFillEmpty emits a LOUD diagnostic so a
// deferred pair reaching the stub at runtime is NEVER silent.
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
    out->manifold_key = StubMarkerForTier(tier);  // deferred-pair sentinel (see above)
}

// Emit a LOUD, once-per-shape-pair diagnostic so a DEFERRED shape pair reaching a
// stub at runtime is surfaced (it produces an empty manifold == no contact, a
// silent correctness gap for the deferred pairs). Gated by a per-(a,b) latch so a
// dense scene does not spam. NOT an abort: the deferral is principled (named
// consumer), so production keeps stepping while the gap is visible.
inline void WarnDeferredPair(const char* tier, ShapeType a, ShapeType b) {
    static bool warned[kShapeTypeCount][kShapeTypeCount] = {};
    const uint32_t ia = static_cast<uint32_t>(a), ib = static_cast<uint32_t>(b);
    if (ia < kShapeTypeCount && ib < kShapeTypeCount && !warned[ia][ib]) {
        warned[ia][ib] = true;
        std::fprintf(stderr,
                     "[nuka_narrowphase] DEFERRED %s pair (ShapeType %u x %u) -> "
                     "EMPTY manifold (NO contact). Unimplemented; named consumer = "
                     "v0.9 trimesh narrowphase / C7b-2.\n", tier, ia, ib);
    }
}

// DEFERRED Analytical stub: only (Plane,Plane) reaches here (degenerate, no finite
// manifold). All other primitive pairs have real C3b handlers.
inline void NarrowphaseStubAnalytical(const CandidatePair& pair,
                                      const ShapeProxyView& geom,
                                      ContactManifold* out) {
    WarnDeferredPair("Analytical", geom.type_a, geom.type_b);
    StubFillEmpty(pair, out, NarrowphaseTier::Analytical);
}

// DEFERRED Convex stub: only TriMesh / HeightField slots reach here (raw-trimesh
// narrowphase deferred to v0.9; convex HULL pairs have the real NarrowphaseConvex).
inline void NarrowphaseStubConvex(const CandidatePair& pair,
                                  const ShapeProxyView& geom,
                                  ContactManifold* out) {
    WarnDeferredPair("Convex", geom.type_a, geom.type_b);
    StubFillEmpty(pair, out, NarrowphaseTier::Convex);
}

// C3d REAL Sdf handler: NarrowphaseSdf -- defined below, AFTER StampSides (the
// shared a/b-stamp helper it reuses). It REPLACES the Sdf stub across the whole
// Sdf plane of the table in MakeNarrowphaseTable() (has_sdf overrides tier, so
// any SDF-equipped pair routes to the Sdf plane). See its header comment there.

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

// Set the manifold sides from the pair (a/b carry type-tagged CollidableRefs)
// AND stamp the D1 stream sort key (§0.1 `manifold_key = (min,max handle)+type`).
// Every REAL handler calls this after filling geometry, so the manifold the C3->C4
// stream carries is self-D1-sortable: manifold_key == the originating
// CandidatePair's stable_key (the same canonical (type,handle) pack). The C3a STUBS
// instead stamp a tier marker into manifold_key (they do NOT call StampSides), so a
// real manifold and a stub manifold are always distinguishable. (Were this left
// 0/unset, the manifold stream's deterministic sort would be unmet — the key is the
// detection->solver identity, not just decoration.)
inline void StampSides(const CandidatePair& pair, ContactManifold* out) {
    out->a = pair.a;
    out->b = pair.b;
    out->manifold_key = MakeStableKey(pair.a, pair.b);
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
    // --- Sphere special case (either side) -- NOT through GJK/EPA. ---
    // A sphere fed to the general boolean-GJK->EPA path has a shallow-penetration
    // non-monotonicity (the curved support breaks EPA's incremental expansion in a
    // ~1mm dead band -> a periodic single-step contact dropout). The robust
    // formulation is the EXACT point-to-convex distance: closest point on the hull
    // to the sphere CENTER, contact iff |center-closest| < radius. This BYPASSES
    // EPA entirely and is monotone at all depths. Only sphere x ConvexHull (and the
    // deferred TriMesh/HeightField hull slots) reaches here: sphere x box and
    // sphere x sphere route to the Analytical tier (SelectTier), and sphere x plane
    // is caught by the plane case above. We special-case the sphere-vs-hull slots.
    // NOTE: only Sphere is routed out of EPA here. capsule x hull (and box x hull /
    // convex x convex) still ride the general EPA path below and retain the SAME
    // shallow-penetration dead band; named consumer for hardening that = C7b-2
    // (real-hand grasp + place-on-counter).
    if (g.type_a == ShapeType::Sphere || g.type_b == ShapeType::Sphere) {
        const bool sphere_is_a = (g.type_a == ShapeType::Sphere);
        const ShapeType hull_t = sphere_is_a ? g.type_b : g.type_a;
        const void* hull_geom  = sphere_is_a ? g.geom_b : g.geom_a;
        // The non-sphere side must be a hull view (ConvexHull / deferred mesh slots
        // map to a hull view). A non-hull other side (e.g. sphere x sphere/box --
        // routed Analytical, never here) falls through to the general path.
        if (hull_t == ShapeType::ConvexHull || hull_t == ShapeType::TriMesh ||
            hull_t == ShapeType::HeightField) {
            const cvx::ConvexHullView* hull =
                static_cast<const cvx::ConvexHullView*>(hull_geom);
            if (hull == nullptr) { out->Clear(); StampSides(pair, out); return; }
            const amf::PrimParams& sphere = sphere_is_a ? g.prim_a : g.prim_b;
            cvx::SphereHull(sphere, *hull, sphere_is_a, out);
            StampSides(pair, out);
            return;
        }
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
// C3d REAL Sdf handler (high-precision single-witness Newton-on-summed-SDF).
// ---------------------------------------------------------------------------
// Wraps find_sdf_contact_newton (sdf_contact.hpp) -- the EXISTING D1, HD,
// differentiable-ready SDF narrowphase that, until C3d, had ZERO stepping
// callsites (architecture §0(c) dead code). This handler is its FIRST caller:
// for an SDF-equipped pair it reads each side's SparseSdfDevice view + body
// frame (from the SdfProxyView the caller aimed geom_a/geom_b at), seeds the
// Newton solve, and converts the single SdfContactResult witness into a 1-point
// ContactManifold (OPEN-I: single witness -> 1 point for v0.8 grasp; multi-point
// via perturbed restarts is v0.9).
//
// INITIAL GUESS. The spec says "midpoint of the pair AABBs". The ShapeProxyView
// carries no AABBs (it is the lean cooked-geometry view), so we use the MIDPOINT
// OF THE TWO BODY FRAME TRANSLATIONS -- the body/piece centers in world. For the
// just-touching SDF-equipped pieces this tier targets (fingertip vs cup piece),
// that midpoint lands in both narrow bands (the bands straddle the contact
// surface, which sits between the two centers). DOCUMENTED CHOICE. (If a future
// caller carries true AABB centers, swapping this one line to their midpoint is
// the localized edit; the contact surface still lies between the AABB centers.)
//   Band caveat (D1, not a regression): find_sdf_contact_newton requires the
//   INITIAL iterate to be in BOTH narrow bands (sdf_contact.hpp:184-187 returns
//   invalid otherwise; the halving line-search only engages after one in-band
//   step). For widely-separated centers the midpoint may miss the bands -> valid
//   == false -> EMPTY manifold (correct "no contact at this seed"); single-
//   witness SDF is a high-precision near-contact tier, not a global search.
//
// NORMAL SIGN (the load-bearing mapping). SdfContactResult.normal is unit and in
// the "a<-b sense" == the SEPARATION direction for body a (sdf_contact.hpp:106,
// 230-238: normalize(grad_b - grad_a)). The ContactManifold convention is
// "separation dir for side A" (contact_manifold.hpp:44). We bind the pair's
// side A -> sdf_a (geom_a) and side B -> sdf_b (geom_b), so SDF body a == pair
// side A: the SDF normal is ALREADY "separation dir for side A" -- NO FLIP. (A
// flip would be needed only if the caller swapped the binding; verified against
// the box-on-ground analytical case (SDF-tier forward gate -> M11): top box A on ground
// box B -> normal.y > 0, i.e. A separates +Y, matching grad_b(+Y)-grad_a(-Y).)
//
// p08-C ADJOINT / DIFFERENTIABILITY SEAM (NOT dropped). The Q5 differentiable
// tier rides p08's IFT-at-convergence adjoint, which needs the converged
// stationarity-residual state (phi_a/phi_b, grad_a/grad_b world, step_curvature)
// -- fields the ContactManifold does NOT carry (it is the solver hand-off, not
// the adjoint state). We do NOT widen ContactManifold for them (blast radius +
// breaks its trivially-copyable D1 contract). The adjoint state is RECOVERED by
// DETERMINISTIC RE-QUERY: find_sdf_contact_newton is D1 (fixed-iter/tol/clamp,
// no atomics), so re-running it from the same (sdf_a, frame_a, sdf_b, frame_b,
// guess) yields a BYTE-IDENTICAL SdfContactResult carrying the full adjoint
// state. That is exactly what test_sdf_contact_adjoint_fd already does (re-runs
// the descent under perturbation). So the differentiability seam is preserved at
// the C4/p08-adjoint consumer via re-query, NOT silently discarded here.
inline void NarrowphaseSdf(const CandidatePair& pair, const ShapeProxyView& g,
                           ContactManifold* out) {
    if (out == nullptr) {
        return;
    }
    out->Clear();
    StampSides(pair, out);
    // Tag the reaction providers from each side's CollidableType static metadata
    // (the Sdf tier is class-blind: rigid piece, articulation-link fingertip,
    // etc. -- the solver reads .react off the ref to build each row side).
    out->a.react = constraint::GetCollidableTypeInfo(pair.a.type).react;
    out->b.react = constraint::GetCollidableTypeInfo(pair.b.type).react;

    // Read the per-side SDF views from the void* seam. Null (unpopulated) ->
    // empty manifold, no crash (mirrors C3c's hull null-guard; also the
    // separated/no-SDF path).
    const auto* sa = static_cast<const SdfProxyView*>(g.geom_a);
    const auto* sb = static_cast<const SdfProxyView*>(g.geom_b);
    if (sa == nullptr || sb == nullptr) {
        return;
    }

    // Initial guess = midpoint of the two body/piece centers (frame translations).
    const math::Vec3 ta = sa->frame.translation;
    const math::Vec3 tb = sb->frame.translation;
    const math::Vec3 guess{0.5f * (ta.x + tb.x), 0.5f * (ta.y + tb.y),
                           0.5f * (ta.z + tb.z)};

    const SdfContactResult r = find_sdf_contact_newton(
        sa->sdf, sa->frame, sb->sdf, sb->frame, guess, SdfContactParams{});

    // No contact (not converged in-band, or not penetrating) -> empty manifold.
    if (!(r.valid && r.penetrating)) {
        return;  // point_count stays 0 (Clear above)
    }

    // Single witness -> 1-point manifold. normal already "sep dir for side A".
    constraint::ContactPoint pt{};
    pt.position    = r.point;
    pt.normal      = r.normal;
    pt.penetration = r.penetration;
    out->AddPoint(pt);  // point_count -> 1
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
            // C3d: the WHOLE Sdf plane defaults to the REAL NarrowphaseSdf handler
            // (not a stub). RATIONALE: SelectTier routes ANY has_sdf==true pair to
            // the Sdf tier REGARDLESS of (typeA,typeB) -- the SDF view is the only
            // geometry the Newton solve reads, so the shapes' enum tags don't pick
            // a different SDF math. One generic handler therefore covers every
            // (typeA,typeB,Sdf) slot (the broad "Sdf plane" the spec asks for), so
            // the right registration is to fill the plane in this default loop --
            // exactly how the table's per-tier default-fill works. NarrowphaseSdf
            // null-guards an unpopulated SdfProxyView seam -> empty manifold, so a
            // slot reached without SDF data still behaves like the old stub (empty
            // manifold) but is no longer the dead-marker stub.
            t.fns[ia][ib][static_cast<uint32_t>(NarrowphaseTier::Sdf)] =
                &NarrowphaseSdf;
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

    // CLOSED (C3-batch review fix): (Capsule,Box)/(Box,Capsule)/(Capsule,Capsule)
    //   have NO closed-form analytical contact, so SelectTier now routes them to the
    //   CONVEX tier (IsConvexOnlyPrimitivePair) and they are registered to the
    //   generic NarrowphaseConvex below -- C3c's GJK/EPA already supports a capsule
    //   side via SupportCapsule (so capsule+box / capsule+capsule are two primitive
    //   SupportProxies through the SAME Minkowski path; 1-pt witness for now, which
    //   is sufficient for v0.8 and is what GJK yields without a swept-segment face).
    //   This removes the silent-empty-manifold landmine before the table reaches
    //   production (C5). (Plane,Plane) stays the Analytical stub by design (two
    //   half-spaces -> no finite manifold).

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

    // C3-batch review fix: capsule×box / capsule×capsule have no closed-form, so
    // SelectTier routes them HERE (Convex). Two primitive SupportProxies (no hull)
    // through the SAME generic NarrowphaseConvex (MakeConvexProxy maps Box->Box,
    // Capsule->Capsule). Closes the C3b/C3c deferral so C5 wiring can't silently
    // emit empty manifolds for these pairs.
    t.fns[kCap][kBox][kCvx]   = &NarrowphaseConvex;   // capsule x box (both orders)
    t.fns[kBox][kCap][kCvx]   = &NarrowphaseConvex;
    t.fns[kCap][kCap][kCvx]   = &NarrowphaseConvex;   // capsule x capsule

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
