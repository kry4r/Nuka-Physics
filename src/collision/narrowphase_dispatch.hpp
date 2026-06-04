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

#include "collision/candidate_pair.hpp"     // CandidatePair, CollidableRef
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
struct ShapeProxyView {
    ShapeType   type_a = ShapeType::Sphere;
    ShapeType   type_b = ShapeType::Sphere;
    const void* geom_a = nullptr;   // SEAM (C3b/c/d): cooked geometry payload, side A
    const void* geom_b = nullptr;   // SEAM (C3b/c/d): cooked geometry payload, side B
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
    // t.fns[(int)ShapeType::Sphere][(int)ShapeType::Sphere]
    //      [(int)NarrowphaseTier::Analytical] = &NarrowphaseSphereSphere;   // C3b
    // ... (one assignment per specialized (typeA,typeB,tier))
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
