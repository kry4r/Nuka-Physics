// ---------------------------------------------------------------------------
// nuka::collision -- the contact stream DRIVER implementation (v0.8 C5a)
// ---------------------------------------------------------------------------
// See contact_stream_driver.hpp for the design + scope (host-only, resolver-
// decoupled, validated-not-wired). This TU holds only the BuildContactManifolds
// body so the header stays light and the dispatch-table/MergeContactParams
// includes do not leak into every consumer.
// ---------------------------------------------------------------------------

#include "collision/contact_stream_driver.hpp"

#include "scene/contact_filter.hpp"  // scene::MergeContactParams + ContactParamsIn

namespace nuka::collision {

namespace {

// Pack one resolved side's MuJoCo per-geom params into the MergeContactParams
// input struct. condim is uint8 in ContactParamsIn (MuJoCo's 1/3/4/6); the
// ResolvedShape carries it as uint32 for convenience -- narrow it here.
scene::ContactParamsIn ToMergeInput(const ResolvedShape& s) {
    scene::ContactParamsIn in;
    in.priority    = s.priority;
    in.condim      = static_cast<uint8_t>(s.condim);
    in.friction_mu = s.friction;
    in.solref[0]   = s.solref[0];
    in.solref[1]   = s.solref[1];
    for (int i = 0; i < 5; ++i) in.solimp[i] = s.solimp[i];
    in.solmix      = s.solmix;
    // margin/gap now carried on ResolvedShape (C5c): forward them so the merge
    // (max of both sides) survives onto the manifold below.
    in.margin      = s.margin;
    in.gap         = s.gap;
    return in;
}

}  // namespace

void BuildContactManifolds(std::span<const CandidatePair> pairs,
                           const ShapeResolver& resolve,
                           std::vector<ContactManifold>* out) {
    if (out == nullptr || !resolve) {
        return;
    }

    // Fixed sequential walk over the (already D1-sorted) pairs: no re-sort, no
    // atomics, deterministic emission order.
    for (const CandidatePair& pair : pairs) {
        // 1) Resolve both sides. Either side without geometry -> skip the pair.
        ResolvedShape ra;
        ResolvedShape rb;
        if (!resolve(pair.a, &ra) || !resolve(pair.b, &rb)) {
            continue;
        }

        // 2) has_sdf is the BOTH-sides contract (narrowphase_dispatch.hpp): the
        //    Sdf tier's Newton solve needs BOTH SDF views, so a one-sided SDF
        //    pair must NOT route to Sdf -- it falls through to Convex/Analytical.
        const bool has_sdf = ra.has_sdf && rb.has_sdf;

        // 3) Build the per-shape view: types + inline prim params + the opaque
        //    geom pointers (hull view / SdfProxyView), side A == pair.a verbatim
        //    (no order canonicalization -- the table registers both orderings and
        //    the handlers keep the normal as "separation dir for side A").
        ShapeProxyView view;
        view.type_a = ra.type;
        view.type_b = rb.type;
        view.geom_a = ra.geom;
        view.geom_b = rb.geom;
        view.prim_a = ra.prim;
        view.prim_b = rb.prim;

        // 4) Resolve the handler through the constexpr dispatch table and invoke
        //    it. The handler fills the manifold geometry AND stamps m.a/m.b +
        //    the D1 manifold_key (via StampSides) -- we do NOT touch those here.
        //    Value-init so the unused points[] tail is deterministic zero bytes
        //    (the D1 two-run memcmp gate relies on this).
        ContactManifold m{};
        const NarrowphaseFn fn = ResolveNarrowphase(ra.type, rb.type, has_sdf);
        fn(pair, view, &m);

        // 5) No contact point -> not a real contact; skip (do not append).
        if (m.point_count == 0u) {
            continue;
        }

        // 6) Merge the two sides' MuJoCo per-geom contact params (priority /
        //    solmix / condim / friction / solref / solimp) into the manifold.
        //    We CALL scene::MergeContactParams -- the same merge the cook-time
        //    filter uses -- never reimplement it. Map merged.friction_mu ->
        //    m.friction, merged.solimp -> m.solimp, and the merged solref onto
        //    EVERY point's solref_timeconst/dampratio. merged.margin/gap are now
        //    carried onto the manifold (C5c); merged.condim still has no field
        //    (consumed downstream by the row builder, not stored here) -> dropped.
        const scene::MergedContactParams merged =
            scene::MergeContactParams(ToMergeInput(ra), ToMergeInput(rb));
        m.friction = merged.friction_mu;
        m.margin   = merged.margin;
        m.gap      = merged.gap;
        for (int i = 0; i < 5; ++i) m.solimp[i] = merged.solimp[i];
        for (uint32_t p = 0; p < m.point_count; ++p) {
            m.points[p].solref_timeconst = merged.solref[0];
            m.points[p].solref_dampratio = merged.solref[1];
        }

        out->push_back(m);
    }
}

}  // namespace nuka::collision
