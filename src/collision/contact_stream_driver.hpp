#pragma once
// ---------------------------------------------------------------------------
// nuka::collision -- the contact stream DRIVER (v0.8 C5a)
// ---------------------------------------------------------------------------
// The piece that closes the v0.8 detection->rows spine on the EMIT side: it
// turns a stream of CandidatePairs (C2 broadphase output) into a stream of
// ContactManifolds by, per pair:
//   1. RESOLVING each side's geometry + world frame + per-geom contact params
//      (via a caller-supplied ShapeResolver -- the decoupling seam, see below),
//   2. deriving has_sdf = a.has_sdf && b.has_sdf (the BOTH-sides contract that
//      narrowphase_dispatch.hpp documents -- one-sided SDF falls through to the
//      analytical/convex tier),
//   3. building a ShapeProxyView and selecting the handler through the constexpr
//      dispatch table (ResolveNarrowphase -> SelectTier -> kNarrowphaseTable),
//   4. invoking the handler to fill the manifold geometry (which also stamps the
//      D1 manifold_key + the a/b sides via the handler's StampSides),
//   5. MERGING the two sides' MuJoCo per-geom contact params into the manifold
//      via scene::MergeContactParams (the same merge the cook-time filter uses),
//   6. appending the manifold if it has >=1 contact point (empty == no contact).
//
// ---------------------------------------------------------------------------
// THE RESOLVER DECOUPLING (the load-bearing design choice).
// ---------------------------------------------------------------------------
// The driver does NOT read cooked tables or world transforms directly. The two
// v0.8 worlds keep that data in DIFFERENT places:
//   - rigid bodies   : CPU BodyState transforms (host),
//   - articulation links : the GPU forward-kinematics `link_pose` buffer.
// So a single hard-wired data source would couple the driver to one world and
// force a rewrite to serve the other. Instead the caller supplies a
// ShapeResolver : CollidableRef -> ResolvedShape. C5c plugs an articulation-link
// resolver (a foot sphere prim at a link offset, world pose from link_pose) and
// the rigid world plugs a BodyState resolver -- the driver body is unchanged.
// (This file's standalone test plugs trivial in-test resolvers that fabricate
// the prims directly, proving both a rigid side and an articulation-link side.)
//
// ---------------------------------------------------------------------------
// SCOPE (C5a, validated-NOT-wired -- same discipline as C1-C4).
// ---------------------------------------------------------------------------
//   - HOST-ONLY. The dispatch table is a table of HOST function pointers today
//     (narrowphase_dispatch.hpp). A DEVICE fn-ptr table + a device driver kernel
//     are DEFERRED (C5 co-resident world / later); this host driver validates
//     the end-to-end emit logic and is the C5c foot-ground path's host driver.
//   - NOT wired into world_stepper.cpp. The v0.7 ad-hoc GenerateContact path
//     stays production until C5c flips it; goldens do not move from C5a.
//   - The COOKED-TABLE-BACKED resolver (rigid BodyState + articulation link_pose)
//     is C5c; here the resolver is an abstract std::function the caller fills.
// ---------------------------------------------------------------------------

#include "collision/candidate_pair.hpp"       // CandidatePair, CollidableRef
#include "collision/narrowphase_dispatch.hpp"  // ShapeProxyView, ResolveNarrowphase
#include "constraint/contact_manifold.hpp"     // ContactManifold
#include "scene/canonical_types.hpp"           // scene::ShapeType

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace nuka::collision {

using constraint::ContactManifold;
using scene::ShapeType;

// ---------------------------------------------------------------------------
// ResolvedShape -- one side, fully resolved: geometry + world frame + per-geom
// MuJoCo contact params + SDF availability. The output of a ShapeResolver and
// the input the driver feeds into the ShapeProxyView + the param merge.
// ---------------------------------------------------------------------------
struct ResolvedShape {
    ShapeType       type = ShapeType::Sphere;
    amf::PrimParams prim;             // half_extents/radius/half_height + baked world frame
    const void*     geom = nullptr;   // convex-hull view / SdfProxyView ptr (Convex/Sdf
                                      // tiers); null for primitives. Passed through
                                      // to ShapeProxyView.geom_a/geom_b verbatim.
    bool            has_sdf = false;  // this side has a cooked SDF (both-sides contract)

    // Per-geom MuJoCo contact params (the MergeContactParams inputs). Defaults are
    // MuJoCo's per-geom defaults so a resolver that fills only geometry still
    // merges to sane params (and the rigid case stays unchanged).
    float   solref[2] = {0.02f, 1.0f};                       // timeconst, dampratio
    float   solimp[5] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};   // dmin,dmax,width,mid,power
    float   friction  = 1.0f;                                 // Coulomb mu
    int32_t priority  = 0;
    float   solmix    = 1.0f;
    uint32_t condim   = 3u;
};

// Resolver: CollidableRef -> ResolvedShape. Returns false if the side has no
// collidable geometry (the pair is then skipped). Caller-provided -- this is the
// transform-source decoupling seam (rigid BodyState / articulation link_pose /
// the standalone test's fabricated prims).
using ShapeResolver = std::function<bool(const CollidableRef&, ResolvedShape*)>;

// ---------------------------------------------------------------------------
// THE DELIVERABLE: drive a CandidatePair span -> a ContactManifold stream.
// ---------------------------------------------------------------------------
// For each pair, in the GIVEN order (the pairs arrive D1-sorted by stable_key
// from C2, so a fixed sequential walk preserves that order -> no re-sort, no
// atomics, deterministic): resolve a & b; skip if either fails to resolve;
// derive has_sdf = a.has_sdf && b.has_sdf; build the ShapeProxyView; resolve the
// handler via the constexpr table; invoke it; skip if it emitted 0 points; merge
// the two sides' contact params into the manifold; append. `out` is appended to
// (not cleared) so a caller can accumulate across broadphase sources; clear it
// first if you want a fresh stream.
void BuildContactManifolds(std::span<const CandidatePair> pairs,
                           const ShapeResolver& resolve,
                           std::vector<ContactManifold>* out);

}  // namespace nuka::collision
