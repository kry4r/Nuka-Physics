#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::BuildRigidCandidatePairs -- the RIGID<->RIGID filtered
// candidate-pair stream (v0.8 C2b)
// ---------------------------------------------------------------------------
// A NEW, VALIDATED-NOT-FORCED broadphase path: build an LBVH over per-SHAPE
// world AABBs (exactly as the production CPU broadphase indexes per-shape AABBs),
// emit each surviving shape-pair as a BODY-pair CandidatePair into the C2a
// CandidatePairStream. This is rigid<->rigid ONLY and is NOT wired into the
// production world_stepper -- the production CPU->GPU broadphase swap rides with
// C5 (just as the p04 LBVH was validated against SAP but never forced default).
//
// THE PIPELINE (all D1):
//   per-shape world AABBs  --LBVH (Karras, thrust::stable_sort)-->  sorted
//   shape-pairs  --filter kernel (per-pair integer test: count->scan->compact,
//   NO append atomics)-->  surviving BODY-pairs  --host explicit-<pair> merge-->
//   BuildCandidatePairStream (C2a; stamps stable_key + thrust::stable_sort) -->
//   D1 byte-exact CandidatePairStream.
//
// THE FILTER (per LBVH shape-pair (sa, sb), MuJoCo collision precedence):
//   1. bitmask: keep iff PassesContactBitmask(contype[sa], conaff[sa],
//      contype[sb], conaff[sb]) (C1c contact_filter.hpp).
//   2. exclude: drop iff the BODY pair (body_ids[sa], body_ids[sb]) is in the
//      policy's excluded_body_pairs (this list already unions authored <exclude>
//      AND parent-child auto-exclude -- C1c built it). Binary-searched on-device
//      over the sorted canonical list.
//   3. explicit <pair>: a force-include -- a <pair> on (sa, sb) (SOURCE-shape
//      ids) is kept EVEN if the bitmask failed (MuJoCo: a <pair> always
//      generates). Applied as a HOST-side merge of the policy.explicit_pairs
//      into the survivor set (the kernel emits the bitmask/exclude survivors; the
//      host union adds back any explicit pair the bitmask dropped). See the .cu
//      for the SOURCE-shape vs COOKED-row note (v0.8 = 1:1, no decomposition).
//
// THE HANDLE MODEL (controller-ratified, §0.1): a CandidatePair side's `handle`
// is the BODY id (the reaction provider C4c maps an impulse to body DOFs). The
// LBVH runs over per-SHAPE AABBs; a surviving shape-pair maps to a body-pair via
// shape_body_ids[shape]. v0.8 production is ~1 collision shape per rigid body, so
// shape-pair ~= body-pair (1:1) and the emit is exact. A MULTI-shape body would
// emit per-shape-pair candidates that map to the SAME body-pair; we dedup
// adjacent identical body-pair candidates after the C2a sort so the emitted
// stream carries each body-pair once (C3 narrowphase re-expands the body's shapes
// [ShapeBacked]). For v0.8's 1-shape/body this dedup is a no-op.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/candidate_pair.hpp"
#include "collision/link_aabb.hpp"   // LinkShapeAabbs (the articulation side input)
#include "phi/device_context.hpp"
#include "scene/cooked_blob.hpp"

#include <cstdint>
#include <vector>

namespace nuka::collision {

// ---------------------------------------------------------------------------
// v0.8 co-residence (commit 2/3): the TYPE-AGNOSTIC tagged broadphase core +
// the artic-link<->rigid mixed entry.
// ---------------------------------------------------------------------------
// C2b emitted RIGID<->RIGID only because the LBVH input was a single role: every
// AABB was a RigidBody whose handle == its body id. This commit generalizes the
// core so each input AABB carries its OWN per-collidable tag (type/react/handle/
// body_id), letting one LBVH+filter+compact pass emit cross-type CandidatePairs
// (ArticulationLink <-> RigidBody) ALONGSIDE same-type ones, consuming commit 1's
// per-link world AABBs (link_aabb.hpp). VALIDATED-NOT-WIRED -- nothing here
// touches world_stepper.
//
// THE TAG SPLIT (the crux): C2b's CompactKernel used shape_body_ids[shape] for
// BOTH the emitted side `handle` AND the filter (same-body drop + exclude). Those
// are now DISTINCT inputs:
//   * `aabb_handles[i]`  -> the value stamped into CollidableRef.handle (and packed
//                           into the stable_key): body-id for RigidBody, the LINK
//                           index for ArticulationLink. (link != body in general.)
//   * `aabb_body_ids[i]` -> used ONLY for the same-body `ba!=bb` drop and the
//                           excluded-body-pair binary search.
// For an all-RigidBody set the two arrays are identical, so the rigid wrapper
// stays BYTE-IDENTICAL to C2b (the existing nuka_lbvh_filtered_pairs_test is the
// regression guard).
//
// HANDLE-GUARD (C2-named-debt consumer): the tagged core throws host-side if any
// `aabb_handles[i] > kCandidateHandleMask` (>= 2^28), since a larger handle would
// alias the type bits in PackSideKey. This is the named consumer of the
// "handle < 2^28 guard" debt C2a deferred to the mint site -- this IS the mint
// site (the rigid wrapper, the mixed entry, and the gate all flow through here).
//
// SCOPE: cross-type explicit-<pair> force-include is OUT OF SCOPE. The explicit
// <pair> merge is rigid-source-shape-specific (policy.explicit_pairs index the
// rigid shape table) and stays in BuildRigidCandidatePairs as caller-supplied
// extra survivors; the mixed entry passes none.

// Per-AABB tag arrays for the tagged core (struct-of-arrays, all parallel to the
// `count` device AABBs in the SAME order). The core reads `react`/`handle` to
// stamp the emitted CollidableRef sides and `body_id`/`contype`/`conaffinity`
// for the filter. (No `type` field needed beyond what react/handle encode -- but
// we carry it so the emitted CollidableRef.type is exact.)
struct CandidateAabbTags {
    const CollidableType*               types;        // [count] emitted side type.
    const constraint::ReactionProviderKind* reacts;   // [count] emitted side react.
    const uint32_t*                     handles;      // [count] emitted side handle.
    const uint32_t*                     body_ids;     // [count] filter body id.
    const uint32_t*                     contypes;     // [count] filter contype.
    const uint32_t*                     conaffinities;// [count] filter conaffinity.
};

// The TYPE-AGNOSTIC core. LBVH over `count` device AABBs -> per-pair bitmask +
// same-body drop + body-pair exclude filter (count->scan->compact, NO append
// atomics) -> emit CandidatePairs whose sides are stamped from the per-AABB tags
// -> UNION the caller-supplied `extra_survivors` (rigid explicit-<pair> body-pairs;
// pass {} for none) -> BuildCandidatePairStream (C2a) -> adjacent dedup -> sorted
// D1 stream. `excluded_body_pairs` is the canonical (min,max) ascending list (the
// rigid wrapper passes policy.excluded_body_pairs). Throws if any handle >= 2^28.
//
//   tags.* arrays must each have `count` entries, parallel to `device_aabbs`.
//   `count == 0` -> empty stream (extra_survivors still merged).
CandidatePairStream BuildCandidatePairsTagged(
    const phi::DeviceContext& context,
    const collision::AABB* device_aabbs, uint32_t count,
    const CandidateAabbTags& tags,
    const std::vector<std::pair<uint32_t, uint32_t>>& excluded_body_pairs,
    const std::vector<CandidatePair>& extra_survivors);

// Build the RIGID<->RIGID filtered candidate-pair stream from per-shape world
// AABBs. LBVH over the AABBs -> per-shape-pair -> filter (C1c bitmask + body-level
// exclude) -> emit body-pair CandidatePairs -> host explicit-<pair> merge ->
// sorted D1 stream. NOT wired into production.
//
//   device_shape_aabbs : `shape_count` collision::AABB in DEVICE memory.
//   shape_body_ids     : CookedShapeTable.body_ids  (shape -> body), HOST array.
//   shape_contypes     : CookedContactParamTable.contypes,           HOST array.
//   shape_conaffinities: CookedContactParamTable.conaffinities,      HOST array.
//   policy             : CookedFilterPolicy (excluded_body_pairs + explicit_pairs).
//                        excluded_body_pairs MUST be the canonical (min,max)
//                        ascending-sorted list C1c produces (binary-searched
//                        on-device). explicit_pairs are in SOURCE-shape space.
//
// All three HOST arrays must have `shape_count` entries (parallel to the cooked
// shape table). `shape_count == 0` -> empty stream.
CandidatePairStream BuildRigidCandidatePairs(
    const phi::DeviceContext& context,
    const collision::AABB* device_shape_aabbs, uint32_t shape_count,
    const uint32_t* shape_body_ids,
    const uint32_t* shape_contypes,
    const uint32_t* shape_conaffinities,
    const scene::CookedFilterPolicy& policy);

// Default-context overload.
CandidatePairStream BuildRigidCandidatePairs(
    const collision::AABB* device_shape_aabbs, uint32_t shape_count,
    const uint32_t* shape_body_ids,
    const uint32_t* shape_contypes,
    const uint32_t* shape_conaffinities,
    const scene::CookedFilterPolicy& policy);

// ---------------------------------------------------------------------------
// The ARTICULATION-LINK <-> RIGID mixed entry (commit 3's gate consumer).
// ---------------------------------------------------------------------------
// Concatenate the RIGID shape AABBs (tagged RigidBody, handle = body id) and the
// articulation LINK shape AABBs (tagged ArticulationLink, handle = LINK index --
// from commit 1's LinkShapeAabbs.link_indices) into ONE tagged AABB set and run
// the tagged core. The emitted stream carries every surviving pair regardless of
// the two sides' types: rigid<->rigid, link<->link, AND link<->rigid, each side's
// CollidableRef stamped with its own (type, react, handle).
//
//   rigid_aabbs / rigid_count    : the rigid shape world AABBs (DEVICE) + their
//                                  per-shape body_ids/contypes/conaffinities (HOST,
//                                  each `rigid_count` long -- the C2b rigid inputs).
//   links                        : commit 1's LinkShapeAabbs (aabbs HOST -> uploaded
//                                  here; shape_body_ids + link_indices parallel).
//                                  Per-link contype/conaffinity is supplied by
//                                  `link_contypes`/`link_conaffinities` (each
//                                  links.aabbs.size() long); pass all-1 for the
//                                  MuJoCo default (everything collides).
//   excluded_body_pairs          : canonical (min,max) ascending exclude list (the
//                                  same body-id space as both sides' body_ids).
//
// Cross-type explicit-<pair> is OUT OF SCOPE (no extra survivors). The rigid side's
// own explicit pairs are NOT applied here either -- callers needing them use
// BuildRigidCandidatePairs for the rigid<->rigid sub-stream. Throws if any handle
// (rigid body id OR link index) >= 2^28.
CandidatePairStream BuildArticulationRigidCandidatePairs(
    const phi::DeviceContext& context,
    const collision::AABB* rigid_aabbs, uint32_t rigid_count,
    const uint32_t* rigid_body_ids,
    const uint32_t* rigid_contypes,
    const uint32_t* rigid_conaffinities,
    const LinkShapeAabbs& links,
    const uint32_t* link_contypes,
    const uint32_t* link_conaffinities,
    const std::vector<std::pair<uint32_t, uint32_t>>& excluded_body_pairs);

} // namespace nuka::collision
