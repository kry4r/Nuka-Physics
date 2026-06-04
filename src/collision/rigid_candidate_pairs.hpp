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
#include "phi/device_context.hpp"
#include "scene/cooked_blob.hpp"

#include <cstdint>

namespace nuka::collision {

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

} // namespace nuka::collision
