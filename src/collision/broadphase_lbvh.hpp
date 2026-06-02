#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu::BroadphaseLbvh -- Linear BVH (Karras 2012) GPU
// broadphase. v0.7 p04.
//
// STANDALONE, VALIDATED entry point. The legacy O(n^2) SAP path in
// collision/gpu/broadphase.cu remains BOTH the default AND the SOLE broadphase
// wired into the runtime (world_stepper) -- this p04 deliberately does NOT add a
// nuka_broadphase_t enum / nuka.h selection field nor flip the default, per the
// controller's "SAP remains the default; prefer the lowest-golden-risk path"
// override of spec exit-criteria 5/6. LBVH is exercised only by its tests for
// now; the engine continues to use SAP unchanged (rigid-body goldens untouched).
//
// This entry point produces a COMPACT, deterministically SORTED candidate-pair
// list whose reduced SET of {min(a,b),max(a,b)} pairs is byte-for-byte equal to
// the SAP overlap set (validated in tests/collision/test_lbvh_vs_sap_pair_set.cpp).
//
// Pipeline: scene-bound reduction -> 30-bit Morton codes -> thrust::stable_sort
// (radix; deterministic) -> Karras radix-tree internal nodes -> bottom-up AABB
// via integer atomic visit-counters -> one-thread-per-leaf overlap traversal
// emitting canonical (a<b) pairs -> thrust::stable_sort of the compact output
// (so the list is byte-identical across runs regardless of emit thread order).
//
// D1: NO float atomics anywhere (lint-guarded via src/collision/** glob). The
// only atomics are uint32 (the visit-counter and the pair-list append index);
// their FIRING ORDER does not affect the RESULT, and the final sort removes any
// residual order dependence in the emitted list.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/dynamic_broadphase.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"

#include <cstdint>
#include <vector>

namespace nuka::collision::gpu {

// Result of an LBVH broadphase build+query: a COMPACT, sorted candidate-pair
// list. `PairCount()` is the number of valid entries in the pair buffer.
class LbvhBroadphaseResult {
public:
    LbvhBroadphaseResult() = default;
    LbvhBroadphaseResult(uint32_t leaf_count,
                         uint32_t pair_count,
                         uint32_t pair_capacity,
                         phi::Buffer pairs);

    LbvhBroadphaseResult(const LbvhBroadphaseResult&) = delete;
    LbvhBroadphaseResult& operator=(const LbvhBroadphaseResult&) = delete;
    LbvhBroadphaseResult(LbvhBroadphaseResult&&) noexcept = default;
    LbvhBroadphaseResult& operator=(LbvhBroadphaseResult&&) noexcept = default;

    uint32_t LeafCount() const { return leaf_count_; }
    uint32_t PairCount() const { return pair_count_; }
    uint32_t PairCapacity() const { return pair_capacity_; }

    // Compact device buffer of `PairCount()` CollisionPair entries, sorted by
    // (body_a, body_b). Bytes beyond PairCount() are unspecified.
    const collision::CollisionPair* DevicePairs() const;

    // Download exactly the PairCount() valid pairs (already sorted, canonical
    // a<b). The reduced SET equals the SAP overlap set.
    std::vector<collision::CollisionPair> DownloadPairs() const;

private:
    uint32_t leaf_count_ = 0;
    uint32_t pair_count_ = 0;
    uint32_t pair_capacity_ = 0;
    phi::Buffer pairs_;
};

// Build an LBVH over `count` device-resident AABBs and return the compact,
// sorted candidate-pair list. `device_aabbs` must point to `count` consecutive
// collision::AABB in DEVICE memory.
//
// `max_pairs_hint` caps the compact output buffer. 0 => use a default heuristic
// (clamped). If the true overlap count exceeds the cap the result is truncated;
// the perf/diff tests size the cap conservatively.
LbvhBroadphaseResult BuildLbvhBroadphase(const phi::DeviceContext& context,
                                         const collision::AABB* device_aabbs,
                                         uint32_t count,
                                         uint32_t max_pairs_hint = 0u);

LbvhBroadphaseResult BuildLbvhBroadphase(const collision::AABB* device_aabbs,
                                         uint32_t count,
                                         uint32_t max_pairs_hint = 0u);

} // namespace nuka::collision::gpu
