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
#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace nuka::collision::gpu {

// Forward declaration of the CUDA-only LBVH node type (defined in
// collision/lbvh_node.cuh). This header is included from g++-compiled host TUs
// (e.g. tests), so LbvhNode can only appear here as an INCOMPLETE-type pointer
// (DeviceNodes() below); the definition lives only in .cu TUs.
struct LbvhNode;

// Result of an LBVH broadphase build+query: a COMPACT, sorted candidate-pair
// list. `PairCount()` is the number of valid entries in the pair buffer.
//
// v0.7 p05: optionally RETAINS the built tree (the flat 2N-1 node array) so a
// cross-system query (particle AABB -> rigid LBVH) can traverse it. The node
// buffer is only populated when the build is invoked via BuildLbvhForQuery (or
// BuildLbvhBroadphase with retain_nodes=true); the default pair-only build path
// destroys it exactly as p04 did (byte-identical behavior, no perf/golden move).
class LbvhBroadphaseResult {
public:
    LbvhBroadphaseResult() = default;
    LbvhBroadphaseResult(uint32_t leaf_count,
                         uint32_t pair_count,
                         uint32_t pair_capacity,
                         phi::Buffer* pairs);
    LbvhBroadphaseResult(uint32_t leaf_count,
                         uint32_t pair_count,
                         uint32_t pair_capacity,
                         phi::Buffer* pairs,
                         phi::Buffer* nodes);
    ~LbvhBroadphaseResult();

    LbvhBroadphaseResult(const LbvhBroadphaseResult&) = delete;
    LbvhBroadphaseResult& operator=(const LbvhBroadphaseResult&) = delete;
    LbvhBroadphaseResult(LbvhBroadphaseResult&& other) noexcept;
    LbvhBroadphaseResult& operator=(LbvhBroadphaseResult&& other) noexcept;

    uint32_t LeafCount() const { return leaf_count_; }
    uint32_t PairCount() const { return pair_count_; }
    uint32_t PairCapacity() const { return pair_capacity_; }

    // Number of internal nodes (LeafCount-1 for LeafCount>=1; 0 otherwise) and
    // total nodes (2*LeafCount-1). The flat node array layout is: indices
    // [0, InternalCount()) are internal nodes, [InternalCount(), NodeCount())
    // are leaves. A leaf's `left` field holds the ORIGINAL body index.
    uint32_t InternalCount() const { return leaf_count_ > 0u ? leaf_count_ - 1u : 0u; }
    uint32_t NodeCount() const { return leaf_count_ > 0u ? 2u * leaf_count_ - 1u : 0u; }

    // Compact device buffer of `PairCount()` CollisionPair entries, sorted by
    // (body_a, body_b). Bytes beyond PairCount() are unspecified.
    const collision::CollisionPair* DevicePairs() const;

    // The built LBVH node tree (flat 2N-1 array), or nullptr if this build did
    // not retain it (default pair-only path). Valid only while this result is
    // alive. Used by the cross-system query to traverse the rigid tree.
    const LbvhNode* DeviceNodes() const;
    bool HasNodes() const;

    // Download exactly the PairCount() valid pairs (already sorted, canonical
    // a<b). The reduced SET equals the SAP overlap set.
    std::vector<collision::CollisionPair> DownloadPairs() const;

private:
    uint32_t leaf_count_ = 0;
    uint32_t pair_count_ = 0;
    uint32_t pair_capacity_ = 0;
    phi::Buffer* pairs_ = nullptr;  // phi v2 opaque handle; freed in the dtor.
    phi::Buffer* nodes_ = nullptr;  // retained tree (null unless retain_nodes set).
};

// Build an LBVH over `count` device-resident AABBs and return the compact,
// sorted candidate-pair list. `device_aabbs` must point to `count` consecutive
// collision::AABB in DEVICE memory.
//
// `max_pairs_hint` caps the compact output buffer. 0 => use a default heuristic
// (clamped). If the true overlap count exceeds the cap the result is truncated;
// the perf/diff tests size the cap conservatively.
LbvhBroadphaseResult BuildLbvhBroadphase(cudaStream_t stream, int device_id,
                                         const collision::AABB* device_aabbs,
                                         uint32_t count,
                                         uint32_t max_pairs_hint = 0u);

LbvhBroadphaseResult BuildLbvhBroadphase(const collision::AABB* device_aabbs,
                                         uint32_t count,
                                         uint32_t max_pairs_hint = 0u);

// v0.7 p05: build an LBVH and RETAIN the node tree for subsequent traversal
// (e.g. cross-system particle->rigid queries). Identical pipeline to
// BuildLbvhBroadphase; the only difference is the result keeps the node buffer
// (DeviceNodes() / HasNodes()) instead of destroying it. The pair list is still
// produced (callers that only need the tree can ignore it).
LbvhBroadphaseResult BuildLbvhForQuery(cudaStream_t stream, int device_id,
                                       const collision::AABB* device_aabbs,
                                       uint32_t count,
                                       uint32_t max_pairs_hint = 0u);

LbvhBroadphaseResult BuildLbvhForQuery(const collision::AABB* device_aabbs,
                                       uint32_t count,
                                       uint32_t max_pairs_hint = 0u);

} // namespace nuka::collision::gpu
