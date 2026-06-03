#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- the ONE stackless parent-pointer (Hapala 2011) LBVH ray-traversal
// state machine, factored out of the p12 RenderDepth kernel so it is REUSED
// UNCHANGED by every consumer (v0.7 p13). p12 renders box leaves; p13 renders
// triangle / sphere / sparse-SDF leaves -- the descent / node-AABB pruning /
// `t`-bound / lowest-index tie-break is byte-identical for both; only the LEAF
// FUNCTION differs. The traversal is templated on a leaf functor (a small struct
// with a __device__ operator(), NOT a lambda -> no --extended-lambda needed) so
// there is literally ONE traversal, two leaves. The p12 gate
// (nuka_rt_traversal_test, depth max-err == 0) proves this extraction did not
// perturb the box path.
//
// State machine (verbatim from p12): move between a node and its parent;
// `from_child` encodes which child we just RETURNED from (0=from above/start,
// 1=from left, 2=from right). At an internal node: descend left, then right, then
// go up -- pruning any child whose bound is missed or whose entry t >= best_t.
// ZERO atomics, sequential local best_t/best_prim -> D1 by construction.
//
// DEVICE-ONLY header (included only from .cu TUs; uses LbvhNode + NodeEntryT).
// ---------------------------------------------------------------------------

#include "collision/lbvh_node.cuh"
#include "math/vec3.hpp"
#include "rt/ray_box.cuh"

namespace nuka::rt {

using ::nuka::collision::gpu::LbvhNode;

// Ray-AABB entry t for node pruning (NaN-guarded; shared with the leaf box math).
__device__ __forceinline__ bool TraverseNodeEntryT(const math::Vec3& origin,
                                                   const math::Vec3& dir,
                                                   const collision::AABB& box,
                                                   float* entry_t) {
    return RtNodeEntryT(origin, dir, box, entry_t);
}

// Stackless traversal of one ray. `internal_count` = N-1 (a flat index >=
// internal_count is a leaf). `leaf(leaf_node, origin, dir, &best_t, &best_prim)`
// tests + possibly-updates the closest hit against the leaf at flat node index
// `leaf_node` (it reads nodes[leaf_node].left == the ORIGINAL prim id). Returns
// closest (best_t, best_prim) via out-params. For N==1 the caller intersects
// node 0 directly (single node IS the leaf).
template <typename LeafFn>
__device__ void TraverseRay(const LbvhNode* __restrict__ nodes,
                            uint32_t internal_count,
                            const math::Vec3& origin,
                            const math::Vec3& dir,
                            float* best_t,
                            uint32_t* best_prim,
                            LeafFn leaf) {
    // Should we descend into `child` (prune if missed or entry t >= best_t)?
    auto want = [&](int32_t child) -> bool {
        float et;
        if (!TraverseNodeEntryT(origin, dir, nodes[child].aabb, &et)) {
            return false;
        }
        return et < *best_t;
    };

    constexpr int kFromParent = 0;
    constexpr int kFromLeft = 1;
    constexpr int kFromRight = 2;

    int32_t current = 0;      // start at root (internal node 0)
    int from = kFromParent;   // how we arrived at `current`

    // Root pruning: if the root bound is entirely missed, nothing to do.
    {
        float et;
        if (!TraverseNodeEntryT(origin, dir, nodes[0].aabb, &et)) {
            return;
        }
    }

    while (true) {
        const LbvhNode node = nodes[current];
        const int32_t left = node.left;
        const int32_t right = node.right;
        const int32_t parent = node.parent;

        int32_t next = -1;
        int next_from = kFromParent;

        if (from == kFromParent) {
            // Just arrived from above: try left child first.
            const bool left_leaf = (static_cast<uint32_t>(left) >= internal_count);
            if (left_leaf) {
                leaf(left, origin, dir, best_t, best_prim);
                from = kFromLeft;
                continue;
            } else if (want(left)) {
                next = left;
                next_from = kFromParent;
            } else {
                from = kFromLeft;
                continue;
            }
        } else if (from == kFromLeft) {
            // Returned from (or skipped) left: try right child.
            const bool right_leaf = (static_cast<uint32_t>(right) >= internal_count);
            if (right_leaf) {
                leaf(right, origin, dir, best_t, best_prim);
                from = kFromRight;
                continue;
            } else if (want(right)) {
                next = right;
                next_from = kFromParent;
            } else {
                from = kFromRight;
                continue;
            }
        } else { // kFromRight: both children done -> go up.
            if (parent < 0) {
                return; // returned to root from its right subtree: done.
            }
            const int32_t pleft = nodes[parent].left;
            next = parent;
            next_from = (pleft == current) ? kFromLeft : kFromRight;
        }

        if (next >= 0) {
            current = next;
            from = next_from;
        }
    }
}

} // namespace nuka::rt
