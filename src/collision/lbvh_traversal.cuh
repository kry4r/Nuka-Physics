#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu -- LBVH overlap-pair query (Task 7.4.4).
//
// One thread per leaf. Each thread walks down from the root using an explicit
// stack, testing the leaf's own AABB against internal-node bounds, and emits a
// candidate pair when it reaches another leaf whose bound overlaps.
//
// Determinism: emission ORDER is thread-scheduling dependent; D1 is restored by
// the caller sorting the compact output list. To make the output a clean SET
// matching SAP's canonical i<j pairs, we emit ONLY when the queried leaf's
// original body index is strictly less than the other leaf's (`a < b`), which
// also de-duplicates the (i,j)/(j,i) double-visit. The append index is a uint32
// atomic (NO float atomics).
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/dynamic_broadphase.hpp"
#include "collision/lbvh_node.cuh"

#include <cstdint>

#include <cuda_runtime.h>

namespace nuka::collision::gpu {

__device__ __forceinline__ bool LbvhOverlaps(const collision::AABB& a,
                                              const collision::AABB& b) {
    if (a.max.x < b.min.x || a.min.x > b.max.x) return false;
    if (a.max.y < b.min.y || a.min.y > b.max.y) return false;
    if (a.max.z < b.min.z || a.min.z > b.max.z) return false;
    return true;
}

// leaf_count == n_leaves. nodes[] is the flat 2N-1 array (internal in [0,N-1),
// leaves in [N-1, 2N-1)). Internal-node count is N-1; root is node 0 (for N>1).
__global__ void LbvhPairQueryKernel(uint32_t leaf_count,
                                    const LbvhNode* __restrict__ nodes,
                                    collision::CollisionPair* __restrict__ out_pairs,
                                    uint32_t* __restrict__ out_pair_count,
                                    uint32_t pair_capacity) {
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= leaf_count) {
        return;
    }
    if (leaf_count < 2u) {
        return;
    }

    const uint32_t internal_count = leaf_count - 1u;
    const uint32_t my_node = internal_count + lane;            // this leaf's node idx
    const collision::AABB query = nodes[my_node].aabb;
    const int32_t my_body = nodes[my_node].left;               // original body id

    // Explicit traversal stack. Depth of a balanced-ish LBVH over N leaves is
    // ~log2(N); 64 is comfortably safe up to far beyond 50k even when skewed,
    // and Morton ordering keeps it shallow in practice.
    int32_t stack[64];
    int32_t top = 0;
    stack[top++] = 0; // root internal node

    while (top > 0) {
        const int32_t node_idx = stack[--top];
        const LbvhNode node = nodes[node_idx];

        // Internal node: descend into overlapping children.
        const int32_t children[2] = {node.left, node.right};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int32_t child = children[c];
            const bool is_leaf = (static_cast<uint32_t>(child) >= internal_count);
            const collision::AABB& child_box = nodes[child].aabb;
            if (!LbvhOverlaps(query, child_box)) {
                continue;
            }
            if (is_leaf) {
                const int32_t other_body = nodes[child].left;
                // Canonical, de-duplicated emit: only the lower-body-id leaf of
                // each overlapping pair appends, matching SAP's i<j set.
                if (my_body < other_body) {
                    const uint32_t slot = atomicAdd(out_pair_count, 1u);
                    if (slot < pair_capacity) {
                        out_pairs[slot] = collision::CollisionPair{
                            static_cast<uint32_t>(my_body),
                            static_cast<uint32_t>(other_body)};
                    }
                }
            } else if (top < 63) {
                stack[top++] = child;
            }
        }
    }
}

} // namespace nuka::collision::gpu
