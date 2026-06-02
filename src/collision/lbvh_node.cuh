#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu -- LBVH node layout + Karras radix-tree device helpers.
//
// Shared by broadphase_lbvh.cu (build + traversal) and lbvh_refit.cu (refit).
// Device-only; included only from .cu TUs. No atomics here -- pure structural
// helpers (delta/LCP). D1-safe.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "math/vec3.hpp"

#include <cstdint>

#include <cuda_runtime.h>

namespace nuka::collision::gpu {

// Karras tree node. For an N-leaf tree there are N leaves and N-1 internal
// nodes. We store one flat array of 2N-1 nodes: indices [0, N-1) are internal
// nodes, [N-1, 2N-1) are leaves. `left`/`right` index into the SAME flat array
// (so a child index >= N-1 is a leaf). `parent` enables bottom-up AABB refit.
// `aabb` is the node bound (leaf bound for leaves; merged child bound for
// internal nodes). For a leaf, `left` holds the ORIGINAL body index of the leaf
// (so traversal can canonicalize pairs against true body ids).
struct LbvhNode {
    int32_t left;   // child node index (internal) OR original body index (leaf)
    int32_t right;  // child node index (internal); unused for leaves
    int32_t parent; // parent node index; -1 for the root
    collision::AABB aabb;
};

// Device-side AABB merge (collision::AABB::Merge is __host__-only -- it uses
// std::min/max). Order-independent fminf/fmaxf, so the bottom-up propagation
// result is the same regardless of which child fires first (D1).
__device__ __forceinline__ collision::AABB LbvhMerge(const collision::AABB& a,
                                                     const collision::AABB& b) {
    collision::AABB out;
    out.min.x = fminf(a.min.x, b.min.x);
    out.min.y = fminf(a.min.y, b.min.y);
    out.min.z = fminf(a.min.z, b.min.z);
    out.max.x = fmaxf(a.max.x, b.max.x);
    out.max.y = fmaxf(a.max.y, b.max.y);
    out.max.z = fmaxf(a.max.z, b.max.z);
    return out;
}

// Longest-common-prefix of the two 30-bit Morton codes at sorted positions
// i and j, with the Karras tie-break: equal codes fall back to comparing the
// indices themselves (so duplicate Morton codes still form a valid tree). The
// returned value is the bit-length of the shared prefix; -1 if j is OOB.
__device__ __forceinline__ int LbvhDelta(int i,
                                          int j,
                                          uint32_t n_leaves,
                                          const uint32_t* __restrict__ morton_sorted) {
    if (j < 0 || j >= static_cast<int>(n_leaves)) {
        return -1;
    }
    const uint32_t ci = morton_sorted[i];
    const uint32_t cj = morton_sorted[j];
    if (ci == cj) {
        // Tie-break on index: 32 bits of code agree, then count leading zeros of
        // (i ^ j) in 32-bit space, offset by the 32 code bits.
        return 32 + static_cast<int>(__clz(static_cast<uint32_t>(i ^ j)));
    }
    return static_cast<int>(__clz(ci ^ cj));
}

} // namespace nuka::collision::gpu
