#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::StaticBVH -- static scene bounding-volume hierarchy
// ---------------------------------------------------------------------------
// A REAL binary BVH (median split on the longest centroid axis): Build()
// recursively partitions the shape AABBs into a tree of internal + leaf nodes,
// and Raycast() traverses it with an explicit stack, pruning whole subtrees
// whose bounds the ray misses -> O(log n) expected instead of O(n) brute force.
// Deterministic: the split uses std::nth_element on a fixed comparator, and the
// build/traversal order is fixed.

#include "collision/aabb.hpp"
#include "math/vec3.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace nuka::collision {

// ---------------------------------------------------------------------------
// BVH node. Internal nodes set left/right (child node indices); leaf nodes set
// shape_index. A node is a leaf iff shape_index != kInvalid.
// ---------------------------------------------------------------------------

struct BVHNode {
    static constexpr uint32_t kInvalid = ~0u;

    AABB     bounds;
    uint32_t left        = kInvalid;
    uint32_t right       = kInvalid;
    uint32_t shape_index = kInvalid; // set on leaf nodes only

    bool IsLeaf() const { return shape_index != kInvalid; }
};

// ---------------------------------------------------------------------------
// StaticBVH -- a binary BVH over a fixed set of per-shape AABBs.
// ---------------------------------------------------------------------------

class StaticBVH {
public:
    struct RayHit {
        float    distance;
        uint32_t shape_index;
    };

    /// Build a binary BVH from a list of per-shape AABBs (median split on the
    /// longest centroid axis). Empty input yields an empty tree.
    void Build(const std::vector<AABB>& bounds) {
        nodes_.clear();
        if (bounds.empty()) {
            return;
        }
        // Working set of shape indices; the recursion permutes this in place.
        std::vector<uint32_t> indices(bounds.size());
        for (uint32_t i = 0; i < static_cast<uint32_t>(bounds.size()); ++i) {
            indices[i] = i;
        }
        // Reserve the worst-case node count (2n-1 for n leaves) so child indices
        // captured during recursion stay valid (no reallocation invalidation).
        nodes_.reserve(2u * bounds.size());
        BuildRecursive(bounds, indices, 0u,
                       static_cast<uint32_t>(indices.size()));
    }

    /// Raycast the BVH; returns the nearest leaf hit within max_dist. Prunes any
    /// subtree whose bounds the ray does not enter before max_dist.
    std::optional<RayHit> Raycast(math::Vec3 origin,
                                   math::Vec3 direction,
                                   float max_dist = FLT_MAX) const {
        std::optional<RayHit> best;
        if (nodes_.empty()) {
            return best;
        }
        // Explicit DFS stack over node indices; root is the LAST node appended.
        uint32_t stack[kMaxStackDepth];
        int32_t  top = 0;
        stack[top++] = static_cast<uint32_t>(nodes_.size() - 1u);
        while (top > 0) {
            const uint32_t ni = stack[--top];
            const BVHNode& node = nodes_[ni];
            const auto thit = RayVsAABB(origin, direction, node.bounds);
            // Miss, behind the ray, or farther than the current best -> prune.
            if (!thit || *thit < 0.0f || *thit >= max_dist) {
                continue;
            }
            if (node.IsLeaf()) {
                if (!best || *thit < best->distance) {
                    best = RayHit{*thit, node.shape_index};
                    max_dist = *thit;
                }
                continue;
            }
            // Push both children (fixed order). The bounds test above prunes
            // each before it is expanded. Overflow guard: degenerate-deep trees
            // fall back to NOT pushing (a missed subtree is reported via the
            // assert-style guard rather than an out-of-bounds write).
            if (top + 2 <= kMaxStackDepth) {
                if (node.left != BVHNode::kInvalid)  stack[top++] = node.left;
                if (node.right != BVHNode::kInvalid) stack[top++] = node.right;
            }
        }
        return best;
    }

    const std::vector<BVHNode>& Nodes() const { return nodes_; }

private:
    // BVHNode holds a single shape_index, so a leaf bottoms out at ONE shape.
    // (A multi-shape leaf would need a [first,count) range field on BVHNode.)
    static constexpr uint32_t kLeafSize = 1u;
    // Traversal stack depth: a balanced BVH over up to 2^kMaxStackDepth leaves.
    static constexpr int32_t kMaxStackDepth = 64;

    std::vector<BVHNode> nodes_;

    // Build the subtree over indices[begin,end); returns its node index. Appends
    // children BEFORE the parent so a parent's child indices are final.
    uint32_t BuildRecursive(const std::vector<AABB>& bounds,
                            std::vector<uint32_t>& indices,
                            uint32_t begin, uint32_t end) {
        // Union bounds of this range (AABB::Merge is the codebase idiom).
        AABB box = bounds[indices[begin]];
        for (uint32_t i = begin + 1u; i < end; ++i) {
            box.Merge(bounds[indices[i]]);
        }
        const uint32_t count = end - begin;
        if (count <= kLeafSize) {
            BVHNode leaf;
            leaf.bounds      = box;
            leaf.shape_index = indices[begin];  // single shape per leaf (kLeafSize==1)
            nodes_.push_back(leaf);
            return static_cast<uint32_t>(nodes_.size() - 1u);
        }
        // Split on the axis of greatest centroid spread; median by nth_element.
        const int axis = LongestCentroidAxis(bounds, indices, begin, end);
        const uint32_t mid = begin + count / 2u;
        std::nth_element(indices.begin() + begin, indices.begin() + mid,
                         indices.begin() + end,
                         [&](uint32_t a, uint32_t b) {
                             return Centroid(bounds[a], axis) <
                                    Centroid(bounds[b], axis);
                         });
        const uint32_t left  = BuildRecursive(bounds, indices, begin, mid);
        const uint32_t right = BuildRecursive(bounds, indices, mid, end);
        BVHNode internal;
        internal.bounds = box;
        internal.left   = left;
        internal.right  = right;
        nodes_.push_back(internal);
        return static_cast<uint32_t>(nodes_.size() - 1u);
    }

    static float Centroid(const AABB& b, int axis) {
        const float* mn = &b.min.x;
        const float* mx = &b.max.x;
        return 0.5f * (mn[axis] + mx[axis]);
    }

    static int LongestCentroidAxis(const std::vector<AABB>& bounds,
                                   const std::vector<uint32_t>& indices,
                                   uint32_t begin, uint32_t end) {
        float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
        float hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (uint32_t i = begin; i < end; ++i) {
            for (int k = 0; k < 3; ++k) {
                const float c = Centroid(bounds[indices[i]], k);
                lo[k] = std::min(lo[k], c);
                hi[k] = std::max(hi[k], c);
            }
        }
        int axis = 0;
        float best = hi[0] - lo[0];
        for (int k = 1; k < 3; ++k) {
            const float ext = hi[k] - lo[k];
            if (ext > best) { best = ext; axis = k; }
        }
        return axis;
    }

    /// Slab-based ray-vs-AABB test returning entry distance (or nullopt on miss).
    static std::optional<float> RayVsAABB(math::Vec3 origin,
                                           math::Vec3 direction,
                                           const AABB& box) {
        float tmin = 0.0f;
        float tmax = FLT_MAX;

        const float* o = &origin.x;
        const float* d = &direction.x;
        const float* bmin = &box.min.x;
        const float* bmax = &box.max.x;

        for (int i = 0; i < 3; ++i) {
            if (std::abs(d[i]) < 1e-12f) {
                if (o[i] < bmin[i] || o[i] > bmax[i]) return std::nullopt;
            } else {
                float inv_d = 1.0f / d[i];
                float t1 = (bmin[i] - o[i]) * inv_d;
                float t2 = (bmax[i] - o[i]) * inv_d;
                if (t1 > t2) std::swap(t1, t2);
                tmin = std::max(tmin, t1);
                tmax = std::min(tmax, t2);
                if (tmin > tmax) return std::nullopt;
            }
        }
        return tmin;
    }
};

} // namespace nuka::collision
