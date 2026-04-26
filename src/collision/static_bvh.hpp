#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::StaticBVH -- Static scene BVH (scaffold)
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "math/vec3.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace nuka::collision {

// ---------------------------------------------------------------------------
// BVH node
// ---------------------------------------------------------------------------

struct BVHNode {
    AABB     bounds;
    uint32_t left        = ~0u;
    uint32_t right       = ~0u;
    uint32_t shape_index = ~0u; // leaf node

    bool IsLeaf() const { return shape_index != ~0u; }
};

// ---------------------------------------------------------------------------
// StaticBVH -- for now a simple linear list of leaf nodes
// ---------------------------------------------------------------------------

class StaticBVH {
public:
    struct RayHit {
        float    distance;
        uint32_t shape_index;
    };

    /// Build a flat BVH from a list of per-shape AABBs.
    void Build(const std::vector<AABB>& bounds) {
        nodes_.clear();
        nodes_.reserve(bounds.size());
        for (uint32_t i = 0; i < static_cast<uint32_t>(bounds.size()); ++i) {
            BVHNode node;
            node.bounds      = bounds[i];
            node.shape_index = i;
            nodes_.push_back(node);
        }
    }

    /// Brute-force raycast against all leaf nodes.
    std::optional<RayHit> Raycast(math::Vec3 origin,
                                   math::Vec3 direction,
                                   float max_dist = FLT_MAX) const {
        std::optional<RayHit> best;
        for (const auto& node : nodes_) {
            if (!node.IsLeaf()) continue;
            auto hit = RayVsAABB(origin, direction, node.bounds);
            if (hit && *hit >= 0.0f && *hit < max_dist) {
                if (!best || *hit < best->distance) {
                    best = RayHit{*hit, node.shape_index};
                    max_dist = *hit;
                }
            }
        }
        return best;
    }

    const std::vector<BVHNode>& Nodes() const { return nodes_; }

private:
    std::vector<BVHNode> nodes_;

    /// Simple slab-based ray-vs-AABB test returning entry distance.
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
