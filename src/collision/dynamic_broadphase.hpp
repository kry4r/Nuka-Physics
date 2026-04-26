#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::DynamicBroadphase -- pair generation for dynamic bodies
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"

#include <cstdint>
#include <vector>

namespace nuka::collision {

struct CollisionPair {
    uint32_t body_a;
    uint32_t body_b;
};

class DynamicBroadphase {
public:
    /// Rebuild the pair list from the current set of body AABBs (brute-force).
    void Update(const std::vector<AABB>& body_aabbs);

    /// Return the most recently computed pairs.
    const std::vector<CollisionPair>& GetPairs() const { return pairs_; }

private:
    std::vector<CollisionPair> pairs_;
};

} // namespace nuka::collision
