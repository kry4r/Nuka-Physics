// ---------------------------------------------------------------------------
// DynamicBroadphase -- brute-force O(n^2) overlap test
// ---------------------------------------------------------------------------

#include "collision/dynamic_broadphase.hpp"

namespace nuka::collision {

void DynamicBroadphase::Update(const std::vector<AABB>& body_aabbs) {
    pairs_.clear();
    const auto n = static_cast<uint32_t>(body_aabbs.size());
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = i + 1; j < n; ++j) {
            if (body_aabbs[i].Overlaps(body_aabbs[j])) {
                pairs_.push_back({i, j});
            }
        }
    }
}

} // namespace nuka::collision
