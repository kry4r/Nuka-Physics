#pragma once

#include <cstdint>

namespace nuka::phi::nkops {

inline constexpr uint32_t kIslandHasArticulation = 1u;
inline constexpr uint32_t kIslandHasMultiplePointTerms = 2u;
inline constexpr uint32_t kIslandWarpWork = kIslandHasArticulation | kIslandHasMultiplePointTerms;

struct IslandRecord {
    uint32_t seg_off;
    uint32_t seg_cnt;
    uint32_t flags;
    uint32_t env;
};
static_assert(sizeof(IslandRecord) == 4u * sizeof(uint32_t));

}  // namespace nuka::phi::nkops
