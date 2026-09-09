#pragma once

#include <cstdint>

namespace nuka::collision {

// Reserved particle-body contact slots; exceeding this capacity reports overflow.
inline constexpr uint32_t kBodyParticleContactSlotsPerParticle = 4u;

} // namespace nuka::collision
