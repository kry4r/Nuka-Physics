#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace nuka {
inline uint64_t CheckedAdd(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a)
        throw std::overflow_error("capacity addition overflow");
    return a + b;
}

inline uint64_t CheckedProduct(std::initializer_list<uint64_t> factors) {
    uint64_t result = 1u;
    for (const auto factor : factors) {
        if (factor != 0u && result > std::numeric_limits<uint64_t>::max() / factor)
            throw std::overflow_error("capacity multiplication overflow");
        result *= factor;
    }
    return result;
}

inline uint64_t CheckedAlignUp(uint64_t bytes, uint64_t alignment) {
    if (alignment == 0u || (alignment & (alignment - 1u)) != 0u)
        throw std::invalid_argument("alignment must be a power of two");
    return CheckedAdd(bytes, alignment - 1u) & ~(alignment - 1u);
}
}  // namespace nuka
