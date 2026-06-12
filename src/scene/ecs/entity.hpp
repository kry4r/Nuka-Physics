#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::EntityId - generational ECS entity handle.
//
// An entity is identified by a slot `index` plus a `gen` (generation) counter.
// When a slot is recycled the Registry bumps its generation so that stale
// handles to the old occupant compare unequal (use-after-free detection). A
// default-constructed handle equals kInvalidEntity.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <functional>

namespace nuka::scene {

struct EntityId {
    uint32_t index = ~uint32_t(0);
    uint32_t gen   = ~uint32_t(0);

    constexpr bool operator==(const EntityId& other) const {
        return index == other.index && gen == other.gen;
    }
    constexpr bool operator!=(const EntityId& other) const {
        return !(*this == other);
    }
};

// Sentinel "no entity" handle (both fields all-ones). The Registry never hands
// out a slot with index == ~0u, so this can never collide with a live entity.
inline constexpr EntityId kInvalidEntity{~uint32_t(0), ~uint32_t(0)};

// Hash functor so EntityId can key unordered_map / unordered_set.
struct EntityIdHash {
    std::size_t operator()(const EntityId& e) const noexcept {
        // Pack the two 32-bit fields into one 64-bit value, then hash. The
        // packing is collision-free over the (index, gen) pair itself.
        const std::uint64_t key =
            (static_cast<std::uint64_t>(e.index) << 32) | e.gen;
        return std::hash<std::uint64_t>{}(key);
    }
};

} // namespace nuka::scene
