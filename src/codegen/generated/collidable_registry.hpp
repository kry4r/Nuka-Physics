// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/collidable/*.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================

#pragma once

#include <cstdint>

// The leaf kind enums (AccelStructureKind / ShapeProxyKind / ReactionProviderKind)
// referenced by the constexpr metadata table below. Hand-authored, no deps ->
// no include cycle (collidable_kinds.hpp -> THIS generated header -> collidable.hpp).
#include "constraint/collidable_kinds.hpp"

namespace nuka::constraint::generated {

// One enumerator per collidable type; the value IS collidable_type_id (the
// kCollidableTypeTable below is indexed by this value, so the ids must be the
// contiguous range 0..kCollidableTypeCount-1, which regen.py asserts).
enum class CollidableType : uint8_t {
    RigidBody = 0u,
    ArticulationLink = 1u,
    Particle = 2u,
    StaticWorld = 3u,
};

inline constexpr uint32_t kCollidableTypeCount = 4u;

// STATIC per-type metadata. Provider KIND selectors only -- NO function pointers
// (the runtime device-fn-ptrs, e.g. the compute_aabbs AABB provider, are
// resolved at world build by C2b/C2c; see src/constraint/collidable.hpp).
struct CollidableTypeInfo {
    CollidableType       type;
    AccelStructureKind   accel;
    ShapeProxyKind       proxy;
    ReactionProviderKind react;
};

// Indexed by static_cast<uint8_t>(CollidableType); ids are contiguous 0..N-1.
inline constexpr CollidableTypeInfo kCollidableTypeTable[kCollidableTypeCount] = {
    { CollidableType::RigidBody, AccelStructureKind::LbvhRigid, ShapeProxyKind::ShapeBacked, ReactionProviderKind::RigidInvMass },
    { CollidableType::ArticulationLink, AccelStructureKind::LbvhRigid, ShapeProxyKind::ShapeBacked, ReactionProviderKind::ArticulationChainJ },
    { CollidableType::Particle, AccelStructureKind::UniformGridParticle, ShapeProxyKind::PointSphere, ReactionProviderKind::ParticleInvMass },
    { CollidableType::StaticWorld, AccelStructureKind::None, ShapeProxyKind::ShapeBacked, ReactionProviderKind::StaticNull },
};

// O(1) metadata lookup (direct index into the id-ordered table).
constexpr const CollidableTypeInfo& GetCollidableTypeInfo(CollidableType t) {
    return kCollidableTypeTable[static_cast<uint8_t>(t)];
}

} // namespace nuka::constraint::generated
