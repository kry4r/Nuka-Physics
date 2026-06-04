#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint -- the runtime collidable reference + the device-fn-ptr seam
// (v0.8 C2a)
// ---------------------------------------------------------------------------
// CollidableRef is the unified, type-tagged handle the broadphase emits and the
// narrowphase/solver consume: it names WHICH collidable (by type + handle) and
// carries the reaction-provider KIND so the solver knows how to build that side
// of a contact row WITHOUT a per-type branch table at the call site.
//
// Include DAG (acyclic):
//   constraint/collidable_kinds.hpp           (leaf: the kind enums)
//     -> codegen/generated/collidable_registry.hpp  (CollidableType + static table)
//       -> constraint/collidable.hpp           (THIS: CollidableRef + the seam)
// THIS header includes BOTH; nothing it includes includes it back.
// ---------------------------------------------------------------------------

#include "codegen/generated/collidable_registry.hpp"  // CollidableType, kCollidableTypeTable
#include "constraint/collidable_kinds.hpp"             // ReactionProviderKind

#include <cstdint>

namespace nuka::constraint {

// Pull the GENERATED type enum into nuka::constraint so callers write
// CollidableType::RigidBody without the ::generated qualifier.
using generated::CollidableType;
using generated::CollidableTypeInfo;
using generated::GetCollidableTypeInfo;
using generated::kCollidableTypeCount;
using generated::kCollidableTypeTable;

// A type-tagged reference to one collidable. `handle` is the per-type id the
// downstream provider resolves: a maximal body id (RigidBody), a GLOBAL link id
// (ArticulationLink), a particle id (Particle), or a static-collider id
// (StaticWorld). `react` is duplicated from the type's static metadata so the
// solver reads it off the ref directly (the broadphase fills it from
// GetCollidableTypeInfo(type).react at emit time). The default is a benign
// (RigidBody, RigidInvMass, sentinel-handle) -- a sentinel handle ~0u marks an
// unset ref.
struct CollidableRef {
    CollidableType       type   = CollidableType::RigidBody;
    ReactionProviderKind react  = ReactionProviderKind::RigidInvMass;
    uint32_t             handle = ~0u;  // body id / global link id / particle id
};

// ---------------------------------------------------------------------------
// §0.4 SEAM: the per-type runtime AABB provider (compute_aabbs) -- DEFERRED.
// ---------------------------------------------------------------------------
// The STATIC registry above (CollidableType + kCollidableTypeTable) is the
// compile-time half. The RUNTIME half is a per-type DEVICE function pointer
// table that maps each CollidableType to its `compute_aabbs` kernel entry (and,
// later, to its narrowphase + reaction providers). Those device-fn-ptrs CANNOT
// live in the constexpr table -- they are resolved against actual world buffers
// at WORLD BUILD.
//
// NAMED DOWNSTREAM CONSUMERS (this seam is NOT an orphan):
//   - C2b builds the per-type broadphase AABB provider table (LbvhRigid for
//     RigidBody/ArticulationLink, UniformGridParticle for Particle, None for
//     StaticWorld -- keyed off GetCollidableTypeInfo(type).accel) and runs it to
//     fill the candidate stream (candidate_pair.hpp) from the real broadphase.
//   - C2c resolves the per-type reaction providers (keyed off .react) so the
//     unified solver builds each contact row's two sides class-blind.
//
// The resolver SIGNATURE below is a forward-declared stub ONLY -- it is the
// shape C2b will implement; C2a does NOT define it (no providers exist yet).
// Keeping it here documents the seam at the type the providers attach to.

// Opaque world-build context (defined by C2b: carries the device buffer views +
// accel structures the providers close over). Forward-declared so this header
// stays dependency-light; C2a does not need its definition.
struct CollidableWorld;

// The runtime AABB-provider entry a built world resolves per CollidableType.
// IMPLEMENTED BY C2b -- declared here, intentionally NOT defined in C2a.
//   void ResolveAabbProviders(const CollidableWorld& world, ...);
// (Signature finalized in C2b once CollidableWorld's buffer layout is fixed.)

} // namespace nuka::constraint
