#pragma once
// ---------------------------------------------------------------------------
// General contact pipeline (Phase 0, R2) — the ONE shape-kind enum.
//
// Before this header the kKind* sentinels were copy-pasted across the CUDA
// collision ops (broadphase.cu, narrowphase_prims.cu, ...), each a private
// `constexpr uint32_t kKind* = N` block — several sources that
// could silently diverge. This header is the single source of truth; the
// surviving sentinel sites include it so a new kind (Heightfield / TrianglePrism)
// is added in ONE place.
//
// The numeric values are the EXISTING wire order the cooked `shape_table` packs
// (kind stored as the u32 bit-pattern of shape_table[row*8 + 0]) and the broad/
// narrowphase already dispatch on: Sphere=0, Capsule=1, Box=2, Plane=3,
// ConvexHull=4, SdfMesh=5 — they mirror scene::CollisionShapeComponent::Kind /
// scene::ShapeType. Heightfield(6) and Cylinder(7) follow; the per-cell heightfield
// prism is a support kind only and never appears in the shape table.
//
// Plain `enum` (not `enum class`) with an explicit u32 underlying type so the
// device kernels can compare it directly against the u32 unpacked from the
// shape_table without a cast, and a host packer can store it as a u32 bit-pattern
// byte-identically to the prior literals.
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::collision {

enum ShapeKind : uint32_t {
    kShapeSphere      = 0u,
    kShapeCapsule     = 1u,
    kShapeBox         = 2u,
    kShapePlane       = 3u,
    kShapeConvexHull  = 4u,
    kShapeSdfMesh     = 5u,
    kShapeHeightfield = 6u,
    kShapeCylinder    = 7u,   // flat caps along local Z; params (radius, half_height)
};

#if defined(__CUDACC__)
#define NUKA_COLLISION_HD __host__ __device__
#else
#define NUKA_COLLISION_HD
#endif

NUKA_COLLISION_HD inline bool ShapeFilterAdmits(uint32_t contype_a, uint32_t conaffinity_a,
                                               uint32_t group_a, uint32_t contype_b,
                                               uint32_t conaffinity_b, uint32_t group_b) {
    if (((contype_a & conaffinity_b) | (contype_b & conaffinity_a)) == 0u) return false;
    const int32_t ga = static_cast<int32_t>(group_a);
    const int32_t gb = static_cast<int32_t>(group_b);
    return !(ga != 0 && gb != 0 && (ga > 0 ? ga != gb && gb > 0 : ga == gb));
}

enum class MeshContactRoute : uint8_t { Analytic, Ogc, Sdf };

NUKA_COLLISION_HD inline MeshContactRoute RouteMeshContact(
    uint32_t kind_a, uint32_t kind_b, uint8_t mode_a, uint8_t mode_b) {
    const bool mesh_a = kind_a == kShapeConvexHull || kind_a == kShapeSdfMesh;
    const bool mesh_b = kind_b == kShapeConvexHull || kind_b == kShapeSdfMesh;
    if (!mesh_a && !mesh_b) return MeshContactRoute::Analytic;
    return mesh_a && mesh_b && mode_a == 1u && mode_b == 1u
        ? MeshContactRoute::Sdf : MeshContactRoute::Ogc;
}

// Primitive pairs with a closed-form manifold; other convex pairs take GJK/EPA.
NUKA_COLLISION_HD inline bool HasAnalyticManifold(uint32_t a, uint32_t b) {
    if (a > b) { const uint32_t t = a; a = b; b = t; }
    return (a == kShapeSphere && (b == kShapeSphere || b == kShapeCapsule || b == kShapeBox || b == kShapePlane)) ||
           (a == kShapeCapsule && (b == kShapeCapsule || b == kShapePlane)) ||
           (a == kShapeBox && (b == kShapeBox || b == kShapePlane)) ||
           (b == kShapeCylinder && (a == kShapeSphere || a == kShapePlane));
}

NUKA_COLLISION_HD inline bool IsConvexShape(uint32_t kind) {
    return kind == kShapeSphere || kind == kShapeCapsule || kind == kShapeBox || kind == kShapeConvexHull ||
           kind == kShapeCylinder;
}

}  // namespace nuka::collision
