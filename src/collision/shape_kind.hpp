#pragma once
// Shared shape kinds retain the scene and cooked shape-table wire values.
// Geometry representation selects contact generation independently of the solver and device backend.

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

// Rigid surface contacts preserve this clearance in their speculative constraints.
inline constexpr float kRigidContactSeparation = 0.0005f;

// Heightfield triangle prisms share this extrusion in detection and broadphase bounds.
NUKA_COLLISION_HD inline float HeightfieldPrismExtrusion(float height_range) {
    return (height_range > 0.0f ? height_range : 0.0f) + 1.0f;
}

NUKA_COLLISION_HD inline bool IsConvexShape(uint32_t kind) {
    return kind == kShapeSphere || kind == kShapeCapsule || kind == kShapeBox || kind == kShapeConvexHull ||
           kind == kShapeCylinder;
}

NUKA_COLLISION_HD inline MeshContactRoute RouteMeshContact(
    uint32_t kind_a, uint32_t kind_b, uint8_t mode_a, uint8_t mode_b) {
    const bool mesh_a = kind_a == kShapeConvexHull || kind_a == kShapeSdfMesh;
    const bool mesh_b = kind_b == kShapeConvexHull || kind_b == kShapeSdfMesh;
    if (mesh_a && mesh_b && mode_a == 1u && mode_b == 1u) return MeshContactRoute::Sdf;
    if (IsConvexShape(kind_a) && IsConvexShape(kind_b)) return MeshContactRoute::Analytic;
    if (!mesh_a && !mesh_b) return MeshContactRoute::Analytic;
    // Rounded primitives query the mesh surface without a tessellated source shape.
    if (kind_a == kShapeSphere || kind_a == kShapeCapsule ||
        kind_b == kShapeSphere || kind_b == kShapeCapsule) return MeshContactRoute::Sdf;
    return MeshContactRoute::Ogc;
}

// Primitive pairs with a closed-form manifold; other convex pairs take GJK/EPA.
NUKA_COLLISION_HD inline bool HasAnalyticManifold(uint32_t a, uint32_t b) {
    if (a > b) { const uint32_t t = a; a = b; b = t; }
    return (a == kShapeSphere && (b == kShapeSphere || b == kShapeCapsule || b == kShapeBox || b == kShapePlane)) ||
           (a == kShapeCapsule && (b == kShapeCapsule || b == kShapePlane)) ||
           (a == kShapeBox && (b == kShapeBox || b == kShapePlane)) ||
           (b == kShapeCylinder && (a == kShapeSphere || a == kShapePlane));
}

}  // namespace nuka::collision
