#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::link_aabb -- per-link / per-shape WORLD-space AABB extraction
// for ARTICULATION collision shapes (v0.8 co-residence PEAK, commit 1/3)
// ---------------------------------------------------------------------------
// The artic-link<->rigid broadphase prerequisite that C2b's rigid<->rigid
// candidate-pair stream deliberately EXCLUDED: it had no per-link AABB, because
// articulation links do not carry a precomputed world AABB the way rigid bodies'
// cooked shapes feed `BuildRigidCandidatePairs(... device_shape_aabbs ...)`.
//
// This module closes that gap. Given a shape's LOCAL params (ShapeType +
// radius/half_extents/half_height + a local Transform relative to its owning
// link) and that link's WORLD Transform (from articulation FK --
// articulation::UpdateWorldLinkPoses), it returns the tight WORLD-space
// `collision::AABB` of the shape -- the SAME `collision::AABB` type the rigid
// broadphase consumes, so commit 2 can feed an articulation's link-shape AABBs
// straight into the rigid AABB stream.
//
// VALIDATED-NOT-WIRED. Nothing here is wired into world_stepper. Commit 2 (a
// separate change) extends the rigid broadphase to consume these AABBs; commit 3
// wires the two-way artic<->rigid gate. The OUTPUT is in the broadphase's input
// format on purpose (see ExtractLinkShapeAabbs's contract below).
//
// DEVICE-CALLABLE BY CONSTRUCTION. The CORE shape->AABB math is `__host__
// __device__` so commit 2 (and later on-device kernels) can call it inside a
// `.cu`. To make "annotated HD" mean "actually compiles on device" we route ALL
// math through:
//   * nuka::math::Vec3's NUKA_MATH_HD ops (the only math type with real HD ops),
//   * our own HD free helpers below (RotateByQuat / RotateByQuatPair), and
//   * direct .min/.max assignment via HD fminf/fmaxf.
// We DO NOT call ANY `Quat::`/`Transform::`/`AABB::` member here, AND WE NEVER
// CONSTRUCT a `math::Quat`: Quat::Rotate and Transform::TransformPoint are
// host-only `inline`; Quat::operator* AND the Quat ctor are `constexpr`-but-not-HD
// (device-callable only under --expt-relaxed-constexpr, which we must not assume);
// AABB::Expand/Merge/FromSphere/FromBox pull host-only std::min/std::max from
// <algorithm>. (An earlier MulQuat returned `math::Quat{...}` thinking it was
// aggregate init -- it actually invokes the host-only ctor and silently
// miscompiled the device box/capsule paths; RotateByQuatPair composes two
// rotations WITHOUT a product Quat instead.) The HD kernel device-check in the
// test PROVES this compiles + runs on-device matching the host path.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/canonical_types.hpp"   // scene::ShapeType
#include "scene/cooked_blob.hpp"       // scene::CookedShapeTable (host glue only)

#include <cstdint>
#include <vector>

#if defined(__CUDACC__)
#define NUKA_LINK_AABB_HD __host__ __device__
#else
#define NUKA_LINK_AABB_HD
#endif

namespace nuka::collision {

// --- HD math helpers (no Quat::/Transform:: methods on the device path) -------

// Rotate vector `v` by the unit quaternion (w,x,y,z). Same formula as
// Quat::Rotate (v + w*(2 qv x v) + qv x (2 qv x v)) but built ONLY from Vec3's
// NUKA_MATH_HD ops so it compiles on device without --expt-relaxed-constexpr.
NUKA_LINK_AABB_HD inline math::Vec3 RotateByQuat(float w, float x, float y, float z,
                                                 math::Vec3 v) {
    const math::Vec3 qv{x, y, z};
    const math::Vec3 t = 2.0f * qv.Cross(v);
    return v + (w * t) + qv.Cross(t);
}

// Convenience overload taking a math::Quat (reads its fields directly; does NOT
// call Quat::Rotate).
NUKA_LINK_AABB_HD inline math::Vec3 RotateByQuat(const math::Quat& q, math::Vec3 v) {
    return RotateByQuat(q.w, q.x, q.y, q.z, v);
}

// Rotate `v` by the COMPOSED rotation (outer * inner) WITHOUT materialising the
// product quaternion: the rotation map is a homomorphism, so
//     R(outer * inner) v  ==  R(outer)( R(inner) v ).
// We MUST NOT form the product as a `math::Quat`: `math::Quat{...}` invokes Quat's
// user-declared constexpr __host__ constructor, which is NOT device-callable
// (nvcc warning #20013) -- on the device that path silently miscompiles (it
// produced a ~2 m AABB error, caught by DeviceCoreMatchesHost). RotateByQuat only
// READS quat fields and returns a Vec3, so chaining two of them is fully HD.
NUKA_LINK_AABB_HD inline math::Vec3 RotateByQuatPair(const math::Quat& outer,
                                                     const math::Quat& inner,
                                                     math::Vec3 v) {
    return RotateByQuat(outer, RotateByQuat(inner, v));
}

// HD scalar min/max so we never reach for host-only std::min/std::max.
NUKA_LINK_AABB_HD inline float MinHd(float a, float b) { return a < b ? a : b; }
NUKA_LINK_AABB_HD inline float MaxHd(float a, float b) { return a > b ? a : b; }

// Grow `box` to include `p` (the HD analogue of AABB::Expand, which is host-only
// because it uses std::min/std::max).
NUKA_LINK_AABB_HD inline void ExpandHd(AABB& box, math::Vec3 p) {
    box.min.x = MinHd(box.min.x, p.x);
    box.min.y = MinHd(box.min.y, p.y);
    box.min.z = MinHd(box.min.z, p.z);
    box.max.x = MaxHd(box.max.x, p.x);
    box.max.y = MaxHd(box.max.y, p.y);
    box.max.z = MaxHd(box.max.z, p.z);
}

// --- the link world pose applied to a shape-local point -----------------------

// World position of a shape-local point: rotate by the link's world rotation,
// then add the link's world position. (HD analogue of Transform::TransformPoint.)
NUKA_LINK_AABB_HD inline math::Vec3 LinkPointToWorld(const math::Transform& link,
                                                     math::Vec3 local_point) {
    return RotateByQuat(link.rotation, local_point) + link.position;
}

// --- per-shape WORLD AABB (the device-callable CORE) --------------------------

// SPHERE world AABB = world_center +/- radius. The world center is the link pose
// applied to the shape's local-transform position; the sphere is
// rotation-invariant so the local/link ROTATION does not affect the extent --
// only the center. This AABB is EXACT (tight): each axis extent == 2*radius.
NUKA_LINK_AABB_HD inline AABB SphereWorldAabb(const math::Transform& link,
                                              const math::Transform& local,
                                              float radius) {
    const math::Vec3 c = LinkPointToWorld(link, local.position);
    AABB box;
    box.min = math::Vec3{c.x - radius, c.y - radius, c.z - radius};
    box.max = math::Vec3{c.x + radius, c.y + radius, c.z + radius};
    return box;
}

// BOX world AABB by transforming all 8 OBB corners (local half_extents under the
// shape's LOCAL rotation, then the link's world rotation+translation) and taking
// the min/max. The box's world orientation is link.rotation * local.rotation, so
// the LOCAL rotation is honored (a rotated box's AABB is larger than its
// axis-aligned one -- the test asserts this). The closed form
// abs(R)*half_extents would be cheaper but the 8-corner walk is unambiguously
// tight and is what the test's independent reference checks against.
NUKA_LINK_AABB_HD inline AABB BoxWorldAabb(const math::Transform& link,
                                           const math::Transform& local,
                                           math::Vec3 half_extents) {
    // Box world orientation = link.rotation * local.rotation; the box center in
    // world = link applied to the local center. We rotate each corner by the
    // composed rotation via RotateByQuatPair (no product Quat -> device-safe).
    const math::Vec3 world_center = LinkPointToWorld(link, local.position);
    AABB box;  // starts inverted (min=+FLT_MAX, max=-FLT_MAX) -> ExpandHd seeds it.
    for (int i = 0; i < 8; ++i) {
        const math::Vec3 local_corner{
            (i & 1) ? half_extents.x : -half_extents.x,
            (i & 2) ? half_extents.y : -half_extents.y,
            (i & 4) ? half_extents.z : -half_extents.z};
        const math::Vec3 world_corner =
            RotateByQuatPair(link.rotation, local.rotation, local_corner) + world_center;
        ExpandHd(box, world_corner);
    }
    return box;
}

// CAPSULE world AABB = union of the two spherical caps. The capsule axis is the
// shape's LOCAL Y (the engine convention -- see analytical_manifold.hpp
// CapsulePlane/CapsuleSphere: "capsule axis = local Y, half_height"); the caps
// sit at the local center +/- half_height along that axis, each padded by radius.
NUKA_LINK_AABB_HD inline AABB CapsuleWorldAabb(const math::Transform& link,
                                               const math::Transform& local,
                                               float radius, float half_height) {
    // Capsule local axis = local Y, expressed in world after local then link rot
    // (composed rotation via RotateByQuatPair -> no product Quat -> device-safe).
    const math::Vec3 axis_world =
        RotateByQuatPair(link.rotation, local.rotation, math::Vec3{0.0f, 1.0f, 0.0f});
    const math::Vec3 world_center = LinkPointToWorld(link, local.position);
    const math::Vec3 cap_a = world_center + (axis_world * half_height);
    const math::Vec3 cap_b = world_center - (axis_world * half_height);
    AABB box;
    // Each cap is a sphere of `radius`; union their two padded boxes.
    box.min = math::Vec3{MinHd(cap_a.x, cap_b.x) - radius,
                         MinHd(cap_a.y, cap_b.y) - radius,
                         MinHd(cap_a.z, cap_b.z) - radius};
    box.max = math::Vec3{MaxHd(cap_a.x, cap_b.x) + radius,
                         MaxHd(cap_a.y, cap_b.y) + radius,
                         MaxHd(cap_a.z, cap_b.z) + radius};
    return box;
}

// --- the by-type dispatch (device-callable) -----------------------------------

// World AABB of ONE shape from its ShapeType + local params + the owning link's
// WORLD pose. Unsupported / non-AABBed types (Plane is unbounded; ConvexHull/
// TriMesh/HeightField have no primitive params here) return an inverted AABB
// (min > max), which Overlaps()/the broadphase treats as "no overlap" -- a safe
// no-op for v0.8's primitive-only articulation collision shapes.
NUKA_LINK_AABB_HD inline AABB ShapeWorldAabb(scene::ShapeType type,
                                             const math::Transform& link,
                                             const math::Transform& local,
                                             math::Vec3 half_extents,
                                             float radius, float half_height) {
    switch (type) {
        case scene::ShapeType::Sphere:
            return SphereWorldAabb(link, local, radius);
        case scene::ShapeType::Box:
            return BoxWorldAabb(link, local, half_extents);
        case scene::ShapeType::Capsule:
            return CapsuleWorldAabb(link, local, radius, half_height);
        default:
            return AABB{};  // inverted sentinel (min=+FLT_MAX, max=-FLT_MAX).
    }
}

// --- host-side convenience: walk a shape table -> world AABBs -----------------
//
// OUTPUT CONTRACT (so commit 2 can feed the rigid broadphase directly):
//   ExtractLinkShapeAabbs walks `shapes` and, for every shape that maps to a
//   live articulation link AND has a primitive type with a bounded AABB
//   (Sphere/Box/Capsule), emits ONE entry. The three output vectors are PARALLEL
//   (same length, same order) and are emitted in ascending shape-table order, so
//   they are layout-compatible with the rigid convention
//   `BuildRigidCandidatePairs(... const collision::AABB* device_shape_aabbs,
//   uint32_t shape_count, const uint32_t* shape_body_ids, ...)`:
//     * `aabbs[i]`         -> the shape's tight WORLD AABB (upload as
//                             device_shape_aabbs),
//     * `shape_body_ids[i]`-> the owning BODY id (mirrors rigid shape_body_ids;
//                             used for the same-body drop + the exclude filter),
//     * `link_indices[i]`  -> the owning LINK index (the ArticulationLink
//                             CandidatePair `handle` space -- distinct from the
//                             body id; commit 2 emits THIS as the side handle so
//                             a contact reaction resolves the chain Jacobian),
//     * `source_shape_ids[i]` -> the GLOBAL shape-table index it came from (so
//                             commit 2 can recover any other shape_table field).
//   Shapes whose body maps to no link, or whose type is unbounded/unsupported
//   (Plane/ConvexHull/TriMesh/HeightField), are SKIPPED (no entry) -- the output
//   is dense, not 1:1 with the shape table.
//
// FK IS THE CALLER'S JOB. `link_world_poses[link]` must already hold the FK world
// pose of every link (e.g. from articulation::UpdateWorldLinkPoses). This keeps
// the convenience pure host glue over the device-callable CORE above. The
// shape->link map is the same one the cook pipeline uses: a shape's owning body
// is shapes.body_ids[shape]; the link that owns that body is the `link` where
// link_body_map[link] == body.
struct LinkShapeAabbs {
    std::vector<AABB>     aabbs;             // per emitted shape, world AABB.
    std::vector<uint32_t> shape_body_ids;    // per emitted shape, owning body id.
    std::vector<uint32_t> link_indices;      // per emitted shape, owning link idx
                                             // (the ArticulationLink handle space).
    std::vector<uint32_t> source_shape_ids;  // per emitted shape, shape-table idx.
};

inline LinkShapeAabbs ExtractLinkShapeAabbs(
    const scene::CookedShapeTable& shapes,
    const std::vector<math::Transform>& link_world_poses,
    const std::vector<uint32_t>& link_body_map) {
    LinkShapeAabbs out;
    const uint32_t shape_count = static_cast<uint32_t>(shapes.types.size());
    const uint32_t link_count = static_cast<uint32_t>(link_body_map.size());
    constexpr uint32_t kInvalidLink = ~0u;

    for (uint32_t shape = 0u; shape < shape_count; ++shape) {
        const scene::ShapeType type = shapes.types[shape];
        // Only Sphere/Box/Capsule carry a bounded primitive AABB here.
        if (type != scene::ShapeType::Sphere &&
            type != scene::ShapeType::Box &&
            type != scene::ShapeType::Capsule) {
            continue;
        }
        // shape -> owning body -> owning link (first link whose body matches).
        const uint32_t body = shapes.body_ids[shape];
        uint32_t owner_link = kInvalidLink;
        for (uint32_t link = 0u; link < link_count; ++link) {
            if (link_body_map[link] == body) {
                owner_link = link;
                break;
            }
        }
        if (owner_link == kInvalidLink) {
            continue;  // body is not an articulation link (e.g. a rigid body).
        }

        const math::Transform& link_pose = link_world_poses[owner_link];
        const math::Transform& local =
            shape < shapes.local_transforms.size()
                ? shapes.local_transforms[shape]
                : math::Transform::Identity();
        const math::Vec3 half_extents =
            shape < shapes.half_extents.size() ? shapes.half_extents[shape]
                                               : math::Vec3::Zero();
        const float radius =
            shape < shapes.radii.size() ? shapes.radii[shape] : 0.0f;
        const float half_height =
            shape < shapes.half_heights.size() ? shapes.half_heights[shape] : 0.0f;

        out.aabbs.push_back(
            ShapeWorldAabb(type, link_pose, local, half_extents, radius, half_height));
        out.shape_body_ids.push_back(body);
        out.link_indices.push_back(owner_link);  // the ArticulationLink handle.
        out.source_shape_ids.push_back(shape);
    }
    return out;
}

} // namespace nuka::collision

#undef NUKA_LINK_AABB_HD
