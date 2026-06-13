#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::cook — FindRestPlacement (M7 T2): the geometric drop-to-rest.
//
// Generalizes the kitchen-counter "seat the object on the support surface"
// computation (the untracked test_scene_compose_h1_cup_table drop-to-rest):
//   1. ObjectRootRelativeMinZ — the object's LOWEST collision/mesh vertex in
//      its ROOT body frame (the cup bottom). Over every collision shape owned
//      by the object subtree, in the object root frame.
//   2. SupportSurfaceTopZ — a support body's collision AABB TOP z in world.
//   3. FindRestPlacement — place the object root so its lowest point rests at
//      support_top_z + clearance (the object bottom seats `clearance` above the
//      support top).
//
// PURE geometry, host-side, DETERMINISTIC (fixed iteration order over the
// SceneIR records; no RNG, no atomics). Both the test_scene_compose drop-to-
// rest and the cup-on-table gate call the same primitive.
//
// nk_engine lint scope: CUDA-type-free C++.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/canonical_types.hpp"
#include "scene/scene_ir.hpp"

namespace nuka::scene::cook {

// World transform of a body (walk parent chain to the root, compose locals).
// Deterministic; matches the test_scene_compose WorldOf helper.
math::Transform BodyWorldTransform(const SceneIR& scene, BodyId body);

// The object's lowest collision-geometry point expressed in `root`'s frame.
// Covers every CollisionShapeRecord whose body is `root` or a descendant of
// `root`: box corners, sphere (center - r), capsule (center +- (hh+r) along its
// local axis extremes), and mesh vertices (when present). Returns the minimal z
// over all sampled support points. +inf if the object has no collision geometry.
float ObjectRootRelativeMinZ(const SceneIR& scene, BodyId root);

// The world-space TOP z of a support body's collision geometry (the max z over
// the same per-shape support-point set as above, in WORLD coordinates). -inf if
// the support body has no collision geometry. Used to resolve a support's
// resting surface height from its cooked collision AABB.
float SupportBodyTopZ(const SceneIR& scene, BodyId support);

// The world-space TOP z of a single support SHAPE (box/sphere/capsule/mesh).
// Same support-point convention as SupportBodyTopZ but for one named shape (the
// kitchen-counter top shape selection path resolves a specific top shape).
float SupportShapeTopZ(const SceneIR& scene, ShapeId support_shape);

// --- the rest placement primitive (3 overloads, cleanest-signature trio) ----

// (A) Pure scalar form: given the object's root-relative min-z and a target XY,
// return the object-root world transform that seats the object bottom at
// `support_top_z + clearance`. Identity rotation. The lowest-level building
// block (the scalar the gate asserts on).
math::Transform FindRestPlacement(float object_root_relative_min_z,
                                  const math::Vec3& xy_target,
                                  float support_top_z,
                                  float clearance = 1.0e-3f);

// (B) SceneIR form resolving the support's top from a support BODY's AABB. The
// XY target defaults to the support's world XY center (drop straight down onto
// the surface centre); pass an explicit xy when seating off-centre (the kitchen
// front-offset). Returns the object root world transform.
math::Transform FindRestPlacement(const SceneIR& scene,
                                  BodyId object_root,
                                  BodyId support_body,
                                  float clearance = 1.0e-3f);

// (C) SceneIR form resolving the support's top from a specific support SHAPE
// (the kitchen counter-top shape), object seated at an explicit XY target.
math::Transform FindRestPlacement(const SceneIR& scene,
                                  BodyId object_root,
                                  ShapeId support_shape,
                                  const math::Vec3& xy_target,
                                  float clearance = 1.0e-3f);

}  // namespace nuka::scene::cook
