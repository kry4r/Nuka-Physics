// ---------------------------------------------------------------------------
// nuka::scene::cook — FindRestPlacement implementation (M7 T2).
// ---------------------------------------------------------------------------

#include "scene/cook/placement.hpp"

#include <array>
#include <limits>

namespace nuka::scene::cook {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Is `body` == `root` or a transitive descendant of `root` (parent chain)?
bool IsInSubtree(const SceneIR& scene, BodyId body, BodyId root) {
    BodyId cur = body;
    // Bounded walk (cycles cannot occur in a cooked tree, but cap defensively).
    for (size_t guard = 0; guard <= scene.RigidBodyCount() && cur != kInvalidBody;
         ++guard) {
        if (cur == root) return true;
        cur = scene.GetBody(cur).parent_id;
    }
    return false;
}

// Sample the SUPPORT POINTS of one collision shape in the shape's LOCAL frame:
// the extreme geometry points whose extrema bound the shape's z-extent. The
// caller maps each into the frame it needs (root-local for min, world for top).
//   * Box      -> 8 corners (+- half extents).
//   * Sphere   -> center +- r along each axis (6 points; z extrema suffice but
//                 the full set keeps the helper shape-agnostic).
//   * Capsule  -> the two cap centers (+- half_height on the LOCAL z axis, the
//                 cooker's capsule axis convention) each +- r.
//   * Mesh     -> the source vertices (when retained).
// Appends to `out`. Deterministic order.
void AppendShapeSupportPointsLocal(const CollisionShapeRecord& shape,
                                   std::vector<math::Vec3>& out) {
    using math::Vec3;
    switch (shape.type) {
        case ShapeType::Box: {
            const Vec3 h = shape.half_extents;
            const std::array<Vec3, 8> corners = {{
                {-h.x, -h.y, -h.z}, {h.x, -h.y, -h.z},
                {h.x, h.y, -h.z},   {-h.x, h.y, -h.z},
                {-h.x, -h.y, h.z},  {h.x, -h.y, h.z},
                {h.x, h.y, h.z},    {-h.x, h.y, h.z},
            }};
            for (const Vec3& c : corners) out.push_back(c);
            break;
        }
        case ShapeType::Sphere: {
            const float r = shape.radius;
            out.push_back({r, 0.0f, 0.0f});
            out.push_back({-r, 0.0f, 0.0f});
            out.push_back({0.0f, r, 0.0f});
            out.push_back({0.0f, -r, 0.0f});
            out.push_back({0.0f, 0.0f, r});
            out.push_back({0.0f, 0.0f, -r});
            break;
        }
        case ShapeType::Capsule: {
            const float r = shape.radius;
            const float hh = shape.half_height;
            // Local z-axis capsule (the cooker convention); the two spherical
            // caps centered at +- hh, each bounded by +- r on every axis.
            for (float s : {-1.0f, 1.0f}) {
                const float cz = s * hh;
                out.push_back({0.0f, 0.0f, cz + r});
                out.push_back({0.0f, 0.0f, cz - r});
                out.push_back({r, 0.0f, cz});
                out.push_back({-r, 0.0f, cz});
                out.push_back({0.0f, r, cz});
                out.push_back({0.0f, -r, cz});
            }
            break;
        }
        case ShapeType::ConvexHull:
        case ShapeType::TriMesh:
        case ShapeType::HeightField:
        case ShapeType::Plane:
        default: {
            // Mesh-backed shapes: the source vertices when present. A Plane (or
            // an unretained mesh) contributes nothing here (no bounded z-extent
            // for a plane; the support comes from a box/sphere counter top).
            const std::vector<float>& v = shape.mesh_vertices;
            for (size_t i = 0; i + 2 < v.size(); i += 3) {
                out.push_back({v[i + 0], v[i + 1], v[i + 2]});
            }
            break;
        }
    }
}

}  // namespace

math::Transform BodyWorldTransform(const SceneIR& scene, BodyId body) {
    // Collect the parent chain root..body, compose locals top-down.
    std::vector<BodyId> chain;
    BodyId cur = body;
    for (size_t guard = 0; guard <= scene.RigidBodyCount() && cur != kInvalidBody;
         ++guard) {
        chain.push_back(cur);
        cur = scene.GetBody(cur).parent_id;
    }
    math::Transform world = math::Transform::Identity();
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        world = world * scene.GetBody(*it).local_transform;
    }
    return world;
}

float ObjectRootRelativeMinZ(const SceneIR& scene, BodyId root) {
    if (root == kInvalidBody || root >= scene.RigidBodyCount()) return kInf;
    const math::Transform root_inv = BodyWorldTransform(scene, root).Inverse();
    float min_z = kInf;
    std::vector<math::Vec3> local;
    for (size_t s = 0; s < scene.ShapeCount(); ++s) {
        const CollisionShapeRecord& shape = scene.GetShape(static_cast<ShapeId>(s));
        if (shape.body_id == kInvalidBody) continue;
        if (!IsInSubtree(scene, shape.body_id, root)) continue;
        const math::Transform shape_world =
            BodyWorldTransform(scene, shape.body_id) * shape.local_transform;
        local.clear();
        AppendShapeSupportPointsLocal(shape, local);
        for (const math::Vec3& p : local) {
            const math::Vec3 in_root =
                root_inv.TransformPoint(shape_world.TransformPoint(p));
            if (in_root.z < min_z) min_z = in_root.z;
        }
    }
    return min_z;
}

float SupportShapeTopZ(const SceneIR& scene, ShapeId support_shape) {
    if (support_shape == kInvalidShape || support_shape >= scene.ShapeCount()) {
        return -kInf;
    }
    const CollisionShapeRecord& shape = scene.GetShape(support_shape);
    if (shape.body_id == kInvalidBody) return -kInf;
    const math::Transform shape_world =
        BodyWorldTransform(scene, shape.body_id) * shape.local_transform;
    std::vector<math::Vec3> local;
    AppendShapeSupportPointsLocal(shape, local);
    float top_z = -kInf;
    for (const math::Vec3& p : local) {
        const float z = shape_world.TransformPoint(p).z;
        if (z > top_z) top_z = z;
    }
    return top_z;
}

float SupportBodyTopZ(const SceneIR& scene, BodyId support) {
    if (support == kInvalidBody || support >= scene.RigidBodyCount()) return -kInf;
    float top_z = -kInf;
    for (size_t s = 0; s < scene.ShapeCount(); ++s) {
        const CollisionShapeRecord& shape = scene.GetShape(static_cast<ShapeId>(s));
        if (shape.body_id != support) continue;
        const float z = SupportShapeTopZ(scene, static_cast<ShapeId>(s));
        if (z > top_z) top_z = z;
    }
    return top_z;
}

math::Transform FindRestPlacement(float object_root_relative_min_z,
                                  const math::Vec3& xy_target,
                                  float support_top_z,
                                  float clearance) {
    math::Transform t = math::Transform::Identity();
    t.position = math::Vec3{xy_target.x, xy_target.y,
                            support_top_z - object_root_relative_min_z + clearance};
    return t;
}

math::Transform FindRestPlacement(const SceneIR& scene, BodyId object_root,
                                  BodyId support_body, float clearance) {
    const float min_z = ObjectRootRelativeMinZ(scene, object_root);
    const float top_z = SupportBodyTopZ(scene, support_body);
    // XY target = the support body's world XY centre (drop straight down).
    const math::Vec3 support_origin =
        BodyWorldTransform(scene, support_body).position;
    return FindRestPlacement(min_z, support_origin, top_z, clearance);
}

math::Transform FindRestPlacement(const SceneIR& scene, BodyId object_root,
                                  ShapeId support_shape,
                                  const math::Vec3& xy_target, float clearance) {
    const float min_z = ObjectRootRelativeMinZ(scene, object_root);
    const float top_z = SupportShapeTopZ(scene, support_shape);
    return FindRestPlacement(min_z, xy_target, top_z, clearance);
}

}  // namespace nuka::scene::cook
