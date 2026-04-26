// ---------------------------------------------------------------------------
// nuka::scene::CookScene – implementation
// ---------------------------------------------------------------------------

#include "scene/cooker.hpp"

namespace nuka::scene {

CookedBlob CookScene(const SceneIR& scene) {
    CookedBlob blob;

    // -- Bodies -------------------------------------------------------------
    const auto& bodies = scene.Bodies();
    blob.body_count = static_cast<uint32_t>(bodies.size());

    blob.bodies.poses.reserve(bodies.size());
    blob.bodies.inv_masses.reserve(bodies.size());
    blob.bodies.inv_inertias.reserve(bodies.size());
    blob.bodies.is_static.reserve(bodies.size());

    for (const auto& b : bodies) {
        blob.bodies.poses.push_back(b.local_transform);

        if (b.is_static) {
            blob.bodies.inv_masses.push_back(0.0f);
            blob.bodies.inv_inertias.push_back(math::Vec3::Zero());
        } else {
            blob.bodies.inv_masses.push_back(
                (b.mass > 0.0f) ? (1.0f / b.mass) : 0.0f);
            blob.bodies.inv_inertias.push_back({
                (b.inertia.x > 0.0f) ? (1.0f / b.inertia.x) : 0.0f,
                (b.inertia.y > 0.0f) ? (1.0f / b.inertia.y) : 0.0f,
                (b.inertia.z > 0.0f) ? (1.0f / b.inertia.z) : 0.0f
            });
        }

        blob.bodies.is_static.push_back(b.is_static ? uint8_t(1) : uint8_t(0));
    }

    // -- Joints -------------------------------------------------------------
    const auto& joints = scene.Joints();
    blob.joint_count = static_cast<uint32_t>(joints.size());

    blob.joints.types.reserve(joints.size());
    blob.joints.parent_bodies.reserve(joints.size());
    blob.joints.child_bodies.reserve(joints.size());
    blob.joints.axes.reserve(joints.size());
    blob.joints.lower_limits.reserve(joints.size());
    blob.joints.upper_limits.reserve(joints.size());

    for (const auto& j : joints) {
        blob.joints.types.push_back(j.type);
        blob.joints.parent_bodies.push_back(j.parent_body);
        blob.joints.child_bodies.push_back(j.child_body);
        blob.joints.axes.push_back(j.axis);
        blob.joints.lower_limits.push_back(j.lower_limit);
        blob.joints.upper_limits.push_back(j.upper_limit);
    }

    // -- Shapes -------------------------------------------------------------
    const auto& shapes = scene.Shapes();
    blob.shape_count = static_cast<uint32_t>(shapes.size());

    blob.shapes.types.reserve(shapes.size());
    blob.shapes.body_ids.reserve(shapes.size());
    blob.shapes.local_transforms.reserve(shapes.size());
    blob.shapes.half_extents.reserve(shapes.size());
    blob.shapes.radii.reserve(shapes.size());

    for (const auto& s : shapes) {
        blob.shapes.types.push_back(s.type);
        blob.shapes.body_ids.push_back(s.body_id);
        blob.shapes.local_transforms.push_back(s.local_transform);
        blob.shapes.half_extents.push_back(s.half_extents);
        blob.shapes.radii.push_back(s.radius);
    }

    return blob;
}

} // namespace nuka::scene
