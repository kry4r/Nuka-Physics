// ---------------------------------------------------------------------------
// nuka::scene::CookScene implementation
// ---------------------------------------------------------------------------

#include "scene/cooker.hpp"

#include "import/cooker/convex_decomposition.hpp"

namespace nuka::scene {

namespace {

math::Transform ResolveWorldTransform(const SceneIR& scene, BodyId body_id) {
    const auto& body = scene.GetBody(body_id);
    if (body.parent_id == kInvalidBody) {
        return body.local_transform;
    }
    return ResolveWorldTransform(scene, body.parent_id) * body.local_transform;
}

import::cooker::DecomposeMode ToCookerMode(DecomposeMode mode) {
    switch (mode) {
        case DecomposeMode::Force: return import::cooker::DecomposeMode::Force;
        case DecomposeMode::Skip:  return import::cooker::DecomposeMode::Skip;
        case DecomposeMode::Auto:  break;
    }
    return import::cooker::DecomposeMode::Auto;
}

// Append one convex hull (vertices/indices/volume) into the cooked geometry
// table and return its index.
uint32_t AppendConvexGeometry(CookedConvexGeometry& geom,
                              const std::vector<float>& vertices,
                              const std::vector<uint32_t>& indices,
                              float volume) {
    const uint32_t index = geom.Count();
    geom.vertex_offsets.push_back(static_cast<uint32_t>(geom.vertices.size() / 3));
    geom.vertex_counts.push_back(static_cast<uint32_t>(vertices.size() / 3));
    geom.index_offsets.push_back(static_cast<uint32_t>(geom.indices.size()));
    geom.index_counts.push_back(static_cast<uint32_t>(indices.size()));
    geom.volumes.push_back(volume);
    geom.vertices.insert(geom.vertices.end(), vertices.begin(), vertices.end());
    geom.indices.insert(geom.indices.end(), indices.begin(), indices.end());
    return index;
}

// Push one cooked shape row (all parallel arrays stay the same length).
void PushShapeRow(CookedShapeTable& shapes,
                  const CollisionShapeRecord& src,
                  ShapeType type,
                  uint32_t convex_geometry_index) {
    shapes.types.push_back(type);
    shapes.body_ids.push_back(src.body_id);
    shapes.material_ids.push_back(src.material_id);
    shapes.local_transforms.push_back(src.local_transform);
    shapes.half_extents.push_back(src.half_extents);
    shapes.radii.push_back(src.radius);
    shapes.half_heights.push_back(src.half_height);
    shapes.convex_geometry_indices.push_back(convex_geometry_index);
}

} // namespace

CookedBlob CookScene(const SceneIR& scene) {
    CookedBlob blob;

    const auto& bodies = scene.Bodies();
    blob.body_count = static_cast<uint32_t>(bodies.size());
    blob.bodies.poses.reserve(bodies.size());
    blob.bodies.local_poses.reserve(bodies.size());
    blob.bodies.inertial_frames.reserve(bodies.size());
    blob.bodies.masses.reserve(bodies.size());
    blob.bodies.inertias.reserve(bodies.size());
    blob.bodies.inv_masses.reserve(bodies.size());
    blob.bodies.inv_inertias.reserve(bodies.size());
    blob.bodies.is_static.reserve(bodies.size());

    for (const auto& b : bodies) {
        blob.bodies.poses.push_back(ResolveWorldTransform(scene, b.id));
        blob.bodies.local_poses.push_back(b.local_transform);
        blob.bodies.inertial_frames.push_back(b.inertial_transform);
        blob.bodies.masses.push_back(b.is_static ? 0.0f : b.mass);
        blob.bodies.inertias.push_back(b.is_static ? math::Vec3::Zero() : b.inertia);

        if (b.is_static) {
            blob.bodies.inv_masses.push_back(0.0f);
            blob.bodies.inv_inertias.push_back(math::Vec3::Zero());
        } else {
            blob.bodies.inv_masses.push_back((b.mass > 0.0f) ? (1.0f / b.mass) : 0.0f);
            blob.bodies.inv_inertias.push_back({
                (b.inertia.x > 0.0f) ? (1.0f / b.inertia.x) : 0.0f,
                (b.inertia.y > 0.0f) ? (1.0f / b.inertia.y) : 0.0f,
                (b.inertia.z > 0.0f) ? (1.0f / b.inertia.z) : 0.0f
            });
        }

        blob.bodies.is_static.push_back(b.is_static ? uint8_t(1) : uint8_t(0));
    }

    const auto& joints = scene.Joints();
    blob.joint_count = static_cast<uint32_t>(joints.size());
    blob.joints.types.reserve(joints.size());
    blob.joints.parent_bodies.reserve(joints.size());
    blob.joints.child_bodies.reserve(joints.size());
    blob.joints.axes.reserve(joints.size());
    blob.joints.parent_frames.reserve(joints.size());
    blob.joints.child_frames.reserve(joints.size());
    blob.joints.lower_limits.reserve(joints.size());
    blob.joints.upper_limits.reserve(joints.size());
    blob.joints.dampings.reserve(joints.size());
    blob.joints.armatures.reserve(joints.size());
    blob.joints.initial_positions.reserve(joints.size());

    for (const auto& j : joints) {
        blob.joints.types.push_back(j.type);
        blob.joints.parent_bodies.push_back(j.parent_body);
        blob.joints.child_bodies.push_back(j.child_body);
        blob.joints.axes.push_back(j.axis);
        blob.joints.parent_frames.push_back(j.parent_frame);
        blob.joints.child_frames.push_back(j.child_frame);
        blob.joints.lower_limits.push_back(j.lower_limit);
        blob.joints.upper_limits.push_back(j.upper_limit);
        blob.joints.dampings.push_back(j.damping);
        blob.joints.armatures.push_back(j.armature);
        blob.joints.initial_positions.push_back(j.initial_position);
    }

    const auto& shapes = scene.Shapes();
    blob.shapes.types.reserve(shapes.size());
    blob.shapes.body_ids.reserve(shapes.size());
    blob.shapes.material_ids.reserve(shapes.size());
    blob.shapes.local_transforms.reserve(shapes.size());
    blob.shapes.half_extents.reserve(shapes.size());
    blob.shapes.radii.reserve(shapes.size());
    blob.shapes.half_heights.reserve(shapes.size());
    blob.shapes.convex_geometry_indices.reserve(shapes.size());

    for (const auto& s : shapes) {
        const bool is_mesh =
            (s.type == ShapeType::TriMesh || s.type == ShapeType::ConvexHull);
        const bool has_geometry = !s.mesh_vertices.empty() && !s.mesh_indices.empty();

        // Non-mesh shapes (and mesh shapes lacking geometry — e.g. importers
        // that do not yet load mesh files) pass through unchanged, one row.
        if (!is_mesh || !has_geometry) {
            PushShapeRow(blob.shapes, s, s.type, kNoConvexGeometry);
            continue;
        }

        const import::cooker::DecomposeMode mode = ToCookerMode(s.decompose_mode);

        if (mode == import::cooker::DecomposeMode::Skip) {
            // Treat the source mesh as a single convex piece (store its own
            // geometry as one ConvexHull; no V-HACD run).
            const uint32_t geom_index = AppendConvexGeometry(
                blob.convex_geometry, s.mesh_vertices, s.mesh_indices, 0.0f);
            PushShapeRow(blob.shapes, s, ShapeType::ConvexHull, geom_index);
            continue;
        }

        // Auto / Force => run V-HACD. (A convex input naturally yields 1 piece,
        // so Auto ~= Force at this phase.)
        import::cooker::ConvexDecompositionParams params;
        params.max_pieces = s.decompose_max_pieces;
        const auto result = import::cooker::DecomposeMesh(
            s.mesh_vertices.data(),
            static_cast<uint32_t>(s.mesh_vertices.size() / 3),
            s.mesh_indices.data(),
            static_cast<uint32_t>(s.mesh_indices.size() / 3),
            params);

        if (!result.succeeded || result.pieces.empty()) {
            // Decomposition failed: fall back to passing the mesh through as a
            // single ConvexHull carrying its own geometry (never drop the shape).
            const uint32_t geom_index = AppendConvexGeometry(
                blob.convex_geometry, s.mesh_vertices, s.mesh_indices, 0.0f);
            PushShapeRow(blob.shapes, s, ShapeType::ConvexHull, geom_index);
            continue;
        }

        for (const auto& piece : result.pieces) {
            const uint32_t geom_index = AppendConvexGeometry(
                blob.convex_geometry, piece.vertices, piece.indices, piece.volume);
            PushShapeRow(blob.shapes, s, ShapeType::ConvexHull, geom_index);
        }
    }

    blob.shape_count = static_cast<uint32_t>(blob.shapes.types.size());

    const auto& sensors = scene.Sensors();
    blob.sensor_count = static_cast<uint32_t>(sensors.size());
    blob.sensors.types.reserve(sensors.size());
    blob.sensors.attached_bodies.reserve(sensors.size());
    blob.sensors.local_transforms.reserve(sensors.size());
    blob.sensors.sample_rates_hz.reserve(sensors.size());

    for (const auto& s : sensors) {
        blob.sensors.types.push_back(s.type);
        blob.sensors.attached_bodies.push_back(s.attached_body);
        blob.sensors.local_transforms.push_back(s.local_transform);
        blob.sensors.sample_rates_hz.push_back(s.sample_rate_hz);
    }

    const auto& materials = scene.Materials();
    blob.material_count = static_cast<uint32_t>(materials.size());
    blob.materials.base_colors.reserve(materials.size());
    blob.materials.alphas.reserve(materials.size());
    blob.materials.roughnesses.reserve(materials.size());
    blob.materials.metallics.reserve(materials.size());

    for (const auto& m : materials) {
        blob.materials.base_colors.push_back(m.base_color);
        blob.materials.alphas.push_back(m.alpha);
        blob.materials.roughnesses.push_back(m.roughness);
        blob.materials.metallics.push_back(m.metallic);
    }

    const auto& cameras = scene.Cameras();
    blob.camera_count = static_cast<uint32_t>(cameras.size());
    blob.cameras.attached_bodies.reserve(cameras.size());
    blob.cameras.local_transforms.reserve(cameras.size());
    blob.cameras.vertical_fovs_degrees.reserve(cameras.size());
    blob.cameras.near_clips.reserve(cameras.size());
    blob.cameras.far_clips.reserve(cameras.size());

    for (const auto& c : cameras) {
        blob.cameras.attached_bodies.push_back(c.attached_body);
        blob.cameras.local_transforms.push_back(c.local_transform);
        blob.cameras.vertical_fovs_degrees.push_back(c.vertical_fov_degrees);
        blob.cameras.near_clips.push_back(c.near_clip);
        blob.cameras.far_clips.push_back(c.far_clip);
    }

    const auto& lights = scene.Lights();
    blob.light_count = static_cast<uint32_t>(lights.size());
    blob.lights.types.reserve(lights.size());
    blob.lights.attached_bodies.reserve(lights.size());
    blob.lights.local_transforms.reserve(lights.size());
    blob.lights.colors.reserve(lights.size());
    blob.lights.intensities.reserve(lights.size());

    for (const auto& l : lights) {
        blob.lights.types.push_back(l.type);
        blob.lights.attached_bodies.push_back(l.attached_body);
        blob.lights.local_transforms.push_back(l.local_transform);
        blob.lights.colors.push_back(l.color);
        blob.lights.intensities.push_back(l.intensity);
    }

    const auto& actuators = scene.Actuators();
    blob.actuator_count = static_cast<uint32_t>(actuators.size());
    blob.actuators.types.reserve(actuators.size());
    blob.actuators.joint_ids.reserve(actuators.size());
    blob.actuators.gains.reserve(actuators.size());
    blob.actuators.force_limits.reserve(actuators.size());

    for (const auto& a : actuators) {
        blob.actuators.types.push_back(a.type);
        blob.actuators.joint_ids.push_back(a.joint_id);
        blob.actuators.gains.push_back(a.gain);
        blob.actuators.force_limits.push_back(a.force_limit);
    }

    return blob;
}

} // namespace nuka::scene
