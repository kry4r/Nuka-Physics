// ---------------------------------------------------------------------------
// nuka::app::BuildDebugVisualization implementation
// ---------------------------------------------------------------------------

#include "apps/debug_shell/debug_visualization.hpp"

#include "collision/aabb.hpp"
#include "scene/canonical_types.hpp"

#include <algorithm>

namespace nuka::app {

namespace {

constexpr uint32_t kCollisionShapeColor = 0x4CC9F0FF;
constexpr uint32_t kShapeAabbColor = 0xF9C74FFF;
constexpr uint32_t kJointAxisColor = 0x90BE6DFF;
constexpr uint32_t kCenterOfMassColor = 0xF8961EFF;
constexpr uint32_t kConstraintErrorColor = 0xF94144FF;

collision::AABB ProxyAabb(const render::DebugProxy& proxy) {
    switch (proxy.shape_type) {
    case scene::ShapeType::Sphere:
        return collision::AABB::FromSphere(proxy.world_transform.position, proxy.radius);
    case scene::ShapeType::Capsule: {
        const math::Vec3 axis = proxy.world_transform.TransformDirection(math::Vec3::UnitZ()).Normalized();
        const math::Vec3 end_a = proxy.world_transform.position - axis * proxy.half_height;
        const math::Vec3 end_b = proxy.world_transform.position + axis * proxy.half_height;
        collision::AABB result = collision::AABB::FromSphere(end_a, proxy.radius);
        result.Merge(collision::AABB::FromSphere(end_b, proxy.radius));
        return result;
    }
    case scene::ShapeType::Box:
    case scene::ShapeType::Plane:
    case scene::ShapeType::ConvexHull:
    case scene::ShapeType::TriMesh:
    case scene::ShapeType::HeightField:
        return collision::AABB::FromBox(proxy.world_transform, proxy.half_extents);
    }

    return collision::AABB::FromBox(proxy.world_transform, proxy.half_extents);
}

math::Vec3 BodyWorldPosition(const scene::SceneGraph& graph, scene::BodyId body_id) {
    if (body_id == scene::kInvalidBody || body_id >= graph.NodeCount()) {
        return math::Vec3::Zero();
    }
    return graph.GetNode(body_id).world_transform.position;
}

void AppendCollisionShapes(const render::RenderScene& render_scene, DebugDrawList& list) {
    for (const auto& proxy : render_scene.debug_proxies) {
        switch (proxy.shape_type) {
        case scene::ShapeType::Sphere:
            list.AddSphere(proxy.world_transform.position, proxy.radius, kCollisionShapeColor);
            break;
        case scene::ShapeType::Capsule:
            list.AddCapsule(proxy.world_transform.position,
                            proxy.world_transform.TransformDirection(math::Vec3::UnitZ()),
                            proxy.radius,
                            proxy.half_height,
                            kCollisionShapeColor);
            break;
        case scene::ShapeType::Box:
        case scene::ShapeType::Plane:
        case scene::ShapeType::ConvexHull:
        case scene::ShapeType::TriMesh:
        case scene::ShapeType::HeightField:
            list.AddBox(proxy.world_transform.position, proxy.half_extents, kCollisionShapeColor);
            break;
        }
    }
}

void AppendShapeAabbs(const render::RenderScene& render_scene, DebugDrawList& list) {
    for (const auto& proxy : render_scene.debug_proxies) {
        list.AddAABB(ProxyAabb(proxy), kShapeAabbColor);
    }
}

void AppendJointAxes(const runtime::PhysicsWorld& physics_world,
                     const scene::SceneGraph& graph,
                     float axis_length,
                     DebugDrawList& list) {
    const auto joint_count = std::min(
        physics_world.joint_table.child_bodies.size(),
        physics_world.joint_table.axes.size());

    for (size_t i = 0; i < joint_count; ++i) {
        const auto child_body = physics_world.joint_table.child_bodies[i];
        if (child_body == scene::kInvalidBody || child_body >= graph.NodeCount()) {
            continue;
        }

        const auto& node = graph.GetNode(child_body);
        const math::Vec3 origin = node.world_transform.position;
        const math::Vec3 axis = node.world_transform.TransformDirection(
            physics_world.joint_table.axes[i].Normalized());
        list.AddLine(origin, origin + axis * axis_length, kJointAxisColor);
    }
}

void AppendCentersOfMass(const runtime::PhysicsWorld& physics_world,
                         const scene::SceneGraph& graph,
                         float radius,
                         DebugDrawList& list) {
    const auto body_count = std::min<size_t>(physics_world.body_count, graph.NodeCount());
    for (size_t i = 0; i < body_count; ++i) {
        list.AddSphere(graph.GetNode(static_cast<scene::BodyId>(i)).world_transform.position,
                       radius,
                       kCenterOfMassColor);
    }
}

void AppendContactPoints(std::span<const constraint::ContactManifold> manifolds,
                         DebugDrawList& list) {
    for (const auto& manifold : manifolds) {
        for (uint32_t i = 0; i < manifold.point_count; ++i) {
            const auto& point = manifold.points[i];
            list.AddContactPoint(point.position, point.normal, point.penetration);
        }
    }
}

void AppendConstraintErrors(const constraint::RowBuffers& rows,
                            const scene::SceneGraph& graph,
                            float error_scale,
                            DebugDrawList& list) {
    for (uint32_t row_index = 0; row_index < rows.RowCount(); ++row_index) {
        const auto body_pair = rows.BodiesForRow(row_index);
        const bool has_a = body_pair.body_a < graph.NodeCount();
        const bool has_b = body_pair.body_b < graph.NodeCount();
        if (!has_a && !has_b) {
            continue;
        }

        const math::Vec3 origin = has_a
            ? BodyWorldPosition(graph, body_pair.body_a)
            : BodyWorldPosition(graph, body_pair.body_b);

        const auto jacobian = rows.JacobianForRowBody(row_index, has_a ? 0u : 1u);
        math::Vec3 direction = jacobian.linear;
        if (direction.LengthSq() < 1e-12f) {
            direction = jacobian.angular;
        }
        if (direction.LengthSq() < 1e-12f) {
            continue;
        }

        const auto& row = rows.rows[row_index];
        const float magnitude = row.rhs != 0.0f ? row.rhs : row.lambda;
        if (magnitude == 0.0f) {
            continue;
        }

        const math::Vec3 endpoint = origin + direction.Normalized() * (magnitude * error_scale);
        list.AddLine(origin, endpoint, kConstraintErrorColor);
    }
}

} // namespace

DebugDrawList BuildDebugVisualization(const DebugVisualizationInput& input,
                                      const DebugVisualizationOptions& options) {
    DebugDrawList list;

    if (input.render_scene != nullptr) {
        if (options.draw_collision_shapes) {
            AppendCollisionShapes(*input.render_scene, list);
        }
        if (options.draw_shape_aabbs) {
            AppendShapeAabbs(*input.render_scene, list);
        }
    }

    if (input.physics_world != nullptr && input.scene_graph != nullptr) {
        if (options.draw_joint_axes) {
            AppendJointAxes(*input.physics_world,
                            *input.scene_graph,
                            options.joint_axis_length,
                            list);
        }
        if (options.draw_centers_of_mass) {
            AppendCentersOfMass(*input.physics_world,
                                *input.scene_graph,
                                options.center_of_mass_radius,
                                list);
        }
    }

    if (options.draw_contact_points) {
        AppendContactPoints(input.contact_manifolds, list);
    }

    if (options.draw_constraint_errors &&
        input.scene_graph != nullptr &&
        input.constraint_rows != nullptr) {
        AppendConstraintErrors(*input.constraint_rows,
                               *input.scene_graph,
                               options.constraint_error_scale,
                               list);
    }

    return list;
}

} // namespace nuka::app
