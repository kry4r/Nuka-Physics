// ---------------------------------------------------------------------------
// nuka::runtime::world_stepper implementation
// ---------------------------------------------------------------------------

#include "runtime/world_stepper.hpp"

#include "collision/aabb.hpp"
#include "collision/dynamic_broadphase.hpp"
#include "constraint/contact_builder.hpp"
#include "runtime/rigid/integrator.hpp"
#include "solver/rigid_solver.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nuka::runtime {

namespace {

struct ShapeProxy {
    uint32_t shape_index = 0;
    uint32_t body_index = 0;
    scene::ShapeType type = scene::ShapeType::Box;
    math::Transform world_transform = math::Transform::Identity();
    math::Vec3 half_extents = {0.5f, 0.5f, 0.5f};
    float radius = 0.5f;
    collision::AABB aabb;
};

void ClearAccumulators(WorldInstance& instance, size_t body_index) {
    if (body_index < instance.forces.size()) {
        instance.forces[body_index] = math::Vec3::Zero();
    }
    if (body_index < instance.torques.size()) {
        instance.torques[body_index] = math::Vec3::Zero();
    }
}

rigid::BodyState LoadBodyState(const WorldTemplate& world_template,
                               const WorldInstance& instance,
                               size_t body_index) {
    rigid::BodyState body;
    body.inv_mass = body_index < world_template.body_table.inv_masses.size()
        ? world_template.body_table.inv_masses[body_index]
        : 0.0f;
    body.inv_inertia = body_index < world_template.body_table.inv_inertias.size()
        ? world_template.body_table.inv_inertias[body_index]
        : math::Vec3::Zero();
    body.position = instance.poses[body_index].position;
    body.orientation = instance.poses[body_index].rotation;
    body.linear_velocity = body_index < instance.linear_velocities.size()
        ? instance.linear_velocities[body_index]
        : math::Vec3::Zero();
    body.angular_velocity = body_index < instance.angular_velocities.size()
        ? instance.angular_velocities[body_index]
        : math::Vec3::Zero();
    body.force = body_index < instance.forces.size()
        ? instance.forces[body_index]
        : math::Vec3::Zero();
    body.torque = body_index < instance.torques.size()
        ? instance.torques[body_index]
        : math::Vec3::Zero();
    return body;
}

void StoreBodyState(const rigid::BodyState& body,
                    WorldInstance& instance,
                    size_t body_index) {
    instance.poses[body_index].position = body.position;
    instance.poses[body_index].rotation = body.orientation;
    if (body_index < instance.linear_velocities.size()) {
        instance.linear_velocities[body_index] = body.linear_velocity;
    }
    if (body_index < instance.angular_velocities.size()) {
        instance.angular_velocities[body_index] = body.angular_velocity;
    }
}

bool IsBodyDynamic(const rigid::BodyState& body) {
    return body.inv_mass > 0.0f;
}

math::Transform BodyPoseOrIdentity(const WorldInstance& instance, uint32_t body_index) {
    if (body_index < instance.poses.size()) {
        return instance.poses[body_index];
    }
    return math::Transform::Identity();
}

collision::AABB ShapeAabb(const ShapeProxy& shape) {
    if (shape.type == scene::ShapeType::Sphere) {
        return collision::AABB::FromSphere(shape.world_transform.position, shape.radius);
    }
    if (shape.type == scene::ShapeType::Plane) {
        collision::AABB result;
        const float plane_y = shape.world_transform.position.y;
        result.min = {-1.0e6f, plane_y - 0.01f, -1.0e6f};
        result.max = { 1.0e6f, plane_y + 0.01f,  1.0e6f};
        return result;
    }

    return collision::AABB::FromBox(shape.world_transform, shape.half_extents);
}

std::vector<ShapeProxy> BuildShapeProxies(const WorldTemplate& world_template,
                                          const WorldInstance& instance) {
    std::vector<ShapeProxy> shapes;
    shapes.reserve(world_template.shape_count);

    for (uint32_t shape_index = 0; shape_index < world_template.shape_count; ++shape_index) {
        if (shape_index >= world_template.shape_table.body_ids.size()) {
            continue;
        }

        const uint32_t body_index = world_template.shape_table.body_ids[shape_index];
        if (body_index >= instance.body_count) {
            continue;
        }

        ShapeProxy shape;
        shape.shape_index = shape_index;
        shape.body_index = body_index;
        if (shape_index < world_template.shape_table.types.size()) {
            shape.type = world_template.shape_table.types[shape_index];
        }
        if (shape_index < world_template.shape_table.local_transforms.size()) {
            shape.world_transform =
                BodyPoseOrIdentity(instance, body_index)
                * world_template.shape_table.local_transforms[shape_index];
        }
        if (shape_index < world_template.shape_table.half_extents.size()) {
            shape.half_extents = world_template.shape_table.half_extents[shape_index];
        }
        if (shape_index < world_template.shape_table.radii.size()) {
            shape.radius = world_template.shape_table.radii[shape_index];
        }
        shape.aabb = ShapeAabb(shape);
        shapes.push_back(shape);
    }

    return shapes;
}

bool IsPlane(const ShapeProxy& shape) {
    return shape.type == scene::ShapeType::Plane;
}

bool IsSphere(const ShapeProxy& shape) {
    return shape.type == scene::ShapeType::Sphere;
}

bool IsBoxLike(const ShapeProxy& shape) {
    return shape.type == scene::ShapeType::Box || shape.type == scene::ShapeType::Capsule;
}

math::Vec3 ClosestPointOnAabb(math::Vec3 point, const collision::AABB& box) {
    return {
        std::clamp(point.x, box.min.x, box.max.x),
        std::clamp(point.y, box.min.y, box.max.y),
        std::clamp(point.z, box.min.z, box.max.z)
    };
}

void AddPointIfPenetrating(const ShapeProxy& dynamic_shape,
                           const ShapeProxy& static_shape,
                           const math::Vec3& position,
                           const math::Vec3& normal_from_static_to_dynamic,
                           float penetration,
                           std::vector<constraint::ContactManifold>& manifolds) {
    if (penetration <= 0.0f) {
        return;
    }

    constraint::ContactPoint point;
    point.position = position;
    point.normal = normal_from_static_to_dynamic.Normalized();
    point.penetration = penetration;

    constraint::ContactManifold manifold;
    manifold.body_a = dynamic_shape.body_index;
    manifold.body_b = static_shape.body_index;
    manifold.AddPoint(point);
    manifolds.push_back(manifold);
}

void GeneratePlaneContact(const ShapeProxy& dynamic_shape,
                          const ShapeProxy& plane_shape,
                          std::vector<constraint::ContactManifold>& manifolds) {
    const float plane_y = plane_shape.world_transform.position.y;
    if (IsSphere(dynamic_shape)) {
        const float bottom = dynamic_shape.world_transform.position.y - dynamic_shape.radius;
        AddPointIfPenetrating(dynamic_shape,
                              plane_shape,
                              {dynamic_shape.world_transform.position.x,
                               plane_y,
                               dynamic_shape.world_transform.position.z},
                              math::Vec3::UnitY(),
                              plane_y - bottom,
                              manifolds);
        return;
    }

    if (IsBoxLike(dynamic_shape)) {
        const float bottom = dynamic_shape.aabb.min.y;
        const math::Vec3 center = dynamic_shape.aabb.Center();
        AddPointIfPenetrating(dynamic_shape,
                              plane_shape,
                              {center.x, plane_y, center.z},
                              math::Vec3::UnitY(),
                              plane_y - bottom,
                              manifolds);
    }
}

void GenerateSphereSphereContact(const ShapeProxy& a,
                                 const ShapeProxy& b,
                                 std::vector<constraint::ContactManifold>& manifolds) {
    const math::Vec3 delta = a.world_transform.position - b.world_transform.position;
    const float distance = delta.Length();
    const float combined_radius = a.radius + b.radius;
    const float penetration = combined_radius - distance;
    if (penetration <= 0.0f) {
        return;
    }

    const math::Vec3 normal = distance > 1.0e-6f ? delta / distance : math::Vec3::UnitY();
    const math::Vec3 point = b.world_transform.position + normal * b.radius;

    constraint::ContactManifold manifold;
    manifold.body_a = a.body_index;
    manifold.body_b = b.body_index;
    manifold.AddPoint({point, normal, penetration});
    manifolds.push_back(manifold);
}

void GenerateSphereBoxContact(const ShapeProxy& sphere,
                              const ShapeProxy& box,
                              std::vector<constraint::ContactManifold>& manifolds) {
    const math::Vec3 closest = ClosestPointOnAabb(sphere.world_transform.position, box.aabb);
    const math::Vec3 delta = sphere.world_transform.position - closest;
    const float distance = delta.Length();
    const float penetration = sphere.radius - distance;
    if (penetration <= 0.0f) {
        return;
    }

    const math::Vec3 normal = distance > 1.0e-6f
        ? delta / distance
        : math::Vec3::UnitY();

    constraint::ContactManifold manifold;
    manifold.body_a = sphere.body_index;
    manifold.body_b = box.body_index;
    manifold.AddPoint({closest, normal, penetration});
    manifolds.push_back(manifold);
}

void GenerateBoxBoxContact(const ShapeProxy& a,
                           const ShapeProxy& b,
                           std::vector<constraint::ContactManifold>& manifolds) {
    const float overlap_x = std::min(a.aabb.max.x, b.aabb.max.x) - std::max(a.aabb.min.x, b.aabb.min.x);
    const float overlap_y = std::min(a.aabb.max.y, b.aabb.max.y) - std::max(a.aabb.min.y, b.aabb.min.y);
    const float overlap_z = std::min(a.aabb.max.z, b.aabb.max.z) - std::max(a.aabb.min.z, b.aabb.min.z);
    if (overlap_x <= 0.0f || overlap_y <= 0.0f || overlap_z <= 0.0f) {
        return;
    }

    math::Vec3 normal = math::Vec3::UnitX();
    float penetration = overlap_x;
    if (overlap_y < penetration) {
        normal = math::Vec3::UnitY();
        penetration = overlap_y;
    }
    if (overlap_z < penetration) {
        normal = math::Vec3::UnitZ();
        penetration = overlap_z;
    }

    const math::Vec3 center_delta = a.aabb.Center() - b.aabb.Center();
    if (center_delta.Dot(normal) < 0.0f) {
        normal = -normal;
    }

    const math::Vec3 point = {
        (std::max(a.aabb.min.x, b.aabb.min.x) + std::min(a.aabb.max.x, b.aabb.max.x)) * 0.5f,
        (std::max(a.aabb.min.y, b.aabb.min.y) + std::min(a.aabb.max.y, b.aabb.max.y)) * 0.5f,
        (std::max(a.aabb.min.z, b.aabb.min.z) + std::min(a.aabb.max.z, b.aabb.max.z)) * 0.5f
    };

    constraint::ContactManifold manifold;
    manifold.body_a = a.body_index;
    manifold.body_b = b.body_index;
    manifold.AddPoint({point, normal, penetration});
    manifolds.push_back(manifold);
}

void GenerateContact(const ShapeProxy& first,
                     const ShapeProxy& second,
                     const std::vector<rigid::BodyState>& bodies,
                     std::vector<constraint::ContactManifold>& manifolds) {
    if (first.body_index == second.body_index) {
        return;
    }

    const bool first_dynamic = first.body_index < bodies.size() && IsBodyDynamic(bodies[first.body_index]);
    const bool second_dynamic = second.body_index < bodies.size() && IsBodyDynamic(bodies[second.body_index]);
    if (!first_dynamic && !second_dynamic) {
        return;
    }

    const ShapeProxy* a = &first;
    const ShapeProxy* b = &second;
    if (!first_dynamic && second_dynamic) {
        a = &second;
        b = &first;
    }

    if (IsPlane(*b)) {
        GeneratePlaneContact(*a, *b, manifolds);
        return;
    }
    if (IsPlane(*a)) {
        GeneratePlaneContact(*b, *a, manifolds);
        return;
    }

    if (IsSphere(*a) && IsSphere(*b)) {
        GenerateSphereSphereContact(*a, *b, manifolds);
        return;
    }
    if (IsSphere(*a) && IsBoxLike(*b)) {
        GenerateSphereBoxContact(*a, *b, manifolds);
        return;
    }
    if (IsBoxLike(*a) && IsSphere(*b)) {
        GenerateSphereBoxContact(*b, *a, manifolds);
        return;
    }
    if (IsBoxLike(*a) && IsBoxLike(*b)) {
        GenerateBoxBoxContact(*a, *b, manifolds);
    }
}

void StoreBodyStates(const std::vector<rigid::BodyState>& bodies,
                     WorldInstance& instance) {
    const size_t body_count = std::min(bodies.size(), instance.poses.size());
    for (size_t body_index = 0; body_index < body_count; ++body_index) {
        StoreBodyState(bodies[body_index], instance, body_index);
    }
}

std::vector<rigid::BodyState> LoadBodyStates(const WorldTemplate& world_template,
                                             const WorldInstance& instance,
                                             size_t body_count) {
    std::vector<rigid::BodyState> bodies;
    bodies.reserve(body_count);
    for (size_t body_index = 0; body_index < body_count; ++body_index) {
        bodies.push_back(LoadBodyState(world_template, instance, body_index));
    }
    return bodies;
}

} // namespace

WorldStepReport StepWorldInstance(const WorldTemplate& world_template,
                                  WorldInstance& instance,
                                  const WorldStepOptions& options) {
    WorldStepReport report;
    if (options.dt <= 0.0f || options.step_count == 0) {
        return report;
    }

    const size_t body_count = std::min<size_t>({
        world_template.body_count,
        instance.body_count,
        instance.poses.size()
    });

    for (uint32_t step = 0; step < options.step_count; ++step) {
        auto bodies = LoadBodyStates(world_template, instance, body_count);

        for (auto& body : bodies) {
            rigid::IntegrateGravity(body, options.gravity, options.dt);
            rigid::IntegrateForces(body, options.dt);
        }

        if (options.enable_contacts && world_template.shape_count > 0u) {
            const auto shapes = BuildShapeProxies(world_template, instance);
            std::vector<collision::AABB> shape_aabbs;
            shape_aabbs.reserve(shapes.size());
            for (const auto& shape : shapes) {
                shape_aabbs.push_back(shape.aabb);
            }

            collision::DynamicBroadphase broadphase;
            broadphase.Update(shape_aabbs);

            std::vector<constraint::ContactManifold> manifolds;
            for (const auto& pair : broadphase.GetPairs()) {
                if (pair.body_a >= shapes.size() || pair.body_b >= shapes.size()) {
                    continue;
                }
                GenerateContact(shapes[pair.body_a], shapes[pair.body_b], bodies, manifolds);
            }

            auto blocks = constraint::BuildContactConstraints(manifolds);
            uint32_t row_count = 0;
            for (auto& block : blocks) {
                row_count += block.row_count;
            }

            for (auto& block : blocks) {
                for (uint32_t row = 0; row < block.normal_row_count; ++row) {
                    block.lower_limit[row] = 0.0f;
                    block.upper_limit[row] = 1.0e6f;
                }
                for (uint32_t row = block.first_friction_row;
                     row < block.first_friction_row + block.friction_row_count
                     && row < constraint::ConstraintBlock::kMaxRows;
                     ++row) {
                    block.lower_limit[row] = 0.0f;
                    block.upper_limit[row] = 0.0f;
                }
            }

            solver::SolverConfig solver_config;
            solver_config.velocity_iterations = options.solver_velocity_iterations;
            solver_config.position_iterations = options.solver_position_iterations;
            solver_config.slop = options.solver_slop;
            solver_config.baumgarte = options.solver_baumgarte;
            const auto solve_result = solver::SolveConstraints(blocks, bodies, solver_config);

            report.broadphase_pair_count += static_cast<uint32_t>(broadphase.GetPairs().size());
            report.contact_manifold_count += static_cast<uint32_t>(manifolds.size());
            report.constraint_block_count += static_cast<uint32_t>(blocks.size());
            report.constraint_row_count += row_count;
            report.solver_iterations_used += solve_result.iterations_used;
            report.max_constraint_error = std::max(report.max_constraint_error,
                                                   solve_result.max_penetration);
            for (const auto& manifold : manifolds) {
                report.contact_point_count += manifold.point_count;
            }
        }

        for (auto& body : bodies) {
            rigid::IntegrateVelocity(body, options.dt);
        }
        StoreBodyStates(bodies, instance);

        for (size_t body_index = 0; body_index < body_count; ++body_index) {
            if (options.clear_forces_after_step) {
                ClearAccumulators(instance, body_index);
            }
        }

        ++report.simulated_step_count;
    }

    return report;
}

} // namespace nuka::runtime
