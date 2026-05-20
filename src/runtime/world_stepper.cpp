// ---------------------------------------------------------------------------
// nuka::runtime::world_stepper implementation
// ---------------------------------------------------------------------------

#include "runtime/world_stepper.hpp"

#include "runtime/rigid/integrator.hpp"

#include <algorithm>

namespace nuka::runtime {

namespace {

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

} // namespace

void StepWorldInstance(const WorldTemplate& world_template,
                       WorldInstance& instance,
                       const WorldStepOptions& options) {
    if (options.dt <= 0.0f || options.step_count == 0) {
        return;
    }

    const size_t body_count = std::min<size_t>({
        world_template.body_count,
        instance.body_count,
        instance.poses.size()
    });

    for (uint32_t step = 0; step < options.step_count; ++step) {
        for (size_t body_index = 0; body_index < body_count; ++body_index) {
            rigid::BodyState body = LoadBodyState(world_template, instance, body_index);
            rigid::StepBody(body, options.gravity, options.dt);
            StoreBodyState(body, instance, body_index);
            if (options.clear_forces_after_step) {
                ClearAccumulators(instance, body_index);
            }
        }
    }
}

} // namespace nuka::runtime
