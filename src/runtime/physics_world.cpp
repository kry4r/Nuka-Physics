// ---------------------------------------------------------------------------
// nuka::runtime::PhysicsWorld implementation
// ---------------------------------------------------------------------------

#include "runtime/physics_world.hpp"

#include "core/diagnostics/invariants.hpp"

namespace nuka::runtime {

PhysicsWorld::PhysicsWorld() = default;

PhysicsWorld::~PhysicsWorld() = default;

PhysicsWorld::PhysicsWorld(const PhysicsWorld& other)
    : body_count(other.body_count),
      joint_count(other.joint_count),
      shape_count(other.shape_count),
      actuator_count(other.actuator_count),
      sensor_count(other.sensor_count),
      body_table(other.body_table),
      joint_table(other.joint_table),
      shape_table(other.shape_table),
      actuator_table(other.actuator_table),
      sensor_table(other.sensor_table),
      runtime_world(other.runtime_world),
      invariant_trace_sink_(other.invariant_trace_sink_) {
    if (other.invariant_sampler_) {
        invariant_sampler_ =
            std::make_unique<core::diagnostics::InvariantSampler>(
                other.invariant_sampler_->Config());
    }
}

PhysicsWorld& PhysicsWorld::operator=(const PhysicsWorld& other) {
    if (this == &other) {
        return *this;
    }
    body_count = other.body_count;
    joint_count = other.joint_count;
    shape_count = other.shape_count;
    actuator_count = other.actuator_count;
    sensor_count = other.sensor_count;
    body_table = other.body_table;
    joint_table = other.joint_table;
    shape_table = other.shape_table;
    actuator_table = other.actuator_table;
    sensor_table = other.sensor_table;
    runtime_world = other.runtime_world;
    invariant_trace_sink_ = other.invariant_trace_sink_;
    if (other.invariant_sampler_) {
        invariant_sampler_ =
            std::make_unique<core::diagnostics::InvariantSampler>(
                other.invariant_sampler_->Config());
    } else {
        invariant_sampler_.reset();
    }
    return *this;
}

PhysicsWorld::PhysicsWorld(PhysicsWorld&&) noexcept = default;

PhysicsWorld& PhysicsWorld::operator=(PhysicsWorld&&) noexcept = default;

PhysicsWorld BuildPhysicsWorld(const scene::CookedBlob& blob) {
    PhysicsWorld world;
    world.body_count = blob.body_count;
    world.joint_count = blob.joint_count;
    world.shape_count = blob.shape_count;
    world.actuator_count = blob.actuator_count;
    world.sensor_count = blob.sensor_count;
    world.body_table = blob.bodies;
    world.joint_table = blob.joints;
    world.shape_table = blob.shapes;
    world.actuator_table = blob.actuators;
    world.sensor_table = blob.sensors;
    world.runtime_world = BuildWorld(blob);
    return world;
}

void PhysicsWorld::ConfigureInvariantSampling(
    const core::diagnostics::InvariantConfig& config) {
    invariant_sampler_ = std::make_unique<core::diagnostics::InvariantSampler>(config);
}

void PhysicsWorld::SetInvariantTraceSink(core::diagnostics::TraceSink* sink) {
    invariant_trace_sink_ = sink;
}

core::diagnostics::InvariantSampler* PhysicsWorld::InvariantSampler() {
    return invariant_sampler_.get();
}

const core::diagnostics::InvariantSampler* PhysicsWorld::InvariantSampler() const {
    return invariant_sampler_.get();
}

core::diagnostics::TraceSink* PhysicsWorld::InvariantTraceSink() const {
    return invariant_trace_sink_;
}

} // namespace nuka::runtime
