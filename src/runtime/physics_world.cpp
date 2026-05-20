// ---------------------------------------------------------------------------
// nuka::runtime::PhysicsWorld implementation
// ---------------------------------------------------------------------------

#include "runtime/physics_world.hpp"

namespace nuka::runtime {

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

} // namespace nuka::runtime
