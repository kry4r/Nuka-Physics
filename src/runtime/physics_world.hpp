#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::PhysicsWorld - physics-facing compiled scene view
// ---------------------------------------------------------------------------

#include "runtime/world_builder.hpp"
#include "scene/cooked_blob.hpp"

#include <cstdint>

namespace nuka::runtime {

struct PhysicsWorld {
    uint32_t body_count = 0;
    uint32_t joint_count = 0;
    uint32_t shape_count = 0;
    uint32_t actuator_count = 0;
    uint32_t sensor_count = 0;

    scene::CookedBodyTable body_table;
    scene::CookedJointTable joint_table;
    scene::CookedShapeTable shape_table;
    scene::CookedActuatorTable actuator_table;
    scene::CookedSensorTable sensor_table;

    BuiltWorld runtime_world;
};

PhysicsWorld BuildPhysicsWorld(const scene::CookedBlob& blob);

} // namespace nuka::runtime
