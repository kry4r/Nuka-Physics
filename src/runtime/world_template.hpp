#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::WorldTemplate – read-only topology from cooked scene
// ---------------------------------------------------------------------------

#include "scene/cooked_blob.hpp"

#include <cstdint>

namespace nuka::runtime {

struct WorldTemplate {
    uint32_t body_count  = 0;
    uint32_t joint_count = 0;
    uint32_t shape_count = 0;
    uint32_t actuator_count = 0;

    // Cooked data references (SoA tables)
    scene::CookedBodyTable  body_table;
    scene::CookedJointTable joint_table;
    scene::CookedShapeTable shape_table;
    scene::CookedActuatorTable actuator_table;
};

} // namespace nuka::runtime
