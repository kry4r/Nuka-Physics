#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::WorldTemplate – read-only topology from cooked scene
// ---------------------------------------------------------------------------

#include "scene/cooked_blob.hpp"
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>
#include <vector>

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
    std::vector<articulation::ArticulationCookedTopology> articulations;
};

} // namespace nuka::runtime
