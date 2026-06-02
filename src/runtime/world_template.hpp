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
    // v0.7 p06/p07: ConvexHull hull geometry + per-unique-piece narrow-band SDFs.
    // Threaded through so p08 (Newton-summed-SDF contact) can upload + sample
    // them. The cooked tables hold the byte-reproducible golden SDF cells; an
    // upload step (later) builds the device-side SparseSdfDevice views.
    scene::CookedConvexGeometry convex_geometry;
    scene::CookedSdfTable   sdf_table;
    std::vector<articulation::ArticulationCookedTopology> articulations;
};

} // namespace nuka::runtime
