#pragma once
// ---------------------------------------------------------------------------
// nuka::scene – Cooked SoA blob for runtime consumption
// ---------------------------------------------------------------------------

#include "scene/canonical_types.hpp"
#include "math/transform.hpp"

#include <vector>

namespace nuka::scene {

struct CookedBodyTable {
    std::vector<math::Transform> poses;       // initial pose per body
    std::vector<float>           inv_masses;   // 1/mass, 0 for static
    std::vector<math::Vec3>      inv_inertias; // diagonal inverse inertia
    std::vector<uint8_t>         is_static;    // flags
};

struct CookedJointTable {
    std::vector<JointType>  types;
    std::vector<BodyId>     parent_bodies;
    std::vector<BodyId>     child_bodies;
    std::vector<math::Vec3> axes;
    std::vector<float>      lower_limits;
    std::vector<float>      upper_limits;
};

struct CookedShapeTable {
    std::vector<ShapeType>       types;
    std::vector<BodyId>          body_ids;
    std::vector<math::Transform> local_transforms;
    std::vector<math::Vec3>      half_extents;
    std::vector<float>           radii;
};

struct CookedBlob {
    uint32_t body_count  = 0;
    uint32_t joint_count = 0;
    uint32_t shape_count = 0;

    CookedBodyTable  bodies;
    CookedJointTable joints;
    CookedShapeTable shapes;
};

} // namespace nuka::scene
