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
    std::vector<math::Transform> local_poses; // body frame relative to parent
    std::vector<math::Transform> inertial_frames; // COM/principal inertia frame in body
    std::vector<float>           masses;       // mass, 0 for static
    std::vector<math::Vec3>      inertias;     // diagonal inertia in body frame
    std::vector<float>           inv_masses;   // 1/mass, 0 for static
    std::vector<math::Vec3>      inv_inertias; // diagonal inverse inertia
    std::vector<uint8_t>         is_static;    // flags
};

struct CookedJointTable {
    std::vector<JointType>  types;
    std::vector<BodyId>     parent_bodies;
    std::vector<BodyId>     child_bodies;
    std::vector<math::Vec3> axes;
    std::vector<math::Transform> parent_frames;
    std::vector<math::Transform> child_frames;
    std::vector<float>      lower_limits;
    std::vector<float>      upper_limits;
    std::vector<float>      dampings;
    std::vector<float>      armatures;
    std::vector<float>      initial_positions;
};

struct CookedShapeTable {
    std::vector<ShapeType>       types;
    std::vector<BodyId>          body_ids;
    std::vector<MaterialId>      material_ids;
    std::vector<math::Transform> local_transforms;
    std::vector<math::Vec3>      half_extents;
    std::vector<float>           radii;
    std::vector<float>           half_heights;
};

struct CookedSensorTable {
    std::vector<SensorType>      types;
    std::vector<BodyId>          attached_bodies;
    std::vector<math::Transform> local_transforms;
    std::vector<float>           sample_rates_hz;
};

struct CookedMaterialTable {
    std::vector<math::Vec3> base_colors;
    std::vector<float>      alphas;
    std::vector<float>      roughnesses;
    std::vector<float>      metallics;
};

struct CookedCameraTable {
    std::vector<BodyId>          attached_bodies;
    std::vector<math::Transform> local_transforms;
    std::vector<float>           vertical_fovs_degrees;
    std::vector<float>           near_clips;
    std::vector<float>           far_clips;
};

struct CookedLightTable {
    std::vector<LightType>       types;
    std::vector<BodyId>          attached_bodies;
    std::vector<math::Transform> local_transforms;
    std::vector<math::Vec3>      colors;
    std::vector<float>           intensities;
};

struct CookedActuatorTable {
    std::vector<ActuatorType> types;
    std::vector<JointId>      joint_ids;
    std::vector<float>        gains;
    std::vector<float>        force_limits;
};

struct CookedBlob {
    uint32_t body_count  = 0;
    uint32_t joint_count = 0;
    uint32_t shape_count = 0;
    uint32_t sensor_count = 0;
    uint32_t material_count = 0;
    uint32_t camera_count = 0;
    uint32_t light_count = 0;
    uint32_t actuator_count = 0;

    CookedBodyTable  bodies;
    CookedJointTable joints;
    CookedShapeTable shapes;
    CookedSensorTable sensors;
    CookedMaterialTable materials;
    CookedCameraTable cameras;
    CookedLightTable lights;
    CookedActuatorTable actuators;
};

} // namespace nuka::scene
