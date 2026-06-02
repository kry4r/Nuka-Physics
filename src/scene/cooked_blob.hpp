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

// Sentinel: a shape row that carries no convex-hull geometry.
constexpr uint32_t kNoConvexGeometry = ~uint32_t(0);

struct CookedShapeTable {
    std::vector<ShapeType>       types;
    std::vector<BodyId>          body_ids;
    std::vector<MaterialId>      material_ids;
    std::vector<math::Transform> local_transforms;
    std::vector<math::Vec3>      half_extents;
    std::vector<float>           radii;
    std::vector<float>           half_heights;
    // Per-shape index into CookedConvexGeometry (kNoConvexGeometry for shapes
    // without hull geometry, e.g. Sphere/Box/Plane/Capsule). Parallel to the
    // arrays above. v0.7 p06.
    std::vector<uint32_t>        convex_geometry_indices;
};

// Flat geometry storage for ConvexHull shapes (one entry per ConvexHull row).
// Vertices are x,y,z triples; indices are triangles. Each hull's slice is
// [vertex_offsets[i]*3, +vertex_counts[i]*3) into `vertices` and
// [index_offsets[i], +index_counts[i]) into `indices`. v0.7 p06.
struct CookedConvexGeometry {
    std::vector<float>    vertices;        // flat x,y,z triples for all hulls
    std::vector<uint32_t> indices;         // flat triangle indices for all hulls
    std::vector<uint32_t> vertex_offsets;  // per-hull start vertex (in vertices/3)
    std::vector<uint32_t> vertex_counts;   // per-hull vertex count
    std::vector<uint32_t> index_offsets;   // per-hull start index (in indices)
    std::vector<uint32_t> index_counts;    // per-hull index count
    std::vector<float>    volumes;         // per-hull volume
    uint32_t Count() const { return static_cast<uint32_t>(vertex_counts.size()); }
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
    CookedConvexGeometry convex_geometry;  // v0.7 p06: hull geometry for ConvexHull rows
    CookedSensorTable sensors;
    CookedMaterialTable materials;
    CookedCameraTable cameras;
    CookedLightTable lights;
    CookedActuatorTable actuators;
};

} // namespace nuka::scene
