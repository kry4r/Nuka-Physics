#pragma once
// ---------------------------------------------------------------------------
// nuka::scene – Cooked SoA blob for runtime consumption
// ---------------------------------------------------------------------------

#include "scene/canonical_types.hpp"
#include "scene/contact_filter.hpp"   // MergedContactParams (v0.8 C1c)
#include "math/transform.hpp"

#include <algorithm>
#include <utility>
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

// Sentinel: a convex-geometry piece that carries no cooked SDF.
constexpr uint32_t kNoSdf = ~uint32_t(0);

// Cooked sparse narrow-band SDFs (v0.7 p07). One logical SDF per UNIQUE convex
// geometry piece (deduplicated by content hash). All SDFs' cells are stored
// flat and concatenated; SDF `s` owns cells [key_offsets[s], +key_counts[s]).
// Per-SDF cell_keys slice is ASCENDING-sorted so the runtime sampler can binary
// search it (see runtime/sdf/sparse_sdf_query.cuh). Values + gradients are
// parallel to keys. Geometry is in MESH-LOCAL frame (origins are local-space).
//
// `piece_sdf_indices` maps each CookedConvexGeometry piece -> its SDF index
// (kNoSdf if no SDF was cooked for that piece). Two identical pieces share one
// SDF index (dedup), so this vector may contain repeated indices.
struct CookedSdfTable {
    // Per-SDF header (parallel arrays; length == sdf_count).
    std::vector<math::Vec3> origins;       // local-space corner of voxel (0,0,0)
    std::vector<float>      voxel_sizes;    // edge length per SDF
    std::vector<uint32_t>   dims_x;         // full grid dims (most cells empty)
    std::vector<uint32_t>   dims_y;
    std::vector<uint32_t>   dims_z;
    std::vector<uint32_t>   key_offsets;    // start into cell_keys/values/gradients
    std::vector<uint32_t>   key_counts;     // narrow-band cell count for this SDF

    // Flat cell storage (concatenated across all SDFs).
    std::vector<uint64_t>   cell_keys;      // PackSdfCellKey(i,j,k); sorted per SDF
    std::vector<float>      cell_values;    // signed distance per cell
    std::vector<math::Vec3> cell_gradients; // precomputed central-diff gradient

    // Per convex-geometry-piece SDF index (parallel to CookedConvexGeometry
    // pieces). kNoSdf if no SDF for that piece.
    std::vector<uint32_t>   piece_sdf_indices;

    uint32_t Count() const { return static_cast<uint32_t>(voxel_sizes.size()); }
};

// Per-shape contact parameters (v0.8 C1a). EXACTLY one row per cooked shape
// row, parallel to CookedShapeTable (so a mesh that decomposed into N convex
// pieces has N identical contact-param rows, one per piece). `frictions` holds
// the RESOLVED per-shape μ (see cooker.cpp friction-resolution precedence).
// `solimp` is FLATTENED: 5 floats per shape, row-major, so shape `i`'s slice is
// [i*5, i*5+5). `solref` is split into solref0 (timeconst) + solref1 (dampratio).
struct CookedContactParamTable {
    std::vector<uint32_t> contypes;
    std::vector<uint32_t> conaffinities;
    std::vector<int32_t>  groups;
    std::vector<float>    solref0;       // timeconst
    std::vector<float>    solref1;       // dampratio
    std::vector<float>    solimp;        // FLATTENED: 5 floats per shape (row-major)
    std::vector<float>    frictions;     // RESOLVED per-shape μ
    std::vector<uint8_t>  condims;
    std::vector<int32_t>  priorities;
    std::vector<float>    solmix;
    std::vector<float>    margins;
    std::vector<float>    gaps;
};

// --- v0.8 C1c: cook-time filtered-pair policy + system-pair matrix --------
//
// The filter POLICY is the baked output of MuJoCo's collision filtering
// precedence (research doc §8): explicit <pair> overrides > <exclude> >
// parent-child auto-exclude > contype/conaffinity bitmask. C1c bakes the
// first three at cook time; the bitmask test stays a per-pair runtime check
// (PassesContactBitmask, contact_filter.hpp) since it depends only on the
// per-shape contype/conaffinity already in CookedContactParamTable.

// Broad classification of a body's owning simulation subsystem, for the
// forward-looking cross-system enable matrix (consumed by C2). v0.8 has no
// MJCF source that authors this; the matrix defaults all-enabled.
enum class SystemKind : uint8_t {
    Rigid      = 0,
    Articulated = 1,
    Soft       = 2,
    Cloth      = 3,
    Fluid      = 4
};

// Symmetric 5x5 enable matrix over SystemKind pairs. Default: ALL true (every
// system collides with every system). A forward-looking enable seam for C2.
struct SystemPairMatrix {
    bool enabled[5][5];
    SystemPairMatrix() {
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) enabled[i][j] = true;
    }
};

// Set/clear a SystemKind pair symmetrically (both (a,b) and (b,a)).
inline void SetSystemPairEnabled(SystemPairMatrix& m, SystemKind a, SystemKind b,
                                 bool enabled) {
    const auto ia = static_cast<int>(a);
    const auto ib = static_cast<int>(b);
    m.enabled[ia][ib] = enabled;
    m.enabled[ib][ia] = enabled;
}

inline bool IsSystemPairEnabled(const SystemPairMatrix& m, SystemKind a,
                                SystemKind b) {
    return m.enabled[static_cast<int>(a)][static_cast<int>(b)];
}

// An authored <contact><pair> explicit override, baked into the policy. geom1/
// geom2 are in SOURCE-shape space (the ShapeIds the <pair> named), canonicalized
// as (min,max) and sorted. params holds the as-authored override values copied
// straight across (C1b pre-filled MuJoCo defaults for omitted attrs); v0.8 does
// NOT geom-merge omitted <pair> attributes (see BuildFilteredPairPolicy doc).
//
// SOURCE-shape vs COOKED-row space: a <pair> on a mesh that V-HACD decomposed
// into N convex pieces maps to N cooked rows (CookedShapeTable). That source ->
// cooked-row expansion is a C2 consumption concern; this struct stays in source
// space (the space the SceneIR authored), and C2 must bridge it via the cook's
// per-row provenance.
struct CookedExplicitPair {
    ShapeId geom1 = kInvalidShape;
    ShapeId geom2 = kInvalidShape;
    MergedContactParams params;
};

struct CookedFilterPolicy {
    // Union of authored <exclude> body pairs and parent-child auto-excludes,
    // each canonicalized (min,max), deduplicated, and ascending-sorted so the
    // lookup can binary search.
    std::vector<std::pair<BodyId, BodyId>> excluded_body_pairs;
    // Authored <contact><pair> overrides, canonical (min,max) on (geom1,geom2)
    // and ascending-sorted for binary search.
    std::vector<CookedExplicitPair>        explicit_pairs;
    // Cross-system enable matrix; default all-enabled.
    SystemPairMatrix                       system_pairs;
};

// Binary-search the sorted canonical excluded_body_pairs for (a,b) in any order.
inline bool IsBodyPairExcluded(const CookedFilterPolicy& policy, BodyId a,
                               BodyId b) {
    if (b < a) std::swap(a, b);
    const std::pair<BodyId, BodyId> key{a, b};
    return std::binary_search(policy.excluded_body_pairs.begin(),
                              policy.excluded_body_pairs.end(), key);
}

// Binary-search the sorted canonical explicit_pairs for (g1,g2) in any order.
// Returns nullptr if no explicit pair was authored for that geom pair.
inline const CookedExplicitPair* FindExplicitPair(const CookedFilterPolicy& policy,
                                                  ShapeId g1, ShapeId g2) {
    if (g2 < g1) std::swap(g1, g2);
    const auto it = std::lower_bound(
        policy.explicit_pairs.begin(), policy.explicit_pairs.end(), g1,
        [](const CookedExplicitPair& p, ShapeId v) { return p.geom1 < v; });
    for (auto cur = it; cur != policy.explicit_pairs.end() && cur->geom1 == g1;
         ++cur) {
        if (cur->geom2 == g2) return &(*cur);
    }
    return nullptr;
}

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
    CookedContactParamTable contact_params;  // v0.8 C1a: per-shape contact params (parallel to shapes)
    CookedFilterPolicy filter_policy;        // v0.8 C1c: baked filter/exclude/explicit-pair policy
    CookedConvexGeometry convex_geometry;  // v0.7 p06: hull geometry for ConvexHull rows
    CookedSdfTable    sdfs;                 // v0.7 p07: narrow-band SDFs per unique piece
    CookedSensorTable sensors;
    CookedMaterialTable materials;
    CookedCameraTable cameras;
    CookedLightTable lights;
    CookedActuatorTable actuators;
};

} // namespace nuka::scene
