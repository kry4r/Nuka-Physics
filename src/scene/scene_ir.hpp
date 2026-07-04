#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::SceneIR – Canonical intermediate representation for scenes
//
// M2b: SceneIR is now a FACADE with a dual representation. The record vectors
// below REMAIN the storage and the legacy read API (zero churn for the cooker /
// render / compose / oracle paths). In addition, every Add* mutator
// write-through-builds the new structural world: ECS entities + components
// (scene/ecs) hung on a scene TREE (scene/graph). The tree + Registry are the
// structural authority that .nks Save/Load (M2c) consumes; the records keep
// full legacy fidelity (contact metadata, inline mesh geometry, decompose mode
// stay record-only) while the components carry the five-component projection.
//
// Copy semantics: SceneIR is copyable. Because the tree is a shared_ptr node
// graph (a shallow copy would alias nodes across instances and leave the copy's
// Registry backrefs pointing into the source tree), the copy ctor / assignment
// REBUILD tree_ + ecs_ from the record vectors via the same write-through path
// Add* uses (see scene_ir.cpp). This deep-copy is exactly what Compose() needs
// for purity (it returns by value and callers assert the base is unchanged).
// ---------------------------------------------------------------------------

#include "scene/asset/asset_ref.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cook/settle_spec.hpp"   // SettleSpec (plain data; persisted IC seam)
#include "scene/ecs/entity.hpp"
#include "scene/ecs/registry.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_metadata.hpp"     // SceneInitialState
#include "math/transform.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nuka::scene {

// ---------------------------------------------------------------------------
// Record types
// ---------------------------------------------------------------------------

struct RigidBodyRecord {
    std::string name;
    BodyId id                              = kInvalidBody;
    BodyId parent_id                       = kInvalidBody;
    math::Transform local_transform        = math::Transform::Identity();
    math::Transform inertial_transform     = math::Transform::Identity();
    float mass                             = 1.0f;
    math::Vec3 inertia                     = {1.0f, 1.0f, 1.0f};
    bool is_static                         = false;
};

struct CollisionShapeRecord {
    ShapeId id                             = 0;
    BodyId body_id                         = kInvalidBody;
    MaterialId material_id                 = kInvalidMaterial;
    std::string name;
    ShapeType type                         = ShapeType::Box;
    math::Transform local_transform        = math::Transform::Identity();
    math::Vec3 half_extents                = {0.5f, 0.5f, 0.5f};
    float radius                           = 0.5f;
    float half_height                      = 0.5f;

    // -- Convex decomposition (mesh shapes; v0.7 p06) -----------------------
    // Authored intent (nuka:decompose) + the upper piece bound (nuka:decompose:
    // max_pieces). For ConvexHull / TriMesh shapes, mesh_vertices / mesh_indices
    // carry the source geometry; when present and the mode warrants it, the
    // cooker runs V-HACD and emits one ConvexHull per piece. Empty geometry =>
    // the shape is passed through unchanged (importers do not yet load mesh
    // files, so geometry is normally supplied programmatically / by future
    // mesh-file loading).
    DecomposeMode decompose_mode           = DecomposeMode::Auto;
    uint32_t      decompose_max_pieces     = 32;
    std::vector<float>    mesh_vertices;   // x,y,z triples (source mesh)
    std::vector<uint32_t> mesh_indices;    // triangle indices (source mesh)
    // Authored per-vertex normals (x,y,z triples, 1:1 with mesh_vertices in
    // declaration order; empty => none). Threaded to the .nka MESH normals
    // stream by SaveShape; empty keeps the cooked .nka byte-identical.
    std::vector<float>    mesh_normals;

    // -- Visual-mesh asset reference (M8.5 T5: the visual-mesh cook) ---------
    // For a VISUAL-only geom (contype==0 && conaffinity==0) carrying triangle
    // geometry, .nks Save routes the triangles to a .nka MESH chunk (vs the CMSH
    // chunk a colliding mesh uses) and stores the resolved AssetRef TEXT here at
    // Load time ("<nka-path>#MESH/<idx>", with nka_path the full sibling path).
    // ProjectShape parses it into VisualMeshComponent.mesh so the render consumer
    // (render_world.cpp) decodes real triangles instead of a placeholder box.
    // NOT serialized as a record field (the on-disk authority is the JSON "mesh"
    // ref on the visual_mesh node); empty for collision shapes and for visual
    // geoms with no triangles.
    std::string           visual_mesh_ref;

    // -- Contact metadata (v0.8 C1a) ----------------------------------------
    // Per-SHAPE contact parameters, matching MuJoCo's per-geom semantics
    // (per-shape strictly generalizes per-material and gives a material-less
    // collision geom somewhere to store its params). Defaults are MuJoCo's.
    // The cooker copies these verbatim into CookedContactParamTable, EXCEPT
    // friction_mu, which is RESOLVED at cook time (see cooker.cpp precedence).
    uint32_t contype          = 1;
    uint32_t conaffinity      = 1;
    int32_t  collision_group  = 0;
    float    solref[2]        = {0.02f, 1.0f};                     // timeconst, dampratio
    float    solimp[5]        = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f}; // dmin,dmax,width,mid,power
    float    friction_mu      = -1.0f;   // SENTINEL: <0 => inherit material μ; >=0 => explicit per-shape override
    int32_t  priority         = 0;
    float    solmix           = 1.0f;
    float    margin           = 0.0f;
    float    gap              = 0.0f;
    uint8_t  condim           = 3;       // {1,3,4,6}; v0.8 uses 1 & 3
};

struct JointRecord {
    std::string name;
    JointId id                             = kInvalidJoint;
    BodyId parent_body                     = kInvalidBody;
    BodyId child_body                      = kInvalidBody;
    JointType type                         = JointType::Revolute;
    math::Vec3 axis                        = {0.0f, 0.0f, 1.0f};
    math::Transform parent_frame           = math::Transform::Identity();
    math::Transform child_frame            = math::Transform::Identity();
    float lower_limit                      = -3.14159f;
    float upper_limit                      =  3.14159f;
    float damping                          = 0.0f;
    float armature                         = 0.0f;
    float stiffness                        = 0.0f;
    float initial_position                 = 0.0f;
    float frictionloss                     = 0.0f;   // MuJoCo joint dry friction
};

// Render-camera intrinsics carried inside a Camera/Depth SensorDesc. Width/height
// are 0 until authored; vFOV is the physCamera-derived vertical field of view.
struct CameraIntrinsics {
    uint16_t width                         = 0;
    uint16_t height                        = 0;
    float vfov_degrees                     = 45.0f;
    float near_clip                        = 0.01f;
    float far_clip                         = 1000.0f;
    uint8_t distortion                     = 0;
    float k1                               = 0.0f;
    float k2                               = 0.0f;
};

// Ray-fan pattern for a Lidar/RangeScan SensorDesc (1-ray rangefinder uses
// az_count=el_count=1); dir_table_off points at an external direction table.
struct LidarPattern {
    uint16_t az_count                      = 0;
    uint16_t el_count                      = 0;
    float az_min                           = 0.0f;
    float az_max                           = 0.0f;
    float el_min                           = 0.0f;
    float el_max                           = 0.0f;
    float min_range                        = 0.0f;
    float max_range                        = 100.0f;
    uint32_t dir_table_off                 = ~0u;
};

// Unified sensor descriptor: ONE IR for analytic + render sensors. mount/
// mount_index name the FK pose for local_offset; cam/lidar are render payloads.
struct SensorDesc {
    std::string name;
    SensorId id                            = 0;
    SensorType type                        = SensorType::Imu;
    MountFrame mount                       = MountFrame::Link;
    uint32_t mount_index                   = kInvalidBody;
    math::Transform local_offset           = math::Transform::Identity();
    CameraIntrinsics cam;
    LidarPattern lidar;
    uint32_t aov_mask                      = 0;
    uint32_t update_period                 = 1;
    float sample_rate_hz                   = 0.0f;
};

struct MaterialRecord {
    std::string name;
    MaterialId id                          = kInvalidMaterial;
    math::Vec3 base_color                  = {1.0f, 1.0f, 1.0f};
    float alpha                            = 1.0f;
    float roughness                        = 0.5f;
    float metallic                         = 0.0f;

    // Render-only beauty fields (1:1 with RenderMaterial). emissive lights both
    // tiers; sheen/transmission/ior/absorption are consumed by the offline beauty
    // path. Defaults are strict no-ops, so a material lacking these keys on disk
    // reconstructs the identical RenderMaterial (no cook / render change).
    math::Vec3 emissive                    = {0.0f, 0.0f, 0.0f};
    float sheen                            = 0.0f;
    float transmission                     = 0.0f;
    float ior                              = 1.0f;
    math::Vec3 absorption                  = {0.0f, 0.0f, 0.0f};

    // Per-material default friction coefficient μ (owner Q8 "per-material μ").
    // Resolved into the per-shape cooked μ at cook time when a shape does not
    // carry an explicit per-shape override (see cooker.cpp precedence). (v0.8 C1a)
    float friction_mu                      = 1.0f;

    // Image-file PBR maps (paths, resolved against the .nks dir at load). Empty
    // (default) reconstructs the identical untextured RenderMaterial. uv_scale
    // tiles the maps; triplanar projects them in body-local space for geometry
    // without authored UVs (boxes / terrain / particle surfaces).
    std::string albedo_map;
    std::string roughness_map;
    std::string normal_map;
    float uv_scale                         = 1.0f;
    bool  triplanar                        = true;
};

// Scene-level HDR environment backing the offline beauty render (equirect
// background + sky-dome light). `hdri` empty (default) keeps the procedural
// studio sky. `use_scene_materials` renders instances with their AUTHORED
// materials instead of the studio hero palette override (default false keeps
// every existing scene's beauty frames unchanged). Authored metadata like
// initial_state/terrain: never cooked.
struct EnvironmentRecord {
    std::string hdri;                      // path, resolved against the .nks dir
    float yaw_deg                          = 0.0f;
    float intensity                        = 1.0f;
    bool  use_scene_materials              = false;

    // Offline beauty look levers (opt-in; neutral defaults keep existing beauty
    // frames byte-identical). render_beauty maps each onto the studio tracer.
    bool  ibl_full_fill                    = false;  // env-miss fill at `intensity`, not the dim sky
    float exposure_ev                      = 0.0f;   // post exposure in stops
    float grade                            = 0.0f;   // post filmic contrast/saturation strength
    float sun_disc                         = 0.0f;   // sky sun-disc radiance scale, keyed to the key light
};

struct CameraRecord {
    std::string name;
    CameraId id                            = 0;
    BodyId attached_body                   = kInvalidBody;
    math::Transform local_transform        = math::Transform::Identity();
    float vertical_fov_degrees             = 45.0f;
    float near_clip                        = 0.01f;
    float far_clip                         = 1000.0f;
};

struct LightRecord {
    std::string name;
    LightId id                             = 0;
    LightType type                         = LightType::Point;
    BodyId attached_body                   = kInvalidBody;
    math::Transform local_transform        = math::Transform::Identity();
    math::Vec3 color                       = {1.0f, 1.0f, 1.0f};
    float intensity                        = 1.0f;
};

struct ActuatorRecord {
    std::string name;
    ActuatorId id                          = 0;
    JointId joint_id                       = kInvalidJoint;
    ActuatorType type                      = ActuatorType::Motor;
    float gain                             = 1.0f;
    float force_limit                      = 0.0f;
};

// An authored <contact><pair> explicit contact override (v0.8 C1b). MuJoCo's
// <pair> always carries a concrete parameter set (an absent attribute uses the
// MuJoCo default, NOT the per-shape -1 friction sentinel), so the fields below
// are initialized to MuJoCo's pair defaults. friction_mu holds only the FIRST
// (isotropic tangential) friction component; spin/roll friction is dropped (see
// the importer comment). geom1/geom2 are the resolved ShapeIds of the two named
// geoms. The filter/merge POLICY that consumes these overrides is C1c; C1b only
// stores them.
struct ContactPairOverride {
    ShapeId geom1       = kInvalidShape;
    ShapeId geom2       = kInvalidShape;
    uint8_t condim      = 3;
    float   friction_mu = 1.0f;                          // <pair friction>[0]
    float   solref[2]   = {0.02f, 1.0f};
    float   solimp[5]   = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    float   margin      = 0.0f;
    float   gap         = 0.0f;
};

// ---------------------------------------------------------------------------
// Media records (cloth / soft tet / fluid). Each field maps 1:1 to an existing
// cook input; one geometry/material variant is used per MediaRecord::Kind.
// ---------------------------------------------------------------------------

// Constitutive material for an XPBD medium (cloth uses distance+bend, tet uses
// distance+volume — one block). Fields map 1:1 to cook::XpbdCookInput.
struct MediaXpbdMaterial {
    float    particle_mass     = 0.0f;
    float    friction          = 0.6f;   // body<->soft contact mu
    float    distance_alpha    = 0.0f;   // distance-constraint compliance
    float    bend_alpha        = 0.0f;   // cloth bend compliance
    float    volume_alpha      = 0.0f;   // tet volume compliance
    uint16_t iters             = 1u;     // solver iterations
    float    aero_drag_normal  = 0.0f;
    float    aero_drag_tangent = 0.0f;
    float    aero_drag_max_dv  = 0.0f;
};

// Constitutive material for a PBF fluid. rest_density / support_scale / iters /
// friction map to cook::PbfCookInput; the wall fields carry box confinement.
struct MediaPbfMaterial {
    float    rest_density      = 0.0f;
    float    support_scale     = 0.0f;   // support_radius = support_scale * spacing
    uint16_t iters             = 4u;
    float    friction          = 0.0f;
    bool     clamp_overdensity = true;
    // Box confinement; the cook generates a pinned boundary slab from boundary_layers.
    bool       walls_enabled   = false;
    math::Vec3 walls_min{0.0f, 0.0f, 0.0f};
    math::Vec3 walls_max{0.0f, 0.0f, 0.0f};
    float      floor_z         = 0.0f;
    uint32_t   boundary_layers = 0u;
};

// Constitutive material for an MLS-MPM medium. Mirrors cook::MpmMaterialInput
// plus the cook::MpmCookInput grid/floor scalars.
struct MediaMpmMaterial {
    float youngs       = 0.0f;
    float poisson      = 0.0f;
    float density      = 0.0f;
    float dp_friction  = 0.0f;   // granular internal friction angle (deg).
    float dp_cohesion  = 0.0f;   // granular cohesion stress (Pa; 0 = cohesionless).
    float model_kind   = 0.0f;   // 0 corotated, 2 neo-hookean, 3 fluid, 4 granular DP
    float bulk_modulus = 0.0f;
    float tait_gamma   = 0.0f;
    float viscosity    = 0.0f;
    float      dx             = 0.0f;   // uniform grid node spacing
    uint32_t   substeps       = 1u;
    math::Vec3 floor_normal{0.0f, 0.0f, 1.0f};
    float      floor_d        = 0.0f;
    float      floor_friction = 0.4f;
};

// Render-skin metadata for surface baking. The cloth fields mirror the particle
// SurfaceTopology (inflation + Laplacian relax); skin_mesh refs a baked tet skin.
struct MediaRenderSkin {
    float    normal_offset = 0.0f;   // outward inflation along the smooth normal
    uint32_t smooth_iters  = 0u;     // Laplacian relaxation passes (render-only)
    float    smooth_lambda = 0.5f;   // per-pass blend weight in [0,1]
    AssetRef skin_mesh;              // embedded tet skin chunk; empty => particle surface
};

// A first-class media entry: WHAT it is (kind), HOW it solves (method),
// procedural-or-baked geometry, a constitutive material, and a render skin.
struct MediaRecord {
    std::string name;
    MediaId     id = kInvalidMedia;

    // WHAT it is (solver routing) and HOW it solves (one method per medium).
    // Granular is an MLS-MPM Drucker-Prager sand/gravel bed (box geometry, model_kind 4).
    enum class Kind   : uint8_t { Cloth, SoftTet, Fluid, Granular };
    enum class Method : uint8_t { Xpbd, Pbf, MlsMpm };
    Kind   kind   = Kind::Cloth;
    Method method = Method::Xpbd;

    // Procedural geometry (one populated per kind); a baked .obj/.nka mesh
    // medium references its chunk via `baked` instead.
    // Which node set is pinned (inv_mass 0): none (a free drape), the perimeter
    // (a taut membrane), or ONE grid edge (a hung curtain / flag).
    enum class ClothPin : uint8_t { FromFree = 0, None, Perimeter,
                                    EdgeX0, EdgeX1, EdgeY0, EdgeY1 };
    struct ClothGrid {
        uint32_t   nx = 0u, ny = 0u;
        float      spacing = 0.0f;
        math::Vec3 origin{0.0f, 0.0f, 0.0f};
        bool       free = false;
        // FromFree defers to the legacy `free` flag; any other value overrides it.
        ClothPin   pin = ClothPin::FromFree;
    };
    struct TetSphere {
        math::Vec3 center{0.0f, 0.0f, 0.0f};
        float      radius = 0.0f;
        uint32_t   cells = 0u;
        float      cell_len = 0.0f;
    };
    // A box of particles sampled on a uniform lattice; the geometry for a Fluid
    // (PBF/MPM) and a Granular (MPM sand/gravel) medium alike.
    struct FluidBox {
        math::Vec3 min{0.0f, 0.0f, 0.0f};
        math::Vec3 max{0.0f, 0.0f, 0.0f};
        float      spacing = 0.0f;
    };
    ClothGrid cloth_grid{};
    TetSphere tet_sphere{};
    FluidBox  fluid_box{};
    AssetRef  baked;            // .nka tet/cloth chunk; empty => procedural.

    MediaXpbdMaterial xpbd{};
    MediaPbfMaterial  pbf{};
    MediaMpmMaterial  mpm{};

    MediaRenderSkin render_skin{};
    uint32_t        render_material_id = kInvalidMaterial;
};

// A first-class terrain field: the parametric elevation grid the cook bakes into a
// static heightfield collidable (the ONE terrain::HeightField source). Fields mirror
// terrain::TerrainGenConfig 1:1 -- the engine has NO terrain-type enum: a flat field
// is every amplitude 0, a staircase is ring_rise>0, the composite rolling field is
// feature_cell>0. base_z is the absolute ground floor. An optional grayscale image
// source fills the grid instead when image_path is set.
struct TerrainRecord {
    std::string name;
    uint32_t   nrow = 0u, ncol = 0u;       // grid rows (+Y) / cols (+X).
    float      cell = 0.0f;                 // square cell spacing (cell_x == cell_y).
    math::Vec3 origin{0.0f, 0.0f, 0.0f};    // world corner cell (0,0).
    float      base_z = 0.0f;               // absolute z of the ground floor.

    // Parametric feature amplitudes (each independent; 0 == absent).
    float    grade_x = 0.0f, grade_y = 0.0f;
    float    ring_rise = 0.0f, ring_width = 0.0f, ring_platform = 0.0f;
    uint32_t ring_count = 0u;
    float    bump_height = 0.0f, bump_cell = 0.0f;
    float    feature_cell = 0.0f, feature_margin = 0.0f;
    uint32_t feature_seed = 0u;
    uint32_t curric_levels = 0u, curric_types = 0u;

    // Optional grayscale height-map source. Non-empty => the image fills the grid
    // (placed by radius/elevation) and the parametric amplitudes above are ignored.
    std::string image_path;
    float       image_radius_x = 0.0f, image_radius_y = 0.0f;
    float       image_elevation_z = 0.0f;
};

// A first-class scripting node: inline source the editor's ScriptHost execs +
// dispatches (on_ready/on_step/on_reset), a stable id that survives a reload, and
// an explicit parent tree-path to hang the /script node under ("" => root). Persisted
// by .nks; ignored by the cook (scripts never reach the physics model).
struct ScriptRecord {
    std::string name = "script";
    ScriptId    id   = kInvalidScript;
    std::string source;
    uint64_t    stable_id   = 0;
    std::string parent_path;
};

// ---------------------------------------------------------------------------
// SceneIR
// ---------------------------------------------------------------------------

class SceneIR {
public:
    // -- lifecycle ----------------------------------------------------------
    // Default-constructible; copy DEEP-rebuilds the facade (tree_ + ecs_) from
    // the records; move transfers everything. (See the file header + .cpp.)
    SceneIR();
    SceneIR(const SceneIR& other);
    SceneIR& operator=(const SceneIR& other);
    SceneIR(SceneIR&&) noexcept = default;
    SceneIR& operator=(SceneIR&&) noexcept = default;
    ~SceneIR() = default;

    // -- mutators -----------------------------------------------------------
    BodyId   AddRigidBody(std::string name);
    BodyId   AddRigidBody(RigidBodyRecord record);

    ShapeId  AddCollisionShape(BodyId body, ShapeType type);
    ShapeId  AddCollisionShape(CollisionShapeRecord record);

    JointId  AddJoint(std::string name, BodyId parent, BodyId child);
    JointId  AddJoint(JointRecord record);

    SensorId AddSensor(std::string name, BodyId body);
    SensorId AddSensor(SensorDesc record);
    MaterialId AddMaterial(MaterialRecord record);
    CameraId AddCamera(CameraRecord record);
    LightId AddLight(LightRecord record);
    ActuatorId AddActuator(ActuatorRecord record);
    MediaId  AddMedia(MediaRecord record);
    ScriptId AddScript(ScriptRecord record);

    // Authored terrain field (scene-global config, like gravity; NOT a tree node).
    // Persisted by .nks, consumed by the cook (CookToModel bakes it into the static
    // heightfield collidable). Copied verbatim by the copy ctor / assignment.
    TerrainId AddTerrain(TerrainRecord record);

    // Record an explicit collision-exclusion between two bodies. Stored
    // canonicalized as (min,max) so call order does not matter. The filter
    // POLICY that consumes this list (and the <contact><pair> explicit-override
    // list) is C1b/C1c; C1a only stores the authored excludes. (v0.8 C1a)
    void AddExcludePair(BodyId a, BodyId b);

    // Record an authored <contact><pair> explicit contact override (v0.8 C1b).
    // Stored verbatim in authoring order; the merge/policy that consumes these
    // is C1c.
    void AddContactPair(ContactPairOverride pair);

    // Remove the body subtree at `root` + every record referencing it; survivors
    // re-densify with all cross-references remapped. False when `root` is invalid.
    bool RemoveBodySubtree(BodyId root);

    // Remove one script record (ids re-densify, facade re-projects). False when
    // `id` is invalid.
    bool RemoveScript(ScriptId id);

    // -- counts -------------------------------------------------------------
    size_t RigidBodyCount() const;
    size_t JointCount()     const;
    size_t ShapeCount()     const;
    size_t SensorCount()    const;
    size_t MaterialCount()  const;
    size_t CameraCount()    const;
    size_t LightCount()     const;
    size_t ActuatorCount()  const;
    size_t MediaCount()     const;
    size_t TerrainCount()   const;
    size_t ScriptCount()    const;

    // -- accessors ----------------------------------------------------------
    const RigidBodyRecord&      GetBody(BodyId id)    const;
    const JointRecord&          GetJoint(JointId id)  const;
    const CollisionShapeRecord& GetShape(ShapeId id)  const;
    const SensorDesc&           GetSensor(SensorId id) const;
    const MaterialRecord&       GetMaterial(MaterialId id) const;
    const CameraRecord&         GetCamera(CameraId id) const;
    const LightRecord&          GetLight(LightId id) const;
    const ActuatorRecord&       GetActuator(ActuatorId id) const;
    const MediaRecord&          GetMedia(MediaId id) const;
    const TerrainRecord&        GetTerrain(TerrainId id) const;
    const ScriptRecord&         GetScript(ScriptId id) const;

    // Mutable record access. Mutating a record through these BYPASSES the Add*
    // write-through, so each Get*Mut marks the facade DIRTY; the next facade
    // read (Tree / Ecs / EntityOf*) lazily re-projects tree_ + ecs_ from the
    // records. NOTE: the re-projection creates fresh entities/nodes — EntityIds
    // and SceneNode pointers obtained BEFORE a Get*Mut are invalidated by it.
    RigidBodyRecord&      GetBodyMut(BodyId id);
    JointRecord&          GetJointMut(JointId id);
    CollisionShapeRecord& GetShapeMut(ShapeId id);
    SensorDesc&           GetSensorMut(SensorId id);
    MaterialRecord&       GetMaterialMut(MaterialId id);
    CameraRecord&         GetCameraMut(CameraId id);
    LightRecord&          GetLightMut(LightId id);
    ActuatorRecord&       GetActuatorMut(ActuatorId id);
    MediaRecord&          GetMediaMut(MediaId id);
    TerrainRecord&        GetTerrainMut(TerrainId id);
    ScriptRecord&         GetScriptMut(ScriptId id);

    // -- bulk accessors -----------------------------------------------------
    const std::vector<RigidBodyRecord>&      Bodies()  const;
    const std::vector<JointRecord>&          Joints()  const;
    const std::vector<CollisionShapeRecord>& Shapes()  const;
    const std::vector<SensorDesc>&           Sensors() const;
    const std::vector<MaterialRecord>&       Materials() const;
    const std::vector<CameraRecord>&         Cameras() const;
    const std::vector<LightRecord>&          Lights() const;
    const std::vector<ActuatorRecord>&       Actuators() const;
    const std::vector<MediaRecord>&          Media() const;
    const std::vector<TerrainRecord>&        Terrain() const;
    const std::vector<ScriptRecord>&         Scripts() const;
    const std::vector<std::pair<BodyId, BodyId>>& ExcludePairs() const;
    const std::vector<ContactPairOverride>&       ContactPairs() const;

    // -- authored scene METADATA (M7: not cook-fidelity records) -------------
    // The per-articulation settled IC (initial_state, R4-baked) and the
    // deterministic-settle directive (settle). Persisted by .nks Save/Load,
    // consumed by the cook (IC seed). NOT part of the facade (the tree /
    // Registry / cook are unaffected): plain authoring data copied verbatim by
    // the copy ctor / assignment. Mutable access does NOT mark the facade dirty.
    const SceneInitialState&     InitialState() const { return initial_state_; }
    SceneInitialState&           InitialStateMut() { return initial_state_; }
    const cook::SettleSpec&      Settle() const { return settle_; }
    cook::SettleSpec&            SettleMut() { return settle_; }
    const EnvironmentRecord&     Environment() const { return environment_; }
    EnvironmentRecord&           EnvironmentMut() { return environment_; }

    // -- facade: structural world (M2b) -------------------------------------
    // The scene TREE and the ECS Registry built by the Add* write-through. The
    // tree is the hierarchy authority; the Registry holds the five-component
    // projection. Records are authoritative: Add* write-through keeps the
    // facade current, and record mutation via Get*Mut marks it dirty so these
    // readers lazily re-project before returning. There is deliberately NO
    // mutable facade access (no TreeMut/EcsMut): mutating the derived
    // representation directly would silently diverge from the records.
    const SceneGraph& Tree() const { EnsureFacade(); return tree_; }
    const Registry&   Ecs()  const { EnsureFacade(); return ecs_; }

    // record-id -> projected entity (kInvalidEntity if the record produced no
    // entity, which cannot happen for bodies/shapes/joints today).
    EntityId EntityOfBody(BodyId id)   const;
    EntityId EntityOfShape(ShapeId id) const;
    EntityId EntityOfJoint(JointId id) const;
    EntityId EntityOfMedia(MediaId id) const;
    EntityId EntityOfScript(ScriptId id) const;

private:
    // Rebuild tree_ + ecs_ from the current record vectors (used by the copy
    // ctor / assignment and by the move-from default; see scene_ir.cpp).
    void RebuildFacade();

    // Lazy facade resync: re-projects when a Get*Mut marked the facade dirty.
    void EnsureFacade() const;

    // -- per-record write-through helpers (build the tree + ECS for one
    //    just-appended record). Defined in scene_ir.cpp. -------------------
    void ProjectBody(const RigidBodyRecord& rec);
    void ProjectShape(const CollisionShapeRecord& rec);
    void ProjectJoint(const JointRecord& rec);
    void ProjectMaterial(const MaterialRecord& rec);
    void ProjectCamera(const CameraRecord& rec);
    void ProjectLight(const LightRecord& rec);
    void ProjectActuator(const ActuatorRecord& rec);
    void ProjectMedia(const MediaRecord& rec);
    void ProjectScript(const ScriptRecord& rec);

    std::vector<RigidBodyRecord>      bodies_;
    std::vector<JointRecord>          joints_;
    std::vector<CollisionShapeRecord> shapes_;
    std::vector<SensorDesc>           sensors_;
    std::vector<MaterialRecord>       materials_;
    std::vector<CameraRecord>         cameras_;
    std::vector<LightRecord>          lights_;
    std::vector<ActuatorRecord>       actuators_;
    std::vector<MediaRecord>          media_;
    std::vector<ScriptRecord>         scripts_;
    std::vector<std::pair<BodyId, BodyId>> exclude_pairs_;
    std::vector<ContactPairOverride>       contact_pairs_;

    // -- authored metadata (M7; not facade-derived, copied verbatim) --------
    SceneInitialState  initial_state_;
    cook::SettleSpec   settle_;
    std::vector<TerrainRecord> terrain_;
    EnvironmentRecord  environment_;

    // -- facade state -------------------------------------------------------
    SceneGraph tree_;
    Registry   ecs_;
    bool       facade_dirty_ = false;  // set by Get*Mut; cleared by EnsureFacade

    // record-id -> entity, and BodyId -> its body node (so a child body / shape
    // / joint can be hung under its parent). Index == record id (dense).
    std::vector<EntityId>                    body_entity_;
    std::vector<EntityId>                    shape_entity_;
    std::vector<EntityId>                    joint_entity_;
    std::vector<EntityId>                    media_entity_;
    std::vector<EntityId>                    script_entity_;
    std::vector<std::shared_ptr<SceneNode>>  body_node_;

    // MaterialId -> {physics-material id, render-material id} in the Registry's
    // asset tables. Both default to Registry::kNoSlot until a material projects.
    struct MaterialIds {
        uint32_t phys   = Registry::kNoSlot;
        uint32_t render = Registry::kNoSlot;
    };
    std::vector<MaterialIds>                 material_ids_;
};

} // namespace nuka::scene
