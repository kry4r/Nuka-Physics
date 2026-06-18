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
};

struct SensorRecord {
    std::string name;
    SensorId id                            = 0;
    BodyId attached_body                   = kInvalidBody;
    SensorType type                        = SensorType::Imu;
    math::Transform local_transform        = math::Transform::Identity();
    float sample_rate_hz                   = 0.0f;
};

struct MaterialRecord {
    std::string name;
    MaterialId id                          = kInvalidMaterial;
    math::Vec3 base_color                  = {1.0f, 1.0f, 1.0f};
    float alpha                            = 1.0f;
    float roughness                        = 0.5f;
    float metallic                         = 0.0f;

    // Per-material default friction coefficient μ (owner Q8 "per-material μ").
    // Resolved into the per-shape cooked μ at cook time when a shape does not
    // carry an explicit per-shape override (see cooker.cpp precedence). (v0.8 C1a)
    float friction_mu                      = 1.0f;
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
    SensorId AddSensor(SensorRecord record);
    MaterialId AddMaterial(MaterialRecord record);
    CameraId AddCamera(CameraRecord record);
    LightId AddLight(LightRecord record);
    ActuatorId AddActuator(ActuatorRecord record);

    // Record an explicit collision-exclusion between two bodies. Stored
    // canonicalized as (min,max) so call order does not matter. The filter
    // POLICY that consumes this list (and the <contact><pair> explicit-override
    // list) is C1b/C1c; C1a only stores the authored excludes. (v0.8 C1a)
    void AddExcludePair(BodyId a, BodyId b);

    // Record an authored <contact><pair> explicit contact override (v0.8 C1b).
    // Stored verbatim in authoring order; the merge/policy that consumes these
    // is C1c.
    void AddContactPair(ContactPairOverride pair);

    // -- counts -------------------------------------------------------------
    size_t RigidBodyCount() const;
    size_t JointCount()     const;
    size_t ShapeCount()     const;
    size_t SensorCount()    const;
    size_t MaterialCount()  const;
    size_t CameraCount()    const;
    size_t LightCount()     const;
    size_t ActuatorCount()  const;

    // -- accessors ----------------------------------------------------------
    const RigidBodyRecord&      GetBody(BodyId id)    const;
    const JointRecord&          GetJoint(JointId id)  const;
    const CollisionShapeRecord& GetShape(ShapeId id)  const;
    const SensorRecord&         GetSensor(SensorId id) const;
    const MaterialRecord&       GetMaterial(MaterialId id) const;
    const CameraRecord&         GetCamera(CameraId id) const;
    const LightRecord&          GetLight(LightId id) const;
    const ActuatorRecord&       GetActuator(ActuatorId id) const;

    // Mutable record access. Mutating a record through these BYPASSES the Add*
    // write-through, so each Get*Mut marks the facade DIRTY; the next facade
    // read (Tree / Ecs / EntityOf*) lazily re-projects tree_ + ecs_ from the
    // records. NOTE: the re-projection creates fresh entities/nodes — EntityIds
    // and SceneNode pointers obtained BEFORE a Get*Mut are invalidated by it.
    RigidBodyRecord&      GetBodyMut(BodyId id);
    JointRecord&          GetJointMut(JointId id);
    CollisionShapeRecord& GetShapeMut(ShapeId id);
    SensorRecord&         GetSensorMut(SensorId id);
    MaterialRecord&       GetMaterialMut(MaterialId id);
    CameraRecord&         GetCameraMut(CameraId id);
    LightRecord&          GetLightMut(LightId id);
    ActuatorRecord&       GetActuatorMut(ActuatorId id);

    // -- bulk accessors -----------------------------------------------------
    const std::vector<RigidBodyRecord>&      Bodies()  const;
    const std::vector<JointRecord>&          Joints()  const;
    const std::vector<CollisionShapeRecord>& Shapes()  const;
    const std::vector<SensorRecord>&         Sensors() const;
    const std::vector<MaterialRecord>&       Materials() const;
    const std::vector<CameraRecord>&         Cameras() const;
    const std::vector<LightRecord>&          Lights() const;
    const std::vector<ActuatorRecord>&       Actuators() const;
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

    std::vector<RigidBodyRecord>      bodies_;
    std::vector<JointRecord>          joints_;
    std::vector<CollisionShapeRecord> shapes_;
    std::vector<SensorRecord>         sensors_;
    std::vector<MaterialRecord>       materials_;
    std::vector<CameraRecord>         cameras_;
    std::vector<LightRecord>          lights_;
    std::vector<ActuatorRecord>       actuators_;
    std::vector<std::pair<BodyId, BodyId>> exclude_pairs_;
    std::vector<ContactPairOverride>       contact_pairs_;

    // -- authored metadata (M7; not facade-derived, copied verbatim) --------
    SceneInitialState  initial_state_;
    cook::SettleSpec   settle_;

    // -- facade state -------------------------------------------------------
    SceneGraph tree_;
    Registry   ecs_;
    bool       facade_dirty_ = false;  // set by Get*Mut; cleared by EnsureFacade

    // record-id -> entity, and BodyId -> its body node (so a child body / shape
    // / joint can be hung under its parent). Index == record id (dense).
    std::vector<EntityId>                    body_entity_;
    std::vector<EntityId>                    shape_entity_;
    std::vector<EntityId>                    joint_entity_;
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
