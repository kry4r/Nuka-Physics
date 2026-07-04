#pragma once
// ---------------------------------------------------------------------------
// nuka::scene - ECS component contract (M2a frozen).
//
// The component set is CLOSED (this is a physics engine, not a general game
// engine), so the Registry stores one explicit dense pool per type below. The
// node tree (scene/graph/scene_graph.hpp) is the single authority for
// hierarchy: there is deliberately NO HierarchyComponent here, and `path` is
// always DERIVED via SceneGraph::PathOf. NameComponent::name mirrors the node
// name (the node is authoritative).
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/asset/asset_ref.hpp"
#include "scene/ecs/entity.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::scene {

// -- identity / spatial -----------------------------------------------------

struct NameComponent {
    std::string name;  // single segment, synced with the owning node's name
};

struct TransformComponent {
    math::Transform local;            // relative to the PARENT NODE
    math::Vec3      scale{1.0f, 1.0f, 1.0f};
};

// -- materials (standalone, id-referenced assets) ---------------------------

struct PhysicsMaterial {
    float static_friction  = 0.5f;
    float dynamic_friction = 0.5f;
    float restitution      = 0.0f;

    enum class Combine : uint8_t { Average, Min, Multiply, Max };
    Combine friction_combine    = Combine::Average;
    Combine restitution_combine = Combine::Average;

    // Soft (compliant) contact params -> maps to the existing solref/solimp.
    float compliant_stiffness = 0.0f;
    float compliant_damping   = 0.0f;

    float density = 1000.0f;

    // SDF bake directives.
    float    sdf_cell_size = 0.005f;
    uint16_t sdf_min_res   = 32;
    uint16_t sdf_max_res   = 128;
};

struct RenderMaterial {
    float base_color[4]{0.8f, 0.8f, 0.8f, 1.0f};
    float metallic  = 0.0f;
    float roughness = 0.5f;
    float emissive[3]{0.0f, 0.0f, 0.0f};
    float opacity = 1.0f;

    // Grazing-angle retroreflective sheen weight (silk/satin luster), consumed only
    // by the offline beauty path. 0 (default) is a strict no-op opaque surface.
    float sheen = 0.0f;

    // Refractive dielectric (water/glass) consumed only by the offline beauty
    // path; transmission==0 (default) is a strict no-op opaque surface.
    float transmission = 0.0f;       // 0 = opaque; 1 = fully transmissive
    float ior          = 1.0f;       // refractive index (water ~1.33)
    float absorption[3]{0.0f, 0.0f, 0.0f};  // per-metre Beer-Lambert extinction

    // Image-file PBR texture maps (file paths; empty => untextured flat PBR).
    // Consumed by the offline beauty tracer: albedo is sRGB and multiplies
    // base_color, roughness/normal are linear. uv_scale tiles the maps (tiles per
    // world metre under triplanar; per UV unit otherwise); triplanar projects the
    // maps in world space for geometry without UVs (boxes/terrain/particle skins).
    std::string albedo_map;
    std::string roughness_map;
    std::string normal_map;
    float uv_scale  = 1.0f;
    bool  triplanar = true;
};

// -- per-entity components ---------------------------------------------------

struct VisualMeshComponent {
    AssetRef mesh;
    uint32_t render_material_id = ~uint32_t(0);

    // M10: PRIMITIVE visual fallback. When `mesh` is empty but the source geom is a
    // box/sphere/capsule/plane VISUAL primitive (e.g. the Robocasa kitchen
    // counters / walls / floor, which are primitive geoms with no MESH asset), the
    // SceneIR records its shape here so render_world.cpp tessellates it instead of
    // skipping -- the skip left primitive-heavy scenes invisible except for their
    // few mesh appliances. kind None => a mesh-backed (or non-renderable) visual,
    // unchanged. params layout mirrors CollisionShapeComponent: sphere(r) /
    // capsule(r,half_height) / box|plane(hx,hy,hz). PROJECTED from records (never
    // serialized), so .nks/.nka roundtrip + cook goldens are unaffected.
    enum class PrimKind : uint8_t { None = 0, Sphere, Capsule, Box, Plane };
    PrimKind prim_kind = PrimKind::None;
    float    prim_params[4]{0.0f, 0.0f, 0.0f, 0.0f};
};

struct CollisionShapeComponent {
    enum class Kind : uint8_t { Sphere, Capsule, Box, Plane, ConvexHull, SdfMesh };
    Kind     kind = Kind::Sphere;
    float    params[4]{0.0f, 0.0f, 0.0f, 0.0f};
    AssetRef cooked;                          // hull / SDF reference into .nka
    uint32_t physics_material_id = ~uint32_t(0);
    // Authored render material for a collision-only body's proxy visual (a body
    // that ships no VisualMeshComponent). kNoSlot => the shared default appearance.
    uint32_t render_material_id  = ~uint32_t(0);
    uint32_t group = 0;                       // replaces magic handles 7000/8500/9000+
    uint32_t mask  = 0;
};

struct RigidBodyComponent {
    float           mass = 0.0f;
    math::Vec3      inertia_diag{0.0f, 0.0f, 0.0f};
    math::Transform inertial_frame = math::Transform::Identity();
    bool            kinematic        = false;
    float           linear_damping   = 0.0f;
    float           angular_damping  = 0.0f;
};

struct JointComponent {
    enum Kind { Revolute, Prismatic, Fixed, Spherical, Free };
    Kind       kind = Fixed;
    EntityId   parent_body = kInvalidEntity;
    EntityId   child_body  = kInvalidEntity;
    math::Vec3 axis{0.0f, 0.0f, 1.0f};
    float      limit_lo = 0.0f;
    float      limit_hi = 0.0f;
    float      damping  = 0.0f;
    float      armature = 0.0f;
    float      friction = 0.0f;
};

struct ActuatorComponent {
    enum Mode { PD, Torque, Velocity };
    Mode  mode = PD;
    float stiffness = 0.0f;
    float damping   = 0.0f;
    float effort_limit     = 0.0f;   // dual clamp (Isaac T9):
    float effort_limit_sim = 0.0f;   //   real-hardware vs sim-side limit
    float velocity_limit   = 0.0f;
};

struct SystemKindComponent {
    enum K { Rigid, Articulated, Soft, Cloth, Fluid };
    K kind = Rigid;                  // morph -> solver routing
};

struct SoftBodyComponent {
    AssetRef tet_or_cloth_mesh;
    float    young            = 0.0f;
    float    poisson          = 0.0f;
    float    xpbd_compliance  = 0.0f;
    // Per-medium solver selection. Xpbd (default) = the existing position-based
    // soft solver; MlsMpm = the continuum MPM medium. Default preserves the
    // byte-identical cook for every existing scene.
    enum class SimMethod : uint8_t { Xpbd = 0, MlsMpm = 1 };
    SimMethod sim_method = SimMethod::Xpbd;
};

struct FluidComponent {
    float particle_size   = 0.0f;
    float rho0            = 0.0f;
    float viscosity       = 0.0f;
    float surface_tension = 0.0f;
};

struct InitialStateComponent {
    std::vector<float> qpos;                         // settle output
    math::Transform    root = math::Transform::Identity();
};

// CameraComponent / LightComponent mirror the existing scene::CameraRecord /
// LightRecord (scene/scene_ir.hpp). The records' `attached_body` is NOT a
// component field: attachment is encoded structurally by the scene TREE (the
// camera/light node is parented under the attached body's node — see
// SceneIR::ProjectCamera/ProjectLight); every other field maps 1:1.

struct CameraComponent {
    math::Transform local_transform     = math::Transform::Identity();
    float           vertical_fov_degrees = 45.0f;
    float           near_clip            = 0.01f;
    float           far_clip             = 1000.0f;
};

struct LightComponent {
    enum class Type : uint8_t { Point, Directional, Spot, Area };
    Type            type            = Type::Point;
    math::Transform local_transform = math::Transform::Identity();
    math::Vec3      color           = {1.0f, 1.0f, 1.0f};
    float           intensity       = 1.0f;
};

// Scripting node payload: inline source + a stable id that survives a reload. The
// blessed reversal of the otherwise-closed component set (scripts live in scenes).
struct ScriptComponent {
    std::string source;          // inline Python source (UTF-8)
    uint64_t    stable_id = 0;   // serialized node identity, stable across reload
};

} // namespace nuka::scene
