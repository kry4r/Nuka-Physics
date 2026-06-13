#pragma once
// ---------------------------------------------------------------------------
// nk::Model — the immutable cook product (plan §3.3).
//
// Model is the cook-constant half of an nk::World: joint/link topology, link
// inertias + local poses, shape tables, XPBD constraint templates, PBF params,
// the physics-material bucket table initial values, the contact filter policy,
// the max_* capacities, and the field schema. It holds host-side std::vectors
// and a single UploadTo() that PACKS the model-owned tables into ONE device
// phi::Buffer (256B-aligned sections, deterministic layout) and fills a
// phi::ModelView of typed device pointers into that buffer.
//
// PURE C++ — zero CUDA tokens. All device memory via phi::BufferType/BufferI,
// no execution here (Model only allocates + uploads its constant tables).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"   // phi::ModelView, BufferType, Buffer (forward + wrappers)
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/generated/arena_layout.hpp"
#include "nk/model/generated/views.hpp"

namespace nuka::nk {

// Fixed-capacity policy (plan §3.3): every per-env array is sized to its max
// slot count; runtime `*_count` watermarks mask the unused tail. Capacities are
// a Model property (deterministic for a given cook product + env_count).
struct ModelCapacities {
    uint32_t env_count       = 1;   // number of replicated envs (env-major).
    uint32_t dofs_per_env    = 0;   // articulation generalized DOF count / env.
    uint32_t links_per_env   = 0;   // articulation link count / env.
    uint32_t bodies_per_env  = 0;   // movable rigid body count / env.
    uint32_t max_contacts_per_env = 0;  // contact-slot capacity / env.
    uint32_t max_rows_per_env     = 0;  // row-slot capacity / env.
    uint32_t max_hull_verts       = 0;  // convex-hull vertex pool capacity (global).
    uint32_t particles_per_env    = 0;  // XPBD/PBF particle count / env.
    uint32_t dist_cons_per_env    = 0;  // XPBD distance-constraint count / env.
    uint32_t bend_cons_per_env    = 0;  // XPBD bend-constraint count / env.
    uint32_t vol_cons_per_env     = 0;  // XPBD volume-constraint count / env.
    uint32_t num_material_buckets = 0;  // physics-material bucket table rows.
    uint32_t obs_width            = 64; // per-env observation export width.

    // M5 — pair-driven broadphase / SDF main-path capacities (GLOBAL tables,
    // per-env template, base-relative; the union slot-template family leaves
    // them 0). Sized by the cooked pair-driven scene; the SDF/SAMP grids are 0
    // for a scene with no SdfMesh collision shape (the union scene).
    uint32_t max_bodies_total     = 0;  // collidable body rows / env (shape_table).
    uint32_t max_excluded_pairs   = 0;  // cooked filter exclude-list length.
    uint32_t max_samp_points      = 0;  // SDF sampling-point pool size.
    uint32_t max_sdf_grids        = 0;  // cooked sparse-SDF grid count.
    uint32_t max_sdf_cells        = 0;  // total cooked narrow-band SDF cells.

    // M6 — particle uniform-grid cell capacity PER ENV (= the cooked
    // grid_dims product; 0 for a scene with no PBF grid). Sizes the
    // grid_cell_start/grid_cell_end arena fields (cells x env_count); the
    // ParticleGridBuild op fails LOUDLY if the live dims product exceeds it.
    uint32_t max_grid_cells       = 0;

    // Resolve a FieldPer count-unit to a concrete per-env element count using
    // these capacities (env-major; the env multiplier is applied by the Arena /
    // UploadTo packer, NOT folded in here -- this returns the PER-ENV count).
    uint32_t PerEnvCount(FieldPer per) const;
    // Total element count across all envs for a field (per-env x env_count, with
    // scalar fields counted once per env unless they are global). mat_buckets is
    // the one symbolic per:scalar field; its count = num_material_buckets*8.
    uint64_t ElementCount(FieldId id) const;
};

// A cook-time articulation template (one per env, replicated). The arrays are
// the SINGLE-ENV BuildArticulationHostState product transcribed 1:1 (M3b), so
// the staged Model bytes match the legacy UploadArticulationState bytes exactly
// (the byte-exact kernel-port contract).
struct ModelArticulation {
    // per-link topology (length == links_per_env)
    std::vector<uint32_t>         parent_link;       // articulation-LOCAL; ~0u at the root
    std::vector<uint8_t>          joint_type;        // ArticulationJointType (u8-backed)
    std::vector<math::Vec3>       joint_axis;
    std::vector<math::Vec3>       parent_offset;     // parent_frame.position (host-state build)
    std::vector<uint32_t>         link_body;         // owning rigid-body row
    std::vector<math::Transform>  link_local_pose;
    std::vector<math::Transform>  link_inertial_frame;
    std::vector<float>            link_inertia_spatial;  // 36 floats / link (flat)
    std::vector<float>            joint_damping;
    std::vector<float>            joint_armature;
    std::vector<float>            initial_q;             // per LINK (scalar slot per link)
    std::vector<math::Transform>  initial_link_pose;     // cook rest pose per link
    // M4 (union family): the SETTLED initial velocity state (the legacy
    // gripper_proto carries it after the factory's oracle settle pre-roll).
    // Empty == zero-velocity start (the M3 CookToModel path, unchanged).
    std::vector<float>            initial_qdot;          // per LINK (scalar slot)
    std::vector<float>            initial_link_velocity; // 6 floats / link (flat)
    math::Transform               base_pose = math::Transform::Identity();  // root world pose
    uint32_t                      dof_count  = 0;        // generalized DOFs (floating root = 6)
    uint32_t                      link_count = 0;
    uint32_t                      root_link  = ~uint32_t(0);
};

// One cooked foot-sphere row (legacy articulation::FootShape 1:1: base-relative
// calf link + sphere center offset in the calf frame + radius). Staged into the
// foot_shape Model field as 5 packed scalars {link bits, off.xyz, radius}.
struct ModelFootShape {
    uint32_t   calf_local_link = 0;
    math::Vec3 local_offset{};
    float      radius = 0.0f;
};

// Cook-seeded PD hold-drive template (per template link; legacy
// BuildHoldDriveTargets / CookGo2 HoldDrives 1:1). Seeded into the Data
// drive_* persistent fields at World construction (replicated env-major).
struct ModelHoldDrives {
    std::vector<float> targets;       // = initial_q (hold the cooked stance)
    std::vector<float> stiffness;     // actuator gain (Position actuators)
    std::vector<float> damping;       // 2*sqrt(gain) (DefaultDriveDamping)
    std::vector<float> force_limits;
};

// A cooked shape row (primitive params + optional hull/SDF asset reference).
struct ModelShape {
    uint8_t          kind = 0;          // scene::ShapeType
    uint32_t         body_row = ~uint32_t(0);
    uint32_t         material_bucket = 0;
    math::Transform  local_transform = math::Transform::Identity();
    math::Vec3       half_extents{0.5f, 0.5f, 0.5f};
    float            radius = 0.5f;
    float            half_height = 0.5f;
    uint32_t         convex_geometry_index = ~uint32_t(0);
    uint32_t         sdf_index = ~uint32_t(0);
};

// A physics-material bucket row (8 floats: μs, μd, restitution, compliant_k,
// compliant_d, density, sdf_cell_size, reserved). The mat_buckets device field
// is num_buckets x 8 floats (plan §3.3 example).
struct ModelMaterialBucket {
    float values[8] = {0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 1000.0f, 0.005f, 0.0f};
};

// ---------------------------------------------------------------------------
// M4: the UNION (CSR compliant) contact family — Model-side authoring tables.
// ---------------------------------------------------------------------------

// Which contact pipeline this Model runs through the AssembleRows /
// SolveRowsBlockIsland ops:
//   FusedFoot — the M3 articulation foot pipeline (sphere x ground detection,
//               ArticulatedContactRow records, the fused block-per-articulation
//               PGS — the legacy batched articulated foot semantics, goldens
//               byte-exact).
//   UnionCsr  — the union compliant-CSR pipeline (feet x ground + finger x
//               hull + box x plane in ONE solve — the legacy coresident union
//               world semantics).
//   PairDriven — the M5 generalized broadphase->narrowphase pipeline (per-env
//               LBVH candidate_pairs -> analytic prim dispatch + SDF main path).
//               ADDITIVE; the union slot-template path is untouched.
enum class ContactFamily : uint8_t { FusedFoot = 0, UnionCsr = 1, PairDriven = 2 };

// One union contact-pair slot (the per-env contact template, replicated across
// envs; the legacy UnionSceneTemplate fingertips/feet/table classes 1:1).
// Worst-case manifold points: sphere classes -> 1, box x plane -> 4. Each point
// expands to 1 normal row + (condim>=2 ? 2*(condim-1) : 0) friction-spoke rows
// (the EmitCompliantContactRows layout: ALL normals first, then ALL spokes).
struct UnionSlot {
    enum Class : uint32_t {
        kInactive       = 0,
        kFootSpherePlane = 1,  // artic sphere (link+offset+radius) x +Z plane.
        kFingerSphereHull = 2, // artic sphere x convex hull on body `body`.
        kBodyBoxPlane   = 3,   // rigid box (body+offset+half) x +Z plane.
        // M6 particle coupling classes: side a == a PARTICLE sphere (the slot's
        // `link` field carries the LOCAL particle index; detection forms its
        // world pos from particle_pos[env*particles_per_env + link]); side b is
        // a static +Z plane (kParticleSpherePlane) or a rigid box (the slot's
        // `body` field; kParticleSphereBox). Radius == particle contact radius.
        kParticleSpherePlane = 4,  // particle sphere x +Z plane (static floor).
        kParticleSphereBox   = 5,  // particle sphere x rigid box (body+offset+half).
    };
    uint32_t   cls = kInactive;
    uint32_t   link = ~0u;            // template-local link (sphere classes).
    uint32_t   body = ~0u;            // template-local body row (hull owner / box).
    uint32_t   condim = 3;            // 1 (frictionless) or 3 (4 spokes).
    math::Vec3 offset{};              // sphere local offset / box-center offset.
    float      radius = 0.0f;         // sphere radius.
    math::Vec3 box_half{};            // box half extents (kBodyBoxPlane).
    float      plane_height = 0.0f;   // +Z plane height (ground / table z).
    float      mu = 0.5f;             // per-class friction stamp (RowMaterial.friction).
    uint32_t   flags = 0;             // bit0: emission gated on the table_enabled field.

    // Worst-case manifold points / rows for this slot.
    uint32_t MaxPoints() const { return cls == kBodyBoxPlane ? 4u : 1u; }
    uint32_t RowsPerPoint() const {
        return 1u + (condim >= 2u ? 2u * (condim - 1u) : 0u);
    }
    uint32_t MaxRows() const { return MaxPoints() * RowsPerPoint(); }
};
inline constexpr uint32_t kUnionSlotGatedOnTable = 1u;  // UnionSlot::flags bit0.

class Model {
public:
    Model() = default;

    // -- cook-product host tables (filled by CookToModel) -------------------
    ModelCapacities                 capacities;
    ModelArticulation               articulation;       // the per-env template.
    std::vector<ModelShape>         shapes;             // cooked shape rows (per env).
    std::vector<ModelMaterialBucket> material_buckets;  // (num_buckets) bucket table.
    std::vector<uint32_t>           body_material_bucket;  // per body-row bucket index.

    // -- articulation contact pipeline config (M3b foot sphere x ground) -----
    // feet: derived by CookToModel (every Sphere shape owned by an articulation
    // link, the T2/T6 derivation). ground_height/friction/baumgarte are scene/
    // run config a caller may override BEFORE World construction (the Model is
    // mutable until then). foot count of 0 disables contact generation (the
    // detection kernel then zero-fills every slot — the single-env oracle path).
    std::vector<ModelFootShape>     feet;
    ModelHoldDrives                 hold_drives;
    float ground_height          = 0.0f;
    float friction_coefficient   = 0.8f;   // legacy kContactFriction
    float baumgarte_max_velocity = 3.0e38f; // ~+inf (legacy default non-binding)

    // -- M6: particle (XPBD soft + PBF fluid) cook product --------------------
    // The XPBD constraint templates (dist/bend/vol, owner:model) are staged by
    // UploadTo; the particle initial state (pos/prev/vel/inv_mass) is seeded by
    // World::SeedInitialState. PBF params are resolved into the ParticleGridBuild
    // + Pbf* op params by Pipeline::Build. A scene with no particles leaves these
    // empty and particles_per_env == 0 (Pipeline emits no particle ops).
    // Coupled: particles co-step against rigid/artic bodies through the unified
    // row solve (the kUSlotParticleSphere* contact classes); the internal XPBD
    // soft constraints (if any) still run. None/Xpbd/Pbf are the standalone modes.
    enum class ParticleMode : uint8_t { None = 0, Xpbd = 1, Pbf = 2, Coupled = 3 };
    // In Coupled mode, which internal dynamics run alongside the contact coupling:
    // None (free point masses), Xpbd (soft constraints), Pbf (fluid density).
    enum class CoupledInternal : uint8_t { None = 0, Xpbd = 1, Pbf = 2 };
    struct ModelParticles {
        ParticleMode mode = ParticleMode::None;
        CoupledInternal coupled_internal = CoupledInternal::None;
        // Initial per-particle state (env-major replicated by SeedInitialState;
        // these are the SINGLE-ENV template, length == particles_per_env).
        std::vector<math::Vec3> initial_pos;
        std::vector<math::Vec3> initial_vel;
        std::vector<float>      inv_mass;
        // XPBD constraint templates (single-env; staged + dispatched per env).
        std::vector<uint32_t> dist_a, dist_b;     // distance endpoints
        std::vector<float>    dist_rest, dist_alpha;
        std::vector<uint32_t> bend_particles;     // 4 / bend constraint
        std::vector<math::Vec3> bend_gradients;   // 4 / bend constraint
        std::vector<float>    bend_alpha;
        std::vector<uint32_t> vol_particles;      // 4 / volume constraint
        std::vector<float>    vol_rest6, vol_alpha;
        uint16_t xpbd_iters = 1;
        // PBF fluid params (Macklin & Mueller 2013 + p10-B polish).
        float    pbf_rest_density   = 0.0f;   // rho0 (0 disables PBF)
        float    pbf_support_radius = 0.0f;   // h (== grid query radius / cell)
        float    pbf_particle_mass  = 0.0f;
        float    pbf_cfm_epsilon    = 1.0e-6f;
        uint16_t pbf_iters          = 4;
        bool     pbf_clamp_overdensity = true;
        float    pbf_xsph_viscosity = 0.0f;
        float    pbf_surface_tension= 0.0f;
        // Uniform grid domain (cook-derived; the grid is built every step over
        // the predicted positions but sized from this cooked AABB + cell size).
        math::Vec3 grid_min{0.0f, 0.0f, 0.0f};
        uint32_t   grid_dims[3] = {0u, 0u, 0u};
        float      cell_size  = 0.0f;        // == support radius (>= query radius).
        float      query_radius = 0.0f;      // == support radius.
        // Boundary floor (z-up; the M5 grid + particle ops are z-up).
        bool  boundary_enabled = false;
        float floor_z = 0.0f;
    };
    ModelParticles particles;

    // -- M4: contact family + union (CSR compliant) tables --------------------
    ContactFamily contact_family = ContactFamily::FusedFoot;
    // Drive mode the ApplyDrives op runs (0 = position PD hold drive, the M3
    // batched articulated path; 1 = direct torque drive, the union world's
    // LaunchApplyTorqueDriveKernels path — drive_target carries the torque).
    uint32_t drive_mode = 0;
    std::vector<UnionSlot> union_slots;   // per-env contact-pair template.
    std::vector<float>     hull_verts;    // mesh-local hull verts (xyz packed).
    // Merged contact params for the union rows (the legacy BuildContactManifolds
    // MergeContactParams(default,default) product — per-class mu comes from the
    // slot's mu stamp, exactly as the legacy class stamp overwrote it).
    float union_solref[2] = {0.02f, 1.0f};
    float union_solimp[5] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    bool  table_enabled_default = true;   // seeds the per-env table_enabled field.
    // Per-env movable rigid body initial state (template; replicated env-major).
    struct BodyInit {
        math::Transform pose = math::Transform::Identity();
        math::Vec3 linear_velocity{};
        math::Vec3 angular_velocity{};
        float      inv_mass = 0.0f;
        math::Vec3 inv_inertia{};
    };
    std::vector<BodyInit> body_init;
    // The flat-DOF -> (template-local link, base component) maps (the legacy
    // DofIndexOf prefix-sum inverse; component == ~0u for scalar joints). Cooked
    // by the union cook; staged into the dof_to_link/dof_to_component fields.
    std::vector<uint32_t> dof_to_link;
    std::vector<uint32_t> dof_to_component;

    // -- M4: the device-resident solve schedule (filled by nk::SolveSchedule
    // BEFORE World::UploadTo; staged into the island_row_offsets /
    // island_color_segments / row_order Model fields). See solve/schedule.hpp.
    std::vector<uint32_t> schedule_row_order;       // [total rows in order]
    std::vector<uint32_t> schedule_color_segments;  // pairs {offset, count}
    std::vector<uint32_t> schedule_islands;         // quads {seg_off, seg_cnt, flags, env}
    uint32_t schedule_island_count  = 0;
    uint32_t schedule_segment_count = 0;
    // Convex hull geometry + SDF grids are referenced by ModelShape indices and
    // packed into the .nka by the cooker; the device upload of those large
    // assets is the collision milestone's (M5) business, not M3a's.

    // Filter policy (plan §3.3): cross-env collision flag + excluded pairs.
    // M3a carried the flag; M5 carries the SORTED (lo<<32|hi) device exclude
    // list (staged into the excluded_pairs field) for the pair-driven query.
    bool filter_cross_env = false;
    std::vector<uint64_t> excluded_pairs;   // SORTED ascending canonical keys.

    // -- M5: pair-driven generalized collision tables (the union slot-template
    // family leaves these empty). shape_table_rows: one PairDrivenShape per
    // collidable body row (cooked from CollisionShapeComponent). samp_points/
    // samp_ranges: the SDF sampling-point pool + per-body slice. sdf_*: the
    // cooked sparse narrow-band SDF (mirrors scene::CookedSdfTable). --------
    struct PairDrivenShape {
        uint32_t   kind = 0;             // CollisionShapeComponent::Kind.
        float      params[4] = {0, 0, 0, 0};  // sphere r / capsule r,hh / box he.
        uint32_t   contype = 1;
        uint32_t   conaffinity = 1;
        uint32_t   sdf_grid = ~0u;       // sdf_headers index, or ~0 (analytic).
    };
    std::vector<PairDrivenShape> shape_table_rows;
    std::vector<float>           samp_points;     // xyz packed.
    std::vector<uint32_t>        samp_ranges;     // {offset,count} per body row.
    // Cooked sparse-SDF grids (host staging; uploaded by UploadTo into the
    // sdf_* Model fields — the SdfDeviceWorld upload duties moved INTO Model).
    struct SdfGrid {
        math::Vec3 origin{};
        float      voxel_size = 0.0f;
        uint32_t   dims[3] = {0, 0, 0};
        uint32_t   cell_offset = 0;      // base into the flat cell arrays.
        uint32_t   cell_count = 0;
    };
    std::vector<SdfGrid>     sdf_grids;
    std::vector<uint64_t>    sdf_cell_keys;       // flat ASCENDING per-grid.
    std::vector<float>       sdf_cell_values;     // flat signed distances.
    std::vector<math::Vec3>  sdf_cell_gradients;  // flat gradients.

    // The field schema is the generated FieldId enum + arena_layout table; Model
    // exposes the capacities the layout multiplies by. (No per-field storage here
    // beyond the constant tables above.)

    // -- device upload ------------------------------------------------------
    // Pack ALL model-owned (owner==Model) fields into ONE device buffer via the
    // BufferType allocator, upload them, and FILL `out_view` with typed device
    // pointers into that buffer. 256B-aligned sections, deterministic layout
    // (iteration in FieldId order). The returned Buffer is owned by the Model and
    // freed in the dtor. Returns Status::Ok / OutOfMemory.
    phi::Status UploadTo(phi::BufferType* bt, phi::ModelView* out_view);

    // The packed device buffer (null until UploadTo). Exposed for World teardown.
    phi::Buffer* DeviceBuffer() const { return device_buffer_; }

    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;

    // -- deterministic segment layout (shared with the test's determinism gate)
    // One packed section per model-owned field, in FieldId order, each 256B
    // aligned. Computed purely from `capacities`; identical inputs => identical
    // table. Public so the acceptance test can assert two builds match.
    struct Segment {
        FieldId  field;
        uint64_t offset;   // byte offset into the device buffer
        uint64_t bytes;    // section size (element_count x elem_size)
    };
    std::vector<Segment> ComputeModelSegments(uint64_t* total_bytes) const;

private:
    // Fill the per-field host staging bytes for one model-owned field into `dst`
    // at the segment offset (transcribes the host tables above into the packed
    // layout). Unimplemented (zero-filled) sections are left as the buffer's
    // memset(0) default -- the kernel-port milestone (M3b) wires the real packing
    // as each op lands; M3a guarantees the LAYOUT + pointers + determinism.
    void StageModelField(FieldId id, const Segment& seg, std::vector<uint8_t>& host) const;

    phi::Buffer* device_buffer_ = nullptr;
};

} // namespace nuka::nk
