#pragma once
// ---------------------------------------------------------------------------
// nk::Model — the immutable cook product (plan §3.3).
//
// Model is the cook-constant half of an nk::World: joint/link topology, link
// inertias + local poses, shape tables, VBD and MLS-MPM particle data,
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
#include <string>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/contact/contact_profile.hpp"
#include "nk/material/mpm_material.hpp"
#include "nk/solve/vertex_block.hpp"
#include "sensor/state_types.hpp"
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
    uint32_t integration_substeps = 1u;
    // Sweeps per env the solver sweep trace holds; widened when sweep telemetry is demanded.
    uint32_t solver_trace_sweeps = 0u;
    // Multi-articulation co-residence (K Go2 in ONE env): the number of SEPARATE
    // articulations resident in each env. The forward kernels are grid =
    // articulation_count == articulations_per_env * env_count, each dog keeping
    // its OWN max_dof^2 M-tile (dofs_per_env stays the per-DOG DOF, NEVER summed,
    // so the per-artic 64-DOF CRBA cap is safe for K separate ~18-DOF dogs). The
    // DEFAULT 1 is the legacy single-robot-per-env shape: every per:articulation
    // / per:articulation_dof2 field then has articulations_per_env == 1 entries,
    // identical to the prior per:env / per:env_dof2 layout (the K==1 D1 invariant).
    uint32_t articulations_per_env = 1;
    uint32_t dofs_per_env    = 0;   // PER-ARTICULATION generalized DOF (max single-dog DOF).
    uint32_t links_per_env   = 0;   // articulation link count / env (SUM across co-resident dogs).
    uint32_t bodies_per_env  = 0;   // movable rigid body count / env.
    uint32_t max_contacts_per_env = 0;  // contact-slot capacity / env.
    uint32_t rigid_contact_rows_per_env = 0u;
    uint32_t distinct_owner_rows_per_env = 0u;  // rows whose sides can be two moving owners
    uint32_t max_rows_per_env     = 0;  // row-slot capacity / env.
    uint32_t foot_shape_count = 0u;
    uint32_t joint_limit_rows_per_env = 0; // stable lower/upper slots when any bound exists
    uint32_t joint_friction_rows_per_env = 0; // one scalar slot per link when friction is authored
    uint32_t joint_drive_rows_per_env = 0;
    uint32_t mimic_couplings_per_env = 0;  // links whose joint follows another; they take no rows
    bool inverse_dynamics_controls = false;
    // A row joins two articulated owners only when some filtered shape pair does;
    // otherwise both row sides share one Jacobian.
    bool articulated_contact_pairs = true;
    uint32_t max_hull_verts       = 0;  // convex-hull vertex pool capacity (global).
    uint32_t max_mesh_triangles   = 0;
    uint32_t max_mesh_bvh_nodes   = 0;
    uint32_t max_mesh_edges       = 0;
    uint32_t max_mesh_edge_nodes  = 0;
    uint32_t mesh_vertex_source_count = 0u;
    uint32_t mesh_edge_source_count = 0u;
    uint32_t particles_per_env    = 0;  // VBD/MPM particle count / env.
    // Cloth aerodynamic-drag element count / env (one per surface triangle). 0 ==
    // no drag (the drag op is not emitted -> a drag-free cook is byte-identical).
    uint32_t aero_tris_per_env           = 0;
    uint32_t particle_surfaces_per_env = 0u;
    uint32_t particle_surface_triangles = 0u;
    uint32_t particle_surface_nodes_per_env = 0u;
    uint32_t particle_surface_edges_per_env = 0u;
    uint32_t particle_surface_edge_nodes_per_env = 0u;
    uint32_t particle_surface_neighbours_per_env = 0u;
    uint32_t point_endpoints_per_env = 0u;
    uint32_t point_endpoint_terms_per_env = 0u;
    // Vertex blocks occupy particles [vbd_particle_begin, + vbd_vertices_per_env) of each env.
    uint32_t vbd_particle_begin = 0u;
    uint32_t vbd_vertices_per_env = 0u;
    // Collidable bodies without a cooked surface mesh; vertex blocks meet them in the particle narrowphase.
    uint32_t analytic_collidable_bodies = 0u;
    uint32_t vbd_dynamic_vertices_per_env = 0u;
    uint32_t vbd_elements_per_env = 0u;
    uint32_t vbd_incidence_per_env = 0u;
    uint32_t vbd_colors = 0u;
    // Coarse levels of the vertex blocks, their nodes and node children; the last level is solved
    // densely with vbd_coarse_dense_nodes nodes, or zero when it stays too large.
    uint32_t vbd_coarse_levels = 0u;
    uint32_t vbd_coarse_nodes = 0u;
    uint32_t vbd_coarse_children = 0u;
    uint32_t vbd_coarse_dense_nodes = 0u;
    uint32_t num_material_buckets = 0;  // physics-material bucket table rows.
    uint32_t obs_width            = 64; // per-env observation export width.

    // M5 — pair-driven broadphase / SDF main-path capacities (GLOBAL tables,
    // per-env template, base-relative; the union slot-template family leaves
    // them 0). Sized by the cooked pair-driven scene; the SDF/SAMP grids are 0
    // for a scene with no SdfMesh collision shape (the union scene).
    uint32_t max_bodies_total     = 0;  // collidable body rows / env (shape_table).
    uint32_t max_excluded_pairs   = 0;  // cooked filter exclude-list length.
    uint32_t max_contact_pairs    = 0;  // authored contact pair count.
    uint32_t max_samp_points      = 0;  // SDF sampling-point pool size.
    uint32_t max_sdf_grids        = 0;  // cooked sparse-SDF grid count.
    uint32_t max_sdf_cells        = 0;  // total cooked narrow-band SDF cells.

    // Byte size of the pair_sort_scratch field (the LbvhQueryPairs canonicalizing
    // radix sort over the rigid slot capacity; sized at World construct,
    // 0 == non-PairDriven world -> the emitted stream stays as-is).
    uint64_t pair_sort_scratch_bytes = 0;
    uint64_t lbvh_sort_scratch_bytes = 0;
    uint64_t contact_cache_scratch_bytes = 0;
    uint64_t contact_index_scratch_bytes = 0;
    // The contact index exists once a contact readout is demanded, in buffers of its own.
    bool contact_readout = false;
    // Coulomb block responses exist only while a solve measures contact residuals.
    bool contact_response = false;
    uint64_t pair_sample_chunk_words = 0;
    uint64_t block_descent_scratch_words = 0;

    // MLS-MPM background grid node count PER ENV (the cooked grid dims product; 0
    // for a non-MPM world). Sizes the grid_mass/momentum/velocity/force fields.
    uint32_t mpm_grid_nodes_per_env = 0;
    uint32_t mpm_contact_capacity_per_env = 0;  // Shared pool; multiple contacts may use one material point.
    uint32_t ogc_contacts_per_env = 0u;
    // Mesh contact slots particle surfaces reserve beyond one per vertex and edge, from their pair budgets.
    uint32_t ogc_surface_reserve_per_env = 0u;
    // Implicit stress cells, one row block per occupied grid cell, at the tail of each env's row span.
    uint32_t mpm_stress_cells_per_env = 0;
    // Byte size of the mpm_sort_scratch field (the P2G deterministic-gather cub
    // sort temp + out buffers; sized at World construct; 0 == no MPM particles).
    uint64_t mpm_grid_sort_scratch_bytes = 0;
    // MLS-MPM material-table row count (indexed by particle_material_id); 0 for a
    // non-MPM world -> zero-byte mpm_material_table segment.
    uint32_t mpm_material_count = 0;
    bool mpm_plastic_state = false;

    // H1 (general contact pipeline Phase 0) — total cooked heightfield-grid cells
    // (sum over all heightfield collidables of nrow*ncol). Sizes the GLOBAL
    // `heights[]` Model field. 0 for every current scene (the cook fill is H2,
    // Phase 2) -> the heights segment is zero bytes (byte-inert).
    uint32_t max_heightfield_cells = 0;

    // Resolve a FieldPer count-unit to a concrete per-env element count using
    // these capacities (env-major; the env multiplier is applied by the Arena /
    // UploadTo packer, NOT folded in here -- this returns the PER-ENV count).
    uint64_t PerEnvCount(FieldPer per) const;
    phi::Status Validate(std::string* reason = nullptr) const;
    // Total element count across all envs for a field (per-env x env_count, with
    // scalar fields counted once per env unless they are global). mat_buckets is
    // the one symbolic per:scalar field; its count = num_material_buckets*8.
    uint64_t ElementCount(FieldId id) const;
    // Body-particle contact slots of one env above the rigid budget; grid-owned particles hold none.
    // Vertex blocks reserve them only when an analytic body exists; meshed bodies meet them in OGC.
    uint64_t ParticleContactReserve(uint32_t grid_particles) const;
    uint32_t OgcPointSlotCapacity() const {
        return particle_surfaces_per_env > 0u ? ogc_contacts_per_env : 0u;
    }
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
    std::vector<float>            joint_frictionloss;    // MuJoCo joint dry friction bound
    std::vector<float>            joint_stiffness;
    std::vector<float>            joint_spring_reference;
    std::vector<constraint::LawSource> joint_law_source;
    std::vector<constraint::SolverReference> joint_limit_reference;
    std::vector<constraint::SolverReference> joint_friction_reference;
    std::vector<float>            joint_limit_margin;
    std::vector<float>            joint_limit_lower;
    std::vector<float>            joint_limit_upper;
    std::vector<uint8_t>          joint_limit_flags;     // bit0 lower, bit1 upper
    std::vector<uint32_t>         mimic_source_link;
    std::vector<float>            mimic_multiplier;
    std::vector<float>            mimic_offset;
    std::vector<float>            initial_q;             // per LINK (scalar slot per link)
    std::vector<math::Transform>  initial_link_pose;     // cook rest pose per link
    // M4 (union family): the SETTLED initial velocity state (the legacy
    // gripper_proto carries it after the factory's oracle settle pre-roll).
    // Empty == zero-velocity start (the M3 CookToModel path, unchanged).
    std::vector<float>            initial_qdot;          // per LINK (scalar slot)
    std::vector<float>            initial_link_velocity; // 6 floats / link (flat)
    math::Transform               base_pose = math::Transform::Identity();  // root world pose (artic 0).
    uint32_t                      dof_count  = 0;        // PER-ARTIC generalized DOFs (max single-dog; floating root = 6)
    uint32_t                      link_count = 0;        // TOTAL links across co-resident articulations.
    uint32_t                      root_link  = ~uint32_t(0);
    // Multi-articulation co-residence (K Go2 in one env). When the cooked scene
    // holds >1 disjoint kinematic tree, these carry the per-articulation
    // bookkeeping (BuildArticulationHostState builds them over ALL topologies):
    //   base_poses              : K root world poses (one per co-resident dog).
    //   articulation_link_count : K link counts (one per dog).
    //   articulation_link_offset: K base offsets into the flat per-env link arrays.
    //   articulation_count      : K == base_poses.size().
    // For the legacy single-robot scene articulation_count == 1, base_poses ==
    // { base_pose }, and the staging collapses to the prior per:env layout (D1).
    std::vector<math::Transform>  base_poses;            // K root world poses.
    std::vector<uint32_t>         articulation_link_count;   // K per-dog link counts.
    std::vector<uint32_t>         articulation_link_offset;  // K per-dog flat link offsets.
    // Per-template-link LOCAL articulation index (length == link_count). Built by
    // BuildArticulationHostState over all co-resident topologies; staged into the
    // link_to_articulation field with the +e*K per-replica offset. Empty / all-0
    // for the single-articulation scene (link l -> artic 0).
    std::vector<uint32_t>         link_to_articulation;
    uint32_t                      articulation_count = 1;
    // Per-template-link collision geometry (per template-link; length ==
    // link_count when populated, else empty). ONE primitive collision shape per
    // link in the link's LOCAL frame, the source the GENERAL contact path uses:
    // SyncLinkBodyPose poses (link_pose o link_geom_local) into the link's body
    // row so the link enters the LBVH as a collidable body for body<->body contact.
    //   link_geom_kind   : scene::ShapeType (0 == none; only Sphere/Box/Capsule
    //                      are collidable here).
    //   link_geom_params : 4 packed f32 per link (sphere r / capsule r,hh / box he).
    //   link_geom_local  : the shape's LOCAL transform relative to its link.
    // Empty for any cook without per-link collision geometry, so the field stays
    // zero-filled and every such scene is byte-identical.
    std::vector<uint32_t>         link_geom_kind;
    std::vector<float>            link_geom_params;   // 4 f32 / link (flat)
    std::vector<math::Transform>  link_geom_local;
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
    std::vector<float> joint_force_limits;  // joint-level range on the summed drive (MuJoCo actfrcrange)
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
    uint8_t          mesh_contact_mode = 0u;
};

// Compatibility aliases for hand-built models. Production contact reads the
// complete ContactProfileV1 record through the represented collidable.
inline constexpr uint32_t kBucketStaticMu = kContactProfileMu1;
inline constexpr uint32_t kBucketDynamicMu = kContactProfileMu2;
inline constexpr uint32_t kBucketDensity = kContactProfileFlags;

struct ModelMaterialBucket {
    static constexpr uint32_t kValueCount = kContactProfileWordCount;
    float values[kValueCount];

    ModelMaterialBucket() { PackContactProfile(ContactProfileV1{}, values); }
    explicit ModelMaterialBucket(const ContactProfileV1& profile) {
        PackContactProfile(profile, values);
    }
    ContactProfileV1 Profile() const { return UnpackContactProfile(values); }
};

bool CanonicalizeContactProfileForUpload(const ModelMaterialBucket& source,
                                         ModelMaterialBucket* canonical);

// ---------------------------------------------------------------------------
// M4: the UNION (CSR compliant) contact family — Model-side authoring tables.
// ---------------------------------------------------------------------------

// Which contact pipeline this Model runs through the AssembleRows /
// BlockDescentSolve ops:
//   FusedFoot — DEAD (L1-b): the legacy M3 articulation foot pipeline runtime
//               was DELETED. The enum value is retained (not renumbered) until
//               the L1-d enum collapse; no runtime branch reads it anymore.
//   UnionCsr  — the union compliant-CSR pipeline (feet x ground + finger x
//               hull + box x plane in ONE solve — the legacy coresident union
//               world semantics).
//   PairDriven — the M5 generalized broadphase->narrowphase pipeline (per-env
//               LBVH candidate_pairs -> analytic prim dispatch + SDF main path).
//               The GENERAL default for every non-union cooked model.
enum class ContactFamily : uint8_t { FusedFoot = 0, UnionCsr = 1, PairDriven = 2 };

// L1-c: the UnionSlot struct (the legacy union contact-pair template) + the
// kUnionSlotGatedOnTable flag were DELETED here together with the entire UnionCsr
// row-assembly pipeline. The ONE general path is PairDriven (broadphase ->
// narrowphase manifolds -> EmitPairDrivenRows), which carries no slot template.

// H1 (general contact pipeline Phase 0): the general heightfield collidable's
// grid descriptor POD, mirroring Newton's HeightfieldData (utils/heightfield.py).
// One per cooked heightfield. The flat height grid itself rides the `heights[]`
// Model field (row-major, h[r*ncol + c]); this POD locates + scales it. DATA POD
// ONLY in Phase 0 — no cook fills it (H2, Phase 2) and no contact kernel reads
// it (H3, Phase 2). Kept a trivially-copyable aggregate so a future cook can
// stage it into a device record alongside the height grid.
struct HeightfieldData {
    math::Vec3 origin{};         // world-space corner (cell (0,0) base) of the grid.
    float      cell_size = 0.0f; // uniform XY cell spacing.
    uint32_t   nrow = 0u;        // grid rows (Y).
    uint32_t   ncol = 0u;        // grid cols (X).
    float      min_z = 0.0f;     // height-grid min (for the big finite AABB).
    float      max_z = 0.0f;     // height-grid max.
    uint32_t   data_offset = 0u; // base index into the flat `heights[]` field.
};

class Model {
public:
    constraint::SolverOptions solver_options;
    Model() = default;
    std::vector<sensor::StateSensorDesc> state_sensors;

    // -- cook-product host tables (filled by CookToModel) -------------------
    ModelCapacities                 capacities;
    ModelArticulation               articulation;       // the per-env template.
    std::vector<ModelShape>         shapes;             // cooked shape rows (per env).
    std::vector<ModelMaterialBucket> material_buckets;  // (num_buckets) bucket table.
    std::vector<std::string> contact_profile_shapes;
    std::vector<uint32_t>           body_material_bucket;  // per body-row bucket index.
    std::vector<MpmMaterial>        mpm_materials;      // (mpm_material_count) MPM table.

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
    float overlap_recovery_velocity = 0.0f; // speculative overlap push-out bound, m/s (0 => none)

    // -- particle cook product: vertex-block cloth/cable and MLS-MPM material points.
    // MpmVbd assigns [0, n_mpm) to grid transfer and [n_mpm, P) to the vertex blocks.
    enum class ParticleMode : uint8_t { None = 0, Vbd = 1, Mpm = 5, MpmVbd = 6 };
    struct ModelParticles {
        ParticleMode mode = ParticleMode::None;
        // Single-env initial state, replicated env-major by SeedInitialState; initial_pos is
        // also the rest shape that self contact excludes neighbours by.
        std::vector<math::Vec3> initial_pos;
        std::vector<math::Vec3> initial_vel;
        std::vector<float>      inv_mass;
        std::vector<collision::MeshSurfaceInfo> surface_info;
        std::vector<uint32_t> surface_triangles;
        std::vector<collision::MeshBvhNode> surface_tree;
        std::vector<collision::MeshEdgeInfo> surface_edge_info;
        std::vector<collision::MeshEdge> surface_edges;
        std::vector<collision::MeshBvhNode> surface_edge_tree;
        std::vector<uint32_t> surface_triangle_edges;
        std::vector<uint32_t> surface_triangle_vertex_owner;
        std::vector<float> surface_thickness;
        std::vector<float> surface_self_thickness;
        std::vector<uint32_t> surface_self_contact;   // 1 where the surface meets particle surfaces
        // Sorted rest-edge neighbours per particle, within twice (self contact distance + longest edge).
        // Rest-close features among these neighbours do not generate self contacts.
        std::vector<uint32_t> surface_neighbour_offsets;
        std::vector<uint32_t> surface_neighbours;
        // Per-particle contact half thickness against bodies and against particle surfaces, and friction;
        // each surface thickness is the maximum over its vertices.
        std::vector<float> half_thickness;
        std::vector<float> self_half_thickness;
        std::vector<float> friction;
        // MLS-MPM per-particle init (single-env template; replicated env-major).
        // F seeded identity, vol0 from the sampling lattice, material_id indexes
        // mpm_materials. Empty for a non-MPM cook (C is the arena zero default).
        std::vector<float>      initial_F;           // 9 floats/particle (identity)
        std::vector<float>      initial_C;
        std::vector<float>      initial_plastic_F, initial_equivalent_plastic_strain;
        std::vector<float>      initial_vol0;         // 1/particle
        std::vector<uint32_t>   initial_material_id;  // 1/particle
        // Cloth AERODYNAMIC-DRAG element templates (single-env; one per surface
        // triangle). aero_tri_verts = 3 particle indices / triangle, aero_tri_area
        // = rest-space area / triangle. Empty for a drag-free cook.
        std::vector<uint32_t>   aero_tri_verts;     // 3 / triangle
        std::vector<float>      aero_tri_area;      // 1 / triangle
        std::vector<uint32_t>   aero_particle_offset;
        std::vector<uint32_t>   aero_particle_count;
        std::vector<uint32_t>   aero_incident_tri;
        // Vertex-block elastic elements and static color classes (see vertex_block.hpp).
        std::vector<VbdElement> vbd_elements;
        std::vector<uint32_t>   vbd_incidence_offsets;
        std::vector<uint32_t>   vbd_incidence;
        std::vector<uint32_t>   vbd_color_vertices;
        std::vector<uint32_t>   vbd_color_segments;  // (first, count) per color
        // Coarse levels (see vertex_block_hierarchy.hpp): node range per level, (node, weight) parents
        // per level and vertex with ~0u unused, and each node's (vertex, weight) children.
        std::vector<uint32_t>   vbd_coarse_level_nodes;
        std::vector<uint32_t>   vbd_coarse_parents;
        std::vector<float>      vbd_coarse_weights;
        std::vector<uint32_t>   vbd_coarse_child_offsets;
        std::vector<uint32_t>   vbd_coarse_children;
        std::vector<float>      vbd_coarse_child_weights;
        // Anisotropic air-drag coefficients (lumped 0.5*rho*Cn, 0.5*rho*Ct). Both
        // default 0 -> the drag op is inert (a drag-free world stays byte-identical).
        // Cn >> Ct (normal-dominant) is what destabilizes a flat falling sheet into
        // flutter. drag_max_dv caps the per-step impulse magnitude (0 == uncapped).
        float    aero_drag_normal    = 0.0f;
        float    aero_drag_tangent   = 0.0f;
        float    aero_drag_max_dv    = 0.0f;
        // MpmVbd divides grid-owned [0, n_mpm_particles) from row-coupled particles.
        uint32_t n_mpm_particles = 0u;
        // Environment-private MPM grid geometry supplies the shared MpmParams.
        math::Vec3 mpm_grid_min{0.0f, 0.0f, 0.0f};
        uint32_t   mpm_grid_dims[3] = {0u, 0u, 0u};
        float      mpm_cell_size = 0.0f;
        // Preferred common substeps; the pipeline also honors the solver's request.
        uint32_t   mpm_substeps = 1u;
        // The floor constrains material-point gaps through their grid velocity interpolation.
        math::Vec3 mpm_floor_normal{0.0f, 0.0f, 1.0f};
        float      mpm_floor_d = 0.0f;
        float      mpm_floor_friction = 0.4f;
        // Finite-mass contact uses the material point's geometric gap and Coulomb friction.
        // The query margin expands detection without changing the physical surface.
        float      mpm_body_friction = 0.4f;
        float      mpm_body_band = 0.0f;        // Query margin in metres; 0 defaults to dx.
        // Free-fall diagnostic: disable ONLY the dynamic-body BC (the static-plane
        // BC stays on). A test sets it to prove the held-up state is BC-caused.
        bool       mpm_bite_disable_dynamic_bc = false;
        // Body<->particle contact diameter: a particle is a sphere of d_min/2 (0 => off).
        float    pp_contact_d_min      = 0.0f;
    };
    ModelParticles particles;

    // The low particle slice owned by grid transfer, excluded from particle contact rows.
    uint32_t MpmParticlesPerEnv() const;

    // -- M4: contact family + union (CSR compliant) tables --------------------
    // L1-b: the FUSED runtime path is deleted; PairDriven is the general default.
    // (The FusedFoot enum value is retained but dead until the L1-d enum collapse.)
    ContactFamily contact_family = ContactFamily::PairDriven;
    // Drive mode the articulation pipeline runs. OSC (4) uses a dedicated
    // free-dynamics/CRBA/task-drive schedule; the other values retain the
    // affine position/torque presets.
    uint32_t drive_mode = 0;
    // Articulation-local task link selected by nuka_world_desc_t::osc_task_link.
    uint32_t osc_task_link = 0;
    std::vector<float>     hull_verts;    // shared collision vertices, xyz packed
    std::vector<collision::MeshSurfaceInfo> mesh_surface_info;
    std::vector<uint32_t> mesh_triangles;
    std::vector<collision::MeshBvhNode> mesh_bvh_nodes;
    std::vector<collision::MeshEdgeInfo> mesh_edge_info;
    std::vector<collision::MeshEdge> mesh_edges;
    std::vector<collision::MeshBvhNode> mesh_edge_nodes;
    std::vector<uint64_t> mesh_vertex_sources;
    std::vector<uint64_t> mesh_edge_sources;
    std::vector<uint32_t> mesh_triangle_edges;
    std::vector<uint32_t> mesh_triangle_vertex_owner;
    // The ONE general path's contact-compliance defaults (solref/solimp). The
    // PairDriven row emitter (EmitPairDrivenRows) feeds these to ComputeCompliantRow.
    // (L1-c renamed these from union_solref/union_solimp; the values are unchanged.)
    float contact_solref[2] = {0.02f, 1.0f};
    float contact_solimp[5] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    // Per-env movable rigid body initial state (template; replicated env-major).
    struct BodyInit {
        math::Transform pose = math::Transform::Identity();
        math::Vec3 linear_velocity{};
        math::Vec3 angular_velocity{};
        float      inv_mass = 0.0f;
        math::Vec3 inv_inertia{};
        math::Transform inertial_frame = math::Transform::Identity();
    };
    std::vector<BodyInit> body_init;
    // The flat-DOF -> (template-local link, base component) maps (the legacy
    // DofIndexOf prefix-sum inverse; component == ~0u for scalar joints). Cooked
    // by the union cook; staged into the dof_to_link/dof_to_component fields.
    std::vector<uint32_t> dof_to_link;
    std::vector<uint32_t> dof_to_component;

    // Convex hull geometry + SDF grids are referenced by ModelShape indices and
    // packed into the .nka by the cooker; the device upload of those large
    // assets is the collision milestone's (M5) business, not M3a's.

    // Filter policy (plan §3.3): cross-env collision flag + excluded pairs.
    // M3a carried the flag; M5 carries the SORTED (lo<<32|hi) device exclude
    // list (staged into the excluded_pairs field) for the pair-driven query.
    bool filter_cross_env = false;
    std::vector<uint64_t> excluded_pairs;   // SORTED ascending canonical keys.
    // Authored contact pairs: SORTED canonical row keys, each with its contact profile.
    std::vector<uint64_t> contact_pair_keys;
    std::vector<uint32_t> contact_pair_profiles;

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
        // R1 (general contact pipeline Phase 0): the shape->body indirection +
        // collision group. body_id resolves the collidable to its owning body
        // row (-1 == static: ground plane / heightfield, no reaction side).
        // group is the signed collision-group filter key (Newton test_group_pair).
        // APPENDED after the original 6 members so the staged lanes 0..7 are
        // byte-identical; body_id/group pack into the new lanes 8/9. Default
        // body_id == -1 (static) is OVERRIDDEN by the cook for real body rows.
        int32_t    body_id = -1;         // owning body row, or -1 == static.
        uint32_t   group = 0;            // signed collision-group filter key.
        // L-RECON-D (general contact pipeline): the per-shape slice into the
        // concatenated `hull_verts` pool for a ConvexHull/TriMesh row, so the
        // cvx narrowphase uses THIS shape's verts (not one global hull). A
        // non-hull shape (sphere/box/capsule/plane/heightfield) leaves count 0.
        // APPENDED after body_id/group so the staged lanes 0..9 stay byte-
        // identical; offset/count pack into the new lanes 10/11. count == 0 is
        // a NO-OP for a hull-free scene (go2: every row keeps the default 0).
        uint32_t   hull_vert_offset = 0; // base vertex index into hull_verts/3.
        uint32_t   hull_vert_count  = 0; // vertex count (0 == not a hull row).
        uint32_t   contact_profile_index = 0; // ContactProfileV1 table row.
        uint8_t    mesh_contact_mode = 0u;
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

    // R3 (general contact pipeline Phase 0): the shape->body inverse tables (the
    // S5 assembly-seam input). TEMPLATE-local, length == bodies_per_env when
    // populated (a cook with an articulation fills them; empty otherwise ->
    // StageModelField leaves the section ~0u-default-free... actually 0-filled,
    // see model.cpp where the staging defaults unset rows to ~0u). For each body
    // row b owned by template link l:  body_to_link[b] = l, body_to_articulation
    // [b] = the LOCAL articulation index of l. A free-rigid / static body row =
    // ~0u in both. Staged into the body_to_link / body_to_articulation fields
    // (tiled env-major, template-local values). INERT in Phase 0.
    std::vector<uint32_t>    body_to_link;          // template-local link, or ~0u.
    std::vector<uint32_t>    body_to_articulation;  // template-local artic, or ~0u.

    // Multi-collidable proxies: a body row that is an EXTRA collision shape of an
    // owner (an articulation link, or a free-rigid body). The pose source is
    // body_collidable_link[row] (template-local link) OR body_collidable_body[row]
    // (template-local body row); both ~0u when the row is NOT a proxy.
    // body_collidable_local[row] = the shape's owner-local offset. SyncLinkBodyPose
    // poses body_pose[row] = owner_pose o local. All ~0u for a single-collidable
    // scene -> byte-identical.
    std::vector<uint32_t>       body_collidable_link;   // template-local link, or ~0u.
    std::vector<math::Transform> body_collidable_local; // shape offset in the owner frame.
    std::vector<uint32_t>       body_collidable_body;   // template-local body row, or ~0u.

    // H1 (general contact pipeline Phase 0): cooked heightfield collidables + the
    // flat height grid that backs the `heights[]` field. EMPTY for every current
    // scene (the cook fill is H2, Phase 2) -> max_heightfield_cells == 0 ->
    // the heights field is a zero-byte segment (byte-inert). The HeightfieldData
    // POD locates each grid into the flat `heightfield_heights` pool.
    std::vector<HeightfieldData> heightfields;          // cooked heightfield descriptors.
    std::vector<float>           heightfield_heights;   // flat row-major grid pool.

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
    phi::Status ValidateTopology(std::string* reason = nullptr) const;
    phi::Status ValidateVertexBlocks(std::string* reason = nullptr) const;
    phi::Status ValidateVertexBlockLevels(std::string* reason = nullptr) const;
    void BuildAeroAdjacency();

    // The packed device buffer (null until UploadTo). Exposed for World teardown.
    phi::Buffer* DeviceBuffer() const { return device_buffer_; }
    phi::Status DownloadFieldStatus(FieldId id, void* destination, uint64_t bytes,
                                    uint64_t byte_offset = 0u) const;

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
