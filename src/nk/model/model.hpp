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
    uint32_t max_rows_per_env     = 0;  // row-slot capacity / env.
    uint32_t max_hull_verts       = 0;  // convex-hull vertex pool capacity (global).
    uint32_t particles_per_env    = 0;  // XPBD/PBF particle count / env.
    uint32_t dist_cons_per_env    = 0;  // XPBD distance-constraint count / env.
    uint32_t bend_cons_per_env    = 0;  // XPBD bend-constraint count / env.
    uint32_t vol_cons_per_env     = 0;  // XPBD volume-constraint count / env.
    // M9 T11 shape-match (XPBD id 9) capacities: per-env cluster slot count +
    // the total flat cluster-MEMBER pool size (sum_c n_c). 0 == no shape-match.
    uint32_t shape_match_slots_per_env   = 0;  // XPBD shape-match cluster count / env.
    uint32_t shape_match_members_per_env = 0;  // flat cluster-member pool / env.
    // Graph-coloring color counts per XPBD family (single-env template; a color
    // is an independent constraint set sharing no particle). Built by
    // nk::XpbdColoring; size the per-family color-segment tables (pairs/color).
    uint32_t xpbd_dist_colors = 0;
    uint32_t xpbd_bend_colors = 0;
    uint32_t xpbd_vol_colors  = 0;
    uint32_t xpbd_sm_colors   = 0;
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

    // Byte size of the grid_sort_scratch arena field (the ParticleGridBuild cub
    // sort/scan temp + out buffers; sized at World construct, 0 == no particles).
    uint64_t grid_sort_scratch_bytes = 0;

    // MLS-MPM background grid node count PER ENV (the cooked grid dims product; 0
    // for a non-MPM world). Sizes the grid_mass/momentum/velocity/force fields.
    uint32_t mpm_grid_nodes_per_env = 0;
    // Byte size of the mpm_sort_scratch field (the P2G deterministic-gather cub
    // sort temp + out buffers; sized at World construct; 0 == no MPM particles).
    uint64_t mpm_grid_sort_scratch_bytes = 0;
    // MLS-MPM material-table row count (indexed by particle_material_id); 0 for a
    // non-MPM world -> zero-byte mpm_material_table segment.
    uint32_t mpm_material_count = 0;

    // H1 (general contact pipeline Phase 0) — total cooked heightfield-grid cells
    // (sum over all heightfield collidables of nrow*ncol). Sizes the GLOBAL
    // `heights[]` Model field. 0 for every current scene (the cook fill is H2,
    // Phase 2) -> the heights segment is zero bytes (byte-inert).
    uint32_t max_heightfield_cells = 0;

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

// Named lanes of a physics-material bucket row (the per-shape contact contract).
// Positional indices into ModelMaterialBucket::values; kBucketSlotCount is the
// row width (mat_buckets device field = num_buckets x kBucketSlotCount, plan §3.3).
enum BucketSlot : uint32_t {
    kBucketStaticMu    = 0u,  // static friction μs
    kBucketDynamicMu   = 1u,  // dynamic friction μd
    kBucketRestitution = 2u,  // restitution (cooked from the authored material)
    kBucketTimeconst   = 3u,  // solref[0] (contact timeconst)
    kBucketDampratio   = 4u,  // solref[1] (contact dampratio)
    kBucketDensity     = 5u,  // material density
    kBucketMargin      = 6u,  // contact margin
    kBucketReserved    = 7u,  // reserved lane
    kBucketSlotCount   = 8u,  // row width (== values[] extent)
};

// A physics-material bucket row (kBucketSlotCount floats; lanes named by
// BucketSlot). The mat_buckets device field is num_buckets x kBucketSlotCount.
struct ModelMaterialBucket {
    static constexpr uint32_t kValueCount = kBucketSlotCount;  // floats per row.
    float values[kValueCount] = {0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 1000.0f, 0.005f, 0.0f};
};

// One MLS-MPM material: elastic moduli + Drucker-Prager params + weakly-compressible
// fluid params, indexed by the per-particle material id. model_kind selects which
// the constitutive branch reads; fluid (3) reads bulk_modulus/tait_gamma/viscosity.
struct MpmMaterial {
    static constexpr uint32_t kValueCount = 9u;  // f32 count of this POD (table stride).
    float youngs = 0.0f, poisson = 0.0f, density = 0.0f;
    float dp_friction = 0.0f, dp_cohesion = 0.0f;
    // 0 = fixed-corotated elastic, 1 = Drucker-Prager (reserved), 2 = Neo-Hookean,
    // 3 = weakly-compressible fluid (Tait EOS).
    float model_kind = 0.0f;
    float bulk_modulus = 0.0f, tait_gamma = 0.0f, viscosity = 0.0f;
};

// ---------------------------------------------------------------------------
// M4: the UNION (CSR compliant) contact family — Model-side authoring tables.
// ---------------------------------------------------------------------------

// Which contact pipeline this Model runs through the AssembleRows /
// SolveRowsBlockIsland ops:
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
    Model() = default;

    // -- cook-product host tables (filled by CookToModel) -------------------
    ModelCapacities                 capacities;
    ModelArticulation               articulation;       // the per-env template.
    std::vector<ModelShape>         shapes;             // cooked shape rows (per env).
    std::vector<ModelMaterialBucket> material_buckets;  // (num_buckets) bucket table.
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

    // -- M6: particle (XPBD soft + PBF fluid) cook product --------------------
    // The XPBD constraint templates (dist/bend/vol, owner:model) are staged by
    // UploadTo; the particle initial state (pos/prev/vel/inv_mass) is seeded by
    // World::SeedInitialState. PBF params are resolved into the ParticleGridBuild
    // + Pbf* op params by Pipeline::Build. A scene with no particles leaves these
    // empty and particles_per_env == 0 (Pipeline emits no particle ops).
    // Coupled: particles co-step against rigid/artic bodies through the unified
    // row solve (the kUSlotParticleSphere* contact classes); the internal XPBD
    // soft constraints (if any) still run. None/Xpbd/Pbf are the standalone modes.
    // M9 T11 SoftFluid: ONE Model holds BOTH a soft (XPBD) particle set AND a
    // fluid (PBF) particle set co-resident in a contiguous [soft | fluid] layout
    // with split index n_soft_particles (the fluid occupies [n_soft, P)). The
    // XPBD solve runs over the soft slice (edge-based constraints reference only
    // soft particle indices) and the PBF density/lambda/neighbor solve is SCOPED
    // to the fluid slice. This mirrors the co-step's existing [xpbd | pbf] union
    // (split n_x) 1:1, so the Phase-2 id-10 cross-contact port is near-verbatim
    // (global g < n_soft => soft, else fluid).
    enum class ParticleMode : uint8_t { None = 0, Xpbd = 1, Pbf = 2, Coupled = 3,
                                        SoftFluid = 4, Mpm = 5 };
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
        // MLS-MPM per-particle init (single-env template; replicated env-major).
        // F seeded identity, vol0 from the sampling lattice, material_id indexes
        // mpm_materials. Empty for a non-MPM cook (C is the arena zero default).
        std::vector<float>      initial_F;           // 9 floats/particle (identity)
        std::vector<float>      initial_vol0;         // 1/particle
        std::vector<uint32_t>   initial_material_id;  // 1/particle
        // XPBD constraint templates (single-env; staged + dispatched per env).
        std::vector<uint32_t> dist_a, dist_b;     // distance endpoints
        std::vector<float>    dist_rest, dist_alpha;
        std::vector<uint32_t> bend_particles;     // 4 / bend constraint
        std::vector<math::Vec3> bend_gradients;   // 4 / bend constraint
        std::vector<float>    bend_alpha;
        std::vector<uint32_t> vol_particles;      // 4 / volume constraint
        std::vector<float>    vol_rest6, vol_alpha;
        // M9 T11 XPBD SHAPE-MATCH (id 9) cluster templates (single-env; CSR
        // layout, mirrors the legacy soft-upload flatten 1:1). Per-cluster:
        // sm_cluster_offset/size into the flat member pool, sm_stiffness goal-pull
        // fraction, sm_rest_centroid c0. Per-member (flat sum_c n_c pool):
        // sm_particles index, sm_rest_q = x_i^0 - c0, sm_mass weight m_i.
        std::vector<uint32_t>   sm_cluster_offset;  // per cluster
        std::vector<uint32_t>   sm_cluster_size;    // per cluster (n_c)
        std::vector<float>      sm_stiffness;       // per cluster (s in [0,1])
        std::vector<math::Vec3> sm_rest_centroid;   // per cluster (c0)
        std::vector<uint32_t>   sm_particles;       // flat pool (sum n_c)
        std::vector<math::Vec3> sm_rest_q;          // flat pool (q_i = x_i^0 - c0)
        std::vector<float>      sm_mass;            // flat pool (m_i weight)
        uint16_t xpbd_iters = 1;
        // M9 T11 two-system [soft | fluid] split index: the fluid particles
        // occupy [n_soft_particles, particles_per_env). For SoftFluid mode the
        // PBF density/lambda/neighbor ops are scoped to the fluid slice; the XPBD
        // constraints (edge-based) reference only the soft slice. For the
        // single-system modes (Xpbd/Pbf/Coupled) it is 0 (Xpbd) or unused.
        uint32_t n_soft_particles = 0u;
        // Per-system body<->particle contact mu: soft finite (a foot grips cloth),
        // fluid ~0 (a foot slides); mixed with the body side by solmix=max.
        float    soft_friction  = 0.6f;
        float    fluid_friction = 0.0f;
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
        // MLS-MPM env-private background grid descriptor (its OWN fields, NOT the PBF
        // domain above). The grid-transfer coupling provider builds MpmStepParams from these.
        math::Vec3 mpm_grid_min{0.0f, 0.0f, 0.0f};
        uint32_t   mpm_grid_dims[3] = {0u, 0u, 0u};
        float      mpm_cell_size = 0.0f;
        // MLS-MPM internal explicit substeps per World.Step (CFL headroom for a stiff
        // medium); the cook sets it from the material/dt. 0 => 1 substep.
        uint32_t   mpm_substeps = 1u;
        // MLS-MPM static floor plane (z-up: n=(0,0,1), d=floor height). The grid BC
        // projects node velocity against this plane (no-penetration + Coulomb mu).
        math::Vec3 mpm_floor_normal{0.0f, 0.0f, 1.0f};
        float      mpm_floor_d = 0.0f;
        float      mpm_floor_friction = 0.4f;
        // MLS-MPM dynamic-body grid BC: a cooked body's SDF is rasterized onto the
        // grid and the node velocity projected onto its surface velocity (Coulomb
        // mpm_body_friction, |phi| band mpm_body_band cells). The grid provider
        // turns the BC on whenever a collidable body co-resides with the medium.
        float      mpm_body_friction = 0.4f;
        float      mpm_body_band = 0.0f;        // 0 => the provider defaults it to dx.
        // Free-fall diagnostic: disable ONLY the dynamic-body BC (the static-plane
        // BC stays on). A test sets it to prove the held-up state is BC-caused.
        bool       mpm_bite_disable_dynamic_bc = false;
        // M9 T11 Phase 2 — id-10 CROSS-SYSTEM particle-particle CONTACT params
        // (the op-ified cross-system particle-particle co-step). The
        // class-blind unilateral non-penetration co-step runs AFTER finalize over
        // the FULL [soft | fluid] union; ONLY the SoftFluid mode emits it. d_min
        // == 2*contact_radius (uniform radius); <= 0 disables the op. The grid
        // neighbor list (built over query_radius) must cover d_min, so the cook
        // sets query_radius/cell_size >= d_min for a SoftFluid scene.
        float    pp_contact_d_min      = 0.0f;   // 2*contact_radius (0 => off)
        float    pp_contact_compliance = 0.0f;   // XPBD alpha (0 => rigid)
        uint32_t pp_contact_iters      = 1u;     // Jacobi gather+apply sweeps
    };
    ModelParticles particles;

    // -- M4: contact family + union (CSR compliant) tables --------------------
    // L1-b: the FUSED runtime path is deleted; PairDriven is the general default.
    // (The FusedFoot enum value is retained but dead until the L1-d enum collapse.)
    ContactFamily contact_family = ContactFamily::PairDriven;
    // Drive mode the ApplyDrives op runs (0 = position PD hold drive, the M3
    // batched articulated path; 1 = direct torque drive, the union world's
    // LaunchApplyTorqueDriveKernels path — drive_target carries the torque).
    uint32_t drive_mode = 0;
    std::vector<float>     hull_verts;    // mesh-local hull verts (xyz packed).
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
    // XPBD graph-coloring color-segment tables (filled by nk::XpbdColoring BEFORE
    // UploadTo; staged into the *_color_segments Model fields). Each is a flat
    // {offset, count} u32 PAIR per color over the single-env constraint template.
    std::vector<uint32_t> dist_color_segments;
    std::vector<uint32_t> bend_color_segments;
    std::vector<uint32_t> vol_color_segments;
    std::vector<uint32_t> sm_color_segments;
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
