#pragma once
// Closed physics operators and their trivially copyable parameter blocks.
// Operator numbering and resource contracts share one backend-independent catalog.

#include <cstdint>
#include "phi/articulation_contract.hpp"
#include "collision/mesh_surface_types.hpp"
#include "sensor/observation_types.hpp"
#include "sensor/state_types.hpp"
#include "nk/readout/energy_ledger.hpp"
#include "nk/readout/physics_diagnostics.hpp"

namespace nuka::phi {

// ---------------------------------------------------------------------------
// Cooked-table row strides — host-safe single source of truth shared by the
// host Model builder (model.cpp staging / ElementCount) and the device readers
// (prims_types.cuh LoadPrimShape, narrowphase_sdf.cu LoadSdfGrid). Naming the
// strides here (a pure-C++ header includable by both the host .cpp and the .cu
// readers) replaces the bare literals so a lane addition bumps ONE constant.
// ---------------------------------------------------------------------------
// shape_table packed f32 / body row {kind, p0..p3, contype, conaffinity,
// sdf_grid, body_id, group, hull_vert_offset, hull_vert_count, contact_profile_index}.
inline constexpr uint32_t kShapeTableRowStride = 13u;
// SDF header packed f32 / grid {origin.xyz, voxel_size, dims.xyz, cell_offset}.
inline constexpr uint32_t kSdfHeaderStride = 8u;

// ---------------------------------------------------------------------------
// env_status diagnostic bits (the per-env readout field, flags:[readout]). Ops
// that can silently lose work under a cooked-capacity miss OR their bit here so
// the host surfaces it post-step (never a silent drop / clamp). Each writer
// clears ONLY its own bit at op start (order-independent across ops in a step).
// Per-articulation / scalar diagnostics use env slot 0 (always present).
// ---------------------------------------------------------------------------
inline constexpr uint32_t kEnvStatusPairOverflow     = 1u << 0;  // candidate_pairs dropped
inline constexpr uint32_t kEnvStatusDofOverflow      = 1u << 2;  // artic dof > max_dof (CRBA)
inline constexpr uint32_t kEnvStatusMpmGridEscape    = 1u << 3;  // MPM open-face escape / invalid F
// A collidable has no supported surface representation and is skipped by MPM.
// The bit name is retained for compatibility; analytic primitives are supported.
inline constexpr uint32_t kEnvStatusMpmOneWayBody    = 1u << 4;
inline constexpr uint32_t kEnvStatusGyroFailure      = 1u << 5;
// Invalid contact ownership or endpoint indices stay latched until world reset.
inline constexpr uint32_t kEnvStatusInvalidEndpoint  = 1u << 6;
// Missing or invalid sampled contact geometry stays latched until world reset.
inline constexpr uint32_t kEnvStatusContactGeometryUnavailable = 1u << 7;
inline constexpr uint32_t kEnvStatusConstitutiveFailure = 1u << 8;
inline constexpr uint32_t kEnvStatusGridContactOverflow = 1u << 9;
inline constexpr uint32_t kEnvStatusControlFailure = 1u << 10;
inline constexpr uint32_t kEnvStatusSensorQueueOverflow = 1u << 11;
inline constexpr uint32_t kEnvStatusDatFailure = 1u << 12;
inline constexpr uint32_t kEnvStatusSolverFailure = 1u << 13;
inline constexpr uint32_t kBodyGyroNotConverged      = 1u;
inline constexpr uint32_t kBodyGyroInvalidInput      = 2u;

// ---------------------------------------------------------------------------
// NkOp — the closed op set. uint16_t backing so an op id fits a single field
// and the enum can be stored in compact tables.
// ---------------------------------------------------------------------------
enum class NkOp : uint16_t {
#define NUKA_OP(name, read, write) name,
#include "phi/op_catalog.inc"
#undef NUKA_OP
    Count
};

// ---------------------------------------------------------------------------
// Per-op parameter PODs.
//
// All are trivially-copyable aggregates. `reserved` fields keep the structs
// non-empty and stable in size while their real fields are designed in later
// op params; they MUST stay POD.
// ---------------------------------------------------------------------------

// --- articulated-body dynamics ------------------------------------------
// Note: ops carry the launch-geometry counts (total_link_count /
// articulation_count / max_dof) in their params because ModelView/DataView are
// pure pointer aggregates; the Pipeline fills them from the Model capacities.
struct ApplyDrivesParams {
    float dt;
    uint32_t total_link_count;
    uint32_t defer_velocity_damping;
    uint32_t mode;
    uint32_t links_per_env;
};

struct ApplyDynamicsDrivesParams {
    uint32_t max_dof;
    uint32_t articulation_count;
    uint32_t total_link_count;
    uint32_t task_link;
    uint32_t mode;
    uint32_t links_per_env;
    float gravity[3];
};

struct ReadoutDrivesParams {
    float dt;
    uint32_t total_link_count;
    uint32_t links_per_env;
    uint32_t rows_per_env;
    uint32_t first_drive_row;
};

struct ArticulationDynamicsParams {
    float dt;
    uint32_t max_dof;
    uint32_t integrate_velocity = 1u;
    uint32_t fold_drive_damping = 0u;
    uint32_t reuse_mass = 0u;
    float    gravity[3];   // world-frame acceleration in m/s^2
    uint32_t articulation_count;
    uint32_t total_link_count;
    uint32_t apply_link_wrench = 0u;
};

struct IntegrateVelocityParams {
    float    dt;
    float    gravity[3];
    uint32_t total_link_count;
    uint32_t articulation_count;
    // Free-body velocity integration consumes world-frame COM force and torque.
    uint32_t total_body_count;
    uint32_t clear_body_forces = 1u;
};

struct SnapshotStepVelocityParams {
    uint32_t env_count;
    uint32_t articulation_count;
    uint32_t max_dof;
    uint32_t base_link_count;
    uint32_t total_body_count;
    uint32_t total_particle_count;
};

inline constexpr uint32_t kAccumulateMpmImpulse = 1u << 0;
inline constexpr uint32_t kAccumulateOutputs = 1u << 1;
inline constexpr uint32_t kAccumulateLinkWrench = 1u << 2;
inline constexpr uint32_t kAccumulateJointLimit = 1u << 3;
inline constexpr uint32_t kAccumulateMpmOutput = 1u << 4;
inline constexpr uint32_t kFinalizeBodyForces = 1u << 5;
inline constexpr uint32_t kFinalizeLinkWrench = 1u << 6;

struct AccumulateStepParams {
    uint32_t env_count = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t links_per_env = 0u;
    uint32_t artics_per_env = 0u;
    uint32_t flags = 0u;
    uint32_t first = 0u;
    uint32_t last = 0u;
    float substep_dt = 0.0f;
    float inv_outer_dt = 0.0f;
};

struct FkWorldPosesParams {
    uint32_t articulation_count;
    uint32_t total_link_count;
    uint32_t articulations_per_env;
    uint32_t selected_env_count;  // 0 = all; otherwise index reset_env_ids
};

struct FkLinkVelocitiesParams {
    uint32_t articulation_count;
    uint32_t total_link_count;
};

struct IntegratePositionParams {
    float    dt;
    uint32_t total_link_count;
    uint32_t articulation_count;
    // Free bodies drift in the principal inertia frame after all physical impulses.
    uint32_t total_body_count;
    // Pseudo rotation changes geometry while preserving physical world angular momentum.
    uint32_t pos_pass;
    uint32_t env_count = 1u;
    // Nonzero when coupled joints take their root's coordinate after the advance.
    uint32_t mimic_couplings = 0u;
};

// Runs after the step mass is factored: each coupled DOF follows its root, the mass and
// factor describe Z^T M Z, and the step velocity is projected onto Z.
struct MimicReduceParams {
    uint32_t max_dof;
    uint32_t articulation_count;
    uint32_t total_link_count;
};

// --- broadphase / spatial acceleration ----------------------------------
// the broadphase ops (BuildAabbs/LbvhBuild/LbvhQueryPairs) and the SDF
// narrowphase EARLY-EXIT unless family == kContactFamilyPairDriven (the union
// slot-template and fused-foot paths do their own detection and never read the
// pair stream). The pair-driven ops carry their launch geometry in params (the
// views are pure pointer aggregates).
struct BuildAabbsParams {
    float    margin;            // AABB inflation margin
    uint32_t family;            // kContactFamily* (PairDriven => build)
    uint32_t env_count;
    uint32_t bodies_per_env;    // collidable body rows / env (shape_table)
};

struct LbvhBuildParams {
    uint32_t family;            // kContactFamily* (PairDriven => build)
    uint32_t env_count;
    uint32_t bodies_per_env;
    uint64_t workspace_bytes = 0u;
};

struct LbvhQueryPairsParams {
    uint32_t max_pairs;         // per-env capacity of the output pair stream
    uint32_t family;            // kContactFamily* (PairDriven => query)
    uint32_t env_count;
    uint32_t bodies_per_env;
    uint32_t max_contacts_per_env;  // candidate_pairs slot stride / env
    uint32_t filter_cross_env;  // 1 => drop pairs spanning envs (env-major)
    uint32_t excluded_count;    // sorted exclude-list length (excluded_pairs)
    // Rigid emission cap (<= the slot stride): body<->body pairs fill only
    // [0, rigid_slot_cap); the body<->particle narrowphase owns [rigid_slot_cap,
    // stride). Equals the stride when no particles -> byte-identical.
    uint32_t rigid_slot_cap;
    uint64_t workspace_bytes;
};

// Byte size of the pre-allocated pair_sort_scratch the LbvhQueryPairs op draws its
// canonicalizing cub radix sort (u64 key -> u32 slot perm) + per-env prefix + pair
// snapshot from. Host-callable (defined in broadphase.cu) so the World sizes the
// field BEFORE allocation -> no mid-capture cudaMalloc. 0 for empty inputs.
uint64_t PairSortScratchBytes(uint32_t total_sort_slots, uint32_t env_count);
uint64_t LbvhSortScratchBytes(uint32_t env_count, uint32_t bodies_per_env);
uint64_t ContactCacheScratchBytes(uint32_t point_count, uint32_t env_count);
uint64_t OgcOrderScratchBytes(uint32_t slot_capacity, uint32_t env_count);
uint64_t ContactIndexScratchBytes(uint32_t row_count, uint32_t env_count);

// All MPM operations share one physical interval and environment-private grid.
// Prediction owns grid initialization; commit alone advances particles and history.
struct ParticleSurfacesParams {
    uint32_t env_count;
    uint32_t particles_per_env;
    uint32_t surfaces_per_env;
    uint32_t triangles_per_env;
    uint32_t nodes_per_env;
    uint32_t edge_nodes_per_env;
    uint32_t position_source = 0u;
};

// Share of a DAT query radius that two sides of an undetected pair may jointly travel.
inline constexpr float kDatRelaxation = 0.9f;

struct OgcDetectParams {
    float dt = 0.0f;
    float margin = 0.0f;
    float relaxation = kDatRelaxation;
    uint32_t env_count = 0u;
    uint32_t particles_per_env = 0u;
    uint32_t surfaces_per_env = 0u;
    uint32_t triangles_per_env = 0u;
    uint32_t nodes_per_env = 0u;
    uint32_t edges_per_env = 0u;
    uint32_t edge_nodes_per_env = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t links_per_env = 0u;
    uint32_t articulations_per_env = 0u;
    uint32_t mesh_vertex_sources = 0u;
    uint32_t mesh_edge_sources = 0u;
    uint32_t mesh_vertices = 0u;
    uint32_t mesh_triangles = 0u;
    uint32_t mesh_nodes = 0u;
    uint32_t mesh_edges = 0u;
    uint32_t mesh_edge_nodes = 0u;
    uint32_t excluded_pairs = 0u;
    uint32_t slot_stride = 0u;
    uint32_t slot_base = 0u;
    uint32_t slot_capacity = 0u;
    uint32_t point_endpoints_per_env = 0u;
    uint32_t point_endpoint_terms_per_env = 0u;
    // OGC endpoints follow the MPM endpoints of each env pool.
    uint32_t point_endpoint_first = 0u;
    uint32_t point_endpoint_term_first = 0u;
    uint64_t workspace_bytes = 0u;
};

struct DatTruncateParams {
    float dt = 0.0f;
    float relaxation = kDatRelaxation;
    float margin = 0.0f;
    uint32_t env_count = 0u;
    uint32_t particles_per_env = 0u;
    uint32_t vbd_particle_begin = 0u;
    uint32_t vbd_vertices_per_env = 0u;
    uint32_t surfaces_per_env = 0u;
    uint32_t triangles_per_env = 0u;
    uint32_t nodes_per_env = 0u;
    uint32_t edges_per_env = 0u;
    uint32_t edge_nodes_per_env = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t links_per_env = 0u;
    uint32_t articulations_per_env = 0u;
    uint32_t max_dof = 0u;
    uint32_t mesh_vertices = 0u;
    uint32_t mesh_triangles = 0u;
    uint32_t mesh_nodes = 0u;
    uint32_t mesh_edges = 0u;
    uint32_t mesh_edge_nodes = 0u;
    uint32_t mesh_vertex_sources = 0u;
    uint32_t mesh_edge_sources = 0u;
    uint32_t excluded_pairs = 0u;
    uint32_t slot_stride = 0u;
    uint32_t slot_base = 0u;
    uint32_t slot_capacity = 0u;
};

struct MpmParams {
    uint32_t contact_slot_base;
    uint32_t contact_capacity;
    uint32_t contact_slots_per_env;
    uint32_t full_row_slot_count;
    uint32_t rigid_rows_per_env;
    uint32_t rows_per_env;
    uint32_t particle_count;     // total env-major particles (0 => inert no-op)
    uint32_t particles_per_env;  // per-env stride (for the env-offset cell key)
    // Grid-owned particles occupy [0, mpm_particles_per_env) in each environment.
    // Zero selects the complete per-environment particle range.
    uint32_t mpm_particles_per_env;
    uint32_t env_count;
    uint32_t nodes_per_env;      // per-env grid node count (mpm_grid_nodes_per_env)
    uint32_t grid_dims[3];       // per-env node resolution (nx, ny, nz)
    float    grid_origin[3];     // world-space corner of node (0,0,0)
    float    dx;                 // uniform node spacing
    float    dt;                 // common physics interval
    uint32_t mode;               // kParticleModeMpm or kParticleModeMpmVbd.
    uint32_t substeps;           // zero or one; subdivision belongs to the pipeline
    uint32_t material_count;     // mpm_material_table rows (indexed by particle_material_id)
    float    gravity[3];         // world-frame gravity applied on the grid each substep
    // Static boundaries emit unilateral velocity constraints and external impulse readout.
    float    plane_n[3];
    float    plane_d;
    float    plane_mu;
    // Every supported collidable within the band emits a shared contact block.
    uint32_t dynamic_body_bc;
    // Disables collidable exchange and reaction while retaining the static floor.
    uint32_t bite_disable_dynamic_bc;
    uint32_t bodies_per_env;     // collidable body rows / env (the BC body loop).
    uint32_t sdf_grid_count;     // available SDF descriptors; primitives need none.
    uint32_t sdf_cell_total;
    collision::MeshGeometryCounts mesh_geometry;
    float    body_mu;            // Coulomb friction the body BC clamps the tangent by.
    float    body_band;          // broad query margin; does not change physical separation.
    // Articulation dimensions for endpoint validation and reaction reference points.
    uint32_t artic_count;        // GLOBAL articulations (artics_per_env * env_count).
    uint32_t max_dof;            // per-articulation generalized DOF (the m_inv tile side).
    uint32_t base_link_count;    // links per env (global link = env*base_link_count+tmpl).
    uint32_t artics_per_env;     // co-resident articulations per env (>=1 when artic).
    uint32_t particle_surfaces_per_env;
    uint32_t particle_surface_triangles;
    uint32_t particle_surface_nodes_per_env;
    uint32_t point_endpoints_per_env;
    uint32_t point_endpoint_terms_per_env;
    // Implicit stress cells per env, row blocks at the env row tail; zero transfers stress explicitly.
    uint32_t stress_cells_per_env = 0u;
};

// Workspace bytes for MPM particles, grid nodes, contacts, reaction targets and stress cells.
// Query before state allocation; zero particles require no workspace.
uint64_t MpmSortScratchBytes(uint32_t particle_count, uint32_t node_count,
                             uint64_t collidables_per_env, uint64_t contact_count,
                             uint64_t reaction_target_count, uint64_t stress_cell_count);

// --- narrowphase / contact rows -----------------------------------------
// Contact-family selector shared by the narrowphase / assemble / solve params
// (mirrors nk::ContactFamily; a plain u32 so the POD stays header-light).
// 0 = FusedFoot (DEAD: the articulation foot pipeline RUNTIME was deleted in
// ; the constant is retained, dead, until the enum collapse — no
// op dispatches on it anymore)
// 1 = UnionCsr (union compliant-CSR pipeline)
// 2 = PairDriven (generalized broadphase->narrowphase: BuildAabbs/Lbvh*/
// candidate_pairs -> NarrowphasePrimitives (amf:: analytic set + sphere x
// hull) + NarrowphaseSdf (SAMP x SDF grid). The general default for every
// non-union cooked model. The broadphase + SDF ops EARLY-EXIT for UnionCsr
// so that gate-pinned path stays bit-identical.)
inline constexpr uint32_t kContactFamilyFusedFoot  = 0u;  // dead constant.
inline constexpr uint32_t kContactFamilyUnionCsr   = 1u;
inline constexpr uint32_t kContactFamilyPairDriven = 2u;

struct NarrowphasePrimitivesParams {
    float contact_margin;
    uint8_t max_contacts_per_pair;
    // first batch (foot sphere x ground plane, the production Go2/H1 path):
    float    ground_height;
    uint32_t foot_count;        // active rows of the Model foot_shape table
    uint32_t env_count;
    uint32_t base_link_count;   // links per env (replica stride)
    // union family (kContactFamilyUnionCsr): per-(env x union-slot) analytic
    // detection (foot sphere x plane / finger sphere x hull / body box x plane).
    uint32_t family;            // kContactFamily*
    uint32_t union_slot_count;  // union slots per env (Model union_slots size)
    uint32_t rigid_slot_cap;    // body<->body live cap (<= stride; == stride if no particles)
    uint32_t bodies_per_env;
    uint32_t hull_vert_count;   // live verts of the hull_verts pool
    // particle coupling (the kUSlotParticleSphere* union classes form the
    // particle side's world sphere from particle_pos[env*particles_per_env+link]).
    uint32_t particles_per_env;
    // these two fields drove the (now-deleted) FUSED foot slot re-keying.
    // They are retained in the POD for layout stability (no op reads them anymore;
    // multi-body dog<->dog collision rides the GENERAL PairDriven path). K ==
    // articulations_per_env; max_foot_contacts == the per-artic stride.
    uint32_t articulations_per_env;  // K (dead: FUSED foot slot re-key removed)
    uint32_t max_foot_contacts;      // == kMaxFootContactsPerEnv (dead with FUSED)
    uint32_t max_body_samples;      // largest per-body slice of the surface sample pool
    uint32_t convex_pairs;          // nonzero when some filtered pair needs the convex narrowphase
};

// General contact pipeline (B2): SyncLinkBodyPose. One thread per
// articulation link copies its FK world pose (link_pose[l]) into its owning
// body_pose row (body_pose[link_body[l]]), composing the cooked link_geom_local
// offset so an offset collidable is posed in world space. This is what makes
// articulation links visible to the LBVH (BuildAabbsKernel reads body_pose),
// the load-bearing prerequisite for general body<->body contact. EARLY-EXITS
// unless family == kContactFamilyPairDriven (so the UnionCsr graph is
// byte-untouched before the general path is wired). link_body is the MODEL table (template-local body
// row per link); the kernel adds the per-env body offset for the global body_pose
// index.
struct SyncLinkBodyPoseParams {
    uint32_t family;          // kContactFamily* (PairDriven => run; else early-exit)
    uint32_t env_count;
    uint32_t links_per_env;   // per-env link stride (the kernel grids env*links)
    uint32_t bodies_per_env;  // body rows per env (the per-env body stride)
    uint32_t selected_env_count;  // 0 = all; otherwise index reset_env_ids
};

struct NarrowphaseHeightfieldParams {
    uint32_t family;            // kContactFamily* (PairDriven => run)
    uint32_t env_count;
    uint32_t slot_stride;       // candidate slots per env (== max_contacts_per_env)
    uint32_t rigid_slot_cap;    // body<->body live cap (<= stride; == stride if no particles)
    uint32_t bodies_per_env;
    uint32_t hull_vert_count;   // cooked hull pool (convex-hull convex side)
    float    contact_margin;
    // -- the heightfield descriptor (HeightfieldData mirror) ------------------
    uint32_t has_heightfield;   // 0 == no cooked heightfield (op no-op)
    float    origin_x, origin_y, origin_z;  // local (0,0) corner (descriptor.origin)
    float    cell_size;
    uint32_t nrow, ncol;
    float    min_z, max_z;      // LOCAL z-range (for the prism z + extrude)
    uint32_t data_offset;       // base index into the `heights` field
    uint32_t hf_body_row;       // the heightfield collidable's body row (side b)
};

// Body/artic <-> particle narrowphase. One thread per (env x env-local particle):
// it builds the particle's query AABB (sphere of particle_radius), traverses the
// env's arena LBVH for overlapping collidable bodies (the cross_system_query CSR
// pattern: insertion-sorted, capped at max_candidates, N<2 direct-scan fallback),
// and for each candidate body runs the sphere-vs-shape manifold (the particle is a
// SPHERE of its radius on the ONE path). The manifold is written into the
// particle's RESERVED contact-slot sub-range [slot_base +
// (pi - particle_row_base)*cands_per_particle, ...+cands_per_particle) so the
// body<->particle slots occupy a DETERMINISTIC
// sub-range relative to the racy rigid-rigid slots within each env block; the
// per-particle base is a fixed (non-atomic) function of the particle index, so the
// stream is bit-D1 by construction (no sort). Side A == the particle (global id +
// kUContactSideParticle tag), side B == the body collidable. EARLY-EXITS unless
// family == kContactFamilyPairDriven. A particle exceeding its candidate budget
// ORs kEnvStatusPairOverflow (never a silent drop).
struct NarrowphaseBodyParticleParams {
    uint32_t family;              // kContactFamily* (PairDriven => run)
    uint32_t env_count;
    uint32_t bodies_per_env;      // collidable rows / env (LBVH leaf count)
    uint32_t particles_per_env;   // per-env particle stride
    uint32_t slot_stride;         // candidate slots / env (== max_contacts_per_env)
    uint32_t particle_slot_base;  // first contact slot reserved for particles / env
    uint32_t cands_per_particle;  // reserved contact slots per particle
    float    particle_radius;     // the particle collision radius (sphere on the path)
    float    contact_margin;
    // MpmVbd: the first env-local particle that generates body rows. The MPM slice
    // [0, particle_row_base) couples via the grid, NOT rows; it owns no reserved
    // slots (the cook exempts it from the budget). 0 for every other mode.
    uint32_t particle_row_base;
    uint32_t excluded_particle_begin;
    uint32_t excluded_particle_count;
    // Vertex-block particles take their surface thickness and a motion-bounded query, and skip meshed bodies.
    uint32_t vbd_particle_begin;
    uint32_t vbd_particle_count;
    uint32_t particle_surfaces;
    float    dt;
    // 1 == launch ONE WARP per particle (the giant-hull SupportHull scan runs warp-
    // cooperatively); 0 == one thread per particle (analytic-only collider worlds,
    // no wide hull to split). BOTH paths are byte-identical; set by the cook-time max
    // hull vcount so it is a model property, not a per-scene branch.
    uint32_t warp_per_particle;
    // -- the heightfield descriptor (HeightfieldData mirror, same names/types as
    // NarrowphaseHeightfieldParams) so a sphere particle walks the cooked grid. ---
    uint32_t has_heightfield;   // 0 == no cooked heightfield (the field arm is inert)
    float    origin_x, origin_y, origin_z;  // local (0,0) corner (descriptor.origin)
    float    cell_size;
    uint32_t nrow, ncol;
    float    min_z, max_z;      // LOCAL z-range (for the cell corner heights)
    uint32_t data_offset;       // base index into the `heights` field
    uint32_t sdf_grid_count;    // available SDF descriptors
    uint32_t sdf_cell_total;
    collision::MeshGeometryCounts mesh_geometry;
};

// Sampled surfaces query the opposing analytic/SDF geometry within each pair.
struct NarrowphaseSdfParams {
    float   contact_margin;
    uint8_t max_contacts_per_pair;
    // -- appended launch geometry -----------------------------------
    uint32_t family;            // kContactFamily* (PairDriven => sample)
    uint32_t env_count;
    uint32_t bodies_per_env;
    uint32_t max_contacts_per_env;  // ucontact slot stride / env
    uint32_t rigid_slot_cap;    // body<->body live cap (<= stride; == stride if no particles)
    uint32_t sdf_grid_count;    // available SDF descriptors
    uint32_t sample_point_count; // available surface samples
    uint32_t sdf_cell_total;
    collision::MeshGeometryCounts mesh_geometry;
    uint32_t max_body_samples;  // largest per-body slice of the sample pool
};

// Word count of the pair_sample_chunks field the NarrowphaseSdf op flags sample chunks in.
// Host-callable (defined in narrowphase_sdf.cu) so the World sizes it before allocation.
uint64_t PairSampleChunkWords(const NarrowphaseSdfParams& params);

// Vertex blocks occupy particles [begin, begin + vertices) of each env; vertices == 0 when absent.
struct VertexBlockLayout {
    uint32_t begin = 0u;
    uint32_t vertices = 0u;
    uint32_t dynamic_vertices = 0u;
    uint32_t colors = 0u;
    uint32_t particles_per_env = 0u;
    uint32_t env_count = 0u;
    uint32_t elements = 0u;
    // Coarse levels of the vertex blocks, their nodes and the nodes of the dense last level.
    uint32_t coarse_levels = 0u;
    uint32_t coarse_nodes = 0u;
    uint32_t coarse_dense_nodes = 0u;
};

struct AssembleRowsParams {
    uint32_t grid_nodes_per_env = 0u;
    float    dt;
    uint32_t slot_count;        // total contact slots (env_count * max_contacts)
    uint32_t max_dof;           // chain-Jacobian dof_stride (== dofs_per_env)
    uint32_t env_count;
    uint32_t articulation_count;
    uint32_t total_link_count;
    // union family:
    uint32_t family;            // kContactFamily*
    uint32_t union_slot_count;  // union slots per env
    uint32_t rows_per_env;      // row slots per env (== max_rows_per_env)
    uint32_t contact_rows_per_env; // fixed contact footprint before joint rows
    uint32_t joint_limit_rows_per_env;
    uint32_t joint_friction_rows_per_env;
    uint32_t joint_drive_rows_per_env;
    // Nonzero when articulation rows act on the reduced coordinates MimicReduce prepared.
    uint32_t mimic_couplings_per_env;
    // Rigid slots use their manifold row ranges; the compact tail reserves one point.
    // Equal to union_slot_count when the model has no compact reserve.
    uint32_t full_row_slot_count;
    uint32_t ogc_slot_count = 0u;
    uint32_t bodies_per_env;
    uint32_t base_link_count;   // links per env (replica stride)
    float    solref[2];         // merged contact solref (union family)
    float    solimp[5];         // merged contact solimp (union family)
    uint32_t particles_per_env; // particle coupling (kUSlotParticleSphere*).
    uint32_t num_material_buckets;  // 0 == no authored materials -> model defaults.
    // Cap on the contact normal aref so a deep contact recovers over several steps,
    // not in one fling. +inf default == non-binding (byte-identical). Model property.
    float    baumgarte_max_velocity;
    float    contact_margin;
    // Nonzero when the split-impulse position pass runs; it then removes joint drift and overlap.
    uint32_t position_pass = 0u;
    // Separating speed (m/s) at which speculative rows recover overlap; 0 recovers none.
    float overlap_recovery_velocity = 0.0f;
    uint32_t point_endpoints_per_env = 0u;
    uint32_t point_endpoint_terms_per_env = 0u;
    VertexBlockLayout vertex_blocks{};
    // Nonzero when the solve measures contact residuals against the Coulomb block response.
    uint32_t measure_contact_response = 0u;
};

struct BlockDescentSolveParams {
    float dt;
    uint32_t iterations = 10u;
    uint32_t env_count = 0u;
    uint32_t rows_per_env = 0u;
    uint32_t total_particle_count = 0u;
    uint32_t total_body_count = 0u;
    uint32_t total_grid_count = 0u;
    uint32_t articulation_count = 0u;
    uint32_t max_dof = 0u;
    uint32_t base_link_count = 0u;
    uint32_t contact_rows_per_env = 0u;
    uint32_t distinct_owner_rows_per_env = 0u;
    uint32_t joint_limit_rows_per_env = 0u;
    uint64_t incidence_per_env = 0u;
    uint32_t measure_contact_residual = 0u;
    float penalty_scale = 100.0f;
    uint64_t workspace_words = 0u;
    VertexBlockLayout vertex_blocks{};
    uint32_t measure_vertex_audit = 0u;
    float acceleration_spectral_radius = 0.0f;
    uint32_t mimic_couplings = 0u;
    uint32_t max_independent_dof = kMaxArticulationDof;
    uint32_t material_cells_per_env = 0u;
    uint32_t grid_particles_per_env = 0u;
    uint32_t grid_dims[3] = {};
    uint32_t clear_failure_diagnostics = 1u;
    uint32_t substep_index = 0u;
    // Split-impulse iterations after the velocity solve, as the row solve's position pass.
    uint32_t position_iterations = 0u;
    float position_beta = 0.0f;
    float position_slop = 0.0f;
    float position_max_velocity = 0.0f;
    // Sweeps stop once the vertex equations and contact rows hold to this velocity; zero runs them all.
    float velocity_tolerance = 0.0f;
};

// Environments of rigid bodies and articulations without particles or grid nodes, whose articulations fit a
// warp of coordinates, solve as one warp tile each; only the tiles run position iterations.
inline bool BlockDescentTileRoute(uint32_t particles, uint32_t grid_nodes, uint32_t blocks, uint32_t max_dof) {
    return particles == 0u && grid_nodes == 0u && blocks > 0u && max_dof <= 32u;
}

// Outer iterations of the warp tiles when the caller sets none.
inline constexpr uint32_t kBlockDescentTileIterations = 6u;

uint64_t BlockDescentScratchWords(const BlockDescentSolveParams& params);

// Particle modes select material constraints and ownership; all row-coupled particles share integration.
inline constexpr uint32_t kParticleModeNone = 0u;
inline constexpr uint32_t kParticleModeVbd = 1u;
inline constexpr uint32_t kParticleModeMpm = 5u;
inline constexpr uint32_t kParticleModeMpmVbd = 6u;

// Cloth anisotropic AERODYNAMIC DRAG (pre-predict velocity impulse). One thread
// per surface triangle forms the outward normal + mean velocity from the current
// positions and applies F = -(Cn*|v_n|*v_n + Ct*|v_t|*v_t)*A as dv = (F/m)*dt to
// the 3 vertices (atomic scatter; a vertex is shared by several triangles). The
// normal-dominant anisotropy (drag_normal >> drag_tangent) destabilizes a flat
// falling sheet into flutter. Emitted ONLY when tri_count>0 and a coeff is set
// (a drag-free world never emits it -> byte-identical).
struct AeroDragParams {
    float    dt;
    float    drag_normal;      // lumped 0.5*rho*Cn (normal-dominant)
    float    drag_tangent;     // lumped 0.5*rho*Ct
    float    max_dv;           // per-step impulse clamp (0 == uncapped)
    uint32_t tri_count;        // total env-major aero triangles (0 == inert)
    uint32_t particle_count;
};

struct ParticlePredictParams {
    float    dt;
    float    gravity[3];
    uint32_t mode;             // kParticleMode*
    uint32_t particle_count;   // total env-major particles
    uint32_t particles_per_env;// per-env particle stride
    // MpmVbd: the per-env MPM slice count. The VBD predict SKIPS the low slice
    // [0, n_mpm) (the transfer loop owns it); 0 for every non-MpmVbd mode.
    uint32_t n_mpm_particles;
    VertexBlockLayout vertex_blocks{};  // predicted by ClothPredict instead
};

struct ClothStepParams {
    float dt;
    float gravity[3];
    VertexBlockLayout layout;
    uint32_t integrator;
    uint32_t truncation;  // nonzero when displacement truncation precedes the finalize
};

// Non-MPM particles share one projected position buffer and one step-start position.
struct ParticleProjectionVelocityParams {
    float dt;
    uint32_t particle_count;
    uint32_t particles_per_env;
    uint32_t active_begin_per_env;
};

struct ParticleContactDeltaParams {
    float dt;
    uint32_t particle_count;
    uint32_t particles_per_env;
    uint32_t active_begin_per_env;
};

struct ParticleFinalizeParams {
    float    dt;
    uint32_t mode;             // kParticleMode*
    uint32_t particle_count;
    uint32_t particles_per_env;
    // Split-impulse position pass active: carry the per-particle pseudo velocity
    // onto the final position (non-penetration push-out). 0 leaves it byte-identical.
    uint32_t pos_pass;
    // MpmVbd: the per-env MPM slice count. The VBD finalize SKIPS the low slice
    // [0, n_mpm) (the transfer loop set its final pos/vel); 0 for non-MpmVbd.
    uint32_t n_mpm_particles;
};

// --- readout / RL substrate ---------------------------------------------
struct ReadoutContactWrenchParams {
    float    dt;                // force = impulse / dt
    uint32_t env_count;
    uint32_t base_link_count;
    uint32_t max_contacts_per_env;
    uint32_t rows_per_env;      // per-env solver row-slot span (max_rows_per_env)
    uint32_t full_row_slot_count;  // number of rigid slots before the compact tail
    uint32_t contact_rows_per_env;
    uint64_t workspace_bytes;
};

struct ReadoutEnergyLedgerParams {
    nk::EnergyStage stage = nk::EnergyStage::Begin;
    float dt = 0.0f;
    float gravity[3]{};
    uint32_t env_count = 0u;
    uint32_t substeps = 1u;
    uint32_t slot = 0u;
    uint32_t particles_per_env = 0u;
    uint32_t surfaces_per_env = 0u;
    uint32_t vbd_begin = 0u;
    uint32_t vbd_vertices = 0u;
    uint32_t vbd_elements = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t links_per_env = 0u;
    uint32_t artics_per_env = 0u;
    uint32_t dofs_per_artic = 0u;
    uint32_t rows_per_env = 0u;
    uint32_t contact_rows = 0u;
    uint32_t limit_rows = 0u;
    uint32_t friction_rows = 0u;
    uint32_t drive_rows = 0u;
    uint32_t has_aero = 0u;
    uint32_t has_dat = 0u;
    uint32_t pos_pass = 0u;
    uint32_t coverage_status = 0u;
    uint32_t physics_diagnostics = 0u;
    uint32_t cloth_integrator = 0u;
    uint32_t contact_slots_per_env = 0u;
    uint32_t rigid_contact_slots = 0u;
    uint32_t ogc_contact_slots = 0u;
    float pos_slop = 0.0f;
    float particle_radius = 0.0f;
    uint32_t sdf_grid_count = 0u;
    uint32_t sdf_cell_total = 0u;
    collision::MeshGeometryCounts mesh_geometry{};
};

struct ExportObsParams {
    uint32_t env_count;
    uint32_t base_link_count;
    uint32_t obs_width;         // floats per env in obs_buffer
};

struct ReadoutMotionParams {
    sensor::MotionFrame* frames = nullptr;
    uint32_t env_count = 0u;
    uint32_t links_per_env = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t articulations_per_env = 0u;
};

struct SampleStateSensorParams {
    const sensor::MotionFrame* before = nullptr;
    const sensor::MotionFrame* after = nullptr;
    const sensor::WrenchImpulse* contact_impulses = nullptr;
    const sensor::WrenchImpulse* transmitted_impulses = nullptr;
    sensor::TactileState* tactile = nullptr;
    const double* times = nullptr;
    float* values = nullptr;
    sensor::ObservationNoiseState* noise = nullptr;
    sensor::StateSensorRuntime* runtime = nullptr;
    sensor::StateSensorPacket* queue = nullptr;
    sensor::StateSensorDesc desc;
    math::Vec3 gravity;
    double interval = 0.0;
    uint32_t env_count = 0u;
    uint32_t frames_per_env = 0u;
    uint32_t links_per_env = 0u;
    uint32_t channel = 0u;
    uint32_t value_count = 0u;
    uint32_t queue_capacity = 0u;
};

struct SensorContactLayout {
    uint32_t env_count = 0u;
    uint32_t links_per_env = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t articulations_per_env = 0u;
    uint32_t slots_per_env = 0u;
    uint32_t rigid_slots_per_env = 0u;
    uint32_t rows_per_env = 0u;
    uint32_t contact_rows_per_env = 0u;
};

struct ReadoutSensorWrenchesParams : SensorContactLayout {
    const sensor::MotionFrame* before = nullptr;
    const sensor::MotionFrame* after = nullptr;
    sensor::WrenchImpulse* contact_impulses = nullptr;
    sensor::WrenchImpulse* transmitted_impulses = nullptr;
    math::Vec3 gravity;
    double interval = 0.0;
    uint32_t link_contacts = 0u;
    uint32_t body_contacts = 0u;
    uint32_t joint_loads = 0u;
};

struct ReadoutContactRegionParams : SensorContactLayout {
    const sensor::MotionFrame* before = nullptr;
    const sensor::MotionFrame* after = nullptr;
    sensor::TactileState* states = nullptr;
    sensor::TactileConfig config;
    math::Transform local_offset;
    uint32_t frame = 0u;
    sensor::StateSensorKind kind = sensor::StateSensorKind::Touch;
};

struct AdvanceSensorTimeParams {
    double* times = nullptr;
    double interval = 0.0;
    uint32_t env_count = 0u;
};

struct SampleObservationParams {
    const float* source = nullptr;
    float* values = nullptr;
    sensor::ObservationNoiseState* noise_state = nullptr;
    sensor::ObservationStamp* stamps = nullptr;
    uint32_t env_count = 0u;
    uint32_t values_per_env = 0u;
    uint32_t channel = 0u;
    double sample_interval = 0.0;
    float temperature = 25.0f;
    sensor::ObservationConfig config;
};

struct ResetObservationParams {
    float* values = nullptr;
    sensor::ObservationNoiseState* noise_state = nullptr;
    sensor::ObservationStamp* stamps = nullptr;
    const uint32_t* env_ids = nullptr;
    uint32_t selected_count = 0u;
    uint32_t values_per_env = 0u;
};

// ReadoutUnionContactObsParams (the union-only per-env contact obs params)
// was DELETED here along with its op + kernel. Grasp/union moved to RL.

// Restore selected environments and invalidate their contact state.
struct ResetEnvsParams {
    uint32_t energy_substeps = 1u;
    uint32_t count;
    uint32_t env_count;
    uint32_t base_link_count;       // links per env
    uint32_t lambda_stride;         // rows per env
    uint32_t contact_slot_count;    // slots per env
    uint32_t articulation_count;    // total roots
    uint32_t articulations_per_env;
    uint32_t dofs_per_articulation;
    uint32_t use_env_ids;           // 0 = consecutive envs; 1 = reset_env_ids
    uint32_t body_count;            // bodies per env
    uint32_t particle_count;        // particles per env
    uint64_t ic_seed;
    uint32_t ic_episode;
    uint32_t jitter_body_index;
    float    jitter_body_xyz[3];    // symmetric half-ranges; zero disables jitter
    float    jitter_base_pos[3];
    float    jitter_q;
    uint32_t point_endpoints_per_env = 0u;
    uint32_t point_endpoint_terms_per_env = 0u;
};

struct SnapshotStateParams {
    uint32_t total_link_count;
    uint32_t env_count;
    uint32_t articulation_count;
    uint32_t total_body_count;
    uint32_t total_particle_count;
};

// Restore all environments with the same state lifecycle as a masked reset.
struct RestoreStateParams {
    uint32_t energy_substeps = 1u;
    uint32_t total_link_count;
    uint32_t env_count;
    uint32_t articulation_count;
    uint32_t dofs_per_articulation;
    uint32_t row_slot_count;
    uint32_t contact_slot_count;
    uint32_t total_body_count;
    uint32_t total_particle_count;
    uint32_t point_endpoints_per_env = 0u;
    uint32_t point_endpoint_terms_per_env = 0u;
};

struct ContactWarmStartParams {
    uint32_t phase;              // 0 = prepare rows, 1 = commit solved blocks
    uint32_t env_count;
    uint32_t slot_count;         // contact slots per env
    uint32_t rows_per_env;
    uint32_t full_row_slot_count;
    uint32_t decay_steps;        // absent contacts expire after this many steps
    uint64_t workspace_bytes = 0u;
};

// --- domain randomization -----------------------------------------------
struct RandomizeMaterialBucketsParams {
    uint64_t seed;
};

struct RandomizeBodyParamsParams {
    uint64_t seed;
    float range;           // +/- fractional perturbation
};

} // namespace nuka::phi
