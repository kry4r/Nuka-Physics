#pragma once
// ---------------------------------------------------------------------------
// PHI v2 — op schema: the closed enumeration of physics ops the backend layer
// dispatches, plus one POD `<Op>Params` struct per op.
//
// ggml/llama.cpp analogy: NkOp is the GGML_OP enum; each <Op>Params is the
// op-specific parameter block that travels with an OpCall (phi/backend.hpp).
//
// CONTRACT (, frozen plan spec):
// * NkOp has exactly 30 named ops + a trailing `Count` sentinel. (
// appended StepBackward, the diffsim contact-free single-step adjoint; 
// Appended ParticleParticleContact, the cross-system
// particle-particle non-penetration co-step. Optional RL initial-condition randomization appended
// ReadoutUnionContactObs, the union-only per-env contact observation
// readout — strictly ADDITIVE, emitted ONLY for the UnionCsr family. 
// appended ApplyImplicitDamping, the standalone backward-Euler joint
// viscous-damping velocity correction extracted from the deleted FUSED
// contact solve — strictly ADDITIVE, emitted ONLY when fold_drive_damping.)
// * Every op gets a trivially-copyable aggregate `<Op>Params`. Two are
// spec-fixed (SolveRowsBlockIslandParams, NarrowphaseSdfParams); the rest
// carry a minimal plausible field set where obvious, or a reserved POD
// otherwise. They are fleshed out when each op is actually implemented
// (-). DO NOT depend on any field below being final outside .
// * Pure C++ — ZERO CUDA types. The op *implementations* (+) live in the
// backend and receive `const void* params` re-cast to the matching POD.
//
// This header is included by both host (.cpp) and device (.cu) TUs, so it must
// stay free of STL containers and of anything not trivially copyable.
// ---------------------------------------------------------------------------

#include <cstdint>
#include "collision/mesh_surface_types.hpp"
#include "sensor/observation_types.hpp"
#include "sensor/state_types.hpp"

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
inline constexpr uint32_t kEnvStatusNeighborOverflow = 1u << 1;  // particle neighbor dropped
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
inline constexpr uint32_t kBodyGyroNotConverged      = 1u;
inline constexpr uint32_t kBodyGyroInvalidInput      = 2u;
inline constexpr uint32_t kDefaultParticleNeighborBudget = 32u;

// ---------------------------------------------------------------------------
// NkOp — the closed op set. uint16_t backing so an op id fits a single field
// and the enum can be stored in compact tables.
// ---------------------------------------------------------------------------
enum class NkOp : uint16_t {
    // --- articulated-body dynamics --------------------------------------
    ApplyDrives,           // PD / motor drives -> generalized forces
    ApplyDynamicsDrives,   // computed-torque and task control from free ABA + M^-1
    AbaForward,            // Featherstone ABA forward dynamics (q,qd -> qdd)
    IntegrateVelocity,     // qd += qdd * dt
    FkWorldPoses,          // forward kinematics -> per-link world poses
    IntegratePosition,     // q += qd * dt
    CrbaComputeM,          // composite rigid-body M (joint-space inertia)
    CrbaFactorM,           // LTDL / Cholesky factorization of M
    ApplyImplicitDamping,  // standalone backward-Euler joint viscous-damping
                           // velocity correction qdot -= dt*(M+dt*C)^-1*(C*qdot).
                           // General articulation physics that USED to ride
                           // inside the deleted FUSED contact solve kernel; now a
                           // standalone op gated on fold_drive_damping (so the
                           // damping survives the FUSED-path deletion). Reads the
                           // factored (M+dt*C)^-1 = data.m_inv (must run AFTER
                           // CrbaFactorM) and the per-DOF c_j = data.drive_damping.

    // --- broadphase / spatial acceleration ------------------------------
    BuildAabbs,            // per-collidable world-space AABB build
    LbvhBuild,             // Karras LBVH build over the AABB set
    LbvhQueryPairs,        // overlap query -> candidate pair stream
    ParticleGridBuild,     // uniform spatial hash grid for particles

    // --- narrowphase / contact rows -------------------------------------
    NarrowphasePrimitives, // analytic primitive x primitive narrowphase
    NarrowphaseSdf,        // SDF / mesh narrowphase
    ContactTangentBasis,   // per-contact tangent frame
    AssembleRows,          // constraint-row assembly (J, bounds, bias)
    SolveRowsBlockIsland,  // block-island PGS/row solve

    // --- particle (XPBD / PBF) substep ----------------------------------
    ParticleAeroDrag,      // cloth anisotropic air-drag velocity impulse (pre-predict)
    ParticlePredict,       // predict x* = x + v*dt + g
    XpbdProject,           // XPBD constraint projection sweep
    PbfDensityLambda,      // PBF density constraint lambda
    PbfApplyDelta,         // PBF position delta apply
    ParticleFinalize,      // v = (x* - x)/dt; commit x

    // MLS-MPM grid state remains available between prediction and commit.
    MpmPredict,            // P2G, stress and external-force grid prediction.
    MpmExchange,           // Grid boundary impulse and endpoint reaction.
    MpmCommit,             // G2P, particle advection and material history.

    // --- readout / RL substrate -----------------------------------------
    ReadoutContactWrench,  // per-link contact wrench readout
    ExportObs,             // whole-body observation export
    ResetEnvs,             // per-env reset to initial state
    SnapshotState,         // capture full world state
    RestoreState,          // restore from a snapshot
    // ReadoutUnionContactObs (union-only per-env contact obs) was DELETED.
    // Removing the enum value is safe — NkOp ids are runtime-only dispatch, never
    // serialized (goldens byte-exact across the enum shifts).

    // --- domain randomization -------------------------------------------
    RandomizeMaterialBuckets, // per-env material bucket randomization
    RandomizeBodyParams,      // per-env body/inertia randomization

    // --- differentiable rollout ---------------------------------
    StepBackward,          // diffsim contact-free single-step reverse adjoint

    // --- cross-system particle contact (Cross-system) -----------------
    ParticleParticleContact, // class-blind unilateral non-penetration co-step

    // --- general contact pipeline (B2) --------------------------
    SyncLinkBodyPose,      // copy each articulation link's FK world pose into its
                           // owning body_pose row (composing link_geom_local) so
                           // artic links enter the LBVH as collidables. Gated to
                           // the PairDriven family (early-exit otherwise), so the
                           // UnionCsr graph is byte-untouched.

    // --- body/artic <-> particle contact --------------------------------
    NarrowphaseBodyParticle, // body/artic <-> soft/fluid particle narrowphase: one
                            // thread per (env x particle) traverses the env LBVH for
                            // overlapping collidables, treats the particle as a sphere
                            // of its radius, and writes the sphere-vs-shape manifold
                            // into the particle's RESERVED contact-slot sub-range with
                            // the particle-side index-kind tag + global particle id.
                            // Runs AFTER the rigid narrowphase + BEFORE AssembleRows;
                            // PairDriven-family-gated (early-exit otherwise).

    // --- general contact pipeline (H3) --------------------------
    NarrowphaseHeightfield, // per-cell heightfield midphase: for each broadphase
                            // (convex, heightfield) candidate pair, walk the
                            // overlapped grid cells, emit 2 TRIANGLE_PRISM
                            // colliders/cell, route each through the EXISTING cvx
                            // GJK/EPA, and write the manifolds into the unified
                            // ucontact_* buffer (side a = convex, side b = the
                            // static heightfield). APPENDED after SyncLinkBodyPose
                            // (stable prior NkOp values). PairDriven-family-gated
                            // (early-exit for UnionCsr).

    // --- dynamic solve islanding --------------------------------------
    BuildSolveIslands,      // per-step connected-components over the ACTIVE contact
                            // rows -> one island per independent component, emitted
                            // as the dynamic schedule SolveRowsBlockIsland reads.
                            // Runs AFTER AssembleRows + BEFORE SolveRowsBlockIsland;
                            // PairDriven-family-gated. APPENDED after the prior ops
                            // (stable NkOp values; execution order is the pipeline's,
                            // not the enum value).
    ContactWarmStart,       // ContactId-indexed warm-start prepare/commit.
    SnapshotStepVelocity,   // Capture the contact acceleration reference before forces.
    ParticleProjectionVelocity, // Publish projected particle velocities before contact solving.
    ParticleContactDelta,   // Apply only the new contact impulse to particle working positions.
    AccumulateStep,         // Aggregate impulses and diagnostics across physical intervals.
    FkLinkVelocities,       // Refresh link spatial velocities from current generalized state.

    ReadoutDrives,         // actual bounded effort from the common velocity solve
    RefitParticleSurfaces,
    SampleObservation,
    ResetObservation,
    ReadoutMotion,
    SampleStateSensor,
    AdvanceSensorTime,
    ReadoutSensorWrenches,
    ReadoutContactRegion,

    Count                    // sentinel: number of ops (NOT an op)
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

struct AbaForwardParams {
    float    gravity[3];   // world-frame acceleration in m/s^2
    uint32_t articulation_count;
    uint32_t total_link_count;
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
};

struct CrbaComputeMParams {
    float    dt;                  // dt*C fold (implicit joint damping)
    uint32_t max_dof;             // M tile stride (== dofs_per_env)
    uint32_t articulation_count;
    uint32_t total_link_count;
    // 1 => fold dt * drive_damping into the joint diagonals so the factored
    // inverse is (M + dt*C)^-1 (both production paths). 0 => pure CRBA M.
    uint32_t fold_drive_damping;
};

struct CrbaFactorMParams {
    uint32_t max_dof;
    uint32_t articulation_count;
};

// Standalone implicit joint-damping op. Applies the backward-Euler joint
// viscous-damping velocity correction qdot -= dt*(M+dt*C)^-1*(C*qdot) using the
// factored inverse data.m_inv (== (M+dt*C)^-1 when CrbaComputeM folded dt*C, so
// this op MUST be scheduled AFTER CrbaFactorM and BEFORE the row solve /
// IntegratePosition) and the per-DOF c_j == data.drive_damping. The float
// sequence is the deleted FUSED solve kernel's implicit-damping seed +
// write-back, so a zero-contact world's trajectory is byte-identical to the
// legacy standalone-damping order (factor -> damping -> integrate).
struct ApplyImplicitDampingParams {
    float    dt;
    uint32_t max_dof;             // dof_stride == the M tile stride
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

struct ParticleGridBuildParams {
    float    cell_size;         // uniform grid cell edge length
    float    query_radius;      // neighbor search radius
    uint32_t particle_count;    // total particles (env-major)
    float    grid_min[3];       // grid lower corner
    uint32_t grid_dims[3];      // grid resolution (PER ENV)
    // which position field the grid is built over. PBF builds the neighbor
    // list on the PREDICTED positions (legacy PBF step order), so pos_source == 1
    // routes the op to pbf_predicted_pos; 0 == particle_pos (the default).
    uint32_t pos_source;        // 0 = particle_pos, 1 = pbf_predicted_pos
    // Env-private grids (review fix): cell keys are offset env*cells so envs
    // never share a cell (env-major replicated particles would otherwise see
    // their own clones as neighbors). cells_capacity mirrors
    // ModelCapacities::max_grid_cells (the grid_cell_start/end per-env arena
    // sizing); the op fails LOUDLY when the live dims product exceeds it or
    // when cells*env_count overflows the u32 cell key.
    uint32_t env_count;
    uint32_t particles_per_env;
    uint32_t cells_capacity;    // per-env cell capacity (max_grid_cells)
    uint32_t neighbor_capacity; // total neighbor indices reserved per environment
};
inline constexpr uint32_t kGridPosSourceParticlePos = 0u;
inline constexpr uint32_t kGridPosSourcePbfPredicted = 1u;

// Byte size of the pre-allocated scratch ParticleGridBuild's cub radix sort +
// exclusive scan draw temp storage + sort output buffers from, for
// `particle_count` keys (cub size queries + the two out buffers, 256B-aligned).
// Host-callable (defined in broadphase.cu) so the World sizes the grid_sort_scratch
// arena field BEFORE allocation -> no mid-capture cudaMalloc. 0 for 0 particles.
uint64_t GridSortScratchBytes(uint32_t particle_count);

// Byte size of the pre-allocated island_cub_temp the BuildSolveIslands op draws its
// cub radix-sort temp storage from, for `total_rows` row slots. Host-callable
// (defined in build_islands.cu) so the World sizes the field BEFORE allocation ->
// no mid-capture cudaMalloc. 0 for 0 rows.
uint64_t IslandSortScratchBytes(uint32_t total_rows);

// Byte size of the pre-allocated pair_sort_scratch the LbvhQueryPairs op draws its
// canonicalizing cub radix sort (u64 key -> u32 slot perm) + per-env prefix + pair
// snapshot from. Host-callable (defined in broadphase.cu) so the World sizes the
// field BEFORE allocation -> no mid-capture cudaMalloc. 0 for empty inputs.
uint64_t PairSortScratchBytes(uint32_t total_sort_slots, uint32_t env_count);
uint64_t LbvhSortScratchBytes(uint32_t env_count, uint32_t bodies_per_env);
uint64_t ContactCacheScratchBytes(uint32_t point_count, uint32_t env_count);
uint64_t ContactIndexScratchBytes(uint32_t row_count, uint32_t env_count);
uint64_t SolverVelocityScratchBytes(uint32_t body_count, uint32_t particle_count,
                                    uint32_t grid_count);

// Dynamic solve-island build (connected components over the active contact rows).
// All launch geometry is a fixed function of the capacities (graph-capturable); the
// op partitions the rows by union-find each step and emits the device schedule the
// SolveRowsBlockIsland op reads (island_rows/island_quads/island_tiles/island_count).
struct BuildSolveIslandsParams {
    uint32_t family;             // kContactFamily* (PairDriven runs; else early-exit)
    uint32_t env_count;
    uint32_t rows_per_env;       // row-slot capacity per env
    uint32_t articulation_count; // global artic count (== artics_per_env * env_count)
    uint32_t bodies_per_env;     // movable rigid body count per env
    uint32_t particles_per_env;  // particle count per env
    uint32_t grid_nodes_per_env = 0u;
};

// All MPM operations share one physical interval and environment-private grid.
// Prediction owns grid initialization; commit alone advances particles and history.
struct ParticleSurfacesParams {
    uint32_t env_count;
    uint32_t particles_per_env;
    uint32_t surfaces_per_env;
    uint32_t triangles_per_env;
    uint32_t nodes_per_env;
};

struct MpmParams {
    uint32_t contact_slot_base;
    uint32_t contact_capacity;
    uint32_t contact_slots_per_env;
    uint32_t full_row_slot_count;
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
    uint32_t mode;               // kParticleModeMpm or kParticleModeMpmXpbd.
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
};

// Workspace bytes for the actual MPM particle and grid node counts.
// Query before state allocation; zero particles require no workspace.
uint64_t MpmSortScratchBytes(uint32_t particle_count, uint32_t node_count);

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

// General contact pipeline (H3): the per-cell heightfield midphase. One
// thread per (env x candidate slot); when the slot's broadphase pair is (convex,
// heightfield) it projects the convex's world AABB into heightfield-LOCAL space,
// maps it to the overlapped grid cell range, and for EACH cell emits its 2
// TRIANGLE_PRISM colliders (corner heights -> the Newton (p00,p10,p11)+(p00,p11,
// p01) layout) routed through the EXISTING cvx GJK/EPA. The resulting <=4-pt
// manifolds are fanned across FREE trailing ucontact slots (atomically reserved
// from the per-env candidate-slot tail), so a body straddling a STEP gets
// contacts from BOTH the tread AND the riser prisms. Side a == the convex body
// (its body_id resolves the reaction side); side b == the heightfield (static,
// body_id==-1 -> no reaction). The HeightfieldData descriptor travels in the
// params (one heightfield/model in the first scope); the height grid rides the
// model `heights` field. EARLY-EXITS unless family == kContactFamilyPairDriven.
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
    // The PBF/SoftFluid fluid-slice predict writes the gravity-integrated position
    // into pbf_predicted_pos, leaving particle_pos at step-start. Route the fluid
    // slice's detection at the predicted position (consistent with the grid build's
    // pos_source) so a fast fluid particle brakes within one step instead of lagging.
    uint32_t fluid_pos_source;    // 1 == fluid slice reads pbf_predicted_pos
    uint32_t n_soft_particles;    // per-env soft/fluid split (fluid is [n_soft, P))
    // MpmXpbd: the first env-local particle that generates body rows. The MPM slice
    // [0, particle_row_base) couples via the grid, NOT rows; it owns no reserved
    // slots (the cook exempts it from the budget). 0 for every other mode.
    uint32_t particle_row_base;
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

struct ContactTangentBasisParams {
    uint32_t slot_count;        // env_count * max_contacts_per_env
    // the FUSED tangent kernel was deleted. The op is now enqueued ONLY for
    // the PairDriven family (the union family emits its tangent spokes inside
    // AssembleRows), and it unconditionally reads the unified ucontact_normal
    // (elem:4) -> ucontact_tangent1/2. The field is retained for layout/diagnostics.
    uint32_t family = 0u;
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
    // Slots [0, full_row_slot_count) use the rigid 4-point/20-row layout; the
    // body-particle provider's reserved tail uses its exact 1-point/5-row layout.
    // Equal to union_slot_count when the model has no body-particle reserve.
    uint32_t full_row_slot_count;
    uint32_t bodies_per_env;
    uint32_t base_link_count;   // links per env (replica stride)
    float    solref[2];         // merged contact solref (union family)
    float    solimp[5];         // merged contact solimp (union family)
    uint32_t particles_per_env; // particle coupling (kUSlotParticleSphere*).
    uint32_t num_material_buckets;  // 0 == no authored materials -> model defaults.
    // Per-system body<->particle friction; n_soft_particles splits per-env
    // [soft|fluid] so a side reads soft vs fluid mu, mixed with the body by solmix=max.
    uint32_t n_soft_particles;       // per-env soft/fluid split (0 => no soft slice).
    float    particle_soft_friction; // soft/cloth mu (finite: a foot grips/drags).
    float    particle_fluid_friction;// fluid mu (~0: a foot slides tangentially).
    // Cap on the contact normal aref so a deep contact recovers over several steps,
    // not in one fling. +inf default == non-binding (byte-identical). Model property.
    float    baumgarte_max_velocity;
    uint32_t point_endpoints_per_env = 0u;
    uint32_t point_endpoint_terms_per_env = 0u;
};

// Spec-fixed semantic fields : {dt, vel_iters, pos_iters}. The fields BELOW
// the spec triplet are the precedent LAUNCH-GEOMETRY transport (op_schema
// header note: "ops carry the launch-geometry counts in their params because
// ModelView/DataView are pure pointer aggregates") plus the per-family solver
// constants that are Model properties (the fused family's legacy knobs). They
// are filled by Pipeline::Build from the Model — the three semantic fields
// keep their spec meaning and position.
struct SolveRowsBlockIslandParams {
    float dt;
    uint16_t vel_iters;
    uint16_t pos_iters;
    // -- appended launch geometry + Model-derived solver constants ------
    uint32_t family;             // kContactFamily*
    uint32_t total_islands;      // union family: schedule island count (grid x)
    uint32_t max_dof;            // dof_stride == the M tile stride
    uint32_t env_count;
    uint32_t articulation_count;
    uint32_t rows_per_env;       // row slots per env (union family)
    uint32_t contact_rows_per_env; // fixed contact footprint before joint rows
    uint32_t joint_limit_rows_per_env;
    uint32_t base_link_count;    // links per env (qdot scatter)
    uint32_t total_body_count;   // env-major rigid body count
    // Fused-family legacy knobs (Model properties; unused by the union family):
    float    friction_coefficient;
    float    baumgarte_max_velocity;
    // (: the FUSED-family `apply_implicit_damping` knob was REMOVED — the
    // implicit joint-damping seed moved out of the deleted FUSED solve kernel
    // into the standalone NkOp::ApplyImplicitDamping op; nothing in the solve op
    // reads it any more.)
    // The solver's per-articulation contact-slot STRIDE (slot_base = articulation *
    // contact_slots_per_artic), the data-driven quotient max_contacts_per_env /
    // articulations_per_env bounded by kMaxContactsPerArtic. At K<=1 ==
    // kMaxFootContactsPerEnv (4) -> byte-identical; at K>1 the grown stride gives
    // each co-resident articulation room for end-effector + body-terrain +
    // inter-articulation rows in one block.
    uint32_t contact_slots_per_artic;
    // Split-impulse position pass (PairDriven only; pos_iters>0). total_particle_count
    // sizes the pseudo-velocity memsets; beta/slop tune the geometric projection
    // (target pseudo separating vel = beta*max(depth-slop,0)/dt).
    uint32_t total_particle_count;
    float    pos_beta;
    float    pos_slop;
    // Sweep convergence bound in velocity units (m/s); 0 sweeps the full budget.
    float    vel_tolerance;
    // Validation hook (NUKA_FORCE_STATIC_ISLANDS): run the conservative cook-time
    // one-island-per-env schedule instead of the dynamic CC pass, to A/B them. 0 == off.
    uint32_t force_static_islands;
    uint32_t continue_impulses = 0u; // Retain this step's applied impulses across solver calls.
    uint32_t measure_contact_residual = 0u;
    uint32_t total_grid_count = 0u;
    uint64_t workspace_bytes = 0u;
};

// Word count of the solve_color_scratch field the dynamic island solve colors live rows in.
// Host-callable (defined in solve_rows.cu) so the World sizes it before allocation.
uint64_t SolveColorScratchWords(const SolveRowsBlockIslandParams& params);

// Particle modes select material constraints and ownership; all row-coupled particles share integration.
inline constexpr uint32_t kParticleModeNone = 0u;
inline constexpr uint32_t kParticleModeXpbd = 1u;
inline constexpr uint32_t kParticleModePbf = 2u;
inline constexpr uint32_t kParticleModeCoupled = 3u;
inline constexpr uint32_t kParticleModeSoftFluid = 4u;
inline constexpr uint32_t kParticleModeMpm = 5u;
inline constexpr uint32_t kParticleModeMpmXpbd = 6u;

// The internal material controls which projection equations apply to coupled point masses.
inline constexpr uint32_t kCoupledInternalNone = 0u;
inline constexpr uint32_t kCoupledInternalXpbd = 1u;
inline constexpr uint32_t kCoupledInternalPbf = 2u;

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
    uint32_t coupled_internal; // kCoupledInternal* (coupled mode only)
    // The material split is independent of the shared prediction and finalization kernels.
    uint32_t n_soft_particles; // per-env soft count (split index)
    uint32_t particles_per_env;// per-env particle stride
    // MpmXpbd: the per-env MPM slice count. The XPBD predict SKIPS the low slice
    // [0, n_mpm) (the transfer loop owns it); 0 for every non-MpmXpbd mode.
    uint32_t n_mpm_particles;
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

struct XpbdProjectParams {
    float    dt;
    uint16_t iters;            // XPBD Gauss-Seidel sweep count
    uint32_t dist_con_count;   // total env-major distance constraints
    uint32_t bend_con_count;   // total env-major bend constraints
    uint32_t vol_con_count;    // total env-major volume constraints
    // Environment-major shape-match clusters run after distance, bend, and volume.
    uint32_t shape_match_cluster_count;
    // Each color contains independent constraints; colors execute in order.
    // Per-environment strides map local constraint and cluster indices to global indices.
    uint32_t dist_colors;
    uint32_t bend_colors;
    uint32_t vol_colors;
    uint32_t sm_colors;
    uint32_t env_count;
    uint32_t dist_cons_per_env;
    uint32_t bend_cons_per_env;
    uint32_t vol_cons_per_env;
    uint32_t sm_clusters_per_env;
    uint32_t sm_members_per_env;
    // Host {offset,count} color tables remain valid for the world's lifetime.
    // Backends use their workload bounds while preserving ordered color dependencies.
    const uint32_t* dist_color_segments;
    const uint32_t* bend_color_segments;
    const uint32_t* vol_color_segments;
    const uint32_t* sm_color_segments;
    uint32_t iteration_start = 0u; // Zero starts the step's lambda accumulation.
};

struct PbfDensityLambdaParams {
    float    rest_density;     // rho0
    float    relaxation;       // CFM-style epsilon (cfm_epsilon)
    float    support_radius;   // SPH support radius h (== grid query radius)
    float    particle_mass;    // uniform per-particle mass
    uint32_t particle_count;
    uint16_t iters;            // density-projection iterations
    uint16_t clamp_overdensity;// 1 => clamp C_i >= 0 (no surface cohesion pull)
    float    dt;               // substep dt (for the boundary clamp + apply)
    uint32_t boundary_enabled; // 1 => apply the floor clamp in the apply pass
    float    floor_z;          // boundary floor (the grid is z-up; legacy y).
    // SoftFluid: scope the density/lambda solve to the fluid slice. A soft
    // particle (within-env local index < n_soft) must NOT contribute to fluid
    // density. 0 == single-system PBF (every particle is fluid).
    uint32_t n_soft_particles;  // per-env soft count (fluid slice = [n_soft, per_env))
    uint32_t particles_per_env; // per-env particle stride
};

struct PbfApplyDeltaParams {
    float    support_radius;
    float    particle_mass;
    uint32_t particle_count;
    uint32_t boundary_enabled;
    float    floor_z;
    // SoftFluid: fluid slice scope (see PbfDensityLambdaParams).
    uint32_t n_soft_particles;
    uint32_t particles_per_env;
};

struct ParticleFinalizeParams {
    float    dt;
    uint32_t mode;             // kParticleMode*
    uint32_t particle_count;
    uint32_t coupled_internal; // kCoupledInternal* (coupled mode only)
    // PBF post-finalize polish (gated inert when the coefficient is 0).
    float    support_radius;
    float    particle_mass;
    float    xsph_viscosity_c;     // XSPH velocity-smoothing coefficient
    float    surface_tension_gamma;// Akinci cohesion coefficient
    float    rest_density;         // rho0 (for the XSPH density normalization)
    // Velocity polish is restricted to the fluid material range.
    uint32_t n_soft_particles;
    uint32_t particles_per_env;
    // Split-impulse position pass active: carry the per-particle pseudo velocity
    // onto the final position (non-penetration push-out). 0 leaves it byte-identical.
    uint32_t pos_pass;
    // MpmXpbd: the per-env MPM slice count. The XPBD finalize SKIPS the low slice
    // [0, n_mpm) (the transfer loop set its final pos/vel); 0 for non-MpmXpbd.
    uint32_t n_mpm_particles;
};

// Particle contact gathers mass-weighted corrections from the current projected positions.
// Both Jacobi passes use the shared neighbor CSR and temporary density-correction storage.
struct ParticleParticleContactParams {
    // Minimum contact distance d_min == 2*contact_radius (uniform-radius). A pair
    // (i,j) penetrates iff |p_i - p_j| < d_min. <= 0 => the op is inert.
    float    contact_distance_d_min;
    // XPBD compliance alpha (1/stiffness); 0 == rigid (the full -C correction).
    // The position-based co-step uses alpha_tilde = compliance_alpha (dt folded 1).
    float    compliance_alpha;
    uint32_t solver_iterations; // full Jacobi gather+apply sweeps per call (>=1).
    uint32_t mode;              // kParticleMode* (only SoftFluid runs the op).
    uint32_t particle_count;    // total env-major union particles.
    // The per-env [soft | fluid] split + stride; the contact row is class-blind so
    // these are carried only for symmetry with the other particle ops + future
    // per-env scoping. The math reads the FULL union (no soft/fluid branch).
    uint32_t n_soft_particles;
    uint32_t particles_per_env;
};

// --- readout / RL substrate ---------------------------------------------
struct ReadoutContactWrenchParams {
    float    dt;                // force = impulse / dt
    uint32_t env_count;
    uint32_t base_link_count;
    uint32_t max_contacts_per_env;
    uint32_t rows_per_env;      // per-env solver row-slot span (max_rows_per_env)
    uint32_t full_row_slot_count;  // rigid slots [0, this) use the 12-row layout
    uint64_t workspace_bytes;
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
    uint32_t has_particle_grid = 0u;
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
    uint32_t total_link_count;
    uint32_t env_count;
    uint32_t articulation_count;
    uint32_t dofs_per_articulation;
    uint32_t row_slot_count;
    uint32_t contact_slot_count;
    uint32_t total_body_count;
    uint32_t total_particle_count;
    uint32_t has_particle_grid = 0u;
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

// --- differentiable rollout -------------------------------------
// StepBackward: the contact-free single-step reverse adjoint. The articulation
// device state (q/qdot/link_*/joint_*) comes from ModelView/DataView like every
// other articulation op; the EXTRA buffers the adjoint needs are caller-owned
// scratch that does NOT live in the arena (the per-step pre-state snapshots, the
// drive descriptors, the dI/dmass slope, and the in/out gradient buffers), so
// they travel as raw device pointers in the params. The pointer/scalar/flag set
// MIRRORS diffsim::StepBackwardInputs + StepBackwardGrads 1:1 (see
// diffsim/step_backward.hpp); the op unpacks them back into those two structs and
// launches the SINGLE kernel the direct host launcher also drives (so the op path
// and the direct path are byte-identical by single-source). math::Transform*
// fields are carried as void* to keep op_schema.hpp math-header-free; the op
// reinterpret_casts them (the layout is fixed: Vec3 position + Quat rotation).
struct StepBackwardParams {
    uint32_t total_link_count;
    uint32_t articulation_count;
    float    dt;
    float    gravity_z;
    // flags (uint32_t to stay trivially-copyable + header-light).
    uint32_t has_drive;        // 1 => convert dL/dtau through the PD drive
    uint32_t has_integrate;    // 1 => reverse the velocity+position integrators
    uint32_t enable_q_channel; // 1 => back-prop the link_xup = JointTransform(q) path
    // --- StepBackwardInputs pointers (const device buffers) --------------
    const float* q_pre;
    const float* qdot_pre;
    const float* v_root_pre;     // may be null (fixed-base fallback)
    const void*  base_pose_pre;  // const math::Transform* (orientation channel; may be null)
    const float* drive_targets;
    const float* drive_stiffness;
    const float* drive_damping;
    const float* drive_force_limits;
    const float* dI_dmass;
    const float* grad_qddot_seed; // read only when has_integrate == 0; may be null
    // --- StepBackwardGrads pointers (in/out device buffers) --------------
    float* grad_q_out;
    float* grad_qdot_out;
    float* grad_target_out;
    float* grad_mass_out;
    float* grad_tau_out;
    float* grad_link_velocity_out;
    // per-articulation base-pose adjoint (7 floats/art: 3 pos + 4 quat); void* to
    // mirror StepBackwardGrads::grad_base_pose_out (a float*); may be null.
    void*  grad_base_pose_out;
};

} // namespace nuka::phi
