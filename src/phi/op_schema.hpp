#pragma once
// ---------------------------------------------------------------------------
// PHI v2 — op schema: the closed enumeration of physics ops the backend layer
// dispatches, plus one POD `<Op>Params` struct per op.
//
// ggml/llama.cpp analogy: NkOp is the GGML_OP enum; each <Op>Params is the
// op-specific parameter block that travels with an OpCall (phi/backend.hpp).
//
// CONTRACT (M1, frozen plan spec):
//   * NkOp has exactly 30 named ops + a trailing `Count` sentinel. (M9 T7
//     appended StepBackward, the diffsim contact-free single-step adjoint; M9
//     T11 Phase 2 appended ParticleParticleContact, the id-10 cross-system
//     particle-particle non-penetration co-step. M10 RL-completion appended
//     ReadoutUnionContactObs, the union-only per-env contact observation
//     readout — strictly ADDITIVE, emitted ONLY for the UnionCsr family.)
//   * Every op gets a trivially-copyable aggregate `<Op>Params`. Two are
//     spec-fixed (SolveRowsBlockIslandParams, NarrowphaseSdfParams); the rest
//     carry a minimal plausible field set where obvious, or a reserved POD
//     otherwise. They are fleshed out when each op is actually implemented
//     (M3-M6). DO NOT depend on any field below being final outside M1.
//   * Pure C++ — ZERO CUDA types. The op *implementations* (M3+) live in the
//     backend and receive `const void* params` re-cast to the matching POD.
//
// This header is included by both host (.cpp) and device (.cu) TUs, so it must
// stay free of STL containers and of anything not trivially copyable.
// ---------------------------------------------------------------------------

#include <cstdint>

// The SHARED procedural-terrain heightfield params (Go2-on-stairs Phase 1).
// terrain_field.hpp is __host__ __device__-portable (CUDA qualifiers gated on
// __CUDACC__, the philox.cuh precedent), so it stays ZERO-CUDA-type when this
// header is compiled by the host toolchain. nuka::terrain::TerrainParams is a
// trivially-copyable POD of 6 floats (the "no STL / trivially copyable" rule).
#include "sensor/terrain/terrain_field.hpp"

namespace nuka::phi {

// ---------------------------------------------------------------------------
// NkOp — the closed op set. uint16_t backing so an op id fits a single field
// and the enum can be stored in compact tables.
// ---------------------------------------------------------------------------
enum class NkOp : uint16_t {
    // --- articulated-body dynamics --------------------------------------
    ApplyDrives,           // PD / motor drives -> generalized forces
    AbaForward,            // Featherstone ABA forward dynamics (q,qd -> qdd)
    IntegrateVelocity,     // qd += qdd * dt
    FkWorldPoses,          // forward kinematics -> per-link world poses
    IntegratePosition,     // q  += qd  * dt
    CrbaComputeM,          // composite rigid-body M (joint-space inertia)
    CrbaFactorM,           // LTDL / Cholesky factorization of M

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
    ParticlePredict,       // predict x* = x + v*dt + g
    XpbdProject,           // XPBD constraint projection sweep
    PbfDensityLambda,      // PBF density constraint lambda
    PbfApplyDelta,         // PBF position delta apply
    ParticleFinalize,      // v = (x* - x)/dt; commit x

    // --- readout / RL substrate -----------------------------------------
    ReadoutContactWrench,  // per-link contact wrench readout
    ExportObs,             // whole-body observation export
    ResetEnvs,             // per-env reset to initial state
    SnapshotState,         // capture full world state
    RestoreState,          // restore from a snapshot
    ReadoutUnionContactObs,// union-only per-env contact obs (finger force + foot state)

    // --- domain randomization -------------------------------------------
    RandomizeMaterialBuckets, // per-env material bucket randomization
    RandomizeBodyParams,      // per-env body/inertia randomization

    // --- differentiable rollout (M9 T7) ---------------------------------
    StepBackward,          // diffsim contact-free single-step reverse adjoint

    // --- cross-system particle contact (M9 T11 Phase 2) -----------------
    ParticleParticleContact, // id-10 class-blind unilateral non-penetration co-step

    Count                  // sentinel: number of ops (NOT an op)
};

// ---------------------------------------------------------------------------
// Per-op parameter PODs.
//
// All are trivially-copyable aggregates. `reserved` fields keep the structs
// non-empty and stable in size while their real fields are designed in later
// milestones; they MUST stay POD.
// ---------------------------------------------------------------------------

// --- articulated-body dynamics ------------------------------------------
// M3b note: ops carry the launch-geometry counts (total_link_count /
// articulation_count / max_dof) in their params because ModelView/DataView are
// pure pointer aggregates; the Pipeline fills them from the Model capacities.
struct ApplyDrivesParams {
    float    dt;
    uint32_t total_link_count;
    // 1 => the drive emits ONLY the Kp stiffness torque; -Kd*qdot is applied
    // IMPLICITLY downstream (the production batched + single-env paths). 0 =>
    // explicit -Kd*qdot in the drive (the legacy pre-implicit oracle form).
    uint32_t defer_velocity_damping;
    // M4: drive mode. 0 = position PD hold drive (ApplyPositionDriveKernel, the
    // M3 batched articulated path). 1 = direct torque drive (the union
    // world's LaunchApplyTorqueDriveKernels port: tau = clamp(drive_target,
    // +/-drive_force_limit) — drive_target carries the per-link torque).
    uint32_t mode;
};

struct AbaForwardParams {
    float    gravity[3];   // world-frame gravity vector (gravity_z = gravity[2])
    uint32_t articulation_count;
    uint32_t total_link_count;
};

struct IntegrateVelocityParams {
    float    dt;
    float    gravity_z;    // floating-base velocity integrate re-derives a_grav
    uint32_t total_link_count;
    uint32_t articulation_count;
    // M4: movable rigid-body gravity velocity-kick arm (the union world's
    // per-body `linear_velocity.z += g*dt` for inv_mass > 0). 0 = no bodies.
    uint32_t total_body_count;
};

struct FkWorldPosesParams {
    uint32_t articulation_count;
    uint32_t total_link_count;
};

struct IntegratePositionParams {
    float    dt;
    uint32_t total_link_count;
    uint32_t articulation_count;
    // M4: movable rigid-body symplectic-Euler position arm (the union world's
    // IntegrateBodyPosition port). 0 = no bodies.
    uint32_t total_body_count;
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

// --- broadphase / spatial acceleration ----------------------------------
// M5: the broadphase ops (BuildAabbs/LbvhBuild/LbvhQueryPairs) and the SDF
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
};

struct LbvhQueryPairsParams {
    uint32_t max_pairs;         // per-env capacity of the output pair stream
    uint32_t family;            // kContactFamily* (PairDriven => query)
    uint32_t env_count;
    uint32_t bodies_per_env;
    uint32_t max_contacts_per_env;  // candidate_pairs slot stride / env
    uint32_t filter_cross_env;  // 1 => drop pairs spanning envs (env-major)
    uint32_t excluded_count;    // sorted exclude-list length (excluded_pairs)
};

struct ParticleGridBuildParams {
    float    cell_size;         // uniform grid cell edge length
    float    query_radius;      // neighbor search radius
    uint32_t particle_count;    // total particles (env-major)
    float    grid_min[3];       // grid lower corner
    uint32_t grid_dims[3];      // grid resolution (PER ENV)
    // M6: which position field the grid is built over. PBF builds the neighbor
    // list on the PREDICTED positions (legacy PBF step order), so pos_source == 1
    // routes the op to pbf_predicted_pos; 0 == particle_pos (the M5 default).
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
};
inline constexpr uint32_t kGridPosSourceParticlePos = 0u;
inline constexpr uint32_t kGridPosSourcePbfPredicted = 1u;

// --- narrowphase / contact rows -----------------------------------------
// Contact-family selector shared by the narrowphase / assemble / solve params
// (mirrors nk::ContactFamily; a plain u32 so the POD stays header-light).
//   0 = FusedFoot  (M3 articulation foot pipeline, goldens byte-exact)
//   1 = UnionCsr   (M4 union compliant-CSR pipeline)
//   2 = PairDriven (M5 generalized broadphase->narrowphase: BuildAabbs/Lbvh*/
//       candidate_pairs -> NarrowphasePrimitives (amf:: analytic set + sphere x
//       hull) + NarrowphaseSdf (SAMP x SDF grid). ADDITIVE; the union family's
//       slot-template path is untouched. The broadphase + SDF ops EARLY-EXIT
//       for FusedFoot/UnionCsr so those gate-pinned paths stay bit-identical.)
inline constexpr uint32_t kContactFamilyFusedFoot  = 0u;
inline constexpr uint32_t kContactFamilyUnionCsr   = 1u;
inline constexpr uint32_t kContactFamilyPairDriven = 2u;

struct NarrowphasePrimitivesParams {
    float contact_margin;
    uint8_t max_contacts_per_pair;
    // M3b first batch (foot sphere x ground plane, the production Go2/H1 path):
    float    ground_height;
    // Go2-on-stairs Phase 1: the SHARED procedural-terrain params. The FusedFoot
    // detection kernel samples nuka::terrain::SampleTerrainHeight(env_type, x, y,
    // terrain) instead of the scalar ground plane; the per-env terrain TYPE is the
    // env_terrain_type DataView field. DEFAULT (all-zero except ground_height ==
    // ground_height) + every env's type seeded 0 (Flat) => SampleTerrainHeight
    // returns exactly ground_height => byte-identical to the legacy scalar plane
    // (the D1 guarantee).
    ::nuka::terrain::TerrainParams terrain;
    uint32_t foot_count;        // active rows of the Model foot_shape table
    uint32_t env_count;
    uint32_t base_link_count;   // links per env (replica stride)
    // M4 union family (kContactFamilyUnionCsr): per-(env x union-slot) analytic
    // detection (foot sphere x plane / finger sphere x hull / body box x plane).
    uint32_t family;            // kContactFamily*
    uint32_t union_slot_count;  // union slots per env (Model union_slots size)
    uint32_t bodies_per_env;
    uint32_t hull_vert_count;   // live verts of the hull_verts pool
    // M6: particle coupling (the kUSlotParticleSphere* union classes form the
    // particle side's world sphere from particle_pos[env*particles_per_env+link]).
    uint32_t particles_per_env;
};

// Spec-fixed semantic fields (M1): {contact_margin, max_contacts_per_pair}.
// M5 appends the pair-driven launch geometry: the op samples each sampling
// shape's SAMP point slice against the OTHER shape's cooked SDF grid (plan
// §3.5). EARLY-EXITS unless family == kContactFamilyPairDriven.
struct NarrowphaseSdfParams {
    float   contact_margin;
    uint8_t max_contacts_per_pair;
    // -- appended launch geometry (M5) -----------------------------------
    uint32_t family;            // kContactFamily* (PairDriven => sample)
    uint32_t env_count;
    uint32_t bodies_per_env;
    uint32_t max_contacts_per_env;  // ucontact slot stride / env
};

struct ContactTangentBasisParams {
    uint32_t slot_count;        // env_count * max_contacts_per_env
};

struct AssembleRowsParams {
    float    dt;
    uint32_t slot_count;        // total contact slots (env_count * max_contacts)
    uint32_t max_dof;           // chain-Jacobian dof_stride (== dofs_per_env)
    uint32_t env_count;
    uint32_t articulation_count;
    uint32_t total_link_count;
    // M4 union family:
    uint32_t family;            // kContactFamily*
    uint32_t union_slot_count;  // union slots per env
    uint32_t rows_per_env;      // row slots per env (== max_rows_per_env)
    uint32_t bodies_per_env;
    uint32_t base_link_count;   // links per env (replica stride)
    float    solref[2];         // merged contact solref (union family)
    float    solimp[5];         // merged contact solimp (union family)
    uint32_t particles_per_env; // M6 particle coupling (kUSlotParticleSphere*).
};

// Spec-fixed semantic fields (M1): {dt, vel_iters, pos_iters}. The fields BELOW
// the spec triplet are the M3b-precedent LAUNCH-GEOMETRY transport (op_schema
// header note: "ops carry the launch-geometry counts in their params because
// ModelView/DataView are pure pointer aggregates") plus the per-family solver
// constants that are Model properties (the fused family's legacy knobs). They
// are filled by Pipeline::Build from the Model — the three semantic fields
// keep their spec meaning and position.
struct SolveRowsBlockIslandParams {
    float dt;
    uint16_t vel_iters;
    uint16_t pos_iters;
    // -- appended launch geometry + Model-derived solver constants (M4) ------
    uint32_t family;             // kContactFamily*
    uint32_t total_islands;      // union family: schedule island count (grid x)
    uint32_t max_dof;            // dof_stride == the M tile stride
    uint32_t env_count;
    uint32_t articulation_count;
    uint32_t rows_per_env;       // row slots per env (union family)
    uint32_t base_link_count;    // links per env (qdot scatter)
    uint32_t total_body_count;   // env-major rigid body count
    // Fused-family legacy knobs (Model properties; unused by the union family):
    float    friction_coefficient;
    float    baumgarte_max_velocity;
    // 1 => apply the implicit joint-damping seed (drive_damping as per-DOF c_j,
    // requires m_inv == (M + dt*C)^-1). 0 => contacts-only.
    uint32_t apply_implicit_damping;
};

// --- particle (XPBD / PBF) substep --------------------------------------
// M6 prediction-mode selector (the XPBD vs PBF predict semantics DIVERGE — the
// XPBD predict folds gravity into the position WITHOUT mutating velocity and
// snapshots prev_pos, while PBF kicks the velocity by g*dt then predicts into a
// SEPARATE pbf_predicted_pos buffer). The op param carries the mode so the ONE
// op TU dispatches the correct legacy kernel body.
inline constexpr uint32_t kParticleModeNone = 0u;  // no particles (early-exit)
inline constexpr uint32_t kParticleModeXpbd = 1u;  // XPBD soft/cloth predict
inline constexpr uint32_t kParticleModePbf  = 2u;  // PBF fluid predict
// COUPLED mode (M6): particles co-step against rigid/artic bodies through the
// unified row solve (the ParticleInvMass arm). The contact solve corrects the
// particle VELOCITY between the position predict and the position finalize (the
// legacy unified_costep pre/couple/post ordering, reproduced inside the fixed
// pipeline). The pre-contact velocity is saved into the DEDICATED particle_v_pre
// scratch field (NOT pbf_predicted_pos, which the PBF density projection owns in
// coupled mode — see fields.yaml) so ParticleFinalize can compose the PBD
// (XPBD soft-constraint / PBF density) velocity with the contact velocity
// delta — exactly v_final = (pos_projected - prev)/dt + (v_contact - v_pre).
inline constexpr uint32_t kParticleModeCoupled = 3u;
// M9 T11 SOFT+FLUID co-residence mode: ONE Model holds a soft (XPBD) particle
// set in [0, n_soft) and a fluid (PBF) particle set in [n_soft, particles_per_env)
// per env (contiguous [soft | fluid] split, mirrors the co-step's [xpbd | pbf]
// union with split n_x 1:1). The predict/finalize ops branch per-particle on the
// within-env local index vs n_soft (soft => XPBD predict/correct; fluid => PBF
// predict/finalize). The PBF density/lambda/neighbor solve is SCOPED to the fluid
// slice (a soft particle must NOT contribute to fluid density); the XPBD
// constraints are edge-based so they only touch the soft slice. The id-10
// cross-contact (Phase 2) runs over the FULL union.
inline constexpr uint32_t kParticleModeSoftFluid = 4u;

// Common particle launch geometry (the views are pure pointer aggregates, so
// every particle op carries its counts). particle_count == total env-major
// particles; the constraint counts are total env-major XPBD constraint slots.
// Coupled internal-dynamics sub-type (which projected-position buffer the
// coupled finalize composes): none (free point masses) / xpbd (particle_pos) /
// pbf (pbf_predicted_pos). Only meaningful when mode == kParticleModeCoupled.
inline constexpr uint32_t kCoupledInternalNone = 0u;
inline constexpr uint32_t kCoupledInternalXpbd = 1u;
inline constexpr uint32_t kCoupledInternalPbf  = 2u;

struct ParticlePredictParams {
    float    dt;
    float    gravity[3];
    uint32_t mode;             // kParticleMode*
    uint32_t particle_count;   // total env-major particles
    uint32_t coupled_internal; // kCoupledInternal* (coupled mode only)
    // M9 T11 SoftFluid: the per-env [soft | fluid] split + per-env stride. The
    // SoftFluid predict/finalize kernels branch per-particle on (i % per_env) vs
    // n_soft. 0 for the single-system modes (unused).
    uint32_t n_soft_particles; // per-env soft count (split index)
    uint32_t particles_per_env;// per-env particle stride
};

struct XpbdProjectParams {
    float    dt;
    uint16_t iters;            // XPBD Gauss-Seidel sweep count
    uint32_t dist_con_count;   // total env-major distance constraints
    uint32_t bend_con_count;   // total env-major bend constraints
    uint32_t vol_con_count;    // total env-major volume constraints
    // M9 T11: total env-major shape-match cluster count (XPBD id 9). 0 == none.
    // Solved LAST in the XPBD sweep (after dist/bend/vol), the legacy
    // XPBD-sweep order, so it pulls the projected config toward the rigid goal.
    uint32_t shape_match_cluster_count;
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
    float    floor_z;          // boundary floor (the M5 grid is z-up; legacy y).
    // M9 T11 SoftFluid: scope the density/lambda solve to the fluid slice. A soft
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
    // M9 T11 SoftFluid: fluid slice scope (see PbfDensityLambdaParams).
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
    // M9 T11 SoftFluid: the per-env [soft | fluid] split + per-env stride. The
    // SoftFluid finalize branches per-particle (soft => XPBD correct; fluid =>
    // PBF finalize). The polish passes are scoped to the fluid slice. 0 == single.
    uint32_t n_soft_particles;
    uint32_t particles_per_env;
};

// M9 T11 Phase 2 — id-10 cross-system particle-particle CONTACT (the op-ified
// legacy cross-system particle co-step). A class-blind
// unilateral non-penetration row over the FULL [soft | fluid] union (the row math
// does NOT branch on soft vs fluid). Mirrors ParticleParticleContactParams 1:1
// (contact_distance_d_min / compliance_alpha / solver_iterations) + the particle
// launch geometry. Emitted ONLY in kParticleModeSoftFluid (the single-system Xpbd/
// Pbf paths never carry this op, so they stay byte-identical). Reads particle_pos
// (the committed union positions post-finalize) + particle_inv_mass + the union
// grid CSR; writes a per-particle Jacobi half-correction into pbf_position_delta
// (free at the post-finalize slot) then applies it own-index (no float atomics).
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
    // per-env scoping. The id-10 math reads the FULL union (no soft/fluid branch).
    uint32_t n_soft_particles;
    uint32_t particles_per_env;
};

// --- readout / RL substrate ---------------------------------------------
struct ReadoutContactWrenchParams {
    float    dt;                // force = impulse / dt
    uint32_t env_count;
    uint32_t base_link_count;
    uint32_t max_contacts_per_env;
};

struct ExportObsParams {
    uint32_t env_count;
    uint32_t base_link_count;
    uint32_t obs_width;         // floats per env in obs_buffer
};

// M10 RL-completion: ReadoutUnionContactObs — union-only per-env contact obs.
// After a union StepPlanned, write a per-env contact-observation slice into
// obs_buffer for the RL grasp reward. The slice layout per env (at obs_offset,
// inside the per-env obs_buffer row of `obs_width` floats):
//   [ n_fingers finger normal-impulse floats | n_feet foot contact-scalar floats ]
// Each finger entry = the sum of the slot's NORMAL-row impulses (lambda) over
// the finger slot's max_pts normal rows (the force-closure signal). Each foot
// entry = 1.0 if the foot slot has >0 manifold contacts this step else 0.0.
// Strictly ADDITIVE + UnionCsr-gated: never emitted for fused/pair-driven.
struct ReadoutUnionContactObsParams {
    uint32_t env_count;
    uint32_t union_slot_count;  // union slots per env (== max_contacts_per_env)
    uint32_t rows_per_env;      // row slots per env (== max_rows_per_env)
    uint32_t n_feet;            // foot slots [0, n_feet)
    uint32_t n_fingers;         // finger slots [n_feet, n_feet+n_fingers)
    uint32_t obs_offset;        // float offset of the slice in the per-env row
    uint32_t obs_width;         // per-env obs_buffer stride in floats (== 64)
    uint32_t max_pts;           // normal rows per finger slot (sphere-hull == 1)
};

// ResetEnvs: per-env masked snapshot restore. The env-id list is uploaded into
// the reset_env_ids field (scratch) by World::Reset BEFORE the dispatch; count
// is how many leading entries are valid.
struct ResetEnvsParams {
    uint32_t count;
    uint32_t base_link_count;
    uint32_t lambda_stride;     // row slots per env (== max_rows_per_env)
    uint32_t articulation_count;
    // M7 T1: movable rigid-body restore arm. body_count is the PER-ENV body
    // stride (bodies_per_env); ResetEnvsKernel restores each reset env's body
    // slice [env*body_count, env*body_count+body_count) from the snapshot_body_*
    // fields. 0 => no bodies (the articulation-only path stays byte-identical).
    uint32_t body_count;        // bodies per env (snapshot_body_* slice stride)
    // M10 RL-completion: OPTIONAL per-env Philox initial-condition randomization,
    // applied ON TOP of the snapshot restore. ALL fields default-zero (this POD
    // is value-init), and EACH perturbation is gated `if (half != 0)` in the
    // kernel, so an all-zero params (every existing go2/fused caller) yields the
    // VERBATIM snapshot copy — byte-identical to the pre-M10 reset.
    uint64_t ic_seed;           // Philox seed (0 default; combined with ic_episode)
    uint32_t ic_episode;        // bumped each Reset() so successive resets differ
    uint32_t cup_body_index;    // which body slot the cup-XY jitter targets (0)
    float    jitter_cup_xy[2];  // +/- half-range for cup body pose x/y (0 => off)
    float    jitter_base_pos[3];// +/- half-range for base pose x/y/z (0 => off)
    float    jitter_q;          // +/- half-range for each generalized coord (0=off)
};

struct SnapshotStateParams {
    uint32_t total_link_count;
    uint32_t env_count;
    // M7 T1: env-major total movable rigid-body count (bodies_per_env*env_count).
    // OpSnapshotState appends the body_pose/lin/ang D2D copies after the four
    // articulation copies. 0 => no bodies (articulation snapshot byte-identical).
    uint32_t total_body_count;
};

// RestoreState: bulk snapshot -> live restore + clear the carried accumulators
// (qddot / tau / lambda), the legacy batched articulated Reset 1:1.
struct RestoreStateParams {
    uint32_t total_link_count;
    uint32_t env_count;
    uint32_t row_slot_count;    // env_count * max_rows_per_env (lambda clear)
    // M7 T1: env-major total movable rigid-body count (bodies_per_env*env_count).
    // OpRestoreState appends the body_pose/lin/ang snapshot->live copies after
    // the articulation restore. 0 => no bodies (articulation restore unchanged).
    uint32_t total_body_count;
};

// --- domain randomization -----------------------------------------------
struct RandomizeMaterialBucketsParams {
    uint64_t seed;
};

struct RandomizeBodyParamsParams {
    uint64_t seed;
    float range;           // +/- fractional perturbation
};

// --- differentiable rollout (M9 T7) -------------------------------------
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
