#pragma once
// ---------------------------------------------------------------------------
// PHI v2 — op schema: the closed enumeration of physics ops the backend layer
// dispatches, plus one POD `<Op>Params` struct per op.
//
// ggml/llama.cpp analogy: NkOp is the GGML_OP enum; each <Op>Params is the
// op-specific parameter block that travels with an OpCall (phi/backend.hpp).
//
// CONTRACT (M1, frozen plan spec):
//   * NkOp has exactly 28 named ops + a trailing `Count` sentinel.
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

    // --- domain randomization -------------------------------------------
    RandomizeMaterialBuckets, // per-env material bucket randomization
    RandomizeBodyParams,      // per-env body/inertia randomization

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
    // M3 BatchedArticulatedWorld path). 1 = direct torque drive (the union
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
    uint32_t grid_dims[3];      // grid resolution
};

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
    uint32_t foot_count;        // active rows of the Model foot_shape table
    uint32_t env_count;
    uint32_t base_link_count;   // links per env (replica stride)
    // M4 union family (kContactFamilyUnionCsr): per-(env x union-slot) analytic
    // detection (foot sphere x plane / finger sphere x hull / body box x plane).
    uint32_t family;            // kContactFamily*
    uint32_t union_slot_count;  // union slots per env (Model union_slots size)
    uint32_t bodies_per_env;
    uint32_t hull_vert_count;   // live verts of the hull_verts pool
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
struct ParticlePredictParams {
    float dt;
    float gravity[3];
};

struct XpbdProjectParams {
    float dt;
    uint16_t iters;
};

struct PbfDensityLambdaParams {
    float rest_density;
    float relaxation;      // CFM-style epsilon
};

struct PbfApplyDeltaParams {
    uint32_t reserved;
};

struct ParticleFinalizeParams {
    float dt;
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

// ResetEnvs: per-env masked snapshot restore. The env-id list is uploaded into
// the reset_env_ids field (scratch) by World::Reset BEFORE the dispatch; count
// is how many leading entries are valid.
struct ResetEnvsParams {
    uint32_t count;
    uint32_t base_link_count;
    uint32_t lambda_stride;     // row slots per env (== max_rows_per_env)
    uint32_t articulation_count;
};

struct SnapshotStateParams {
    uint32_t total_link_count;
    uint32_t env_count;
};

// RestoreState: bulk snapshot -> live restore + clear the carried accumulators
// (qddot / tau / lambda), the legacy BatchedArticulatedWorld::Reset 1:1.
struct RestoreStateParams {
    uint32_t total_link_count;
    uint32_t env_count;
    uint32_t row_slot_count;    // env_count * max_rows_per_env (lambda clear)
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
