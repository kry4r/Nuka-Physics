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
};

struct FkWorldPosesParams {
    uint32_t articulation_count;
    uint32_t total_link_count;
};

struct IntegratePositionParams {
    float    dt;
    uint32_t total_link_count;
    uint32_t articulation_count;
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
struct BuildAabbsParams {
    float margin;          // AABB inflation margin
};

struct LbvhBuildParams {
    uint32_t reserved;
};

struct LbvhQueryPairsParams {
    uint32_t max_pairs;    // capacity of the output pair stream
};

struct ParticleGridBuildParams {
    float cell_size;       // uniform grid cell edge length
};

// --- narrowphase / contact rows -----------------------------------------
struct NarrowphasePrimitivesParams {
    float contact_margin;
    uint8_t max_contacts_per_pair;
    // M3b first batch (foot sphere x ground plane, the production Go2/H1 path):
    float    ground_height;
    uint32_t foot_count;        // active rows of the Model foot_shape table
    uint32_t env_count;
    uint32_t base_link_count;   // links per env (replica stride)
};

// Spec-fixed (M1).
struct NarrowphaseSdfParams {
    float contact_margin;
    uint8_t max_contacts_per_pair;
};

struct ContactTangentBasisParams {
    uint32_t slot_count;        // env_count * max_contacts_per_env
};

struct AssembleRowsParams {
    float    dt;
    uint32_t slot_count;
    uint32_t max_dof;           // chain-Jacobian dof_stride (== dofs_per_env)
    uint32_t env_count;
    uint32_t articulation_count;
    uint32_t total_link_count;
};

// Spec-fixed (M1).
struct SolveRowsBlockIslandParams {
    float dt;
    uint16_t vel_iters;
    uint16_t pos_iters;
};

// TRANSITIONAL (M3b -> deleted in M4): the params of the ported legacy fused
// solver `SolveArticulatedContactRows` (ops/solve_articulated.cu), which is
// ROUTED THROUGH THE SolveRowsBlockIsland PIPELINE SLOT as its interim
// implementation (the §3.2 enum has no transitional slot). The M4
// SolveRowsBlockIslandParams above stays spec-fixed and untouched; M4 replaces
// the interim op + this struct together.
struct SolveArticulatedParams {
    float    dt;
    float    friction_coefficient;
    float    baumgarte_max_velocity;
    uint32_t max_dof;             // dof_stride == the M tile stride
    uint32_t articulation_count;
    uint32_t env_count;
    // 1 => apply the implicit joint-damping seed (drive_damping as per-DOF c_j,
    // requires m_inv == (M + dt*C)^-1). 0 => contacts-only (legacy pre-implicit).
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
