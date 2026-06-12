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
struct ApplyDrivesParams {
    float dt;
};

struct AbaForwardParams {
    float gravity[3];      // world-frame gravity vector
};

struct IntegrateVelocityParams {
    float dt;
};

struct FkWorldPosesParams {
    uint32_t reserved;
};

struct IntegratePositionParams {
    float dt;
};

struct CrbaComputeMParams {
    uint32_t reserved;
};

struct CrbaFactorMParams {
    uint32_t reserved;
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
};

// Spec-fixed (M1).
struct NarrowphaseSdfParams {
    float contact_margin;
    uint8_t max_contacts_per_pair;
};

struct ContactTangentBasisParams {
    uint32_t reserved;
};

struct AssembleRowsParams {
    float dt;
};

// Spec-fixed (M1).
struct SolveRowsBlockIslandParams {
    float dt;
    uint16_t vel_iters;
    uint16_t pos_iters;
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
    uint32_t reserved;
};

struct ExportObsParams {
    uint32_t reserved;
};

struct ResetEnvsParams {
    uint32_t reserved;
};

struct SnapshotStateParams {
    uint32_t reserved;
};

struct RestoreStateParams {
    uint32_t reserved;
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
