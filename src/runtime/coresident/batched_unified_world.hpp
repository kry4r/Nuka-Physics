#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::coresident -- BatchedUnifiedWorld (v0.8 P2). The GENERAL,
// scene-driven, BATCHED world: N parallel envs, each a co-resident articulation
// + movable rigid bodies + statics, stepped TOGETHER through the unified
// collision/contact spine (the SAME UnifiedSolve the single-instance
// UnifiedCoResidentStepper proved). ADDITIVE -- a NEW translation unit.
// ---------------------------------------------------------------------------
// WHY THIS EXISTS. RL training must run in the GENERAL world (owner principle),
// not the standing-specialized BatchedArticulatedWorld (foot<->ground only, no
// movable rigid body). The single-instance UnifiedCoResidentStepper has the
// general two-way contact physics (articulation<->rigid<->static via UnifiedSolve)
// but is NOT batched. This class batches it: N envs on one set of concatenated
// buffers, so a single UnifiedSolve launch advances all envs at once.
//
// THE BATCHING MECHANISM (validated, ZERO solver change -- see the roadmap
// docs/plans/2026-06-09-engine-general-world-assessment-roadmap.md §7). Each env
// gets a GLOBALLY-UNIQUE body-id range: env e's rigid bodies live at flat indices
// [e*k, (e+1)*k) of one env-major BodyState SoA. The row solver's graph coloring
// (RowsConflict, row_scheduler.cu) keys on the raw row body indices, so rows from
// different envs share NO body id -> never conflict -> color into parallel groups
// -> one SolveRowsComponentSweepKernel launch advances all envs (G1d throughput
// increment: one CUDA block per connected component = per env, byte-identical to
// the former single-block sweep restricted to that env; D1: the per-env disjoint
// constraint graphs make the colored PGS provably equal to N independent
// solves). The articulation side uses the SAME convention the
// co-resident emitter uses (synthetic body key = total_body_count + art_index,
// with art_index = the env's articulation index, env-major M^-1 / qdot tiles per
// row_articulation_refs.hpp) so same-articulation rows within an env serialize and
// cross-env articulation rows parallelize.
//
// IT REUSES THE PRODUCTION INTEGRATOR / SOLVER VERBATIM. Articulation dynamics go
// through the shared FeatherstoneAba:: static methods (the same ones
// BatchedArticulatedWorld and UnifiedCoResidentStepper call). Rigid bodies use the
// SAME symplectic-Euler scheme as UnifiedCoResidentStepper::IntegrateBoxPosition
// (gravity velocity-kick before the contact phase; position + quaternion advance
// after) so an N=1 BatchedUnifiedWorld matches the co-resident oracle BYTE-FOR-BYTE.
// Contact resolves through EmitCompliantContactRows -> UnifiedSolve. NO new physics.
//
// INCREMENTAL BUILD (roadmap §7).
//   * P2.1 (THIS): the skeleton + per-env MOVABLE RIGID BODIES under gravity
//     (free-fall, NO contact, NO articulation yet). Establishes the env-major
//     buffer layout + the deterministic step loop. Validated vs the analytic
//     symplectic-Euler trajectory + per-env independence + D1 byte-exact.
//   * P2.2 : batched rigid<->static contact (cup<->ground/table).
//   * P2.3 : batched articulation<->rigid contact (the grasp crux).
//   * P2.4 : scene->bodies builder + full grasp scene + GPU narrowphase + the
//     multi-block batched solver kernel (RL-scale throughput).
//
// ADDITIVE. Does NOT modify BatchedArticulatedWorld, world_stepper, UnifiedSolve,
// any FeatherstoneAba method, the row solver/scheduler, or any golden.
// ---------------------------------------------------------------------------

#include "core/perf/perf_recorder.hpp"  // PerfRecorder (host wall-clock stage attribution)
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"  // ArticulationHostState / DeviceBuffers
#include "runtime/coresident/unified_coresident_stepper.hpp"  // CoResidentFingertip / CoResidentCup
#include "runtime/rigid/body_state.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::coresident {

// The cup XY reset-jitter half-box default (m, ~+/-2.5 cm). Hoisted to namespace
// scope so BatchedSceneTemplate (defined before BatchedUnifiedWorld) can default its
// per-axis jitter fields to it, AND so BatchedUnifiedWorld::kResetCupJitterM (the
// named constant the gate test reads) can alias it -- the SAME literal both places.
inline constexpr float kDefaultResetCupJitterM = 0.025f;

// The per-env scene template, replicated across all envs at construction. P2.1
// scope: rigid bodies only. (P2.2+ extends with articulation proto, fingertips,
// cup hull, and static colliders -- additive fields, no layout churn for the
// rigid SoA which stays the leading env-major block.)
struct BatchedSceneTemplate {
    // The k rigid bodies that make up ONE env (e.g. the cup). Replicated into the
    // env-major SoA at construction; per-env initial-condition perturbation is then
    // applied via BodyMut() before the first Step().
    std::vector<runtime::rigid::BodyState> bodies_per_env;

    // ----- P2.2: the static +Z ground plane the FIRST per-env body rests on -------
    // ADDITIVE; inert by default. P2.2 inserts a batched box x static-plane contact
    // phase: each env's local body 0 is treated as a BOX (the stable C3b BoxPlane
    // narrowphase -- a flat-bottom cup proxy that sidesteps the hull-vs-plane
    // coplanar-bottom-face instability) resting on ONE static world +Z plane at
    // `ground_height`. The box uses `box_half_extent` (per-axis half-extents; a
    // cup-sized cube ~0.045 m is a good default). When `has_ground` is false (the
    // default), NO box<->ground pair is ever emitted -> the contact phase stays empty
    // -> the P2.1 free-fall path is byte-for-byte unchanged. Only local body 0 of each
    // env participates (the one movable rigid body per env in the P2.2 scope).
    bool       has_ground     = false;   // false -> NO contact (pure P2.1 free-fall).
    math::Vec3 box_half_extent{0.045f, 0.045f, 0.045f};  // body-0 box half-extents (m).
    float      ground_height  = 0.0f;    // static +Z plane z (the box bottom rests here).

    // ----- P2.3a: the per-env articulation<->rigid GRASP contact phase ------------
    // ADDITIVE; inert by default (has_grasp=false -> NO grasp pair ever emitted, the
    // P2.1/P2.2 paths are byte-for-byte unchanged). The grasp scope is a FIXED-base
    // 2-finger gripper (a CoResident articulation proto, replicated per env) that
    // pinches ONE movable rigid CUP (a convex hull == one of the bodies_per_env
    // bodies, selected by `cup_local_index`) by FINGER FRICTION ALONE -- NO table.
    // Each step: re-apply the constant grip torque to tau (the drive path) -> ABA ->
    // velocity integrate (gripper + cup gravity kick) -> per-finger sphere x cup-hull
    // narrowphase -> EmitCompliantContactRows(condim=3) -> per-row finger chain-J +
    // cup body index -> ONE UnifiedSolve -> scatter qdot -> position integrate. This
    // is the N=1 batched analog of UnifiedCoResidentStepper::StepGrasp (the validated
    // oracle); P2.3b adds an env loop + env-major tile concatenation, no restructure.
    bool has_grasp = false;  // false -> NO grasp (P2.1/P2.2 path preserved byte-exact).

    // The gripper articulation proto (a CoResident fixed-base 2-finger gripper, built
    // the SAME way the grasp-hold spike's BuildGripper does). Replicated per env.
    articulation::ArticulationHostState gripper_proto;

    // The N fingertip sphere descriptors (link / local offset / radius / broadphase
    // handle) -- the pinch contacts. Kept SPHERES (cvx::SphereHull EPA-bypass).
    std::vector<CoResidentFingertip> fingertips;

    // The movable convex-hull CUP geometry (mesh-local hull verts, COM-centered) +
    // which bodies_per_env body IS the cup (its env-major BodyState carries the live
    // pose / mass / inertia). The cup BodyState lives in bodies_per_env[cup_local_index].
    CoResidentCup cup;
    uint32_t      cup_local_index = 0u;  // which bodies_per_env body is the cup.

    // The constant grip torque + optional drive force limits, PER DEVICE LINK (length
    // == gripper link count; 0 on non-finger links). Re-applied to tau each step.
    std::vector<float> grip_torque;
    std::vector<float> drive_force_limits;

    // Per-contact isotropic friction coefficient (stamped into RowMaterial.friction)
    // + the contact dimension (3 -> 1 normal row + a 4-spoke friction pyramid / point;
    // FRICTION is what holds the cup -- a frictionless condim=1 grasp could not).
    float    friction_mu = 0.6f;
    uint32_t condim      = 3u;

    // ----- G1b UNION: feet x static-ground rows (inert unless has_feet) -----------
    // ADDITIVE; inert by default (has_feet=false -> NO foot pair is ever emitted, so
    // every existing template path is byte-for-byte unchanged). When enabled, the grasp
    // branch ALSO emits foot-sphere x ground-plane rows (the oracle StepStandGrasp's
    // foot row class: ArticulationChainJ <-> StaticNull) into the SAME per-env append ->
    // one-solve loop as the finger rows. Feet are authored ankle contact spheres (the
    // fingertip pattern, spec §4); the foot narrowphase is HOST emission (~4 spheres/env
    // on the trivial C3b SpherePlane -- a GPU foot kernel is the NAMED G1f deferral,
    // gated on a measured throughput finding). Gated on has_feet (NOT a new has_union
    // flag) so the G1 increments compose (G1b feet, G1c fingers+cup, G1d table).
    // Requires has_grasp (the articulation lives there); has_feet without has_grasp is a
    // LOUD construction error, never a silent no-op.
    bool                              has_feet = false;   // false -> no foot pair emitted.
    std::vector<CoResidentFootSphere> feet;               // authored ankle spheres (toe/heel).
    CoResidentGround                  ground;             // static +Z plane (height + bp id).
    float                             foot_mu = 0.8f;     // per-foot isotropic friction.

    // ----- G1d UNION: cup(-proxy-box) x static-TABLE rows (inert unless has_table) --
    // ADDITIVE; inert by default (has_table=false -> NO table pair is ever emitted, so
    // every existing template path is byte-for-byte unchanged). When enabled, the grasp
    // branch ALSO emits ONE cup x table-plane row class per env (the oracle
    // StepStandGrasp's third class: RigidInvMass <-> StaticNull) into the SAME per-env
    // append -> one-solve loop, AFTER the feet + finger rows (the oracle's drive_pairs
    // order, :1455-1459 -- row-layout parity). The cup side uses the FLAT-BOTTOM PROXY
    // box (C3b BoxPlane at the LIVE cup pose; a real mug rests on its flat base --
    // sidesteps the hull-vs-plane coplanar-bottom-rim instability, named engine debt)
    // when any cup_table_proxy_half component > 0; zero half-extents fall back to the
    // detailed hull id (hull x plane). HOST emission like the P2.2 ground branch (one
    // trivial box x plane per env). The mid-run toggle is SetTableEnabled (the lift
    // choreography: cup rests -> hand closes -> table removed -> friction-only hold).
    // has_table without has_grasp is a LOUD construction error (the cup lives there).
    bool       has_table     = false;   // false -> no cup x table pair ever emitted.
    float      table_height  = 0.0f;    // static +Z plane z (the cup bottom rests here).
    float      table_mu      = 0.6f;    // cup<->table friction (cup sits, no slide).
    uint32_t   table_broadphase_id = 8500u;  // distinct from cup/proxy/ground/links.
    math::Vec3 cup_table_proxy_half{};       // box half-extents; all-zero -> hull id.
    math::Vec3 cup_table_proxy_offset{};     // box-center offset in the cup body frame.
    uint32_t   cup_table_proxy_id = 7001u;   // distinct from the cup hull id (7000).

    // ----- A5a: per-axis cup RESET JITTER half-box (m), read by ResetEnvs ----------
    // ResetEnvs perturbs each reset cup's X by +/-reset_jitter_x and its Y by
    // +/-reset_jitter_y (independent uniform draws, X then Y, per-env mt19937_64).
    // BOTH default to kDefaultResetCupJitterM (0.025) so the legacy isotropic +/-2.5 cm
    // jitter is byte-identical (the range-parameterized distributions equal the old
    // single one at the default). Anisotropic settings (e.g. X=0.025, Y=0) let the RL
    // env request jitter along the gripper's ACTUATED (X) axis only -- the un-actuated
    // Y jitter is unsaveable by the X-only gripper, so shrinking it raises the policy's
    // reachable catch rate WITHOUT touching the discriminative timing IC.
    float    reset_jitter_x = kDefaultResetCupJitterM;
    float    reset_jitter_y = kDefaultResetCupJitterM;
};

// Per-env grasp metrics (the P2.3a HOLD / NO-TABLE / BITE gates read this). Populated
// by ResolveBatchedGraspContact -- the SAME quantities StepGrasp reports, computed
// while the rows / lambdas are live (a test CANNOT reconstruct the contact impulse
// after Step() integrates). For N=1 there is one entry; P2.3b makes this per-env.
struct BatchedGraspEnvReport {
    uint32_t finger_contacts = 0u;        // # cup<->fingertip manifold points this step.
    uint32_t finger_row_count = 0u;       // # finger contact rows (normal + spokes).
    bool     any_static_row = false;      // a row carried a StaticNull side (the P2.3a
                                          // NO-TABLE gate's tripwire; with G1b feet the
                                          // foot<->ground rows set it too, mirroring the
                                          // oracle StepStandGrasp :1622).
    double   cup_vertical_impulse = 0.0;  // Σ over finger rows of λ * j_cup.linear.z.
    double   cup_dvz_impulse = 0.0;       // m_cup * (vz_after_solve - vz_pre_contact).
    double   cup_vz = 0.0;                // cup vertical velocity AFTER this step.
    double   cup_z  = 0.0;                // cup height AFTER this step.
    float    max_lambda = 0.0f;           // peak finger normal-row impulse this step.
    // ----- A1 RL substrate: the FORCE-CLOSURE signal (per-fingertip normal impulse) -----
    // Σ over THIS fingertip's NORMAL rows (NOT the friction spokes) of the row lambda --
    // the squeeze impulse each finger delivers. Both fingers' impulse > 0 with the cup
    // held == grasped (not batted by one finger). Bucketed host-side here (the SAME place
    // cup_vertical_impulse / max_lambda are extracted), keyed by the fingertip the row's
    // chain-J slot belongs to; the solver is NOT touched. Indexed by fingertip order
    // (size == num_fingertips); Σ over fingers == max-style aggregate cross-check.
    std::vector<float> finger_normal_impulse;  // per-fingertip normal-row impulse (>=0).
    // ----- G1b UNION: the foot<->ground standing-support fields (the oracle's S3
    // stand fields, CoResidentStepReport hpp:141-142, classified by (react,react)
    // exactly as StepStandGrasp :1730-1738). Zero unless the template has has_feet.
    // At a quasi-static stance foot_normal_impulse_sum balances m_total*g*dt -- the
    // G1b force-balance gate number.
    double   foot_normal_impulse_sum = 0.0;  // Σ over foot NORMAL rows of λ this step.
    uint32_t foot_normal_rows = 0u;          // # foot NORMAL rows (== foot contact count).
    // ----- G1d UNION: the cup<->TABLE support fields (the oracle's table metrics,
    // CoResidentStepReport hpp:123-130, classified by (react,react) -- a cup x table
    // row is (RigidInvMass, StaticNull), distinct from BOTH the foot class (ChainJ,
    // StaticNull) and the finger class (ChainJ, RigidInvMass) -- exactly as the
    // StepStandGrasp report walk :1717-1728). Zero unless the template has has_table.
    // While the cup RESTS, table_vertical_impulse balances m_cup*g*dt; after
    // SetTableEnabled(false) it is EXACTLY 0 (no rows) and the finger vertical
    // impulse must close the triangle -- the G1d lift-choreography gate numbers.
    uint32_t table_row_count = 0u;       // # cup<->table NORMAL rows this step.
    float    table_lambda = 0.0f;        // max cup<->table NORMAL-row impulse.
    double   table_vertical_impulse = 0.0;  // Σ over ALL table rows of λ * j_cup.z.
};

// ----- A1 RL substrate: the batched per-step OBS readout (the throughput-critical piece) ---
// Every DEVICE-resident quantity the downstream H1GraspEnv force-closure-HOLD reward needs,
// exported in ONE bulk device->host copy per buffer for ALL envs (no per-env loop -> the
// P2.4b sync-storm elimination is preserved). Cup pose/vel stay on the host accessors
// Body(env,cup_local) / GraspReports(); this struct carries ONLY the device-resident signals.
// All vectors are env-major; sized once by ExportObsState and reused.
struct ObsStateBatch {
    std::vector<float> q;                   // finger joint q       [dof_stride per env].
    std::vector<float> qdot;                // finger joint qdot    [dof_stride per env].
    std::vector<float> fingertip_world_pos; // fingertip world xyz  [3*num_fingertips per env].
    std::vector<float> finger_normal_impulse;  // per-finger normal impulse [num_fingertips per env].
    // ----- G2 UNION additions (ADDITIVE; zero-filled for a Fixed-root gripper) -----
    // The articulation root state the whole-body RL obs needs, exported from the SAME
    // bulk downloads (base_pose: one small env-major Transform copy; base_vel: the
    // root slot of the link_velocity download ExportObsState already does when
    // base_dof > 0). For a FIXED-root articulation base_pose still reports the live
    // (constant) root transform; base_vel stays zero (the fixed root never moves).
    std::vector<float> base_pose;  // [7 per env] px,py,pz, qw,qx,qy,qz.
    std::vector<float> base_vel;   // [6 per env] root spatial velocity (== qdot cols 0..5
                                   // for a FloatingBase root; zeros for a Fixed root).
};

// The batched general world. Owns N envs of co-resident state and advances them in
// lockstep. P2.1: only the per-env rigid BodyState block exists.
class BatchedUnifiedWorld {
public:
    BatchedUnifiedWorld(const phi::DeviceContext& context,
                        const BatchedSceneTemplate& scene_template,
                        uint32_t env_count,
                        float gravity_z,
                        float dt);

    // Advance EVERY env one step. P2.1: per-env rigid gravity velocity-kick +
    // symplectic-Euler position/orientation integrate (NO contact). Deterministic
    // (D1): the bodies are advanced in fixed env-major index order, no atomics.
    // A1 (grasp): applies the LAST actions set by SetActions (the template grip torque
    // until the first SetActions call), so a world that never calls SetActions is
    // BYTE-IDENTICAL to today (GATE-1).
    void Step();

    // ----- A1 RL substrate: PER-STEP PER-ENV ACTION INJECTION --------------------
    // Set the per-env per-finger DRIVE TORQUE for ALL envs from a host array in ONE
    // device upload. `actions` is env_count*DofStride() floats ENV-MAJOR: action
    // [e*DofStride()+d] is env e's drive torque on the gripper joint at flat-qdot DOF
    // index d (the SAME prefix-sum index DofIndexOf produces -- so d maps to a finger
    // prismatic joint). The torque is scattered into the env-major per-link drive buffer
    // the grip kernel reads (global link e*BaseLinkCount()+link), REPLACING the constant
    // template grip torque. No-op when !has_grasp_ or count != env_count*DofStride().
    // NEVER a per-env host loop -- one cudaMemcpy-class upload. After SetActions, Step()
    // uses these actions every step until the next SetActions / ResetEnvs.
    void SetActions(const float* actions, size_t count);
    // Convenience: set actions then advance one step (the RL per-step driver).
    void Step(const float* actions, size_t count) {
        SetActions(actions, count);
        Step();
    }
    // The per-env action width (== the gripper's joint DOF count). Actions are
    // env_count*ActionDim() floats env-major.
    uint32_t ActionDim() const { return dof_stride_; }
    uint32_t NumFingertips() const { return static_cast<uint32_t>(fingertips_.size()); }

    uint32_t EnvCount() const { return env_count_; }
    uint32_t BodiesPerEnv() const { return bodies_per_env_; }

    // The flat env-major index of env e's local body i: e*BodiesPerEnv() + i.
    uint32_t BodyIndex(uint32_t env, uint32_t local) const {
        return env * bodies_per_env_ + local;
    }

    // Read / mutate one env's rigid body (env-major). BodyMut() is for per-env
    // initial-condition setup / perturbation before stepping.
    const runtime::rigid::BodyState& Body(uint32_t env, uint32_t local) const {
        return bodies_[BodyIndex(env, local)];
    }
    runtime::rigid::BodyState& BodyMut(uint32_t env, uint32_t local) {
        return bodies_[BodyIndex(env, local)];
    }

    // The whole env-major rigid SoA (read-only). Size == env_count * bodies_per_env.
    const std::vector<runtime::rigid::BodyState>& Bodies() const { return bodies_; }

    // ----- P2.3a grasp observability (the gates read these) ----------------------
    // The per-env grasp report from the LAST Step() (populated only when has_grasp_).
    // For N=1 size()==1. The cup is bodies_per_env[cup_local_index].
    const std::vector<BatchedGraspEnvReport>& GraspReports() const {
        return grasp_reports_;
    }
    // Snapshot env e's gripper articulation host state (q / qdot / base_pose /
    // link_velocity). For per-env byte/tolerance comparison of the gripper joints. A
    // no-op (leaves *out unchanged) when !has_grasp_ or env >= env_count_.
    void DownloadGripper(uint32_t env, articulation::ArticulationHostState* out) const;

    // ----- G1b: per-env FLOATING-BASE initial-condition seam ----------------------
    // Overwrite env e's articulation base pose on the consolidated env-major device
    // (base_pose[e]; the FloatingBase root's live transform). The articulation analog
    // of BodyMut(): per-env IC setup BEFORE stepping (e.g. the G1b MIXED gate seats one
    // env's base high so its feet never touch -- the foot-airborne tile-gap env; the
    // future RL reset randomizes base ICs through the same seam). One bulk
    // download-modify-upload of the base_pose buffer, OFF the per-step path. No-op when
    // !has_grasp_ or env >= env_count_. NOT called by any pre-G1b path -> byte-compat.
    void SetGripperBasePose(uint32_t env, const math::Transform& pose);
    // Convenience: env 0 (the N=1 case). Keeps the existing A1/A2 call sites unchanged.
    void DownloadGripper(articulation::ArticulationHostState* out) const {
        DownloadGripper(0u, out);
    }

    // ----- G1d: the mid-run TABLE toggle (mirrors the oracle's SetTableEnabled, the
    // lift choreography's "remove the table" step: the next Step() stops emitting the
    // cup x table pair, table_vertical_impulse goes EXACTLY to 0, and the finger
    // friction must carry the cup alone). Per-env state (a host byte vector -- OFF
    // the per-step hot path; the emission gate reads it per env). The all-envs form
    // is the N=1 / lockstep-choreography call; the per-env form serves MIXED gates +
    // future RL stage resets. Emission ALSO requires has_table (the oracle's
    // emit_table = has_table && table_enabled_ && have_cup), so toggling a
    // has_table=false world stays inert -> byte-compat.
    void SetTableEnabled(bool enabled);                 // all envs.
    void SetTableEnabled(uint32_t env, bool enabled);   // one env.
    bool TableEnabled(uint32_t env = 0u) const {
        return env < table_enabled_.size() && table_enabled_[env] != 0u;
    }

    // ----- A1 RL substrate: BATCHED OBS EXPORT (the per-step RL readout) ----------
    // Fill `out` with every DEVICE-resident obs signal for ALL envs in ONE bulk
    // device->host copy per buffer (q, qdot, fingertip_world_pos download from the
    // env-major device buffers; finger_normal_impulse is read from the LAST Step()'s
    // host-side report aggregation). NEVER a per-env DownloadGripper loop -- that would
    // re-create the O(N) sync storm P2.4b eliminated. Sizes `out`'s vectors once
    // (env_count*DofStride / env_count*3*NumFingertips / env_count*NumFingertips) and
    // reuses them across calls. Cup pose/vel come from Body(env,cup_local) /
    // GraspReports() (already host-resident). No-op (clears `out`) when !has_grasp_.
    void ExportObsState(ObsStateBatch& out) const;

    // ----- A1 RL substrate: PER-ENV AUTORESET with deterministic randomization ----
    // For each env in `env_ids`: (a) restore the cup BodyState to a seed-randomized
    // initial pose (cup XY jittered within +/-kResetCupJitterM of the template grasp
    // position, cup Z + orientation reset to the template, velocity zeroed); (b) restore
    // that env's gripper device slice to the template (q to the nominal open config,
    // qdot=0, link_velocity=0). The device write is BATCHED over the listed envs (one
    // consolidated download-modify-upload, NOT a per-env synchronous round-trip), kept
    // OFF the per-step critical path. Randomization is DETERMINISTIC given `seed`
    // (per-env std::mt19937 seeded from seed XOR env-id) so two ResetEnvs calls with the
    // same seed produce identical states (D1). No-op when !has_grasp_.
    void ResetEnvs(const std::vector<uint32_t>& env_ids, uint64_t seed);
    // The DEFAULT cup XY randomization half-box (m) ResetEnvs jitters within (small,
    // ~+/-2.5 cm). The LIVE per-axis magnitudes are reset_jitter_x_/reset_jitter_y_
    // (from the scene template, each defaulting to THIS); this named constant is kept
    // for the gate test's box check and as the canonical default literal.
    static constexpr float kResetCupJitterM = kDefaultResetCupJitterM;

    // ----- P2.4a perf-gate instrumentation (ADDITIVE; physics-neutral) -----------
    // Per-tag HOST WALL-CLOCK aggregator (mirrors BatchedArticulatedWorld::Perf). The
    // Step / ResolveBatchedGraspContact stages already Synchronize internally, so a
    // chrono RAII timer bracketing each stage captures the REAL host round-trip cost
    // (the per-env download/upload + CopyToHost storms a CUDA-event timer would miss).
    // Canonical disjoint tags: pose_download / artic_download / crba_minv /
    // chain_jacobian / aba_integrate / narrowphase / row_assembly / row_solver /
    // scatter_integrate. The recorder is a pure-host sample bucket (no CUDA, no float
    // ops, no extra syncs) -> the 16 correctness gates stay byte-exact. A perf TU reads
    // this to test the "articulation sync storm dominates, row_solver is negligible"
    // hypothesis (roadmap §7). Use Perf().Reset() to clear warm-up samples.
    core::perf::PerfRecorder& Perf() { return perf_; }
    const core::perf::PerfRecorder& Perf() const { return perf_; }

private:
    const phi::DeviceContext& context_;
    uint32_t env_count_;
    uint32_t bodies_per_env_;
    float gravity_z_;
    float dt_;

    // ----- P2.2 static ground (inert unless has_ground_) -------------------------
    bool       has_ground_;
    math::Vec3 box_half_extent_;
    float      ground_height_;

    // ----- P2.3a/P2.3b/P2.4b articulation<->rigid GRASP (inert unless has_grasp_) -------
    // P2.4b: ONE consolidated env-major device-resident multi-gripper state (env_device_,
    // articulation_count == env_count_, built once from ReplicateArticulationHostState).
    // REPLACES the P2.3b std::vector<ArticulationDeviceBuffers> (one device per env), whose
    // per-finger-per-env upload+sync+copyback was the P2.4a sync storm. The grasp resolver now
    // runs the grip drive / ABA / IntegrateVelocity / FK / CRBA / chain-J as BATCHED launches
    // over all N grippers in ONE call each; the live host snapshot is downloaded ONCE per step
    // (sliced per env for the host narrowphase / scatter). The replicated proto -> uniform
    // dof_stride / base_link_count_ across envs, which is WHAT makes the env-major M^-1 (@
    // e*dof_stride^2) / qdot (@ e*dof_stride) tiling valid (the row solver / batched CRBA index
    // those buffers by art_index = e); asserted at construction.
    bool       has_grasp_ = false;
    uint32_t   cup_local_index_ = 0u;
    uint32_t   dof_stride_  = 0u;
    uint32_t   base_dof_    = 0u;   // root DOF count (FloatingBase=6, Fixed gripper=0).
    uint32_t   root_link_   = 0u;
    uint32_t   link_count_  = 0u;   // per-env (single-articulation) gripper link count.
    float      friction_mu_ = 0.6f;
    uint32_t   condim_      = 3u;
    // ----- G1b UNION: feet x static-ground (inert unless has_feet_) ----------------
    // Captured verbatim from the template. The per-env foot emission lives in the grasp
    // branch (the feet share the env's articulation); has_feet_ without has_grasp_ is
    // rejected LOUDLY at construction (a silent no-op would be a dishonest scene).
    bool       has_feet_ = false;
    std::vector<CoResidentFootSphere> feet_;       // authored ankle spheres.
    CoResidentGround foot_ground_;                 // static +Z plane (height + bp id).
    float      foot_mu_ = 0.8f;                    // per-foot isotropic friction.
    // ----- G1d UNION: cup(-proxy) x static-table (inert unless has_table_) ----------
    // Captured verbatim from the template. The per-env table emission lives in the
    // grasp branch (the cup lives there); has_table_ without has_grasp_ is rejected
    // LOUDLY at construction. table_enabled_ is the per-env LIVE toggle (init ==
    // has_table_, the oracle's table_enabled_ = stand_grasp.has_table).
    bool       has_table_ = false;
    float      table_height_ = 0.0f;
    float      table_mu_ = 0.6f;
    uint32_t   table_broadphase_id_ = 8500u;
    math::Vec3 cup_table_proxy_half_{};
    math::Vec3 cup_table_proxy_offset_{};
    uint32_t   cup_table_proxy_id_ = 7001u;
    std::vector<uint8_t> table_enabled_;           // per-env live toggle (host-only).
    // A5a: the LIVE per-axis cup reset-jitter half-box (m), captured from the scene
    // template. ResetEnvs draws X within +/-reset_jitter_x_ then Y within
    // +/-reset_jitter_y_. Both default to kDefaultResetCupJitterM -> the legacy
    // isotropic +/-2.5 cm jitter is byte-identical (range-parameterized distributions).
    float      reset_jitter_x_ = kDefaultResetCupJitterM;
    float      reset_jitter_y_ = kDefaultResetCupJitterM;
    articulation::ArticulationHostState gripper_proto_;      // refresh-able CPU mirror (ONE articulation).
    // ----- P2.4b: ONE consolidated env-major device-resident multi-gripper state ------
    // The P2.4b throughput increment. The per-env std::vector<ArticulationDeviceBuffers>
    // (P2.3b) became a per-finger-per-env upload+sync+copyback STORM (P2.4a measured
    // chain_jacobian=53.4% at N=32). It is REPLACED by ONE ArticulationDeviceBuffers
    // holding env_count_ replicated grippers (built ONCE at construction from
    // ReplicateArticulationHostState(gripper_proto_, env_count_), articulation_count ==
    // env_count_, replica e's links at [e*base_link_count_, (e+1)*base_link_count_)),
    // mirroring BatchedArticulatedWorld::device_ VERBATIM. The already-per-articulation-
    // batched kernels (FeatherstoneAba::*, UpdateWorldLinkPoses, ComputeArticulationInertiaM,
    // FactorArticulationInertiaM, ComputeContactChainJacobians) then advance ALL N envs in
    // ONE launch each over this state -- NO per-step re-upload (the per-step work is the
    // in-place tau/q/qdot update + batched launches). The host snapshot env_live_ is
    // downloaded ONCE per step (consolidated) for the host narrowphase / scatter.
    articulation::ArticulationDeviceBuffers env_device_;  // ONE env-major multi-gripper device.
    uint32_t   base_link_count_ = 0u;  // per-env (proto) link count; replica stride.
    std::vector<CoResidentFingertip> fingertips_;
    CoResidentCup grasp_cup_;
    phi::Buffer   grip_torque_dev_;   // env-major per-link constant grip torque (device).
    phi::Buffer   grip_limits_dev_;   // env-major per-link drive force limits (device).
    // ----- A1 RL substrate: per-step per-env ACTION drive (replaces grip_torque_dev_ at
    // the drive call). Env-major per-link layout (length env_count_*base_link_count_, the
    // SAME the kernel reads), DEFAULT-INITIALIZED to a byte-identical copy of the replicated
    // template grip torque -> a world that never calls SetActions drives EXACTLY the constant
    // grip (GATE-1 byte-identity). SetActions scatters env_count_*dof_stride_ host actions
    // (env-major, DOF-indexed) into this buffer in ONE upload. action_torque_host_ is the
    // CPU mirror used by SetActions to map DOF index -> global link before the upload.
    phi::Buffer        action_torque_dev_;   // env-major per-link drive torque (the LIVE drive).
    std::vector<float> action_torque_host_;  // CPU mirror (env_count_*base_link_count_).
    // The gripper joint-DOF -> articulation-LOCAL link map (length dof_stride_): flat-qdot
    // DOF index d is driven on link dof_to_link_[d]. Built once from the proto's joint types
    // (the inverse of DofIndexOf restricted to single-DOF prismatic/revolute finger joints).
    std::vector<uint32_t> dof_to_link_;
    // ----- A1 RL substrate: the reset template (the nominal cup IC + the proto open config) ---
    // The per-env cup BodyState the scene was seeded with (template body cup_local_index_),
    // captured at construction so ResetEnvs can restore + jitter it without a scene rebuild.
    runtime::rigid::BodyState reset_cup_template_;
    // ----- A1 RL substrate: retained fingertip WORLD positions (host-resident) -------------
    // The resolver computes finger_centers = lp.position + lp.rotation.Rotate(ft.local_offset)
    // ON THE HOST during the narrowphase prep (from the ONE batched FK download). We retain
    // them here env-major [num_fingertips*3 per env] in that same loop, so ExportObsState
    // surfaces them with NO extra device round-trip (they are already host-computed). Reused
    // each Step(). mutable so the const ExportObsState reads them; the resolver fills them.
    mutable std::vector<float> fingertip_world_host_;  // env_count_*num_fingertips*3 (env-major).
    // ----- A1 RL substrate: ExportObsState scratch (reused; the bulk q/qdot/link_vel download) ---
    // ExportObsState copies ONLY the q / qdot / link_velocity device buffers it actually reads
    // (NOT a full DownloadArticulationState of all ~20 SoA arrays) into these env-major host
    // mirrors via direct CopyToHost -- the "ONE bulk download from the env-major device buffers"
    // contract. Sized once at construction (env_count_*base_link_count_); mutable so the const
    // ExportObsState can fill them.
    mutable std::vector<float> obs_q_scratch_;       // env_count_*base_link_count_ (device q).
    mutable std::vector<float> obs_qdot_scratch_;    // env_count_*base_link_count_ (device qdot).
    mutable std::vector<articulation::LinkSpatialVel> obs_linkvel_scratch_;  // env-major link vel.
    // G2: base_pose download scratch (env_count_ Transforms -- one per articulation;
    // tiny). Filled by ExportObsState for the base_pose/base_vel obs fields.
    mutable std::vector<math::Transform> obs_basepose_scratch_;
    // ----- P2.4b persistent batched articulation scratch (allocated once, reused) ------
    // Mirrors BatchedArticulatedWorld's world_pose_/m_/m_inv_/composite_ layout. world_pose_
    // is Transform[env_count_*base_link_count_] (ONE batched UpdateWorldLinkPoses output);
    // m_/m_inv_ are env-major CRBA tiles float[env_count_*dof_stride_^2] (one batched
    // ComputeArticulationInertiaM/Factor over all N); composite_ is the CRBA scratch.
    phi::Buffer   world_pose_dev_;    // Transform[env_count_*base_link_count_] (batched FK).
    phi::Buffer   crba_composite_;    // LinkSpatialInertia[env_count_*base_link_count_] (CRBA scratch).
    phi::Buffer   crba_m_;            // float[env_count_*dof_stride_^2] (CRBA M tiles).
    phi::Buffer   crba_m_inv_;        // float[env_count_*dof_stride_^2] (CRBA M^-1 tiles).
    std::vector<BatchedGraspEnvReport> grasp_reports_;  // last-Step() per-env metrics.

    // ----- P2.4a perf instrumentation (ADDITIVE; mutable -> const Step path can time) ---
    // Host wall-clock per-tag aggregator. mutable so the timer scopes can feed it without
    // making Step() non-const (Step() is already non-const, but the file-static stage
    // call-sites are bracketed via a member-visible recorder). Pure-host: no physics.
    mutable core::perf::PerfRecorder perf_;

    // The env-major rigid BodyState SoA: env e's bodies at [e*k, (e+1)*k). This is
    // the leading block of the concatenated buffer the batched UnifiedSolve will
    // consume in P2.2+; the per-env body-id offset is exactly BodyIndex().
    std::vector<runtime::rigid::BodyState> bodies_;

    // ----- P2.2: the batched box<->static-ground contact phase -------------------
    // For EVERY env e (in fixed env-major order, D1): narrowphase env e's local body 0
    // (a BOX) against the ONE static +Z plane -> EmitCompliantContactRows (condim=1,
    // APPENDING to the shared rows/sides) -> OVERWRITE the appended rows' body indices
    // (rigid side -> BodyIndex(e,0); static side -> kInvalidBodyIndex). After ALL envs,
    // ONE UnifiedSolve over the concatenated rows + the full env-major `bodies_`
    // mutates the velocities in place. Mirrors UnifiedCoResidentStepper::Step()'s
    // box<->ground branch EXACTLY (same PrimParams, same cfg/inputs, same body-index
    // wiring) so an N=1 world is byte-identical to the single-instance reference. A
    // no-op when !has_ground_ (P2.1 free-fall path preserved).
    void ResolveBatchedGroundContact();

    // ----- P2.3a: the batched articulation<->rigid GRASP contact phase -----------
    // The N=1 batched analog of UnifiedCoResidentStepper::StepGrasp. Mirrors the
    // oracle's exact stage order: (stage 0) re-apply the constant grip torque to tau
    // via LaunchApplyTorqueDriveKernels (the DRIVE path); (stage 1/2) ABA accelerations
    // WITH the grip tau; (stage 3) velocity integrate (gripper halves + cup gravity
    // kick); (stage 4-6) download live state + FK poses -> per-finger sphere x cup-hull
    // narrowphase; (stage 7) EmitCompliantContactRows(condim=3) + stamp friction_mu;
    // wire each finger row (cup side -> BodyIndex(e,cup_local), finger side -> coloring
    // key total_body_count+art_index + a chain-J slot, art_index=e); build minv (CRBA
    // M^-1) + the flat prefix-sum qdot; (stage 10) ONE UnifiedSolve; scatter qdot ->
    // device link_velocity/qdot. The POSITION integrate (gripper device IntegratePosition
    // + IntegrateFloatingBasePose + cup IntegrateBodyPosition) happens in Step() after.
    // A no-op when !has_grasp_ (the P2.1/P2.2 paths stay byte-for-byte unchanged). The
    // cup's vertical velocity BEFORE the contact solve is returned so a HOLD gate can
    // read the force balance (the finger vertical impulse must lift it back).
    void ResolveBatchedGraspContact();

    // The flat-qdot prefix-sum DOF index of a gripper device link: Σ JointDofCount over
    // the gripper links in [root_link_, link). REPLICATED byte-for-byte from
    // UnifiedCoResidentStepper::DofIndexOf (a member reading the proto's joint_type).
    uint32_t DofIndexOf(uint32_t link) const;

    // Advance ONE rigid body by the symplectic-Euler position step (gravity has
    // already kicked the velocity). BYTE-IDENTICAL to
    // UnifiedCoResidentStepper::IntegrateBoxPosition so an N=1 world matches the
    // co-resident oracle exactly.
    void IntegrateBodyPosition(runtime::rigid::BodyState& body) const;
};

}  // namespace nuka::runtime::coresident
