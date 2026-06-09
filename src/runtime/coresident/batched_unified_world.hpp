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
// -> a single SolveRowsSweepKernel advances all envs, deterministically (D1: the
// per-env disjoint constraint graphs make the colored PGS provably equal to N
// independent solves). The articulation side uses the SAME convention the
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
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"  // ArticulationHostState / DeviceBuffers
#include "runtime/coresident/unified_coresident_stepper.hpp"  // CoResidentFingertip / CoResidentCup
#include "runtime/rigid/body_state.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::coresident {

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
};

// Per-env grasp metrics (the P2.3a HOLD / NO-TABLE / BITE gates read this). Populated
// by ResolveBatchedGraspContact -- the SAME quantities StepGrasp reports, computed
// while the rows / lambdas are live (a test CANNOT reconstruct the contact impulse
// after Step() integrates). For N=1 there is one entry; P2.3b makes this per-env.
struct BatchedGraspEnvReport {
    uint32_t finger_contacts = 0u;        // # cup<->fingertip manifold points this step.
    uint32_t finger_row_count = 0u;       // # finger contact rows (normal + spokes).
    bool     any_static_row = false;      // a row carried a StaticNull side (a table!).
    double   cup_vertical_impulse = 0.0;  // Σ over finger rows of λ * j_cup.linear.z.
    double   cup_dvz_impulse = 0.0;       // m_cup * (vz_after_solve - vz_pre_contact).
    double   cup_vz = 0.0;                // cup vertical velocity AFTER this step.
    double   cup_z  = 0.0;                // cup height AFTER this step.
    float    max_lambda = 0.0f;           // peak finger normal-row impulse this step.
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
    void Step();

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
    // Convenience: env 0 (the N=1 case). Keeps the existing A1/A2 call sites unchanged.
    void DownloadGripper(articulation::ArticulationHostState* out) const {
        DownloadGripper(0u, out);
    }

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
