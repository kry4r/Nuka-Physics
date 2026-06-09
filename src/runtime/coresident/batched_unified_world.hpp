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

#include "math/vec3.hpp"
#include "phi/device_context.hpp"
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

    // Advance ONE rigid body by the symplectic-Euler position step (gravity has
    // already kicked the velocity). BYTE-IDENTICAL to
    // UnifiedCoResidentStepper::IntegrateBoxPosition so an N=1 world matches the
    // co-resident oracle exactly.
    void IntegrateBodyPosition(runtime::rigid::BodyState& body) const;
};

}  // namespace nuka::runtime::coresident
