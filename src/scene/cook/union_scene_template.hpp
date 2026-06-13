#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::coresident -- H1UnionDriveEntry + UnionSceneTemplate.
//
// COOK-PRODUCT PLAIN-DATA STRUCTS, relocated here in M9 T3 (controller
// refinement) out of the DOOMED src/runtime/coresident/batched_unified_world.hpp.
// They are the PRODUCT TYPES of the .nks union cook (src/scene/cook/union_cook,
// CookSceneToUnionTemplate -> CookedUnionScene.tmpl IS a UnionSceneTemplate,
// .drive_hold/rest/close ARE std::vector<H1UnionDriveEntry>), consumed identically
// by BuildNkUnionModel + BatchedUnifiedWorld + the C-ABI union/grasp worlds + the
// parity/perf/grasp gates. They are PLAIN DATA -- no kernels, no device handles,
// no phi::Buffer, no CUDA types -- which is exactly WHY they can live under
// src/scene/cook (the host-only cook directory) while their old home in the
// coresident dir is deleted.
//
// WHY THE MOVE. M9 deletes the coresident directory WHOLE (incl. the
// BatchedUnifiedWorld class) at T11. union_cook MUST NOT depend on that doomed
// directory (the decouple), and the C-ABI union path + the surviving union gates
// must keep reading these product types. Relocating the two plain-data structs
// here (a structural move ONLY -- byte-identical field layout, same order, same
// types, same nuka::runtime::coresident namespace so every consumer's
// `coresident::` qualifier keeps resolving) unblocks the coresident-dir delete
// without building the not-yet-existent native CookToModel->UnionCsr cook.
//
// LIFETIME. Alive until the union path migrates to the native nk cook (the M9 exit
// gate zeroes UnionSceneTemplate's grep when CookToModel->UnionCsr lands). The
// coresident dir + the BatchedUnifiedWorld class die at T11; these structs survive
// here (still consumed) until that native cook replaces them. The transitional
// batched_unified_world.hpp #includes THIS header (it no longer DEFINES these
// structs but still SEES them, since BatchedUnifiedWorld consumes them).
//
// NAMESPACE NOTE. Kept in nuka::runtime::coresident (NOT moved to
// nuka::scene::cook) ON PURPOSE: the structs reference the CoResident* descriptors
// (CoResidentFingertip/Cup/FootSphere/Ground) which live in
// nuka::runtime::coresident; keeping the same namespace means the relocation is
// pure #include churn (no qualifier edits at any consumer), preserving byte-exact
// cook fidelity with the least risk. The coresident-named namespace is a
// transitional artifact -- it disappears with the structs at the native-cook swap.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "runtime/articulation/articulation_state.hpp"   // ArticulationHostState
#include "runtime/rigid/body_state.hpp"
#include "scene/cook/coresident_descriptors.hpp"  // CoResident{Fingertip,Cup,FootSphere,Ground}

#include <cstdint>
#include <vector>

namespace nuka::runtime::coresident {

// The cup XY reset-jitter half-box default (m, ~+/-2.5 cm). Hoisted to namespace
// scope so UnionSceneTemplate (defined before BatchedUnifiedWorld) can default its
// per-axis jitter fields to it, AND so BatchedUnifiedWorld::kResetCupJitterM (the
// named constant the gate test reads) can alias it -- the SAME literal both places.
inline constexpr float kDefaultResetCupJitterM = 0.025f;

// One entry of a reference PD drive table (the python-facing choreography seam):
// tau = kp*(target - q[dof]) - kd*qdot[dof], clamped to +/-tlim when tlim > 0.
// `dof` is the flat action column (DofIndexOf order, the SetActions layout);
// `grip` marks the 12 wrap-driven close links (the BITE kill-switch columns).
// The exported q/qdot at column `dof` ARE the link's q/qdot, so a host/python PD
// can be computed straight off the obs export. A choreography aid only -- G3's
// RL policy replaces every table.
//
// RELOCATED here (M7 T6) from the now-deleted h1_union_scene_factory.hpp: it is
// a plain-data product type the union cook (CookedUnionScene) + the C-ABI union
// world + the parity/perf gates carry alongside the UnionSceneTemplate, so it
// belongs with the template (alive to M9 per R1/R2).
struct H1UnionDriveEntry {
    uint32_t link = ~0u;   // device link index.
    uint32_t dof = ~0u;    // flat action/obs column (prefix-sum DofIndexOf).
    float target = 0.0f;
    float kp = 0.0f;
    float kd = 0.0f;
    float tlim = 0.0f;     // physical |tau| clamp (0 -> unclamped).
    uint8_t grip = 0u;     // 1 == a wrap-driven grip-close link.
};

// The per-env scene template, replicated across all envs at construction. P2.1
// scope: rigid bodies only. (P2.2+ extends with articulation proto, fingertips,
// cup hull, and static colliders -- additive fields, no layout churn for the
// rigid SoA which stays the leading env-major block.)
struct UnionSceneTemplate {
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

}  // namespace nuka::runtime::coresident
