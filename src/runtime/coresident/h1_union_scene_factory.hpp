#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::coresident -- the H1 WHOLE-BODY UNION SCENE FACTORY (v0.8 G2).
// The PRODUCTION home of the G1c/G1d-proven union scene authoring: a floating-
// base 51-DOF whole-body H1 (h1_with_hand MJCF cook, mesh geometry dropped,
// the 12 wrap-driven finger joints pinned to the H1.1 damping/armature) standing
// on a static ground with the 10.9 cm cup (C7a usda hull, scaled 1.8x) SETTLED
// in the curled right hand and a static table seated 2 mm into the settled cup
// bottom -- the COMPLETE union template (feet x ground + finger x cup +
// cup-proxy x table) BatchedUnifiedWorld steps in ONE solve.
//
// WHY THIS EXISTS (the G2 construction seam). The authoring choreography --
// bent stance -> foot-sphere authoring -> base seat at FK -> right-hand curl ->
// placement search at the SEAT FK -> 200-step ORACLE SETTLE pre-roll (stance
// drive + ankle CoP law + curl hold) -> hand-frame cup carry -> table height
// from the settled cup bottom -- was proven TEST-SIDE
// (tests/coresident/h1_union_scene_shared.hpp, the G1 parity gates). The C-ABI
// union world (nuka_union_world_create) cannot include a tests/ header, so the
// authoring is REPRODUCED here VERBATIM (same float expressions, same call
// order, same constants) and a dedicated gate
// (tests/coresident/test_h1_union_scene_factory.cpp) asserts the produced
// BatchedSceneTemplate is BYTE-IDENTICAL to the shared-header authoring -- a
// drifted copy fails loudly there.
//
// DETERMINISTIC. Same inputs -> byte-same template (the cook, the placement
// search, and the D1 settle pre-roll are all deterministic). The settle runs at
// the VALIDATED constants (g=-9.81, dt=1/240) regardless of what dt/gravity the
// world is later constructed with -- the authoring defines the initial state,
// the world's own integrator constants govern stepping.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/coresident/batched_unified_world.hpp"  // BatchedSceneTemplate

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::coresident {

// The default repo-relative assets (the SAME paths the test-side authoring and
// the C7b/H1.x gates use). A caller may pass absolute paths instead.
inline constexpr const char* kH1UnionMjcfDefault =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
inline constexpr const char* kH1UnionCupDefault =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";

// The validated integrator constants the authoring (settle pre-roll) runs at.
inline constexpr float kH1UnionGravityZ = -9.81f;
inline constexpr float kH1UnionDt = 1.0f / 240.0f;
// The locked scene constants (H1.1b/H1.2): 0.2 kg cup, mu=0.8, 1.8x cup scale.
inline constexpr float kH1UnionCupMass = 0.2f;
// The proven grip choreography offsets (rad): the H1.1 close target past the
// curl, and the G1d REST back-off that opens the wrap for a clean table rest.
inline constexpr float kH1UnionCloseOffset = 0.18f;
inline constexpr float kH1UnionRestBackOffset = -0.25f;

// One entry of a reference PD drive table (the python-facing choreography seam):
// tau = kp*(target - q[dof]) - kd*qdot[dof], clamped to +/-tlim when tlim > 0.
// `dof` is the flat action column (DofIndexOf order, the SetActions layout);
// `grip` marks the 12 wrap-driven close links (the BITE kill-switch columns).
// The exported q/qdot at column `dof` ARE the link's q/qdot, so a host/python
// PD can be computed straight off the obs export. Ankle entries are a POSTURE
// stand-in for the test-side CoP balance law (which needs an FK CoM probe a
// remote caller does not have): target = settled ankle q biased by the settle's
// final CoP feedforward, kp/kd/tlim per the real ankle actuator. A choreography
// aid only -- G3's RL policy replaces every table.
struct H1UnionDriveEntry {
    uint32_t link = ~0u;   // device link index.
    uint32_t dof = ~0u;    // flat action/obs column (prefix-sum DofIndexOf).
    float target = 0.0f;
    float kp = 0.0f;
    float kd = 0.0f;
    float tlim = 0.0f;     // physical |tau| clamp (0 -> unclamped).
    uint8_t grip = 0u;     // 1 == a wrap-driven grip-close link.
};

// The assembled production union scene: the BatchedSceneTemplate (settled
// proto + cup + feet + ground + table) plus the metadata an RL substrate needs.
struct H1UnionScene {
    BatchedSceneTemplate tmpl;

    // Articulation shape metadata.
    uint32_t dof_stride = 0u;   // 51 == 6 base + 45 joints (the action width).
    uint32_t base_dof = 0u;     // 6 (FloatingBase root).
    uint32_t root_link = 0u;
    uint32_t link_count = 0u;   // per-env device links.

    // Scene scalars.
    double total_mass = 0.0;    // Σ link masses (m*g*dt force-balance reference).
    float cup_mass = kH1UnionCupMass;
    float poly_cx = 0.0f;       // seat-time foot-polygon center x (CoP reference).
    math::Transform seat_pose{};  // pre-settle seated base pose.
    bool place_found = false;   // the placement search succeeded (else throw).
    float table_height = 0.0f;  // == tmpl.table_height (convenience mirror).

    // The 12 wrap-driven grip links + their flat action columns.
    std::vector<uint32_t> grip_links;
    std::vector<uint32_t> grip_dofs;

    // Reference choreography drive tables (hold = curl held at the settled q;
    // rest = wrap backed off kH1UnionRestBackOffset; close = the H1.1 close PD
    // at +kH1UnionCloseOffset). Same entry set, different grip targets.
    std::vector<H1UnionDriveEntry> drive_hold;
    std::vector<H1UnionDriveEntry> drive_rest;
    std::vector<H1UnionDriveEntry> drive_close;

    // Per-DOF |torque| limits (length dof_stride; 0 on the 6 dead base columns)
    // -- the "random torques within per-joint limits" envelope: legs at the MJCF
    // motor limits (hip 200 / knee 300 / ankle 40), torso 200, shoulder 40
    // (yaw 18), elbow 18, wrist 6, fingers 1.
    std::vector<float> dof_torque_limit;

    // The settle's final per-ankle CoP torque (the posture-table feedforward
    // baked into the ankle entries' targets; surfaced for diagnostics).
    float ankle_settle_tau[2] = {0.0f, 0.0f};
};

// Build the complete union scene (cook -> stance -> seat -> curl -> placement
// search -> 200-step oracle settle -> carry -> table). Throws std::runtime_error
// on a missing/unloadable asset or a failed placement search (LOUD, never a
// silently degenerate scene). Deterministic: same inputs -> byte-same result.
H1UnionScene BuildH1UnionScene(const phi::DeviceContext& context,
                               const std::string& h1_mjcf = kH1UnionMjcfDefault,
                               const std::string& cup_usda = kH1UnionCupDefault);

}  // namespace nuka::runtime::coresident
