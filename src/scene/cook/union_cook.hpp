#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::cook — CookSceneToUnionTemplate (M7 T4): THE CRUX.
//
// Turns the AUTHORED h1_cup_table.nks (loaded into a SceneIR: the expanded
// `imports h1_with_hand.xml` + the cup / table tree bodies + the GraspConfig
// metadata block + the baked initial_state) into the union scene template the
// co-resident world / nk bridge step — REPLACING the 1026-line
// h1_union_scene_factory's BuildH1UnionScene authoring (cook -> stance -> seat
// -> curl -> placement search -> 200-step settle -> carry -> table).
//
// WHERE THE FACTORY DERIVES, THIS COOK READS (controller ruling R4: the settled
// stance IC + the cup placement are BAKED in the .nks — the cook does NOT run a
// settle loop or a placement search). The factory's gripper_proto is the product
// of a 200-step quasi-static settle pre-roll; this cook reads the SETTLED per-
// link q + base pose straight out of the SceneIR's initial_state, the 30 wrap +
// 4 foot contact spheres + the three PD drive tables straight out of the
// GraspConfig block, and the settled cup pose straight off the cup tree node.
// The cook therefore reproduces MakeUnionTemplate's OUTPUT (the
// UnionSceneTemplate) WITHOUT the factory's empirical FK/placement/settle
// machinery — proven equivalent by the permanent grasp gate
// tests/scenario/h1_grasp_lift.cpp (T5; the transitional cook==factory gate that
// originally pinned the ~1e-8 equivalence was deleted with the factory in T6).
//
// PER CONTROLLER R1/R2/R3 the cook PRODUCES a coresident::UnionSceneTemplate
// (the struct + BuildNkUnionModel + BatchedUnifiedWorld all stay alive to M9):
// BuildNkUnionModel(cooked.tmpl) and BatchedUnifiedWorld(cooked.tmpl) behave
// identically to the factory's. The native CookToModel->UnionCsr path is M9.
//
// nk_engine lint scope: CUDA-type-free host C++. Depends ONLY on the relocated
// plain-data product types (UnionSceneTemplate, H1UnionDriveEntry, in
// src/scene/cook/union_scene_template.hpp; CoResident* descriptors pulled in
// transitively) — NO kernels, NO device, NO dependency on the doomed coresident
// directory (the M9 T3 decouple: union_cook no longer includes
// runtime/coresident/batched_unified_world.hpp).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

#include "scene/cook/union_scene_template.hpp"  // UnionSceneTemplate + H1UnionDriveEntry
#include "scene/scene_ir.hpp"

namespace nuka::scene::cook {

// The post-factory home of the union product (the consumer-facing surface
// H1UnionScene exposed, moved OUT of the doomed factory header into src/scene/
// cook so T5's re-pointed consumers — the parity oracle / perf gate / C-ABI —
// and T6's factory deletion work). Field-for-field the metadata an RL substrate
// + the parity oracle read off BuildH1UnionScene's result, with `tmpl` the
// UnionSceneTemplate BuildNkUnionModel / BatchedUnifiedWorld consume.
struct CookedUnionScene {
    runtime::coresident::UnionSceneTemplate tmpl;

    // Articulation shape metadata.
    uint32_t dof_stride = 0u;   // 51 == 6 base + 45 joints (the action width).
    uint32_t base_dof   = 0u;   // 6 (FloatingBase root).
    uint32_t root_link  = 0u;
    uint32_t link_count = 0u;   // per-env device links.

    // Scene scalars.
    double total_mass = 0.0;    // Σ link masses (m*g*dt force-balance reference).
    float  cup_mass   = 0.0f;
    float  poly_cx    = 0.0f;   // seat-time foot-polygon center x (CoP reference).
    bool   place_found = false; // the authored scene carried a placed cup.
    float  table_height = 0.0f; // == tmpl.table_height (convenience mirror).

    // The 12 wrap-driven grip links + their flat action columns (resolved).
    std::vector<uint32_t> grip_links;
    std::vector<uint32_t> grip_dofs;

    // Reference choreography drive tables (hold / rest / close), the SAME
    // H1UnionDriveEntry layout the parity oracle / perf gate / C-ABI read:
    // {link, dof, target, kp, kd, tlim, grip}, link/dof resolved to device.
    std::vector<runtime::coresident::H1UnionDriveEntry> drive_hold;
    std::vector<runtime::coresident::H1UnionDriveEntry> drive_rest;
    std::vector<runtime::coresident::H1UnionDriveEntry> drive_close;

    // Per-DOF |torque| limits (length dof_stride; 0 on the 6 dead base columns).
    std::vector<float> dof_torque_limit;

    // The settle's final per-ankle CoP torque (read from the GraspConfig; the
    // factory baked it into the ankle drive entries' targets, surfaced here for
    // diagnostics / python parity).
    float ankle_settle_tau[2] = {0.0f, 0.0f};
};

// Cook the loaded h1_cup_table SceneIR into the union template, replicated
// across `env_count` envs (clamped to 1 if <= 0). The SceneIR MUST carry a
// present GraspConfig (scene.Grasp().present) + the matching initial_state, or
// the cook throws std::runtime_error (LOUD, never a silently degenerate scene —
// mirrors BuildH1UnionScene's throw posture). Deterministic: same SceneIR ->
// same product.
CookedUnionScene CookSceneToUnionTemplate(const SceneIR& scene, int env_count);

}  // namespace nuka::scene::cook
