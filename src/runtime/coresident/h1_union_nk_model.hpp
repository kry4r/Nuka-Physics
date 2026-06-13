#pragma once
// ---------------------------------------------------------------------------
// DEPRECATED(M9): src/runtime/coresident/ is deleted WHOLE at M9. KEPT ALIVE
// through M8 (controller R1/R2) — the M7 factory death deletes only the SCENE
// AUTHORING (h1_union_scene_factory); this bridge survives because it maps the
// BatchedSceneTemplate the .nks cook (CookSceneToUnionTemplate) now produces
// into the UnionCsr nk::Model the union gates (h1_union_parity / nk_union_n1 /
// h1_grasp_lift) step. The native CookToModel->UnionCsr path that retires it is
// M9. nk core itself never includes coresident headers.
// ---------------------------------------------------------------------------
// nuka::runtime::coresident — BatchedSceneTemplate -> nk::Model (M4).
//
// The TRANSITIONAL cook bridge for the M4 union gates: converts the union scene
// product (BatchedSceneTemplate — the settled gripper_proto + fingertips + feet
// + cup hull + table, exactly what BatchedUnifiedWorld consumes; now produced by
// the M7 .nks cook, formerly by the deleted factory) into an nk::Model of the
// UnionCsr contact family, so the SAME scene steps through CookToModel-shaped
// nk::World ops and the h1_union_parity / nk_union_n1 / h1_grasp_lift gates can
// compare the two worlds on identical initial state.
//
// MAPPING (the legacy semantics, 1:1):
//   * articulation template  <- gripper_proto (single articulation; incl. the
//     SETTLED q/qdot/link_velocity/base_pose — the factory pre-roll product)
//   * drive                  <- torque mode; drive_target seeded with the
//     template grip_torque, drive_force_limit with drive_force_limits
//   * union slots            <- feet (FootSpherePlane, foot_mu, ground height)
//                               then fingertips (FingerSphereHull, friction_mu)
//                               then the cup-proxy table box (BodyBoxPlane,
//                               table_mu, gated on table_enabled) — the legacy
//                               emission order (oracle drive_pairs order)
//   * bodies                 <- bodies_per_env BodyState template (the cup)
//   * condim                 <- the template condim for every class (the
//     legacy grasp branch's single ContactRowComplianceInputs)
// ---------------------------------------------------------------------------

#include "nk/model/model.hpp"
#include "runtime/coresident/batched_unified_world.hpp"  // BatchedSceneTemplate

namespace nuka::runtime::coresident {

// Build the UnionCsr nk::Model for `tmpl`. env_count seeds the capacity (the
// nk::World ctor can override it). Throws std::runtime_error on a template the
// union family cannot represent (no grasp articulation, oversized hull, ...).
nk::Model BuildNkUnionModel(const BatchedSceneTemplate& tmpl,
                            uint32_t env_count = 1u);

}  // namespace nuka::runtime::coresident
