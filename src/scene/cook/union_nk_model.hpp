#pragma once
// ---------------------------------------------------------------------------
// RELOCATED (M9 T11-core-b1) from src/runtime/coresident/h1_union_nk_model.{hpp,cpp}
// into src/scene/cook/union_nk_model.{hpp,cpp}: this surviving union-cook bridge
// must NOT live in the DOOMED coresident directory (T11-core-b2 deletes that dir
// WHOLE). It depends only on the surviving cook product (UnionSceneTemplate + the
// CoResident* descriptors, both under src/scene/cook) + nk model headers; it no
// longer includes any coresident CLASS header (batched_unified_world.hpp /
// unified_coresident_stepper.hpp).
//
// LIFETIME: this bridge maps the UnionSceneTemplate the .nks cook
// (CookSceneToUnionTemplate) produces into the UnionCsr nk::Model the union gates
// (h1_union_parity / nk_union_n1 / h1_grasp_lift) step. The native
// CookToModel->UnionCsr path that retires it is M9. The namespace is kept as
// nuka::runtime::coresident (a transitional artifact, not the M9 gate symbol) so
// no consumer qualifier churns -- preserving byte-exact cook fidelity.
// ---------------------------------------------------------------------------
// nuka::runtime::coresident — UnionSceneTemplate -> nk::Model (M4).
//
// The TRANSITIONAL cook bridge for the M4 union gates: converts the union scene
// product (UnionSceneTemplate — the settled gripper_proto + fingertips + feet
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
#include "scene/cook/union_scene_template.hpp"  // UnionSceneTemplate (+ CoResident* descriptors)

namespace nuka::runtime::coresident {

// Build the UnionCsr nk::Model for `tmpl`. env_count seeds the capacity (the
// nk::World ctor can override it). Throws std::runtime_error on a template the
// union family cannot represent (no grasp articulation, oversized hull, ...).
nk::Model BuildNkUnionModel(const UnionSceneTemplate& tmpl,
                            uint32_t env_count = 1u);

}  // namespace nuka::runtime::coresident
