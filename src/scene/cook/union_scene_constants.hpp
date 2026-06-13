#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::cook -- the H1 union scene CONSTANTS (M7 T6).
//
// RELOCATED here from the now-deleted h1_union_scene_factory.hpp so the
// re-pointed union consumers (the perf gate nk_union_n1, the parity oracle
// h1_union_parity, the permanent grasp gate h1_grasp_lift, the C-ABI union
// world) keep their integrator constants + default asset / scene paths after
// the 1026-line factory dies. The cook (CookSceneToUnionTemplate) reads the
// authored h1_cup_table.nks; these are the validated scalars the factory baked.
//
// Pure host header (constexpr only); the surviving home for the constants that
// outlive src/runtime/coresident/ (deleted whole at M9).
// ---------------------------------------------------------------------------

namespace nuka::scene::cook {

// The authored union scene + the default repo-relative assets it imports (the
// .nks `imports h1_with_hand.xml` + the cup hull the cook re-resolves). The
// .nks IS the scene; the asset paths are the file-existence guard the gates +
// the C-ABI check before cooking.
inline constexpr const char* kH1UnionNksDefault =
    "examples/scenes/h1_cup_table.nks";
inline constexpr const char* kH1UnionMjcfDefault =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
inline constexpr const char* kH1UnionCupDefault =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";

// The validated integrator constants the authoring (settle pre-roll) ran at and
// the union worlds step at (g=-9.81, dt=1/240).
inline constexpr float kH1UnionGravityZ = -9.81f;
inline constexpr float kH1UnionDt = 1.0f / 240.0f;

// The locked scene constants (H1.1b/H1.2): 0.2 kg cup, 1.8x cup scale (~10.9 cm).
inline constexpr float kH1UnionCupMass = 0.2f;

// The proven grip choreography offsets (rad): the H1.1 close target past the
// curl, and the G1d REST back-off that opens the wrap for a clean table rest.
inline constexpr float kH1UnionCloseOffset = 0.18f;
inline constexpr float kH1UnionRestBackOffset = -0.25f;

}  // namespace nuka::scene::cook
