#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::coresident -- the GRASP SCENE FACTORY (v0.8 A2). The shared,
// library-resident builder for the VALIDATED batched grasp scene the
// BatchedUnifiedWorld is trained on: the C7a cup (convex hull) + a synthetic
// fixed-base 2-finger gripper that pinches it by friction alone (NO table).
//
// WHY THIS EXISTS (the equivalence guarantee). This logic was previously
// TEST-ONLY (inline helpers in tests/coresident/test_batched_unified_world.cpp).
// Any consumer building the grasp scene MUST build the SAME UnionSceneTemplate
// the 21 nuka_batched_unified_world_test gates exercise -- a subtly different
// template means a downstream env would run on an UNVALIDATED scene (a silent
// correctness failure). So the factory is PROMOTED here, compiled into libnuka
// (g++-14), and the test DELEGATES to it: the 21 gates now run THROUGH this code,
// keeping their byte-identical results AND certifying the shared scene.
//
// PURE HOST. No CUDA in the signatures (CoResidentCup carries plain host vectors;
// ArticulationHostState is the host SoA mirror). The only device dependency is
// CookScene's transient cooking pass inside LoadGraspCupHull. Host-compiles in a
// plain g++-14 TU (the A2 Step-0b gate proved batched_unified_world.hpp does too).
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "runtime/articulation/articulation_state.hpp"   // ArticulationHostState
#include "runtime/coresident/unified_coresident_stepper.hpp"  // CoResident{Cup,Fingertip} / GraspConfig
#include "scene/cook/union_scene_template.hpp"   // UnionSceneTemplate (M9 T3 relocation)
#include "runtime/rigid/body_state.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::coresident {

// The C7a cup mass parameterization the validated gates use (cup_local_index body).
// A small 0.2 kg cup pinched by friction; the inertia is the box-approx from the
// cooked hull half-extents (see BuildGraspSceneBundle). Exposed so a caller / the
// C-ABI can report it (e.g. for a force-balance reward) without re-deriving it.
inline constexpr float kGraspCupMass    = 0.2f;
inline constexpr float kGraspCupInvMass = 1.0f / kGraspCupMass;
// The default validated cup asset (repo-relative). The C-ABI takes the path as an
// argument (so a Python caller can pass an absolute path); the test passes this.
inline constexpr const char* kGraspCupDefaultAsset =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";

// The C7a cup loaded -> a single cooked ConvexHull: mesh-local hull verts + bbox.
// (Promoted verbatim from the test helper; the cook pass runs CookScene with the
// mesh decompose SKIPPED so the cup stays ONE convex hull.)
struct GraspCupHull {
    std::vector<float> verts;   // flat xyz, length 3*vcount.
    math::Vec3 lo{}, hi{};      // mesh-local AABB of the hull verts.
    uint32_t vcount = 0u;
};

// Load + cook the cup asset at `asset_path` into a single convex hull. Returns an
// empty hull (vcount==0) when the asset is absent / produces no convex geometry --
// the caller decides whether that is fatal (the gates GTEST_SKIP; the C-ABI returns
// NUKA_RESULT_FILE_NOT_FOUND).
GraspCupHull LoadGraspCupHull(const std::string& asset_path = kGraspCupDefaultAsset);

// The synthetic FIXED-base 2-finger gripper (the SAME proto the spike / gates use).
// `host` is the cooked articulation host SoA; finger_link[i] is the device-link
// index of finger i; fingertip_local[i] is finger i's tip offset in its link frame.
struct GraspGripper {
    articulation::ArticulationHostState host;
    uint32_t   finger_link[2];
    math::Vec3 fingertip_local[2];
    float      fingertip_radius = 0.008f;
};

// Build the gripper proto positioned so its two fingertips pinch a cup of half-x
// `cup_half_x` at `base_pos` (a 1.5 mm shallow pre-pose -> contact live at step 0).
GraspGripper BuildGraspGripper(const math::Vec3& base_pos, float cup_half_x,
                               float fingertip_radius);

// The assembled grasp scene: the gripper + the GraspConfig the oracle consumes
// (fingertips, cup hull COM-centered, cup BodyState, per-link grip torque, friction).
struct GraspSceneBundle {
    GraspGripper gripper;
    GraspConfig  config;
    // A5a: per-axis cup RESET-JITTER half-box (m) carried into the UnionSceneTemplate
    // (read by ResetEnvs). Default == kDefaultResetCupJitterM so MakeGraspTemplate yields
    // the legacy isotropic +/-2.5 cm jitter byte-identically.
    float reset_jitter_x = kDefaultResetCupJitterM;
    float reset_jitter_y = kDefaultResetCupJitterM;
};

// Assemble the validated grasp scene from a cup hull + the grip force (constant
// per-finger drive torque, N) + the friction coefficient `mu`. With
// `cup_start_z_offset == 0` this is BYTE-IDENTICAL to the test helper it replaces
// -> the A1 byte-exact / A2 1e-5 gates are preserved.
//
// `cup_start_z_offset` (A3, default 0) is the DISCRIMINATIVE-IC knob: it raises the
// cup's INITIAL body Z (and therefore the ResetEnvs template Z, captured from this
// same BodyState) by that many metres ABOVE the fingertip catch plane (z=0.20),
// WITHOUT moving the fingertips (their pre-pose math keeps using the fixed 0.20
// catch-plane Z). So at offset>0 the cup starts above OPEN fingertips with NO
// contact at t=0 and DESCENDS under gravity -> a timed close catches it but a
// constant-max slam shuts on empty space before the cup arrives (proven non-trivial
// in A3). At offset==0 every field is bit-identical to the validated scene.
//
// `reset_jitter_x` / `reset_jitter_y` (A5a, default kDefaultResetCupJitterM == 0.025)
// set the per-axis cup RESET-JITTER half-box (m) ResetEnvs uses. Both at the default
// reproduce the legacy isotropic +/-2.5 cm jitter byte-identically; an anisotropic
// setting (e.g. x=0.025, y=0) jitters only along the gripper's actuated X axis so the
// X-only gripper can actually catch the jittered cups. These are reset-only -- they do
// NOT touch the create-time scene, so the byte gates (which never vary them) are
// unaffected, and the cup_start_z_offset / fingertip math is untouched.
GraspSceneBundle BuildGraspSceneBundle(const GraspCupHull& hull, float grip_force,
                                       float mu, float cup_start_z_offset = 0.0f,
                                       float reset_jitter_x = kDefaultResetCupJitterM,
                                       float reset_jitter_y = kDefaultResetCupJitterM);

// Populate a UnionSceneTemplate from the bundle (cup == the single per-env body,
// cup_local_index 0; the gripper proto + fingertips + grip torque + friction). The
// SAME inputs the A2 oracle uses -> an N=1 BatchedUnifiedWorld is byte-exact.
UnionSceneTemplate MakeGraspTemplate(const GraspSceneBundle& bundle);

}  // namespace nuka::runtime::coresident
