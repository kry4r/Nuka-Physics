#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::coresident -- the GRASP SCENE FACTORY (v0.8 A2). The shared,
// library-resident builder for the VALIDATED batched grasp scene the
// BatchedUnifiedWorld is trained on: the C7a cup (convex hull) + a synthetic
// fixed-base 2-finger gripper that pinches it by friction alone (NO table).
//
// WHY THIS EXISTS (the equivalence guarantee). This logic was previously
// TEST-ONLY (inline helpers in tests/coresident/test_batched_unified_world.cpp).
// The C-ABI `nuka_grasp_world_create` MUST build the SAME BatchedSceneTemplate
// the 21 nuka_batched_unified_world_test gates exercise -- a subtly different
// template means a Python PPO env would train on an UNVALIDATED scene (a silent
// correctness failure). So the factory is PROMOTED here, compiled into libnuka
// (g++-14), and the test DELEGATES to it: the 21 gates now run THROUGH this code,
// keeping their byte-identical results AND certifying the C-ABI scene.
//
// PURE HOST. No CUDA in the signatures (CoResidentCup carries plain host vectors;
// ArticulationHostState is the host SoA mirror). The only device dependency is
// CookScene's transient cooking pass inside LoadGraspCupHull. Host-compiles in a
// plain g++-14 TU (the A2 Step-0b gate proved batched_unified_world.hpp does too).
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "runtime/articulation/articulation_state.hpp"   // ArticulationHostState
#include "runtime/coresident/batched_unified_world.hpp"   // BatchedSceneTemplate
#include "runtime/coresident/unified_coresident_stepper.hpp"  // CoResident{Cup,Fingertip} / GraspConfig
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
};

// Assemble the validated grasp scene from a cup hull + the grip force (constant
// per-finger drive torque, N) + the friction coefficient `mu`. BYTE-IDENTICAL to
// the test helper it replaces -> the A1 byte-exact / A2 1e-5 gates are preserved.
GraspSceneBundle BuildGraspSceneBundle(const GraspCupHull& hull, float grip_force,
                                       float mu);

// Populate a BatchedSceneTemplate from the bundle (cup == the single per-env body,
// cup_local_index 0; the gripper proto + fingertips + grip torque + friction). The
// SAME inputs the A2 oracle uses -> an N=1 BatchedUnifiedWorld is byte-exact.
BatchedSceneTemplate MakeGraspTemplate(const GraspSceneBundle& bundle);

}  // namespace nuka::runtime::coresident
