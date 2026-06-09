// ---------------------------------------------------------------------------
// v0.8 -- the POLICY -> CO-RESIDENT INFERENCE BRIDGE SPIKE (C++ DEPLOY side).
//
// THE GATE: before funding a multi-week PPO standing campaign we must prove the
// SEAM that retires at deploy actually works. PPO trains a PyTorch NN in the
// BATCHED reduced-16-DOF world (python, nuka.World); the demo runs that policy in
// the CO-RESIDENT full-51-DOF world (this C++ UnifiedCoResidentStepper, NOT bound
// to python). The bridge is flat f32 weights + a ~50-line C++ forward (tiny_mlp.hpp)
// -- NOT python-bind / NOT ONNX / NOT torch-at-deploy.
//
// TWO pass-while-broken TRAPS this gate is built to DEFEAT (each with a BITE):
//   TRAP 1 -- "stands" does NOT test the bridge. The CoM/CoP law has huge gain
//     margin, so an APPROX-right C++ forward (transposed weight, wrong activation,
//     f32/f64 slip, row/col-major) STILL stands -> a green-washed broken export.
//     -> we require EXACT numerical parity (1e-5) on goldens (Gate 1), NOT "stands".
//   TRAP 2 -- the bridge must CROSS the train<->deploy boundary. The MLP is fit in
//     the BATCHED reduced world (what PPO emits) and deployed in the co-resident
//     FULL world via an explicit obs-projection + leg-index map. The real risk =
//     leg ordering reduced-cook vs full-cook differs -> an index PERMUTATION is
//     INVISIBLE to "stands". -> Piece A proves the reduced 1..10 == the full
//     by-name legs are structurally identical (the index map is the identity for
//     legs), and the obs-builder routes leg slots through a leg-index array that
//     the permutation BITE swaps.
//
// PIECES:
//   A   -- structural index-correspondence proof: cook BOTH the reduced + full
//          MJCF, resolve the 10 contract legs BY NAME on both, assert joint_axis /
//          parent_offset / mass identical (the reduced legs are 1..10; the full
//          cook is 46 links DFS-order so legs are NOT 1..10 -> by-name is mandatory).
//   B   -- GATE 1 (the bridge verdict, EXACT): load the exported f32 weights +
//          goldens; (1a) rebuild obs_cpp from the raw shared-state with the C++
//          obs-builder, assert max|obs_cpp - obs_py| < 1e-5; (1b) run the C++ MLP
//          forward on obs_py, assert max|action_cpp - action_py| < 1e-5.
//   BITE -- prove Gate 1 BITES: (i) a leg-index PERMUTATION in the obs-builder and
//          (ii) a TRANSPOSED W1 each produce diffs >> 1e-5.
//   C   -- GATE 2: deploy the exported PPO MLP in the FULL co-resident world
//          (Download -> obs_cpp -> MLP -> normalized action -> per-link leg torque
//          -> SetGripTorque -> Step), hold the upper body with the SAME hold-PD as
//          costand, ~1000 steps, and assert S4 standing metrics: tilt<15deg,
//          4/4 foot contacts, physical torque limits, stable height, no NaN.
//
// ADDITIVE: a NEW translation unit. It COMPILES the production stepper + the host
// contact driver (same as test_h1_costand_transfer.cpp) but EDITS nothing. The
// current PPO goldens/weights are produced by
// examples/training/export_h1_tiny_actor_bridge.py (regenerable, gitignored); the
// exporter is the source of truth for the normalized-action bridge artifact.
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"
#include "tiny_mlp.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace coresident = nuka::runtime::coresident;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kGravityZ = -9.81f;
constexpr float kDt = 0.005f;

// The shared obs/action contract dimensions (== export_h1_tiny_actor_bridge.py).
constexpr uint32_t kObsDim = 32u;
constexpr uint32_t kActDim = 10u;

// poly center PINNED scalar (full-cook seat-time value; printed by costand SEAT).
// Used IDENTICALLY in the Python training obs and the C++ Gate-2 obs.
constexpr float kPolyCx = 0.0770f;
constexpr float kPolyCy = 0.0f;

const std::string kFullMjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
const std::string kReducedMjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_reduced_train.xml";
const std::string kWeightsBin = "python/spikes/out/h1_bridge_mlp.bin";
const std::string kGoldensTxt = "python/spikes/out/h1_bridge_goldens.txt";

bool FileExists(const std::string& p) { return std::filesystem::exists(p); }

// The 10 contract leg JOINTS in the pinned order. The cooked link names append
// "_link"; the reduced legs are links 1..10, the full legs are resolved by name.
const std::array<std::string, 10> kLegLinkNames = {
    "left_hip_yaw_link",  "left_hip_roll_link",  "left_hip_pitch_link",
    "left_knee_link",     "left_ankle_link",     "right_hip_yaw_link",
    "right_hip_roll_link", "right_hip_pitch_link", "right_knee_link",
    "right_ankle_link"};

// ---- physical MJCF torque limits (N*m) per leg slot, contract order -----------
float LegLimit(uint32_t slot) {
    // slots: 0 hip_yaw,1 roll,2 pitch,3 knee,4 ankle (x2 for right).
    const uint32_t j = slot % 5u;
    if (j == 3u) return 300.0f;  // knee.
    if (j == 4u) return 40.0f;   // ankle.
    return 200.0f;               // hip yaw/roll/pitch.
}

// ===========================================================================
// COOK helpers (mirror test_h1_costand_transfer.cpp's LoadH1Floating, by name).
// ===========================================================================
struct Cooked {
    articulation::ArticulationHostState host;
    nuka::scene::SceneIR scene;
};

Cooked LoadFloating(const std::string& mjcf) {
    Cooked out;
    out.scene = nuka::import::LoadMjcf(mjcf);
    // Drop mesh geometry (dodge the V-HACD wall; we only need inertia + topology).
    for (size_t i = 0u; i < out.scene.ShapeCount(); ++i) {
        auto& s = out.scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        s.mesh_vertices.clear();
        s.mesh_indices.clear();
    }
    const auto blob = nuka::scene::CookScene(out.scene);
    auto topos = articulation::CookArticulations(blob);
    out.host = articulation::BuildArticulationHostState(topos, blob.bodies);
    return out;
}

std::string LinkName(const Cooked& c, uint32_t link) {
    if (c.host.link_body.size() <= link) return "?";
    const uint32_t body = c.host.link_body[link];
    if (body < c.scene.Bodies().size()) return c.scene.GetBody(body).name;
    return "?";
}
uint32_t LinkByName(const Cooked& c, const std::string& name) {
    for (uint32_t l = 0u; l < c.host.TotalLinkCount(); ++l)
        if (LinkName(c, l) == name) return l;
    return kInvalidLink;
}

// Resolve the 10 contract leg links (by name) into device link indices.
std::array<uint32_t, 10> ResolveLegLinks(const Cooked& c) {
    std::array<uint32_t, 10> out{};
    for (uint32_t s = 0u; s < 10u; ++s) out[s] = LinkByName(c, kLegLinkNames[s]);
    return out;
}

const char* JointTypeName(articulation::ArticulationJointType t) {
    switch (t) {
        case articulation::ArticulationJointType::Revolute: return "Revolute";
        case articulation::ArticulationJointType::Prismatic: return "Prismatic";
        case articulation::ArticulationJointType::Fixed: return "Fixed";
        case articulation::ArticulationJointType::FloatingBase: return "FloatingBase";
    }
    return "?";
}

// ===========================================================================
// The SHARED obs-builder (the ONE function used by Gate 1a AND Gate 2). It indexes
// the leg slots through a LEG-INDEX ARRAY -- the permutation BITE swaps two entries
// in that array (defeating TRAP 2's invisible-permutation). raw == obs (no
// normalization), so Gate 1a's parity is trivially exact BY DESIGN; the obs-builder
// teeth live entirely in the permutation BITE going through this same array.
//
//   obs[0:4]   = quat (qw,qx,qy,qz)
//   obs[4:10]  = base spatial vel (omega-first) [wx,wy,wz, vx,vy,vz]
//   obs[10:12] = (base_x - poly_cx, base_y - poly_cy)
//   obs[12:22] = leg qpos in leg-index order
//   obs[22:32] = leg qvel in leg-index order
// ===========================================================================
std::vector<float> BuildObs(const std::array<float, 4>& quat,
                            const std::array<float, 6>& spatial_vel,
                            float base_x, float base_y,
                            const std::array<float, 10>& leg_qpos,
                            const std::array<float, 10>& leg_qvel,
                            const std::array<uint32_t, 10>& leg_index) {
    std::vector<float> obs(kObsDim, 0.0f);
    for (uint32_t i = 0u; i < 4u; ++i) obs[i] = quat[i];
    for (uint32_t i = 0u; i < 6u; ++i) obs[4u + i] = spatial_vel[i];
    obs[10] = base_x - kPolyCx;
    obs[11] = base_y - kPolyCy;
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t j = leg_index[s];   // the leg-index map (identity normally).
        obs[12u + s] = leg_qpos[j];
        obs[22u + s] = leg_qvel[j];
    }
    return obs;
}

// The identity leg-index map (the correct one: reduced 1..10 == full by-name).
std::array<uint32_t, 10> IdentityLegIndex() {
    std::array<uint32_t, 10> m{};
    for (uint32_t s = 0u; s < 10u; ++s) m[s] = s;
    return m;
}

// ===========================================================================
// Golden + weight IO.
// ===========================================================================
struct Golden {
    std::array<float, 32> raw{};
    std::array<float, 32> obs_py{};
    std::array<float, 10> action_py{};
};

std::vector<Golden> LoadGoldens(const std::string& path) {
    std::ifstream f(path);
    std::vector<Golden> out;
    std::string header;
    std::getline(f, header);
    std::istringstream hs(header);
    uint32_t k = 0u, od = 0u, ad = 0u;
    hs >> k >> od >> ad;
    EXPECT_EQ(od, kObsDim);
    EXPECT_EQ(ad, kActDim);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // format: raw(32) | obs(32) | action(10).
        std::replace(line.begin(), line.end(), '|', ' ');
        std::istringstream ls(line);
        Golden g;
        for (uint32_t i = 0u; i < 32u; ++i) ls >> g.raw[i];
        for (uint32_t i = 0u; i < 32u; ++i) ls >> g.obs_py[i];
        for (uint32_t i = 0u; i < 10u; ++i) ls >> g.action_py[i];
        out.push_back(g);
    }
    return out;
}

// Decompose a golden's RAW shared-state (32) back into the obs-builder inputs.
void RawToInputs(const std::array<float, 32>& raw, std::array<float, 4>* quat,
                 std::array<float, 6>* spatial_vel, float* base_x, float* base_y,
                 std::array<float, 10>* leg_qpos, std::array<float, 10>* leg_qvel) {
    for (uint32_t i = 0u; i < 4u; ++i) (*quat)[i] = raw[i];
    for (uint32_t i = 0u; i < 6u; ++i) (*spatial_vel)[i] = raw[4u + i];
    // raw[10],raw[11] are ALREADY base_{x,y} - poly_center; reconstruct absolute so
    // the C++ obs-builder (which re-subtracts poly_center) reproduces the same slot.
    *base_x = raw[10] + kPolyCx;
    *base_y = raw[11] + kPolyCy;
    for (uint32_t i = 0u; i < 10u; ++i) (*leg_qpos)[i] = raw[12u + i];
    for (uint32_t i = 0u; i < 10u; ++i) (*leg_qvel)[i] = raw[22u + i];
}

// ===========================================================================
// PIECE A -- structural index-correspondence proof (defeats TRAP 2's permutation).
// ===========================================================================
TEST(H1BridgeSpike, PieceAStructuralLegCorrespondence) {
    if (!FileExists(kFullMjcf)) GTEST_SKIP() << "h1_with_hand asset not available";
    if (!FileExists(kReducedMjcf))
        GTEST_SKIP() << "h1_reduced_train asset not available (run gen_h1_reduced_train.py)";

    const Cooked full = LoadFloating(kFullMjcf);
    const Cooked red = LoadFloating(kReducedMjcf);

    std::printf("\n#### PIECE A: leg-index correspondence (reduced 1..10 vs full by-name) ####\n");
    std::printf("  reduced cook: %u links ; full cook: %u links\n",
                red.host.TotalLinkCount(), full.host.TotalLinkCount());

    const auto red_legs = ResolveLegLinks(red);
    const auto full_legs = ResolveLegLinks(full);

    std::printf("  %-22s | %-18s | %-18s\n", "contract leg", "REDUCED", "FULL");
    std::printf("  %-22s | idx jtype  mass  | idx jtype  mass\n", "");
    float max_axis_diff = 0.0f, max_off_diff = 0.0f, max_mass_diff = 0.0f;
    bool all_resolved = true;
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t rl = red_legs[s], fl = full_legs[s];
        if (rl == kInvalidLink || fl == kInvalidLink) {
            all_resolved = false;
            std::printf("  %-22s | UNRESOLVED (red=%u full=%u)\n",
                        kLegLinkNames[s].c_str(), rl, fl);
            continue;
        }
        const auto& ra = red.host.joint_axis[rl];
        const auto& fa = full.host.joint_axis[fl];
        const auto& ro = red.host.parent_offset[rl];
        const auto& fo = full.host.parent_offset[fl];
        const float rmass = red.host.link_inertia[rl].I[3u * 6u + 3u];
        const float fmass = full.host.link_inertia[fl].I[3u * 6u + 3u];
        const float ad = std::max({std::fabs(ra.x - fa.x), std::fabs(ra.y - fa.y),
                                   std::fabs(ra.z - fa.z)});
        const float od = std::max({std::fabs(ro.x - fo.x), std::fabs(ro.y - fo.y),
                                   std::fabs(ro.z - fo.z)});
        const float md = std::fabs(rmass - fmass);
        max_axis_diff = std::max(max_axis_diff, ad);
        max_off_diff = std::max(max_off_diff, od);
        max_mass_diff = std::max(max_mass_diff, md);
        std::printf("  %-22s | %3u %-9s %5.3f | %3u %-9s %5.3f  "
                    "[axis|%.1e off|%.1e mass|%.1e]\n",
                    kLegLinkNames[s].c_str(),
                    rl, JointTypeName(red.host.joint_type[rl]), rmass,
                    fl, JointTypeName(full.host.joint_type[fl]), fmass,
                    ad, od, md);
    }
    std::printf("  REDUCED leg device indices :");
    for (uint32_t s = 0u; s < 10u; ++s) std::printf(" %u", red_legs[s]);
    std::printf("\n  FULL    leg device indices :");
    for (uint32_t s = 0u; s < 10u; ++s) std::printf(" %u", full_legs[s]);
    std::printf("\n  => max diff: joint_axis=%.2e parent_offset=%.2e mass=%.2e\n",
                max_axis_diff, max_off_diff, max_mass_diff);

    ASSERT_TRUE(all_resolved) << "a contract leg link did not resolve in one of the cooks";
    // The reduced legs are the contiguous 1..10 the python side uses (q[1:11]).
    for (uint32_t s = 0u; s < 10u; ++s)
        EXPECT_EQ(red_legs[s], s + 1u)
            << "reduced leg slot " << s << " (" << kLegLinkNames[s]
            << ") not at link " << (s + 1u);
    // The structural match: the reduced<->full leg index map is the IDENTITY for the
    // contract obs/action indices (the python action/obs indices == the C++ deploy
    // indices). A mismatch here is a real boundary finding, not a tuning chore.
    EXPECT_LT(max_axis_diff, 1e-5f) << "leg joint_axis differs reduced vs full cook";
    EXPECT_LT(max_off_diff, 1e-5f) << "leg parent_offset differs reduced vs full cook";
    EXPECT_LT(max_mass_diff, 1e-4f) << "leg mass differs reduced vs full cook";
}

// ===========================================================================
// PIECE B -- GATE 1: the bridge proof (EXACT, the load-bearing gate).
// ===========================================================================
TEST(H1BridgeSpike, Gate1BridgeParity) {
    if (!FileExists(kWeightsBin) || !FileExists(kGoldensTxt))
        GTEST_SKIP() << "bridge artifacts missing (run examples/training/export_h1_tiny_actor_bridge.py)";

    const auto mlp = nuka::test::TinyMlp<32, 64, 10>::Load(kWeightsBin);
    const auto goldens = LoadGoldens(kGoldensTxt);
    ASSERT_GE(goldens.size(), 8u) << "need a handful of goldens";

    const auto id = IdentityLegIndex();
    float max_obs_diff = 0.0f, max_act_diff = 0.0f;
    for (const auto& g : goldens) {
        // Gate 1a: rebuild obs_cpp from the RAW shared-state via the C++ obs-builder.
        std::array<float, 4> quat;
        std::array<float, 6> svel;
        float bx, by;
        std::array<float, 10> qpos, qvel;
        RawToInputs(g.raw, &quat, &svel, &bx, &by, &qpos, &qvel);
        const auto obs_cpp = BuildObs(quat, svel, bx, by, qpos, qvel, id);
        for (uint32_t i = 0u; i < kObsDim; ++i)
            max_obs_diff = std::max(max_obs_diff,
                                    std::fabs(obs_cpp[i] - g.obs_py[i]));

        // Gate 1b: run the C++ MLP forward on obs_py, compare to action_py.
        const std::vector<float> obs_py(g.obs_py.begin(), g.obs_py.end());
        const auto act_cpp = mlp.Forward(obs_py);
        for (uint32_t i = 0u; i < kActDim; ++i)
            max_act_diff = std::max(max_act_diff,
                                    std::fabs(act_cpp[i] - g.action_py[i]));
    }
    std::printf("\n#### GATE 1: bridge parity over %zu goldens ####\n", goldens.size());
    std::printf("  [1a] max|obs_cpp - obs_py|     = %.3e  (bar 1e-5)\n", max_obs_diff);
    std::printf("  [1b] max|action_cpp - action_py| = %.3e  (bar 1e-5)\n", max_act_diff);
    std::printf("  => the full chain mlp_cpp(obs_cpp(raw)) ~= action_py holds.\n");

    EXPECT_LT(max_obs_diff, 1e-5f) << "Gate 1a: C++ obs-builder disagrees with python obs";
    EXPECT_LT(max_act_diff, 1e-5f) << "Gate 1b: C++ MLP forward disagrees with python";
}

// ===========================================================================
// PIECE B BITE -- prove Gate 1 DETECTS the named failure modes (anti-green-wash).
// ===========================================================================
TEST(H1BridgeSpike, Gate1BiteDetectsBrokenBridge) {
    if (!FileExists(kWeightsBin) || !FileExists(kGoldensTxt))
        GTEST_SKIP() << "bridge artifacts missing (run examples/training/export_h1_tiny_actor_bridge.py)";

    const auto mlp = nuka::test::TinyMlp<32, 64, 10>::Load(kWeightsBin);
    const auto goldens = LoadGoldens(kGoldensTxt);
    ASSERT_GE(goldens.size(), 8u);

    std::printf("\n#### GATE 1 BITE: broken-variant detection ####\n");

    // ---- BITE (i): a deliberate leg-index PERMUTATION in the obs-builder. -------
    // Swap leg slots 2<->3 (left_hip_pitch <-> left_knee): their qpos differ by ~1.1
    // rad (-0.40 vs +0.70) in EVERY golden, so the swap hits BOTH obs 14<->15 (qpos)
    // AND obs 24<->25 (qvel) with a large, unambiguous signal. (Swapping near-equal
    // steady-state slots would be vacuous -> we pick the maximally-distinct pair.)
    auto perm = IdentityLegIndex();
    std::swap(perm[2], perm[3]);
    float perm_obs_diff = 0.0f;
    for (const auto& g : goldens) {
        std::array<float, 4> quat;
        std::array<float, 6> svel;
        float bx, by;
        std::array<float, 10> qpos, qvel;
        RawToInputs(g.raw, &quat, &svel, &bx, &by, &qpos, &qvel);
        const auto obs_perm = BuildObs(quat, svel, bx, by, qpos, qvel, perm);
        for (uint32_t i = 0u; i < kObsDim; ++i)
            perm_obs_diff = std::max(perm_obs_diff,
                                     std::fabs(obs_perm[i] - g.obs_py[i]));
    }
    std::printf("  [BITE i] leg-index permutation (slots 2<->3): max|obs_diff| = %.3e\n",
                perm_obs_diff);
    EXPECT_GT(perm_obs_diff, 0.1f)
        << "BITE(i) VACUOUS: the leg-index permutation produced obs diff < 0.1 -- the "
           "obs-builder is not routing legs through the index array, or the swapped "
           "slots are near-equal. Gate 1a does not bite.";

    // ---- BITE (ii): a TRANSPOSED W1 in the forward. -----------------------------
    // The canonical row/col-major confusion: W1 is [H=64, IN=32] row-major (2048
    // floats); the classic bug LOADS the SAME 2048 floats as if they were stored
    // [IN=32, H=64] and uses them as [H, IN]. We model that exactly: the buggy
    // W1_bug[o, i] reads the float a [32,64]-stored matrix would have at logical
    // [i, o] -- i.e. w1[i * H + o]. Every one of the 2048 floats is used (i in
    // [0,32), o in [0,64) -> indices 0..2047), and it is genuinely THE transpose-
    // load bug, not an index-wrapping approximation.
    std::vector<float> flat = mlp.Flat();
    {
        std::vector<float> w1(flat.begin(), flat.begin() + 64 * 32);
        std::vector<float> w1t(64 * 32);
        for (uint32_t o = 0u; o < 64u; ++o)
            for (uint32_t i = 0u; i < 32u; ++i)
                w1t[o * 32u + i] = w1[i * 64u + o];  // [32,64]-stored read as [64,32].
        std::copy(w1t.begin(), w1t.end(), flat.begin());
    }
    const auto mlp_bug = nuka::test::TinyMlp<32, 64, 10>::FromFlat(flat);
    float trans_act_diff = 0.0f;
    for (const auto& g : goldens) {
        const std::vector<float> obs_py(g.obs_py.begin(), g.obs_py.end());
        const auto act_bug = mlp_bug.Forward(obs_py);
        for (uint32_t i = 0u; i < kActDim; ++i)
            trans_act_diff = std::max(trans_act_diff,
                                      std::fabs(act_bug[i] - g.action_py[i]));
    }
    std::printf("  [BITE ii] transposed-W1 forward: max|action_diff| = %.3e\n",
                trans_act_diff);
    EXPECT_GT(trans_act_diff, 1e-2f)
        << "BITE(ii) VACUOUS: a transposed W1 produced action diff < 1e-2 -- Gate 1b "
           "does not bite (the forward is insensitive to the W1 layout).";

    std::printf("  => both broken variants exceed 1e-5 by orders of magnitude: Gate 1 BITES.\n");
}

// ===========================================================================
// PIECE C scaffold -- the FULL co-resident standing scene (mirrors costand's
// BuildStandScene, by name). Reused for Gate 2.
// ===========================================================================
constexpr float kStanceHipPitch = -0.40f;
constexpr float kStanceKnee = 0.70f;
constexpr float kStanceAnkle = -0.30f;
constexpr float kKpHold = 50.0f, kKdHold = 4.0f;
constexpr float kFootSphereR = 0.025f;
constexpr float kFootBottomZ = -0.055f;
constexpr float kFootToeX = 0.10f;
constexpr float kFootHeelX = -0.10f;

std::vector<Transform> ForwardKinematics(const nuka::phi::DeviceContext& context,
                                         const articulation::ArticulationHostState& host) {
    const uint32_t link_count = host.TotalLinkCount();
    auto device = articulation::UploadArticulationState(context, host);
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                               nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, device.View(),
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> poses(link_count);
    pose_buf.CopyToHost(poses.data(), poses.size() * sizeof(Transform));
    return poses;
}

struct CoMModel {
    std::vector<float> mass;
    std::vector<Vec3> com_local;
    double total = 0.0;
};
CoMModel BuildCoMModel(const articulation::ArticulationHostState& host) {
    CoMModel m;
    const uint32_t n = host.TotalLinkCount();
    m.mass.resize(n);
    m.com_local.resize(n);
    for (uint32_t l = 0u; l < n; ++l) {
        m.mass[l] = host.link_inertia[l].I[3u * 6u + 3u];
        m.com_local[l] = host.link_inertial_frame[l].position;
        m.total += m.mass[l];
    }
    return m;
}
Vec3 WholeBodyCoM(const CoMModel& m, const std::vector<Transform>& poses) {
    Vec3 acc{0, 0, 0};
    for (uint32_t l = 0u; l < poses.size(); ++l) {
        const Vec3 cw = poses[l].position + poses[l].rotation.Rotate(m.com_local[l]);
        acc += cw * m.mass[l];
    }
    return acc * static_cast<float>(1.0 / m.total);
}
double TiltDeg(const Quat& q) {
    const double gz = 1.0 - 2.0 * (static_cast<double>(q.x) * q.x +
                                   static_cast<double>(q.y) * q.y);
    return std::acos(std::max(-1.0, std::min(1.0, gz))) * 180.0 / M_PI;
}

struct StandScene {
    Cooked h1;
    articulation::ArticulationHostState host;
    coresident::StandConfig config;
    CoMModel com;
    std::array<uint32_t, 10> leg_links{};   // by-name device links (contract order).
    std::vector<uint32_t> ankle_links;      // [left_ankle, right_ankle].
    std::vector<uint32_t> hold_links;       // every OTHER actuated DOF (torso/arms/...).
    std::vector<float> hold_target;
    std::vector<float> hold_tlim;
    float seat_base_z = 0.0f;
};

float HoldLimitFor(const std::string& nm) {
    if (nm.find("torso") != std::string::npos) return 200.0f;
    if (nm.find("elbow") != std::string::npos) return 18.0f;
    if (nm.find("shoulder_yaw") != std::string::npos) return 18.0f;
    if (nm.find("shoulder") != std::string::npos) return 40.0f;
    return 1.0f;
}

StandScene BuildStandScene(const nuka::phi::DeviceContext& context) {
    StandScene sc;
    sc.h1 = LoadFloating(kFullMjcf);
    sc.host = sc.h1.host;
    for (auto& v : sc.host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (float& qd : sc.host.qdot) qd = 0.0f;
    sc.host.base_pose[0].rotation = Quat::Identity();

    sc.leg_links = ResolveLegLinks(sc.h1);
    // Apply the bent stance to both legs (hip_pitch/knee/ankle); yaw/roll = 0.
    auto setq = [&](uint32_t l, float v) { if (l != kInvalidLink) sc.host.q[l] = v; };
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t j = s % 5u;
        float v = 0.0f;
        if (j == 2u) v = kStanceHipPitch;
        else if (j == 3u) v = kStanceKnee;
        else if (j == 4u) v = kStanceAnkle;
        setq(sc.leg_links[s], v);
    }
    sc.com = BuildCoMModel(sc.host);

    const uint32_t l_ankle = sc.leg_links[4];   // left_ankle.
    const uint32_t r_ankle = sc.leg_links[9];   // right_ankle.
    sc.ankle_links = {l_ankle, r_ankle};
    uint32_t handle = 9000u;
    auto add_foot = [&](uint32_t link, float x) {
        coresident::CoResidentFootSphere fs;
        fs.link = link;
        fs.broadphase_handle = handle++;
        fs.local_offset = Vec3{x, 0.0f, kFootBottomZ};
        fs.radius = kFootSphereR;
        sc.config.feet.push_back(fs);
    };
    for (uint32_t a : sc.ankle_links) { add_foot(a, kFootToeX); add_foot(a, kFootHeelX); }

    auto poses = ForwardKinematics(context, sc.host);
    float min_bottom = std::numeric_limits<float>::infinity();
    for (const auto& fs : sc.config.feet) {
        const Vec3 c = poses[fs.link].position + poses[fs.link].rotation.Rotate(fs.local_offset);
        min_bottom = std::min(min_bottom, c.z - fs.radius);
    }
    constexpr float kGroundZ = 0.0f;
    constexpr float kRestPen = 0.002f;
    sc.host.base_pose[0].position.z -= (min_bottom - (kGroundZ - kRestPen));
    sc.seat_base_z = sc.host.base_pose[0].position.z;
    sc.config.ground.height = kGroundZ;
    sc.config.friction_mu = 0.8f;
    sc.config.condim = 3u;

    // HOLD every other actuated DOF at its cooked rest q (the lumped->articulated gap).
    auto is_leg = [&](uint32_t l) {
        return std::find(sc.leg_links.begin(), sc.leg_links.end(), l) != sc.leg_links.end();
    };
    const uint32_t root = sc.host.articulation_link_offset[0];
    for (uint32_t l = 0u; l < sc.host.TotalLinkCount(); ++l) {
        if (l == root) continue;
        if (articulation::ArticulationJointDofCount(sc.host.joint_type[l]) == 0u) continue;
        if (is_leg(l)) continue;
        sc.hold_links.push_back(l);
        sc.hold_target.push_back(sc.host.q[l]);
        sc.hold_tlim.push_back(HoldLimitFor(LinkName(sc.h1, l)));
    }

    const uint32_t link_count = sc.host.TotalLinkCount();
    sc.config.drive_torque.assign(link_count, 0.0f);
    sc.config.drive_force_limits.assign(link_count, 0.0f);
    return sc;
}

// Assemble the Gate-2 obs from the live FULL host state (projects the full state to
// the shared subset; legs = the by-name leg links). Uses the SAME BuildObs as Gate 1a.
std::vector<float> Gate2Obs(const StandScene& sc,
                            const articulation::ArticulationHostState& st,
                            const std::array<uint32_t, 10>& leg_index) {
    const uint32_t root = sc.host.articulation_link_offset[0];
    std::array<float, 4> quat = {st.base_pose[0].rotation.w, st.base_pose[0].rotation.x,
                                 st.base_pose[0].rotation.y, st.base_pose[0].rotation.z};
    // base spatial velocity (omega-first), root-link body frame == link_velocity[root].
    std::array<float, 6> svel{};
    for (uint32_t i = 0u; i < 6u; ++i) svel[i] = st.link_velocity[root].v[i];
    const float bx = st.base_pose[0].position.x;
    const float by = st.base_pose[0].position.y;
    std::array<float, 10> qpos{}, qvel{};
    for (uint32_t s = 0u; s < 10u; ++s) {
        const uint32_t l = sc.leg_links[s];
        qpos[s] = st.q[l];          // per-link scalar q (NOT DofIndexOf).
        qvel[s] = st.qdot[l];
    }
    return BuildObs(quat, svel, bx, by, qpos, qvel, leg_index);
}

// ===========================================================================
// GATE 2: deploy the MLP in the FULL co-resident world and enforce S4 standing.
// ===========================================================================
TEST(H1BridgeSpike, Gate2ClosedLoopStandSoft) {
    if (!FileExists(kFullMjcf)) GTEST_SKIP() << "h1_with_hand asset not available";
    if (!FileExists(kWeightsBin))
        GTEST_SKIP() << "bridge weights missing (run examples/training/export_h1_tiny_actor_bridge.py)";

    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const auto mlp = nuka::test::TinyMlp<32, 64, 10>::Load(kWeightsBin);
    StandScene sc = BuildStandScene(context);
    const auto id = IdentityLegIndex();

    std::printf("\n#### GATE 2: closed-loop MLP stand in the FULL co-resident world ####\n");
    std::printf("  total mass=%.2f kg  seat z=%.4f  poly_cx(pinned)=%.4f  legs(by-name):",
                sc.com.total, sc.seat_base_z, kPolyCx);
    for (uint32_t s = 0u; s < 10u; ++s) std::printf(" %u", sc.leg_links[s]);
    std::printf("\n");

    coresident::UnifiedCoResidentStepper stepper(context, sc.host, sc.config,
                                                 kGravityZ, kDt);
    uint32_t policy_decimation = 1u;
    if (const char* env_decim = std::getenv("NUKA_H1_POLICY_DECIMATION")) {
        const int parsed = std::atoi(env_decim);
        if (parsed > 0) policy_decimation = static_cast<uint32_t>(parsed);
    }
    const uint32_t kControlSteps = 1000u;
    const uint32_t kSteps = kControlSteps * policy_decimation;
    const float half_len = 0.5f * (kFootToeX - kFootHeelX);

    articulation::ArticulationHostState st;
    stepper.Download(&st);
    const double base_z0 = st.base_pose[0].position.z;
    double base_z_min = base_z0, base_z_max = base_z0, base_z_final = base_z0;
    double tilt_max = 0.0, tilt_final = 0.0, com_ex_max = 0.0, min_edge = 1e9;
    uint32_t min_contacts = 4u, low_contact_steps = 0u;
    float peak_ankle = 0.0f, peak_knee = 0.0f, peak_hip = 0.0f;
    std::array<double, 4> min_foot_gap = {1e9, 1e9, 1e9, 1e9};
    std::array<double, 4> final_foot_gap = {0.0, 0.0, 0.0, 0.0};
    bool went_nan = false;

    // Diagnostic-only cross-world report: compare learned PPO leg torques against
    // the old handcoded stand controller on the same live full-world states. This is
    // NOT an acceptance metric; Gate 2 accepts on S4 behavior (upright, 4/4 contacts,
    // torque limits, no NaN). The diagnostic is retained only as a contrast against
    // the previous controller family.
    double sq_err = 0.0, sq_ref = 0.0;
    uint64_t n_pairs = 0u;
    double sq_err_id = 0.0, sq_ref_id = 0.0;
    uint64_t n_pairs_id = 0u;

    const bool trace = std::getenv("NUKA_H1_BRIDGE_TRACE") != nullptr;
    Vec3 com_prev{0, 0, 0};
    bool have_com_prev = false;
    std::vector<float> act(kActDim, 0.0f);
    std::vector<float> obs(kObsDim, 0.0f);
    for (uint32_t k = 0u; k < kSteps; ++k) {
        stepper.Download(&st);
        const double tilt_pre = TiltDeg(st.base_pose[0].rotation);  // in-distribution gate.
        if ((k % policy_decimation) == 0u) {
            obs = Gate2Obs(sc, st, id);
            act = mlp.Forward(obs);   // 10 normalized leg actions (contract order).
            for (float v : obs) went_nan = went_nan || !std::isfinite(v);
            for (float v : act) went_nan = went_nan || !std::isfinite(v);
            if (went_nan) {
                std::printf("    t=%u !! nonfinite obs/action\n", k + 1);
                break;
            }
        }

        // Build the per-link torque vector: legs from normalized MLP actions scaled
        // by the MJCF torque limits, then clamped to physical limits. The policy action
        // is held for policy_decimation physics ticks. The deploy gate defaults to
        // 200 Hz bridge inference; set NUKA_H1_POLICY_DECIMATION=4 to mirror training.
        std::vector<float> tau(st.TotalLinkCount(), 0.0f);
        for (uint32_t s = 0u; s < 10u; ++s) {
            const float lim = LegLimit(s);
            float u = std::max(-1.0f, std::min(1.0f, act[s])) * lim;
            u = std::max(-lim, std::min(lim, u));
            tau[sc.leg_links[s]] = u;
            const uint32_t j = s % 5u;
            const float au = std::fabs(u);
            if (j == 4u) peak_ankle = std::max(peak_ankle, au);
            else if (j == 3u) peak_knee = std::max(peak_knee, au);
            else peak_hip = std::max(peak_hip, au);
        }
        for (size_t i = 0u; i < sc.hold_links.size(); ++i) {
            const uint32_t l = sc.hold_links[i];
            float u = kKpHold * (sc.hold_target[i] - st.q[l]) - kKdHold * st.qdot[l];
            const float lim = sc.hold_tlim[i];
            tau[l] = std::max(-lim, std::min(lim, u));
        }

        // Diagnostic probe: the handcoded leg torque on THIS state
        // (the SAME law the batched gate + costand controller apply: posture PD legs
        // + CoP ankle from world CoM, incl. the CoM-velocity damping term via a
        // finite-diff of the whole-body CoM across steps).
        {
            auto poses = ForwardKinematics(context, st);
            const Vec3 com = WholeBodyCoM(sc.com, poses);
            const Vec3 com_v = have_com_prev ? (com - com_prev) * (1.0f / kDt)
                                             : Vec3{0, 0, 0};
            com_prev = com;
            have_com_prev = true;
            const float ex = com.x - kPolyCx;
            const float evx = com_v.x;
            const float kp[5] = {150.f, 200.f, 200.f, 300.f, 0.f};
            const float kd[5] = {8.f, 12.f, 14.f, 18.f, 0.f};
            const float tgt[5] = {0.f, 0.f, kStanceHipPitch, kStanceKnee, kStanceAnkle};
            const bool in_dist = tilt_pre < 15.0;  // pre-fall == training regime.
            for (uint32_t s = 0u; s < 10u; ++s) {
                const uint32_t j = s % 5u;
                const uint32_t l = sc.leg_links[s];
                float ref;
                if (j == 4u) ref = 320.0f * ex + 50.0f * evx - 1.5f * st.qdot[l];  // CoP.
                else ref = kp[j] * (tgt[j] - st.q[l]) - kd[j] * st.qdot[l];
                const float lim = LegLimit(s);
                ref = std::max(-lim, std::min(lim, ref));
                const float mlp_tau = std::max(-1.0f, std::min(1.0f, act[s])) * lim;
                const float d = mlp_tau - ref;
                sq_err += static_cast<double>(d) * d;
                sq_ref += static_cast<double>(ref) * ref;
                ++n_pairs;
                if (in_dist) {
                    sq_err_id += static_cast<double>(d) * d;
                    sq_ref_id += static_cast<double>(ref) * ref;
                    ++n_pairs_id;
                }
            }
        }

        stepper.SetGripTorque(tau);
        const auto rep = stepper.Step();
        stepper.Download(&st);

        const double tilt = TiltDeg(st.base_pose[0].rotation);
        const double bz = st.base_pose[0].position.z;
        auto poses2 = ForwardKinematics(context, st);
        for (size_t fi = 0u; fi < sc.config.feet.size() && fi < min_foot_gap.size(); ++fi) {
            const auto& fs = sc.config.feet[fi];
            const Vec3 fc = poses2[fs.link].position + poses2[fs.link].rotation.Rotate(fs.local_offset);
            const double gap = static_cast<double>(fc.z - fs.radius - sc.config.ground.height);
            min_foot_gap[fi] = std::min(min_foot_gap[fi], gap);
            final_foot_gap[fi] = gap;
        }
        const Vec3 com2 = WholeBodyCoM(sc.com, poses2);
        const double exv = com2.x - kPolyCx;
        const double edge = half_len - std::fabs(exv);
        if (!std::isfinite(tilt) || !std::isfinite(bz) || !std::isfinite(exv) ||
            !std::isfinite(edge)) {
            went_nan = true;
            std::printf("    t=%u !! nonfinite pose/com\n", k + 1);
            break;
        }
        for (uint32_t s = 0u; s < 10u; ++s) {
            const uint32_t l = sc.leg_links[s];
            if (!std::isfinite(st.q[l]) || !std::isfinite(st.qdot[l])) went_nan = true;
        }
        const uint32_t root = sc.host.articulation_link_offset[0];
        for (uint32_t i = 0u; i < 6u; ++i) {
            if (!std::isfinite(st.link_velocity[root].v[i])) went_nan = true;
        }
        if (went_nan) {
            std::printf("    t=%u !! nonfinite joint/root velocity state\n", k + 1);
            break;
        }
        tilt_max = std::max(tilt_max, tilt);
        tilt_final = tilt;
        base_z_min = std::min(base_z_min, bz);
        base_z_max = std::max(base_z_max, bz);
        base_z_final = bz;
        com_ex_max = std::max(com_ex_max, std::fabs(exv));
        min_edge = std::min(min_edge, edge);
        min_contacts = std::min(min_contacts, rep.foot_normal_rows);
        if (rep.foot_normal_rows < 4u) ++low_contact_steps;

        if ((k + 1u) % (100u * policy_decimation) == 0u || k < 4u) {
            std::printf("    t=%4u ctrl=%4u tilt=%5.2f z=%+.4f com_ex=%+.4f edge=%+.4f "
                        "contacts=%u/4 peak_an=%5.1f\n",
                        k + 1, (k + 1u) / policy_decimation, tilt, bz, exv, edge,
                        rep.foot_normal_rows, peak_ankle);
        }
        if (trace && (k < 8u || (k + 1u) == 100u * policy_decimation ||
                      (k + 1u) == kSteps)) {
            std::printf("    [TRACE t=%u ctrl=%u] obs0_12=", k + 1u, (k + 1u) / policy_decimation);
            for (uint32_t i = 0u; i < 12u; ++i) std::printf(" %.6f", obs[i]);
            std::printf(" action=");
            for (uint32_t i = 0u; i < kActDim; ++i) std::printf(" %.6f", act[i]);
            std::printf(" q=");
            for (uint32_t s = 0u; s < 10u; ++s) std::printf(" %.6f", st.q[sc.leg_links[s]]);
            std::printf(" qd=");
            for (uint32_t s = 0u; s < 10u; ++s) std::printf(" %.6f", st.qdot[sc.leg_links[s]]);
            std::printf(" foot_gap=%.6f %.6f %.6f %.6f\n",
                        final_foot_gap[0], final_foot_gap[1], final_foot_gap[2], final_foot_gap[3]);
        }
    }

    const double base_z_sink = base_z0 - base_z_min;
    const double mlp_rms = (n_pairs > 0u) ? std::sqrt(sq_err / n_pairs) : 0.0;
    const double ref_rms = (n_pairs > 0u) ? std::sqrt(sq_ref / n_pairs) : 0.0;
    const double mlp_rms_id = (n_pairs_id > 0u) ? std::sqrt(sq_err_id / n_pairs_id) : 0.0;
    const double ref_rms_id = (n_pairs_id > 0u) ? std::sqrt(sq_ref_id / n_pairs_id) : 0.0;

    const bool upright = !went_nan && tilt_max < 15.0 && tilt_final < 15.0;
    const bool height_stable = std::fabs(base_z_final - base_z0) < 0.12 && base_z_sink < 0.12;
    const bool com_in_poly = min_edge > 0.0;
    const bool contacts_held = min_contacts >= 2u;
    const bool contacts_4of4 = min_contacts >= 4u;
    const bool within_torque = peak_ankle <= 40.0f + 1e-2f && peak_knee <= 300.0f + 1e-2f
                               && peak_hip <= 200.0f + 1e-2f;
    const bool stood = upright && height_stable && com_in_poly && contacts_held;

    std::printf("  => [GATE2] tilt_max=%.2f tf=%.2f z %.4f->%.4f (min %.4f sink %.4f) "
                "com_ex_max=%.4f min_edge=%+.4f min_ct=%u/4 low_ct=%u | "
                "peak ankle=%.1f/40 knee=%.1f/300 hip=%.1f/200 | nan=%d\n",
                tilt_max, tilt_final, base_z0, base_z_final, base_z_min, base_z_sink,
                com_ex_max, min_edge, min_contacts, low_contact_steps,
                peak_ankle, peak_knee, peak_hip, went_nan ? 1 : 0);
    std::printf("  => [GATE2] verdict: stood=%d within_torque=%d contacts_4of4=%d "
                "(upright=%d height=%d com_in_poly=%d contacts_ge2=%d)\n",
                stood ? 1 : 0, within_torque ? 1 : 0, contacts_4of4 ? 1 : 0,
                upright ? 1 : 0, height_stable ? 1 : 0, com_in_poly ? 1 : 0,
                contacts_held ? 1 : 0);
    std::printf("  => [GATE2] foot gap min/final (Ltoe,Lheel,Rtoe,Rheel) = "
                "[%.4f/%.4f %.4f/%.4f %.4f/%.4f %.4f/%.4f] m\n",
                min_foot_gap[0], final_foot_gap[0], min_foot_gap[1], final_foot_gap[1],
                min_foot_gap[2], final_foot_gap[2], min_foot_gap[3], final_foot_gap[3]);
    // This RMS is a coarse learned-policy-vs-legacy-controller diagnostic. It is
    // intentionally separate from the hard S4 deploy-standing assertions below.
    std::printf("  => [GATE2 diagnostic] learned-vs-handcoded leg-torque RMS "
                "(cross-world diagnostic, NOT an acceptance metric): tilt<15deg "
                "window (%llu steps) = %.2f N*m (ref %.2f) | full window "
                "(%llu steps) = %.1f N*m (ref %.1f). Gate 1 + Gate 2 assertions "
                "are the bridge/deploy verdict.\n",
                static_cast<unsigned long long>(n_pairs_id / 10u), mlp_rms_id, ref_rms_id,
                static_cast<unsigned long long>(n_pairs / 10u), mlp_rms, ref_rms);

    if (stood) {
        std::printf("  => [GATE2] STANDS: the bridged MLP closes the loop in the full world.\n");
    } else {
        std::printf("  => [GATE2] does NOT satisfy S4 deploy-standing metrics.\n");
    }
    ASSERT_TRUE(stood);
    ASSERT_TRUE(within_torque);
    ASSERT_TRUE(contacts_4of4);
}

}  // namespace
