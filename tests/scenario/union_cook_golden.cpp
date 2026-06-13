// ---------------------------------------------------------------------------
// union_cook_golden — GOLDEN regression gate for the .nks union cook.
//
// REPLACES the M7-deleted tests/scenario/union_cook_matches_factory.cpp (which
// pinned CookSceneToUnionTemplate(.nks) == BuildH1UnionScene() to ~1e-8 before
// T6 deleted both it and the 1026-line factory). After that deletion NOTHING
// tightly pinned the cook output: h1_union_parity builds BOTH worlds from the
// SAME cooked tmpl (an IC error cancels in the diff — it cannot catch a cook
// bug), and h1_grasp_lift's absolute physics bands are loose (±35% force
// balance, 3.7–11× disturbance headroom) — a moderately-wrong settled q could
// keep them all green.
//
// This gate closes that hole with a GOLDEN-comparison oracle (D1 discipline):
// the golden below was generated ONCE from the T4-verified-correct cook and is
// NEVER regenerated — a RED here == a cook / .nks regression (NOT a golden to
// refresh). The cook is deterministic (two cook runs are byte-identical, every
// hex-float literal matched), so the floats reproduce to near-ULP; we still
// compare them with a TIGHT EXPECT_NEAR (kFloatTol 1e-6, far below the physics
// bands) per the spec, and EXPECT_EQ every integer / count / index / order
// field. The .nks stores the authored values %.9g-binary32-lossless and the
// cook copies them verbatim, so these are genuinely reproducible.
//
// PINNED (the regression-sensitive cook outputs): dof_stride / base_dof /
// root_link / link_count; total_mass; cup_mass; poly_cx; place_found;
// table_height; the grip_dofs / grip_links (12 each); the 30 fingertips + 4
// feet (resolved link + broadphase handle + local offset + radius, IN ORDER —
// the union slot emission order); the cup BodyState (settled pose + inv_mass +
// inv_inertia); the settled articulation q (46, the baked IC) + base_pose; the
// three drive tables (45 entries each: link/dof/grip exact, target/kp/kd/tlim
// near); dof_torque_limit (51); the cup hull vertex count + a checksum; and the
// emitted UnionCsr slot order/count (4 FootSpherePlane + 30 FingerSphereHull +
// 1 BodyBoxPlane == 35) via BuildNkUnionModel. Asset-gated SKIP.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "runtime/coresident/h1_union_nk_model.hpp"
#include "scene/cook/union_cook.hpp"
#include "scene/cook/union_scene_constants.hpp"  // constants (post-factory home)
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace coresident = nuka::runtime::coresident;
namespace cook = nuka::scene::cook;
namespace nk = nuka::nk;

constexpr const char* kNksPath = "examples/scenes/h1_cup_table.nks";
// TIGHT: ~1e-6 is orders of magnitude below the h1_grasp_lift physics bands
// (±35% force balance) yet generous vs the observed byte-exact reproducibility.
constexpr float  kFloatTol = 1.0e-6f;
constexpr double kDoubleTol = 1.0e-6;

bool AssetsAvailable() {
    return std::filesystem::exists(kNksPath) &&
           std::filesystem::exists(cook::kH1UnionMjcfDefault) &&
           std::filesystem::exists(cook::kH1UnionCupDefault);
}

// ===========================================================================
// THE GOLDEN — generated ONCE from the T4-verified-correct cook
// (nuka_union_cook_golden_gen, deleted after capture). NEVER regenerate. Float
// literals are hex (bit-faithful to the cook output). A row in the fingertip /
// foot / drive tables is {link, bp/dof, ...}.
// ===========================================================================
struct Sphere { uint32_t link, bp; float ox, oy, oz, r; };
struct Drive  { uint32_t link, dof; float target, kp, kd, tlim; uint32_t grip; };

constexpr uint32_t kDofStride = 51u;
constexpr uint32_t kBaseDof   = 6u;
constexpr uint32_t kRootLink  = 0u;
constexpr uint32_t kLinkCount = 46u;
constexpr double   kTotalMass = 0x1.a21f4f001853bp+5;   // 52.265287399999998
constexpr float    kCupMass   = 0x1.99999ap-3f;          // 0.200000003
constexpr float    kPolyCx    = 0x1.3b80f4p-4f;          // 0.0770272762
constexpr bool     kPlaceFound = true;
constexpr float    kTableHeight = 0x1.00f49p+0f;         // 1.00373173
constexpr float    kAnkleTau0 = -0x1.5e43cep+3f;         // -10.9457769
constexpr float    kAnkleTau1 = -0x1.5e5f66p+3f;         // -10.9491453
constexpr float    kFrictionMu = 0x1.99999ap-1f;         // 0.800000012
constexpr float    kFootMu    = 0x1.99999ap-1f;          // 0.800000012
constexpr float    kTableMu   = 0x1.333334p-1f;          // 0.600000024
constexpr float    kGroundHeight = 0x0p+0f;
constexpr uint32_t kCondim    = 3u;
constexpr uint32_t kCupLocalIndex = 0u;
constexpr uint32_t kCupBp = 7000u, kGroundBp = 8000u, kTableBp = 8500u, kProxyBp = 7001u;

// cup BodyState (bodies_per_env[0]).
constexpr float kCupInvMass = 0x1.4p+2f;                 // 5
constexpr float kCupPx = 0x1.ac2182p-2f, kCupPy = -0x1.914178p-3f, kCupPz = 0x1.13ef68p+0f;
constexpr float kCupQw = 0x1.ffeep-1f, kCupQx = -0x1.a550c6p-13f,
                kCupQy = -0x1.c50a96p-7f, kCupQz = -0x1.2b625cp-7f;
constexpr float kCupIIx = 0x1.ae603ep+10f, kCupIIy = 0x1.ae603ep+10f, kCupIIz = 0x1.418066p+11f;

constexpr uint32_t kGripDofs[12]  = {39u,40u,41u,42u,43u,44u,45u,46u,47u,48u,49u,50u};
constexpr uint32_t kGripLinks[12] = {34u,35u,36u,37u,38u,39u,40u,41u,42u,43u,44u,45u};

constexpr Sphere kFingertips[30] = {
    {38u,9000u, 0x1.89374cp-8f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {38u,9001u, 0x1.0624dep-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {38u,9002u, 0x1.a9fbe8p-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {39u,9003u, 0x1.89374cp-8f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {39u,9004u, 0x1.0624dep-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {39u,9005u, 0x1.a9fbe8p-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {40u,9006u, 0x1.89374cp-8f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {40u,9007u, 0x1.0624dep-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {40u,9008u, 0x1.a9fbe8p-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {41u,9009u, 0x1.89374cp-8f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {41u,9010u, 0x1.0624dep-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {41u,9011u, 0x1.a9fbe8p-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {42u,9012u, 0x1.89374cp-8f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {42u,9013u, 0x1.0624dep-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {42u,9014u, 0x1.a9fbe8p-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {43u,9015u, 0x1.89374cp-8f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {43u,9016u, 0x1.0624dep-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {43u,9017u, 0x1.a9fbe8p-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {44u,9018u, 0x1.89374cp-8f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {44u,9019u, 0x1.0624dep-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {44u,9020u, 0x1.a9fbe8p-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {45u,9021u, 0x1.89374cp-8f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {45u,9022u, 0x1.0624dep-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {45u,9023u, 0x1.a9fbe8p-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {36u,9024u, 0x1.0624dep-8f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {36u,9025u, 0x1.89374cp-7f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {36u,9026u, 0x1.47ae14p-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {37u,9027u, 0x1.0624dep-8f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {37u,9028u, 0x1.89374cp-7f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
    {37u,9029u, 0x1.47ae14p-6f,0x0p+0f,0x0p+0f, 0x1.89374cp-8f},
};
constexpr Sphere kFeet[4] = {
    {5u,12000u,  0x1.99999ap-4f,0x0p+0f,-0x1.c28f5cp-5f, 0x1.99999ap-6f},
    {5u,12001u, -0x1.99999ap-4f,0x0p+0f,-0x1.c28f5cp-5f, 0x1.99999ap-6f},
    {10u,12002u, 0x1.99999ap-4f,0x0p+0f,-0x1.c28f5cp-5f, 0x1.99999ap-6f},
    {10u,12003u,-0x1.99999ap-4f,0x0p+0f,-0x1.c28f5cp-5f, 0x1.99999ap-6f},
};

constexpr uint32_t kProtoLinkCount = 46u;
constexpr float kProtoQ[46] = {
    0x0p+0f,0x1.ebe368p-15f,0x1.29242cp-11f,-0x1.5c3d66p-2f,0x1.bf343p-1f,-0x1.c0e898p-2f,
    0x1.0f8c0ap-14f,0x1.2903bcp-11f,-0x1.5abf2p-2f,0x1.bdad42p-1f,-0x1.bf5044p-2f,0x1.7aa7e6p-10f,
    0x1.7ec60cp-5f,-0x1.60c544p-14f,0x1.45e15ep-13f,0x1.3151c8p-5f,0x1.295de6p-13f,0x1.9aab68p-16f,
    0x1.1d0b02p-14f,0x1.fae506p-18f,0x1.8dbe0ep-19f,-0x1.bd075ap-23f,-0x1.8f938p-23f,-0x1.848d2cp-23f,
    -0x1.09301ep-23f,-0x1.17465p-23f,-0x1.08dbb8p-26f,-0x1.38dafep-25f,0x1.7ceb0ap-24f,0x1.7dd966p-5f,
    0x1.7129f8p-12f,0x1.1103ap-15f,0x1.3132bep-5f,-0x1.6bd9bcp-12f,0x1.003ffap+0f,0x1.003364p-1f,
    0x1.b2431ep-2f,0x1.b23338p-2f,0x1.1ebcfcp-1f,0x1.51ed8p-1f,0x1.1ebb4p-1f,0x1.51ecdep-1f,
    0x1.1eb7ep-1f,0x1.51eb5ep-1f,0x1.1eb5b4p-1f,0x1.51ea96p-1f,
};
// base_pose[0]: pos.xyz, quat.wxyz.
constexpr float kProtoBasePose[7] = {
    0x1.5a260ep-7f,0x1.3b4d62p-10f,0x1.f16da2p-1f,
    0x1.ff43eap-1f,-0x1.bc1da6p-11f,-0x1.b6a2a6p-5f,-0x1.6c460ep-16f,
};

constexpr Drive kDriveHold[45] = {
    {1u,6u, 0x0p+0f,0x1.2cp+7f,0x1p+3f,0x1.9p+7f, 0u},{2u,7u, 0x0p+0f,0x1.9p+7f,0x1.8p+3f,0x1.9p+7f, 0u},
    {3u,8u, -0x1.99999ap-2f,0x1.9p+7f,0x1.cp+3f,0x1.9p+7f, 0u},{4u,9u, 0x1.666666p-1f,0x1.2cp+8f,0x1.2p+4f,0x1.2cp+8f, 0u},
    {6u,11u, 0x0p+0f,0x1.2cp+7f,0x1p+3f,0x1.9p+7f, 0u},{7u,12u, 0x0p+0f,0x1.9p+7f,0x1.8p+3f,0x1.9p+7f, 0u},
    {8u,13u, -0x1.99999ap-2f,0x1.9p+7f,0x1.cp+3f,0x1.9p+7f, 0u},{9u,14u, 0x1.666666p-1f,0x1.2cp+8f,0x1.2p+4f,0x1.2cp+8f, 0u},
    {11u,16u, 0x1.7aa7e6p-10f,0x1.9p+5f,0x1p+2f,0x1.9p+7f, 0u},{12u,17u, 0x1.7ec60cp-5f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},
    {13u,18u, -0x1.60c544p-14f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},{14u,19u, 0x1.45e15ep-13f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},
    {15u,20u, 0x1.3151c8p-5f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},{16u,21u, 0x1.295de6p-13f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {17u,22u, 0x1.9aab68p-16f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{18u,23u, 0x1.1d0b02p-14f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {19u,24u, 0x1.fae506p-18f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{20u,25u, 0x1.8dbe0ep-19f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {21u,26u, -0x1.bd075ap-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{22u,27u, -0x1.8f938p-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {23u,28u, -0x1.848d2cp-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{24u,29u, -0x1.09301ep-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {25u,30u, -0x1.17465p-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{26u,31u, -0x1.08dbb8p-26f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {27u,32u, -0x1.38dafep-25f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{28u,33u, 0x1.7ceb0ap-24f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {29u,34u, 0x1.7dd966p-5f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},{30u,35u, 0x1.7129f8p-12f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},
    {31u,36u, 0x1.1103ap-15f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},{32u,37u, 0x1.3132bep-5f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},
    {33u,38u, -0x1.6bd9bcp-12f,0x1.9p+5f,0x1p+2f,0x1.8p+2f, 0u},{34u,39u, 0x1.003ffap+0f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {35u,40u, 0x1.003364p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{36u,41u, 0x1.b2431ep-2f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {37u,42u, 0x1.b23338p-2f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{38u,43u, 0x1.1ebcfcp-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {39u,44u, 0x1.51ed8p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{40u,45u, 0x1.1ebb4p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {41u,46u, 0x1.51ecdep-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{42u,47u, 0x1.1eb7ep-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {43u,48u, 0x1.51eb5ep-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{44u,49u, 0x1.1eb5b4p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {45u,50u, 0x1.51ea96p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{5u,10u, -0x1.dcee04p-2f,0x1.9p+8f,0x1.ep+4f,0x1.4p+5f, 0u},
    {10u,15u, -0x1.db57e6p-2f,0x1.9p+8f,0x1.ep+4f,0x1.4p+5f, 0u},
};
constexpr Drive kDriveRest[45] = {
    {1u,6u, 0x0p+0f,0x1.2cp+7f,0x1p+3f,0x1.9p+7f, 0u},{2u,7u, 0x0p+0f,0x1.9p+7f,0x1.8p+3f,0x1.9p+7f, 0u},
    {3u,8u, -0x1.99999ap-2f,0x1.9p+7f,0x1.cp+3f,0x1.9p+7f, 0u},{4u,9u, 0x1.666666p-1f,0x1.2cp+8f,0x1.2p+4f,0x1.2cp+8f, 0u},
    {6u,11u, 0x0p+0f,0x1.2cp+7f,0x1p+3f,0x1.9p+7f, 0u},{7u,12u, 0x0p+0f,0x1.9p+7f,0x1.8p+3f,0x1.9p+7f, 0u},
    {8u,13u, -0x1.99999ap-2f,0x1.9p+7f,0x1.cp+3f,0x1.9p+7f, 0u},{9u,14u, 0x1.666666p-1f,0x1.2cp+8f,0x1.2p+4f,0x1.2cp+8f, 0u},
    {11u,16u, 0x1.7aa7e6p-10f,0x1.9p+5f,0x1p+2f,0x1.9p+7f, 0u},{12u,17u, 0x1.7ec60cp-5f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},
    {13u,18u, -0x1.60c544p-14f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},{14u,19u, 0x1.45e15ep-13f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},
    {15u,20u, 0x1.3151c8p-5f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},{16u,21u, 0x1.295de6p-13f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {17u,22u, 0x1.9aab68p-16f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{18u,23u, 0x1.1d0b02p-14f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {19u,24u, 0x1.fae506p-18f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{20u,25u, 0x1.8dbe0ep-19f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {21u,26u, -0x1.bd075ap-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{22u,27u, -0x1.8f938p-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {23u,28u, -0x1.848d2cp-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{24u,29u, -0x1.09301ep-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {25u,30u, -0x1.17465p-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{26u,31u, -0x1.08dbb8p-26f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {27u,32u, -0x1.38dafep-25f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{28u,33u, 0x1.7ceb0ap-24f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {29u,34u, 0x1.7dd966p-5f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},{30u,35u, 0x1.7129f8p-12f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},
    {31u,36u, 0x1.1103ap-15f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},{32u,37u, 0x1.3132bep-5f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},
    {33u,38u, -0x1.6bd9bcp-12f,0x1.9p+5f,0x1p+2f,0x1.8p+2f, 0u},{34u,39u, 0x1.807ff4p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {35u,40u, 0x1.0066c8p-2f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{36u,41u, 0x1.64863cp-3f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {37u,42u, 0x1.64667p-3f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{38u,43u, 0x1.3d79f8p-2f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {39u,44u, 0x1.a3dbp-2f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{40u,45u, 0x1.3d768p-2f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {41u,46u, 0x1.a3d9bcp-2f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{42u,47u, 0x1.3d6fcp-2f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {43u,48u, 0x1.a3d6bcp-2f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{44u,49u, 0x1.3d6b68p-2f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {45u,50u, 0x1.a3d52cp-2f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{5u,10u, -0x1.dcee04p-2f,0x1.9p+8f,0x1.ep+4f,0x1.4p+5f, 0u},
    {10u,15u, -0x1.db57e6p-2f,0x1.9p+8f,0x1.ep+4f,0x1.4p+5f, 0u},
};
constexpr Drive kDriveClose[45] = {
    {1u,6u, 0x0p+0f,0x1.2cp+7f,0x1p+3f,0x1.9p+7f, 0u},{2u,7u, 0x0p+0f,0x1.9p+7f,0x1.8p+3f,0x1.9p+7f, 0u},
    {3u,8u, -0x1.99999ap-2f,0x1.9p+7f,0x1.cp+3f,0x1.9p+7f, 0u},{4u,9u, 0x1.666666p-1f,0x1.2cp+8f,0x1.2p+4f,0x1.2cp+8f, 0u},
    {6u,11u, 0x0p+0f,0x1.2cp+7f,0x1p+3f,0x1.9p+7f, 0u},{7u,12u, 0x0p+0f,0x1.9p+7f,0x1.8p+3f,0x1.9p+7f, 0u},
    {8u,13u, -0x1.99999ap-2f,0x1.9p+7f,0x1.cp+3f,0x1.9p+7f, 0u},{9u,14u, 0x1.666666p-1f,0x1.2cp+8f,0x1.2p+4f,0x1.2cp+8f, 0u},
    {11u,16u, 0x1.7aa7e6p-10f,0x1.9p+5f,0x1p+2f,0x1.9p+7f, 0u},{12u,17u, 0x1.7ec60cp-5f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},
    {13u,18u, -0x1.60c544p-14f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},{14u,19u, 0x1.45e15ep-13f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},
    {15u,20u, 0x1.3151c8p-5f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},{16u,21u, 0x1.295de6p-13f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {17u,22u, 0x1.9aab68p-16f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{18u,23u, 0x1.1d0b02p-14f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {19u,24u, 0x1.fae506p-18f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{20u,25u, 0x1.8dbe0ep-19f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {21u,26u, -0x1.bd075ap-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{22u,27u, -0x1.8f938p-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {23u,28u, -0x1.848d2cp-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{24u,29u, -0x1.09301ep-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {25u,30u, -0x1.17465p-23f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{26u,31u, -0x1.08dbb8p-26f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {27u,32u, -0x1.38dafep-25f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},{28u,33u, 0x1.7ceb0ap-24f,0x1.9p+5f,0x1p+2f,0x1p+0f, 0u},
    {29u,34u, 0x1.7dd966p-5f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},{30u,35u, 0x1.7129f8p-12f,0x1.9p+5f,0x1p+2f,0x1.4p+5f, 0u},
    {31u,36u, 0x1.1103ap-15f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},{32u,37u, 0x1.3132bep-5f,0x1.9p+5f,0x1p+2f,0x1.2p+4f, 0u},
    {33u,38u, -0x1.6bd9bcp-12f,0x1.9p+5f,0x1p+2f,0x1.8p+2f, 0u},{34u,39u, 0x1.2e5474p+0f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {35u,40u, 0x1.5c5c5ap-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{36u,41u, 0x1.354a84p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {37u,42u, 0x1.354292p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{38u,43u, 0x1.7ae5f2p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {39u,44u, 0x1.ae1676p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{40u,45u, 0x1.7ae436p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {41u,46u, 0x1.ae15d4p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{42u,47u, 0x1.7ae0d6p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {43u,48u, 0x1.ae1454p-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{44u,49u, 0x1.7adeaap-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},
    {45u,50u, 0x1.ae138cp-1f,0x1p+2f,0x1.99999ap-2f,0x0p+0f, 1u},{5u,10u, -0x1.dcee04p-2f,0x1.9p+8f,0x1.ep+4f,0x1.4p+5f, 0u},
    {10u,15u, -0x1.db57e6p-2f,0x1.9p+8f,0x1.ep+4f,0x1.4p+5f, 0u},
};
constexpr float kDofTorqueLimit[51] = {
    0x0p+0f,0x0p+0f,0x0p+0f,0x0p+0f,0x0p+0f,0x0p+0f,0x1.9p+7f,0x1.9p+7f,0x1.9p+7f,0x1.2cp+8f,
    0x1.4p+5f,0x1.9p+7f,0x1.9p+7f,0x1.9p+7f,0x1.2cp+8f,0x1.4p+5f,0x1.9p+7f,0x1.4p+5f,0x1.4p+5f,0x1.2p+4f,
    0x1.2p+4f,0x1.8p+2f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,
    0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1.4p+5f,0x1.4p+5f,0x1.2p+4f,0x1.2p+4f,0x1.8p+2f,0x1p+0f,
    0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,0x1p+0f,
};
// The cup hull is a pure cook-passthrough of the cup mesh (5388 floats); pin its
// vertex count + a deterministic checksum (cheaper than 5388 literals, still a
// LOUD tripwire on any silent geometry change).
constexpr uint32_t kHullVertCount = 5388u;          // == 1796 vertices * 3.
constexpr double   kHullChecksum  = -0x1.01e6c68f8p+3;  // Σ verts (-8.0594208529219031).

// helper: track worst float deviation so the [PASS] line can report it.
double g_max_dev = 0.0;
const char* g_max_dev_what = "(none)";
void Near(float got, float want, const char* what) {
    const double d = std::abs(static_cast<double>(got) - static_cast<double>(want));
    if (d > g_max_dev) { g_max_dev = d; g_max_dev_what = what; }
    EXPECT_NEAR(got, want, kFloatTol) << what;
}
void NearD(double got, double want, const char* what) {
    const double d = std::abs(got - want);
    if (d > g_max_dev) { g_max_dev = d; g_max_dev_what = what; }
    EXPECT_NEAR(got, want, kDoubleTol) << what;
}
void CmpSphere(const char* tag, size_t i, uint32_t link, uint32_t bp,
               float ox, float oy, float oz, float r, const Sphere& g) {
    EXPECT_EQ(link, g.link) << tag << " link " << i;
    EXPECT_EQ(bp, g.bp) << tag << " bp " << i;
    Near(ox, g.ox, tag); Near(oy, g.oy, tag); Near(oz, g.oz, tag);
    Near(r, g.r, tag);
}
void CmpDrive(const char* tag, size_t i, const coresident::H1UnionDriveEntry& e,
              const Drive& g) {
    EXPECT_EQ(e.link, g.link) << tag << " link " << i;
    EXPECT_EQ(e.dof, g.dof) << tag << " dof " << i;
    EXPECT_EQ(static_cast<uint32_t>(e.grip), g.grip) << tag << " grip " << i;
    Near(e.target, g.target, tag);
    Near(e.kp, g.kp, tag);
    Near(e.kd, g.kd, tag);
    Near(e.tlim, g.tlim, tag);
}

}  // namespace

// The GOLDEN gate: cook the .nks and pin every regression-sensitive field to the
// committed golden (EXACT ints/order, TIGHT-NEAR floats). A RED is a cook bug.
TEST(UnionCookGolden, CookMatchesCommittedGolden) {
    if (!AssetsAvailable())
        GTEST_SKIP() << "h1_cup_table.nks / h1_with_hand / cup not present";

    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    const cook::CookedUnionScene c = cook::CookSceneToUnionTemplate(scene, 1);
    const coresident::BatchedSceneTemplate& t = c.tmpl;

    // ----- shape metadata (EXACT) -----
    EXPECT_EQ(c.dof_stride, kDofStride);
    EXPECT_EQ(c.base_dof, kBaseDof);
    EXPECT_EQ(c.root_link, kRootLink);
    EXPECT_EQ(c.link_count, kLinkCount);

    // ----- scene scalars -----
    NearD(c.total_mass, kTotalMass, "total_mass");
    Near(c.cup_mass, kCupMass, "cup_mass");
    Near(c.poly_cx, kPolyCx, "poly_cx");
    EXPECT_EQ(c.place_found, kPlaceFound);
    Near(c.table_height, kTableHeight, "table_height");
    Near(t.table_height, kTableHeight, "tmpl.table_height");
    Near(c.ankle_settle_tau[0], kAnkleTau0, "ankle_tau0");
    Near(c.ankle_settle_tau[1], kAnkleTau1, "ankle_tau1");
    Near(t.friction_mu, kFrictionMu, "friction_mu");
    Near(t.foot_mu, kFootMu, "foot_mu");
    Near(t.table_mu, kTableMu, "table_mu");
    Near(t.ground.height, kGroundHeight, "ground.height");
    EXPECT_EQ(t.condim, kCondim);
    EXPECT_EQ(t.cup_local_index, kCupLocalIndex);
    EXPECT_TRUE(t.has_grasp);
    EXPECT_FALSE(t.has_ground);
    EXPECT_TRUE(t.has_feet);
    EXPECT_TRUE(t.has_table);
    EXPECT_EQ(t.cup.broadphase_body_id, kCupBp);
    EXPECT_EQ(t.ground.broadphase_id, kGroundBp);
    EXPECT_EQ(t.table_broadphase_id, kTableBp);
    EXPECT_EQ(t.cup_table_proxy_id, kProxyBp);

    // ----- cup BodyState (settled pose + mass + inertia) -----
    ASSERT_EQ(t.bodies_per_env.size(), 1u);
    const auto& b = t.bodies_per_env[0];
    Near(b.inv_mass, kCupInvMass, "cup.inv_mass");
    Near(b.position.x, kCupPx, "cup.pos.x");
    Near(b.position.y, kCupPy, "cup.pos.y");
    Near(b.position.z, kCupPz, "cup.pos.z");
    Near(b.orientation.w, kCupQw, "cup.quat.w");
    Near(b.orientation.x, kCupQx, "cup.quat.x");
    Near(b.orientation.y, kCupQy, "cup.quat.y");
    Near(b.orientation.z, kCupQz, "cup.quat.z");
    Near(b.inv_inertia.x, kCupIIx, "cup.ii.x");
    Near(b.inv_inertia.y, kCupIIy, "cup.ii.y");
    Near(b.inv_inertia.z, kCupIIz, "cup.ii.z");

    // ----- grip dofs / links (12, EXACT) -----
    ASSERT_EQ(c.grip_dofs.size(), 12u);
    ASSERT_EQ(c.grip_links.size(), 12u);
    for (size_t i = 0; i < 12u; ++i) {
        EXPECT_EQ(c.grip_dofs[i], kGripDofs[i]) << "grip_dof " << i;
        EXPECT_EQ(c.grip_links[i], kGripLinks[i]) << "grip_link " << i;
    }

    // ----- fingertips (30) + feet (4): resolved link + bp + offset + radius, IN
    // ORDER (the union slot emission order). -----
    ASSERT_EQ(t.fingertips.size(), 30u);
    for (size_t i = 0; i < 30u; ++i) {
        const auto& f = t.fingertips[i];
        CmpSphere("fingertip", i, f.link, f.broadphase_handle, f.local_offset.x,
                  f.local_offset.y, f.local_offset.z, f.radius, kFingertips[i]);
    }
    ASSERT_EQ(t.feet.size(), 4u);
    for (size_t i = 0; i < 4u; ++i) {
        const auto& f = t.feet[i];
        CmpSphere("foot", i, f.link, f.broadphase_handle, f.local_offset.x,
                  f.local_offset.y, f.local_offset.z, f.radius, kFeet[i]);
    }

    // ----- the settled articulation q (the baked IC) + base_pose -----
    const auto& gp = t.gripper_proto;
    EXPECT_EQ(gp.TotalLinkCount(), kProtoLinkCount);
    ASSERT_EQ(gp.q.size(), 46u);
    for (size_t i = 0; i < 46u; ++i) Near(gp.q[i], kProtoQ[i], "proto.q");
    ASSERT_EQ(gp.base_pose.size(), 1u);
    {
        const auto& bp = gp.base_pose[0];
        Near(bp.position.x, kProtoBasePose[0], "proto.base.px");
        Near(bp.position.y, kProtoBasePose[1], "proto.base.py");
        Near(bp.position.z, kProtoBasePose[2], "proto.base.pz");
        Near(bp.rotation.w, kProtoBasePose[3], "proto.base.qw");
        Near(bp.rotation.x, kProtoBasePose[4], "proto.base.qx");
        Near(bp.rotation.y, kProtoBasePose[5], "proto.base.qy");
        Near(bp.rotation.z, kProtoBasePose[6], "proto.base.qz");
    }

    // ----- the three drive tables (45 each; link/dof/grip EXACT, gains NEAR) ----
    ASSERT_EQ(c.drive_hold.size(), 45u);
    ASSERT_EQ(c.drive_rest.size(), 45u);
    ASSERT_EQ(c.drive_close.size(), 45u);
    for (size_t i = 0; i < 45u; ++i) {
        CmpDrive("drive_hold", i, c.drive_hold[i], kDriveHold[i]);
        CmpDrive("drive_rest", i, c.drive_rest[i], kDriveRest[i]);
        CmpDrive("drive_close", i, c.drive_close[i], kDriveClose[i]);
    }

    // ----- per-DOF torque limits (51) -----
    ASSERT_EQ(c.dof_torque_limit.size(), 51u);
    for (size_t i = 0; i < 51u; ++i)
        Near(c.dof_torque_limit[i], kDofTorqueLimit[i], "dof_torque_limit");

    // ----- cup hull: vertex count + checksum (geometry tripwire) -----
    EXPECT_EQ(t.cup.hull_verts.size(), static_cast<size_t>(kHullVertCount));
    double hull_sum = 0.0;
    for (float v : t.cup.hull_verts) hull_sum += static_cast<double>(v);
    NearD(hull_sum, kHullChecksum, "cup.hull_checksum");

    std::printf("[COOK GOLDEN] dof=%u links=%u fingertips=%zu feet=%zu "
                "cup_z=%.6f table_z=%.6f total_mass=%.6f | hull_verts=%zu "
                "hull_sum=%.6f | max float dev=%.3e on %s\n",
                c.dof_stride, c.link_count, t.fingertips.size(), t.feet.size(),
                b.position.z, c.table_height, c.total_mass, t.cup.hull_verts.size(),
                hull_sum, g_max_dev, g_max_dev_what);
}

// The emitted UnionCsr slot order/count: BuildNkUnionModel(cooked.tmpl) -> 4
// FootSpherePlane + 30 FingerSphereHull + 1 BodyBoxPlane == 35, IN THAT ORDER
// (the union slot emission order the spec asks to pin). Deterministic from the
// cooked template; a reorder/miscount here is a cook-bridge regression.
TEST(UnionCookGolden, CookedTemplateEmitsGoldenUnionSlots) {
    if (!AssetsAvailable())
        GTEST_SKIP() << "h1_cup_table.nks / h1_with_hand / cup not present";

    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    const cook::CookedUnionScene c = cook::CookSceneToUnionTemplate(scene, 1);
    const nk::Model model = coresident::BuildNkUnionModel(c.tmpl, 1u);

    EXPECT_EQ(model.contact_family, nk::ContactFamily::UnionCsr);
    ASSERT_EQ(model.union_slots.size(), 35u);
    for (uint32_t i = 0u; i < 4u; ++i)
        EXPECT_EQ(model.union_slots[i].cls, nk::UnionSlot::kFootSpherePlane) << i;
    for (uint32_t i = 4u; i < 34u; ++i)
        EXPECT_EQ(model.union_slots[i].cls, nk::UnionSlot::kFingerSphereHull) << i;
    EXPECT_EQ(model.union_slots[34].cls, nk::UnionSlot::kBodyBoxPlane);
    EXPECT_EQ(model.capacities.dofs_per_env, kDofStride);
    EXPECT_EQ(model.capacities.links_per_env, kLinkCount);
    EXPECT_EQ(model.capacities.bodies_per_env, 1u);
    EXPECT_TRUE(model.table_enabled_default);
    ASSERT_EQ(model.body_init.size(), 1u);

    std::printf("[COOK GOLDEN NK] slots=%zu (4 foot + 30 finger + 1 box) "
                "rows/env=%u hull_verts=%u\n",
                model.union_slots.size(), model.capacities.max_rows_per_env,
                model.capacities.max_hull_verts);
}
