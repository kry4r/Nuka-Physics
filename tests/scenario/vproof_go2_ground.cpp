// ---------------------------------------------------------------------------
// V-PROOF — go2 feet on FLAT ground: LEGACY (FusedFoot) vs GENERAL (PairDriven +
// CookHeightfieldGrid) equivalence harness.
//
// The one-general-solver-spec (§4 / §L0) collapses the THREE contact families
// (FusedFoot / UnionCsr / PairDriven) onto the ONE general path. Before the
// collapse re-baselines the byte-pinned go2 golden, we must PROVE the general
// path reproduces the legacy foot-on-ground MOTION to a physical tolerance
// (spec §8.2 default: joint |dq| <= 1e-3 rad, base pose <= 1e-3 m). This harness
// cooks ONE go2 scene two ways and steps both under gravity + the cooked PD hold
// drive, then asserts per-joint + base agreement.
//
//   LEGACY  : CookToModel(scene, 1)            (default family == FusedFoot)
//             + keep cooked feet + ground_height = kGround  (the analytic
//             sphere-bottom-vs-+Z-plane foot detection -> fused per-articulation
//             PGS, the byte-pinned go2 path).
//   GENERAL : CookToModel(scene, 1, {PairDriven})  (LBVH -> cvx GJK/EPA + the
//             general mixed-island solve) + CookHeightfieldGrid(kTerrainFlat) at
//             the SAME height + capacity grow (the heightfield static collidable
//             enters the broadphase as one big leaf; the foot spheres collide
//             against its per-cell TRIANGLE_PRISM via the SAME cvx narrowphase).
//
// HONESTY (spec §"HONESTY IS THE DELIVERABLE"): if the general path does NOT match
// within tolerance this FAILS with the worst joint+contact-count diagnostic. We do
// NOT loosen the tolerance, fake agreement, or special-case to force a pass — the
// FAIL precisely tells the collapse what it must reconcile (foot-contact-point
// convention, Baumgarte/slop/margin, etc.).
//
// go2_stand cooks a FIXED-base articulation; with the ground seated at the
// perf-gate kGround = 0.31 the feet load against the surface (the legacy
// NkWorldBatchedContactStepPlannedByteExact perf-gate seat) so this exercises
// REAL foot-vs-ground contacts on both paths.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include "import/usd_importer.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "sensor/terrain/terrain_field.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace terrain = nuka::terrain;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr float kDt = 1.0f / 240.0f;
constexpr float kGravityZ = -9.81f;
constexpr float kGround = 0.31f;     // perf-gate seat (feet load on both paths).
constexpr uint32_t kSteps = 200u;
constexpr float kJointTol = 1.0e-3f;  // spec §8.2 default.
constexpr float kBaseTol = 1.0e-3f;   // spec §8.2 default.

std::filesystem::path StandScenePath() {
    return std::filesystem::path(NUKA_SOURCE_DIR) / "examples" / "scenes" /
           "go2_stand.usda";
}

struct Backend {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
};
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = kDt;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = kGravityZ;
    cfg.contact_margin = 0.0f;
    return cfg;
}

// The cooked-from-scene LEGACY (FusedFoot) go2: keep feet + seat the ground.
nk::Model CookLegacy(const nuka::scene::SceneIR& scene) {
    nk::Model m = cook::CookToModel(scene, 1).model;
    EXPECT_EQ(m.feet.size(), 4u) << "go2 cooks 4 foot spheres";
    m.ground_height = kGround;
    return m;
}

// The cooked-from-scene GENERAL (PairDriven) go2 + a FLAT heightfield collidable
// at the SAME height. The heightfield is appended at the body row that the
// PairDriven cook used for the static ground PLANE (== bodies_per_env), so it
// REPLACES that plane row 1:1; body_init is pre-grown to bodies_per_env so the
// heightfield's static pose seeds at its matching index, and bodies_per_env is
// grown to admit it into the LBVH broadphase (spec API anchor).
nk::Model CookGeneral(const nuka::scene::SceneIR& scene, uint32_t* out_hf_row) {
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    nk::Model m = cook::CookToModel(scene, 1, opt).model;
    const uint32_t orig_bodies = m.capacities.bodies_per_env;

    // Pre-grow body_init so the heightfield lands at index orig_bodies (the
    // static ground PLANE's shape_table row) and its static pose is seeded there.
    // The articulation-link body rows [0, orig_bodies) get default body_init
    // (their pose is overwritten every step by SyncLinkBodyPose from FK).
    m.body_init.resize(orig_bodies);

    terrain::TerrainParams tp = terrain::DefaultTerrainParams();
    tp.ground_height = kGround;
    const uint32_t hf_row = cook::CookHeightfieldGrid(
        m, tp, terrain::kTerrainFlat, /*nrow=*/21u, /*ncol=*/21u, /*cell=*/0.25f,
        /*center=*/Vec3{0, 0, kGround});
    EXPECT_EQ(hf_row, orig_bodies)
        << "heightfield must replace the static ground PLANE row at bodies_per_env";
    if (out_hf_row) *out_hf_row = hf_row;

    nk::ModelCapacities& cap = m.capacities;
    cap.bodies_per_env = static_cast<uint32_t>(m.body_init.size());
    cap.max_bodies_total = static_cast<uint32_t>(m.shape_table_rows.size());
    cap.max_contacts_per_env = 32u;
    cap.max_rows_per_env = 32u * nk::kPairDrivenRowsPerSlot;
    return m;
}

struct Snap {
    std::vector<float> q;     // per-link joint coords
    Transform base;          // articulation base pose
    uint32_t contacts = 0u;  // active contact count
    bool finite = true;
};

bool Finite(const Transform& t) {
    return std::isfinite(t.position.x) && std::isfinite(t.position.y) &&
           std::isfinite(t.position.z) && std::isfinite(t.rotation.w);
}

Snap StepAndRead(nk::World& w) {
    Snap r;
    const uint32_t L = w.GetModel().capacities.links_per_env;
    for (uint32_t s = 0; s < kSteps; ++s) {
        const nk::StepResult sr = w.Step();
        EXPECT_TRUE(sr.AllOk()) << "step " << s;
        if (!sr.AllOk()) { r.finite = false; return r; }
    }
    r.q.assign(L, 0.0f);
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Q, r.q.data(),
                                          L * sizeof(float)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BasePose, &r.base,
                                          sizeof(Transform)));
    w.GetData().DownloadField(nk::FieldId::ContactCount, &r.contacts,
                              sizeof(uint32_t));
    r.finite = true;
    for (float v : r.q) if (!std::isfinite(v)) r.finite = false;
    if (!Finite(r.base)) r.finite = false;
    return r;
}

}  // namespace

// ===========================================================================
// THE V-PROOF: legacy FusedFoot vs general PairDriven+heightfield must agree to
// the physical tolerance, with REAL foot-vs-ground contacts loaded on both sides.
// ===========================================================================
TEST(VProofGo2Ground, GeneralReproducesFusedFootOnFlatGround) {
    if (!std::filesystem::exists(StandScenePath()))
        GTEST_SKIP() << "go2_stand.usda not present";
    Backend b = GetBackend();
    if (b.backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    const auto scene = nuka::import::LoadUsd(StandScenePath().string());

    // ---- LEGACY (FusedFoot) ----
    Snap leg;
    {
        nk::Model m = CookLegacy(scene);
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        ASSERT_TRUE(w.Ready());
        leg = StepAndRead(w);
    }

    // ---- GENERAL (PairDriven + flat heightfield) ----
    uint32_t hf_row = ~0u;
    Snap gen;
    {
        nk::Model m = CookGeneral(scene, &hf_row);
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        ASSERT_TRUE(w.Ready());
        gen = StepAndRead(w);
    }

    ASSERT_EQ(leg.q.size(), gen.q.size());
    const uint32_t L = static_cast<uint32_t>(leg.q.size());

    // worst per-joint divergence + worst base-position component.
    float worst_dq = 0.0f;
    uint32_t worst_j = 0u;
    for (uint32_t j = 0; j < L; ++j) {
        const float d = std::abs(gen.q[j] - leg.q[j]);
        if (d > worst_dq) { worst_dq = d; worst_j = j; }
    }
    const Vec3 dbase = gen.base.position - leg.base.position;
    const float worst_dbase =
        std::max({std::abs(dbase.x), std::abs(dbase.y), std::abs(dbase.z)});

    std::fprintf(stderr,
        "[VPROOF go2-ground] hf_row=%u | LEGACY contacts=%u GENERAL contacts=%u | "
        "worst |dq|=%.6e rad at link %u (gen=%.6f leg=%.6f) | worst |dbase|=%.6e m "
        "(d=%.4f,%.4f,%.4f)\n",
        hf_row, leg.contacts, gen.contacts, worst_dq, worst_j,
        gen.q[worst_j], leg.q[worst_j], worst_dbase, dbase.x, dbase.y, dbase.z);
    // Per-link breakdown for any joint beyond tolerance (the collapse's worklist).
    for (uint32_t j = 0; j < L; ++j) {
        const float d = std::abs(gen.q[j] - leg.q[j]);
        if (d > kJointTol)
            std::fprintf(stderr,
                "    [diff] link %u: gen=%.6f leg=%.6f |d|=%.6e\n",
                j, gen.q[j], leg.q[j], d);
    }

    // (1) Both sides FINITE.
    ASSERT_TRUE(leg.finite) << "LEGACY side produced NaN/inf";
    ASSERT_TRUE(gen.finite) << "GENERAL side produced NaN/inf";

    // (2) FEET LOAD on both sides (the general side must NOT fall through the
    // heightfield — a 0-contact general side would be a silent no-collision).
    EXPECT_GT(leg.contacts, 0u) << "LEGACY feet did not load on the ground";
    EXPECT_GT(gen.contacts, 0u)
        << "GENERAL feet did not load (fell through the flat heightfield)";

    // (3) The EQUIVALENCE assertion (spec §8.2 default tolerance — NOT loosened).
    EXPECT_LE(worst_dq, kJointTol)
        << "GENERAL path joint trajectory diverges from FusedFoot by " << worst_dq
        << " rad (> " << kJointTol << ") at link " << worst_j
        << " — the family collapse must reconcile this BEFORE re-baselining the "
           "go2 golden (legacy contacts=" << leg.contacts
        << ", general contacts=" << gen.contacts << ").";
    EXPECT_LE(worst_dbase, kBaseTol)
        << "GENERAL path base position diverges from FusedFoot by " << worst_dbase
        << " m (> " << kBaseTol << ").";
}

// ===========================================================================
// Two-run BIT-IDENTITY on the GENERAL side (the new path's own determinism / D1).
// ===========================================================================
TEST(VProofGo2Ground, GeneralPathTwoRunByteIdentical) {
    if (!std::filesystem::exists(StandScenePath()))
        GTEST_SKIP() << "go2_stand.usda not present";
    Backend b = GetBackend();
    if (b.backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    const auto scene = nuka::import::LoadUsd(StandScenePath().string());
    auto run = [&]() -> Snap {
        uint32_t hf = ~0u;
        nk::Model m = CookGeneral(scene, &hf);
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        EXPECT_TRUE(w.Ready());
        return StepAndRead(w);
    };
    const Snap a = run();
    const Snap c = run();
    ASSERT_EQ(a.q.size(), c.q.size());
    const int q_cmp =
        std::memcmp(a.q.data(), c.q.data(), a.q.size() * sizeof(float));
    const int base_cmp = std::memcmp(&a.base, &c.base, sizeof(Transform));
    std::fprintf(stderr,
        "[VPROOF go2-ground] general two-run: q memcmp=%d base memcmp=%d "
        "(0 == byte-identical)\n",
        q_cmp, base_cmp);
    EXPECT_EQ(q_cmp, 0) << "general-path q diverged across two runs (non-deterministic)";
    EXPECT_EQ(base_cmp, 0) << "general-path base pose diverged across two runs";
}
