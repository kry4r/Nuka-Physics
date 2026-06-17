// ---------------------------------------------------------------------------
// V-PROOF — go2 feet on PyramidStairs TERRAIN: GENERAL (PairDriven +
// CookHeightfieldGrid(kTerrainPyramidStairs)) physical-validity harness. The
// terrain companion to vproof_go2_ground.
//
// L1-b: the legacy FUSED foot-vs-terrain RUNTIME was DELETED, so the former
// LEGACY-vs-GENERAL equivalence half can no longer be PRODUCED. Per the L1-b spec
// this test is converted to assert the GENERAL path ALONE is physically valid on
// a discretized stair terrain: FINITE, feet LOAD (contacts > 0), and NO sink /
// tunnel through the surface beyond a small slop.
//
//   GENERAL : CookToModel(scene, 1, {PairDriven}) + CookHeightfieldGrid(
//             kTerrainPyramidStairs) + capacity grow. The feet collide against the
//             cooked per-cell TRIANGLE_PRISM grid via the cvx GJK/EPA narrowphase
//             -> general mixed-island solve.
//
// We seat a WIDE elevated platform (half_plat >> the dog footprint) so all four
// feet rest on the FLAT platform TOP at a single height. The platform top is
// seated near the foot level (~0.32 m) so the fixed-base feet reach it.
//
// HONESTY: we do NOT loosen tolerances or fake a pass — a foot that falls through,
// a NaN, or a deep sink FAILS with a diagnostic.
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
constexpr uint32_t kSteps = 200u;

constexpr float kSinkTol = 0.05f;  // a real clip-through is tens of cm.

std::filesystem::path StandScenePath() {
    return std::filesystem::path(NUKA_SOURCE_DIR) / "examples" / "scenes" /
           "go2_stand.usda";
}

// PyramidStairs seated so the WIDE flat platform TOP sits near the foot level.
terrain::TerrainParams TerrainSpec() {
    terrain::TerrainParams tp = terrain::DefaultTerrainParams();
    tp.ground_height = 0.0f;
    tp.step_height = 0.04f;      // 8 rings * 0.04 = 0.32 m platform top.
    tp.step_width = 0.40f;
    tp.platform_width = 2.0f;    // half_plat 1.0 >> the dog footprint (~0.2 m).
    return tp;
}
// The flat platform-top height the dog stands on (== ground + 8*step_height).
float PlatformTop() {
    const terrain::TerrainParams tp = TerrainSpec();
    return tp.ground_height +
           static_cast<float>(terrain::kPyramidRings) * tp.step_height;
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

struct Snap {
    std::vector<float> q;
    Transform base;
    std::vector<Transform> link_pose;  // for the sink-through check.
    uint32_t contacts = 0u;
    bool finite = true;
};

bool Finite(const Transform& t) {
    return std::isfinite(t.position.x) && std::isfinite(t.position.y) &&
           std::isfinite(t.position.z) && std::isfinite(t.rotation.w);
}

Snap ReadRun(nk::World& w) {
    Snap r;
    const uint32_t L = w.GetModel().capacities.links_per_env;
    r.q.assign(L, 0.0f);
    r.link_pose.assign(L, Transform::Identity());
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Q, r.q.data(),
                                          L * sizeof(float)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BasePose, &r.base,
                                          sizeof(Transform)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::LinkPose,
                                          r.link_pose.data(),
                                          L * sizeof(Transform)));
    w.GetData().DownloadField(nk::FieldId::ContactCount, &r.contacts,
                              sizeof(uint32_t));
    r.finite = Finite(r.base);
    for (float v : r.q) if (!std::isfinite(v)) r.finite = false;
    return r;
}

// GENERAL PairDriven go2 + CookHeightfieldGrid(PyramidStairs).
Snap RunGeneral(const nuka::scene::SceneIR& scene, Backend b, uint32_t* out_hf) {
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    nk::Model m = cook::CookToModel(scene, 1, opt).model;
    const uint32_t orig_bodies = m.capacities.bodies_per_env;
    m.body_init.resize(orig_bodies);  // heightfield seeds at index orig_bodies.

    const terrain::TerrainParams tp = TerrainSpec();
    // A grid wide enough to cover the dog footprint + the first rings; cell 0.25.
    const uint32_t hf_row = cook::CookHeightfieldGrid(
        m, tp, terrain::kTerrainPyramidStairs, /*nrow=*/41u, /*ncol=*/41u,
        /*cell=*/0.25f, /*center=*/Vec3{0, 0, 0});
    EXPECT_EQ(hf_row, orig_bodies);
    if (out_hf) *out_hf = hf_row;

    nk::ModelCapacities& cap = m.capacities;
    cap.bodies_per_env = static_cast<uint32_t>(m.body_init.size());
    cap.max_bodies_total = static_cast<uint32_t>(m.shape_table_rows.size());
    cap.max_contacts_per_env = 32u;
    cap.max_rows_per_env = 32u * nk::kPairDrivenRowsPerSlot;

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    EXPECT_TRUE(w.Ready());
    for (uint32_t s = 0; s < kSteps; ++s) EXPECT_TRUE(w.Step().AllOk()) << s;
    return ReadRun(w);
}

// Lowest world-Z of every collidable foot sphere, and its sink below the local
// terrain surface. feet are the cooked foot spheres (model.feet has the link +
// offset + radius). Returns the worst (most-negative) clearance over all feet.
float WorstFootSink(const Snap& r, const std::vector<nk::ModelFootShape>& feet,
                    const terrain::TerrainParams& tp) {
    float worst = 1.0e9f;
    for (const auto& f : feet) {
        if (f.calf_local_link >= r.link_pose.size()) continue;
        const Transform lp = r.link_pose[f.calf_local_link];
        const Vec3 c = lp.TransformPoint(f.local_offset);  // foot sphere center.
        const float bottom = c.z - f.radius;
        const float surf = terrain::SampleTerrainHeight(
            terrain::kTerrainPyramidStairs, c.x, c.y, tp);
        worst = std::min(worst, bottom - surf);
    }
    return worst;
}

}  // namespace

// ===========================================================================
// THE V-PROOF (terrain): the GENERAL heightfield path is physically valid for a
// go2 standing on a discretized stair terrain — finite, feet load, no sink-through.
// (L1-b: the FUSED reference half was deleted; this is the surviving general-path
// validity assertion.)
// ===========================================================================
TEST(VProofGo2Terrain, GeneralPathPhysicallyValidOnPyramidStairs) {
    if (!std::filesystem::exists(StandScenePath()))
        GTEST_SKIP() << "go2_stand.usda not present";
    Backend b = GetBackend();
    if (b.backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    const auto scene = nuka::import::LoadUsd(StandScenePath().string());
    // feet table (link/offset/radius) for the sink-through check.
    const std::vector<nk::ModelFootShape> feet =
        cook::CookToModel(scene, 1).model.feet;
    EXPECT_EQ(feet.size(), 4u);
    const terrain::TerrainParams tp = TerrainSpec();

    uint32_t hf_row = ~0u;
    const Snap gen = RunGeneral(scene, b, &hf_row);

    const float gen_sink = WorstFootSink(gen, feet, tp);

    std::fprintf(stderr,
        "[VPROOF go2-terrain] platform_top=%.3f hf_row=%u | GENERAL contacts=%u | "
        "base=(%.4f,%.4f,%.4f) | foot-sink=%.4f m (<0 == 穿模)\n",
        PlatformTop(), hf_row, gen.contacts, gen.base.position.x,
        gen.base.position.y, gen.base.position.z, gen_sink);

    // (1) finite.
    ASSERT_TRUE(gen.finite) << "GENERAL side produced NaN/inf";

    // (2) feet load (general must not fall through the terrain grid).
    EXPECT_GT(gen.contacts, 0u)
        << "GENERAL feet did not load (fell through the heightfield terrain)";

    // (3) NO sink-through (the body never tunnels the surface beyond a small slop).
    EXPECT_GT(gen_sink, -kSinkTol) << "GENERAL foot sank THROUGH the terrain (穿模)";
}
