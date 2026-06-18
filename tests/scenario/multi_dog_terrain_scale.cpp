// ---------------------------------------------------------------------------
// SCALE harness: K=20 Go2 articulations co-resident in ONE env of ONE general
// nk::World on PyramidStairs terrain. The PairDriven block-island solver builds
// ONE island per env holding ALL of that env's rows; with K=20 dogs the per-env
// row slice is ~22400 rows. This test pins the storage cost of that island so the
// world can be CONSTRUCTED (no dynamic-shared overflow) and STEPPED.
//
// It cooks 20 composed go2 into one env (the multi_dog_costep recipe), seats a
// general heightfield collidable (PyramidStairs), sizes the per-env candidate /
// row budget the SAME way the broadphase does (collidables * 4 candidate pairs,
// each slot kPairDrivenRowsPerSlot rows), steps ~100 steps, and asserts the world
// is Ready (no cudaFuncSetAttribute overflow), every step is Ok, the downloaded
// state is finite, feet load, and no foot sinks below the terrain beyond a slop.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "import/usd_importer.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"
#include "scene/terrain/heightfield.hpp"
#include "scene/terrain/heightfield_loaders.hpp"
#include "scene/terrain/heightfield_sample.hpp"

namespace {

namespace nphi = nuka::phi;
namespace nk = nuka::nk;
namespace cook = nuka::scene::cook;
namespace terrain = nuka::terrain;
using nuka::math::Transform;
using nuka::math::Vec3;

// The broadphase candidate-pair budget per collidable (the cook's
// kCandidatePairsPerCollidable / lbvh_traversal kCandidatePairsPerCollidable).
constexpr uint32_t kCandidatePairsPerCollidable = 4u;
constexpr float kSinkTol = 0.05f;  // a real clip-through is tens of cm.

std::filesystem::path Go2ScenePath() {
    return std::filesystem::path(NUKA_SOURCE_DIR) / "examples" / "scenes" /
           "go2_stand.usda";
}

nk::Model CookKDogs(uint32_t k, float spacing) {
    nuka::scene::SceneIR scene = nuka::import::LoadUsd(Go2ScenePath().string());
    for (uint32_t i = 1; i < k; ++i) {
        Transform place = Transform::Identity();
        place.position.x = spacing * static_cast<float>(i);
        scene = nuka::scene::Compose(scene,
                                     nuka::import::LoadUsd(Go2ScenePath().string()),
                                     place, "dog" + std::to_string(i) + "_");
    }
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    return cook::CookToModel(scene, 1, opt).model;
}

// A Chebyshev staircase HeightField (8 rings * 0.04 = 0.32 m plateau) spanning the
// 20-dog line. Pure parameters -- no terrain-type selector.
terrain::HeightField TerrainHeightField(uint32_t nrow, uint32_t ncol, float cell) {
    terrain::TerrainGenConfig cfg;
    cfg.nrow = nrow;
    cfg.ncol = ncol;
    cfg.cell_x = cell;
    cfg.cell_y = cell;
    cfg.origin = Vec3{-0.5f * static_cast<float>(ncol - 1u) * cell,
                      -0.5f * static_cast<float>(nrow - 1u) * cell, 0.0f};
    cfg.base_z = 0.0f;
    cfg.ring_rise = 0.04f;
    cfg.ring_width = 0.40f;
    cfg.ring_platform = 2.0f;
    cfg.ring_count = 8u;
    terrain::HeightField hf;
    terrain::GenerateHeightField(cfg, hf);
    return hf;
}

}  // namespace

// ===========================================================================
// THE SCALE GATE: 20 dogs = ~22400 rows in ONE per-env island. The world MUST
// construct (no dynamic-shared overflow on the PairDriven solver) and step.
// ===========================================================================
TEST(MultiDogTerrainScale, TwentyDogsStepOnPyramidStairsNoSharedOverflow) {
    if (!std::filesystem::exists(Go2ScenePath())) {
        GTEST_SKIP() << "go2_stand.usda not present";
    }
    nphi::Device* dev = nphi::InitBestDevice();
    if (dev == nullptr) GTEST_SKIP() << "no compute device";
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    // Cook 20 go2 (composed 1.0 m apart in X) into ONE env on the general path.
    const uint32_t K = 20u;
    const float spacing = 1.0f;
    nk::Model model = CookKDogs(K, spacing);
    ASSERT_EQ(model.capacities.articulations_per_env, K);
    ASSERT_EQ(model.contact_family, nk::ContactFamily::PairDriven);
    const std::vector<nk::ModelFootShape> feet = model.feet;  // for the sink check.
    const uint32_t L = model.capacities.links_per_env;

    // Seat a general staircase heightfield collidable spanning all 20 dogs.
    const terrain::HeightField hf = TerrainHeightField(161u, 41u, 0.25f);
    const uint32_t orig_bodies = model.capacities.bodies_per_env;
    model.body_init.resize(orig_bodies);
    const uint32_t hf_row = cook::CookHeightfieldGrid(model, hf);
    ASSERT_EQ(hf_row, orig_bodies);

    nk::ModelCapacities& cap = model.capacities;
    cap.bodies_per_env = static_cast<uint32_t>(model.body_init.size());
    cap.max_bodies_total = static_cast<uint32_t>(model.shape_table_rows.size());
    // REALISTIC budget: the broadphase emits up to collidables * 4 candidate pairs,
    // each candidate slot expands to kPairDrivenRowsPerSlot rows. With ~280
    // collidables (20 dogs * ~14 collidable links + ground + heightfield) this is
    // ~1120 candidate slots and ~22400 rows -- the per-env island the solver stages.
    cap.max_contacts_per_env = cap.bodies_per_env * kCandidatePairsPerCollidable;
    cap.max_rows_per_env =
        cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;

    std::fprintf(stderr,
        "[scale 20-dog] K=%u L=%u bodies=%u max_contacts=%u max_rows=%u\n",
        K, L, cap.bodies_per_env, cap.max_contacts_per_env, cap.max_rows_per_env);

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u;
    cfg.max_pairs = 2048u;

    // Capture per-link collision geometry for the sink-through check before move.
    const std::vector<uint32_t> geom_kind = model.articulation.link_geom_kind;
    const std::vector<float> geom_params = model.articulation.link_geom_params;
    const std::vector<Transform> geom_local = model.articulation.link_geom_local;

    nk::World world(std::move(model), 1u, dev, backend, cfg);
    // (1) READY: the PairDriven solver's dynamic-shared carve must NOT overflow the
    // device opt-in limit for a ~22400-row per-env island (the bug this fixes).
    ASSERT_TRUE(world.Ready())
        << "world not Ready -- the 20-dog per-env island overflowed dynamic shared";

    // (2) STEP ~100 steps, all Ok.
    const uint32_t kSteps = 100u;
    for (uint32_t s = 0; s < kSteps; ++s) {
        ASSERT_TRUE(world.Step().AllOk()) << "step " << s;
    }

    // (3) FINITE state.
    std::vector<float> q(L, 0.0f);
    std::vector<Transform> link_pose(L);
    std::vector<float> base_pose(K * 7u, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Q, q.data(),
                                              L * sizeof(float)));
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::LinkPose,
                                              link_pose.data(),
                                              L * sizeof(Transform)));
    ASSERT_TRUE(world.GetData().DownloadField(
        nk::FieldId::BasePose, base_pose.data(), base_pose.size() * sizeof(float)));
    for (float v : q) ASSERT_TRUE(std::isfinite(v)) << "non-finite q";
    for (float v : base_pose) ASSERT_TRUE(std::isfinite(v)) << "non-finite base";

    // (4) feet load.
    uint32_t contacts = 0u;
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::ContactCount, &contacts,
                                              sizeof(uint32_t)));

    // (5) no foot sinks below the terrain beyond a slop. The feet table is per-DOG
    // (link indices are template-local); apply each foot to every co-resident dog's
    // link block (links_per_dog == L / K) and check the worst clearance over dog A.
    const uint32_t links_per_dog = L / K;
    float worst_sink = 1.0e9f;
    for (const auto& f : feet) {
        if (f.calf_local_link >= links_per_dog) continue;
        const Transform lp = link_pose[f.calf_local_link];  // dog A's link block.
        const Vec3 c = lp.TransformPoint(f.local_offset);
        const float bottom = c.z - f.radius;
        const float surf = terrain::SampleHeightFieldZ(hf, c.x, c.y);
        worst_sink = std::min(worst_sink, bottom - surf);
    }

    std::fprintf(stderr,
        "[scale 20-dog] contacts=%u base0=(%.4f,%.4f,%.4f) worst-foot-sink=%.4f m "
        "(<0 == 穿模)\n",
        contacts, base_pose[0], base_pose[1], base_pose[2], worst_sink);

    EXPECT_GT(contacts, 0u) << "feet did not load (fell through the terrain)";
    EXPECT_GT(worst_sink, -kSinkTol) << "a dog-A foot sank THROUGH the terrain (穿模)";
}
