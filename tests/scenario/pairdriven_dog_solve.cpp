// ---------------------------------------------------------------------------
// General contact pipeline — Phase 1B SOLVER, the ARTICULATION cases.
//
// The mixed-island coupling proof on REAL dogs, all through the GENERAL path
// (LBVH -> cvx narrowphase -> mixed island) — the ONE multi-body contact path
// (the dog_dog hack was DELETED in L1/D1):
//   (a) TWO co-resident floating-base dogs whose trunk boxes overlap collide via
//       the GENERAL path (artic x artic) and push apart with two-way reaction (the
//       reaction scatters into BOTH dogs' per-articulation qdot tiles -- S3).
//   (c) a FREE rigid box dropped on a dog's back PUSHES the dog (a MIXED island:
//       the free-rigid 6-DOF side coupled to the artic chain-J side -- the box's
//       weight loads the dog, the dog's trunk moves).
//
// Both cook with CookContactFamily::PairDriven. The migrated multi_dog_costep.cpp
// also runs its multi-dog cases on this SAME general path (the dog_dog hack is gone).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
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

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Transform;
using nuka::math::Vec3;

std::filesystem::path Go2FloatScenePath() {
    return std::filesystem::path(NUKA_SOURCE_DIR) / "examples" / "scenes" /
           "go2_float.usda";
}

// Cook K floating-base go2 (composed at distinct X) into ONE env via the GENERAL
// PairDriven family (the B1 cook overload). Returns the cooked Model.
nk::Model CookKFloatDogsPairDriven(uint32_t k, float spacing) {
    nuka::scene::SceneIR scene = nuka::import::LoadUsd(Go2FloatScenePath().string());
    for (uint32_t i = 1; i < k; ++i) {
        Transform place = Transform::Identity();
        place.position.x = spacing * static_cast<float>(i);
        scene = nuka::scene::Compose(scene,
                                     nuka::import::LoadUsd(Go2FloatScenePath().string()),
                                     place, "dog" + std::to_string(i) + "_");
    }
    nuka::scene::cook::CookToModelOptions opt;
    opt.contact_family = nuka::scene::cook::CookContactFamily::PairDriven;
    return nuka::scene::cook::CookToModel(scene, 1, opt).model;
}

nk::Pipeline::SolverConfig Cfg(float gz) {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = gz;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u;
    cfg.max_pairs = 64u;
    return cfg;
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

bool HaveAsset() { return std::filesystem::exists(Go2FloatScenePath()); }

}  // namespace

// --- (a) two dogs collide via the GENERAL path (artic x artic) ---------------
TEST(PairDrivenDogSolve, TwoDogsCollideViaGeneralPath) {
    if (!HaveAsset()) GTEST_SKIP() << "go2_float.usda not present";
    Backend b = GetBackend();
    if (b.backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    // Two dogs 0.15 m apart in X; trunk boxes 0.20 (half 0.10) -> 0.05 overlap.
    // With the GENERAL contact path (the ONE multi-body path) the trunks
    // MUST push apart, and the two-way reaction injects velocity into BOTH dogs.
    nk::Model model = CookKFloatDogsPairDriven(2u, 0.15f);
    ASSERT_EQ(model.capacities.articulations_per_env, 2u);
    ASSERT_EQ(model.contact_family, nk::ContactFamily::PairDriven);
    const uint32_t L = model.capacities.links_per_env;

    nk::World world(std::move(model), 1u, b.dev, b.backend, Cfg(0.0f));
    ASSERT_TRUE(world.Ready());

    // Spawn root positions (base_pose) before stepping.
    std::vector<float> base0(2u * 7u, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BasePose, base0.data(),
                                              base0.size() * sizeof(float)));
    const float spawn0 = base0[0 * 7 + 0];
    const float spawn1 = base0[1 * 7 + 0];

    for (uint32_t s = 0; s < 60u; ++s) ASSERT_TRUE(world.Step().AllOk()) << s;

    std::vector<float> base1(2u * 7u, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BasePose, base1.data(),
                                              base1.size() * sizeof(float)));
    const float x0 = base1[0 * 7 + 0];
    const float x1 = base1[1 * 7 + 0];
    const float sep0 = std::abs(spawn1 - spawn0);
    const float sep1 = std::abs(x1 - x0);

    // qdot energy (any nonzero -> the contact injected velocity).
    std::vector<float> qdot(L, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Qdot, qdot.data(),
                                              L * sizeof(float)));
    float qnorm = 0.0f;
    for (float q : qdot) if (std::isfinite(q)) qnorm += q * q;
    qnorm = std::sqrt(qnorm);

    std::fprintf(stderr,
                 "[pd two-dogs] spawn_sep=%.4f -> sep=%.4f  x0 %.4f->%.4f  x1 %.4f->%.4f"
                 "  qnorm=%.5f\n",
                 sep0, sep1, spawn0, x0, spawn1, x1, qnorm);

    // (1) finite (no explosion).
    for (uint32_t i = 0; i < 2u * 7u; ++i) ASSERT_TRUE(std::isfinite(base1[i]));
    // (2) PUSH-APART via the general path: separation grew past the spawn gap.
    EXPECT_GT(sep1, sep0 + 1.0e-3f)
        << "overlapping dogs must push apart through the GENERAL contact path";
    // (3) TWO-WAY: BOTH dogs moved from spawn (the reaction reached both qdot tiles).
    EXPECT_LT(x0, spawn0 - 1.0e-4f) << "dog A must be pushed in -X";
    EXPECT_GT(x1, spawn1 + 1.0e-4f) << "dog B must be pushed in +X";
}

// --- (a-control) far-apart dogs do NOT spuriously collide --------------------
TEST(PairDrivenDogSolve, FarApartDogsNoSpuriousContact) {
    if (!HaveAsset()) GTEST_SKIP() << "go2_float.usda not present";
    Backend b = GetBackend();
    if (b.backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    nk::Model model = CookKFloatDogsPairDriven(2u, 1.5f);  // 1.5 m apart: no overlap.
    const uint32_t L = model.capacities.links_per_env;
    nk::World world(std::move(model), 1u, b.dev, b.backend, Cfg(0.0f));
    ASSERT_TRUE(world.Ready());

    std::vector<float> base0(2u * 7u, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BasePose, base0.data(),
                                              base0.size() * sizeof(float)));
    for (uint32_t s = 0; s < 60u; ++s) ASSERT_TRUE(world.Step().AllOk()) << s;
    std::vector<float> base1(2u * 7u, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BasePose, base1.data(),
                                              base1.size() * sizeof(float)));
    std::vector<float> qdot(L, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Qdot, qdot.data(),
                                              L * sizeof(float)));
    float qnorm = 0.0f;
    for (float q : qdot) qnorm += q * q;
    qnorm = std::sqrt(qnorm);
    std::fprintf(stderr, "[pd two-dogs control] sep=%.4f qnorm=%.7f\n",
                 std::abs(base1[7] - base1[0]), qnorm);
    EXPECT_NEAR(std::abs(base1[7] - base1[0]), std::abs(base0[7] - base0[0]), 1.0e-2f)
        << "far-apart dogs must NOT move (no spurious general contact)";
    EXPECT_LT(qnorm, 1.0e-3f) << "far-apart dogs must stay at rest";
}

// --- two-run byte-identity for the PairDriven artic-artic path ---------------
TEST(PairDrivenDogSolve, TwoDogsTwoRunsByteIdentical) {
    if (!HaveAsset()) GTEST_SKIP() << "go2_float.usda not present";
    Backend b = GetBackend();
    if (b.backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    auto run = [&](std::vector<float>* base, std::vector<float>* qdot) -> bool {
        nk::Model model = CookKFloatDogsPairDriven(2u, 0.15f);
        const uint32_t L = model.capacities.links_per_env;
        nk::World world(std::move(model), 1u, b.dev, b.backend, Cfg(0.0f));
        if (!world.Ready()) return false;
        for (uint32_t s = 0; s < 40u; ++s)
            if (!world.Step().AllOk()) return false;
        base->assign(2u * 7u, 0.0f);
        qdot->assign(L, 0.0f);
        return world.GetData().DownloadField(nk::FieldId::BasePose, base->data(),
                                             base->size() * sizeof(float)) &&
               world.GetData().DownloadField(nk::FieldId::Qdot, qdot->data(),
                                             qdot->size() * sizeof(float));
    };
    std::vector<float> ba, qa, bc, qc;
    ASSERT_TRUE(run(&ba, &qa));
    ASSERT_TRUE(run(&bc, &qc));
    ASSERT_EQ(ba.size(), bc.size());
    ASSERT_EQ(qa.size(), qc.size());
    EXPECT_EQ(std::memcmp(ba.data(), bc.data(), ba.size() * sizeof(float)), 0)
        << "two-dog base_pose differs across runs (PairDriven path not deterministic)";
    EXPECT_EQ(std::memcmp(qa.data(), qc.data(), qa.size() * sizeof(float)), 0)
        << "two-dog qdot differs across runs";
}

// --- (c) a free box on a dog's back PUSHES the dog (MIXED island) -------------
TEST(PairDrivenDogSolve, FreeBoxOnDogBackPushesDog) {
    if (!HaveAsset()) GTEST_SKIP() << "go2_float.usda not present";
    Backend b = GetBackend();
    if (b.backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    // ONE dog (K==1) + a heavy free box dropped onto its trunk. The MIXED island
    // couples the free-rigid 6-DOF box side to the dog's artic chain-J side: the
    // box's weight loads the dog so the dog's trunk is pushed down (its base z
    // drops) -- the two-way mixed coupling on the ONE general contact path.
    nuka::scene::SceneIR scene = nuka::import::LoadUsd(Go2FloatScenePath().string());
    nuka::scene::cook::CookToModelOptions opt;
    opt.contact_family = nuka::scene::cook::CookContactFamily::PairDriven;
    nk::Model model = nuka::scene::cook::CookToModel(scene, 1, opt).model;
    ASSERT_EQ(model.capacities.articulations_per_env, 1u);
    const uint32_t L = model.capacities.links_per_env;

    // Append a free rigid box ABOVE the dog's trunk, given a DOWNWARD velocity so it
    // slams into the dog's back. Gravity OFF so the ONLY force is the box<->dog
    // contact: the MIXED island (free-rigid 6-DOF box side coupled to the dog's
    // floating-base chain-J side). The two-way reaction must (a) push the dog's
    // trunk DOWN (-Z base velocity) and (b) DECELERATE the box (its -Z speed drops).
    // The dog's trunk root spawn is z~0.445; place the box at z=0.60 moving -Z.
    nk::ModelCapacities& cap = model.capacities;
    const uint32_t box_row = cap.bodies_per_env;  // appended after the dog's body rows.
    {
        // body_init must be sized to ALL body rows (SeedInitialState indexes by
        // body row). The dog's link-body rows [0, box_row) are frozen (inv_mass 0,
        // integrated by FK); only the box at box_row is a movable free rigid body.
        model.body_init.assign(box_row + 1u, nk::Model::BodyInit{});
        nk::Model::BodyInit& bi = model.body_init[box_row];
        bi.pose = Transform::Identity();
        bi.pose.position = Vec3{0.0f, 0.0f, 0.60f};
        bi.linear_velocity = Vec3{0.0f, 0.0f, -1.0f};  // slam down onto the back.
        bi.inv_mass = 1.0f / 5.0f;          // 5 kg (heavy enough to load the dog).
        const float half = 0.08f;
        const float ii = 1.0f / (5.0f * half * half);
        bi.inv_inertia = Vec3{ii, ii, ii};
        nk::Model::PairDrivenShape sh;
        sh.kind = 2u;  // Box
        sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
        sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
        sh.body_id = static_cast<int32_t>(box_row);
        sh.group = 0u;
        model.shape_table_rows.insert(
            model.shape_table_rows.begin() + box_row, sh);
    }
    cap.bodies_per_env += 1u;
    cap.max_bodies_total = static_cast<uint32_t>(model.shape_table_rows.size());

    nk::World world(std::move(model), 1u, b.dev, b.backend, Cfg(0.0f));
    ASSERT_TRUE(world.Ready());

    const uint32_t bodies = cap.bodies_per_env;
    std::vector<float> linkvel0(L * 6u, 0.0f);  // spatial6 per link: trunk root = link 0.
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::LinkVelocity, linkvel0.data(),
                                              linkvel0.size() * sizeof(float)));

    for (uint32_t s = 0; s < 80u; ++s) ASSERT_TRUE(world.Step().AllOk()) << s;

    std::vector<Vec3> blin(bodies, Vec3::Zero());
    std::vector<Transform> bx(bodies, Transform::Identity());
    std::vector<float> linkvel1(L * 6u, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BodyLinearVelocity, blin.data(),
                                              blin.size() * sizeof(Vec3)));
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BodyPose, bx.data(),
                                              bx.size() * sizeof(Transform)));
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::LinkVelocity, linkvel1.data(),
                                              linkvel1.size() * sizeof(float)));
    // The trunk root is link 0; its spatial6 linear velocity (v) is the base lin vel.
    // Spatial6 layout is [w(0..2), v(3..5)] (omega-first); the linear Z is index 5.
    const float dog_vz = linkvel1[5];
    const float box_vz = blin[box_row].z;
    const float box_z = bx[box_row].position.z;

    std::fprintf(stderr,
                 "[pd box-on-dog] box_z=%.4f box_vz %.4f->%.4f  dog_root_vz=%.5f\n",
                 box_z, -1.0f, box_vz, dog_vz);

    for (auto& v : blin) { ASSERT_TRUE(std::isfinite(v.x) && std::isfinite(v.z)); }
    ASSERT_TRUE(std::isfinite(dog_vz));
    // (1) the box DECELERATED from -1.0 m/s (the dog's back resisted it via the
    // contact) -- it is no longer in free flight at the launch speed.
    EXPECT_GT(box_vz, -1.0f + 1.0e-2f)
        << "the box must decelerate on hitting the dog (contact reaction on the box)";
    // (2) MIXED COUPLING: the dog's trunk root received a DOWNWARD push from the box
    // (the free-rigid reaction scattered into the artic floating-base side). A dog
    // with no contact stays at rest (dog_root_vz == 0); here it is driven negative.
    EXPECT_LT(dog_vz, -1.0e-3f)
        << "the box impact must push the dog's trunk down (mixed-island two-way reaction)";
}
