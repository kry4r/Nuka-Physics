// ---------------------------------------------------------------------------
// WP1 multi-articulation FOUNDATION spike: K Go2 articulations co-resident in
// ONE env of ONE general nk::World (the dog-dog collision foundation, owner
// bottom-line). This proves the cook + model + schema re-key + staging put K
// SEPARATE Go2 articulations into one env (articulation_count == K) and that
// they CO-STEP independently under gravity.
//
// IMPORTANT SCOPE: this is the FORWARD-DYNAMICS co-residence foundation, NOT
// dog-dog contact (that is the later contact crux, WP5-8). The dogs do NOT
// physically touch here -- the spike validates that K articulations co-step in
// one general world with each dog keeping its OWN per-articulation tile (the
// 64-DOF cap stays safe; dofs_per_env stays the per-DOG DOF, never summed), and
// that one dog's joints do NOT move another dog's joints (no qdot-tile aliasing).
//
// The 2 dogs are built by composing the go2_stand scene with itself at an XY
// offset (Compose => 2 disjoint kinematic trees => CookArticulations => 2
// topologies => the multi-artic cook). go2_stand cooks as a FIXED-base 12-DOF
// articulation, so under gravity the (revolute) legs swing while the base stays
// pinned -- ample independent dynamics to test co-residence + non-aliasing.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "import/usd_importer.hpp"
#include "math/transform.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nphi = nuka::phi;
namespace nk = nuka::nk;

std::filesystem::path Go2ScenePath() {
    return std::filesystem::path(NUKA_SOURCE_DIR) / "examples" / "scenes" /
           "go2_stand.usda";
}

// The FLOATING-BASE Go2 variant (FloatingBase root => the trunk can TRANSLATE,
// so two co-resident dogs whose trunk boxes overlap can PHYSICALLY push apart).
std::filesystem::path Go2FloatScenePath() {
    return std::filesystem::path(NUKA_SOURCE_DIR) / "examples" / "scenes" /
           "go2_float.usda";
}

// Cook K go2 (composed at distinct XY) into ONE env. Returns the cooked Model.
nk::Model CookKDogs(uint32_t k, float spacing) {
    nuka::scene::SceneIR scene = nuka::import::LoadUsd(Go2ScenePath().string());
    for (uint32_t i = 1; i < k; ++i) {
        nuka::math::Transform place = nuka::math::Transform::Identity();
        place.position.x = spacing * static_cast<float>(i);
        scene = nuka::scene::Compose(scene, nuka::import::LoadUsd(Go2ScenePath().string()),
                                     place, "dog" + std::to_string(i) + "_");
    }
    return nuka::scene::cook::CookToModel(scene, 1).model;
}

// Cook K FLOATING-BASE go2 placed `spacing` apart in X into ONE env.
nk::Model CookKFloatDogs(uint32_t k, float spacing) {
    nuka::scene::SceneIR scene = nuka::import::LoadUsd(Go2FloatScenePath().string());
    for (uint32_t i = 1; i < k; ++i) {
        nuka::math::Transform place = nuka::math::Transform::Identity();
        place.position.x = spacing * static_cast<float>(i);
        scene = nuka::scene::Compose(scene, nuka::import::LoadUsd(Go2FloatScenePath().string()),
                                     place, "dog" + std::to_string(i) + "_");
    }
    return nuka::scene::cook::CookToModel(scene, 1).model;
}

bool AllFinite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

}  // namespace

// --- The cook foundation (host-only): K disjoint dogs => K articulations ------
TEST(MultiDogCoStep, CookEmitsKSeparateArticulationsPerDogDof) {
    if (!std::filesystem::exists(Go2ScenePath())) {
        GTEST_SKIP() << "go2_stand.usda not present";
    }
    const nk::Model one = CookKDogs(1u, 1.5f);
    const nk::Model two = CookKDogs(2u, 1.5f);
    const nk::Model three = CookKDogs(3u, 1.5f);

    // ONE dog == the legacy single-robot shape (K==1 D1 invariant).
    EXPECT_EQ(one.capacities.articulations_per_env, 1u);
    EXPECT_EQ(one.articulation.articulation_count, 1u);

    // K dogs => K separate articulations, dofs_per_env STAYS the per-DOG DOF
    // (NOT summed -- the per-artic 64-DOF cap is safe), links_per_env == sum.
    EXPECT_EQ(two.capacities.articulations_per_env, 2u);
    EXPECT_EQ(two.articulation.articulation_count, 2u);
    EXPECT_EQ(two.capacities.dofs_per_env, one.capacities.dofs_per_env)
        << "dofs_per_env must be the per-DOG DOF, never summed across dogs";
    EXPECT_EQ(two.capacities.links_per_env, one.capacities.links_per_env * 2u);

    EXPECT_EQ(three.capacities.articulations_per_env, 3u);
    EXPECT_EQ(three.capacities.dofs_per_env, one.capacities.dofs_per_env);
    EXPECT_EQ(three.capacities.links_per_env, one.capacities.links_per_env * 3u);

    // Per-articulation link offsets are the running prefix sum (the second dog's
    // links start AFTER the first dog's), and base_poses carry the distinct
    // spawn placements from Compose.
    ASSERT_EQ(two.articulation.articulation_link_offset.size(), 2u);
    EXPECT_EQ(two.articulation.articulation_link_offset[0], 0u);
    EXPECT_EQ(two.articulation.articulation_link_offset[1],
              one.capacities.links_per_env);
    ASSERT_EQ(two.articulation.base_poses.size(), 2u);
    EXPECT_NE(two.articulation.base_poses[0].position.x,
              two.articulation.base_poses[1].position.x)
        << "the two dogs must have DISTINCT root poses";
}

// --- The co-step spike (GPU): 2 dogs co-step finite + INDEPENDENT -------------
TEST(MultiDogCoStep, TwoDogsCoStepFiniteAndIndependent) {
    if (!std::filesystem::exists(Go2ScenePath())) {
        GTEST_SKIP() << "go2_stand.usda not present";
    }
    nphi::Device* dev = nphi::InitBestDevice();
    if (dev == nullptr) {
        GTEST_SKIP() << "no compute device";
    }
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    nk::Model model = CookKDogs(2u, 1.5f);
    ASSERT_EQ(model.capacities.articulations_per_env, 2u);
    const uint32_t L = model.capacities.links_per_env;        // 2 * single-dog links
    const uint32_t Lsingle = L / 2u;                          // links of one dog
    ASSERT_GT(Lsingle, 0u);

    // WP1 SCOPE: drop the foot-ground contact set. The FUSED foot pipeline is
    // env-keyed and hard-capped at kMaxFootContactsPerEnv (4); K dogs (8 feet)
    // exceed it. Multi-dog FOOT/leg contact (the K*4 slot growth + env->articulation
    // slot re-key + capsule-capsule narrowphase + two-articulation reaction) is the
    // LATER contact crux (WP5-8), NOT this forward-dynamics co-residence foundation.
    // With feet cleared (foot_count == 0) the detection kernel zero-fills and the
    // dogs co-step under gravity + drive with NO contact -- exactly what this spike
    // validates (K articulations co-step in ONE general world, independent tiles).
    model.feet.clear();

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[2] = -9.81f;

    nk::World world(std::move(model), 1u, dev, backend, cfg);
    ASSERT_TRUE(world.Ready());

    // Drive ONLY dog A (links [0, Lsingle)) toward a non-rest joint target;
    // leave dog B (links [Lsingle, L)) at its cooked rest target. If the two
    // articulations aliased into a shared qdot/M tile, driving A would perturb B.
    std::vector<float> drive_target(L, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::DriveTarget,
                                              drive_target.data(),
                                              L * sizeof(float)));
    const std::vector<float> rest_target = drive_target;  // cooked hold targets.
    for (uint32_t l = 0; l < Lsingle; ++l) {
        drive_target[l] = rest_target[l] + 0.30f;  // push dog A's joints.
    }
    ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::DriveTarget,
                                            drive_target.data(),
                                            L * sizeof(float)));

    // Co-step ~200 steps under gravity + the asymmetric drive.
    for (uint32_t s = 0; s < 200u; ++s) {
        const nk::StepResult r = world.Step();
        ASSERT_TRUE(r.AllOk()) << "step " << s << " had a non-Ok op";
    }

    std::vector<float> q(L, 0.0f), qd(L, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Q, q.data(),
                                              L * sizeof(float)));
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Qdot, qd.data(),
                                              L * sizeof(float)));
    std::vector<float> base_pose(2u * 7u, 0.0f);  // K==2 root transforms (xyzw+xyz).
    ASSERT_TRUE(world.GetData().DownloadField(
        nk::FieldId::BasePose, base_pose.data(), base_pose.size() * sizeof(float)));

    // (a) FINITE: both dogs' full state is finite (no NaN/Inf from co-step).
    EXPECT_TRUE(AllFinite(q)) << "joint positions must stay finite";
    EXPECT_TRUE(AllFinite(qd)) << "joint velocities must stay finite";
    EXPECT_TRUE(AllFinite(base_pose)) << "both root poses must stay finite";

    // Split per-dog joint slices.
    std::vector<float> qa(q.begin(), q.begin() + Lsingle);
    std::vector<float> qb(q.begin() + Lsingle, q.end());

    // (b) DOG A MOVED: the driven dog's joints left their cooked rest (the drive
    // and gravity actually advanced its state -- the world really stepped).
    float dog_a_moved = 0.0f;
    for (uint32_t l = 0; l < Lsingle; ++l) {
        dog_a_moved = std::max(dog_a_moved, std::abs(qa[l] - rest_target[l]));
    }
    EXPECT_GT(dog_a_moved, 1.0e-3f) << "the driven dog A must have moved";

    // (c) INDEPENDENCE: dog B (undriven) must NOT have been dragged by dog A's
    // drive. With aliased tiles, A's large drive impulse would bleed into B. B
    // only sees gravity on its own (separately-pinned) revolute legs, so its
    // deviation must be FAR smaller than A's driven deviation -- and crucially
    // its joints must not track A's joints.
    float dog_b_moved = 0.0f;
    float ab_coupling = 0.0f;
    for (uint32_t l = 0; l < Lsingle; ++l) {
        dog_b_moved = std::max(dog_b_moved, std::abs(qb[l] - rest_target[l]));
        // if B aliased A, qb[l] would chase qa[l] (same driven target).
        ab_coupling = std::max(ab_coupling, std::abs(qb[l] - qa[l]));
    }
    EXPECT_LT(dog_b_moved, dog_a_moved)
        << "undriven dog B must move LESS than driven dog A (no tile aliasing)";
    EXPECT_GT(ab_coupling, 1.0e-2f)
        << "dog B must NOT track dog A's driven joints (independent tiles)";
}

// ===========================================================================
// WP5/WP6/WP8 dog-dog PHYSICAL COLLISION spike (the owner bottom-line). TWO
// FLOATING-BASE Go2 dropped into each other (trunk boxes overlapping) must
// PHYSICALLY PUSH APART: trunk separation grows, BOTH dogs' qdot changes
// (two-way momentum exchange), and a far-apart control run shows NO push.
// Free-fall (feet cleared) isolates the dog-dog reaction from foot-ground.
// ===========================================================================

namespace {

// Step K floating-base dogs (feet cleared) for `steps` and return their final
// per-dog root X positions (from BasePose) + the L2 norm of the joint qdot
// (the kinetic signature). `spacing` sets the X gap between the two dogs.
struct DogRunResult {
    std::vector<float> root_x;   // K root X positions (BasePose).
    float qdot_norm = 0.0f;      // ||qdot|| over all dogs.
    bool ok = true;
};

DogRunResult RunFloatDogs(nphi::Device* dev, nphi::Backend* backend, uint32_t k,
                          float spacing, uint32_t steps) {
    DogRunResult out;
    nk::Model model = CookKFloatDogs(k, spacing);
    const uint32_t L = model.capacities.links_per_env;
    // Free-fall: drop foot-ground contact (the env-keyed fused foot cap is a
    // named WP1 debt; this isolates the dog-DOG reaction, the WP5-8 deliverable).
    model.feet.clear();

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[2] = 0.0f;  // no gravity: isolate the dog-dog contact impulse
                            // (overlapping trunks must push apart on their own).
    cfg.contact_margin = 0.0f;

    nk::World world(std::move(model), 1u, dev, backend, cfg);
    if (!world.Ready()) { out.ok = false; return out; }

    for (uint32_t s = 0; s < steps; ++s) {
        const nk::StepResult r = world.Step();
        if (!r.AllOk()) { out.ok = false; return out; }
    }

    std::vector<float> base_pose(k * 7u, 0.0f);  // K root transforms (pos+quat).
    std::vector<float> qd(L, 0.0f);
    out.ok = world.GetData().DownloadField(nk::FieldId::BasePose, base_pose.data(),
                                           base_pose.size() * sizeof(float)) &&
             world.GetData().DownloadField(nk::FieldId::Qdot, qd.data(),
                                           L * sizeof(float));
    out.root_x.resize(k);
    for (uint32_t a = 0; a < k; ++a) out.root_x[a] = base_pose[a * 7u + 0u];
    float n2 = 0.0f;
    for (float v : qd) n2 += v * v;
    out.qdot_norm = std::sqrt(n2);
    return out;
}

}  // namespace

// --- The cook records per-link collision geometry (WP5/WP6 source) ------------
TEST(MultiDogContact, CookEmitsPerLinkCollisionGeometry) {
    if (!std::filesystem::exists(Go2FloatScenePath())) {
        GTEST_SKIP() << "go2_float.usda not present";
    }
    const nk::Model two = CookKFloatDogs(2u, 0.15f);
    ASSERT_EQ(two.capacities.articulations_per_env, 2u);

    // The contact slot budget grew to K * kMaxFootContactsPerEnv so the fused
    // per-articulation solver's slot_base = artic*4 stays in bounds for K dogs.
    EXPECT_EQ(two.capacities.max_contacts_per_env, 8u);   // 2 * 4
    EXPECT_EQ(two.capacities.max_rows_per_env, 24u);      // 2 * 12

    // K==1 collapses to the legacy single-robot budget (the D1 invariant).
    const nk::Model one = CookKFloatDogs(1u, 0.15f);
    EXPECT_EQ(one.capacities.max_contacts_per_env, 4u);
    EXPECT_EQ(one.capacities.max_rows_per_env, 12u);

    // Per-link collision geometry is populated: each dog's trunk cooks a Box
    // (link_geom_kind == ShapeType::Box + 1 == 3) so dog-dog narrowphase has a
    // collidable to pose from link_pose. At least one link per dog is collidable.
    ASSERT_FALSE(two.articulation.link_geom_kind.empty());
    uint32_t collidable_links = 0u;
    for (uint32_t kind : two.articulation.link_geom_kind) {
        if (kind != 0u) ++collidable_links;
    }
    EXPECT_GE(collidable_links, 2u)
        << "each dog must cook >=1 collidable link (trunk box / foot sphere)";
}

// --- The two-way physical reaction spike (GPU) --------------------------------
TEST(MultiDogContact, TwoDogsPushApartAndExchangeMomentum) {
    if (!std::filesystem::exists(Go2FloatScenePath())) {
        GTEST_SKIP() << "go2_float.usda not present";
    }
    nphi::Device* dev = nphi::InitBestDevice();
    if (dev == nullptr) {
        GTEST_SKIP() << "no compute device";
    }
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    // (A) OVERLAPPING: two dogs 0.15 m apart in X. Trunk boxes are 0.20 m cubes
    // (half-extent 0.10), so the centers 0.15 m apart => 0.05 m interpenetration.
    // With contact ON the dogs MUST push apart along X.
    const uint32_t kSteps = 30u;
    DogRunResult overlap = RunFloatDogs(dev, backend, 2u, 0.15f, kSteps);
    ASSERT_TRUE(overlap.ok) << "overlapping co-step must run clean";
    ASSERT_EQ(overlap.root_x.size(), 2u);

    const float sep_overlap = std::abs(overlap.root_x[1] - overlap.root_x[0]);
    const float radii_sum = 0.10f + 0.10f;  // the two trunk box half-extents.

    std::fprintf(stderr,
                 "[dog-dog spike] OVERLAP: root_x={%.4f, %.4f} sep=%.4f "
                 "(spawn 0.15, radii_sum %.2f)  qdot_norm=%.5f\n",
                 overlap.root_x[0], overlap.root_x[1], sep_overlap, radii_sum,
                 overlap.qdot_norm);

    // (1) PUSH-APART: after the contact engages the trunk separation must EXCEED
    // the initial 0.15 m gap (they were forced apart) and approach the
    // non-interpenetrating separation (Sum of half-extents, minus a small slop).
    EXPECT_GT(sep_overlap, 0.15f + 1.0e-3f)
        << "overlapping dogs MUST push apart (separation grew past the 0.15 spawn)";
    EXPECT_GE(sep_overlap, radii_sum - 0.03f)
        << "the dogs must separate to ~non-interpenetrating distance (Sum radii - slop)";

    // (2a) TWO-WAY (the WP8 crux): BOTH dogs moved from their spawn (dog A at
    // x=0 went negative, dog B at x=0.15 went positive) -- the contact reaction
    // scattered into BOTH per-articulation qdot tiles, not just one. This is the
    // exact thing single-tile solvers cannot do; here each dog's fused solver
    // block resolves its OWN slot block into its OWN tile.
    EXPECT_LT(overlap.root_x[0], 0.0f - 1.0e-3f)
        << "dog A (spawn x=0) must be pushed in -X by the contact";
    EXPECT_GT(overlap.root_x[1], 0.15f + 1.0e-3f)
        << "dog B (spawn x=0.15) must be pushed in +X by the contact";

    // (2b) MOMENTUM EXCHANGE: qdot is nonzero (the contact reaction injected
    // velocity). With gravity off and feet cleared, the ONLY force is the dog-dog
    // contact, so a nonzero qdot_norm IS the two-way reaction itself.
    EXPECT_GT(overlap.qdot_norm, 1.0e-3f)
        << "the dog-dog contact must inject velocity into BOTH dogs' qdot tiles";

    // (B) FAR-APART CONTROL: two dogs 1.5 m apart never overlap => NO contact.
    // Separation stays at the spawn gap and qdot stays ~0 (no spurious push).
    DogRunResult apart = RunFloatDogs(dev, backend, 2u, 1.5f, kSteps);
    ASSERT_TRUE(apart.ok) << "far-apart co-step must run clean";
    const float sep_apart = std::abs(apart.root_x[1] - apart.root_x[0]);
    std::fprintf(stderr,
                 "[dog-dog spike] CONTROL: root_x={%.4f, %.4f} sep=%.4f "
                 "(spawn 1.5)  qdot_norm=%.7f\n",
                 apart.root_x[0], apart.root_x[1], sep_apart, apart.qdot_norm);
    EXPECT_NEAR(sep_apart, 1.5f, 1.0e-2f)
        << "far-apart dogs must NOT move (no spurious dog-dog contact)";
    EXPECT_LT(apart.qdot_norm, 1.0e-4f)
        << "far-apart dogs must stay at rest (the dog-dog op is inert with no overlap)";

    // The push-apart separation must DOMINATE the control's drift: the contact
    // produced a real, large displacement the no-contact case never shows.
    EXPECT_GT(sep_overlap, 0.15f + 0.01f);
}
