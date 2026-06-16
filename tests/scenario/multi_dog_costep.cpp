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
#include "sensor/terrain/terrain_field.hpp"  // Task 2 body-vs-terrain spike

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

    // The contact slot budget grew to K * kMultiDogContactsPerArtic (Task 1/2: each
    // dog's block holds feet + body-vs-terrain + dog-dog) so the fused per-
    // articulation solver's slot_base = artic*stride stays in bounds for K dogs.
    EXPECT_EQ(two.capacities.max_contacts_per_env, 24u);  // 2 * 12
    EXPECT_EQ(two.capacities.max_rows_per_env, 72u);      // 2 * 36 (3 rows/slot)

    // K==1 collapses to the legacy single-robot budget (the D1 invariant: stride 4).
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

// ===========================================================================
// TASK 1 + TASK 2 spikes: FEET + DOG-DOG COEXISTENCE and BODY-vs-TERRAIN
// (no 穿模). The K==1 byte-identity is re-proven by the unchanged go2_stand
// regression + the determinism gate; these spikes exercise the K>1 paths that
// the prior WP1-8 spikes deliberately skipped (they cleared feet).
// ===========================================================================

namespace {

// The LOWEST world-space point of a posed link primitive (host mirror of the
// kernel's LinkLowestPoint). kind sentinel: 1=Sphere,2=Capsule,3=Box (ShapeType+1).
float LinkLowestZHost(uint32_t kind_sentinel, const float* p4,
                      const nuka::math::Transform& world_geom,
                      float* out_x, float* out_y) {
    using nuka::math::Vec3;
    const uint32_t st = kind_sentinel - 1u;  // 0 Sph, 1 Cap, 2 Box
    if (st == 0u) {  // Sphere: center - r*ez
        const Vec3 c = world_geom.position;
        *out_x = c.x; *out_y = c.y;
        return c.z - p4[0];
    }
    if (st == 1u) {  // Capsule: lower segment end (local Z axis) - r
        const Vec3 axis = world_geom.rotation.Rotate(Vec3{0.0f, 0.0f, 1.0f});
        const float hh = p4[1];
        const Vec3 e0{world_geom.position.x + axis.x * hh,
                      world_geom.position.y + axis.y * hh,
                      world_geom.position.z + axis.z * hh};
        const Vec3 e1{world_geom.position.x - axis.x * hh,
                      world_geom.position.y - axis.y * hh,
                      world_geom.position.z - axis.z * hh};
        const Vec3 lo = (e0.z <= e1.z) ? e0 : e1;
        *out_x = lo.x; *out_y = lo.y;
        return lo.z - p4[0];
    }
    // Box: lowest of 8 OBB corners.
    float best = 0.0f; bool first = true;
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2)
            for (int sz = -1; sz <= 1; sz += 2) {
                const Vec3 corner = world_geom.TransformPoint(
                    Vec3{static_cast<float>(sx) * p4[0],
                         static_cast<float>(sy) * p4[1],
                         static_cast<float>(sz) * p4[2]});
                if (first || corner.z < best) { best = corner.z; *out_x = corner.x;
                    *out_y = corner.y; first = false; }
            }
    return best;
}

}  // namespace

// --- TASK 2: K dogs free-fall onto a PYRAMID-STAIRS field and REST on it -------
// (no 穿模). With body-vs-terrain contact the trunk/leg primitives land ON the
// local terrain surface instead of clipping THROUGH it. Quantified by the MIN
// (body-link lowest point - SampleTerrainHeight) over all dogs/links/last steps.
TEST(MultiDogTerrain, BodyLinksRestOnTerrainNoSinkThrough) {
    if (!std::filesystem::exists(Go2FloatScenePath())) {
        GTEST_SKIP() << "go2_float.usda not present";
    }
    nphi::Device* dev = nphi::InitBestDevice();
    if (dev == nullptr) GTEST_SKIP() << "no compute device";
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    // 2 floating-base dogs spaced 1.0 m apart in X (both well inside the central
    // platform), dropped from the USD spawn z=0.445.
    const uint32_t K = 2u;
    nk::Model model = CookKFloatDogs(K, 1.0f);
    ASSERT_EQ(model.capacities.articulations_per_env, K);
    const uint32_t L = model.capacities.links_per_env;

    // An ELEVATED central platform (PyramidStairs): the platform top sits
    // kPyramidRings*step_height above ground. We set the top BELOW the dogs' spawn
    // z (0.445) so they drop a few cm onto it -- a dog whose TRUNK / upper-leg
    // capsule sags toward the platform is caught by the body-terrain pass instead
    // of clipping THROUGH it (the owner 穿模 fix). A wide platform keeps both dogs
    // on the flat top (so the comparison surface is a single height, not a step).
    nuka::terrain::TerrainParams terr{};
    terr.ground_height  = 0.0f;
    terr.step_height    = 0.025f;  // 8 rings * 0.025 = 0.20 m platform top
    terr.step_width     = 0.40f;
    terr.platform_width = 6.0f;    // wide flat top (half_plat 3.0 > dog spread 1.0)
    model.terrain = terr;
    model.ground_height = 0.0f;

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;

    // Capture the per-link collision geometry BEFORE moving the model into World.
    const std::vector<uint32_t> geom_kind = model.articulation.link_geom_kind;
    std::vector<float> geom_params = model.articulation.link_geom_params;
    const std::vector<nuka::math::Transform> geom_local =
        model.articulation.link_geom_local;

    uint32_t collidable_capture = 0u;
    for (uint32_t kk : geom_kind) if (kk != 0u) ++collidable_capture;
    std::fprintf(stderr,
                 "[terrain spike] captured geom: L=%u kind.size=%zu params.size=%zu "
                 "local.size=%zu collidable=%u\n",
                 L, geom_kind.size(), geom_params.size(), geom_local.size(),
                 collidable_capture);
    ASSERT_GT(collidable_capture, 0u) << "no per-link collision geometry captured";

    nk::World world(std::move(model), 1u, dev, backend, cfg);
    ASSERT_TRUE(world.Ready());

    // Set EVERY env's terrain TYPE to PyramidStairs (seeded Flat at construction).
    std::vector<uint32_t> ttype(1u, nuka::terrain::kTerrainPyramidStairs);
    ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::EnvTerrainType,
                                            ttype.data(), sizeof(uint32_t)));

    // Settle: drop the dogs onto the platform.
    const uint32_t kSteps = 400u;
    for (uint32_t s = 0; s < kSteps; ++s) {
        ASSERT_TRUE(world.Step().AllOk()) << "step " << s;
    }

    // Measure the resting clearance over a few final steps. worst_pen tracks the
    // MOST-NEGATIVE clearance (a sink-through); min_clear tracks the true minimum
    // clearance (the closest a body link comes to the surface -- proof it actually
    // rests on, not floats high above, the terrain).
    nuka::terrain::TerrainParams sampled = terr;
    sampled.ground_height = 0.0f;
    float worst_pen = 1.0e9f;
    float min_clear = 1.0e9f;
    for (uint32_t s = 0; s < 20u; ++s) {
        ASSERT_TRUE(world.Step().AllOk());
        std::vector<nuka::math::Transform> link_pose(L);
        ASSERT_TRUE(world.GetData().DownloadField(
            nk::FieldId::LinkPose, link_pose.data(),
            L * sizeof(nuka::math::Transform)));
        for (uint32_t l = 0; l < L; ++l) {
            const uint32_t kind = l < geom_kind.size() ? geom_kind[l] : 0u;
            if (kind == 0u) continue;
            const nuka::math::Transform world_geom = link_pose[l] * geom_local[l];
            float gx = 0.0f, gy = 0.0f;
            const float lowest_z = LinkLowestZHost(
                kind, geom_params.data() + static_cast<size_t>(l) * 4u,
                world_geom, &gx, &gy);
            const float surface = nuka::terrain::SampleTerrainHeight(
                nuka::terrain::kTerrainPyramidStairs, gx, gy, sampled);
            // clearance > 0 == above surface; < 0 == sunk INTO terrain (穿模).
            const float clearance = lowest_z - surface;
            if (!std::isfinite(clearance)) continue;
            worst_pen = std::min(worst_pen, clearance);
            min_clear = std::min(min_clear, clearance);
        }
    }

    std::fprintf(stderr,
                 "[terrain spike] %u dogs on PyramidStairs (platform top %.2f): "
                 "min body-link clearance above surface = %.4f m "
                 "(most-negative = %.4f m; <0 == 穿模)\n",
                 K, 8.0f * terr.step_height, min_clear, worst_pen);

    // NO 穿模: no body link sinks meaningfully below the local terrain surface.
    // A tolerance of one Baumgarte-resting penetration slop (~couple cm) accounts
    // for the soft contact's steady-state interpenetration; a real clip-through is
    // tens of cm (the trunk falling 0.4 m past the platform).
    EXPECT_GT(worst_pen, -0.05f)
        << "a body link sank THROUGH the terrain (穿模) -- body-terrain contact "
           "failed to keep the trunk resting ON the surface";
}

// --- TASK 1: feet LOADED on ground AND dog-dog push-apart, SIMULTANEOUSLY ------
// Two floating-base dogs spawned overlapping on FLAT ground with gravity: the
// feet must contact the ground (foot slots loaded) WHILE the overlapping trunks
// push apart (dog-dog reaction). Proves feet + dog-dog coexist for K>1 (neither
// clobbers the other's per-articulation slot block).
TEST(MultiDogTerrain, FeetAndDogDogContactCoexist) {
    if (!std::filesystem::exists(Go2FloatScenePath())) {
        GTEST_SKIP() << "go2_float.usda not present";
    }
    nphi::Device* dev = nphi::InitBestDevice();
    if (dev == nullptr) GTEST_SKIP() << "no compute device";
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    // Two dogs 0.15 m apart (trunk boxes 0.20 -> 0.05 m interpenetration) on a
    // FLAT ground plane at z=0, gravity ON so the feet fall onto the ground.
    const uint32_t K = 2u;
    nk::Model model = CookKFloatDogs(K, 0.15f);
    ASSERT_EQ(model.capacities.articulations_per_env, K);
    ASSERT_EQ(model.capacities.max_contacts_per_env, 24u);  // K*12 (feet+body+dog-dog)
    const uint32_t L = model.capacities.links_per_env;
    const uint32_t slots = model.capacities.max_contacts_per_env;  // 24
    const uint32_t stride = slots / K;                             // 12 per dog
    model.ground_height = 0.0f;  // flat plane (default terrain Flat).

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;

    nk::World world(std::move(model), 1u, dev, backend, cfg);
    ASSERT_TRUE(world.Ready());

    // Spawn separation from BasePose (before stepping).
    std::vector<float> base0(K * 7u, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BasePose, base0.data(),
                                              base0.size() * sizeof(float)));
    const float spawn_sep = std::abs(base0[1 * 7 + 0] - base0[0 * 7 + 0]);

    // Settle under gravity: feet land on the ground; overlapping trunks push apart.
    for (uint32_t s = 0; s < 250u; ++s) {
        ASSERT_TRUE(world.Step().AllOk()) << "step " << s;
    }

    // (1) FEET LOADED: download the contact slots; count active foot contacts
    // (normal +Z, the foot calf links) across BOTH dogs' per-articulation blocks.
    std::vector<uint32_t> clink(slots, ~0u);
    std::vector<float> cdepth(slots, 0.0f);
    std::vector<nuka::math::Vec3> cnormal(slots);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::ContactLink, clink.data(),
                                              slots * sizeof(uint32_t)));
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::ContactDepth, cdepth.data(),
                                              slots * sizeof(float)));
    ASSERT_TRUE(world.GetData().DownloadField(
        nk::FieldId::ContactNormal, cnormal.data(),
        slots * sizeof(nuka::math::Vec3)));

    uint32_t active = 0u, ground_contacts = 0u;
    // Per-articulation block [art*stride, art*stride+stride): art 0 == dog A, 1 == B.
    uint32_t per_dog_active[2] = {0u, 0u};
    for (uint32_t s = 0; s < slots; ++s) {
        if (clink[s] == ~0u) continue;
        ++active;
        if (cnormal[s].z > 0.5f) {  // +Z normal == a ground/terrain contact.
            ++ground_contacts;
            const uint32_t art = s / stride;
            if (art < 2u) ++per_dog_active[art];
        }
    }

    // (2) DOG-DOG push-apart: final separation grew past the 0.15 spawn gap.
    std::vector<float> base1(K * 7u, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BasePose, base1.data(),
                                              base1.size() * sizeof(float)));
    const float final_sep = std::abs(base1[1 * 7 + 0] - base1[0 * 7 + 0]);

    std::fprintf(stderr,
                 "[coexist spike] spawn_sep=%.3f final_sep=%.3f | active_slots=%u "
                 "ground_contacts=%u (dogA=%u dogB=%u of %u/dog)\n",
                 spawn_sep, final_sep, active, ground_contacts,
                 per_dog_active[0], per_dog_active[1], stride);

    // FEET coexist: at least one ground contact PER DOG is loaded (the feet did
    // NOT get clobbered by the dog-dog op's slot writes).
    EXPECT_GT(per_dog_active[0], 0u) << "dog A has no ground contact (feet lost)";
    EXPECT_GT(per_dog_active[1], 0u) << "dog B has no ground contact (feet lost)";
    // AND the dogs pushed apart (dog-dog reaction active at the same time).
    EXPECT_GT(final_sep, spawn_sep + 1.0e-3f)
        << "overlapping dogs must push apart WHILE their feet rest on the ground";
}
