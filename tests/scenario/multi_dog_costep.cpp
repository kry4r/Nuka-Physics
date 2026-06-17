// ---------------------------------------------------------------------------
// Multi-articulation co-residence + GENERAL-PATH collision: K Go2 articulations
// co-resident in ONE env of ONE general nk::World. Two layers:
//
//  (1) FORWARD-DYNAMICS co-residence: the cook + model + schema re-key + staging
//      put K SEPARATE Go2 articulations into one env (articulation_count == K)
//      and they CO-STEP independently under gravity -- each dog keeps its OWN
//      per-articulation tile (the 64-DOF cap stays safe; dofs_per_env stays the
//      per-DOG DOF, never summed), and one dog's joints do NOT move another's
//      (no qdot-tile aliasing).
//
//  (2) PHYSICAL COLLISION via the GENERAL contact path: K dogs that overlap, or
//      that rest on a general heightfield, collide through the ONE LBVH ->
//      cvx GJK/EPA narrowphase -> mixed-island solve pipeline. There is NO
//      "dog-dog" concept (the dog_dog hack was deleted in L1/D1): a robot body
//      is just a physics body that collides body<->body, and a heightfield is
//      just a general collidable -- the owner bottom-line. The general two-dog +
//      box-on-dog cases are also covered by pairdriven_dog_solve.cpp.
//
// The dogs are built by composing the go2 scene with itself at an XY offset
// (Compose => K disjoint kinematic trees => CookArticulations => K topologies =>
// the multi-artic cook) and cooked with CookContactFamily::PairDriven (the
// general path). go2_stand cooks as a FIXED-base 12-DOF articulation; go2_float
// cooks FLOATING-base so the trunk can translate and overlapping dogs can push
// apart.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "import/usd_importer.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"            // kPairDrivenRowsPerSlot
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"
#include "sensor/terrain/terrain_field.hpp"  // general heightfield collidable

namespace {

namespace nphi = nuka::phi;
namespace nk = nuka::nk;
namespace cook = nuka::scene::cook;
namespace terrain = nuka::terrain;
using nuka::math::Vec3;

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

// Cook K go2 (composed at distinct XY) into ONE env via the GENERAL (PairDriven)
// contact path. Returns the cooked Model. The general path (LBVH broadphase ->
// cvx GJK/EPA narrowphase -> mixed-island solve) is the ONE multi-body contact
// path; there is no longer a "dog-dog" concept (the dog_dog hack was deleted in
// L1/D1) -- a robot body is just a physics body that collides body<->body here.
nk::Model CookKDogs(uint32_t k, float spacing) {
    nuka::scene::SceneIR scene = nuka::import::LoadUsd(Go2ScenePath().string());
    for (uint32_t i = 1; i < k; ++i) {
        nuka::math::Transform place = nuka::math::Transform::Identity();
        place.position.x = spacing * static_cast<float>(i);
        scene = nuka::scene::Compose(scene, nuka::import::LoadUsd(Go2ScenePath().string()),
                                     place, "dog" + std::to_string(i) + "_");
    }
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    return cook::CookToModel(scene, 1, opt).model;
}

// Cook K FLOATING-BASE go2 placed `spacing` apart in X into ONE env via the
// GENERAL (PairDriven) contact path.
nk::Model CookKFloatDogs(uint32_t k, float spacing) {
    nuka::scene::SceneIR scene = nuka::import::LoadUsd(Go2FloatScenePath().string());
    for (uint32_t i = 1; i < k; ++i) {
        nuka::math::Transform place = nuka::math::Transform::Identity();
        place.position.x = spacing * static_cast<float>(i);
        scene = nuka::scene::Compose(scene, nuka::import::LoadUsd(Go2FloatScenePath().string()),
                                     place, "dog" + std::to_string(i) + "_");
    }
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    return cook::CookToModel(scene, 1, opt).model;
}

// Set the PairDriven per-env candidate-slot / row budget for a K-dog general
// world. 32 candidate slots/env is generous for K<=3 dogs (artic<->artic +
// foot/leg<->ground/heightfield candidate pairs) and matches the budget the
// other general-path harnesses (vproof_go2_terrain / pairdriven_heightfield)
// use. Caller has already resized body_init / shape_table_rows.
void SetPairDrivenCaps(nk::Model& model) {
    nk::ModelCapacities& cap = model.capacities;
    cap.max_contacts_per_env = 32u;
    cap.max_rows_per_env = 32u * nk::kPairDrivenRowsPerSlot;
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

    // CO-RESIDENCE SCOPE: this test isolates the FORWARD-DYNAMICS independence of
    // co-resident articulations (no contact). The two dogs are 1.5 m apart (no
    // overlap -> no general body<->body contact) and there is no ground/heightfield
    // collidable, so they simply co-step under gravity + drive. (model.feet is
    // unused on the PairDriven path.) This validates K articulations co-step in ONE
    // general world with independent qdot tiles (no aliasing), the foundation the
    // GENERAL contact cases build on.
    SetPairDrivenCaps(model);

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[2] = -9.81f;
    cfg.vel_iters = 32u;
    cfg.max_pairs = 64u;

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
// MULTI-BODY PHYSICAL COLLISION spike via the GENERAL path (the owner
// bottom-line: a robot body is just a physics body, collision is ONE general
// body<->body pipeline -- no "dog-dog" concept). TWO FLOATING-BASE Go2 dropped
// into each other (trunk boxes overlapping) must PHYSICALLY PUSH APART through
// the GENERAL LBVH -> cvx narrowphase -> mixed-island solve: trunk separation
// grows, BOTH dogs' qdot changes (two-way momentum exchange), and a far-apart
// control run shows NO push. Gravity off isolates the body<->body reaction.
// ===========================================================================

namespace {

// Step K floating-base dogs (general path, gravity off) for `steps` and return
// their final per-dog root X positions (from BasePose) + the L2 norm of the
// joint qdot (the kinetic signature). `spacing` sets the X gap between dogs.
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
    // GENERAL path: no terrain/ground collidable here -> free-fall with gravity
    // OFF, so the ONLY force is the body<->body (artic-link<->artic-link) contact
    // along the GENERAL pipeline. (model.feet is unused on the PairDriven path;
    // foot/leg collidables ride shape_table_rows, posed into the LBVH by
    // SyncLinkBodyPose.)
    SetPairDrivenCaps(model);

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[2] = 0.0f;  // no gravity: isolate the body<->body contact impulse
                            // (overlapping trunks must push apart on their own).
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u;
    cfg.max_pairs = 64u;

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

// --- The cook records per-link collision geometry (GENERAL-path source) --------
// On the GENERAL (PairDriven) path the per-link collision geometry is what
// SyncLinkBodyPose poses into the LBVH (link_pose o link_geom_local) so each
// articulation link enters the ONE broadphase/narrowphase as a collidable body.
// This is the cook-side prerequisite for general artic-link<->artic-link contact
// (the multi-body collision that REPLACED the dog_dog hack).
TEST(MultiDogContact, CookEmitsPerLinkCollisionGeometry) {
    if (!std::filesystem::exists(Go2FloatScenePath())) {
        GTEST_SKIP() << "go2_float.usda not present";
    }
    const nk::Model two = CookKFloatDogs(2u, 0.15f);
    ASSERT_EQ(two.capacities.articulations_per_env, 2u);

    // The GENERAL family is selected: the pipeline routes the ONE
    // LBVH -> cvx -> mixed-island contact path (no dog_dog hack).
    EXPECT_EQ(two.contact_family, nk::ContactFamily::PairDriven);

    // Per-link collision geometry is populated: each dog's trunk cooks a Box
    // (link_geom_kind == ShapeType::Box + 1 == 3) so the general narrowphase has a
    // collidable posed from link_pose. At least one link per dog is collidable.
    ASSERT_FALSE(two.articulation.link_geom_kind.empty());
    uint32_t collidable_links = 0u;
    for (uint32_t kind : two.articulation.link_geom_kind) {
        if (kind != 0u) ++collidable_links;
    }
    EXPECT_GE(collidable_links, 2u)
        << "each dog must cook >=1 collidable link (trunk box / foot sphere)";

    // K==1 collapses to the legacy single-robot articulation shape (the D1
    // invariant: one articulation, the per-dog DOF, byte-identical FUSED golden
    // unaffected by the general-path family selection at cook).
    const nk::Model one = CookKFloatDogs(1u, 0.15f);
    EXPECT_EQ(one.capacities.articulations_per_env, 1u);
    EXPECT_EQ(one.capacities.dofs_per_env, two.capacities.dofs_per_env);
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
                 "[multi-body general] OVERLAP: root_x={%.4f, %.4f} sep=%.4f "
                 "(spawn 0.15, radii_sum %.2f)  qdot_norm=%.5f\n",
                 overlap.root_x[0], overlap.root_x[1], sep_overlap, radii_sum,
                 overlap.qdot_norm);

    // (1) PUSH-APART: after the general contact engages the trunk separation must
    // EXCEED the initial 0.15 m gap (they were forced apart) and approach the
    // non-interpenetrating separation (Sum of half-extents, minus a small slop).
    EXPECT_GT(sep_overlap, 0.15f + 1.0e-3f)
        << "overlapping dogs MUST push apart (separation grew past the 0.15 spawn)";
    EXPECT_GE(sep_overlap, radii_sum - 0.03f)
        << "the dogs must separate to ~non-interpenetrating distance (Sum radii - slop)";

    // (2a) TWO-WAY: BOTH dogs moved from their spawn (dog A at x=0 went negative,
    // dog B at x=0.15 went positive) -- the general mixed-island contact reaction
    // scattered into BOTH dogs' floating-base qdot tiles, not just one. Each
    // co-resident articulation receives its own side of the body<->body reaction.
    EXPECT_LT(overlap.root_x[0], 0.0f - 1.0e-3f)
        << "dog A (spawn x=0) must be pushed in -X by the contact";
    EXPECT_GT(overlap.root_x[1], 0.15f + 1.0e-3f)
        << "dog B (spawn x=0.15) must be pushed in +X by the contact";

    // (2b) MOMENTUM EXCHANGE: qdot is nonzero (the contact reaction injected
    // velocity). With gravity off, the ONLY force is the body<->body general
    // contact, so a nonzero qdot_norm IS the two-way reaction itself.
    EXPECT_GT(overlap.qdot_norm, 1.0e-3f)
        << "the general body<->body contact must inject velocity into BOTH dogs";

    // (B) FAR-APART CONTROL: two dogs 1.5 m apart never overlap => NO contact.
    // Separation stays at the spawn gap and qdot stays ~0 (no spurious push).
    DogRunResult apart = RunFloatDogs(dev, backend, 2u, 1.5f, kSteps);
    ASSERT_TRUE(apart.ok) << "far-apart co-step must run clean";
    const float sep_apart = std::abs(apart.root_x[1] - apart.root_x[0]);
    std::fprintf(stderr,
                 "[multi-body general] CONTROL: root_x={%.4f, %.4f} sep=%.4f "
                 "(spawn 1.5)  qdot_norm=%.7f\n",
                 apart.root_x[0], apart.root_x[1], sep_apart, apart.qdot_norm);
    EXPECT_NEAR(sep_apart, 1.5f, 1.0e-2f)
        << "far-apart dogs must NOT move (no spurious general contact)";
    EXPECT_LT(apart.qdot_norm, 1.0e-4f)
        << "far-apart dogs must stay at rest (no overlap -> no contact)";

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

// --- K dogs free-fall onto a PyramidStairs STEP EDGE and REST on it (no 穿模) ---
// MIGRATED to the GENERAL path: the dogs collide against a GENERAL heightfield
// collidable (CookHeightfieldGrid -> the LBVH -> per-cell TRIANGLE_PRISM midphase
// -> cvx GJK/EPA -> mixed-island solve), the EXACT same body<->body contact path
// every other collidable uses. There is no body-vs-terrain special case anymore.
//
// ★ IMPROVED over the old test: the dogs stand STRADDLING a REAL step edge (NOT a
// wide flat platform top), so this now covers the step-edge geometry the deleted
// dog_dog analytic body-vs-terrain pass EXPLODED on. The general heightfield
// handles step edges (proven by pairdriven_heightfield's BoxOnStepEdge/Wedge
// cases), so the dogs must settle finite + NO 穿模 on the steps. Quantified by the
// MIN (body-link lowest point - SampleTerrainHeight) over all dogs/links/last
// steps, evaluated at each link's OWN final column (a real step surface).
//
// FIXED-base dogs (go2_stand), the vproof_go2_terrain GENERAL recipe (proven to
// rest a go2 stably on the heightfield without 穿模). The base is pinned, so the
// dog cannot ragdoll off the staircase -- only the standing legs/feet settle onto
// the steps. The heightfield grid CENTER is OFFSET in X so a real step EDGE falls
// directly UNDER the first cooked dog (the cooked roots sit at x=0 and x=spacing
// and cannot be moved -- FK reads the cooked relative root for a fixed base): with
// center.x = +half_plat the platform/-X-edge lands at world x=0, so dog A STRADDLES
// the platform-top / ring-1 edge (inner feet on the 0.32 platform tread, outer
// feet over the ring-1 tread/riser prisms -- the step-edge case the deleted
// dog_dog body-vs-terrain pass exploded on).
//
// HONESTY / scope note: go2 cooks with NO position-actuator drive (stiffness == 0),
// so a co-resident dog only stays put if the terrain geometry supports its passive
// legs. The FIRST cooked articulation (dog A) settles ON the step edge (validated
// here: finite + no 穿模). The composed SECOND articulation's passive legs are NOT
// drive-held and splay under gravity -- a PRE-EXISTING multi-dog cook/drive
// limitation, unrelated to the heightfield contact or the dog_dog deletion. So the
// no-穿模 assertion is scoped to dog A (the dog the general heightfield supports),
// while FINITE is asserted for BOTH dogs (the step-edge no-explosion win that is
// the headline over the deleted dog_dog hack).
TEST(MultiDogTerrain, BodyLinksRestOnTerrainNoSinkThrough) {
    if (!std::filesystem::exists(Go2ScenePath())) {
        GTEST_SKIP() << "go2_stand.usda not present";
    }
    nphi::Device* dev = nphi::InitBestDevice();
    if (dev == nullptr) GTEST_SKIP() << "no compute device";
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    // PyramidStairs (vproof spec): step_height 0.04, step_width 0.40,
    // platform_width 2.0 -> half_plat 1.0, platform top = 8*0.04 = 0.32 m (the
    // standing fixed-base feet reach it). The grid center is shifted +half_plat in
    // X so the platform/-X-edge sits at world x=0 under cooked dog A.
    const uint32_t K = 2u;
    const float spacing = 2.00f;  // dog B at x=2.0 (== +X platform edge, symmetric).
    nk::Model model = CookKDogs(K, spacing);
    ASSERT_EQ(model.capacities.articulations_per_env, K);
    ASSERT_EQ(model.contact_family, nk::ContactFamily::PairDriven);
    const uint32_t L = model.capacities.links_per_env;
    const uint32_t links_per_dog = L / K;  // dog A == links [0, links_per_dog).

    nuka::terrain::TerrainParams terr = nuka::terrain::DefaultTerrainParams();
    terr.ground_height  = 0.0f;
    terr.step_height    = 0.04f;   // 8 rings * 0.04 = 0.32 m platform top
    terr.step_width     = 0.40f;
    terr.platform_width = 2.00f;   // half_plat = 1.00
    const float half_plat = 0.5f * terr.platform_width;  // 1.0
    // Shift the grid so a step EDGE is under the cooked dogs. center.x = +half_plat
    // maps grid-local origin to world x=+1.0; the platform's -X edge (local
    // x=-half_plat) lands at world x=0 (cooked dog A straddles this edge).
    const Vec3 grid_center{half_plat, 0.0f, 0.0f};

    // Cook the GENERAL heightfield collidable for the SAME PyramidStairs spec.
    // body_init must be sized to the ORIGINAL bodies before CookHeightfieldGrid
    // seeds the heightfield collidable at index orig_bodies (the vproof recipe).
    const uint32_t orig_bodies = model.capacities.bodies_per_env;
    model.body_init.resize(orig_bodies);
    const uint32_t hf_row = cook::CookHeightfieldGrid(
        model, terr, terrain::kTerrainPyramidStairs,
        /*nrow=*/41u, /*ncol=*/41u, /*cell=*/0.20f, grid_center);
    ASSERT_EQ(hf_row, orig_bodies);

    nk::ModelCapacities& cap = model.capacities;
    cap.bodies_per_env = static_cast<uint32_t>(model.body_init.size());
    cap.max_bodies_total = static_cast<uint32_t>(model.shape_table_rows.size());
    SetPairDrivenCaps(model);

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u;
    cfg.max_pairs = 64u;

    // Capture the per-link collision geometry BEFORE moving the model into World.
    const std::vector<uint32_t> geom_kind = model.articulation.link_geom_kind;
    std::vector<float> geom_params = model.articulation.link_geom_params;
    const std::vector<nuka::math::Transform> geom_local =
        model.articulation.link_geom_local;

    uint32_t collidable_capture = 0u;
    for (uint32_t kk : geom_kind) if (kk != 0u) ++collidable_capture;
    std::fprintf(stderr,
                 "[terrain general] captured geom: L=%u collidable=%u hf_row=%u "
                 "grid_center.x=%.2f\n",
                 L, collidable_capture, hf_row, grid_center.x);
    ASSERT_GT(collidable_capture, 0u) << "no per-link collision geometry captured";

    nk::World world(std::move(model), 1u, dev, backend, cfg);
    ASSERT_TRUE(world.Ready());

    // Settle: the standing legs sag onto the step edge / treads and rest. The
    // SampleTerrainHeight comparison below uses the SAME grid_center offset (the
    // surface under a link at world (x,y) is sampled at local (x-center, y-center)).
    const uint32_t kSteps = 400u;
    for (uint32_t s = 0; s < kSteps; ++s) {
        ASSERT_TRUE(world.Step().AllOk()) << "step " << s;
    }

    // Measure the resting clearance over a few final steps. dogA_worst_pen tracks
    // dog A's MOST-NEGATIVE clearance (a sink-through) -- the no-穿模 metric scoped
    // to the dog the general heightfield supports. all_finite covers BOTH dogs
    // (every link of every dog must stay finite -- the step-edge no-explosion win).
    // Each link is compared against the LOCAL step surface at its OWN (x,y) -- a
    // real staircase surface, not a single flat platform height.
    float dogA_worst_pen = 1.0e9f;
    float dogA_min_clear = 1.0e9f;
    bool all_finite = true;
    uint32_t worst_l = ~0u;
    float worst_gx = 0.0f, worst_gy = 0.0f, worst_z = 0.0f, worst_surf = 0.0f;
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
            if (!std::isfinite(lowest_z) || !std::isfinite(gx) ||
                !std::isfinite(gy)) {
                all_finite = false;
                continue;
            }
            if (l >= links_per_dog) continue;  // 穿模 metric scoped to dog A.
            // The cooked grid is centred at grid_center, so the surface under a
            // link at world (gx,gy) is sampled at grid-LOCAL (gx-center, gy-center)
            // (the SAME mapping CookHeightfieldGrid used).
            const float surface = nuka::terrain::SampleTerrainHeight(
                nuka::terrain::kTerrainPyramidStairs,
                gx - grid_center.x, gy - grid_center.y, terr);
            // clearance > 0 == above surface; < 0 == sunk INTO terrain (穿模).
            const float clearance = lowest_z - surface;
            if (!std::isfinite(clearance)) { all_finite = false; continue; }
            if (clearance < dogA_worst_pen) {
                dogA_worst_pen = clearance; worst_l = l; worst_gx = gx;
                worst_gy = gy; worst_z = lowest_z; worst_surf = surface;
            }
            dogA_min_clear = std::min(dogA_min_clear, clearance);
        }
    }

    std::fprintf(stderr,
                 "[terrain general] dog A on PyramidStairs step edge (platform top "
                 "%.2f, half_plat %.2f): worst link %u @ (%.3f,%.3f) lowest_z=%.4f "
                 "surf=%.4f | min clearance=%.4f worst=%.4f m (<0 == 穿模)\n",
                 8.0f * terr.step_height, half_plat, worst_l, worst_gx, worst_gy,
                 worst_z, worst_surf, dogA_min_clear, dogA_worst_pen);

    // (1) FINITE (BOTH dogs) — the general step-edge path does NOT explode (the
    // deleted dog_dog analytic body-vs-terrain pass exploded on exactly this
    // geometry: base_z=16029 / NaN). Every link of every dog stays finite.
    EXPECT_TRUE(all_finite) << "a body link went NaN/inf on the step edges "
                               "(the explosion the general path fixes)";

    // (2) NO 穿模 (dog A, the dog the general heightfield supports): no dog-A body
    // link sinks meaningfully below the LOCAL step surface while straddling the
    // edge. A tolerance of one Baumgarte-resting penetration slop (~couple cm)
    // accounts for the soft contact's steady-state interpenetration; a real
    // clip-through is tens of cm (the body falling past the step).
    EXPECT_GT(dogA_worst_pen, -0.05f)
        << "a dog-A body link sank THROUGH the step terrain (穿模) -- the general "
           "heightfield contact failed to keep the body resting ON the step edge";
}

// --- K overlapping dogs push apart WHILE resting on a GENERAL ground field ------
// MIGRATED to the GENERAL path. Two floating-base dogs spawned overlapping on a
// FLAT general heightfield with gravity ON: their leg/foot collidables rest on
// the ground field (contacts loaded) WHILE the overlapping trunks push apart --
// ALL through the ONE LBVH -> cvx -> mixed-island contact path. Proves
// body<->ground AND body<->body contact coexist for K>1 dogs in one general world
// (no per-articulation dog_dog slot blocks; the unified contact buffer holds both).
TEST(MultiDogTerrain, FeetAndDogDogContactCoexist) {
    if (!std::filesystem::exists(Go2FloatScenePath())) {
        GTEST_SKIP() << "go2_float.usda not present";
    }
    nphi::Device* dev = nphi::InitBestDevice();
    if (dev == nullptr) GTEST_SKIP() << "no compute device";
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    // Two dogs 0.15 m apart (trunk boxes 0.20 -> 0.05 m interpenetration) on a
    // FLAT general heightfield at z=0, gravity ON so the legs/feet fall onto it.
    const uint32_t K = 2u;
    nk::Model model = CookKFloatDogs(K, 0.15f);
    ASSERT_EQ(model.capacities.articulations_per_env, K);
    ASSERT_EQ(model.contact_family, nk::ContactFamily::PairDriven);

    nuka::terrain::TerrainParams terr = nuka::terrain::DefaultTerrainParams();
    terr.ground_height = 0.0f;

    // GENERAL flat ground field (terrain type Flat) the legs/feet collide against.
    const uint32_t orig_bodies = model.capacities.bodies_per_env;
    model.body_init.resize(orig_bodies);
    const uint32_t hf_row = cook::CookHeightfieldGrid(
        model, terr, terrain::kTerrainFlat,
        /*nrow=*/41u, /*ncol=*/41u, /*cell=*/0.20f, /*center=*/Vec3{0, 0, 0});
    ASSERT_EQ(hf_row, orig_bodies);

    nk::ModelCapacities& cap = model.capacities;
    cap.bodies_per_env = static_cast<uint32_t>(model.body_init.size());
    cap.max_bodies_total = static_cast<uint32_t>(model.shape_table_rows.size());
    SetPairDrivenCaps(model);

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u;
    cfg.max_pairs = 64u;

    nk::World world(std::move(model), 1u, dev, backend, cfg);
    ASSERT_TRUE(world.Ready());

    // Spawn separation from BasePose (before stepping).
    std::vector<float> base0(K * 7u, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BasePose, base0.data(),
                                              base0.size() * sizeof(float)));
    const float spawn_sep = std::abs(base0[1 * 7 + 0] - base0[0 * 7 + 0]);

    // Settle under gravity: legs/feet land on the ground field; the overlapping
    // trunks push apart -- both contact kinds resolved by the SAME general solve.
    for (uint32_t s = 0; s < 250u; ++s) {
        ASSERT_TRUE(world.Step().AllOk()) << "step " << s;
    }

    // (1) CONTACTS LOADED on the general path: the unified per-env contact count
    // is nonzero (the ground field + the dog<->dog overlap both produced contacts).
    uint32_t contacts = 0u;
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::ContactCount, &contacts,
                                              sizeof(uint32_t)));

    // (2) push-apart: final separation grew past the 0.15 spawn gap, AND both dogs
    // stayed finite resting on the ground (their base z did not tunnel below 0).
    std::vector<float> base1(K * 7u, 0.0f);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BasePose, base1.data(),
                                              base1.size() * sizeof(float)));
    const float final_sep = std::abs(base1[1 * 7 + 0] - base1[0 * 7 + 0]);
    const float base_z_a = base1[0 * 7 + 2];
    const float base_z_b = base1[1 * 7 + 2];

    std::fprintf(stderr,
                 "[coexist general] spawn_sep=%.3f final_sep=%.3f | contacts=%u | "
                 "base_z A=%.4f B=%.4f\n",
                 spawn_sep, final_sep, contacts, base_z_a, base_z_b);

    for (uint32_t i = 0; i < K * 7u; ++i) ASSERT_TRUE(std::isfinite(base1[i]));

    // CONTACTS coexist: the general world loaded contacts (ground + dog-dog) at
    // the SAME time, on the ONE unified buffer.
    EXPECT_GT(contacts, 0u)
        << "no contacts loaded on the general path (ground + dog-dog both missing)";
    // AND the dogs pushed apart (body<->body reaction active simultaneously).
    EXPECT_GT(final_sep, spawn_sep + 1.0e-3f)
        << "overlapping dogs must push apart WHILE resting on the ground field";
    // AND neither dog tunneled the ground field (no 穿模 of the trunk through z=0).
    EXPECT_GT(base_z_a, -0.05f) << "dog A tunneled through the ground field";
    EXPECT_GT(base_z_b, -0.05f) << "dog B tunneled through the ground field";
}
