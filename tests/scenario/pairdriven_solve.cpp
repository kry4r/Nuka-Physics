// ---------------------------------------------------------------------------
// General contact pipeline — Phase 1B SOLVER + INTEGRATION gate.
//
// Proves the GENERAL mixed-island solver: a detected PairDriven manifold
// (broadphase LBVH -> cvx narrowphase -> the unified contact buffer) becomes a
// COUPLED two-sided NkRow assembled by OpAssembleRowsPairDriven (S1/S2/S5) and
// solved by the generalized SolveRowsBlockIsland (S3 per-articulation qdot tiles
// + S4 two-key schedule), scattering reaction into BOTH sides. NO special-casing:
// a free rigid body, an articulation link, and a static collidable are all just
// physics bodies fed into the EXISTING coupling engine (SolveUnionRowWarp).
//
// Cases (the Phase 1 exit gate, §9):
//   (b1) TWO free rigid boxes collide head-on -> push apart + SYMMETRIC momentum
//        exchange (free x free, no articulation).
//   (b2) a free box resting on a STATIC ground plane stays (free x static; the
//        static side scatters no reaction, the box is held up).
//   (e)  TWO-RUN byte-identity for the PairDriven solve path (it does NOT inherit
//        single-dog D1 -- proven here directly on q/qdot/body velocity/lambda).
//
// The dog cases (artic x rigid box-on-dog mixed island; artic x artic two dogs)
// live in pairdriven_dog_solve.cpp (they need the go2 USD asset).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>

#include "collision/shape_kind.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindBox   = 2u;  // collision::ShapeKind Box
constexpr uint32_t kKindPlane = 3u;  // collision::ShapeKind Plane

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

nk::Pipeline::SolverConfig Cfg(float gz) {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = gz;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u;
    cfg.max_pairs = 32u;
    return cfg;
}

void AddBox(nk::Model& m, const Vec3& pos, const Vec3& vel, float half,
            float mass, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.linear_velocity = vel;
    bi.inv_mass = mass > 0.0f ? 1.0f / mass : 0.0f;
    if (mass > 0.0f) {
        const float ii = 1.0f / (mass * half * half);  // solid-ish box diag.
        bi.inv_inertia = Vec3{ii, ii, ii};
    } else {
        bi.inv_inertia = Vec3{0, 0, 0};
    }
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id;
    sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

constexpr uint32_t kKindCapsule = 1u;  // collision::ShapeKind Capsule

// A capsule lying horizontally along Y, then rotated by yaw_z about world Z.
void AddCapsule(nk::Model& m, const Vec3& pos, float radius, float half_height,
                float yaw_z, float mass, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.pose.rotation = nuka::math::Quat::FromAxisAngle(Vec3{0, 0, 1}, yaw_z) *
        nuka::math::Quat::FromAxisAngle(Vec3{1, 0, 0}, -1.57079632679f);
    bi.inv_mass = mass > 0.0f ? 1.0f / mass : 0.0f;
    // Small light body -> large inv-inertia, so a spurious contact torque would
    // tumble it hard (the ejection signature); the fix keeps it torque-free.
    const float ii = mass > 0.0f ? 1.0f / (mass * radius * radius) : 0.0f;
    bi.inv_inertia = Vec3{ii, ii, ii};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindCapsule;
    sh.params[0] = radius; sh.params[1] = half_height; sh.params[2] = radius;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// A large static box (immovable plate); top face at pos.z + half_z.
void AddStaticBox(nk::Model& m, const Vec3& pos, const Vec3& half, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = 0.0f; bi.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half.x; sh.params[1] = half.y; sh.params[2] = half.z;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

void AddGroundPlane(nk::Model& m, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();      // plane at z=0.
    bi.inv_mass = 0.0f;                   // static (immovable; rigid arm im==0 no-op).
    bi.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindPlane;
    sh.params[0] = 0.0f; sh.params[1] = 0.0f; sh.params[2] = 0.0f;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id;                 // a REGULAR (im==0) body row, in the LBVH.
    sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

void FinishBodyModel(nk::Model& m) {
    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = 16u;       // candidate slots / unified buffer.
    cap.max_rows_per_env = 16u * nk::kPairDrivenRowsPerSlot;  // general row budget.
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;            // env-local pairs.
}

struct Snapshot {
    bool ok = false;
    std::vector<Vec3> lin;   // body linear velocity [bodies]
    std::vector<Transform> pose;  // body pose [bodies]
    std::vector<float> lambda;    // [rows]
};

Snapshot Download(nk::World& w, uint32_t bodies, uint32_t rows) {
    Snapshot s;
    s.lin.assign(bodies, Vec3::Zero());
    s.pose.assign(bodies, Transform::Identity());
    s.lambda.assign(rows, 0.0f);
    nk::Data& d = w.GetData();
    s.ok = d.DownloadField(nk::FieldId::BodyLinearVelocity, s.lin.data(),
                           s.lin.size() * sizeof(Vec3)) &&
           d.DownloadField(nk::FieldId::BodyPose, s.pose.data(),
                           s.pose.size() * sizeof(Transform)) &&
           d.DownloadField(nk::FieldId::Lambda, s.lambda.data(),
                           s.lambda.size() * sizeof(float));
    return s;
}

}  // namespace

// --- (b1) two free boxes collide + symmetric momentum exchange ---------------
TEST(PairDrivenSolve, TwoFreeBoxesCollideExchangeMomentum) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    // Box A at x=-0.09 moving +X (toward B); box B at x=+0.09 moving -X. Half 0.10
    // each -> centers 0.18 apart, faces touch at 0.20 reach -> 0.02 overlap. Equal
    // mass, head-on, no gravity: a symmetric elastic-ish exchange (PGS) MUST push
    // them apart and reverse the sign of each box's x-velocity contribution.
    nk::Model m;
    AddBox(m, Vec3{-0.09f, 0, 0}, Vec3{+0.5f, 0, 0}, 0.10f, 1.0f, 0);
    AddBox(m, Vec3{+0.09f, 0, 0}, Vec3{-0.5f, 0, 0}, 0.10f, 1.0f, 1);
    FinishBodyModel(m);
    const uint32_t bodies = 2u, rows = m.capacities.max_rows_per_env;

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg(0.0f));
    ASSERT_TRUE(w.Ready());
    for (uint32_t s = 0; s < 60u; ++s) ASSERT_TRUE(w.Step().AllOk()) << s;

    const Snapshot snap = Download(w, bodies, rows);
    ASSERT_TRUE(snap.ok);
    const float xa = snap.pose[0].position.x;
    const float xb = snap.pose[1].position.x;
    const float sep = std::abs(xb - xa);
    std::fprintf(stderr,
                 "[pd free-boxes] xA=%.4f xB=%.4f sep=%.4f vA.x=%.4f vB.x=%.4f\n",
                 xa, xb, sep, snap.lin[0].x, snap.lin[1].x);

    // (1) PUSH-APART: separation grew past the 0.18 spawn gap toward ~0.20 (2*half).
    EXPECT_GT(sep, 0.18f + 1.0e-3f) << "overlapping boxes must push apart";
    // (2) TWO-WAY: box A (was +X) ends moving -X, box B (was -X) ends moving +X
    // (the contact reversed BOTH, the symmetric momentum exchange).
    EXPECT_LT(snap.lin[0].x, 0.0f) << "box A reaction must reverse its +X motion";
    EXPECT_GT(snap.lin[1].x, 0.0f) << "box B reaction must reverse its -X motion";
    // (3) SYMMETRY: equal mass, head-on -> the two post-contact x-speeds match.
    EXPECT_NEAR(std::abs(snap.lin[0].x), std::abs(snap.lin[1].x), 0.05f)
        << "equal-mass head-on collision must exchange momentum symmetrically";
    // (4) momentum: total x-momentum ~0 (started +0.5 + -0.5 == 0, no external X).
    EXPECT_NEAR(snap.lin[0].x + snap.lin[1].x, 0.0f, 0.05f)
        << "x-momentum must be conserved (sum ~ 0)";
}

// --- (b2) a free box rests on a static ground plane (free x static) ----------
TEST(PairDrivenSolve, FreeBoxRestsOnStaticGround) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    for (uint32_t representation = 0u; representation < 3u; ++representation) {
        SCOPED_TRACE(representation);
        const uint32_t box = representation == 2u ? 0u : 1u;
        const uint32_t ground = 1u - box;
        nk::Model m;
        if (ground == 0u) AddGroundPlane(m, 0);
        AddBox(m, Vec3{0, 0, 0.105f}, Vec3{0, 0, 0}, 0.10f, 1.0f, box);
        if (ground == 1u) AddGroundPlane(m, 1);
        if (representation != 0u) {
            auto& shape = m.shape_table_rows[box];
            shape.kind = nuka::collision::kShapeSdfMesh;
            // A conservative bound must not become the collision surface.
            shape.params[0] = 0.70f;
            shape.params[1] = shape.params[2] = shape.params[3] = 0.40f;
            for (float x : {-0.10f, 0.10f})
                for (float y : {-0.10f, 0.10f})
                    for (float z : {-0.10f, 0.10f})
                        m.samp_points.insert(m.samp_points.end(), {x, y, z});
            m.samp_ranges.assign(4u, 0u);
            m.samp_ranges[box * 2u + 1u] = 8u;
            m.capacities.max_samp_points = 8u;
        }
        FinishBodyModel(m);
        const uint32_t rows = m.capacities.max_rows_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg(-9.81f));
        ASSERT_TRUE(w.Ready());
        for (uint32_t s = 0; s < 240u; ++s) ASSERT_TRUE(w.Step().AllOk()) << s;

        const Snapshot snap = Download(w, 2u, rows);
        ASSERT_TRUE(snap.ok);
        uint32_t status = 0u;
        ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::EnvStatus, &status, sizeof(status)));
        EXPECT_EQ(status, 0u);
        const float box_z = snap.pose[box].position.z;
        const float ground_z = snap.pose[ground].position.z;
        std::fprintf(stderr, "[pd box-on-ground] representation=%u box_z=%.4f ground_z=%.4f vbox.z=%.5f\n",
                     representation, box_z, ground_z, snap.lin[box].z);
        EXPECT_GT(box_z, 0.10f - 0.03f) << "box sank through the ground";
        EXPECT_LT(box_z, 0.10f + 0.03f) << "box floats above its resting height";
        EXPECT_NEAR(ground_z, 0.0f, 1.0e-5f) << "static ground must not move";
        EXPECT_NEAR(snap.lin[ground].x, 0.0f, 1.0e-5f);
        EXPECT_NEAR(snap.lin[ground].z, 0.0f, 1.0e-5f);
        EXPECT_LT(std::abs(snap.lin[box].z), 0.2f) << "box must come to rest on the ground";
    }
}

TEST(PairDrivenSolve, CookedConcaveAndOpenSurfacesSupportBodiesAndParticles) {
    const auto backend = GetBackend();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    namespace scene = nuka::scene;
    namespace cook = nuka::scene::cook;
    constexpr uint32_t envs = 2u, steps = 120u;
    constexpr float radius = 0.02f;
    struct StoredScene {
        std::filesystem::path nks_path, nka_path;
        ~StoredScene() {
            std::error_code error;
            std::filesystem::remove(nks_path, error);
            std::filesystem::remove(nka_path, error);
        }
    };
    for (uint32_t surface_case = 0u; surface_case < 3u; ++surface_case) {
        SCOPED_TRACE(surface_case);
        const float side = surface_case == 2u ? -1.0f : 1.0f;
        const Vec3 offset{0.07f, -0.03f, 0.02f};
        scene::SceneIR source;
        scene::RigidBodyRecord wall;
        wall.is_static = true;
        wall.local_transform.position = offset * -1.0f;
        const auto wall_id = source.AddRigidBody(wall);
        scene::CollisionShapeRecord mesh;
        mesh.body_id = wall_id;
        mesh.type = scene::ShapeType::TriMesh;
        mesh.local_transform.position = offset;
        if (surface_case == 0u) {
            const Vec3 polygon[] = {{-0.4f, 0, -0.1f}, {0.4f, 0, -0.1f}, {0.4f, 0, 0},
                {-0.2f, 0, 0}, {-0.2f, 0, 0.4f}, {-0.4f, 0, 0.4f}};
            for (float y : {-0.4f, 0.4f})
                for (auto p : polygon) mesh.mesh_vertices.insert(mesh.mesh_vertices.end(), {p.x, y, p.z});
            for (uint32_t i = 1u; i < 5u; ++i) {
                mesh.mesh_indices.insert(mesh.mesh_indices.end(), {0u, i, i + 1u,
                    6u, i + 7u, i + 6u});
            }
            for (uint32_t i = 0u; i < 6u; ++i) {
                const uint32_t j = (i + 1u) % 6u;
                mesh.mesh_indices.insert(mesh.mesh_indices.end(), {i, j, j + 6u, i, j + 6u, i + 6u});
            }
        } else {
            mesh.mesh_vertices = {-0.4f, -0.4f, 0, 0.4f, -0.4f, 0,
                                   0.4f, 0.4f, 0, -0.4f, 0.4f, 0};
            mesh.mesh_indices = {0u, 1u, 2u, 0u, 2u, 3u};
        }
        source.AddCollisionShape(mesh);
        scene::CollisionShapeRecord other;
        other.body_id = wall_id;
        other.type = scene::ShapeType::Sphere;
        other.radius = 0.01f;
        other.local_transform.position = {4.0f, 0, 0};
        source.AddCollisionShape(other);
        scene::RigidBodyRecord sphere;
        sphere.local_transform.position = {0.1f, -0.08f, side * 0.15f};
        const float inertia = 0.4f * sphere.mass * radius * radius;
        sphere.inertia = {inertia, inertia, inertia};
        const auto sphere_id = source.AddRigidBody(sphere);
        scene::CollisionShapeRecord ball;
        ball.body_id = sphere_id;
        ball.type = scene::ShapeType::Sphere;
        ball.radius = radius;
        source.AddCollisionShape(ball);
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto stem = std::filesystem::temp_directory_path() /
            ("nuka_surface_" + std::to_string(stamp) + "_" + std::to_string(surface_case));
        StoredScene stored{stem.string() + ".nks", stem.string() + ".nka"};
        scene::nks::Save(source, stored.nks_path.string());
        const auto loaded = scene::nks::Load(stored.nks_path.string());
        ASSERT_EQ(loaded.Shapes()[0].mesh_vertices, mesh.mesh_vertices);
        ASSERT_EQ(loaded.Shapes()[0].mesh_indices, mesh.mesh_indices);
        ASSERT_EQ(loaded.Shapes()[0].type, scene::ShapeType::TriMesh);
        auto model = std::move(cook::CookSceneToModel(loaded, envs, {}).model);
        ASSERT_GT(model.capacities.bodies_per_env, source.Bodies().size());
        ASSERT_GT(model.capacities.max_mesh_triangles, 0u);
        ASSERT_EQ(model.capacities.max_sdf_grids, 0u);
        for (uint32_t body = 0u; body < model.mesh_surface_info.size(); ++body) {
            if (model.mesh_surface_info[body].triangle_count == 0u) continue;
            EXPECT_EQ(model.shape_table_rows[body].kind, nuka::collision::kShapeSdfMesh);
            EXPECT_EQ((model.mesh_surface_info[body].flags & nuka::collision::kMeshSurfaceClosed) != 0u,
                      surface_case == 0u);
        }
        cook::XpbdCookInput particle;
        particle.positions = {{0.1f, 0.08f, side * 0.15f}};
        particle.velocities = {{0, 0, 0}};
        particle.inv_mass = {100.0f};
        cook::CookXpbdParticles(model, envs, particle);
        model.particles.pp_contact_d_min = 2.0f * radius;
        const uint32_t bodies = model.capacities.bodies_per_env;
        auto cfg = Cfg(-side * 9.81f);
        cfg.dt = 1.0f / 480.0f;
        nk::World world(std::move(model), envs, backend.dev, backend.backend, cfg);
        ASSERT_TRUE(world.Ready()) << world.CreationError();
        ASSERT_EQ(world.SetExecutionMode(nk::World::ExecutionMode::Graph), nphi::Status::Ok);
        std::vector<Transform> poses(envs * bodies);
        std::vector<Vec3> positions(envs), velocities(envs * bodies);
        std::vector<uint32_t> status(envs);
        float max_penetration = 0.0f;
        const auto read = [&]() {
            return world.GetData().DownloadField(nk::FieldId::BodyPose, poses.data(), poses.size() * sizeof(Transform)) &&
                world.GetData().DownloadField(nk::FieldId::ParticlePos, positions.data(), positions.size() * sizeof(Vec3));
        };
        for (uint32_t step = 0u; step < steps; ++step) {
            ASSERT_TRUE(world.Step().AllOk());
            ASSERT_TRUE(read());
            for (uint32_t env = 0u; env < envs; ++env) {
                max_penetration = std::max(max_penetration,
                    radius - std::min(side * poses[env * bodies + sphere_id].position.z,
                                      side * positions[env].z));
            }
        }
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::EnvStatus, status.data(), status.size() * sizeof(uint32_t)));
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BodyLinearVelocity,
            velocities.data(), velocities.size() * sizeof(Vec3)));
        for (uint32_t env = 0u; env < envs; ++env) {
            EXPECT_EQ(status[env], 0u);
            EXPECT_NEAR(side * poses[env * bodies + sphere_id].position.z, radius, 0.003f);
            EXPECT_NEAR(side * positions[env].z, radius, 0.003f);
            EXPECT_LT(std::fabs(velocities[env * bodies + sphere_id].z), 0.1f);
        }
        EXPECT_LT(max_penetration, 0.006f);
        std::fprintf(stderr, "[triangle surface] case=%u rigid_z=%.7f particle_z=%.7f penetration=%.7f\n",
            surface_case, poses[sphere_id].position.z, positions[0].z, max_penetration);
        const auto before = poses;
        const auto particle_before = positions;
        ASSERT_EQ(world.Reset({0u}), nphi::Status::Ok);
        ASSERT_TRUE(read());
        EXPECT_NEAR(poses[sphere_id].position.z, side * 0.15f, 1e-7f);
        EXPECT_NEAR(positions[0].z, side * 0.15f, 1e-7f);
        EXPECT_EQ(poses[bodies + sphere_id].position.z, before[bodies + sphere_id].position.z);
        EXPECT_EQ(positions[1].z, particle_before[1].z);
        for (uint32_t step = 0u; step < steps; ++step) ASSERT_TRUE(world.Step().AllOk());
        ASSERT_TRUE(read());
        EXPECT_NEAR(poses[sphere_id].position.z, before[sphere_id].position.z, 1e-6f);
        EXPECT_NEAR(positions[0].z, particle_before[0].z, 1e-6f);
    }
}

// --- (e) two-run byte-identity for the PairDriven solve path ------------------
TEST(PairDrivenSolve, TwoRunsByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    auto run = [&]() -> Snapshot {
        nk::Model m;
        AddGroundPlane(m, 0);
        AddBox(m, Vec3{-0.05f, 0, 0.12f}, Vec3{+0.3f, 0, 0}, 0.10f, 1.0f, 1);
        AddBox(m, Vec3{+0.05f, 0, 0.40f}, Vec3{-0.3f, 0, 0}, 0.10f, 1.0f, 2);
        FinishBodyModel(m);
        const uint32_t bodies = 3u, rows = m.capacities.max_rows_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg(-9.81f));
        if (!w.Ready()) return {};
        for (uint32_t s = 0; s < 120u; ++s)
            if (!w.Step().AllOk()) return {};
        return Download(w, bodies, rows);
    };

    const Snapshot a = run();
    const Snapshot c = run();
    ASSERT_TRUE(a.ok && c.ok);

    auto same = [](const auto& x, const auto& y) {
        return x.size() == y.size() &&
               std::memcmp(x.data(), y.data(),
                           x.size() * sizeof(typename std::decay_t<decltype(x)>::value_type)) == 0;
    };
    EXPECT_TRUE(same(a.lin, c.lin)) << "body velocity differs across runs";
    EXPECT_TRUE(same(a.pose, c.pose)) << "body pose differs across runs";
    EXPECT_TRUE(same(a.lambda, c.lambda)) << "contact lambda differs across runs";
}

// --- capsule/box pile on a static plate settles (no ejection) ----------------
// The general reproduction of the ejection bug: small light capsules (bolts) lying
// flat on a box plate, in a dense multi-contact island. Capsule x box was the ONE
// EPA-path resting contact, and its single off-centre over-deep witness injected a
// torque that tumbled a light capsule through the plate and flung it at m/s. The
// closest-feature manifold (2 symmetric endpoint points, true depth) keeps the pile
// at rest. If this bug returns, a body tunnels the plate and ejects.
TEST(PairDrivenSolve, CapsuleBoxPileSettlesNoEject) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    nk::Model m;
    AddStaticBox(m, Vec3{0, 0, -0.5f}, Vec3{1.0f, 1.0f, 0.5f}, 0);  // plate top z=0
    const float r = 0.02f, hh = 0.03f;
    int32_t id = 1;
    // A 3x3 raft of flat bolts at assorted yaws resting at z=r. Grid pitch 0.14 >
    // 2*(hh+r)=0.10, so no capsule overlaps another at spawn (a bolt is a bolt on a
    // plate -- the resting posture that surfaced the bug, no spawn penetration).
    const float yaw[9] = {0.0f, 0.7f, 1.4f, 2.1f, 2.8f, 0.35f, 1.05f, 1.75f, 2.45f};
    int k = 0;
    for (int ix = 0; ix < 3; ++ix)
        for (int iy = 0; iy < 3; ++iy) {
            const Vec3 p{-0.14f + 0.14f * ix, -0.14f + 0.14f * iy, r};
            AddCapsule(m, p, r, hh, yaw[k++], 0.02f, id++);
        }
    // A few small boxes (nuts/washers) on a separate row, clear of the bolts.
    AddBox(m, Vec3{-0.14f, 0.34f, 0.015f}, Vec3{0, 0, 0}, 0.015f, 0.02f, id++);
    AddBox(m, Vec3{0.0f, 0.34f, 0.015f}, Vec3{0, 0, 0}, 0.015f, 0.02f, id++);
    AddBox(m, Vec3{0.14f, 0.34f, 0.02f}, Vec3{0, 0, 0}, 0.02f, 0.02f, id++);

    FinishBodyModel(m);
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    m.capacities.max_contacts_per_env = 96u;
    m.capacities.max_rows_per_env = 96u * nk::kPairDrivenRowsPerSlot;

    nk::Pipeline::SolverConfig cfg = Cfg(-9.81f);
    cfg.max_pairs = 192u;
    nk::World w(std::move(m), 1u, b.dev, b.backend, cfg);
    ASSERT_TRUE(w.Ready());
    float worst_v = 0.0f;
    for (uint32_t s = 0; s < 400u; ++s) {
        ASSERT_TRUE(w.Step().AllOk()) << s;
        if (s >= 350u) {  // settled window
            const Snapshot sn = Download(w, bodies, m.capacities.max_rows_per_env);
            for (uint32_t i = 1; i < bodies; ++i)
                worst_v = std::max(worst_v, std::sqrt(sn.lin[i].Dot(sn.lin[i])));
        }
    }
    const Snapshot snap = Download(w, bodies, m.capacities.max_rows_per_env);
    ASSERT_TRUE(snap.ok);
    float min_z = 1e9f, max_xy = 0.0f;
    bool finite = true;
    for (uint32_t i = 1; i < bodies; ++i) {
        const Vec3 p = snap.pose[i].position;
        finite &= std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        min_z = std::min(min_z, p.z);
        max_xy = std::max(max_xy, std::max(std::abs(p.x), std::abs(p.y)));
    }
    std::fprintf(stderr, "[pd capsule-pile] bodies=%u min_z=%.4f max_xy=%.4f worst_v=%.4f\n",
                 bodies, min_z, max_xy, worst_v);
    // The EPA torque bug tunnelled a bolt through the plate and flung it at metres/s
    // (min_z << 0, max_xy >> cluster, worst_v ~ 5). The closest-feature manifold keeps
    // every bolt ON the plate, in the cluster, and sub-launch. (A residual ~0.5 m/s
    // jitter of these 20 g bolts is a dense light-body settling debt, NOT ejection.)
    EXPECT_TRUE(finite) << "a body position went NaN/inf";
    EXPECT_GT(min_z, -0.03f) << "a body tunnelled through the plate (穿模/eject)";
    EXPECT_LT(max_xy, 0.6f) << "a body was flung out of the cluster (ejection)";
    EXPECT_LT(worst_v, 1.5f) << "a body launched off the pile (ejection, not jitter)";
}
