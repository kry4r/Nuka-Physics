// Complete contact steps check separation, momentum, resting geometry and repeatability.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>

#include "collision/shape_kind.hpp"
#include "collision/mesh_surface.hpp"
#include "import/cooker/mesh_surface_cooker.hpp"
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
    cfg.velocity_iterations_override = true;
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

TEST(PairDrivenSolve, TwoFreeBoxesCollideExchangeMomentum) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    // Equal masses approach with zero total momentum and 0.02 m of initial overlap.
    // Zero restitution dissipates the approach velocity while split impulses separate them.
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

    EXPECT_GT(sep, 0.18f + 1.0e-3f) << "overlapping boxes must push apart";
    EXPECT_NEAR(snap.lin[0].x, 0.0f, 1.0e-3f);
    EXPECT_NEAR(snap.lin[1].x, 0.0f, 1.0e-3f);
    EXPECT_NEAR(snap.lin[0].x + snap.lin[1].x, 0.0f, 2.0e-4f)
        << "x-momentum must be conserved (sum ~ 0)";
    const float energy = 0.5f * (snap.lin[0].LengthSq() + snap.lin[1].LengthSq());
    EXPECT_LT(energy, 1.0e-6f);
}

// --- (b2) a free box rests on a static ground plane (free x static) ----------
TEST(PairDrivenSolve, ParticleImpactPreservesPhysicalMomentumAcrossIntegrators) {
    const auto backend = GetBackend();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    namespace scene = nuka::scene;
    namespace cook = nuka::scene::cook;
    for (uint32_t variant = 0u; variant < 3u; ++variant) {
        for (float body_mass : {0.1f, 1.0f, 100.0f}) {
            SCOPED_TRACE(::testing::Message() << "variant=" << variant << " mass=" << body_mass);
            scene::SceneIR source;
            scene::RigidBodyRecord body;
            body.mass = body_mass;
            body.inertia = Vec3{1, 1, 1} * (0.4f * body_mass * 0.05f * 0.05f);
            const uint32_t owner = source.AddRigidBody(body);
            scene::CollisionShapeRecord sphere;
            sphere.body_id = owner;
            sphere.type = scene::ShapeType::Sphere;
            sphere.radius = 0.05f;
            source.AddCollisionShape(sphere);
            auto model = std::move(cook::CookSceneToModel(source, 1u, {}).model);
            cook::VbdCookInput particle;
            particle.positions = {{0.075f, 0, 0}};
            particle.velocities = {{-1, 0, 0}};
            particle.inv_mass = {1.0f};
            particle.half_thickness = {0.02f};
            particle.friction = {0.0f};
            particle.vbd_count = variant == 0u ? 0u : 1u;
            cook::CookVbdParticles(model, 1u, particle);
            model.particles.pp_contact_d_min = 0.04f;
            auto cfg = Cfg(0.0f);
            cfg.dt = 0.001f;
            cfg.vel_iters = 128u;
            cfg.velocity_iterations_override = true;
            cfg.cloth_integrator = variant == 2u ? 1u : 0u;
            nk::World world(std::move(model), 1u, backend.dev, backend.backend, cfg);
            ASSERT_TRUE(world.Ready()) << world.CreationError();
            ASSERT_EQ(world.SetExecutionMode(nk::World::ExecutionMode::Graph), nphi::Status::Ok);
            const size_t bodies = world.GetModel().capacities.bodies_per_env;
            std::vector<Vec3> rigid_velocity(bodies), angular(bodies);
            std::vector<Transform> poses(bodies);
            Vec3 position{}, velocity{};
            for (uint32_t step = 0u; step < 40u; ++step) {
                ASSERT_TRUE(world.Step().AllOk());
                ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::ParticleVel, &velocity, sizeof(velocity)));
                ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::ParticlePos, &position, sizeof(position)));
                ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BodyLinearVelocity,
                    rigid_velocity.data(), bodies * sizeof(Vec3)));
                ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BodyAngularVelocity,
                    angular.data(), bodies * sizeof(Vec3)));
                ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::BodyPose,
                    poses.data(), bodies * sizeof(Transform)));
                const auto momentum = velocity + rigid_velocity[owner] * body_mass;
                EXPECT_LT((momentum - Vec3{-1, 0, 0}).Length(), 2e-5f);
                const double kinetic = 0.5 * velocity.LengthSq() +
                    0.5 * body_mass * rigid_velocity[owner].LengthSq() +
                    0.2 * body_mass * 0.05 * 0.05 * angular[owner].LengthSq();
                EXPECT_LE(kinetic, 0.5 + 2e-5);
                EXPECT_GE((position - poses[owner].position).Length(), 0.07f - 2e-4f);
            }
            const float expected = -1.0f / (1.0f + body_mass);
            EXPECT_NEAR(velocity.x, expected, 2e-4f);
            EXPECT_NEAR(rigid_velocity[owner].x, expected, 2e-4f);
        }
    }
}

TEST(PairDrivenSolve, DampedSpringConvergesToPhysicalVelocityLaw) {
    const auto backend = GetBackend();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    namespace scene = nuka::scene;
    namespace cook = nuka::scene::cook;
    constexpr float initial = 1.05f;
    constexpr double duration = 0.1, stiffness = 1000.0, alpha = 10.0, frequency = 30.0;
    const double amplitude = double(initial) - 1.0;
    const double exact_position = 1.0 + amplitude * std::exp(-alpha * duration) *
        (std::cos(frequency * duration) + alpha / frequency * std::sin(frequency * duration));
    const double exact_velocity = -amplitude * std::exp(-alpha * duration) *
        stiffness / frequency * std::sin(frequency * duration);
    for (uint32_t integrator : {0u, 1u}) {
        double previous_error = 0.0;
        for (uint32_t steps : {20u, 40u, 80u}) {
            SCOPED_TRACE(::testing::Message() << "integrator=" << integrator << " steps=" << steps);
            auto model = std::move(cook::CookSceneToModel(scene::SceneIR{}, 1u, {}).model);
            cook::VbdCookInput input;
            input.positions = {{0, 0, 0}, {initial, 0, 0}};
            input.velocities.resize(2u);
            input.inv_mass = {0.0f, 1.0f};
            input.half_thickness = {0.0f, 0.0f};
            input.friction = {0.0f, 0.0f};
            input.vbd_count = 2u;
            nk::VbdElement spring;
            spring.vertex[0] = 0u;
            spring.vertex[1] = 1u;
            spring.rest[0] = 1.0f;
            spring.rest[1] = static_cast<float>(stiffness);
            spring.damping = 0.02f;
            input.vbd_elements.push_back(spring);
            cook::CookVbdParticles(model, 1u, input);
            auto cfg = Cfg(0.0f);
            cfg.dt = static_cast<float>(duration / steps);
            cfg.vel_iters = 128u;
            cfg.cloth_integrator = integrator;
            nk::World world(std::move(model), 1u, backend.dev, backend.backend, cfg);
            ASSERT_TRUE(world.Ready()) << world.CreationError();
            ASSERT_EQ(world.SetExecutionMode(nk::World::ExecutionMode::Graph), nphi::Status::Ok);
            for (uint32_t step = 0u; step < steps; ++step) ASSERT_TRUE(world.Step().AllOk()) << step;
            Vec3 position[2], velocity[2];
            ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::ParticlePos, position, sizeof(position)));
            ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::ParticleVel, velocity, sizeof(velocity)));
            const double error = std::hypot(double(position[1].x) - exact_position,
                                           (double(velocity[1].x) - exact_velocity) / std::sqrt(stiffness));
            std::fprintf(stderr, "[damped spring] integrator=%u steps=%u x=%.9g v=%.9g error=%.9g\n",
                         integrator, steps, position[1].x, velocity[1].x, error);
            EXPECT_NEAR(position[0].LengthSq(), 0.0f, 1e-12f);
            EXPECT_NEAR(velocity[0].LengthSq(), 0.0f, 1e-12f);
            if (previous_error > 0.0) EXPECT_LT(error, previous_error * (integrator == 0u ? 0.35 : 0.6));
            if (steps == 80u) EXPECT_LT(error, integrator == 0u ? 5e-5 : 2e-3);
            previous_error = error;
        }
    }
}

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
            namespace scene = nuka::scene;
            scene::SceneIR source;
            for (uint32_t body = 0u; body < 2u; ++body) {
                scene::RigidBodyRecord record;
                record.is_static = body == ground;
                record.local_transform.position = {0, 0, body == box ? 0.105f : 0.0f};
                record.mass = 1.0f;
                record.inertia = {0.006666667f, 0.006666667f, 0.006666667f};
                scene::CollisionShapeRecord shape;
                shape.body_id = source.AddRigidBody(record);
                shape.type = body == ground ? scene::ShapeType::Plane : scene::ShapeType::TriMesh;
                if (body == box) {
                    shape.decompose_mode = scene::DecomposeMode::Skip;
                    shape.mesh_vertices = {-.1f,-.1f,-.1f, .1f,-.1f,-.1f, .1f,.1f,-.1f, -.1f,.1f,-.1f,
                        -.1f,-.1f,.1f, .1f,-.1f,.1f, .1f,.1f,.1f, -.1f,.1f,.1f};
                    shape.mesh_indices = {0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
                        1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7};
                }
                source.AddCollisionShape(shape);
            }
            m = std::move(scene::cook::CookSceneToModel(source, 1u, {}).model);
            ASSERT_EQ(m.shape_table_rows[box].kind, nuka::collision::kShapeSdfMesh);
            ASSERT_EQ(m.mesh_surface_info[box].triangle_count, 12u);
            // Conservative bounds do not become the collision surface.
            m.shape_table_rows[box].params[0] = 0.70f;
            m.shape_table_rows[box].params[1] = m.shape_table_rows[box].params[2] =
                m.shape_table_rows[box].params[3] = 0.40f;
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

TEST(PairDrivenSolve, InteriorSurfaceNormalIsOrthogonalAtFloatSpacing) {
    const float vertices[] = {1, 1, 1, 2, 1, 1.5f, 1, 2, 1.25f};
    const uint32_t triangles[] = {0u, 1u, 2u};
    const auto mesh = nuka::import::cooker::CookMeshSurface(vertices, 3u, triangles, 1u);
    const nuka::collision::MeshSurfaceView view{vertices, triangles, mesh.nodes.data(),
        {3u, 1u, mesh.info.node_count}};
    const float plane_z = 1.1875f;
    for (bool oriented : {false, true}) {
        auto info = mesh.info;
        if (oriented) info.flags |= nuka::collision::kMeshSurfaceOriented;
        for (float direction : {-1.0f, 1.0f}) {
            const Vec3 point{1.25f, 1.25f, std::nextafter(plane_z, plane_z + direction)};
            const auto surface = nuka::collision::QueryMeshSurface(view, info, point);
            ASSERT_TRUE(surface.valid);
            ASSERT_EQ(surface.feature, 6u);
            const double height = (double(point.z) - plane_z) / std::sqrt(1.3125);
            EXPECT_NEAR(surface.distance, oriented ? height : std::abs(height), 1e-13);
            EXPECT_NEAR(surface.normal.Dot({1, 0, 0.5f}), 0.0f, 1e-7f);
            EXPECT_NEAR(surface.normal.Dot({0, 1, 0.25f}), 0.0f, 1e-7f);
            EXPECT_GT(surface.normal.z * (oriented ? 1.0f : direction), 0.0f);
            EXPECT_NEAR(surface.normal.LengthSq(), 1.0f, 2e-7f);
        }
    }
}

TEST(PairDrivenSolve, CookedConcaveAndOpenSurfacesSupportBodiesAndParticles) {
    const auto backend = GetBackend();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    namespace scene = nuka::scene;
    namespace cook = nuka::scene::cook;
    const float edge_vertices[] = {0, 0, -1, 0, 0, 1, -0.6f, -0.8f, 0, 0, -1, 0};
    const uint32_t edge_triangles[] = {1, 0, 2, 0, 1, 3};
    auto edge = nuka::import::cooker::CookMeshSurface(edge_vertices, 4u, edge_triangles, 2u);
    edge.info.flags |= nuka::collision::kMeshSurfaceOriented;
    const nuka::collision::MeshSurfaceView edge_view{edge_vertices, edge_triangles,
        edge.nodes.data(), {4u, 2u, edge.info.node_count}};
    const auto exterior = nuka::collision::QueryMeshSurface(edge_view, edge.info, {0.01f, 0, 0});
    ASSERT_TRUE(exterior.valid);
    EXPECT_NEAR(exterior.distance, 0.01f, 1e-7f);
    EXPECT_GT(exterior.normal.x, 0.99f);
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
    for (uint32_t variant = 0u; variant < 8u; ++variant) {
        SCOPED_TRACE(variant);
        const uint32_t surface_case = variant % 4u;
        const bool vertex_block = variant >= 4u;
        const float side = surface_case == 2u ? -1.0f : 1.0f;
        const float initial_height = surface_case == 3u ? -0.003f : side * 0.15f;
        const Vec3 offset{0.07f, -0.03f, 0.02f};
        scene::SceneIR source;
        scene::RigidBodyRecord wall;
        wall.is_static = true;
        wall.local_transform.position = offset * -1.0f;
        const auto wall_id = source.AddRigidBody(wall);
        scene::CollisionShapeRecord mesh;
        mesh.body_id = wall_id;
        mesh.type = scene::ShapeType::TriMesh;
        mesh.mesh_oriented = surface_case == 3u;
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
        sphere.local_transform.position = {0.1f, -0.08f, initial_height};
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
        ASSERT_EQ(loaded.Shapes()[0].mesh_oriented, mesh.mesh_oriented);
        auto model = std::move(cook::CookSceneToModel(loaded, envs, {}).model);
        ASSERT_GT(model.capacities.bodies_per_env, source.Bodies().size());
        ASSERT_GT(model.capacities.max_mesh_triangles, 0u);
        ASSERT_EQ(model.capacities.max_sdf_grids, 0u);
        for (uint32_t body = 0u; body < model.mesh_surface_info.size(); ++body) {
            if (model.mesh_surface_info[body].triangle_count == 0u) continue;
            EXPECT_EQ(model.shape_table_rows[body].kind, nuka::collision::kShapeSdfMesh);
            EXPECT_EQ((model.mesh_surface_info[body].flags & nuka::collision::kMeshSurfaceClosed) != 0u,
                      surface_case == 0u);
            EXPECT_EQ((model.mesh_surface_info[body].flags & nuka::collision::kMeshSurfaceOriented) != 0u,
                      mesh.mesh_oriented);
        }
        cook::VbdCookInput particle;
        particle.positions = {{0.1f, 0.08f, initial_height}};
        particle.velocities = {{0, 0, 0}};
        particle.inv_mass = {100.0f};
        particle.half_thickness = {radius};
        particle.friction = {0.6f};
        particle.vbd_count = vertex_block ? 1u : 0u;
        cook::CookVbdParticles(model, envs, particle);
        model.particles.pp_contact_d_min = 2.0f * radius;
        const uint32_t bodies = model.capacities.bodies_per_env;
        auto cfg = Cfg(-side * 9.81f);
        cfg.dt = 1.0f / 480.0f;
        nk::World world(std::move(model), envs, backend.dev, backend.backend, cfg);
        ASSERT_TRUE(world.Ready()) << world.CreationError();
        ASSERT_EQ(world.SetExecutionMode(nk::World::ExecutionMode::Graph), nphi::Status::Ok);
        std::vector<Transform> poses(envs * bodies);
        std::vector<Vec3> positions(envs), particle_velocities(envs), velocities(envs * bodies);
        std::vector<uint32_t> status(envs);
        float max_penetration = 0.0f;
        float max_contact_rebound = 0.0f;
        const auto read = [&]() {
            return world.GetData().DownloadField(nk::FieldId::BodyPose, poses.data(), poses.size() * sizeof(Transform)) &&
                world.GetData().DownloadField(nk::FieldId::ParticlePos, positions.data(), positions.size() * sizeof(Vec3)) &&
                world.GetData().DownloadField(nk::FieldId::ParticleVel, particle_velocities.data(), particle_velocities.size() * sizeof(Vec3));
        };
        for (uint32_t step = 0u; step < steps; ++step) {
            ASSERT_TRUE(world.Step().AllOk());
            ASSERT_TRUE(read());
            if (step == 0u && surface_case != 3u)
                for (uint32_t env = 0u; env < envs; ++env)
                    EXPECT_NEAR(positions[env].z, initial_height + cfg.gravity[2] * cfg.dt * cfg.dt, 5e-8f);
            for (uint32_t env = 0u; env < envs; ++env) {
                if (side * positions[env].z <= radius + 0.002f)
                    max_contact_rebound = std::max(max_contact_rebound, side * particle_velocities[env].z);
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
        EXPECT_LT(max_penetration, surface_case == 3u ? radius - initial_height : 0.006f);
        EXPECT_LE(max_contact_rebound, 1e-3f);
        std::fprintf(stderr, "[triangle surface] case=%u vertex_block=%u rigid_z=%.7f particle_z=%.7f penetration=%.7f\n",
            surface_case, vertex_block ? 1u : 0u, poses[sphere_id].position.z, positions[0].z, max_penetration);
        const auto before = poses;
        const auto particle_before = positions;
        ASSERT_EQ(world.Reset({0u}), nphi::Status::Ok);
        ASSERT_TRUE(read());
        EXPECT_NEAR(poses[sphere_id].position.z, initial_height, 1e-7f);
        EXPECT_NEAR(positions[0].z, initial_height, 1e-7f);
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
