#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "import/usd_importer.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"

namespace {
namespace nk = nuka::nk;
namespace phi = nuka::phi;
namespace scene = nuka::scene;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

struct Backend {
    phi::Device* device = phi::InitBestDevice();
    phi::Backend* backend = device ? phi::DeviceInitBackend(device, nullptr) : nullptr;
};

Backend& Device() {
    static Backend backend;
    return backend;
}

class IslandMode {
public:
    explicit IslandMode(bool force_static) {
        const char* previous = std::getenv("NUKA_FORCE_STATIC_ISLANDS");
        had_value_ = previous != nullptr;
        if (previous) previous_ = previous;
        Set(force_static ? "1" : "0");
    }
    ~IslandMode() { Set(had_value_ ? previous_.c_str() : nullptr); }
private:
    static void Set(const char* value) {
#ifdef _WIN32
        _putenv_s("NUKA_FORCE_STATIC_ISLANDS", value ? value : "");
#else
        if (value) setenv("NUKA_FORCE_STATIC_ISLANDS", value, 1);
        else unsetenv("NUKA_FORCE_STATIC_ISLANDS");
#endif
    }
    bool had_value_ = false;
    std::string previous_;
};

nk::Model CookFreeBody(const scene::RigidBodyRecord& body, bool with_floor) {
    scene::SceneIR source;
    if (with_floor) {
        scene::RigidBodyRecord floor;
        floor.name = "floor";
        floor.is_static = true;
        floor.local_transform.position = {0, 0, -0.1f};
        scene::CollisionShapeRecord shape;
        shape.body_id = source.AddRigidBody(floor);
        shape.type = scene::ShapeType::Box;
        shape.half_extents = {2, 2, 0.1f};
        source.AddCollisionShape(shape);
    }
    scene::CollisionShapeRecord shape;
    shape.body_id = source.AddRigidBody(body);
    shape.type = scene::ShapeType::Sphere;
    shape.radius = 0.1f;
    source.AddCollisionShape(shape);
    scene::cook::CookToModelOptions options;
    options.enable_contacts = with_floor;
    return scene::cook::CookToModel(source, 1, options).model;
}

nk::Pipeline::SolverConfig Config() {
    nk::Pipeline::SolverConfig config;
    config.dt = 0.002f;
    config.gravity[0] = config.gravity[1] = config.gravity[2] = 0.0f;
    config.pos_iters = 0u;
    return config;
}

template <typename T>
std::vector<T> Read(nk::World& world, nk::FieldId field, size_t count) {
    std::vector<T> values(count);
    EXPECT_TRUE(world.GetData().DownloadField(field, values.data(), count * sizeof(T)));
    return values;
}

void ExpectVectorNear(Vec3 actual, Vec3 expected, float tolerance) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}
}  // namespace

TEST(FreeRigidDynamics, WorldAngularVelocityRotatesAboutTheAuthoredCenterOfMass) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (Vec3 offset : {Vec3{}, Vec3{0.025f, -0.01f, 0.026f}}) {
        scene::RigidBodyRecord body;
        body.name = "free_body";
        body.mass = 1.0f;
        body.inertia = {1, 1, 1};
        body.local_transform.position = {0, 0, 2};
        body.local_transform.rotation = Quat::FromAxisAngle({0, 1, 0}, 1.57079632679f);
        body.inertial_transform.position = offset;
        auto model = CookFreeBody(body, false);
        ASSERT_EQ(model.body_init.size(), 1u);
        const Vec3 velocity{0.1f, -0.03f, 0.02f};
        const Vec3 omega{1, 0, 0};
        model.body_init[0].linear_velocity = velocity;
        model.body_init[0].angular_velocity = omega;
        const auto config = Config();
        nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
        ASSERT_TRUE(world.Ready());
        constexpr uint32_t steps = 20u;
        for (uint32_t i = 0; i < steps; ++i) ASSERT_TRUE(world.Step().AllOk());
        const auto poses = Read<Transform>(world, nk::FieldId::BodyPose, 2u);
        const float duration = steps * config.dt;
        const Quat expected_rotation =
            Quat::FromAxisAngle(omega, duration) * body.local_transform.rotation;
        const Vec3 expected_com = body.local_transform.position +
            body.local_transform.rotation.Rotate(offset) + velocity * duration;
        for (const auto& pose : poses) {
            ExpectVectorNear(pose.position + pose.rotation.Rotate(offset), expected_com, 3.0e-6f);
            EXPECT_NEAR(pose.rotation.w, expected_rotation.w, 2.0e-6f);
            EXPECT_NEAR(pose.rotation.x, expected_rotation.x, 2.0e-6f);
            EXPECT_NEAR(pose.rotation.y, expected_rotation.y, 2.0e-6f);
            EXPECT_NEAR(pose.rotation.z, expected_rotation.z, 2.0e-6f);
        }
    }
}

TEST(FreeRigidDynamics, WorldWrenchesAndGravityAreCovariantAndConsumedOnce) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (Quat rotation : {Quat::Identity(), Quat::FromAxisAngle({1, 2, -3}, 1.1f)}) {
        scene::RigidBodyRecord body;
        body.name = "free_body";
        body.mass = 2.0f;
        body.inertia = {0.4f, 0.7f, 1.2f};
        body.local_transform.position = rotation.Rotate({0.3f, -0.4f, 1.5f});
        body.local_transform.rotation = rotation * Quat::FromAxisAngle({2, 1, 3}, 0.6f);
        body.inertial_transform.position = {0.04f, -0.02f, 0.01f};
        body.inertial_transform.rotation = Quat::FromAxisAngle({3, -2, 1}, 0.7f);
        auto config = Config();
        config.dt = 0.001f;
        const Vec3 gravity = rotation.Rotate({1.7f, -2.3f, -8.1f});
        config.gravity[0] = gravity.x;
        config.gravity[1] = gravity.y;
        config.gravity[2] = gravity.z;
        const Vec3 force = rotation.Rotate({0.7f, -0.9f, 1.1f});
        const Vec3 torque = rotation.Rotate(Vec3{0.2f, -0.3f, 0.1f}.Cross({0.7f, -0.9f, 1.1f}));
        nk::World world(CookFreeBody(body, false), 2u, backend.device, backend.backend, config);
        ASSERT_TRUE(world.Ready());
        const std::vector<Vec3> forces{force, Vec3{}}, torques{torque, Vec3{}};
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BodyForce, forces.data(), forces.size() * sizeof(Vec3)));
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BodyTorque, torques.data(), torques.size() * sizeof(Vec3)));
        ASSERT_TRUE(world.Step().AllOk());
        auto velocity = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 2u);
        const auto omega = Read<Vec3>(world, nk::FieldId::BodyAngularVelocity, 2u);
        const auto pose = Read<Transform>(world, nk::FieldId::BodyPose, 2u);
        ExpectVectorNear(velocity[0], (gravity + force / body.mass) * config.dt, 1.0e-6f);
        ExpectVectorNear(velocity[1], gravity * config.dt, 1.0e-6f);
        const Quat principal = body.local_transform.rotation * body.inertial_transform.rotation;
        Vec3 principal_torque = principal.Conjugate().Rotate(torque);
        principal_torque.x /= body.inertia.x;
        principal_torque.y /= body.inertia.y;
        principal_torque.z /= body.inertia.z;
        ExpectVectorNear(omega[0], principal.Rotate(principal_torque) * config.dt, 1.0e-6f);
        ExpectVectorNear(omega[1], Vec3{}, 1.0e-7f);
        const Vec3 initial_com = body.local_transform.TransformPoint(body.inertial_transform.position);
        for (uint32_t env = 0; env < 2u; ++env)
            ExpectVectorNear(pose[env].TransformPoint(body.inertial_transform.position),
                             initial_com + velocity[env] * config.dt, 1.0e-6f);
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyForce, 2u), std::vector<Vec3>(2u));
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyTorque, 2u), std::vector<Vec3>(2u));
        ASSERT_TRUE(world.Step().AllOk());
        velocity = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 2u);
        ExpectVectorNear(velocity[0], gravity * (2.0f * config.dt) + force * (config.dt / body.mass), 1.0e-6f);
        ExpectVectorNear(velocity[1], gravity * (2.0f * config.dt), 1.0e-6f);
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BodyForce, forces.data(), forces.size() * sizeof(Vec3)));
        ASSERT_EQ(world.Reset({1u}), phi::Status::Ok);
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyForce, 2u), forces);
        ASSERT_EQ(world.Reset({0u}), phi::Status::Ok);
        EXPECT_EQ(Read<Vec3>(world, nk::FieldId::BodyForce, 2u), std::vector<Vec3>(2u));
    }
}

TEST(FreeRigidDynamics, FixedAndFloatingArticulationsRespectRotatedGravity) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    const Quat rotation = Quat::FromAxisAngle({1, -3, 2}, 1.1f);
    const Vec3 gravity{2.3f, -4.1f, -8.2f};
    const Vec3 rotated_gravity = rotation.Rotate(gravity);
    for (const char* filename : {"go2_stand.usda", "go2_float.usda"}) {
        SCOPED_TRACE(filename);
        const auto path = std::filesystem::path(NUKA_SOURCE_DIR) / "examples/scenes" / filename;
        ASSERT_TRUE(std::filesystem::exists(path));
        const auto source = nuka::import::LoadUsd(path.string());
        const auto rotated = scene::Compose(scene::SceneIR{}, source, Transform{{}, rotation}, "rotated_");
        scene::cook::CookToModelOptions options;
        options.enable_contacts = false;
        auto config = Config();
        config.dt = 0.0005f;
        config.gravity[0] = gravity.x;
        config.gravity[1] = gravity.y;
        config.gravity[2] = gravity.z;
        nk::World original(scene::cook::CookToModel(source, 1, options).model,
                           2u, backend.device, backend.backend, config);
        config.gravity[0] = rotated_gravity.x;
        config.gravity[1] = rotated_gravity.y;
        config.gravity[2] = rotated_gravity.z;
        nk::World transformed(scene::cook::CookToModel(rotated, 1, options).model,
                              2u, backend.device, backend.backend, config);
        ASSERT_TRUE(original.Ready());
        ASSERT_TRUE(transformed.Ready());
        ASSERT_TRUE(original.Step().AllOk());
        ASSERT_TRUE(transformed.Step().AllOk());
        const uint32_t links = original.GetModel().capacities.links_per_env * 2u;
        const auto qdot = Read<float>(original, nk::FieldId::Qdot, links);
        const auto rotated_qdot = Read<float>(transformed, nk::FieldId::Qdot, links);
        for (uint32_t link = 0; link < links; ++link)
            EXPECT_NEAR(rotated_qdot[link], qdot[link], 2.0e-6f) << "link=" << link;
        const auto velocity = Read<nk::Spatial6>(original, nk::FieldId::LinkVelocity, links);
        const auto rotated_velocity = Read<nk::Spatial6>(transformed, nk::FieldId::LinkVelocity, links);
        for (uint32_t link = 0; link < links; ++link)
            for (uint32_t axis = 0; axis < 6u; ++axis)
                EXPECT_NEAR(rotated_velocity[link].v[axis], velocity[link].v[axis], 2.0e-6f)
                    << "link=" << link << " axis=" << axis;
    }
}

TEST(FreeRigidDynamics, ContactImpulseUsesTheCenterOfMassAndWorldInertia) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (bool force_static : {false, true}) {
        IslandMode mode(force_static);
        for (bool offset_center : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "static=" << force_static
                         << " offset=" << offset_center);
            scene::RigidBodyRecord body;
            body.name = "free_body";
            body.mass = 0.5f;
            body.inertia = {0.003f, 0.007f, 0.011f};
            body.local_transform.position = {0, 0, 0.0995f};
            body.local_transform.rotation = Quat::FromAxisAngle({1, 2, 3}, 1.05f);
            body.inertial_transform.rotation = Quat::FromAxisAngle({3, 1, 2}, 0.66f);
            if (offset_center) body.inertial_transform.position = {0.025f, -0.01f, 0.015f};
            auto model = CookFreeBody(body, true);
            ASSERT_EQ(model.body_init.size(), 2u);
            const auto capacity = model.capacities;
            const Vec3 incoming{0.2f, -0.1f, -0.2f};
            model.body_init[1].linear_velocity = incoming;
            nk::World world(std::move(model), 2u, backend.device, backend.backend, Config());
            ASSERT_TRUE(world.Ready());
            ASSERT_TRUE(world.Step().AllOk());
            const auto linear = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 4u);
            const auto angular = Read<Vec3>(world, nk::FieldId::BodyAngularVelocity, 4u);
            const auto rows = Read<nk::NkRow>(world, nk::FieldId::Urows, 2u * capacity.max_rows_per_env);
            const auto impulses = Read<float>(world, nk::FieldId::Lambda, rows.size());
            const auto counts = Read<uint32_t>(world, nk::FieldId::UcontactCount,
                                              2u * capacity.max_contacts_per_env);
            const auto points = Read<Vec3>(world, nk::FieldId::UcontactPoint, 4u * counts.size());
            const Vec3 center = body.local_transform.position +
                body.local_transform.rotation.Rotate(body.inertial_transform.position);
            const Quat principal_world =
                body.local_transform.rotation * body.inertial_transform.rotation;
            for (uint32_t env = 0; env < 2u; ++env) {
                Vec3 impulse{}, torque{};
                for (uint32_t slot = 0; slot < capacity.max_contacts_per_env; ++slot) {
                    const uint32_t contact = env * capacity.max_contacts_per_env + slot;
                    for (uint32_t point = 0; point < counts[contact]; ++point) {
                        const Vec3 lever = points[4u * contact + point] - center;
                        for (uint32_t axis = 0; axis < 3u; ++axis) {
                            const uint32_t row_index = env * capacity.max_rows_per_env +
                                slot * nk::kPairDrivenRowsPerSlot +
                                axis * nk::kPairDrivenPtsPerSlot + point;
                            const auto& row = rows[row_index];
                            for (const auto& side : {row.a, row.b}) {
                                if (side.kind != nk::kNkSideRigid || side.index != env * 2u + 1u) continue;
                                const Vec3 applied = side.jlin * impulses[row_index];
                                impulse += applied;
                                torque += lever.Cross(applied);
                            }
                        }
                    }
                }
                ASSERT_GT(impulse.LengthSq(), 1.0e-6f);
                Vec3 local_torque = principal_world.Conjugate().Rotate(torque);
                local_torque.x /= body.inertia.x;
                local_torque.y /= body.inertia.y;
                local_torque.z /= body.inertia.z;
                ExpectVectorNear(linear[env * 2u + 1u], incoming + impulse / body.mass, 2.0e-5f);
                ExpectVectorNear(angular[env * 2u + 1u], principal_world.Rotate(local_torque), 3.0e-5f);
            }
        }
    }
}

TEST(FreeRigidDynamics, PairSortingPreservesTheFollowingInertiaField) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (uint32_t environments : {1u, 2u}) {
        for (uint32_t slots : {32u, 512u}) {
            SCOPED_TRACE(::testing::Message() << "envs=" << environments << " slots=" << slots);
            scene::RigidBodyRecord body;
            body.name = "free_body";
            body.mass = 1.0f;
            body.inertia = {0.01f, 0.01f, 0.01f};
            body.local_transform.position = {0, 0, 0.0995f};
            auto model = CookFreeBody(body, true);
            model.capacities.max_contacts_per_env = slots;
            model.capacities.max_rows_per_env = slots * nk::kPairDrivenRowsPerSlot;
            nk::World world(std::move(model), environments, backend.device, backend.backend, Config());
            ASSERT_TRUE(world.Ready());
            const auto required = world.GetModel().capacities.pair_sort_scratch_bytes;
            ASSERT_GT(required, environments);
            uint64_t available = 0u;
            for (const auto& segment : world.GetData().Segments()) {
                if (segment.field == nk::FieldId::PairSortScratch) available = segment.bytes;
            }
            ASSERT_GE(available, required) << "GPU sorting must stay inside its allocated segment";
            for (uint32_t step = 0; step < 3u; ++step) {
                ASSERT_TRUE(world.Step().AllOk());
                const auto inertia = Read<nuka::math::SymmetricMat3>(
                    world, nk::FieldId::BodyWorldInvInertia, environments * 2u);
                for (uint32_t env = 0; env < environments; ++env) {
                    const auto& fixed = inertia[env * 2u];
                    EXPECT_FLOAT_EQ(fixed.xx, 0.0f);
                    EXPECT_FLOAT_EQ(fixed.yy, 0.0f);
                    EXPECT_FLOAT_EQ(fixed.zz, 0.0f);
                    const auto& value = inertia[env * 2u + 1u];
                    EXPECT_NEAR(value.xx, 100.0f, 1.0e-4f);
                    EXPECT_NEAR(value.yy, 100.0f, 1.0e-4f);
                    EXPECT_NEAR(value.zz, 100.0f, 1.0e-4f);
                    EXPECT_NEAR(value.xy, 0.0f, 1.0e-4f);
                    EXPECT_NEAR(value.xz, 0.0f, 1.0e-4f);
                    EXPECT_NEAR(value.yz, 0.0f, 1.0e-4f);
                }
            }
        }
    }
}
