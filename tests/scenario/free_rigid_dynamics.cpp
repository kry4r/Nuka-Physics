#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "scene/cook/cook_to_model.hpp"
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
