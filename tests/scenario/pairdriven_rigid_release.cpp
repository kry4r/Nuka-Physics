#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "collision/shape_kind.hpp"
#include "math/transform.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"

namespace {
namespace nk = nuka::nk;
namespace phi = nuka::phi;
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

void AddBody(nk::Model& model, uint32_t kind, Vec3 position, float size, float mass) {
    nk::Model::BodyInit body;
    body.pose = Transform::Identity();
    body.pose.position = position;
    body.inv_mass = mass > 0.0f ? 1.0f / mass : 0.0f;
    const float ii = mass > 0.0f ? 1.0f / (0.4f * mass * size * size) : 0.0f;
    body.inv_inertia = {ii, ii, ii};
    nk::Model::PairDrivenShape shape;
    shape.kind = kind;
    shape.params[0] = shape.params[1] = shape.params[2] = size;
    shape.contype = shape.conaffinity = 1u;
    shape.sdf_grid = ~0u;
    shape.body_id = static_cast<int32_t>(model.body_init.size());
    model.shape_table_rows.push_back(shape);
    model.body_init.push_back(body);
}

void Finish(nk::Model& model) {
    auto& capacity = model.capacities;
    capacity.env_count = 1u;
    capacity.bodies_per_env = static_cast<uint32_t>(model.body_init.size());
    capacity.max_bodies_total = capacity.bodies_per_env;
    capacity.max_contacts_per_env = 32u;
    capacity.max_rows_per_env = 32u * nk::kPairDrivenRowsPerSlot;
    model.contact_family = nk::ContactFamily::PairDriven;
    model.filter_cross_env = true;
}

nk::Pipeline::SolverConfig Config() {
    nk::Pipeline::SolverConfig config;
    config.dt = 0.005f;
    config.gravity[0] = config.gravity[1] = config.gravity[2] = 0.0f;
    config.max_pairs = 32u;
    config.pos_iters = 4u;
    return config;
}

template <typename T>
std::vector<T> Read(nk::World& world, nk::FieldId field, size_t count) {
    std::vector<T> result(count);
    EXPECT_TRUE(world.GetData().DownloadField(field, result.data(), count * sizeof(T)));
    return result;
}
}  // namespace

// Multiple simultaneous four-point manifolds exercise cross-slot writes, including
// contacts beyond the first environment and inactive candidate slots.
TEST(PairDrivenRigidRelease, ManifoldPointsStayOnTheContactingGeometry) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    nk::Model model;
    AddBody(model, nuka::collision::kShapePlane, {0, 0, 0}, 0.0f, 0.0f);
    model.body_init[0].pose.rotation = nuka::math::Quat::FromAxisAngle(
        Vec3{1, 0, 0}, 1.57079632679f);
    for (uint32_t i = 0; i < 6u; ++i)
        AddBody(model, nuka::collision::kShapeBox,
                {static_cast<float>(i), 0, 0.195f}, 0.2f, 0.2f);
    Finish(model);
    nk::World world(std::move(model), 2u, backend.device, backend.backend, Config());
    ASSERT_TRUE(world.Ready());
    for (uint32_t step = 0; step < 4u; ++step) {
        ASSERT_TRUE(world.Step().AllOk());
        const auto counts = Read<uint32_t>(world, nk::FieldId::UcontactCount, 64u);
        const auto points = Read<Vec3>(world, nk::FieldId::UcontactPoint, 256u);
        uint32_t active = 0u;
        for (uint32_t slot = 0; slot < counts.size(); ++slot) {
            for (uint32_t point = 0; point < counts[slot]; ++point) {
                ++active;
                const Vec3 p = points[slot * 4u + point];
                EXPECT_LT(std::abs(p.z), 0.02f) << "slot=" << slot << " point=" << point;
                EXPECT_LT(std::abs(p.y), 0.21f) << "slot=" << slot << " point=" << point;
                EXPECT_GE(p.x, -0.21f);
                EXPECT_LE(p.x, 5.21f);
            }
        }
        EXPECT_GT(active, 0u);
    }
}

// Physical impulses and geometric push-out preserve the center of mass without
// external forces, for either body order and both island schedules.
TEST(PairDrivenRigidRelease, SeparationPreservesCenterOfMassForBothIslandSchedules) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (bool force_static : {false, true}) {
        IslandMode mode(force_static);
        for (bool reverse : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "static=" << force_static << " reverse=" << reverse);
            nk::Model model;
            AddBody(model, nuka::collision::kShapeSphere,
                    {reverse ? 0.09f : -0.09f, 0, 0}, 0.1f, reverse ? 0.3f : 0.2f);
            AddBody(model, nuka::collision::kShapeSphere,
                    {reverse ? -0.09f : 0.09f, 0, 0}, 0.1f, reverse ? 0.2f : 0.3f);
            Finish(model);
            nk::World world(std::move(model), 2u, backend.device, backend.backend, Config());
            ASSERT_TRUE(world.Ready());
            for (uint32_t step = 0; step < 20u; ++step) ASSERT_TRUE(world.Step().AllOk());
            const auto poses = Read<Transform>(world, nk::FieldId::BodyPose, 4u);
            const auto velocities = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 4u);
            for (uint32_t env = 0; env < 2u; ++env) {
                const uint32_t left = env * 2u + (reverse ? 1u : 0u);
                const uint32_t right = env * 2u + (reverse ? 0u : 1u);
                const float center = (0.2f * poses[left].position.x + 0.3f * poses[right].position.x) / 0.5f;
                EXPECT_NEAR(center, 0.018f, 2.0e-6f) << "env=" << env;
                EXPECT_NEAR(0.2f * velocities[left].x + 0.3f * velocities[right].x,
                            0.0f, 2.0e-6f);
                EXPECT_GT(poses[right].position.x - poses[left].position.x, 0.199f);
            }
        }
    }
}

TEST(PairDrivenRigidRelease, ContactReadoutMatchesSolvedRowsAcrossEnvironments) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    nk::Model model;
    AddBody(model, nuka::collision::kShapePlane, {0, 0, 0}, 0.0f, 0.0f);
    model.body_init[0].pose.rotation = nuka::math::Quat::FromAxisAngle(
        Vec3{1, 0, 0}, 1.57079632679f);
    AddBody(model, nuka::collision::kShapeBox, {1, 0, 0.195f}, 0.2f, 0.2f);
    model.body_init[1].linear_velocity = {0.1f, 0.05f, 0};
    Finish(model);
    // Row padding must not be mistaken for the next environment's contacts.
    model.capacities.max_rows_per_env += 16u;
    const uint32_t row_stride = model.capacities.max_rows_per_env;
    auto config = Config();
    config.gravity[2] = -9.81f;
    nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
    ASSERT_TRUE(world.Ready());
    ASSERT_TRUE(world.Step().AllOk());
    // Requesting geometry alone must enable its producer, including rigid-only worlds.
    ASSERT_NE(world.FieldPtr(nk::FieldId::ContactPoint), nullptr);
    ASSERT_TRUE(world.Step().AllOk());
    const auto counts = Read<uint32_t>(world, nk::FieldId::UcontactCount, 64u);
    const auto points = Read<Vec3>(world, nk::FieldId::ContactPoint, 64u);
    const auto manifold = Read<Vec3>(world, nk::FieldId::UcontactPoint, 256u);
    const auto forces = Read<Vec3>(world, nk::FieldId::ContactForce, 64u);
    const auto links = Read<uint32_t>(world, nk::FieldId::ContactLink, 64u);
    const auto kinds_a = Read<uint32_t>(world, nk::FieldId::ContactSideAKind, 64u);
    const auto kinds_b = Read<uint32_t>(world, nk::FieldId::ContactSideBKind, 64u);
    const auto indices_a = Read<uint32_t>(world, nk::FieldId::ContactSideAIndex, 64u);
    const auto indices_b = Read<uint32_t>(world, nk::FieldId::ContactSideBIndex, 64u);
    const auto impulses = Read<float>(world, nk::FieldId::Lambda, row_stride * 2u);
    uint32_t active = 0u;
    for (uint32_t slot = 0; slot < counts.size(); ++slot) {
        EXPECT_EQ(links[slot], ~0u) << "rigid-only contact has no articulation link";
        if (counts[slot] == 0u) {
            EXPECT_EQ(kinds_a[slot], nk::kNkSideStatic);
            EXPECT_EQ(kinds_b[slot], nk::kNkSideStatic);
            EXPECT_EQ(indices_a[slot], ~0u);
            EXPECT_EQ(indices_b[slot], ~0u);
            continue;
        }
        const uint32_t body_base = (slot / 32u) * 2u;
        EXPECT_EQ(kinds_a[slot], nk::kNkSideRigid);
        EXPECT_EQ(kinds_b[slot], nk::kNkSideRigid);
        EXPECT_TRUE((indices_a[slot] == body_base && indices_b[slot] == body_base + 1u) ||
                    (indices_b[slot] == body_base && indices_a[slot] == body_base + 1u));
        ++active;
        const uint32_t base = (slot / 32u) * row_stride +
                              (slot % 32u) * nk::kPairDrivenRowsPerSlot;
        float normal = 0, tangent1 = 0, tangent2 = 0;
        for (uint32_t point = 0; point < counts[slot]; ++point) {
            normal += impulses[base + point];
            tangent1 += impulses[base + nk::kPairDrivenPtsPerSlot + point];
            tangent2 += impulses[base + 2u * nk::kPairDrivenPtsPerSlot + point];
        }
        EXPECT_GT(normal, 1.0e-6f);
        EXPECT_NEAR(forces[slot].x, normal / config.dt, 1.0e-4f) << "slot=" << slot;
        EXPECT_NEAR(forces[slot].y, tangent1 / config.dt, 1.0e-4f);
        EXPECT_NEAR(forces[slot].z, tangent2 / config.dt, 1.0e-4f);
        EXPECT_FLOAT_EQ(points[slot].x, manifold[slot * 4u].x);
        EXPECT_FLOAT_EQ(points[slot].z, manifold[slot * 4u].z);
    }
    EXPECT_EQ(active, 2u);
}

// Implicit damping must not reverse or amplify the incoming normal velocity.
// Scalar and block contacts must agree under either island schedule.
TEST(PairDrivenRigidRelease, ImplicitDampingRemainsStableForScalarAndBlockContacts) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    constexpr float approach = -0.2f;
    constexpr float dt = 0.002f;
    constexpr float impedance = 0.95f;
    for (float damping : {0.0f, 100.0f, 10000.0f}) {
        for (float mass : {0.2f, 2.0f}) {
            float reference = 0.0f;
            bool have_reference = false;
            for (bool force_static : {false, true}) {
                IslandMode mode(force_static);
                for (uint32_t condim : {1u, 3u}) {
                        SCOPED_TRACE(::testing::Message() << "static=" << force_static
                            << " condim=" << condim << " damping=" << damping
                            << " mass=" << mass);
                        nk::Model model;
                        AddBody(model, nuka::collision::kShapePlane, {0, 0, 0}, 0.0f, 0.0f);
                        model.body_init[0].pose.rotation = nuka::math::Quat::FromAxisAngle(
                            Vec3{1, 0, 0}, 1.57079632679f);
                        AddBody(model, nuka::collision::kShapeSphere,
                                {0, 0, 0.0995f}, 0.1f, mass);
                        model.body_init[1].linear_velocity = {0, 0, approach};
                        nk::ContactProfileV1 profile;
                        profile.condim = condim;
                        profile.solref[0] = 0.0f;
                        profile.solref[1] = -damping * impedance;
                        profile.solimp[0] = profile.solimp[1] = impedance;
                        model.material_buckets.emplace_back(profile);
                        Finish(model);
                        model.capacities.num_material_buckets = 1u;
                        auto config = Config();
                        config.dt = dt;
                        config.pos_iters = 0u;
                        nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
                        ASSERT_TRUE(world.Ready());
                        ASSERT_TRUE(world.Step().AllOk());
                        const auto velocity = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 4u);
                        for (uint32_t env = 0; env < 2u; ++env) {
                            const float actual = velocity[env * 2u + 1u].z;
                            EXPECT_GE(actual, approach - 3.0e-6f);
                            EXPECT_LE(actual, 3.0e-6f);
                            if (have_reference) EXPECT_NEAR(actual, reference, 3.0e-6f);
                            else {
                                reference = actual;
                                have_reference = true;
                            }
                        }
                }
            }
        }
    }
}

// Soft contact must retain incoming velocity and scale its impulse with mass.
// The analytic decay includes external gravity and both island schedules.
TEST(PairDrivenRigidRelease, CompliantDampingPreservesReferenceVelocityAcrossMasses) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    constexpr float approach = -0.2f;
    constexpr float dt = 0.002f;
    constexpr float impedance = 0.95f;
    for (bool force_static : {false, true}) {
        IslandMode mode(force_static);
        for (uint32_t condim : {1u, 3u}) {
            for (float damping : {0.0f, 100.0f, 10000.0f}) {
                for (float mass : {0.2f, 2.0f}) {
                    for (float gravity : {0.0f, -9.81f}) {
                        SCOPED_TRACE(::testing::Message() << "static=" << force_static
                            << " condim=" << condim << " damping=" << damping
                            << " mass=" << mass << " gravity=" << gravity);
                        nk::Model model;
                        AddBody(model, nuka::collision::kShapePlane, {0, 0, 0}, 0.0f, 0.0f);
                        model.body_init[0].pose.rotation = nuka::math::Quat::FromAxisAngle(
                            Vec3{1, 0, 0}, 1.57079632679f);
                        AddBody(model, nuka::collision::kShapeSphere,
                                {0, 0, 0.0995f}, 0.1f, mass);
                        model.body_init[1].linear_velocity = {0, 0, approach};
                        nk::ContactProfileV1 profile;
                        profile.condim = condim;
                        profile.solref[0] = 0.0f;
                        profile.solref[1] = -damping * impedance;
                        profile.solimp[0] = profile.solimp[1] = impedance;
                        model.material_buckets.emplace_back(profile);
                        Finish(model);
                        model.capacities.num_material_buckets = 1u;
                        auto config = Config();
                        config.dt = dt;
                        config.gravity[2] = gravity;
                        config.pos_iters = 0u;
                        nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
                        ASSERT_TRUE(world.Ready());
                        ASSERT_TRUE(world.Step().AllOk());
                        const auto velocity = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 4u);
                        const float expected =
                            (approach + (1.0f - impedance) * gravity * dt) /
                            (1.0f + impedance * damping * dt);
                        for (uint32_t env = 0; env < 2u; ++env) {
                            EXPECT_NEAR(velocity[env * 2u + 1u].z, expected, 3.0e-6f);
                        }
                    }
                }
            }
        }
    }
}
