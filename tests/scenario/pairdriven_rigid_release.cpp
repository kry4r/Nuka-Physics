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

// Position sweeps converge to mass-weighted separation without external forces,
// independently of body order; the production default must meet the final bound.
TEST(PairDrivenRigidRelease, SeparationPreservesCenterOfMass) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    for (bool reverse : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "reverse=" << reverse);
        double previous_error = 1.0;
        const uint16_t final_iterations = Config().pos_iters;
        for (uint16_t iterations : std::vector<uint16_t>{4u, 8u, 16u, final_iterations}) {
            SCOPED_TRACE(::testing::Message() << "position iterations=" << iterations);
            nk::Model model;
            AddBody(model, nuka::collision::kShapeSphere,
                    {reverse ? 0.09f : -0.09f, 0, 0}, 0.1f, reverse ? 0.3f : 0.2f);
            AddBody(model, nuka::collision::kShapeSphere,
                    {reverse ? -0.09f : 0.09f, 0, 0}, 0.1f, reverse ? 0.2f : 0.3f);
            Finish(model);
            auto config = Config();
            config.pos_iters = iterations;
            nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
            ASSERT_TRUE(world.Ready());
            for (uint32_t step = 0; step < 20u; ++step) ASSERT_TRUE(world.Step().AllOk());
            const auto poses = Read<Transform>(world, nk::FieldId::BodyPose, 4u);
            const auto velocities = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 4u);
            double error = 0.0;
            for (uint32_t env = 0; env < 2u; ++env) {
                const uint32_t left = env * 2u + (reverse ? 1u : 0u);
                const uint32_t right = env * 2u + (reverse ? 0u : 1u);
                const float center = (0.2f * poses[left].position.x + 0.3f * poses[right].position.x) / 0.5f;
                error = std::fmax(error, std::abs(double(center) - 0.018));
                if (iterations == final_iterations) EXPECT_NEAR(center, 0.018f, 2.0e-6f) << "env=" << env;
                EXPECT_NEAR(0.2f * velocities[left].x + 0.3f * velocities[right].x, 0.0f, 2.0e-6f);
                EXPECT_GT(poses[right].position.x - poses[left].position.x, 0.199f);
            }
            RecordProperty("center_error_pm_" + std::to_string(reverse) + "_" + std::to_string(iterations),
                           std::to_string(std::llround(error * 1.0e12)));
            EXPECT_LE(error, std::fmax(0.65 * previous_error, 2.0e-7));
            previous_error = error;
        }
    }
}

TEST(PairDrivenRigidRelease, ContactReadoutMatchesSolvedRowsAcrossEnvironments) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    nk::Model model;
    AddBody(model, nuka::collision::kShapePlane, {0, 0, 0}, 0.0f, 0.0f);
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
    const auto ranges = Read<uint32_t>(world, nk::FieldId::UcontactRowRange, 128u);
    const auto point_impulses = Read<Vec3>(world, nk::FieldId::UcontactImpulse, 256u);
    const auto normals = Read<Vec3>(world, nk::FieldId::UcontactNormal, 256u);
    const auto endpoints_a = Read<uint32_t>(world, nk::FieldId::UcontactA, 256u);
    const auto endpoints_b = Read<uint32_t>(world, nk::FieldId::UcontactB, 256u);
    uint32_t active = 0u;
    for (uint32_t slot = 0; slot < counts.size(); ++slot) {
        EXPECT_EQ(links[slot], ~0u) << "rigid-only contact has no articulation link";
        for (uint32_t point = counts[slot]; point < nk::kPairDrivenPtsPerSlot; ++point)
            EXPECT_FLOAT_EQ(point_impulses[slot * nk::kPairDrivenPtsPerSlot + point].LengthSq(), 0.0f);
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
        const uint32_t base = ranges[slot * 2u];
        const uint32_t stride = ranges[slot * 2u + 1u];
        EXPECT_EQ(stride, counts[slot]);
        float normal = 0, tangent1 = 0, tangent2 = 0;
        Vec3 box_impulse{};
        for (uint32_t point = 0; point < counts[slot]; ++point) {
            const uint32_t index = slot * nk::kPairDrivenPtsPerSlot + point;
            EXPECT_EQ(endpoints_a[index], indices_a[slot] - body_base);
            EXPECT_EQ(endpoints_b[index], indices_b[slot] - body_base);
            EXPECT_NEAR(point_impulses[index].Dot(normals[index]), impulses[base + point], 1.0e-6f);
            box_impulse += point_impulses[index] * (endpoints_a[index] == 1u ? 1.0f : -1.0f);
            normal += impulses[base + point];
            tangent1 += impulses[base + stride + point];
            tangent2 += impulses[base + 2u * stride + point];
        }
        EXPECT_GT(normal, 1.0e-6f);
        EXPECT_NEAR(forces[slot].x, normal / config.dt, 1.0e-4f) << "slot=" << slot;
        EXPECT_NEAR(forces[slot].y, tangent1 / config.dt, 1.0e-4f);
        EXPECT_NEAR(forces[slot].z, tangent2 / config.dt, 1.0e-4f);
        EXPECT_FLOAT_EQ(points[slot].x, manifold[slot * 4u].x);
        EXPECT_FLOAT_EQ(points[slot].z, manifold[slot * 4u].z);
        EXPECT_LT(box_impulse.x, 0.0f);
        EXPECT_LT(box_impulse.y, 0.0f);
        EXPECT_GT(box_impulse.z, 0.0f);
    }
    EXPECT_EQ(active, 2u);
}

// Acceleration-level contact damping uses the incoming velocity, including under-resolved inputs.
TEST(PairDrivenRigidRelease, MuJoCoDampingMatchesAccelerationReferenceAcrossMasses) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    constexpr float approach = -0.2f;
    constexpr float dt = 0.002f;
    constexpr float impedance = 0.95f;
    for (float damping : {0.0f, 100.0f, 10000.0f}) {
        for (float mass : {0.2f, 2.0f}) {
            for (float gravity : {0.0f, -9.81f}) {
                for (uint32_t condim : {1u, 3u}) {
                    SCOPED_TRACE(::testing::Message() << "condim=" << condim << " damping=" << damping
                        << " mass=" << mass);
                    nk::Model model;
                    AddBody(model, nuka::collision::kShapePlane, {0, 0, 0}, 0.0f, 0.0f);
                    AddBody(model, nuka::collision::kShapeSphere,
                            {0, 0, 0.0995f}, 0.1f, mass);
                    model.body_init[1].linear_velocity = {0, 0, approach};
                    nk::ContactProfileV1 profile;
                    profile.law_source = nuka::constraint::LawSource::MuJoCo;
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
                    const float expected = approach * (1.0f - impedance * damping * dt) +
                        (1.0f - impedance) * gravity * dt;
                    for (uint32_t env = 0; env < 2u; ++env) {
                        EXPECT_NEAR(velocity[env * 2u + 1u].z, expected, 3.0e-6f);
                    }
                }
            }
        }
    }
}

// Resolved time steps converge to dv/dt = -impedance*damping*v + (1-impedance)*gravity.
TEST(PairDrivenRigidRelease, ResolvedContactDampingConvergesToAnalyticDecay) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    constexpr float approach = -0.2f;
    constexpr float impedance = 0.95f;
    for (uint32_t condim : {1u, 3u}) {
        for (float damping : {100.0f, 10000.0f}) {
            for (float mass : {0.2f, 2.0f}) {
                for (float gravity : {0.0f, -9.81f}) {
                    SCOPED_TRACE(::testing::Message() << "condim=" << condim << " damping=" << damping
                        << " mass=" << mass << " gravity=" << gravity);
                    const float duration = 1.0f / damping;
                    const float decay = std::exp(-impedance * damping * duration);
                    const float expected = approach * decay +
                        (1.0f - impedance) * gravity / (impedance * damping) * (1.0f - decay);
                    float previous_error = 1.0f;
                    for (uint32_t steps : {16u, 32u, 64u}) {
                        const float dt = duration / static_cast<float>(steps);
                        nk::Model model;
                        AddBody(model, nuka::collision::kShapePlane, {0, 0, 0}, 0.0f, 0.0f);
                        AddBody(model, nuka::collision::kShapeSphere,
                                {0, 0, 0.0995f}, 0.1f, mass);
                        model.body_init[1].linear_velocity = {0, 0, approach};
                        nk::ContactProfileV1 profile;
                        profile.law_source = nuka::constraint::LawSource::MuJoCo;
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
                        for (uint32_t step = 0u; step < steps; ++step) ASSERT_TRUE(world.Step().AllOk());
                        const auto velocity = Read<Vec3>(world, nk::FieldId::BodyLinearVelocity, 4u);
                        float error = 0.0f;
                        for (uint32_t env = 0; env < 2u; ++env) {
                            const float actual = velocity[env * 2u + 1u].z;
                            EXPECT_GE(actual, approach);
                            EXPECT_LE(actual, 0.0f);
                            error = std::fmax(error, std::abs(actual - expected));
                        }
                        EXPECT_LT(error, 0.65f * previous_error);
                        previous_error = error;
                    }
                    EXPECT_LT(previous_error, 1.0e-3f);
                }
            }
        }
    }
}
