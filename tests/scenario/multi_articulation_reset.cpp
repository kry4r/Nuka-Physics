#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "import/usd_importer.hpp"
#include "import/mjcf_importer.hpp"
#include "math/symmetric_mat3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"

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
    static Backend result;
    return result;
}

nk::Model Model(uint32_t articulations = 2u) {
    const auto path = std::filesystem::path(NUKA_SOURCE_DIR) / "examples/scenes/go2_float.usda";
    auto scene = nuka::import::LoadUsd(path.string());
    for (uint32_t i = 1u; i < articulations; ++i) {
        auto placement = Transform::Identity();
        placement.position.x = 1.5f * i;
        scene = nuka::scene::Compose(scene, nuka::import::LoadUsd(path.string()),
                                     placement, "robot" + std::to_string(i) + "_");
    }
    nuka::scene::cook::CookToModelOptions options;
    return nuka::scene::cook::CookToModel(scene, 1, options).model;
}

nk::Model ArticulatedModelWithoutColliders() {
    const auto path = std::filesystem::path(NUKA_SOURCE_DIR) / "tests/data/osc_passive_damping.xml";
    auto placement = Transform::Identity();
    placement.position.x = 1.5f;
    auto scene = nuka::scene::Compose(nuka::import::LoadMjcf(path.string()),
        nuka::import::LoadMjcf(path.string()), placement, "second_");
    nuka::scene::cook::CookToModelOptions options;
    options.enable_contacts = false;
    auto model = nuka::scene::cook::CookToModel(scene, 1, options).model;
    model.articulation.initial_qdot.assign(model.capacities.links_per_env, 0.02f);
    return model;
}

nk::Model ParticleModel() {
    nk::Model model;
    auto& particles = model.particles;
    particles.mode = nk::Model::ParticleMode::Mpm;
    for (uint32_t i = 0u; i < 8u; ++i) {
        particles.initial_pos.push_back({0.25f + 0.05f * (i & 1u),
            0.25f + 0.05f * ((i >> 1u) & 1u), 0.25f + 0.05f * (i >> 2u)});
        particles.initial_vel.push_back({0.03f * i, -0.02f * i, 0.01f * i});
    }
    const auto count = particles.initial_pos.size();
    particles.inv_mass.assign(count, 1.0f);
    particles.initial_vol0.assign(count, 0.001f);
    particles.initial_material_id.assign(count, 0u);
    particles.mpm_grid_dims[0] = particles.mpm_grid_dims[1] = particles.mpm_grid_dims[2] = 6u;
    particles.mpm_cell_size = 0.1f;
    nk::MpmMaterial material;
    material.youngs = 1.0e4f;
    material.poisson = 0.3f;
    material.density = 1000.0f;
    model.mpm_materials = {material};
    model.capacities.articulations_per_env = 0u;
    model.capacities.particles_per_env = static_cast<uint32_t>(count);
    model.capacities.mpm_grid_nodes_per_env = 6u * 6u * 6u;
    model.capacities.mpm_material_count = 1u;
    return model;
}

nk::Model FreeBodyModel() {
    nuka::scene::SceneIR scene;
    nuka::scene::RigidBodyRecord body;
    body.name = "body";
    body.mass = 0.5f;
    body.inertia = {0.003f, 0.007f, 0.011f};
    body.local_transform.position = {0.1f, 0.2f, 2.0f};
    body.local_transform.rotation = nuka::math::Quat::FromAxisAngle({2, -1, 3}, 0.8f);
    body.inertial_transform.position = {0.02f, -0.01f, 0.03f};
    body.inertial_transform.rotation = nuka::math::Quat::FromAxisAngle({1, 2, 3}, 0.63f);
    nuka::scene::CollisionShapeRecord shape;
    shape.body_id = scene.AddRigidBody(body);
    shape.type = nuka::scene::ShapeType::Sphere;
    shape.radius = 0.1f;
    scene.AddCollisionShape(shape);
    shape.radius = 0.06f;
    shape.local_transform.position = {0.1f, 0.03f, 0.02f};
    scene.AddCollisionShape(shape);
    auto model = nuka::scene::cook::CookToModel(scene, 1, {}).model;
    model.capacities.articulations_per_env = 0u;
    for (auto& initial : model.body_init) {
        if (initial.inv_mass <= 0.0f) continue;
        initial.linear_velocity = {0.1f, -0.2f, 0.05f};
        initial.angular_velocity = {0.5f, 0.3f, -0.4f};
    }
    return model;
}

nk::Pipeline::SolverConfig Config() {
    nk::Pipeline::SolverConfig result;
    result.dt = 0.002f;
    return result;
}

std::vector<uint8_t> Read(nk::World& world, nk::FieldId field) {
    for (const auto& segment : world.GetData().Segments()) {
        if (segment.field != field) continue;
        std::vector<uint8_t> result(segment.bytes);
        if (!result.empty())
            EXPECT_TRUE(world.GetData().DownloadField(field, result.data(), result.size()));
        return result;
    }
    ADD_FAILURE() << "missing field " << static_cast<uint32_t>(field);
    return {};
}

void RefreshPoses(nk::World& world) {
    for (const auto& call : world.GetPipeline().Calls()) {
        if (call.op == phi::NkOp::FkWorldPoses || call.op == phi::NkOp::SyncLinkBodyPose)
            ASSERT_EQ(world.DispatchOp(call.op, call.params), phi::Status::Ok);
    }
}

const std::vector<nk::FieldId> kStateFields{
    nk::FieldId::BasePose, nk::FieldId::Q, nk::FieldId::Qdot,
    nk::FieldId::LinkVelocity, nk::FieldId::LinkPose, nk::FieldId::BodyPose,
    nk::FieldId::BodyLinearVelocity, nk::FieldId::BodyAngularVelocity,
    nk::FieldId::DriveTarget};

using State = std::vector<std::vector<uint8_t>>;

const std::vector<nk::FieldId> kParticleStateFields{
    nk::FieldId::ParticlePos, nk::FieldId::ParticlePrevPos, nk::FieldId::ParticleVel,
    nk::FieldId::ParticleF, nk::FieldId::ParticleC, nk::FieldId::ParticlePlastic};

const std::vector<nk::FieldId> kParticleSnapshotFields{
    nk::FieldId::SnapshotParticlePos, nk::FieldId::SnapshotParticlePrevPos,
    nk::FieldId::SnapshotParticleVel, nk::FieldId::SnapshotParticleF,
    nk::FieldId::SnapshotParticleC, nk::FieldId::SnapshotParticlePlastic};

State ReadState(nk::World& world, const std::vector<nk::FieldId>& fields = kStateFields) {
    State result;
    for (auto field : fields) result.push_back(Read(world, field));
    return result;
}

void ExpectSelectedState(nk::World& world, const State& initial, const State& before,
                         const std::vector<uint32_t>& selected,
                         const std::vector<nk::FieldId>& fields = kStateFields) {
    const auto after = ReadState(world, fields);
    for (size_t f = 0u; f < fields.size(); ++f) {
        SCOPED_TRACE(static_cast<uint32_t>(fields[f]));
        ASSERT_EQ(after[f].size(), before[f].size());
        const size_t stride = after[f].size() / world.EnvCount();
        if (stride == 0u) continue;
        for (uint32_t env = 0u; env < world.EnvCount(); ++env) {
            const bool reset = selected.empty() ||
                std::find(selected.begin(), selected.end(), env) != selected.end();
            const auto& expected = reset ? initial[f] : before[f];
            EXPECT_EQ(std::memcmp(after[f].data() + env * stride,
                                  expected.data() + env * stride, stride), 0)
                << "env=" << env << " selected=" << reset;
        }
    }
}

void Step(nk::World& world, uint32_t count, bool graph = false) {
    for (uint32_t step = 0; step < count; ++step) {
        if (graph) ASSERT_EQ(world.StepPlanned(), phi::Status::Ok);
        else ASSERT_TRUE(world.Step().AllOk());
    }
}

void FillParticleState(nk::World& world, float bias) {
    for (size_t field = 0; field < kParticleStateFields.size(); ++field) {
        const auto bytes = Read(world, kParticleStateFields[field]).size();
        ASSERT_GT(bytes, 0u);
        ASSERT_EQ(bytes % sizeof(float), 0u);
        std::vector<float> values(bytes / sizeof(float));
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = bias + 0.25f * field + 0.03125f * (i + 1u);
        ASSERT_TRUE(world.GetData().UploadField(kParticleStateFields[field], values.data(), bytes));
    }
}
}  // namespace

TEST(MultiArticulationReset, SnapshotContainsEveryRoot) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    nk::World world(Model(), 3u, device.device, device.backend, Config());
    ASSERT_TRUE(world.Ready());
    ASSERT_EQ(world.GetModel().capacities.articulations_per_env, 2u);
    EXPECT_EQ(Read(world, nk::FieldId::BasePose), Read(world, nk::FieldId::SnapshotBasePose));
}

TEST(MultiArticulationReset, MaskedResetPreservesOtherEnvironments) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    nk::World world(Model(), 3u, device.device, device.backend, Config());
    ASSERT_TRUE(world.Ready());
    RefreshPoses(world);
    const auto initial = ReadState(world);
    for (const auto& selection : std::vector<std::vector<uint32_t>>{
             {0u}, {1u}, {2u}, {2u, 0u, 2u, 1u}, {}}) {
        Step(world, 4u);
        const auto before = ReadState(world);
        ASSERT_NE(initial[0], before[0]);
        ASSERT_EQ(world.Reset(selection), phi::Status::Ok);
        ExpectSelectedState(world, initial, before, selection);
    }
}

TEST(MultiArticulationReset, DuplicateIdsDoNotTruncateTheSelection) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    nk::World world(Model(), 3u, device.device, device.backend, Config());
    ASSERT_TRUE(world.Ready());
    RefreshPoses(world);
    const auto initial = ReadState(world);
    Step(world, 4u);
    const auto before = ReadState(world);
    ASSERT_EQ(world.Reset({2u, 2u, 2u, 0u}), phi::Status::Ok);
    ExpectSelectedState(world, initial, before, {0u, 2u});
}

TEST(MultiArticulationReset, InvalidIdLeavesTheWorldUnchanged) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    nk::World world(Model(), 3u, device.device, device.backend, Config());
    ASSERT_TRUE(world.Ready());
    Step(world, 4u);
    const auto before = ReadState(world);
    ASSERT_EQ(world.Reset({0u, 3u}), phi::Status::Failed);
    EXPECT_EQ(before, ReadState(world));
}

TEST(MultiArticulationReset, FirstStepAfterResetMatchesANewWorld) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    for (const auto& selection : std::vector<std::vector<uint32_t>>{{}, {2u, 0u, 1u}}) {
        nk::World world(Model(), 3u, device.device, device.backend, Config());
        nk::World fresh(Model(), 3u, device.device, device.backend, Config());
        ASSERT_TRUE(world.Ready());
        ASSERT_TRUE(fresh.Ready());
        Step(world, 8u);
        ASSERT_EQ(world.Reset(selection), phi::Status::Ok);
        Step(world, 1u);
        Step(fresh, 1u);
        const auto expected = ReadState(fresh);
        ExpectSelectedState(world, expected, expected, {});
    }
}

TEST(MultiArticulationReset, CapturedContactStepsRemainValidAfterReset) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    nk::World world(Model(), 3u, device.device, device.backend, Config());
    nk::World fresh(Model(), 3u, device.device, device.backend, Config());
    ASSERT_TRUE(world.Ready());
    ASSERT_TRUE(fresh.Ready());
    const auto capture = world.StepPlanned();
    if (capture == phi::Status::Unsupported)
        GTEST_SKIP() << "contact pipeline cannot be captured; LBVH workspace is a separate requirement";
    ASSERT_EQ(capture, phi::Status::Ok);
    Step(world, 3u, true);
    ASSERT_EQ(world.Reset({0u, 1u, 2u}), phi::Status::Ok);
    Step(world, 1u, true);
    Step(fresh, 1u);
    const auto expected = ReadState(fresh);
    ExpectSelectedState(world, expected, expected, {});
}

TEST(MultiArticulationReset, CapturedDynamicsStepsRemainValidAfterReset) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    nk::World world(ArticulatedModelWithoutColliders(), 3u, device.device, device.backend, Config());
    nk::World fresh(ArticulatedModelWithoutColliders(), 3u, device.device, device.backend, Config());
    ASSERT_TRUE(world.Ready());
    ASSERT_TRUE(fresh.Ready());
    const auto initial_q = Read(world, nk::FieldId::Q);
    ASSERT_EQ(world.StepPlanned(), phi::Status::Ok);
    Step(world, 3u, true);
    ASSERT_NE(initial_q, Read(world, nk::FieldId::Q));
    ASSERT_EQ(world.Reset({0u, 1u, 2u}), phi::Status::Ok);
    Step(world, 1u, true);
    Step(fresh, 1u);
    const auto expected = ReadState(fresh);
    ExpectSelectedState(world, expected, expected, {});
}

TEST(MultiArticulationReset, ClearsContactHistoryAndReadoutOnlyForSelectedEnvironments) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    nk::World world(Model(), 3u, device.device, device.backend, Config());
    ASSERT_TRUE(world.Ready());
    const std::vector<nk::FieldId> zero_fields{
        nk::FieldId::Lambda, nk::FieldId::ContactCachePair, nk::FieldId::ContactCacheLambda,
        nk::FieldId::ContactCacheFeature, nk::FieldId::ContactCacheMaterial, nk::FieldId::ContactCacheAge,
        nk::FieldId::ContactCount, nk::FieldId::UcontactCount, nk::FieldId::EnvStatus,
        nk::FieldId::ContactForce, nk::FieldId::ContactPoint, nk::FieldId::ContactNormal,
        nk::FieldId::LinkContactWrench, nk::FieldId::QdotPseudo, nk::FieldId::QdotPseudoFlat,
        nk::FieldId::LinkVelocityPseudo, nk::FieldId::BodyPseudoLinearVelocity,
        nk::FieldId::BodyPseudoAngularVelocity, nk::FieldId::RowPseudoLambda};
    for (auto field : zero_fields) {
        auto value = Read(world, field);
        std::fill(value.begin(), value.end(), 0x35u);
        ASSERT_TRUE(world.GetData().UploadField(field, value.data(), value.size()));
    }
    auto targets = Read(world, nk::FieldId::DriveTarget);
    std::fill(targets.begin(), targets.end(), 0x35u);
    ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::DriveTarget, targets.data(), targets.size()));
    ASSERT_EQ(world.Reset({1u}), phi::Status::Ok);
    EXPECT_EQ(targets, Read(world, nk::FieldId::DriveTarget));
    for (auto field : zero_fields) {
        const auto value = Read(world, field);
        const auto stride = value.size() / world.EnvCount();
        for (uint32_t env = 0u; env < world.EnvCount(); ++env) {
            EXPECT_TRUE(std::all_of(value.begin() + env * stride, value.begin() + (env + 1u) * stride,
                [env](uint8_t byte) { return byte == (env == 1u ? 0u : 0x35u); }))
                << "field=" << static_cast<uint32_t>(field) << " env=" << env;
        }
    }
    const auto owners = Read(world, nk::FieldId::ContactSideAIndex);
    const auto stride = owners.size() / world.EnvCount();
    EXPECT_TRUE(std::all_of(owners.begin() + stride, owners.begin() + 2u * stride,
                            [](uint8_t byte) { return byte == 0xffu; }));
}

TEST(MultiArticulationReset, ParticleStateRoundTripsWithoutArticulations) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    nk::World world(ParticleModel(), 3u, device.device, device.backend, Config());
    ASSERT_TRUE(world.Ready());
    ASSERT_EQ(world.GetModel().capacities.articulations_per_env, 0u);
    ASSERT_EQ(ReadState(world, kParticleStateFields), ReadState(world, kParticleSnapshotFields));

    FillParticleState(world, 1.0f);
    phi::SnapshotStateParams snapshot{};
    snapshot.env_count = world.EnvCount();
    snapshot.total_particle_count = world.GetModel().capacities.particles_per_env * world.EnvCount();
    ASSERT_EQ(world.DispatchOp(phi::NkOp::SnapshotState, &snapshot), phi::Status::Ok);
    const auto initial = ReadState(world, kParticleStateFields);
    ASSERT_EQ(initial, ReadState(world, kParticleSnapshotFields));
    const std::vector<nk::FieldId> transient_fields{nk::FieldId::ParticlePseudoVel, nk::FieldId::EnvStatus};
    const auto cleared = ReadState(world, transient_fields);

    for (const auto& selection : std::vector<std::vector<uint32_t>>{
             {0u}, {1u}, {2u}, {2u, 0u, 2u}, {}}) {
        FillParticleState(world, 11.0f);
        for (auto field : transient_fields) {
            auto bytes = Read(world, field);
            std::fill(bytes.begin(), bytes.end(), 0x35u);
            ASSERT_TRUE(world.GetData().UploadField(field, bytes.data(), bytes.size()));
        }
        const auto before = ReadState(world, kParticleStateFields);
        const auto transient_before = ReadState(world, transient_fields);
        for (size_t field = 0; field < before.size(); ++field) ASSERT_NE(before[field], initial[field]);
        ASSERT_EQ(world.Reset(selection), phi::Status::Ok);
        ExpectSelectedState(world, initial, before, selection, kParticleStateFields);
        ExpectSelectedState(world, cleared, transient_before, selection, transient_fields);
        EXPECT_EQ(initial, ReadState(world, kParticleSnapshotFields));
    }
}

TEST(MultiArticulationReset, ParticleStepAfterResetMatchesANewWorld) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    for (const auto& selection : std::vector<std::vector<uint32_t>>{{1u}, {}}) {
        nk::World world(ParticleModel(), 3u, device.device, device.backend, Config());
        nk::World fresh(ParticleModel(), 3u, device.device, device.backend, Config());
        nk::World uninterrupted(ParticleModel(), 3u, device.device, device.backend, Config());
        ASSERT_TRUE(world.Ready());
        ASSERT_TRUE(fresh.Ready());
        ASSERT_TRUE(uninterrupted.Ready());
        const auto initial = ReadState(world, kParticleStateFields);
        Step(world, 6u);
        ASSERT_NE(initial[0], Read(world, nk::FieldId::ParticlePos));
        ASSERT_EQ(world.Reset(selection), phi::Status::Ok);
        Step(world, 1u);
        Step(fresh, 1u);
        Step(uninterrupted, 7u);
        ExpectSelectedState(world, ReadState(fresh, kParticleStateFields),
            ReadState(uninterrupted, kParticleStateFields), selection, kParticleStateFields);
    }
}

TEST(MultiArticulationReset, FreeBodyInertiaAndProxiesRestoreOnlySelectedEnvironments) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    nk::World world(FreeBodyModel(), 3u, device.device, device.backend, Config());
    ASSERT_TRUE(world.Ready());
    ASSERT_EQ(world.GetModel().capacities.articulations_per_env, 0u);
    const auto& bodies = world.GetModel().body_init;
    ASSERT_EQ(bodies.size(), world.GetModel().capacities.bodies_per_env);
    ASSERT_EQ(std::count_if(bodies.begin(), bodies.end(),
                           [](const auto& body) { return body.inv_mass > 0.0f; }), 1);
    const auto& proxy_owners = world.GetModel().body_collidable_body;
    ASSERT_EQ(std::count_if(proxy_owners.begin(), proxy_owners.end(),
                           [](uint32_t owner) { return owner != ~0u; }), 2);
    RefreshPoses(world);
    const auto initial = ReadState(world);
    const auto initial_poses = Read(world, nk::FieldId::BodyPose);
    for (const auto& selection : std::vector<std::vector<uint32_t>>{{1u}, {2u, 0u}, {}}) {
        Step(world, 7u);
        const auto before = ReadState(world);
        const auto inertia_before = Read(world, nk::FieldId::BodyWorldInvInertia);
        ASSERT_NE(initial_poses, Read(world, nk::FieldId::BodyPose));
        ASSERT_EQ(world.Reset(selection), phi::Status::Ok);
        ExpectSelectedState(world, initial, before, selection);
        const auto inertia_after = Read(world, nk::FieldId::BodyWorldInvInertia);
        const size_t stride = inertia_after.size() / world.EnvCount();
        for (uint32_t env = 0u; env < world.EnvCount(); ++env) {
            const bool reset = selection.empty() ||
                std::find(selection.begin(), selection.end(), env) != selection.end();
            if (!reset) {
                EXPECT_EQ(std::memcmp(inertia_before.data() + env * stride,
                                      inertia_after.data() + env * stride, stride), 0);
                continue;
            }
            for (size_t body = 0; body < bodies.size(); ++body) {
                nuka::math::SymmetricMat3 actual;
                std::memcpy(&actual, inertia_after.data() + env * stride + body * sizeof(actual), sizeof(actual));
                const auto rotation = (bodies[body].pose.rotation * bodies[body].inertial_frame.rotation).Normalized();
                const auto inverse = bodies[body].inv_inertia;
                for (Vec3 axis : {Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}}) {
                    const auto local = rotation.Conjugate().Rotate(axis);
                    const auto expected = rotation.Rotate({local.x * inverse.x, local.y * inverse.y, local.z * inverse.z});
                    const auto result = actual.Multiply(axis);
                    EXPECT_NEAR(result.x, expected.x, 5.0e-4f);
                    EXPECT_NEAR(result.y, expected.y, 5.0e-4f);
                    EXPECT_NEAR(result.z, expected.z, 5.0e-4f);
                }
            }
        }
    }
}

TEST(MultiArticulationReset, ReportsResetCostAndMemory) {
    auto& device = Device();
    if (!device.backend) GTEST_SKIP() << "no CUDA backend";
    for (uint32_t envs : {1u, 3u, 16u}) {
        for (uint32_t articulations : {1u, 2u}) {
            nk::World world(Model(articulations), envs, device.device, device.backend, Config());
            ASSERT_TRUE(world.Ready());
            const std::string prefix = "E" + std::to_string(envs) + "K" + std::to_string(articulations);
            const auto expected_roots = Read(world, nk::FieldId::BasePose);
            uint64_t bytes[3]{};
            nk::Arena::ComputeSegments(world.GetModel().capacities, bytes);
            for (size_t arena = 0; arena < 3u; ++arena)
                RecordProperty(prefix + "_arena" + std::to_string(arena) + "_bytes", std::to_string(bytes[arena]));
            for (bool masked : {false, true}) {
                const std::vector<uint32_t> selection = masked ? std::vector<uint32_t>{envs - 1u}
                                                               : std::vector<uint32_t>{};
                for (uint32_t i = 0u; i < 10u; ++i) ASSERT_EQ(world.Reset(selection), phi::Status::Ok);
                phi::BackendSynchronize(device.backend);
                const auto start = std::chrono::steady_clock::now();
                constexpr uint32_t count = 200u;
                for (uint32_t i = 0u; i < count; ++i) ASSERT_EQ(world.Reset(selection), phi::Status::Ok);
                phi::BackendSynchronize(device.backend);
                const double us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count() / count;
                RecordProperty(prefix + (masked ? "_masked_us" : "_full_us"), std::to_string(us));
            }
            const bool valid = Read(world, nk::FieldId::BasePose) == expected_roots;
            RecordProperty(prefix + "_roots_valid", valid ? "true" : "false");
            EXPECT_TRUE(valid) << prefix;
        }
    }
}
