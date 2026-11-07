#include <gtest/gtest.h>

#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "import/mjcf_importer.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/readout/energy_ledger.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"

namespace {
namespace nk = nuka::nk;
namespace phi = nuka::phi;

struct Backend {
    phi::Device* device = phi::InitBestDevice();
    phi::Backend* backend = device ? phi::DeviceInitBackend(device, nullptr) : nullptr;
};
Backend& Device() {
    static Backend backend;
    return backend;
}

void CheckDryFrictionResponse(bool coupled, bool compliant) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    const auto fixture = std::filesystem::path(NUKA_SOURCE_DIR) /
        "tests/data/osc_passive_damping.xml";
    for (uint32_t drive_mode : {1u, 4u}) {
        for (uint32_t fold_damping : {0u, 1u}) {
            for (float speed : {0.05f, 0.0001f}) {
                SCOPED_TRACE(::testing::Message() << "coupled=" << coupled << " compliant=" << compliant
                    << " mode=" << drive_mode
                    << " fold=" << fold_damping << " speed=" << speed);
                auto scene = nuka::import::LoadMjcf(fixture.string());
                auto& joint = scene.GetJointMut(scene.Joints().back().id);
                joint.damping = 0.0f;
                joint.frictionloss = 1.0f;
                joint.law_source = compliant ? nuka::constraint::LawSource::MuJoCo
                                            : nuka::constraint::LawSource::None;
                joint.friction_reference = {};
                if (coupled) joint.axis = {1.0f, 0.0f, 0.0f};
                nuka::scene::cook::CookToModelOptions options;
                options.enable_contacts = false;
                auto model = nuka::scene::cook::CookToModel(scene, 1, options).model;
                ASSERT_EQ(model.capacities.max_contacts_per_env, 0u);
                ASSERT_FLOAT_EQ(model.articulation.joint_frictionloss[2], 1.0f);
                model.drive_mode = drive_mode;
                model.osc_task_link = 1u;
                nk::Pipeline::SolverConfig config;
                config.dt = 0.002f;
                config.fold_drive_damping = fold_damping;
                config.gravity[0] = config.gravity[1] = config.gravity[2] = 0.0f;
                nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
                ASSERT_TRUE(world.Ready());
                auto& data = world.GetData();
                const std::vector<float> zero(6, 0.0f);
                const std::vector<float> velocity{0.0f, 0.0f, speed, 0.0f, 0.0f, -speed};
                ASSERT_TRUE(data.UploadField(nk::FieldId::Qdot, velocity.data(), velocity.size() * sizeof(float)));
                for (auto field : {nk::FieldId::DriveStiffness, nk::FieldId::DriveDamping, nk::FieldId::DriveTarget}) {
                    ASSERT_TRUE(data.UploadField(field, zero.data(), zero.size() * sizeof(float)));
                }
                ASSERT_TRUE(world.Step().AllOk());
                std::vector<float> after(6);
                ASSERT_TRUE(data.DownloadField(nk::FieldId::Qdot, after.data(), after.size() * sizeof(float)));
                const float inverse_diagonal = coupled ? 1.1f / 1.2f : 1.0f / 1.1f;
                const float inverse_cross = coupled ? -0.1f / 1.2f : 0.0f;
                for (uint32_t env = 0; env < 2u; ++env) {
                    const uint32_t hand = env * 3u + 1u;
                    const uint32_t finger = hand + 1u;
                    // The imported law uses a damped acceleration reference and positive impedance compliance.
                    const float stopping_impulse = compliant
                        ? config.dt * (2.0f / (0.95f * 0.02f)) * speed / (inverse_diagonal / 0.9f)
                        : speed / inverse_diagonal;
                    const float impulse = -std::copysign(std::min(config.dt, stopping_impulse), velocity[finger]);
                    EXPECT_NEAR(after[finger], velocity[finger] + inverse_diagonal * impulse, 2.0e-6f);
                    EXPECT_NEAR(after[hand], inverse_cross * impulse, 2.0e-6f);
                }
            }
        }
    }
}
}

// OSC compensation must preserve passive damping on an unpowered child joint.
// Orthogonal sliders give diagonal inertia and an analytic implicit damping step.
TEST(OscPassiveDamping, CompensationPreservesUnpoweredGripperDamping) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    const auto fixture = std::filesystem::path(NUKA_SOURCE_DIR) /
        "tests/data/osc_passive_damping.xml";
    const auto scene = nuka::import::LoadMjcf(fixture.string());
    nuka::scene::cook::CookToModelOptions options;
    options.enable_contacts = false;
    auto model = nuka::scene::cook::CookToModel(scene, 1, options).model;
    ASSERT_EQ(model.articulation.link_count, 3u);
    ASSERT_EQ(model.articulation.dof_count, 2u);
    ASSERT_FLOAT_EQ(model.articulation.joint_damping[2], 100.0f);
    ASSERT_FLOAT_EQ(model.articulation.joint_armature[2], 1.0f);
    model.drive_mode = 4u;  // OSC
    model.osc_task_link = 1u;
    nk::Pipeline::SolverConfig config;
    config.dt = 0.002f;
    config.gravity[0] = config.gravity[1] = config.gravity[2] = 0.0f;
    nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
    ASSERT_TRUE(world.Ready());
    auto& data = world.GetData();
    const std::vector<float> zero(6, 0.0f);
    const std::vector<float> velocity{0.0f, 0.0f, 0.05f, 0.0f, 0.0f, -0.03f};
    ASSERT_TRUE(data.UploadField(nk::FieldId::Qdot, velocity.data(), velocity.size() * sizeof(float)));
    ASSERT_TRUE(data.UploadField(nk::FieldId::DriveStiffness, zero.data(), zero.size() * sizeof(float)));
    ASSERT_TRUE(data.UploadField(nk::FieldId::DriveDamping, zero.data(), zero.size() * sizeof(float)));
    ASSERT_TRUE(world.Step().AllOk());
    std::vector<float> after(6), effort(6);
    ASSERT_TRUE(data.DownloadField(nk::FieldId::Qdot, after.data(), after.size() * sizeof(float)));
    ASSERT_TRUE(data.DownloadField(nk::FieldId::ActuatorEffort, effort.data(), effort.size() * sizeof(float)));
    for (uint32_t env = 0; env < 2u; ++env) {
        const auto finger = env * 3u + 2u;
        const float expected = velocity[finger] / (1.0f + config.dt * 100.0f / 1.1f);
        EXPECT_NEAR(effort[finger], 0.0f, 2.0e-5f);
        EXPECT_NEAR(after[finger], expected, 2.0e-6f);
    }
}

TEST(OscPassiveDamping, GravityCompensationPreservesBodyWeightsAndJointForceLimits) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    const auto fixture = std::filesystem::path(NUKA_SOURCE_DIR) / "tests/data/gravity_compensation.xml";
    const auto imported = nuka::import::LoadMjcf(fixture.string());
    ASSERT_EQ(imported.Joints().size(), 2u);
    EXPECT_TRUE(imported.Joints().front().actuator_gravity_compensation);
    const auto saved = std::filesystem::temp_directory_path() / "nuka_gravity_compensation.nks";
    nuka::scene::nks::Save(imported, saved.string());
    const auto restored = nuka::scene::nks::Load(saved.string());
    for (uint32_t body = 0u; body < imported.Bodies().size(); ++body)
        EXPECT_FLOAT_EQ(imported.GetBody(body).gravity_compensation, restored.GetBody(body).gravity_compensation);
    EXPECT_TRUE(restored.Joints().front().actuator_gravity_compensation);
    std::filesystem::remove(saved);
    std::filesystem::remove(saved.parent_path() / (saved.stem().string() + ".nka"));
    for (bool actuated : {false, true}) {
      for (float joint_limit : {0.0f, 10.0f}) {
        SCOPED_TRACE(::testing::Message() << "actuated=" << actuated << " joint_limit=" << joint_limit);
        auto source = restored;
        source.GetJointMut(source.Joints().front().id).actuator_gravity_compensation = actuated;
        nuka::scene::cook::CookToModelOptions options;
        options.enable_contacts = false;
        auto model = nuka::scene::cook::CookToModel(source, 1u, options).model;
        const uint32_t links = model.capacities.links_per_env;
        ASSERT_EQ(links, 3u);
        model.drive_mode = 1u;
        model.hold_drives.targets.assign(links, 0.0f);
        model.hold_drives.targets[1] = 4.0f;
        model.hold_drives.stiffness.assign(links, 0.0f);
        model.hold_drives.damping.assign(links, 0.0f);
        model.hold_drives.force_limits.assign(links, 3.0f);
        model.hold_drives.joint_force_limits.assign(links, joint_limit);
        model.articulation.initial_qdot.assign(links, 0.0f);
        model.articulation.initial_qdot[1] = 0.1f;
        nk::Pipeline::SolverConfig config;
        config.dt = 0.001f;
        config.pos_iters = 0u;
        config.gravity[0] = config.gravity[1] = 0.0f;
        config.gravity[2] = -9.81f;
        nk::World world(std::move(model), 2u, backend.device, backend.backend, config);
        ASSERT_TRUE(world.Ready());
        ASSERT_NE(world.FieldPtr(nk::FieldId::EnergyLedger), nullptr);
        const float compensation = 19.62f;
        const float requested = 3.0f + (actuated ? compensation : 0.0f);
        const float effort = joint_limit > 0.0f ? std::min(requested, joint_limit) : requested;
        const float speed = 0.1f + config.dt *
            (-29.43f + effort + (actuated ? 0.0f : compensation)) / 3.0f;
        for (bool planned : {false, true}) {
            if (planned) ASSERT_EQ(world.Reset({0u, 1u}), phi::Status::Ok);
            std::vector<float> command(links * 2u, 0.0f);
            command[1] = command[links + 1u] = 4.0f;
            ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::DriveTarget, command.data(), command.size() * sizeof(float)));
            ASSERT_EQ(planned ? world.StepPlanned() : world.Step().result, phi::Status::Ok);
            std::vector<float> velocity(links * 2u), applied(links * 2u), projected(links * 2u);
            auto& data = world.GetData();
            ASSERT_TRUE(data.DownloadField(nk::FieldId::Qdot, velocity.data(), velocity.size() * sizeof(float)));
            ASSERT_TRUE(data.DownloadField(nk::FieldId::ActuatorEffort, applied.data(), applied.size() * sizeof(float)));
            ASSERT_TRUE(data.DownloadField(nk::FieldId::JointGravityForce, projected.data(), projected.size() * sizeof(float)));
            std::vector<float> ledger(2u * nk::kEnergyColumnCount);
            ASSERT_TRUE(data.DownloadField(nk::FieldId::EnergyLedger, ledger.data(), ledger.size() * sizeof(float)));
            for (uint32_t env = 0u; env < 2u; ++env) {
                EXPECT_NEAR(velocity[env * links + 1u], speed, 2.0e-6f);
                EXPECT_NEAR(applied[env * links + 1u], effort, 2.0e-5f);
                EXPECT_NEAR(projected[env * links + 1u], actuated ? compensation : 0.0f, 2.0e-5f);
                const auto value = [&](nk::EnergyColumn c) { return ledger[env * nk::kEnergyColumnCount + uint32_t(c)]; };
                EXPECT_NEAR(value(nk::EnergyColumn::ExternalWork),
                    actuated ? 0.0f : 0.5f * compensation * (0.1f + speed) * config.dt, 2.0e-7f);
                EXPECT_NEAR(value(nk::EnergyColumn::DriveWork), effort * speed * config.dt, 2.0e-7f);
            }
        }
      }
    }
}

TEST(JointDryFriction, DecayAndStickingWithoutContactsInBothControlModes) {
    CheckDryFrictionResponse(false, false);
    CheckDryFrictionResponse(false, true);
}

TEST(JointDryFriction, ImpulsePropagatesThroughTheCoupledMassMatrix) {
    CheckDryFrictionResponse(true, false);
    CheckDryFrictionResponse(true, true);
}

namespace {
nuka::scene::SceneIR SliderControlScene(bool floating, uint32_t joints = 1u,
                                        uint32_t articulations = 1u, bool rotational = false) {
    namespace scene = nuka::scene;
    scene::SceneIR result;
    for (uint32_t tree = 0u; tree < articulations; ++tree) {
        scene::RigidBodyRecord base;
        base.name = "base_" + std::to_string(tree);
        base.mass = 3.0f;
        base.inertia = {1.0f, 1.0f, 1.0f};
        base.is_static = !floating;
        base.local_transform.position = {float(tree), 0.0f, 0.0f};
        auto parent = result.AddRigidBody(base);
        for (uint32_t index = 0u; index < joints; ++index) {
            scene::RigidBodyRecord body;
            body.name = "slider_" + std::to_string(tree) + "_" + std::to_string(index);
            body.parent_id = parent;
            body.mass = 2.0f;
            body.inertia = {0.2f, 0.2f, 0.2f};
            const auto child = result.AddRigidBody(body);
            scene::JointRecord joint;
            joint.name = "joint_" + body.name;
            joint.parent_body = parent;
            joint.child_body = child;
            joint.type = rotational ? scene::JointType::Revolute : scene::JointType::Prismatic;
            joint.axis = index % 3u == 0u ? nuka::math::Vec3{1, 0, 0}
                : index % 3u == 1u ? nuka::math::Vec3{0, 1, 0} : nuka::math::Vec3{0, 0, 1};
            joint.armature = 0.4f;
            joint.damping = joints == 1u ? 0.7f : 0.0f;
            result.AddJoint(joint);
            parent = child;
        }
    }
    return result;
}

template <class T>
std::vector<T> ControlField(nk::World& world, nk::FieldId field) {
    const auto bytes = world.GetModel().capacities.ElementCount(field) * nk::LayoutOf(field).elem_size;
    std::vector<T> values(bytes / sizeof(T));
    EXPECT_TRUE(world.GetData().DownloadField(field, values.data(), bytes));
    return values;
}

void WriteControl(nk::World& world, nk::FieldId field, const std::vector<float>& values) {
    ASSERT_TRUE(world.GetData().UploadField(field, values.data(), values.size() * sizeof(float)));
}
}

TEST(ControlModes, BoundedEffortMatchesAnalyticFixedAndFloatingMassResponse) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    nuka::scene::cook::CookToModelOptions options;
    options.enable_contacts = false;
    for (bool rotational : {false, true}) for (bool floating : {false, true}) {
        for (uint32_t mode = 0u; mode < 6u; ++mode) {
            for (float limit : {0.0f, 0.25f}) {
                SCOPED_TRACE(::testing::Message() << "rotational=" << rotational << " floating=" << floating
                             << " mode=" << mode << " limit=" << limit);
                auto model = nuka::scene::cook::CookToModel(SliderControlScene(floating, 1u, 1u, rotational), 1u, options).model;
                model.drive_mode = mode;
                model.osc_task_link = 1u;
                nk::Pipeline::SolverConfig cfg;
                cfg.dt = 0.01f;
                cfg.gravity[0] = cfg.gravity[1] = cfg.gravity[2] = 0.0f;
                nk::World world(std::move(model), 2u, backend.device, backend.backend, cfg);
                ASSERT_TRUE(world.Ready()) << world.CreationError();
                ASSERT_EQ(world.GetModel().capacities.links_per_env, 2u);
                WriteControl(world, nk::FieldId::Q, {0, 0.1f, 0, 0.1f});
                WriteControl(world, nk::FieldId::Qdot, {0, 0.2f, 0, -0.2f});
                WriteControl(world, nk::FieldId::DriveTarget, {0, mode == 1u || mode == 5u ? 5.0f : 0.3f,
                                                            0, mode == 1u || mode == 5u ? -5.0f : 0.3f});
                WriteControl(world, nk::FieldId::VelocityTarget, {0, -0.1f, 0, -0.1f});
                WriteControl(world, nk::FieldId::AccelerationTarget, {0, 0.8f, 0, 0.8f});
                WriteControl(world, nk::FieldId::DriveStiffness, {0, 20, 0, 20});
                WriteControl(world, nk::FieldId::DriveDamping, {0, 4, 0, 4});
                WriteControl(world, nk::FieldId::DriveForceLimit, {0, limit, 0, limit});
                WriteControl(world, nk::FieldId::ActuatorNoloadSpeed, {0, 0.3f, 0, 0.3f});
                WriteControl(world, nk::FieldId::JointF, {0, 0.25f, 0, 0.25f});
                WriteControl(world, nk::FieldId::TaskTarget, {0.3f, 0, 0, 0.3f, 0, 0});
                if (rotational) {
                    const float c = std::cos(0.175f), s = std::sin(0.175f);
                    WriteControl(world, nk::FieldId::TaskRotationTarget, {c, s, 0, 0, -c, -s, 0, 0});
                    const float lc = std::cos(0.025f), ls = std::sin(0.025f);
                    WriteControl(world, nk::FieldId::TaskLocalPose, {0.02f, 0, 0, lc, ls, 0, 0,
                                                                    0.02f, 0, 0, lc, ls, 0, 0});
                }
                ASSERT_TRUE(world.Step().AllOk());
                const auto speed = ControlField<float>(world, nk::FieldId::Qdot);
                const auto effort = ControlField<float>(world, nk::FieldId::ActuatorEffort);
                const auto requested = ControlField<float>(world, nk::FieldId::ActuatorEffortRequested);
                const auto saturated = ControlField<float>(world, nk::FieldId::ActuatorSaturated);
                const auto velocity = ControlField<float>(world, nk::FieldId::LinkVelocity);
                EXPECT_EQ(ControlField<uint32_t>(world, nk::FieldId::EnvStatus), std::vector<uint32_t>(2u));
                const float base_inertia = rotational ? 1.0f : 3.0f;
                const float tip_inertia = rotational ? 0.2f : 2.0f;
                const float mass = 0.4f + (floating
                    ? base_inertia * tip_inertia / (base_inertia + tip_inertia) : tip_inertia);
                const float task_response = (floating ? base_inertia / (base_inertia + tip_inertia) : 1.0f) / mass;
                for (uint32_t env = 0u; env < 2u; ++env) {
                    const uint32_t link = env * 2u + 1u;
                    const float initial = env == 0u ? 0.2f : -0.2f;
                    const float free_speed = initial + cfg.dt * (0.25f - 0.7f * initial) / mass;
                    float expected_effort = 0.0f;
                    if (mode == 0u) {
                        const float impedance = 4.0f + cfg.dt * 20.0f;
                        expected_effort = (4.0f - impedance * (0.1f + free_speed)) /
                                          (1.0f + cfg.dt * impedance / mass);
                    }
                    if (mode == 1u || mode == 5u) expected_effort = env == 0u ? 5.0f : -5.0f;
                    if (mode == 2u) expected_effort = (-2.0f - 20.0f * free_speed) / (1.0f + cfg.dt * 20.0f / mass);
                    if (mode == 3u) expected_effort = mass * (0.8f + 4.0f + 4.0f * (-0.1f - initial));
                    if (mode == 4u) expected_effort = (4.0f - 4.0f * initial) / task_response;
                    float lower = -limit, upper = limit;
                    if (mode == 5u && limit > 0.0f) {
                        lower = std::clamp(-limit * (1.0f + initial / 0.3f), -limit, limit);
                        upper = std::clamp(limit * (1.0f - initial / 0.3f), -limit, limit);
                    }
                    if (limit > 0.0f) expected_effort = std::clamp(expected_effort, lower, upper);
                    const float expected_speed = free_speed + cfg.dt * expected_effort / mass;
                    EXPECT_NEAR(effort[link], expected_effort, 8.0e-5f);
                    EXPECT_NEAR(speed[link], expected_speed, 3.0e-6f);
                    EXPECT_EQ(saturated[link] != 0.0f, limit > 0.0f &&
                              (requested[link] < lower || requested[link] > upper));
                    if (floating) {
                        const float root_speed = velocity[env * 12u + (rotational ? 0u : 3u)];
                        EXPECT_NEAR((base_inertia + tip_inertia) * root_speed + tip_inertia * speed[link],
                                    tip_inertia * initial, 2.0e-5f);
                    }
                }
            }
        }
    }
}

TEST(ControlModes, ImplicitDampingEffortAndWorkMatchAnalyticImpulse) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    nuka::scene::cook::CookToModelOptions options;
    options.enable_contacts = false;
    for (bool derivative : {false, true}) for (float limit : {0.0f, 0.25f}) {
      for (float joint_limit : {0.0f, 0.15f}) {
        SCOPED_TRACE(::testing::Message() << "derivative=" << derivative << " limit=" << limit
                     << " joint_limit=" << joint_limit);
        auto source = SliderControlScene(false, 1u, 1u, false);
        source.SolverOptionsMut().law_source = nuka::constraint::LawSource::MuJoCo;
        source.SolverOptionsMut().integrator = derivative ? nuka::constraint::Integrator::ImplicitFast
                                                         : nuka::constraint::Integrator::Euler;
        auto model = nuka::scene::cook::CookToModel(source, 1u, options).model;
        model.drive_mode = 0u;
        model.hold_drives.joint_force_limits = {0.0f, joint_limit};
        nk::Pipeline::SolverConfig cfg;
        cfg.dt = 0.01f;
        cfg.gravity[0] = cfg.gravity[1] = cfg.gravity[2] = 0.0f;
        nk::World world(std::move(model), 2u, backend.device, backend.backend, cfg);
        ASSERT_TRUE(world.Ready()) << world.CreationError();
        ASSERT_NE(world.FieldPtr(nk::FieldId::EnergyLedger), nullptr);
        for (bool planned : {false, true}) {
            if (planned) ASSERT_EQ(world.Reset({0u, 1u}), phi::Status::Ok);
            WriteControl(world, nk::FieldId::Q, {0, 0.1f, 0, 0.1f});
            WriteControl(world, nk::FieldId::Qdot, {0, 0.2f, 0, -0.2f});
            WriteControl(world, nk::FieldId::DriveTarget, {0, 0.3f, 0, 0.3f});
            WriteControl(world, nk::FieldId::VelocityTarget, {0, -0.1f, 0, -0.1f});
            WriteControl(world, nk::FieldId::DriveStiffness, {0, 20, 0, 20});
            WriteControl(world, nk::FieldId::DriveDamping, {0, 4, 0, 4});
            WriteControl(world, nk::FieldId::DriveForceLimit, {0, limit, 0, limit});
            WriteControl(world, nk::FieldId::JointF, {0, 0.25f, 0, 0.25f});
            ASSERT_EQ(planned ? world.StepPlanned() : world.Step().result, phi::Status::Ok);
            const auto speed = ControlField<float>(world, nk::FieldId::Qdot);
            const auto effort = ControlField<float>(world, nk::FieldId::ActuatorEffort);
            const auto requested = ControlField<float>(world, nk::FieldId::ActuatorEffortRequested);
            const auto ledger = ControlField<float>(world, nk::FieldId::EnergyLedger);
            for (uint32_t env = 0u; env < 2u; ++env) {
                const uint32_t link = env * 2u + 1u;
                const double initial = env == 0u ? 0.2 : -0.2;
                const double command = 4.0 + 4.0 * (-0.1 - initial);
                const double gain = derivative && (limit == 0.0f || std::abs(command) <= limit) ? 4.0 : 0.0;
                double applied = limit > 0.0f ? std::clamp(command, -double(limit), double(limit)) : command;
                if (joint_limit > 0.0f) applied = std::clamp(applied, -double(joint_limit), double(joint_limit));
                const double end = (2.4 * initial + cfg.dt * (0.25 + applied + gain * initial)) /
                                   (2.4 + cfg.dt * (0.7 + gain));
                const double correction = gain * (end - initial);
                EXPECT_NEAR(speed[link], end, 2.0e-6);
                EXPECT_NEAR(effort[link], applied - correction, 3.0e-5);
                EXPECT_NEAR(requested[link], command - correction, 3.0e-5);
                const auto value = [&](nk::EnergyColumn column) { return ledger[env * nk::kEnergyColumnCount + uint32_t(column)]; };
                EXPECT_NEAR(value(nk::EnergyColumn::DriveWork), (applied - correction) * end * cfg.dt, 2.0e-7);
                EXPECT_NEAR(value(nk::EnergyColumn::ExternalWork), 0.25 * end * cfg.dt, 2.0e-7);
                EXPECT_NEAR(value(nk::EnergyColumn::PassiveLoss), 0.7 * end * end * cfg.dt, 2.0e-7);
            }
        }
      }
    }
}

TEST(ControlModes, MotorEnvelopePreservesBrakingInEveryQuadrant) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    nuka::scene::cook::CookToModelOptions options;
    options.enable_contacts = false;
    auto model = nuka::scene::cook::CookToModel(SliderControlScene(false), 1u, options).model;
    model.drive_mode = 5u;
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 0.002f;
    cfg.gravity[0] = cfg.gravity[1] = cfg.gravity[2] = 0.0f;
    nk::World world(std::move(model), 1u, backend.device, backend.backend, cfg);
    ASSERT_TRUE(world.Ready()) << world.CreationError();
    WriteControl(world, nk::FieldId::DriveForceLimit, {0, 2});
    WriteControl(world, nk::FieldId::ActuatorNoloadSpeed, {0, 1});
    for (float speed : {-1.5f, -0.5f, 0.5f, 1.5f}) {
        for (float command : {-10.0f, 0.0f, 10.0f}) {
            SCOPED_TRACE(::testing::Message() << "speed=" << speed << " command=" << command);
            ASSERT_EQ(world.Reset(), phi::Status::Ok);
            WriteControl(world, nk::FieldId::Qdot, {0, speed});
            WriteControl(world, nk::FieldId::DriveTarget, {0, command});
            ASSERT_TRUE(world.Step().AllOk());
            const float lower = std::clamp(-2.0f * (1.0f + speed), -2.0f, 2.0f);
            const float upper = std::clamp(2.0f * (1.0f - speed), -2.0f, 2.0f);
            EXPECT_NEAR(ControlField<float>(world, nk::FieldId::ActuatorEffort)[1],
                        std::clamp(command, lower, upper), 2.0e-6f);
        }
    }
}

TEST(ControlModes, OscMultipleWideArticulationsGraphAndMaskedReset) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    nuka::scene::cook::CookToModelOptions options;
    options.enable_contacts = false;
    auto make_model = [&]() {
        auto model = nuka::scene::cook::CookToModel(SliderControlScene(true, 20u, 2u), 2u, options).model;
        model.drive_mode = 4u;
        model.osc_task_link = 20u;
        return model;
    };
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 0.001f;
    cfg.gravity[0] = cfg.gravity[1] = cfg.gravity[2] = 0.0f;
    nk::World eager(make_model(), 2u, backend.device, backend.backend, cfg);
    nk::World graph(make_model(), 2u, backend.device, backend.backend, cfg);
    ASSERT_TRUE(eager.Ready()) << eager.CreationError();
    ASSERT_TRUE(graph.Ready()) << graph.CreationError();
    ASSERT_GT(graph.GetModel().capacities.dofs_per_env, 18u);
    ASSERT_EQ(graph.GetModel().capacities.articulations_per_env, 2u);
    const auto initial = ControlField<float>(graph, nk::FieldId::Q);
    for (auto* world : {&eager, &graph}) {
        WriteControl(*world, nk::FieldId::DriveStiffness, std::vector<float>(84u, 20.0f));
        WriteControl(*world, nk::FieldId::DriveDamping, std::vector<float>(84u, 4.0f));
        WriteControl(*world, nk::FieldId::TaskTarget, {0.001f, 0, 0, 1.002f, 0, 0,
                                                     0.003f, 0, 0, 1.004f, 0, 0});
        WriteControl(*world, nk::FieldId::TaskRotationTarget, std::vector<float>(16u, 0.0f));
    }
    const auto targets = ControlField<float>(graph, nk::FieldId::TaskTarget);
    ASSERT_EQ(targets.size(), 12u);
    ASSERT_EQ(graph.SetExecutionMode(nk::World::ExecutionMode::Graph), phi::Status::Ok);
    for (uint32_t step = 0u; step < 8u; ++step) {
        for (auto* world : {&eager, &graph}) {
            ASSERT_EQ(world->StepConfigured(), phi::Status::Ok);
            EXPECT_EQ(ControlField<uint32_t>(*world, nk::FieldId::EnvStatus), std::vector<uint32_t>(2u));
        }
        EXPECT_EQ(ControlField<float>(eager, nk::FieldId::Q), ControlField<float>(graph, nk::FieldId::Q));
        EXPECT_EQ(ControlField<float>(eager, nk::FieldId::BasePose), ControlField<float>(graph, nk::FieldId::BasePose));
    }
    const auto before = ControlField<float>(graph, nk::FieldId::Q);
    ASSERT_EQ(graph.Reset({1u}), phi::Status::Ok);
    const auto after = ControlField<float>(graph, nk::FieldId::Q);
    EXPECT_TRUE(std::equal(before.begin(), before.begin() + 42u, after.begin()));
    EXPECT_TRUE(std::equal(initial.begin() + 42u, initial.end(), after.begin() + 42u));
    EXPECT_EQ(ControlField<float>(graph, nk::FieldId::TaskTarget), targets);
    const auto efforts = ControlField<float>(graph, nk::FieldId::ActuatorEffort);
    EXPECT_TRUE(std::all_of(efforts.begin() + 42u, efforts.end(), [](float value) { return value == 0.0f; }));
    auto invalid = make_model();
    invalid.osc_task_link = 21u;
    nk::World rejected(std::move(invalid), 2u, backend.device, backend.backend, cfg);
    EXPECT_FALSE(rejected.Ready());
    EXPECT_EQ(rejected.CreationStatus(), phi::Status::InvalidArgument);
}
