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
#include "scene/cook/cook_to_model.hpp"

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

void CheckDryFrictionResponse(bool coupled) {
    auto& backend = Device();
    if (!backend.backend) GTEST_SKIP() << "no CUDA backend";
    const auto fixture = std::filesystem::path(NUKA_SOURCE_DIR) /
        "tests/data/osc_passive_damping.xml";
    for (bool force_static : {false, true}) {
        IslandMode schedule(force_static);
        for (uint32_t drive_mode : {1u, 4u}) {
            for (uint32_t fold_damping : {0u, 1u}) {
                for (float speed : {0.05f, 0.0001f}) {
                    SCOPED_TRACE(::testing::Message() << "coupled=" << coupled
                        << " static=" << force_static << " mode=" << drive_mode
                        << " fold=" << fold_damping << " speed=" << speed);
                    auto scene = nuka::import::LoadMjcf(fixture.string());
                    auto& joint = scene.GetJointMut(scene.Joints().back().id);
                    joint.damping = 0.0f;
                    joint.frictionloss = 1.0f;
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
                        const float impulse = -std::copysign(
                            std::min(config.dt, speed / inverse_diagonal), velocity[finger]);
                        EXPECT_NEAR(after[finger], velocity[finger] + inverse_diagonal * impulse, 2.0e-6f);
                        EXPECT_NEAR(after[hand], inverse_cross * impulse, 2.0e-6f);
                    }
                }
            }
        }
    }
}
}

// OSC compensation must preserve passive damping on an unpowered child joint.
// Orthogonal sliders give diagonal inertia and an analytic one-step decay.
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
        const float expected = velocity[finger] * (1.0f - config.dt * 100.0f / 1.1f);
        EXPECT_NEAR(effort[finger], 0.0f, 2.0e-5f);
        EXPECT_NEAR(after[finger], expected, 2.0e-6f);
    }
}

TEST(JointDryFriction, DecayAndStickingWithoutContactsInBothControlModes) {
    CheckDryFrictionResponse(false);
}

TEST(JointDryFriction, ImpulsePropagatesThroughTheCoupledMassMatrix) {
    CheckDryFrictionResponse(true);
}
