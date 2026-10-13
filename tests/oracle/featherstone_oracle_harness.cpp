#include "oracle/golden_trajectory.hpp"

#include "import/mjcf_importer.hpp"
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/scoped_device_guard.hpp"
#include <cuda_runtime.h>
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/op_schema.hpp"
#include "runtime/articulation/control_mode.hpp"
#include "scene/cook/cook_to_model.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

std::filesystem::path MenageriePath(const char* relative_path) {
    if (const char* root = std::getenv("NUKA_MENAGERIE_ROOT"))
        return std::filesystem::path(root) / relative_path;
    return SourcePath(".nuka-assets/mujoco_menagerie") / relative_path;
}

std::filesystem::path GoldenPath(const char* filename) {
    if (const char* golden_dir = std::getenv("NUKA_GOLDEN_DIR")) {
        if (golden_dir[0] != '\0') {
            return std::filesystem::path(golden_dir) / filename;
        }
    }
    return SourcePath("tests/oracle/golden") / filename;
}

struct FloatingBaseAbaResult {
    std::vector<float> leg_qddot;     // Nuka per-leg-DOF accel (leg-major).
    std::array<float, 6> base_accel_mjx{};  // base accel in MJX [lin(world); ang(body)] convention.
    std::array<float, 6> base_accel_nuka_spatial{};  // raw Nuka [omega; v_lin] body-frame.
    nuka::math::Quat base_rotation{};
};

// v' = R(q) v, w-first Hamilton.
nuka::math::Vec3 RotateByQuat(const nuka::math::Quat& q, const nuka::math::Vec3& v) {
    const float n2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    nuka::math::Quat u = q;
    if (n2 > 1.0e-12f) {
        const float inv = 1.0f / std::sqrt(n2);
        u.w *= inv; u.x *= inv; u.y *= inv; u.z *= inv;
    }
    const nuka::math::Vec3 qv{u.x, u.y, u.z};
    const auto cross = [](const nuka::math::Vec3& a, const nuka::math::Vec3& b) {
        return nuka::math::Vec3{a.y * b.z - a.z * b.y,
                                a.z * b.x - a.x * b.z,
                                a.x * b.y - a.y * b.x};
    };
    const nuka::math::Vec3 t{2.0f * cross(qv, v).x, 2.0f * cross(qv, v).y,
                             2.0f * cross(qv, v).z};
    const nuka::math::Vec3 c = cross(qv, t);
    return nuka::math::Vec3{v.x + u.w * t.x + c.x,
                            v.y + u.w * t.y + c.y,
                            v.z + u.w * t.z + c.z};
}

// v_body = R(q)^T v_world.
nuka::math::Vec3 RotateByQuatInverse(const nuka::math::Quat& q, const nuka::math::Vec3& v) {
    nuka::math::Quat inv{q.w, -q.x, -q.y, -q.z};
    return RotateByQuat(inv, v);
}

} // namespace

TEST(FeatherstoneOracle, GoldenFilesAreOwnerProvided) {
    const auto go2 = GoldenPath("featherstone_go2_random_sample.bin");
    const auto h1 = GoldenPath("featherstone_h1_random_sample.bin");
    const auto stand = GoldenPath("go2_stand_5s.bin");
    if (!std::filesystem::exists(go2) ||
        !std::filesystem::exists(h1) ||
        !std::filesystem::exists(stand)) {
        GTEST_SKIP() << "v0.1 golden files are owner-protected and not present";
    }

    for (const auto& path : {go2, h1, stand}) {
        const auto golden = nuka::tests::oracle::LoadGoldenTrajectory(path);
        EXPECT_GT(golden.sample_count, 0u) << path;
        EXPECT_GT(golden.qpos_count, 0u) << path;
        if (golden.kind == nuka::tests::oracle::GoldenKind::RandomQacc) {
            EXPECT_GT(golden.qvel_count, 0u) << path;
            EXPECT_GT(golden.qacc_count, 0u) << path;
        }
        EXPECT_EQ(golden.payload.size(),
                  static_cast<size_t>(golden.sample_count) * golden.RecordFloatCount())
            << path;
    }
}

TEST(FeatherstoneOracle, RandomSampleGoldenShapeMatchesV1Contract) {
    const auto go2 = GoldenPath("featherstone_go2_random_sample.bin");
    const auto h1 = GoldenPath("featherstone_h1_random_sample.bin");
    if (!std::filesystem::exists(go2) || !std::filesystem::exists(h1)) {
        GTEST_SKIP() << "v0.1 random-sample golden files are owner-protected and not present";
    }

    for (const auto& path : {go2, h1}) {
        const auto golden = nuka::tests::oracle::LoadGoldenTrajectory(path);
        EXPECT_EQ(golden.kind, nuka::tests::oracle::GoldenKind::RandomQacc) << path;
        EXPECT_EQ(golden.sample_count, 1000u) << path;
        EXPECT_EQ(golden.qvel_count, golden.qacc_count) << path;
    }
}

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;

struct NkBackendFixture {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;

    NkBackendFixture() {
        dev = nphi::InitBestDevice();
        if (dev != nullptr) {
            backend = nphi::DeviceInitBackend(dev, nullptr);
        }
    }
    ~NkBackendFixture() {
        if (backend != nullptr) {
            nphi::BackendFree(backend);
        }
    }
};

nk::Model CookNkModelFixedBase(const std::filesystem::path& model_path) {
    auto scene = nuka::import::LoadMjcf(model_path.string());
    for (const auto& body : scene.Bodies())
        if (body.parent_id == nuka::scene::kInvalidBody) scene.GetBodyMut(body.id).is_static = true;
    return std::move(nuka::scene::cook::CookToModel(scene, 1u).model);
}

std::vector<float> DynamicsQddot(nk::World& world, const float* record,
                              const nuka::tests::oracle::GoldenTrajectory& golden,
                              nphi::Backend* backend) {
    const uint32_t link_count = world.GetModel().capacities.links_per_env;
    EXPECT_EQ(link_count, golden.qpos_count) << "cooked link count vs golden nq";
    const float* qpos = record;
    const float* qvel = qpos + golden.qpos_count;
    const float* tau = qvel + golden.qvel_count;

    const size_t bytes = static_cast<size_t>(link_count) * sizeof(float);
    EXPECT_TRUE(world.GetData().UploadField(nk::FieldId::Q, qpos, bytes));
    EXPECT_TRUE(world.GetData().UploadField(nk::FieldId::Qdot, qvel, bytes));
    EXPECT_TRUE(world.GetData().UploadField(nk::FieldId::Tau, tau, bytes));

    nphi::ArticulationDynamicsParams dynamics{};
    dynamics.max_dof = world.GetModel().capacities.dofs_per_env;
    dynamics.integrate_velocity = 0u;
    dynamics.gravity[0] = 0.0f;
    dynamics.gravity[1] = 0.0f;
    dynamics.gravity[2] = -9.81f;
    dynamics.articulation_count = 1u;
    dynamics.total_link_count = link_count;
    EXPECT_EQ(world.DispatchOp(nphi::NkOp::ArticulationDynamics, &dynamics), nphi::Status::Ok);
    nphi::BackendSynchronize(backend);

    std::vector<float> qddot(link_count, 0.0f);
    EXPECT_TRUE(world.GetData().DownloadField(nk::FieldId::Qddot, qddot.data(), bytes));
    return qddot;
}

}  // namespace

TEST(FeatherstoneOracle, FixedBaseDynamicsMatchGolden) {
    const auto go2_model =
        MenageriePath("unitree_go2/go2_mjx.xml");
    const auto h1_model =
        MenageriePath("unitree_h1/h1.xml");
    const auto go2_golden = GoldenPath("featherstone_go2_random_sample.bin");
    const auto h1_golden = GoldenPath("featherstone_h1_random_sample.bin");
    if (!std::filesystem::exists(go2_model) || !std::filesystem::exists(h1_model)) {
        GTEST_SKIP() << "MuJoCo Menagerie Go2/H1 assets are not available";
    }
    if (!std::filesystem::exists(go2_golden) || !std::filesystem::exists(h1_golden)) {
        GTEST_SKIP() << "v0.1 random-sample golden files are owner-protected and not present";
    }

    NkBackendFixture fx;
    ASSERT_NE(fx.dev, nullptr);
    ASSERT_NE(fx.backend, nullptr);

    const struct {
        const std::filesystem::path* model;
        const std::filesystem::path* golden;
        const char* label;
    } cases[] = {{&go2_model, &go2_golden, "go2"}, {&h1_model, &h1_golden, "h1"}};

    for (const auto& c : cases) {
        const auto golden = nuka::tests::oracle::LoadGoldenTrajectory(*c.golden);
        ASSERT_EQ(golden.kind, nuka::tests::oracle::GoldenKind::RandomQacc);
        constexpr uint32_t kMaxSamplesCheckedInTest = 32u;
        const uint32_t sample_count =
            std::min(golden.sample_count, kMaxSamplesCheckedInTest);
        ASSERT_GT(sample_count, 0u);

        nk::Model model = CookNkModelFixedBase(*c.model);
        nk::World world(std::move(model), 1u, fx.dev, fx.backend);
        ASSERT_TRUE(world.Ready()) << c.label << ": " << world.CreationError();

        float max_abs = 0.0f;
        for (uint32_t sample = 0u; sample < sample_count; ++sample) {
            const float* record = golden.Record(sample);
            const float* qacc = record + golden.qpos_count +
                                golden.qvel_count + golden.qvel_count;
            const auto nk_qddot = DynamicsQddot(world, record, golden, fx.backend);
            ASSERT_EQ(nk_qddot.size(), golden.qacc_count) << c.label;
            for (uint32_t dof = 0u; dof < golden.qacc_count; ++dof) {
            max_abs = std::max(max_abs, std::abs(nk_qddot[dof] - qacc[dof]));
            }
        }
        EXPECT_LE(max_abs, 1.0e-3f)
            << c.label << ": nk::World articulation qddot differs from MJX golden";
    }
}

TEST(FeatherstoneOracle, FloatingBaseDynamicsMatchGolden) {
    const auto go2_model =
        MenageriePath("unitree_go2/go2_mjx.xml");
    const auto go2_golden = GoldenPath("featherstone_go2_floating_random_sample.bin");
    if (!std::filesystem::exists(go2_model)) {
        GTEST_SKIP() << "MuJoCo Menagerie Go2 assets are not available";
    }
    if (!std::filesystem::exists(go2_golden)) {
        GTEST_SKIP() << "floating-base golden not present";
    }

    NkBackendFixture fx;
    ASSERT_NE(fx.dev, nullptr);
    ASSERT_NE(fx.backend, nullptr);

    const auto golden = nuka::tests::oracle::LoadGoldenTrajectory(go2_golden);
    ASSERT_EQ(golden.kind, nuka::tests::oracle::GoldenKind::RandomQacc);
    const uint32_t nq = golden.qpos_count;
    const uint32_t nv = golden.qvel_count;
    constexpr uint32_t kMaxSamples = 32u;
    const uint32_t sample_count = std::min(golden.sample_count, kMaxSamples);
    ASSERT_GT(sample_count, 0u);

    // Natural cook (root must stay FloatingBase).
    const auto scene = nuka::import::LoadMjcf(go2_model.string());
    auto cooked = nuka::scene::cook::CookToModel(scene, 1);
    ASSERT_FALSE(cooked.model.articulation.joint_type.empty());
    ASSERT_EQ(cooked.model.articulation.joint_type[0],
              static_cast<uint8_t>(
                  nuka::runtime::articulation::ArticulationJointType::FloatingBase));
    const uint32_t link_count = cooked.model.capacities.links_per_env;
    const uint32_t leg_dofs = link_count - 1u;
    ASSERT_EQ(nv, leg_dofs + 6u);

    nk::World world(std::move(cooked.model), 1u, fx.dev, fx.backend);
    ASSERT_TRUE(world.Ready());

    float leg_max_abs = 0.0f;
    float base_lin_max_abs = 0.0f;
    float base_ang_max_abs = 0.0f;
    for (uint32_t sample = 0u; sample < sample_count; ++sample) {
        const float* record = golden.Record(sample);
        const float* qpos = record;
        const float* qvel = qpos + nq;
        const float* tau = qvel + nv;
        const float* qacc = record + static_cast<size_t>(nq) + nv + nv;

        nuka::math::Transform base_pose;
        base_pose.position = nuka::math::Vec3{qpos[0], qpos[1], qpos[2]};
        nuka::math::Quat base_rot{qpos[3], qpos[4], qpos[5], qpos[6]};
        {
            const float n = std::sqrt(base_rot.w * base_rot.w + base_rot.x * base_rot.x +
                                      base_rot.y * base_rot.y + base_rot.z * base_rot.z);
            const float inv = n > 0.0f ? 1.0f / n : 1.0f;
            base_rot.w *= inv; base_rot.x *= inv; base_rot.y *= inv; base_rot.z *= inv;
        }
        base_pose.rotation = base_rot;

        const nuka::math::Vec3 v_lin_world{qvel[0], qvel[1], qvel[2]};
        const nuka::math::Vec3 omega_body{qvel[3], qvel[4], qvel[5]};
        const nuka::math::Vec3 v_lin_body = RotateByQuatInverse(base_rot, v_lin_world);
        float root_vel[6] = {omega_body.x, omega_body.y, omega_body.z,
                             v_lin_body.x, v_lin_body.y, v_lin_body.z};

        std::vector<float> q(link_count, 0.0f), qd(link_count, 0.0f), tq(link_count, 0.0f);
        for (uint32_t i = 0u; i < leg_dofs; ++i) {
            q[1u + i] = qpos[7u + i];
            qd[1u + i] = qvel[6u + i];
            tq[1u + i] = tau[6u + i];
        }
        const size_t bytes = static_cast<size_t>(link_count) * sizeof(float);
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::Q, q.data(), bytes));
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::Qdot, qd.data(), bytes));
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::Tau, tq.data(), bytes));
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::BasePose, &base_pose,
                                                sizeof(base_pose)));
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::LinkVelocity, root_vel,
                                                sizeof(root_vel)));  // root = link 0

        nphi::ArticulationDynamicsParams dynamics{};
        dynamics.max_dof = world.GetModel().capacities.dofs_per_env;
        dynamics.integrate_velocity = 0u;
        dynamics.gravity[2] = -9.81f;
        dynamics.articulation_count = 1u;
        dynamics.total_link_count = link_count;
        ASSERT_EQ(world.DispatchOp(nphi::NkOp::ArticulationDynamics, &dynamics), nphi::Status::Ok);
        nphi::BackendSynchronize(fx.backend);

        std::vector<float> qddot(link_count, 0.0f);
        float a0[6] = {0, 0, 0, 0, 0, 0};
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Qddot, qddot.data(), bytes));
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::LinkAcceleration, a0,
                                                  sizeof(a0)));  // root = link 0

        const nuka::math::Vec3 g_world{0.0f, 0.0f, 9.81f};  // -kGravityZ
        const nuka::math::Vec3 g_body = RotateByQuatInverse(base_rot, g_world);
        float real[6];
        real[0] = a0[0]; real[1] = a0[1]; real[2] = a0[2];
        real[3] = a0[3] - g_body.x;
        real[4] = a0[4] - g_body.y;
        real[5] = a0[5] - g_body.z;
        const nuka::math::Vec3 omega_dot{real[0], real[1], real[2]};
        const nuka::math::Vec3 a_lin_body_spatial{real[3], real[4], real[5]};
        const nuka::math::Vec3 omega_cross_v{
            omega_body.y * v_lin_body.z - omega_body.z * v_lin_body.y,
            omega_body.z * v_lin_body.x - omega_body.x * v_lin_body.z,
            omega_body.x * v_lin_body.y - omega_body.y * v_lin_body.x};
        const nuka::math::Vec3 a_lin_body{
            a_lin_body_spatial.x + omega_cross_v.x,
            a_lin_body_spatial.y + omega_cross_v.y,
            a_lin_body_spatial.z + omega_cross_v.z};
        const nuka::math::Vec3 a_lin_world = RotateByQuat(base_rot, a_lin_body);
        const std::array<float, 6> base_accel_mjx = {
            a_lin_world.x, a_lin_world.y, a_lin_world.z,
            omega_dot.x, omega_dot.y, omega_dot.z};

        for (uint32_t i = 0u; i < leg_dofs; ++i) {
            leg_max_abs = std::max(leg_max_abs,
                                   std::abs(qddot[1u + i] - qacc[6u + i]));
        }

        for (uint32_t i = 0u; i < 3u; ++i) {
            base_lin_max_abs = std::max(base_lin_max_abs,
                                        std::abs(base_accel_mjx[i] - qacc[i]));
            base_ang_max_abs = std::max(base_ang_max_abs,
                                        std::abs(base_accel_mjx[3u + i] - qacc[3u + i]));
        }
    }

    std::cout << "[NkFloatingBase go2] over " << sample_count
              << " samples: leg_max_abs=" << leg_max_abs
              << "  base_lin_max_abs=" << base_lin_max_abs
              << "  base_ang_max_abs=" << base_ang_max_abs << std::endl;
    EXPECT_LE(leg_max_abs, 1.0e-3f);
    EXPECT_LE(base_lin_max_abs, 1.0e-3f);
    EXPECT_LE(base_ang_max_abs, 1.0e-3f);
}

namespace {

// Download the nk world's per-link q (single env) as a vector.
std::vector<float> NkDownloadQ(nk::World& world) {
    const uint32_t L = world.GetModel().capacities.links_per_env;
    std::vector<float> q(static_cast<size_t>(L) * world.EnvCount(), 0.0f);
    EXPECT_TRUE(world.GetData().DownloadField(nk::FieldId::Q, q.data(),
                                              q.size() * sizeof(float)));
    return q;
}

}  // namespace

TEST(FeatherstoneOracle, Go2StandMatchesGolden) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    const auto golden_path = GoldenPath("go2_stand_5s.bin");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    if (!std::filesystem::exists(golden_path)) {
        GTEST_SKIP() << "v0.1 Go2 stand golden trajectory is owner-protected and not present";
    }

    namespace articulation = nuka::runtime::articulation;
    constexpr float kDt = 1.0f / 240.0f;
    constexpr float kGravityZ = -9.81f;
    constexpr uint32_t kSteps = 1200u;

    NkBackendFixture fx;
    ASSERT_NE(fx.dev, nullptr);
    ASSERT_NE(fx.backend, nullptr);

    auto scene = nuka::import::LoadUsd(scene_path.string());
    for (const auto& joint : scene.Joints()) {
        auto& unconstrained = scene.GetJointMut(joint.id);
        unconstrained.has_lower_limit = unconstrained.has_upper_limit = false;
        unconstrained.frictionloss = 0.0f;
    }

    auto MakeStandWorld = [&]() {
        nuka::scene::cook::CookToModelOptions opts;
        opts.enable_contacts = false;
        auto cooked = nuka::scene::cook::CookToModel(scene, 1, opts);
        cooked.model.feet.clear();
        cooked.model.drive_mode = static_cast<uint32_t>(articulation::ControlMode::Velocity);
        nk::Pipeline::SolverConfig cfg;
        cfg.dt = kDt;
        cfg.gravity[2] = kGravityZ;
        return std::make_unique<nk::World>(std::move(cooked.model), 1u, fx.dev,
                                           fx.backend, cfg);
    };
    auto world = MakeStandWorld();
    ASSERT_TRUE(world->Ready());
    const uint32_t L = world->GetModel().capacities.links_per_env;
    ASSERT_GT(L, 12u);

    const size_t bytes = size_t{L} * sizeof(float);
    const auto target = NkDownloadQ(*world);
    std::vector<float> stiffness(L), damping(L), limits(L), zero(L), effort(L);
    ASSERT_TRUE(world->GetData().DownloadField(nk::FieldId::DriveStiffness, stiffness.data(), bytes));
    ASSERT_TRUE(world->GetData().DownloadField(nk::FieldId::DriveDamping, damping.data(), bytes));
    ASSERT_TRUE(world->GetData().DownloadField(nk::FieldId::DriveForceLimit, limits.data(), bytes));
    // The reference applies clamped explicit spring torque and implicit viscous damping.
    // A velocity actuator plus external spring effort expresses that same input through the production pipeline.
    auto ConfigureReferenceDrive = [&](nk::World& instance) {
        EXPECT_TRUE(instance.GetData().UploadField(nk::FieldId::DriveStiffness, damping.data(), bytes));
        EXPECT_TRUE(instance.GetData().UploadField(nk::FieldId::DriveForceLimit, zero.data(), bytes));
        EXPECT_TRUE(instance.GetData().UploadField(nk::FieldId::VelocityTarget, zero.data(), bytes));
    };
    auto ApplyReferenceSpring = [&](nk::World& instance) {
        const auto q = NkDownloadQ(instance);
        for (uint32_t i = 0u; i < L; ++i) {
            effort[i] = stiffness[i] * (target[i] - q[i]);
            if (limits[i] > 0.0f) effort[i] = std::clamp(effort[i], -limits[i], limits[i]);
        }
        EXPECT_TRUE(instance.GetData().UploadField(nk::FieldId::JointF, effort.data(), bytes));
    };
    ConfigureReferenceDrive(*world);

    std::vector<float> nk_traj;
    nk_traj.reserve(static_cast<size_t>(kSteps) * L);
    for (uint32_t step = 0; step < kSteps; ++step) {
        ApplyReferenceSpring(*world);
        const nk::StepResult res = world->Step();
        ASSERT_TRUE(res.AllOk()) << "nk Step op failure at step " << step;
        const auto q = NkDownloadQ(*world);
        nk_traj.insert(nk_traj.end(), q.begin(), q.end());
    }

    // --- (c) StepPlanned twin: byte-identical trajectory over all 1200 steps.
    auto planned = MakeStandWorld();
    ASSERT_TRUE(planned->Ready());
    ConfigureReferenceDrive(*planned);
    std::vector<float> plan_traj;
    plan_traj.reserve(nk_traj.size());
    for (uint32_t step = 0; step < kSteps; ++step) {
        ApplyReferenceSpring(*planned);
        ASSERT_EQ(planned->StepPlanned(), nphi::Status::Ok)
            << "StepPlanned failed at step " << step;
        const auto q = NkDownloadQ(*planned);
        plan_traj.insert(plan_traj.end(), q.begin(), q.end());
    }
    ASSERT_EQ(plan_traj.size(), nk_traj.size());
    EXPECT_EQ(std::memcmp(plan_traj.data(), nk_traj.data(),
                          nk_traj.size() * sizeof(float)),
              0)
        << "StepPlanned (CUDA-graph plan) trajectory is NOT byte-identical to Step";

    const auto golden = nuka::tests::oracle::LoadGoldenTrajectory(golden_path);
    ASSERT_EQ(golden.kind, nuka::tests::oracle::GoldenKind::JointTrajectory);
    ASSERT_EQ(golden.sample_count, kSteps);
    ASSERT_EQ(golden.qpos_count, L);
    float max_abs = 0.0f;
    uint32_t worst_step = 0u, worst_joint = 0u;
    for (uint32_t step = 0; step < golden.sample_count; ++step) {
        const float* record = golden.Record(step);
        const size_t offset = static_cast<size_t>(step) * golden.qpos_count;
        for (uint32_t joint = 0; joint < golden.qpos_count; ++joint) {
            const float error = std::abs(nk_traj[offset + joint] - record[joint]);
            if (error > max_abs) { max_abs = error; worst_step = step; worst_joint = joint; }
        }
    }
    std::cout << "[NkGo2Stand5s] golden max_abs=" << max_abs << std::endl;
    EXPECT_LE(max_abs, 1.0e-4f)
        << "nk::World Go2 stand joint trajectory differs from MJX golden at step " << worst_step
        << " joint " << worst_joint << " actual " << nk_traj[size_t{worst_step} * L + worst_joint]
        << " reference " << golden.Record(worst_step)[worst_joint];
}

TEST(FeatherstoneOracle, NkWorldBatchedContactStepPlannedByteExact) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }

    constexpr float kDt = 1.0f / 240.0f;
    constexpr float kGravityZ = -9.81f;
    constexpr float kGround = 0.31f;            // perf-gate seat (active contacts)
    constexpr float kBaumgarteMaxVel = 3.0f;
    constexpr uint32_t kEnvs = 4u;
    constexpr uint32_t kSteps = 200u;

    NkBackendFixture fx;
    ASSERT_NE(fx.dev, nullptr);
    ASSERT_NE(fx.backend, nullptr);

    const auto scene = nuka::import::LoadUsd(scene_path.string());

    // --- nk world: cooked feet kept; ground/baumgarte = the perf-gate values.
    auto MakeContactWorld = [&]() {
        auto cooked = nuka::scene::cook::CookToModel(scene, 1);
        EXPECT_EQ(cooked.model.feet.size(), 4u) << "go2 cooks 4 foot spheres";
        cooked.model.ground_height = kGround;
        cooked.model.baumgarte_max_velocity = kBaumgarteMaxVel;
        nk::Pipeline::SolverConfig cfg;
        cfg.dt = kDt;
        cfg.gravity[2] = kGravityZ;
        return std::make_unique<nk::World>(std::move(cooked.model), kEnvs, fx.dev,
                                           fx.backend, cfg);
    };
    auto world_ptr = MakeContactWorld();
    nk::World& world = *world_ptr;
    ASSERT_TRUE(world.Ready());
    const uint32_t L = world.GetModel().capacities.links_per_env;
    const uint32_t total_links = L * kEnvs;

    for (uint32_t step = 0; step < kSteps; ++step) {
        const nk::StepResult res = world.Step();
        ASSERT_TRUE(res.AllOk()) << "nk Step op failure at step " << step;
    }
    nphi::BackendSynchronize(fx.backend);

    auto planned = MakeContactWorld();
    ASSERT_TRUE(planned->Ready());
    for (uint32_t step = 0; step < kSteps; ++step) {
        ASSERT_EQ(planned->StepPlanned(), nphi::Status::Ok)
            << "StepPlanned failed at step " << step;
    }
    nphi::BackendSynchronize(fx.backend);

    const uint32_t row_slots = world.GetModel().capacities.max_rows_per_env * kEnvs;
    std::vector<float> q_step(total_links), q_plan(total_links);
    std::vector<float> l_step(row_slots), l_plan(row_slots);
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Q, q_step.data(),
                                              q_step.size() * 4));
    ASSERT_TRUE(planned->GetData().DownloadField(nk::FieldId::Q, q_plan.data(),
                                                 q_plan.size() * 4));
    ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Lambda, l_step.data(),
                                              l_step.size() * 4));
    ASSERT_TRUE(planned->GetData().DownloadField(nk::FieldId::Lambda, l_plan.data(),
                                                 l_plan.size() * 4));
    EXPECT_EQ(std::memcmp(q_step.data(), q_plan.data(), q_step.size() * 4), 0)
        << "StepPlanned q diverges from Step under active contacts";
    EXPECT_EQ(std::memcmp(l_step.data(), l_plan.data(), l_step.size() * 4), 0)
        << "StepPlanned lambda diverges from Step under active contacts";

    // Active-contact sanity: the seat actually fires contacts (the gate is not
    // vacuously comparing zeros).
    float lambda_sum = 0.0f;
    for (float v : l_step) lambda_sum += std::abs(v);
    EXPECT_GT(lambda_sum, 0.0f) << "no contact impulse — the contact gate is vacuous";
}
