// ---------------------------------------------------------------------------
// v0.1 foundation e2e validation
// ---------------------------------------------------------------------------

#include "core/diagnostics/invariants.hpp"
#include "core/diagnostics/trace_sink.hpp"
#include "runtime/physics_world.hpp"
#include "runtime/world_builder.hpp"
#include "runtime/world_stepper.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace diagnostics = nuka::core::diagnostics;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

nuka::scene::SceneIR BuildConservativeSingleBodyScene() {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord body;
    body.name = "conservative_body";
    body.mass = 1.0f;
    body.inertia = {1.0f, 1.0f, 1.0f};
    body.local_transform.position = {0.0f, 5.0f, 0.0f};
    scene.AddRigidBody(std::move(body));

    return scene;
}

nuka::scene::SceneIR BuildSingleFallingBoxSmokeScene() {
    nuka::scene::SceneIR scene = BuildConservativeSingleBodyScene();

    nuka::scene::CollisionShapeRecord shape;
    shape.body_id = 0;
    shape.type = nuka::scene::ShapeType::Box;
    shape.half_extents = {0.5f, 0.5f, 0.5f};
    scene.AddCollisionShape(std::move(shape));

    return scene;
}

nuka::scene::SceneIR BuildDoublePendulumSmokeScene() {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord base;
    base.name = "base";
    base.is_static = true;
    const auto base_id = scene.AddRigidBody(std::move(base));

    nuka::scene::RigidBodyRecord upper;
    upper.name = "upper_link";
    upper.mass = 1.0f;
    upper.inertia = {0.2f, 0.2f, 0.2f};
    upper.local_transform.position = {0.0f, -0.5f, 0.0f};
    const auto upper_id = scene.AddRigidBody(std::move(upper));

    nuka::scene::RigidBodyRecord lower;
    lower.name = "lower_link";
    lower.mass = 1.0f;
    lower.inertia = {0.2f, 0.2f, 0.2f};
    lower.local_transform.position = {0.0f, -1.5f, 0.0f};
    const auto lower_id = scene.AddRigidBody(std::move(lower));

    nuka::scene::JointRecord shoulder;
    shoulder.name = "shoulder";
    shoulder.type = nuka::scene::JointType::Revolute;
    shoulder.parent_body = base_id;
    shoulder.child_body = upper_id;
    shoulder.axis = nuka::math::Vec3::UnitZ();
    shoulder.lower_limit = -1.2f;
    shoulder.upper_limit = 1.2f;
    scene.AddJoint(std::move(shoulder));

    nuka::scene::JointRecord elbow;
    elbow.name = "elbow";
    elbow.type = nuka::scene::JointType::Revolute;
    elbow.parent_body = upper_id;
    elbow.child_body = lower_id;
    elbow.axis = nuka::math::Vec3::UnitZ();
    elbow.lower_limit = -1.4f;
    elbow.upper_limit = 1.4f;
    scene.AddJoint(std::move(elbow));

    return scene;
}

nuka::scene::SceneIR BuildTwoBodyCollisionSmokeScene() {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord left;
    left.name = "left_sphere";
    left.mass = 1.0f;
    left.inertia = {0.4f, 0.4f, 0.4f};
    left.local_transform.position = {-1.0f, 0.0f, 0.0f};
    const auto left_id = scene.AddRigidBody(std::move(left));

    nuka::scene::CollisionShapeRecord left_shape;
    left_shape.body_id = left_id;
    left_shape.type = nuka::scene::ShapeType::Sphere;
    left_shape.radius = 0.5f;
    scene.AddCollisionShape(std::move(left_shape));

    nuka::scene::RigidBodyRecord right;
    right.name = "right_sphere";
    right.mass = 1.0f;
    right.inertia = {0.4f, 0.4f, 0.4f};
    right.local_transform.position = {1.0f, 0.0f, 0.0f};
    const auto right_id = scene.AddRigidBody(std::move(right));

    nuka::scene::CollisionShapeRecord right_shape;
    right_shape.body_id = right_id;
    right_shape.type = nuka::scene::ShapeType::Sphere;
    right_shape.radius = 0.5f;
    scene.AddCollisionShape(std::move(right_shape));

    return scene;
}

nuka::scene::SceneIR BuildJointRangeScene() {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord base;
    base.name = "base";
    base.is_static = true;
    const auto base_id = scene.AddRigidBody(std::move(base));

    nuka::scene::RigidBodyRecord link;
    link.name = "link";
    link.mass = 1.0f;
    link.inertia = {1.0f, 1.0f, 1.0f};
    link.local_transform.position = {0.0f, 1.0f, 0.0f};
    link.local_transform.rotation =
        nuka::math::Quat::FromAxisAngle(nuka::math::Vec3::UnitZ(), 0.1f);
    const auto link_id = scene.AddRigidBody(std::move(link));

    nuka::scene::JointRecord joint;
    joint.name = "hinge";
    joint.type = nuka::scene::JointType::Revolute;
    joint.parent_body = base_id;
    joint.child_body = link_id;
    joint.axis = nuka::math::Vec3::UnitZ();
    joint.lower_limit = -0.5f;
    joint.upper_limit = 0.5f;
    scene.AddJoint(std::move(joint));

    return scene;
}

std::vector<diagnostics::InvariantSample> SampleWorld(
    nuka::runtime::BuiltWorld& world,
    diagnostics::InvariantConfig config,
    uint32_t step_index,
    nuka::math::Vec3 gravity = nuka::math::Vec3::Zero(),
    const nuka::runtime::WorldStepReport* report = nullptr) {
    diagnostics::InvariantSampler sampler(config);
    std::vector<diagnostics::InvariantSample> samples;
    const diagnostics::InvariantWorldView view{
        &world.template_view,
        &world.instance,
        report,
        gravity
    };
    sampler.Sample(view, step_index, &samples);
    return samples;
}

const diagnostics::InvariantSample* FindSample(
    const std::vector<diagnostics::InvariantSample>& samples,
    diagnostics::Invariant which) {
    for (const auto& sample : samples) {
        if (sample.which == which) {
            return &sample;
        }
    }
    return nullptr;
}

int64_t RunStepperMicros(nuka::runtime::BuiltWorld& world,
                         const nuka::runtime::WorldStepOptions& options,
                         int step_count,
                         diagnostics::InvariantSampler* sampler) {
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < step_count; ++step) {
        nuka::runtime::StepWorldInstance(world.template_view,
                                         world.instance,
                                         options,
                                         sampler,
                                         nullptr);
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void ExpectNoViolations(const std::vector<diagnostics::InvariantSample>& samples,
                        const char* scene_name) {
    for (const auto& sample : samples) {
        EXPECT_FALSE(sample.violation)
            << scene_name << " " << diagnostics::InvariantName(sample.which)
            << " value=" << sample.value
            << " threshold=" << sample.threshold;
    }
}

} // namespace

TEST(V01FoundationE2E, SmokeSceneContractsPassInvariantChecks) {
    constexpr std::array<const char*, 3> kSmokeScenes = {
        "tests/data/smoke/single_falling_box.yaml",
        "tests/data/smoke/double_pendulum.yaml",
        "tests/data/smoke/two_body_collision.yaml",
    };

    for (const char* relative_path : kSmokeScenes) {
        const auto path = SourcePath(relative_path);
        ASSERT_TRUE(std::filesystem::exists(path)) << path.string();
        const auto size = std::filesystem::file_size(path);
        EXPECT_GT(size, 0u) << path.string();
    }

    diagnostics::InvariantConfig config;
    config.enabled = true;
    config.sample_every_steps = 1;
    config.trace_violations_only = true;
    config.abort_on_nan = false;
    config.thresholds.energy_drift_rel = 0.02f;
    config.thresholds.momentum_drift_abs = 1.0e-3f;
    config.thresholds.constraint_residual_abs = 1.0e-3f;
    config.thresholds.joint_range_slop_rad = 1.0e-4f;
    config.thresholds.velocity_max = 100.0f;
    config.thresholds.position_max = 100.0f;

    auto falling_box = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildSingleFallingBoxSmokeScene()));
    auto falling_config = config;
    falling_config.thresholds.momentum_drift_abs = 1000.0f;
    diagnostics::InvariantSampler falling_sampler(falling_config);
    diagnostics::InMemoryRingTraceSink falling_sink(16);
    nuka::runtime::WorldStepOptions falling_options;
    falling_options.gravity = {0.0f, -9.81f, 0.0f};
    falling_options.dt = 1.0f / 240.0f;
    falling_options.step_count = 240;
    falling_options.enable_contacts = false;
    nuka::runtime::StepWorldInstance(falling_box.template_view,
                                     falling_box.instance,
                                     falling_options,
                                     &falling_sampler,
                                     &falling_sink);
    ExpectNoViolations(falling_sink.Samples(), "single_falling_box");

    auto pendulum = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildDoublePendulumSmokeScene()));
    const auto pendulum_samples = SampleWorld(pendulum,
                                              config,
                                              1u,
                                              {0.0f, -9.81f, 0.0f});
    ExpectNoViolations(pendulum_samples, "double_pendulum");

    auto collision = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildTwoBodyCollisionSmokeScene()));
    collision.instance.linear_velocities[0] = {1.0f, 0.0f, 0.0f};
    collision.instance.linear_velocities[1] = {-1.0f, 0.0f, 0.0f};
    diagnostics::InvariantSampler collision_sampler(config);
    diagnostics::InMemoryRingTraceSink collision_sink(16);
    nuka::runtime::WorldStepOptions collision_options;
    collision_options.gravity = nuka::math::Vec3::Zero();
    collision_options.dt = 1.0f / 240.0f;
    collision_options.step_count = 240;
    collision_options.enable_contacts = false;
    nuka::runtime::StepWorldInstance(collision.template_view,
                                     collision.instance,
                                     collision_options,
                                     &collision_sampler,
                                     &collision_sink);
    ExpectNoViolations(collision_sink.Samples(), "two_body_collision");
}

TEST(V01FoundationE2E, V2SamplesOperationalInvariantsAndTracesCsv) {
    auto world = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildConservativeSingleBodyScene()));

    diagnostics::InvariantConfig config;
    config.enabled = true;
    config.sample_every_steps = 1;
    config.trace_violations_only = false;
    config.thresholds.energy_drift_rel = 0.02f;
    config.thresholds.momentum_drift_abs = 1.0e-3f;
    config.thresholds.constraint_residual_abs = 1.0e-3f;
    config.thresholds.joint_range_slop_rad = 1.0e-4f;
    config.thresholds.velocity_max = 100.0f;
    config.thresholds.position_max = 100.0f;

    diagnostics::InvariantSampler sampler(config);
    diagnostics::InMemoryRingTraceSink ring_sink(64);

    nuka::runtime::WorldStepOptions options;
    options.gravity = nuka::math::Vec3::Zero();
    options.dt = 1.0f / 120.0f;
    options.step_count = 4;
    options.enable_contacts = false;

    const auto report = nuka::runtime::StepWorldInstance(world.template_view,
                                                         world.instance,
                                                         options,
                                                         &sampler,
                                                         &ring_sink);
    EXPECT_EQ(report.simulated_step_count, 4u);

    const auto samples = ring_sink.Samples();
    ASSERT_GE(samples.size(), 8u);
    EXPECT_NE(FindSample(samples, diagnostics::Invariant::Energy), nullptr);
    EXPECT_NE(FindSample(samples, diagnostics::Invariant::LinearMomentum), nullptr);
    EXPECT_NE(FindSample(samples, diagnostics::Invariant::AngularMomentum), nullptr);
    EXPECT_NE(FindSample(samples, diagnostics::Invariant::ConstraintResidual), nullptr);
    EXPECT_NE(FindSample(samples, diagnostics::Invariant::JointRange), nullptr);
    EXPECT_NE(FindSample(samples, diagnostics::Invariant::NanInf), nullptr);
    EXPECT_NE(FindSample(samples, diagnostics::Invariant::VelocityEnvelope), nullptr);
    EXPECT_NE(FindSample(samples, diagnostics::Invariant::PositionEnvelope), nullptr);
    for (const auto& sample : samples) {
        EXPECT_FALSE(sample.violation) << diagnostics::InvariantName(sample.which);
    }

    const auto csv_path =
        std::filesystem::temp_directory_path() / "nuka_v01_invariants_e2e.csv";
    {
        diagnostics::CsvTraceSink csv_sink(csv_path.string());
        for (const auto& sample : samples) {
            csv_sink.OnSample(sample);
        }
        csv_sink.Flush();
    }

    std::ifstream csv(csv_path);
    ASSERT_TRUE(csv.good());
    std::string header;
    std::getline(csv, header);
    EXPECT_EQ(header, "step,env_id,invariant,value,threshold,violation");
    std::string first_row;
    std::getline(csv, first_row);
    EXPECT_NE(first_row.find(",energy,"), std::string::npos);
    std::filesystem::remove(csv_path);
}

TEST(V01FoundationE2E, V2DetectsNanWithinSamplingWindow) {
    auto world = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildConservativeSingleBodyScene()));
    ASSERT_FALSE(world.instance.linear_velocities.empty());
    world.instance.linear_velocities[0].x = std::numeric_limits<float>::quiet_NaN();

    diagnostics::InvariantConfig config;
    config.enabled = true;
    config.sample_every_steps = 1;
    config.trace_violations_only = false;
    config.abort_on_nan = false;

    const auto samples = SampleWorld(world, config, 1u);
    const auto* nan_sample = FindSample(samples, diagnostics::Invariant::NanInf);
    ASSERT_NE(nan_sample, nullptr);
    EXPECT_TRUE(nan_sample->violation);
    EXPECT_GT(nan_sample->value, 0.0f);
}

TEST(V01FoundationE2E, V2FlagsConstraintJointAndEnvelopeViolations) {
    auto world = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildJointRangeScene()));
    ASSERT_GT(world.instance.body_count, 1u);
    world.instance.poses[1].rotation =
        nuka::math::Quat::FromAxisAngle(nuka::math::Vec3::UnitZ(), 1.0f);
    world.instance.linear_velocities[1] = {5.0f, 0.0f, 0.0f};
    world.instance.poses[1].position = {12.0f, 1.0f, 0.0f};

    nuka::runtime::WorldStepReport report;
    report.max_constraint_error = 0.25f;

    diagnostics::InvariantConfig config;
    config.enabled = true;
    config.sample_every_steps = 1;
    config.trace_violations_only = false;
    config.thresholds.constraint_residual_abs = 0.01f;
    config.thresholds.joint_range_slop_rad = 0.01f;
    config.thresholds.velocity_max = 1.0f;
    config.thresholds.position_max = 10.0f;

    const auto samples = SampleWorld(world, config, 1u, nuka::math::Vec3::Zero(), &report);
    const auto* residual = FindSample(samples, diagnostics::Invariant::ConstraintResidual);
    const auto* joint = FindSample(samples, diagnostics::Invariant::JointRange);
    const auto* velocity = FindSample(samples, diagnostics::Invariant::VelocityEnvelope);
    const auto* position = FindSample(samples, diagnostics::Invariant::PositionEnvelope);
    ASSERT_NE(residual, nullptr);
    ASSERT_NE(joint, nullptr);
    ASSERT_NE(velocity, nullptr);
    ASSERT_NE(position, nullptr);
    EXPECT_TRUE(residual->violation);
    EXPECT_TRUE(joint->violation);
    EXPECT_TRUE(velocity->violation);
    EXPECT_TRUE(position->violation);
}

TEST(V01FoundationE2E, PhysicsWorldSurfacesConfigurableSampler) {
    const auto blob = nuka::scene::CookScene(BuildConservativeSingleBodyScene());
    auto physics_world = nuka::runtime::BuildPhysicsWorld(blob);

    diagnostics::InvariantConfig config;
    EXPECT_FALSE(config.enabled);
    config.enabled = true;
    config.sample_every_steps = 3;

    diagnostics::NullTraceSink sink;
    physics_world.ConfigureInvariantSampling(config);
    physics_world.SetInvariantTraceSink(&sink);

    ASSERT_NE(physics_world.InvariantSampler(), nullptr);
    EXPECT_TRUE(physics_world.InvariantSampler()->Config().enabled);
    EXPECT_EQ(physics_world.InvariantSampler()->Config().sample_every_steps, 3u);
    EXPECT_EQ(physics_world.InvariantTraceSink(), &sink);

    auto copied = physics_world;
    ASSERT_NE(copied.InvariantSampler(), nullptr);
    EXPECT_EQ(copied.InvariantSampler()->Config().sample_every_steps, 3u);
    EXPECT_EQ(copied.InvariantTraceSink(), &sink);
}

TEST(V01FoundationE2E, V2SamplingOverheadStaysWithinBudget) {
    nuka::scene::SceneIR scene;
    constexpr int body_count = 48;
    for (int i = 0; i < body_count; ++i) {
        nuka::scene::RigidBodyRecord body;
        body.name = "body";
        body.mass = 1.0f;
        body.inertia = {1.0f, 1.0f, 1.0f};
        body.local_transform.position = {static_cast<float>(i), 3.0f, 0.0f};
        scene.AddRigidBody(std::move(body));
    }

    const auto blob = nuka::scene::CookScene(scene);
    auto baseline_world = nuka::runtime::BuildWorld(blob);
    auto disabled_world = nuka::runtime::BuildWorld(blob);
    auto enabled_world = nuka::runtime::BuildWorld(blob);

    nuka::runtime::WorldStepOptions options;
    options.gravity = nuka::math::Vec3::Zero();
    options.dt = 1.0f / 240.0f;
    options.step_count = 1;
    options.enable_contacts = false;

    constexpr int step_count = 320;
    const int64_t baseline_us = RunStepperMicros(baseline_world,
                                                 options,
                                                 step_count,
                                                 nullptr);

    diagnostics::InvariantConfig disabled_config;
    disabled_config.enabled = false;
    diagnostics::InvariantSampler disabled_sampler(disabled_config);
    const int64_t disabled_us = RunStepperMicros(disabled_world,
                                                 options,
                                                 step_count,
                                                 &disabled_sampler);

    diagnostics::InvariantConfig enabled_config;
    enabled_config.enabled = true;
    enabled_config.trace_violations_only = true;
    diagnostics::InvariantSampler enabled_sampler(enabled_config);
    const int64_t enabled_us = RunStepperMicros(enabled_world,
                                                options,
                                                step_count,
                                                &enabled_sampler);

    const int64_t disabled_budget = baseline_us + std::max<int64_t>(1500, baseline_us / 100);
    const int64_t enabled_budget = baseline_us + std::max<int64_t>(3000, baseline_us / 20);
    EXPECT_LE(disabled_us, disabled_budget)
        << "disabled V2 overhead exceeded 1% budget: baseline="
        << baseline_us << "us disabled=" << disabled_us << "us";
    EXPECT_LE(enabled_us, enabled_budget)
        << "sampled V2 overhead exceeded 5% budget: baseline="
        << baseline_us << "us enabled=" << enabled_us << "us";
}
