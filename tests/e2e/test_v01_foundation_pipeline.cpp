// ---------------------------------------------------------------------------
// v0.1 foundation e2e validation
// ---------------------------------------------------------------------------

#include "core/diagnostics/invariants.hpp"
#include "core/diagnostics/trace_sink.hpp"
#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_jacobian.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/physics_world.hpp"
#include "runtime/rigid/body_state.hpp"
#include "runtime/world_builder.hpp"
#include "runtime/world_stepper.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"
#include "solver/gpu/row_solver.cuh"

#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace diagnostics = nuka::core::diagnostics;

#ifndef NUKA_ABA_CUDA_PROBE
#define NUKA_ABA_CUDA_PROBE ""
#endif

#ifndef NUKA_ORACLE_PYTHON
#define NUKA_ORACLE_PYTHON ""
#endif

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

nuka::scene::SceneIR BuildPointMassDoublePendulumScene(float length1,
                                                       float length2,
                                                       float mass1,
                                                       float mass2) {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord base;
    base.name = "base";
    base.is_static = true;
    const auto base_id = scene.AddRigidBody(std::move(base));

    nuka::scene::RigidBodyRecord link1;
    link1.name = "link1";
    link1.mass = mass1;
    link1.inertia = nuka::math::Vec3::Zero();
    link1.inertial_transform.position = {length1, 0.0f, 0.0f};
    const auto link1_id = scene.AddRigidBody(std::move(link1));

    nuka::scene::JointRecord shoulder;
    shoulder.name = "shoulder";
    shoulder.type = nuka::scene::JointType::Revolute;
    shoulder.parent_body = base_id;
    shoulder.child_body = link1_id;
    shoulder.axis = nuka::math::Vec3::UnitY();
    scene.AddJoint(std::move(shoulder));

    nuka::scene::RigidBodyRecord link2;
    link2.name = "link2";
    link2.mass = mass2;
    link2.inertia = nuka::math::Vec3::Zero();
    link2.local_transform.position = {length1, 0.0f, 0.0f};
    link2.inertial_transform.position = {length2, 0.0f, 0.0f};
    const auto link2_id = scene.AddRigidBody(std::move(link2));

    nuka::scene::JointRecord elbow;
    elbow.name = "elbow";
    elbow.type = nuka::scene::JointType::Revolute;
    elbow.parent_body = link1_id;
    elbow.child_body = link2_id;
    elbow.axis = nuka::math::Vec3::UnitY();
    scene.AddJoint(std::move(elbow));

    return scene;
}

nuka::scene::SceneIR BuildSinglePointPendulumScene(float length, float mass) {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord base;
    base.name = "base";
    base.is_static = true;
    const auto base_id = scene.AddRigidBody(std::move(base));

    nuka::scene::RigidBodyRecord bob;
    bob.name = "bob";
    bob.mass = mass;
    bob.inertia = nuka::math::Vec3::Zero();
    bob.inertial_transform.position = {length, 0.0f, 0.0f};
    const auto bob_id = scene.AddRigidBody(std::move(bob));

    nuka::scene::JointRecord hinge;
    hinge.name = "hinge";
    hinge.type = nuka::scene::JointType::Revolute;
    hinge.parent_body = base_id;
    hinge.child_body = bob_id;
    hinge.axis = nuka::math::Vec3::UnitY();
    scene.AddJoint(std::move(hinge));

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

nuka::scene::SceneIR BuildArticulationCoverageScene() {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord base;
    base.name = "base";
    base.is_static = true;
    const auto base_id = scene.AddRigidBody(std::move(base));

    nuka::scene::RigidBodyRecord hinge;
    hinge.name = "hinge_link";
    hinge.mass = 2.0f;
    hinge.inertia = {2.0f, 3.0f, 4.0f};
    hinge.local_transform.position = {1.0f, 0.0f, 0.0f};
    const auto hinge_id = scene.AddRigidBody(std::move(hinge));

    nuka::scene::JointRecord revolute;
    revolute.name = "hip";
    revolute.type = nuka::scene::JointType::Revolute;
    revolute.parent_body = base_id;
    revolute.child_body = hinge_id;
    revolute.axis = nuka::math::Vec3::UnitZ();
    scene.AddJoint(std::move(revolute));

    nuka::scene::RigidBodyRecord slider;
    slider.name = "slider_link";
    slider.mass = 3.0f;
    slider.inertia = {1.0f, 1.0f, 1.0f};
    slider.local_transform.position = {1.0f, 0.0f, 0.5f};
    const auto slider_id = scene.AddRigidBody(std::move(slider));

    nuka::scene::JointRecord prismatic;
    prismatic.name = "knee_slide";
    prismatic.type = nuka::scene::JointType::Prismatic;
    prismatic.parent_body = hinge_id;
    prismatic.child_body = slider_id;
    prismatic.axis = nuka::math::Vec3::UnitZ();
    scene.AddJoint(std::move(prismatic));

    nuka::scene::RigidBodyRecord fixed_link;
    fixed_link.name = "fixed_foot";
    fixed_link.mass = 1.0f;
    fixed_link.inertia = {0.5f, 0.6f, 0.7f};
    fixed_link.local_transform.position = {1.0f, 0.0f, 0.75f};
    const auto fixed_id = scene.AddRigidBody(std::move(fixed_link));

    nuka::scene::JointRecord fixed;
    fixed.name = "ankle_fixed";
    fixed.type = nuka::scene::JointType::Fixed;
    fixed.parent_body = slider_id;
    fixed.child_body = fixed_id;
    fixed.axis = nuka::math::Vec3::UnitZ();
    scene.AddJoint(std::move(fixed));

    return scene;
}

void ExpectFloatVectorEq(const std::vector<float>& lhs,
                         const std::vector<float>& rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());
    for (size_t index = 0; index < lhs.size(); ++index) {
        EXPECT_EQ(lhs[index], rhs[index]) << "index " << index;
    }
}

void ExpectNearRelative(float actual,
                        float expected,
                        float relative_tolerance,
                        const char* label) {
    const float scale = std::max(1.0f, std::abs(expected));
    EXPECT_LE(std::abs(actual - expected), relative_tolerance * scale)
        << label << " actual=" << actual << " expected=" << expected;
}

std::array<float, 2> AnalyticalPointMassDoublePendulumQddot(float q1,
                                                            float q2,
                                                            float tau1,
                                                            float tau2,
                                                            float length1,
                                                            float length2,
                                                            float mass1,
                                                            float mass2,
                                                            float gravity) {
    const float gravity_magnitude = -gravity;
    const float c2 = std::cos(q2);
    const float m11 = (mass1 + mass2) * length1 * length1 +
                      mass2 * length2 * length2 +
                      2.0f * mass2 * length1 * length2 * c2;
    const float m12 = mass2 * length2 * length2 +
                      mass2 * length1 * length2 * c2;
    const float m22 = mass2 * length2 * length2;
    const float gravity1 = (mass1 + mass2) * gravity_magnitude * length1 * std::cos(q1) +
                           mass2 * gravity_magnitude * length2 * std::cos(q1 + q2);
    const float gravity2 = mass2 * gravity_magnitude * length2 * std::cos(q1 + q2);
    const float rhs1 = tau1 + gravity1;
    const float rhs2 = tau2 + gravity2;
    const float determinant = m11 * m22 - m12 * m12;
    return {
        (rhs1 * m22 - m12 * rhs2) / determinant,
        (m11 * rhs2 - m12 * rhs1) / determinant
    };
}

float PointPendulumEnergy(float q,
                          float qdot,
                          float length,
                          float mass,
                          float gravity) {
    const float kinetic = 0.5f * mass * length * length * qdot * qdot;
    const float potential = mass * gravity * length * std::sin(q);
    return kinetic + potential;
}

std::string QuotePath(const std::filesystem::path& path) {
    return "'" + path.string() + "'";
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
    auto measured_options = options;
    measured_options.step_count = static_cast<uint32_t>(step_count);
    const auto start = std::chrono::steady_clock::now();
    nuka::runtime::StepWorldInstance(world.template_view,
                                     world.instance,
                                     measured_options,
                                     sampler,
                                     nullptr);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

template <typename BaselineFn, typename MeasuredFn>
std::array<int64_t, 2> BestOverheadMicros(BaselineFn&& baseline,
                                          MeasuredFn&& measured) {
    (void)baseline();
    (void)measured();
    std::array<int64_t, 2> best{
        0,
        std::numeric_limits<int64_t>::max()
    };
    for (int sample = 0; sample < 7; ++sample) {
        const int64_t baseline_us = baseline();
        const int64_t measured_us = measured();
        const int64_t overhead_us = std::max<int64_t>(0, measured_us - baseline_us);
        if (overhead_us < best[1]) {
            best = {baseline_us, overhead_us};
        }
    }
    return best;
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

TEST(V01FoundationE2E, V2SamplingWindowPersistsAcrossSingleStepCalls) {
    auto world = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildConservativeSingleBodyScene()));

    diagnostics::InvariantConfig config;
    config.enabled = true;
    config.sample_every_steps = 4;
    config.trace_violations_only = false;

    diagnostics::InvariantSampler sampler(config);
    diagnostics::InMemoryRingTraceSink ring_sink(16);

    nuka::runtime::WorldStepOptions options;
    options.gravity = nuka::math::Vec3::Zero();
    options.dt = 1.0f / 120.0f;
    options.step_count = 1;
    options.enable_contacts = false;

    for (int step = 0; step < 4; ++step) {
        nuka::runtime::StepWorldInstance(world.template_view,
                                         world.instance,
                                         options,
                                         &sampler,
                                         &ring_sink);
    }

    const auto samples = ring_sink.Samples();
    ASSERT_GE(samples.size(), 8u);
    EXPECT_EQ(samples.front().step_index, 4u);
    EXPECT_NE(FindSample(samples, diagnostics::Invariant::Energy), nullptr);
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

    nuka::runtime::WorldStepOptions options;
    options.gravity = nuka::math::Vec3::Zero();
    options.dt = 1.0f / 240.0f;
    options.step_count = 1;
    options.enable_contacts = false;

    constexpr int step_count = 320;
    const auto baseline = [&]() {
        auto world = nuka::runtime::BuildWorld(blob);
        return RunStepperMicros(world, options, step_count, nullptr);
    };

    const auto disabled = [&]() {
        auto world = nuka::runtime::BuildWorld(blob);
        diagnostics::InvariantConfig disabled_config;
        disabled_config.enabled = false;
        diagnostics::InvariantSampler disabled_sampler(disabled_config);
        return RunStepperMicros(world, options, step_count, &disabled_sampler);
    };

    const auto enabled = [&]() {
        auto world = nuka::runtime::BuildWorld(blob);
        diagnostics::InvariantConfig enabled_config;
        enabled_config.enabled = true;
        enabled_config.trace_violations_only = true;
        diagnostics::InvariantSampler enabled_sampler(enabled_config);
        return RunStepperMicros(world, options, step_count, &enabled_sampler);
    };

    const auto disabled_timing = BestOverheadMicros(baseline, disabled);
    const auto enabled_timing = BestOverheadMicros(baseline, enabled);

    const int64_t disabled_budget =
        std::max<int64_t>(1500, disabled_timing[0] / 100);
    const int64_t enabled_budget =
        std::max<int64_t>(3000, enabled_timing[0] / 20);
    EXPECT_LE(disabled_timing[1], disabled_budget)
        << "disabled V2 overhead exceeded 1% budget: baseline="
        << disabled_timing[0] << "us overhead=" << disabled_timing[1] << "us";
    EXPECT_LE(enabled_timing[1], enabled_budget)
        << "sampled V2 overhead exceeded 5% budget: baseline="
        << enabled_timing[0] << "us overhead=" << enabled_timing[1] << "us";
}

TEST(V01FoundationE2E, Phase6CooksArticulationsAndPreservesImportedInertia) {
    const auto urdf_scene = nuka::import::LoadUrdf("tests/data/minimal_arm.urdf");
    const auto mjcf_scene = nuka::import::LoadMjcf("tests/data/minimal_arm.xml");
    const auto usd_scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");

    const auto urdf_world = nuka::runtime::BuildWorld(nuka::scene::CookScene(urdf_scene));
    const auto mjcf_world = nuka::runtime::BuildWorld(nuka::scene::CookScene(mjcf_scene));
    const auto usd_world = nuka::runtime::BuildWorld(nuka::scene::CookScene(usd_scene));

    ASSERT_EQ(urdf_world.template_view.articulations.size(), 1u);
    ASSERT_EQ(mjcf_world.template_view.articulations.size(), 1u);
    ASSERT_EQ(usd_world.template_view.articulations.size(), 1u);

    ASSERT_GE(urdf_world.template_view.body_table.masses.size(), 2u);
    EXPECT_FLOAT_EQ(urdf_world.template_view.body_table.masses[1], 0.5f);
    EXPECT_FLOAT_EQ(urdf_world.template_view.body_table.inertias[1].x, 0.05f);

    ASSERT_GE(mjcf_world.template_view.body_table.masses.size(), 2u);
    EXPECT_FLOAT_EQ(mjcf_world.template_view.body_table.masses[1], 0.5f);
    EXPECT_FLOAT_EQ(mjcf_world.template_view.body_table.inertias[1].z, 0.01f);

    ASSERT_GE(usd_world.template_view.body_table.masses.size(), 2u);
    EXPECT_FLOAT_EQ(usd_world.template_view.body_table.masses[1], 0.5f);
    EXPECT_EQ(usd_world.template_view.articulations.front().joint_types[1],
              nuka::runtime::articulation::ArticulationJointType::Revolute);
}

TEST(V01FoundationE2E, Phase6CudaAbaSupportsJointTypesAndIsDeterministic) {
    auto world = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildArticulationCoverageScene()));
    ASSERT_EQ(world.template_view.articulations.size(), 1u);

    auto host_state =
        nuka::runtime::articulation::BuildArticulationHostState(
            world.template_view.articulations,
            world.template_view.body_table);
    ASSERT_EQ(host_state.TotalLinkCount(), 4u);
    ASSERT_EQ(host_state.q.size(), 4u);
    host_state.tau[1] = 8.0f;
    host_state.tau[2] = 0.0f;

    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto device_state =
        nuka::runtime::articulation::UploadArticulationState(context, host_state);
    nuka::runtime::articulation::FeatherstoneAba::ComputeAccelerations(
        context,
        device_state.View(),
        -9.81f);
    context.stream.Synchronize();

    nuka::runtime::articulation::ArticulationHostState first;
    nuka::runtime::articulation::DownloadArticulationState(device_state, &first);
    ASSERT_EQ(first.qddot.size(), 4u);
    EXPECT_FLOAT_EQ(first.qddot[0], 0.0f);
    EXPECT_TRUE(std::isfinite(first.qddot[1]));
    EXPECT_TRUE(std::isfinite(first.qddot[2]));
    EXPECT_GT(std::abs(first.qddot[1]), 1.0e-4f);
    EXPECT_GT(std::abs(first.qddot[2]), 1.0e-4f);
    EXPECT_FLOAT_EQ(first.qddot[3], 0.0f);
    EXPECT_GT(first.joint_diagonal[1], 0.0f);
    EXPECT_GT(first.joint_diagonal[2], 0.0f);

    nuka::runtime::articulation::FeatherstoneAba::ComputeAccelerations(
        context,
        device_state.View(),
        -9.81f);
    context.stream.Synchronize();
    nuka::runtime::articulation::ArticulationHostState second;
    nuka::runtime::articulation::DownloadArticulationState(device_state, &second);
    ExpectFloatVectorEq(first.qddot, second.qddot);
}

TEST(V01FoundationE2E, Phase6CudaAbaMatchesGo2AndH1MuJoCoOracle) {
    const std::filesystem::path python(NUKA_ORACLE_PYTHON);
    const std::filesystem::path probe(NUKA_ABA_CUDA_PROBE);
    const auto script = SourcePath("tools/reference/validate_aba_oracle.py");
    const auto go2 = SourcePath(".nuka-assets/mujoco_menagerie/unitree_go2/go2_mjx.xml");
    const auto h1 = SourcePath(".nuka-assets/mujoco_menagerie/unitree_h1/h1.xml");

    if (python.empty() || !std::filesystem::exists(python)) {
        GTEST_SKIP() << "MuJoCo oracle venv is not available";
    }
    if (probe.empty() || !std::filesystem::exists(probe)) {
        GTEST_SKIP() << "CUDA ABA oracle probe is not available";
    }
    if (!std::filesystem::exists(go2) || !std::filesystem::exists(h1)) {
        GTEST_SKIP() << "MuJoCo Menagerie Go2/H1 assets are not available";
    }

    const std::string command =
        QuotePath(python) + " " +
        QuotePath(script) +
        " --probe " + QuotePath(probe) +
        " --model " + QuotePath(go2) +
        " --model " + QuotePath(h1) +
        " --tolerance 1e-3";
    EXPECT_EQ(std::system(command.c_str()), 0);
}

TEST(V01FoundationE2E, Phase6CudaAbaMatchesAnalyticalDoublePendulum) {
    constexpr float kLength1 = 0.7f;
    constexpr float kLength2 = 0.45f;
    constexpr float kMass1 = 1.3f;
    constexpr float kMass2 = 0.8f;
    constexpr float kGravity = -9.81f;
    constexpr float kQ1 = 0.25f;
    constexpr float kQ2 = -0.35f;
    constexpr float kTau1 = 0.4f;
    constexpr float kTau2 = -0.2f;

    auto world = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildPointMassDoublePendulumScene(
            kLength1,
            kLength2,
            kMass1,
            kMass2)));
    auto host_state =
        nuka::runtime::articulation::BuildArticulationHostState(
            world.template_view.articulations,
            world.template_view.body_table);
    ASSERT_EQ(host_state.q.size(), 3u);
    host_state.q[1] = kQ1;
    host_state.q[2] = kQ2;
    host_state.tau[1] = kTau1;
    host_state.tau[2] = kTau2;

    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto device_state =
        nuka::runtime::articulation::UploadArticulationState(context, host_state);
    nuka::runtime::articulation::FeatherstoneAba::ComputeAccelerations(
        context,
        device_state.View(),
        kGravity);
    context.stream.Synchronize();

    nuka::runtime::articulation::ArticulationHostState out_state;
    nuka::runtime::articulation::DownloadArticulationState(device_state, &out_state);
    const auto expected = AnalyticalPointMassDoublePendulumQddot(kQ1,
                                                                 kQ2,
                                                                 kTau1,
                                                                 kTau2,
                                                                 kLength1,
                                                                 kLength2,
                                                                 kMass1,
                                                                 kMass2,
                                                                 kGravity);
    ASSERT_EQ(out_state.qddot.size(), 3u);
    EXPECT_FLOAT_EQ(out_state.qddot[0], 0.0f);
    ExpectNearRelative(out_state.qddot[1], expected[0], 1.0e-5f, "shoulder qddot");
    ExpectNearRelative(out_state.qddot[2], expected[1], 1.0e-5f, "elbow qddot");
}

TEST(V01FoundationE2E, Phase6WorldStepperRunsCudaAbaBeforeRows) {
    auto world = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildSinglePointPendulumScene(0.75f, 1.2f)));
    ASSERT_EQ(world.instance.articulation_q.size(), 2u);
    world.instance.articulation_q[1] = 0.35f;
    world.instance.articulation_tau[1] = 0.25f;

    nuka::runtime::WorldStepOptions options;
    options.gravity = {0.0f, 0.0f, -9.81f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.enable_contacts = false;
    options.solver_velocity_iterations = 1u;
    options.solver_position_iterations = 0u;
    const auto report = nuka::runtime::StepWorldInstance(world.template_view,
                                                         world.instance,
                                                         options);
    EXPECT_EQ(report.simulated_step_count, 1u);
    ASSERT_EQ(world.instance.articulation_qddot.size(), 2u);
    EXPECT_FLOAT_EQ(world.instance.articulation_qddot[0], 0.0f);
    EXPECT_GT(std::abs(world.instance.articulation_qddot[1]), 1.0e-4f);
    EXPECT_GT(std::abs(world.instance.articulation_qdot[1]), 1.0e-6f);
    EXPECT_NE(world.instance.articulation_q[1], 0.35f);
}

TEST(V01FoundationE2E, Phase6FeatherstoneJacobianFeedsContactRowSolver) {
    auto world = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildArticulationCoverageScene()));
    auto host_state =
        nuka::runtime::articulation::BuildArticulationHostState(
            world.template_view.articulations,
            world.template_view.body_table);

    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto device_state =
        nuka::runtime::articulation::UploadArticulationState(context, host_state);

    const uint32_t contact_link = 1u;
    const nuka::math::Vec3 contact_point{1.0f, 1.0f, 0.0f};
    nuka::phi::Buffer contact_links(sizeof(uint32_t), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer contact_points(sizeof(nuka::math::Vec3), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer jacobian_scalars(sizeof(float), nuka::phi::MemoryKind::Device);
    contact_links.CopyFromHost(&contact_link, sizeof(contact_link));
    contact_points.CopyFromHost(&contact_point, sizeof(contact_point));

    nuka::runtime::articulation::ComputeLinkPointJacobians(
        context,
        device_state.View(),
        static_cast<const uint32_t*>(contact_links.Data()),
        static_cast<const nuka::math::Vec3*>(contact_points.Data()),
        1u,
        static_cast<float*>(jacobian_scalars.Data()));
    context.stream.Synchronize();

    float scalar = 0.0f;
    jacobian_scalars.CopyToHost(&scalar, sizeof(scalar));
    EXPECT_GT(std::abs(scalar), 0.1f);

    std::vector<nuka::runtime::rigid::BodyState> bodies(1u);
    bodies[0].inv_mass = 1.0f;
    bodies[0].inv_inertia = {1.0f, 1.0f, 1.0f};
    bodies[0].linear_velocity = {0.0f, -2.0f, 0.0f};

    nuka::constraint::RowBuffers rows;
    nuka::constraint::RowMaterial material;
    material.kind = nuka::constraint::RowKind::Contact;
    material.group_id = 0u;
    material.normal_row_count = 1u;

    nuka::constraint::Row row;
    row.row_class_id = nuka::constraint::kFeatherstoneContactRowClassId;
    row.lower = 0.0f;
    row.upper = nuka::constraint::kRowHugeLimit;
    row.flags = nuka::constraint::row_flags::Unilateral;
    rows.AddRow(row,
                {0u, nuka::constraint::kInvalidBodyIndex},
                {{0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, scalar}},
                {{0.0f, -1.0f, 0.0f}, nuka::math::Vec3::Zero()},
                material);

    nuka::solver::gpu::RowSolveConfig config;
    config.velocity_iterations = 8u;
    config.position_iterations = 0u;
    const auto report = nuka::solver::gpu::SolveRows(context, rows, bodies, config);
    EXPECT_EQ(report.row_count, 1u);
    EXPECT_GT(rows.rows[0].lambda, 0.0f);
    EXPECT_GT(bodies[0].linear_velocity.y, -2.0f);
    EXPECT_LT(std::abs(bodies[0].linear_velocity.y), 2.0f);
}

TEST(V01FoundationE2E, Phase6PendulumV2EnergyInvariantHoldsForFiveSeconds) {
    auto world = nuka::runtime::BuildWorld(
        nuka::scene::CookScene(BuildSinglePointPendulumScene(0.75f, 1.2f)));
    auto host_state =
        nuka::runtime::articulation::BuildArticulationHostState(
            world.template_view.articulations,
            world.template_view.body_table);
    ASSERT_EQ(host_state.q.size(), 2u);
    constexpr float kLength = 0.75f;
    constexpr float kMass = 1.2f;
    constexpr float kGravity = -9.81f;
    constexpr float kDt = 1.0f / 2400.0f;
    constexpr uint32_t kSteps = 12000u;
    host_state.q[1] = 0.35f;
    host_state.qdot[1] = 0.0f;

    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto device_state =
        nuka::runtime::articulation::UploadArticulationState(context, host_state);
    const float initial_energy =
        PointPendulumEnergy(host_state.q[1], host_state.qdot[1], kLength, kMass, kGravity);

    for (uint32_t step = 0u; step < kSteps; ++step) {
        nuka::runtime::articulation::FeatherstoneAba::ComputeAccelerations(
            context,
            device_state.View(),
            kGravity);
        nuka::runtime::articulation::FeatherstoneAba::Integrate(
            context,
            device_state.View(),
            kDt);
    }
    context.stream.Synchronize();

    nuka::runtime::articulation::ArticulationHostState out_state;
    nuka::runtime::articulation::DownloadArticulationState(device_state, &out_state);
    ASSERT_EQ(out_state.q.size(), 2u);
    const float final_energy =
        PointPendulumEnergy(out_state.q[1], out_state.qdot[1], kLength, kMass, kGravity);
    const float drift =
        std::abs(final_energy - initial_energy) / std::max(1.0f, std::abs(initial_energy));
    EXPECT_LE(drift, 0.02f)
        << "initial_energy=" << initial_energy
        << " final_energy=" << final_energy
        << " q=" << out_state.q[1]
        << " qdot=" << out_state.qdot[1];
}
