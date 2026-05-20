// ---------------------------------------------------------------------------
// Performance test: Integration step timing
// ---------------------------------------------------------------------------

#include "runtime/rigid/integrator.hpp"
#include "constraint/constraint_block.hpp"
#include "math/vec3.hpp"
#include "runtime/world_builder.hpp"
#include "runtime/world_stepper.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"
#include "solver/rigid_solver.hpp"

#include <gtest/gtest.h>
#include <chrono>
#include <vector>

using namespace nuka;
using namespace nuka::runtime::rigid;

// ---------------------------------------------------------------------------
// 100 steps on 10 bodies should complete well under 1 second
// ---------------------------------------------------------------------------

TEST(StepTiming, HundredStepsUnderOneSecond) {
    const int num_bodies = 10;
    const int num_steps  = 100;
    const float dt       = 0.01f;
    const math::Vec3 gravity{0.0f, -9.81f, 0.0f};

    std::vector<BodyState> bodies(num_bodies);
    for (int i = 0; i < num_bodies; ++i) {
        bodies[i].inv_mass         = 1.0f;
        bodies[i].inv_inertia      = {1.0f, 1.0f, 1.0f};
        bodies[i].position         = {static_cast<float>(i), 10.0f, 0.0f};
        bodies[i].orientation      = math::Quat::Identity();
        bodies[i].linear_velocity  = {0.0f, 0.0f, 0.0f};
        bodies[i].angular_velocity = {0.0f, 0.0f, 0.0f};
        bodies[i].force            = {0.0f, 0.0f, 0.0f};
        bodies[i].torque           = {0.0f, 0.0f, 0.0f};
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < num_steps; ++step) {
        for (auto& body : bodies) {
            StepBody(body, gravity, dt);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_LT(ms, 1000) << "100 steps on 10 bodies took " << ms << " ms";
}

// ---------------------------------------------------------------------------
// 1000 steps on a single body stays under 1 second
// ---------------------------------------------------------------------------

TEST(StepTiming, ThousandStepsSingleBody) {
    const int num_steps = 1000;
    const float dt      = 0.001f;
    const math::Vec3 gravity{0.0f, -9.81f, 0.0f};

    BodyState body{};
    body.inv_mass         = 1.0f;
    body.inv_inertia      = {1.0f, 1.0f, 1.0f};
    body.position         = {0.0f, 100.0f, 0.0f};
    body.orientation      = math::Quat::Identity();
    body.linear_velocity  = {0.0f, 0.0f, 0.0f};
    body.angular_velocity = {0.0f, 0.0f, 0.0f};
    body.force            = {0.0f, 0.0f, 0.0f};
    body.torque           = {0.0f, 0.0f, 0.0f};

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < num_steps; ++step) {
        StepBody(body, gravity, dt);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_LT(ms, 1000) << "1000 steps on 1 body took " << ms << " ms";
}

TEST(StepTiming, ContactMaterialSolveUnderOneSecond) {
    constexpr int body_count = 64;
    constexpr int step_count = 120;

    std::vector<BodyState> bodies(body_count);
    std::vector<constraint::ConstraintBlock> contacts(body_count);

    for (int i = 0; i < body_count; ++i) {
        bodies[i].inv_mass = 1.0f;
        bodies[i].inv_inertia = {1.0f, 1.0f, 1.0f};
        bodies[i].linear_velocity = {
            1.0f + static_cast<float>(i % 7),
            -2.0f - 0.01f * static_cast<float>(i),
            0.5f
        };

        auto& contact = contacts[i];
        contact.type = constraint::ConstraintType::Contact;
        contact.body_a = static_cast<uint32_t>(i);
        contact.body_b = ~0u;
        contact.row_count = 3;
        contact.normal_row_count = 1;
        contact.first_friction_row = 1;
        contact.friction_row_count = 2;
        contact.friction = 0.6f;
        contact.restitution = 0.1f;

        contact.jacobian_linear_a[0] = {0.0f, 1.0f, 0.0f};
        contact.jacobian_linear_b[0] = {0.0f, -1.0f, 0.0f};
        contact.lower_limit[0] = 0.0f;
        contact.upper_limit[0] = 1e6f;

        contact.jacobian_linear_a[1] = {1.0f, 0.0f, 0.0f};
        contact.jacobian_linear_b[1] = {-1.0f, 0.0f, 0.0f};
        contact.jacobian_linear_a[2] = {0.0f, 0.0f, 1.0f};
        contact.jacobian_linear_b[2] = {0.0f, 0.0f, -1.0f};
    }

    solver::SolverConfig config{};
    config.velocity_iterations = 12;
    config.position_iterations = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < step_count; ++step) {
        auto step_contacts = contacts;
        solver::SolveConstraints(step_contacts, bodies, config);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_LT(ms, 1000) << "contact material solve took " << ms << " ms";
}

TEST(StepTiming, RuntimeContactPipelineUnderOneSecond) {
    scene::SceneIR scene;

    scene::RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    const auto ground_id = scene.AddRigidBody(std::move(ground));

    scene::CollisionShapeRecord plane;
    plane.body_id = ground_id;
    plane.type = scene::ShapeType::Plane;
    scene.AddCollisionShape(std::move(plane));

    constexpr int dynamic_body_count = 48;
    for (int i = 0; i < dynamic_body_count; ++i) {
        scene::RigidBodyRecord body;
        body.name = "box";
        body.mass = 1.0f;
        body.inertia = {1.0f, 1.0f, 1.0f};
        body.local_transform.position = {
            static_cast<float>(i % 12) * 1.25f,
            0.45f + static_cast<float>(i / 12) * 0.02f,
            static_cast<float>(i / 12) * 1.25f
        };
        const auto body_id = scene.AddRigidBody(std::move(body));

        scene::CollisionShapeRecord shape;
        shape.body_id = body_id;
        shape.type = scene::ShapeType::Box;
        shape.half_extents = {0.5f, 0.5f, 0.5f};
        scene.AddCollisionShape(std::move(shape));
    }

    auto world = runtime::BuildWorld(scene::CookScene(scene));
    for (uint32_t body_id = 1; body_id < world.instance.body_count; ++body_id) {
        world.instance.linear_velocities[body_id] = {0.2f, -1.0f, 0.1f};
    }

    runtime::WorldStepOptions options;
    options.gravity = {0.0f, 0.0f, 0.0f};
    options.dt = 1.0f / 120.0f;
    options.solver_velocity_iterations = 8;
    options.solver_position_iterations = 4;

    runtime::WorldStepReport last_report;
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int step_count = 120;
    for (int step = 0; step < step_count; ++step) {
        last_report = runtime::StepWorldInstance(world.template_view,
                                                 world.instance,
                                                 options);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_GT(last_report.contact_manifold_count, 0u);
    EXPECT_GT(last_report.constraint_row_count, 0u);
    EXPECT_LT(ms, 1000) << "runtime contact pipeline took " << ms << " ms";
}

TEST(StepTiming, RuntimeJointDrivePipelineUnderOneSecond) {
    scene::SceneIR scene;
    constexpr int joint_count = 32;

    for (int i = 0; i < joint_count; ++i) {
        scene::RigidBodyRecord parent;
        parent.name = "parent";
        parent.is_static = true;
        parent.local_transform.position = {static_cast<float>(i), 0.0f, 0.0f};
        const auto parent_id = scene.AddRigidBody(std::move(parent));

        scene::RigidBodyRecord child;
        child.name = "child";
        child.mass = 1.0f;
        child.inertia = {1.0f, 1.0f, 1.0f};
        child.local_transform.position = {static_cast<float>(i), 0.02f, 0.0f};
        const auto child_id = scene.AddRigidBody(std::move(child));

        scene::JointRecord joint;
        joint.name = "hinge";
        joint.type = scene::JointType::Revolute;
        joint.parent_body = parent_id;
        joint.child_body = child_id;
        joint.axis = math::Vec3::UnitZ();
        const auto joint_id = scene.AddJoint(std::move(joint));

        scene::ActuatorRecord actuator;
        actuator.name = "drive";
        actuator.type = scene::ActuatorType::Velocity;
        actuator.joint_id = joint_id;
        actuator.gain = 2.0f;
        actuator.force_limit = 10.0f;
        scene.AddActuator(std::move(actuator));
    }

    auto world = runtime::BuildWorld(scene::CookScene(scene));

    runtime::WorldStepOptions options;
    options.gravity = {0.0f, 0.0f, 0.0f};
    options.dt = 1.0f / 120.0f;
    options.enable_contacts = false;
    options.solver_velocity_iterations = 8;
    options.solver_position_iterations = 4;
    options.solver_baumgarte = 0.5f;
    options.solver_slop = 0.0f;

    runtime::WorldStepReport last_report;
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int step_count = 120;
    for (int step = 0; step < step_count; ++step) {
        last_report = runtime::StepWorldInstance(world.template_view,
                                                 world.instance,
                                                 options);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(last_report.joint_constraint_count, static_cast<uint32_t>(joint_count));
    EXPECT_EQ(last_report.drive_constraint_count, static_cast<uint32_t>(joint_count));
    EXPECT_GT(world.instance.angular_velocities[1].z, 0.0f);
    EXPECT_LT(ms, 1000) << "runtime joint-drive pipeline took " << ms << " ms";
}
