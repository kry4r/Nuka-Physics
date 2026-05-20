// ---------------------------------------------------------------------------
// Performance test: Integration step timing
// ---------------------------------------------------------------------------

#include "runtime/rigid/integrator.hpp"
#include "constraint/constraint_block.hpp"
#include "math/vec3.hpp"
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
