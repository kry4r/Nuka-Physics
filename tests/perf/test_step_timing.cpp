// ---------------------------------------------------------------------------
// Performance test: Integration step timing
// ---------------------------------------------------------------------------

#include "runtime/rigid/integrator.hpp"
#include "math/vec3.hpp"

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
