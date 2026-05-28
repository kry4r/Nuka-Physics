// ---------------------------------------------------------------------------
// Regression test: Two boxes stacked on a ground plane
// ---------------------------------------------------------------------------

#include "solver/rigid_solver.hpp"
#include "runtime/rigid/integrator.hpp"
#include "constraint/row_buffers.hpp"
#include "math/vec3.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace nuka;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Build a minimal two-box stack scenario:
//   body 0: bottom box sitting on ground plane (y = 0.5)
//   body 1: top box sitting on bottom box   (y = 1.5)
// Each box is 1m tall.  Ground is at y = 0.

struct StackResult {
    float max_penetration;
    float top_box_y;
    float bottom_box_y;
};

StackResult RunRestStackRegression() {
    std::vector<runtime::rigid::BodyState> bodies(2);

    // Bottom box: mass = 2 kg, resting on ground at y = 0.5
    bodies[0].inv_mass         = 0.5f;
    bodies[0].inv_inertia      = {6.0f, 6.0f, 6.0f};
    bodies[0].position         = {0.0f, 0.5f, 0.0f};
    bodies[0].linear_velocity  = {0.0f, 0.0f, 0.0f};
    bodies[0].angular_velocity = {0.0f, 0.0f, 0.0f};
    bodies[0].force            = {0.0f, 0.0f, 0.0f};
    bodies[0].torque           = {0.0f, 0.0f, 0.0f};

    // Top box: mass = 1 kg, sitting on top of bottom box at y = 1.5
    bodies[1].inv_mass         = 1.0f;
    bodies[1].inv_inertia      = {6.0f, 6.0f, 6.0f};
    bodies[1].position         = {0.0f, 1.5f, 0.0f};
    bodies[1].linear_velocity  = {0.0f, 0.0f, 0.0f};
    bodies[1].angular_velocity = {0.0f, 0.0f, 0.0f};
    bodies[1].force            = {0.0f, 0.0f, 0.0f};
    bodies[1].torque           = {0.0f, 0.0f, 0.0f};

    constraint::RowBuffers rows;
    constraint::Row row;
    row.row_class_id = constraint::kMaximalContactRowClassId;
    row.lower = 0.0f;
    row.upper = constraint::kRowHugeLimit;
    row.flags = constraint::row_flags::Unilateral;
    constraint::RowMaterial material;
    material.kind = constraint::RowKind::Contact;
    material.normal_row_count = 1u;

    material.group_id = 0u;
    rows.AddRow(row,
                {0u, constraint::kInvalidBodyIndex},
                {{0.0f, 1.0f, 0.0f}, math::Vec3::Zero()},
                {{0.0f, -1.0f, 0.0f}, math::Vec3::Zero()},
                material);

    material.group_id = 1u;
    rows.AddRow(row,
                {1u, 0u},
                {{0.0f, 1.0f, 0.0f}, math::Vec3::Zero()},
                {{0.0f, -1.0f, 0.0f}, math::Vec3::Zero()},
                material);

    auto result = solver::SolveConstraints(rows, bodies);

    return {result.max_penetration, bodies[1].position.y, bodies[0].position.y};
}

}  // namespace

// ---------------------------------------------------------------------------
// Penetration is bounded after solving
// ---------------------------------------------------------------------------

TEST(RestStackRegression, PenetrationIsBounded) {
    const auto result = RunRestStackRegression();
    EXPECT_LT(result.max_penetration, 1e-2f);
}

// ---------------------------------------------------------------------------
// Top box stays above bottom box
// ---------------------------------------------------------------------------

TEST(RestStackRegression, TopBoxStaysAboveBottomBox) {
    const auto result = RunRestStackRegression();
    EXPECT_GT(result.top_box_y, result.bottom_box_y);
}

// ---------------------------------------------------------------------------
// Bottom box stays above ground
// ---------------------------------------------------------------------------

TEST(RestStackRegression, BottomBoxStaysAboveGround) {
    const auto result = RunRestStackRegression();
    EXPECT_GE(result.bottom_box_y, 0.0f);
}
