// ---------------------------------------------------------------------------
// Tests: GPU row solver resting contact behavior
// ---------------------------------------------------------------------------

#include "solver/rigid_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

nuka::constraint::RowBuffers BuildGroundContactRows(float penetration) {
    nuka::constraint::RowBuffers rows;
    nuka::constraint::Row row;
    row.row_class_id = nuka::constraint::kMaximalContactRowClassId;
    row.lower = 0.0f;
    row.upper = nuka::constraint::kRowHugeLimit;
    row.flags = nuka::constraint::row_flags::Unilateral;

    nuka::constraint::RowMaterial material;
    material.kind = nuka::constraint::RowKind::Contact;
    material.group_id = 0u;
    material.normal_row_count = 1u;
    material.position_error = penetration;

    rows.AddRow(row,
                {0u, nuka::constraint::kInvalidBodyIndex},
                {{0.0f, 1.0f, 0.0f}, nuka::math::Vec3::Zero()},
                {{0.0f, -1.0f, 0.0f}, nuka::math::Vec3::Zero()},
                material);
    return rows;
}

} // namespace

TEST(RigidSolver, RestingContactPreventsFurtherPenetration) {
    auto result = nuka::solver::SolveUnitGroundContact();
    EXPECT_LE(result.max_penetration, 1.0e-3f);
}

TEST(RigidSolver, IterationsReported) {
    auto result = nuka::solver::SolveUnitGroundContact();
    EXPECT_GE(result.iterations_used, 1u);
    EXPECT_LE(result.iterations_used, 100u);
}

TEST(RigidSolver, SlopToleranceRespected) {
    std::vector<nuka::runtime::rigid::BodyState> bodies(1);
    bodies[0].inv_mass = 1.0f;
    bodies[0].inv_inertia = {6.0f, 6.0f, 6.0f};
    bodies[0].position = {0.0f, -0.003f, 0.0f};

    auto rows = BuildGroundContactRows(0.003f);
    nuka::solver::SolverConfig config{};
    config.slop = 0.005f;

    nuka::solver::SolveConstraints(rows, bodies, config);

    EXPECT_LE(std::abs(bodies[0].position.y), 0.01f);
}

TEST(RigidSolver, EmptyInputReturnsZero) {
    nuka::constraint::RowBuffers rows;
    std::vector<nuka::runtime::rigid::BodyState> bodies;
    auto result = nuka::solver::SolveConstraints(rows, bodies);
    EXPECT_FLOAT_EQ(result.max_penetration, 0.0f);
    EXPECT_EQ(result.iterations_used, 0u);
}
