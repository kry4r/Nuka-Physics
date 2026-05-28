// ---------------------------------------------------------------------------
// Performance test: CUDA Universal Row solver timing
// ---------------------------------------------------------------------------

#include "constraint/row_buffers.hpp"
#include "runtime/rigid/body_state.hpp"
#include "solver/gpu/row_solver.cuh"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <vector>

using namespace nuka;

TEST(CudaSolverTiming, UniversalRowsUnderOneSecond) {
    constexpr int kBodyCount = 128;
    constexpr int kIterationCount = 30;

    std::vector<runtime::rigid::BodyState> bodies(kBodyCount);
    constraint::RowBuffers rows;
    for (int body_index = 0; body_index < kBodyCount; ++body_index) {
        bodies[body_index].inv_mass = 1.0f;
        bodies[body_index].inv_inertia = {1.0f, 1.0f, 1.0f};
        bodies[body_index].linear_velocity = {0.0f, -1.0f, 0.0f};

        constraint::Row row;
        row.row_class_id = constraint::kMaximalContactRowClassId;
        row.lower = 0.0f;
        row.upper = constraint::kRowHugeLimit;
        row.flags = constraint::row_flags::Unilateral;
        constraint::RowMaterial material;
        material.kind = constraint::RowKind::Contact;
        material.group_id = rows.RowCount();
        material.normal_row_count = 1u;
        rows.AddRow(row,
                    {static_cast<uint32_t>(body_index),
                     constraint::kInvalidBodyIndex},
                    {{0.0f, 1.0f, 0.0f}, math::Vec3::Zero()},
                    {{0.0f, -1.0f, 0.0f}, math::Vec3::Zero()},
                    material);
    }

    solver::gpu::RowSolveConfig config;
    config.velocity_iterations = 8u;
    config.position_iterations = 2u;

    solver::gpu::RowSolveReport last_report;
    const auto context = phi::MakeDefaultDeviceContext();
    {
        auto warmup_rows = rows;
        auto warmup_bodies = bodies;
        solver::gpu::SolveRows(context, warmup_rows, warmup_bodies, config);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < kIterationCount; ++iteration) {
        auto iteration_rows = rows;
        auto iteration_bodies = bodies;
        last_report = solver::gpu::SolveRows(context,
                                             iteration_rows,
                                             iteration_bodies,
                                             config);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(last_report.row_count, static_cast<uint32_t>(kBodyCount));
    EXPECT_EQ(last_report.velocity_iterations, config.velocity_iterations);
    EXPECT_EQ(last_report.position_iterations, config.position_iterations);
    EXPECT_EQ(last_report.row_scheduler_report.row_kind,
              runtime::gpu::CudaConstraintRowBufferKind::UniversalRowCsr);
    EXPECT_EQ(last_report.row_scheduler_report.row_layout,
              runtime::gpu::CudaConstraintRowLayout::UniversalRowCsr);
    EXPECT_EQ(last_report.row_scheduler_report.schedule_mode,
              runtime::gpu::CudaConstraintRowScheduleMode::IslandColoredSweep);
    EXPECT_EQ(last_report.row_scheduler_report.configured_iterations,
              config.velocity_iterations);
    EXPECT_EQ(last_report.row_scheduler_report.executed_iterations,
              config.velocity_iterations);
    EXPECT_GE(last_report.row_scheduler_report.active_row_count,
              last_report.row_count);
    EXPECT_TRUE(std::isfinite(last_report.row_scheduler_report.max_residual));
    EXPECT_LT(ms, 1000) << "CUDA row solver pipeline took " << ms << " ms";
}
