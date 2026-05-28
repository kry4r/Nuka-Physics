// ---------------------------------------------------------------------------
// nuka::solver – host adapter for CUDA row solving
// ---------------------------------------------------------------------------

#include "solver/rigid_solver.hpp"

#include "constraint/row_builder.hpp"

#if defined(NUKA_HAS_CUDA_RUNTIME)
#include "phi/device_context.hpp"
#include "solver/gpu/row_solver.cuh"
#endif

#include <cmath>
#include <stdexcept>

namespace nuka::solver {

SolveResult SolveConstraints(
    constraint::RowBuffers& rows,
    std::vector<runtime::rigid::BodyState>& bodies,
    const SolverConfig& config) {
    if (rows.RowCount() == 0u || bodies.empty()) {
        return {0.0f, 0u};
    }

#if defined(NUKA_HAS_CUDA_RUNTIME)
    auto context = phi::MakeDefaultDeviceContext();
    gpu::RowSolveConfig gpu_config;
    gpu_config.velocity_iterations = config.velocity_iterations;
    gpu_config.position_iterations = config.position_iterations;
    gpu_config.slop = config.slop;
    gpu_config.baumgarte = config.baumgarte;
    const auto report = gpu::SolveRows(context, rows, bodies, gpu_config);
    return {report.max_position_error, report.velocity_iterations};
#else
    (void)rows;
    (void)bodies;
    (void)config;
    throw std::runtime_error(
        "SolveConstraints requires the CUDA row solver; CPU production solving is disabled");
#endif
}

SolveResult SolveUnitGroundContact() {
    std::vector<runtime::rigid::BodyState> bodies(1);
    auto& body = bodies[0];
    body.inv_mass = 1.0f;
    body.inv_inertia = {6.0f, 6.0f, 6.0f};
    body.position = {0.0f, 0.0f, 0.0f};
    body.linear_velocity = {0.0f, -9.81f * (1.0f / 60.0f), 0.0f};

    constraint::RowBuffers rows;
    constraint::Row row;
    row.row_class_id = constraint::kMaximalContactRowClassId;
    row.lower = 0.0f;
    row.upper = constraint::kRowHugeLimit;
    row.flags = constraint::row_flags::Unilateral;
    constraint::RowMaterial material;
    material.kind = constraint::RowKind::Contact;
    material.group_id = 0u;
    material.normal_row_count = 1u;
    material.position_error = 0.0f;
    rows.AddRow(row,
                {0u, constraint::kInvalidBodyIndex},
                {{0.0f, 1.0f, 0.0f}, math::Vec3::Zero()},
                {{0.0f, -1.0f, 0.0f}, math::Vec3::Zero()},
                material);

    SolverConfig config{};
    config.velocity_iterations = 10u;
    config.position_iterations = 4u;
    auto result = SolveConstraints(rows, bodies, config);
    result.max_penetration = std::abs(bodies[0].position.y);
    return result;
}

} // namespace nuka::solver
