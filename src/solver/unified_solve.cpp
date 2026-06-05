// ---------------------------------------------------------------------------
// nuka::solver -- UnifiedSolve host adapter (v0.8 C5a). VALIDATED, NOT WIRED.
// ---------------------------------------------------------------------------
// Mirrors solver::SolveConstraints (rigid_solver.cpp), but additionally threads
// the per-row ContactRowSides stream to the GPU so the compliant branch can
// dispatch each contact side's reaction by side.react. See unified_solve.hpp.
// ---------------------------------------------------------------------------

#include "solver/unified_solve.hpp"

#if defined(NUKA_HAS_CUDA_RUNTIME)
#include "phi/device_context.hpp"
#include "solver/gpu/row_solver.cuh"
#endif

#include <stdexcept>

namespace nuka::solver {

void UnifiedSolve(const SolveContext& ctx, const solver::SolverConfig& config) {
    if (ctx.rows == nullptr || ctx.state == nullptr) {
        return;
    }
    if (ctx.rows->RowCount() == 0u || ctx.state->empty()) {
        return;
    }

#if defined(NUKA_HAS_CUDA_RUNTIME)
    auto context = phi::MakeDefaultDeviceContext();
    gpu::RowSolveConfig gpu_config;
    gpu_config.velocity_iterations = config.velocity_iterations;
    gpu_config.position_iterations = config.position_iterations;
    gpu_config.slop = config.slop;
    gpu_config.baumgarte = config.baumgarte;
    // C5a: the compliant branch scales each row's aref (acceleration) by dt to a
    // velocity-target bias for the impulse PGS (done locally in the kernel, no
    // buffer mutation). Legacy SolveConstraints leaves dt=0 (byte-identical).
    gpu_config.dt = ctx.dt;

    const constraint::ContactRowSides* sides_ptr =
        ctx.sides != nullptr ? ctx.sides->data() : nullptr;
    const uint32_t sides_count =
        ctx.sides != nullptr ? static_cast<uint32_t>(ctx.sides->size()) : 0u;

    gpu::SolveRowsWithSides(context,
                            *ctx.rows,
                            ctx.state->data(),
                            static_cast<uint32_t>(ctx.state->size()),
                            sides_ptr,
                            sides_count,
                            gpu_config);
#else
    (void)config;
    throw std::runtime_error(
        "UnifiedSolve requires the CUDA row solver; CPU production solving is disabled");
#endif
}

} // namespace nuka::solver
