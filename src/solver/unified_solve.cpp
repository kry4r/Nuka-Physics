// DEPRECATED(M9): SURVIVES ONLY for the legacy BatchedUnifiedWorld (the M4-M9
// parity ORACLE, edited again in M5) + its co-resident/test consumers. The nk
// core consumes NONE of this: the M4 SolveRowsBlockIsland/AssembleRows ops +
// nk::SolveSchedule (src/nk/solve/) carry the migrated row math + scheduling.
// Deleted with src/runtime/coresident/ at M9. Do NOT add new consumers.
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
    if (ctx.rows->RowCount() == 0u) {
        return;
    }
    // v0.8 C5c-1: the articulation-only case. The go2 foot-ground path is
    // link<->static -- there are NO rigid bodies (state is empty), but the
    // articulation reaction buffers ARE present, so the solve must still run (the
    // chain-J arm fires; the rigid arm is inert via ValidBody(body, 0) == false).
    // C5a/C5b/legacy rigid callers always supply a non-empty `state`, so they keep
    // the original `state->empty()` early-return behaviour byte-for-byte (the new
    // branch is reachable ONLY when state is empty AND articulation data exists).
    const auto& art = ctx.articulation;
    const bool have_articulation =
        art.art_refs != nullptr && !art.art_refs->empty() && art.dof_stride > 0u;
    // v0.8 C6b: a particle-only coupling solve (no rigid bodies, no articulation)
    // must still run -- e.g. particle<->particle / particle<->static coupling.
    const auto& part = ctx.particles;
    const bool have_particles =
        part.inv_mass != nullptr && part.velocities != nullptr &&
        !part.inv_mass->empty() && !part.velocities->empty();
    if (ctx.state->empty() && !have_articulation && !have_particles) {
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

    // v0.8 C5b-core: build the device-side articulation descriptor from the host
    // vectors/spans. Empty (the default SolveContext) -> the articulation arms
    // never fire and the solve is byte-identical to the C5a rigid-only path.
    gpu::RowArticulationData art_data;
    if (have_articulation) {
        art_data.art_refs = art.art_refs->data();
        art_data.art_refs_count = static_cast<uint32_t>(art.art_refs->size());
        if (art.chain_jacobians != nullptr) {
            art_data.chain_jacobians = art.chain_jacobians->data();
            art_data.chain_jacobians_count =
                static_cast<uint32_t>(art.chain_jacobians->size());
        }
        if (art.inertia_m_inv != nullptr) {
            art_data.inertia_m_inv = art.inertia_m_inv->data();
            art_data.inertia_m_inv_count =
                static_cast<uint32_t>(art.inertia_m_inv->size());
        }
        if (art.qdot != nullptr) {
            art_data.qdot = art.qdot->data();
            art_data.qdot_count = static_cast<uint32_t>(art.qdot->size());
        }
        art_data.dof_stride = art.dof_stride;
    }

    // v0.8 C6b: build the device-side particle descriptor from the host vectors.
    // Empty (the default SolveContext) -> the particle arm never fires.
    gpu::RowParticleData part_data;
    if (have_particles) {
        part_data.inv_mass = part.inv_mass->data();
        part_data.velocities = part.velocities->data();
        part_data.count = static_cast<uint32_t>(part.velocities->size());
    }

    gpu::SolveRowsWithSides(context,
                            *ctx.rows,
                            ctx.state->data(),
                            static_cast<uint32_t>(ctx.state->size()),
                            sides_ptr,
                            sides_count,
                            gpu_config,
                            art_data,
                            part_data);
#else
    (void)config;
    throw std::runtime_error(
        "UnifiedSolve requires the CUDA row solver; CPU production solving is disabled");
#endif
}

} // namespace nuka::solver
