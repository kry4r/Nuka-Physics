// ---------------------------------------------------------------------------
// nuka::diffsim -- CG convergence diagnostics (host summarizer) -- v0.7 p01
// ---------------------------------------------------------------------------
//
// Host-side aggregation of a downloaded CgConvergenceReport into a compact
// CgDiagnosticsSummary (max / total iters, worst residual, converged / stalled /
// diverged counts). Pure CPU; the device buffers are produced by the opt-in
// diagnostics side-channel in sparse_solver_cg.cu. Deterministic: a fixed-order
// pass over block_count entries.
// ---------------------------------------------------------------------------

#include "diffsim/solver/cg_diagnostics.hpp"

#include <algorithm>

namespace nuka::diffsim {

CgDiagnosticsSummary SummarizeCgDiagnostics(uint32_t block_count,
                                            const uint32_t* iters_run,
                                            const float* final_rel_resid,
                                            const uint8_t* status,
                                            const uint32_t* block_dim) {
    CgDiagnosticsSummary s;
    for (uint32_t blk = 0u; blk < block_count; ++blk) {
        // Skip empty (n==0) blocks when block_dim is provided -- they do not
        // represent a real solve.
        if (block_dim != nullptr && block_dim[blk] == 0u) continue;
        ++s.blocks;
        if (iters_run != nullptr) {
            const uint32_t it = iters_run[blk];
            s.max_iters_run = std::max(s.max_iters_run, it);
            s.total_iters_run += it;
        }
        if (final_rel_resid != nullptr) {
            s.max_final_rel_resid =
                std::max(s.max_final_rel_resid, final_rel_resid[blk]);
        }
        if (status != nullptr) {
            const uint8_t st = status[blk];
            if (st & kCgStatusConverged) ++s.converged_blocks;
            if (st & kCgStatusMaxIterReached) ++s.maxiter_blocks;
            if (st & kCgStatusStalled) ++s.stalled_blocks;
            if (st & kCgStatusDiverged) ++s.diverged_blocks;
        }
    }
    return s;
}

}  // namespace nuka::diffsim
