#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- CG convergence diagnostics (host surface) -- v0.7 p01
// ---------------------------------------------------------------------------
//
// The general (non-IFT) sparse-solver path needs to OBSERVE convergence on
// larger / worse-conditioned blocks without changing the bit pattern of the
// solve itself. This header is the HOST-side surface of that opt-in side-channel:
// the per-block report struct (CgConvergenceReport) lives in
// sparse_solver_backend.hpp next to SolveParams; here we add small host helpers
// to (a) size + allocate the device diagnostic buffers and (b) summarize a
// downloaded report (max iters, worst residual, any stalled / diverged block).
//
// DETERMINISM CONTRACT: the diagnostics are written by a FIXED-ORDER, per-block,
// lane-0 store path that reuses the existing WarpButterflySum residual already
// computed by the CG loop. They add NO new reduction and NO cross-block coupling,
// so they are two-run byte-exact like the solve. The presence of a non-null
// diagnostics pointer does NOT alter x: the math is identical, only an extra
// (independent) store is performed. See cg_diagnostics.cuh for the device side.
// ---------------------------------------------------------------------------

#include "diffsim/sparse_solver_backend.hpp"

#include <cstdint>

namespace nuka::diffsim {

// Host-side summary of a downloaded CgConvergenceReport (block_count entries of
// iters_run / final_rel_resid / status). Pure aggregation; no device work.
struct CgDiagnosticsSummary {
    uint32_t blocks = 0u;            // number of non-empty blocks summarized
    uint32_t max_iters_run = 0u;     // worst iteration count over the batch
    uint32_t total_iters_run = 0u;   // sum of iterations (for an average)
    float max_final_rel_resid = 0.0f;  // worst final relative residual
    uint32_t converged_blocks = 0u;  // blocks whose tol early-exit fired
    uint32_t maxiter_blocks = 0u;    // blocks that exhausted the budget
    uint32_t stalled_blocks = 0u;    // blocks flagged kCgStatusStalled
    uint32_t diverged_blocks = 0u;   // blocks flagged kCgStatusDiverged
};

// Aggregate HOST-resident report arrays (each length block_count; any may be
// null to skip that channel). `block_dim` (length block_count, null => count
// all) is used to skip empty (n==0) blocks. Deterministic, allocation-free.
CgDiagnosticsSummary SummarizeCgDiagnostics(uint32_t block_count,
                                            const uint32_t* iters_run,
                                            const float* final_rel_resid,
                                            const uint8_t* status,
                                            const uint32_t* block_dim);

}  // namespace nuka::diffsim
