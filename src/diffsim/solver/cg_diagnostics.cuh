#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- CG convergence diagnostics (device side) -- v0.7 p01
// ---------------------------------------------------------------------------
//
// Device-side helpers for the opt-in CG convergence side-channel. These are
// pulled INTO the CG kernel (sparse_solver_cg.cu) by lane 0 of each warp-per-
// block solve. They consume the residual the CG loop ALREADY computes via the
// existing WarpButterflySum (no new reduction, no new __shfl pattern -> the
// accumulation order is untouched and D1 is preserved) and emit per-block
// diagnostics ONLY when the destination buffer pointer is non-null.
//
// D1: every store is performed by lane 0 in fixed block order to a per-block
// slot (blk index), so two runs of the same batch produce byte-identical
// diagnostics. No atomics. With all diagnostic pointers null the kernel never
// calls these and emits the exact v0.5 bit pattern.
//
// Stall / divergence detection runs INSIDE the CG loop against running scalars
// (best-so-far residual + the previous residual) with FIXED-CONSTANT thresholds,
// so the flags are a deterministic function of the (data-dependent but run-
// invariant) residual trajectory.
// ---------------------------------------------------------------------------

#include "diffsim/sparse_solver_backend.hpp"

#include <cmath>
#include <cstdint>

namespace nuka::diffsim {

// DIVERGENCE / BREAKDOWN is a MAGNITUDE-FREE SIGN test, NOT a residual-size
// threshold. The unpreconditioned residual 2-norm ||r_k|| that CG produces is
// genuinely NON-MONOTONIC and can spike >10x ||b|| on a healthy ill-conditioned
// SPD block (measured), so NO fixed multiple of ||r|| distinguishes a benign
// transient from a real failure. The TRUE failure signature is loss of positive-
// definiteness: for an SPD A and SPD M, the CG scalars p^T A p (pAp) and r^T M^-1 r
// (rz) are STRICTLY POSITIVE until convergence. If the solve-time guard trips
// (pAp <= floor OR rz <= floor) BEFORE the residual has converged, the operator /
// preconditioner is not SPD -- that IS the divergence/breakdown flag (and doubles
// as the solver's runtime SPD detection). At genuine convergence rz->0 LEGITIMATELY,
// so the flag is gated on "not yet converged" to avoid a false positive at the win.
// All compares are on deterministic values -> D1 preserved.

// STALL is detected on the PRECONDITIONED residual rz = r^T M^-1 r -- the quantity
// CG actually drives to zero (unlike ||r||, rz does not spike). A block is flagged
// STALLED if rz fails to drop below this fraction of the previous iteration's rz
// for kStallWindow consecutive iterations while NOT yet converged (an honestly
// crawling ill-conditioned block). Fixed constants => deterministic.
__device__ constexpr double kStallProgress = 0.9999;  // < 0.01% rz reduction/step
__device__ constexpr uint32_t kStallWindow = 6u;

// Running per-block diagnostic accumulator (a lane-0 local; never cross-warp).
// Tracks the trajectory needed for the stall / divergence flags + history.
struct CgDiagState {
    double r0_sq = 0.0;       // initial ||r||^2 (== ||b||^2 here, x0 = 0)
    double r_min_sq = 0.0;    // smallest ||r||^2 seen (a reported statistic)
    double rz_prev = 0.0;     // previous iteration's rz (for the stall window)
    uint32_t stall_run = 0u;  // consecutive low-progress iterations
    uint8_t status = kCgStatusNone;
};

// Initialize the diag state from the initial squared residual r0_sq (== ||b||^2)
// and the initial preconditioned residual rz0 (== r0^T M^-1 r0).
__device__ __forceinline__ void CgDiagInit(CgDiagState& s, double r0_sq,
                                           double rz0) {
    s.r0_sq = r0_sq;
    s.r_min_sq = r0_sq;
    s.rz_prev = rz0;
    s.stall_run = 0u;
    s.status = kCgStatusNone;
}

// Per-iteration update (lane 0 only). Observes the new squared residual rr, the
// new preconditioned residual rz, and whether the SPD guard broke down this
// iteration, at iteration `iter`. Maintains the running min + the stall window +
// the divergence/breakdown flag, and optionally records ||r_k||/||b|| into the
// history row.
//
//   rr         : ||r||^2 AFTER this iteration (the loop's early-exit value).
//   rz         : r^T M^-1 r AFTER this iteration (the preconditioned residual).
//   broke_down : the solve-time SPD guard tripped this iter (pAp<=floor||rz<=floor).
//   converged  : the deterministic tol early-exit fired this iter.
//   iter       : 0-based iteration index.
//   hist       : optional float* row (length history_cap); null => skip.
//   history_cap: capacity of `hist`.
__device__ __forceinline__ void CgDiagOnIter(CgDiagState& s, double rr, double rz,
                                             bool broke_down, bool converged,
                                             uint32_t iter, float* hist,
                                             uint32_t history_cap) {
    // Record ||r_k||/||b|| into the history row (capped). sqrt of the ratio; the
    // denominator floor mirrors the host summarizer's tiny-||b|| guard.
    if (hist != nullptr && iter < history_cap) {
        const double denom = (s.r0_sq > 0.0) ? s.r0_sq : 1.0;
        hist[iter] = static_cast<float>(sqrt(rr / denom));
    }
    if (rr < s.r_min_sq) s.r_min_sq = rr;
    // Divergence / breakdown: the SPD guard tripped BEFORE converging => the
    // operator/preconditioner lost positive-definiteness (a non-SPD system). At
    // genuine convergence rz->0 legitimately, hence the not-converged gate.
    if (broke_down && !converged) {
        s.status |= kCgStatusDiverged;
    }
    // Stall: the PRECONDITIONED residual rz failed to make progress for a sustained
    // run while not yet converged. Warm-up (iter 0) seeds rz_prev.
    if (iter > 0u && !converged) {
        if (rz >= kStallProgress * s.rz_prev) {
            ++s.stall_run;
            if (s.stall_run >= kStallWindow) s.status |= kCgStatusStalled;
        } else {
            s.stall_run = 0u;
        }
    }
    s.rz_prev = rz;
}

// Finalize + write the per-block report (lane 0, fixed block slot `blk`). `rr_final`
// is the last squared residual the loop produced; `iters_run` the count actually
// executed; `converged` whether the tol early-exit fired (vs budget exhaustion).
__device__ __forceinline__ void CgDiagFinalize(const CgDiagState& s,
                                               const CgConvergenceReport& rep,
                                               uint32_t blk, double rr_final,
                                               uint32_t iters_run,
                                               bool converged) {
    uint8_t status = s.status;
    status |= converged ? kCgStatusConverged : kCgStatusMaxIterReached;
    if (rep.iters_run != nullptr) rep.iters_run[blk] = iters_run;
    if (rep.final_rel_resid != nullptr) {
        const double denom = (s.r0_sq > 0.0) ? s.r0_sq : 1.0;
        rep.final_rel_resid[blk] = static_cast<float>(sqrt(rr_final / denom));
    }
    if (rep.status != nullptr) rep.status[blk] = status;
}

}  // namespace nuka::diffsim
