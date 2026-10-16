#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- OPT-IN render timing probe for the two-level tracer. Splits ONE
// RenderFrame call's on-stream GPU cost into TLAS-update (build-or-refit) vs the
// closest-hit kernel, via CUDA events on the render stream. DEFAULT OFF: when
// disabled the tracer records nothing and runs byte-identically (no events, no
// extra syncs) so the FP64 goldens stay byte-exact. This instruments the ONE
// general render path; it does not fork or duplicate it.
// ---------------------------------------------------------------------------

namespace nuka::rt {

// Accumulated GPU milliseconds for sensor and beauty rendering, split per stage.
// `calls` counts launches since the last TakeRenderTiming.
struct RenderTiming {
    double tlas_ms = 0.0;    // EnsureFrameTlas: refresh tables + build-or-refit
    double kernel_ms = 0.0;  // closest-hit or stochastic beauty trace
    unsigned long long calls = 0u;
};

// Enable/disable the probe (process-global). Off by default.
void SetRenderTimingEnabled(bool enabled);
bool RenderTimingEnabled();

// Read the accumulated split and zero it for the next measurement window.
RenderTiming TakeRenderTiming();

}  // namespace nuka::rt
