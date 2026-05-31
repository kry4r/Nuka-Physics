// V3 finite-difference validation for the box-clamp PGS row classes
// (maximal_joint = dense_adjoint; maximal_contact / featherstone_contact =
// stop_grad_on_event). All three generated adjoints share the single-row PGS
// law  u = lambda + effective_mass*(rhs - jv);  lambda_new = clamp(u, lower, upper),
// so one generic, template-over-the-generated-pair harness validates all three.
//
// As in the maximal_drive slice (fd_harness.hpp), the away-from-boundary batch
// central-differences the generated primal and compares to the generated
// analytical adjoint (seed v=1 -> the adjoint returns the row of partials). For
// the two stop_grad_on_event classes there is a SEPARATE check (R5) that AT/over
// the event (box) boundary the analytical adjoint returns exactly zero -- the
// stop-gradient. maximal_joint is a bilateral equality row (unbounded box), so it
// only runs the away-from-boundary + determinism checks (no event case), exactly
// like the dense drive slice.

#pragma once

#include <cstdint>

namespace nuka::codegen::v3 {

// Aggregate finite-difference vs analytical-adjoint result over a batch.
struct BoxPgsFdResult {
    float max_rel_err = 0.0f;        // worst element-wise relative error
    uint32_t worst_case_index = 0u;  // batch index achieving max_rel_err
    uint32_t worst_param_index = 0u; // which grad component (0..3) was worst
    uint32_t case_count = 0u;        // number of cases evaluated
};

// Sampling bounds for one box-PGS FD batch. Defaults keep every central-
// difference stencil strictly INSIDE the box (for the bounded contact rows) and
// keep the effective_mass partial (= rhs - jv) away from zero so its relative
// error stays well-conditioned -- mirroring the drive harness's R9 sampling.
struct BoxPgsFdConfig {
    uint32_t num_cases = 100u;
    uint64_t seed = 0xC0FFEEu;
    // Central-difference step. The PGS update is piecewise-LINEAR in every
    // differentiable variable, so central-difference truncation error is exactly
    // zero in exact arithmetic; the only residual is float32 cancellation.
    float fd_delta = 1.0e-2f;

    // Sample ranges. lambda + m_eff*(rhs - jv) is the unclamped impulse u; we
    // sample so |u| sits comfortably inside the box even after the worst-case
    // stencil excursion across all four variables.
    float jv_abs_max = 0.5f;
    float rhs_abs_max = 0.5f;
    float lambda_min = 1.0f;       // away-from-edge floor (interior of [0, hi])
    float lambda_max = 3.0f;
    float meff_min = 0.5f;         // away-from-zero floor (well-conditioned grads)
    float meff_max = 2.0f;

    // Box for the away-from-boundary batch. Wide enough that the sampled u and
    // its stencil stay strictly interior (asserted per case).
    float lower = -50.0f;
    float upper = 50.0f;
};

// --- maximal_joint (dense_adjoint, bilateral equality / unbounded box) -------
BoxPgsFdResult ValidateMaximalJointFd(const BoxPgsFdConfig& config);

// --- maximal_contact (stop_grad_on_event) ------------------------------------
BoxPgsFdResult ValidateMaximalContactFd(const BoxPgsFdConfig& config);
// Worst analytical-adjoint magnitude when the row is driven strictly PAST a box
// edge (the active-set / friction-cone event). The stop-gradient indicator is
// zero, so every analytical grad component must be exactly 0. Returns the max
// |grad| over a saturated batch (expected: 0). Caller asserts == 0.
float MaxEventStoppedMaximalContactMagnitude(const BoxPgsFdConfig& config);

// --- featherstone_contact (stop_grad_on_event, articulated chain J) ----------
BoxPgsFdResult ValidateFeatherstoneContactFd(const BoxPgsFdConfig& config);
float MaxEventStoppedFeatherstoneContactMagnitude(const BoxPgsFdConfig& config);

} // namespace nuka::codegen::v3
