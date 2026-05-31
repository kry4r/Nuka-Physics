// V3 finite-difference validation harness for generated reverse-mode adjoints.
//
// The codegen pipeline emits, per row class, a matched (primal f, analytical
// adjoint vᵀ·∂f/∂x) pair as __host__ __device__ device functions. This harness
// proves the analytical adjoint agrees with a central-difference numerical
// Jacobian of the SAME primal, within a relative tolerance. It is the V3 spec's
// RowClassFdValidator (Phase-1 Task 5.1.5), instantiated here for the
// maximal_drive (dense_adjoint) vertical slice.
//
// For a scalar output tau = f(x), the adjoint with seed v=1 returns exactly the
// row of partials [∂tau/∂x_i]; the numerical Jacobian central-differences each
// x_i. We compare element-by-element. R9: the force-limit clamp is non-smooth at
// saturation, so the 1e-3 comparison samples cases strictly inside the band; a
// separate check asserts the adjoint is the correct sub-gradient (exactly zero)
// when the drive is saturated.

#pragma once

#include <cstdint>

namespace nuka::codegen::v3 {

// Aggregate finite-difference vs analytical-adjoint result over a batch.
struct FdValidationResult {
    float max_rel_err = 0.0f;       // worst element-wise relative error
    uint32_t worst_case_index = 0u; // batch index achieving max_rel_err
    uint32_t worst_param_index = 0u;// which grad component (0..4) was worst
    uint32_t case_count = 0u;       // number of cases evaluated
};

// Sampling bounds for one maximal_drive FD test batch. Defaults keep every
// central-difference stencil strictly inside the force-limit band and keep the
// gain-gradient partials (which scale with (target-q) and qdot) away from zero
// so relative error is well-conditioned.
struct MaximalDriveFdConfig {
    uint32_t num_cases = 100u;
    uint64_t seed = 0xC0FFEEu;
    // Central-difference step δ. The PD law is piecewise-LINEAR in every
    // differentiable variable, so central-difference truncation error is exactly
    // zero in exact arithmetic; the only residual is float32 cancellation, which
    // SHRINKS as δ grows. δ=0.1 keeps the whole stencil far inside the band
    // (worst excursion Kp*δ <= 2.5 << force_limit=1000) while driving the
    // cancellation residual ~2 orders below the 1e-3 gate.
    float fd_delta = 1.0e-1f;

    // Sample ranges (magnitudes chosen so |u| stays well below force_limit even
    // after the worst-case stencil excursion across all five variables).
    float q_abs_max = 0.5f;
    float qdot_abs_min = 0.05f;     // away-from-zero floor (well-conditioned Kd grad)
    float qdot_abs_max = 0.5f;
    float target_gap_min = 0.05f;   // |drive_target - q| floor (Kp grad)
    float target_gap_max = 0.5f;
    float kp_min = 5.0f;
    float kp_max = 25.0f;
    float kd_min = 0.2f;
    float kd_max = 2.0f;
    float force_limit = 1000.0f;    // large -> stencil stays interior
};

// Runs the away-from-clamp FD validation on the maximal_drive primal/adjoint
// pair. Numerical Jacobian computed in double for headroom; comparison in the
// engine's float domain. Pure host evaluation of the __host__ __device__ pair.
FdValidationResult ValidateMaximalDriveFd(const MaximalDriveFdConfig& config);

// Worst-case adjoint magnitude when the drive is driven strictly PAST the
// force limit (saturated). The clamp sub-gradient is zero, so every analytical
// grad component must be exactly 0. Returns the max |grad| over a saturated
// batch (expected: 0). Caller asserts == 0.
float MaxSaturatedAdjointMagnitude(const MaximalDriveFdConfig& config);

} // namespace nuka::codegen::v3
