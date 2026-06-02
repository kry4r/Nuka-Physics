// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/xpbd_bend.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================
//
// Reverse-mode matched pair for XPBDBendRow (dense_adjoint).
//
//   primal  : lambda_new = clamp(u, lower, upper)
//             where u = lambda + effective_mass * (-C - alpha_tilde * lambda)
//   adjoint : grad_x_i = (du/dx_i) * kInsideBand * seed
//             for x_i in C, lambda, effective_mass, alpha_tilde
//
// The unclamped law `u` is the single source of truth shared by the primal, the
// in-band test, and the derivative_rules (authored as du/dx_i in the IR).
// The clamp is non-smooth at saturation; kInsideBand is its sub-gradient
// indicator (1 strictly inside the band, 0 when saturated), so a saturated row
// yields zero gradient on every input/param.
//
// Both functions are __host__ __device__ so the V3 finite-difference harness
// (src/codegen/v3_validation) validates the SAME definitions it will run on the
// device. D1: one row per thread, distinct outputs, no atomics.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nuka::solver::generated {

// Differentiable inputs + params for the xpbd_bend law.
struct XPBDBendAdjInputs {
    float C = 0.0f;
    float lambda = 0.0f;
    float effective_mass = 0.0f;
    float alpha_tilde = 0.0f;
    float lower = 0.0f;
    float upper = 0.0f;
};

// Per-input gradient bundle (vᵀ·∂f/∂x). One entry per grad_input_field.
struct XPBDBendAdjGrads {
    float grad_C = 0.0f;
    float grad_lambda = 0.0f;
    float grad_effective_mass = 0.0f;
    float grad_alpha_tilde = 0.0f;
};

// Un-clamped law u (shared interior of primal and adjoint).
__host__ __device__ __forceinline__ float
xpbd_bend_unclamped_eval(const XPBDBendAdjInputs& in) {
    const float C = in.C;
    const float lambda = in.lambda;
    const float effective_mass = in.effective_mass;
    const float alpha_tilde = in.alpha_tilde;
    const float lower = in.lower;
    const float upper = in.upper;
    (void)C;
    (void)lambda;
    (void)effective_mass;
    (void)alpha_tilde;
    (void)lower;
    (void)upper;
    return lambda + effective_mass * (-C - alpha_tilde * lambda);
}

// Primal forward-update f(inputs, params) -> lambda_new.
__host__ __device__ __forceinline__ float
xpbd_bend_forward_eval(const XPBDBendAdjInputs& in) {
    const float u = xpbd_bend_unclamped_eval(in);
    return fminf(fmaxf(u, in.lower), in.upper);
}

// Clamp sub-gradient indicator: 1 strictly inside the band, else 0.
__host__ __device__ __forceinline__ float
xpbd_bend_clamp_active(const XPBDBendAdjInputs& in) {
    const float u = xpbd_bend_unclamped_eval(in);
    return (u <= in.lower || u >= in.upper) ? 0.0f : 1.0f;
}

// Reverse-mode adjoint: grad_x_i = (du/dx_i) * kInsideBand * seed,
// with seed = dL/dlambda_new.
__host__ __device__ __forceinline__ XPBDBendAdjGrads
xpbd_bend_adjoint_eval(const XPBDBendAdjInputs& in, float seed) {
    const float C = in.C;
    const float lambda = in.lambda;
    const float effective_mass = in.effective_mass;
    const float alpha_tilde = in.alpha_tilde;
    const float lower = in.lower;
    const float upper = in.upper;
    const float kClampActive = xpbd_bend_clamp_active(in);

    XPBDBendAdjGrads g;
    g.grad_C = (-effective_mass) * kClampActive * seed;
    g.grad_lambda = (1.0f - effective_mass * alpha_tilde) * kClampActive * seed;
    g.grad_effective_mass = ((-C - alpha_tilde * lambda)) * kClampActive * seed;
    g.grad_alpha_tilde = (-effective_mass * lambda) * kClampActive * seed;
    (void)C;
    (void)lambda;
    (void)effective_mass;
    (void)alpha_tilde;
    (void)lower;
    (void)upper;
    return g;
}

} // namespace nuka::solver::generated
