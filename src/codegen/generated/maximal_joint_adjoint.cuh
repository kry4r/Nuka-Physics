// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/maximal_joint.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================
//
// Reverse-mode matched pair for MaximalJointRow (dense_adjoint).
//
//   primal  : lambda_new = clamp(u, lower, upper)
//             where u = lambda + effective_mass * (rhs - jv)
//   adjoint : grad_x_i = (du/dx_i) * kInsideBand * seed
//             for x_i in jv, rhs, lambda, effective_mass
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

// Differentiable inputs + params for the maximal_joint law.
struct MaximalJointAdjInputs {
    float jv = 0.0f;
    float rhs = 0.0f;
    float lambda = 0.0f;
    float effective_mass = 0.0f;
    float lower = 0.0f;
    float upper = 0.0f;
};

// Per-input gradient bundle (vᵀ·∂f/∂x). One entry per grad_input_field.
struct MaximalJointAdjGrads {
    float grad_jv = 0.0f;
    float grad_rhs = 0.0f;
    float grad_lambda = 0.0f;
    float grad_effective_mass = 0.0f;
};

// Un-clamped law u (shared interior of primal and adjoint).
__host__ __device__ __forceinline__ float
maximal_joint_unclamped_eval(const MaximalJointAdjInputs& in) {
    const float jv = in.jv;
    const float rhs = in.rhs;
    const float lambda = in.lambda;
    const float effective_mass = in.effective_mass;
    const float lower = in.lower;
    const float upper = in.upper;
    (void)jv;
    (void)rhs;
    (void)lambda;
    (void)effective_mass;
    (void)lower;
    (void)upper;
    return lambda + effective_mass * (rhs - jv);
}

// Primal forward-update f(inputs, params) -> lambda_new.
__host__ __device__ __forceinline__ float
maximal_joint_forward_eval(const MaximalJointAdjInputs& in) {
    const float u = maximal_joint_unclamped_eval(in);
    return fminf(fmaxf(u, in.lower), in.upper);
}

// Clamp sub-gradient indicator: 1 strictly inside the band, else 0.
__host__ __device__ __forceinline__ float
maximal_joint_clamp_active(const MaximalJointAdjInputs& in) {
    const float u = maximal_joint_unclamped_eval(in);
    return (u <= in.lower || u >= in.upper) ? 0.0f : 1.0f;
}

// Reverse-mode adjoint: grad_x_i = (du/dx_i) * kInsideBand * seed,
// with seed = dL/dlambda_new.
__host__ __device__ __forceinline__ MaximalJointAdjGrads
maximal_joint_adjoint_eval(const MaximalJointAdjInputs& in, float seed) {
    const float jv = in.jv;
    const float rhs = in.rhs;
    const float lambda = in.lambda;
    const float effective_mass = in.effective_mass;
    const float lower = in.lower;
    const float upper = in.upper;
    const float kClampActive = maximal_joint_clamp_active(in);

    MaximalJointAdjGrads g;
    g.grad_jv = (-effective_mass) * kClampActive * seed;
    g.grad_rhs = (effective_mass) * kClampActive * seed;
    g.grad_lambda = (1.0f) * kClampActive * seed;
    g.grad_effective_mass = ((rhs - jv)) * kClampActive * seed;
    (void)jv;
    (void)rhs;
    (void)lambda;
    (void)effective_mass;
    (void)lower;
    (void)upper;
    return g;
}

} // namespace nuka::solver::generated
