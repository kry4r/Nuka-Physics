// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/xpbd_shape_match.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================
//
// Reverse-mode matched pair for XPBDShapeMatchRow (dense_adjoint).
//
//   primal  : position_new = clamp(u, lower, upper)
//             where u = position + stiffness * (goal - position)
//   adjoint : grad_x_i = (du/dx_i) * kInsideBand * seed
//             for x_i in position, goal, stiffness
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

// Differentiable inputs + params for the xpbd_shape_match law.
struct XPBDShapeMatchAdjInputs {
    float position = 0.0f;
    float goal = 0.0f;
    float stiffness = 0.0f;
    float lower = 0.0f;
    float upper = 0.0f;
};

// Per-input gradient bundle (vᵀ·∂f/∂x). One entry per grad_input_field.
struct XPBDShapeMatchAdjGrads {
    float grad_position = 0.0f;
    float grad_goal = 0.0f;
    float grad_stiffness = 0.0f;
};

// Un-clamped law u (shared interior of primal and adjoint).
__host__ __device__ __forceinline__ float
xpbd_shape_match_unclamped_eval(const XPBDShapeMatchAdjInputs& in) {
    const float position = in.position;
    const float goal = in.goal;
    const float stiffness = in.stiffness;
    const float lower = in.lower;
    const float upper = in.upper;
    (void)position;
    (void)goal;
    (void)stiffness;
    (void)lower;
    (void)upper;
    return position + stiffness * (goal - position);
}

// Primal forward-update f(inputs, params) -> position_new.
__host__ __device__ __forceinline__ float
xpbd_shape_match_forward_eval(const XPBDShapeMatchAdjInputs& in) {
    const float u = xpbd_shape_match_unclamped_eval(in);
    return fminf(fmaxf(u, in.lower), in.upper);
}

// Clamp sub-gradient indicator: 1 strictly inside the band, else 0.
__host__ __device__ __forceinline__ float
xpbd_shape_match_clamp_active(const XPBDShapeMatchAdjInputs& in) {
    const float u = xpbd_shape_match_unclamped_eval(in);
    return (u <= in.lower || u >= in.upper) ? 0.0f : 1.0f;
}

// Reverse-mode adjoint: grad_x_i = (du/dx_i) * kInsideBand * seed,
// with seed = dL/dposition_new.
__host__ __device__ __forceinline__ XPBDShapeMatchAdjGrads
xpbd_shape_match_adjoint_eval(const XPBDShapeMatchAdjInputs& in, float seed) {
    const float position = in.position;
    const float goal = in.goal;
    const float stiffness = in.stiffness;
    const float lower = in.lower;
    const float upper = in.upper;
    const float kClampActive = xpbd_shape_match_clamp_active(in);

    XPBDShapeMatchAdjGrads g;
    g.grad_position = (1.0f - stiffness) * kClampActive * seed;
    g.grad_goal = (stiffness) * kClampActive * seed;
    g.grad_stiffness = ((goal - position)) * kClampActive * seed;
    (void)position;
    (void)goal;
    (void)stiffness;
    (void)lower;
    (void)upper;
    return g;
}

} // namespace nuka::solver::generated
