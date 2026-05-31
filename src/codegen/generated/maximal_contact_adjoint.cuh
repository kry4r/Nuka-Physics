// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/maximal_contact.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================
//
// Reverse-mode matched pair for MaximalContactRow (stop_grad_on_event).
//
//   primal  : lambda_new = clamp(u, lower, upper)
//             where u = lambda + effective_mass * (rhs - jv)
//   adjoint : grad_x_i = (du/dx_i) * kInsideBand * seed
//             for x_i in jv, rhs, lambda, effective_mass
//
// The unclamped law `u` is the single source of truth shared by the primal, the
// in-band test, and the derivative_rules (authored as du/dx_i in the IR).
// EVENT (stop_grad_on_event): the clamp encodes an active-set / friction-cone
// event. kInsideBand is the stop-gradient indicator -- 1 strictly inside the box
// (the row was UNCLAMPED, gradient flows), 0 at/over a band edge (the event bit
// is set, gradient is stopped). The __global__ kernel ADDITIONALLY ANDs the
// stored Row.event_flag so a row flagged as an event contributes zero gradient.
//
// Both functions are __host__ __device__ so the V3 finite-difference harness
// (src/codegen/v3_validation) validates the SAME definitions it will run on the
// device. D1: one row per thread, distinct outputs, no atomics.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nuka::solver::generated {

// Differentiable inputs + params for the maximal_contact law.
struct MaximalContactAdjInputs {
    float jv = 0.0f;
    float rhs = 0.0f;
    float lambda = 0.0f;
    float effective_mass = 0.0f;
    float lower = 0.0f;
    float upper = 0.0f;
};

// Per-input gradient bundle (vᵀ·∂f/∂x). One entry per grad_input_field.
struct MaximalContactAdjGrads {
    float grad_jv = 0.0f;
    float grad_rhs = 0.0f;
    float grad_lambda = 0.0f;
    float grad_effective_mass = 0.0f;
};

// Un-clamped law u (shared interior of primal and adjoint).
__host__ __device__ __forceinline__ float
maximal_contact_unclamped_eval(const MaximalContactAdjInputs& in) {
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
maximal_contact_forward_eval(const MaximalContactAdjInputs& in) {
    const float u = maximal_contact_unclamped_eval(in);
    return fminf(fmaxf(u, in.lower), in.upper);
}

// Inside-band (stop-gradient) indicator: 1 strictly inside the box [lower, upper]
// (row UNCLAMPED), 0 at/over a band edge (active-set / friction-cone event).
__host__ __device__ __forceinline__ float
maximal_contact_clamp_active(const MaximalContactAdjInputs& in) {
    const float u = maximal_contact_unclamped_eval(in);
    return (u <= in.lower || u >= in.upper) ? 0.0f : 1.0f;
}

// Reverse-mode adjoint: grad_x_i = (du/dx_i) * kInsideBand * seed,
// with seed = dL/dlambda_new.
// kInsideBand here is the stop-gradient on the active-set / friction-cone event:
// a row clamped at a band edge contributes zero gradient (the event bit). The
// host pair validates this; the __global__ kernel further ANDs Row.event_flag.
__host__ __device__ __forceinline__ MaximalContactAdjGrads
maximal_contact_adjoint_eval(const MaximalContactAdjInputs& in, float seed) {
    const float jv = in.jv;
    const float rhs = in.rhs;
    const float lambda = in.lambda;
    const float effective_mass = in.effective_mass;
    const float lower = in.lower;
    const float upper = in.upper;
    const float kClampActive = maximal_contact_clamp_active(in);

    MaximalContactAdjGrads g;
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
