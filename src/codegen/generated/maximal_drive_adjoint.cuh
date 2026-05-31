// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/maximal_drive.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================
//
// Reverse-mode matched pair for MaximalDriveRow (dense_adjoint).
//
//   primal  : tau = clamp(u, -force_limit, +force_limit)
//             where u = drive_stiffness * (drive_target - q) - drive_damping * qdot
//   adjoint : grad_x_i = (du/dx_i) * kClampActive * seed
//             for x_i in q, qdot, drive_target, drive_stiffness, drive_damping
//
// The unclamped law `u` is the single source of truth shared by the primal,
// the saturation test, and the derivative_rules (authored as du/dx_i in the IR).
// The force-limit clamp is non-smooth at saturation; kClampActive is its
// sub-gradient indicator (1 strictly inside the band, 0 when saturated), so a
// saturated drive yields zero gradient on every input/param.
//
// Both functions are __host__ __device__ so the V3 finite-difference harness
// (src/codegen/v3_validation) validates the SAME definitions it will run on the
// device. D1: one row per thread, distinct outputs, no atomics.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nuka::solver::generated {

// Differentiable inputs + params for the maximal_drive law.
struct MaximalDriveAdjInputs {
    float q = 0.0f;
    float qdot = 0.0f;
    float drive_target = 0.0f;
    float drive_stiffness = 0.0f;
    float drive_damping = 0.0f;
    float force_limit = 0.0f;
};

// Per-input gradient bundle (vᵀ·∂f/∂x). One entry per grad_input_field.
struct MaximalDriveAdjGrads {
    float grad_q = 0.0f;
    float grad_qdot = 0.0f;
    float grad_drive_target = 0.0f;
    float grad_drive_stiffness = 0.0f;
    float grad_drive_damping = 0.0f;
};

// Un-clamped law u (shared interior of primal and adjoint).
__host__ __device__ __forceinline__ float
maximal_drive_unclamped_eval(const MaximalDriveAdjInputs& in) {
    const float q = in.q;
    const float qdot = in.qdot;
    const float drive_target = in.drive_target;
    const float drive_stiffness = in.drive_stiffness;
    const float drive_damping = in.drive_damping;
    const float force_limit = in.force_limit;
    (void)q;
    (void)qdot;
    (void)drive_target;
    (void)drive_stiffness;
    (void)drive_damping;
    (void)force_limit;
    return drive_stiffness * (drive_target - q) - drive_damping * qdot;
}

// Primal forward-update f(inputs, params) -> tau.
__host__ __device__ __forceinline__ float
maximal_drive_forward_eval(const MaximalDriveAdjInputs& in) {
    const float u = maximal_drive_unclamped_eval(in);
    if (in.force_limit > 0.0f) {
        return fminf(fmaxf(u, -in.force_limit), in.force_limit);
    }
    return u;
}

// Clamp sub-gradient indicator: 1 strictly inside the force-limit band, else 0.
__host__ __device__ __forceinline__ float
maximal_drive_clamp_active(const MaximalDriveAdjInputs& in) {
    if (in.force_limit <= 0.0f) {
        return 1.0f;  // no active limit -> always differentiable
    }
    const float u = maximal_drive_unclamped_eval(in);
    return (u <= -in.force_limit || u >= in.force_limit) ? 0.0f : 1.0f;
}

// Reverse-mode adjoint: grad_x_i = (du/dx_i) * kClampActive * seed,
// with seed = dL/dtau.
__host__ __device__ __forceinline__ MaximalDriveAdjGrads
maximal_drive_adjoint_eval(const MaximalDriveAdjInputs& in, float seed) {
    const float q = in.q;
    const float qdot = in.qdot;
    const float drive_target = in.drive_target;
    const float drive_stiffness = in.drive_stiffness;
    const float drive_damping = in.drive_damping;
    const float force_limit = in.force_limit;
    const float kClampActive = maximal_drive_clamp_active(in);

    MaximalDriveAdjGrads g;
    g.grad_q = (-drive_stiffness) * kClampActive * seed;
    g.grad_qdot = (-drive_damping) * kClampActive * seed;
    g.grad_drive_target = (drive_stiffness) * kClampActive * seed;
    g.grad_drive_stiffness = ((drive_target - q)) * kClampActive * seed;
    g.grad_drive_damping = (-qdot) * kClampActive * seed;
    (void)q;
    (void)qdot;
    (void)drive_target;
    (void)drive_stiffness;
    (void)drive_damping;
    (void)force_limit;
    return g;
}

} // namespace nuka::solver::generated
