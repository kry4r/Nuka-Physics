// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/xpbd_volume.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================
//
// Reverse-mode adjoint kernel for XPBDVolumeRow (dense_adjoint).
// One row per thread; each thread reads its inputs + the upstream seed and
// writes distinct per-input gradient outputs. D1: fixed order, no atomics.
// The differentiable law + its analytical adjoint live in the paired header
// (validated against finite differences by src/codegen/v3_validation).

#include "codegen/generated/row_class_registry.hpp"
#include "codegen/generated/xpbd_volume_adjoint.cuh"

#include <cstdint>

namespace nuka::solver::generated {

__global__ void xpbd_volume_adjoint_kernel(
    const Row* __restrict__ rows,
    const float* __restrict__ C_data,
    const float* __restrict__ lambda_data,
    const float* __restrict__ effective_mass_data,
    const float* __restrict__ alpha_tilde_data,
    const float* __restrict__ lower_data,
    const float* __restrict__ upper_data,
    const float* __restrict__ seed_data,
    float* __restrict__ grad_C_out,
    float* __restrict__ grad_lambda_out,
    float* __restrict__ grad_effective_mass_out,
    float* __restrict__ grad_alpha_tilde_out,
    uint32_t row_count)
{
    const uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= row_count) {
        return;
    }

    const Row& row = rows[row_idx];
    if (row.row_class_id != 8u) {
        return;
    }
    if (row.adjoint_kernel_id != 7u) {
        return;
    }

    XPBDVolumeAdjInputs in;
    in.C = C_data[row_idx];
    in.lambda = lambda_data[row_idx];
    in.effective_mass = effective_mass_data[row_idx];
    in.alpha_tilde = alpha_tilde_data[row_idx];
    in.lower = lower_data[row_idx];
    in.upper = upper_data[row_idx];

    XPBDVolumeAdjGrads g =
        xpbd_volume_adjoint_eval(in, seed_data[row_idx]);

    grad_C_out[row_idx] = g.grad_C;
    grad_lambda_out[row_idx] = g.grad_lambda;
    grad_effective_mass_out[row_idx] = g.grad_effective_mass;
    grad_alpha_tilde_out[row_idx] = g.grad_alpha_tilde;
    (void)row;
}

} // namespace nuka::solver::generated
