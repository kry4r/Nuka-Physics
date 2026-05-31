// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/maximal_drive.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================
//
// Reverse-mode adjoint kernel for MaximalDriveRow (dense_adjoint).
// One row per thread; each thread reads its inputs + the upstream seed and
// writes distinct per-input gradient outputs. D1: fixed order, no atomics.
// The differentiable law + its analytical adjoint live in the paired header
// (validated against finite differences by src/codegen/v3_validation).

#include "codegen/generated/row_class_registry.hpp"
#include "codegen/generated/maximal_drive_adjoint.cuh"

#include <cstdint>

namespace nuka::solver::generated {

__global__ void maximal_drive_adjoint_kernel(
    const Row* __restrict__ rows,
    const float* __restrict__ q_data,
    const float* __restrict__ qdot_data,
    const float* __restrict__ drive_target_data,
    const float* __restrict__ drive_stiffness_data,
    const float* __restrict__ drive_damping_data,
    const float* __restrict__ force_limit_data,
    const float* __restrict__ seed_data,
    float* __restrict__ grad_q_out,
    float* __restrict__ grad_qdot_out,
    float* __restrict__ grad_drive_target_out,
    float* __restrict__ grad_drive_stiffness_out,
    float* __restrict__ grad_drive_damping_out,
    uint32_t row_count)
{
    const uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= row_count) {
        return;
    }

    const Row& row = rows[row_idx];
    if (row.row_class_id != 2u) {
        return;
    }
    if (row.adjoint_kernel_id != 3u) {
        return;
    }

    MaximalDriveAdjInputs in;
    in.q = q_data[row_idx];
    in.qdot = qdot_data[row_idx];
    in.drive_target = drive_target_data[row_idx];
    in.drive_stiffness = drive_stiffness_data[row_idx];
    in.drive_damping = drive_damping_data[row_idx];
    in.force_limit = force_limit_data[row_idx];

    MaximalDriveAdjGrads g =
        maximal_drive_adjoint_eval(in, seed_data[row_idx]);

    grad_q_out[row_idx] = g.grad_q;
    grad_qdot_out[row_idx] = g.grad_qdot;
    grad_drive_target_out[row_idx] = g.grad_drive_target;
    grad_drive_stiffness_out[row_idx] = g.grad_drive_stiffness;
    grad_drive_damping_out[row_idx] = g.grad_drive_damping;
    (void)row;
}

} // namespace nuka::solver::generated
