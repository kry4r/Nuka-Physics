// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/xpbd_shape_match.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================
//
// Reverse-mode adjoint kernel for XPBDShapeMatchRow (dense_adjoint).
// One row per thread; each thread reads its inputs + the upstream seed and
// writes distinct per-input gradient outputs. D1: fixed order, no atomics.
// The differentiable law + its analytical adjoint live in the paired header
// (validated against finite differences by src/codegen/v3_validation).

#include "codegen/generated/row_class_registry.hpp"
#include "codegen/generated/xpbd_shape_match_adjoint.cuh"

#include <cstdint>

namespace nuka::solver::generated {

__global__ void xpbd_shape_match_adjoint_kernel(
    const Row* __restrict__ rows,
    const float* __restrict__ position_data,
    const float* __restrict__ goal_data,
    const float* __restrict__ stiffness_data,
    const float* __restrict__ lower_data,
    const float* __restrict__ upper_data,
    const float* __restrict__ seed_data,
    float* __restrict__ grad_position_out,
    float* __restrict__ grad_goal_out,
    float* __restrict__ grad_stiffness_out,
    uint32_t row_count)
{
    const uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= row_count) {
        return;
    }

    const Row& row = rows[row_idx];
    if (row.row_class_id != 9u) {
        return;
    }
    if (row.adjoint_kernel_id != 8u) {
        return;
    }

    XPBDShapeMatchAdjInputs in;
    in.position = position_data[row_idx];
    in.goal = goal_data[row_idx];
    in.stiffness = stiffness_data[row_idx];
    in.lower = lower_data[row_idx];
    in.upper = upper_data[row_idx];

    XPBDShapeMatchAdjGrads g =
        xpbd_shape_match_adjoint_eval(in, seed_data[row_idx]);

    grad_position_out[row_idx] = g.grad_position;
    grad_goal_out[row_idx] = g.grad_goal;
    grad_stiffness_out[row_idx] = g.grad_stiffness;
    (void)row;
}

} // namespace nuka::solver::generated
