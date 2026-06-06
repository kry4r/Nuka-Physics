// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/xpbd_cosserat_stretch_shear.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================

#include "codegen/generated/row_class_registry.hpp"

#include <cstdint>

namespace nuka::solver::generated {

__global__ void xpbd_cosserat_stretch_shear_forward_kernel(
    const Row* __restrict__ rows,
    const uint32_t* __restrict__ body_a_data,
    const uint32_t* __restrict__ body_b_data,
    const float* __restrict__ jacobian_linear_a_data,
    const float* __restrict__ jacobian_angular_a_data,
    const float* __restrict__ jacobian_linear_b_data,
    const float* __restrict__ jacobian_angular_b_data,
    const float* __restrict__ rhs_data,
    const float* __restrict__ lower_data,
    const float* __restrict__ upper_data,
    const float* __restrict__ lambda_data,
    const float* __restrict__ effective_mass_data,
    float* __restrict__ lambda_out,
    float* __restrict__ position_error_out,
    uint32_t row_count)
{
    const uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= row_count) {
        return;
    }

    const Row& row = rows[row_idx];
    if (row.row_class_id != 11u) {
        return;
    }

    (void)body_a_data;
    (void)body_b_data;
    (void)jacobian_linear_a_data;
    (void)jacobian_angular_a_data;
    (void)jacobian_linear_b_data;
    (void)jacobian_angular_b_data;
    (void)rhs_data;
    (void)lower_data;
    (void)upper_data;
    (void)lambda_data;
    (void)effective_mass_data;
    (void)lambda_out;
    (void)position_error_out;
    (void)row;

    // Behavior-equivalent row update is wired in Phase 5 with the CSR migration
    // diff-test bridge. Phase 2 deliberately emits compile/linkable stubs only.
    // Reserved strategy: body-segmented serial Jacobi per row.
}

} // namespace nuka::solver::generated
