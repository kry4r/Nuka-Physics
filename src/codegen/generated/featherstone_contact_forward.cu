// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/featherstone_contact.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================

#include "codegen/generated/row_class_registry.hpp"

#include <cstdint>

namespace nuka::solver::generated {

__global__ void featherstone_contact_forward_kernel(
    const Row* __restrict__ rows,
    const uint32_t* __restrict__ maximal_body_data,
    const uint32_t* __restrict__ articulation_id_data,
    const uint32_t* __restrict__ link_index_data,
    const float* __restrict__ jacobian_linear_body_data,
    const float* __restrict__ jacobian_angular_body_data,
    const float* __restrict__ jacobian_chain_scalar_data,
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
    if (row.row_class_id != 3u) {
        return;
    }

    (void)maximal_body_data;
    (void)articulation_id_data;
    (void)link_index_data;
    (void)jacobian_linear_body_data;
    (void)jacobian_angular_body_data;
    (void)jacobian_chain_scalar_data;
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
    // Reserved friction model: 4-side friction pyramid.
}

} // namespace nuka::solver::generated
