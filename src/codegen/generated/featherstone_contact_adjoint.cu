// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/featherstone_contact.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================
//
// Reverse-mode adjoint kernel for FeatherstoneContactRow (stop_grad_on_event).
// One row per thread; each thread reads its inputs + the upstream seed and
// writes distinct per-input gradient outputs. D1: fixed order, no atomics.
// The differentiable law + its analytical adjoint live in the paired header
// (validated against finite differences by src/codegen/v3_validation).
//
// stop_grad_on_event: in addition to the in-band (active-set / friction-cone)
// indicator computed inside the analytical adjoint, this kernel zeros the whole
// gradient bundle when the row's stored event bit (Row.event_flag_field) is set,
// so a row tagged as an event contributes no gradient. Per p01 SCOPE: this kernel
// writes ONLY distinct per-row outputs (no cross-row scatter / no float atomics);
// the cross-DOF reduction belongs to p02's full multi-row backward.

#include "codegen/generated/row_class_registry.hpp"
#include "codegen/generated/featherstone_contact_adjoint.cuh"

#include <cstdint>

namespace nuka::solver::generated {

__global__ void featherstone_contact_adjoint_kernel(
    const Row* __restrict__ rows,
    const float* __restrict__ jv_data,
    const float* __restrict__ rhs_data,
    const float* __restrict__ lambda_data,
    const float* __restrict__ effective_mass_data,
    const float* __restrict__ lower_data,
    const float* __restrict__ upper_data,
    const float* __restrict__ seed_data,
    float* __restrict__ grad_jv_out,
    float* __restrict__ grad_rhs_out,
    float* __restrict__ grad_lambda_out,
    float* __restrict__ grad_effective_mass_out,
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
    if (row.adjoint_kernel_id != 4u) {
        return;
    }

    FeatherstoneContactAdjInputs in;
    in.jv = jv_data[row_idx];
    in.rhs = rhs_data[row_idx];
    in.lambda = lambda_data[row_idx];
    in.effective_mass = effective_mass_data[row_idx];
    in.lower = lower_data[row_idx];
    in.upper = upper_data[row_idx];

    FeatherstoneContactAdjGrads g =
        featherstone_contact_adjoint_eval(in, seed_data[row_idx]);

    // Stop-gradient on a recorded event: zero the whole bundle when the row's
    // event bit is set (the in-band indicator inside _adjoint_eval already
    // stops the gradient at a freshly-detected band edge; this honours an
    // event recorded upstream on the forward row).
    if (row.event_flag_field != 0u) {
        FeatherstoneContactAdjGrads zero;
        g = zero;
    }
    grad_jv_out[row_idx] = g.grad_jv;
    grad_rhs_out[row_idx] = g.grad_rhs;
    grad_lambda_out[row_idx] = g.grad_lambda;
    grad_effective_mass_out[row_idx] = g.grad_effective_mass;
    (void)row;
}

} // namespace nuka::solver::generated
