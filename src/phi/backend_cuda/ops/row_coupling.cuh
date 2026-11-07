#pragma once

#include <cstdint>

#include "math/cuda_vec_ops.cuh"
#include "math/symmetric_mat3.hpp"
#include "phi/backend_cuda/ops/articulation_mass.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"

namespace nuka::phi::nkops {

// Forward solves L^-1 J of a row's articulated sides; couplings are D^-1-weighted dots of them.
struct RowForward {
    const float* side[2] = {nullptr, nullptr};
};

__device__ inline RowForward ForwardRowSides(const NkRow& row, uint32_t row_index,
    const float* __restrict__ chain_jacobian, const float* __restrict__ chain_jacobian_b,
    const float* __restrict__ mass_factor, uint32_t dof_stride, float* storage) {
    RowForward forward;
    const NkRowSide sides[2] = {row.a, row.b};
    for (uint32_t s = 0u; s < 2u; ++s) {
        if (sides[s].kind != nk::kNkSideArtic) continue;
        float* const y = storage + size_t{s} * dof_stride;
        ForwardMass(mass_factor + size_t{sides[s].index} * dof_stride * dof_stride, dof_stride,
                    (s == 0u ? chain_jacobian : chain_jacobian_b) + size_t{row_index} * dof_stride, nullptr, y);
        forward.side[s] = y;
    }
    return forward;
}

__device__ inline float PairDrivenSideCoupling(
    const NkRowSide& lhs, const float* lhs_forward,
    const NkRowSide& rhs, const float* rhs_forward,
    const float* __restrict__ mass_factor,
    const float* __restrict__ body_inv_mass,
    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
    PointMassView point_masses, uint32_t dof_stride) {
    if (PointMassView::IsPointSide(lhs.kind) && PointMassView::IsPointSide(rhs.kind))
        return point_masses.Coupling(lhs, rhs);
    if (lhs.kind != rhs.kind || lhs.index != rhs.index) return 0.0f;
    switch (lhs.kind) {
        case nk::kNkSideRigid: {
            const float im = body_inv_mass[lhs.index];
            return im * math::gpu::Dot(lhs.jlin, rhs.jlin) +
                   math::gpu::Dot(lhs.jang, body_world_inv_inertia[lhs.index].Multiply(rhs.jang));
        }
        case nk::kNkSideArtic: {
            const float* const factor = mass_factor + size_t{lhs.index} * dof_stride * dof_stride;
            return MassDot(factor, dof_stride, lhs_forward, rhs_forward);
        }
        case nk::kNkSideParticle:
        case nk::kNkSideGrid: {
            return point_masses.Quad(lhs.kind, lhs.index, lhs.jlin, rhs.jlin);
        }
        default:
            return 0.0f;
    }
}

// Same-side couplings a caller already summed (bit s for side s) replace their merge.
struct SharedSideCoupling {
    uint32_t known = 0u;
    float value[2] = {0.0f, 0.0f};
};

// Row coupling J_l M^-1 J_r^T summed over both sides of each row.
__device__ inline float PairDrivenRowCoupling(
    const NkRow& lhs, const RowForward& lhs_forward, const NkRow& rhs, const RowForward& rhs_forward,
    const float* __restrict__ mass_factor,
    const float* __restrict__ body_inv_mass,
    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
    PointMassView point_masses, uint32_t dof_stride, SharedSideCoupling shared = {}) {
    const NkRowSide lhs_sides[2] = {lhs.a, lhs.b};
    const NkRowSide rhs_sides[2] = {rhs.a, rhs.b};
    float coupling = 0.0f;
    for (uint32_t i = 0u; i < 2u; ++i) {
        for (uint32_t j = 0u; j < 2u; ++j) {
            coupling += i == j && ((shared.known >> i) & 1u) != 0u ? shared.value[i]
                : PairDrivenSideCoupling(
                      lhs_sides[i], lhs_forward.side[i], rhs_sides[j], rhs_forward.side[j],
                      mass_factor, body_inv_mass, body_world_inv_inertia, point_masses, dof_stride);
        }
    }
    return coupling;
}

}  // namespace nuka::phi::nkops
