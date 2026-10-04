#pragma once

__device__ MaterialRowState& MaterialState(BlockDescentSolveParams p, BlockScratch s,
                                          uint32_t slot) {
    const uint32_t env = slot / p.rows_per_env;
    const uint32_t first = p.rows_per_env - p.material_cells_per_env * nk::kMpmStressRowsPerCell;
    const uint32_t cell = (slot % p.rows_per_env - first) / nk::kMpmStressRowsPerCell;
    return s.material_state[size_t{env} * p.material_cells_per_env + cell];
}

__device__ nk::augmented::BallTerm<nk::kMpmDeviatorComponents> EvaluateMaterialDeviator(
    DataView data, BlockScratch s, uint32_t slot, const MaterialRowState& state,
    const Vec3* jacobians, Vec3 move) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const NkRow head = rows[slot];
    double dual[nk::kMpmDeviatorComponents], residual[nk::kMpmDeviatorComponents];
    for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k) {
        dual[k] = data.lambda[slot + 1u + k];
        residual[k] = state.residual[1u + k] -
            (jacobians != nullptr ? jacobians[1u + k].Dot(move) : 0.0f);
    }
    const double radius = head.friction_secondary < FLT_MAX
        ? fmax(double(head.mu) * state.pressure_bound + head.friction_secondary, 0.0) : DBL_MAX;
    return nk::augmented::EvaluateBall(dual, double(s.penalty[slot]), residual,
        double(rows[slot + 1u].compliance_alpha), radius);
}

__global__ void CacheMaterialRowsKernel(DataView data, BlockDescentSolveParams p,
                                       BlockScratch s, bool snapshot, uint32_t cache_color) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t stride = gridDim.x * blockDim.x / warpSize;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    auto points = PointMasses(data);
    if (snapshot) {
        points.grid_velocity = s.grid_snapshot;
        points.particle_velocity = s.particle_snapshot;
    }
    for (uint32_t item = warp; item < *s.active_count; item += stride) {
        const uint32_t slot = s.active_rows[item];
        if (!RowUsesCacheColor(s, slot, cache_color)) continue;
        const NkRow head = rows[slot];
        if (!(head.flags & nk::nk_row_flags::kMaterialBlock)) continue;
        double moment[6] = {};
        for (uint32_t at = lane; at < points.Count(head.a); at += warpSize) {
            const auto term = points.At(head.a, at);
            const Vec3 velocity = points.Velocity(term.kind)[term.index];
            const Vec3 g = term.jacobian;
            moment[0] += double(velocity.x) * g.x;
            moment[1] += double(velocity.y) * g.y;
            moment[2] += double(velocity.z) * g.z;
            moment[3] += 0.5 * (double(velocity.x) * g.y + double(velocity.y) * g.x);
            moment[4] += 0.5 * (double(velocity.x) * g.z + double(velocity.z) * g.x);
            moment[5] += 0.5 * (double(velocity.y) * g.z + double(velocity.z) * g.y);
        }
        float gradient[6];
        for (uint32_t k = 0u; k < 6u; ++k) gradient[k] = static_cast<float>(SumDenseBlockWarp(moment[k]));
        if (lane != 0u) continue;
        float rates[nk::kMpmStressRowsPerCell];
        nk::material::MpmStressBlockRates(gradient, rates);
        auto& state = MaterialState(p, s, slot);
        bool valid = true;
        for (uint32_t k = 0u; k < nk::kMpmStressRowsPerCell; ++k) {
            state.residual[k] = rows[slot + k].rhs * p.dt - rates[k];
            valid &= fabsf(state.residual[k]) <= FLT_MAX;
        }
        const auto pressure = nk::augmented::EvaluateScalar(data.lambda[slot], s.penalty[slot],
            state.residual[0], head.compliance_alpha, head.lower, head.upper, s.scalar_response[slot]);
        state.pressure_bound = pressure.impulse;
        if (snapshot) {
            const auto deviator = EvaluateMaterialDeviator(data, s, slot, state, nullptr, {});
            s.row_state[slot].potential = pressure.potential + deviator.potential;
        }
        valid &= fabsf(state.pressure_bound) <= FLT_MAX;
        if (!valid) {
            const auto term = points.At(head.a, 0u);
            RecordBlockFailure(data, p, s, Owner(p, term.kind, term.index),
                nk::BlockSolveFailure::InvalidEquation, slot);
        }
    }
}

__device__ double EvaluateGridRow(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                  size_t at, Vec3 move,
                                  Vec3* impulse, SymmetricMat3* curvature) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint32_t slot = s.incidence[at];
    const NkRow head = rows[slot];
    if (!(head.flags & nk::nk_row_flags::kMaterialBlock)) {
        const LocalTerm term = LoadPointLocalTerm(data, p, s, at);
        return EvaluateLocal(term, move, impulse, curvature);
    }
    const auto& state = MaterialState(p, s, slot);
    Vec3 jacobian[nk::kMpmStressRowsPerCell];
    jacobian[0] = LoadPointIncidenceJacobian(s, at, 0u);
    for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k) {
        float components[3];
        nk::material::MpmDeviatorGradient(k, jacobian[0].x, jacobian[0].y, jacobian[0].z, components);
        jacobian[1u + k] = {components[0], components[1], components[2]};
    }
    const auto pressure = nk::augmented::EvaluateScalar(data.lambda[slot], s.penalty[slot],
        state.residual[0] - jacobian[0].Dot(move), head.compliance_alpha, head.lower, head.upper,
        s.scalar_response[slot]);
    const auto deviator = EvaluateMaterialDeviator(data, s, slot, state, jacobian, move);
    if (impulse != nullptr) {
        *impulse += jacobian[0] * pressure.impulse;
        AddOuter(*curvature, jacobian[0], pressure.curvature);
        for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k) {
            const Vec3 j = jacobian[1u + k];
            *impulse += j * deviator.impulse[k];
            AddOuter(*curvature, j, deviator.curvature[k][k]);
            for (uint32_t l = k + 1u; l < nk::kMpmDeviatorComponents; ++l) {
                const Vec3 other = jacobian[1u + l];
                const float cross = deviator.curvature[k][l];
                AddOuter(*curvature, j + other, cross);
                AddOuter(*curvature, j, -cross);
                AddOuter(*curvature, other, -cross);
            }
        }
    }
    return pressure.potential + deviator.potential;
}

__device__ void UpdateMaterialDual(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                  uint32_t slot) {
    const auto deviator = EvaluateMaterialDeviator(data, s, slot, MaterialState(p, s, slot), nullptr, {});
    data.lambda[slot] = MaterialState(p, s, slot).pressure_bound;
    for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k)
        data.lambda[slot + 1u + k] = deviator.impulse[k];
}
