#pragma once

__device__ MaterialRowState& MaterialState(BlockDescentSolveParams p, BlockScratch s,
                                          uint32_t slot) {
    const uint32_t env = slot / p.rows_per_env;
    const uint32_t first = p.rows_per_env - p.material_cells_per_env * nk::kMpmStressRowsPerCell;
    const uint32_t cell = (slot % p.rows_per_env - first) / nk::kMpmStressRowsPerCell;
    return s.material_state[size_t{env} * p.material_cells_per_env + cell];
}

// The deviator ball's yield radius from the cell's pressure bound; DBL_MAX without plasticity.
__device__ double MaterialYieldRadius(const NkRow& head, const MaterialRowState& state) {
    return head.friction_secondary < FLT_MAX
        ? fmax(double(head.mu) * state.pressure_bound + head.friction_secondary, 0.0) : DBL_MAX;
}

__device__ nk::augmented::BallTerm<nk::kMpmDeviatorComponents> EvaluateMaterialDeviator(
    DataView data, BlockScratch s, uint32_t slot, const MaterialRowState& state) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    double dual[nk::kMpmDeviatorComponents], residual[nk::kMpmDeviatorComponents];
    for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k) {
        dual[k] = data.lambda[slot + 1u + k];
        residual[k] = state.residual[1u + k];
    }
    return nk::augmented::EvaluateBall(dual, double(s.penalty[slot]), residual,
        double(rows[slot + 1u].compliance_alpha), MaterialYieldRadius(rows[slot], state));
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
    const bool all = cache_color == kAllCacheColors;
    const uint32_t first = all ? 0u : s.cache_offsets[cache_color];
    const uint32_t count = all ? *s.active_count : s.cache_offsets[cache_color + 1u] - first;
    for (uint32_t item = warp; item < count; item += stride) {
        const uint32_t slot = all ? s.active_rows[item] : s.cache_rows[first + item];
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
        valid &= fabsf(state.pressure_bound) <= FLT_MAX;
        if (!valid) {
            const auto term = points.At(head.a, 0u);
            RecordBlockFailure(data, p, s, Owner(p, term.kind, term.index),
                nk::BlockSolveFailure::InvalidEquation, slot);
        }
    }
}

// A grid incidence's potential change along the node move, integrated from its cached state; with `impulse`,
// the move is zero and only its response is added. Point rows run entirely on float scales.
__device__ double EvaluateGridRow(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                  size_t at, Vec3 move,
                                  Vec3* impulse, SymmetricMat3* curvature) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint32_t slot = s.incidence[at];
    const NkRow head = rows[slot];
    if (!(head.flags & nk::nk_row_flags::kMaterialBlock)) {
        const LocalTerm term = LoadPointLocalTerm(data, p, s, at);
        const auto response = nk::augmented::ComputeFloatScalarResponse(term.penalty, term.compliance);
        if (impulse == nullptr)
            return EvaluateDropChange(term, {term.jacobian[0].Dot(move), term.jacobian[1].Dot(move),
                                             term.jacobian[2].Dot(move)}, response);
        Vec3 f{};
        SymmetricMat3 c{};
        EvaluateAugmentedResponse(term, nk::augmented::EvaluateScalarImpulse(term.dual.x, term.penalty,
            term.residual.x, term.compliance, term.lower, term.upper, response), {}, &f, &c);
        AddLocalResponse(term.jacobian, f, c, impulse, curvature);
        return 0.0;
    }
    const auto& state = MaterialState(p, s, slot);
    Vec3 jacobian[nk::kMpmStressRowsPerCell];
    jacobian[0] = LoadPointIncidenceJacobian(s, at, 0u);
    for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k) {
        float components[3];
        nk::material::MpmDeviatorGradient(k, jacobian[0].x, jacobian[0].y, jacobian[0].z, components);
        jacobian[1u + k] = {components[0], components[1], components[2]};
    }
    if (impulse == nullptr) {
        double dual[nk::kMpmDeviatorComponents], residual[nk::kMpmDeviatorComponents];
        double drop[nk::kMpmDeviatorComponents];
        for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k) {
            dual[k] = data.lambda[slot + 1u + k];
            residual[k] = state.residual[1u + k];
            drop[k] = jacobian[1u + k].Dot(move);
        }
        return nk::augmented::ScalarPotentialChange(data.lambda[slot], s.penalty[slot], state.residual[0],
                   jacobian[0].Dot(move), head.compliance_alpha, head.lower, head.upper, s.scalar_response[slot]) +
               nk::augmented::BallPotentialChange(dual, double(s.penalty[slot]), residual, drop,
                   double(rows[slot + 1u].compliance_alpha), MaterialYieldRadius(head, state));
    }
    const auto pressure = nk::augmented::EvaluateScalar(data.lambda[slot], s.penalty[slot],
        state.residual[0], head.compliance_alpha, head.lower, head.upper, s.scalar_response[slot]);
    const auto deviator = EvaluateMaterialDeviator(data, s, slot, state);
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
    return 0.0;
}

// Sets a material cell's multipliers to its projected impulses. Returns their step over the residual scales,
// the compliant stress rows' volume-rate residual, per cell face area as a velocity.
__device__ float UpdateMaterialDual(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                   uint32_t slot) {
    const auto& state = MaterialState(p, s, slot);
    const auto deviator = EvaluateMaterialDeviator(data, s, slot, state);
    const float rho = s.penalty[slot];
    const float compliance = reinterpret_cast<const NkRow*>(data.urows)[slot + 1u].compliance_alpha;
    const float deviator_scale = rho > 0.0f ? rho / fmaf(rho, compliance, 1.0f) : 0.0f;
    float square = 0.0f;
    for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k) {
        const float change = MultiplierSteps({data.lambda[slot + 1u + k], 0.0f, 0.0f},
            {deviator.impulse[k], 0.0f, 0.0f}, deviator_scale, rho, 0.0f, 0.0f).x;
        square = fmaf(change, change, square);
        data.lambda[slot + 1u + k] = deviator.impulse[k];
    }
    const float pressure = MultiplierSteps({data.lambda[slot], 0.0f, 0.0f}, {state.pressure_bound, 0.0f, 0.0f},
        static_cast<float>(s.scalar_response[slot].residual_scale), rho, 0.0f, 0.0f).x;
    data.lambda[slot] = state.pressure_bound;
    return fmaxf(pressure, sqrtf(square)) / (p.grid_cell_size * p.grid_cell_size);
}
