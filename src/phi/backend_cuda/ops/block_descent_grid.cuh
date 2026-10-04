#pragma once

__host__ __device__ uint32_t GridColorCapacity(const BlockDescentSolveParams& p, uint32_t color) {
    const uint32_t lattice = color < nk::kMpmLatticeStencilNodes ? 0u : 1u;
    const uint32_t width = nk::kMpmStencilWidth + lattice;
    uint32_t residue = color - (lattice == 0u ? 0u : nk::kMpmLatticeStencilNodes);
    uint64_t capacity = p.env_count;
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
        const uint32_t first = residue % width;
        residue /= width;
        const uint32_t count = p.grid_dims[axis] > first
            ? 1u + (p.grid_dims[axis] - 1u - first) / width : 0u;
        capacity *= count;
    }
    return static_cast<uint32_t>(capacity);
}

__global__ void InitializeGridColorScheduleKernel(
    DataView, BlockDescentSolveParams p, BlockScratch s) {
    for (uint32_t color = blockIdx.x * blockDim.x + threadIdx.x;
         color < nk::kMpmCellStencilNodes; color += gridDim.x * blockDim.x) {
        uint32_t first = 0u;
        for (uint32_t prior = 0u; prior < color; ++prior) first += GridColorCapacity(p, prior);
        s.grid_color_offsets[color] = first;
        s.grid_active_count[color] = 0u;
    }
}

__global__ void InitializeGridBlocksKernel(
    DataView data, BlockDescentSolveParams p, BlockScratch s) {
    for (size_t node = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
         node < p.total_grid_count; node += size_t{gridDim.x} * blockDim.x) {
        const Vec3 velocity = data.grid_velocity[node];
        s.grid_free[node] = velocity;
        s.grid_snapshot[node] = velocity;
        const float inverse_mass = data.grid_inv_mass[node];
        const uint32_t owner = p.total_particle_count + p.total_body_count + p.articulation_count +
                               static_cast<uint32_t>(node);
        if (!Finite(velocity) || !(inverse_mass >= 0.0f && inverse_mass <= FLT_MAX)) {
            RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidMass);
            continue;
        }
        if (inverse_mass == 0.0f) continue;
        const double mass = 1.0 / static_cast<double>(inverse_mass);
        if (!(mass > 0.0 && mass <= double(FLT_MAX))) {
            RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidMass);
            continue;
        }
        const uint32_t color = GridNodeColor(p, static_cast<uint32_t>(node));
        const uint32_t item = atomicAdd(s.grid_active_count + color, 1u);
        s.grid_active[s.grid_color_offsets[color] + item] = static_cast<uint32_t>(node);
    }
}

__global__ void DescendGridBlocksKernel(
    DataView data, BlockDescentSolveParams p, BlockScratch s, uint32_t color) {
    const uint32_t lane = threadIdx.x % warpSize;
    const size_t warp = (size_t{blockIdx.x} * blockDim.x + threadIdx.x) / warpSize;
    const size_t stride = size_t{gridDim.x} * blockDim.x / warpSize;
    const uint32_t active_count = s.grid_active_count[color];
    const uint32_t first = s.grid_color_offsets[color];
    for (size_t item = warp; item < active_count; item += stride) {
        const uint32_t node = s.grid_active[first + item];
        const uint32_t owner = p.total_particle_count + p.total_body_count + p.articulation_count + node;
        const uint32_t begin = s.offsets[owner];
        const uint32_t end = s.offsets[owner + 1u];
        const Vec3 snapshot = s.grid_snapshot[node];
        const Vec3 free_velocity = s.grid_free[node];
        const double mass = 1.0 / static_cast<double>(data.grid_inv_mass[node]);
        Vec3 impulse{};
        SymmetricMat3 hessian{};
        bool rows_finite = true;
        uint32_t dominant_row = ~0u;
        double dominant_trace = -DBL_MAX;
        for (size_t at = size_t{begin} + lane; at < end; at += warpSize) {
            Vec3 row_impulse{};
            SymmetricMat3 row_curvature{};
            const double potential = EvaluateGridRow(data, p, s, at,
                                                     {}, &row_impulse, &row_curvature);
            rows_finite &= isfinite(potential);
            if (!isfinite(potential))
                RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidRow, s.incidence[at]);
            impulse += row_impulse;
            nk::vbd::AddTo(hessian, row_curvature);
            const double trace = double(row_curvature.xx) + row_curvature.yy + row_curvature.zz;
            const uint32_t slot = s.incidence[at];
            if (dominant_row == ~0u || trace > dominant_trace ||
                ((trace == dominant_trace || (isnan(trace) && isnan(dominant_trace))) && slot < dominant_row)) {
                dominant_row = slot;
                dominant_trace = trace;
            }
        }
        impulse = {WarpSum(impulse.x), WarpSum(impulse.y), WarpSum(impulse.z)};
        hessian = {WarpSum(hessian.xx), WarpSum(hessian.yy), WarpSum(hessian.zz),
                   WarpSum(hessian.xy), WarpSum(hessian.xz), WarpSum(hessian.yz)};
        for (uint32_t offset = warpSize / 2u; offset > 0u; offset /= 2u) {
            const uint32_t other_row = __shfl_down_sync(0xffffffffu, dominant_row, offset);
            const double other_trace = __shfl_down_sync(0xffffffffu, dominant_trace, offset);
            if (lane + offset < warpSize && other_row != ~0u &&
                (dominant_row == ~0u || other_trace > dominant_trace ||
                 ((other_trace == dominant_trace || (isnan(other_trace) && isnan(dominant_trace))) &&
                  other_row < dominant_row))) {
                dominant_row = other_row;
                dominant_trace = other_trace;
            }
        }
        if (!__all_sync(0xffffffffu, rows_finite)) {
            if (lane == 0u) RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidRow);
            continue;
        }
        Vec3 direction{};
        Vec3 force{};
        double slope = 0.0;
        uint32_t valid = 0u;
        if (lane == 0u) {
            const double fx = -mass * (double(snapshot.x) - double(free_velocity.x)) + impulse.x;
            const double fy = -mass * (double(snapshot.y) - double(free_velocity.y)) + impulse.y;
            const double fz = -mass * (double(snapshot.z) - double(free_velocity.z)) + impulse.z;
            const bool force_finite = fx >= -double(FLT_MAX) && fx <= double(FLT_MAX) &&
                fy >= -double(FLT_MAX) && fy <= double(FLT_MAX) &&
                fz >= -double(FLT_MAX) && fz <= double(FLT_MAX);
            nk::vbd::AddIdentity(hessian, static_cast<float>(mass));
            SymmetricMat3 inverse{};
            const bool equation_valid = mass > 0.0 && mass <= double(FLT_MAX) && force_finite &&
                Finite({hessian.xx, hessian.yy, hessian.zz}) &&
                Finite({hessian.xy, hessian.xz, hessian.yz});
            nk::BlockSolveFailure reason = equation_valid ? nk::BlockSolveFailure::Factorization
                                                          : nk::BlockSolveFailure::InvalidEquation;
            if (equation_valid && nk::vbd::Invert(hessian, 0.0f, &inverse)) {
                force = {static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(fz)};
                direction = inverse.Multiply(force);
                slope = -(double(force.x) * direction.x + double(force.y) * direction.y +
                          double(force.z) * direction.z);
                reason = nk::BlockSolveFailure::InvalidDirection;
                if (Finite(direction) && isfinite(slope)) valid = 1u;
            }
            if (valid == 0u) {
                double equation[nk::kBlockSolveFailureEquationColumnCount] = {
                    mass, fx, fy, fz,
                    hessian.xx, hessian.yy, hessian.zz, hessian.xy, hessian.xz, hessian.yz,
                    snapshot.x, snapshot.y, snapshot.z,
                    free_velocity.x, free_velocity.y, free_velocity.z};
                using Column = nk::BlockSolveFailureEquationColumn;
                equation[static_cast<uint32_t>(Column::DominantRow)] = dominant_row;
                if (dominant_row != ~0u) {
                    const NkRow dominant = reinterpret_cast<const NkRow*>(data.urows)[dominant_row];
                    const auto points = PointMasses(data);
                    equation[static_cast<uint32_t>(Column::DominantPenalty)] = s.penalty[dominant_row];
                    equation[static_cast<uint32_t>(Column::DominantResponseA)] =
                        SideResponse(dominant.a, data, p, points, dominant_row, 0u);
                    equation[static_cast<uint32_t>(Column::DominantResponseB)] =
                        SideResponse(dominant.b, data, p, points, dominant_row, 1u);
                    if (dominant.flags & nk::nk_row_flags::kMaterialBlock) {
                        equation[static_cast<uint32_t>(Column::DominantResidualNormal)] =
                            MaterialState(p, s, dominant_row).residual[0];
                        equation[static_cast<uint32_t>(Column::DominantDualNormal)] = data.lambda[dominant_row];
                    } else {
                        const LocalTerm term = LoadLocalTerm(data, p, s, dominant_row, ~0u, true);
                        equation[static_cast<uint32_t>(Column::DominantResidualNormal)] = term.residual.x;
                        equation[static_cast<uint32_t>(Column::DominantDualNormal)] = term.dual.x;
                    }
                    equation[static_cast<uint32_t>(Column::DominantCurvatureTrace)] = dominant_trace;
                }
                RecordBlockFailure(data, p, s, owner, reason, ~0u, equation);
            }
        }
        valid = __shfl_sync(0xffffffffu, valid, 0u);
        direction = {__shfl_sync(0xffffffffu, direction.x, 0u),
                     __shfl_sync(0xffffffffu, direction.y, 0u),
                     __shfl_sync(0xffffffffu, direction.z, 0u)};
        force = {__shfl_sync(0xffffffffu, force.x, 0u),
                 __shfl_sync(0xffffffffu, force.y, 0u),
                 __shfl_sync(0xffffffffu, force.z, 0u)};
        slope = __shfl_sync(0xffffffffu, slope, 0u);
        if (valid == 0u || !(slope < 0.0)) continue;

        float scale = 1.0f;
        Vec3 candidate{};
        bool accepted = false;
        bool failed = false;
        for (uint32_t halving = 0u; halving <= kVertexStepHalvings; ++halving) {
            candidate = {nk::vbd::TrialValue(snapshot.x, direction.x, scale),
                         nk::vbd::TrialValue(snapshot.y, direction.y, scale),
                         nk::vbd::TrialValue(snapshot.z, direction.z, scale)};
            const Vec3 move = candidate - snapshot;
            const double dx = double(candidate.x) - snapshot.x;
            const double dy = double(candidate.y) - snapshot.y;
            const double dz = double(candidate.z) - snapshot.z;
            const double actual_slope = -(double(force.x) * dx + double(force.y) * dy +
                                          double(force.z) * dz);
            double row_change = 0.0;
            bool base_finite = true;
            bool potentials_finite = Finite(candidate) && Finite(move);
            for (size_t at = size_t{begin} + lane; at < end; at += warpSize) {
                const uint32_t slot = s.incidence[at];
                const double before = s.row_state[slot].potential;
                const double after = EvaluateGridRow(data, p, s, at, move, nullptr, nullptr);
                base_finite &= isfinite(before);
                potentials_finite &= isfinite(before) && isfinite(after);
                if (!isfinite(before))
                    RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidPotential, slot);
                row_change += after - before;
            }
            row_change = SumDenseBlockWarp(row_change);
            const double mass_change = 0.5 * mass * (
                dx * (2.0 * (double(snapshot.x) - double(free_velocity.x)) + dx) +
                dy * (2.0 * (double(snapshot.y) - double(free_velocity.y)) + dy) +
                dz * (2.0 * (double(snapshot.z) - double(free_velocity.z)) + dz));
            const double change = mass_change + row_change;
            if (!__all_sync(0xffffffffu, potentials_finite && isfinite(change) && isfinite(actual_slope))) {
                if (!__all_sync(0xffffffffu, base_finite) || halving == kVertexStepHalvings) {
                    if (lane == 0u)
                        RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidPotential);
                    failed = true;
                    break;
                }
                scale *= 0.5f;
                continue;
            }
            if (actual_slope <= 0.0 && change <= double(kVertexStepDecrease) * actual_slope) {
                accepted = true;
                break;
            }
            scale *= 0.5f;
        }
        if (failed || !accepted) continue;

        if (lane == 0u) data.grid_velocity[node] = candidate;
    }
}
