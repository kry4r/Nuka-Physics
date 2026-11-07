#pragma once

__host__ __device__ uint32_t GridColorCapacity(const BlockDescentSolveParams& p, uint32_t color) {
    const uint32_t lattice = color < nk::kMpmLatticeStencilNodes ? 0u : 1u;
    const uint32_t width = GridLatticeWidth(p, lattice);
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
         color < GridColorCount(p); color += gridDim.x * blockDim.x) {
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
        // A node without rows already sits at the minimum of its inertia term, its free velocity.
        if (s.offsets[owner + 1u] == s.offsets[owner]) continue;
        const uint32_t color = GridNodeColor(p, static_cast<uint32_t>(node));
        const uint32_t item = atomicAdd(s.grid_active_count + color, 1u);
        s.grid_active[s.grid_color_offsets[color] + item] = static_cast<uint32_t>(node);
    }
}

// nk::vbd::Invert at floor zero with lane k < 6 forming its k-th cofactor and inverse entry, so each double
// product issues once per warp; every lane returns the same inverse.
__device__ bool InvertAcrossLanes(const SymmetricMat3& m, uint32_t lane, SymmetricMat3* out) {
    // Cofactor k is a * b - c * d in Invert's order cxx, cyy, czz, cxy, cxz, cyz.
    const uint32_t k = lane < 6u ? lane : 0u;
    const float a = k == 0u ? m.yy : k <= 2u ? m.xx : k == 3u ? m.xz : m.xy;
    const float b = k <= 1u ? m.zz : k == 2u ? m.yy : k <= 4u ? m.yz : m.xz;
    const float c = k == 0u ? m.yz : k == 1u || k == 4u ? m.xz : k <= 3u ? m.xy : m.xx;
    const float d = k == 0u || k == 5u ? m.yz : k == 1u ? m.xz : k == 2u ? m.xy : k == 3u ? m.zz : m.yy;
    const double cofactor = double(a) * double(b) - double(c) * double(d);
    const double cxx = __shfl_sync(0xffffffffu, cofactor, 0u);
    const double cxy = __shfl_sync(0xffffffffu, cofactor, 3u);
    const double cxz = __shfl_sync(0xffffffffu, cofactor, 4u);
    const double det = double(m.xx) * cxx + double(m.xy) * cxy + double(m.xz) * cxz;
    if (!(det > 0.0 && det <= DBL_MAX)) return false;
    const double inv = 1.0 / det;
    const double value = cofactor * inv;
    if (!__all_sync(0xffffffffu, lane >= 6u || (value >= -FLT_MAX && value <= FLT_MAX))) return false;
    const float entry = float(value);
    *out = {__shfl_sync(0xffffffffu, entry, 0u), __shfl_sync(0xffffffffu, entry, 1u),
            __shfl_sync(0xffffffffu, entry, 2u), __shfl_sync(0xffffffffu, entry, 3u),
            __shfl_sync(0xffffffffu, entry, 4u), __shfl_sync(0xffffffffu, entry, 5u)};
    return true;
}

// The incident row of largest curvature trace (lowest slot on ties), searched only for a failed node's record.
__device__ uint32_t DominantGridRow(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                    uint32_t begin, uint32_t end, uint32_t lane, double* trace) {
    uint32_t dominant_row = ~0u;
    double dominant_trace = -DBL_MAX;
    for (size_t at = size_t{begin} + lane; at < end; at += warpSize) {
        double3 row_impulse{};
        SymmetricMat3 row_curvature{};
        EvaluateGridRow(data, p, s, at, {}, &row_impulse, &row_curvature);
        const double row_trace = double(row_curvature.xx) + row_curvature.yy + row_curvature.zz;
        const uint32_t slot = s.incidence[at];
        if (dominant_row == ~0u || row_trace > dominant_trace ||
            ((row_trace == dominant_trace || (isnan(row_trace) && isnan(dominant_trace))) && slot < dominant_row)) {
            dominant_row = slot;
            dominant_trace = row_trace;
        }
    }
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
    *trace = dominant_trace;
    return dominant_row;
}

__global__ void DescendGridBlocksKernel(
    DataView data, BlockDescentSolveParams p, BlockScratch s, uint32_t color) {
    const uint32_t lane = threadIdx.x % warpSize;
    const size_t warp = (size_t{blockIdx.x} * blockDim.x + threadIdx.x) / warpSize;
    const size_t stride = size_t{gridDim.x} * blockDim.x / warpSize;
    const uint32_t active_count = s.grid_active_count[color];
    const uint32_t first = s.grid_color_offsets[color];
    const uint32_t nodes_per_env = p.total_grid_count / p.env_count;
    // A warp keeps failed-node residuals until its environment changes.
    uint32_t env = ~0u, residual = 0u;
    const auto raise = [&](uint32_t node_env, float step) {
        if (node_env != env && env != ~0u) {
            if (lane == 0u) RaiseResidual(s, env, kResidualMomentum, residual);
            residual = 0u;
        }
        env = node_env;
        residual = max(residual, ResidualBits(step));
    };
    for (size_t item = warp; item < active_count; item += stride) {
        const uint32_t node = s.grid_active[first + item];
        const uint32_t node_env = node / nodes_per_env;
        if (!EnvRunning(data, node_env)) continue;
        const uint32_t owner = p.total_particle_count + p.total_body_count + p.articulation_count + node;
        const uint32_t begin = s.offsets[owner];
        const uint32_t end = s.offsets[owner + 1u];
        const Vec3 snapshot = s.grid_snapshot[node];
        const Vec3 free_velocity = s.grid_free[node];
        const double mass = 1.0 / double(data.grid_inv_mass[node]);
        double3 impulse{};
        SymmetricMat3 hessian{};
        bool rows_finite = true;
        for (size_t at = size_t{begin} + lane; at < end; at += warpSize) {
            double3 row_impulse{};
            SymmetricMat3 row_curvature{};
            const double change = EvaluateGridRow(data, p, s, at, {}, &row_impulse, &row_curvature);
            const bool finite = isfinite(change) && isfinite(row_impulse.x) &&
                isfinite(row_impulse.y) && isfinite(row_impulse.z);
            rows_finite &= finite;
            if (!finite)
                RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidRow, s.incidence[at]);
            impulse.x += row_impulse.x;
            impulse.y += row_impulse.y;
            impulse.z += row_impulse.z;
            nk::vbd::AddTo(hessian, row_curvature);
        }
        impulse = {WarpSum(impulse.x), WarpSum(impulse.y), WarpSum(impulse.z)};
        hessian = {WarpSum(hessian.xx), WarpSum(hessian.yy), WarpSum(hessian.zz),
                   WarpSum(hessian.xy), WarpSum(hessian.xz), WarpSum(hessian.yz)};
        if (!__all_sync(0xffffffffu, rows_finite)) {
            if (lane == 0u) RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidRow);
            raise(node_env, FLT_MAX);
            continue;
        }
        // Opposing material impulses cancel in double before the local float solve.
        const Vec3 force{float(impulse.x - mass * (double(snapshot.x) - free_velocity.x)),
                         float(impulse.y - mass * (double(snapshot.y) - free_velocity.y)),
                         float(impulse.z - mass * (double(snapshot.z) - free_velocity.z))};
        Vec3 direction{};
        float slope = 0.0f;
        bool valid = false;
        nk::vbd::AddIdentity(hessian, float(mass));
        SymmetricMat3 inverse{};
        const bool equation_valid = mass > 0.0f && mass <= FLT_MAX && Finite(force) &&
            Finite({hessian.xx, hessian.yy, hessian.zz}) &&
            Finite({hessian.xy, hessian.xz, hessian.yz});
        nk::BlockSolveFailure reason = equation_valid ? nk::BlockSolveFailure::Factorization
                                                      : nk::BlockSolveFailure::InvalidEquation;
        if (equation_valid && InvertAcrossLanes(hessian, lane, &inverse)) {
            direction = inverse.Multiply(force);
            slope = -force.Dot(direction);
            reason = nk::BlockSolveFailure::InvalidDirection;
            valid = Finite(direction) && fabsf(slope) <= FLT_MAX;
        }
        if (!valid) raise(node_env, FLT_MAX);
        if (!valid) {
            double dominant_trace = 0.0;
            const uint32_t dominant_row = DominantGridRow(data, p, s, begin, end, lane, &dominant_trace);
            if (lane == 0u) {
                double equation[nk::kBlockSolveFailureEquationColumnCount] = {
                    mass, force.x, force.y, force.z,
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
            continue;
        }
        if (!(slope < 0.0f)) continue;

        float scale = 1.0f;
        Vec3 candidate{};
        bool accepted = false;
        bool failed = false;
        for (uint32_t halving = 0u; halving <= kVertexStepHalvings; ++halving) {
            candidate = {fmaf(scale, direction.x, snapshot.x), fmaf(scale, direction.y, snapshot.y),
                         fmaf(scale, direction.z, snapshot.z)};
            const Vec3 move = candidate - snapshot;
            const double actual_slope = -(double(force.x) * move.x + double(force.y) * move.y +
                double(force.z) * move.z);
            double row_change = 0.0;
            for (size_t at = size_t{begin} + lane; at < end; at += warpSize)
                row_change += EvaluateGridRow(data, p, s, at, move, nullptr, nullptr);
            const double change = 0.5 * mass * (
                double(move.x) * (2.0 * (double(snapshot.x) - free_velocity.x) + move.x) +
                double(move.y) * (2.0 * (double(snapshot.y) - free_velocity.y) + move.y) +
                double(move.z) * (2.0 * (double(snapshot.z) - free_velocity.z) + move.z)) +
                WarpSum(row_change);
            if (!__all_sync(0xffffffffu, Finite(candidate) && Finite(move) && isfinite(change) &&
                                         isfinite(actual_slope))) {
                bool base_finite = true;
                for (size_t at = size_t{begin} + lane; at < end; at += warpSize) {
                    const bool finite = isfinite(EvaluateGridRow(data, p, s, at, {}, nullptr, nullptr));
                    base_finite &= finite;
                    if (!finite)
                        RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidPotential,
                                           s.incidence[at]);
                }
                if (!__all_sync(0xffffffffu, base_finite) || halving == kVertexStepHalvings) {
                    if (lane == 0u)
                        RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidPotential);
                    failed = true;
                    break;
                }
                scale *= 0.5f;
                continue;
            }
            if (actual_slope <= 0.0f && change <= kVertexStepDecrease * actual_slope) {
                accepted = true;
                break;
            }
            scale *= 0.5f;
        }
        if (failed || !accepted) continue;

        if (lane == 0u) data.grid_velocity[node] = candidate;
        // Same-color nodes share no rows, so each point row takes this node's move without races.
        const Vec3 move = candidate - snapshot;
        const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
        for (size_t at = size_t{begin} + lane; at < end; at += warpSize) {
            const uint32_t slot = s.incidence[at];
            const NkRow row = rows[slot];
            if (row.flags & nk::nk_row_flags::kMaterialBlock) continue;
            const LocalTerm term = LoadPointLocalTerm(data, p, s, at);
            AugmentedRowState state = s.row_state[slot];
            state.residual = {term.residual.x - term.jacobian[0].Dot(move),
                              term.residual.y - term.jacobian[1].Dot(move),
                              term.residual.z - term.jacobian[2].Dot(move)};
            if (!term.contact) state.residual.y = state.residual.z = 0.0f;
            state.normal_bound = nk::augmented::EvaluateScalarImpulse(state.dual.x, term.penalty, state.residual.x,
                term.compliance, term.lower, term.upper,
                nk::augmented::ComputeFloatScalarResponse(term.penalty, term.compliance)).impulse;
            s.row_state[slot] = state;
        }
    }
    if (lane == 0u && env != ~0u) RaiseResidual(s, env, kResidualMomentum, residual);
}

// The accepted node velocity must match the free momentum plus the final physical row impulses.
__global__ void MeasureGridPrimalsKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    bool prune = false;
    if (!MeasuredSweep(p, s, &prune)) return;
    const uint32_t lane = threadIdx.x % warpSize;
    const size_t first = (size_t{blockIdx.x} * blockDim.x + threadIdx.x) / warpSize;
    const size_t stride = size_t{gridDim.x} * blockDim.x / warpSize;
    const uint32_t nodes_per_env = p.total_grid_count / p.env_count;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    for (size_t node = first; node < p.total_grid_count; node += stride) {
        const uint32_t env = node / nodes_per_env;
        const bool skip = !EnvRunning(data, env) || data.grid_inv_mass[node] == 0.0f ||
            (prune && Past(p, s, env, kResidualMomentum, 0u));
        if (__shfl_sync(0xffffffffu, skip ? 1u : 0u, 0u) != 0u) continue;
        const uint32_t owner = p.total_particle_count + p.total_body_count + p.articulation_count + node;
        double x = 0.0, y = 0.0, z = 0.0;
        for (size_t at = size_t{s.offsets[owner]} + lane; at < s.offsets[owner + 1u]; at += warpSize) {
            const uint32_t slot = s.incidence[at];
            const NkRow row = rows[slot];
            const bool material = (row.flags & nk::nk_row_flags::kMaterialBlock) != 0u;
            const uint32_t axes = material ? nk::kMpmStressRowsPerCell :
                (row.flags & nk::nk_row_flags::kBlockNormal) ? 3u : 1u;
            const Vec3 gradient = LoadPointIncidenceJacobian(s, at, 0u);
            for (uint32_t axis = 0u; axis < axes; ++axis) {
                Vec3 jacobian = gradient;
                if (material && axis > 0u) {
                    float value[3];
                    nk::material::MpmDeviatorGradient(axis - 1u, gradient.x, gradient.y, gradient.z, value);
                    jacobian = {value[0], value[1], value[2]};
                } else if (axis > 0u) jacobian = LoadPointIncidenceJacobian(s, at, axis);
                const uint32_t index = slot + axis * (material ? 1u : row.group_normal_count);
                const double impulse = material ? data.lambda[index] : s.dual_base[index];
                x += double(jacobian.x) * impulse;
                y += double(jacobian.y) * impulse;
                z += double(jacobian.z) * impulse;
            }
        }
        x = SumDenseBlockWarp(x); y = SumDenseBlockWarp(y); z = SumDenseBlockWarp(z);
        if (lane != 0u) continue;
        const Vec3 velocity = data.grid_velocity[node], free = s.grid_free[node];
        const double inverse_mass = data.grid_inv_mass[node];
        const Vec3 residual{float(double(velocity.x) - free.x - inverse_mass * x),
            float(double(velocity.y) - free.y - inverse_mass * y),
            float(double(velocity.z) - free.z - inverse_mass * z)};
        RaiseResidual(s, env, kResidualMomentum, ResidualBits(Finite(residual) ? StepSize(residual) : FLT_MAX));
    }
}
