#pragma once

__global__ void InitializeRigidBlocksKernel(
    DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t body = blockIdx.x;
    if (body >= p.total_body_count || threadIdx.x != 0u) return;
    const size_t base = size_t{body} * kRigidBlockDof;
    const Vec3 linear = data.body_linear_velocity[body];
    const Vec3 angular = data.body_angular_velocity[body];
    s.rigid_free[base] = s.rigid_snapshot[base] = linear.x;
    s.rigid_free[base + 1u] = s.rigid_snapshot[base + 1u] = linear.y;
    s.rigid_free[base + 2u] = s.rigid_snapshot[base + 2u] = linear.z;
    s.rigid_free[base + 3u] = s.rigid_snapshot[base + 3u] = angular.x;
    s.rigid_free[base + 4u] = s.rigid_snapshot[base + 4u] = angular.y;
    s.rigid_free[base + 5u] = s.rigid_snapshot[base + 5u] = angular.z;
    s.rigid_dimensions[body] = 0u;
    const float inverse_mass = data.body_inv_mass[body];
    const uint32_t owner = p.total_particle_count + body;
    if (!Finite(linear) || !Finite(angular)) {
        RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidMass);
        return;
    }
    if (inverse_mass == 0.0f) return;
    const double mass = 1.0 / static_cast<double>(inverse_mass);
    SymmetricMat3 inertia{};
    const bool valid = inverse_mass > 0.0f && inverse_mass <= FLT_MAX &&
        mass > 0.0 && mass <= double(FLT_MAX) &&
        nk::vbd::Invert(data.body_world_inv_inertia[body], 0.0f, &inertia) &&
        inertia.xx > 0.0f && inertia.yy > 0.0f && inertia.zz > 0.0f &&
        double(inertia.xx) * inertia.yy > double(inertia.xy) * inertia.xy &&
        Finite({inertia.xx, inertia.yy, inertia.zz}) &&
        Finite({inertia.xy, inertia.xz, inertia.yz});
    if (!valid) {
        RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidMass);
        return;
    }
    const float rotational[3][3] = {
        {inertia.xx, inertia.xy, inertia.xz},
        {inertia.xy, inertia.yy, inertia.yz},
        {inertia.xz, inertia.yz, inertia.zz}};
    for (uint32_t row = 0u; row < kRigidBlockDof; ++row) {
        for (uint32_t column = 0u; column < kRigidBlockDof; ++column) {
            float value = 0.0f;
            if (row < 3u && column < 3u && row == column) value = static_cast<float>(mass);
            if (row >= 3u && column >= 3u) value = rotational[row - 3u][column - 3u];
            s.rigid_mass[base * kRigidBlockDof + size_t{row} * kRigidBlockDof + column] = value;
        }
    }
    s.rigid_dimensions[body] = kRigidBlockDof;
}

constexpr uint32_t kRigidAssemblyTileRows = 32u;
constexpr uint32_t kRigidAssemblyAxes = 3u;
constexpr uint32_t kRigidAssemblyJacobianCount = kRigidAssemblyAxes * kRigidBlockDof;
constexpr uint32_t kRigidAssemblyCoefficientCount = 7u;

struct RigidAssemblyRow {
    uint32_t row_slot, normal_count, axes;
    float jacobian[kRigidAssemblyJacobianCount];
    float coefficients[kRigidAssemblyCoefficientCount];
};

__global__ void DescendRigidBlocksKernel(
    DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t body = blockIdx.x;
    if (body >= p.total_body_count || s.rigid_dimensions[body] == 0u) return;
    const size_t base = size_t{body} * kRigidBlockDof;
    const size_t matrix_base = base * kRigidBlockDof;
    const float* mass = s.rigid_mass + matrix_base;
    float* matrix = s.rigid_matrix + matrix_base;
    float* force = s.rigid_force + base;
    float* direction = s.rigid_direction + base;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint32_t owner = p.total_particle_count + body;
    const uint32_t begin = s.offsets[owner];
    const uint32_t end = s.offsets[owner + 1u];
    __shared__ RigidAssemblyRow row_tile[kRigidAssemblyTileRows];
    const bool matrix_entry = threadIdx.x < kRigidBlockDof * kRigidBlockDof;
    const bool force_entry = threadIdx.x < kRigidBlockDof;
    double matrix_value = matrix_entry ? static_cast<double>(mass[threadIdx.x]) : 0.0;
    double force_value = 0.0;
    if (force_entry) {
        for (uint32_t column = 0u; column < kRigidBlockDof; ++column) {
            const double difference = static_cast<double>(s.rigid_snapshot[base + column]) -
                                      static_cast<double>(s.rigid_free[base + column]);
            force_value -= static_cast<double>(mass[threadIdx.x * kRigidBlockDof + column]) * difference;
        }
    }
    for (uint32_t tile_begin = begin; tile_begin < end;) {
        const uint32_t tile_count = end - tile_begin < kRigidAssemblyTileRows
            ? end - tile_begin : kRigidAssemblyTileRows;
        for (uint32_t row = threadIdx.x; row < tile_count; row += blockDim.x) {
            RigidAssemblyRow& cached = row_tile[row];
            cached.row_slot = s.incidence[tile_begin + row];
            const NkRow normal = rows[cached.row_slot];
            cached.normal_count = normal.group_normal_count;
            cached.axes = (normal.flags & nk::nk_row_flags::kBlockNormal) != 0u ? kRigidAssemblyAxes : 1u;
            const auto& coefficient = s.row_coefficients[cached.row_slot];
            cached.coefficients[0] = coefficient.impulse.x;
            cached.coefficients[1] = coefficient.impulse.y;
            cached.coefficients[2] = coefficient.impulse.z;
            cached.coefficients[3] = coefficient.curvature.xx;
            cached.coefficients[4] = coefficient.curvature.yy;
            cached.coefficients[5] = coefficient.curvature.yz;
            cached.coefficients[6] = coefficient.curvature.zz;
        }
        __syncthreads();
        for (uint32_t entry = threadIdx.x; entry < tile_count * kRigidAssemblyJacobianCount;
             entry += blockDim.x) {
            RigidAssemblyRow& cached = row_tile[entry / kRigidAssemblyJacobianCount];
            const uint32_t index = entry % kRigidAssemblyJacobianCount;
            const uint32_t axis = index % kRigidAssemblyAxes;
            const uint32_t dof = index / kRigidAssemblyAxes;
            cached.jacobian[index] = axis < cached.axes
                ? RigidJacobian(data, cached.row_slot + axis * cached.normal_count, body, dof) : 0.0f;
        }
        __syncthreads();
        for (uint32_t row = 0u; row < tile_count; ++row) {
            const float* jacobian = row_tile[row].jacobian;
            const float* coefficients = row_tile[row].coefficients;
            if (matrix_entry) {
                const float* jr = jacobian + (threadIdx.x / kRigidBlockDof) * 3u;
                const float* jc = jacobian + (threadIdx.x % kRigidBlockDof) * 3u;
                matrix_value += static_cast<double>(jr[0]) * coefficients[3] * jc[0] +
                                static_cast<double>(jr[1]) * coefficients[4] * jc[1] +
                                static_cast<double>(coefficients[5]) *
                                    (static_cast<double>(jr[1]) * jc[2] + static_cast<double>(jr[2]) * jc[1]) +
                                static_cast<double>(jr[2]) * coefficients[6] * jc[2];
            }
            if (force_entry) {
                const float* jr = jacobian + threadIdx.x * 3u;
                force_value += static_cast<double>(jr[0]) * coefficients[0] +
                               static_cast<double>(jr[1]) * coefficients[1] +
                               static_cast<double>(jr[2]) * coefficients[2];
            }
        }
        __syncthreads();
        tile_begin += tile_count;
    }
    bool assembly_finite = true;
    if (matrix_entry) {
        const bool finite = matrix_value >= -double(FLT_MAX) && matrix_value <= double(FLT_MAX);
        assembly_finite &= finite;
        matrix[threadIdx.x] = finite ? static_cast<float>(matrix_value) : 0.0f;
    }
    if (force_entry) {
        const bool finite = force_value >= -double(FLT_MAX) && force_value <= double(FLT_MAX);
        assembly_finite &= finite;
        force[threadIdx.x] = finite ? static_cast<float>(force_value) : 0.0f;
    }
    if (__syncthreads_or(!assembly_finite)) {
        if (threadIdx.x == 0u) RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidEquation);
        return;
    }
    if (!SolveDensePositiveBlock(matrix, force, direction, s.rigid_diagonal + base,
                                 kRigidBlockDof, kRigidBlockDof)) {
        if (threadIdx.x == 0u) RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::Factorization);
        return;
    }
    const double slope_part = force_entry ? -static_cast<double>(force[threadIdx.x]) * direction[threadIdx.x] : 0.0;
    const double slope = SumDenseBlock(slope_part);
    if (!isfinite(slope)) {
        if (threadIdx.x == 0u) RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidDirection);
        return;
    }
    if (!(slope < 0.0)) return;

    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    const uint32_t warps = blockDim.x / warpSize;
    const uint32_t row_group_width = RowGroupWidth(kRigidBlockDof);
    const uint32_t row_groups = warpSize / row_group_width;
    const uint32_t row_group = lane / row_group_width;
    const uint32_t group_lane = lane % row_group_width;
    float scale = 1.0f;
    bool accepted = false;
    float* candidate = s.rigid_diagonal + base;
    for (uint32_t halving = 0u; halving <= kVertexStepHalvings; ++halving) {
        bool candidate_finite = true;
        if (force_entry) {
            candidate[threadIdx.x] = nk::vbd::TrialValue(s.rigid_snapshot[base + threadIdx.x],
                                                        direction[threadIdx.x], scale);
            candidate_finite = isfinite(candidate[threadIdx.x]);
        }
        __syncthreads();
        double change_part = 0.0;
        double actual_slope_part = 0.0;
        if (force_entry) {
            const double delta = static_cast<double>(candidate[threadIdx.x]) -
                                 static_cast<double>(s.rigid_snapshot[base + threadIdx.x]);
            actual_slope_part = -static_cast<double>(force[threadIdx.x]) * delta;
            double mass_change = 0.0;
            for (uint32_t column = 0u; column < kRigidBlockDof; ++column) {
                const double difference = static_cast<double>(s.rigid_snapshot[base + column]) -
                                          static_cast<double>(s.rigid_free[base + column]);
                const double column_delta = static_cast<double>(candidate[column]) -
                                            static_cast<double>(s.rigid_snapshot[base + column]);
                mass_change += static_cast<double>(mass[threadIdx.x * kRigidBlockDof + column]) *
                               (2.0 * difference + column_delta);
            }
            change_part = 0.5 * delta * mass_change;
        }
        bool rows_finite = true, base_finite = true;
        for (size_t row_begin = size_t{begin} + warp; row_begin < end; row_begin += size_t{warps} * row_groups) {
            const size_t at = row_begin + size_t{row_group} * warps;
            const bool row_valid = at < end;
            uint32_t slot = 0u, row_axes = 0u;
            NkRow normal{};
            if (row_valid) {
                slot = s.incidence[at];
                normal = rows[slot];
                row_axes = (normal.flags & nk::nk_row_flags::kBlockNormal) != 0u ? 3u : 1u;
            }
            double row_dot[3] = {};
            if (row_valid) {
                for (uint32_t dof = group_lane; dof < kRigidBlockDof; dof += row_group_width) {
                    const double delta = static_cast<double>(candidate[dof]) -
                                         static_cast<double>(s.rigid_snapshot[base + dof]);
                    for (uint32_t axis = 0u; axis < row_axes; ++axis)
                        row_dot[axis] = static_cast<double>(RigidJacobian(
                            data, slot + axis * normal.group_normal_count, body, dof)) * delta;
                }
            }
            for (uint32_t axis = 0u; axis < 3u; ++axis)
                row_dot[axis] = SumDenseRowGroup(row_dot[axis], row_group_width);
            double row_change = 0.0;
            uint32_t add_row_change = 0u;
            if (row_valid && group_lane == 0u) {
                const double before = s.row_state[slot].potential;
                base_finite &= isfinite(before);
                rows_finite &= isfinite(before);
                if (!isfinite(before))
                    RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidPotential, slot);
                bool move_finite = true;
                for (uint32_t axis = 0u; axis < 3u; ++axis) {
                    move_finite &= row_dot[axis] >= -double(FLT_MAX) && row_dot[axis] <= double(FLT_MAX);
                }
                rows_finite &= move_finite;
                if (move_finite) {
                    const Vec3 row_move{static_cast<float>(row_dot[0]),
                                        static_cast<float>(row_dot[1]),
                                        static_cast<float>(row_dot[2])};
                    const LocalTerm term = LoadLocalTerm(data, p, s, slot, ~0u, true);
                    const double after = EvaluateLocalResidual(term, row_move);
                    row_change = after - before;
                    rows_finite &= isfinite(after) && isfinite(row_change);
                    add_row_change = isfinite(before) && isfinite(after) ? 1u : 0u;
                }
            }
            for (uint32_t group = 0u; group < row_groups; ++group) {
                const double next_change = __shfl_sync(0xffffffffu, row_change, group * row_group_width);
                const uint32_t add_change = __shfl_sync(0xffffffffu, add_row_change, group * row_group_width);
                if (lane == 0u && add_change != 0u) change_part += next_change;
            }
        }
        const double actual_slope = SumDenseBlock(actual_slope_part);
        const double change = SumDenseBlock(change_part);
        const bool invalid_candidate = __syncthreads_or(!candidate_finite);
        if (__syncthreads_or(!rows_finite || !isfinite(change) || !isfinite(actual_slope)) || invalid_candidate) {
            const bool invalid_base = __syncthreads_or(!base_finite);
            if (invalid_base || halving == kVertexStepHalvings) {
                if (threadIdx.x == 0u) RecordBlockFailure(data, p, s, owner,
                    !invalid_base && invalid_candidate ? nk::BlockSolveFailure::InvalidCandidate : nk::BlockSolveFailure::InvalidPotential);
                return;
            }
            scale *= 0.5f;
            continue;
        }
        if (actual_slope <= 0.0 && change <= static_cast<double>(kVertexStepDecrease) * actual_slope) {
            accepted = true;
            break;
        }
        scale *= 0.5f;
    }
    if (!accepted) return;

    if (threadIdx.x == 0u) {
        data.body_linear_velocity[body] = {candidate[0], candidate[1], candidate[2]};
        data.body_angular_velocity[body] = {candidate[3], candidate[4], candidate[5]};
    }
}
