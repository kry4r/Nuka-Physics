#pragma once

__global__ void InitializeArticulationBlocksKernel(
    ArticulationDeviceState state, DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t articulation = blockIdx.x;
    if (articulation >= p.articulation_count || threadIdx.x != 0u) return;
    const size_t base = size_t{articulation} * p.max_dof;
    uint32_t dimension = 0u;
    for (uint32_t dof = 0u; dof < p.max_dof; ++dof) {
        uint32_t link, component;
        if (!ArticulationDofLocation(state, articulation, dof, &link, &component)) continue;
        if (p.mimic_couplings != 0u && data.mimic_root_dof[base + dof] != dof) continue;
        s.articulation_roots[base + dimension++] = dof;
    }
    s.articulation_dimensions[articulation] = dimension;
    for (uint32_t compact = dimension; compact < p.max_dof; ++compact)
        s.articulation_roots[base + compact] = ~0u;
}

// Each articulation owns a square grid of 8x8 tiles at the full DOF stride.
__global__ void AssembleArticulationBlocksKernel(
    DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t tiles = (p.max_dof + 7u) / 8u;
    if (tiles == 0u) return;
    const uint32_t articulation = blockIdx.x / (tiles * tiles);
    if (articulation >= p.articulation_count) return;
    const uint32_t tile_row = (blockIdx.x / tiles) % tiles;
    const uint32_t tile_column = blockIdx.x % tiles;
    const uint32_t dimension = s.articulation_dimensions[articulation];
    const size_t base = size_t{articulation} * p.max_dof;
    const size_t matrix_base = base * p.max_dof;
    const float* mass = (p.mimic_couplings != 0u ? data.m_reduced : data.m) + matrix_base;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint32_t owner = p.total_particle_count + p.total_body_count + articulation;
    const uint32_t begin = s.offsets[owner];
    const uint32_t end = s.offsets[owner + 1u];
    constexpr uint32_t kAssemblyTileRows = 32u;
    constexpr uint32_t kTileDofs = 8u;
    constexpr uint32_t kRowAxes = 3u;
    constexpr uint32_t kSideJacobianSize = kTileDofs * kRowAxes;
    struct AssemblyRow {
        uint32_t slot, normal_count, axes;
        float row_jacobian[kSideJacobianSize], column_jacobian[kSideJacobianSize];
        float impulse_x, impulse_y, impulse_z;
        float curvature_xx, curvature_yy, curvature_yz, curvature_zz;
    };
    __shared__ AssemblyRow row_tile[kAssemblyTileRows];

    const uint32_t compact_row = tile_row * 8u + threadIdx.x / 8u;
    const uint32_t compact_column = tile_column * 8u + threadIdx.x % 8u;
    const bool matrix_entry = threadIdx.x < 64u && compact_row < dimension && compact_column < dimension;
    double matrix_value = 0.0;
    if (matrix_entry) {
        const uint32_t root_row = s.articulation_roots[base + compact_row];
        const uint32_t root_column = s.articulation_roots[base + compact_column];
        matrix_value = mass[size_t{root_row} * p.max_dof + root_column];
    }
    const uint32_t force_row = tile_row * 8u + threadIdx.x;
    const bool force_entry = tile_column == 0u && threadIdx.x < 8u && force_row < dimension;
    double force_value = 0.0;
    if (force_entry) {
        const uint32_t root_row = s.articulation_roots[base + force_row];
        for (uint32_t compact = 0u; compact < dimension; ++compact) {
            const uint32_t root_column = s.articulation_roots[base + compact];
            const double difference = static_cast<double>(s.articulation_snapshot[base + root_column]) -
                                      static_cast<double>(s.articulation_free[base + root_column]);
            force_value -= static_cast<double>(mass[size_t{root_row} * p.max_dof + root_column]) * difference;
        }
    }

    for (uint32_t at = begin; at < end;) {
        const uint32_t count = min(kAssemblyTileRows, end - at);
        for (uint32_t item = threadIdx.x; item < count; item += blockDim.x) {
            auto& cached = row_tile[item];
            cached.slot = s.incidence[at + item];
            const NkRow normal = rows[cached.slot];
            cached.normal_count = normal.group_normal_count;
            cached.axes = (normal.flags & nk::nk_row_flags::kBlockNormal) != 0u ? kRowAxes : 1u;
            const auto& coefficient = s.row_coefficients[cached.slot];
            cached.impulse_x = coefficient.impulse.x;
            cached.impulse_y = coefficient.impulse.y;
            cached.impulse_z = coefficient.impulse.z;
            cached.curvature_xx = coefficient.curvature.xx;
            cached.curvature_yy = coefficient.curvature.yy;
            cached.curvature_yz = coefficient.curvature.yz;
            cached.curvature_zz = coefficient.curvature.zz;
        }
        __syncthreads();
        for (uint32_t item = threadIdx.x; item < count * 2u * kSideJacobianSize; item += blockDim.x) {
            auto& cached = row_tile[item / (2u * kSideJacobianSize)];
            const uint32_t side_item = item % (2u * kSideJacobianSize);
            const bool row_side = side_item < kSideJacobianSize;
            const uint32_t local = side_item % kSideJacobianSize;
            const uint32_t axis = local % kRowAxes;
            const uint32_t compact = (row_side ? tile_row : tile_column) * kTileDofs + local / kRowAxes;
            float jacobian = 0.0f;
            if (compact < dimension && axis < cached.axes) {
                const uint32_t root = s.articulation_roots[base + compact];
                jacobian = ArticulationJacobian(data, p, cached.slot + axis * cached.normal_count,
                                                articulation, root);
            }
            (row_side ? cached.row_jacobian : cached.column_jacobian)[local] = jacobian;
        }
        __syncthreads();
        for (uint32_t item = 0u; item < count; ++item) {
            const auto& cached = row_tile[item];
            if (matrix_entry) {
                const float* jr = cached.row_jacobian + (threadIdx.x / kTileDofs) * kRowAxes;
                const float* jc = cached.column_jacobian + (threadIdx.x % kTileDofs) * kRowAxes;
                matrix_value += static_cast<double>(jr[0]) * cached.curvature_xx * jc[0] +
                                static_cast<double>(jr[1]) * cached.curvature_yy * jc[1] +
                                static_cast<double>(cached.curvature_yz) *
                                    (static_cast<double>(jr[1]) * jc[2] + static_cast<double>(jr[2]) * jc[1]) +
                                static_cast<double>(jr[2]) * cached.curvature_zz * jc[2];
            }
            if (force_entry) {
                const float* jr = cached.row_jacobian + threadIdx.x * kRowAxes;
                force_value += static_cast<double>(jr[0]) * cached.impulse_x +
                               static_cast<double>(jr[1]) * cached.impulse_y +
                               static_cast<double>(jr[2]) * cached.impulse_z;
            }
        }
        __syncthreads();
        at += count;
    }
    if (matrix_entry) {
        s.articulation_matrix[matrix_base + size_t{compact_row} * p.max_dof + compact_column] =
            static_cast<float>(matrix_value);
        if (!isfinite(static_cast<float>(matrix_value)))
            RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidEquation);
    }
    if (force_entry) {
        s.articulation_force[base + force_row] = static_cast<float>(force_value);
        if (!isfinite(static_cast<float>(force_value)))
            RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidEquation);
    }
}

__global__ void DescendArticulationsKernel(
    ArticulationDeviceState state, DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t articulation = blockIdx.x;
    if (articulation >= p.articulation_count) return;
    const uint32_t owner = p.total_particle_count + p.total_body_count + articulation;
    const uint32_t dimension = s.articulation_dimensions[articulation];
    const size_t base = size_t{articulation} * p.max_dof;
    const size_t matrix_base = base * p.max_dof;
    float* direction = s.articulation_direction + base;
    const float* force = s.articulation_force + base;
    if (!SolveDensePositiveBlock(s.articulation_matrix + matrix_base, force, direction,
                                s.articulation_diagonal + base, dimension, p.max_dof)) {
        if (threadIdx.x == 0u) RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::Factorization);
        return;
    }
    double slope_part = 0.0;
    for (uint32_t compact = threadIdx.x; compact < dimension; compact += blockDim.x)
        slope_part -= static_cast<double>(force[compact]) * direction[compact];
    const double slope = SumDenseBlock(slope_part);
    if (!isfinite(slope)) {
        if (threadIdx.x == 0u) RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidDirection);
        return;
    }
    if (!(slope < 0.0)) return;

    const float* mass = (p.mimic_couplings != 0u ? data.m_reduced : data.m) + matrix_base;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint32_t begin = s.offsets[owner];
    const uint32_t end = s.offsets[owner + 1u];
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    const uint32_t warps = blockDim.x / warpSize;
    const uint32_t row_group_width = RowGroupWidth(dimension);
    const uint32_t row_groups = warpSize / row_group_width;
    const uint32_t row_group = lane / row_group_width;
    const uint32_t group_lane = lane % row_group_width;
    float scale = 1.0f;
    bool accepted = false;
    float* candidate = s.articulation_diagonal + base;
    for (uint32_t halving = 0u; halving <= kVertexStepHalvings; ++halving) {
        bool candidate_finite = true;
        for (uint32_t compact = threadIdx.x; compact < dimension; compact += blockDim.x) {
            const uint32_t root = s.articulation_roots[base + compact];
            candidate[compact] = nk::vbd::TrialValue(s.articulation_snapshot[base + root], direction[compact], scale);
            candidate_finite &= isfinite(candidate[compact]);
        }
        __syncthreads();
        for (uint32_t dof = threadIdx.x; dof < p.max_dof; dof += blockDim.x) {
            uint32_t link, component;
            if (!ArticulationDofLocation(state, articulation, dof, &link, &component)) continue;
            const uint32_t root = p.mimic_couplings != 0u ? data.mimic_root_dof[base + dof] : dof;
            double root_value = s.articulation_snapshot[base + root];
            for (uint32_t compact = 0u; compact < dimension; ++compact)
                if (s.articulation_roots[base + compact] == root) root_value = candidate[compact];
            candidate_finite &= root_value >= -double(FLT_MAX) && root_value <= double(FLT_MAX);
            const double value = p.mimic_couplings != 0u
                ? root_value * data.mimic_root_scale[base + dof] : root_value;
            candidate_finite &= value >= -double(FLT_MAX) && value <= double(FLT_MAX);
        }
        double change_part = 0.0;
        double actual_slope_part = 0.0;
        for (uint32_t compact = threadIdx.x; compact < dimension; compact += blockDim.x) {
            const uint32_t root_row = s.articulation_roots[base + compact];
            const double delta = static_cast<double>(candidate[compact]) -
                                 static_cast<double>(s.articulation_snapshot[base + root_row]);
            actual_slope_part -= static_cast<double>(force[compact]) * delta;
            double mass_change = 0.0;
            for (uint32_t column = 0u; column < dimension; ++column) {
                const uint32_t root_column = s.articulation_roots[base + column];
                const double difference = static_cast<double>(s.articulation_snapshot[base + root_column]) -
                                          static_cast<double>(s.articulation_free[base + root_column]);
                const double column_delta = static_cast<double>(candidate[column]) -
                                            static_cast<double>(s.articulation_snapshot[base + root_column]);
                mass_change += static_cast<double>(mass[size_t{root_row} * p.max_dof + root_column]) *
                               (2.0 * difference + column_delta);
            }
            change_part += 0.5 * delta * mass_change;
        }
        bool row_moves_finite = true, base_finite = true;
        for (size_t row_begin = size_t{begin} + warp; row_begin < end; row_begin += size_t{warps} * row_groups) {
            const size_t at = row_begin + size_t{row_group} * warps;
            const bool row_valid = at < end;
            uint32_t slot = 0u, axes = 0u;
            NkRow normal{};
            if (row_valid) {
                slot = s.incidence[at];
                normal = rows[slot];
                axes = (normal.flags & nk::nk_row_flags::kBlockNormal) != 0u ? 3u : 1u;
            }
            double row_dot[3] = {};
            if (row_valid) {
                for (uint32_t compact = group_lane; compact < dimension; compact += row_group_width) {
                    const uint32_t root = s.articulation_roots[base + compact];
                    const double delta = static_cast<double>(candidate[compact]) -
                                         static_cast<double>(s.articulation_snapshot[base + root]);
                    for (uint32_t axis = 0u; axis < axes; ++axis)
                        row_dot[axis] += static_cast<double>(ArticulationJacobian(
                            data, p, slot + axis * normal.group_normal_count, articulation, root)) * delta;
                }
            }
            for (uint32_t axis = 0u; axis < 3u; ++axis)
                row_dot[axis] = SumDenseRowGroup(row_dot[axis], row_group_width);
            double row_change = 0.0;
            uint32_t add_row_change = 0u;
            if (row_valid && group_lane == 0u) {
                const double before = s.row_state[slot].potential;
                base_finite &= isfinite(before);
                row_moves_finite &= isfinite(before);
                if (!isfinite(before))
                    RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidPotential, slot);
                const Vec3 row_move{static_cast<float>(row_dot[0]),
                                    static_cast<float>(row_dot[1]),
                                    static_cast<float>(row_dot[2])};
                row_moves_finite &= Finite(row_move);
                if (Finite(row_move)) {
                    const LocalTerm term = LoadLocalTerm(data, p, s, slot, ~0u, true);
                    const double after = EvaluateLocalResidual(term, row_move);
                    row_change = after - before;
                    row_moves_finite &= isfinite(after) && isfinite(row_change);
                    add_row_change = 1u;
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
        if (__syncthreads_or(!row_moves_finite || !isfinite(change) || !isfinite(actual_slope)) || invalid_candidate) {
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

    for (uint32_t dof = threadIdx.x; dof < p.max_dof; dof += blockDim.x) {
        uint32_t link, component;
        if (!ArticulationDofLocation(state, articulation, dof, &link, &component)) {
            data.qdot_flat[base + dof] = 0.0f;
            continue;
        }
        const uint32_t root = p.mimic_couplings != 0u ? data.mimic_root_dof[base + dof] : dof;
        double next_value = 0.0;
        for (uint32_t compact = 0u; compact < dimension; ++compact) {
            if (s.articulation_roots[base + compact] == root) {
                next_value = candidate[compact];
                break;
            }
        }
        if (p.mimic_couplings != 0u) next_value *= data.mimic_root_scale[base + dof];
        const float next = static_cast<float>(next_value);
        data.qdot_flat[base + dof] = next;
        if (component == ~0u) state.qdot[link] = next;
        else state.link_velocity[link].v[component] = next;
    }
}

__global__ void WriteBlockJointLimitImpulsesKernel(DataView data, BlockDescentSolveParams p) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint64_t total_links = uint64_t{p.base_link_count} * p.env_count;
    for (uint64_t link = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x; link < total_links;
         link += uint64_t{gridDim.x} * blockDim.x) {
        const uint32_t env = static_cast<uint32_t>(link / p.base_link_count);
        const uint32_t local_link = static_cast<uint32_t>(link % p.base_link_count);
        const size_t lower = size_t{env} * p.rows_per_env + p.contact_rows_per_env + size_t{local_link} * 2u;
        data.joint_limit_impulse[2u * link] =
            (rows[lower].flags & nk::nk_row_flags::kActive) != 0u ? data.lambda[lower] : 0.0f;
        data.joint_limit_impulse[2u * link + 1u] =
            (rows[lower + 1u].flags & nk::nk_row_flags::kActive) != 0u ? data.lambda[lower + 1u] : 0.0f;
    }
}
