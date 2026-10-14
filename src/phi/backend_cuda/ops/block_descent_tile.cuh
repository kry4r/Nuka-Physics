#pragma once

// One warp solves an environment whose blocks fit a warp of coordinates, all outer iterations in a launch:
// blocks descend in Gauss-Seidel order, then the duals update; a lane holds a block coordinate or a head row.
constexpr uint32_t kNoTileBlock = ~0u;
constexpr uint32_t kNoTileIncidence = 2u;

__device__ __forceinline__ void SetAxisComponent(Vec3& v, uint32_t axis, float value) {
    if (axis == 0u) v.x = value;
    else if (axis == 1u) v.y = value;
    else v.z = value;
}

// A float sum kept unevaluated as value + error with error-free transformations.
struct TileSum {
    float value = 0.0f;
    float error = 0.0f;
};

__device__ __forceinline__ void TileAdd(TileSum& s, float x) {
    const float sum = s.value + x;
    const float part = sum - s.value;
    s.error += (s.value - (sum - part)) + (x - part);
    s.value = sum;
}

__device__ __forceinline__ void TileAddProduct(TileSum& s, float a, float b) {
    const float product = __fmul_rn(a, b);
    s.error += fmaf(a, b, -product);
    TileAdd(s, product);
}

// Every lane receives the same warp total; exchanges are symmetric and the added errors are exact.
__device__ __forceinline__ float TileWarpTotal(TileSum s) {
    for (uint32_t offset = kTileLanes / 2u; offset > 0u; offset /= 2u) {
        const float value = __shfl_xor_sync(~0u, s.value, offset);
        s.error += __shfl_xor_sync(~0u, s.error, offset);
        TileAdd(s, value);
    }
    return s.value + s.error;
}

__device__ __forceinline__ float TileWarpSum(float value) {
    for (uint32_t offset = kTileLanes / 2u; offset > 0u; offset /= 2u)
        value += __shfl_xor_sync(~0u, value, offset);
    return value;
}

__device__ __forceinline__ Vec3 TileWarpSum3(Vec3 v) {
    for (uint32_t offset = kTileLanes / 2u; offset > 0u; offset /= 2u) {
        v.x += __shfl_xor_sync(~0u, v.x, offset);
        v.y += __shfl_xor_sync(~0u, v.y, offset);
        v.z += __shfl_xor_sync(~0u, v.z, offset);
    }
    return v;
}

__device__ __forceinline__ float TileWarpMax(float value) {
    for (uint32_t offset = kTileLanes / 2u; offset > 0u; offset /= 2u)
        value = fmaxf(value, __shfl_xor_sync(~0u, value, offset));
    return value;
}

struct TileContext {
    uint32_t env;
    uint32_t lane;
    uint32_t iteration;
    uint32_t heads;
    uint32_t bodies;
    uint32_t articulations;
    uint32_t first_body;
    uint32_t first_articulation;
    uint32_t capacity;
    uint32_t staged;
    uint32_t width;
    float* fields;
    float* jacobian;
    float* mass;
    float* factor;
    __device__ float& Value(uint32_t field, uint32_t h) const {
        return fields[size_t{h / kTileHeadGroup} * kTileFields * kTileHeadGroup +
                      size_t{field} * kTileHeadGroup + h % kTileHeadGroup];
    }
    __device__ uint32_t& Index(uint32_t field, uint32_t h) const {
        return reinterpret_cast<uint32_t&>(Value(field, h));
    }
    __device__ Vec3 Load3(uint32_t field, uint32_t h) const {
        return {Value(field, h), Value(field + 1u, h), Value(field + 2u, h)};
    }
    __device__ void Store3(uint32_t field, uint32_t h, Vec3 v) const {
        Value(field, h) = v.x;
        Value(field + 1u, h) = v.y;
        Value(field + 2u, h) = v.z;
    }
    __device__ nk::augmented::FloatScalarResponse Response(uint32_t h) const {
        return {Value(kTileDualScale, h), Value(kTileResidualScale, h)};
    }
    // The head's incidence (0 or 1) on `block`, or kNoTileIncidence when it does not touch it.
    __device__ uint32_t Incidence(uint32_t h, uint32_t block) const {
        return Index(kTileOwner, h) == block ? 0u : Index(kTileOwner + 1u, h) == block ? 1u : kNoTileIncidence;
    }
};

// A tile block's coordinates: a rigid body's six, or an articulation's compact (mimic root) ones.
struct TileBlock {
    uint32_t index;
    uint32_t local;
    uint32_t owner;
    uint32_t dimension;
    uint32_t mass_stride;
    const uint32_t* roots;
    const float* mass;
    const float* free;
    float* velocity;
    __device__ float Mass(uint32_t i, uint32_t j) const {
        return roots == nullptr ? mass[i * kRigidBlockDof + j] : mass[size_t{roots[i]} * mass_stride + roots[j]];
    }
    __device__ float Jacobian(DataView data, BlockDescentSolveParams p, const NkRow& row,
                              uint32_t slot, uint32_t i) const {
        float value = 0.0f;
        if (roots == nullptr) {
            const NkRowSide sides[2] = {row.a, row.b};
            for (const NkRowSide& side : sides) {
                if (side.kind == kNkSideRigid && side.index == index)
                    value += AxisComponent(i < 3u ? side.jlin : side.jang, i % 3u);
            }
        } else {
            const size_t at = size_t{slot} * p.max_dof + roots[i];
            if (row.a.kind == kNkSideArtic && row.a.index == index)
                value += static_cast<const float*>(data.chain_jacobian)[at];
            if (row.b.kind == kNkSideArtic && row.b.index == index)
                value += static_cast<const float*>(data.chain_jacobian_b)[at];
        }
        return value;
    }
};

// Block `local` of the tile: its rigid bodies first, then its articulations. The position pass moves pseudo
// velocities that start at rest without a free velocity.
__device__ TileBlock LoadTileBlock(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                   const TileContext& c, uint32_t local, bool position = false) {
    TileBlock b{};
    b.local = local;
    if (local < c.bodies) {
        b.index = c.first_body + local;
        b.owner = p.total_particle_count + b.index;
        b.dimension = s.rigid_dimensions[b.index];
        b.mass_stride = kRigidBlockDof;
        const size_t base = size_t{b.index} * kRigidBlockDof;
        b.mass = s.rigid_mass + base * kRigidBlockDof;
        b.free = position ? nullptr : s.rigid_free + base;
        b.velocity = (position ? s.rigid_pseudo : s.rigid_velocity) + base;
        return b;
    }
    b.index = c.first_articulation + local - c.bodies;
    b.owner = p.total_particle_count + p.total_body_count + b.index;
    b.dimension = s.articulation_dimensions[b.index];
    b.mass_stride = p.max_dof;
    const size_t base = size_t{b.index} * p.max_dof;
    b.roots = s.articulation_roots + base;
    b.mass = (p.mimic_couplings != 0u ? data.m_reduced : data.m) + base * p.max_dof;
    b.free = position ? nullptr : s.articulation_free + base;
    b.velocity = (position ? s.articulation_pseudo : s.articulation_velocity) + base;
    return b;
}

// A side's tile block when it moves in the solve, else kNoTileBlock.
__device__ uint32_t TileOwner(BlockScratch s, const TileContext& c, const NkRowSide& side) {
    if (side.kind == kNkSideRigid && s.rigid_dimensions[side.index] != 0u) return side.index - c.first_body;
    if (side.kind == kNkSideArtic && s.articulation_dimensions[side.index] != 0u)
        return c.bodies + side.index - c.first_articulation;
    return kNoTileBlock;
}

// RecordBlockFailure at the tile's iteration; the kernel's scratch parameter stays read-only.
__device__ void RecordTileFailure(DataView data, BlockDescentSolveParams p, BlockScratch s, const TileContext& c,
                                  uint32_t owner, nk::BlockSolveFailure reason) {
    s.diagnostic_iteration = c.iteration;
    RecordBlockFailure(data, p, s, owner, reason);
}

// A head's coefficients and state as one lane evaluates it.
struct TileHead {
    uint32_t axes;
    float rho;
    float compliance;
    float lower;
    float upper;
    float mu_first;
    float mu_second;
    float normal_bound;
    Vec3 residual;
    Vec3 dual;
    Vec3 move;
    Vec3 step;
    nk::augmented::FloatScalarResponse response;
};

__device__ TileHead LoadTileHead(const TileContext& c, uint32_t h) {
    TileHead t;
    t.axes = c.Index(kTileAxes, h);
    t.rho = c.Value(kTileRho, h);
    t.compliance = c.Value(kTileCompliance, h);
    t.lower = c.Value(kTileLower, h);
    t.upper = c.Value(kTileUpper, h);
    t.mu_first = c.Value(kTileMuFirst, h);
    t.mu_second = c.Value(kTileMuSecond, h);
    t.normal_bound = c.Value(kTileNormalBound, h);
    t.residual = c.Load3(kTileResidual, h);
    t.dual = c.Load3(kTileDual, h);
    t.move = c.Load3(kTileMove, h);
    t.step = c.Load3(kTileStep, h);
    t.response = c.Response(h);
    return t;
}

__device__ LocalTerm TileLocalTerm(const TileHead& t) {
    LocalTerm term{};
    term.residual = t.residual;
    term.dual = t.dual;
    term.penalty = t.rho;
    term.compliance = t.compliance;
    term.lower = t.lower;
    term.upper = t.upper;
    term.mu_first = t.mu_first;
    term.mu_second = t.mu_second;
    term.normal_bound = t.normal_bound;
    term.contact = t.axes == 3u;
    return term;
}

// EvaluateAugmentedResponse at a move of the head, its normal impulse evaluated in float like the cone.
__device__ void TileResponse(const TileHead& t, Vec3 move, Vec3* impulse, SymmetricMat3* curvature) {
    EvaluateAugmentedResponse(TileLocalTerm(t), nk::augmented::EvaluateScalarImpulse(t.dual.x, t.rho,
        t.residual.x - move.x, t.compliance, t.lower, t.upper, t.response), move, impulse, curvature);
}

// Lane c's share of a row side's velocity J u at the tile's current velocities.
__device__ float TileSideVelocity(BlockDescentSolveParams p, BlockScratch s, const NkRowSide& side,
                                  const float* jacobian, uint32_t slot, uint32_t c) {
    if (side.kind == kNkSideRigid) {
        if (c >= kRigidBlockDof) return 0.0f;
        const float j = c < 3u ? AxisComponent(side.jlin, c) : AxisComponent(side.jang, c - 3u);
        return j * s.rigid_velocity[size_t{side.index} * kRigidBlockDof + c];
    }
    if (side.kind != kNkSideArtic || c >= s.articulation_dimensions[side.index]) return 0.0f;
    const size_t base = size_t{side.index} * p.max_dof;
    return jacobian[size_t{slot} * p.max_dof + s.articulation_roots[base + c]] * s.articulation_velocity[base + c];
}

// RowInertialResponse in float: J M^-1 J^T over the row's rigid bodies and articulations, with both sides
// on one owner summed first; `coupling` is the product of the two owners' shares.
__device__ float TileInertialResponse(DataView data, BlockDescentSolveParams p, const NkRow& row, uint32_t slot,
                                      float* coupling) {
    const NkRowSide sides[2] = {row.a, row.b};
    const bool shared = row.a.kind == row.b.kind && row.a.index == row.b.index;
    const size_t at = size_t{slot} * p.max_dof;
    const auto* jacobian_b = static_cast<const float*>(data.chain_jacobian_b) + at;
    float response = 0.0f, shares[2] = {0.0f, 0.0f};
    bool valid = true;
    for (uint32_t i = 0u; i < (shared ? 1u : 2u); ++i) {
        const NkRowSide& side = sides[i];
        float contribution = 0.0f;
        if (side.kind == kNkSideRigid) {
            const Vec3 linear = shared ? side.jlin + row.b.jlin : side.jlin;
            const Vec3 angular = shared ? side.jang + row.b.jang : side.jang;
            contribution = data.body_inv_mass[side.index] * linear.Dot(linear) +
                           angular.Dot(data.body_world_inv_inertia[side.index].Multiply(angular));
        } else if (side.kind == kNkSideArtic) {
            const auto* jacobian = i == 0u ? static_cast<const float*>(data.chain_jacobian) + at : jacobian_b;
            const float* const factor = data.m_factor + size_t{side.index} * p.max_dof * p.max_dof;
            contribution = MassCoupling<float>(factor, p.max_dof, jacobian, jacobian,
                                               shared ? jacobian_b : nullptr, shared ? jacobian_b : nullptr);
        }
        valid &= contribution >= 0.0f && isfinite(contribution);
        response += contribution;
        shares[i] = contribution;
    }
    *coupling = shares[0] * shares[1];
    return valid && isfinite(response) ? response : nanf("");
}

// Head row setup as CountIncidenceKernel in float: penalty, response and the row's fixed coefficients; the
// double response stays for the residual measure.
__device__ void InitializeTileHead(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                   const TileContext& c, uint32_t slot, uint32_t h) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const NkRow row = rows[slot];
    const uint32_t stride = row.group_normal_count;
    const uint32_t axes = (row.flags & nk::nk_row_flags::kBlockNormal) != 0u ? 3u : 1u;
    const uint32_t owner = TileOwner(s, c, row.a);
    const uint32_t other = TileOwner(s, c, row.b);
    float coupling = 0.0f;
    const float response = TileInertialResponse(data, p, row, slot, &coupling);
    const float coupled = coupling > 0.0f
        ? (response + sqrtf(response * response + 12.0f * coupling)) / (6.0f * coupling) : FLT_MAX;
    float rho = response > 0.0f ? fminf(p.penalty_scale / response, coupled) : 0.0f;
    if (!(response == 0.0f || (response > 0.0f && rho > 0.0f && rho <= FLT_MAX))) {
        atomicOr(data.env_status + c.env, kEnvStatusSolverFailure);
        VisitOwners(row, PointMasses(data), p, [&](uint32_t failed) {
            RecordBlockFailure(data, p, s, failed, nk::BlockSolveFailure::InvalidRow, slot);
        });
        rho = nanf("");
    }
    const float compliance = row.compliance_alpha * RowStateScale(data, p, slot);
    const auto scalar = nk::augmented::ComputeFloatScalarResponse(rho, compliance);
    s.penalty[slot] = rho;
    if (p.measure_contact_residual != 0u)
        s.scalar_response[slot] = nk::augmented::ComputeScalarResponse(rho, compliance);
    c.Index(kTileSlot, h) = slot;
    c.Index(kTileStride, h) = stride;
    c.Index(kTileAxes, h) = axes;
    // Both sides on one block form a single incidence whose Jacobian sums them.
    c.Index(kTileOwner, h) = owner;
    c.Index(kTileOwner + 1u, h) = other == owner ? kNoTileBlock : other;
    c.Value(kTileRho, h) = rho;
    c.Value(kTileCompliance, h) = compliance;
    c.Value(kTileLower, h) = row.lower;
    c.Value(kTileUpper, h) = row.upper;
    c.Value(kTileMuFirst, h) = row.mu;
    c.Value(kTileMuSecond, h) = row.friction_secondary;
    c.Value(kTileNormalBound, h) = 0.0f;
    c.Value(kTileDualScale, h) = scalar.dual_scale;
    c.Value(kTileResidualScale, h) = scalar.residual_scale;
    c.Store3(kTileDual, h, {data.lambda[slot], axes == 3u ? data.lambda[slot + stride] : 0.0f,
                            axes == 3u ? data.lambda[slot + 2u * stride] : 0.0f});
    c.Store3(kTileResidual, h, {});
    c.Store3(kTileMove, h, {});
    c.Store3(kTileStep, h, {});
}

// Jacobians on moving blocks and residuals at free velocities, including kinematic sides. Single-coordinate
// rows store their coordinate and value; other rows store one vector of block coordinates per axis.
__device__ void InitializeTileRows(DataView data, BlockDescentSolveParams p, BlockScratch s, const TileContext& c) {
    const auto* row_data = reinterpret_cast<const NkRow*>(data.urows);
    const auto* chain_a = static_cast<const float*>(data.chain_jacobian);
    const auto* chain_b = static_cast<const float*>(data.chain_jacobian_b);
    uint32_t rows = 0u;
    for (uint32_t h = 0u; h < c.heads; ++h) {
        const uint32_t slot = c.Index(kTileSlot, h);
        const uint32_t stride = c.Index(kTileStride, h);
        const uint32_t axes = c.Index(kTileAxes, h);
        const float scale = RowStateScale(data, p, slot);
        Vec3 velocity{}, rhs{};
        for (uint32_t incidence = 0u; incidence < 2u; ++incidence) {
            const uint32_t block = c.Index(kTileOwner + incidence, h);
            TileBlock b{};
            if (block != kNoTileBlock) b = LoadTileBlock(data, p, s, c, block);
            const bool coordinate = block != kNoTileBlock && c.lane < b.dimension;
            float value[3]{};
#pragma unroll
            for (uint32_t axis = 0u; axis < 3u; ++axis) {
                if (axis >= axes) continue;
                const uint32_t row_slot = slot + axis * stride;
                const NkRow& row = row_data[row_slot];
                if (coordinate) value[axis] = b.Jacobian(data, p, row, row_slot, c.lane);
                if (incidence == 0u) {
                    SetAxisComponent(velocity, axis, TileSideVelocity(p, s, row.a, chain_a, row_slot, c.lane) +
                                                     TileSideVelocity(p, s, row.b, chain_b, row_slot, c.lane));
                    SetAxisComponent(rhs, axis, __fmul_rn(__fmul_rn(row.rhs, p.dt), axis == 0u ? scale : 1.0f));
                }
            }
            if (block == kNoTileBlock) continue;
            const uint32_t support = __ballot_sync(~0u, value[0] != 0.0f);
            uint32_t row = ~0u, unit = ~0u;
            float unit_value = 0.0f;
            if (axes == 1u && __popc(support) <= 1u) {
                unit = support != 0u ? __ffs(support) - 1u : 0u;
                unit_value = __shfl_sync(~0u, value[0], unit);
            } else {
                row = rows;
#pragma unroll
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                    if (axis < axes && c.lane < c.width) c.jacobian[size_t{row + axis} * c.width + c.lane] = value[axis];
                rows += axes;
            }
            if (c.lane == 0u) {
                c.Index(kTileRow + incidence, h) = row;
                c.Index(kTileUnit + incidence, h) = unit;
                c.Value(kTileUnitValue + incidence, h) = unit_value;
            }
        }
        velocity = TileWarpSum3(velocity);
        if (c.lane == 0u)
            c.Store3(kTileResidual, h, {__fsub_rn(rhs.x, velocity.x), __fsub_rn(rhs.y, velocity.y),
                                        __fsub_rn(rhs.z, velocity.z)});
    }
    __syncwarp();
}

// Free velocities, block coordinates, the environment's active head rows in slot order with their
// Jacobians, and their residuals at the free velocities.
__device__ void InitializeTile(ArticulationDeviceState state, DataView data, BlockDescentSolveParams p,
                               BlockScratch s, TileContext& c) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    if (c.lane < kSolverCountWords)
        data.solver_color_counts[c.env * kSolverCountWords + c.lane] = c.lane == 0u ? p.vertex_blocks.colors : 0u;
    if (c.lane < kSolverPhaseWords) data.solver_phase_time[c.env * kSolverPhaseWords + c.lane] = 0u;
    if (c.lane == 0u) data.vbd_velocity_sweep_count[c.env] = p.iterations;
    if (p.clear_failure_diagnostics != 0u) {
        for (uint32_t k = c.lane; k < c.bodies + c.articulations; k += kTileLanes) {
            const uint32_t owner = p.total_particle_count + (k < c.bodies
                ? c.first_body + k : p.total_body_count + c.first_articulation + k - c.bodies);
            s.failure_reasons[owner] = 0u;
            s.failure_rows[owner] = ~0u;
            s.failure_substeps[owner] = ~0u;
            for (uint32_t column = 0u; column < nk::kBlockSolveFailureEquationColumnCount; ++column)
                s.failure_equations[size_t{owner} * nk::kBlockSolveFailureEquationColumnCount + column] = 0.0;
        }
        __syncwarp();
    }
    for (uint32_t k = c.lane; k < c.bodies; k += kTileLanes) {
        const uint32_t body = c.first_body + k;
        InitializeRigidBlock(data, p, s, body);
        for (uint32_t i = 0u; i < kRigidBlockDof; ++i)
            s.rigid_velocity[size_t{body} * kRigidBlockDof + i] = s.rigid_free[size_t{body} * kRigidBlockDof + i];
    }
    for (uint32_t k = 0u; k < c.articulations; ++k) {
        const uint32_t articulation = c.first_articulation + k;
        const size_t base = size_t{articulation} * p.max_dof;
        uint32_t link = 0u, component = 0u;
        const bool located = c.lane < p.max_dof &&
            ArticulationDofLocation(state, articulation, c.lane, &link, &component);
        const float value = located
            ? (component != ~0u ? state.link_velocity[link].v[component] : state.qdot[link]) : 0.0f;
        const bool root = located && (p.mimic_couplings == 0u || data.mimic_root_dof[base + c.lane] == c.lane);
        const uint32_t roots = __ballot_sync(~0u, root);
        if (root) {
            const uint32_t compact = __popc(roots & ((1u << c.lane) - 1u));
            s.articulation_roots[base + compact] = c.lane;
            s.articulation_free[base + compact] = value;
            s.articulation_velocity[base + compact] = value;
        }
        if (c.lane == 0u) s.articulation_dimensions[articulation] = __popc(roots);
    }
    __syncwarp();
    const uint32_t first = c.env * p.rows_per_env;
    const uint32_t bound = ContactRowBound(data.contact_row_extent, c.env, p.contact_rows_per_env);
    uint32_t heads = 0u;
    for (uint32_t range = 0u; range < 2u; ++range) {
        const uint32_t begin = range == 0u ? 0u : p.contact_rows_per_env;
        const uint32_t end = range == 0u ? bound : p.rows_per_env;
        for (uint32_t base = begin; base < end; base += kTileLanes) {
            const uint32_t local = base + c.lane;
            const uint32_t flags = local < end ? rows[first + local].flags : 0u;
            const bool head = (flags & nk::nk_row_flags::kActive) != 0u &&
                              (flags & nk::nk_row_flags::kBlockTangent) == 0u;
            const uint32_t mask = __ballot_sync(~0u, head);
            if (head) InitializeTileHead(data, p, s, c, first + local,
                heads + __popc(mask & ((1u << c.lane) - 1u)));
            heads += __popc(mask);
        }
    }
    c.heads = heads;
    __syncwarp();
    if (p.measure_contact_residual != 0u) {
        for (uint32_t base = 0u; base < heads; base += kTileLanes) {
            uint32_t at = 0u;
            if (c.lane == 0u) at = atomicAdd(s.active_count, min(kTileLanes, heads - base));
            at = __shfl_sync(~0u, at, 0u);
            if (base + c.lane < heads) s.active_rows[at + c.lane] = c.Index(kTileSlot, base + c.lane);
        }
    }
    InitializeTileRows(data, p, s, c);
}

// The normal bound each head on `block` holds through its solve, and a zero move.
__device__ void StartTileBlock(const TileContext& c, uint32_t block) {
    for (uint32_t h = c.lane; h < c.heads; h += kTileLanes) {
        if (c.Incidence(h, block) == kNoTileIncidence) continue;
        c.Value(kTileNormalBound, h) = nk::augmented::EvaluateScalarImpulse(c.Value(kTileDual, h),
            c.Value(kTileRho, h), c.Value(kTileResidual, h), c.Value(kTileCompliance, h), c.Value(kTileLower, h),
            c.Value(kTileUpper, h), c.Response(h)).impulse;
        c.Store3(kTileMove, h, {});
    }
    __syncwarp();
}

// The block's moves enter the residuals of the heads it touches.
__device__ void EndTileBlock(const TileContext& c, uint32_t block) {
    for (uint32_t h = c.lane; h < c.heads; h += kTileLanes) {
        if (c.Incidence(h, block) == kNoTileIncidence) continue;
        const Vec3 residual = c.Load3(kTileResidual, h);
        const Vec3 move = c.Load3(kTileMove, h);
        c.Store3(kTileResidual, h, {__fsub_rn(residual.x, move.x), __fsub_rn(residual.y, move.y),
                                    __fsub_rn(residual.z, move.z)});
    }
    __syncwarp();
}

// Lane i's row of H = M + sum J^T C J and its force g_i = -[M(u - u_free)]_i + sum J_i f at the current
// moves, as AssembleDenseBlock; `converged` reports every |g_i| within rounding of its terms.
template <uint32_t kN>
__device__ bool AssembleTileBlock(const TileContext& c, const TileBlock& b, float u, float free, float (&h)[kN],
                                  float* force, float* inertia, bool* converged) {
    constexpr uint32_t kS = kN + 1u;
    const bool coordinate = c.lane < b.dimension;
    const bool row = c.lane < kN;
    const float difference = u - free;
    const float part = difference - u;
    const float difference_error = (u - (difference - part)) + (-free - part);
    TileSum inertial{};
    float scale = 0.0f;
#pragma unroll
    for (uint32_t k = 0u; k < kN; ++k) {
        const float mass = row ? c.mass[c.lane * kS + k] : 0.0f;
        const float high = __shfl_sync(~0u, difference, k);
        const float low = __shfl_sync(~0u, difference_error, k);
        h[k] = mass;
        TileAddProduct(inertial, mass, high);
        TileAddProduct(inertial, mass, low);
        scale += fabsf(mass) * fabsf(high);
    }
    TileSum g{-inertial.value, -inertial.error};
    float diagonal = 0.0f;
    for (uint32_t chunk = 0u; chunk < c.heads; chunk += kTileLanes) {
        const uint32_t own = chunk + c.lane;
        const uint32_t incidence = own < c.heads ? c.Incidence(own, b.local) : kNoTileIncidence;
        Vec3 f{}, noise{};
        SymmetricMat3 curvature{};
        uint32_t unit = ~0u, first = 0u, axes = 1u;
        float unit_value = 0.0f;
        if (incidence != kNoTileIncidence) {
            const TileHead t = LoadTileHead(c, own);
            TileResponse(t, t.move, &f, &curvature);
            // A rounding of the residual or move changes the impulse by the curvature times it.
            noise = {fabsf(f.x) + curvature.xx * (fabsf(t.residual.x) + fabsf(t.move.x)),
                     fabsf(f.y) + curvature.yy * (fabsf(t.residual.y) + fabsf(t.move.y)),
                     fabsf(f.z) + curvature.zz * (fabsf(t.residual.z) + fabsf(t.move.z))};
            unit = c.Index(kTileUnit + incidence, own);
            first = c.Index(kTileRow + incidence, own);
            unit_value = c.Value(kTileUnitValue + incidence, own);
            axes = t.axes;
        }
        // A row on one coordinate adds to that coordinate's diagonal and force only.
        for (uint32_t pending = __ballot_sync(~0u, incidence != kNoTileIncidence && unit != ~0u); pending != 0u;
             pending &= pending - 1u) {
            const uint32_t source = __ffs(pending) - 1u;
            const uint32_t target = __shfl_sync(~0u, unit, source);
            const float v = __shfl_sync(~0u, unit_value, source);
            const float fx = __shfl_sync(~0u, f.x, source);
            const float cxx = __shfl_sync(~0u, curvature.xx, source);
            const float nx = __shfl_sync(~0u, noise.x, source);
            if (c.lane == target) {
                diagonal += v * cxx * v;
                TileAddProduct(g, v, fx);
                scale += fabsf(v) * nx;
            }
        }
        for (uint32_t pending = __ballot_sync(~0u, incidence != kNoTileIncidence && unit == ~0u); pending != 0u;
             pending &= pending - 1u) {
            const uint32_t source = __ffs(pending) - 1u;
            const bool staged = c.lane < c.width;
            const float* jacobian = c.jacobian + size_t{__shfl_sync(~0u, first, source)} * c.width +
                                    (staged ? c.lane : 0u);
            const uint32_t source_axes = __shfl_sync(~0u, axes, source);
            const float fx = __shfl_sync(~0u, f.x, source), cxx = __shfl_sync(~0u, curvature.xx, source);
            const float nx = __shfl_sync(~0u, noise.x, source);
            const float j0 = staged ? jacobian[0] : 0.0f;
            TileAddProduct(g, j0, fx);
            scale += fabsf(j0) * nx;
            if (source_axes == 1u) {
                const float a0 = j0 * cxx;
#pragma unroll
                for (uint32_t k = 0u; k < kN; ++k) h[k] += a0 * __shfl_sync(~0u, j0, k);
                continue;
            }
            const float fy = __shfl_sync(~0u, f.y, source), fz = __shfl_sync(~0u, f.z, source);
            const float cyy = __shfl_sync(~0u, curvature.yy, source), czz = __shfl_sync(~0u, curvature.zz, source);
            const float cyz = __shfl_sync(~0u, curvature.yz, source);
            const float ny = __shfl_sync(~0u, noise.y, source), nz = __shfl_sync(~0u, noise.z, source);
            const float j1 = staged ? jacobian[c.width] : 0.0f, j2 = staged ? jacobian[2u * c.width] : 0.0f;
            TileAddProduct(g, j1, fy);
            TileAddProduct(g, j2, fz);
            scale += fabsf(j1) * ny + fabsf(j2) * nz;
            const float a0 = j0 * cxx, a1 = j1 * cyy + j2 * cyz, a2 = j2 * czz + j1 * cyz;
#pragma unroll
            for (uint32_t k = 0u; k < kN; ++k)
                h[k] += a0 * __shfl_sync(~0u, j0, k) + a1 * __shfl_sync(~0u, j1, k) + a2 * __shfl_sync(~0u, j2, k);
        }
    }
#pragma unroll
    for (uint32_t k = 0u; k < kN; ++k) h[k] += c.lane == k ? diagonal : 0.0f;
    *force = coordinate ? g.value + g.error : 0.0f;
    *inertia = coordinate ? inertial.value + inertial.error : 0.0f;
    bool finite = isfinite(*force);
    // A float velocity is off its optimum by up to a rounding, which moves g by H times it.
#pragma unroll
    for (uint32_t k = 0u; k < kN; ++k) {
        finite &= isfinite(h[k]);
        scale += fabsf(h[k]) * fabsf(__shfl_sync(~0u, u, k));
    }
    *converged = __all_sync(~0u, !coordinate || fabsf(*force) <= kDenseRoundings * FLT_EPSILON * scale);
    return __all_sync(~0u, finite);
}

// Cholesky factor in place: lane i keeps row i of L and the reciprocal of its diagonal; false at a
// nonpositive pivot.
template <uint32_t kN>
__device__ bool FactorTileBlock(float (&h)[kN], float* inverse, uint32_t lane, uint32_t dimension) {
    bool valid = true;
    float reciprocal = 0.0f;
#pragma unroll
    for (uint32_t k = 0u; k < kN; ++k) {
        if (k >= dimension) break;
        float value = h[k];
#pragma unroll
        for (uint32_t j = 0u; j < k; ++j) value -= h[j] * __shfl_sync(~0u, h[j], k);
        const float pivot = __shfl_sync(~0u, value, k);
        valid &= pivot > 0.0f && pivot <= FLT_MAX;
        const float root = sqrtf(valid ? pivot : 1.0f);
        const float scale = 1.0f / root;
        h[k] = lane == k ? root : value * scale;
        reciprocal = lane == k ? scale : reciprocal;
    }
    *inverse = reciprocal;
    return valid;
}

// d = H^-1 g: forward substitution on L's rows, back substitution on its columns stored transposed.
template <uint32_t kN>
__device__ float SolveTileBlock(const float (&h)[kN], float inverse, float g, const TileContext& c, uint32_t dimension) {
    constexpr uint32_t kS = kN + 1u;
    float r = g;
#pragma unroll
    for (uint32_t j = 0u; j < kN; ++j) {
        if (j >= dimension) break;
        const float y = __shfl_sync(~0u, r * inverse, j);
        r = c.lane == j ? y : c.lane > j ? fmaf(-h[j], y, r) : r;
        if (c.lane > j && c.lane < kN) c.factor[j * kS + c.lane] = h[j];
    }
    __syncwarp();
#pragma unroll
    for (uint32_t k = kN; k > 0u; --k) {
        if (k > dimension) continue;
        const uint32_t j = k - 1u;
        const float x = __shfl_sync(~0u, r * inverse, j);
        const float l = c.lane < j ? c.factor[c.lane * kS + j] : 0.0f;
        r = c.lane == j ? x : fmaf(-l, x, r);
    }
    __syncwarp();
    return r;
}

// w = J d of every head on the block: a row on one coordinate scales that coordinate's step, other rows
// sum over the warp.
__device__ void TileRowSteps(const TileContext& c, uint32_t block, float d) {
    for (uint32_t chunk = 0u; chunk < c.heads; chunk += kTileLanes) {
        const uint32_t own = chunk + c.lane;
        const uint32_t incidence = own < c.heads ? c.Incidence(own, block) : kNoTileIncidence;
        const bool touches = incidence != kNoTileIncidence;
        const uint32_t unit = touches ? c.Index(kTileUnit + incidence, own) : ~0u;
        const uint32_t first = touches ? c.Index(kTileRow + incidence, own) : 0u;
        const uint32_t axes = touches ? c.Index(kTileAxes, own) : 1u;
        const float along = __shfl_sync(~0u, d, unit != ~0u ? unit : 0u);
        Vec3 step{};
        if (unit != ~0u) step.x = c.Value(kTileUnitValue + incidence, own) * along;
        for (uint32_t pending = __ballot_sync(~0u, touches && unit == ~0u); pending != 0u; pending &= pending - 1u) {
            const uint32_t source = __ffs(pending) - 1u;
            const bool staged = c.lane < c.width;
            const float* jacobian = c.jacobian + size_t{__shfl_sync(~0u, first, source)} * c.width +
                                    (staged ? c.lane : 0u);
            const float j0 = staged ? jacobian[0] : 0.0f;
            Vec3 w{};
            if (__shfl_sync(~0u, axes, source) == 1u) w.x = TileWarpSum(j0 * d);
            else w = TileWarpSum3({j0 * d, (staged ? jacobian[c.width] : 0.0f) * d,
                                   (staged ? jacobian[2u * c.width] : 0.0f) * d});
            if (c.lane == source) step = w;
        }
        if (touches) c.Store3(kTileStep, own, step);
    }
    __syncwarp();
}

// Slope and curvature of the local objective at `alpha` along the step, as DenseLineDerivatives.
__device__ void TileLineDerivatives(const TileContext& c, uint32_t block, float alpha, float inertial_slope,
                                    float inertial_curvature, const TileHead& resident, bool resident_incidence,
                                    float* slope, float* curvature, float* magnitude) {
    TileSum row_slope{};
    float row_curvature = 0.0f, row_magnitude = 0.0f;
    for (uint32_t h = c.lane; h < c.heads; h += kTileLanes) {
        if (h < kTileLanes ? !resident_incidence : c.Incidence(h, block) == kNoTileIncidence) continue;
        const TileHead t = h < kTileLanes ? resident : LoadTileHead(c, h);
        const Vec3 trial{fmaf(alpha, t.step.x, t.move.x), fmaf(alpha, t.step.y, t.move.y),
                         fmaf(alpha, t.step.z, t.move.z)};
        Vec3 f{};
        SymmetricMat3 response{};
        TileResponse(t, trial, &f, &response);
        TileAddProduct(row_slope, f.x, t.step.x);
        TileAddProduct(row_slope, f.y, t.step.y);
        TileAddProduct(row_slope, f.z, t.step.z);
        row_magnitude += fabsf(f.x * t.step.x) + fabsf(f.y * t.step.y) + fabsf(f.z * t.step.z);
        row_curvature += response.xx * t.step.x * t.step.x + response.yy * t.step.y * t.step.y +
                         response.zz * t.step.z * t.step.z + 2.0f * response.yz * t.step.y * t.step.z;
    }
    TileSum total{-row_slope.value, -row_slope.error};
    if (c.lane == 0u) {
        TileAddProduct(total, alpha, inertial_curvature);
        TileAdd(total, inertial_slope);
    }
    *slope = TileWarpTotal(total);
    *curvature = inertial_curvature + TileWarpSum(row_curvature);
    *magnitude = fabsf(inertial_slope) + alpha * inertial_curvature + TileWarpSum(row_magnitude);
}

// Safeguarded Newton iteration on the slope as SearchDenseLine; 0 when the step does not descend.
__device__ float SearchTileLine(const TileContext& c, uint32_t block, float inertial_slope,
                                float inertial_curvature) {
    const bool incident = c.lane < c.heads && c.Incidence(c.lane, block) != kNoTileIncidence;
    TileHead resident{};
    if (incident) resident = LoadTileHead(c, c.lane);
    float slope = 0.0f, curvature = 0.0f, magnitude = 0.0f;
    TileLineDerivatives(c, block, 0.0f, inertial_slope, inertial_curvature, resident, incident,
                        &slope, &curvature, &magnitude);
    const float initial = slope;
    if (!(initial < 0.0f)) return 0.0f;
    float low = 0.0f, high = 0.0f, alpha = 1.0f;
    bool bounded = false;
    for (uint32_t step = 0u; step < kDenseLineSearchSteps; ++step) {
        TileLineDerivatives(c, block, alpha, inertial_slope, inertial_curvature, resident, incident,
                            &slope, &curvature, &magnitude);
        if (!isfinite(slope) || !isfinite(curvature)) {
            high = alpha;
            bounded = true;
            alpha = 0.5f * (low + high);
            continue;
        }
        if (fabsf(slope) <= fmaxf(static_cast<float>(kDenseLineSearchTolerance) * -initial,
                                  kDenseRoundings * FLT_EPSILON * magnitude)) return alpha;
        if (slope < 0.0f) low = alpha;
        else {
            high = alpha;
            bounded = true;
        }
        if (bounded && high - low <= FLT_EPSILON * high) break;
        float next = alpha - slope / curvature;
        if (!(next > low && (!bounded || next < high))) next = bounded ? 0.5f * (low + high) : 2.0f * alpha;
        alpha = next;
    }
    return low;
}

// Newton steps with an exact line search on the block's local augmented Lagrangian, as DescendDenseBlock.
template <uint32_t kN>
__device__ void DescendTileBlock(DataView data, BlockDescentSolveParams p, BlockScratch s, TileContext& c,
                                 const TileBlock& b) {
    constexpr uint32_t kS = kN + 1u;
    const uint32_t n = b.dimension;
    const bool coordinate = c.lane < n;
    // Coordinates past the block's carry unit mass and no rows, so their step stays zero.
    if (b.local != c.staged) {
        for (uint32_t i = 0u; i < kN; ++i)
            if (c.lane < kN)
                c.mass[i * kS + c.lane] = i < n && coordinate ? b.Mass(i, c.lane) : i == c.lane ? 1.0f : 0.0f;
        c.staged = b.local;
    }
    StartTileBlock(c, b.local);
    const float start = coordinate ? b.velocity[c.lane] : 0.0f;
    const float free = coordinate && b.free != nullptr ? b.free[c.lane] : 0.0f;
    float u = start;
    nk::BlockSolveFailure failure = nk::BlockSolveFailure::None;
    float first_decrement = 0.0f;
    for (uint32_t step = 0u; step < kDenseNewtonSteps; ++step) {
        float h[kN];
        float force = 0.0f, inertia = 0.0f, inverse = 0.0f;
        bool converged = false;
        if (!AssembleTileBlock<kN>(c, b, u, free, h, &force, &inertia, &converged)) {
            failure = nk::BlockSolveFailure::InvalidEquation;
            break;
        }
        if (converged) break;
        if (!FactorTileBlock<kN>(h, &inverse, c.lane, n)) {
            failure = nk::BlockSolveFailure::Factorization;
            break;
        }
        const float d = SolveTileBlock<kN>(h, inverse, force, c, n);
        float mass_step = 0.0f;
#pragma unroll
        for (uint32_t k = 0u; k < kN; ++k)
            mass_step += (c.lane < kN ? c.mass[c.lane * kS + k] : 0.0f) * __shfl_sync(~0u, d, k);
        TileSum decrement{}, slope{}, curvature{};
        if (coordinate) {
            TileAddProduct(decrement, force, d);
            TileAddProduct(slope, inertia, d);
            TileAddProduct(curvature, d, mass_step);
        }
        const float total_decrement = TileWarpTotal(decrement);
        const float inertial_slope = TileWarpTotal(slope);
        const float inertial_curvature = TileWarpTotal(curvature);
        if (!isfinite(total_decrement) || !isfinite(inertial_slope) || !isfinite(inertial_curvature)) {
            failure = nk::BlockSolveFailure::InvalidDirection;
            break;
        }
        if (!(total_decrement > 0.0f)) break;
        if (step == 0u) first_decrement = total_decrement;
        else if (total_decrement <= static_cast<float>(kDenseDecrementRatio) * first_decrement) break;
        TileRowSteps(c, b.local, d);
        const float alpha = SearchTileLine(c, b.local, inertial_slope, inertial_curvature);
        const float after = fmaf(alpha, d, u);
        const bool finite = !coordinate || isfinite(after);
        const float largest_step = TileWarpMax(coordinate ? fabsf(after - u) : 0.0f);
        const float largest_value =
            TileWarpMax(coordinate ? fmaxf(fabsf(after), fmaxf(fabsf(start), fabsf(free))) : 0.0f);
        if (!__all_sync(~0u, finite)) {
            failure = nk::BlockSolveFailure::InvalidCandidate;
            break;
        }
        if (!(alpha > 0.0f)) break;
        u = after;
        for (uint32_t head = c.lane; head < c.heads; head += kTileLanes) {
            if (c.Incidence(head, b.local) == kNoTileIncidence) continue;
            const Vec3 move = c.Load3(kTileMove, head);
            const Vec3 row_step = c.Load3(kTileStep, head);
            c.Store3(kTileMove, head, {fmaf(alpha, row_step.x, move.x), fmaf(alpha, row_step.y, move.y),
                                       fmaf(alpha, row_step.z, move.z)});
        }
        __syncwarp();
        if (largest_step <= kDenseRoundings * FLT_EPSILON * largest_value) break;
    }
    EndTileBlock(c, b.local);
    if (failure != nk::BlockSolveFailure::None && c.lane == 0u) RecordTileFailure(data, p, s, c, b.owner, failure);
    if (coordinate) b.velocity[c.lane] = u;
    __syncwarp();
}

// Dual update as DualUpdateKernel at the velocities every block reached in this iteration; position pass
// impulses stay in the tile.
__device__ void UpdateTileDuals(DataView data, const TileContext& c, bool velocity = true) {
    for (uint32_t h = c.lane; h < c.heads; h += kTileLanes) {
        const TileHead t = LoadTileHead(c, h);
        const uint32_t slot = c.Index(kTileSlot, h);
        const uint32_t stride = c.Index(kTileStride, h);
        const float normal = nk::augmented::EvaluateScalarImpulse(t.dual.x, t.rho, t.residual.x, t.compliance,
                                                                  t.lower, t.upper, t.response).impulse;
        if (velocity) data.lambda[slot] = normal;
        Vec3 dual{normal, 0.0f, 0.0f};
        if (t.axes == 3u) {
            const auto f = nk::augmented::EvaluateTangentImpulse(t.dual, t.rho, t.residual, normal,
                                                                 t.mu_first, t.mu_second);
            data.lambda[slot + stride] = f.impulse.y;
            data.lambda[slot + 2u * stride] = f.impulse.z;
            dual.y = f.impulse.y;
            dual.z = f.impulse.z;
        }
        c.Store3(kTileDual, h, dual);
    }
    __syncwarp();
}

// Final velocities as the tails of DescendArticulationsKernel and DescendRigidBlocksKernel, then the
// joint limit impulses as WriteBlockJointLimitImpulsesKernel.
__device__ void FinishTile(ArticulationDeviceState state, DataView data, BlockDescentSolveParams p,
                           BlockScratch s, const TileContext& c) {
    for (uint32_t k = 0u; k < c.articulations; ++k) {
        const uint32_t articulation = c.first_articulation + k;
        const size_t base = size_t{articulation} * p.max_dof;
        const uint32_t dof = c.lane;
        uint32_t link = 0u, component = 0u;
        const bool located = dof < p.max_dof && ArticulationDofLocation(state, articulation, dof, &link, &component);
        float value = 0.0f;
        if (located) {
            const uint32_t root = p.mimic_couplings != 0u ? data.mimic_root_dof[base + dof] : dof;
            for (uint32_t compact = 0u; compact < s.articulation_dimensions[articulation]; ++compact) {
                if (s.articulation_roots[base + compact] == root) {
                    value = s.articulation_velocity[base + compact];
                    break;
                }
            }
            if (p.mimic_couplings != 0u) value *= data.mimic_root_scale[base + dof];
        }
        if (__any_sync(~0u, !(fabsf(value) <= FLT_MAX))) {
            if (c.lane == 0u)
                RecordTileFailure(data, p, s, c, p.total_particle_count + p.total_body_count + articulation,
                                  nk::BlockSolveFailure::InvalidCandidate);
            continue;
        }
        if (dof >= p.max_dof) continue;
        data.qdot_flat[base + dof] = value;
        if (!located) continue;
        if (component == ~0u) state.qdot[link] = value;
        else state.link_velocity[link].v[component] = value;
    }
    for (uint32_t k = c.lane; k < c.bodies; k += kTileLanes) {
        const uint32_t body = c.first_body + k;
        if (s.rigid_dimensions[body] == 0u) continue;
        const float* u = s.rigid_velocity + size_t{body} * kRigidBlockDof;
        data.body_linear_velocity[body] = {u[0], u[1], u[2]};
        data.body_angular_velocity[body] = {u[3], u[4], u[5]};
    }
    if (p.base_link_count == 0u || p.joint_limit_rows_per_env < 2u * p.base_link_count ||
        data.joint_limit_impulse == nullptr) return;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    for (uint32_t local_link = c.lane; local_link < p.base_link_count; local_link += kTileLanes) {
        const uint64_t link = uint64_t{c.env} * p.base_link_count + local_link;
        const size_t lower = size_t{c.env} * p.rows_per_env + p.contact_rows_per_env + size_t{local_link} * 2u;
        data.joint_limit_impulse[2u * link] =
            (rows[lower].flags & nk::nk_row_flags::kActive) != 0u ? data.lambda[lower] : 0.0f;
        data.joint_limit_impulse[2u * link + 1u] =
            (rows[lower + 1u].flags & nk::nk_row_flags::kActive) != 0u ? data.lambda[lower + 1u] : 0.0f;
    }
}

// Rows live as in the row solve's position pass push out from rest by beta of their excess over the slop; the
// pass takes no iteration without a push and one unless a live row couples two blocks, as block solves are exact.
__device__ uint32_t InitializeTilePositions(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                            const TileContext& c) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    bool push = false, coupled = false;
    for (uint32_t h = c.lane; h < c.heads; h += kTileLanes) {
        const uint32_t slot = c.Index(kTileSlot, h);
        const float penetration = data.row_penetration[slot];
        const bool position =
            (rows[slot].flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kVelocityOnly)) == 0u &&
            (penetration > p.position_slop || c.Value(kTileDual, h) != 0.0f);
        const float bias = position ? fminf(p.position_beta * fmaxf(penetration - p.position_slop, 0.0f) / p.dt,
                                            p.position_max_velocity) : 0.0f;
        const auto scalar = nk::augmented::ComputeFloatScalarResponse(c.Value(kTileRho, h), 0.0f);
        if (!position) c.Index(kTileOwner, h) = c.Index(kTileOwner + 1u, h) = kNoTileBlock;
        coupled |= c.Index(kTileOwner, h) != kNoTileBlock && c.Index(kTileOwner + 1u, h) != kNoTileBlock;
        c.Index(kTileAxes, h) = 1u;
        c.Value(kTileCompliance, h) = 0.0f;
        c.Value(kTileLower, h) = 0.0f;
        c.Value(kTileUpper, h) = FLT_MAX;
        c.Value(kTileDualScale, h) = scalar.dual_scale;
        c.Value(kTileResidualScale, h) = scalar.residual_scale;
        c.Store3(kTileResidual, h, {bias, 0.0f, 0.0f});
        c.Store3(kTileDual, h, {});
        push |= bias > 0.0f;
    }
    for (uint32_t k = c.lane; k < c.bodies * kRigidBlockDof; k += kTileLanes)
        s.rigid_pseudo[size_t{c.first_body} * kRigidBlockDof + k] = 0.0f;
    for (uint32_t k = 0u; k < c.articulations; ++k)
        if (c.lane < p.max_dof) s.articulation_pseudo[size_t{c.first_articulation + k} * p.max_dof + c.lane] = 0.0f;
    __syncwarp();
    if (!__any_sync(~0u, push)) return 0u;
    return __any_sync(~0u, coupled) ? p.position_iterations : 1u;
}

// Pseudo velocities for IntegratePosition, expanded from block coordinates as FinishTile.
__device__ void FinishTilePositions(ArticulationDeviceState state, DataView data, BlockDescentSolveParams p,
                                    BlockScratch s, const TileContext& c) {
    auto* link_pseudo = reinterpret_cast<LinkSpatialVel*>(data.link_velocity_pseudo);
    for (uint32_t k = 0u; k < c.articulations; ++k) {
        const uint32_t articulation = c.first_articulation + k;
        const size_t base = size_t{articulation} * p.max_dof;
        uint32_t link = 0u, component = 0u;
        if (c.lane >= p.max_dof || !ArticulationDofLocation(state, articulation, c.lane, &link, &component)) continue;
        const uint32_t root = p.mimic_couplings != 0u ? data.mimic_root_dof[base + c.lane] : c.lane;
        float value = 0.0f;
        for (uint32_t compact = 0u; compact < s.articulation_dimensions[articulation]; ++compact)
            if (s.articulation_roots[base + compact] == root) value = s.articulation_pseudo[base + compact];
        if (p.mimic_couplings != 0u) value *= data.mimic_root_scale[base + c.lane];
        if (component == ~0u) data.qdot_pseudo[link] = value;
        else link_pseudo[link].v[component] = value;
    }
    for (uint32_t k = c.lane; k < c.bodies; k += kTileLanes) {
        const uint32_t body = c.first_body + k;
        const float* u = s.rigid_pseudo + size_t{body} * kRigidBlockDof;
        data.body_pseudo_linear_velocity[body] = {u[0], u[1], u[2]};
        data.body_pseudo_angular_velocity[body] = {u[3], u[4], u[5]};
    }
}

// One Gauss-Seidel pass over the tile's blocks.
template <uint32_t kN>
__device__ void DescendTileBlocks(DataView data, BlockDescentSolveParams p, BlockScratch s, TileContext& c,
                                  bool position) {
    for (uint32_t base = 0u; base < c.bodies + c.articulations; base += kTileLanes) {
        const uint32_t own = base + c.lane;
        const uint32_t dimension = own < c.bodies ? s.rigid_dimensions[c.first_body + own]
            : own < c.bodies + c.articulations
                ? s.articulation_dimensions[c.first_articulation + own - c.bodies] : 0u;
        for (uint32_t pending = __ballot_sync(~0u, dimension != 0u); pending != 0u; pending &= pending - 1u) {
            const uint32_t local = base + __ffs(pending) - 1u;
            const TileBlock b = LoadTileBlock(data, p, s, c, local, position);
            if (local < c.bodies) DescendTileBlock<kRigidBlockDof>(data, p, s, c, b);
            else DescendTileBlock<kN>(data, p, s, c, b);
        }
    }
}

// An environment's velocity iterations, then its split-impulse position pass.
template <uint32_t kN>
__device__ void DescendTile(ArticulationDeviceState state, DataView data, BlockDescentSolveParams p, BlockScratch s,
                            TileContext& c) {
    InitializeTile(state, data, p, s, c);
    for (uint32_t iteration = 0u; iteration < p.iterations; ++iteration) {
        c.iteration = iteration;
        DescendTileBlocks<kN>(data, p, s, c, false);
        UpdateTileDuals(data, c);
    }
    FinishTile(state, data, p, s, c);
    if (p.position_iterations == 0u) return;
    const uint32_t positions = InitializeTilePositions(data, p, s, c);
    for (uint32_t iteration = 0u; iteration < positions; ++iteration) {
        c.iteration = p.iterations + iteration;
        DescendTileBlocks<kN>(data, p, s, c, true);
        if (iteration + 1u < positions) UpdateTileDuals(data, c, false);
    }
    FinishTilePositions(state, data, p, s, c);
}

constexpr uint32_t kOrderTileThreads = 1024u;
constexpr uint32_t kTileCostBuckets = 128u;

// A solve's cycles in quarter octaves.
__device__ uint32_t TileCostBucket(uint32_t cycles) {
    const uint32_t octave = cycles != 0u ? 31u - __clz(cycles) : 0u;
    return octave * 4u + (octave >= 2u ? (cycles >> (octave - 2u)) & 3u : 0u);
}

// Environments costliest first by their last solve, so the resident warps start the longest solves first.
__global__ void __launch_bounds__(kOrderTileThreads) OrderTilesKernel(BlockDescentSolveParams p, BlockScratch s) {
    __shared__ uint32_t offsets[kTileCostBuckets];
    for (uint32_t k = threadIdx.x; k < kTileCostBuckets; k += blockDim.x) offsets[k] = 0u;
    __syncthreads();
    for (uint32_t env = threadIdx.x; env < p.env_count; env += blockDim.x)
        atomicAdd(offsets + TileCostBucket(s.tile_cost[env]), 1u);
    __syncthreads();
    if (threadIdx.x == 0u) {
        uint32_t total = 0u;
        for (uint32_t k = kTileCostBuckets; k-- > 0u;) {
            const uint32_t count = offsets[k];
            offsets[k] = total;
            total += count;
        }
    }
    __syncthreads();
    for (uint32_t env = threadIdx.x; env < p.env_count; env += blockDim.x)
        s.tile_order[atomicAdd(offsets + TileCostBucket(s.tile_cost[env]), 1u)] = env;
}

// kN is the articulation coordinates rounded up to a multiple of four; rigid bodies descend at six. Resident
// warps take environments in OrderTilesKernel's order, so a costly environment holds back only its own warp.
template <uint32_t kN>
__global__ void __launch_bounds__(kTileWarps * kTileLanes) DescendDenseTilesKernel(
    ArticulationDeviceState state, DataView data, BlockDescentSolveParams p, BlockScratch s) {
    extern __shared__ float tile_shared[];
    const uint32_t warp = threadIdx.x / kTileLanes;
    const uint32_t lane = threadIdx.x % kTileLanes;
    const uint32_t slot = blockIdx.x * kTileWarps + warp;
    for (;;) {
        uint32_t ticket = 0u;
        if (lane == 0u) ticket = atomicAdd(s.tile_next, 1u);
        ticket = __shfl_sync(~0u, ticket, 0u);
        if (ticket >= p.env_count) return;
        const uint32_t env = s.tile_order[ticket];
        TileContext c{};
        c.env = env;
        c.lane = lane;
        c.bodies = p.total_body_count / p.env_count;
        c.articulations = p.articulation_count / p.env_count;
        c.first_body = c.env * c.bodies;
        c.first_articulation = c.env * c.articulations;
        c.capacity = s.tile_head_capacity;
        c.staged = kNoTileBlock;
        c.width = kN;
        c.fields = s.tile_fields + size_t{slot} * kTileFields * c.capacity;
        c.jacobian = s.tile_jacobian + size_t{slot} * s.tile_row_capacity * kN;
        c.mass = tile_shared + size_t{warp} * 2u * kN * (kN + 1u);
        c.factor = c.mass + kN * (kN + 1u);
        c.iteration = ~0u;
        const long long start = clock64();
        DescendTile<kN>(state, data, p, s, c);
        if (lane == 0u) s.tile_cost[env] = static_cast<uint32_t>(min(clock64() - start, 0xffffffffll));
    }
}

template <uint32_t kN>
constexpr size_t TileSharedBytes() {
    return size_t{kTileWarps} * 2u * kN * (kN + 1u) * sizeof(float);
}

// Blocks of the tile kernel resident for p's environments, with the cache carveout the launch uses.
template <uint32_t kN>
cudaError_t TileResidentBlocks(const BlockDescentSolveParams& p, uint32_t* blocks) {
    const cudaError_t status =
        PreferCacheCarveout(DescendDenseTilesKernel<kN>, kTileWarps * kTileLanes, TileSharedBytes<kN>());
    if (status != cudaSuccess) return status;
    return ResidentGridSize(DescendDenseTilesKernel<kN>, kTileWarps * kTileLanes, TileSharedBytes<kN>(),
                            (p.env_count + kTileWarps - 1u) / kTileWarps, blocks);
}

template <uint32_t kN>
cudaError_t LaunchDenseTiles(ArticulationDeviceState state, DataView data, BlockDescentSolveParams p,
                             BlockScratch s, cudaStream_t stream) {
    uint32_t blocks = 0u;
    const cudaError_t status = TileResidentBlocks<kN>(p, &blocks);
    if (status != cudaSuccess) return status;
    LaunchCuda(DescendDenseTilesKernel<kN>, dim3(blocks), dim3(kTileWarps * kTileLanes), TileSharedBytes<kN>(),
               stream, state, data, p, s);
    return cudaGetLastError();
}

// Calls visit with p's tile width as an integral constant.
template <typename Visit>
cudaError_t VisitTileWidth(const BlockDescentSolveParams& p, Visit&& visit) {
    switch (TileWidth(p)) {
        case 8u: return visit(std::integral_constant<uint32_t, 8u>{});
        case 12u: return visit(std::integral_constant<uint32_t, 12u>{});
        case 16u: return visit(std::integral_constant<uint32_t, 16u>{});
        case 20u: return visit(std::integral_constant<uint32_t, 20u>{});
        case 24u: return visit(std::integral_constant<uint32_t, 24u>{});
        case 28u: return visit(std::integral_constant<uint32_t, 28u>{});
        default: return visit(std::integral_constant<uint32_t, 32u>{});
    }
}

uint32_t TileWarpSlots(const BlockDescentSolveParams& p) {
    uint32_t blocks = 0u;
    const cudaError_t status = VisitTileWidth(p, [&](auto width) {
        return TileResidentBlocks<decltype(width)::value>(p, &blocks);
    });
    return status == cudaSuccess ? blocks * kTileWarps : 0u;
}
