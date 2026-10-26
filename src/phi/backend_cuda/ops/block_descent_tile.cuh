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

// Coordinates a lane holds in a tile of width kN: lane l holds coordinates l, l + 32 and so on.
template <uint32_t kN>
__host__ __device__ constexpr uint32_t TileSpan() {
    return (kN + kTileLanes - 1u) / kTileLanes;
}

// A chunk of heads lists up to three Jacobian rows each on a block; half of them stage at a time.
constexpr uint32_t kTileChunkRows = 3u * kTileLanes;
constexpr uint32_t kTileStageRows = kTileChunkRows / 2u;

// Shared floats per warp: the mass and factor buffers, then past a warp of coordinates a chunk's row list
// and its staged rows.
template <uint32_t kN>
__host__ __device__ constexpr uint32_t TileWarpFloats() {
    return 2u * kN * (kN + 1u) + (TileSpan<kN>() > 1u ? kTileChunkRows * 6u + kTileStageRows * kN : 0u);
}

// Warps per block of the tile kernel: up to kTileWarps whose shared buffers fit the 48 KiB of shared
// memory a block takes without opting in.
template <uint32_t kN>
__host__ __device__ constexpr uint32_t TileWarps() {
    const uint32_t fit = 48u * 1024u / (TileWarpFloats<kN>() * uint32_t{sizeof(float)});
    return fit < kTileWarps ? fit : kTileWarps;
}

// Coordinate k of a per-lane array laid out as above, for a k the whole warp shares.
template <uint32_t kSpan>
__device__ __forceinline__ float TileCoordinate(const float (&x)[kSpan], uint32_t k) {
    float value = x[0];
#pragma unroll
    for (uint32_t r = 1u; r < kSpan; ++r) value = k / kTileLanes == r ? x[r] : value;
    return __shfl_sync(~0u, value, k % kTileLanes);
}

// A lane's share of a dot product over its coordinates.
template <uint32_t kSpan>
__device__ __forceinline__ float TileDot(const float (&a)[kSpan], const float (&b)[kSpan]) {
    float value = a[0] * b[0];
#pragma unroll
    for (uint32_t r = 1u; r < kSpan; ++r) value += a[r] * b[r];
    return value;
}

struct TileContext;

// Producer and worker warps rendezvous from distinct control paths at the same named barrier.
__device__ __forceinline__ void TileWorkSync() {
    asm volatile("barrier.cta.sync 1;" ::: "memory");
}

struct TileChunkWork {
    bool stop;
    uint32_t dimension, begin, end;
    float* factor;
    const float* stage;
    const uint2* rows;
    const float4* terms;
    const TileContext* heads;
};

// Coordinates above a warp share the Hessian products across a block, retaining each sum's row order.
template <uint32_t kN>
__device__ void AssembleTileChunk(const TileChunkWork& work) {
    const uint32_t n = work.dimension;
    for (uint32_t item = threadIdx.x; item < n * n; item += blockDim.x) {
        const uint32_t i = item / n, j = item % n;
        if (i > j) continue;
        float value = work.factor[i * (kN + 1u) + j];
        for (uint32_t t = work.begin; t < work.end; ++t) {
            const auto term = work.terms[t];
            const float* self = work.stage + (t - work.begin) * kN;
            const float* partner = work.stage + (work.rows[t].y - work.begin) * kN;
            const float weighted = self[i] * term.x + partner[i] * term.y;
            value = fmaf(weighted, self[j], value);
        }
        work.factor[i * (kN + 1u) + j] = value;
    }
}

template <uint32_t kN>
__host__ __device__ constexpr uint32_t TileThreads() {
    return TileSpan<kN>() > 1u ? kRigidThreads : TileWarps<kN>() * kTileLanes;
}

// Warps of a team: up to kTileTeamWarps whose shared buffers fit as TileWarps; a wide tile descends alone.
template <uint32_t kN>
__host__ __device__ constexpr uint32_t TileTeamWarps() {
    const uint32_t fit = 48u * 1024u / (TileWarpFloats<kN>() * uint32_t{sizeof(float)});
    return TileSpan<kN>() > 1u ? 1u : fit < kTileTeamWarps ? fit : kTileTeamWarps;
}

// Threads of the tile kernel's largest block, alone or in teams.
template <uint32_t kN>
__host__ __device__ constexpr uint32_t TileBoundThreads() {
    constexpr uint32_t team = TileTeamWarps<kN>() * kTileLanes;
    return TileThreads<kN>() > team ? TileThreads<kN>() : team;
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
    // Warps descending the environment together and this warp's place among them.
    uint32_t team;
    uint32_t member;
    // The largest block coordinate change of the running sweep, its momentum residual.
    float momentum;
    float* fields;
    float* jacobian;
    // Each block's heads in head order run from its predecessor's incident_end to its own.
    uint32_t* incident_end;
    uint32_t* incident;
    // Each block's color, the blocks in color order and each color's end in that order.
    uint32_t* color;
    uint32_t* order;
    uint32_t* color_end;
    // Lane k holds color k's end in the order; `colors` marks the colors that hold blocks.
    uint32_t color_stop;
    uint32_t colors;
    // One slot per team warp for the team's exchanges.
    uint4* team_share;
    float* mass;
    float* factor;
    uint2* chunk_rows;
    float4* chunk_terms;
    float* stage;
    TileChunkWork* chunk_work;
    TileContext* shared_context;
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
    __device__ uint32_t IncidentBegin(uint32_t block) const {
        return block == 0u ? 0u : incident_end[block - 1u];
    }
};

// Barrier over the warps descending one environment; a lone warp only converges.
__device__ __forceinline__ void TeamSync(const TileContext& c) {
    if (c.team > 1u) __syncthreads();
    else __syncwarp();
}

// The team's largest of each warp-uniform component; residual bits order as their nonnegative floats.
__device__ uint4 TeamMax(const TileContext& c, uint4 value) {
    if (c.team == 1u) return value;
    if (c.lane == 0u) c.team_share[c.member] = value;
    __syncthreads();
    for (uint32_t member = 0u; member < c.team; ++member) {
        const uint4 other = c.team_share[member];
        value = make_uint4(max(value.x, other.x), max(value.y, other.y), max(value.z, other.z),
                           max(value.w, other.w));
    }
    __syncthreads();
    return value;
}

// The first warp's warp-uniform value, across the team.
__device__ uint32_t TeamBroadcast(const TileContext& c, uint32_t value) {
    if (c.team == 1u) return value;
    if (c.member == 0u && c.lane == 0u) c.team_share[0].x = value;
    __syncthreads();
    value = c.team_share[0].x;
    __syncthreads();
    return value;
}

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
            // The chains are inputs of the solve, so they read through the read-only cache.
            if (row.a.kind == kNkSideArtic && row.a.index == index)
                value += __ldg(static_cast<const float*>(data.chain_jacobian) + at);
            if (row.b.kind == kNkSideArtic && row.b.index == index)
                value += __ldg(static_cast<const float*>(data.chain_jacobian_b) + at);
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
    term.radius = t.normal_bound;
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

// A row side's whole velocity J u, summed in coordinate order.
__device__ float TileSideSpeed(BlockDescentSolveParams p, BlockScratch s, const NkRowSide& side,
                               const float* jacobian, uint32_t slot) {
    const uint32_t count = side.kind == kNkSideRigid ? kRigidBlockDof
        : side.kind == kNkSideArtic ? s.articulation_dimensions[side.index] : 0u;
    float along = 0.0f;
    for (uint32_t c = 0u; c < count; ++c) along += TileSideVelocity(p, s, side, jacobian, slot, c);
    return along;
}

// RowInertialResponse in float: J M^-1 J^T over the row's rigid bodies and articulations, with both sides
// on one owner summed first; `coupling` is the product of the two owners' shares, `largest` the larger share.
__device__ float TileInertialResponse(DataView data, BlockDescentSolveParams p, const NkRow& row, uint32_t slot,
                                      float* coupling, float* largest) {
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
    *largest = fmaxf(shares[0], shares[1]);
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
    float coupling = 0.0f, largest = 0.0f;
    const float response = TileInertialResponse(data, p, row, slot, &coupling, &largest);
    const float coupled = coupling > 0.0f
        ? (response + sqrtf(response * response + 12.0f * coupling)) / (6.0f * coupling) : FLT_MAX;
    float rho = response > 0.0f ? fminf(static_cast<float>(kPenaltyConditionCap) / largest, coupled) : 0.0f;
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
    c.Store3(kTileBase, h, c.Load3(kTileDual, h));
    c.Store3(kTileResidual, h, {});
    c.Store3(kTileMove, h, {});
    c.Store3(kTileStep, h, {});
    c.Store3(kTileRelaxationStep, h, {});
    c.Value(kTileRelaxationWeight, h) = 0.0f;
}

template <uint32_t kN>
__device__ void ExecuteTileWork(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                const TileChunkWork& work) {
    if (work.heads != nullptr) {
        const auto& c = *work.heads;
        for (uint32_t h = threadIdx.x; h < c.heads; h += blockDim.x)
            InitializeTileHead(data, p, s, c, c.Index(kTileSlot, h), h);
    } else {
        AssembleTileChunk<kN>(work);
    }
}

// Lists each block's heads in head order, so a block solve visits only its own heads; a head whose two
// sides share a block lists once.
__device__ void IndexTileIncidence(const TileContext& c) {
    const uint32_t blocks = c.bodies + c.articulations;
    // One block lists every head that still moves it, in head order.
    if (blocks == 1u) {
        uint32_t listed = 0u;
        for (uint32_t base = 0u; base < c.heads; base += kTileLanes) {
            const uint32_t h = base + c.lane;
            const bool moves = h < c.heads &&
                (c.Index(kTileOwner, h) != kNoTileBlock || c.Index(kTileOwner + 1u, h) != kNoTileBlock);
            const uint32_t mask = __ballot_sync(~0u, moves);
            if (moves) c.incident[listed + __popc(mask & ((1u << c.lane) - 1u))] = h;
            listed += __popc(mask);
        }
        if (c.lane == 0u) c.incident_end[0] = listed;
        __syncwarp();
        return;
    }
    for (uint32_t block = c.lane; block < blocks; block += kTileLanes) c.incident_end[block] = 0u;
    __syncwarp();
    for (uint32_t h = c.lane; h < c.heads; h += kTileLanes)
        for (uint32_t side = 0u; side < 2u; ++side)
            if (c.Index(kTileOwner + side, h) != kNoTileBlock)
                atomicAdd(c.incident_end + c.Index(kTileOwner + side, h), 1u);
    __syncwarp();
    uint32_t total = 0u;
    for (uint32_t base = 0u; base < blocks; base += kTileLanes) {
        const uint32_t count = base + c.lane < blocks ? c.incident_end[base + c.lane] : 0u;
        uint32_t through = count;
        for (uint32_t offset = 1u; offset < kTileLanes; offset <<= 1u) {
            const uint32_t other = __shfl_up_sync(~0u, through, offset);
            if (c.lane >= offset) through += other;
        }
        if (base + c.lane < blocks) c.incident_end[base + c.lane] = total + through - count;
        total += __shfl_sync(~0u, through, kTileLanes - 1u);
    }
    __syncwarp();
    // Each block's cursor starts at its list and ends at its end; a chunk's lanes place heads in lane order.
    for (uint32_t chunk = 0u; chunk < c.heads; chunk += kTileLanes) {
        const uint32_t h = chunk + c.lane;
        const uint32_t first = h < c.heads ? c.Index(kTileOwner, h) : kNoTileBlock;
        const uint32_t second = h < c.heads ? c.Index(kTileOwner + 1u, h) : kNoTileBlock;
        uint32_t before_first = 0u, before_second = 0u;
        for (uint32_t other = 0u; other < kTileLanes; ++other) {
            const uint32_t a = __shfl_sync(~0u, first, other), b = __shfl_sync(~0u, second, other);
            if (other < c.lane) {
                before_first += (a == first) + (b == first);
                before_second += (a == second) + (b == second);
            }
        }
        const uint32_t at_first = first != kNoTileBlock ? c.incident_end[first] + before_first : 0u;
        const uint32_t at_second = second != kNoTileBlock ? c.incident_end[second] + before_second : 0u;
        __syncwarp();
        if (first != kNoTileBlock) {
            c.incident[at_first] = h;
            atomicAdd(c.incident_end + first, 1u);
        }
        if (second != kNoTileBlock) {
            c.incident[at_second] = h;
            atomicAdd(c.incident_end + second, 1u);
        }
        __syncwarp();
    }
}

// Greedy colors in block order, each moving block taking the least color no lower neighbor holds, so a
// color's blocks share no head; past the last free color a block joins the last, which descends in order.
__device__ void ColorTileBlocks(BlockScratch s, const TileContext& c) {
    const uint32_t blocks = c.bodies + c.articulations;
    for (uint32_t block = 0u; block < blocks; ++block) {
        uint32_t taken = 0u;
        for (uint32_t k = c.IncidentBegin(block) + c.lane; k < c.incident_end[block]; k += kTileLanes) {
            const uint32_t h = c.incident[k];
            const uint32_t first = c.Index(kTileOwner, h);
            const uint32_t other = first == block ? c.Index(kTileOwner + 1u, h) : first;
            if (other < block) taken |= 1u << c.color[other];
        }
        taken = __reduce_or_sync(~0u, taken);
        const bool moving = block < c.bodies ? s.rigid_dimensions[c.first_body + block] != 0u
                                             : s.articulation_dimensions[c.first_articulation + block - c.bodies] != 0u;
        if (c.lane == 0u)
            c.color[block] = moving ? min(static_cast<uint32_t>(__ffs(~taken)) - 1u, kTileColors - 1u) : kNoTileColor;
        __syncwarp();
    }
    c.color_end[c.lane] = 0u;
    __syncwarp();
    for (uint32_t block = c.lane; block < blocks; block += kTileLanes)
        if (c.color[block] != kNoTileColor) atomicAdd(c.color_end + c.color[block], 1u);
    __syncwarp();
    const uint32_t count = c.color_end[c.lane];
    uint32_t through = count;
    for (uint32_t offset = 1u; offset < kTileLanes; offset <<= 1u) {
        const uint32_t other = __shfl_up_sync(~0u, through, offset);
        if (c.lane >= offset) through += other;
    }
    c.color_end[c.lane] = through - count;
    __syncwarp();
    // Each color's cursor starts at its blocks and ends at its end; a chunk's lanes place blocks in lane order.
    for (uint32_t base = 0u; base < blocks; base += kTileLanes) {
        const uint32_t block = base + c.lane;
        const uint32_t color = block < blocks ? c.color[block] : kNoTileColor;
        const uint32_t peers = __match_any_sync(~0u, color);
        const uint32_t before = __popc(peers & ((1u << c.lane) - 1u));
        if (color != kNoTileColor) c.order[c.color_end[color] + before] = block;
        __syncwarp();
        if (color != kNoTileColor && before == 0u) c.color_end[color] += __popc(peers);
        __syncwarp();
    }
}

// A warp's copy of the color ends, a lane per color.
__device__ void LoadTileColors(TileContext& c) {
    c.color_stop = c.color_end[c.lane];
    const uint32_t start = __shfl_up_sync(~0u, c.color_stop, 1u);
    c.colors = __ballot_sync(~0u, c.color_stop != (c.lane == 0u ? 0u : start));
}

// Jacobians on moving blocks and residuals at free velocities, including kinematic sides. Single-coordinate
// rows store their coordinate and value; other rows store one vector of block coordinates per axis.
template <uint32_t kN>
__device__ void InitializeTileRows(DataView data, BlockDescentSolveParams p, BlockScratch s, const TileContext& c) {
    constexpr uint32_t kSpan = TileSpan<kN>();
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
            // A side off the tile's blocks keeps the empty block, whose dimension is zero.
            float value[3][kSpan]{};
#pragma unroll
            for (uint32_t axis = 0u; axis < 3u; ++axis) {
                if (axis >= axes) continue;
                const uint32_t row_slot = slot + axis * stride;
                const NkRow& row = row_data[row_slot];
#pragma unroll
                for (uint32_t r = 0u; r < kSpan; ++r) {
                    const uint32_t i = c.lane + r * kTileLanes;
                    if (i < b.dimension) value[axis][r] = b.Jacobian(data, p, row, row_slot, i);
                }
                if (incidence == 0u) {
                    float along = TileSideVelocity(p, s, row.a, chain_a, row_slot, c.lane) +
                                  TileSideVelocity(p, s, row.b, chain_b, row_slot, c.lane);
#pragma unroll
                    for (uint32_t r = 1u; r < kSpan; ++r)
                        along += TileSideVelocity(p, s, row.a, chain_a, row_slot, c.lane + r * kTileLanes) +
                                 TileSideVelocity(p, s, row.b, chain_b, row_slot, c.lane + r * kTileLanes);
                    SetAxisComponent(velocity, axis, along);
                    SetAxisComponent(rhs, axis, __fmul_rn(__fmul_rn(row.rhs, p.dt), axis == 0u ? scale : 1.0f));
                }
            }
            if (block == kNoTileBlock) continue;
            uint32_t support[kSpan], supported = 0u;
#pragma unroll
            for (uint32_t r = 0u; r < kSpan; ++r) {
                support[r] = __ballot_sync(~0u, value[0][r] != 0.0f);
                supported += __popc(support[r]);
            }
            uint32_t row = ~0u, unit = ~0u;
            float unit_value = 0.0f;
            if (axes == 1u && supported <= 1u) {
                unit = 0u;
#pragma unroll
                for (uint32_t r = 0u; r < kSpan; ++r)
                    if (support[r] != 0u) unit = r * kTileLanes + __ffs(support[r]) - 1u;
                unit_value = TileCoordinate(value[0], unit);
            } else {
                row = rows;
#pragma unroll
                for (uint32_t axis = 0u; axis < 3u; ++axis)
#pragma unroll
                    for (uint32_t r = 0u; r < kSpan; ++r)
                        if (axis < axes && c.lane + r * kTileLanes < c.width)
                            c.jacobian[size_t{row + axis} * c.width + c.lane + r * kTileLanes] = value[axis][r];
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

// InitializeTileRows past a warp of coordinates, a lane per head walking its block coordinates; a warp scan
// places the Jacobian rows in head order.
template <uint32_t kN>
__device__ void InitializeWideTileRows(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                       const TileContext& c) {
    const auto* row_data = reinterpret_cast<const NkRow*>(data.urows);
    const auto* chain_a = static_cast<const float*>(data.chain_jacobian);
    const auto* chain_b = static_cast<const float*>(data.chain_jacobian_b);
    uint32_t rows = 0u;
    for (uint32_t chunk = 0u; chunk < c.heads; chunk += kTileLanes) {
        const uint32_t h = chunk + c.lane;
        const bool live = h < c.heads;
        const uint32_t slot = live ? c.Index(kTileSlot, h) : 0u;
        const uint32_t stride = live ? c.Index(kTileStride, h) : 0u;
        const uint32_t axes = live ? c.Index(kTileAxes, h) : 0u;
        const float scale = live ? RowStateScale(data, p, slot) : 0.0f;
        Vec3 velocity{}, rhs{};
        for (uint32_t axis = 0u; axis < axes; ++axis) {
            const uint32_t row_slot = slot + axis * stride;
            const NkRow row = row_data[row_slot];
            SetAxisComponent(velocity, axis, TileSideSpeed(p, s, row.a, chain_a, row_slot) +
                                             TileSideSpeed(p, s, row.b, chain_b, row_slot));
            SetAxisComponent(rhs, axis, __fmul_rn(__fmul_rn(row.rhs, p.dt), axis == 0u ? scale : 1.0f));
        }
        if (live)
            c.Store3(kTileResidual, h, {__fsub_rn(rhs.x, velocity.x), __fsub_rn(rhs.y, velocity.y),
                                        __fsub_rn(rhs.z, velocity.z)});
        TileBlock blocks[2]{};
        uint32_t staged[2] = {0u, 0u};
#pragma unroll
        for (uint32_t incidence = 0u; incidence < 2u; ++incidence) {
            const uint32_t block = live ? c.Index(kTileOwner + incidence, h) : kNoTileBlock;
            if (block == kNoTileBlock) continue;
            blocks[incidence] = LoadTileBlock(data, p, s, c, block);
            const TileBlock& b = blocks[incidence];
            const NkRow head = row_data[slot];
            uint32_t supported = 0u, unit = 0u;
            float unit_value = 0.0f;
            for (uint32_t i = 0u; i < b.dimension; ++i) {
                const float value = b.Jacobian(data, p, head, slot, i);
                if (value == 0.0f) continue;
                ++supported;
                unit = i;
                unit_value = value;
            }
            const bool single = axes == 1u && supported <= 1u;
            staged[incidence] = single ? 0u : axes;
            c.Index(kTileRow + incidence, h) = ~0u;
            c.Index(kTileUnit + incidence, h) = single ? unit : ~0u;
            c.Value(kTileUnitValue + incidence, h) = single ? unit_value : 0.0f;
        }
        const uint32_t own = staged[0] + staged[1];
        uint32_t through = own;
        for (uint32_t offset = 1u; offset < kTileLanes; offset <<= 1u) {
            const uint32_t other = __shfl_up_sync(~0u, through, offset);
            if (c.lane >= offset) through += other;
        }
        uint32_t row = rows + through - own;
        rows += __shfl_sync(~0u, through, kTileLanes - 1u);
#pragma unroll
        for (uint32_t incidence = 0u; incidence < 2u; ++incidence) {
            if (staged[incidence] == 0u) continue;
            const TileBlock& b = blocks[incidence];
            c.Index(kTileRow + incidence, h) = row;
            for (uint32_t axis = 0u; axis < axes; ++axis) {
                const uint32_t row_slot = slot + axis * stride;
                const NkRow axis_row = row_data[row_slot];
                float* const out = c.jacobian + size_t{row + axis} * c.width;
                for (uint32_t i = 0u; i < c.width; ++i)
                    out[i] = i < b.dimension ? b.Jacobian(data, p, axis_row, row_slot, i) : 0.0f;
            }
            row += axes;
        }
    }
    __syncwarp();
}

// Free velocities, block coordinates, the environment's active head rows in slot order with their
// Jacobians, and their residuals at the free velocities.
template <uint32_t kN>
__device__ void InitializeTile(ArticulationDeviceState state, DataView data, BlockDescentSolveParams p,
                               BlockScratch s, TileContext& c) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    if (c.lane < kSolverCountWords)
        data.solver_color_counts[c.env * kSolverCountWords + c.lane] = c.lane == 0u ? p.vertex_blocks.colors : 0u;
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
        uint32_t compacted = 0u;
        for (uint32_t first = 0u; first < p.max_dof; first += kTileLanes) {
            const uint32_t dof = first + c.lane;
            uint32_t link = 0u, component = 0u;
            const bool located = dof < p.max_dof &&
                ArticulationDofLocation(state, articulation, dof, &link, &component);
            const float value = located
                ? (component != ~0u ? state.link_velocity[link].v[component] : state.qdot[link]) : 0.0f;
            const bool root = located && (p.mimic_couplings == 0u || data.mimic_root_dof[base + dof] == dof);
            const uint32_t roots = __ballot_sync(~0u, root);
            if (root) {
                const uint32_t compact = compacted + __popc(roots & ((1u << c.lane) - 1u));
                s.articulation_roots[base + compact] = dof;
                s.articulation_free[base + compact] = value;
                s.articulation_velocity[base + compact] = value;
            }
            compacted += __popc(roots);
        }
        if (c.lane == 0u) s.articulation_dimensions[articulation] = compacted;
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
            if (head) c.Index(kTileSlot, heads + __popc(mask & ((1u << c.lane) - 1u))) = first + local;
            heads += __popc(mask);
        }
    }
    c.heads = heads;
    __syncwarp();
    if constexpr (TileSpan<kN>() > 1u) {
        if (c.lane == 0u) {
            *c.shared_context = c;
            *c.chunk_work = {false, 0u, 0u, 0u, nullptr, nullptr, nullptr, nullptr, c.shared_context};
        }
        TileWorkSync();
        ExecuteTileWork<kN>(data, p, s, *c.chunk_work);
        TileWorkSync();
    } else {
        for (uint32_t h = c.lane; h < heads; h += kTileLanes)
            InitializeTileHead(data, p, s, c, c.Index(kTileSlot, h), h);
        __syncwarp();
    }
    IndexTileIncidence(c);
    ColorTileBlocks(s, c);
    if (p.measure_contact_residual != 0u) {
        for (uint32_t base = 0u; base < heads; base += kTileLanes) {
            uint32_t at = 0u;
            if (c.lane == 0u) at = atomicAdd(s.active_count, min(kTileLanes, heads - base));
            at = __shfl_sync(~0u, at, 0u);
            if (base + c.lane < heads) s.active_rows[at + c.lane] = c.Index(kTileSlot, base + c.lane);
        }
    }
    if constexpr (TileSpan<kN>() == 1u) InitializeTileRows<kN>(data, p, s, c);
    else InitializeWideTileRows<kN>(data, p, s, c);
}

// Hold the current normal multiplier as the friction radius through the block solve, as LoadLocalTerm.
__device__ void StartTileBlock(const TileContext& c, uint32_t block) {
    for (uint32_t k = c.IncidentBegin(block) + c.lane; k < c.incident_end[block]; k += kTileLanes) {
        const uint32_t h = c.incident[k];
        c.Value(kTileNormalBound, h) = c.Value(kTileDual, h);
        c.Store3(kTileMove, h, {});
    }
    __syncwarp();
}

// The block's moves enter the residuals of the heads it touches.
__device__ void EndTileBlock(const TileContext& c, uint32_t block) {
    for (uint32_t k = c.IncidentBegin(block) + c.lane; k < c.incident_end[block]; k += kTileLanes) {
        const uint32_t h = c.incident[k];
        const Vec3 residual = c.Load3(kTileResidual, h);
        const Vec3 move = c.Load3(kTileMove, h);
        c.Store3(kTileResidual, h, {__fsub_rn(residual.x, move.x), __fsub_rn(residual.y, move.y),
                                    __fsub_rn(residual.z, move.z)});
    }
    __syncwarp();
}

// Exact zero responses need no matrix work; every head is reevaluated at each Newton iterate.
__device__ bool TileResponseContributes(Vec3 f, SymmetricMat3 curvature, Vec3 noise) {
    return f.x != 0.0f || f.y != 0.0f || f.z != 0.0f || curvature.xx != 0.0f ||
           curvature.yy != 0.0f || curvature.zz != 0.0f || curvature.yz != 0.0f ||
           noise.x != 0.0f || noise.y != 0.0f || noise.z != 0.0f;
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
    const uint32_t end = c.incident_end[b.local];
    for (uint32_t chunk = c.IncidentBegin(b.local); chunk < end; chunk += kTileLanes) {
        const bool listed = chunk + c.lane < end;
        const uint32_t own = listed ? c.incident[chunk + c.lane] : 0u;
        const uint32_t incidence = listed ? c.Incidence(own, b.local) : kNoTileIncidence;
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
        const bool contributes = incidence != kNoTileIncidence && TileResponseContributes(f, curvature, noise);
        // A row on one coordinate adds to that coordinate's diagonal and force only.
        for (uint32_t pending = __ballot_sync(~0u, contributes && unit != ~0u); pending != 0u;
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
        for (uint32_t pending = __ballot_sync(~0u, contributes && unit == ~0u); pending != 0u;
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

// AssembleTileBlock past a warp of coordinates: H's upper triangle over the block's own coordinates goes to
// the warp's factor buffer, lane i holding rows i, i + 32, ...; `pattern` marks each row's nonzero columns.
template <uint32_t kN, uint32_t kSpan = TileSpan<kN>()>
__device__ bool AssembleWideTileBlock(const TileContext& c, const TileBlock& b, const float (&u)[kSpan],
                                      const float (&free)[kSpan], float (&force)[kSpan], float (&inertia)[kSpan],
                                      uint64_t (&pattern)[kSpan], bool* converged) {
    static_assert(kN <= 64u, "a row pattern is one 64-bit mask");
    constexpr uint32_t kS = kN + 1u;
    const uint32_t n = b.dimension;
    float* const h = c.factor;
    bool coordinate[kSpan];
    float difference[kSpan], difference_error[kSpan], scale[kSpan], diagonal[kSpan];
    TileSum inertial[kSpan], g[kSpan];
#pragma unroll
    for (uint32_t r = 0u; r < kSpan; ++r) {
        coordinate[r] = c.lane + r * kTileLanes < n;
        difference[r] = u[r] - free[r];
        const float part = difference[r] - u[r];
        difference_error[r] = (u[r] - (difference[r] - part)) + (-free[r] - part);
        scale[r] = 0.0f;
        diagonal[r] = 0.0f;
    }
    for (uint32_t k = 0u; k < n; ++k) {
        const float high = TileCoordinate(difference, k);
        const float low = TileCoordinate(difference_error, k);
#pragma unroll
        for (uint32_t r = 0u; r < kSpan; ++r) {
            if (!coordinate[r]) continue;
            const uint32_t i = c.lane + r * kTileLanes;
            const float mass = c.mass[i * kS + k];
            h[i * kS + k] = mass;
            TileAddProduct(inertial[r], mass, high);
            TileAddProduct(inertial[r], mass, low);
            scale[r] += fabsf(mass) * fabsf(high);
        }
    }
#pragma unroll
    for (uint32_t r = 0u; r < kSpan; ++r) g[r] = {-inertial[r].value, -inertial[r].error};
    const uint32_t end = c.incident_end[b.local];
    for (uint32_t chunk = c.IncidentBegin(b.local); chunk < end; chunk += kTileLanes) {
        const bool listed = chunk + c.lane < end;
        const uint32_t own = listed ? c.incident[chunk + c.lane] : 0u;
        const uint32_t incidence = listed ? c.Incidence(own, b.local) : kNoTileIncidence;
        Vec3 f{}, noise{};
        SymmetricMat3 curvature{};
        uint32_t unit = ~0u, first = 0u, axes = 1u;
        float unit_value = 0.0f;
        if (incidence != kNoTileIncidence) {
            const TileHead t = LoadTileHead(c, own);
            TileResponse(t, t.move, &f, &curvature);
            noise = {fabsf(f.x) + curvature.xx * (fabsf(t.residual.x) + fabsf(t.move.x)),
                     fabsf(f.y) + curvature.yy * (fabsf(t.residual.y) + fabsf(t.move.y)),
                     fabsf(f.z) + curvature.zz * (fabsf(t.residual.z) + fabsf(t.move.z))};
            unit = c.Index(kTileUnit + incidence, own);
            first = c.Index(kTileRow + incidence, own);
            unit_value = c.Value(kTileUnitValue + incidence, own);
            axes = t.axes;
        }
        const bool contributes = incidence != kNoTileIncidence && TileResponseContributes(f, curvature, noise);
        for (uint32_t pending = __ballot_sync(~0u, contributes && unit != ~0u); pending != 0u;
             pending &= pending - 1u) {
            const uint32_t source = __ffs(pending) - 1u;
            const uint32_t target = __shfl_sync(~0u, unit, source);
            const float v = __shfl_sync(~0u, unit_value, source);
            const float fx = __shfl_sync(~0u, f.x, source);
            const float cxx = __shfl_sync(~0u, curvature.xx, source);
            const float nx = __shfl_sync(~0u, noise.x, source);
#pragma unroll
            for (uint32_t r = 0u; r < kSpan; ++r) {
                if (c.lane + r * kTileLanes != target) continue;
                diagonal[r] += v * cxx * v;
                TileAddProduct(g[r], v, fx);
                scale[r] += fabsf(v) * nx;
            }
        }
        // The chunk's Jacobian rows on the block in head order, each with the list position of the other
        // tangent axis's row, and its axis's curvature, coupling to that row, force and noise.
        const uint32_t count = contributes && unit == ~0u ? axes : 0u;
        uint32_t through = count;
        for (uint32_t offset = 1u; offset < kTileLanes; offset <<= 1u) {
            const uint32_t other = __shfl_up_sync(~0u, through, offset);
            if (c.lane >= offset) through += other;
        }
        const uint32_t rows = __shfl_sync(~0u, through, kTileLanes - 1u);
        const uint32_t half = __shfl_sync(~0u, through, kTileLanes / 2u - 1u);
        __syncwarp();
        for (uint32_t axis = 0u; axis < count; ++axis) {
            const uint32_t at = through - count + axis;
            c.chunk_rows[at] = {first + axis, axis == 0u ? at : axis == 1u ? at + 1u : at - 1u};
            c.chunk_terms[at] = {axis == 0u ? curvature.xx : axis == 1u ? curvature.yy : curvature.zz,
                                 axis == 0u ? 0.0f : curvature.yz, AxisComponent(f, axis), AxisComponent(noise, axis)};
        }
        // Half of the chunk's heads stage together, keeping each tangent pair in one shared batch.
        for (uint32_t part = 0u; part < 2u; ++part) {
            const uint32_t begin = part == 0u ? 0u : half, end = part == 0u ? half : rows;
            if (begin == end) continue;
            __syncwarp();
            constexpr uint32_t kBatch = 16u;
            for (uint32_t first_row = begin; first_row < end; first_row += kBatch) {
                float staged[kBatch][kSpan];
#pragma unroll
                for (uint32_t k = 0u; k < kBatch; ++k) {
                    const float* const from = first_row + k < end
                        ? c.jacobian + size_t{c.chunk_rows[first_row + k].x} * c.width : nullptr;
#pragma unroll
                    for (uint32_t r = 0u; r < kSpan; ++r)
                        staged[k][r] = first_row + k < end && c.lane + r * kTileLanes < c.width
                            ? from[c.lane + r * kTileLanes] : 0.0f;
                }
#pragma unroll
                for (uint32_t k = 0u; k < kBatch; ++k)
#pragma unroll
                    for (uint32_t r = 0u; r < kSpan; ++r)
                        if (first_row + k < end && c.lane + r * kTileLanes < kN)
                            c.stage[(first_row + k - begin) * kN + c.lane + r * kTileLanes] = staged[k][r];
            }
            if (c.lane == 0u) *c.chunk_work = {false, n, begin, end, c.factor, c.stage, c.chunk_rows, c.chunk_terms};
            TileWorkSync();
            AssembleTileChunk<kN>(*c.chunk_work);
            TileWorkSync();
            for (uint32_t t = begin; t < end; ++t) {
                const float4 term = c.chunk_terms[t];
                const float* const self = c.stage + (t - begin) * kN;
#pragma unroll
                for (uint32_t r = 0u; r < kSpan; ++r) {
                    const uint32_t i = c.lane + r * kTileLanes;
                    const float j = i < kN ? self[i] : 0.0f;
                    TileAddProduct(g[r], j, term.z);
                    scale[r] += fabsf(j) * term.w;
                }
            }
        }
    }
    bool finite = true, settled = true;
#pragma unroll
    for (uint32_t r = 0u; r < kSpan; ++r) {
        const uint32_t i = c.lane + r * kTileLanes;
        if (coordinate[r]) h[i * kS + i] += diagonal[r];
        force[r] = coordinate[r] ? g[r].value + g[r].error : 0.0f;
        inertia[r] = coordinate[r] ? inertial[r].value + inertial[r].error : 0.0f;
        finite &= isfinite(force[r]);
        pattern[r] = 0u;
    }
    // A float velocity is off its optimum by up to a rounding, which moves g by H times it.
    for (uint32_t k = 0u; k < n; ++k) {
        const float along = fabsf(TileCoordinate(u, k));
#pragma unroll
        for (uint32_t r = 0u; r < kSpan; ++r) {
            if (!coordinate[r]) continue;
            const uint32_t i = c.lane + r * kTileLanes;
            const float entry = h[min(i, k) * kS + max(i, k)];
            finite &= isfinite(entry);
            scale[r] += fabsf(entry) * along;
            if (entry != 0.0f) pattern[r] |= uint64_t{1} << k;
        }
    }
#pragma unroll
    for (uint32_t r = 0u; r < kSpan; ++r)
        settled &= !coordinate[r] || fabsf(force[r]) <= kDenseRoundings * FLT_EPSILON * scale[r];
    *converged = __all_sync(~0u, settled);
    return __all_sync(~0u, finite);
}

// FactorTileBlock past a warp of coordinates as H = U U^T in the factor buffer's upper triangle. Columns go
// from the last coordinate down, so a tree's leaves go first and its pattern takes no fill; only rows on a
// column's pattern take it, and entries off the pattern stay exact zeros.
template <uint32_t kN, uint32_t kSpan = TileSpan<kN>()>
__device__ bool FactorWideTileBlock(const TileContext& c, float (&inverse)[kSpan], uint64_t (&pattern)[kSpan],
                                    uint32_t n) {
    constexpr uint32_t kS = kN + 1u;
    float* const h = c.factor;
    bool valid = true;
#pragma unroll
    for (uint32_t r = 0u; r < kSpan; ++r) inverse[r] = 0.0f;
    __syncwarp();
    for (uint32_t k = n; k-- > 0u;) {
        uint64_t column = 0u;
#pragma unroll
        for (uint32_t r = 0u; r < kSpan; ++r) {
            const bool above = c.lane + r * kTileLanes < k && (pattern[r] >> k & 1u) != 0u;
            column |= uint64_t{__ballot_sync(~0u, above)} << (r * kTileLanes);
        }
        const float* const pivot_row = h + k * kS;
        float value[kSpan];
        bool member[kSpan];
#pragma unroll
        for (uint32_t r = 0u; r < kSpan; ++r) {
            const uint32_t i = c.lane + r * kTileLanes;
            member[r] = i == k || (i < k && (column >> i & 1u) != 0u);
            value[r] = 0.0f;
            if (!member[r]) continue;
            const float* const own = h + i * kS;
            float v = own[k];
#pragma unroll 4
            for (uint32_t j = k + 1u; j < n; ++j) v -= own[j] * pivot_row[j];
            value[r] = v;
        }
        const float pivot = TileCoordinate(value, k);
        valid &= pivot > 0.0f && pivot <= FLT_MAX;
        const float root = sqrtf(valid ? pivot : 1.0f);
        const float scale = 1.0f / root;
        // Eliminating column k joins the rows above it on the column.
#pragma unroll
        for (uint32_t r = 0u; r < kSpan; ++r) {
            const uint32_t i = c.lane + r * kTileLanes;
            if (i == k) inverse[r] = scale;
            if (member[r]) h[i * kS + k] = i == k ? root : value[r] * scale;
            if (member[r] && i != k) pattern[r] |= column;
        }
        __syncwarp();
    }
    return valid;
}

// SolveTileBlock past a warp of coordinates on the factor buffer's U: U w = g up its columns, then
// U^T d = w down its rows.
template <uint32_t kN, uint32_t kSpan = TileSpan<kN>()>
__device__ void SolveWideTileBlock(const TileContext& c, const float (&inverse)[kSpan], const float (&g)[kSpan],
                                   uint32_t n, float (&d)[kSpan]) {
    constexpr uint32_t kS = kN + 1u;
    const float* const h = c.factor;
    float r[kSpan], scaled[kSpan];
#pragma unroll
    for (uint32_t s = 0u; s < kSpan; ++s) r[s] = g[s];
    for (uint32_t j = n; j-- > 0u;) {
#pragma unroll
        for (uint32_t s = 0u; s < kSpan; ++s) scaled[s] = r[s] * inverse[s];
        const float y = TileCoordinate(scaled, j);
#pragma unroll
        for (uint32_t s = 0u; s < kSpan; ++s) {
            const uint32_t i = c.lane + s * kTileLanes;
            if (i == j) r[s] = y;
            else if (i < j) r[s] = fmaf(-h[i * kS + j], y, r[s]);
        }
    }
    for (uint32_t j = 0u; j < n; ++j) {
#pragma unroll
        for (uint32_t s = 0u; s < kSpan; ++s) scaled[s] = r[s] * inverse[s];
        const float x = TileCoordinate(scaled, j);
#pragma unroll
        for (uint32_t s = 0u; s < kSpan; ++s) {
            const uint32_t i = c.lane + s * kTileLanes;
            if (i == j) r[s] = x;
            else if (i > j && i < n) r[s] = fmaf(-h[j * kS + i], x, r[s]);
        }
    }
    __syncwarp();
#pragma unroll
    for (uint32_t s = 0u; s < kSpan; ++s) d[s] = r[s];
}

// The Newton direction d = H^-1 g at u and the forces g: H in registers within a warp of coordinates, in the
// factor buffer past it.
template <uint32_t kN, uint32_t kSpan = TileSpan<kN>()>
__device__ nk::BlockSolveFailure TileNewtonDirection(const TileContext& c, const TileBlock& b, const float (&u)[kSpan],
                                                     const float (&free)[kSpan], float (&force)[kSpan],
                                                     float (&inertia)[kSpan], float (&d)[kSpan], bool* converged) {
    if constexpr (kSpan == 1u) {
        float h[kN];
        float inverse = 0.0f;
        if (!AssembleTileBlock<kN>(c, b, u[0], free[0], h, &force[0], &inertia[0], converged))
            return nk::BlockSolveFailure::InvalidEquation;
        if (*converged) return nk::BlockSolveFailure::None;
        if (!FactorTileBlock<kN>(h, &inverse, c.lane, b.dimension)) return nk::BlockSolveFailure::Factorization;
        d[0] = SolveTileBlock<kN>(h, inverse, force[0], c, b.dimension);
    } else {
        float inverse[kSpan];
        uint64_t pattern[kSpan];
        if (!AssembleWideTileBlock<kN>(c, b, u, free, force, inertia, pattern, converged))
            return nk::BlockSolveFailure::InvalidEquation;
        if (*converged) return nk::BlockSolveFailure::None;
        if (!FactorWideTileBlock<kN>(c, inverse, pattern, b.dimension)) return nk::BlockSolveFailure::Factorization;
        SolveWideTileBlock<kN>(c, inverse, force, b.dimension, d);
    }
    return nk::BlockSolveFailure::None;
}

// w = J d of every head on the block: a row on one coordinate scales that coordinate's step, other rows
// sum over the warp.
template <uint32_t kSpan>
__device__ void TileRowSteps(const TileContext& c, uint32_t block, const float (&d)[kSpan]) {
    const uint32_t end = c.incident_end[block];
    for (uint32_t chunk = c.IncidentBegin(block); chunk < end; chunk += kTileLanes) {
        const bool listed = chunk + c.lane < end;
        const uint32_t own = listed ? c.incident[chunk + c.lane] : 0u;
        const uint32_t incidence = listed ? c.Incidence(own, block) : kNoTileIncidence;
        const bool touches = incidence != kNoTileIncidence;
        const uint32_t unit = touches ? c.Index(kTileUnit + incidence, own) : ~0u;
        const uint32_t first = touches ? c.Index(kTileRow + incidence, own) : 0u;
        const uint32_t axes = touches ? c.Index(kTileAxes, own) : 1u;
        // Each lane reads its own head's coordinate, so every part of d crosses the warp.
        const uint32_t coordinate = unit != ~0u ? unit : 0u;
        float along = __shfl_sync(~0u, d[0], coordinate % kTileLanes);
#pragma unroll
        for (uint32_t r = 1u; r < kSpan; ++r) {
            const float part = __shfl_sync(~0u, d[r], coordinate % kTileLanes);
            along = coordinate / kTileLanes == r ? part : along;
        }
        Vec3 step{};
        if (unit != ~0u) step.x = c.Value(kTileUnitValue + incidence, own) * along;
        for (uint32_t pending = __ballot_sync(~0u, touches && unit == ~0u); pending != 0u; pending &= pending - 1u) {
            const uint32_t source = __ffs(pending) - 1u;
            const float* jacobian = c.jacobian + size_t{__shfl_sync(~0u, first, source)} * c.width;
            float j0[kSpan];
#pragma unroll
            for (uint32_t r = 0u; r < kSpan; ++r)
                j0[r] = c.lane + r * kTileLanes < c.width ? jacobian[c.lane + r * kTileLanes] : 0.0f;
            Vec3 w{};
            if (__shfl_sync(~0u, axes, source) == 1u) {
                w.x = TileWarpSum(TileDot(j0, d));
            } else {
                float j1[kSpan], j2[kSpan];
#pragma unroll
                for (uint32_t r = 0u; r < kSpan; ++r) {
                    const uint32_t i = c.lane + r * kTileLanes;
                    j1[r] = i < c.width ? jacobian[c.width + i] : 0.0f;
                    j2[r] = i < c.width ? jacobian[2u * c.width + i] : 0.0f;
                }
                w = TileWarpSum3({TileDot(j0, d), TileDot(j1, d), TileDot(j2, d)});
            }
            if (c.lane == source) step = w;
        }
        if (touches) c.Store3(kTileStep, own, step);
    }
    __syncwarp();
}

// TileRowSteps past a warp of coordinates: the step goes through the factor buffer, and a lane takes a head.
template <uint32_t kN, uint32_t kSpan = TileSpan<kN>()>
__device__ void WideTileRowSteps(const TileContext& c, uint32_t block, uint32_t n, const float (&d)[kSpan]) {
    float* const staged = c.factor;
#pragma unroll
    for (uint32_t r = 0u; r < kSpan; ++r)
        if (c.lane + r * kTileLanes < kN) staged[c.lane + r * kTileLanes] = d[r];
    __syncwarp();
    for (uint32_t k = c.IncidentBegin(block) + c.lane; k < c.incident_end[block]; k += kTileLanes) {
        const uint32_t h = c.incident[k];
        const uint32_t incidence = c.Incidence(h, block);
        const uint32_t unit = c.Index(kTileUnit + incidence, h);
        Vec3 step{};
        if (unit != ~0u) {
            step.x = c.Value(kTileUnitValue + incidence, h) * staged[unit];
        } else {
            const float* const jacobian = c.jacobian + size_t{c.Index(kTileRow + incidence, h)} * c.width;
            const bool three = c.Index(kTileAxes, h) == 3u;
            for (uint32_t i = 0u; i < n; ++i) {
                step.x = fmaf(jacobian[i], staged[i], step.x);
                if (three) {
                    step.y = fmaf(jacobian[c.width + i], staged[i], step.y);
                    step.z = fmaf(jacobian[2u * c.width + i], staged[i], step.z);
                }
            }
        }
        c.Store3(kTileStep, h, step);
    }
    __syncwarp();
}

// Slope and curvature of the local objective at `alpha` along the step, as DenseLineDerivatives; a lane's
// first listed head is `resident`.
__device__ void TileLineDerivatives(const TileContext& c, uint32_t block, float alpha, float inertial_slope,
                                    float inertial_curvature, const TileHead& resident,
                                    float* slope, float* curvature, float* magnitude) {
    TileSum row_slope{};
    float row_curvature = 0.0f, row_magnitude = 0.0f;
    const uint32_t begin = c.IncidentBegin(block);
    for (uint32_t k = begin + c.lane; k < c.incident_end[block]; k += kTileLanes) {
        const TileHead t = k < begin + kTileLanes ? resident : LoadTileHead(c, c.incident[k]);
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
    const uint32_t first = c.IncidentBegin(block) + c.lane;
    TileHead resident{};
    if (first < c.incident_end[block]) resident = LoadTileHead(c, c.incident[first]);
    float slope = 0.0f, curvature = 0.0f, magnitude = 0.0f;
    TileLineDerivatives(c, block, 0.0f, inertial_slope, inertial_curvature, resident,
                        &slope, &curvature, &magnitude);
    const float initial = slope;
    if (!(initial < 0.0f)) return 0.0f;
    float low = 0.0f, high = 0.0f, alpha = 1.0f;
    bool bounded = false;
    for (uint32_t step = 0u; step < kDenseLineSearchSteps; ++step) {
        TileLineDerivatives(c, block, alpha, inertial_slope, inertial_curvature, resident,
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
    constexpr uint32_t kSpan = TileSpan<kN>();
    const uint32_t n = b.dimension;
    bool coordinate[kSpan];
#pragma unroll
    for (uint32_t r = 0u; r < kSpan; ++r) coordinate[r] = c.lane + r * kTileLanes < n;
    // Coordinates past the block's carry unit mass and no rows, so their step stays zero.
    if (b.local != c.staged) {
        for (uint32_t i = 0u; i < kN; ++i)
#pragma unroll
            for (uint32_t r = 0u; r < kSpan; ++r) {
                const uint32_t j = c.lane + r * kTileLanes;
                if (j < kN) c.mass[i * kS + j] = i < n && coordinate[r] ? b.Mass(i, j) : i == j ? 1.0f : 0.0f;
            }
        c.staged = b.local;
    }
    StartTileBlock(c, b.local);
    float start[kSpan], free[kSpan], u[kSpan];
#pragma unroll
    for (uint32_t r = 0u; r < kSpan; ++r) {
        start[r] = coordinate[r] ? b.velocity[c.lane + r * kTileLanes] : 0.0f;
        free[r] = coordinate[r] && b.free != nullptr ? b.free[c.lane + r * kTileLanes] : 0.0f;
        u[r] = start[r];
    }
    nk::BlockSolveFailure failure = nk::BlockSolveFailure::None;
    for (uint32_t step = 0u; step < kDenseNewtonSteps; ++step) {
        float force[kSpan], inertia[kSpan], d[kSpan], mass_step[kSpan];
        bool converged = false;
        failure = TileNewtonDirection<kN>(c, b, u, free, force, inertia, d, &converged);
        if (failure != nk::BlockSolveFailure::None || converged) break;
#pragma unroll
        for (uint32_t r = 0u; r < kSpan; ++r) mass_step[r] = 0.0f;
        // Past a warp, coordinates beyond the block's have zero step and mass.
#pragma unroll
        for (uint32_t k = 0u; k < (kSpan == 1u ? kN : n); ++k) {
            const float along = TileCoordinate(d, k);
#pragma unroll
            for (uint32_t r = 0u; r < kSpan; ++r) {
                const uint32_t i = c.lane + r * kTileLanes;
                mass_step[r] += (i < kN ? c.mass[i * kS + k] : 0.0f) * along;
            }
        }
        TileSum decrement{}, slope{}, curvature{};
#pragma unroll
        for (uint32_t r = 0u; r < kSpan; ++r) {
            if (!coordinate[r]) continue;
            TileAddProduct(decrement, force[r], d[r]);
            TileAddProduct(slope, inertia[r], d[r]);
            TileAddProduct(curvature, d[r], mass_step[r]);
        }
        const float total_decrement = TileWarpTotal(decrement);
        const float inertial_slope = TileWarpTotal(slope);
        const float inertial_curvature = TileWarpTotal(curvature);
        if (!isfinite(total_decrement) || !isfinite(inertial_slope) || !isfinite(inertial_curvature)) {
            failure = nk::BlockSolveFailure::InvalidDirection;
            break;
        }
        if (!(total_decrement > 0.0f)) break;
        if constexpr (kSpan == 1u) TileRowSteps(c, b.local, d);
        else WideTileRowSteps<kN>(c, b.local, n, d);
        const float alpha = SearchTileLine(c, b.local, inertial_slope, inertial_curvature);
        float after[kSpan];
        bool finite = true, rounded = true;
#pragma unroll
        for (uint32_t r = 0u; r < kSpan; ++r) {
            after[r] = fmaf(alpha, d[r], u[r]);
            finite &= !coordinate[r] || isfinite(after[r]);
            rounded &= !coordinate[r] || fabsf(after[r] - u[r]) <=
                kDenseRoundings * FLT_EPSILON * fmaxf(fabsf(after[r]), fabsf(u[r]));
        }
        rounded = __all_sync(~0u, rounded);
        if (!__all_sync(~0u, finite)) {
            failure = nk::BlockSolveFailure::InvalidCandidate;
            break;
        }
        if (!(alpha > 0.0f)) break;
#pragma unroll
        for (uint32_t r = 0u; r < kSpan; ++r) u[r] = after[r];
        for (uint32_t k = c.IncidentBegin(b.local) + c.lane; k < c.incident_end[b.local]; k += kTileLanes) {
            const uint32_t head = c.incident[k];
            const Vec3 move = c.Load3(kTileMove, head);
            const Vec3 row_step = c.Load3(kTileStep, head);
            c.Store3(kTileMove, head, {fmaf(alpha, row_step.x, move.x), fmaf(alpha, row_step.y, move.y),
                                       fmaf(alpha, row_step.z, move.z)});
        }
        __syncwarp();
        if (rounded) break;
    }
    EndTileBlock(c, b.local);
    if (failure != nk::BlockSolveFailure::None && c.lane == 0u) RecordTileFailure(data, p, s, c, b.owner, failure);
    float moved = failure == nk::BlockSolveFailure::None ? 0.0f : FLT_MAX;
#pragma unroll
    for (uint32_t r = 0u; r < kSpan; ++r)
        if (coordinate[r]) moved = fmaxf(moved, __uint_as_float(ResidualBits(u[r] - start[r])));
    c.momentum = fmaxf(c.momentum, TileWarpMax(moved));
#pragma unroll
    for (uint32_t r = 0u; r < kSpan; ++r)
        if (coordinate[r]) b.velocity[c.lane + r * kTileLanes] = u[r];
    __syncwarp();
}

// Dual update as DualUpdateKernel at the velocities every block reached in this iteration; position pass
// impulses stay in the tile. Returns the contact normal, contact tangent and other rows' residuals, then the
// largest multiplier step. A team's warps take alternate warps of heads.
__device__ float4 UpdateTileDuals(DataView data, const TileContext& c, float beta, bool velocity = true) {
    uint32_t steps[kResidualClasses] = {0u, 0u, 0u, 0u};
    for (uint32_t h = c.member * kTileLanes + c.lane; h < c.heads; h += c.team * kTileLanes) {
        const TileHead t = LoadTileHead(c, h);
        const uint32_t slot = c.Index(kTileSlot, h);
        const uint32_t stride = c.Index(kTileStride, h);
        const bool contact = t.axes == 3u;
        const float normal = nk::augmented::EvaluateScalarImpulse(t.dual.x, t.rho, t.residual.x, t.compliance,
                                                                  t.lower, t.upper, t.response).impulse;
        nk::augmented::ContactRelaxation relaxation{c.Load3(kTileRelaxationStep, h), c.Value(kTileRelaxationWeight, h)};
        const Vec3 dual = ProjectRowDual(t.dual, normal, t.rho, t.residual, contact, t.mu_first, t.mu_second,
                                         &relaxation);
        const float2 step = MultiplierSteps(t.dual, dual, t.response.residual_scale, t.rho,
                                            contact ? t.mu_first : 0.0f, contact ? t.mu_second : 0.0f);
        steps[kResidualMomentum] = max(steps[kResidualMomentum], ResidualBits(fmaxf(step.x, step.y)));
        if (!contact) steps[kResidualJoint] = max(steps[kResidualJoint], ResidualBits(step.x));
        if (contact) {
            c.Store3(kTileRelaxationStep, h, relaxation.step);
            c.Value(kTileRelaxationWeight, h) = relaxation.weight;
            data.lambda[slot + stride] = dual.y;
            data.lambda[slot + 2u * stride] = dual.z;
            // Natural velocities without the contact response: a loaded normal must not move, a free one must
            // not approach; friction must not drive its own axis, and inside the cone the slip is the tangent step.
            const Vec3 v{-t.residual.x + t.compliance * dual.x, -t.residual.y, -t.residual.z};
            steps[kResidualNormal] = max(steps[kResidualNormal],
                                         ResidualBits(dual.x > 0.0f ? fabsf(v.x) : fmaxf(-v.x, 0.0f)));
            steps[kResidualTangent] = max(steps[kResidualTangent],
                ResidualBits(fmaxf(step.y, fmaxf(Along(dual.y, v.y), Along(dual.z, v.z)))));
        }
        if (velocity) data.lambda[slot] = dual.x;
        // The blocks descend against the extrapolated multipliers; data.lambda keeps the unextrapolated ones.
        c.Store3(kTileDual, h, ExtrapolateDual(c.Load3(kTileBase, h), dual, beta, t.lower, t.upper, contact,
                                               t.mu_first, t.mu_second));
        c.Store3(kTileBase, h, dual);
    }
    __syncwarp();
    const uint4 largest = TeamMax(c, make_uint4(__reduce_max_sync(~0u, steps[kResidualNormal]),
                                                __reduce_max_sync(~0u, steps[kResidualTangent]),
                                                __reduce_max_sync(~0u, steps[kResidualJoint]),
                                                __reduce_max_sync(~0u, steps[kResidualMomentum])));
    return {__uint_as_float(largest.x), __uint_as_float(largest.y), __uint_as_float(largest.z),
            __uint_as_float(largest.w)};
}

// Final velocities as the tails of DescendArticulationsKernel and DescendRigidBlocksKernel, then the
// joint limit impulses as WriteBlockJointLimitImpulsesKernel.
template <uint32_t kN>
__device__ void FinishTile(ArticulationDeviceState state, DataView data, BlockDescentSolveParams p,
                           BlockScratch s, const TileContext& c) {
    constexpr uint32_t kSpan = TileSpan<kN>();
    for (uint32_t k = 0u; k < c.articulations; ++k) {
        const uint32_t articulation = c.first_articulation + k;
        const size_t base = size_t{articulation} * p.max_dof;
        uint32_t link[kSpan], component[kSpan];
        bool located[kSpan], invalid = false;
        float value[kSpan];
#pragma unroll
        for (uint32_t r = 0u; r < kSpan; ++r) {
            const uint32_t dof = c.lane + r * kTileLanes;
            link[r] = component[r] = 0u;
            located[r] = dof < p.max_dof &&
                ArticulationDofLocation(state, articulation, dof, &link[r], &component[r]);
            value[r] = 0.0f;
            if (located[r]) {
                const uint32_t root = p.mimic_couplings != 0u ? data.mimic_root_dof[base + dof] : dof;
                for (uint32_t compact = 0u; compact < s.articulation_dimensions[articulation]; ++compact) {
                    if (s.articulation_roots[base + compact] == root) {
                        value[r] = s.articulation_velocity[base + compact];
                        break;
                    }
                }
                if (p.mimic_couplings != 0u) value[r] *= data.mimic_root_scale[base + dof];
            }
            invalid |= !(fabsf(value[r]) <= FLT_MAX);
        }
        if (__any_sync(~0u, invalid)) {
            if (c.lane == 0u)
                RecordTileFailure(data, p, s, c, p.total_particle_count + p.total_body_count + articulation,
                                  nk::BlockSolveFailure::InvalidCandidate);
            continue;
        }
#pragma unroll
        for (uint32_t r = 0u; r < kSpan; ++r) {
            const uint32_t dof = c.lane + r * kTileLanes;
            if (dof >= p.max_dof) continue;
            data.qdot_flat[base + dof] = value[r];
            if (!located[r]) continue;
            if (component[r] == ~0u) state.qdot[link[r]] = value[r];
            else state.link_velocity[link[r]].v[component[r]] = value[r];
        }
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
        // A speculative row's gap already includes its separation shell; a second dead zone can erase it.
        const float slop = (rows[slot].flags & nk::nk_row_flags::kSpeculative) != 0u ? 0.0f : p.position_slop;
        const bool position =
            (rows[slot].flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kVelocityOnly)) == 0u &&
            (penetration > slop || c.Value(kTileDual, h) != 0.0f);
        const float bias = position ? fminf(p.position_beta * fmaxf(penetration - slop, 0.0f) / p.dt,
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
        for (uint32_t dof = c.lane; dof < p.max_dof; dof += kTileLanes)
            s.articulation_pseudo[size_t{c.first_articulation + k} * p.max_dof + dof] = 0.0f;
    __syncwarp();
    if (!__any_sync(~0u, push)) return 0u;
    IndexTileIncidence(c);
    return __any_sync(~0u, coupled) ? p.position_iterations : 1u;
}

// Pseudo velocities for IntegratePosition, expanded from block coordinates as FinishTile.
__device__ void FinishTilePositions(ArticulationDeviceState state, DataView data, BlockDescentSolveParams p,
                                    BlockScratch s, const TileContext& c) {
    auto* link_pseudo = reinterpret_cast<LinkSpatialVel*>(data.link_velocity_pseudo);
    for (uint32_t k = 0u; k < c.articulations; ++k) {
        const uint32_t articulation = c.first_articulation + k;
        const size_t base = size_t{articulation} * p.max_dof;
        for (uint32_t dof = c.lane; dof < p.max_dof; dof += kTileLanes) {
            uint32_t link = 0u, component = 0u;
            if (!ArticulationDofLocation(state, articulation, dof, &link, &component)) continue;
            const uint32_t root = p.mimic_couplings != 0u ? data.mimic_root_dof[base + dof] : dof;
            float value = 0.0f;
            for (uint32_t compact = 0u; compact < s.articulation_dimensions[articulation]; ++compact)
                if (s.articulation_roots[base + compact] == root) value = s.articulation_pseudo[base + compact];
            if (p.mimic_couplings != 0u) value *= data.mimic_root_scale[base + dof];
            if (component == ~0u) data.qdot_pseudo[link] = value;
            else link_pseudo[link].v[component] = value;
        }
    }
    for (uint32_t k = c.lane; k < c.bodies; k += kTileLanes) {
        const uint32_t body = c.first_body + k;
        const float* u = s.rigid_pseudo + size_t{body} * kRigidBlockDof;
        data.body_pseudo_linear_velocity[body] = {u[0], u[1], u[2]};
        data.body_pseudo_angular_velocity[body] = {u[3], u[4], u[5]};
    }
}

// One Gauss-Seidel pass over the tile's blocks in color order. A color's blocks share no head, so a team's
// warps take alternate ones; the last color may hold neighbors, so its first warp descends it in order.
template <uint32_t kN>
__device__ void DescendTileBlocks(DataView data, BlockDescentSolveParams p, BlockScratch s, TileContext& c,
                                  bool position) {
    for (uint32_t pending = c.colors; pending != 0u; pending &= pending - 1u) {
        const uint32_t color = __ffs(pending) - 1u;
        const uint32_t before = __shfl_sync(~0u, c.color_stop, (color + kTileColors - 1u) % kTileColors);
        const uint32_t begin = color == 0u ? 0u : before, end = __shfl_sync(~0u, c.color_stop, color);
        const bool shared = color + 1u < kTileColors;
        for (uint32_t k = shared ? begin + c.member : c.member == 0u ? begin : end; k < end;
             k += shared ? c.team : 1u) {
            const uint32_t local = c.order[k];
            const TileBlock b = LoadTileBlock(data, p, s, c, local, position);
            if (local < c.bodies) DescendTileBlock<kRigidBlockDof>(data, p, s, c, b);
            else DescendTileBlock<kN>(data, p, s, c, b);
        }
        TeamSync(c);
    }
}

// Nanoseconds of a timed tile solve per phase; Stamp closes the running phase into `phase`.
struct TilePhaseClock {
    bool timed;
    unsigned long long last;
    unsigned long long total[kSolverPhaseWords];
    __device__ explicit TilePhaseClock(bool on) : timed(on), last(on ? GlobalTimer() : 0u), total{} {}
    __device__ void Stamp(uint32_t phase) {
        if (!timed) return;
        const unsigned long long now = GlobalTimer();
        total[phase] += now - last;
        last = now;
    }
};

// An environment's velocity sweeps until its residuals pass the tolerance or the budget is spent, as the
// sweep solve ends each env, then its split-impulse position pass. A team's first warp sets up and finishes.
template <uint32_t kN>
__device__ void DescendTile(ArticulationDeviceState state, DataView data, BlockDescentSolveParams p, BlockScratch s,
                            TileContext& c) {
    TilePhaseClock clock(p.telemetry != 0u);
    if (c.member == 0u) InitializeTile<kN>(state, data, p, s, c);
    c.heads = TeamBroadcast(c, c.heads);
    LoadTileColors(c);
    clock.Stamp(kPhaseSetup);
    // Lane k holds the residual of class k.
    float residual = 0.0f;
    uint32_t sweeps = 0u;
    DualMomentum momentum = StartDualMomentum();
    float step = FLT_MAX;
    while (sweeps < p.iterations) {
        c.iteration = sweeps++;
        c.momentum = 0.0f;
        DescendTileBlocks<kN>(data, p, s, c, false);
        c.momentum = __uint_as_float(TeamMax(c, make_uint4(__float_as_uint(c.momentum), 0u, 0u, 0u)).x);
        clock.Stamp(kPhaseDense);
        AdvanceDualMomentum(momentum, step);
        clock.Stamp(kPhaseMomentum);
        const float4 rows = UpdateTileDuals(data, c, momentum.beta);
        step = rows.w;
        clock.Stamp(kPhaseDual);
        residual = c.lane == kResidualNormal ? rows.x : c.lane == kResidualTangent ? rows.y
                 : c.lane == kResidualJoint ? rows.z : c.momentum;
        if ((p.telemetry & nk::kTelemetrySweeps) != 0u && sweeps <= p.trace_sweeps && c.member == 0u &&
            c.lane < kResidualClasses)
            data.solver_sweep_trace[(size_t{c.env} * p.trace_sweeps + sweeps - 1u) * kResidualClasses + c.lane] =
                residual;
        const float largest = fmaxf(fmaxf(rows.x, rows.y), fmaxf(rows.z, c.momentum));
        clock.Stamp(kPhaseCheck);
        if (largest < p.velocity_tolerance) break;
    }
    if (c.member == 0u) {
        if (c.lane < kResidualClasses) data.solver_residual[c.env * kResidualClasses + c.lane] = residual;
        if (c.lane == 0u) data.vbd_velocity_sweep_count[c.env] = sweeps;
        FinishTile<kN>(state, data, p, s, c);
    }
    if (p.position_iterations != 0u) {
        const uint32_t positions = TeamBroadcast(c, c.member == 0u ? InitializeTilePositions(data, p, s, c) : 0u);
        for (uint32_t iteration = 0u; iteration < positions; ++iteration) {
            c.iteration = sweeps + iteration;
            DescendTileBlocks<kN>(data, p, s, c, true);
            if (iteration + 1u < positions) UpdateTileDuals(data, c, 0.0f, false);
        }
        if (c.member == 0u) FinishTilePositions(state, data, p, s, c);
    }
    clock.Stamp(kPhaseDense);
    if (c.member == 0u)
        for (uint32_t phase = c.lane; phase < kSolverPhaseWords; phase += kTileLanes)
            data.solver_phase_time[c.env * kSolverPhaseWords + phase] = clock.total[phase];
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

// Resident tiles take environments in cost order, one per warp or one per team of warps; wide blocks
// cooperate on Hessian assembly. A tile retains the same block order and convergence checks at every width.
template <uint32_t kN>
__global__ void __launch_bounds__(TileBoundThreads<kN>()) DescendDenseTilesKernel(
    ArticulationDeviceState state, DataView data, BlockDescentSolveParams p, BlockScratch s, uint32_t team) {
    extern __shared__ __align__(16) float tile_shared[];
    __shared__ TileChunkWork chunk_work;
    __shared__ TileContext shared_context;
    __shared__ uint4 team_share[kTileTeamWarps];
    const uint32_t warp = threadIdx.x / kTileLanes;
    const uint32_t lane = threadIdx.x % kTileLanes;
    if constexpr (TileSpan<kN>() > 1u) {
        if (warp != 0u) {
            for (;;) {
                TileWorkSync();
                if (chunk_work.stop) return;
                ExecuteTileWork<kN>(data, p, s, chunk_work);
                TileWorkSync();
            }
        }
    }
    const uint32_t slot = team > 1u ? blockIdx.x : blockIdx.x * TileWarps<kN>() + warp;
    for (;;) {
        uint32_t ticket = 0u;
        if (lane == 0u && warp % team == 0u) ticket = atomicAdd(s.tile_next, 1u);
        ticket = __shfl_sync(~0u, ticket, 0u);
        if (team > 1u) {
            if (threadIdx.x == 0u) team_share[0].x = ticket;
            __syncthreads();
            ticket = team_share[0].x;
            __syncthreads();
        }
        if (ticket >= p.env_count) {
            if constexpr (TileSpan<kN>() > 1u) {
                if (lane == 0u) chunk_work.stop = true;
                TileWorkSync();
            }
            return;
        }
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
        c.width = TileJacobianWidth(p);
        c.team = team;
        c.member = team > 1u ? warp : 0u;
        c.fields = s.tile_fields + size_t{slot} * kTileFields * c.capacity;
        c.jacobian = s.tile_jacobian + size_t{slot} * s.tile_row_capacity * c.width;
        c.incident_end = s.tile_incidence + size_t{slot} * s.tile_incidence_stride;
        c.incident = c.incident_end + c.bodies + c.articulations;
        c.color = c.incident + 2u * c.capacity;
        c.order = c.color + c.bodies + c.articulations;
        c.color_end = c.order + c.bodies + c.articulations;
        c.team_share = team_share;
        c.mass = tile_shared + size_t{warp} * TileWarpFloats<kN>();
        c.factor = c.mass + kN * (kN + 1u);
        c.chunk_terms = reinterpret_cast<float4*>(c.factor + kN * (kN + 1u));
        c.chunk_rows = reinterpret_cast<uint2*>(c.chunk_terms + kTileChunkRows);
        c.stage = reinterpret_cast<float*>(c.chunk_rows + kTileChunkRows);
        c.chunk_work = TileSpan<kN>() > 1u ? &chunk_work : nullptr;
        c.shared_context = TileSpan<kN>() > 1u ? &shared_context : nullptr;
        c.iteration = ~0u;
        const long long start = clock64();
        DescendTile<kN>(state, data, p, s, c);
        if (lane == 0u && c.member == 0u) s.tile_cost[env] = static_cast<uint32_t>(min(clock64() - start, 0xffffffffll));
    }
}

template <uint32_t kN>
constexpr size_t TileSharedBytes(uint32_t warps = TileWarps<kN>()) {
    return size_t{warps} * TileWarpFloats<kN>() * sizeof(float);
}

// Blocks of the tile kernel resident for p's environments, with the cache carveout the launch uses.
template <uint32_t kN>
cudaError_t TileResidentBlocks(const BlockDescentSolveParams& p, uint32_t* blocks) {
    constexpr uint32_t warps = TileWarps<kN>();
    const cudaError_t status =
        PreferCacheCarveout(DescendDenseTilesKernel<kN>, TileThreads<kN>(), TileSharedBytes<kN>());
    if (status != cudaSuccess) return status;
    return ResidentGridSize(DescendDenseTilesKernel<kN>, TileThreads<kN>(), TileSharedBytes<kN>(),
                            (p.env_count + warps - 1u) / warps, blocks);
}

// Warps per environment: the widest team, with at least two blocks per warp, that keeps every environment
// resident within the slots of one warp per environment; else one warp each.
template <uint32_t kN>
cudaError_t TileTeam(const BlockDescentSolveParams& p, uint32_t* team, uint32_t* blocks) {
    *team = 1u;
    cudaError_t status = TileResidentBlocks<kN>(p, blocks);
    if (status != cudaSuccess) return status;
    const uint32_t per_env = (p.total_body_count + p.articulation_count) / p.env_count;
    const uint32_t slots = *blocks * TileWarps<kN>();
    for (uint32_t width = kTileTeamWarps; width > 1u; width /= 2u) {
        if (width > TileTeamWarps<kN>() || 2u * width > per_env || p.env_count > slots) continue;
        uint32_t resident = 0u;
        status = ResidentGridSize(DescendDenseTilesKernel<kN>, width * kTileLanes, TileSharedBytes<kN>(width),
                                  p.env_count, &resident);
        if (status != cudaSuccess) return status;
        if (resident < p.env_count) continue;
        *team = width;
        *blocks = p.env_count;
        return PreferCacheCarveout(DescendDenseTilesKernel<kN>, width * kTileLanes, TileSharedBytes<kN>(width));
    }
    return cudaSuccess;
}

template <uint32_t kN>
cudaError_t LaunchDenseTiles(ArticulationDeviceState state, DataView data, BlockDescentSolveParams p,
                             BlockScratch s, cudaStream_t stream) {
    uint32_t team = 1u, blocks = 0u;
    const cudaError_t status = TileTeam<kN>(p, &team, &blocks);
    if (status != cudaSuccess) return status;
    const uint32_t warps = team > 1u ? team : TileWarps<kN>();
    LaunchCuda(DescendDenseTilesKernel<kN>, dim3(blocks), dim3(team > 1u ? warps * kTileLanes : TileThreads<kN>()),
               TileSharedBytes<kN>(warps), stream, state, data, p, s, team);
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
        case 32u: return visit(std::integral_constant<uint32_t, 32u>{});
        case 48u: return visit(std::integral_constant<uint32_t, 48u>{});
        default: return visit(std::integral_constant<uint32_t, kMaxArticulationDof>{});
    }
}

uint32_t TileWarpSlots(const BlockDescentSolveParams& p) {
    uint32_t slots = 0u;
    const cudaError_t status = VisitTileWidth(p, [&](auto width) {
        uint32_t blocks = 0u;
        const cudaError_t resident = TileResidentBlocks<decltype(width)::value>(p, &blocks);
        slots = blocks * TileWarps<decltype(width)::value>();
        return resident;
    });
    return status == cudaSuccess ? slots : 0u;
}
