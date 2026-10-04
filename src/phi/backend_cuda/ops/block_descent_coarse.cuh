#pragma once
// Galerkin correction of each point-owner family along one uniform translation per environment.
// Elastic and Rayleigh energies depend only on rate differences, so inertia and rows span it exactly.

__device__ inline double WarpSumDouble(double value) {
    for (uint32_t offset = warpSize / 2u; offset > 0u; offset /= 2u)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

// Fixed-order sum over the first kThreads threads; every thread of the block calls it and
// thread 0 receives the sum.
__device__ inline double BlockSumDouble(double value, double* shared) {
    value = WarpSumDouble(value);
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    __syncthreads();
    if (lane == 0u && warp < kThreads / 32u) shared[warp] = value;
    __syncthreads();
    double sum = 0.0;
    if (threadIdx.x == 0u)
        for (uint32_t w = 0u; w < kThreads / 32u; ++w) sum += shared[w];
    return sum;
}

// BlockSumDouble of every value at once, sharing its two barriers; thread 0 receives the sums.
template <uint32_t kCount>
__device__ inline void BlockSumsDouble(double (&values)[kCount], double (&shared)[kCount][kThreads / 32u]) {
    for (uint32_t k = 0u; k < kCount; ++k) values[k] = WarpSumDouble(values[k]);
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    __syncthreads();
    if (lane == 0u && warp < kThreads / 32u)
        for (uint32_t k = 0u; k < kCount; ++k) shared[k][warp] = values[k];
    __syncthreads();
    if (threadIdx.x != 0u) return;
    for (uint32_t k = 0u; k < kCount; ++k) {
        double sum = 0.0;
        for (uint32_t w = 0u; w < kThreads / 32u; ++w) sum += shared[k][w];
        values[k] = sum;
    }
}

// Helper threads per owner thread in the coarse sums: they evaluate the owner's rows, which the
// owner adds in its own order.
constexpr uint32_t kCoarseHelpers = 4u;

__device__ inline uint32_t CoarseFamily(const BlockDescentSolveParams& p, uint32_t particle) {
    const VertexBlockLayout& l = p.vertex_blocks;
    const uint32_t local = particle % (p.total_particle_count / p.env_count);
    if (local < p.grid_particles_per_env) return kNoCoarseFamily;
    return l.vertices > 0u && local >= l.begin && local - l.begin < l.vertices
        ? kCoarseVertexFamily : kCoarseParticleFamily;
}

// Inertia and free rate exactly as the particle descent uses them.
__device__ inline bool CoarseParticle(const VertexBlockView& b, const DataView& data,
                                      const BlockDescentSolveParams& p, const BlockScratch& s,
                                      uint32_t particle, uint32_t family,
                                      float* inertia, Vec3* free_rate) {
    if (particle >= p.total_particle_count || CoarseFamily(p, particle) != family ||
        !(data.particle_inv_mass[particle] > 0.0f)) return false;
    if (family == kCoarseVertexFamily) {
        const uint32_t env = particle / b.layout.particles_per_env;
        const uint32_t vertex = particle % b.layout.particles_per_env - b.layout.begin;
        *inertia = b.inertia[b.Slot(env, vertex)];
        *free_rate = b.free_rate[b.Slot(env, vertex)];
    } else {
        *inertia = 1.0f / (data.particle_inv_mass[particle] * p.dt * p.dt);
        *free_rate = s.particle_free[particle];
    }
    return true;
}

// The family's share of each row axis: the sum of its dynamic particle Jacobians.
__device__ inline bool CoarseRowJacobians(const DataView& data, const BlockDescentSolveParams& p,
                                          PointMassView points, uint32_t slot, Vec3 jacobian[3],
                                          uint32_t family) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const NkRow normal = rows[slot];
    const bool contact = (normal.flags & nk::nk_row_flags::kBlockNormal) != 0u;
    bool touched = false;
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
        jacobian[axis] = {};
        if (axis > 0u && !contact) continue;
        const NkRow row = rows[slot + axis * normal.group_normal_count];
        const NkRowSide sides[2] = {row.a, row.b};
        for (const NkRowSide& side : sides) {
            if (!PointMassView::IsPointSide(side.kind)) continue;
            for (uint32_t term = 0u; term < points.Count(side); ++term) {
                const auto entry = points.At(side, term);
                if (entry.kind != kNkSideParticle || entry.index >= p.total_particle_count ||
                    CoarseFamily(p, entry.index) != family ||
                    !(data.particle_inv_mass[entry.index] > 0.0f)) continue;
                jacobian[axis] += entry.jacobian;
                touched = true;
            }
        }
    }
    return touched;
}

// A selected row head with the family's Jacobians; tangent axes travel with their normal head.
__device__ inline bool CoarseRowTerm(const DataView& data, const BlockDescentSolveParams& p,
                                     const BlockScratch& s, PointMassView points, uint32_t slot,
                                     uint32_t family, LocalTerm* term) {
    Vec3 jacobian[3];
    if (!CoarseRowJacobians(data, p, points, slot, jacobian, family)) return false;
    *term = LoadLocalTerm(data, p, s, slot, ~0u, false);
    for (uint32_t axis = 0u; axis < 3u; ++axis) term->jacobian[axis] = jacobian[axis];
    return true;
}

// First selected row at or after `slot`; the selection is in slot order.
__device__ inline uint32_t CoarseRowLowerBound(const BlockScratch& s, uint32_t count, uint64_t slot) {
    uint32_t low = 0u, high = count;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2u;
        if (s.coarse_rows[middle] < slot) low = middle + 1u;
        else high = middle;
    }
    return low;
}

// Each environment's range in the selected rows, found once per solve.
__global__ void CoarseRowRangesKernel(BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t count = *s.coarse_row_count;
    for (uint32_t env = blockIdx.x * blockDim.x + threadIdx.x; env <= p.env_count;
         env += gridDim.x * blockDim.x)
        s.coarse_row_offsets[env] = CoarseRowLowerBound(s, count, uint64_t{env} * p.rows_per_env);
}

// The environment's selected rows split into equal contiguous chunks over its blocks.
__device__ inline void CoarseRowChunk(const BlockDescentSolveParams& p, const BlockScratch& s,
                                      uint32_t env, uint32_t part, uint32_t* begin, uint32_t* end) {
    const uint32_t first = s.coarse_row_offsets[env];
    const uint32_t last = s.coarse_row_offsets[env + 1u];
    const uint64_t chunk = (uint64_t{last - first} + s.coarse_parts - 1u) / s.coarse_parts;
    const uint64_t from = first + part * chunk;
    *begin = static_cast<uint32_t>(from < last ? from : last);
    *end = static_cast<uint32_t>(from + chunk < last ? from + chunk : last);
}

// An element joining moving and fixed vertices stores energy under translation, so its mesh is anchored.
__global__ void MarkCoarseAnchorsKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                                        BlockScratch s) {
    const VertexBlockView b = Vertices(model, data, p);
    const uint64_t count = uint64_t{b.layout.elements} * p.env_count;
    for (uint64_t item = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x; item < count;
         item += uint64_t{gridDim.x} * blockDim.x) {
        const uint32_t env = static_cast<uint32_t>(item / b.layout.elements);
        const nk::VbdElement element = b.elements[item % b.layout.elements];
        uint32_t moving = 0u;
        const uint32_t vertices = nk::VbdElementVertexCount(element.kind);
        for (uint32_t j = 0u; j < vertices; ++j)
            moving += data.particle_inv_mass[b.Particle(env, element.vertex[j])] > 0.0f ? 1u : 0u;
        if (moving > 0u && moving < vertices) atomicExch(s.coarse_anchored + env, 1u);
    }
}

__device__ inline float CoarseTrialScale(uint32_t trial) {
    return ldexpf(1.0f, -static_cast<int>(trial));
}

// Blocks own fixed strided slices of one environment, so every partial sum has a fixed order.
// Owner threads keep the sums; each round evaluates kCoarseHelpers rows per owner at once.
__global__ void __launch_bounds__(kThreads * kCoarseHelpers)
CoarseGradientKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                     BlockScratch s, uint32_t family) {
    __shared__ double shared[kCoarseGradientTerms][kThreads / 32u];
    __shared__ float row_terms[kCoarseHelpers][kCoarseGradientTerms][kThreads];
    __shared__ bool row_live[kCoarseHelpers][kThreads];
    const uint32_t owner = threadIdx.x % kThreads;
    const uint32_t helper = threadIdx.x / kThreads;
    const uint32_t env = blockIdx.x / s.coarse_parts;
    const uint32_t part = blockIdx.x % s.coarse_parts;
    const uint32_t stride = s.coarse_parts * kThreads;
    const VertexBlockView b = Vertices(model, data, p);
    const auto points = PointMasses(data);
    const uint32_t per_env = p.total_particle_count / p.env_count;
    const bool anchored = family == kCoarseVertexFamily && s.coarse_anchored[env] != 0u;
    double sums[kCoarseGradientTerms] = {};
    for (uint32_t local = part * kThreads + owner; helper == 0u && !anchored && local < per_env;
         local += stride) {
        const uint32_t particle = env * per_env + local;
        float inertia = 0.0f;
        Vec3 free_rate{};
        if (!CoarseParticle(b, data, p, s, particle, family, &inertia, &free_rate)) continue;
        const Vec3 u = data.particle_vel[particle];
        const double mass = double(inertia) * p.dt * p.dt;
        sums[0] -= mass * (double(u.x) - free_rate.x);
        sums[1] -= mass * (double(u.y) - free_rate.y);
        sums[2] -= mass * (double(u.z) - free_rate.z);
        sums[3] += mass;
        sums[4] += mass;
        sums[5] += mass;
    }
    uint32_t begin = 0u, end = 0u;
    if (!anchored) CoarseRowChunk(p, s, env, part, &begin, &end);
    for (uint32_t round = begin; round < end; round += kThreads * kCoarseHelpers) {
        const uint32_t item = round + helper * kThreads + owner;
        LocalTerm term{};
        const bool live = item < end &&
            CoarseRowTerm(data, p, s, points, s.coarse_rows[item], family, &term);
        if (live) {
            Vec3 impulse{};
            SymmetricMat3 curvature{};
            EvaluateLocalResponse(term, {}, &impulse, &curvature);
            const float values[kCoarseGradientTerms] = {impulse.x, impulse.y, impulse.z,
                curvature.xx, curvature.yy, curvature.zz, curvature.xy, curvature.xz, curvature.yz};
            for (uint32_t k = 0u; k < kCoarseGradientTerms; ++k) row_terms[helper][k][owner] = values[k];
        }
        row_live[helper][owner] = live;
        __syncthreads();
        for (uint32_t h = 0u; helper == 0u && h < kCoarseHelpers; ++h) {
            if (!row_live[h][owner]) continue;
            for (uint32_t k = 0u; k < kCoarseGradientTerms; ++k) sums[k] += row_terms[h][k][owner];
        }
        __syncthreads();
    }
    BlockSumsDouble(sums, shared);
    if (threadIdx.x != 0u) return;
    double* out = s.coarse_partials + size_t{blockIdx.x} * kCoarseTrials;
    for (uint32_t k = 0u; k < kCoarseGradientTerms; ++k) out[k] = sums[k];
}

// Sums one environment's partials of `term` in a fixed thread-strided order; thread 0 gets it.
__device__ inline double CoarsePartialSum(const BlockScratch& s, uint32_t env, uint32_t term,
                                          double* shared) {
    double sum = 0.0;
    for (uint32_t part = threadIdx.x; part < s.coarse_parts; part += blockDim.x)
        sum += s.coarse_partials[(size_t{env} * s.coarse_parts + part) * kCoarseTrials + term];
    return BlockSumDouble(sum, shared);
}

// One block per environment.
__global__ void CoarseDirectionKernel(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                      uint32_t family) {
    __shared__ double shared[kCoarseGradientTerms][kThreads / 32u];
    const uint32_t env = blockIdx.x;
    double terms[kCoarseGradientTerms] = {};
    for (uint32_t part = threadIdx.x; part < s.coarse_parts; part += blockDim.x)
        for (uint32_t k = 0u; k < kCoarseGradientTerms; ++k)
            terms[k] += s.coarse_partials[(size_t{env} * s.coarse_parts + part) * kCoarseTrials + k];
    BlockSumsDouble(terms, shared);
    if (threadIdx.x != 0u) return;
    CoarseState& state = s.coarse[env];
    const double* f = terms;
    const double* h = terms + 3;
    state.slope = 0.0;
    state.scale = 0.0f;
    for (uint32_t axis = 0u; axis < 3u; ++axis) state.direction[axis] = 0.0f;
    if (!(h[0] > 0.0) || (family == kCoarseVertexFamily && s.coarse_anchored[env] != 0u)) return;
    const double cxx = h[1] * h[2] - h[5] * h[5], cyy = h[0] * h[2] - h[4] * h[4];
    const double czz = h[0] * h[1] - h[3] * h[3], cxy = h[4] * h[5] - h[3] * h[2];
    const double cxz = h[3] * h[5] - h[4] * h[1], cyz = h[3] * h[4] - h[0] * h[5];
    const double det = h[0] * cxx + h[3] * cxy + h[4] * cxz;
    const double d[3] = {(cxx * f[0] + cxy * f[1] + cxz * f[2]) / det,
                         (cxy * f[0] + cyy * f[1] + cyz * f[2]) / det,
                         (cxz * f[0] + cyz * f[1] + czz * f[2]) / det};
    bool valid = det > 0.0 && det <= DBL_MAX;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
        valid &= fabs(d[axis]) <= double(FLT_MAX) && fabs(f[axis]) <= DBL_MAX;
    if (!valid) {
        atomicOr(data.env_status + env, kEnvStatusSolverFailure);
        return;
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis) state.direction[axis] = static_cast<float>(d[axis]);
    state.slope = -(f[0] * state.direction[0] + f[1] * state.direction[1] +
                    f[2] * state.direction[2]);
}

// The unit step, one particle or row per owner thread; most environments accept it.
__global__ void __launch_bounds__(kThreads * kCoarseHelpers)
CoarseUnitTrialKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                      BlockScratch s, uint32_t family) {
    __shared__ double shared[kThreads / 32u];
    __shared__ double row_changes[kCoarseHelpers][kThreads];
    __shared__ bool row_live[kCoarseHelpers][kThreads];
    const uint32_t owner = threadIdx.x % kThreads;
    const uint32_t helper = threadIdx.x / kThreads;
    const uint32_t env = blockIdx.x / s.coarse_parts;
    const uint32_t part = blockIdx.x % s.coarse_parts;
    const uint32_t stride = s.coarse_parts * kThreads;
    const CoarseState state = s.coarse[env];
    const Vec3 direction{state.direction[0], state.direction[1], state.direction[2]};
    const bool descending = state.slope < 0.0;
    const VertexBlockView b = Vertices(model, data, p);
    const auto points = PointMasses(data);
    const uint32_t per_env = p.total_particle_count / p.env_count;
    double change = 0.0;
    for (uint32_t local = part * kThreads + owner; helper == 0u && descending && local < per_env;
         local += stride) {
        const uint32_t particle = env * per_env + local;
        float inertia = 0.0f;
        Vec3 free_rate{};
        if (!CoarseParticle(b, data, p, s, particle, family, &inertia, &free_rate)) continue;
        const Vec3 u = data.particle_vel[particle];
        const Vec3 candidate{nk::vbd::TrialValue(u.x, direction.x, 1.0f),
                             nk::vbd::TrialValue(u.y, direction.y, 1.0f),
                             nk::vbd::TrialValue(u.z, direction.z, 1.0f)};
        change += nk::vbd::InertialEnergyChange(u, free_rate, candidate - u, inertia, p.dt);
    }
    uint32_t begin = 0u, end = 0u;
    if (descending) CoarseRowChunk(p, s, env, part, &begin, &end);
    for (uint32_t round = begin; round < end; round += kThreads * kCoarseHelpers) {
        const uint32_t item = round + helper * kThreads + owner;
        LocalTerm term{};
        const bool live = item < end &&
            CoarseRowTerm(data, p, s, points, s.coarse_rows[item], family, &term);
        if (live) row_changes[helper][owner] = EvaluateLocalChange(term, direction);
        row_live[helper][owner] = live;
        __syncthreads();
        for (uint32_t h = 0u; helper == 0u && h < kCoarseHelpers; ++h)
            if (row_live[h][owner]) change += row_changes[h][owner];
        __syncthreads();
    }
    const double sum = BlockSumDouble(change, shared);
    if (threadIdx.x == 0u) s.coarse_partials[size_t{blockIdx.x} * kCoarseTrials] = sum;
}

// Halving search where the unit step failed. Lanes are trials: a warp loads each particle or row
// once and evaluates every scale on the rounded candidate the apply kernel would write.
__global__ void CoarseTrialsKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                                   BlockScratch s, uint32_t family) {
    static_assert(kCoarseTrials <= 32u);
    __shared__ double shared[kThreads / 32u][32];
    const uint32_t env = blockIdx.x / s.coarse_parts;
    const uint32_t part = blockIdx.x % s.coarse_parts;
    const uint32_t lane = threadIdx.x % 32u, warp = threadIdx.x / 32u, warps = blockDim.x / 32u;
    const float scale = CoarseTrialScale(lane < kCoarseTrials ? lane : 0u);
    const CoarseState state = s.coarse[env];
    const Vec3 direction{state.direction[0], state.direction[1], state.direction[2]};
    const bool descending = state.scale < 0.0f;
    if (!descending) return;
    const VertexBlockView b = Vertices(model, data, p);
    const auto points = PointMasses(data);
    const uint32_t per_env = p.total_particle_count / p.env_count;
    double change = 0.0;
    for (uint32_t local = part * warps + warp; descending && local < per_env;
         local += s.coarse_parts * warps) {
        const uint32_t particle = env * per_env + local;
        float inertia = 0.0f;
        Vec3 free_rate{};
        if (!CoarseParticle(b, data, p, s, particle, family, &inertia, &free_rate)) continue;
        const Vec3 u = data.particle_vel[particle];
        const Vec3 candidate{nk::vbd::TrialValue(u.x, direction.x, scale),
                             nk::vbd::TrialValue(u.y, direction.y, scale),
                             nk::vbd::TrialValue(u.z, direction.z, scale)};
        change += nk::vbd::InertialEnergyChange(u, free_rate, candidate - u, inertia, p.dt);
    }
    uint32_t begin = 0u, end = 0u;
    if (descending) CoarseRowChunk(p, s, env, part, &begin, &end);
    for (uint32_t item = begin + warp; item < end; item += warps) {
        LocalTerm term{};
        if (!CoarseRowTerm(data, p, s, points, s.coarse_rows[item], family, &term)) continue;
        change += EvaluateLocalChange(term, direction * scale);
    }
    shared[warp][lane] = change;
    __syncthreads();
    if (threadIdx.x >= kCoarseTrials) return;
    double sum = 0.0;
    for (uint32_t w = 0u; w < warps; ++w) sum += shared[w][threadIdx.x];
    s.coarse_partials[size_t{blockIdx.x} * kCoarseTrials + threadIdx.x] = sum;
}

// One block per environment; the first trial meeting the sufficient decrease is accepted.
// After the unit trial a failed environment gets scale -1, which requests the search.
__global__ void CoarseAcceptKernel(BlockDescentSolveParams p, BlockScratch s, bool search) {
    __shared__ double shared[kThreads / 32u];
    const uint32_t env = blockIdx.x;
    const double slope = s.coarse[env].slope;
    if (search && !(s.coarse[env].scale < 0.0f)) return;
    if (!(slope < 0.0)) {
        if (threadIdx.x == 0u) s.coarse[env].scale = 0.0f;
        return;
    }
    if (!search) {
        const double change = CoarsePartialSum(s, env, 0u, shared);
        if (threadIdx.x == 0u)
            s.coarse[env].scale = isfinite(change) &&
                change <= double(kVertexStepDecrease) * slope ? 1.0f : -1.0f;
        return;
    }
    float accepted = 0.0f;
    for (uint32_t trial = 0u; trial < kCoarseTrials; ++trial) {
        const double change = CoarsePartialSum(s, env, trial, shared);
        const float scale = CoarseTrialScale(trial);
        if (threadIdx.x == 0u && accepted == 0.0f && isfinite(change) &&
            change <= double(kVertexStepDecrease) * scale * slope) accepted = scale;
    }
    if (threadIdx.x == 0u) s.coarse[env].scale = accepted;
}

__global__ void CoarseApplyKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                                  BlockScratch s, uint32_t family) {
    const VertexBlockView b = Vertices(model, data, p);
    const uint32_t per_env = p.total_particle_count / p.env_count;
    for (uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x; particle < p.total_particle_count;
         particle += gridDim.x * blockDim.x) {
        float inertia = 0.0f;
        Vec3 free_rate{};
        if (!CoarseParticle(b, data, p, s, particle, family, &inertia, &free_rate)) continue;
        const CoarseState& state = s.coarse[particle / per_env];
        if (!(state.scale > 0.0f)) continue;
        const Vec3 u = data.particle_vel[particle];
        data.particle_vel[particle] = {nk::vbd::TrialValue(u.x, state.direction[0], state.scale),
                                       nk::vbd::TrialValue(u.y, state.direction[1], state.scale),
                                       nk::vbd::TrialValue(u.z, state.direction[2], state.scale)};
    }
}
