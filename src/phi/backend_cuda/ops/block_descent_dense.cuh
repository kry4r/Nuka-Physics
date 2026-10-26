#pragma once

__device__ void CountDenseWork(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                uint32_t env, uint32_t kind) {
    if ((p.telemetry & nk::kTelemetrySweeps) != 0u && threadIdx.x == 0u) {
        const uint32_t sweep = s.sweep_state[kSweepCount];
        if (sweep < p.trace_sweeps)
            atomicAdd(data.solver_work_trace + (size_t{env} * p.trace_sweeps + sweep) * kSolverWorkWords + kind, 1u);
    }
}

// Dense blocks (articulations, rigid bodies) minimize their convex local augmented Lagrangian with
// Newton steps and an exact line search, so stiff rows that activate along a step enter the next Hessian.
constexpr uint32_t kDenseNewtonSteps = 16u;
constexpr uint32_t kDenseLineSearchSteps = 32u;
constexpr uint32_t kDenseTileRows = 32u;
constexpr double kDenseLineSearchTolerance = 1.0e-6;
// Gradients and steps within this many float roundings of their terms are converged.
constexpr float kDenseRoundings = 8.0f;

// Dense owners keep each incidence's row move, step, Jacobian bound, parked slack and working list
// entry in the point Jacobian columns, which only point owners otherwise use.
constexpr uint32_t kDenseMoveColumn = 0u;
constexpr uint32_t kDenseStepColumn = 3u;
constexpr uint32_t kDenseBoundColumn = 6u;
constexpr uint32_t kDenseSlackColumn = 7u;
constexpr uint32_t kDenseRowsColumn = 8u;

__device__ Vec3 LoadIncidenceVec3(BlockScratch s, size_t at, uint32_t column) {
    const float* value = s.point_jacobian + uint64_t{column} * s.incidence_capacity + at;
    return {value[0], value[s.incidence_capacity], value[2u * s.incidence_capacity]};
}

__device__ void StoreIncidenceVec3(BlockScratch s, size_t at, uint32_t column, Vec3 v) {
    float* value = s.point_jacobian + uint64_t{column} * s.incidence_capacity + at;
    value[0] = v.x;
    value[s.incidence_capacity] = v.y;
    value[2u * s.incidence_capacity] = v.z;
}

__device__ float& DenseIncidenceScalar(BlockScratch s, size_t at, uint32_t column) {
    return s.point_jacobian[uint64_t{column} * s.incidence_capacity + at];
}

__device__ uint32_t* DenseRows(BlockScratch s) {
    return reinterpret_cast<uint32_t*>(s.point_jacobian + uint64_t{kDenseRowsColumn} * s.incidence_capacity);
}

// Each dense incidence's normal Jacobian 1-norm, which bounds its row move by the block's largest
// velocity change.
__global__ void CacheDenseIncidenceBoundsKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t first = p.total_particle_count;
    const uint32_t last = first + p.total_body_count + p.articulation_count;
    const uint64_t end = s.offsets[last];
    for (uint64_t at = s.offsets[first] + uint64_t{blockIdx.x} * blockDim.x + threadIdx.x; at < end;
         at += uint64_t{gridDim.x} * blockDim.x) {
        uint32_t owner = first, above = last;
        while (above - owner > 1u) {
            const uint32_t middle = owner + (above - owner) / 2u;
            if (s.offsets[middle] <= at) owner = middle;
            else above = middle;
        }
        const uint32_t slot = s.incidence[at];
        double bound = 0.0;
        if (owner < first + p.total_body_count) {
            for (uint32_t dof = 0u; dof < kRigidBlockDof; ++dof)
                bound += fabsf(RigidJacobian(data, slot, owner - first, dof));
        } else {
            for (uint32_t dof = 0u; dof < p.max_dof; ++dof)
                bound += fabsf(ArticulationJacobian(data, p, slot, owner - first - p.total_body_count, dof));
        }
        DenseIncidenceScalar(s, at, kDenseBoundColumn) = static_cast<float>(bound);
    }
}

// Every thread receives the CTA sums, added in warp order.
__device__ void SumDenseBlock3(double* a, double* b, double* c) {
    __shared__ double partial[3][32];
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    const uint32_t warps = blockDim.x / warpSize;
    const double sums[3] = {SumDenseBlockWarp(*a), SumDenseBlockWarp(*b), SumDenseBlockWarp(*c)};
    if (lane == 0u)
        for (uint32_t k = 0u; k < 3u; ++k) partial[k][warp] = sums[k];
    __syncthreads();
    double total[3] = {};
    for (uint32_t w = 0u; w < warps; ++w)
        for (uint32_t k = 0u; k < 3u; ++k) total[k] += partial[k][w];
    __syncthreads();
    *a = total[0];
    *b = total[1];
    *c = total[2];
}

__device__ void MaxDenseBlock2(float* a, float* b) {
    __shared__ float partial[2][32];
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    const uint32_t warps = blockDim.x / warpSize;
    float x = *a, y = *b;
    for (uint32_t offset = warpSize / 2u; offset > 0u; offset /= 2u) {
        x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, offset));
        y = fmaxf(y, __shfl_down_sync(0xffffffffu, y, offset));
    }
    if (lane == 0u) {
        partial[0][warp] = x;
        partial[1][warp] = y;
    }
    __syncthreads();
    x = 0.0f;
    y = 0.0f;
    for (uint32_t w = 0u; w < warps; ++w) {
        x = fmaxf(x, partial[0][w]);
        y = fmaxf(y, partial[1][w]);
    }
    __syncthreads();
    *a = x;
    *b = y;
}

// Writes H = M + sum J^T C J (lower triangle) and g = -M(u - u_free) + sum J^T f at the current
// velocity; `converged` reports every |g_i| within rounding of the terms that form it.
template <typename Block>
__device__ bool AssembleDenseBlock(const Block& b, DataView data, BlockDescentSolveParams p,
                                   BlockScratch s, uint32_t working, double* gradient, double* inertia,
                                   bool* converged) {
    constexpr uint32_t kMaxDof = Block::kMaxDof;
    constexpr uint32_t kEntries = (kMaxDof * (kMaxDof + 1u) / 2u + kThreads - 1u) / kThreads;
    static_assert(kMaxDof <= kThreads && kMaxDof <= 0xffffu);
    __shared__ float jacobian[kDenseTileRows][3][kMaxDof];
    __shared__ float response[kDenseTileRows][7];
    __shared__ float noise[kDenseTileRows][3];
    __shared__ uint32_t tile_slot[kDenseTileRows];
    __shared__ uint32_t tile_axes[kDenseTileRows];
    __shared__ uint32_t tile_stride[kDenseTileRows];
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint32_t n = b.dimension;
    const float* u = b.velocity;
    uint32_t pair[kEntries];
    float h[kEntries];
#pragma unroll
    for (uint32_t q = 0u; q < kEntries; ++q) {
        const uint32_t e = threadIdx.x + q * blockDim.x;
        pair[q] = ~0u;
        h[q] = 0.0f;
        if (e < n * (n + 1u) / 2u) {
            uint32_t i = static_cast<uint32_t>((sqrtf(8.0f * static_cast<float>(e) + 1.0f) - 1.0f) * 0.5f);
            while (i * (i + 1u) / 2u > e) --i;
            while ((i + 1u) * (i + 2u) / 2u <= e) ++i;
            const uint32_t j = e - i * (i + 1u) / 2u;
            pair[q] = i << 16u | j;
            h[q] = b.Mass(i, j);
        }
    }
    double force = 0.0, inertial = 0.0;
    float scale = 0.0f;
    if (threadIdx.x < n) {
        for (uint32_t k = 0u; k < n; ++k) {
            const double difference = static_cast<double>(u[k]) - static_cast<double>(b.Free(k));
            const float mass = b.Mass(threadIdx.x, k);
            inertial += static_cast<double>(mass) * difference;
            scale += fabsf(mass) * static_cast<float>(fabs(difference));
        }
        force = -inertial;
    }
    for (uint32_t first = 0u; first < working; first += kDenseTileRows) {
        const uint32_t count = min(kDenseTileRows, working - first);
        if (threadIdx.x < count) {
            const size_t at = DenseRows(s)[size_t{b.begin} + first + threadIdx.x];
            const uint32_t slot = s.incidence[at];
            const LocalTerm term = LoadLocalTerm(data, p, s, slot, ~0u, true);
            const Vec3 move = LoadIncidenceVec3(s, at, kDenseMoveColumn);
            Vec3 f{};
            SymmetricMat3 c{};
            EvaluateAugmentedResponse(term, move, &f, &c);
            float* r = response[threadIdx.x];
            r[0] = f.x;
            r[1] = f.y;
            r[2] = f.z;
            r[3] = c.xx;
            r[4] = c.yy;
            r[5] = c.zz;
            r[6] = c.yz;
            // A rounding of the residual or move changes the impulse by the curvature times it.
            noise[threadIdx.x][0] = fabsf(f.x) + c.xx * (fabsf(term.residual.x) + fabsf(move.x));
            noise[threadIdx.x][1] = fabsf(f.y) + c.yy * (fabsf(term.residual.y) + fabsf(move.y));
            noise[threadIdx.x][2] = fabsf(f.z) + c.zz * (fabsf(term.residual.z) + fabsf(move.z));
            tile_slot[threadIdx.x] = slot;
            tile_axes[threadIdx.x] = term.contact ? 3u : 1u;
            tile_stride[threadIdx.x] = rows[slot].group_normal_count;
        }
        __syncthreads();
        for (uint32_t item = threadIdx.x; item < count * 3u * n; item += blockDim.x) {
            const uint32_t r = item / (3u * n);
            const uint32_t axis = item / n % 3u;
            const uint32_t i = item % n;
            jacobian[r][axis][i] = axis < tile_axes[r]
                ? b.Jacobian(data, p, tile_slot[r] + axis * tile_stride[r], i) : 0.0f;
        }
        __syncthreads();
        for (uint32_t r = 0u; r < count; ++r) {
            const float cxx = response[r][3], cyy = response[r][4];
            const float czz = response[r][5], cyz = response[r][6];
            const bool tangent = tile_axes[r] > 1u;
#pragma unroll
            for (uint32_t q = 0u; q < kEntries; ++q) {
                if (pair[q] == ~0u) continue;
                const uint32_t i = pair[q] >> 16u, j = pair[q] & 0xffffu;
                float value = jacobian[r][0][i] * cxx * jacobian[r][0][j];
                if (tangent)
                    value += jacobian[r][1][i] * cyy * jacobian[r][1][j] +
                             jacobian[r][2][i] * czz * jacobian[r][2][j] +
                             cyz * (jacobian[r][1][i] * jacobian[r][2][j] + jacobian[r][2][i] * jacobian[r][1][j]);
                h[q] += value;
            }
            if (threadIdx.x < n) {
                for (uint32_t axis = 0u; axis < tile_axes[r]; ++axis) {
                    force += static_cast<double>(jacobian[r][axis][threadIdx.x]) * response[r][axis];
                    scale += fabsf(jacobian[r][axis][threadIdx.x]) * noise[r][axis];
                }
            }
        }
        __syncthreads();
    }
    bool finite = true;
#pragma unroll
    for (uint32_t q = 0u; q < kEntries; ++q) {
        if (pair[q] == ~0u) continue;
        b.matrix[size_t{pair[q] >> 16u} * b.stride + (pair[q] & 0xffffu)] = h[q];
        finite &= isfinite(h[q]);
    }
    if (threadIdx.x < n) {
        gradient[threadIdx.x] = force;
        inertia[threadIdx.x] = inertial;
        b.force[threadIdx.x] = static_cast<float>(force);
        finite &= isfinite(static_cast<float>(force));
    }
    __syncthreads();
    bool settled = true;
    if (threadIdx.x < n) {
        // A float velocity is off its optimum by up to a rounding, which moves g by H times it.
        for (uint32_t k = 0u; k < n; ++k)
            scale += fabsf(b.matrix[size_t{max(threadIdx.x, k)} * b.stride + min(threadIdx.x, k)]) * fabsf(u[k]);
        settled = fabs(force) <= static_cast<double>(kDenseRoundings * FLT_EPSILON) * scale;
    }
    *converged = __syncthreads_and(settled) != 0;
    return __syncthreads_or(!finite) == 0;
}

// AssembleDenseBlock for rigid-sized blocks: each thread folds whole rows into its own sums, which the
// CTA then adds in warp order, so the assembly scales with the rows instead of a 32-row tile.
template <typename Block>
__device__ bool AssembleSmallDenseBlock(const Block& b, DataView data, BlockDescentSolveParams p,
                                        BlockScratch s, uint32_t working, double* gradient, double* inertia,
                                        bool* converged) {
    constexpr uint32_t kMaxDof = Block::kMaxDof;
    constexpr uint32_t kPairs = kMaxDof * (kMaxDof + 1u) / 2u;
    constexpr uint32_t kSums = kPairs + 2u * kMaxDof;
    __shared__ double partial[32][kSums];
    __shared__ double total[kSums];
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint32_t n = b.dimension;
    const float* u = b.velocity;
    double sums[kSums] = {};
    for (uint32_t item = threadIdx.x; item < working; item += blockDim.x) {
        const size_t at = DenseRows(s)[size_t{b.begin} + item];
        const uint32_t slot = s.incidence[at];
        const LocalTerm term = LoadLocalTerm(data, p, s, slot, ~0u, true);
        const Vec3 move = LoadIncidenceVec3(s, at, kDenseMoveColumn);
        Vec3 f{};
        SymmetricMat3 c{};
        EvaluateAugmentedResponse(term, move, &f, &c);
        const uint32_t axes = term.contact ? 3u : 1u;
        const uint32_t stride = rows[slot].group_normal_count;
        const float response[3] = {f.x, f.y, f.z};
        const float noise[3] = {fabsf(f.x) + c.xx * (fabsf(term.residual.x) + fabsf(move.x)),
                                fabsf(f.y) + c.yy * (fabsf(term.residual.y) + fabsf(move.y)),
                                fabsf(f.z) + c.zz * (fabsf(term.residual.z) + fabsf(move.z))};
        float jacobian[3][kMaxDof];
        for (uint32_t axis = 0u; axis < 3u; ++axis)
            for (uint32_t i = 0u; i < kMaxDof; ++i)
                jacobian[axis][i] = axis < axes && i < n ? b.Jacobian(data, p, slot + axis * stride, i) : 0.0f;
        for (uint32_t i = 0u, e = 0u; i < kMaxDof; ++i) {
            for (uint32_t j = 0u; j <= i; ++j, ++e) {
                float value = jacobian[0][i] * c.xx * jacobian[0][j];
                if (axes > 1u)
                    value += jacobian[1][i] * c.yy * jacobian[1][j] + jacobian[2][i] * c.zz * jacobian[2][j] +
                             c.yz * (jacobian[1][i] * jacobian[2][j] + jacobian[2][i] * jacobian[1][j]);
                sums[e] += value;
            }
            for (uint32_t axis = 0u; axis < axes; ++axis) {
                sums[kPairs + i] += static_cast<double>(jacobian[axis][i]) * response[axis];
                sums[kPairs + kMaxDof + i] += fabsf(jacobian[axis][i]) * noise[axis];
            }
        }
    }
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    for (uint32_t k = 0u; k < kSums; ++k) {
        const double sum = SumDenseBlockWarp(sums[k]);
        if (lane == 0u) partial[warp][k] = sum;
    }
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < kSums; k += blockDim.x) {
        double value = 0.0;
        for (uint32_t w = 0u; w < blockDim.x / warpSize; ++w) value += partial[w][k];
        total[k] = value;
    }
    __syncthreads();
    bool finite = true;
    for (uint32_t e = threadIdx.x; e < n * (n + 1u) / 2u; e += blockDim.x) {
        uint32_t i = 0u;
        while ((i + 1u) * (i + 2u) / 2u <= e) ++i;
        const uint32_t j = e - i * (i + 1u) / 2u;
        const float h = b.Mass(i, j) + static_cast<float>(total[e]);
        b.matrix[size_t{i} * b.stride + j] = h;
        finite &= isfinite(h);
    }
    double force = 0.0;
    float scale = 0.0f;
    if (threadIdx.x < n) {
        double inertial = 0.0;
        for (uint32_t k = 0u; k < n; ++k) {
            const double difference = static_cast<double>(u[k]) - static_cast<double>(b.Free(k));
            const float mass = b.Mass(threadIdx.x, k);
            inertial += static_cast<double>(mass) * difference;
            scale += fabsf(mass) * static_cast<float>(fabs(difference));
        }
        force = total[kPairs + threadIdx.x] - inertial;
        scale += static_cast<float>(total[kPairs + kMaxDof + threadIdx.x]);
        gradient[threadIdx.x] = force;
        inertia[threadIdx.x] = inertial;
        b.force[threadIdx.x] = static_cast<float>(force);
        finite &= isfinite(static_cast<float>(force));
    }
    __syncthreads();
    bool settled = true;
    if (threadIdx.x < n) {
        for (uint32_t k = 0u; k < n; ++k)
            scale += fabsf(b.matrix[size_t{max(threadIdx.x, k)} * b.stride + min(threadIdx.x, k)]) * fabsf(u[k]);
        settled = fabs(force) <= static_cast<double>(kDenseRoundings * FLT_EPSILON) * scale;
    }
    *converged = __syncthreads_and(settled) != 0;
    return __syncthreads_or(!finite) == 0;
}

// w_j = J_j d for every incidence; a row group of lanes sums each row.
template <typename Block>
__device__ void StoreDenseRowSteps(const Block& b, DataView data, BlockDescentSolveParams p,
                                   BlockScratch s, uint32_t working) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    const uint32_t warps = blockDim.x / warpSize;
    const uint32_t width = RowGroupWidth(b.dimension);
    const uint32_t groups = warpSize / width;
    const uint32_t group = lane / width;
    const uint32_t group_lane = lane % width;
    for (uint32_t row_begin = warp; row_begin < working; row_begin += warps * groups) {
        const uint32_t item = row_begin + group * warps;
        const bool valid = item < working;
        size_t at = 0u;
        uint32_t slot = 0u, axes = 0u, stride = 0u;
        if (valid) {
            at = DenseRows(s)[size_t{b.begin} + item];
            slot = s.incidence[at];
            const NkRow head = rows[slot];
            axes = (head.flags & nk::nk_row_flags::kBlockNormal) != 0u ? 3u : 1u;
            stride = head.group_normal_count;
        }
        double dot[3] = {};
        if (valid) {
            for (uint32_t i = group_lane; i < b.dimension; i += width)
                for (uint32_t axis = 0u; axis < axes; ++axis)
                    dot[axis] += static_cast<double>(b.Jacobian(data, p, slot + axis * stride, i)) * b.direction[i];
        }
        for (uint32_t axis = 0u; axis < 3u; ++axis) dot[axis] = SumDenseRowGroup(dot[axis], width);
        if (valid && group_lane == 0u)
            StoreIncidenceVec3(s, at, kDenseStepColumn, {static_cast<float>(dot[0]), static_cast<float>(dot[1]),
                                                         static_cast<float>(dot[2])});
    }
}

// Slope and curvature of the local objective at scale `alpha` along the step; `magnitude` bounds
// the terms whose roundings the slope carries.
template <typename Block>
__device__ void DenseLineDerivatives(const Block& b, DataView data, BlockDescentSolveParams p,
                                     BlockScratch s, uint32_t working, double alpha, double inertial_slope,
                                     double inertial_curvature, double* slope, double* curvature,
                                     double* magnitude) {
    double row_slope = 0.0, row_curvature = 0.0, row_magnitude = 0.0;
    for (uint32_t item = threadIdx.x; item < working; item += blockDim.x) {
        const size_t at = DenseRows(s)[size_t{b.begin} + item];
        const LocalTerm term = LoadLocalTerm(data, p, s, s.incidence[at], ~0u, true);
        const Vec3 move = LoadIncidenceVec3(s, at, kDenseMoveColumn);
        const Vec3 step = LoadIncidenceVec3(s, at, kDenseStepColumn);
        const Vec3 trial{static_cast<float>(move.x + alpha * step.x),
                         static_cast<float>(move.y + alpha * step.y),
                         static_cast<float>(move.z + alpha * step.z)};
        Vec3 f{};
        SymmetricMat3 c{};
        EvaluateAugmentedResponse(term, trial, &f, &c);
        const double fx = static_cast<double>(f.x) * step.x;
        const double fy = static_cast<double>(f.y) * step.y;
        const double fz = static_cast<double>(f.z) * step.z;
        row_slope += fx + fy + fz;
        row_magnitude += fabs(fx) + fabs(fy) + fabs(fz);
        row_curvature += static_cast<double>(c.xx) * step.x * step.x +
                         static_cast<double>(c.yy) * step.y * step.y +
                         static_cast<double>(c.zz) * step.z * step.z +
                         2.0 * static_cast<double>(c.yz) * step.y * step.z;
    }
    SumDenseBlock3(&row_slope, &row_curvature, &row_magnitude);
    *slope = inertial_slope + alpha * inertial_curvature - row_slope;
    *curvature = inertial_curvature + row_curvature;
    *magnitude = fabs(inertial_slope) + alpha * inertial_curvature + row_magnitude;
}

// Safeguarded Newton iteration on the slope, which increases monotonically along the step because
// the local objective is convex. Returns the accepted scale, or 0 when the step does not descend.
template <typename Block>
__device__ double SearchDenseLine(const Block& b, DataView data, BlockDescentSolveParams p,
                                  BlockScratch s, uint32_t working, double inertial_slope,
                                  double inertial_curvature) {
    double slope = 0.0, curvature = 0.0, magnitude = 0.0;
    CountDenseWork(data, p, s, b.env, kWorkLine);
    DenseLineDerivatives(b, data, p, s, working, 0.0, inertial_slope, inertial_curvature, &slope, &curvature,
                         &magnitude);
    const double initial = slope;
    if (!(initial < 0.0)) return 0.0;
    double low = 0.0, high = 0.0, alpha = 1.0;
    bool bounded = false;
    for (uint32_t step = 0u; step < kDenseLineSearchSteps; ++step) {
        CountDenseWork(data, p, s, b.env, kWorkLine);
        DenseLineDerivatives(b, data, p, s, working, alpha, inertial_slope, inertial_curvature, &slope, &curvature,
                             &magnitude);
        if (!isfinite(slope) || !isfinite(curvature)) {
            high = alpha;
            bounded = true;
            alpha = 0.5 * (low + high);
            continue;
        }
        if (fabs(slope) <= fmax(kDenseLineSearchTolerance * -initial,
                                static_cast<double>(kDenseRoundings * FLT_EPSILON) * magnitude)) return alpha;
        if (slope < 0.0) low = alpha;
        else {
            high = alpha;
            bounded = true;
        }
        if (bounded && high - low <= static_cast<double>(FLT_EPSILON) * high) break;
        double next = alpha - slope / curvature;
        if (!(next > low && (!bounded || next < high))) next = bounded ? 0.5 * (low + high) : 2.0 * alpha;
        alpha = next;
    }
    return low;
}

// Lists the flagged incidences of a CTA-wide chunk from `base` in thread order; returns their number.
__device__ uint32_t AppendDenseRows(BlockScratch s, size_t base, bool flag, size_t at) {
    __shared__ uint32_t warp_count[32];
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    const uint32_t ballot = __ballot_sync(~0u, flag);
    if (lane == 0u) warp_count[warp] = __popc(ballot);
    __syncthreads();
    uint32_t before = 0u, total = 0u;
    for (uint32_t w = 0u; w < blockDim.x / warpSize; ++w) {
        before += w < warp ? warp_count[w] : 0u;
        total += warp_count[w];
    }
    if (flag) DenseRows(s)[base + before + __popc(ballot & ((1u << lane) - 1u))] = static_cast<uint32_t>(at);
    __syncthreads();
    return total;
}

// The row move a dense row must exceed before its impulse leaves a zero bound, while its friction has
// no radius; 0 when the row may act at once.
__device__ float DenseRowSlack(DataView data, BlockDescentSolveParams p, BlockScratch s, uint32_t slot) {
    const LocalTerm t = LoadLocalTerm(data, p, s, slot, ~0u, true);
    if (t.contact && !(t.radius <= 0.0f)) return 0.0f;
    if (t.penalty <= 0.0f) return INFINITY;
    const double k = t.response.residual_scale;
    const double x = t.response.dual_scale * t.dual.x + k * t.residual.x;
    if (t.lower == 0.0f && x < 0.0) return static_cast<float>(-x / k);
    if (t.upper == 0.0f && x > 0.0) return static_cast<float>(x / k);
    return 0.0f;
}

// Rows with a slack wait outside the Newton working set, which lists the other rows in order with
// zero moves; returns its size.
template <typename Block>
__device__ uint32_t ParkDenseRows(const Block& b, DataView data, BlockDescentSolveParams p, BlockScratch s) {
    uint32_t working = 0u;
    for (size_t first = b.begin; first < b.end; first += blockDim.x) {
        const size_t at = first + threadIdx.x;
        bool work = false;
        if (at < b.end) {
            const float slack = DenseRowSlack(data, p, s, s.incidence[at]);
            work = !(slack > 0.0f);
            DenseIncidenceScalar(s, at, kDenseSlackColumn) = work ? 0.0f : slack;
            if (work) StoreIncidenceVec3(s, at, kDenseMoveColumn, {});
        }
        working += AppendDenseRows(s, size_t{b.begin} + working, work, at);
    }
    __syncthreads();
    return working;
}

// Parked rows whose bounded move reaches their slack are evaluated at the block's velocity change and
// join the working set when they act; returns whether any joined.
template <typename Block>
__device__ bool WakeDenseRows(const Block& b, DataView data, BlockDescentSolveParams p, BlockScratch s,
                              uint32_t* working) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    float change = 0.0f, unused = 0.0f;
    for (uint32_t i = threadIdx.x; i < b.dimension; i += blockDim.x)
        change = fmaxf(change, fabsf(b.velocity[i] - b.Snapshot(i)));
    MaxDenseBlock2(&change, &unused);
    const uint32_t before = *working;
    for (size_t first = b.begin; first < b.end; first += blockDim.x) {
        const size_t at = first + threadIdx.x;
        bool wake = false;
        if (at < b.end) {
            const float slack = DenseIncidenceScalar(s, at, kDenseSlackColumn);
            const double reach = static_cast<double>(DenseIncidenceScalar(s, at, kDenseBoundColumn)) * change *
                (1.0 + kDenseRoundings * FLT_EPSILON);
            if (slack > 0.0f && !(reach < slack)) {
                const uint32_t slot = s.incidence[at];
                const LocalTerm term = LoadLocalTerm(data, p, s, slot, ~0u, true);
                const uint32_t stride = rows[slot].group_normal_count;
                double move[3] = {};
                for (uint32_t axis = 0u; axis < (term.contact ? 3u : 1u); ++axis)
                    for (uint32_t i = 0u; i < b.dimension; ++i)
                        move[axis] += static_cast<double>(b.Jacobian(data, p, slot + axis * stride, i)) *
                                      (static_cast<double>(b.velocity[i]) - b.Snapshot(i));
                const Vec3 row_move{static_cast<float>(move[0]), static_cast<float>(move[1]),
                                    static_cast<float>(move[2])};
                Vec3 f{};
                SymmetricMat3 c{};
                EvaluateAugmentedResponse(term, row_move, &f, &c);
                wake = f.x != 0.0f || c.xx != 0.0f;
                if (wake) {
                    StoreIncidenceVec3(s, at, kDenseMoveColumn, row_move);
                    DenseIncidenceScalar(s, at, kDenseSlackColumn) = 0.0f;
                }
            }
        }
        *working += AppendDenseRows(s, size_t{b.begin} + *working, wake, at);
    }
    return *working > before;
}

// Newton steps with an exact line search over the working rows from b.velocity; returns the first failure.
template <typename Block>
__device__ nk::BlockSolveFailure NewtonDenseBlock(const Block& b, DataView data, BlockDescentSolveParams p,
                                                  BlockScratch s, uint32_t working) {
    constexpr uint32_t kMaxDof = Block::kMaxDof;
    __shared__ double gradient[kMaxDof];
    __shared__ double inertia[kMaxDof];
    const uint32_t n = b.dimension;
    nk::BlockSolveFailure failure = nk::BlockSolveFailure::None;
    for (uint32_t step = 0u; step < kDenseNewtonSteps; ++step) {
        CountDenseWork(data, p, s, b.env, kWorkNewton);
        bool converged = false;
        bool assembled = false;
        if constexpr (kMaxDof <= kRigidBlockDof)
            assembled = AssembleSmallDenseBlock(b, data, p, s, working, gradient, inertia, &converged);
        else
            assembled = AssembleDenseBlock(b, data, p, s, working, gradient, inertia, &converged);
        if (!assembled) {
            failure = nk::BlockSolveFailure::InvalidEquation;
            break;
        }
        if (converged) break;
        CountDenseWork(data, p, s, b.env, kWorkFactor);
        if (!SolveDensePositiveBlock<kMaxDof>(b.matrix, b.force, b.direction, n, b.stride)) {
            failure = nk::BlockSolveFailure::Factorization;
            break;
        }
        double decrement = 0.0, inertial_slope = 0.0, inertial_curvature = 0.0;
        if (threadIdx.x < n) {
            const double d = b.direction[threadIdx.x];
            double mass_step = 0.0;
            for (uint32_t k = 0u; k < n; ++k)
                mass_step += static_cast<double>(b.Mass(threadIdx.x, k)) * b.direction[k];
            decrement = gradient[threadIdx.x] * d;
            inertial_slope = inertia[threadIdx.x] * d;
            inertial_curvature = d * mass_step;
        }
        SumDenseBlock3(&decrement, &inertial_slope, &inertial_curvature);
        if (!isfinite(decrement) || !isfinite(inertial_slope) || !isfinite(inertial_curvature)) {
            failure = nk::BlockSolveFailure::InvalidDirection;
            break;
        }
        if (!(decrement > 0.0)) break;
        StoreDenseRowSteps(b, data, p, s, working);
        __syncthreads();
        const double alpha = SearchDenseLine(b, data, p, s, working, inertial_slope, inertial_curvature);
        bool rounded = true;
        bool finite = true;
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            const float before = b.velocity[i];
            const float after = static_cast<float>(static_cast<double>(before) + alpha * b.direction[i]);
            finite &= isfinite(after);
            rounded &= fabsf(after - before) <=
                kDenseRoundings * FLT_EPSILON * fmaxf(fabsf(after), fabsf(before));
        }
        rounded = __syncthreads_and(rounded) != 0;
        if (__syncthreads_or(!finite)) {
            failure = nk::BlockSolveFailure::InvalidCandidate;
            break;
        }
        if (!(alpha > 0.0)) break;
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
            b.velocity[i] = static_cast<float>(static_cast<double>(b.velocity[i]) + alpha * b.direction[i]);
        for (uint32_t item = threadIdx.x; item < working; item += blockDim.x) {
            const size_t at = DenseRows(s)[size_t{b.begin} + item];
            const Vec3 move = LoadIncidenceVec3(s, at, kDenseMoveColumn);
            const Vec3 row_step = LoadIncidenceVec3(s, at, kDenseStepColumn);
            StoreIncidenceVec3(s, at, kDenseMoveColumn, {static_cast<float>(move.x + alpha * row_step.x),
                                                        static_cast<float>(move.y + alpha * row_step.y),
                                                        static_cast<float>(move.z + alpha * row_step.z)});
        }
        __syncthreads();
        if (rounded) break;
    }
    __syncthreads();
    return failure;
}

// Leaves the block's final velocity in b.velocity (compact coordinates); returns the first failure.
template <typename Block>
__device__ nk::BlockSolveFailure DescendDenseBlock(const Block& b, DataView data, BlockDescentSolveParams p,
                                                   BlockScratch s) {
    for (uint32_t i = threadIdx.x; i < b.dimension; i += blockDim.x) b.velocity[i] = b.Snapshot(i);
    uint32_t working = ParkDenseRows(b, data, p, s);
    nk::BlockSolveFailure failure = nk::BlockSolveFailure::None;
    do failure = NewtonDenseBlock(b, data, p, s, working);
    while (failure == nk::BlockSolveFailure::None && WakeDenseRows(b, data, p, s, &working));
    return failure;
}

// The largest coordinate change of a block's solve, its momentum residual; a failed solve has the largest.
template <typename Block>
__device__ uint32_t DenseBlockResidual(const Block& b, nk::BlockSolveFailure failure) {
    uint32_t bits = failure == nk::BlockSolveFailure::None ? 0u : ResidualBits(FLT_MAX);
    for (uint32_t i = 0u; i < b.dimension; ++i) bits = max(bits, ResidualBits(b.velocity[i] - b.Snapshot(i)));
    return bits;
}
