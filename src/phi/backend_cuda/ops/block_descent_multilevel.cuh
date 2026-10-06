#pragma once
// Multilevel Galerkin correction of the vertex blocks: levels restrict the momentum residual through their
// weights, upper levels take node block steps and the last a dense solve; a plane search takes the step.

// Nodes one element reaches on a level: the parents of its vertices.
constexpr uint32_t kMultilevelElementNodes = 4u * nk::kVbdCoarseParents;
constexpr uint32_t kMultilevelRowNodes = kMultilevelRowPoints * nk::kVbdCoarseParents;
// Row Jacobian components up to this keep every product of a row's coupling blocks finite.
constexpr float kMultilevelRowJacobian = 65536.0f;
// Count bit of a touch entry whose row group moves more vertex-block particles than the entry holds.
constexpr uint8_t kMultilevelTouchOverflow = 0x80u;
// Largest dense-level rank. The factor kernel takes it in panels of 32 pivot rows; an inverse column keeps
// it in lane-strided register slots, lane x holding rows x + 32k in slot k.
constexpr uint32_t kMultilevelDenseRank = 3u * nk::kVbdCoarseDenseNodes;
constexpr uint32_t kMultilevelFactorWarps = 16u;
constexpr uint32_t kMultilevelFactorPanel = 32u;
constexpr uint32_t kMultilevelFactorColumns = (kMultilevelDenseRank + 31u) / 32u;
// Inverse columns a block solves against its shared copy of the dense factor, one per warp; few warps per
// block leave each column's chain of dependent pivots its own issue slots.
constexpr uint32_t kMultilevelInverseWarps = 4u;
// Trials after the unit step, each at the minimizer of the quadratic through the slope and the last one.
constexpr uint32_t kMultilevelBacktracks = 3u;
// The plane is solved when its curvature determinant keeps this share of the diagonal product.
constexpr double kMultilevelIndependence = 1.0e-6;
// Pivots of the scaled single-precision factor below this are rounding noise.
constexpr float kMultilevelPivot = 1.0e-6f;
constexpr int32_t kMultilevelUnassembled = INT32_MIN;
static_assert(kMultilevelCorrectThreads % 32u == 0u && kMultilevelFactorPanel == 32u &&
              kMultilevelRowPoints < kMultilevelTouchOverflow);

// Upper-triangle entry (i, j), i <= j, of a rank-wide matrix stored by rows.
__device__ inline uint32_t MultilevelRowPacked(uint32_t i, uint32_t j, uint32_t rank) {
    return i * (2u * rank - i + 1u) / 2u + (j - i);
}

// A magnitude for the operator bound; a non-finite value gives infinity.
__device__ inline float MultilevelMagnitude(float value) {
    return fabsf(value) <= FLT_MAX ? fabsf(value) : INFINITY;
}

// One plane-kernel thread's items: the first blocks stride over their chunk of the environment's selected
// rows by index, the others take one vertex, then element, per thread.
template <typename VertexVisit, typename ElementVisit, typename RowVisit>
__device__ inline void VisitPlaneItems(const BlockScratch& s, const VertexBlockLayout& l, uint32_t env,
                                       uint32_t part, VertexVisit vertex, ElementVisit element,
                                       RowVisit row) {
    const uint32_t row_parts = s.multilevel_parts - s.multilevel_item_parts;
    if (part >= row_parts) {
        const uint32_t item = (part - row_parts) * kThreads + threadIdx.x;
        if (item < l.vertices) vertex(item);
        else if (item - l.vertices < l.elements) element(item - l.vertices);
        return;
    }
    uint32_t begin = 0u, end = 0u;
    CoarseRowChunk(s, row_parts, env, part, &begin, &end);
    for (uint32_t item = begin + threadIdx.x; item < end; item += kThreads) row(item);
}

// Plane kernels launch the vertex and element items of every environment before its row groups; parts keep
// their numbering, so the block completing an environment adds the same ordered partials. True for rows.
__device__ inline bool MultilevelPlanePart(const BlockDescentSolveParams& p, const BlockScratch& s,
                                           uint32_t* env, uint32_t* part) {
    const uint32_t row_parts = s.multilevel_parts - s.multilevel_item_parts;
    const uint32_t item_blocks = p.env_count * s.multilevel_item_parts;
    const bool rows = blockIdx.x >= item_blocks;
    *env = rows ? (blockIdx.x - item_blocks) / row_parts : blockIdx.x / s.multilevel_item_parts;
    *part = rows ? (blockIdx.x - item_blocks) % row_parts : row_parts + blockIdx.x % s.multilevel_item_parts;
    return rows;
}

// Counts a finished block of an environment's `parts` after its writes; true in the block completing the
// environment, which resets the count for the next kernel.
__device__ inline bool MultilevelArrive(const BlockScratch& s, uint32_t env, uint32_t parts) {
    __threadfence();
    if (atomicAdd(&s.multilevel[env].done, 1u) + 1u != parts) return false;
    s.multilevel[env].done = 0u;
    __threadfence();
    return true;
}

// One shuffle round of two WarpSumDouble trees: lanes with the offset bit clear add as the tree of `low`
// does, the others as the tree of `high` does, each add keeping its operands and their order.
__device__ inline double MultilevelPairFold(double low, double high, uint32_t offset) {
    const bool upper = (threadIdx.x & offset) != 0u;
    const double other = __shfl_xor_sync(0xffffffffu, upper ? low : high, offset);
    return (upper ? other : low) + (upper ? high : other);
}

// Lane holding plane sum k's warp tree after MultilevelWarpSums<first>.
__host__ __device__ constexpr uint32_t MultilevelSumLane(uint32_t first, uint32_t k) {
    return first == 0u ? (k == 0u ? 0u : k == 1u ? 16u : k == 2u ? 8u : k == 3u ? 24u : 4u)
                       : (k == 2u ? 0u : k == 3u ? 16u : 8u);
}

// The WarpSumDouble trees of plane sums kFirst to 4, two per shuffle round while two remain.
template <uint32_t kFirst>
__device__ inline double MultilevelWarpSums(const double (&v)[kMultilevelSums]) {
    static_assert(kFirst == 0u || kFirst == 2u);
    if constexpr (kFirst == 2u) {
        // Trees of +0.0 in every lane are +0.0.
        if (__all_sync(0xffffffffu, (__double_as_longlong(v[2]) | __double_as_longlong(v[3]) |
                                     __double_as_longlong(v[4])) == 0ll)) return 0.0;
    }
    double tail = v[4] + __shfl_down_sync(0xffffffffu, v[4], 16u);
    double tree;
    if constexpr (kFirst == 0u) {
        tree = MultilevelPairFold(MultilevelPairFold(v[0], v[1], 16u), MultilevelPairFold(v[2], v[3], 16u), 8u);
        tail += __shfl_down_sync(0xffffffffu, tail, 8u);
        tree = MultilevelPairFold(tree, tail, 4u);
    } else {
        tree = MultilevelPairFold(MultilevelPairFold(v[2], v[3], 16u), tail, 8u);
        tree += __shfl_down_sync(0xffffffffu, tree, 4u);
    }
    tree += __shfl_down_sync(0xffffffffu, tree, 2u);
    tree += __shfl_down_sync(0xffffffffu, tree, 1u);
    return tree;
}

// BlockSumsDouble of plane sums kFirst to 4, each warp total added by its own lane of warp 0; sums below
// kFirst are +0.0 in every thread and stay so.
template <uint32_t kFirst>
__device__ inline void MultilevelBlockSums(double (&values)[kMultilevelSums],
                                           double (&shared)[kMultilevelSums][kThreads / 32u]) {
    const double tree = MultilevelWarpSums<kFirst>(values);
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    __syncthreads();
#pragma unroll
    for (uint32_t k = kFirst; k < kMultilevelSums; ++k)
        if (lane == MultilevelSumLane(kFirst, k)) shared[k][warp] = tree;
    __syncthreads();
    if (warp != 0u) return;
    double sum = 0.0;
    if (lane >= kFirst && lane < kMultilevelSums)
        for (uint32_t w = 0u; w < kThreads / 32u; ++w) sum += shared[lane][w];
#pragma unroll
    for (uint32_t k = kFirst; k < kMultilevelSums; ++k) {
        const double total = __shfl_sync(0xffffffffu, sum, k);
        if (lane == 0u) values[k] = total;
    }
}

// An environment's block partials of the first kCount sums, added in a fixed thread-strided order and
// read past the L1 cache since other blocks wrote them; thread 0 gets the sums.
template <uint32_t kCount>
__device__ inline void MultilevelPartialSums(const BlockScratch& s, uint32_t env, double (&sums)[kCount],
                                             double (&shared)[kCount][kThreads / 32u]) {
    for (uint32_t k = 0u; k < kCount; ++k) sums[k] = 0.0;
#pragma unroll 4
    for (uint32_t part = threadIdx.x; part < s.multilevel_parts; part += blockDim.x)
        for (uint32_t k = 0u; k < kCount; ++k)
            sums[k] += __ldcg(s.multilevel_partials +
                              (size_t{env} * s.multilevel_parts + part) * kMultilevelSums + k);
    if constexpr (kCount == kMultilevelSums) MultilevelBlockSums<0u>(sums, shared);
    else BlockSumsDouble(sums, shared);
}

__device__ inline uint32_t MultilevelPacked(uint32_t i, uint32_t j) { return j * (j + 1u) / 2u + i; }

__device__ inline float AxisComponent(Vec3 v, uint32_t axis) {
    return axis == 0u ? v.x : axis == 1u ? v.y : v.z;
}

// Vertices the color sweeps descend; the correction moves only these.
__device__ inline bool MultilevelDynamic(const BlockScratch& s, uint32_t particle) {
    return s.owner_cache_color[particle] != kNoCacheColor;
}

__device__ inline bool IsMultilevelVertex(const BlockDescentSolveParams& p, const BlockScratch& s,
                                          uint32_t particle) {
    const VertexBlockLayout& l = p.vertex_blocks;
    if (particle >= p.total_particle_count) return false;
    const uint32_t local = particle % l.particles_per_env;
    return local >= l.begin && local - l.begin < l.vertices && MultilevelDynamic(s, particle);
}

// Index of a vertex-block particle in the per-vertex arrays.
__device__ inline uint32_t VertexParticleSlot(const BlockDescentSolveParams& p, uint32_t particle) {
    const VertexBlockLayout& l = p.vertex_blocks;
    return particle / l.particles_per_env * l.vertices + particle % l.particles_per_env - l.begin;
}

// The plane step of one vertex at `scale`, rounded explicitly so every kernel forms the same candidate.
__device__ inline Vec3 MultilevelCandidate(const BlockScratch& s, const MultilevelState& state,
                                           uint32_t slot, Vec3 u, float scale) {
    const Vec3 d = s.multilevel_direction[slot], q = s.multilevel_previous[slot];
    const Vec3 step{__fmaf_rn(state.beta, q.x, __fmul_rn(state.alpha, d.x)),
                    __fmaf_rn(state.beta, q.y, __fmul_rn(state.alpha, d.y)),
                    __fmaf_rn(state.beta, q.z, __fmul_rn(state.alpha, d.z))};
    return {nk::vbd::TrialValue(u.x, step.x, scale), nk::vbd::TrialValue(u.y, step.y, scale),
            nk::vbd::TrialValue(u.z, step.z, scale)};
}

// A vertex's rate change under the plane step at `scale`; vertices the sweeps skip keep their rate.
__device__ inline Vec3 MultilevelMove(const DataView& data, const BlockScratch& s,
                                      const MultilevelState& state, uint32_t particle, uint32_t slot,
                                      float scale) {
    if (!MultilevelDynamic(s, particle)) return {};
    const Vec3 u = data.particle_vel[particle];
    return MultilevelCandidate(s, state, slot, u, scale) - u;
}

// Entry of an owner's incidence segment that holds the row group headed by `slot`, or ~0u: a segment
// lists its distinct head slots in ascending order.
__device__ inline uint32_t IncidenceEntry(const BlockScratch& s, uint32_t owner, uint32_t slot) {
    uint32_t first = IncidenceOffset(s.offsets, s.incidence_capacity, owner);
    const uint32_t end = IncidenceOffset(s.offsets, s.incidence_capacity, owner + 1u);
    for (uint32_t last = end; first < last;) {
        const uint32_t middle = first + (last - first) / 2u;
        if (s.incidence[middle] < slot) first = middle + 1u;
        else last = middle;
    }
    return first < end && s.incidence[first] == slot ? first : ~0u;
}

// Frozen row parameters and the current residual/dual form the same local term used by the descent.
__device__ inline LocalTerm LoadMultilevelRowTerm(const BlockScratch& s, uint32_t item,
                                                  const AugmentedRowState& state) {
    const MultilevelRowTerm cached = s.multilevel_row_terms[item];
    LocalTerm term{};
    term.penalty = cached.penalty;
    term.compliance = cached.compliance;
    term.lower = cached.lower;
    term.upper = cached.upper;
    term.mu_first = cached.mu_first;
    term.mu_second = cached.mu_second;
    term.contact = cached.contact;
    term.response = cached.response;
    term.residual = state.residual;
    term.dual = state.dual;
    term.normal_bound = state.normal_bound;
    return term;
}

// Touch entries hold a vertex-array slot and a Jacobian incidence index, in side and term order.
// Row coefficients are fixed during the solve and cached with the selected group.
__global__ void MultilevelTouchKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const auto points = PointMasses(data);
    const uint32_t count = *s.coarse_row_count;
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.coarse_rows[item];
        const NkRow normal = rows[slot];
        const float scale = RowStateScale(data, p, slot);
        s.multilevel_row_terms[item] = {s.penalty[slot], normal.compliance_alpha * scale, normal.lower,
            normal.upper, normal.mu, normal.friction_secondary,
            (normal.flags & nk::nk_row_flags::kBlockNormal) != 0u, s.scalar_response[slot]};
        const NkRowSide sides[2] = {normal.a, normal.b};
        uint32_t touched = 0u;
        bool overflow = false;
        for (uint32_t side = 0u; side < 2u; ++side) {
            if (!PointMassView::IsPointSide(sides[side].kind)) continue;
            for (uint32_t term = 0u; term < points.Count(sides[side]); ++term) {
                const auto entry = points.At(sides[side], term);
                if (entry.kind != kNkSideParticle || !IsMultilevelVertex(p, s, entry.index)) continue;
                bool repeated = false;
                for (uint32_t prior = 0u; prior < term; ++prior) {
                    const auto before = points.At(sides[side], prior);
                    repeated |= before.kind == entry.kind && before.index == entry.index;
                }
                for (uint32_t other = 0u; side == 1u && other < points.Count(sides[0]); ++other) {
                    const auto before = points.At(sides[0], other);
                    repeated |= before.kind == entry.kind && before.index == entry.index;
                }
                if (repeated) continue;
                const uint32_t at = IncidenceEntry(s, entry.index, slot);
                if (touched == kMultilevelRowPoints || at == ~0u) {
                    overflow = true;
                    continue;
                }
                s.multilevel_touch[size_t{touched++} * s.rows + item] = make_uint2(VertexParticleSlot(p, entry.index), at);
            }
        }
        s.multilevel_touch_count[item] =
            static_cast<uint8_t>(touched | (overflow ? kMultilevelTouchOverflow : 0u));
    }
}

// Visits a selected group's vertex-array slots in touch order, with their Jacobian on every axis.
template <typename Visitor>
__device__ inline void VisitRowTouch(const BlockScratch& s, uint32_t item, Visitor visit) {
    const uint32_t count = s.multilevel_touch_count[item] & (kMultilevelTouchOverflow - 1u);
    for (uint32_t k = 0u; k < count; ++k) {
        const uint2 entry = s.multilevel_touch[size_t{k} * s.rows + item];
        const Vec3 jacobian[3] = {LoadPointIncidenceJacobian(s, entry.y, 0u),
                                  LoadPointIncidenceJacobian(s, entry.y, 1u),
                                  LoadPointIncidenceJacobian(s, entry.y, 2u)};
        visit(entry.x, jacobian);
    }
}

// x^T C y with the row-axis curvature (xx, yy, zz, yz) of EvaluateAugmentedResponse.
__device__ inline double RowForm(const float* c, Vec3 x, Vec3 y) {
    return double(c[0]) * x.x * y.x + double(c[1]) * x.y * y.y + double(c[2]) * x.z * y.z +
           double(c[3]) * (double(x.y) * y.z + double(x.z) * y.y);
}

// Coupling of two coarse nodes through a row's curvature, given each node's axis Jacobians.
__device__ inline void RowCoarseBlock(const Vec3* row, const Vec3* column, const float* c, float scale,
                                      float (&block)[3][3]) {
    for (uint32_t i = 0u; i < 3u; ++i)
        for (uint32_t j = 0u; j < 3u; ++j) {
            const float x1 = AxisComponent(row[1], i), x2 = AxisComponent(row[2], i);
            const float y1 = AxisComponent(column[1], j), y2 = AxisComponent(column[2], j);
            block[i][j] = scale * (c[0] * AxisComponent(row[0], i) * AxisComponent(column[0], j) +
                                   c[1] * x1 * y1 + c[2] * x2 * y2 + c[3] * (x1 * y2 + x2 * y1));
        }
}

__device__ inline const nk::vbd::MembraneStartState* ElementMembraneStart(const VertexBlockView& b,
                                                                           uint32_t env, uint32_t element) {
    return b.membrane_start != nullptr ? b.membrane_start + size_t{env} * b.layout.elements + element
                                       : nullptr;
}

// Interval-start geometry of an element, the frame of its Rayleigh metric.
__device__ inline nk::vbd::ElementGeometry ElementStartGeometry(const VertexBlockView& b,
                                                                const nk::VbdElement& element,
                                                                uint32_t env) {
    nk::vbd::ElementGeometry geometry;
    for (uint32_t j = 0u; j < nk::VbdElementVertexCount(element.kind); ++j)
        geometry.start[j] = b.start[b.Particle(env, element.vertex[j])];
    return geometry;
}

// Variations of a unit move along axis dof % 3 of vertex 1 + dof / 3 with vertex 0 held.
__device__ inline nk::vbd::ElementVariation UnitVariation(const nk::VbdElement& e,
                                                          const nk::vbd::ElementCurvature& c,
                                                          uint32_t dof) {
    Vec3 x[4] = {};
    x[1u + dof / 3u] = {dof % 3u == 0u ? 1.0f : 0.0f, dof % 3u == 1u ? 1.0f : 0.0f,
                        dof % 3u == 2u ? 1.0f : 0.0f};
    return nk::vbd::Vary(e, c, x);
}

// Gauss-Newton plus Rayleigh Hessian of one element kind over its vertex moves relative to vertex 0,
// stored as an upper triangle by entry; returns its absolute sum over both triangles.
template <uint32_t kKind>
__device__ inline float StoreKindHessian(nk::VbdElement e, const nk::vbd::ElementCurvature& gn,
                                         const nk::vbd::ElementCurvature& rayleigh, bool damped,
                                         float damping, float* out, size_t stride) {
    constexpr uint32_t kDofs = 3u * (nk::VbdElementVertexCount(kKind) - 1u);
    e.kind = kKind;
    float total = 0.0f;
#pragma unroll
    for (uint32_t q = 0u; q < kDofs; ++q) {
        const nk::vbd::ElementVariation gq = UnitVariation(e, gn, q);
        const nk::vbd::ElementVariation rq =
            damped ? UnitVariation(e, rayleigh, q) : nk::vbd::ElementVariation{};
#pragma unroll
        for (uint32_t i = 0u; i <= q; ++i) {
            float value = nk::vbd::ElementForm(e, gn, UnitVariation(e, gn, i), gq);
            if (damped)
                value += damping * nk::vbd::ElementForm(e, rayleigh, UnitVariation(e, rayleigh, i), rq);
            out[MultilevelPacked(i, q) * stride] = value;
            total += (i == q ? 1.0f : 2.0f) * fabsf(value);
        }
    }
    return total;
}

// An element's relative Hessian into the multilevel scratch. Relative node weights lie in [-1, 1], so the
// returned absolute entry sum bounds every coupling the element gives two coarse nodes.
__device__ inline float StoreElementHessian(const VertexBlockView& b, const DataView& data,
                                            const BlockScratch& s, uint32_t envs, uint32_t env,
                                            uint32_t index) {
    const nk::VbdElement element = b.elements[index];
    const auto* start = ElementMembraneStart(b, env, index);
    const auto gn = nk::vbd::GaussNewtonCurvature(
        element, ElementStepGeometry(b, data.particle_vel, element, env), start);
    const bool damped = element.damping > 0.0f;
    const auto rayleigh = damped
        ? nk::vbd::RayleighCurvature(element, ElementStartGeometry(b, element, env), start)
        : nk::vbd::ElementCurvature{};
    const float damping = element.damping / b.dt;
    float* out = s.multilevel_element_hessian + size_t{env} * b.layout.elements + index;
    const size_t stride = size_t{envs} * b.layout.elements;
    switch (element.kind) {
    case nk::kVbdTriangle:
        return StoreKindHessian<nk::kVbdTriangle>(element, gn, rayleigh, damped, damping, out, stride);
    case nk::kVbdHinge:
        return StoreKindHessian<nk::kVbdHinge>(element, gn, rayleigh, damped, damping, out, stride);
    case nk::kVbdSpring:
        return StoreKindHessian<nk::kVbdSpring>(element, gn, rayleigh, damped, damping, out, stride);
    default:
        return StoreKindHessian<nk::kVbdRodBend>(element, gn, rayleigh, damped, damping, out, stride);
    }
}

// Curvature products of an element between the correction and the previous step, in rate units.
__device__ inline void AddElementForms(const VertexBlockView& b, const DataView& data,
                                       const BlockScratch& s, uint32_t env, uint32_t index,
                                       double (&sums)[kMultilevelSums]) {
    const nk::VbdElement element = b.elements[index];
    const auto* start = ElementMembraneStart(b, env, index);
    Vec3 d[4], q[4];
    for (uint32_t j = 0u; j < nk::VbdElementVertexCount(element.kind); ++j) {
        const uint32_t slot = b.Slot(env, element.vertex[j]);
        d[j] = s.multilevel_direction[slot];
        q[j] = s.multilevel_previous[slot];
    }
    const auto gn = nk::vbd::GaussNewtonCurvature(
        element, ElementStepGeometry(b, data.particle_vel, element, env), start);
    const auto gd = nk::vbd::Vary(element, gn, d), gq = nk::vbd::Vary(element, gn, q);
    const double step = double(b.dt) * b.dt;
    sums[2] += step * nk::vbd::ElementForm(element, gn, gd, gd);
    sums[3] += step * nk::vbd::ElementForm(element, gn, gd, gq);
    sums[4] += step * nk::vbd::ElementForm(element, gn, gq, gq);
    if (!(element.damping > 0.0f)) return;
    const auto rayleigh = nk::vbd::RayleighCurvature(element, ElementStartGeometry(b, element, env), start);
    const auto rd = nk::vbd::Vary(element, rayleigh, d), rq = nk::vbd::Vary(element, rayleigh, q);
    const double damping = double(element.damping) * b.dt;
    sums[2] += damping * nk::vbd::ElementForm(element, rayleigh, rd, rd);
    sums[3] += damping * nk::vbd::ElementForm(element, rayleigh, rd, rq);
    sums[4] += damping * nk::vbd::ElementForm(element, rayleigh, rq, rq);
}

// Elastic and Rayleigh energy change of an element as its vertices take the plane step at `scale`.
__device__ inline float MultilevelElementChange(const VertexBlockView& b, const DataView& data,
                                                const BlockScratch& s, const MultilevelState& state,
                                                uint32_t env, uint32_t index, float scale) {
    const nk::VbdElement element = b.elements[index];
    const auto* start = ElementMembraneStart(b, env, index);
    Vec3 rate[4], move[4], shift[4];
    for (uint32_t j = 0u; j < nk::VbdElementVertexCount(element.kind); ++j) {
        const uint32_t particle = b.Particle(env, element.vertex[j]);
        rate[j] = data.particle_vel[particle];
        move[j] = MultilevelMove(data, s, state, particle, b.Slot(env, element.vertex[j]), scale);
        shift[j] = move[j] * b.dt;
    }
    float change = nk::vbd::ElementMovesEnergyChange(
        element, ElementStepGeometry(b, data.particle_vel, element, env), shift, start);
    if (element.damping > 0.0f) {
        const auto rayleigh = nk::vbd::RayleighCurvature(element, ElementStartGeometry(b, element, env), start);
        const auto vm = nk::vbd::Vary(element, rayleigh, move), vu = nk::vbd::Vary(element, rayleigh, rate);
        change += element.damping * b.dt * (nk::vbd::ElementForm(element, rayleigh, vm, vu) +
                                             0.5f * nk::vbd::ElementForm(element, rayleigh, vm, vm));
    }
    return change;
}

// Four lanes share a selected row group: three evaluate its axes, and the first stores the live state
// and its response without reading the state back from global memory.
__global__ void MultilevelRowsKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint32_t axis = threadIdx.x % kRowAxisLanes;
    const uint32_t mask = ((1u << kRowAxisLanes) - 1u) << (threadIdx.x % 32u - axis);
    const uint32_t count = *s.coarse_row_count;
    for (uint32_t item = (blockIdx.x * blockDim.x + threadIdx.x) / kRowAxisLanes; item < count;
         item += gridDim.x * blockDim.x / kRowAxisLanes) {
        const uint32_t slot = s.coarse_rows[item];
        if (s.multilevel_touch_count[item] == 0u) continue;
        const NkRow normal = rows[slot];
        const bool contact = (normal.flags & nk::nk_row_flags::kBlockNormal) != 0u;
        const float scale = RowStateScale(data, p, slot);
        float residual = 0.0f, dual = 0.0f;
        if (axis < (contact ? 3u : 1u))
            AugmentedRowAxis(data, p, s, normal, slot, axis, scale, false, &residual, &dual);
        AugmentedRowState state{};
        state.residual = {residual, __shfl_sync(mask, residual, 1, kRowAxisLanes),
                                    __shfl_sync(mask, residual, 2, kRowAxisLanes)};
        state.dual = {dual, __shfl_sync(mask, dual, 1, kRowAxisLanes),
                            __shfl_sync(mask, dual, 2, kRowAxisLanes)};
        if (axis != 0u) continue;
        // The bound is the normal term at zero move, so the response reuses it.
        const nk::augmented::ScalarTerm bound = AugmentedRowBoundTerm(s, normal, slot, scale, state);
        state.normal_bound = bound.impulse;
        s.row_state[slot] = state;
        Vec3 impulse{};
        SymmetricMat3 curvature{};
        EvaluateAugmentedResponse(LoadMultilevelRowTerm(s, item, state), bound, {}, &impulse, &curvature);
        s.multilevel_row_impulse[slot] = impulse;
        float* stored = s.multilevel_row_curvature + 4u * size_t{slot};
        stored[0] = curvature.xx;
        stored[1] = curvature.yy;
        stored[2] = curvature.zz;
        stored[3] = curvature.yz;
    }
}

// Slots for every local vertex of each element, indexed by its packed incidence.
constexpr uint32_t kMultilevelElementSlots = nk::PackVbdIncidence(1u, 0u);

// Element gradients for every incidence at the current rates; the force gathers them in its original
// lane-strided order, including the complete Rayleigh action.
__global__ void MultilevelElementGradientKernel(ModelView model, DataView data,
                                                BlockDescentSolveParams p, BlockScratch s) {
    const VertexBlockView b = Vertices(model, data, p);
    const uint64_t slots = uint64_t{kMultilevelElementSlots} * b.layout.elements;
    for (uint64_t item = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x; item < slots * p.env_count;
         item += uint64_t{gridDim.x} * blockDim.x) {
        const uint32_t env = static_cast<uint32_t>(item / slots), packed = static_cast<uint32_t>(item % slots);
        const nk::VbdElement element = b.elements[nk::VbdIncidenceElement(packed)];
        const uint32_t local = nk::VbdIncidenceLocal(packed);
        if (local >= nk::VbdElementVertexCount(element.kind)) continue;
        const uint32_t particle = b.Particle(env, element.vertex[local]);
        Vec3 gradient;
        SymmetricMat3 curvature;
        VertexElementBlock(b, data.particle_vel, env, data.particle_vel[particle], packed, gradient, curvature);
        s.multilevel_element_gradient[item] = gradient;
    }
}

// Vertices of a momentum-residual tile: warps sum the incidences of its vertices in turn, then one thread per
// vertex adds the inertial term.
constexpr uint32_t kMultilevelForceTile = 8u;

// Momentum residual of every vertex at the current rates, one warp per vertex sum; skipped vertices get none.
// Ten blocks per SM bound the registers so one wave holds more tiles.
__global__ void __launch_bounds__(kThreads, 10) MultilevelForceKernel(ModelView model, DataView data,
                                                                      BlockDescentSolveParams p, BlockScratch s) {
    __shared__ Vec3 gradients[kMultilevelForceTile], impulses[kMultilevelForceTile];
    __shared__ bool moving[kMultilevelForceTile];
    const uint32_t lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
    const VertexBlockView b = Vertices(model, data, p);
    const uint32_t items = b.layout.vertices * p.env_count;
    for (uint32_t tile = blockIdx.x * kMultilevelForceTile; tile < items; tile += gridDim.x * kMultilevelForceTile) {
        const uint32_t owned = tile + threadIdx.x;
        const bool owner = threadIdx.x < kMultilevelForceTile && owned < items;
        for (uint32_t local = warp; local < kMultilevelForceTile && tile + local < items; local += kThreads / 32u) {
            const uint32_t item = tile + local;
            const uint32_t env = item / b.layout.vertices, vertex = item % b.layout.vertices;
            const uint32_t particle = b.Particle(env, vertex);
            const bool dynamic = MultilevelDynamic(s, particle);
            if (lane == 0u) moving[local] = dynamic;
            if (!dynamic) continue;
            Vec3 gradient{}, impulse{};
            const Vec3* terms = s.multilevel_element_gradient +
                uint64_t{env} * kMultilevelElementSlots * b.layout.elements;
            const uint32_t begin = IncidenceOffset(s.offsets, s.incidence_capacity, particle);
            const uint32_t end = IncidenceOffset(s.offsets, s.incidence_capacity, particle + 1u);
            for (uint32_t at = b.offsets[vertex] + lane; at < b.offsets[vertex + 1u]; at += warpSize)
                gradient = gradient + terms[b.incidence[at]];
            for (uint32_t at = begin + lane; at < end; at += warpSize) {
                const Vec3 f = s.multilevel_row_impulse[s.incidence[at]];
                impulse += LoadPointIncidenceJacobian(s, at, 0u) * f.x + LoadPointIncidenceJacobian(s, at, 1u) * f.y +
                           LoadPointIncidenceJacobian(s, at, 2u) * f.z;
            }
            gradient = {WarpSum(gradient.x), WarpSum(gradient.y), WarpSum(gradient.z)};
            impulse = {WarpSum(impulse.x), WarpSum(impulse.y), WarpSum(impulse.z)};
            if (lane != 0u) continue;
            gradients[local] = gradient;
            impulses[local] = impulse;
        }
        __syncthreads();
        if (owner) {
            const uint32_t particle = b.Particle(owned / b.layout.vertices, owned % b.layout.vertices);
            Vec3 force{};
            if (moving[threadIdx.x]) {
                const Vec3 gradient = gradients[threadIdx.x], impulse = impulses[threadIdx.x];
                force = -nk::vbd::InertialGradient(data.particle_vel[particle], b.free_rate[owned], b.inertia[owned],
                                                   p.dt) - gradient + impulse / p.dt;
                if (!Finite(force))
                    RecordBlockFailure(data, p, s, particle, nk::BlockSolveFailure::InvalidEquation);
            }
            s.multilevel_force[owned] = Finite(force) ? force : Vec3{};
        }
        __syncthreads();
    }
}

// The node of an element parent slot, selected without indexing so the slots stay in registers.
__device__ inline uint32_t SlotNode(const uint32_t (&node)[kMultilevelElementNodes], uint32_t slot) {
    uint32_t value = ~0u;
#pragma unroll
    for (uint32_t k = 0u; k < kMultilevelElementNodes; ++k) value = k == slot ? node[k] : value;
    return value;
}

// A node's weights at element vertices 1 to 3 less its weight at vertex 0, from the parent slots.
__device__ inline void RelativeWeights(const uint32_t (&node)[kMultilevelElementNodes],
                                       const float (&weight)[kMultilevelElementNodes], uint32_t target,
                                       float (&relative)[3]) {
    float at[4];
#pragma unroll
    for (uint32_t j = 0u; j < 4u; ++j) {
        at[j] = 0.0f;
#pragma unroll
        for (uint32_t k = 0u; k < nk::kVbdCoarseParents; ++k)
            if (node[nk::kVbdCoarseParents * j + k] == target) at[j] = weight[nk::kVbdCoarseParents * j + k];
    }
#pragma unroll
    for (uint32_t j = 0u; j < 3u; ++j) relative[j] = at[j + 1u] - at[0];
}

// Coupling block of two nodes through an element's relative Hessian, given their relative weights.
__device__ inline void ElementNodeBlock(const float (&h)[kMultilevelElementHessian], const float (&row)[3],
                                        const float (&column)[3], float (&block)[3][3]) {
#pragma unroll
    for (uint32_t i = 0u; i < 3u; ++i)
#pragma unroll
        for (uint32_t k = 0u; k < 3u; ++k) {
            float value = 0.0f;
#pragma unroll
            for (uint32_t j = 0u; j < 3u; ++j)
#pragma unroll
                for (uint32_t m = 0u; m < 3u; ++m) {
                    const uint32_t a = 3u * j + i, c = 3u * m + k;
                    value = fmaf(row[j] * column[m],
                                 h[a <= c ? MultilevelPacked(a, c) : MultilevelPacked(c, a)], value);
                }
            block[i][k] = value;
        }
}

// A row group's moving vertex-block particles with their axis Jacobians.
struct RowTouch {
    uint32_t count;
    uint32_t vertex[kMultilevelRowPoints];
    Vec3 jacobian[kMultilevelRowPoints][3];
};

// Adds node couplings to an environment's operator in fixed point: a symmetric block per node above the
// dense level, the packed upper triangle on it. A non-finite coupling fails the environment.
struct OperatorSink {
    unsigned long long* entries;
    uint32_t* status;
    uint32_t dense_first;
    float low, high;

    // The scale 2^exponent applies as two float powers of two: the product is exact wherever the double one
    // rounds to a nonzero integer, so both conversions agree. Conversions precede their atomics.
    template <uint32_t kCount>
    __device__ void Add(const size_t (&at)[kCount], const float (&value)[kCount]) const {
        long long fixed[kCount];
        bool failed = false;
#pragma unroll
        for (uint32_t k = 0u; k < kCount; ++k) {
            failed |= !(fabsf(value[k]) <= FLT_MAX);
            fixed[k] = __float2ll_rn(__fmul_rn(__fmul_rn(value[k], low), high));
        }
        if (failed) atomicOr(status, kEnvStatusSolverFailure);
#pragma unroll
        for (uint32_t k = 0u; k < kCount; ++k)
            if (fixed[k] != 0ll && fabsf(value[k]) <= FLT_MAX)
                atomicAdd(entries + at[k], static_cast<unsigned long long>(fixed[k]));
    }

    __device__ void AddBlock(uint32_t row, uint32_t column, const float (&block)[3][3]) const {
        if (row < dense_first) {
            const size_t at[6] = {6u * size_t{row}, 6u * size_t{row} + 1u, 6u * size_t{row} + 2u,
                                  6u * size_t{row} + 3u, 6u * size_t{row} + 4u, 6u * size_t{row} + 5u};
            const float value[6] = {block[0][0], block[1][1], block[2][2], block[0][1], block[0][2], block[1][2]};
            Add(at, value);
            return;
        }
        const bool swap = row > column;
        const uint32_t a = (swap ? column : row) - dense_first, c = (swap ? row : column) - dense_first;
        size_t at[9];
        float value[9];
#pragma unroll
        for (uint32_t i = 0u; i < 3u; ++i)
#pragma unroll
            for (uint32_t j = 0u; j < 3u; ++j) {
                const bool upper = 3u * a + i <= 3u * c + j;
                at[3u * i + j] = upper ? 6u * size_t{dense_first} + MultilevelPacked(3u * a + i, 3u * c + j) : 0u;
                value[3u * i + j] = upper ? (swap ? block[j][i] : block[i][j]) : 0.0f;
            }
        Add(at, value);
    }
};

// 2^k as a float for k within the normal exponent range.
__device__ inline float MultilevelPowerOfTwo(int32_t k) { return __int_as_float((127 + k) << 23); }

// One vertex's inertia on one level through its parent weights.
__device__ inline void AssembleVertex(const ModelView& model, const BlockScratch& s, const VertexBlockView& b,
                                      const OperatorSink& sink, uint32_t env, uint32_t vertex,
                                      uint32_t level, bool pairs) {
    if (!MultilevelDynamic(s, b.Particle(env, vertex))) return;
    const float inertia = b.inertia[b.Slot(env, vertex)];
    const size_t at = (size_t{level} * b.layout.vertices + vertex) * nk::kVbdCoarseParents;
    for (uint32_t a = 0u; a < nk::kVbdCoarseParents; ++a)
        for (uint32_t c = a; c < (pairs ? nk::kVbdCoarseParents : a + 1u); ++c) {
            const uint32_t row = model.vbd_coarse_parents[at + a];
            const uint32_t column = model.vbd_coarse_parents[at + c];
            if (row == ~0u || column == ~0u) continue;
            const float value = inertia * model.vbd_coarse_weights[at + a] * model.vbd_coarse_weights[at + c];
            const float block[3][3] = {{value, 0.0f, 0.0f}, {0.0f, value, 0.0f}, {0.0f, 0.0f, value}};
            sink.AddBlock(row, column, block);
        }
}

// One element's couplings on every level from its relative Hessian: the block of each distinct node it
// reaches, or of every pair of them on the dense level. Vertices the sweeps skip carry no weight.
__device__ inline void AssembleElement(const ModelView& model, const BlockScratch& s,
                                       const VertexBlockView& b, const OperatorSink& sink, uint32_t envs,
                                       uint32_t env, uint32_t index, bool dense) {
    const nk::VbdElement element = b.elements[index];
    const uint32_t count = nk::VbdElementVertexCount(element.kind);
    bool live[4], any = false;
#pragma unroll
    for (uint32_t j = 0u; j < 4u; ++j) {
        live[j] = j < count && MultilevelDynamic(s, b.Particle(env, element.vertex[j]));
        any = any || live[j];
    }
    if (!any) return;
    const uint32_t dofs = 3u * (count - 1u);
    const float* in = s.multilevel_element_hessian + size_t{env} * b.layout.elements + index;
    const size_t stride = size_t{envs} * b.layout.elements;
    float h[kMultilevelElementHessian];
#pragma unroll
    for (uint32_t k = 0u; k < kMultilevelElementHessian; ++k)
        h[k] = k < dofs * (dofs + 1u) / 2u ? in[k * stride] : 0.0f;
    for (uint32_t level = 0u; level < b.layout.coarse_levels; ++level) {
        uint32_t node[kMultilevelElementNodes];
        float weight[kMultilevelElementNodes];
#pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const size_t at = (size_t{level} * b.layout.vertices + (live[j] ? element.vertex[j] : 0u)) *
                              nk::kVbdCoarseParents;
#pragma unroll
            for (uint32_t k = 0u; k < nk::kVbdCoarseParents; ++k) {
                node[nk::kVbdCoarseParents * j + k] = live[j] ? model.vbd_coarse_parents[at + k] : ~0u;
                weight[nk::kVbdCoarseParents * j + k] = live[j] ? model.vbd_coarse_weights[at + k] : 0.0f;
            }
        }
        uint32_t firsts = 0u;
#pragma unroll
        for (uint32_t a = 0u; a < kMultilevelElementNodes; ++a) {
            bool first = node[a] != ~0u;
#pragma unroll
            for (uint32_t c = 0u; c < a; ++c) first = first && node[c] != node[a];
            if (first) firsts |= 1u << a;
        }
        const bool pairs = dense && level + 1u == b.layout.coarse_levels;
        for (uint32_t rest = firsts; rest != 0u; rest &= rest - 1u) {
            const uint32_t row = SlotNode(node, __ffs(rest) - 1u);
            float a[3], block[3][3];
            RelativeWeights(node, weight, row, a);
            if (!pairs) {
                ElementNodeBlock(h, a, a, block);
                sink.AddBlock(row, row, block);
                continue;
            }
            for (uint32_t more = rest; more != 0u; more &= more - 1u) {
                const uint32_t column = SlotNode(node, __ffs(more) - 1u);
                float c[3];
                RelativeWeights(node, weight, column, c);
                ElementNodeBlock(h, a, c, block);
                sink.AddBlock(row, column, block);
            }
        }
    }
}

// A power-of-two lane group shares a row's touch Jacobians, parent topology and coarse Jacobians within one
// warp, one lane loading each touch entry.
constexpr uint32_t kMultilevelRowNodeLanes = 8u;
static_assert(32u % kMultilevelRowNodeLanes == 0u && kMultilevelRowNodes <= 32u &&
              kMultilevelRowPoints <= kMultilevelRowNodeLanes);
struct MultilevelRowNodeTile {
    Vec3 touch[kMultilevelRowPoints][3];
    uint32_t parent[kMultilevelRowNodes];
    float weight[kMultilevelRowNodes];
    uint32_t node[kMultilevelRowNodes];
    Vec3 jacobian[kMultilevelRowNodes][3];
};
// Resident assembly blocks per multiprocessor, as many as their shared row tiles allow.
constexpr uint32_t kMultilevelAssembleBlocks = 4u;

// Lanes build distinct nodes in parent order on every level, then retain the original touch order in each
// axis sum. Loads independent of the touch count issue first, and each level reads the next one's parents.
__device__ inline void AssembleRow(const ModelView& model, const DataView& data,
                                   const BlockDescentSolveParams& p, const BlockScratch& s,
                                   const OperatorSink& sink, MultilevelRowNodeTile& tile,
                                   uint32_t env, uint32_t item, bool dense) {
    const uint32_t lane = threadIdx.x % kMultilevelRowNodeLanes;
    const uint32_t mask = ((1u << kMultilevelRowNodeLanes) - 1u) << (threadIdx.x % 32u - lane);
    const VertexBlockLayout& l = p.vertex_blocks;
    const uint32_t touched = s.multilevel_touch_count[item];
    const uint2 entry = s.multilevel_touch[size_t{lane} * s.rows + item];
    const float* stored = s.multilevel_row_curvature + 4u * size_t{s.coarse_rows[item]};
    const float curvature[4] = {stored[0], stored[1], stored[2], stored[3]};
    if ((touched & kMultilevelTouchOverflow) != 0u && lane == 0u)
        atomicOr(data.env_status + env, kEnvStatusSolverFailure);
    if (touched == 0u || (touched & kMultilevelTouchOverflow) != 0u) return;
    const float scale = 1.0f / (p.dt * p.dt);
    const bool curved = !(curvature[0] == 0.0f && curvature[1] == 0.0f && curvature[2] == 0.0f &&
                          curvature[3] == 0.0f);
    const uint32_t vertex = lane < touched ? entry.x - env * l.vertices : 0u;
    uint32_t parent[nk::kVbdCoarseParents] = {};
    float weight[nk::kVbdCoarseParents] = {};
    const auto load_parents = [&](uint32_t level) {
        const size_t at = (size_t{level} * l.vertices + vertex) * nk::kVbdCoarseParents;
        for (uint32_t k = 0u; k < nk::kVbdCoarseParents; ++k) {
            parent[k] = model.vbd_coarse_parents[at + k];
            weight[k] = model.vbd_coarse_weights[at + k];
        }
    };
    bool bounded = true;
    if (lane < touched) {
        // A curved row assembles, so its first parents load beside its Jacobians.
        if (curved) load_parents(0u);
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            const Vec3 jacobian = LoadPointIncidenceJacobian(s, entry.y, axis);
            tile.touch[lane][axis] = jacobian;
            bounded &= fabsf(jacobian.x) <= kMultilevelRowJacobian && fabsf(jacobian.y) <= kMultilevelRowJacobian &&
                       fabsf(jacobian.z) <= kMultilevelRowJacobian;
        }
    }
    // Without curvature the row adds only zero couplings while its Jacobians keep every product finite.
    const bool vanish = __all_sync(mask, bounded) && !curved && scale <= FLT_MAX;
    __syncwarp(mask);
    if (vanish) return;
    if (lane < touched && !curved) load_parents(0u);
    const uint32_t parents = touched * nk::kVbdCoarseParents;
    for (uint32_t level = 0u; level < l.coarse_levels; ++level) {
        if (lane < touched) {
            for (uint32_t k = 0u; k < nk::kVbdCoarseParents; ++k) {
                tile.parent[nk::kVbdCoarseParents * lane + k] = parent[k];
                tile.weight[nk::kVbdCoarseParents * lane + k] = weight[k];
            }
            if (level + 1u < l.coarse_levels) load_parents(level + 1u);
        }
        __syncwarp(mask);
        uint32_t firsts = 0u;
        for (uint32_t i = lane; i < parents; i += kMultilevelRowNodeLanes) {
            const uint32_t node = tile.parent[i];
            bool first = node != ~0u;
            for (uint32_t prior = 0u; prior < i; ++prior) first &= tile.parent[prior] != node;
            if (first) firsts |= 1u << i;
        }
        for (uint32_t offset = kMultilevelRowNodeLanes / 2u; offset > 0u; offset /= 2u)
            firsts |= __shfl_xor_sync(mask, firsts, offset, kMultilevelRowNodeLanes);
        const uint32_t count = __popc(firsts);
        for (uint32_t i = lane; i < count; i += kMultilevelRowNodeLanes) {
            uint32_t rest = firsts;
            for (uint32_t prior = 0u; prior < i; ++prior) rest &= rest - 1u;
            const uint32_t node = tile.parent[__ffs(rest) - 1u];
            Vec3 coarse[3] = {};
            for (uint32_t touch = 0u; touch < touched; ++touch)
                for (uint32_t k = 0u; k < nk::kVbdCoarseParents; ++k) {
                    const uint32_t at = touch * nk::kVbdCoarseParents + k;
                    if (tile.parent[at] == node)
                        for (uint32_t axis = 0u; axis < 3u; ++axis)
                            coarse[axis] += tile.touch[touch][axis] * tile.weight[at];
                }
            tile.node[i] = node;
            for (uint32_t axis = 0u; axis < 3u; ++axis) tile.jacobian[i][axis] = coarse[axis];
        }
        __syncwarp(mask);
        const bool pairs = dense && level + 1u == l.coarse_levels;
        for (uint32_t row = lane; row < count; row += kMultilevelRowNodeLanes) {
            if (!pairs) {
                float block[3][3];
                RowCoarseBlock(tile.jacobian[row], tile.jacobian[row], stored, scale, block);
                sink.AddBlock(tile.node[row], tile.node[row], block);
            } else {
                for (uint32_t column = row; column < count; ++column) {
                    float block[3][3];
                    RowCoarseBlock(tile.jacobian[row], tile.jacobian[column], stored, scale, block);
                    sink.AddBlock(tile.node[row], tile.node[column], block);
                }
            }
        }
        __syncwarp(mask);
    }
}

// Element relative Hessians for the assembly and a bound on every coupling it adds: an element's absolute
// Hessian sum, a vertex's inertia, and a row's curvature on its summed absolute Jacobian components.
__global__ void MultilevelBoundKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                                      BlockScratch s) {
    uint32_t env = 0u, part = 0u;
    MultilevelPlanePart(p, s, &env, &part);
    const VertexBlockView b = Vertices(model, data, p);
    float bound = 0.0f;
    VisitPlaneItems(s, b.layout, env, part,
        [&](uint32_t vertex) {
            if (MultilevelDynamic(s, b.Particle(env, vertex)))
                bound = fmaxf(bound, MultilevelMagnitude(b.inertia[b.Slot(env, vertex)]));
        },
        [&](uint32_t index) {
            bound = fmaxf(bound,
                          MultilevelMagnitude(StoreElementHessian(b, data, s, p.env_count, env, index)));
        },
        [&](uint32_t item) {
            float reach[3] = {};
            bool moved = false;
            VisitRowTouch(s, item, [&](uint32_t, const Vec3* jacobian) {
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                    reach[axis] += fmaxf(fabsf(jacobian[axis].x),
                                         fmaxf(fabsf(jacobian[axis].y), fabsf(jacobian[axis].z)));
                moved = true;
            });
            if (!moved) return;
            const float* c = s.multilevel_row_curvature + 4u * size_t{s.coarse_rows[item]};
            const float coupling = fabsf(c[0]) * reach[0] * reach[0] + fabsf(c[1]) * reach[1] * reach[1] +
                                   fabsf(c[2]) * reach[2] * reach[2] + 2.0f * fabsf(c[3]) * reach[1] * reach[2];
            bound = fmaxf(bound, MultilevelMagnitude(coupling / (p.dt * p.dt)));
        });
    for (uint32_t offset = 16u; offset > 0u; offset /= 2u)
        bound = fmaxf(bound, __shfl_down_sync(0xffffffffu, bound, offset));
    if (threadIdx.x % 32u == 0u) atomicMax(&s.multilevel[env].bound, __float_as_uint(bound));
}

// Vertex inertia and element Hessians take the first blocks of the launch, one item per thread; row groups take
// the rest, a block per row part and level of an environment assembling its share of rows on every level.
__global__ void __launch_bounds__(kThreads, kMultilevelAssembleBlocks)
MultilevelAssembleKernel(ModelView model, DataView data, BlockDescentSolveParams p, BlockScratch s) {
    __shared__ MultilevelRowNodeTile tiles[kThreads / kMultilevelRowNodeLanes];
    const VertexBlockView b = Vertices(model, data, p);
    const VertexBlockLayout& l = b.layout;
    const uint32_t row_parts = (s.multilevel_parts - s.multilevel_item_parts) * l.coarse_levels;
    const uint32_t item_blocks = p.env_count * s.multilevel_item_parts;
    const bool rows = blockIdx.x >= item_blocks;
    const uint32_t block = rows ? blockIdx.x - item_blocks : blockIdx.x;
    const uint32_t env = rows ? block / row_parts : block / s.multilevel_item_parts;
    const int32_t exponent = s.multilevel[env].exponent;
    if (exponent == kMultilevelUnassembled) return;
    const OperatorSink sink{s.multilevel_operator + size_t{env} * MultilevelOperatorEntries(l),
                            data.env_status + env, l.coarse_nodes - l.coarse_dense_nodes,
                            MultilevelPowerOfTwo(exponent / 2), MultilevelPowerOfTwo(exponent - exponent / 2)};
    const bool dense = l.coarse_dense_nodes > 0u;
    if (rows) {
        uint32_t begin = 0u, end = 0u;
        CoarseRowChunk(s, row_parts, env, block % row_parts, &begin, &end);
        const uint32_t group_index = threadIdx.x / kMultilevelRowNodeLanes;
        for (uint32_t item = begin + group_index; item < end; item += blockDim.x / kMultilevelRowNodeLanes)
            AssembleRow(model, data, p, s, sink, tiles[group_index], env, item, dense);
        return;
    }
    const uint32_t item = block % s.multilevel_item_parts * kThreads + threadIdx.x;
    if (item < l.vertices) {
        for (uint32_t level = 0u; level < l.coarse_levels; ++level)
            AssembleVertex(model, s, b, sink, env, item, level, dense && level + 1u == l.coarse_levels);
    } else if (item - l.vertices < l.elements) {
        AssembleElement(model, s, b, sink, p.env_count, env, item - l.vertices, dense);
    }
}

// The fixed-point scale keeps every operator sum of an environment below 2^62.
__global__ void MultilevelScaleKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const VertexBlockLayout& l = p.vertex_blocks;
    const double terms = double(l.elements) + l.vertices + p.rows_per_env + 1.0;
    for (uint32_t env = blockIdx.x * blockDim.x + threadIdx.x; env < p.env_count;
         env += gridDim.x * blockDim.x) {
        MultilevelState& state = s.multilevel[env];
        const double total = double(__uint_as_float(state.bound)) * terms;
        if (!(total <= DBL_MAX)) {
            state.exponent = kMultilevelUnassembled;
            atomicOr(data.env_status + env, kEnvStatusSolverFailure);
            continue;
        }
        int exponent = 0;
        frexp(total, &exponent);
        state.exponent = 62 - exponent;
    }
}

// Inverts each node block above the dense level; a block that is not positive definite takes no step. Items
// past them give the dense level's rows their Jacobi scales in single precision, kept after its factor.
__global__ void MultilevelJacobiFactorKernel(BlockDescentSolveParams p, BlockScratch s) {
    const VertexBlockLayout& l = p.vertex_blocks;
    const uint32_t jacobi = l.coarse_nodes - l.coarse_dense_nodes, rank = 3u * l.coarse_dense_nodes;
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < (jacobi + rank) * p.env_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t env = item / (jacobi + rank), node = item % (jacobi + rank);
        const int32_t exponent = s.multilevel[env].exponent;
        const unsigned long long* entries = s.multilevel_operator + size_t{env} * MultilevelOperatorEntries(l);
        if (node >= jacobi) {
            if (exponent == kMultilevelUnassembled) continue;
            const uint32_t i = node - jacobi;
            const double diagonal = ldexp(
                double(static_cast<long long>(entries[6u * size_t{jacobi} + MultilevelPacked(i, i)])), -exponent);
            const double scale = diagonal > 0.0 ? 1.0 / sqrt(diagonal) : 0.0;
            float* scales = s.multilevel_dense_factor + size_t{env} * MultilevelDenseFactorWords(rank) +
                            rank * (rank + 1u) / 2u;
            scales[i] = scale <= double(FLT_MAX) ? static_cast<float>(scale) : 0.0f;
            continue;
        }
        SymmetricMat3 inverse{};
        if (exponent != kMultilevelUnassembled) {
            float values[6];
            for (uint32_t k = 0u; k < 6u; ++k)
                values[k] = static_cast<float>(
                    ldexp(double(static_cast<long long>(entries[6u * size_t{node} + k])), -exponent));
            nk::vbd::Invert({values[0], values[1], values[2], values[3], values[4], values[5]}, 0.0f, &inverse);
        }
        s.multilevel_block_inverse[size_t{env} * jacobi + node] = inverse;
    }
}

// The Jacobi-scaled dense level in single precision, one packed entry per thread. A node without moving
// children keeps a unit row, so its step stays zero.
__global__ void MultilevelDenseScaleKernel(BlockDescentSolveParams p, BlockScratch s) {
    const VertexBlockLayout& l = p.vertex_blocks;
    const uint32_t first = l.coarse_nodes - l.coarse_dense_nodes, rank = 3u * l.coarse_dense_nodes;
    const uint32_t packed = rank * (rank + 1u) / 2u;
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < packed * p.env_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t env = item / packed, k = item % packed;
        const int32_t exponent = s.multilevel[env].exponent;
        if (exponent == kMultilevelUnassembled) continue;
        uint32_t j = static_cast<uint32_t>((sqrtf(8.0f * k + 1.0f) - 1.0f) * 0.5f);
        while (j * (j + 1u) / 2u > k) --j;
        while ((j + 1u) * (j + 2u) / 2u <= k) ++j;
        const uint32_t i = k - j * (j + 1u) / 2u;
        float* factor = s.multilevel_dense_factor + size_t{env} * MultilevelDenseFactorWords(rank);
        const unsigned long long entry =
            s.multilevel_operator[size_t{env} * MultilevelOperatorEntries(l) + 6u * size_t{first} + k];
        const double value = ldexp(double(static_cast<long long>(entry)), -exponent);
        factor[k] = i == j ? 1.0f
                           : static_cast<float>(value * double(factor[packed + i]) * double(factor[packed + j]));
    }
}

// Shared bytes of the dense factor kernel: the scaled level, then 16-byte aligned the rows of one pivot panel
// over its trailing columns.
__host__ __device__ inline uint32_t MultilevelFactorPanelStride(uint32_t rank) {
    return rank > kMultilevelFactorPanel ? (rank - kMultilevelFactorPanel + 3u) / 4u * 4u : 0u;
}
__host__ __device__ inline size_t MultilevelFactorPanelOffset(uint32_t rank) {
    return (size_t{rank} * (rank + 1u) / 2u * sizeof(float) + 15u) / 16u * 16u;
}
__host__ __device__ inline size_t MultilevelFactorSharedBytes(uint32_t rank) {
    return MultilevelFactorPanelOffset(rank) +
           size_t{kMultilevelFactorPanel} * MultilevelFactorPanelStride(rank) * sizeof(float);
}

// Copies `count` words into shared memory with several loads in flight per thread.
__device__ inline void MultilevelLoadShared(float* to, const float* from, uint32_t count) {
    constexpr uint32_t kBatch = 8u;
    for (uint32_t base = threadIdx.x; base < count; base += kBatch * blockDim.x) {
        float value[kBatch];
#pragma unroll
        for (uint32_t b = 0u; b < kBatch; ++b) {
            const uint32_t k = base + b * blockDim.x;
            value[b] = k < count ? from[k] : 0.0f;
        }
#pragma unroll
        for (uint32_t b = 0u; b < kBatch; ++b)
            if (base + b * blockDim.x < count) to[base + b * blockDim.x] = value[b];
    }
}

// Copies words [first, count) into shared memory as thread `thread` of `threads`, several loads in flight.
__device__ inline void MultilevelLoadShared(float* to, const float* from, uint32_t first, uint32_t count,
                                            uint32_t thread, uint32_t threads) {
    constexpr uint32_t kBatch = 8u;
    for (uint32_t base = first + thread; base < count; base += kBatch * threads) {
        float value[kBatch];
#pragma unroll
        for (uint32_t b = 0u; b < kBatch; ++b) {
            const uint32_t k = base + b * threads;
            value[b] = k < count ? from[k] : 0.0f;
        }
#pragma unroll
        for (uint32_t b = 0u; b < kBatch; ++b)
            if (base + b * threads < count) to[base + b * threads] = value[b];
    }
}

// Named barrier 1 of the factor block: warp 0 arrives once its panel columns are written and the other
// warps wait there before reading them.
__device__ __forceinline__ void MultilevelPanelArrive() {
    __syncwarp();
    asm volatile("bar.arrive 1, %0;" ::"r"(32u * kMultilevelFactorWarps) : "memory");
}
__device__ __forceinline__ void MultilevelPanelWait() {
    __syncwarp();
    asm volatile("bar.sync 1, %0;" ::"r"(32u * kMultilevelFactorWarps) : "memory");
}

// Updates of rows kRow and below of a register column by pivot row kStep, `row` holding the pivot row in
// groups of four columns; rows below a diagonal-block lane's own column are scratch.
template <uint32_t kStep, uint32_t kRow>
__device__ __forceinline__ void MultilevelDiagonalRows(float (&d)[kMultilevelFactorPanel],
                                                       const float4 (&row)[kMultilevelFactorPanel / 4u]) {
    if constexpr (kRow < kMultilevelFactorPanel) {
        const float4 quad = row[kRow / 4u];
        const float along = kRow % 4u == 0u ? quad.x : kRow % 4u == 1u ? quad.y : kRow % 4u == 2u ? quad.z : quad.w;
        d[kRow] = fmaf(-along, d[kStep], d[kRow]);
        MultilevelDiagonalRows<kStep, kRow + 1u>(d, row);
    }
}

// Groups kQuad and later of four pivot-row columns.
template <uint32_t kQuad>
__device__ __forceinline__ void MultilevelDiagonalRow(float4 (&row)[kMultilevelFactorPanel / 4u],
                                                      const float4* shared) {
    if constexpr (kQuad < kMultilevelFactorPanel / 4u) {
        row[kQuad] = shared[kQuad];
        MultilevelDiagonalRow<kQuad + 1u>(row, shared);
    }
}

// Pivot kStep and the later ones of a diagonal block `width` wide: every lane roots the broadcast pivot, lanes
// past it divide; the pivot row goes to row kStep of `rows` and updates all rows below, the next pivot first.
template <uint32_t kStep>
__device__ __forceinline__ void MultilevelDiagonalPivots(float (&d)[kMultilevelFactorPanel], float* rows,
                                                         uint32_t width, uint32_t lane, float pivot,
                                                         bool* failed) {
    constexpr uint32_t kPanel = kMultilevelFactorPanel;
    if constexpr (kStep < kPanel) {
        if (kStep < width) {
            const float root = sqrtf(pivot);
            *failed |= !(pivot > kMultilevelPivot);
            const float divided = math::Quotient(lane > kStep ? d[kStep] : 0.0f, root);
            d[kStep] = lane == kStep ? root : divided;
            float next = 0.0f;
            if constexpr (kStep + 1u < kPanel)
                next = __shfl_sync(0xffffffffu, fmaf(-d[kStep], d[kStep], d[kStep + 1u]), kStep + 1u);
            rows[kStep * kPanel + lane] = d[kStep];
            __syncwarp();
            float4 row[kPanel / 4u];
            MultilevelDiagonalRow<(kStep + 1u) / 4u>(row, reinterpret_cast<const float4*>(rows + kStep * kPanel));
            MultilevelDiagonalRows<kStep, kStep + 1u>(d, row);
            MultilevelDiagonalPivots<kStep + 1u>(d, rows, width, lane, next, failed);
        }
    }
}

// Rows kStep and later of a register column against a diagonal block kept by rows in `rows`, in pivot order:
// each is divided by its root, then updates the rows below it.
template <uint32_t kStep>
__device__ __forceinline__ void MultilevelSolvePivots(float (&n)[kMultilevelFactorPanel], const float* rows) {
    constexpr uint32_t kPanel = kMultilevelFactorPanel;
    if constexpr (kStep < kPanel) {
        float4 row[kPanel / 4u];
        MultilevelDiagonalRow<kStep / 4u>(row, reinterpret_cast<const float4*>(rows + kStep * kPanel));
        const float4 quad = row[kStep / 4u];
        const float root = kStep % 4u == 0u ? quad.x : kStep % 4u == 1u ? quad.y : kStep % 4u == 2u ? quad.z : quad.w;
        n[kStep] = math::Quotient(n[kStep], root);
        MultilevelDiagonalRows<kStep, kStep + 1u>(n, row);
        MultilevelSolvePivots<kStep + 1u>(n, rows);
    }
}

// Updates of every row of a register column by rows kStep and later of `rows` in pivot order, each times
// the column's own entry `n` in that row.
template <uint32_t kStep>
__device__ __forceinline__ void MultilevelBlockUpdates(float (&d)[kMultilevelFactorPanel],
                                                       const float (&n)[kMultilevelFactorPanel], const float* rows) {
    constexpr uint32_t kPanel = kMultilevelFactorPanel;
    if constexpr (kStep < kPanel) {
        const float4* quads = reinterpret_cast<const float4*>(rows + kStep * kPanel);
#pragma unroll
        for (uint32_t q = 0u; q < kPanel / 4u; ++q) {
            const float4 quad = quads[q];
            d[4u * q] = fmaf(-quad.x, n[kStep], d[4u * q]);
            d[4u * q + 1u] = fmaf(-quad.y, n[kStep], d[4u * q + 1u]);
            d[4u * q + 2u] = fmaf(-quad.z, n[kStep], d[4u * q + 2u]);
            d[4u * q + 3u] = fmaf(-quad.w, n[kStep], d[4u * q + 3u]);
        }
        MultilevelBlockUpdates<kStep + 1u>(d, n, rows);
    }
}

// Updates of a trailing column's panel rows kRow and below by panel row kStep.
template <uint32_t kStep, uint32_t kRow>
__device__ __forceinline__ void MultilevelColumnRows(float (&d)[kMultilevelFactorPanel], const float* scaled,
                                                     uint32_t base) {
    if constexpr (kRow < kMultilevelFactorPanel) {
        d[kRow] = fmaf(-scaled[MultilevelPacked(base + kStep, base + kRow)], d[kStep], d[kRow]);
        MultilevelColumnRows<kStep, kRow + 1u>(d, scaled, base);
    }
}

// Panel rows kStep and later of a trailing column, each divided by its root before it updates the rest.
// The next root and the next row's coefficient are read before the division that precedes their use.
template <uint32_t kStep>
__device__ __forceinline__ void MultilevelColumnPivots(float (&d)[kMultilevelFactorPanel], const float* scaled,
                                                       uint32_t base, float root) {
    if constexpr (kStep < kMultilevelFactorPanel) {
        float next_root = 0.0f, along = 0.0f;
        if constexpr (kStep + 1u < kMultilevelFactorPanel) {
            next_root = scaled[MultilevelPacked(base + kStep + 1u, base + kStep + 1u)];
            along = scaled[MultilevelPacked(base + kStep, base + kStep + 1u)];
        }
        d[kStep] = math::Quotient(d[kStep], root);
        if constexpr (kStep + 1u < kMultilevelFactorPanel) d[kStep + 1u] = fmaf(-along, d[kStep], d[kStep + 1u]);
        MultilevelColumnRows<kStep, kStep + 2u>(d, scaled, base);
        MultilevelColumnPivots<kStep + 1u>(d, scaled, base, next_root);
    }
}

// Warp 0 factors the diagonal block at `base` in registers, lane x holding column base + x; past the first block,
// these columns are first solved against the previous panel, kept for the other warps and updated by it.
__device__ inline bool MultilevelFactorBlock(float* scaled, float* panel, float* rows, const float* stored,
                                             uint32_t stride, uint32_t base, uint32_t width, uint32_t lane) {
    constexpr uint32_t kPanel = kMultilevelFactorPanel;
    const uint32_t j = base + lane;
    float d[kPanel];
    if (base == 0u) {
#pragma unroll
        for (uint32_t m = 0u; m < kPanel; ++m)
            d[m] = lane < width && m <= lane ? stored[MultilevelPacked(m, j)] : 0.0f;
    } else {
        float n[kPanel];
#pragma unroll
        for (uint32_t m = 0u; m < kPanel; ++m)
            n[m] = lane < width ? scaled[MultilevelPacked(base - kPanel + m, j)] : 0.0f;
        MultilevelSolvePivots<0u>(n, rows);
#pragma unroll
        for (uint32_t m = 0u; m < kPanel; ++m)
            if (lane < width) {
                scaled[MultilevelPacked(base - kPanel + m, j)] = n[m];
                panel[m * stride + lane] = n[m];
            }
        MultilevelPanelArrive();
#pragma unroll
        for (uint32_t m = 0u; m < kPanel; ++m)
            d[m] = lane < width && m <= lane ? scaled[MultilevelPacked(base + m, j)] : 0.0f;
        __syncwarp();
#pragma unroll
        for (uint32_t m = 0u; m < kPanel; ++m) rows[m * kPanel + lane] = n[m];
        __syncwarp();
        MultilevelBlockUpdates<0u>(d, n, rows);
        __syncwarp();
    }
    bool failed = false;
    MultilevelDiagonalPivots<0u>(d, rows, width, lane, __shfl_sync(0xffffffffu, d[0], 0u), &failed);
#pragma unroll
    for (uint32_t m = 0u; m < kPanel; ++m)
        if (lane < width && m <= lane) scaled[MultilevelPacked(base + m, j)] = d[m];
    return failed;
}

// Trailing column `column` of the panel at `base`: its panel rows take their in-panel updates in pivot
// order, each then divided by its root, and are also kept by rows for the trailing update.
__device__ inline void MultilevelFactorColumn(float* scaled, float* panel, uint32_t stride, uint32_t base,
                                              uint32_t column) {
    constexpr uint32_t kPanel = kMultilevelFactorPanel;
    const uint32_t j = base + kPanel + column;
    float d[kPanel];
#pragma unroll
    for (uint32_t m = 0u; m < kPanel; ++m) d[m] = scaled[MultilevelPacked(base + m, j)];
    MultilevelColumnPivots<0u>(d, scaled, base, scaled[MultilevelPacked(base, base)]);
#pragma unroll
    for (uint32_t m = 0u; m < kPanel; ++m) {
        scaled[MultilevelPacked(base + m, j)] = d[m];
        panel[m * stride + column] = d[m];
    }
}

// Tile `tile` of the trailing upper triangle past its first `lead` tile columns, numbered row by row in 4 x 4
// blocks, takes the updates of every panel row in pivot order.
__device__ inline void MultilevelFactorTile(float* scaled, const float* panel, uint32_t stride, uint32_t base,
                                            uint32_t rank, uint32_t tiles, uint32_t lead, uint32_t tile) {
    constexpr uint32_t kPanel = kMultilevelFactorPanel;
    const uint32_t wide = tiles - lead;
    uint32_t row = tile / wide, column = lead + tile % wide;
    if (row >= lead) {
        uint32_t rest = tile - lead * wide;
        for (row = lead; rest >= tiles - row; ++row) rest -= tiles - row;
        column = row + rest;
    }
    const uint32_t i0 = base + kPanel + 4u * row, j0 = base + kPanel + 4u * column;
    float a[4][4];
#pragma unroll
    for (uint32_t x = 0u; x < 4u; ++x)
#pragma unroll
        for (uint32_t y = 0u; y < 4u; ++y)
            a[x][y] = i0 + x <= j0 + y && j0 + y < rank ? scaled[MultilevelPacked(i0 + x, j0 + y)] : 0.0f;
#pragma unroll 4
    for (uint32_t m = 0u; m < kPanel; ++m) {
        const float4 r = *reinterpret_cast<const float4*>(panel + m * stride + 4u * row);
        const float4 c = *reinterpret_cast<const float4*>(panel + m * stride + 4u * column);
        const float u[4] = {r.x, r.y, r.z, r.w}, v[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
        for (uint32_t x = 0u; x < 4u; ++x)
#pragma unroll
            for (uint32_t y = 0u; y < 4u; ++y) a[x][y] = fmaf(-u[x], v[y], a[x][y]);
    }
#pragma unroll
    for (uint32_t x = 0u; x < 4u; ++x)
#pragma unroll
        for (uint32_t y = 0u; y < 4u; ++y)
            if (i0 + x <= j0 + y && j0 + y < rank) scaled[MultilevelPacked(i0 + x, j0 + y)] = a[x][y];
}

// One block per environment factors the Jacobi-scaled dense level by panels of 32 pivot rows, every entry in
// ascending pivot order; warp 0 factors each next diagonal block while the other warps update the rest.
__global__ void __launch_bounds__(32u * kMultilevelFactorWarps, 1)
MultilevelDenseFactorKernel(BlockDescentSolveParams p, BlockScratch s) {
    constexpr uint32_t kPanel = kMultilevelFactorPanel;
    extern __shared__ float4 factor_words[];
    __shared__ __align__(16) float factor_rows[kPanel * kPanel];
    __shared__ bool factor_failed;
    const VertexBlockLayout& l = p.vertex_blocks;
    const uint32_t rank = 3u * l.coarse_dense_nodes;
    const uint32_t packed = rank * (rank + 1u) / 2u, stride = MultilevelFactorPanelStride(rank);
    float* scaled = reinterpret_cast<float*>(factor_words);
    float* panel = reinterpret_cast<float*>(reinterpret_cast<char*>(factor_words) + MultilevelFactorPanelOffset(rank));
    const uint32_t lane = threadIdx.x % 32u, warp = threadIdx.x / 32u, updaters = blockDim.x - 32u;
    for (uint32_t env = blockIdx.x; env < p.env_count; env += gridDim.x) {
        bool failed = s.multilevel[env].exponent == kMultilevelUnassembled;
        float* stored = s.multilevel_dense_factor + size_t{env} * MultilevelDenseFactorWords(rank);
        if (threadIdx.x == 0u) factor_failed = failed;
        for (uint32_t base = 0u; !failed && base < rank; base += kPanel) {
            const uint32_t width = rank - base < kPanel ? rank - base : kPanel, trailing = rank - base;
            if (warp == 0u) {
                const bool block_failed =
                    MultilevelFactorBlock(scaled, panel, factor_rows, stored, stride, base, width, lane);
                if (block_failed && lane == 0u) factor_failed = true;
            } else if (base == 0u) {
                MultilevelLoadShared(scaled, stored, width * (width + 1u) / 2u, packed, threadIdx.x - 32u, updaters);
            } else {
                for (uint32_t column = threadIdx.x; column < trailing; column += updaters)
                    MultilevelFactorColumn(scaled, panel, stride, base - kPanel, column);
                MultilevelPanelWait();
                const uint32_t tiles = (trailing + 3u) / 4u, lead = kPanel / 4u;
                const uint32_t count = tiles > lead ? (tiles - lead) * (tiles + lead + 1u) / 2u : 0u;
                for (uint32_t tile = threadIdx.x - 32u; tile < count; tile += updaters)
                    MultilevelFactorTile(scaled, panel, stride, base - kPanel, rank, tiles, lead, tile);
            }
            __syncthreads();
            failed = factor_failed;
        }
        for (uint32_t k = threadIdx.x; !failed && k < packed; k += blockDim.x) stored[k] = scaled[k];
        if (threadIdx.x == 0u) s.multilevel[env].dense = failed ? 0u : 1u;
        __syncthreads();
    }
}

// The pivot of inverse step k and the factor entries it applies to this lane's rows: row k of the factor
// on forward steps, column k on back steps, read a step ahead of the broadcast that uses them.
__device__ inline void MultilevelInverseTerms(const float* factor, uint32_t k, uint32_t t, uint32_t rank,
                                              uint32_t lane, bool forward, float* pivot,
                                              float (&coefficient)[kMultilevelFactorColumns]) {
    *pivot = factor[MultilevelPacked(k, k)];
#pragma unroll
    for (uint32_t r = 0u; r < kMultilevelFactorColumns; ++r) {
        const uint32_t i = lane + 32u * r;
        const bool live = forward ? i > k && i < rank : i >= t && i < k;
        coefficient[r] = live ? factor[forward ? MultilevelPacked(k, i) : MultilevelPacked(i, k)] : 0.0f;
    }
}

// Columns of the dense inverse: each warp keeps a column in lane-strided registers, broadcasts each
// pivot value and performs its forward and back updates in order before mirroring and unscaling it.
__global__ void __launch_bounds__(32u * kMultilevelInverseWarps)
MultilevelDenseInverseKernel(BlockDescentSolveParams p, BlockScratch s) {
    extern __shared__ float factor[];
    const VertexBlockLayout& l = p.vertex_blocks;
    const uint32_t rank = 3u * l.coarse_dense_nodes, packed = rank * (rank + 1u) / 2u;
    const uint32_t groups = (rank + kMultilevelInverseWarps - 1u) / kMultilevelInverseWarps;
    const uint32_t lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
    const float* scales = factor + packed;
    for (uint32_t item = blockIdx.x; item < p.env_count * groups; item += gridDim.x) {
        const uint32_t env = item / groups, t = item % groups + warp * groups;
        if (s.multilevel[env].dense == 0u) continue;
        const float* stored = s.multilevel_dense_factor + size_t{env} * MultilevelDenseFactorWords(rank);
        __syncthreads();
        MultilevelLoadShared(factor, stored, packed + rank);
        __syncthreads();
        if (t >= rank) continue;
        float y[kMultilevelFactorColumns];
#pragma unroll
        for (uint32_t r = 0u; r < kMultilevelFactorColumns; ++r) y[r] = lane + 32u * r == t ? 1.0f : 0.0f;
        float pivot = 0.0f, next_pivot = 0.0f;
        float coefficient[kMultilevelFactorColumns] = {}, next[kMultilevelFactorColumns] = {};
#pragma unroll
        for (uint32_t h = 0u; h < kMultilevelFactorColumns; ++h) {
            const uint32_t first = t > 32u * h ? t : 32u * h;
            const uint32_t end = rank < 32u * (h + 1u) ? rank : 32u * (h + 1u);
            if (first < end) MultilevelInverseTerms(factor, first, t, rank, lane, true, &pivot, coefficient);
            for (uint32_t k = first; k < end; ++k) {
                if (k + 1u < end) MultilevelInverseTerms(factor, k + 1u, t, rank, lane, true, &next_pivot, next);
                const float value = math::Quotient(__shfl_sync(0xffffffffu, y[h], k % 32u), pivot);
#pragma unroll
                for (uint32_t r = h; r < kMultilevelFactorColumns; ++r) {
                    const uint32_t i = lane + 32u * r;
                    if (i > k && i < rank) y[r] = fmaf(-coefficient[r], value, y[r]);
                }
                if (lane == k % 32u) y[h] = value;
                pivot = next_pivot;
#pragma unroll
                for (uint32_t r = 0u; r < kMultilevelFactorColumns; ++r) coefficient[r] = next[r];
            }
        }
#pragma unroll
        for (uint32_t h = kMultilevelFactorColumns; h-- > 0u;) {
            const uint32_t low = t > 32u * h ? t : 32u * h;
            const uint32_t end = rank < 32u * (h + 1u) ? rank : 32u * (h + 1u);
            if (low < end) MultilevelInverseTerms(factor, end - 1u, t, rank, lane, false, &pivot, coefficient);
            for (uint32_t k = end; k-- > low;) {
                if (k > low) MultilevelInverseTerms(factor, k - 1u, t, rank, lane, false, &next_pivot, next);
                const float value = math::Quotient(__shfl_sync(0xffffffffu, y[h], k % 32u), pivot);
#pragma unroll
                for (uint32_t r = 0u; r <= h; ++r) {
                    const uint32_t i = lane + 32u * r;
                    if (i >= t && i < k) y[r] = fmaf(-coefficient[r], value, y[r]);
                }
                if (lane == k % 32u) y[h] = value;
                pivot = next_pivot;
#pragma unroll
                for (uint32_t r = 0u; r < kMultilevelFactorColumns; ++r) coefficient[r] = next[r];
            }
        }
        float* inverse = s.multilevel_dense_inverse + size_t{env} * rank * rank;
#pragma unroll
        for (uint32_t r = 0u; r < kMultilevelFactorColumns; ++r) {
            const uint32_t i = lane + 32u * r;
            if (i < t || i >= rank) continue;
            const float value = y[r] * scales[i] * scales[t];
            inverse[size_t{i} * rank + t] = value;
            inverse[size_t{t} * rank + i] = value;
        }
        __syncwarp();
    }
}

// Lane strides of a restriction whose child, weight and force loads are issued together; the products
// still add in the lane's stride order.
constexpr uint32_t kMultilevelRestrictBatch = 8u;

// A node's restricted residual, summed by one warp in a fixed lane-strided order; every lane gets it.
__device__ inline Vec3 RestrictedResidual(const ModelView& model, const BlockScratch& s, uint32_t vertices,
                                          uint32_t env, uint32_t node, uint32_t lane) {
    const Vec3* force = s.multilevel_force + size_t{env} * vertices;
    const uint32_t end = model.vbd_coarse_child_offsets[node + 1u];
    Vec3 residual{};
    for (uint32_t base = model.vbd_coarse_child_offsets[node] + lane; base < end;
         base += 32u * kMultilevelRestrictBatch) {
        uint32_t child[kMultilevelRestrictBatch];
        float weight[kMultilevelRestrictBatch];
        Vec3 value[kMultilevelRestrictBatch];
#pragma unroll
        for (uint32_t k = 0u; k < kMultilevelRestrictBatch; ++k) {
            const uint32_t i = base + 32u * k;
            child[k] = i < end ? model.vbd_coarse_children[i] : 0u;
            weight[k] = i < end ? model.vbd_coarse_child_weights[i] : 0.0f;
        }
#pragma unroll
        for (uint32_t k = 0u; k < kMultilevelRestrictBatch; ++k)
            if (base + 32u * k < end) value[k] = force[child[k]];
#pragma unroll
        for (uint32_t k = 0u; k < kMultilevelRestrictBatch; ++k)
            if (base + 32u * k < end) residual += value[k] * weight[k];
    }
    return {WarpSum(residual.x), WarpSum(residual.y), WarpSum(residual.z)};
}

// Restricted residuals of all coarse nodes, one warp per node; the dense level is staged separately
// so its matrix product reads a complete input before any output is written.
__global__ void MultilevelCorrectKernel(ModelView model, BlockDescentSolveParams p,
                                        BlockScratch s, uint32_t parts) {
    const VertexBlockLayout& l = p.vertex_blocks;
    const uint32_t first = l.coarse_nodes - l.coarse_dense_nodes, rank = 3u * l.coarse_dense_nodes;
    const uint32_t lane = threadIdx.x % 32u, warp = threadIdx.x / 32u, warps = blockDim.x / 32u;
    // Coarser nodes restrict more children, so their blocks come first.
    const uint32_t env = blockIdx.x / parts, node = (parts - 1u - blockIdx.x % parts) * warps + warp;
    if (node >= l.coarse_nodes) return;
    const Vec3 residual = RestrictedResidual(model, s, l.vertices, env, node, lane);
    if (lane != 0u) return;
    if (node < first)
        s.multilevel_correction[size_t{env} * l.coarse_nodes + node] =
            s.multilevel_block_inverse[size_t{env} * first + node].Multiply(residual);
    else {
        float* out = s.multilevel_restricted + size_t{env} * rank + 3u * (node - first);
        out[0] = residual.x;
        out[1] = residual.y;
        out[2] = residual.z;
    }
}

// Warps share 32 contiguous outputs; each keeps the original lane-strided FMA partials, then sums
// them through WarpSum's offsets 16, 8, 4, 2, 1 with an ordered shared-memory reduction.
__global__ void __launch_bounds__(kMultilevelCorrectThreads, 2)
MultilevelDenseCorrectKernel(BlockDescentSolveParams p, BlockScratch s, uint32_t tiles) {
    constexpr uint32_t warps = kMultilevelCorrectThreads / 32u;
    constexpr uint32_t leaves = 32u / warps;
    static_assert(leaves == 4u && warps == 8u);
    __shared__ float restricted[kMultilevelDenseRank];
    __shared__ float partial[warps][32];
    const VertexBlockLayout& l = p.vertex_blocks;
    const uint32_t rank = 3u * l.coarse_dense_nodes, first = l.coarse_nodes - l.coarse_dense_nodes;
    const uint32_t env = blockIdx.x / tiles, lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
    const uint32_t i = blockIdx.x % tiles * 32u + lane;
    for (uint32_t k = threadIdx.x; k < rank; k += blockDim.x)
        restricted[k] = s.multilevel_restricted[size_t{env} * rank + k];
    __syncthreads();
    const uint32_t terms = s.multilevel[env].dense != 0u ? rank : 0u;
    const float* inverse = s.multilevel_dense_inverse + size_t{env} * rank * rank;
    float sums[leaves] = {};
#pragma unroll
    for (uint32_t leaf = 0u; leaf < leaves; ++leaf)
#pragma unroll
        for (uint32_t j = warp + warps * leaf; j < kMultilevelDenseRank; j += 32u)
            if (i < rank && j < terms) sums[leaf] = fmaf(inverse[size_t{j} * rank + i], restricted[j], sums[leaf]);
    partial[warp][lane] = __fadd_rn(__fadd_rn(sums[0], sums[2]), __fadd_rn(sums[1], sums[3]));
    __syncthreads();
    if (warp < 4u) partial[warp][lane] = __fadd_rn(partial[warp][lane], partial[warp + 4u][lane]);
    __syncthreads();
    if (warp < 2u) partial[warp][lane] = __fadd_rn(partial[warp][lane], partial[warp + 2u][lane]);
    __syncthreads();
    if (warp != 0u || i >= rank) return;
    const float value = __fadd_rn(partial[0][lane], partial[1][lane]);
    Vec3& out = s.multilevel_correction[size_t{env} * l.coarse_nodes + first + i / 3u];
    if (i % 3u == 0u) out.x = value;
    else if (i % 3u == 1u) out.y = value;
    else out.z = value;
}

// The summed level corrections interpolated to every moving vertex as a rate direction.
__global__ void MultilevelProlongKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                                        BlockScratch s) {
    const VertexBlockLayout& l = p.vertex_blocks;
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < l.vertices * p.env_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t env = item / l.vertices, vertex = item % l.vertices;
        Vec3 direction{};
        if (MultilevelDynamic(s, env * l.particles_per_env + l.begin + vertex))
            for (uint32_t level = 0u; level < l.coarse_levels; ++level) {
                const size_t at = (size_t{level} * l.vertices + vertex) * nk::kVbdCoarseParents;
                for (uint32_t k = 0u; k < nk::kVbdCoarseParents; ++k) {
                    const uint32_t node = model.vbd_coarse_parents[at + k];
                    if (node == ~0u) continue;
                    direction += s.multilevel_correction[size_t{env} * l.coarse_nodes + node] *
                                 model.vbd_coarse_weights[at + k];
                }
            }
        s.multilevel_direction[item] = direction / p.dt;
    }
}

// Resident blocks per multiprocessor the plane trial kernel is compiled for.
constexpr uint32_t kMultilevelPlaneBlocks = 6u;
// Touches of a selected row group whose gathers the plane forms issue together.
constexpr uint32_t kMultilevelFormsTouches = 4u;

// Row threads retain their original order while adjacent threads read adjacent selected groups, keeping the
// products of each row with the correction and the previous step for the trials; the sums open the unit trial.
__global__ void MultilevelFormsKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                                      BlockScratch s) {
    __shared__ double shared[kMultilevelSums][kThreads / 32u];
    __shared__ bool last;
    uint32_t env = 0u, part = 0u;
    const bool rows = MultilevelPlanePart(p, s, &env, &part);
    double sums[kMultilevelSums] = {};
    if (rows) {
        uint32_t begin = 0u, end = 0u;
        CoarseRowChunk(s, s.multilevel_parts - s.multilevel_item_parts, env, part, &begin, &end);
        for (uint32_t item = begin + threadIdx.x; item < end; item += kThreads) {
            Vec3 d{}, q{};
            const uint32_t count = s.multilevel_touch_count[item] & (kMultilevelTouchOverflow - 1u);
            const float* curvature = s.multilevel_row_curvature + 4u * size_t{s.coarse_rows[item]};
            const float c[4] = {curvature[0], curvature[1], curvature[2], curvature[3]};
            // Each batch issues its touch gathers together; the products still add in touch order.
            for (uint32_t first = 0u; first < count; first += kMultilevelFormsTouches) {
                uint2 entry[kMultilevelFormsTouches];
                Vec3 direction[kMultilevelFormsTouches], previous[kMultilevelFormsTouches];
                Vec3 x[kMultilevelFormsTouches], y[kMultilevelFormsTouches], z[kMultilevelFormsTouches];
#pragma unroll
                for (uint32_t k = 0u; k < kMultilevelFormsTouches; ++k)
                    if (first + k < count) entry[k] = s.multilevel_touch[size_t{first + k} * s.rows + item];
#pragma unroll
                for (uint32_t k = 0u; k < kMultilevelFormsTouches; ++k) {
                    if (first + k >= count) continue;
                    direction[k] = s.multilevel_direction[entry[k].x];
                    previous[k] = s.multilevel_previous[entry[k].x];
                    x[k] = LoadPointIncidenceJacobian(s, entry[k].y, 0u);
                    y[k] = LoadPointIncidenceJacobian(s, entry[k].y, 1u);
                    z[k] = LoadPointIncidenceJacobian(s, entry[k].y, 2u);
                }
#pragma unroll
                for (uint32_t k = 0u; k < kMultilevelFormsTouches; ++k) {
                    if (first + k >= count) continue;
                    d += Vec3{x[k].Dot(direction[k]), y[k].Dot(direction[k]), z[k].Dot(direction[k])};
                    q += Vec3{x[k].Dot(previous[k]), y[k].Dot(previous[k]), z[k].Dot(previous[k])};
                }
            }
            if (count == 0u) continue;
            s.multilevel_row_along[2u * size_t{item}] = d;
            s.multilevel_row_along[2u * size_t{item} + 1u] = q;
            sums[2] += RowForm(c, d, d);
            sums[3] += RowForm(c, d, q);
            sums[4] += RowForm(c, q, q);
        }
    } else {
        const VertexBlockView b = Vertices(model, data, p);
        VisitPlaneItems(s, b.layout, env, part,
            [&](uint32_t vertex) {
                const uint32_t slot = b.Slot(env, vertex);
                if (!MultilevelDynamic(s, b.Particle(env, vertex))) return;
                const Vec3 f = s.multilevel_force[slot], d = s.multilevel_direction[slot];
                const Vec3 q = s.multilevel_previous[slot];
                const double mass = double(b.inertia[slot]) * p.dt * p.dt;
                sums[0] += double(p.dt) * (double(f.x) * d.x + double(f.y) * d.y + double(f.z) * d.z);
                sums[1] += double(p.dt) * (double(f.x) * q.x + double(f.y) * q.y + double(f.z) * q.z);
                sums[2] += mass * (double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z);
                sums[3] += mass * (double(d.x) * q.x + double(d.y) * q.y + double(d.z) * q.z);
                sums[4] += mass * (double(q.x) * q.x + double(q.y) * q.y + double(q.z) * q.z);
            },
            [&](uint32_t element) { AddElementForms(b, data, s, env, element, sums); },
            [&](uint32_t) {});
    }
    // Row and element-only blocks leave the force sums 0 and 1 at +0.0.
    if (rows || (part - (s.multilevel_parts - s.multilevel_item_parts)) * kThreads >= p.vertex_blocks.vertices)
        MultilevelBlockSums<2u>(sums, shared);
    else
        MultilevelBlockSums<0u>(sums, shared);
    if (threadIdx.x == 0u) {
        double* out = s.multilevel_partials + (size_t{env} * s.multilevel_parts + part) * kMultilevelSums;
        for (uint32_t k = 0u; k < kMultilevelSums; ++k) out[k] = sums[k];
        last = MultilevelArrive(s, env, s.multilevel_parts);
    }
    __syncthreads();
    if (!last) return;
    MultilevelPartialSums(s, env, sums, shared);
    if (threadIdx.x != 0u) return;
    MultilevelState& state = s.multilevel[env];
    state.slope = 0.0;
    state.alpha = state.beta = state.scale = state.trial = 0.0f;
    const double fd = sums[0], fq = sums[1], dd = sums[2], dq = sums[3], qq = sums[4];
    const double det = dd * qq - dq * dq;
    double alpha = 0.0, beta = 0.0;
    if (qq > 0.0 && det > kMultilevelIndependence * dd * qq) {
        alpha = (fd * qq - fq * dq) / det;
        beta = (dd * fq - dq * fd) / det;
    } else if (dd > 0.0) {
        alpha = fd / dd;
    }
    bool finite = fabs(alpha) <= double(FLT_MAX) && fabs(beta) <= double(FLT_MAX);
    for (uint32_t k = 0u; k < kMultilevelSums; ++k) finite &= fabs(sums[k]) <= DBL_MAX;
    if (!finite) {
        atomicOr(data.env_status + env, kEnvStatusSolverFailure);
        return;
    }
    state.alpha = static_cast<float>(alpha);
    state.beta = static_cast<float>(beta);
    state.slope = -(double(state.alpha) * fd + double(state.beta) * fq);
    state.trial = state.slope < 0.0 ? 1.0f : 0.0f;
}

// Row drops scale the kept products by the step coefficients and items add their energy changes; a sufficient
// decrease is accepted, otherwise the quadratic minimizer opens the next trial within [0.1, 0.5] of the scale.
__global__ void __launch_bounds__(kThreads, kMultilevelPlaneBlocks)
MultilevelTrialKernel(ModelView model, DataView data, BlockDescentSolveParams p, BlockScratch s, bool final) {
    __shared__ double shared[1][kThreads / 32u];
    __shared__ bool last;
    uint32_t env = 0u, part = 0u;
    const bool rows = MultilevelPlanePart(p, s, &env, &part);
    const MultilevelState state = s.multilevel[env];
    if (!(state.trial > 0.0f)) return;
    double change[1] = {};
    if (rows) {
        uint32_t begin = 0u, end = 0u;
        CoarseRowChunk(s, s.multilevel_parts - s.multilevel_item_parts, env, part, &begin, &end);
        for (uint32_t item = begin + threadIdx.x; item < end; item += kThreads) {
            const uint32_t slot = s.coarse_rows[item];
            if ((s.multilevel_touch_count[item] & (kMultilevelTouchOverflow - 1u)) == 0u) continue;
            const Vec3 d = s.multilevel_row_along[2u * size_t{item}];
            const Vec3 q = s.multilevel_row_along[2u * size_t{item} + 1u];
            const Vec3 drop{__fmul_rn(state.trial, __fmaf_rn(state.beta, q.x, __fmul_rn(state.alpha, d.x))),
                            __fmul_rn(state.trial, __fmaf_rn(state.beta, q.y, __fmul_rn(state.alpha, d.y))),
                            __fmul_rn(state.trial, __fmaf_rn(state.beta, q.z, __fmul_rn(state.alpha, d.z)))};
            change[0] += EvaluateDropChange(LoadMultilevelRowTerm(s, item, s.row_state[slot]), drop);
        }
    } else {
        const VertexBlockView b = Vertices(model, data, p);
        VisitPlaneItems(s, b.layout, env, part,
            [&](uint32_t vertex) {
                const uint32_t particle = b.Particle(env, vertex), slot = b.Slot(env, vertex);
                if (!MultilevelDynamic(s, particle)) return;
                change[0] += nk::vbd::InertialEnergyChange(data.particle_vel[particle], b.free_rate[slot],
                    MultilevelMove(data, s, state, particle, slot, state.trial), b.inertia[slot], p.dt);
            },
            [&](uint32_t element) {
                change[0] += MultilevelElementChange(b, data, s, state, env, element, state.trial);
            },
            [&](uint32_t) {});
    }
    BlockSumsDouble(change, shared);
    if (threadIdx.x == 0u) {
        s.multilevel_partials[(size_t{env} * s.multilevel_parts + part) * kMultilevelSums] = change[0];
        last = MultilevelArrive(s, env, s.multilevel_parts);
    }
    __syncthreads();
    if (!last) return;
    MultilevelPartialSums(s, env, change, shared);
    if (threadIdx.x != 0u) return;
    MultilevelState& out = s.multilevel[env];
    const double trial = state.trial, total = change[0];
    out.trial = 0.0f;
    if (isfinite(total) && total <= double(kVertexStepDecrease) * trial * state.slope) {
        out.scale = state.trial;
        return;
    }
    if (final) return;
    const double next = isfinite(total)
        ? -state.slope * trial * trial / (2.0 * (total - state.slope * trial)) : 0.0;
    out.trial = static_cast<float>(fmin(fmax(next, 0.1 * trial), 0.5 * trial));
}

// Commits the accepted plane step and keeps it as the next previous step; a rejected step leaves none.
__global__ void MultilevelApplyKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const VertexBlockLayout& l = p.vertex_blocks;
    for (uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x; slot < l.vertices * p.env_count;
         slot += gridDim.x * blockDim.x) {
        const uint32_t env = slot / l.vertices;
        const uint32_t particle = env * l.particles_per_env + l.begin + slot % l.vertices;
        const MultilevelState& state = s.multilevel[env];
        Vec3 previous{};
        if (state.scale > 0.0f && MultilevelDynamic(s, particle)) {
            const Vec3 u = data.particle_vel[particle];
            const Vec3 candidate = MultilevelCandidate(s, state, slot, u, state.scale);
            data.particle_vel[particle] = candidate;
            previous = candidate - u;
        }
        s.multilevel_previous[slot] = previous;
    }
}
