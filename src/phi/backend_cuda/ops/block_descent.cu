#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "constraint/coulomb_contact.hpp"
#include "nk/material/mpm_constitutive.hpp"
#include "nk/solve/augmented_row.hpp"
#include "nk/solve/block_row_schedule.hpp"
#include "nk/solve/chebyshev_iteration.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/launch_grid.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/dense_block_solve.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/vertex_blocks.cuh"

namespace nuka::phi {
namespace {

using namespace nkops;
using math::Vec3;
using math::SymmetricMat3;
constexpr uint32_t kThreads = 128u;
constexpr uint32_t kRigidBlockDof = 6u;
constexpr uint32_t kSolverCountWords = 6u;
constexpr uint32_t kSolverPhaseWords = 8u;
constexpr uint32_t kNoCacheColor = ~0u;
constexpr uint32_t kAllCacheColors = ~0u;
constexpr uint32_t kPointJacobianAxes = 3u;
constexpr uint32_t kPointJacobianComponents = 3u;
constexpr uint32_t kPointJacobianWords = kPointJacobianAxes * kPointJacobianComponents;

struct AugmentedRowState {
    Vec3 residual;
    Vec3 dual;
    float normal_bound;
    double potential = 0.0;
};
static_assert(sizeof(AugmentedRowState) == 40u);

struct MaterialRowState {
    float residual[nk::kMpmStressRowsPerCell];
    float pressure_bound;
};

constexpr uint32_t kCoarseVertexFamily = 0u;
constexpr uint32_t kCoarseParticleFamily = 1u;
constexpr uint32_t kCoarseFamilies = 2u;
constexpr uint32_t kNoCoarseFamily = ~0u;
constexpr uint32_t kCoarseTrials = kVertexStepHalvings + 1u;
// Force (3) and symmetric curvature (6) of one translation system.
constexpr uint32_t kCoarseGradientTerms = 9u;
// Blocks per environment that share its live rows, bounded by its reserved row slots.
constexpr uint32_t kCoarseRowParts = 64u;
static_assert(kCoarseTrials >= kCoarseGradientTerms);

struct CoarseState {
    double slope;
    float direction[3];
    float scale;
};
static_assert(sizeof(CoarseState) % sizeof(double) == 0u);

uint32_t CoarseParts(const BlockDescentSolveParams& p) {
    const uint64_t per_env = p.env_count > 0u ? p.total_particle_count / p.env_count : 0u;
    const uint64_t row_parts = std::min<uint64_t>(kCoarseRowParts,
        (uint64_t{p.rows_per_env} + kThreads - 1u) / kThreads);
    const uint64_t parts = std::max((per_env + kThreads - 1u) / kThreads, row_parts);
    return static_cast<uint32_t>(std::max<uint64_t>(1u, parts));
}

// Live point-row heads, selected in slot order so the coarse sums keep a fixed order.
struct CoarseRowSelect {
    const NkRow* rows;
    __device__ bool operator()(uint32_t slot) const {
        const NkRow& row = rows[slot];
        return (row.flags & nk::nk_row_flags::kActive) != 0u &&
               (row.flags & (nk::nk_row_flags::kBlockTangent | nk::nk_row_flags::kMaterialBlock)) == 0u &&
               (PointMassView::IsPointSide(row.a.kind) || PointMassView::IsPointSide(row.b.kind));
    }
};

struct BlockScratch {
    uint32_t* counts;
    uint32_t* offsets;
    uint32_t* incidence;
    uint32_t* sorted_incidence;
    uint32_t* active_rows;
    uint32_t* active_count;
    uint32_t* failure_reasons;
    uint32_t* failure_rows;
    uint32_t* failure_substeps;
    double* failure_equations;
    float* penalty;
    Vec3* particle_snapshot;
    Vec3* particle_free;
    Vec3* particle_previous;
    Vec3* particle_older;
    uint32_t* acceleration_disabled;
    uint32_t* acceleration_iterations;
    uint32_t* acceleration_accepted;
    float* acceleration_ratio;
    double* acceleration_merit;
    uint32_t* articulation_roots;
    uint32_t* articulation_dimensions;
    float* articulation_free;
    float* articulation_snapshot;
    float* articulation_matrix;
    float* articulation_force;
    float* articulation_direction;
    float* articulation_diagonal;
    float* articulation_velocity;
    uint32_t* rigid_dimensions;
    float* rigid_free;
    float* rigid_snapshot;
    float* rigid_mass;
    float* rigid_matrix;
    float* rigid_force;
    float* rigid_direction;
    float* rigid_diagonal;
    float* rigid_velocity;
    Vec3* grid_free;
    Vec3* grid_snapshot;
    uint32_t* grid_active;
    uint32_t* grid_active_count;
    uint32_t* grid_color_offsets;
    MaterialRowState* material_state;
    AugmentedRowState* row_state;
    void* scan;
    size_t scan_bytes;
    void* incidence_sort;
    size_t incidence_sort_bytes;
    uint32_t blocks;
    uint32_t rows;
    uint64_t incidence_capacity;
    uint32_t diagnostic_iteration = ~0u;
    uint32_t* owner_cache_color;
    uint32_t* row_cache_colors;
    uint32_t color_words;
    float* point_jacobian;
    nk::augmented::ScalarResponse* scalar_response;
    CoarseState* coarse;
    double* coarse_partials;
    uint32_t* coarse_anchored;
    uint32_t* coarse_rows;
    uint32_t* coarse_row_count;
    uint8_t* coarse_flags;
    uint32_t* coarse_row_offsets;
    void* coarse_select;
    size_t coarse_select_bytes;
    uint32_t coarse_parts;
};

// Incidence entries are row slots, so sorting the bits that index every slot orders them fully.
int RowSlotBits(uint64_t rows) {
    int bits = 1;
    while (bits < 32 && (uint64_t{1} << bits) < rows) ++bits;
    return bits;
}

constexpr uint32_t kWarpSortKeysPerLane = 8u;
constexpr uint32_t kWarpSortLimit = kWarpSortKeysPerLane * 32u;

__host__ __device__ inline uint32_t IncidenceOffset(const uint32_t* offsets, uint64_t capacity,
                                                    uint32_t segment) {
    return offsets[segment] < capacity ? offsets[segment] : static_cast<uint32_t>(capacity);
}

struct IncidenceBegin {
    const uint32_t* offsets;
    uint64_t capacity;
    __host__ __device__ uint32_t operator()(uint32_t segment) const {
        return IncidenceOffset(offsets, capacity, segment);
    }
};

// Segment ends as the radix sort sees them: segments a warp sorts become empty.
struct LargeIncidenceEnd {
    const uint32_t* offsets;
    uint64_t capacity;
    __host__ __device__ uint32_t operator()(uint32_t segment) const {
        const uint32_t begin = IncidenceOffset(offsets, capacity, segment);
        const uint32_t end = IncidenceOffset(offsets, capacity, segment + 1u);
        return end - begin > kWarpSortLimit ? end : begin;
    }
};
using IncidenceBegins = thrust::transform_iterator<IncidenceBegin, thrust::counting_iterator<uint32_t>>;
using LargeIncidenceEnds =
    thrust::transform_iterator<LargeIncidenceEnd, thrust::counting_iterator<uint32_t>>;

uint64_t BindScratch(const BlockDescentSolveParams& p, uint32_t* base, BlockScratch* result) {
    const uint64_t blocks = uint64_t{p.total_particle_count} + p.total_body_count +
                            p.articulation_count + p.total_grid_count;
    const uint64_t rows = uint64_t{p.rows_per_env} * p.env_count;
    if (blocks >= uint64_t{std::numeric_limits<int>::max()} ||
        rows >= uint64_t{std::numeric_limits<int>::max()}) return 0u;
    const auto schedule = nk::MakeBlockRowScheduleLayout(blocks, rows, p.max_point_terms);
    const uint64_t incidence = schedule.incidence_capacity;
    if (incidence >= uint64_t{std::numeric_limits<uint32_t>::max()}) return 0u;
    const uint64_t word_limit = std::numeric_limits<size_t>::max() / sizeof(uint32_t);
    if (incidence > word_limit) return 0u;
    const bool incidence_terms = p.total_particle_count > 0u || p.total_grid_count > 0u ||
                                 p.total_body_count > 0u || p.articulation_count > 0u;
    if (incidence_terms && incidence > word_limit / kPointJacobianWords) return 0u;
    const uint64_t point_jacobian_words = incidence_terms ? incidence * kPointJacobianWords : 0u;
    const uint64_t cache_colors = uint64_t{p.vertex_blocks.colors} + 2u +
        (p.total_grid_count > 0u ? nk::kMpmCellStencilNodes : 0u);
    if (cache_colors > uint64_t{std::numeric_limits<uint32_t>::max()}) return 0u;
    const uint64_t color_words = (cache_colors + 31u) / 32u;
    if (rows > word_limit / color_words) return 0u;
    const uint64_t row_color_words = rows * color_words;
    const uint64_t dofs = uint64_t{p.articulation_count} * p.max_dof;
    if (dofs >= uint64_t{std::numeric_limits<uint32_t>::max()} ||
        dofs > word_limit || (dofs > 0u && p.max_dof > word_limit / dofs)) return 0u;
    const uint64_t matrix_words = dofs * p.max_dof;
    if (matrix_words > word_limit - dofs ||
        dofs > (word_limit - matrix_words) / 7u) return 0u;
    size_t scan_bytes = 0u;
    if (cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<int>(blocks + 1u)) != cudaSuccess)
        return 0u;
    size_t incidence_sort_bytes = 0u;
    // Owner deduplication bounds each segment by rows, already below CUB's INT_MAX limit.
    if (blocks > 0u && incidence > 0u &&
        cub::DeviceSegmentedRadixSort::SortKeys(nullptr, incidence_sort_bytes,
            static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<int64_t>(incidence), static_cast<int64_t>(blocks),
            IncidenceBegins(thrust::counting_iterator<uint32_t>(0u), IncidenceBegin{nullptr, 0u}),
            LargeIncidenceEnds(thrust::counting_iterator<uint32_t>(0u), LargeIncidenceEnd{nullptr, 0u}),
            0, RowSlotBits(rows)) != cudaSuccess) return 0u;
    size_t coarse_select_bytes = 0u;
    if (p.total_particle_count > 0u && rows > 0u &&
        cub::DeviceSelect::Flagged(nullptr, coarse_select_bytes, thrust::counting_iterator<uint32_t>(0u),
            static_cast<const uint8_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<int>(rows)) != cudaSuccess) return 0u;
    uint64_t at = 0u;
    bool fits = true;
    const auto take = [&](uint64_t words) {
        const uint64_t first = at;
        if (at > word_limit || words > word_limit - at) {
            fits = false;
            return static_cast<uint32_t*>(nullptr);
        }
        at += words;
        return fits && base != nullptr ? base + first : nullptr;
    };
    BlockScratch s{};
    s.counts = take(blocks + 1u);
    s.offsets = take(blocks + 1u);
    s.incidence = take(incidence);
    s.active_rows = take(rows);
    s.active_count = take(1u);
    s.failure_reasons = take(blocks);
    s.failure_rows = take(blocks);
    s.failure_substeps = take(blocks);
    at = (at + 1u) & ~uint64_t{1u};
    s.failure_equations = reinterpret_cast<double*>(take(
        blocks * nk::kBlockSolveFailureEquationColumnCount * (sizeof(double) / sizeof(uint32_t))));
    if (at != schedule.Words()) return 0u;
    s.penalty = reinterpret_cast<float*>(take(rows));
    s.particle_snapshot = reinterpret_cast<Vec3*>(take(3u * p.total_particle_count));
    s.particle_free = reinterpret_cast<Vec3*>(take(3u * p.total_particle_count));
    s.particle_previous = reinterpret_cast<Vec3*>(take(3u * p.total_particle_count));
    s.particle_older = reinterpret_cast<Vec3*>(take(3u * p.total_particle_count));
    s.acceleration_disabled = take(blocks);
    s.acceleration_iterations = take(p.env_count);
    s.acceleration_accepted = take(p.env_count);
    s.acceleration_ratio = reinterpret_cast<float*>(take(p.env_count));
    at = (at + 1u) & ~uint64_t{1u};
    s.acceleration_merit = reinterpret_cast<double*>(take(4u * p.env_count));
    s.articulation_roots = take(dofs);
    s.articulation_dimensions = take(p.articulation_count);
    s.articulation_free = reinterpret_cast<float*>(take(dofs));
    s.articulation_snapshot = reinterpret_cast<float*>(take(dofs));
    s.articulation_matrix = reinterpret_cast<float*>(take(matrix_words));
    s.articulation_force = reinterpret_cast<float*>(take(dofs));
    s.articulation_direction = reinterpret_cast<float*>(take(dofs));
    s.articulation_diagonal = reinterpret_cast<float*>(take(dofs));
    s.articulation_velocity = reinterpret_cast<float*>(take(dofs));
    const uint64_t rigid_dofs = uint64_t{p.total_body_count} * kRigidBlockDof;
    s.rigid_dimensions = take(p.total_body_count);
    s.rigid_free = reinterpret_cast<float*>(take(rigid_dofs));
    s.rigid_snapshot = reinterpret_cast<float*>(take(rigid_dofs));
    s.rigid_mass = reinterpret_cast<float*>(take(rigid_dofs * kRigidBlockDof));
    s.rigid_matrix = reinterpret_cast<float*>(take(rigid_dofs * kRigidBlockDof));
    s.rigid_force = reinterpret_cast<float*>(take(rigid_dofs));
    s.rigid_direction = reinterpret_cast<float*>(take(rigid_dofs));
    s.rigid_diagonal = reinterpret_cast<float*>(take(rigid_dofs));
    s.rigid_velocity = reinterpret_cast<float*>(take(rigid_dofs));
    s.grid_free = reinterpret_cast<Vec3*>(take(3u * uint64_t{p.total_grid_count}));
    s.grid_snapshot = reinterpret_cast<Vec3*>(take(3u * uint64_t{p.total_grid_count}));
    s.grid_active = take(p.total_grid_count);
    s.grid_active_count = take(p.total_grid_count > 0u ? nk::kMpmCellStencilNodes : 0u);
    s.grid_color_offsets = take(p.total_grid_count > 0u ? nk::kMpmCellStencilNodes : 0u);
    s.material_state = reinterpret_cast<MaterialRowState*>(take(
        uint64_t{p.material_cells_per_env} * p.env_count *
        sizeof(MaterialRowState) / sizeof(uint32_t)));
    at = (at + 63u) & ~uint64_t{63u};
    s.scan = take(scan_bytes / sizeof(uint32_t) + (scan_bytes % sizeof(uint32_t) != 0u));
    s.scan_bytes = scan_bytes;
    s.sorted_incidence = take(incidence_sort_bytes > 0u ? incidence : 0u);
    if (incidence_sort_bytes > 0u) at = (at + 63u) & ~uint64_t{63u};
    s.incidence_sort = take(incidence_sort_bytes / sizeof(uint32_t) +
                            (incidence_sort_bytes % sizeof(uint32_t) != 0u));
    s.incidence_sort_bytes = incidence_sort_bytes;
    at = (at + 1u) & ~uint64_t{1u};
    s.row_state = reinterpret_cast<AugmentedRowState*>(take(
        rows * sizeof(AugmentedRowState) / sizeof(uint32_t)));
    s.owner_cache_color = take(blocks);
    s.row_cache_colors = take(row_color_words);
    s.color_words = static_cast<uint32_t>(color_words);
    s.point_jacobian = reinterpret_cast<float*>(take(point_jacobian_words));
    at = (at + 1u) & ~uint64_t{1u};
    s.scalar_response = reinterpret_cast<nk::augmented::ScalarResponse*>(take(
        rows * sizeof(nk::augmented::ScalarResponse) / sizeof(uint32_t)));
    at = (at + 1u) & ~uint64_t{1u};
    const bool coarse = p.total_particle_count > 0u;
    s.coarse_parts = CoarseParts(p);
    if (uint64_t{p.env_count} * s.coarse_parts >= uint64_t{std::numeric_limits<int>::max()}) return 0u;
    s.coarse =reinterpret_cast<CoarseState*>(take(
        coarse ? uint64_t{p.env_count} * sizeof(CoarseState) / sizeof(uint32_t) : 0u));
    s.coarse_partials = reinterpret_cast<double*>(take(coarse
        ? uint64_t{p.env_count} * s.coarse_parts * kCoarseTrials * (sizeof(double) / sizeof(uint32_t))
        : 0u));
    s.coarse_anchored = take(coarse ? p.env_count : 0u);
    s.coarse_rows = take(coarse ? rows : 0u);
    s.coarse_row_count = take(coarse ? 1u : 0u);
    s.coarse_flags = reinterpret_cast<uint8_t*>(take(coarse ? (rows + 3u) / 4u : 0u));
    s.coarse_row_offsets = take(coarse ? uint64_t{p.env_count} + 1u : 0u);
    at = (at + 63u) & ~uint64_t{63u};
    s.coarse_select = take(coarse_select_bytes / sizeof(uint32_t) +
                           (coarse_select_bytes % sizeof(uint32_t) != 0u));
    s.coarse_select_bytes = coarse_select_bytes;
    s.blocks = static_cast<uint32_t>(blocks);
    s.rows = static_cast<uint32_t>(rows);
    s.incidence_capacity = incidence;
    if (!fits || at > word_limit) return 0u;
    if (result != nullptr) *result = s;
    return at;
}

__device__ uint32_t Owner(const BlockDescentSolveParams& p, uint32_t kind, uint32_t index) {
    if (kind == kNkSideParticle) return index;
    if (kind == kNkSideRigid) return p.total_particle_count + index;
    if (kind == kNkSideArtic) return p.total_particle_count + p.total_body_count + index;
    if (kind == kNkSideGrid)
        return p.total_particle_count + p.total_body_count + p.articulation_count + index;
    return ~0u;
}

__device__ uint32_t GridNodeColor(const BlockDescentSolveParams& p, uint32_t node) {
    const uint32_t nodes_per_env = p.total_grid_count / p.env_count;
    const uint32_t lattice_nodes = nodes_per_env / nk::kMpmLattices;
    const uint32_t env_node = node % nodes_per_env;
    const uint32_t lattice = env_node / lattice_nodes;
    const uint32_t local = env_node % lattice_nodes;
    const uint32_t x = local % p.grid_dims[0];
    const uint32_t y = (local / p.grid_dims[0]) % p.grid_dims[1];
    const uint32_t z = static_cast<uint32_t>(local / (size_t{p.grid_dims[0]} * p.grid_dims[1]));
    const uint32_t width = nk::kMpmStencilWidth + lattice;
    const uint32_t first = lattice == 0u ? 0u : nk::kMpmLatticeStencilNodes;
    return first + x % width + width * (y % width + width * (z % width));
}

__device__ bool RowUsesCacheColor(BlockScratch s, uint32_t slot, uint32_t cache_color) {
    return cache_color == kAllCacheColors ||
        (s.row_cache_colors[size_t{slot} * s.color_words + cache_color / 32u] &
         (uint32_t{1u} << (cache_color % 32u))) != 0u;
}

__global__ void InitializeOwnerCacheColorsKernel(
    DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t dense_first = p.total_particle_count;
    const uint32_t grid_first = dense_first + p.total_body_count + p.articulation_count;
    for (uint32_t owner = blockIdx.x * blockDim.x + threadIdx.x; owner < s.blocks;
         owner += gridDim.x * blockDim.x) {
        uint32_t color = kNoCacheColor;
        if (owner < dense_first) {
            const uint32_t local = owner % (p.total_particle_count / p.env_count);
            const bool elastic = local >= p.vertex_blocks.begin &&
                local - p.vertex_blocks.begin < p.vertex_blocks.vertices;
            if (!elastic && local >= p.grid_particles_per_env && data.particle_inv_mass[owner] > 0.0f)
                color = p.vertex_blocks.colors;
        } else if (owner < grid_first) {
            color = p.vertex_blocks.colors + 1u;
        } else {
            const uint32_t node = owner - grid_first;
            if (data.grid_inv_mass[node] > 0.0f)
                color = p.vertex_blocks.colors + 2u + GridNodeColor(p, node);
        }
        s.owner_cache_color[owner] = color;
    }
}

__global__ void InitializeVertexCacheColorsKernel(
    ModelView model, DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint64_t items = uint64_t{p.vertex_blocks.dynamic_vertices} * p.env_count;
    for (size_t item = size_t{blockIdx.x} * blockDim.x + threadIdx.x; item < items;
         item += size_t{gridDim.x} * blockDim.x) {
        const uint32_t env = static_cast<uint32_t>(item / p.vertex_blocks.dynamic_vertices);
        const uint32_t at = static_cast<uint32_t>(item % p.vertex_blocks.dynamic_vertices);
        uint32_t first = 0u, last = p.vertex_blocks.colors;
        while (first < last) {
            const uint32_t middle = first + (last - first) / 2u;
            const size_t segment = size_t{middle} * 2u;
            const uint64_t end = uint64_t{model.vbd_color_segments[segment]} +
                model.vbd_color_segments[segment + 1u];
            if (at >= end) first = middle + 1u;
            else last = middle;
        }
        if (first == p.vertex_blocks.colors) continue;
        const uint32_t vertex = model.vbd_color_vertices[at];
        const uint32_t local = p.vertex_blocks.begin + vertex;
        const uint32_t particle = env * p.vertex_blocks.particles_per_env + local;
        if (local >= p.grid_particles_per_env && data.particle_inv_mass[particle] > 0.0f)
            s.owner_cache_color[particle] = first;
    }
}

__device__ void RecordBlockFailure(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                   uint32_t owner, nk::BlockSolveFailure reason, uint32_t row = ~0u,
                                   const double* equation = nullptr) {
    if (owner >= s.blocks) return;
    if (atomicCAS(s.failure_reasons + owner, 0u, static_cast<uint32_t>(reason)) == 0u) {
        s.failure_rows[owner] = row;
        s.failure_substeps[owner] = p.substep_index;
        for (uint32_t column = 0u; column < nk::kBlockSolveFailureEquationColumnCount; ++column)
            s.failure_equations[size_t{owner} * nk::kBlockSolveFailureEquationColumnCount + column] =
                equation != nullptr ? equation[column] : 0.0;
        s.failure_equations[size_t{owner} * nk::kBlockSolveFailureEquationColumnCount +
            static_cast<uint32_t>(nk::BlockSolveFailureEquationColumn::Iteration)] = s.diagnostic_iteration;
    }
    uint32_t env = 0u;
    if (owner < p.total_particle_count) env = owner / (p.total_particle_count / p.env_count);
    else if (owner < p.total_particle_count + p.total_body_count)
        env = (owner - p.total_particle_count) / (p.total_body_count / p.env_count);
    else if (owner < p.total_particle_count + p.total_body_count + p.articulation_count)
        env = (owner - p.total_particle_count - p.total_body_count) / (p.articulation_count / p.env_count);
    else env = (owner - p.total_particle_count - p.total_body_count - p.articulation_count) /
                   (p.total_grid_count / p.env_count);
    atomicOr(data.env_status + env, kEnvStatusSolverFailure);
}

template <typename Visitor>
__device__ void VisitOwners(const NkRow& row, PointMassView points,
                            const BlockDescentSolveParams& p, Visitor visit) {
    const NkRowSide sides[2] = {row.a, row.b};
    for (uint32_t side = 0u; side < 2u; ++side) {
        const uint32_t count = points.Count(sides[side]);
        for (uint32_t term = 0u; term < count; ++term) {
            const auto entry = points.At(sides[side], term);
            const uint32_t owner = Owner(p, entry.kind, entry.index);
            if (owner == ~0u) continue;
            bool repeated = false;
            for (uint32_t prior = 0u; prior < term; ++prior) {
                const auto entry_before = points.At(sides[side], prior);
                repeated |= entry_before.kind == entry.kind && entry_before.index == entry.index;
            }
            if (side == 1u) {
                for (uint32_t other = 0u; other < points.Count(sides[0]); ++other) {
                    const auto prior = points.At(sides[0], other);
                    repeated |= prior.kind == entry.kind && prior.index == entry.index;
                }
            }
            if (!repeated) visit(owner);
        }
    }
}

__device__ float InertialPointResponse(const PointMassView::Contribution& term,
                                      const DataView& data, const BlockDescentSolveParams& p,
                                      PointMassView points) {
    const float* mass = points.InverseMass(term.kind);
    if (mass == nullptr || !(mass[term.index] > 0.0f)) return 0.0f;
    float response = mass[term.index];
    if (term.kind == kNkSideParticle && p.vertex_blocks.vertices > 0u) {
        const uint32_t local = term.index % p.vertex_blocks.particles_per_env;
        if (local >= p.vertex_blocks.begin && local - p.vertex_blocks.begin < p.vertex_blocks.vertices) {
            const uint32_t slot = term.index / p.vertex_blocks.particles_per_env * p.vertex_blocks.vertices +
                                  local - p.vertex_blocks.begin;
            response = 1.0f / (data.vbd_inertia[slot] * p.dt * p.dt);
        }
    }
    return response;
}

__device__ float SideResponse(const NkRowSide& side, const DataView& data,
                              const BlockDescentSolveParams& p, PointMassView points,
                              uint32_t slot, uint32_t endpoint) {
    if (PointMassView::IsPointSide(side.kind)) {
        float response = 0.0f;
        for (uint32_t i = 0u; i < points.Count(side); ++i) {
            const auto term = points.At(side, i);
            bool repeated = false;
            for (uint32_t prior = 0u; prior < i; ++prior) {
                const auto before = points.At(side, prior);
                repeated |= before.kind == term.kind && before.index == term.index;
            }
            if (repeated) continue;
            Vec3 jacobian = term.jacobian;
            for (uint32_t after = i + 1u; after < points.Count(side); ++after) {
                const auto other = points.At(side, after);
                if (other.kind == term.kind && other.index == term.index) jacobian += other.jacobian;
            }
            response += InertialPointResponse(term, data, p, points) * jacobian.Dot(jacobian);
        }
        return response;
    }
    if (side.kind == kNkSideRigid)
        return data.body_inv_mass[side.index] * side.jlin.Dot(side.jlin) +
               side.jang.Dot(data.body_world_inv_inertia[side.index].Multiply(side.jang));
    if (side.kind == kNkSideArtic) {
        const auto* jacobians = static_cast<const float*>(endpoint == 0u
            ? data.chain_jacobian : data.chain_jacobian_b);
        const auto* response = static_cast<const float*>(endpoint == 0u
            ? data.row_minv_jt : data.row_minv_jt_b);
        double value = 0.0;
        for (uint32_t dof = 0u; dof < p.max_dof; ++dof)
            value += double(jacobians[size_t{slot} * p.max_dof + dof]) *
                response[size_t{slot} * p.max_dof + dof];
        return static_cast<float>(value);
    }
    return 0.0f;
}

__device__ float RowVelocity(const NkRow& row, uint32_t slot, const DataView& data,
                             BlockDescentSolveParams p, PointMassView points,
                             const float* articulation_velocity, const float* rigid_velocity) {
    float velocity = 0.0f;
    const NkRowSide sides[2] = {row.a, row.b};
    for (uint32_t endpoint = 0u; endpoint < 2u; ++endpoint) {
        const NkRowSide side = sides[endpoint];
        if (PointMassView::IsPointSide(side.kind)) velocity += points.RowVelocity(side);
        else if (side.kind == kNkSideRigid) {
            const size_t base = size_t{side.index} * kRigidBlockDof;
            const Vec3 linear = rigid_velocity != nullptr
                ? Vec3{rigid_velocity[base], rigid_velocity[base + 1u], rigid_velocity[base + 2u]}
                : data.body_linear_velocity[side.index];
            const Vec3 angular = rigid_velocity != nullptr
                ? Vec3{rigid_velocity[base + 3u], rigid_velocity[base + 4u], rigid_velocity[base + 5u]}
                : data.body_angular_velocity[side.index];
            velocity += side.jlin.Dot(linear) + side.jang.Dot(angular);
        }
        else if (side.kind == kNkSideArtic) {
            const auto* jacobian = static_cast<const float*>(endpoint == 0u
                ? data.chain_jacobian : data.chain_jacobian_b) + size_t{slot} * p.max_dof;
            const float* u = articulation_velocity + size_t{side.index} * p.max_dof;
            double value = 0.0;
            for (uint32_t dof = 0u; dof < p.max_dof; ++dof)
                value += double(jacobian[dof]) * u[dof];
            velocity += static_cast<float>(value);
        }
    }
    return velocity;
}

__device__ float ArticulationJacobian(DataView data, BlockDescentSolveParams p,
                                      uint32_t slot, uint32_t articulation, uint32_t dof) {
    const NkRow row = reinterpret_cast<const NkRow*>(data.urows)[slot];
    float value = 0.0f;
    const size_t at = size_t{slot} * p.max_dof + dof;
    if (row.a.kind == kNkSideArtic && row.a.index == articulation)
        value += static_cast<const float*>(data.chain_jacobian)[at];
    if (row.b.kind == kNkSideArtic && row.b.index == articulation)
        value += static_cast<const float*>(data.chain_jacobian_b)[at];
    return value;
}

__device__ float RigidJacobian(DataView data, uint32_t slot, uint32_t body, uint32_t dof) {
    const NkRow row = reinterpret_cast<const NkRow*>(data.urows)[slot];
    float value = 0.0f;
    const NkRowSide sides[2] = {row.a, row.b};
    for (const NkRowSide& side : sides) {
        if (side.kind != kNkSideRigid || side.index != body) continue;
        const Vec3 jacobian = dof < 3u ? side.jlin : side.jang;
        const uint32_t component = dof % 3u;
        value += component == 0u ? jacobian.x : component == 1u ? jacobian.y : jacobian.z;
    }
    return value;
}

__device__ Vec3 PointJacobian(const NkRow& row, PointMassView points, uint32_t kind,
                              uint32_t index) {
    Vec3 result{};
    const NkRowSide sides[2] = {row.a, row.b};
    for (const NkRowSide& side : sides) {
        if (!PointMassView::IsPointSide(side.kind)) continue;
        const uint32_t count = points.Count(side);
        uint32_t first = 0u;
        uint32_t last = count;
        while (first < last) {
            const uint32_t middle = first + (last - first) / 2u;
            const auto term = points.At(side, middle);
            if (term.kind < kind || (term.kind == kind && term.index < index))
                first = middle + 1u;
            else
                last = middle;
        }
        for (uint32_t i = first; i < count; ++i) {
            const auto term = points.At(side, i);
            if (term.kind != kind || term.index != index) break;
            result += term.jacobian;
        }
    }
    return result;
}

__device__ Vec3 ParticleJacobian(const NkRow& row, PointMassView points, uint32_t particle) {
    return PointJacobian(row, points, kNkSideParticle, particle);
}

__device__ Vec3 LoadPointIncidenceJacobian(BlockScratch s, size_t at, uint32_t axis) {
    const uint64_t first = uint64_t{axis} * kPointJacobianComponents * s.incidence_capacity + at;
    return {s.point_jacobian[first], s.point_jacobian[first + s.incidence_capacity],
            s.point_jacobian[first + 2u * s.incidence_capacity]};
}

__global__ void CachePointIncidenceJacobiansKernel(
    DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const uint32_t lane = threadIdx.x % warpSize;
    const size_t warp = (size_t{blockIdx.x} * blockDim.x + threadIdx.x) / warpSize;
    const size_t stride = size_t{gridDim.x} * blockDim.x / warpSize;
    const uint64_t count = uint64_t{p.total_particle_count} + p.total_grid_count;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const auto points = PointMasses(data);
    for (size_t item = warp; item < count; item += stride) {
        const uint32_t kind = item < p.total_particle_count ? kNkSideParticle : kNkSideGrid;
        const uint32_t index = static_cast<uint32_t>(item < p.total_particle_count
            ? item : item - p.total_particle_count);
        const uint32_t owner = Owner(p, kind, index);
        if (s.owner_cache_color[owner] == kNoCacheColor) continue;
        for (size_t at = size_t{s.offsets[owner]} + lane; at < s.offsets[owner + 1u]; at += warpSize) {
            const uint32_t slot = s.incidence[at];
            const NkRow head = rows[slot];
            const uint32_t axes = (head.flags & nk::nk_row_flags::kBlockNormal) &&
                !(head.flags & nk::nk_row_flags::kMaterialBlock) ? kPointJacobianAxes : 1u;
            for (uint32_t axis = 0u; axis < kPointJacobianAxes; ++axis) {
                const Vec3 jacobian = axis < axes
                    ? PointJacobian(rows[slot + axis * head.group_normal_count], points, kind, index)
                    : Vec3{};
                const uint64_t first = uint64_t{axis} * kPointJacobianComponents *
                    s.incidence_capacity + at;
                s.point_jacobian[first] = jacobian.x;
                s.point_jacobian[first + s.incidence_capacity] = jacobian.y;
                s.point_jacobian[first + 2u * s.incidence_capacity] = jacobian.z;
            }
        }
    }
}

__device__ double RowInertialResponse(const NkRow& row, const DataView& data,
                                     const BlockDescentSolveParams& p, PointMassView points,
                                     uint32_t slot) {
    double response = 0.0;
    bool valid = true;
    VisitOwners(row, points, p, [&](uint32_t owner) {
        double contribution = 0.0;
        if (owner < p.total_particle_count ||
            owner >= p.total_particle_count + p.total_body_count + p.articulation_count) {
            const uint32_t kind = owner < p.total_particle_count ? kNkSideParticle : kNkSideGrid;
            const uint32_t index = kind == kNkSideParticle ? owner
                : owner - p.total_particle_count - p.total_body_count - p.articulation_count;
            const Vec3 jacobian = PointJacobian(row, points, kind, index);
            const double norm = double(jacobian.x) * jacobian.x + double(jacobian.y) * jacobian.y +
                                double(jacobian.z) * jacobian.z;
            contribution = double(InertialPointResponse(
                PointMassView::Contribution{kind, index, {}}, data, p, points)) * norm;
        } else if (owner < p.total_particle_count + p.total_body_count) {
            const uint32_t body = owner - p.total_particle_count;
            Vec3 linear{}, angular{};
            const NkRowSide sides[2] = {row.a, row.b};
            for (const NkRowSide& side : sides) {
                if (side.kind != kNkSideRigid || side.index != body) continue;
                linear += side.jlin;
                angular += side.jang;
            }
            const double x = angular.x, y = angular.y, z = angular.z;
            const SymmetricMat3 inverse = data.body_world_inv_inertia[body];
            contribution = double(data.body_inv_mass[body]) *
                (double(linear.x) * linear.x + double(linear.y) * linear.y + double(linear.z) * linear.z) +
                x * (double(inverse.xx) * x + double(inverse.xy) * y + double(inverse.xz) * z) +
                y * (double(inverse.xy) * x + double(inverse.yy) * y + double(inverse.yz) * z) +
                z * (double(inverse.xz) * x + double(inverse.yz) * y + double(inverse.zz) * z);
        } else {
            const uint32_t articulation = owner - p.total_particle_count - p.total_body_count;
            const NkRowSide sides[2] = {row.a, row.b};
            for (uint32_t dof = 0u; dof < p.max_dof; ++dof) {
                const size_t at = size_t{slot} * p.max_dof + dof;
                double jacobian = 0.0, inverse_response = 0.0;
                for (uint32_t endpoint = 0u; endpoint < 2u; ++endpoint) {
                    if (sides[endpoint].kind != kNkSideArtic || sides[endpoint].index != articulation) continue;
                    jacobian += static_cast<const float*>(endpoint == 0u
                        ? data.chain_jacobian : data.chain_jacobian_b)[at];
                    inverse_response += static_cast<const float*>(endpoint == 0u
                        ? data.row_minv_jt : data.row_minv_jt_b)[at];
                }
                contribution += jacobian * inverse_response;
            }
        }
        valid &= contribution >= 0.0 && isfinite(contribution);
        response += contribution;
    });
    return valid && isfinite(response) ? response : nan("");
}

__global__ void CountIncidenceKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const auto points = PointMasses(data);
    for (uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x; slot < s.rows;
         slot += gridDim.x * blockDim.x) {
        // Contact rows past the bound are empty and keep the zero coarse flag written earlier.
        const uint32_t env_row = slot % p.rows_per_env;
        if (env_row < p.contact_rows_per_env && env_row >= ContactRowBound(
                data.contact_row_extent, slot / p.rows_per_env, p.contact_rows_per_env)) continue;
        // Every row that can be live passes here, so the coarse selection reads one byte per row.
        if (p.total_particle_count > 0u) s.coarse_flags[slot] = CoarseRowSelect{rows}(slot);
        const uint32_t flags = rows[slot].flags;
        if (!(flags & nk::nk_row_flags::kActive) || (flags & nk::nk_row_flags::kBlockTangent))
            continue;
        uint32_t* cache_colors = s.row_cache_colors + size_t{slot} * s.color_words;
        for (uint32_t word = 0u; word < s.color_words; ++word) cache_colors[word] = 0u;
        const NkRow row = rows[slot];
        if (flags & nk::nk_row_flags::kMaterialBlock) {
            const uint32_t span = p.material_cells_per_env * nk::kMpmStressRowsPerCell;
            const uint32_t first = p.rows_per_env - span;
            const uint32_t local = slot % p.rows_per_env;
            bool valid = span > 0u && local >= first &&
                (local - first) % nk::kMpmStressRowsPerCell == 0u &&
                row.a.kind == kNkSidePointEndpoint && row.b.kind == kNkSideStatic;
            if (valid) {
                for (uint32_t term = 0u; term < points.Count(row.a); ++term)
                    valid &= points.At(row.a, term).kind == kNkSideGrid;
                const float compliance = rows[slot + 1u].compliance_alpha;
                valid &= compliance >= 0.0f && compliance <= FLT_MAX;
                for (uint32_t k = 2u; k < nk::kMpmStressRowsPerCell; ++k)
                    valid &= rows[slot + k].compliance_alpha == compliance;
            }
            if (!valid) {
                atomicOr(data.env_status + slot / p.rows_per_env, kEnvStatusSolverFailure);
                continue;
            }
        }
        const double response = RowInertialResponse(row, data, p, points, slot);
        const double penalty = response > 0.0 ? double(p.penalty_scale) / response : 0.0;
        const bool valid_penalty = response == 0.0 ||
            (penalty > 0.0 && penalty <= double(FLT_MAX) && static_cast<float>(penalty) > 0.0f);
        if (!(response >= 0.0 && isfinite(response)) || !valid_penalty) {
            atomicOr(data.env_status + slot / p.rows_per_env, kEnvStatusSolverFailure);
            VisitOwners(row, points, p, [&](uint32_t owner) {
                RecordBlockFailure(data, p, s, owner, nk::BlockSolveFailure::InvalidRow, slot);
            });
            s.penalty[slot] = nanf("");
        } else {
            s.penalty[slot] = static_cast<float>(penalty);
        }
        const float scale = (flags & nk::nk_row_flags::kMaterialBlock) != 0u
            ? 1.0f : 1.0f / (1.0f + data.row_damping[slot] * p.dt);
        s.scalar_response[slot] = nk::augmented::ComputeScalarResponse(
            s.penalty[slot], row.compliance_alpha * scale);
        VisitOwners(row, points, p, [&](uint32_t owner) {
            atomicAdd(s.counts + owner, 1u);
            const uint32_t color = s.owner_cache_color[owner];
            if (color != kNoCacheColor)
                cache_colors[color / 32u] |= uint32_t{1u} << (color % 32u);
        });
        s.active_rows[atomicAdd(s.active_count, 1u)] = slot;
    }
}

// One warp sorts each segment of at most kWarpSortLimit slots by ranking every key against all of
// them, ties by position, so the result equals any full sort of the segment.
__global__ void SortSmallIncidenceKernel(BlockScratch s) {
    const uint32_t lane = threadIdx.x % 32u;
    for (uint32_t segment = (blockIdx.x * blockDim.x + threadIdx.x) / 32u; segment < s.blocks;
         segment += gridDim.x * blockDim.x / 32u) {
        const uint32_t begin = IncidenceOffset(s.offsets, s.incidence_capacity, segment);
        const uint32_t end = IncidenceOffset(s.offsets, s.incidence_capacity, segment + 1u);
        const uint32_t count = end - begin;
        if (count > kWarpSortLimit) continue;
        uint32_t keys[kWarpSortKeysPerLane], ranks[kWarpSortKeysPerLane];
#pragma unroll
        for (uint32_t k = 0u; k < kWarpSortKeysPerLane; ++k) {
            const uint32_t at = k * 32u + lane;
            keys[k] = at < count ? s.incidence[begin + at] : 0u;
            ranks[k] = 0u;
        }
#pragma unroll
        for (uint32_t group = 0u; group < kWarpSortKeysPerLane; ++group) {
            if (group * 32u >= count) break;
            for (uint32_t source = 0u; source < 32u && group * 32u + source < count; ++source) {
                const uint32_t other = __shfl_sync(~0u, keys[group], source);
                const uint32_t other_at = group * 32u + source;
#pragma unroll
                for (uint32_t k = 0u; k < kWarpSortKeysPerLane; ++k)
                    ranks[k] += other < keys[k] || (other == keys[k] && other_at < k * 32u + lane);
            }
        }
#pragma unroll
        for (uint32_t k = 0u; k < kWarpSortKeysPerLane; ++k) {
            const uint32_t at = k * 32u + lane;
            if (at < count) s.sorted_incidence[begin + ranks[k]] = keys[k];
        }
    }
}

__global__ void FillIncidenceKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const auto points = PointMasses(data);
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.active_rows[item];
        VisitOwners(rows[slot], points, p, [&](uint32_t owner) {
            const uint32_t at = s.offsets[owner] + atomicAdd(s.counts + owner, 1u);
            if (at < s.incidence_capacity) s.incidence[at] = slot;
            else atomicOr(data.env_status + slot / p.rows_per_env, kEnvStatusSolverFailure);
        });
    }
}

__device__ VertexBlockView Vertices(ModelView model, DataView data, BlockDescentSolveParams p) {
    VertexBlockView b;
    b.elements = model.vbd_elements;
    b.membrane_start = data.vbd_membrane_start;
    b.offsets = model.vbd_incidence_offsets;
    b.incidence = model.vbd_incidence;
    b.color_vertices = model.vbd_color_vertices;
    b.color_segments = model.vbd_color_segments;
    b.start = data.particle_prev_pos;
    b.free_rate = data.vbd_free_rate;
    b.inertia = data.vbd_inertia;
    b.effective_step = data.vbd_step;
    b.solver_audit = p.measure_vertex_audit != 0u ? data.vbd_solve_audit : nullptr;
    b.env_status = data.env_status;
    b.layout = p.vertex_blocks;
    b.dt = p.dt;
    return b;
}

__device__ void AddOuter(SymmetricMat3& h, Vec3 j, float scale) {
    h.xx += scale * j.x * j.x;
    h.yy += scale * j.y * j.y;
    h.zz += scale * j.z * j.z;
    h.xy += scale * j.x * j.y;
    h.xz += scale * j.x * j.z;
    h.yz += scale * j.y * j.z;
}

struct LocalTerm {
    Vec3 jacobian[3];
    Vec3 residual;
    Vec3 dual;
    float penalty;
    float compliance;
    float lower;
    float upper;
    float mu_first;
    float mu_second;
    float normal_bound;
    bool contact;
    nk::augmented::ScalarResponse response;
};

__device__ PointMassView RowPointMasses(DataView data, BlockScratch s, bool snapshot) {
    auto points = PointMasses(data);
    if (snapshot) {
        points.particle_velocity = s.particle_snapshot;
        points.grid_velocity = s.grid_snapshot;
    }
    return points;
}

__device__ float RowStateScale(DataView data, BlockDescentSolveParams p, uint32_t slot) {
    return 1.0f / (1.0f + data.row_damping[slot] * p.dt);
}

// Residual and dual of one axis of the row group headed by `slot`. Explicit roundings keep the
// residual identical in every kernel that inlines it.
__device__ void AugmentedRowAxis(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                 const NkRow& normal, uint32_t slot, uint32_t axis, float scale,
                                 bool snapshot, float* residual, float* dual) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint32_t row_slot = slot + axis * normal.group_normal_count;
    const NkRow row = rows[row_slot];
    *residual = __fsub_rn(__fmul_rn(__fmul_rn(row.rhs, p.dt), axis == 0u ? scale : 1.0f),
                RowVelocity(row, row_slot, data, p, RowPointMasses(data, s, snapshot),
                    snapshot ? s.articulation_snapshot : data.qdot_flat,
                    snapshot ? s.rigid_snapshot : nullptr));
    *dual = data.lambda[row_slot];
}

__device__ float AugmentedRowBound(BlockScratch s, const NkRow& normal, uint32_t slot, float scale,
                                   const AugmentedRowState& state) {
    return nk::augmented::EvaluateScalarImpulse(state.dual.x, s.penalty[slot], state.residual.x,
        normal.compliance_alpha * scale, normal.lower, normal.upper, s.scalar_response[slot]).impulse;
}

__device__ AugmentedRowState ComputeAugmentedRowState(
    DataView data, BlockDescentSolveParams p, BlockScratch s, uint32_t slot, bool snapshot) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const NkRow normal = rows[slot];
    AugmentedRowState state{};
    const bool contact = (normal.flags & nk::nk_row_flags::kBlockNormal) != 0u;
    const float scale = RowStateScale(data, p, slot);
    float* residual[] = {&state.residual.x, &state.residual.y, &state.residual.z};
    float* dual[] = {&state.dual.x, &state.dual.y, &state.dual.z};
    for (uint32_t axis = 0u; axis < (contact ? 3u : 1u); ++axis)
        AugmentedRowAxis(data, p, s, normal, slot, axis, scale, snapshot, residual[axis], dual[axis]);
    state.normal_bound = AugmentedRowBound(s, normal, slot, scale, state);
    return state;
}

__device__ LocalTerm LoadLocalTerm(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                   uint32_t slot, uint32_t particle, bool snapshot) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const NkRow normal = rows[slot];
    auto points = PointMasses(data);
    if (snapshot) {
        points.particle_velocity = s.particle_snapshot;
        points.grid_velocity = s.grid_snapshot;
    }
    LocalTerm t{};
    t.contact = (normal.flags & nk::nk_row_flags::kBlockNormal) != 0u;
    t.penalty = s.penalty[slot];
    t.lower = normal.lower;
    t.upper = normal.upper;
    t.mu_first = normal.mu;
    t.mu_second = normal.friction_secondary;
    const float scale = 1.0f / (1.0f + data.row_damping[slot] * p.dt);
    t.compliance = normal.compliance_alpha * scale;
    for (uint32_t axis = 0u; axis < (t.contact ? 3u : 1u); ++axis) {
        const uint32_t row_slot = slot + axis * normal.group_normal_count;
        const NkRow row = rows[row_slot];
        t.jacobian[axis] = particle != ~0u ? ParticleJacobian(row, points, particle) : Vec3{};
    }
    const AugmentedRowState state = snapshot ? s.row_state[slot]
        : ComputeAugmentedRowState(data, p, s, slot, false);
    t.residual = state.residual;
    t.dual = state.dual;
    t.normal_bound = state.normal_bound;
    t.response = s.scalar_response[slot];
    return t;
}

__device__ LocalTerm LoadPointLocalTerm(DataView data, BlockDescentSolveParams p,
                                        BlockScratch s, size_t at) {
    LocalTerm term = LoadLocalTerm(data, p, s, s.incidence[at], ~0u, true);
    for (uint32_t axis = 0u; axis < kPointJacobianAxes; ++axis)
        term.jacobian[axis] = LoadPointIncidenceJacobian(s, at, axis);
    return term;
}

__device__ double EvaluateAugmentedResidual(const LocalTerm& t, Vec3 row_move,
                                            Vec3* impulse, SymmetricMat3* curvature) {
    const float residual = t.residual.x - row_move.x;
    const auto n = nk::augmented::EvaluateScalar(t.dual.x, t.penalty, residual,
                                                t.compliance, t.lower, t.upper, t.response);
    if (impulse != nullptr) {
        *impulse = {n.impulse, 0.0f, 0.0f};
        *curvature = {n.curvature, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }
    if (!t.contact) return n.potential;
    const Vec3 tangent_residual{0.0f, t.residual.y - row_move.y,
                               t.residual.z - row_move.z};
    const auto f = nk::augmented::EvaluateTangent(t.dual, t.penalty, tangent_residual,
                                                  t.normal_bound, t.mu_first, t.mu_second);
    if (impulse != nullptr) {
        impulse->y = f.impulse.y;
        impulse->z = f.impulse.z;
        curvature->yy = f.curvature.yy;
        curvature->zz = f.curvature.zz;
        curvature->yz = f.curvature.yz;
    }
    return n.potential + f.potential;
}

__device__ double EvaluateLocalResidual(const LocalTerm& t, Vec3 row_move) {
    return EvaluateAugmentedResidual(t, row_move, nullptr, nullptr);
}

// Folds a row's impulse and its curvature (xx, yy, zz, yz used) into a point block.
__device__ void AddLocalResponse(const Vec3* jacobian, Vec3 f, const SymmetricMat3& c,
                                 Vec3* impulse, SymmetricMat3* curvature) {
    *impulse += jacobian[0] * f.x + jacobian[1] * f.y + jacobian[2] * f.z;
    AddOuter(*curvature, jacobian[0], c.xx);
    AddOuter(*curvature, jacobian[1], c.yy - c.yz);
    AddOuter(*curvature, jacobian[2], c.zz - c.yz);
    AddOuter(*curvature, jacobian[1] + jacobian[2], c.yz);
}

__device__ double EvaluateLocal(const LocalTerm& t, Vec3 move, Vec3* impulse,
                                 SymmetricMat3* curvature) {
    Vec3 f{};
    SymmetricMat3 c{};
    const double potential = EvaluateAugmentedResidual(t,
        {t.jacobian[0].Dot(move), t.jacobian[1].Dot(move), t.jacobian[2].Dot(move)},
        impulse != nullptr ? &f : nullptr, impulse != nullptr ? &c : nullptr);
    if (impulse != nullptr) AddLocalResponse(t.jacobian, f, c, impulse, curvature);
    return potential;
}

// EvaluateAugmentedResidual's impulse and curvature without the potential; the cone runs in float.
__device__ void EvaluateAugmentedResponse(const LocalTerm& t, Vec3 row_move, Vec3* impulse,
                                          SymmetricMat3* curvature) {
    const auto n = nk::augmented::EvaluateScalarImpulse(t.dual.x, t.penalty, t.residual.x - row_move.x,
        t.compliance, t.lower, t.upper, t.response);
    *impulse = {n.impulse, 0.0f, 0.0f};
    *curvature = {n.curvature, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (!t.contact) return;
    const Vec3 tangent_residual{0.0f, t.residual.y - row_move.y, t.residual.z - row_move.z};
    const auto f = nk::augmented::EvaluateTangentImpulse(t.dual, t.penalty, tangent_residual,
                                                         t.normal_bound, t.mu_first, t.mu_second);
    impulse->y = f.impulse.y;
    impulse->z = f.impulse.z;
    curvature->yy = f.curvature.yy;
    curvature->zz = f.curvature.zz;
    curvature->yz = f.curvature.yz;
}

__device__ void EvaluateLocalResponse(const LocalTerm& t, Vec3 move, Vec3* impulse,
                                      SymmetricMat3* curvature) {
    Vec3 f{};
    SymmetricMat3 c{};
    EvaluateAugmentedResponse(t,
        {t.jacobian[0].Dot(move), t.jacobian[1].Dot(move), t.jacobian[2].Dot(move)}, &f, &c);
    AddLocalResponse(t.jacobian, f, c, impulse, curvature);
}

// EvaluateLocal(t, move) - EvaluateLocal(t, 0), integrated along the move instead of subtracted.
__device__ float EvaluateLocalChange(const LocalTerm& t, Vec3 move) {
    const Vec3 drop{t.jacobian[0].Dot(move), t.jacobian[1].Dot(move), t.jacobian[2].Dot(move)};
    float change = nk::augmented::ScalarPotentialChange(t.dual.x, t.penalty, t.residual.x, drop.x,
        t.compliance, t.lower, t.upper, t.response);
    if (t.contact)
        change += nk::augmented::TangentPotentialChange(t.dual, t.penalty, t.residual, drop,
                                                        t.normal_bound, t.mu_first, t.mu_second);
    return change;
}

// One point incidence's row response at the block rate.
struct PointRowSample {
    Vec3 impulse;
    float curvature[4];
};

__device__ PointRowSample SamplePointRow(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                         size_t at) {
    const LocalTerm term = LoadPointLocalTerm(data, p, s, at);
    Vec3 f{};
    SymmetricMat3 c{};
    PointRowSample sample{};
    EvaluateAugmentedResponse(term,
        {term.jacobian[0].Dot(Vec3{}), term.jacobian[1].Dot(Vec3{}), term.jacobian[2].Dot(Vec3{})},
        &f, &c);
    sample.impulse = f;
    sample.curvature[0] = c.xx;
    sample.curvature[1] = c.yy;
    sample.curvature[2] = c.zz;
    sample.curvature[3] = c.yz;
    return sample;
}

__device__ void AddPointRowSample(BlockScratch s, size_t at, const PointRowSample& sample,
                                  Vec3* impulse, SymmetricMat3* curvature) {
    Vec3 jacobian[kPointJacobianAxes];
    for (uint32_t axis = 0u; axis < kPointJacobianAxes; ++axis)
        jacobian[axis] = LoadPointIncidenceJacobian(s, at, axis);
    const SymmetricMat3 c{sample.curvature[0], sample.curvature[1], sample.curvature[2],
                          0.0f, 0.0f, sample.curvature[3]};
    AddLocalResponse(jacobian, sample.impulse, c, impulse, curvature);
}

__device__ bool Finite(Vec3 value) {
    return fabsf(value.x) <= FLT_MAX && fabsf(value.y) <= FLT_MAX && fabsf(value.z) <= FLT_MAX;
}

#include "phi/backend_cuda/ops/block_descent_dense.cuh"
#include "phi/backend_cuda/ops/block_descent_articulation.cuh"
#include "phi/backend_cuda/ops/block_descent_rigid.cuh"
#include "phi/backend_cuda/ops/block_descent_material.cuh"
#include "phi/backend_cuda/ops/block_descent_grid.cuh"
#include "phi/backend_cuda/ops/block_descent_coarse.cuh"

__global__ void PackBlockRigidSnapshotKernel(DataView data, BlockDescentSolveParams p,
                                            BlockScratch s) {
    for (uint32_t body = blockIdx.x * blockDim.x + threadIdx.x; body < p.total_body_count;
         body += gridDim.x * blockDim.x) {
        const size_t base = size_t{body} * kRigidBlockDof;
        const Vec3 linear = data.body_linear_velocity[body];
        const Vec3 angular = data.body_angular_velocity[body];
        s.rigid_snapshot[base] = linear.x;
        s.rigid_snapshot[base + 1u] = linear.y;
        s.rigid_snapshot[base + 2u] = linear.z;
        s.rigid_snapshot[base + 3u] = angular.x;
        s.rigid_snapshot[base + 4u] = angular.y;
        s.rigid_snapshot[base + 5u] = angular.z;
    }
}

__global__ void PackBlockArticulationVelocityKernel(ArticulationDeviceState state, DataView data,
                                                   BlockDescentSolveParams p, BlockScratch s) {
    const size_t count = size_t{p.articulation_count} * p.max_dof;
    for (size_t at = size_t{blockIdx.x} * blockDim.x + threadIdx.x; at < count;
         at += size_t{gridDim.x} * blockDim.x) {
        const uint32_t articulation = static_cast<uint32_t>(at / p.max_dof);
        uint32_t link = 0u, component = 0u;
        const float value = ArticulationDofLocation(state, articulation,
                static_cast<uint32_t>(at % p.max_dof), &link, &component)
            ? (component != ~0u ? state.link_velocity[link].v[component] : state.qdot[link]) : 0.0f;
        data.qdot_flat[at] = value;
        s.articulation_free[at] = value;
        s.articulation_snapshot[at] = value;
    }
}

// Caches a row's assembled state with, if asked, its potential.
__device__ void CacheAugmentedRow(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                  uint32_t slot, const AugmentedRowState& state, bool potential) {
    s.row_state[slot] = state;
    if (!potential) return;
    s.row_state[slot].potential = EvaluateLocalResidual(LoadLocalTerm(data, p, s, slot, ~0u, true), {});
}

// Rows too few to give every warp scheduler a warp are latency-bound, so adjacent lanes evaluate one
// row's axes and the group's first lane caches it; otherwise each lane caches whole rows.
constexpr uint32_t kRowAxisLanes = 4u;

__global__ void CacheAugmentedRowsKernel(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                        uint32_t cache_color, uint32_t spread_rows) {
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t count = *s.active_count;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    // Particle descents integrate potential changes and dense descents search slopes; only grid
    // descents read the cached potential.
    const bool potential = cache_color > p.vertex_blocks.colors + 1u;
    if (count > spread_rows) {
        for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < count; item += threads) {
            const uint32_t slot = s.active_rows[item];
            if (!RowUsesCacheColor(s, slot, cache_color)) continue;
            const NkRow row = rows[slot];
            if (row.flags & nk::nk_row_flags::kMaterialBlock) continue;
            CacheAugmentedRow(data, p, s, slot, ComputeAugmentedRowState(data, p, s, slot, true), potential);
        }
        return;
    }
    const uint32_t axis = threadIdx.x % kRowAxisLanes;
    const uint32_t group_mask = ((1u << kRowAxisLanes) - 1u) << (threadIdx.x % warpSize - axis);
    for (uint32_t item = (blockIdx.x * blockDim.x + threadIdx.x) / kRowAxisLanes; item < count;
         item += threads / kRowAxisLanes) {
        const uint32_t slot = s.active_rows[item];
        if (!RowUsesCacheColor(s, slot, cache_color)) continue;
        const NkRow row = rows[slot];
        if (row.flags & nk::nk_row_flags::kMaterialBlock) continue;
        const bool contact = (row.flags & nk::nk_row_flags::kBlockNormal) != 0u;
        const float scale = RowStateScale(data, p, slot);
        float residual = 0.0f, dual = 0.0f;
        if (axis < (contact ? 3u : 1u))
            AugmentedRowAxis(data, p, s, row, slot, axis, scale, true, &residual, &dual);
        AugmentedRowState state{};
        state.residual = {residual, __shfl_sync(group_mask, residual, 1, kRowAxisLanes),
                          __shfl_sync(group_mask, residual, 2, kRowAxisLanes)};
        state.dual = {dual, __shfl_sync(group_mask, dual, 1, kRowAxisLanes),
                      __shfl_sync(group_mask, dual, 2, kRowAxisLanes)};
        if (axis != 0u) continue;
        state.normal_bound = AugmentedRowBound(s, row, slot, scale, state);
        CacheAugmentedRow(data, p, s, slot, state, potential);
    }
}

// A block descends a tile of vertices. Its threads first evaluate the tile's element blocks (grouped
// by kind) and row responses densely; warps then sum them in the lane order of a warp per vertex.
constexpr uint32_t kDescendTileVertices = 8u;
constexpr uint32_t kDescendTileElements = 384u;
constexpr uint32_t kDescendTileRows = 256u;
constexpr uint32_t kDescendElementKinds = 4u;
constexpr uint32_t kElementSampleWords = 9u;
constexpr uint32_t kRowSampleWords = 7u;

struct DescendTileVertex {
    uint32_t active;
    uint32_t descend;
    uint32_t trial;
    uint32_t particle;
    uint32_t env;
    uint32_t vertex;
    uint32_t element_begin;
    uint32_t element_count;
    uint32_t element_first;
    uint32_t row_begin;
    uint32_t row_count;
    uint32_t row_first;
    float u[3];
    float free_rate[3];
    float direction[3];
    float primal_force[3];
    float inertia;
    float slope;
};

// The tile vertex owning tile-local sample `k`: the last one whose samples start at or before it.
__device__ uint32_t DescendTileOwner(const DescendTileVertex* tile, uint32_t k, bool rows) {
    uint32_t j = 0u;
    while (j + 1u < kDescendTileVertices &&
           (rows ? tile[j + 1u].row_first : tile[j + 1u].element_first) <= k) ++j;
    return j;
}

__device__ uint32_t DescendElementKind(const VertexBlockView& b, const DescendTileVertex& v,
                                       uint32_t k) {
    const uint32_t packed = b.incidence[v.element_begin + k - v.element_first];
    const uint32_t kind = b.elements[nk::VbdIncidenceElement(packed)].kind;
    return kind < kDescendElementKinds ? kind : kDescendElementKinds - 1u;
}

__device__ Vec3 TileVec3(const float* value) { return {value[0], value[1], value[2]}; }

__device__ void SetTileVec3(float* value, Vec3 v) {
    value[0] = v.x;
    value[1] = v.y;
    value[2] = v.z;
}

template <bool elastic>
__global__ void DescendParticlesKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                                       BlockScratch s, uint32_t color) {
    __shared__ DescendTileVertex tile[kDescendTileVertices];
    __shared__ uint32_t totals[2];
    __shared__ uint32_t kind_cursor[kDescendElementKinds];
    __shared__ uint32_t order[kDescendTileElements];
    __shared__ float element_samples[kDescendTileElements][kElementSampleWords];
    __shared__ float row_samples[kDescendTileRows][kRowSampleWords];
    __shared__ float row_changes[kDescendTileRows];
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    const uint32_t warps = blockDim.x / warpSize;
    const VertexBlockView b = Vertices(model, data, p);
    const uint32_t first = elastic ? b.color_segments[2u * color] : 0u;
    const uint32_t count = elastic ? b.color_segments[2u * color + 1u]
                                  : p.total_particle_count / p.env_count;
    const uint32_t items = count * p.env_count;
    const uint32_t tiles = (items + kDescendTileVertices - 1u) / kDescendTileVertices;
    for (uint32_t tile_index = blockIdx.x; tile_index < tiles; tile_index += gridDim.x) {
        if (threadIdx.x < kDescendTileVertices) {
            DescendTileVertex& v = tile[threadIdx.x];
            const uint32_t item = tile_index * kDescendTileVertices + threadIdx.x;
            v.active = v.descend = v.trial = 0u;
            v.element_count = v.row_count = 0u;
            if (item < items) {
                const uint32_t env = item / count;
                const uint32_t vertex = elastic ? b.color_vertices[first + item % count] : 0u;
                const uint32_t particle = elastic ? b.Particle(env, vertex) : item;
                const uint32_t local = particle % b.layout.particles_per_env;
                const bool skipped = !elastic && (local < p.grid_particles_per_env ||
                    (local >= b.layout.begin && local - b.layout.begin < b.layout.vertices));
                if (!skipped && data.particle_inv_mass[particle] > 0.0f) {
                    v.active = 1u;
                    v.env = env;
                    v.vertex = vertex;
                    v.particle = particle;
                    v.element_begin = elastic ? b.offsets[vertex] : 0u;
                    v.element_count = elastic ? b.offsets[vertex + 1u] - v.element_begin : 0u;
                    v.row_begin = s.offsets[particle];
                    v.row_count = s.offsets[particle + 1u] - v.row_begin;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            uint32_t element_total = 0u, row_total = 0u;
            for (uint32_t j = 0u; j < kDescendTileVertices; ++j) {
                tile[j].element_first = element_total;
                tile[j].row_first = row_total;
                element_total += tile[j].element_count;
                row_total += tile[j].row_count;
            }
            totals[0] = element_total;
            totals[1] = row_total;
        }
        if (threadIdx.x < kDescendElementKinds) kind_cursor[threadIdx.x] = 0u;
        __syncthreads();
        const uint32_t element_total = totals[0], row_total = totals[1];
        const bool dense_elements = elastic && element_total <= kDescendTileElements;
        const bool dense_rows = row_total <= kDescendTileRows;
        const uint32_t dense_element_total = dense_elements ? element_total : 0u;
        const uint32_t dense_total = dense_element_total + (dense_rows ? row_total : 0u);
        if (dense_elements) {
            for (uint32_t k = threadIdx.x; k < element_total; k += blockDim.x)
                atomicAdd(kind_cursor + DescendElementKind(b, tile[DescendTileOwner(tile, k, false)], k), 1u);
            __syncthreads();
            if (threadIdx.x == 0u) {
                uint32_t sum = 0u;
                for (uint32_t kind = 0u; kind < kDescendElementKinds; ++kind) {
                    const uint32_t kind_count = kind_cursor[kind];
                    kind_cursor[kind] = sum;
                    sum += kind_count;
                }
            }
            __syncthreads();
            for (uint32_t k = threadIdx.x; k < element_total; k += blockDim.x)
                order[atomicAdd(kind_cursor + DescendElementKind(b, tile[DescendTileOwner(tile, k, false)], k), 1u)] = k;
            __syncthreads();
        }
        for (uint32_t q = threadIdx.x; q < dense_total; q += blockDim.x) {
            if (q < dense_element_total) {
                const uint32_t k = order[q];
                const DescendTileVertex& v = tile[DescendTileOwner(tile, k, false)];
                Vec3 g;
                SymmetricMat3 h;
                VertexElementBlock(b, data.particle_vel, v.env, data.particle_vel[v.particle],
                                   b.incidence[v.element_begin + k - v.element_first], g, h);
                float* sample = element_samples[k];
                SetTileVec3(sample, g);
                sample[3] = h.xx;
                sample[4] = h.yy;
                sample[5] = h.zz;
                sample[6] = h.xy;
                sample[7] = h.xz;
                sample[8] = h.yz;
            } else {
                const uint32_t k = q - dense_element_total;
                const DescendTileVertex& v = tile[DescendTileOwner(tile, k, true)];
                const PointRowSample row = SamplePointRow(data, p, s, v.row_begin + k - v.row_first);
                float* sample = row_samples[k];
                SetTileVec3(sample, row.impulse);
                for (uint32_t c = 0u; c < 4u; ++c) sample[3u + c] = row.curvature[c];
            }
        }
        __syncthreads();
        for (uint32_t j = warp; j < kDescendTileVertices; j += warps) {
            DescendTileVertex& v = tile[j];
            if (v.active == 0u) continue;
            const uint32_t env = v.env, vertex = v.vertex, particle = v.particle;
            const Vec3 u = s.particle_snapshot[particle];
            const float inertia = elastic ? b.inertia[b.Slot(env, vertex)]
                                          : 1.0f / (data.particle_inv_mass[particle] * p.dt * p.dt);
            const Vec3 free_rate = elastic ? b.free_rate[b.Slot(env, vertex)] : s.particle_free[particle];
            Vec3 gradient{};
            SymmetricMat3 hessian{};
            if constexpr (elastic) {
                if (dense_elements) {
                    for (uint32_t i = lane; i < v.element_count; i += warpSize) {
                        const float* sample = element_samples[v.element_first + i];
                        gradient = gradient + TileVec3(sample);
                        nk::vbd::AddTo(hessian, SymmetricMat3{sample[3], sample[4], sample[5],
                                                              sample[6], sample[7], sample[8]});
                    }
                } else {
                    GatherVertexBlock(b, data.particle_vel, env, vertex, lane, warpSize, gradient, hessian);
                }
            }
            Vec3 impulse{};
            SymmetricMat3 contact_hessian{};
            const uint32_t begin = v.row_begin;
            const uint32_t end = begin + v.row_count;
            const uint32_t cached_at = begin + lane;
            for (uint32_t at = cached_at; at < end; at += warpSize) {
                PointRowSample row{};
                if (dense_rows) {
                    const uint32_t k = v.row_first + (at - begin);
                    row.impulse = TileVec3(row_samples[k]);
                    for (uint32_t c = 0u; c < 4u; ++c) row.curvature[c] = row_samples[k][3u + c];
                } else {
                    row = SamplePointRow(data, p, s, at);
                }
                AddPointRowSample(s, at, row, &impulse, &contact_hessian);
            }
            gradient = {WarpSum(gradient.x), WarpSum(gradient.y), WarpSum(gradient.z)};
            impulse = {WarpSum(impulse.x), WarpSum(impulse.y), WarpSum(impulse.z)};
            nk::vbd::AddTo(hessian, nk::vbd::Scaled(contact_hessian, 1.0f / (p.dt * p.dt)));
            hessian = {WarpSum(hessian.xx), WarpSum(hessian.yy), WarpSum(hessian.zz),
                       WarpSum(hessian.xy), WarpSum(hessian.xz), WarpSum(hessian.yz)};
            if (lane == 0u) {
                Vec3 direction{};
                Vec3 primal_force{};
                float slope = 0.0f;
                uint32_t valid = 0u;
                nk::vbd::AddIdentity(hessian, inertia);
                const Vec3 force = -nk::vbd::InertialGradient(u, free_rate, inertia, p.dt) -
                                   gradient + impulse / p.dt;
                primal_force = force;
                SymmetricMat3 inverse{};
                if (Finite(force) && nk::vbd::Invert(hessian, 0.0f, &inverse)) {
                    direction = inverse.Multiply(force) / p.dt;
                    slope = -p.dt * force.Dot(direction);
                    if (Finite(direction) && fabsf(slope) <= FLT_MAX) valid = 1u;
                    else RecordBlockFailure(data, p, s, particle, nk::BlockSolveFailure::InvalidDirection);
                } else {
                    RecordBlockFailure(data, p, s, particle, Finite(force)
                        ? nk::BlockSolveFailure::Factorization : nk::BlockSolveFailure::InvalidEquation);
                }
                SetTileVec3(v.u, u);
                SetTileVec3(v.free_rate, free_rate);
                SetTileVec3(v.direction, direction);
                SetTileVec3(v.primal_force, primal_force);
                v.inertia = inertia;
                v.slope = slope;
                v.descend = valid;
                v.trial = valid != 0u && slope < 0.0f ? 1u : 0u;
            }
        }
        __syncthreads();
        for (uint32_t q = threadIdx.x; q < dense_total; q += blockDim.x) {
            const bool element = q < dense_element_total;
            const uint32_t k = element ? order[q] : q - dense_element_total;
            const DescendTileVertex& v = tile[DescendTileOwner(tile, k, !element)];
            if (v.trial == 0u) continue;
            const Vec3 u = TileVec3(v.u);
            const Vec3 direction = TileVec3(v.direction);
            const Vec3 candidate{nk::vbd::TrialValue(u.x, direction.x, 1.0f),
                                 nk::vbd::TrialValue(u.y, direction.y, 1.0f),
                                 nk::vbd::TrialValue(u.z, direction.z, 1.0f)};
            const Vec3 move = candidate - u;
            if (element) {
                const VertexElementChangeTerms terms = VertexElementChange(b, data.particle_vel, v.env,
                    b.incidence[v.element_begin + k - v.element_first], u, move);
                element_samples[k][0] = terms.elastic;
                element_samples[k][1] = terms.rayleigh;
                element_samples[k][2] = terms.damping;
            } else {
                row_changes[k] = EvaluateLocalChange(
                    LoadPointLocalTerm(data, p, s, v.row_begin + k - v.row_first), move);
            }
        }
        __syncthreads();
        for (uint32_t j = warp; j < kDescendTileVertices; j += warps) {
            const DescendTileVertex& v = tile[j];
            if (v.active == 0u || v.descend == 0u) continue;
            const uint32_t env = v.env, vertex = v.vertex, particle = v.particle;
            const Vec3 u = TileVec3(v.u);
            const Vec3 free_rate = TileVec3(v.free_rate);
            const Vec3 direction = TileVec3(v.direction);
            const Vec3 primal_force = TileVec3(v.primal_force);
            const float inertia = v.inertia;
            const float slope = v.slope;
            const uint32_t begin = v.row_begin;
            const uint32_t end = begin + v.row_count;
            const uint32_t cached_at = begin + lane;
            float scale = 1.0f;
            Vec3 next = u;
            float last_change = 0.0f;
            uint32_t last_halving = 0u;
            bool accepted = false;
            for (uint32_t halving = 0u; halving <= kVertexStepHalvings && slope < 0.0f; ++halving) {
                const Vec3 candidate{nk::vbd::TrialValue(u.x, direction.x, scale),
                                     nk::vbd::TrialValue(u.y, direction.y, scale),
                                     nk::vbd::TrialValue(u.z, direction.z, scale)};
                const Vec3 move = candidate - u;
                const double trial_slope = __shfl_sync(0xffffffffu, -double(p.dt) * (
                    double(primal_force.x) * move.x + double(primal_force.y) * move.y +
                    double(primal_force.z) * move.z), 0u);
                const bool sampled = halving == 0u;
                float elastic_change = 0.0f;
                if constexpr (elastic) {
                    if (sampled && dense_elements) {
                        for (uint32_t i = lane; i < v.element_count; i += warpSize) {
                            const float* sample = element_samples[v.element_first + i];
                            AddVertexElementChange(b, {sample[0], sample[1], sample[2]}, elastic_change);
                        }
                    } else {
                        elastic_change = VertexEnergyChange(b, data.particle_vel, env, vertex, u, move,
                                                            lane, warpSize);
                    }
                }
                float row_change = 0.0f;
                for (uint32_t at = cached_at; at < end; at += warpSize)
                    row_change += sampled && dense_rows ? row_changes[v.row_first + (at - begin)]
                        : EvaluateLocalChange(LoadPointLocalTerm(data, p, s, at), move);
                const float change = static_cast<float>(
                    double(WarpSum(elastic_change + row_change)) +
                    nk::vbd::InertialEnergyChange(u, free_rate, move, inertia, p.dt));
                last_change = change;
                last_halving = halving;
                if (!Finite(candidate) || !Finite(move) || !isfinite(trial_slope) ||
                    !(fabsf(change) <= FLT_MAX)) {
                    bool base_finite = true;
                    if constexpr (elastic)
                        base_finite = VertexEnergyStateValid(b, data.particle_vel, env, vertex,
                                                            u, lane, warpSize);
                    for (uint32_t at = cached_at; at < end; at += warpSize)
                        base_finite &= isfinite(EvaluateLocalChange(LoadPointLocalTerm(data, p, s, at), {}));
                    if (!__all_sync(0xffffffffu, base_finite) || halving == kVertexStepHalvings) {
                        if (lane == 0u)
                            RecordBlockFailure(data, p, s, particle, nk::BlockSolveFailure::InvalidPotential);
                        break;
                    }
                    scale *= 0.5f;
                    continue;
                }
                if (trial_slope <= 0.0 && double(change) <= double(kVertexStepDecrease) * trial_slope) {
                    next = candidate;
                    accepted = true;
                    break;
                }
                scale *= 0.5f;
            }
            __syncwarp();
            if (lane == 0u) {
                if constexpr (elastic)
                    RecordVertexDescentAudit(b, env, vertex, primal_force, direction, u, next,
                        scale, last_change, last_halving, accepted, slope < 0.0f);
                data.particle_vel[particle] = next;
            }
        }
        __syncthreads();
    }
}

__global__ void DualUpdateKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.active_rows[item];
        if (rows[slot].flags & nk::nk_row_flags::kMaterialBlock) {
            UpdateMaterialDual(data, p, s, slot);
            continue;
        }
        const LocalTerm t = LoadLocalTerm(data, p, s, slot, ~0u, false);
        data.lambda[slot] = t.normal_bound;
        if (t.contact) {
            const auto f = nk::augmented::EvaluateTangentImpulse(t.dual, t.penalty, t.residual,
                                                                 t.normal_bound, t.mu_first, t.mu_second);
            data.lambda[slot + rows[slot].group_normal_count] = f.impulse.y;
            data.lambda[slot + 2u * rows[slot].group_normal_count] = f.impulse.z;
        }
    }
}

__global__ void MarkContactBlocksKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const auto points = PointMasses(data);
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.active_rows[item];
        const NkRow row = rows[slot];
        if (!(row.flags & nk::nk_row_flags::kBlockNormal)) continue;
        const LocalTerm term = LoadLocalTerm(data, p, s, slot, ~0u, false);
        if (!(term.normal_bound > 0.0f)) continue;
        VisitOwners(row, points, p, [&](uint32_t owner) { atomicExch(s.acceleration_disabled + owner, 1u); });
    }
}

__global__ void PrepareAccelerationKernel(BlockDescentSolveParams p, BlockScratch s) {
    for (uint32_t env = blockIdx.x * blockDim.x + threadIdx.x; env < p.env_count;
         env += gridDim.x * blockDim.x)
        s.acceleration_ratio[env] = nk::solve::ChebyshevIterationRatio(
            s.acceleration_iterations[env], s.acceleration_ratio[env], p.acceleration_spectral_radius);
}

__global__ void TrialAccelerationKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    for (uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x; particle < p.total_particle_count;
         particle += gridDim.x * blockDim.x) {
        const uint32_t env = particle / p.vertex_blocks.particles_per_env;
        const Vec3 current = data.particle_vel[particle];
        s.particle_snapshot[particle] = current;
        const bool owned = particle % p.vertex_blocks.particles_per_env >= p.grid_particles_per_env;
        const Vec3 next = owned && data.particle_inv_mass[particle] > 0.0f && s.acceleration_disabled[particle] == 0u
            ? nk::solve::ChebyshevExtrapolate(current, s.particle_older[particle], s.acceleration_ratio[env])
            : current;
        data.particle_vel[particle] = next;
    }
}

__global__ void ParticleMeritKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                                    BlockScratch s, uint32_t sample) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t stride = gridDim.x * blockDim.x / warpSize;
    const VertexBlockView b = Vertices(model, data, p);
    for (uint32_t particle = first; particle < p.total_particle_count; particle += stride) {
        if (!(data.particle_inv_mass[particle] > 0.0f)) continue;
        const uint32_t env = particle / b.layout.particles_per_env;
        const uint32_t local = particle % b.layout.particles_per_env;
        if (local < p.grid_particles_per_env) continue;
        const bool elastic = local >= b.layout.begin && local - b.layout.begin < b.layout.vertices;
        const uint32_t vertex = local - b.layout.begin;
        const Vec3 u = data.particle_vel[particle];
        const float inertia = elastic ? b.inertia[b.Slot(env, vertex)]
            : 1.0f / (data.particle_inv_mass[particle] * p.dt * p.dt);
        const Vec3 free_rate = elastic ? b.free_rate[b.Slot(env, vertex)] : s.particle_free[particle];
        Vec3 gradient{}, impulse{};
        SymmetricMat3 hessian{};
        if (elastic)
            GatherVertexBlock(b, data.particle_vel, env, vertex, lane, warpSize, gradient, hessian);
        for (uint32_t at = s.offsets[particle] + lane; at < s.offsets[particle + 1u]; at += warpSize) {
            const LocalTerm term = LoadLocalTerm(data, p, s, s.incidence[at], particle, false);
            impulse += term.jacobian[0] * term.normal_bound;
            if (term.contact) {
                const auto tangent = nk::augmented::EvaluateTangent(term.dual, term.penalty, term.residual,
                    term.normal_bound, term.mu_first, term.mu_second);
                impulse += term.jacobian[1] * tangent.impulse.y + term.jacobian[2] * tangent.impulse.z;
            }
        }
        gradient = {WarpSum(gradient.x), WarpSum(gradient.y), WarpSum(gradient.z)};
        impulse = {WarpSum(impulse.x), WarpSum(impulse.y), WarpSum(impulse.z)};
        if (lane == 0u) {
            const Vec3 force = -nk::vbd::InertialGradient(u, free_rate, inertia, p.dt) -
                               gradient + impulse / p.dt;
            const double merit = (double(force.x) * force.x + double(force.y) * force.y +
                                  double(force.z) * force.z) / inertia;
            atomicAdd(s.acceleration_merit + 2u * env + sample, merit);
        }
    }
}

__global__ void RowMeritKernel(DataView data, BlockDescentSolveParams p, BlockScratch s,
                               uint32_t sample) {
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.active_rows[item];
        const LocalTerm term = LoadLocalTerm(data, p, s, slot, ~0u, false);
        if (!(term.penalty > 0.0f)) continue;
        const double normal = double(term.normal_bound) - term.dual.x;
        double merit = normal * normal / term.penalty;
        if (term.contact) {
            const auto tangent = nk::augmented::EvaluateTangent(term.dual, term.penalty, term.residual,
                term.normal_bound, term.mu_first, term.mu_second);
            if (term.mu_first > 0.0f) {
                const double delta = (double(tangent.impulse.y) - term.dual.y) / term.mu_first;
                merit += delta * delta / term.penalty;
            }
            if (term.mu_second > 0.0f) {
                const double delta = (double(tangent.impulse.z) - term.dual.z) / term.mu_second;
                merit += delta * delta / term.penalty;
            }
        }
        atomicAdd(s.acceleration_merit + 2u * (slot / p.rows_per_env) + sample, merit);
    }
}

__global__ void AcceptAccelerationKernel(BlockDescentSolveParams p, BlockScratch s) {
    for (uint32_t env = blockIdx.x * blockDim.x + threadIdx.x; env < p.env_count;
         env += gridDim.x * blockDim.x) {
        const bool accepted = s.acceleration_ratio[env] == 1.0f ||
            nk::solve::ChebyshevAccept(s.acceleration_merit[2u * env + 1u], s.acceleration_merit[2u * env]);
        s.acceleration_accepted[env] = accepted ? 1u : 0u;
        s.acceleration_iterations[env] = accepted ? s.acceleration_iterations[env] + 1u : 0u;
    }
}

__global__ void CommitAccelerationKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    for (uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x; particle < p.total_particle_count;
         particle += gridDim.x * blockDim.x) {
        const uint32_t env = particle / p.vertex_blocks.particles_per_env;
        const bool accepted = s.acceleration_accepted[env] != 0u;
        const Vec3 next = accepted ? data.particle_vel[particle] : s.particle_snapshot[particle];
        s.particle_older[particle] = accepted ? s.particle_previous[particle] : next;
        s.particle_previous[particle] = next;
        data.particle_vel[particle] = next;
    }
}

// One warp per particle: lanes form the incidence terms, every lane sums them in incidence order.
__global__ void GatherImpulseKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    if (data.particle_row_impulse == nullptr) return;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const auto points = PointMasses(data);
    const uint32_t lane = threadIdx.x % warpSize;
    const size_t warp = (size_t{blockIdx.x} * blockDim.x + threadIdx.x) / warpSize;
    const size_t stride = size_t{gridDim.x} * blockDim.x / warpSize;
    for (size_t item = warp; item < p.total_particle_count; item += stride) {
        const uint32_t particle = static_cast<uint32_t>(item);
        const uint32_t first = s.offsets[particle];
        const uint32_t last = s.offsets[particle + 1u];
        Vec3 impulse{};
        for (uint32_t chunk = first; chunk < last; chunk += warpSize) {
            Vec3 jacobians[3] = {};
            float weights[3] = {};
            uint32_t axes = 0u;
            if (chunk + lane < last) {
                const uint32_t slot = s.incidence[chunk + lane];
                const NkRow row = rows[slot];
                axes = (row.flags & nk::nk_row_flags::kBlockNormal) ? 3u : 1u;
#pragma unroll
                for (uint32_t axis = 0u; axis < 3u; ++axis) {
                    if (axis >= axes) continue;
                    const uint32_t index = slot + axis * row.group_normal_count;
                    jacobians[axis] = ParticleJacobian(rows[index], points, particle);
                    weights[axis] = data.lambda[index];
                }
            }
            const uint32_t count = min(last - chunk, static_cast<uint32_t>(warpSize));
            for (uint32_t source = 0u; source < count; ++source) {
                const uint32_t source_axes = __shfl_sync(0xffffffffu, axes, source);
#pragma unroll
                for (uint32_t axis = 0u; axis < 3u; ++axis) {
                    const Vec3 jacobian{__shfl_sync(0xffffffffu, jacobians[axis].x, source),
                                        __shfl_sync(0xffffffffu, jacobians[axis].y, source),
                                        __shfl_sync(0xffffffffu, jacobians[axis].z, source)};
                    const float weight = __shfl_sync(0xffffffffu, weights[axis], source);
                    if (axis < source_axes) impulse += jacobian * weight;
                }
            }
        }
        if (lane == 0u) data.particle_row_impulse[particle] = impulse;
    }
}

__global__ void RecordSolveKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                                  BlockScratch s) {
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < p.env_count;
         index += gridDim.x * blockDim.x) {
        data.vbd_velocity_sweep_count[index] = p.iterations;
        data.solver_color_counts[index * kSolverCountWords] = p.vertex_blocks.colors;
    }
}

__global__ void MeasureRowsKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.active_rows[item];
        const NkRow row = rows[slot];
        if (!(row.flags & nk::nk_row_flags::kBlockNormal)) continue;
        const LocalTerm t = LoadLocalTerm(data, p, s, slot, ~0u, false);
        const Vec3 velocity{-t.residual.x + t.compliance * t.dual.x, -t.residual.y, -t.residual.z};
        const auto residual = constraint::EvaluateCoulombContactResidual(
            row.contact_response, velocity, t.dual, row.mu, row.friction_secondary);
        const uint32_t env = slot / p.rows_per_env;
        atomicAdd(data.contact_solve_counts + env * constraint::kContactSolveCountSize, 1u);
        if (!residual.valid)
            atomicAdd(data.contact_solve_counts + env * constraint::kContactSolveCountSize + 1u, 1u);
        const float metrics[] = {residual.normal_natural_velocity, residual.tangent_natural_velocity,
            residual.normal_velocity_violation, residual.normal_impulse_violation,
            residual.normal_complementarity, residual.friction_impulse_violation,
            residual.friction_power_violation, residual.friction_dissipation_work};
        for (uint32_t metric = 0u; metric < constraint::kContactSolveMetricCount; ++metric) {
            const float value = residual.valid ? metrics[metric] : FLT_MAX;
            const unsigned long long packed =
                (static_cast<unsigned long long>(__float_as_uint(value)) << 32u) | (~slot);
            atomicMax(reinterpret_cast<unsigned long long*>(data.contact_solve_metrics +
                env * constraint::kContactSolveMetricCount + metric), packed);
        }
    }
}

__global__ void MeasureVerticesKernel(ModelView model, DataView data, BlockDescentSolveParams p) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t stride = gridDim.x * blockDim.x / warpSize;
    const VertexBlockView b = Vertices(model, data, p);
    for (uint32_t item = first; item < b.layout.dynamic_vertices * p.env_count; item += stride) {
        const uint32_t env = item / b.layout.dynamic_vertices;
        const uint32_t vertex = b.color_vertices[item % b.layout.dynamic_vertices];
        Vec3 u, force;
        SymmetricMat3 hessian, inverse;
        VertexBlockEquationWarp(b, PointMasses(data), nullptr, env, vertex, lane, u, force, hessian);
        const bool valid = Finite(force) && nk::vbd::Invert(hessian, 0.0f, &inverse);
        const Vec3 correction = valid ? inverse.Multiply(force) / p.dt : Vec3{FLT_MAX, FLT_MAX, FLT_MAX};
        if (lane != 0u) continue;
        RecordVertexEquationAudit(b, env, vertex, force, correction);
        const float values[2] = {fmaxf(fabsf(correction.x), fmaxf(fabsf(correction.y), fabsf(correction.z))),
            valid ? fmaxf(fabsf(force.x), fmaxf(fabsf(force.y), fabsf(force.z))) : FLT_MAX};
        for (uint32_t metric = 0u; metric < 2u; ++metric) {
            const unsigned long long packed =
                (static_cast<unsigned long long>(__float_as_uint(values[metric])) << 32u) |
                (~b.Particle(env, vertex));
            atomicMax(reinterpret_cast<unsigned long long*>(data.vbd_solve_metrics + env * 2u + metric), packed);
        }
    }
}

template <typename Kernel, typename... Args>
cudaError_t LaunchWork(Kernel kernel, uint32_t work, uint32_t items_per_block,
                       cudaStream_t stream, Args... args) {
    if (work == 0u) return cudaSuccess;
    uint32_t grid = 0u;
    auto status = ResidentGridSize(kernel, kThreads, 0u,
                                   (work + items_per_block - 1u) / items_per_block, &grid);
    if (status != cudaSuccess) return status;
    LaunchCuda(kernel, dim3(grid), dim3(kThreads), 0u, stream, args...);
    return cudaGetLastError();
}

Status OpBlockDescentSolve(const ModelView& model, const DataView& data,
                           const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const BlockDescentSolveParams*>(params);
    if (p == nullptr || !(p->dt > 0.0f) || p->iterations == 0u || p->env_count == 0u)
        return Status::InvalidArgument;
    if (p->total_particle_count % p->env_count != 0u ||
        p->grid_particles_per_env > p->total_particle_count / p->env_count ||
        uint64_t{p->material_cells_per_env} * nk::kMpmStressRowsPerCell > p->rows_per_env)
        return Status::InvalidArgument;
    const uint64_t vertex_color_items = uint64_t{p->vertex_blocks.dynamic_vertices} * p->env_count;
    if (vertex_color_items > p->total_particle_count ||
        (vertex_color_items > 0u && (model.vbd_color_vertices == nullptr ||
                                  model.vbd_color_segments == nullptr)))
        return Status::InvalidArgument;
    if (p->total_grid_count > 0u) {
        if (p->total_grid_count % p->env_count != 0u || data.grid_inv_mass == nullptr ||
            data.grid_velocity == nullptr) return Status::InvalidArgument;
        const uint64_t nodes = p->total_grid_count / p->env_count;
        uint64_t extent = nk::kMpmLattices;
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            if (p->grid_dims[axis] == 0u || p->grid_dims[axis] > nodes / extent)
                return Status::InvalidArgument;
            extent *= p->grid_dims[axis];
        }
        if (extent != nodes) return Status::InvalidArgument;
    }
    if (p->material_cells_per_env > 0u && (p->total_grid_count == 0u ||
        data.point_endpoint_ranges == nullptr || data.point_endpoint_terms == nullptr))
        return Status::InvalidArgument;
    if (p->total_body_count > 0u && (p->total_body_count % p->env_count != 0u ||
        data.body_inv_mass == nullptr || data.body_world_inv_inertia == nullptr ||
        data.body_linear_velocity == nullptr || data.body_angular_velocity == nullptr))
        return Status::InvalidArgument;
    if (p->articulation_count > 0u && (p->max_dof == 0u ||
        p->articulation_count % p->env_count != 0u || model.articulation_link_offset == nullptr ||
        model.articulation_link_count == nullptr || model.joint_type == nullptr ||
        model.parent_link == nullptr || data.qdot == nullptr || data.qdot_flat == nullptr ||
        data.link_velocity == nullptr || data.m == nullptr || data.m_inv == nullptr ||
        data.chain_jacobian == nullptr || data.chain_jacobian_b == nullptr ||
        data.row_minv_jt == nullptr || data.row_minv_jt_b == nullptr ||
        (p->mimic_couplings > 0u && (data.m_reduced == nullptr ||
         data.mimic_root_dof == nullptr || data.mimic_root_scale == nullptr))))
        return Status::InvalidArgument;
    BlockScratch s{};
    const uint64_t words = BindScratch(*p, data.block_descent_scratch, &s);
    if (words == 0u || words > p->workspace_words || data.block_descent_scratch == nullptr)
        return Status::InvalidArgument;
    const ArticulationDeviceState articulation_state = MakeArticulationDeviceState(
        model, data, p->base_link_count * p->env_count, p->articulation_count);
    const auto clear = [&](void* ptr, size_t bytes) {
        return bytes == 0u || (ptr != nullptr && cudaMemsetAsync(ptr, 0, bytes, stream) == cudaSuccess);
    };
    if (p->measure_vertex_audit != 0u && !clear(data.vbd_solve_audit,
            size_t{p->vertex_blocks.vertices} * p->env_count * nk::kVbdSolveAuditColumnCount * sizeof(float)))
        return Status::Failed;
    if (!clear(s.counts, size_t{s.blocks + 1u} * sizeof(uint32_t)) ||
        !clear(s.active_count, sizeof(uint32_t)) ||
        !clear(data.solver_color_counts, size_t{p->env_count} * kSolverCountWords * sizeof(uint32_t)) ||
        !clear(data.solver_phase_time, size_t{p->env_count} * kSolverPhaseWords * sizeof(uint64_t))) return Status::Failed;
    if (p->clear_failure_diagnostics != 0u &&
        (!clear(s.failure_reasons, size_t{s.blocks} * sizeof(uint32_t)) ||
         !clear(s.failure_equations, size_t{s.blocks} * nk::kBlockSolveFailureEquationColumnCount * sizeof(double)) ||
         (s.blocks > 0u &&
          (cudaMemsetAsync(s.failure_rows, 0xff, size_t{s.blocks} * sizeof(uint32_t), stream) != cudaSuccess ||
           cudaMemsetAsync(s.failure_substeps, 0xff, size_t{s.blocks} * sizeof(uint32_t), stream) != cudaSuccess))))
        return Status::Failed;
    uint32_t spread_rows = 0u;
    if (SchedulerThreads(&spread_rows) != cudaSuccess) return Status::Failed;
    const size_t particle_bytes = size_t{p->total_particle_count} * sizeof(Vec3);
    const size_t grid_bytes = size_t{p->total_grid_count} * sizeof(Vec3);
    if (particle_bytes > 0u && cudaMemcpyAsync(s.particle_free, data.particle_vel, particle_bytes,
            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return Status::Failed;
    const size_t articulation_bytes = size_t{p->articulation_count} * p->max_dof * sizeof(float);
    if (p->articulation_count > 0u) {
        if (LaunchWork(PackBlockArticulationVelocityKernel,
                p->articulation_count * p->max_dof, kThreads, stream,
                articulation_state, data, *p, s) != cudaSuccess) return Status::Failed;
        LaunchCuda(InitializeArticulationBlocksKernel, dim3(p->articulation_count), dim3(kThreads),
                   0u, stream, articulation_state, data, *p, s);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
    }
    if (p->total_body_count > 0u) {
        LaunchCuda(InitializeRigidBlocksKernel, dim3(p->total_body_count), dim3(kThreads),
                   0u, stream, data, *p, s);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
    }
    if (p->total_grid_count > 0u &&
        (LaunchWork(InitializeGridColorScheduleKernel, nk::kMpmCellStencilNodes, kThreads, stream,
             data, *p, s) != cudaSuccess ||
         LaunchWork(InitializeGridBlocksKernel, p->total_grid_count, kThreads, stream,
             data, *p, s) != cudaSuccess)) return Status::Failed;
    const bool accelerate = p->acceleration_spectral_radius > 0.0f &&
                            p->acceleration_spectral_radius < 1.0f;
    if (accelerate && (!clear(s.acceleration_disabled, size_t{s.blocks} * sizeof(uint32_t)) ||
        !clear(s.acceleration_iterations, size_t{p->env_count} * sizeof(uint32_t)) ||
        !clear(s.acceleration_ratio, size_t{p->env_count} * sizeof(float)) ||
        (particle_bytes > 0u &&
         (cudaMemcpyAsync(s.particle_previous, data.particle_vel, particle_bytes,
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
          cudaMemcpyAsync(s.particle_older, data.particle_vel, particle_bytes,
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess)))) return Status::Failed;
    if (LaunchWork(InitializeOwnerCacheColorsKernel, s.blocks, kThreads, stream,
            data, *p, s) != cudaSuccess ||
        LaunchWork(InitializeVertexCacheColorsKernel, static_cast<uint32_t>(vertex_color_items),
            kThreads, stream, model, data, *p, s) != cudaSuccess ||
        LaunchWork(CountIncidenceKernel, s.rows, kThreads, stream, data, *p, s) != cudaSuccess)
        return Status::Failed;
    if (cub::DeviceScan::ExclusiveSum(s.scan, s.scan_bytes, s.counts, s.offsets,
            static_cast<int>(s.blocks + 1u), stream) != cudaSuccess ||
        !clear(s.counts, size_t{s.blocks + 1u} * sizeof(uint32_t))) return Status::Failed;
    if (LaunchWork(FillIncidenceKernel, s.rows, kThreads, stream, data, *p, s) != cudaSuccess)
        return Status::Failed;
    if (s.incidence_sort_bytes > 0u) {
        if (cub::DeviceSegmentedRadixSort::SortKeys(s.incidence_sort, s.incidence_sort_bytes,
                s.incidence, s.sorted_incidence, static_cast<int64_t>(s.incidence_capacity),
                static_cast<int64_t>(s.blocks),
                IncidenceBegins(thrust::counting_iterator<uint32_t>(0u),
                                IncidenceBegin{s.offsets, s.incidence_capacity}),
                LargeIncidenceEnds(thrust::counting_iterator<uint32_t>(0u),
                                   LargeIncidenceEnd{s.offsets, s.incidence_capacity}),
                0, RowSlotBits(s.rows), stream) != cudaSuccess ||
            LaunchWork(SortSmallIncidenceKernel, s.blocks, kThreads / 32u, stream, s) != cudaSuccess)
            return Status::Failed;
        s.incidence = s.sorted_incidence;
    }
    if (LaunchWork(CachePointIncidenceJacobiansKernel,
            p->total_particle_count + p->total_grid_count, kThreads / 32u, stream,
            data, *p, s) != cudaSuccess) return Status::Failed;
    if (p->total_particle_count > 0u &&
        (!clear(s.coarse_anchored, size_t{p->env_count} * sizeof(uint32_t)) ||
         LaunchWork(MarkCoarseAnchorsKernel, p->vertex_blocks.elements * p->env_count, kThreads,
             stream, model, data, *p, s) != cudaSuccess ||
         !clear(s.coarse_row_count, sizeof(uint32_t)) ||
         (s.rows > 0u && cub::DeviceSelect::Flagged(s.coarse_select, s.coarse_select_bytes,
             thrust::counting_iterator<uint32_t>(0u), s.coarse_flags, s.coarse_rows,
             s.coarse_row_count, static_cast<int>(s.rows), stream) != cudaSuccess) ||
         LaunchWork(CoarseRowRangesKernel, p->env_count + 1u, kThreads, stream, *p, s) != cudaSuccess))
        return Status::Failed;
    for (uint32_t iteration = 0u; iteration < p->iterations; ++iteration) {
        s.diagnostic_iteration = iteration;
        if (grid_bytes > 0u && cudaMemcpyAsync(s.grid_snapshot, data.grid_velocity, grid_bytes,
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return Status::Failed;
        if (articulation_bytes > 0u && cudaMemcpyAsync(s.articulation_snapshot, data.qdot_flat,
                articulation_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return Status::Failed;
        if (LaunchWork(PackBlockRigidSnapshotKernel, p->total_body_count, kThreads, stream,
                data, *p, s) != cudaSuccess) return Status::Failed;
        BlockScratch particle_s = s;
        particle_s.particle_snapshot = data.particle_vel;
        for (uint32_t color = 0u; color <= p->vertex_blocks.colors; ++color) {
            const bool elastic = color < p->vertex_blocks.colors;
            const uint32_t work = elastic ? p->vertex_blocks.dynamic_vertices * p->env_count
                                          : p->total_particle_count;
            if (!elastic && p->total_particle_count == p->vertex_blocks.vertices * p->env_count)
                continue;
            if (LaunchWork(CacheAugmentedRowsKernel, s.rows, kThreads / kRowAxisLanes, stream,
                    data, *p, particle_s, color, spread_rows) != cudaSuccess) return Status::Failed;
            const auto descent_status = elastic
                ? LaunchWork(DescendParticlesKernel<true>, work, kDescendTileVertices, stream,
                             model, data, *p, particle_s, color)
                : LaunchWork(DescendParticlesKernel<false>, work, kDescendTileVertices, stream,
                             model, data, *p, particle_s, color);
            if (descent_status != cudaSuccess) return Status::Failed;
        }
        if (p->articulation_count > 0u || p->total_body_count > 0u) {
            if (particle_bytes > 0u && cudaMemcpyAsync(s.particle_snapshot, data.particle_vel,
                    particle_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return Status::Failed;
            if (LaunchWork(CacheAugmentedRowsKernel, s.rows, kThreads / kRowAxisLanes, stream,
                    data, *p, s, p->vertex_blocks.colors + 1u, spread_rows) != cudaSuccess) return Status::Failed;
            if (p->articulation_count > 0u) {
                LaunchCuda(DescendArticulationsKernel, dim3(p->articulation_count), dim3(kThreads),
                           0u, stream, articulation_state, data, *p, s);
                if (cudaGetLastError() != cudaSuccess) return Status::Failed;
            }
            if (p->total_body_count > 0u) {
                LaunchCuda(DescendRigidBlocksKernel, dim3(p->total_body_count), dim3(kThreads),
                           0u, stream, data, *p, s);
                if (cudaGetLastError() != cudaSuccess) return Status::Failed;
            }
        }
        if (p->total_grid_count > 0u) {
            if (articulation_bytes > 0u && cudaMemcpyAsync(s.articulation_snapshot, data.qdot_flat,
                    articulation_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return Status::Failed;
            if (LaunchWork(PackBlockRigidSnapshotKernel, p->total_body_count, kThreads, stream,
                    data, *p, s) != cudaSuccess) return Status::Failed;
            BlockScratch grid_s = s;
            grid_s.grid_snapshot = data.grid_velocity;
            for (uint32_t color = 0u; color < nk::kMpmCellStencilNodes; ++color) {
                const uint32_t cache_color = p->vertex_blocks.colors + 2u + color;
                if (LaunchWork(CacheAugmentedRowsKernel, s.rows, kThreads / kRowAxisLanes, stream,
                        data, *p, grid_s, cache_color, spread_rows) != cudaSuccess) return Status::Failed;
                if (p->material_cells_per_env > 0u &&
                    LaunchWork(CacheMaterialRowsKernel, s.rows, kThreads / 32u, stream,
                        data, *p, grid_s, true, cache_color) != cudaSuccess) return Status::Failed;
                if (LaunchWork(DescendGridBlocksKernel, GridColorCapacity(*p, color), kThreads / 32u,
                        stream, data, *p, grid_s, color) != cudaSuccess) return Status::Failed;
            }
        }
        for (uint32_t family = 0u; family < kCoarseFamilies; ++family) {
            const uint32_t per_env = p->total_particle_count / p->env_count;
            const bool present = family == kCoarseVertexFamily
                ? p->vertex_blocks.dynamic_vertices > 0u
                : per_env > p->grid_particles_per_env + p->vertex_blocks.vertices;
            if (!present) continue;
            const dim3 coarse_grid(p->env_count * s.coarse_parts);
            LaunchCuda(CoarseGradientKernel, coarse_grid, dim3(kThreads * kCoarseHelpers), 0u, stream,
                       model, data, *p, s, family);
            LaunchCuda(CoarseDirectionKernel, dim3(p->env_count), dim3(kThreads), 0u, stream,
                       data, *p, s, family);
            LaunchCuda(CoarseUnitTrialKernel, coarse_grid, dim3(kThreads * kCoarseHelpers), 0u, stream,
                       model, data, *p, s, family);
            LaunchCuda(CoarseAcceptKernel, dim3(p->env_count), dim3(kThreads), 0u, stream,
                       *p, s, false);
            LaunchCuda(CoarseTrialsKernel, coarse_grid, dim3(kThreads), 0u, stream,
                       model, data, *p, s, family);
            LaunchCuda(CoarseAcceptKernel, dim3(p->env_count), dim3(kThreads), 0u, stream,
                       *p, s, true);
            if (cudaGetLastError() != cudaSuccess ||
                LaunchWork(CoarseApplyKernel, p->total_particle_count, kThreads, stream,
                    model, data, *p, s, family) != cudaSuccess) return Status::Failed;
        }
        if (accelerate) {
            if (!clear(s.acceleration_merit, size_t{p->env_count} * 2u * sizeof(double)) ||
                LaunchWork(MarkContactBlocksKernel, s.rows, kThreads, stream, data, *p, s) != cudaSuccess ||
                LaunchWork(PrepareAccelerationKernel, p->env_count, kThreads, stream, *p, s) != cudaSuccess ||
                LaunchWork(ParticleMeritKernel, p->total_particle_count, kThreads / 32u, stream,
                    model, data, *p, s, 0u) != cudaSuccess ||
                LaunchWork(RowMeritKernel, s.rows, kThreads, stream, data, *p, s, 0u) != cudaSuccess ||
                LaunchWork(TrialAccelerationKernel, p->total_particle_count, kThreads, stream, data, *p, s) != cudaSuccess ||
                LaunchWork(ParticleMeritKernel, p->total_particle_count, kThreads / 32u, stream,
                    model, data, *p, s, 1u) != cudaSuccess ||
                LaunchWork(RowMeritKernel, s.rows, kThreads, stream, data, *p, s, 1u) != cudaSuccess ||
                LaunchWork(AcceptAccelerationKernel, p->env_count, kThreads, stream, *p, s) != cudaSuccess ||
                LaunchWork(CommitAccelerationKernel, p->total_particle_count, kThreads, stream, data, *p, s) != cudaSuccess)
                return Status::Failed;
        }
        if (p->material_cells_per_env > 0u &&
            LaunchWork(CacheMaterialRowsKernel, s.rows, kThreads / 32u, stream,
                data, *p, s, false, kAllCacheColors) != cudaSuccess) return Status::Failed;
        if (LaunchWork(DualUpdateKernel, s.rows, kThreads, stream, data, *p, s) != cudaSuccess)
            return Status::Failed;
    }
    if (LaunchWork(GatherImpulseKernel, p->total_particle_count, kThreads / 32u, stream, data, *p, s) != cudaSuccess ||
        LaunchWork(RecordSolveKernel, p->env_count, kThreads, stream,
                   model, data, *p, s) != cudaSuccess) return Status::Failed;
    if (p->base_link_count > 0u && p->joint_limit_rows_per_env >= 2u * p->base_link_count &&
        data.joint_limit_impulse != nullptr &&
        LaunchWork(WriteBlockJointLimitImpulsesKernel, p->base_link_count * p->env_count,
            kThreads, stream, data, *p) != cudaSuccess) return Status::Failed;
    if (p->measure_contact_residual != 0u) {
        if (!clear(data.contact_solve_metrics, size_t{p->env_count} * constraint::kContactSolveMetricCount * sizeof(uint64_t)) ||
            !clear(data.contact_solve_counts, size_t{p->env_count} * constraint::kContactSolveCountSize * sizeof(uint32_t)) ||
            !clear(data.vbd_solve_metrics, size_t{p->env_count} * 2u * sizeof(uint64_t))) return Status::Failed;
        if (LaunchWork(MeasureRowsKernel, s.rows, kThreads, stream, data, *p, s) != cudaSuccess ||
            LaunchWork(MeasureVerticesKernel, p->vertex_blocks.dynamic_vertices * p->env_count,
                kThreads / 32u, stream, model, data, *p) != cudaSuccess) return Status::Failed;
    }
    return Status::Ok;
}

}  // namespace

uint64_t BlockDescentScratchWords(const BlockDescentSolveParams& params) {
    return BindScratch(params, nullptr, nullptr);
}

void RegisterNkBlockDescentOps() {
    SetCudaOp(NkOp::BlockDescentSolve, &OpBlockDescentSolve);
}

}  // namespace nuka::phi
