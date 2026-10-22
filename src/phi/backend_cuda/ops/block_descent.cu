#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <type_traits>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "constraint/coulomb_contact.hpp"
#include "nk/material/mpm_constitutive.hpp"
#include "nk/readout/energy_ledger.hpp"
#include "nk/readout/telemetry.hpp"
#include "nk/solve/augmented_row.hpp"
#include "nk/solve/block_row_schedule.hpp"
#include "nk/solve/vertex_block_forms.hpp"
#include "nk/solve/vertex_block_hierarchy.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/launch_grid.cuh"
#include "phi/backend_cuda/ops/articulation_mass.cuh"
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
// A rigid body solves all its rows in one CTA, so it takes a wider one than the other blocks.
constexpr uint32_t kRigidThreads = 256u;
constexpr uint32_t kSolverCountWords = 6u;
constexpr uint32_t kSolverPhaseWords = 8u;
constexpr uint32_t kNoCacheColor = ~0u;
constexpr uint32_t kAllCacheColors = ~0u;
constexpr uint32_t kRowAxisLanes = 4u;
// Loop words of a solve: finished sweeps and whether another sweep runs.
constexpr uint32_t kSweepCount = 0u, kSweepContinue = 1u, kSweepStateWords = 2u;
// Residual classes of a sweep in velocity units: the contact normal and tangent multiplier steps, the other
// rows' multiplier steps and the blocks' momentum (Newton) steps.
constexpr uint32_t kResidualNormal = 0u, kResidualTangent = 1u, kResidualJoint = 2u, kResidualMomentum = 3u,
                   kResidualClasses = 4u;
// solver_phase_time slots of a timed solve; kPhaseEnd closes its last phase.
constexpr uint32_t kPhaseSetup = 0u, kPhaseVertices = 1u, kPhaseDense = 2u, kPhaseGrid = 3u, kPhaseCoarse = 4u,
                   kPhaseMomentum = 5u, kPhaseDual = 6u, kPhaseCheck = 7u, kPhaseEnd = kSolverPhaseWords;
constexpr uint32_t kPointJacobianAxes = 3u;
constexpr uint32_t kPointJacobianComponents = 3u;
constexpr uint32_t kPointJacobianWords = kPointJacobianAxes * kPointJacobianComponents;
// Penalty scale of rows with grid owners: they tie light nodes to each other and to heavier blocks, where a
// stiffer penalty pins each block to its neighbours' stale velocities and stalls block Gauss-Seidel.
constexpr double kGridRowPenaltyScale = 1.0;
// Largest penalty times a row's most compliant owner share, the condition float block solves keep exact enough.
constexpr double kPenaltyConditionCap = 1.0e2;

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

// One environment's multilevel step: plane coefficients, accepted scale and pending trial, the dense
// factor flag, the fixed-point exponent with its bound, and the finished blocks of the running kernel.
struct MultilevelState {
    double slope;
    float alpha;
    float beta;
    float scale;
    float trial;
    uint32_t dense;
    int32_t exponent;
    uint32_t bound;
    uint32_t done;
};
static_assert(sizeof(MultilevelState) % sizeof(double) == 0u);
// Plane sums: force work along the correction and the previous step, then their three curvatures.
constexpr uint32_t kMultilevelSums = 5u;
// Blocks per environment that share its live rows in the multilevel kernels, about a row per thread.
constexpr uint32_t kMultilevelRowParts = 256u;
// Upper triangle of an element Hessian over the moves of up to three vertices relative to vertex 0.
constexpr uint32_t kMultilevelElementHessian = 45u;
// Threads of a restriction block, one coarse node per warp.
constexpr uint32_t kMultilevelCorrectThreads = 256u;
// Vertex-block particles one row group may move; a group with more fails its environment.
constexpr uint32_t kMultilevelRowPoints = 8u;

// Operator words per environment: a symmetric block per node above the dense level, then the packed
// upper triangle of the dense level.
__host__ __device__ inline uint64_t MultilevelOperatorEntries(const VertexBlockLayout& l) {
    const uint64_t rank = 3u * uint64_t{l.coarse_dense_nodes};
    return 6u * uint64_t{l.coarse_nodes - l.coarse_dense_nodes} + rank * (rank + 1u) / 2u;
}

// Words of one environment's dense factor: the upper factor by columns, then its scales.
__host__ __device__ inline uint64_t MultilevelDenseFactorWords(uint32_t rank) {
    return uint64_t{rank} * (rank + 3u) / 2u;
}

// Blocks per environment of the multilevel plane kernels: up to kMultilevelRowParts blocks striding over
// the rows, then one vertex or element per thread.
uint32_t MultilevelItemParts(const BlockDescentSolveParams& p) {
    const uint64_t items = uint64_t{p.vertex_blocks.vertices} + p.vertex_blocks.elements;
    return static_cast<uint32_t>(std::max<uint64_t>(1u, (items + kThreads - 1u) / kThreads));
}

uint32_t MultilevelParts(const BlockDescentSolveParams& p) {
    return MultilevelItemParts(p) + static_cast<uint32_t>(std::min<uint64_t>(
        kMultilevelRowParts, (uint64_t{p.rows_per_env} + kThreads - 1u) / kThreads));
}

// Blocks per environment of the restriction.
uint32_t MultilevelCorrectParts(const VertexBlockLayout& l) {
    constexpr uint32_t warps = kMultilevelCorrectThreads / 32u;
    return std::max<uint32_t>(1u, (l.coarse_nodes + warps - 1u) / warps);
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

// Row coefficients remain fixed during a solve; its residual, dual and bound are refreshed separately.
struct MultilevelRowTerm {
    float penalty;
    float compliance;
    float lower;
    float upper;
    float mu_first;
    float mu_second;
    bool contact;
    nk::augmented::ScalarResponse response;
};
static_assert(sizeof(MultilevelRowTerm) % sizeof(double) == 0u);

constexpr uint32_t kTileLanes = 32u;
constexpr uint32_t kTileWarps = 4u;
constexpr uint32_t kTileHeadGroup = 8u;
// Per-head fields interleaved in groups of eight: indices, coefficients, normal bound, residual, dual,
// move, step and scalar response of the current block solve, the relaxation and the unextrapolated dual.
constexpr uint32_t kTileSlot = 0u, kTileStride = 1u, kTileAxes = 2u, kTileOwner = 3u, kTileRow = 5u,
                   kTileUnit = 7u, kTileUnitValue = 9u, kTileRho = 11u, kTileCompliance = 12u, kTileLower = 13u,
                   kTileUpper = 14u, kTileMuFirst = 15u, kTileMuSecond = 16u, kTileNormalBound = 17u,
                   kTileResidual = 18u, kTileDual = 21u, kTileMove = 24u, kTileStep = 27u,
                   kTileDualScale = 30u, kTileResidualScale = 31u,
                   kTileRelaxationStep = kTileResidualScale + 1u,
                   kTileRelaxationWeight = kTileRelaxationStep + 3u,
                   kTileBase = kTileRelaxationWeight + 1u,
                   kTileFields = kTileBase + 3u;

bool DenseTileRoute(const BlockDescentSolveParams& p) {
    return BlockDescentTileRoute(p.total_particle_count, p.total_grid_count,
                                 p.articulation_count + p.total_body_count, p.max_dof);
}

// Head rows an environment can hold in whole warps: a contact head takes two tangent rows, every other
// row heads.
uint32_t TileHeadCapacity(const BlockDescentSolveParams& p) {
    const uint32_t heads = p.contact_rows_per_env / 3u + (p.rows_per_env - p.contact_rows_per_env);
    return (heads + kTileLanes - 1u) / kTileLanes * kTileLanes;
}

// Coordinates of a tile block: the articulation's rounded up to four within a warp and the largest articulation's
// past it, and at least a rigid body's six.
uint32_t TileWidth(const BlockDescentSolveParams& p) {
    return p.max_dof <= kTileLanes ? std::max(8u, (p.max_dof + 3u) & ~3u) : kMaxArticulationDof;
}

__host__ __device__ uint32_t TileJacobianWidth(const BlockDescentSolveParams& p) {
    return ((p.max_independent_dof > kRigidBlockDof ? p.max_independent_dof : kRigidBlockDof) + 3u) & ~3u;
}

// Warps the tile kernel keeps resident; each stages one environment at a time in its own slot.
uint32_t TileWarpSlots(const BlockDescentSolveParams& p);

// FISTA momentum of an env's multipliers, restarted when its last multiplier step grew.
struct DualMomentum {
    float t;
    float previous;
    float beta;
    uint32_t restarts;
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
    // The sweep path's unextrapolated multipliers, momentum per env and its running sweep's largest multiplier step.
    float* dual_base;
    DualMomentum* dual_momentum;
    uint32_t* dual_step;
    uint32_t* articulation_roots;
    uint32_t* articulation_dimensions;
    float* articulation_free;
    float* articulation_snapshot;
    float* articulation_matrix;
    float* articulation_force;
    float* articulation_direction;
    float* articulation_velocity;
    uint32_t* rigid_dimensions;
    float* rigid_free;
    float* rigid_snapshot;
    float* rigid_mass;
    float* rigid_matrix;
    float* rigid_force;
    float* rigid_direction;
    float* rigid_velocity;
    Vec3* grid_free;
    Vec3* grid_snapshot;
    uint32_t* grid_active;
    uint32_t* grid_active_count;
    uint32_t* grid_color_offsets;
    MaterialRowState* material_state;
    AugmentedRowState* row_state;
    nk::augmented::ContactRelaxation* contact_relaxation;
    void* scan;
    size_t scan_bytes;
    void* incidence_sort;
    size_t incidence_sort_bytes;
    uint32_t* large_segments;
    uint32_t* large_count;
    uint32_t large_capacity;
    uint32_t blocks;
    uint32_t rows;
    uint64_t incidence_capacity;
    uint32_t diagnostic_iteration = ~0u;
    uint32_t* sweep_state;
    // Per env and residual class, the running sweep's largest residual as float bits.
    uint32_t* env_residual;
    // A timed solve's last stamp and open phase, then nanoseconds per phase.
    unsigned long long* phase_clock;
    uint32_t* owner_cache_color;
    uint32_t* row_cache_colors;
    uint32_t* cache_counts;
    uint32_t* cache_offsets;
    uint32_t* cache_rows;
    uint32_t cache_color_count;
    uint32_t color_words;
    float* point_jacobian;
    nk::augmented::ScalarResponse* scalar_response;
    CoarseState* coarse;
    double* coarse_partials;
    uint32_t* coarse_rows;
    uint32_t* coarse_row_count;
    uint8_t* coarse_flags;
    uint32_t* coarse_row_offsets;
    void* coarse_select;
    size_t coarse_select_bytes;
    uint32_t coarse_parts;
    MultilevelState* multilevel;
    double* multilevel_partials;
    unsigned long long* multilevel_operator;
    Vec3* multilevel_force;
    Vec3* multilevel_direction;
    Vec3* multilevel_previous;
    Vec3* multilevel_correction;
    float* multilevel_restricted;
    SymmetricMat3* multilevel_block_inverse;
    float* multilevel_dense_factor;
    float* multilevel_dense_inverse;
    float* multilevel_element_hessian;
    Vec3* multilevel_element_gradient;
    Vec3* multilevel_row_impulse;
    float* multilevel_row_curvature;
    Vec3* multilevel_row_along;
    uint2* multilevel_touch;
    MultilevelRowTerm* multilevel_row_terms;
    uint8_t* multilevel_touch_count;
    uint32_t multilevel_parts;
    uint32_t multilevel_item_parts;
    float* tile_fields;
    float* tile_jacobian;
    uint32_t* tile_next;
    uint32_t* tile_order;
    uint32_t* tile_cost;
    float* articulation_pseudo;
    float* rigid_pseudo;
    uint32_t tile_head_capacity;
    uint32_t tile_row_capacity;
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

// The radix sort sees only the segments too long for a warp, listed in any order; spare entries are empty.
struct LargeIncidenceBound {
    const uint32_t* offsets;
    const uint32_t* segments;
    const uint32_t* count;
    uint64_t capacity;
    uint32_t end;
    __host__ __device__ uint32_t operator()(uint32_t at) const {
        return at < *count ? IncidenceOffset(offsets, capacity, segments[at] + end) : 0u;
    }
};
using LargeIncidenceBounds =
    thrust::transform_iterator<LargeIncidenceBound, thrust::counting_iterator<uint32_t>>;

uint64_t BindScratch(const BlockDescentSolveParams& p, uint32_t* base, BlockScratch* result) {
    const uint64_t blocks = uint64_t{p.total_particle_count} + p.total_body_count +
                            p.articulation_count + p.total_grid_count;
    const uint64_t rows = uint64_t{p.rows_per_env} * p.env_count;
    if (blocks >= uint64_t{std::numeric_limits<int>::max()} ||
        rows >= uint64_t{std::numeric_limits<int>::max()}) return 0u;
    // Dense tiles keep their rows per environment, without incidence lists, colors or row caches.
    const bool tile = DenseTileRoute(p);
    const auto schedule =
        nk::MakeBlockRowScheduleLayout(blocks, rows, tile ? 0u : p.incidence_per_env, p.env_count);
    const uint64_t incidence = schedule.incidence_capacity;
    if (incidence >= uint64_t{std::numeric_limits<uint32_t>::max()}) return 0u;
    const uint64_t word_limit = std::numeric_limits<size_t>::max() / sizeof(uint32_t);
    if (incidence > word_limit) return 0u;
    const bool incidence_terms = !tile && (p.total_particle_count > 0u || p.total_grid_count > 0u ||
                                           p.total_body_count > 0u || p.articulation_count > 0u);
    if (incidence_terms && incidence > word_limit / kPointJacobianWords) return 0u;
    const uint64_t point_jacobian_words = incidence_terms ? incidence * kPointJacobianWords : 0u;
    const uint64_t cache_colors = uint64_t{p.vertex_blocks.colors} + 2u +
        (p.total_grid_count > 0u ? nk::kMpmCellStencilNodes : 0u);
    if (cache_colors > uint64_t{std::numeric_limits<uint32_t>::max()}) return 0u;
    const uint64_t color_words = (cache_colors + 31u) / 32u;
    if (rows > word_limit / color_words) return 0u;
    const uint64_t row_color_words = tile ? 0u : rows * color_words;
    const uint64_t dofs = uint64_t{p.articulation_count} * p.max_dof;
    if (dofs >= uint64_t{std::numeric_limits<uint32_t>::max()} ||
        dofs > word_limit || (dofs > 0u && p.max_dof > word_limit / dofs)) return 0u;
    const uint64_t matrix_words = dofs * p.max_dof;
    if (matrix_words > word_limit - dofs ||
        dofs > (word_limit - matrix_words) / 7u) return 0u;
    size_t scan_bytes = 0u;
    if (!tile && cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<int>(blocks + 1u)) != cudaSuccess)
        return 0u;
    size_t incidence_sort_bytes = 0u;
    const uint64_t large_capacity = std::min(blocks, incidence / (kWarpSortLimit + 1u) + 1u);
    // Owner deduplication bounds each segment by rows, already below CUB's INT_MAX limit.
    if (!tile && blocks > 0u && incidence > 0u &&
        cub::DeviceSegmentedRadixSort::SortKeys(nullptr, incidence_sort_bytes,
            static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<int64_t>(incidence), static_cast<int64_t>(large_capacity),
            LargeIncidenceBounds(thrust::counting_iterator<uint32_t>(0u), LargeIncidenceBound{}),
            LargeIncidenceBounds(thrust::counting_iterator<uint32_t>(0u), LargeIncidenceBound{}),
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
    // Tiles factor their blocks and keep their multipliers' momentum in shared memory, without the sweeps' search state.
    const uint64_t sweep_blocks = tile ? 0u : blocks, sweep_envs = tile ? 0u : p.env_count;
    const uint64_t sweep_dofs = tile ? 0u : dofs, sweep_matrix = tile ? 0u : matrix_words;
    s.dual_base = reinterpret_cast<float*>(take(tile ? 0u : rows));
    s.dual_momentum = reinterpret_cast<DualMomentum*>(take(sweep_envs * sizeof(DualMomentum) / sizeof(uint32_t)));
    s.dual_step = take(sweep_envs);
    s.articulation_roots = take(dofs);
    s.articulation_dimensions = take(p.articulation_count);
    s.articulation_free = reinterpret_cast<float*>(take(dofs));
    s.articulation_snapshot = reinterpret_cast<float*>(take(sweep_dofs));
    s.articulation_matrix = reinterpret_cast<float*>(take(sweep_matrix));
    s.articulation_force = reinterpret_cast<float*>(take(sweep_dofs));
    s.articulation_direction = reinterpret_cast<float*>(take(sweep_dofs));
    s.articulation_velocity = reinterpret_cast<float*>(take(dofs));
    const uint64_t rigid_dofs = uint64_t{p.total_body_count} * kRigidBlockDof;
    const uint64_t sweep_rigid_dofs = tile ? 0u : rigid_dofs;
    s.rigid_dimensions = take(p.total_body_count);
    s.rigid_free = reinterpret_cast<float*>(take(rigid_dofs));
    s.rigid_snapshot = reinterpret_cast<float*>(take(rigid_dofs));
    s.rigid_mass = reinterpret_cast<float*>(take(rigid_dofs * kRigidBlockDof));
    s.rigid_matrix = reinterpret_cast<float*>(take(sweep_rigid_dofs * kRigidBlockDof));
    s.rigid_force = reinterpret_cast<float*>(take(sweep_rigid_dofs));
    s.rigid_direction = reinterpret_cast<float*>(take(sweep_rigid_dofs));
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
    s.large_capacity = incidence_sort_bytes > 0u ? static_cast<uint32_t>(large_capacity) : 0u;
    s.large_segments = take(s.large_capacity);
    s.large_count = take(s.large_capacity > 0u ? 1u : 0u);
    at = (at + 1u) & ~uint64_t{1u};
    s.row_state = reinterpret_cast<AugmentedRowState*>(take(
        tile ? 0u : rows * sizeof(AugmentedRowState) / sizeof(uint32_t)));
    s.contact_relaxation = reinterpret_cast<nk::augmented::ContactRelaxation*>(take(
        tile ? 0u : rows * sizeof(nk::augmented::ContactRelaxation) / sizeof(uint32_t)));
    s.owner_cache_color = take(sweep_blocks);
    s.row_cache_colors = take(row_color_words);
    s.cache_color_count = tile ? 0u : static_cast<uint32_t>(cache_colors);
    s.cache_counts = take(s.cache_color_count);
    s.cache_offsets = take(tile ? 0u : cache_colors + 1u);
    s.sweep_state = tile ? nullptr : take(kSweepStateWords);
    s.env_residual = take(tile ? 0u : uint64_t{p.env_count} * kResidualClasses);
    at = (at + 1u) & ~uint64_t{1u};
    s.phase_clock = reinterpret_cast<unsigned long long*>(take(tile ? 0u : 2u * (2u + kSolverPhaseWords)));
    s.cache_rows = s.sorted_incidence;
    s.color_words = static_cast<uint32_t>(color_words);
    s.point_jacobian = reinterpret_cast<float*>(take(point_jacobian_words));
    at = (at + 1u) & ~uint64_t{1u};
    // Tiles hold their scalar responses in the head fields; the stored copy serves only the residual measure.
    s.scalar_response = reinterpret_cast<nk::augmented::ScalarResponse*>(take(!tile || p.measure_contact_residual != 0u
        ? rows * sizeof(nk::augmented::ScalarResponse) / sizeof(uint32_t) : 0u));
    // Each head row stages one Jacobian row per incidence, a tile width of block coordinates; only rows
    // between two moving owners have a second incidence.
    at = (at + kTileLanes - 1u) & ~uint64_t{kTileLanes - 1u};
    s.tile_head_capacity = tile ? TileHeadCapacity(p) : 0u;
    s.tile_row_capacity = tile ? p.rows_per_env + p.distinct_owner_rows_per_env : 0u;
    const uint64_t tile_slots = tile ? TileWarpSlots(p) : 0u;
    if (tile && tile_slots == 0u) return 0u;
    s.tile_fields = reinterpret_cast<float*>(take(tile_slots * kTileFields * s.tile_head_capacity));
    s.tile_jacobian = reinterpret_cast<float*>(take(tile_slots * s.tile_row_capacity * TileJacobianWidth(p)));
    s.tile_next = take(tile ? 1u : 0u);
    s.tile_order = take(tile ? p.env_count : 0u);
    s.tile_cost = take(tile ? p.env_count : 0u);
    // Block coordinates of the tiles' pseudo velocities.
    const bool positions = tile && p.position_iterations > 0u;
    s.articulation_pseudo = reinterpret_cast<float*>(take(positions ? dofs : 0u));
    s.rigid_pseudo = reinterpret_cast<float*>(take(positions ? rigid_dofs : 0u));
    const bool coarse = p.total_particle_count > 0u;
    s.coarse_parts = CoarseParts(p);
    if (uint64_t{p.env_count} * s.coarse_parts >= uint64_t{std::numeric_limits<int>::max()}) return 0u;
    s.coarse =reinterpret_cast<CoarseState*>(take(
        coarse ? uint64_t{p.env_count} * sizeof(CoarseState) / sizeof(uint32_t) : 0u));
    s.coarse_partials = reinterpret_cast<double*>(take(coarse
        ? uint64_t{p.env_count} * s.coarse_parts * kCoarseTrials * (sizeof(double) / sizeof(uint32_t))
        : 0u));
    s.coarse_rows = take(coarse ? rows : 0u);
    s.coarse_row_count = take(coarse ? 1u : 0u);
    s.coarse_flags = reinterpret_cast<uint8_t*>(take(coarse ? (rows + 3u) / 4u : 0u));
    s.coarse_row_offsets = take(coarse ? uint64_t{p.env_count} + 1u : 0u);
    const VertexBlockLayout& layout = p.vertex_blocks;
    const bool multilevel = layout.dynamic_vertices > 0u && layout.coarse_levels > 0u;
    s.multilevel_parts = multilevel ? MultilevelParts(p) : 0u;
    s.multilevel_item_parts = multilevel ? MultilevelItemParts(p) : 0u;
    const uint64_t assemble_parts =
        uint64_t{layout.coarse_levels} * (s.multilevel_parts - s.multilevel_item_parts) + s.multilevel_item_parts;
    const uint64_t correct_parts = MultilevelCorrectParts(layout);
    if (uint64_t{p.env_count} * std::max({uint64_t{s.multilevel_parts}, assemble_parts, correct_parts,
                                          MultilevelOperatorEntries(layout)}) >=
            uint64_t{std::numeric_limits<int>::max()} ||
        layout.coarse_dense_nodes > layout.coarse_nodes) return 0u;
    const uint64_t element_slots = uint64_t{nk::PackVbdIncidence(1u, 0u)} * layout.elements * p.env_count;
    if (multilevel && element_slots > uint64_t{std::numeric_limits<uint32_t>::max()} - kThreads + 1u)
        return 0u;
    const uint64_t vertex_words = multilevel ? 3u * uint64_t{layout.vertices} * p.env_count : 0u;
    const uint64_t rank = 3u * uint64_t{layout.coarse_dense_nodes};
    at = (at + 1u) & ~uint64_t{1u};
    s.multilevel = reinterpret_cast<MultilevelState*>(take(
        multilevel ? uint64_t{p.env_count} * sizeof(MultilevelState) / sizeof(uint32_t) : 0u));
    s.multilevel_partials = reinterpret_cast<double*>(take(multilevel
        ? uint64_t{p.env_count} * s.multilevel_parts * kMultilevelSums * (sizeof(double) / sizeof(uint32_t))
        : 0u));
    s.multilevel_operator = reinterpret_cast<unsigned long long*>(take(multilevel
        ? uint64_t{p.env_count} * MultilevelOperatorEntries(layout) * (sizeof(uint64_t) / sizeof(uint32_t))
        : 0u));
    s.multilevel_force = reinterpret_cast<Vec3*>(take(vertex_words));
    s.multilevel_direction = reinterpret_cast<Vec3*>(take(vertex_words));
    s.multilevel_previous = reinterpret_cast<Vec3*>(take(vertex_words));
    s.multilevel_correction = reinterpret_cast<Vec3*>(take(
        multilevel ? 3u * uint64_t{layout.coarse_nodes} * p.env_count : 0u));
    s.multilevel_restricted = reinterpret_cast<float*>(take(multilevel ? rank * p.env_count : 0u));
    s.multilevel_block_inverse = reinterpret_cast<SymmetricMat3*>(take(multilevel
        ? 6u * uint64_t{layout.coarse_nodes - layout.coarse_dense_nodes} * p.env_count : 0u));
    s.multilevel_dense_inverse = reinterpret_cast<float*>(take(multilevel ? rank * rank * p.env_count : 0u));
    s.multilevel_dense_factor = reinterpret_cast<float*>(
        take(multilevel ? MultilevelDenseFactorWords(static_cast<uint32_t>(rank)) * p.env_count : 0u));
    s.multilevel_element_hessian = reinterpret_cast<float*>(take(multilevel
        ? uint64_t{kMultilevelElementHessian} * layout.elements * p.env_count : 0u));
    s.multilevel_element_gradient = reinterpret_cast<Vec3*>(take(multilevel
        ? 3u * element_slots : 0u));
    s.multilevel_row_impulse = reinterpret_cast<Vec3*>(take(multilevel ? 3u * rows : 0u));
    s.multilevel_row_curvature = reinterpret_cast<float*>(take(multilevel ? 4u * rows : 0u));
    s.multilevel_row_along = reinterpret_cast<Vec3*>(take(multilevel ? 6u * rows : 0u));
    at = (at + 1u) & ~uint64_t{1u};
    s.multilevel_touch = reinterpret_cast<uint2*>(
        take(multilevel ? 2u * uint64_t{kMultilevelRowPoints} * rows : 0u));
    s.multilevel_touch_count = reinterpret_cast<uint8_t*>(take(multilevel ? (rows + 3u) / 4u : 0u));
    at = (at + 1u) & ~uint64_t{1u};
    s.multilevel_row_terms = reinterpret_cast<MultilevelRowTerm*>(take(multilevel
        ? rows * sizeof(MultilevelRowTerm) / sizeof(uint32_t) : 0u));
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

// An env sweeps until its residuals pass the tolerance; vbd_velocity_sweep_count then holds its sweeps.
__device__ bool EnvRunning(const DataView& data, uint32_t env) { return data.vbd_velocity_sweep_count[env] == 0u; }

__device__ float StepSize(Vec3 step) { return fmaxf(fabsf(step.x), fmaxf(fabsf(step.y), fabsf(step.z))); }

// A residual's float bits, ordered as integers since residuals are not negative; a non-finite one is the largest.
__device__ uint32_t ResidualBits(float value) { return __float_as_uint(value <= FLT_MAX ? fabsf(value) : FLT_MAX); }

__device__ void RaiseResidual(BlockScratch s, uint32_t env, uint32_t kind, uint32_t bits) {
    atomicMax(s.env_residual + size_t{env} * kResidualClasses + kind, bits);
}

// RaiseResidual from every lane of a warp, once per env the lanes hold; a lane without one passes ~0u.
__device__ void RaiseWarpResidual(BlockScratch s, uint32_t env, uint32_t kind, uint32_t bits) {
    const uint32_t peers = __match_any_sync(~0u, env);
    const uint32_t largest = __reduce_max_sync(peers, bits);
    if (env != ~0u && threadIdx.x % warpSize == __ffs(peers) - 1u) RaiseResidual(s, env, kind, largest);
}

// A row's multiplier steps at its dual update in velocity units: the normal (or scalar) step over its residual
// scale, the tangent steps over their cone penalties rho mu^2, which is the slip velocity inside the cone.
__device__ float2 MultiplierSteps(Vec3 dual, Vec3 next, float scale, float rho, float mu_first, float mu_second) {
    const float normal = scale > 0.0f ? fabsf(next.x - dual.x) / scale : 0.0f;
    const float first = mu_first > 0.0f && rho > 0.0f ? (next.y - dual.y) / (rho * mu_first * mu_first) : 0.0f;
    const float second = mu_second > 0.0f && rho > 0.0f ? (next.z - dual.z) / (rho * mu_second * mu_second) : 0.0f;
    return {normal, hypotf(first, second)};
}

// Velocity of a contact axis along its own impulse, which a converged contact keeps from being positive.
__device__ float Along(float impulse, float velocity) {
    return impulse > 0.0f ? velocity : impulse < 0.0f ? -velocity : 0.0f;
}

__host__ __device__ inline DualMomentum StartDualMomentum() { return {1.0f, FLT_MAX, 0.0f, 0u}; }

// One FISTA step of an env's momentum from its last largest multiplier step; a grown step restarts it at rest.
__device__ void AdvanceDualMomentum(DualMomentum& m, float step) {
    if (step > m.previous) {
        m.t = 1.0f;
        ++m.restarts;
    }
    m.previous = step;
    const float t = 0.5f * (1.0f + sqrtf(fmaf(4.0f * m.t, m.t, 1.0f)));
    m.beta = (m.t - 1.0f) / t;
    m.t = t;
}

// A row's next multiplier at its residual: the bound projection `normal`, the friction cone of that normal
// impulse and, on contacts, the secant relaxation of the step.
__device__ Vec3 ProjectRowDual(Vec3 dual, float normal, float rho, Vec3 residual, bool contact, float mu_first,
                               float mu_second, nk::augmented::ContactRelaxation* relaxation) {
    if (!contact) return {normal, 0.0f, 0.0f};
    const auto f = nk::augmented::EvaluateTangentImpulse(dual, rho, residual, normal, mu_first, mu_second);
    return nk::augmented::RelaxContactImpulse(dual, {normal, f.impulse.y, f.impulse.z}, relaxation);
}

// The Nesterov point beta past `next` away from `base`, projected onto the row's bounds and the friction cone
// of its normal multiplier.
__device__ Vec3 ExtrapolateDual(Vec3 base, Vec3 next, float beta, float lower, float upper, bool contact,
                                float mu_first, float mu_second) {
    if (beta == 0.0f) return next;
    Vec3 y{fmaf(beta, next.x - base.x, next.x), fmaf(beta, next.y - base.y, next.y),
           fmaf(beta, next.z - base.z, next.z)};
    y.x = fminf(fmaxf(y.x, lower), upper);
    if (!contact) return {y.x, 0.0f, 0.0f};
    const float first = mu_first > 0.0f ? y.y / mu_first : 0.0f;
    const float second = mu_second > 0.0f ? y.z / mu_second : 0.0f;
    const float slip = hypotf(first, second), radius = fmaxf(y.x, 0.0f);
    const float scale = slip > radius ? radius / slip : 1.0f;
    return {y.x, mu_first > 0.0f ? y.y * scale : 0.0f, mu_second > 0.0f ? y.z * scale : 0.0f};
}

// atomicMax of `bits` into words[env] from every lane of a warp, once per env the lanes hold; a lane without one
// passes ~0u.
__device__ void RaiseWarpMax(uint32_t* words, uint32_t env, uint32_t bits) {
    const uint32_t peers = __match_any_sync(~0u, env);
    const uint32_t largest = __reduce_max_sync(peers, bits);
    if (env != ~0u && threadIdx.x % warpSize == __ffs(peers) - 1u) atomicMax(words + env, largest);
}

// Whether an env's class already exceeds the tolerance this sweep, so its remaining elements cannot stop it.
__device__ bool Past(const BlockDescentSolveParams& p, BlockScratch s, uint32_t env, uint32_t kind, uint32_t local) {
    const volatile uint32_t* words = s.env_residual + size_t{env} * kResidualClasses;
    return __uint_as_float(max(words[kind], local)) >= p.velocity_tolerance;
}

// Whether a sweep measures the contact and cloth vertex residuals: every sweep under a tolerance or a sweep
// trace, else the last one. Without a trace, `prune` lets an env past the tolerance skip the rest.
__device__ bool MeasuredSweep(const BlockDescentSolveParams& p, BlockScratch s, bool* prune) {
    const bool trace = (p.telemetry & nk::kTelemetrySweeps) != 0u;
    const bool last = s.sweep_state[kSweepCount] + 1u >= p.iterations;
    *prune = p.velocity_tolerance > 0.0f && !trace && !last;
    return p.velocity_tolerance > 0.0f || trace || last;
}

__device__ unsigned long long GlobalTimer() {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// Stress cells reach lattice-1 nodes i-1..i+1; point rows span two nodes per axis on either lattice.
__host__ __device__ inline uint32_t GridLatticeWidth(const BlockDescentSolveParams& p, uint32_t lattice) {
    return nk::kMpmStencilWidth + (lattice != 0u && p.material_cells_per_env > 0u ? 1u : 0u);
}

__host__ __device__ inline uint32_t GridColorCount(const BlockDescentSolveParams& p) {
    const uint32_t width = GridLatticeWidth(p, 1u);
    return nk::kMpmLatticeStencilNodes + width * width * width;
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
    const uint32_t width = GridLatticeWidth(p, lattice);
    const uint32_t first = lattice == 0u ? 0u : nk::kMpmLatticeStencilNodes;
    return first + x % width + width * (y % width + width * (z % width));
}

// Whether every descending owner of a row descends in a grid color.
__device__ bool GridOnlyRow(BlockScratch s, uint32_t colors, uint32_t slot) {
    const uint32_t* words = s.row_cache_colors + size_t{slot} * s.color_words;
    for (uint32_t word = 0u; word < s.color_words; ++word)
        if (words[word] != 0u) return word * 32u + static_cast<uint32_t>(__ffs(words[word])) - 1u >= colors + 2u;
    return false;
}

// The lowest cache color of a row whose owners all descend in particle colors, else kNoCacheColor.
__device__ uint32_t ParticleRowFirstColor(BlockScratch s, uint32_t colors, uint32_t slot) {
    const uint32_t* words = s.row_cache_colors + size_t{slot} * s.color_words;
    uint32_t first = kNoCacheColor;
    for (uint32_t word = 0u; word < s.color_words; ++word) {
        const uint32_t bits = words[word];
        if (bits == 0u) continue;
        if (first == kNoCacheColor) first = word * 32u + static_cast<uint32_t>(__ffs(bits)) - 1u;
        if (word * 32u + 31u - static_cast<uint32_t>(__clz(bits)) >= colors) return kNoCacheColor;
    }
    return first;
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
            // Static bodies never descend, so their rows need no rigid cache.
            const uint32_t body = owner - dense_first;
            if (body >= p.total_body_count || data.body_inv_mass[body] > 0.0f)
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
            static_cast<uint32_t>(nk::BlockSolveFailureEquationColumn::Iteration)] =
            s.sweep_state != nullptr ? s.sweep_state[kSweepCount] : s.diagnostic_iteration;
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
        const float* const J = jacobians + size_t{slot} * p.max_dof;
        const float* const factor = data.m_factor + size_t{side.index} * p.max_dof * p.max_dof;
        return static_cast<float>(MassCoupling<double>(factor, p.max_dof, J, J));
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

// PointJacobian of the first `axes` rows of the group headed by `head`; a contact group's rows share their
// endpoints and differ only in direction, so one search per side serves every axis.
__device__ void PointJacobians(const NkRow* rows, uint32_t slot, const NkRow& head, uint32_t axes,
                               PointMassView points, uint32_t kind, uint32_t index, Vec3* jacobian) {
    const NkRowSide sides[2] = {head.a, head.b};
    for (uint32_t side = 0u; side < 2u; ++side) {
        if (!PointMassView::IsPointSide(sides[side].kind)) continue;
        const uint32_t count = points.Count(sides[side]);
        uint32_t first = 0u;
        uint32_t last = count;
        while (first < last) {
            const uint32_t middle = first + (last - first) / 2u;
            const auto term = points.At(sides[side], middle);
            if (term.kind < kind || (term.kind == kind && term.index < index))
                first = middle + 1u;
            else
                last = middle;
        }
        for (uint32_t i = first; i < count; ++i) {
            const auto term = points.At(sides[side], i);
            if (term.kind != kind || term.index != index) break;
            for (uint32_t axis = 0u; axis < axes; ++axis) {
                const NkRow& row = axis == 0u ? head : rows[slot + axis * head.group_normal_count];
                jacobian[axis] += points.At(side == 0u ? row.a : row.b, i).jacobian;
            }
        }
    }
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
            Vec3 jacobian[kPointJacobianAxes]{};
            PointJacobians(rows, slot, head, axes, points, kind, index, jacobian);
            for (uint32_t axis = 0u; axis < kPointJacobianAxes; ++axis) {
                const uint64_t first = uint64_t{axis} * kPointJacobianComponents *
                    s.incidence_capacity + at;
                s.point_jacobian[first] = jacobian[axis].x;
                s.point_jacobian[first + s.incidence_capacity] = jacobian[axis].y;
                s.point_jacobian[first + 2u * s.incidence_capacity] = jacobian[axis].z;
            }
        }
    }
}

// One owner's share of a row's J M^-1 J^T; sets `grid` when a grid node is the owner.
__device__ double OwnerInertialResponse(const NkRow& row, const DataView& data,
                                        const BlockDescentSolveParams& p, PointMassView points,
                                        uint32_t slot, uint32_t owner, bool* grid) {
    if (owner < p.total_particle_count ||
        owner >= p.total_particle_count + p.total_body_count + p.articulation_count) {
        const uint32_t kind = owner < p.total_particle_count ? kNkSideParticle : kNkSideGrid;
        const uint32_t index = kind == kNkSideParticle ? owner
            : owner - p.total_particle_count - p.total_body_count - p.articulation_count;
        *grid |= kind == kNkSideGrid;
        const Vec3 jacobian = PointJacobian(row, points, kind, index);
        const double norm = double(jacobian.x) * jacobian.x + double(jacobian.y) * jacobian.y +
                            double(jacobian.z) * jacobian.z;
        return double(InertialPointResponse(PointMassView::Contribution{kind, index, {}}, data, p, points)) * norm;
    }
    if (owner < p.total_particle_count + p.total_body_count) {
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
        return double(data.body_inv_mass[body]) *
            (double(linear.x) * linear.x + double(linear.y) * linear.y + double(linear.z) * linear.z) +
            x * (double(inverse.xx) * x + double(inverse.xy) * y + double(inverse.xz) * z) +
            y * (double(inverse.xy) * x + double(inverse.yy) * y + double(inverse.yz) * z) +
            z * (double(inverse.xz) * x + double(inverse.yz) * y + double(inverse.zz) * z);
    }
    const uint32_t articulation = owner - p.total_particle_count - p.total_body_count;
    const size_t at = size_t{slot} * p.max_dof;
    const bool side_a = row.a.kind == kNkSideArtic && row.a.index == articulation;
    const bool side_b = row.b.kind == kNkSideArtic && row.b.index == articulation;
    const float* const J = (side_a ? data.chain_jacobian : data.chain_jacobian_b) + at;
    const float* const extra = side_a && side_b ? data.chain_jacobian_b + at : nullptr;
    const float* const factor = data.m_factor + size_t{articulation} * p.max_dof * p.max_dof;
    return MassCoupling<double>(factor, p.max_dof, J, J, extra, extra);
}

// J M^-1 J^T of a row over its owners; sets `grid` when a grid node owns it and `coupling` to the
// largest product of one owner's share with the other owners' shares, `largest` to the largest share.
__device__ double RowInertialResponse(const NkRow& row, const DataView& data,
                                     const BlockDescentSolveParams& p, PointMassView points,
                                     uint32_t slot, bool* grid, double* coupling, double* largest) {
    double response = 0.0;
    bool valid = true;
    VisitOwners(row, points, p, [&](uint32_t owner) {
        const double contribution = OwnerInertialResponse(row, data, p, points, slot, owner, grid);
        valid &= contribution >= 0.0 && isfinite(contribution);
        response += contribution;
    });
    *coupling = 0.0;
    *largest = 0.0;
    if (!valid || !isfinite(response)) return nan("");
    VisitOwners(row, points, p, [&](uint32_t owner) {
        const double contribution = OwnerInertialResponse(row, data, p, points, slot, owner, grid);
        *coupling = fmax(*coupling, contribution * (response - contribution));
        *largest = fmax(*largest, contribution);
    });
    return response;
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
        bool grid = false;
        double coupling = 0.0, largest = 0.0;
        const double response = RowInertialResponse(row, data, p, points, slot, &grid, &coupling, &largest);
        // Owners a row ties descend against each other's stale velocities at the block Gauss-Seidel factor
        // rho^2 P / (1 + rho W + rho^2 P), P = max w_i (W - w_i); the root below keeps it at most 1/4.
        const double coupled = coupling > 0.0
            ? (response + sqrt(response * response + 12.0 * coupling)) / (6.0 * coupling) : HUGE_VAL;
        const double cap = grid ? kGridRowPenaltyScale / response : kPenaltyConditionCap / largest;
        const double penalty = response > 0.0 ? fmin(cap, coupled) : 0.0;
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
        for (uint32_t word = 0u; word < s.color_words; ++word)
            for (uint32_t bits = cache_colors[word]; bits != 0u; bits &= bits - 1u)
                atomicAdd(s.cache_counts + word * 32u + __ffs(bits) - 1u, 1u);
        s.active_rows[atomicAdd(s.active_count, 1u)] = slot;
    }
}

// Sorted rows return to CSR storage before the temporary array becomes color buckets.
__global__ void PrepareCacheBucketsKernel(BlockScratch s) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t at = item; at < s.offsets[s.blocks]; at += gridDim.x * blockDim.x)
        s.incidence[at] = s.sorted_incidence[at];
    if (item == 0u) {
        uint32_t offset = 0u;
        for (uint32_t color = 0u; color < s.cache_color_count; ++color) {
            s.cache_offsets[color] = offset;
            offset += s.cache_counts[color];
            s.cache_counts[color] = 0u;
        }
        s.cache_offsets[s.cache_color_count] = offset;
    }
}

__global__ void FillCacheBucketsKernel(BlockScratch s) {
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.active_rows[item];
        for (uint32_t word = 0u; word < s.color_words; ++word) {
            const uint32_t colors = s.row_cache_colors[size_t{slot} * s.color_words + word];
            for (uint32_t bits = colors; bits != 0u; bits &= bits - 1u) {
                const uint32_t color = word * 32u + __ffs(bits) - 1u;
                const uint32_t at = s.cache_offsets[color] + atomicAdd(s.cache_counts + color, 1u);
                s.cache_rows[at] = slot;
            }
        }
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
        if (count > kWarpSortLimit) {
            if (lane == 0u) s.large_segments[atomicAdd(s.large_count, 1u)] = segment;
            continue;
        }
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
    BindFieldTables(b, model, data);
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
    // Friction cone radius held through a block solve: the normal multiplier, or a tile's block-start bound.
    float radius;
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

__device__ nk::augmented::ScalarTerm AugmentedRowBoundTerm(BlockScratch s, const NkRow& normal, uint32_t slot,
                                                           float scale, const AugmentedRowState& state) {
    return nk::augmented::EvaluateScalarImpulse(state.dual.x, s.penalty[slot], state.residual.x,
        normal.compliance_alpha * scale, normal.lower, normal.upper, s.scalar_response[slot]);
}

__device__ float AugmentedRowBound(BlockScratch s, const NkRow& normal, uint32_t slot, float scale,
                                   const AugmentedRowState& state) {
    return AugmentedRowBoundTerm(s, normal, slot, scale, state).impulse;
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
                                   uint32_t slot, uint32_t particle, bool snapshot,
                                   const AugmentedRowState* current = nullptr) {
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
    const AugmentedRowState state = current != nullptr ? *current : snapshot ? s.row_state[slot]
        : ComputeAugmentedRowState(data, p, s, slot, false);
    t.residual = state.residual;
    t.dual = state.dual;
    t.normal_bound = state.normal_bound;
    t.radius = state.dual.x;
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

// Folds a row's impulse and its curvature (xx, yy, zz, yz used) into a point block.
__device__ void AddLocalResponse(const Vec3* jacobian, Vec3 f, const SymmetricMat3& c,
                                 Vec3* impulse, SymmetricMat3* curvature) {
    *impulse += jacobian[0] * f.x + jacobian[1] * f.y + jacobian[2] * f.z;
    AddOuter(*curvature, jacobian[0], c.xx);
    AddOuter(*curvature, jacobian[1], c.yy - c.yz);
    AddOuter(*curvature, jacobian[2], c.zz - c.yz);
    AddOuter(*curvature, jacobian[1] + jacobian[2], c.yz);
}

// EvaluateAugmentedResponse given the normal term `n` already evaluated at the residual less row_move.x.
__device__ void EvaluateAugmentedResponse(const LocalTerm& t, const nk::augmented::ScalarTerm& n, Vec3 row_move,
                                          Vec3* impulse, SymmetricMat3* curvature) {
    *impulse = {n.impulse, 0.0f, 0.0f};
    *curvature = {n.curvature, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (!t.contact) return;
    const Vec3 tangent_residual{0.0f, t.residual.y - row_move.y, t.residual.z - row_move.z};
    const auto f = nk::augmented::EvaluateTangentImpulse(t.dual, t.penalty, tangent_residual,
                                                         t.radius, t.mu_first, t.mu_second);
    impulse->y = f.impulse.y;
    impulse->z = f.impulse.z;
    curvature->yy = f.curvature.yy;
    curvature->zz = f.curvature.zz;
    curvature->yz = f.curvature.yz;
}

// A row's impulse and curvature at its residual less row_move; the cone runs in float.
__device__ void EvaluateAugmentedResponse(const LocalTerm& t, Vec3 row_move, Vec3* impulse,
                                          SymmetricMat3* curvature) {
    EvaluateAugmentedResponse(t, nk::augmented::EvaluateScalarImpulse(t.dual.x, t.penalty, t.residual.x - row_move.x,
        t.compliance, t.lower, t.upper, t.response), row_move, impulse, curvature);
}

__device__ void EvaluateLocalResponse(const LocalTerm& t, Vec3 move, Vec3* impulse,
                                      SymmetricMat3* curvature) {
    Vec3 f{};
    SymmetricMat3 c{};
    EvaluateAugmentedResponse(t,
        {t.jacobian[0].Dot(move), t.jacobian[1].Dot(move), t.jacobian[2].Dot(move)}, &f, &c);
    AddLocalResponse(t.jacobian, f, c, impulse, curvature);
}

// The row potential change as its residual drops by `drop`, integrated along it instead of subtracted.
template <typename Response>
__device__ float EvaluateDropChange(const LocalTerm& t, Vec3 drop, const Response& response) {
    float change = nk::augmented::ScalarPotentialChange(t.dual.x, t.penalty, t.residual.x, drop.x,
        t.compliance, t.lower, t.upper, response);
    if (t.contact)
        change += nk::augmented::TangentPotentialChange(t.dual, t.penalty, t.residual, drop,
                                                        t.radius, t.mu_first, t.mu_second);
    return change;
}

__device__ float EvaluateDropChange(const LocalTerm& t, Vec3 drop) {
    return EvaluateDropChange(t, drop, t.response);
}

// The row potential change as the block moves by `move`.
__device__ float EvaluateLocalChange(const LocalTerm& t, Vec3 move) {
    return EvaluateDropChange(t, {t.jacobian[0].Dot(move), t.jacobian[1].Dot(move), t.jacobian[2].Dot(move)});
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
#include "phi/backend_cuda/ops/block_descent_multilevel.cuh"
#include "phi/backend_cuda/ops/block_descent_tile.cuh"

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

// Rows too few to give every warp scheduler a warp are latency-bound, so adjacent lanes evaluate one
// row's axes and the group's first lane caches it; otherwise each lane caches whole rows.
__global__ void CacheAugmentedRowsKernel(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                        uint32_t cache_color, uint32_t spread_rows, bool seeded) {
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t first = s.cache_offsets[cache_color];
    const uint32_t count = s.cache_offsets[cache_color + 1u] - first;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    if (count > spread_rows) {
        for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < count; item += threads) {
            const uint32_t slot = s.cache_rows[first + item];
            if (seeded && ParticleRowFirstColor(s, p.vertex_blocks.colors, slot) == cache_color) continue;
            const NkRow row = rows[slot];
            if (row.flags & nk::nk_row_flags::kMaterialBlock) continue;
            s.row_state[slot] = ComputeAugmentedRowState(data, p, s, slot, true);
        }
        return;
    }
    const uint32_t axis = threadIdx.x % kRowAxisLanes;
    const uint32_t group_mask = ((1u << kRowAxisLanes) - 1u) << (threadIdx.x % warpSize - axis);
    for (uint32_t item = (blockIdx.x * blockDim.x + threadIdx.x) / kRowAxisLanes; item < count;
         item += threads / kRowAxisLanes) {
        const uint32_t slot = s.cache_rows[first + item];
        if (seeded && ParticleRowFirstColor(s, p.vertex_blocks.colors, slot) == cache_color) continue;
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
        s.row_state[slot] = state;
    }
}

// Point rows of grid owners, cached once per grid sweep unless the dual update seeded them; grid
// descents then update them in place.
__global__ void CacheGridRowsKernel(DataView data, BlockDescentSolveParams p, BlockScratch s, bool seeded) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const uint32_t first = p.vertex_blocks.colors + 2u;
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.active_rows[item];
        if (rows[slot].flags & nk::nk_row_flags::kMaterialBlock) continue;
        if (seeded && GridOnlyRow(s, p.vertex_blocks.colors, slot)) continue;
        const uint32_t* words = s.row_cache_colors + size_t{slot} * s.color_words;
        bool grid = false;
        for (uint32_t word = first / 32u; word < s.color_words && !grid; ++word) {
            const uint32_t low = word == first / 32u ? first % 32u : 0u;
            grid = (words[word] >> low) != 0u;
        }
        if (grid) s.row_state[slot] = ComputeAugmentedRowState(data, p, s, slot, true);
    }
}

// A block descends a tile of one vertex per warp. Its threads first evaluate the tile's element blocks
// (grouped by kind) and row responses densely; each warp then sums, solves and searches its own vertex.
constexpr uint32_t kDescendTileVertices = kThreads / 32u;
constexpr uint32_t kDescendTileElements = 384u;
constexpr uint32_t kDescendTileRows = 256u;
// Vertex-block element kinds, then the two material-field kinds.
constexpr uint32_t kDescendVertexKinds = 4u;
constexpr uint32_t kDescendElementKinds = kDescendVertexKinds + 2u;
constexpr uint32_t kElementSampleWords = 9u;
constexpr uint32_t kRowSampleWords = 7u;
constexpr uint32_t kDescendSolveWords = 12u;
static_assert(kDescendTileVertices * kDescendSolveWords <= kDescendTileRows);

struct DescendTileVertex {
    uint32_t active;
    uint32_t descend;
    uint32_t trial;
    uint32_t field;
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
    double unit_slope;
    double unit_move;
};

// The tile vertex owning tile-local sample `k`: the last one whose samples start at or before it.
__device__ uint32_t DescendTileOwner(const DescendTileVertex* tile, uint32_t k, bool rows) {
    uint32_t j = 0u;
    while (j + 1u < kDescendTileVertices &&
           (rows ? tile[j + 1u].row_first : tile[j + 1u].element_first) <= k) ++j;
    return j;
}

__device__ uint32_t TileIncidence(const VertexBlockView& b, const DescendTileVertex& v, uint32_t k) {
    const uint32_t at = v.element_begin + k - v.element_first;
    return v.field != 0u ? b.field_incidence[at] : b.incidence[at];
}

__device__ uint32_t DescendElementKind(const VertexBlockView& b, const DescendTileVertex& v,
                                       uint32_t k) {
    const uint32_t packed = TileIncidence(b, v, k);
    if (v.field != 0u) return kDescendVertexKinds + b.field_elements[nk::MfIncidenceElement(packed)].kind;
    const uint32_t kind = b.elements[nk::VbdIncidenceElement(packed)].kind;
    return kind < kDescendVertexKinds ? kind : kDescendVertexKinds - 1u;
}

__device__ Vec3 TileVec3(const float* value) { return {value[0], value[1], value[2]}; }

__device__ void SetTileVec3(float* value, Vec3 v) {
    value[0] = v.x;
    value[1] = v.y;
    value[2] = v.z;
}

// At scale 1 TrialValue rounds an exact sum to double, then to float; binary64 holds 2*24+2 bits, so
// one float add gives the same value. A NaN sum keeps TrialValue's payload.
__device__ Vec3 VertexTrial(Vec3 u, Vec3 direction, float scale) {
    if (scale == 1.0f) {
        const Vec3 sum{__fadd_rn(u.x, direction.x), __fadd_rn(u.y, direction.y), __fadd_rn(u.z, direction.z)};
        if (sum.x == sum.x && sum.y == sum.y && sum.z == sum.z) return sum;
    }
    return {nk::vbd::TrialValue(u.x, direction.x, scale), nk::vbd::TrialValue(u.y, direction.y, scale),
            nk::vbd::TrialValue(u.z, direction.z, scale)};
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
            v.active = v.descend = v.trial = v.field = 0u;
            v.element_count = v.row_count = 0u;
            if (item < items) {
                const uint32_t env = item / count;
                const uint32_t vertex = elastic ? b.color_vertices[first + item % count] : 0u;
                const uint32_t particle = elastic ? b.Particle(env, vertex) : item;
                const uint32_t local = particle % b.layout.particles_per_env;
                const bool skipped = !elastic && (local < p.grid_particles_per_env ||
                    (local >= b.layout.begin && local - b.layout.begin < b.layout.vertices));
                if (!skipped && data.particle_inv_mass[particle] > 0.0f && EnvRunning(data, env)) {
                    v.active = 1u;
                    v.env = env;
                    v.vertex = vertex;
                    v.particle = particle;
                    VertexElementList list{nullptr, 0u, 0u, false};
                    if (elastic) list = ElementsOf(b, vertex);
                    v.field = list.field ? 1u : 0u;
                    v.element_begin = list.begin;
                    v.element_count = list.end - list.begin;
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
                ListElementBlock(b, data.particle_vel, v.env, data.particle_vel[v.particle], v.field != 0u,
                                 TileIncidence(b, v, k), g, h);
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
                float* sums = row_changes + j * kDescendSolveWords;
                SetTileVec3(sums, gradient);
                SetTileVec3(sums + 3, impulse);
                sums[6] = hessian.xx;
                sums[7] = hessian.yy;
                sums[8] = hessian.zz;
                sums[9] = hessian.xy;
                sums[10] = hessian.xz;
                sums[11] = hessian.yz;
                SetTileVec3(v.u, u);
                SetTileVec3(v.free_rate, free_rate);
                v.inertia = inertia;
            }
        }
        __syncthreads();
        // Each vertex's 3x3 solve runs on its own lane of the first warp, from the sums its warp stored;
        // row_changes is free until the trial pass below.
        if (threadIdx.x < kDescendTileVertices && tile[threadIdx.x].active != 0u) {
            DescendTileVertex& v = tile[threadIdx.x];
            const float* sums = row_changes + threadIdx.x * kDescendSolveWords;
            const Vec3 u = TileVec3(v.u);
            const Vec3 free_rate = TileVec3(v.free_rate);
            const float inertia = v.inertia;
            const Vec3 gradient = TileVec3(sums);
            const Vec3 impulse = TileVec3(sums + 3);
            SymmetricMat3 hessian{sums[6], sums[7], sums[8], sums[9], sums[10], sums[11]};
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
                else RecordBlockFailure(data, p, s, v.particle, nk::BlockSolveFailure::InvalidDirection);
            } else {
                RecordBlockFailure(data, p, s, v.particle, Finite(force)
                    ? nk::BlockSolveFailure::Factorization : nk::BlockSolveFailure::InvalidEquation);
            }
            SetTileVec3(v.direction, direction);
            SetTileVec3(v.primal_force, primal_force);
            v.slope = slope;
            v.descend = valid;
            v.trial = valid != 0u && slope < 0.0f ? 1u : 0u;
            // The line search starts at the unit step, whose slope and inertial factors are formed here once.
            if (v.trial != 0u) {
                const Vec3 move = VertexTrial(u, direction, 1.0f) - u;
                v.unit_slope = -double(p.dt) * (double(primal_force.x) * move.x +
                    double(primal_force.y) * move.y + double(primal_force.z) * move.z);
                v.unit_move = nk::vbd::InertialMoveChange(u, free_rate, move);
            }
        }
        __syncthreads();
        // Unless the sweep measures the vertex equations themselves, the Newton steps are their residuals.
        if (threadIdx.x == 0u && p.measure_contact_residual == 0u) {
            uint32_t env = ~0u, largest = 0u;
            for (uint32_t j = 0u; j < kDescendTileVertices; ++j) {
                const DescendTileVertex& v = tile[j];
                if (v.active == 0u) continue;
                if (v.env != env && env != ~0u) {
                    RaiseResidual(s, env, kResidualMomentum, largest);
                    largest = 0u;
                }
                env = v.env;
                largest = max(largest, ResidualBits(v.descend != 0u ? StepSize(TileVec3(v.direction)) : FLT_MAX));
            }
            if (env != ~0u) RaiseResidual(s, env, kResidualMomentum, largest);
        }
        for (uint32_t q = threadIdx.x; q < dense_total; q += blockDim.x) {
            const bool element = q < dense_element_total;
            const uint32_t k = element ? order[q] : q - dense_element_total;
            const DescendTileVertex& v = tile[DescendTileOwner(tile, k, !element)];
            if (v.trial == 0u) continue;
            const Vec3 u = TileVec3(v.u);
            const Vec3 direction = TileVec3(v.direction);
            const Vec3 candidate = VertexTrial(u, direction, 1.0f);
            const Vec3 move = candidate - u;
            if (element) {
                const VertexElementChangeTerms terms = ListElementChange(b, data.particle_vel, v.env,
                    v.field != 0u, TileIncidence(b, v, k), u, move);
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
            const float smallest = ldexpf(1.0f, -static_cast<int>(kVertexStepHalvings));
            for (uint32_t halving = 0u; scale >= smallest && slope < 0.0f; ++halving) {
                const Vec3 candidate = VertexTrial(u, direction, scale);
                const Vec3 move = candidate - u;
                const double trial_slope = halving == 0u ? v.unit_slope
                    : __shfl_sync(0xffffffffu, -double(p.dt) * (
                          double(primal_force.x) * move.x + double(primal_force.y) * move.y +
                          double(primal_force.z) * move.z), 0u);
                const bool sampled = halving == 0u;
                float elastic_change = 0.0f, size = 0.0f;
                if constexpr (elastic) {
                    if (sampled && dense_elements) {
                        for (uint32_t i = lane; i < v.element_count; i += warpSize) {
                            const float* sample = element_samples[v.element_first + i];
                            AddVertexElementChange(b, {sample[0], sample[1], sample[2]}, elastic_change, &size);
                        }
                    } else {
                        elastic_change = VertexEnergyChange(b, data.particle_vel, env, vertex, u, move,
                                                            lane, warpSize, &size);
                    }
                }
                float row_change = 0.0f;
                for (uint32_t at = cached_at; at < end; at += warpSize) {
                    const float term = sampled && dense_rows ? row_changes[v.row_first + (at - begin)]
                        : EvaluateLocalChange(LoadPointLocalTerm(data, p, s, at), move);
                    row_change += term;
                    size += fabsf(term);
                }
                // The unit step's fused add matches the contracted sum of the general branch.
                const double energy = double(WarpSum(elastic_change + row_change));
                // The float sum rounds by at most `depth` units of its terms' magnitude, and each term about as
                // much inside; backtracking cannot resolve a change within that (Shi et al. 2022, relaxed Armijo).
                const uint32_t depth = 2u * ((v.element_count + warpSize - 1u) / warpSize) +
                                       (v.row_count + warpSize - 1u) / warpSize + 6u;
                const double noise = double(depth) * FLT_EPSILON * double(WarpSum(size));
                const float change = halving == 0u
                    ? static_cast<float>(fma(nk::vbd::InertialEnergyScale(inertia, p.dt), v.unit_move, energy))
                    : static_cast<float>(energy + nk::vbd::InertialEnergyChange(u, free_rate, move, inertia, p.dt));
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
                    if (!__all_sync(0xffffffffu, base_finite) || 0.5f * scale < smallest) {
                        if (lane == 0u)
                            RecordBlockFailure(data, p, s, particle, nk::BlockSolveFailure::InvalidPotential);
                        break;
                    }
                    scale *= 0.5f;
                    continue;
                }
                if (trial_slope <= 0.0 && double(change) <= double(kVertexStepDecrease) * trial_slope + noise) {
                    next = candidate;
                    accepted = true;
                    break;
                }
                // Backtrack to the minimizer of the quadratic matching this trial's slope and change,
                // safeguarded to between a tenth and a half of the rejected step.
                const double curve = double(change) - trial_slope;
                scale *= curve > 0.0 ? static_cast<float>(fmin(0.5, fmax(0.1, -trial_slope / (2.0 * curve)))) : 0.5f;
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

// A seeded row's endpoints keep their velocities until its lowest color's next cache, which would compute
// the state stored here; grid-only rows read the residuals grid descents keep current instead of recomputing.
__global__ void DualUpdateKernel(DataView data, BlockDescentSolveParams p, BlockScratch s, bool seed) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    uint32_t env = ~0u, joint = 0u, step = 0u;
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.active_rows[item];
        const uint32_t row_env = slot / p.rows_per_env;
        if (!EnvRunning(data, row_env)) continue;
        if (rows[slot].flags & nk::nk_row_flags::kMaterialBlock) {
            UpdateMaterialDual(data, p, s, slot);
            continue;
        }
        const bool grid_only = p.total_grid_count != 0u && GridOnlyRow(s, p.vertex_blocks.colors, slot);
        AugmentedRowState current = grid_only ? s.row_state[slot] : ComputeAugmentedRowState(data, p, s, slot, false);
        if (grid_only)
            current.normal_bound = AugmentedRowBound(s, rows[slot], slot, RowStateScale(data, p, slot), current);
        const LocalTerm t = LoadLocalTerm(data, p, s, slot, ~0u, false, &current);
        const uint32_t stride = rows[slot].group_normal_count;
        const Vec3 next = ProjectRowDual(t.dual, t.normal_bound, t.penalty, t.residual, t.contact, t.mu_first,
                                         t.mu_second, s.contact_relaxation + slot);
        if (row_env != env && env != ~0u) {
            RaiseResidual(s, env, kResidualJoint, joint);
            atomicMax(s.dual_step + env, step);
            joint = step = 0u;
        }
        env = row_env;
        // Every row's multiplier step drives the momentum restart; a non-contact row's is its residual too.
        const float2 change = MultiplierSteps(t.dual, next, static_cast<float>(t.response.residual_scale),
            t.penalty, t.contact ? t.mu_first : 0.0f, t.contact ? t.mu_second : 0.0f);
        if (!t.contact) joint = max(joint, ResidualBits(change.x));
        step = max(step, ResidualBits(fmaxf(change.x, change.y)));
        // The primal reads the extrapolated multipliers; the unextrapolated ones stay in dual_base.
        const Vec3 base{s.dual_base[slot], t.contact ? s.dual_base[slot + stride] : 0.0f,
                        t.contact ? s.dual_base[slot + 2u * stride] : 0.0f};
        const Vec3 y = ExtrapolateDual(base, next, s.dual_momentum[row_env].beta, t.lower, t.upper, t.contact,
                                       t.mu_first, t.mu_second);
        s.dual_base[slot] = next.x;
        data.lambda[slot] = y.x;
        if (t.contact) {
            s.dual_base[slot + stride] = next.y;
            s.dual_base[slot + 2u * stride] = next.z;
            data.lambda[slot + stride] = y.y;
            data.lambda[slot + 2u * stride] = y.z;
        }
        if (seed && (p.total_grid_count == 0u
                ? ParticleRowFirstColor(s, p.vertex_blocks.colors, slot) != kNoCacheColor : grid_only)) {
            AugmentedRowState seeded{};
            seeded.residual = t.residual;
            seeded.dual = y;
            seeded.normal_bound = AugmentedRowBound(s, rows[slot], slot, RowStateScale(data, p, slot), seeded);
            s.row_state[slot] = seeded;
        }
    }
    RaiseWarpResidual(s, env, kResidualJoint, joint);
    RaiseWarpMax(s.dual_step, env, step);
}

// No momentum before the first sweep; the first restart test passes against the largest step.
__global__ void StartDualMomentumKernel(BlockDescentSolveParams p, BlockScratch s) {
    for (uint32_t env = blockIdx.x * blockDim.x + threadIdx.x; env < p.env_count;
         env += gridDim.x * blockDim.x) {
        s.dual_momentum[env] = StartDualMomentum();
        s.dual_step[env] = __float_as_uint(FLT_MAX);
    }
}

// A running env's momentum for this sweep's dual update, from the step its last update took.
__global__ void AdvanceDualMomentumKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    for (uint32_t env = blockIdx.x * blockDim.x + threadIdx.x; env < p.env_count;
         env += gridDim.x * blockDim.x) {
        if (!EnvRunning(data, env)) continue;
        AdvanceDualMomentum(s.dual_momentum[env], __uint_as_float(s.dual_step[env]));
        s.dual_step[env] = 0u;
    }
}

// The solve leaves the unextrapolated multipliers to the impulses, the readouts and the next warm start.
__global__ void RestoreDualBaseKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.active_rows[item];
        const NkRow& row = rows[slot];
        if (row.flags & nk::nk_row_flags::kMaterialBlock) continue;
        data.lambda[slot] = s.dual_base[slot];
        if (!(row.flags & nk::nk_row_flags::kBlockNormal)) continue;
        data.lambda[slot + row.group_normal_count] = s.dual_base[slot + row.group_normal_count];
        data.lambda[slot + 2u * row.group_normal_count] = s.dual_base[slot + 2u * row.group_normal_count];
    }
}

// A warp sums a particle's row impulse: lanes form the incidence terms, every lane adds them in incidence order.
__device__ Vec3 GatherParticleImpulse(DataView data, BlockScratch s, uint32_t particle, uint32_t lane) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const auto points = PointMasses(data);
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
    return impulse;
}

__global__ void GatherImpulseKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    if (data.particle_row_impulse == nullptr) return;
    const uint32_t lane = threadIdx.x % warpSize;
    const size_t warp = (size_t{blockIdx.x} * blockDim.x + threadIdx.x) / warpSize;
    const size_t stride = size_t{gridDim.x} * blockDim.x / warpSize;
    for (size_t item = warp; item < p.total_particle_count; item += stride) {
        const uint32_t particle = static_cast<uint32_t>(item);
        const Vec3 impulse = GatherParticleImpulse(data, s, particle, lane);
        if (lane == 0u) data.particle_row_impulse[particle] = impulse;
    }
}

// Every env's color counts, and the phase times of a timed solve.
__global__ void RecordSolveKernel(ModelView model, DataView data, BlockDescentSolveParams p,
                                  BlockScratch s) {
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < p.env_count;
         index += gridDim.x * blockDim.x) {
        data.solver_color_counts[index * kSolverCountWords] = p.vertex_blocks.colors;
        for (uint32_t phase = 0u; p.telemetry != 0u && phase < kSolverPhaseWords; ++phase)
            data.solver_phase_time[index * kSolverPhaseWords + phase] = s.phase_clock[2u + phase];
    }
}

// Coulomb residual of a contact row at the current velocities and multipliers; `rates` gets the largest velocity
// of the normal and of a tangent axis along its own impulse, which the contact audit bounds by the tolerance.
__device__ constraint::CoulombContactResidual ContactRowResidual(DataView data, BlockDescentSolveParams p,
                                                                 BlockScratch s, uint32_t slot, const NkRow& row,
                                                                 float2* rates = nullptr) {
    const LocalTerm t = LoadLocalTerm(data, p, s, slot, ~0u, false);
    const Vec3 velocity{-t.residual.x + t.compliance * t.dual.x, -t.residual.y, -t.residual.z};
    if (rates != nullptr)
        *rates = {Along(t.dual.x, velocity.x), fmaxf(Along(t.dual.y, velocity.y), Along(t.dual.z, velocity.z))};
    return constraint::EvaluateCoulombContactResidual(
        data.row_contact_response[slot], velocity, t.dual, row.mu, row.friction_secondary);
}

// The contact classes of a measured sweep: each row's natural velocities and velocities along its impulses.
__global__ void ContactResidualsKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    bool prune = false;
    if (!MeasuredSweep(p, s, &prune)) return;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    uint32_t env = ~0u, normal = 0u, tangent = 0u;
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.active_rows[item];
        const NkRow row = rows[slot];
        const uint32_t row_env = slot / p.rows_per_env;
        if (!(row.flags & nk::nk_row_flags::kBlockNormal) || !EnvRunning(data, row_env)) continue;
        if (row_env != env && env != ~0u) {
            RaiseResidual(s, env, kResidualNormal, normal);
            RaiseResidual(s, env, kResidualTangent, tangent);
            normal = tangent = 0u;
        }
        env = row_env;
        if (prune && Past(p, s, env, kResidualNormal, normal) && Past(p, s, env, kResidualTangent, tangent)) continue;
        float2 rates{};
        const auto residual = ContactRowResidual(data, p, s, slot, row, &rates);
        normal = max(normal, ResidualBits(residual.valid ? fmaxf(residual.normal_natural_velocity, rates.x) : FLT_MAX));
        tangent = max(tangent, ResidualBits(residual.valid ? fmaxf(residual.tangent_natural_velocity, rates.y) : FLT_MAX));
    }
    RaiseWarpResidual(s, env, kResidualNormal, normal);
    RaiseWarpResidual(s, env, kResidualTangent, tangent);
}

// The cloth vertex part of a measured sweep's momentum class, after every row update of the sweep.
__global__ void VertexResidualsKernel(ModelView model, DataView data, BlockDescentSolveParams p, BlockScratch s) {
    bool prune = false;
    if (!MeasuredSweep(p, s, &prune)) return;
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t stride = gridDim.x * blockDim.x / warpSize;
    const VertexBlockView b = Vertices(model, data, p);
    for (uint32_t item = first; item < b.layout.dynamic_vertices * p.env_count; item += stride) {
        const uint32_t env = item / b.layout.dynamic_vertices;
        const bool skip = !EnvRunning(data, env) || (prune && Past(p, s, env, kResidualMomentum, 0u));
        if (__shfl_sync(0xffffffffu, skip ? 1u : 0u, 0) != 0u) continue;
        const uint32_t vertex = b.color_vertices[item % b.layout.dynamic_vertices];
        const uint32_t particle = b.Particle(env, vertex);
        const Vec3 impulse = GatherParticleImpulse(data, s, particle, lane);
        if (lane == 0u) data.particle_row_impulse[particle] = impulse;
        __syncwarp();
        const uint32_t bits = ResidualBits(VertexBlockStationaryResidualWarp(b, PointMasses(data), env, vertex, lane));
        if (lane == 0u && bits > s.env_residual[size_t{env} * kResidualClasses + kResidualMomentum])
            RaiseResidual(s, env, kResidualMomentum, bits);
    }
}

__global__ void MeasureRowsKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t slot = s.active_rows[item];
        const NkRow row = rows[slot];
        if (!(row.flags & nk::nk_row_flags::kBlockNormal)) continue;
        const auto residual = ContactRowResidual(data, p, s, slot, row);
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

// Terms of each metric's worst row, read back from the slot its packed record names.
__global__ void MeasureWorstRowsKernel(DataView data, BlockDescentSolveParams p, BlockScratch s) {
    constexpr uint32_t kColumns = constraint::kContactSolveWorstColumns;
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
         item < p.env_count * constraint::kContactSolveMetricCount; item += gridDim.x * blockDim.x) {
        const uint64_t packed = data.contact_solve_metrics[item];
        if (packed == 0u) continue;
        const uint32_t slot = ~static_cast<uint32_t>(packed);
        const LocalTerm t = LoadLocalTerm(data, p, s, slot, ~0u, false);
        const Vec3 velocity{-t.residual.x + t.compliance * t.dual.x, -t.residual.y, -t.residual.z};
        const auto f = nk::augmented::EvaluateTangent(t.dual, t.penalty, {0.0f, t.residual.y, t.residual.z},
                                                      t.normal_bound, t.mu_first, t.mu_second);
        const SymmetricMat3& w = data.row_contact_response[slot];
        const float terms[kColumns] = {t.dual.x, t.dual.y, t.dual.z, velocity.x, velocity.y, velocity.z,
            t.normal_bound, f.impulse.y, f.impulse.z, t.penalty, t.compliance, w.xx, w.yy, w.zz, w.xy, w.xz,
            w.yz, t.mu_first, t.mu_second, __uint_as_float(slot)};
        for (uint32_t column = 0u; column < kColumns; ++column)
            data.contact_solve_worst_terms[size_t{item} * kColumns + column] = terms[column];
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

// Closes the open phase of a timed solve at the global timer and opens `phase`; the first stamp only opens.
__global__ void StampPhaseKernel(BlockScratch s, uint32_t phase, bool first) {
    const unsigned long long now = GlobalTimer();
    if (!first) s.phase_clock[2u + s.phase_clock[1]] += now - s.phase_clock[0];
    s.phase_clock[0] = now;
    s.phase_clock[1] = phase;
}

__device__ nk::TelemetryHeader* TelemetryHeaderOf(uint32_t* ring) {
    return reinterpret_cast<nk::TelemetryHeader*>(ring);
}

// The record of the running substep; the solve and the substep's RecordTelemetry fill it.
__device__ nk::TelemetryRecord* OpenTelemetryRecord(uint32_t* ring) {
    const nk::TelemetryHeader* header = TelemetryHeaderOf(ring);
    return reinterpret_cast<nk::TelemetryRecord*>(ring + nk::kTelemetryHeaderWords) +
           header->head % header->capacity;
}

__device__ float PackedValue(uint64_t packed) { return __uint_as_float(static_cast<uint32_t>(packed >> 32u)); }
__device__ uint32_t PackedIndex(uint64_t packed) { return ~static_cast<uint32_t>(packed); }
__device__ uint64_t Larger(uint64_t a, uint64_t b) { return a > b ? a : b; }

// Sample slot of the 1-based sweep n among 1, 2, 3, 4, 6, 8, 12, 16, ...; kTelemetrySamples when unsampled.
__device__ uint32_t SweepSampleSlot(uint32_t n) {
    const uint32_t k = 31u - static_cast<uint32_t>(__clz(n));
    if (n == 1u << k) return k == 0u ? 0u : 2u * k - 1u;
    if (k > 0u && n == 3u << (k - 1u)) return 2u * k;
    return nk::kTelemetrySamples;
}

__global__ void OpenTelemetryKernel(DataView data) {
    auto* words = reinterpret_cast<uint32_t*>(OpenTelemetryRecord(data.telemetry));
    for (uint32_t word = threadIdx.x; word < nk::kTelemetryRecordWords; word += blockDim.x) words[word] = 0u;
    if (threadIdx.x == 0u) TelemetryHeaderOf(data.telemetry)->penetration = 0u;
}

// Ends the sweep of each running env whose residuals pass the tolerance or whose budget is spent, keeping its
// sweeps and residuals; a while node repeats the sweep as long as an env runs.
__global__ void FinishSweepKernel(DataView data, BlockDescentSolveParams p, BlockScratch s,
                                  cudaGraphConditionalHandle loop, bool graph) {
    __shared__ uint32_t running[kThreads];
    __shared__ uint32_t largest[kThreads];
    const uint32_t done = s.sweep_state[kSweepCount] + 1u;
    const bool trace = (p.telemetry & nk::kTelemetrySweeps) != 0u && done <= p.trace_sweeps;
    uint32_t count = 0u, worst = 0u;
    for (uint32_t env = threadIdx.x; env < p.env_count; env += blockDim.x) {
        if (!EnvRunning(data, env)) continue;
        uint32_t* words = s.env_residual + size_t{env} * kResidualClasses;
        uint32_t bits = 0u;
        for (uint32_t kind = 0u; kind < kResidualClasses; ++kind) {
            bits = max(bits, words[kind]);
            if (trace)
                data.solver_sweep_trace[(size_t{env} * p.trace_sweeps + done - 1u) * kResidualClasses + kind] =
                    __uint_as_float(words[kind]);
        }
        if (__uint_as_float(bits) < p.velocity_tolerance || done >= p.iterations) {
            data.vbd_velocity_sweep_count[env] = done;
            for (uint32_t kind = 0u; kind < kResidualClasses; ++kind)
                data.solver_residual[env * kResidualClasses + kind] = __uint_as_float(words[kind]);
        } else {
            ++count;
        }
        worst = max(worst, bits);
        for (uint32_t kind = 0u; kind < kResidualClasses; ++kind) words[kind] = 0u;
    }
    running[threadIdx.x] = count;
    largest[threadIdx.x] = worst;
    __syncthreads();
    for (uint32_t half = blockDim.x / 2u; half > 0u; half /= 2u) {
        if (threadIdx.x < half) {
            running[threadIdx.x] += running[threadIdx.x + half];
            largest[threadIdx.x] = max(largest[threadIdx.x], largest[threadIdx.x + half]);
        }
        __syncthreads();
    }
    if (threadIdx.x != 0u) return;
    const uint32_t again = running[0] > 0u ? 1u : 0u;
    s.sweep_state[kSweepCount] = done;
    s.sweep_state[kSweepContinue] = again;
    if (graph) cudaGraphSetConditional(loop, again);
    const uint32_t slot = SweepSampleSlot(done);
    if ((p.telemetry & nk::kTelemetrySweeps) == 0u || slot >= nk::kTelemetrySamples) return;
    nk::TelemetryRecord* record = OpenTelemetryRecord(data.telemetry);
    record->sample_sweep[slot] = done;
    record->sample_residual[slot] = __uint_as_float(largest[0]);
    record->sample_count = max(record->sample_count, slot + 1u);
}

// The solve's part of the substep record: sweeps, the largest multiplier momentum and the deepest active contact row.
__global__ void TelemetrySolveKernel(DataView data, BlockDescentSolveParams p, BlockScratch s, uint32_t momentum) {
    auto* deepest = reinterpret_cast<unsigned long long*>(&TelemetryHeaderOf(data.telemetry)->penetration);
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    for (uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < *s.active_count;
         item += gridDim.x * blockDim.x) {
        const uint32_t row_slot = s.active_rows[item];
        if (!(rows[row_slot].flags & nk::nk_row_flags::kBlockNormal)) continue;
        const float depth = data.row_penetration[row_slot];
        if (depth > 0.0f)
            atomicMax(deepest, (static_cast<unsigned long long>(__float_as_uint(depth)) << 32u) | ~row_slot);
    }
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    nk::TelemetryRecord* record = OpenTelemetryRecord(data.telemetry);
    record->sweeps = 0u;
    bool converged = true;
    for (uint32_t env = 0u; env < p.env_count; ++env) {
        record->sweeps = max(record->sweeps, data.vbd_velocity_sweep_count[env]);
        for (uint32_t kind = 0u; kind < kResidualClasses; ++kind)
            converged &= data.solver_residual[env * kResidualClasses + kind] < p.velocity_tolerance;
    }
    record->sweep_budget = p.iterations;
    if (converged) record->flags |= nk::kTelemetryConverged;
    float beta = 0.0f;
    for (uint32_t env = 0u; momentum != 0u && env < p.env_count; ++env)
        beta = fmaxf(beta, s.dual_momentum[env].beta);
    record->acceleration = beta;
}

// Completes the substep record from the measured solve and the energy ledger, then commits it.
__global__ void RecordTelemetryKernel(DataView data, RecordTelemetryParams p) {
    using nk::EnergyColumn;
    __shared__ uint64_t maxima[4][kThreads];
    __shared__ float sums[2][kThreads];
    __shared__ uint32_t counts[2][kThreads];
    uint64_t local[4] = {0u, 0u, 0u, 0u};
    float energy = 0.0f, residual = 0.0f;
    uint32_t contacts = 0u, invalid = 0u;
    const auto column = [](EnergyColumn c) { return static_cast<uint32_t>(c); };
    for (uint32_t env = threadIdx.x; env < p.env_count; env += blockDim.x) {
        const uint64_t* metrics = data.contact_solve_metrics + size_t{env} * constraint::kContactSolveMetricCount;
        local[0] = Larger(local[0], data.vbd_solve_metrics[env * 2u]);
        local[1] = Larger(local[1], data.vbd_solve_metrics[env * 2u + 1u]);
        local[2] = Larger(local[2], metrics[0]);
        local[3] = Larger(local[3], metrics[1]);
        contacts += data.contact_solve_counts[env * constraint::kContactSolveCountSize];
        if (p.energy == 0u) continue;
        const float* ledger = data.energy_ledger + (size_t{env} * p.substeps + p.substep) * nk::kEnergyColumnCount;
        energy += ledger[column(EnergyColumn::EndKinetic)] + ledger[column(EnergyColumn::EndGravity)] +
                  ledger[column(EnergyColumn::EndElastic)];
        residual += ledger[column(EnergyColumn::Residual)];
        invalid += ledger[column(EnergyColumn::Valid)] == 1.0f ? 0u : 1u;
    }
    for (uint32_t k = 0u; k < 4u; ++k) maxima[k][threadIdx.x] = local[k];
    sums[0][threadIdx.x] = energy;
    sums[1][threadIdx.x] = residual;
    counts[0][threadIdx.x] = contacts;
    counts[1][threadIdx.x] = invalid;
    __syncthreads();
    for (uint32_t half = blockDim.x / 2u; half > 0u; half /= 2u) {
        if (threadIdx.x < half) {
            const uint32_t other = threadIdx.x + half;
            for (uint32_t k = 0u; k < 4u; ++k) maxima[k][threadIdx.x] = Larger(maxima[k][threadIdx.x], maxima[k][other]);
            for (uint32_t k = 0u; k < 2u; ++k) {
                sums[k][threadIdx.x] += sums[k][other];
                counts[k][threadIdx.x] += counts[k][other];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x != 0u) return;
    nk::TelemetryHeader* header = TelemetryHeaderOf(data.telemetry);
    nk::TelemetryRecord* record = OpenTelemetryRecord(data.telemetry);
    const uint64_t vertex = maxima[0][0], contact = Larger(maxima[2][0], maxima[3][0]);
    record->sequence = header->head;
    record->substep = p.substep;
    record->contacts = counts[0][0];
    record->vertex_residual = vertex != 0u ? PackedValue(vertex) : 0.0f;
    record->vertex_force = maxima[1][0] != 0u ? PackedValue(maxima[1][0]) : 0.0f;
    record->contact_residual = contact != 0u ? PackedValue(contact) : 0.0f;
    record->max_penetration = header->penetration != 0u ? PackedValue(header->penetration) : 0.0f;
    record->penetration_row = header->penetration != 0u ? PackedIndex(header->penetration) : ~0u;
    record->energy = sums[0][0];
    record->energy_residual = sums[1][0];
    if (p.energy != 0u && counts[1][0] == 0u) record->flags |= nk::kTelemetryEnergyValid;
    record->worst_kind = nk::kTelemetryElementNone;
    record->worst_index = ~0u;
    for (uint32_t side = 0u; side < 2u; ++side) {
        record->worst_side_kind[side] = ~0u;
        record->worst_side_index[side] = ~0u;
    }
    if (vertex != 0u && vertex >= contact) {
        record->worst_kind = nk::kTelemetryElementParticle;
        record->worst_index = PackedIndex(vertex);
        record->worst_residual = PackedValue(vertex);
    } else if (contact != 0u) {
        const NkRow row = reinterpret_cast<const NkRow*>(data.urows)[PackedIndex(contact)];
        record->worst_kind = nk::kTelemetryElementRow;
        record->worst_index = PackedIndex(contact);
        record->worst_residual = PackedValue(contact);
        record->worst_side_kind[0] = row.a.kind;
        record->worst_side_index[0] = row.a.index;
        record->worst_side_kind[1] = row.b.kind;
        record->worst_side_index[1] = row.b.index;
    }
    const float alert = p.velocity_tolerance > 0.0f ? p.velocity_tolerance : header->alert_velocity;
    if (record->sweeps >= record->sweep_budget && record->worst_residual > alert)
        record->flags |= nk::kTelemetryCap;
    __threadfence();
    header->head += 1u;
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

// The multilevel correction of the vertex blocks builds its operators and dense factors when asked.
cudaError_t LaunchMultilevelCorrection(const ModelView& model, const DataView& data,
                                       const BlockDescentSolveParams& p, const BlockScratch& s,
                                       bool assemble, cudaStream_t stream) {
    const VertexBlockLayout& l = p.vertex_blocks;
    const uint32_t vertices = l.vertices * p.env_count;
    const uint32_t jacobi = (l.coarse_nodes - l.coarse_dense_nodes) * p.env_count;
    const uint32_t rank = 3u * l.coarse_dense_nodes, packed = rank * (rank + 1u) / 2u;
    const dim3 parts(p.env_count * s.multilevel_parts);
    cudaError_t status = LaunchWork(MultilevelRowsKernel, s.rows, kThreads / kRowAxisLanes, stream, data, p, s);
    if (status == cudaSuccess)
        status = LaunchWork(MultilevelElementGradientKernel,
            static_cast<uint32_t>(uint64_t{nk::PackVbdIncidence(1u, 0u)} * l.elements * p.env_count),
            kThreads, stream, model, data, p, s);
    if (status == cudaSuccess)
        status = LaunchWork(MultilevelForceKernel, vertices, kMultilevelForceTile, stream, model, data, p, s);
    if (status == cudaSuccess && assemble) {
        status = cudaMemsetAsync(s.multilevel_operator, 0, size_t{p.env_count} *
                                 MultilevelOperatorEntries(l) * sizeof(unsigned long long), stream);
        if (status == cudaSuccess) {
            LaunchCuda(MultilevelBoundKernel, parts, dim3(kThreads), 0u, stream, model, data, p, s);
            status = cudaGetLastError();
        }
        if (status == cudaSuccess)
            status = LaunchWork(MultilevelScaleKernel, p.env_count, kThreads, stream, data, p, s);
        if (status == cudaSuccess) {
            const uint32_t row_parts = s.multilevel_parts - s.multilevel_item_parts;
            LaunchCuda(MultilevelAssembleKernel,
                       dim3(p.env_count * (row_parts * l.coarse_levels + s.multilevel_item_parts)),
                       dim3(kThreads), 0u, stream, model, data, p, s);
            status = cudaGetLastError();
        }
        if (status == cudaSuccess)
            status = LaunchWork(MultilevelJacobiFactorKernel, jacobi + rank * p.env_count, kThreads, stream, p, s);
        if (status == cudaSuccess && rank > 0u)
            status = LaunchWork(MultilevelDenseScaleKernel, packed * p.env_count, kThreads, stream, p, s);
        const size_t factor_shared = MultilevelFactorSharedBytes(rank);
        uint32_t factor_grid = 0u;
        if (status == cudaSuccess && rank > 0u)
            status = ResidentGridSize(MultilevelDenseFactorKernel, 32u * kMultilevelFactorWarps, factor_shared,
                                      p.env_count, &factor_grid);
        if (status == cudaSuccess && rank > 0u) {
            LaunchCuda(MultilevelDenseFactorKernel, dim3(factor_grid), dim3(32u * kMultilevelFactorWarps),
                       factor_shared, stream, p, s);
            status = cudaGetLastError();
        }
        const uint32_t groups = (rank + kMultilevelInverseWarps - 1u) / kMultilevelInverseWarps;
        const size_t inverse_shared =
            (size_t{packed} + size_t{rank}) * sizeof(float);
        uint32_t inverse_grid = 0u;
        if (status == cudaSuccess && rank > 0u)
            status = ResidentGridSize(MultilevelDenseInverseKernel, 32u * kMultilevelInverseWarps,
                                      inverse_shared, p.env_count * groups, &inverse_grid);
        if (status == cudaSuccess && rank > 0u) {
            LaunchCuda(MultilevelDenseInverseKernel, dim3(inverse_grid), dim3(32u * kMultilevelInverseWarps),
                       inverse_shared, stream, p, s);
            status = cudaGetLastError();
        }
    }
    if (status == cudaSuccess) {
        const uint32_t correct_parts = MultilevelCorrectParts(l);
        LaunchCuda(MultilevelCorrectKernel, dim3(p.env_count * correct_parts), dim3(kMultilevelCorrectThreads),
                   0u, stream, model, p, s, correct_parts);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess && rank > 0u) {
        const uint32_t tiles = (rank + 31u) / 32u;
        LaunchCuda(MultilevelDenseCorrectKernel, dim3(p.env_count * tiles), dim3(kMultilevelCorrectThreads),
                   0u, stream, p, s, tiles);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess)
        status = LaunchWork(MultilevelProlongKernel, vertices, kThreads, stream, model, data, p, s);
    if (status != cudaSuccess) return status;
    LaunchCuda(MultilevelFormsKernel, parts, dim3(kThreads), 0u, stream, model, data, p, s);
    for (uint32_t trial = 0u; trial <= kMultilevelBacktracks; ++trial)
        LaunchCuda(MultilevelTrialKernel, parts, dim3(kThreads), 0u, stream, model, data, p, s,
                   trial == kMultilevelBacktracks);
    status = cudaGetLastError();
    return status == cudaSuccess
        ? LaunchWork(MultilevelApplyKernel, vertices, kThreads, stream, data, p, s) : status;
}

// Contact and vertex residuals of the finished solve, when asked.
Status MeasureSolve(const ModelView& model, const DataView& data, const BlockDescentSolveParams& p,
                    const BlockScratch& s, cudaStream_t stream) {
    if (p.measure_contact_residual == 0u) return Status::Ok;
    const auto clear = [&](void* ptr, size_t bytes) {
        return bytes == 0u || (ptr != nullptr && cudaMemsetAsync(ptr, 0, bytes, stream) == cudaSuccess);
    };
    if (!clear(data.contact_solve_metrics, size_t{p.env_count} * constraint::kContactSolveMetricCount * sizeof(uint64_t)) ||
        !clear(data.contact_solve_counts, size_t{p.env_count} * constraint::kContactSolveCountSize * sizeof(uint32_t)) ||
        !clear(data.vbd_solve_metrics, size_t{p.env_count} * 2u * sizeof(uint64_t)) ||
        !clear(data.contact_solve_worst_terms, size_t{p.env_count} * constraint::kContactSolveMetricCount *
            constraint::kContactSolveWorstColumns * sizeof(float))) return Status::Failed;
    if (LaunchWork(MeasureRowsKernel, s.rows, kThreads, stream, data, p, s) != cudaSuccess ||
        LaunchWork(MeasureWorstRowsKernel, p.env_count * constraint::kContactSolveMetricCount, kThreads,
            stream, data, p, s) != cudaSuccess ||
        LaunchWork(MeasureVerticesKernel, p.vertex_blocks.dynamic_vertices * p.env_count,
            kThreads / 32u, stream, model, data, p) != cudaSuccess) return Status::Failed;
    return Status::Ok;
}

// The measured solve, then its part of the telemetry record when telemetry is on.
Status MeasureAndRecord(const ModelView& model, const DataView& data, const BlockDescentSolveParams& p,
                        const BlockScratch& s, bool momentum, cudaStream_t stream) {
    if (MeasureSolve(model, data, p, s, stream) != Status::Ok) return Status::Failed;
    if (p.telemetry == 0u) return Status::Ok;
    return LaunchWork(TelemetrySolveKernel, std::max(s.rows, 1u), kThreads, stream, data, p, s,
                      uint32_t{momentum}) == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpRecordTelemetry(const ModelView&, const DataView& data, const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const RecordTelemetryParams*>(params);
    if (p == nullptr || data.telemetry == nullptr) return Status::InvalidArgument;
    LaunchCuda(RecordTelemetryKernel, dim3(1u), dim3(kThreads), 0u, stream, data, *p);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
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
    const VertexBlockLayout& layout = p->vertex_blocks;
    const bool multilevel = layout.dynamic_vertices > 0u && layout.coarse_levels > 0u;
    if (multilevel && (layout.coarse_dense_nodes > nk::kVbdCoarseDenseNodes ||
        layout.coarse_dense_nodes > layout.coarse_nodes ||
        layout.particles_per_env != p->total_particle_count / p->env_count ||
        model.vbd_coarse_parents == nullptr || model.vbd_coarse_weights == nullptr ||
        model.vbd_coarse_child_offsets == nullptr || model.vbd_coarse_children == nullptr ||
        model.vbd_coarse_child_weights == nullptr || data.vbd_inertia == nullptr ||
        data.vbd_free_rate == nullptr))
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
        data.link_velocity == nullptr || data.m == nullptr || data.m_factor == nullptr ||
        data.chain_jacobian == nullptr || data.chain_jacobian_b == nullptr ||
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
    if (p->telemetry != 0u) {
        if (data.telemetry == nullptr) return Status::InvalidArgument;
        LaunchCuda(OpenTelemetryKernel, dim3(1u), dim3(kThreads), 0u, stream, data);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
    }
    if (DenseTileRoute(*p)) {
        if (p->measure_contact_residual != 0u && !clear(s.active_count, sizeof(uint32_t))) return Status::Failed;
        if (!clear(s.tile_next, sizeof(uint32_t))) return Status::Failed;
        LaunchCuda(OrderTilesKernel, dim3(1u), dim3(kOrderTileThreads), 0u, stream, *p, s);
        const cudaError_t launched = VisitTileWidth(*p, [&](auto width) {
            return LaunchDenseTiles<decltype(width)::value>(articulation_state, data, *p, s, stream);
        });
        if (launched != cudaSuccess) return Status::Failed;
        return MeasureAndRecord(model, data, *p, s, false, stream);
    }
    if (p->measure_vertex_audit != 0u && !clear(data.vbd_solve_audit,
            size_t{p->vertex_blocks.vertices} * p->env_count * nk::kVbdSolveAuditColumnCount * sizeof(float)))
        return Status::Failed;
    if (!clear(s.counts, size_t{s.blocks + 1u} * sizeof(uint32_t)) ||
        !clear(s.cache_counts, size_t{s.cache_color_count} * sizeof(uint32_t)) ||
        !clear(s.contact_relaxation, size_t{s.rows} * sizeof(nk::augmented::ContactRelaxation)) ||
        !clear(s.active_count, sizeof(uint32_t)) ||
        !clear(s.sweep_state, kSweepStateWords * sizeof(uint32_t)) ||
        !clear(data.vbd_velocity_sweep_count, size_t{p->env_count} * sizeof(uint32_t)) ||
        !clear(s.env_residual, size_t{p->env_count} * kResidualClasses * sizeof(uint32_t)) ||
        !clear(data.solver_color_counts, size_t{p->env_count} * kSolverCountWords * sizeof(uint32_t)) ||
        !clear(data.solver_phase_time, size_t{p->env_count} * kSolverPhaseWords * sizeof(uint64_t))) return Status::Failed;
    // A timed solve stamps its phase boundaries; the stamps go into the captured sweeps too.
    const auto stamp = [&](uint32_t phase, cudaStream_t stream) {
        if (p->telemetry == 0u) return true;
        LaunchCuda(StampPhaseKernel, dim3(1u), dim3(1u), 0u, stream, s, phase, false);
        return cudaGetLastError() == cudaSuccess;
    };
    if (p->telemetry != 0u) {
        if (!clear(s.phase_clock, (2u + kSolverPhaseWords) * sizeof(unsigned long long))) return Status::Failed;
        LaunchCuda(StampPhaseKernel, dim3(1u), dim3(1u), 0u, stream, s, kPhaseSetup, true);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
    }
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
    // The multipliers start unextrapolated, without momentum; residuals are measured at the unextrapolated ones.
    if ((s.rows > 0u && cudaMemcpyAsync(s.dual_base, data.lambda, size_t{s.rows} * sizeof(float),
            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) ||
        LaunchWork(StartDualMomentumKernel, p->env_count, kThreads, stream, *p, s) != cudaSuccess)
        return Status::Failed;
    DataView measured = data;
    measured.lambda = s.dual_base;
    if (LaunchWork(InitializeOwnerCacheColorsKernel, s.blocks, kThreads, stream,
            data, *p, s) != cudaSuccess ||
        LaunchWork(InitializeVertexCacheColorsKernel, static_cast<uint32_t>(vertex_color_items),
            kThreads, stream, model, data, *p, s) != cudaSuccess ||
        LaunchWork(CountIncidenceKernel, s.rows, kThreads, stream, data, *p, s) != cudaSuccess)
        return Status::Failed;
    if (cub::DeviceScan::ExclusiveSum(s.scan, s.scan_bytes, s.counts, s.offsets,
            static_cast<int>(s.blocks + 1u), stream) != cudaSuccess ||
        !clear(s.counts, size_t{s.blocks + 1u} * sizeof(uint32_t))) return Status::Failed;
    // Grid colors list only the nodes with rows, so they follow the incidence offsets.
    if (p->total_grid_count > 0u &&
        (LaunchWork(InitializeGridColorScheduleKernel, nk::kMpmCellStencilNodes, kThreads, stream,
             data, *p, s) != cudaSuccess ||
         LaunchWork(InitializeGridBlocksKernel, p->total_grid_count, kThreads, stream,
             data, *p, s) != cudaSuccess)) return Status::Failed;
    if (LaunchWork(FillIncidenceKernel, s.rows, kThreads, stream, data, *p, s) != cudaSuccess)
        return Status::Failed;
    if (s.incidence_sort_bytes > 0u) {
        const LargeIncidenceBound bound{s.offsets, s.large_segments, s.large_count, s.incidence_capacity, 0u};
        LargeIncidenceBound end_bound = bound;
        end_bound.end = 1u;
        if (!clear(s.large_count, sizeof(uint32_t)) ||
            LaunchWork(SortSmallIncidenceKernel, s.blocks, kThreads / 32u, stream, s) != cudaSuccess ||
            cub::DeviceSegmentedRadixSort::SortKeys(s.incidence_sort, s.incidence_sort_bytes,
                s.incidence, s.sorted_incidence, static_cast<int64_t>(s.incidence_capacity),
                static_cast<int64_t>(s.large_capacity),
                LargeIncidenceBounds(thrust::counting_iterator<uint32_t>(0u), bound),
                LargeIncidenceBounds(thrust::counting_iterator<uint32_t>(0u), end_bound),
                0, RowSlotBits(s.rows), stream) != cudaSuccess)
            return Status::Failed;
    }
    if (LaunchWork(PrepareCacheBucketsKernel, s.incidence_capacity, kThreads, stream, s) != cudaSuccess ||
        LaunchWork(FillCacheBucketsKernel, s.rows, kThreads, stream, s) != cudaSuccess)
        return Status::Failed;
    if (LaunchWork(CachePointIncidenceJacobiansKernel,
            p->total_particle_count + p->total_grid_count, kThreads / 32u, stream,
            data, *p, s) != cudaSuccess) return Status::Failed;
    if ((p->total_body_count > 0u || p->articulation_count > 0u) &&
        LaunchWork(CacheDenseIncidenceBoundsKernel, s.incidence_capacity, kThreads, stream, data, *p, s) != cudaSuccess)
        return Status::Failed;
    if (p->total_particle_count > 0u &&
        (!clear(s.coarse_row_count, sizeof(uint32_t)) ||
         (s.rows > 0u && cub::DeviceSelect::Flagged(s.coarse_select, s.coarse_select_bytes,
             thrust::counting_iterator<uint32_t>(0u), s.coarse_flags, s.coarse_rows,
             s.coarse_row_count, static_cast<int>(s.rows), stream) != cudaSuccess) ||
         LaunchWork(CoarseRowRangesKernel, p->env_count + 1u, kThreads, stream, *p, s) != cudaSuccess))
        return Status::Failed;
    if (multilevel &&
        (LaunchWork(MultilevelTouchKernel, s.rows, kThreads, stream, data, *p, s) != cudaSuccess ||
         !clear(s.multilevel, size_t{p->env_count} * sizeof(MultilevelState)) ||
         !clear(s.multilevel_previous, size_t{layout.vertices} * p->env_count * sizeof(Vec3))))
        return Status::Failed;
    // Grid descents can move particles that carry no cache color, so particle rows are seeded only
    // without grids; with grids, rows that only grid nodes move are seeded instead.
    const bool seed_rows = p->total_grid_count == 0u;
    // One sweep over every block; the first caches every row and builds the multilevel operators.
    const auto sweep = [&](bool first, cudaStream_t stream) {
        if (!stamp(kPhaseVertices, stream)) return Status::Failed;
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
            if (!elastic && p->total_particle_count / p->env_count ==
                    p->grid_particles_per_env + p->vertex_blocks.vertices)
                continue;
            if (LaunchWork(CacheAugmentedRowsKernel, s.rows, kThreads / kRowAxisLanes, stream,
                    data, *p, particle_s, color, spread_rows, elastic && seed_rows && !first)
                != cudaSuccess) return Status::Failed;
            const auto descent_status = elastic
                ? LaunchWork(DescendParticlesKernel<true>, work, kDescendTileVertices, stream,
                             model, data, *p, particle_s, color)
                : LaunchWork(DescendParticlesKernel<false>, work, kDescendTileVertices, stream,
                             model, data, *p, particle_s, color);
            if (descent_status != cudaSuccess) return Status::Failed;
        }
        if (p->articulation_count > 0u || p->total_body_count > 0u) {
            if (!stamp(kPhaseDense, stream)) return Status::Failed;
            if (particle_bytes > 0u && cudaMemcpyAsync(s.particle_snapshot, data.particle_vel,
                    particle_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return Status::Failed;
            if (LaunchWork(CacheAugmentedRowsKernel, s.rows, kThreads / kRowAxisLanes, stream,
                    data, *p, s, p->vertex_blocks.colors + 1u, spread_rows, false) != cudaSuccess)
                return Status::Failed;
            if (p->articulation_count > 0u) {
                const auto kernel = p->max_independent_dof <= 32u ? DescendArticulationsKernel<32u>
                    : DescendArticulationsKernel<kMaxArticulationDof>;
                LaunchCuda(kernel, dim3(p->articulation_count), dim3(kThreads),
                           0u, stream, articulation_state, data, *p, s);
                if (cudaGetLastError() != cudaSuccess) return Status::Failed;
            }
            if (p->total_body_count > 0u) {
                LaunchCuda(DescendRigidBlocksKernel, dim3(p->total_body_count), dim3(kRigidThreads),
                           0u, stream, data, *p, s);
                if (cudaGetLastError() != cudaSuccess) return Status::Failed;
            }
        }
        if (p->total_grid_count > 0u) {
            if (!stamp(kPhaseGrid, stream)) return Status::Failed;
            if (articulation_bytes > 0u && cudaMemcpyAsync(s.articulation_snapshot, data.qdot_flat,
                    articulation_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return Status::Failed;
            if (LaunchWork(PackBlockRigidSnapshotKernel, p->total_body_count, kThreads, stream,
                    data, *p, s) != cudaSuccess) return Status::Failed;
            BlockScratch grid_s = s;
            grid_s.grid_snapshot = data.grid_velocity;
            if (LaunchWork(CacheGridRowsKernel, s.rows, kThreads, stream, data, *p, grid_s, !first)
                != cudaSuccess) return Status::Failed;
            for (uint32_t color = 0u; color < GridColorCount(*p); ++color) {
                const uint32_t cache_color = p->vertex_blocks.colors + 2u + color;
                if (p->material_cells_per_env > 0u &&
                    LaunchWork(CacheMaterialRowsKernel, s.rows, kThreads / 32u, stream,
                        data, *p, grid_s, true, cache_color) != cudaSuccess) return Status::Failed;
                if (LaunchWork(DescendGridBlocksKernel, GridColorCapacity(*p, color), kThreads / 32u,
                        stream, data, *p, grid_s, color) != cudaSuccess) return Status::Failed;
            }
        }
        if (!stamp(kPhaseCoarse, stream)) return Status::Failed;
        if (multilevel &&
            LaunchMultilevelCorrection(model, data, *p, s, first, stream) != cudaSuccess)
            return Status::Failed;
        if (p->total_particle_count / p->env_count > p->grid_particles_per_env + layout.vertices) {
            const dim3 coarse_grid(p->env_count * s.coarse_parts);
            LaunchCuda(CoarseGradientKernel, coarse_grid, dim3(kThreads * kCoarseHelpers), 0u, stream,
                       data, *p, s);
            LaunchCuda(CoarseDirectionKernel, dim3(p->env_count), dim3(kThreads), 0u, stream,
                       data, *p, s);
            LaunchCuda(CoarseUnitTrialKernel, coarse_grid, dim3(kThreads * kCoarseHelpers), 0u, stream,
                       data, *p, s);
            LaunchCuda(CoarseAcceptKernel, dim3(p->env_count), dim3(kThreads), 0u, stream,
                       *p, s, false);
            LaunchCuda(CoarseTrialsKernel, coarse_grid, dim3(kThreads), 0u, stream,
                       data, *p, s);
            LaunchCuda(CoarseAcceptKernel, dim3(p->env_count), dim3(kThreads), 0u, stream,
                       *p, s, true);
            if (cudaGetLastError() != cudaSuccess ||
                LaunchWork(CoarseApplyKernel, p->total_particle_count, kThreads, stream,
                    data, *p, s) != cudaSuccess) return Status::Failed;
        }
        if (!stamp(kPhaseMomentum, stream) ||
            LaunchWork(AdvanceDualMomentumKernel, p->env_count, kThreads, stream, data, *p, s) != cudaSuccess)
            return Status::Failed;
        if (!stamp(kPhaseDual, stream)) return Status::Failed;
        if (p->material_cells_per_env > 0u &&
            LaunchWork(CacheMaterialRowsKernel, s.rows, kThreads / 32u, stream,
                data, *p, s, false, kAllCacheColors) != cudaSuccess) return Status::Failed;
        // The next solve's first sweep recaches every row, so the last sweep's seeds go unread.
        if (LaunchWork(DualUpdateKernel, s.rows, kThreads, stream, data, *p, s, true) != cudaSuccess)
            return Status::Failed;
        return Status::Ok;
    };
    const auto finish = [&](cudaStream_t stream, cudaGraphConditionalHandle loop, bool graph) {
        if (!stamp(kPhaseCheck, stream)) return Status::Failed;
        // The contact and cloth vertex residuals need the measured contact response.
        if (p->measure_contact_residual != 0u &&
            (LaunchWork(ContactResidualsKernel, s.rows, kThreads, stream, measured, *p, s) != cudaSuccess ||
             LaunchWork(VertexResidualsKernel, p->vertex_blocks.dynamic_vertices * p->env_count, kThreads / 32u,
                 stream, model, measured, *p, s) != cudaSuccess))
            return Status::Failed;
        LaunchCuda(FinishSweepKernel, dim3(1u), dim3(kThreads), 0u, stream, data, *p, s, loop, graph);
        return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
    };
    if (sweep(true, stream) != Status::Ok) return Status::Failed;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) return Status::Failed;
    if (capture == cudaStreamCaptureStatusActive) {
        // A captured solve loops on the device: a while node repeats the sweep that its last kernel allows.
        cudaGraph_t graph = nullptr;
        const cudaGraphNode_t* dependencies = nullptr;
        const cudaGraphEdgeData* edges = nullptr;
        size_t dependency_count = 0u;
        cudaGraphConditionalHandle loop{};
        if (cudaStreamGetCaptureInfo(stream, &capture, nullptr, &graph) != cudaSuccess ||
            cudaGraphConditionalHandleCreate(&loop, graph, 0u, 0u) != cudaSuccess ||
            finish(stream, loop, true) != Status::Ok ||
            cudaStreamGetCaptureInfo(stream, &capture, nullptr, &graph, &dependencies, &edges,
                &dependency_count) != cudaSuccess) return Status::Failed;
        cudaGraphNodeParams node_params{};
        node_params.type = cudaGraphNodeTypeConditional;
        node_params.conditional.handle = loop;
        node_params.conditional.type = cudaGraphCondTypeWhile;
        node_params.conditional.size = 1u;
        cudaGraphNode_t node = nullptr;
        cudaStream_t body = nullptr;
        if (cudaGraphAddNode(&node, graph, dependencies, edges, dependency_count, &node_params) != cudaSuccess ||
            cudaStreamUpdateCaptureDependencies(stream, &node, nullptr, 1u,
                cudaStreamSetCaptureDependencies) != cudaSuccess ||
            cudaStreamCreateWithFlags(&body, cudaStreamNonBlocking) != cudaSuccess) return Status::Failed;
        cudaGraph_t captured = nullptr;
        const bool recorded = cudaStreamBeginCaptureToGraph(body, node_params.conditional.phGraph_out[0],
                nullptr, nullptr, 0u, cudaStreamCaptureModeThreadLocal) == cudaSuccess &&
            sweep(false, body) == Status::Ok && finish(body, loop, true) == Status::Ok;
        const bool ended = cudaStreamEndCapture(body, &captured) == cudaSuccess;
        if (cudaStreamDestroy(body) != cudaSuccess || !recorded || !ended) return Status::Failed;
    } else {
        if (finish(stream, {}, false) != Status::Ok) return Status::Failed;
        // Without a tolerance every env runs the whole budget, so the loop needs no readback.
        const bool check = p->velocity_tolerance > 0.0f;
        for (uint32_t done = 1u; done < p->iterations; ++done) {
            uint32_t again = 1u;
            if (check && (cudaMemcpyAsync(&again, s.sweep_state + kSweepContinue, sizeof(again),
                    cudaMemcpyDeviceToHost, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess))
                return Status::Failed;
            if (again == 0u) break;
            if (sweep(false, stream) != Status::Ok || finish(stream, {}, false) != Status::Ok)
                return Status::Failed;
        }
    }
    if (!stamp(kPhaseEnd, stream)) return Status::Failed;
    if (LaunchWork(RestoreDualBaseKernel, s.rows, kThreads, stream, data, *p, s) != cudaSuccess ||
        LaunchWork(GatherImpulseKernel, p->total_particle_count, kThreads / 32u, stream, data, *p, s) != cudaSuccess ||
        LaunchWork(RecordSolveKernel, p->env_count, kThreads, stream,
                   model, data, *p, s) != cudaSuccess) return Status::Failed;
    if (p->base_link_count > 0u && p->joint_limit_rows_per_env >= 2u * p->base_link_count &&
        data.joint_limit_impulse != nullptr &&
        LaunchWork(WriteBlockJointLimitImpulsesKernel, p->base_link_count * p->env_count,
            kThreads, stream, data, *p) != cudaSuccess) return Status::Failed;
    return MeasureAndRecord(model, data, *p, s, true, stream);
}

}  // namespace

uint64_t BlockDescentScratchWords(const BlockDescentSolveParams& params) {
    return BindScratch(params, nullptr, nullptr);
}

void RegisterNkBlockDescentOps() {
    SetCudaOp(NkOp::BlockDescentSolve, &OpBlockDescentSolve);
    SetCudaOp(NkOp::RecordTelemetry, &OpRecordTelemetry);
}

}  // namespace nuka::phi
