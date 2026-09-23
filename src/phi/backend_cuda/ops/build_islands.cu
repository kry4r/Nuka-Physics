// Connected components include every shared velocity and friction group.

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cub/device/device_radix_sort.cuh>

#include "nk/model/generated/views.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/island_schedule.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi {

namespace {

using ::nuka::nk::NkRow;
using ::nuka::nk::NkRowSide;
using ::nuka::nk::kNkSideArtic;
using ::nuka::nk::kNkSideRigid;
using ::nuka::nk::kNkSideParticle;
using ::nuka::nk::kNkSideGrid;

constexpr uint32_t kBlockSize = 128u;
constexpr uint32_t kSentinel = ~0u;  // empty key / inactive root (sorts to the end).
constexpr uint64_t kScratchAlign = 256u;
inline __host__ uint64_t AlignScratch(uint64_t v) {
    return (v + (kScratchAlign - 1u)) & ~(kScratchAlign - 1u);
}

// Parent reads are atomic without serializing readers through a read-modify-write.
__device__ inline uint32_t AtomicLoad(uint32_t* p) {
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*p).load(cuda::memory_order_relaxed);
}

// Concurrent path-halving find (best-effort halve via atomicCAS; correct even when
// the CAS loses a race — the node still advances toward its root).
__device__ uint32_t Find(uint32_t* parent, uint32_t i) {
    for (;;) {
        const uint32_t p = AtomicLoad(&parent[i]);
        if (p == i) return i;
        const uint32_t gp = AtomicLoad(&parent[p]);
        atomicCAS(&parent[i], p, gp);
        i = p;
    }
}

// Smaller-root-wins union makes the component partition independent of hook order.
__device__ void Unite(uint32_t* parent, uint32_t a, uint32_t b) {
    for (;;) {
        a = Find(parent, a);
        b = Find(parent, b);
        if (a == b) return;
        const uint32_t lo = a < b ? a : b;
        const uint32_t hi = a < b ? b : a;
        if (atomicCAS(&parent[hi], hi, lo) == hi) return;  // hi was still a root.
    }
}

// Read-only find (post-union: no concurrent writers, so plain loads are race-free).
__device__ uint32_t FindReadonly(const uint32_t* parent, uint32_t i) {
    uint32_t p = parent[i];
    while (p != i) { i = p; p = parent[i]; }
    return i;
}

// cc_parent[i] = i. Used twice: the union-find seed, then the radix-sort values_in
// (the identity row ids the sort carries along into island_rows).
__global__ void FillIdentityKernel(uint32_t* arr, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

// Claim a side's GLOBAL coupling key (first active row wins the slot) and union this
// row with the prior claimer. Static sides / out-of-range keys contribute no edge.
// A zero-inverse-mass body never receives an impulse, so it carries no edge either --
// the solver already skips its write, and coupling through it would be fictitious.
__device__ inline void ClaimUnion(uint32_t* parent, const NkRowSide& s, uint32_t row,
                                  uint32_t* artic_first, uint32_t artic_count,
                                  uint32_t* body_first, uint32_t body_count,
                                  uint32_t* particle_first, uint32_t particle_count,
                                  uint32_t* grid_first, uint32_t grid_count,
                                  const nk::PointEndpointRange* ranges,
                                  const nk::PointEndpointTerm* terms,
                                  const float* body_inv_mass) {
    const bool interpolated = s.kind == nk::kNkSidePointEndpoint;
    const uint32_t count = interpolated ? ranges[s.index].count : 1u;
    for (uint32_t i = 0u; i < count; ++i) {
        const auto term = interpolated ? terms[ranges[s.index].first + i] : nk::PointEndpointTerm{};
        const uint32_t kind = interpolated ? term.kind : s.kind;
        const uint32_t index = interpolated ? term.index : s.index;
        uint32_t* table = nullptr;
        uint32_t limit = 0u;
        if (kind == kNkSideArtic)         { table = artic_first;    limit = artic_count; }
        else if (kind == kNkSideRigid)    { table = body_first;     limit = body_count; }
        else if (kind == kNkSideParticle) { table = particle_first; limit = particle_count; }
        else if (kind == kNkSideGrid)     { table = grid_first;     limit = grid_count; }
        if (table == nullptr || index >= limit) continue;
        if (kind == kNkSideRigid && body_inv_mass != nullptr &&
            !(body_inv_mass[index] > 0.0f))
            continue;
        const uint32_t old = atomicCAS(&table[index], kSentinel, row);
        if (old != kSentinel) Unite(parent, row, old);
    }
}

// One thread per row slot. Active rows union by their two sides' coupling keys + the
// friction-group anchor (so a manifold's normals + spokes always share a component).
__global__ void UnionRowsKernel(const NkRow* __restrict__ urows, uint32_t* parent,
                                uint32_t* artic_first, uint32_t artic_count,
                                uint32_t* body_first, uint32_t body_count,
                                uint32_t* particle_first, uint32_t particle_count,
                                uint32_t* grid_first, uint32_t grid_count,
                                const nk::PointEndpointRange* ranges, const nk::PointEndpointTerm* terms,
                                const float* __restrict__ body_inv_mass,
                                uint32_t total_rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    const uint32_t flags = urows[row].flags;
    if (!(flags & nk::nk_row_flags::kActive) || (flags & nk::nk_row_flags::kBlockTangent)) return;
    const NkRow r = urows[row];
    ClaimUnion(parent, r.a, row, artic_first, artic_count, body_first, body_count,
               particle_first, particle_count, grid_first, grid_count, ranges, terms,
               body_inv_mass);
    ClaimUnion(parent, r.b, row, artic_first, artic_count, body_first, body_count,
               particle_first, particle_count, grid_first, grid_count, ranges, terms,
               body_inv_mass);
    const uint32_t gf = r.group_first;
    if (gf != row && gf < total_rows) Unite(parent, row, gf);
}

// One thread per row slot. cc_root = the component root (min row) for active rows,
// SENTINEL for inactive ones (which the sort pushes past the active prefix).
__global__ void FlattenRootsKernel(const NkRow* __restrict__ urows,
                                   const uint32_t* __restrict__ parent,
                                   uint32_t* __restrict__ cc_root, uint32_t total_rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    // A block normal projects its complete friction triplet; tangents need no separate visit.
    const uint32_t flags = urows[row].flags;
    cc_root[row] = ((flags & nk::nk_row_flags::kActive) && !(flags & nk::nk_row_flags::kBlockTangent))
                       ? FindReadonly(parent, row) : kSentinel;
}

__global__ void AccumulateIslandFlagsKernel(
    uint32_t* __restrict__ roots, const NkRow* __restrict__ urows,
    const nk::PointEndpointRange* __restrict__ endpoint_ranges,
    uint32_t total_rows, uint32_t* __restrict__ root_flags) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_rows) return;
    const NkRow& row = urows[i];
    if (!(row.flags & nk::nk_row_flags::kActive)) return;
    // Tangents share their normal's component even though the solve visits only block normals.
    if (row.flags & nk::nk_row_flags::kBlockTangent) {
        roots[i] = roots[row.group_first];
        return;
    }
    const uint32_t root = roots[i];
    uint32_t flags = 0u;
    if (row.a.kind == kNkSideArtic || row.b.kind == kNkSideArtic)
        flags |= nkops::kIslandHasArticulation;
    if ((row.a.kind == nk::kNkSidePointEndpoint && endpoint_ranges[row.a.index].count > 1u) ||
        (row.b.kind == nk::kNkSidePointEndpoint && endpoint_ranges[row.b.index].count > 1u))
        flags |= nkops::kIslandHasMultiplePointTerms;
    if (flags != 0u) atomicOr(&root_flags[root], flags);
}

// Component bounds use binary search; endpoint flags are reduced in parallel.
__global__ void EmitIslandsKernel(const uint32_t* __restrict__ root_sorted,
                                  const uint32_t* __restrict__ rows,
                                  const uint32_t* __restrict__ root_flags,
                                  uint32_t total_rows, uint32_t rows_per_env,
                                  uint32_t* __restrict__ island_count,
                                  nkops::IslandRecord* __restrict__ islands) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_rows) return;
    const uint32_t root = root_sorted[i];
    if (root == kSentinel) return;                 // inactive tail.
    if (i > 0u && root_sorted[i - 1u] == root) return;  // not a component start.
    uint32_t begin = i + 1u, end = total_rows;
    while (begin < end) {
        const uint32_t middle = begin + (end - begin) / 2u;
        if (root_sorted[middle] == root) begin = middle + 1u;
        else end = middle;
    }
    const uint32_t row_cnt = begin - i;
    const uint32_t env = (rows_per_env > 0u) ? (rows[i] / rows_per_env) : 0u;
    const uint32_t comp = atomicAdd(island_count, 1u);  // comp < #components <= total_rows.
    islands[comp] = {i, row_cnt, root_flags[root], env};
}

}  // namespace

uint64_t IslandSortScratchBytes(uint32_t total_rows) {
    if (total_rows == 0u) return 0u;
    size_t bytes = 0u;
    (void)cub::DeviceRadixSort::SortPairs<uint32_t, uint32_t>(
        nullptr, bytes, static_cast<const uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<const uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<int>(total_rows));
    return AlignScratch(bytes);
}

Status OpBuildSolveIslands(const ModelView& /*model*/, const DataView& data,
                           const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const BuildSolveIslandsParams*>(params);
    if (p == nullptr) return Status::Failed;
    // Gated to the ONE general path: a UnionCsr / contact-free model keeps its
    // cook-time schedule (the op is a no-op, op list byte-identical).
    if (p->family != kContactFamilyPairDriven) return Status::Ok;
    const uint32_t total_rows = p->env_count * p->rows_per_env;
    if (total_rows == 0u) return Status::Ok;
    const uint32_t artic_count = p->articulation_count;
    const uint32_t body_count =
        static_cast<uint32_t>(static_cast<uint64_t>(p->bodies_per_env) * p->env_count);
    const uint32_t particle_count =
        static_cast<uint32_t>(static_cast<uint64_t>(p->particles_per_env) * p->env_count);
    const uint32_t grid_count = p->grid_nodes_per_env * p->env_count;
    const uint32_t rblocks = (total_rows + kBlockSize - 1u) / kBlockSize;
    const auto* urows = reinterpret_cast<const NkRow*>(data.urows);

    // Seed parent = identity, key tables = empty, component count = 0.
    LaunchCuda(FillIdentityKernel, dim3(rblocks), dim3(kBlockSize), 0u, stream,
               data.cc_parent, total_rows);
    if (artic_count > 0u) {
        (void)cudaMemsetAsync(data.cc_artic_first, 0xFF,
                              static_cast<size_t>(artic_count) * sizeof(uint32_t), stream);
    }
    if (body_count > 0u) {
        (void)cudaMemsetAsync(data.cc_body_first, 0xFF,
                              static_cast<size_t>(body_count) * sizeof(uint32_t), stream);
    }
    if (particle_count > 0u) {
        (void)cudaMemsetAsync(data.cc_particle_first, 0xFF,
                              static_cast<size_t>(particle_count) * sizeof(uint32_t), stream);
    }
    (void)cudaMemsetAsync(data.island_count, 0, sizeof(uint32_t), stream);
    if (grid_count > 0u)
        (void)cudaMemsetAsync(data.cc_grid_first, 0xFF,
                              static_cast<size_t>(grid_count) * sizeof(uint32_t), stream);

    // Union-find, then flatten each active row to its component root.
    LaunchCuda(UnionRowsKernel, dim3(rblocks), dim3(kBlockSize), 0u, stream, urows,
               data.cc_parent, data.cc_artic_first, artic_count, data.cc_body_first,
               body_count, data.cc_particle_first, particle_count,
               data.cc_grid_first, grid_count,
               data.point_endpoint_ranges, data.point_endpoint_terms,
               static_cast<const float*>(data.body_inv_mass), total_rows);
    LaunchCuda(FlattenRootsKernel, dim3(rblocks), dim3(kBlockSize), 0u, stream, urows,
               data.cc_parent, data.cc_root, total_rows);

    // Stable radix sort (key = root, value = row id) -> rows grouped per component,
    // ASCENDING within. cc_parent is reloaded as the identity values_in.
    LaunchCuda(FillIdentityKernel, dim3(rblocks), dim3(kBlockSize), 0u, stream,
               data.cc_parent, total_rows);
    size_t temp_bytes = 0u;
    (void)cub::DeviceRadixSort::SortPairs<uint32_t, uint32_t>(
        nullptr, temp_bytes, data.cc_root, data.island_root_sorted, data.cc_parent,
        data.island_rows, static_cast<int>(total_rows));
    if (cub::DeviceRadixSort::SortPairs(
            data.island_cub_temp, temp_bytes, data.cc_root, data.island_root_sorted,
            data.cc_parent, data.island_rows, static_cast<int>(total_rows), 0, 32,
            stream) != cudaSuccess) {
        return Status::Failed;
    }

    // Sorting releases cc_parent; reuse it for root flags until island records are emitted.
    if (cudaMemsetAsync(data.cc_parent, 0, static_cast<size_t>(total_rows) * sizeof(uint32_t),
                        stream) != cudaSuccess) return Status::Failed;
    LaunchCuda(AccumulateIslandFlagsKernel, dim3(rblocks), dim3(kBlockSize), 0u, stream,
               data.cc_root, urows, data.point_endpoint_ranges,
               total_rows, data.cc_parent);
    LaunchCuda(EmitIslandsKernel, dim3(rblocks), dim3(kBlockSize), 0u, stream,
               data.island_root_sorted, data.island_rows, data.cc_parent,
               total_rows, p->rows_per_env, data.island_count,
               reinterpret_cast<nkops::IslandRecord*>(data.island_quads));
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

void RegisterNkBuildSolveIslandsOps() {
    SetCudaOp(NkOp::BuildSolveIslands, &OpBuildSolveIslands);
}

}  // namespace nuka::phi
