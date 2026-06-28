// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — BuildSolveIslands (dynamic connected-component schedule).
//
// The PairDriven contact rows are DYNAMIC broadphase candidates, so the cook-time
// schedule conservatively held ALL of an env's rows in ONE island (one block). For
// a multi-articulation scene (30 spatially-disjoint dogs in one env) that serializes
// every contact through a single block. This op re-derives the TRUE islands each
// step, on device, with ZERO host round-trip: a union-find over the ACTIVE rows
// keyed by every coupling a row touches (artic side a/b, rigid body, particle) plus
// the friction-group anchor, then a stable radix sort that groups each component's
// rows CONTIGUOUSLY in ASCENDING slot order. The solve runs one block per component
// in parallel.
//
// BYTE-IDENTITY: components are state-disjoint (the union-find merges every shared
// coupling), so reordering rows across components changes nothing in any component's
// arithmetic; the stable sort preserves each component's ascending row order, which
// is exactly the order the single-island-per-env sweep used — bit-for-bit identical.
// A single-articulation scene is ONE component -> structurally the old single island.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>

#include "nk/model/generated/views.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend_cuda/launch.cuh"
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

constexpr uint32_t kBlockSize = 128u;
constexpr uint32_t kSentinel = ~0u;  // empty key / inactive root (sorts to the end).
constexpr uint64_t kScratchAlign = 256u;
inline __host__ uint64_t AlignScratch(uint64_t v) {
    return (v + (kScratchAlign - 1u)) & ~(kScratchAlign - 1u);
}

// Atomic load of a union-find parent slot (atomicOr-0): keeps EVERY parent access
// atomic so the concurrent hooks/finds never tear under the CUDA memory model.
__device__ inline uint32_t AtomicLoad(uint32_t* p) { return atomicOr(p, 0u); }

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

// Lock-free union, SMALLER-ROOT-WINS (the host RowUnionFind convention: the
// component root is the min row index). The components are determined by the edge
// set, not the union order, so any interleaving yields the same partition.
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
__device__ inline void ClaimUnion(uint32_t* parent, const NkRowSide& s, uint32_t row,
                                  uint32_t* artic_first, uint32_t artic_count,
                                  uint32_t* body_first, uint32_t body_count,
                                  uint32_t* particle_first, uint32_t particle_count) {
    uint32_t* table = nullptr;
    uint32_t limit = 0u;
    if (s.kind == kNkSideArtic)         { table = artic_first;    limit = artic_count; }
    else if (s.kind == kNkSideRigid)    { table = body_first;     limit = body_count; }
    else if (s.kind == kNkSideParticle) { table = particle_first; limit = particle_count; }
    else return;
    if (s.index >= limit) return;
    const uint32_t old = atomicCAS(&table[s.index], kSentinel, row);
    if (old != kSentinel) Unite(parent, row, old);
}

// One thread per row slot. Active rows union by their two sides' coupling keys + the
// friction-group anchor (so a manifold's normals + spokes always share a component).
__global__ void UnionRowsKernel(const NkRow* __restrict__ urows, uint32_t* parent,
                                uint32_t* artic_first, uint32_t artic_count,
                                uint32_t* body_first, uint32_t body_count,
                                uint32_t* particle_first, uint32_t particle_count,
                                uint32_t total_rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    if (!(urows[row].flags & nk::nk_row_flags::kActive)) return;
    const NkRow r = urows[row];
    ClaimUnion(parent, r.a, row, artic_first, artic_count, body_first, body_count,
               particle_first, particle_count);
    ClaimUnion(parent, r.b, row, artic_first, artic_count, body_first, body_count,
               particle_first, particle_count);
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
    cc_root[row] = (urows[row].flags & nk::nk_row_flags::kActive)
                       ? FindReadonly(parent, row) : kSentinel;
}

// One thread per SORTED position. The radix sort grouped equal roots contiguously
// (ascending row within, stable). A position whose root differs from its predecessor
// is a component START: it scans its run, emits the {row_off,row_cnt,flags,env} quad
// at a freshly reserved component id, and stamps the has-artic flag.
__global__ void EmitIslandsKernel(const uint32_t* __restrict__ root_sorted,
                                  const uint32_t* __restrict__ rows,
                                  const NkRow* __restrict__ urows,
                                  uint32_t total_rows, uint32_t rows_per_env,
                                  uint32_t* __restrict__ island_count,
                                  uint32_t* __restrict__ island_quads) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_rows) return;
    const uint32_t root = root_sorted[i];
    if (root == kSentinel) return;                 // inactive tail.
    if (i > 0u && root_sorted[i - 1u] == root) return;  // not a component start.
    uint32_t j = i + 1u;
    while (j < total_rows && root_sorted[j] == root) ++j;
    const uint32_t row_cnt = j - i;
    const uint32_t env = (rows_per_env > 0u) ? (rows[i] / rows_per_env) : 0u;
    uint32_t flags = 0u;
    for (uint32_t k = i; k < j; ++k) {
        const NkRow& r = urows[rows[k]];
        if (r.a.kind == kNkSideArtic || r.b.kind == kNkSideArtic) { flags |= 1u; break; }
    }
    const uint32_t comp = atomicAdd(island_count, 1u);  // comp < #components <= total_rows.
    uint32_t* const q = island_quads + static_cast<size_t>(comp) * 4u;
    q[0] = i;        // row_off into island_rows
    q[1] = row_cnt;
    q[2] = flags;    // bit0 == component has an articulation row
    q[3] = env;
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

    // Union-find, then flatten each active row to its component root.
    LaunchCuda(UnionRowsKernel, dim3(rblocks), dim3(kBlockSize), 0u, stream, urows,
               data.cc_parent, data.cc_artic_first, artic_count, data.cc_body_first,
               body_count, data.cc_particle_first, particle_count, total_rows);
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

    // Emit one island per component span.
    LaunchCuda(EmitIslandsKernel, dim3(rblocks), dim3(kBlockSize), 0u, stream,
               data.island_root_sorted, data.island_rows, urows, total_rows,
               p->rows_per_env, data.island_count, data.island_quads);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

void RegisterNkBuildSolveIslandsOps() {
    SetCudaOp(NkOp::BuildSolveIslands, &OpBuildSolveIslands);
}

}  // namespace nuka::phi
