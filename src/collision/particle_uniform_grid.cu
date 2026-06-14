// ---------------------------------------------------------------------------
// nuka::collision::gpu::BuildParticleUniformGrid implementation (v0.7 p05).
//
// Pipeline (D1-critical points marked):
//   1. compute_cell_keys     -- hash each particle to a cell key + record idx
//   2. stable_sort_by_key    -- (D1) radix sort particles by cell key
//   3. build_cell_ranges     -- per-cell [start,end) from the sorted keys
//   4. count_neighbors       -- pass 1: per-particle in-radius neighbor count
//                               (capped at kParticleGridMaxNeighbors == 32; bumped
//                               from 16 for p10 PBF density estimation)
//   5. exclusive scan        -- (D1) prefix-sum counts -> CSR offsets
//   6. fill_neighbors        -- pass 2: write each particle's neighbors into its
//                               PRIVATE CSR slice, already sorted ascending
//
// D1: the only sort is thrust::stable_sort_by_key (radix; deterministic); the
// only atomic is a uint32 truncation counter; every neighbor list is written
// into a per-particle private slice (no append atomic) and produced sorted by
// the insertion-sort in QueryParticleNeighbors -- so neighbor lists are
// byte-identical run-to-run regardless of warp order. NO float atomics.
// ---------------------------------------------------------------------------

#include "collision/particle_uniform_grid.hpp"

#include "collision/particle_grid_traversal.cuh"
#include "phi/backend.hpp"            // InitBestDevice, DeviceBufferType
#include "phi/buffer_transfer_v2.hpp" // DownloadVectorV2

#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::collision::gpu {

namespace {

constexpr uint32_t kBlockSize = 128u;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// File-local RAII over a phi v2 opaque Buffer*. Stream-0 device buffer type:
// NULL-stream transfers are byte+ordering identical to the legacy synchronous
// memcpy. Frees on every path (incl. CheckCuda throw paths). Base()/Handle()
// mirror the legacy Data(); Release() hands the long-lived result members their
// owned handle.
class OwnedBuffer {
public:
    OwnedBuffer() = default;
    explicit OwnedBuffer(size_t bytes) : bytes_(bytes) {
        buf_ = phi::BufferAlloc(phi::DeviceBufferType(phi::InitBestDevice()), bytes);
    }
    ~OwnedBuffer() { if (buf_ != nullptr) phi::BufferFree(buf_); }
    OwnedBuffer(OwnedBuffer&& o) noexcept : buf_(o.buf_), bytes_(o.bytes_) {
        o.buf_ = nullptr; o.bytes_ = 0u;
    }
    OwnedBuffer& operator=(OwnedBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_ != nullptr) phi::BufferFree(buf_);
            buf_ = o.buf_; bytes_ = o.bytes_; o.buf_ = nullptr; o.bytes_ = 0u;
        }
        return *this;
    }
    OwnedBuffer(const OwnedBuffer&) = delete;
    OwnedBuffer& operator=(const OwnedBuffer&) = delete;
    void*  Base()   const { return buf_ != nullptr ? phi::BufferBase(buf_) : nullptr; }
    size_t Bytes()  const { return bytes_; }
    phi::Buffer* Handle() const { return buf_; }
    phi::Buffer* Release() { phi::Buffer* b = buf_; buf_ = nullptr; bytes_ = 0u; return b; }
private:
    phi::Buffer* buf_ = nullptr;
    size_t bytes_ = 0u;
};

ParticleGridConfigDevice ToDevice(const ParticleGridConfig& cfg) {
    ParticleGridConfigDevice d;
    d.cell_size = make_float3(cfg.cell_size.x, cfg.cell_size.y, cfg.cell_size.z);
    d.inv_cell_size = make_float3(
        cfg.cell_size.x > 0.0f ? 1.0f / cfg.cell_size.x : 0.0f,
        cfg.cell_size.y > 0.0f ? 1.0f / cfg.cell_size.y : 0.0f,
        cfg.cell_size.z > 0.0f ? 1.0f / cfg.cell_size.z : 0.0f);
    d.grid_min = make_float3(cfg.grid_min.x, cfg.grid_min.y, cfg.grid_min.z);
    d.grid_dims = make_uint3(cfg.grid_dims[0], cfg.grid_dims[1], cfg.grid_dims[2]);
    return d;
}

// --- 1. Cell keys -----------------------------------------------------------
__global__ void ComputeCellKeysKernel(uint32_t particle_count,
                                       const math::Vec3* __restrict__ positions,
                                       ParticleGridConfigDevice cfg,
                                       uint32_t* __restrict__ out_cell_keys,
                                       uint32_t* __restrict__ out_particle_idx) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 p = positions[i];
    out_cell_keys[i] = CellKeyFromPos(make_float3(p.x, p.y, p.z), cfg);
    out_particle_idx[i] = i;
}

// --- 3. Cell ranges ---------------------------------------------------------
// One thread per sorted entry. A boundary between cell keys i-1 and i closes the
// previous cell's [.,end) and opens cell key[i]'s [start,.). cell_start/cell_end
// are pre-zeroed to indicate empty cells (start==end==0 for an empty cell key is
// fine since no particle hashes to it). Writes are to distinct cell slots keyed
// by the (sorted) cell key, so they are race-free and order-independent (D1).
__global__ void BuildCellRangesKernel(uint32_t particle_count,
                                       const uint32_t* __restrict__ sorted_cell_keys,
                                       uint32_t* __restrict__ out_cell_start,
                                       uint32_t* __restrict__ out_cell_end) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const uint32_t key = sorted_cell_keys[i];
    if (i == 0u) {
        out_cell_start[key] = 0u;
    } else {
        const uint32_t prev = sorted_cell_keys[i - 1u];
        if (prev != key) {
            out_cell_end[prev] = i;
            out_cell_start[key] = i;
        }
    }
    if (i == particle_count - 1u) {
        out_cell_end[key] = particle_count;
    }
}

// --- 4. Neighbor count (pass 1) ---------------------------------------------
__global__ void CountNeighborsKernel(uint32_t particle_count,
                                      const math::Vec3* __restrict__ positions,
                                      float radius,
                                      ParticleGridConfigDevice cfg,
                                      const uint32_t* __restrict__ cell_start,
                                      const uint32_t* __restrict__ cell_end,
                                      const uint32_t* __restrict__ particle_idx_sorted,
                                      uint32_t max_neighbors,
                                      uint32_t* __restrict__ out_counts) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 pm = positions[i];
    const float3 p = make_float3(pm.x, pm.y, pm.z);
    uint32_t scratch[kParticleGridMaxNeighbors];
    const uint32_t n = QueryParticleNeighbors(
        p, i, radius, cfg, cell_start, cell_end, particle_idx_sorted, positions,
        scratch, max_neighbors, /*out_overflow=*/nullptr);
    out_counts[i] = n;
}

// --- 6. Neighbor fill (pass 2) ----------------------------------------------
__global__ void FillNeighborsKernel(uint32_t particle_count,
                                     const math::Vec3* __restrict__ positions,
                                     float radius,
                                     ParticleGridConfigDevice cfg,
                                     const uint32_t* __restrict__ cell_start,
                                     const uint32_t* __restrict__ cell_end,
                                     const uint32_t* __restrict__ particle_idx_sorted,
                                     uint32_t max_neighbors,
                                     const uint32_t* __restrict__ offsets,
                                     uint32_t* __restrict__ out_neighbor_idx,
                                     uint32_t* __restrict__ out_truncated_count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const math::Vec3 pm = positions[i];
    const float3 p = make_float3(pm.x, pm.y, pm.z);
    bool overflow = false;
    // Write directly into this particle's PRIVATE CSR slice (already sorted
    // ascending by QueryParticleNeighbors). No append atomic -> D1 by region.
    uint32_t* slice = out_neighbor_idx + offsets[i];
    (void)QueryParticleNeighbors(
        p, i, radius, cfg, cell_start, cell_end, particle_idx_sorted, positions,
        slice, max_neighbors, &overflow);
    if (overflow) {
        // uint32 atomic counter -- D1 (its value is order-independent; only the
        // COUNT of truncated particles matters, not which thread bumped it).
        atomicAdd(out_truncated_count, 1u);
    }
}

} // namespace

// --- Result accessors -------------------------------------------------------
ParticleGridResult::ParticleGridResult(uint32_t particle_count,
                                       uint32_t cell_count,
                                       uint32_t neighbor_capacity,
                                       uint32_t total_neighbors,
                                       uint32_t truncated_particle_count,
                                       phi::Buffer* sorted_particle_idx,
                                       phi::Buffer* cell_start,
                                       phi::Buffer* cell_end,
                                       phi::Buffer* neighbor_offsets,
                                       phi::Buffer* neighbor_counts,
                                       phi::Buffer* neighbor_indices)
    : particle_count_(particle_count)
    , cell_count_(cell_count)
    , neighbor_capacity_(neighbor_capacity)
    , total_neighbors_(total_neighbors)
    , truncated_particle_count_(truncated_particle_count)
    , sorted_particle_idx_(sorted_particle_idx)
    , cell_start_(cell_start)
    , cell_end_(cell_end)
    , neighbor_offsets_(neighbor_offsets)
    , neighbor_counts_(neighbor_counts)
    , neighbor_indices_(neighbor_indices) {}

ParticleGridResult::~ParticleGridResult() {
    if (sorted_particle_idx_ != nullptr) { phi::BufferFree(sorted_particle_idx_); }
    if (cell_start_          != nullptr) { phi::BufferFree(cell_start_); }
    if (cell_end_            != nullptr) { phi::BufferFree(cell_end_); }
    if (neighbor_offsets_    != nullptr) { phi::BufferFree(neighbor_offsets_); }
    if (neighbor_counts_     != nullptr) { phi::BufferFree(neighbor_counts_); }
    if (neighbor_indices_    != nullptr) { phi::BufferFree(neighbor_indices_); }
}

ParticleGridResult::ParticleGridResult(ParticleGridResult&& other) noexcept
    : particle_count_(other.particle_count_)
    , cell_count_(other.cell_count_)
    , neighbor_capacity_(other.neighbor_capacity_)
    , total_neighbors_(other.total_neighbors_)
    , truncated_particle_count_(other.truncated_particle_count_)
    , sorted_particle_idx_(other.sorted_particle_idx_)
    , cell_start_(other.cell_start_)
    , cell_end_(other.cell_end_)
    , neighbor_offsets_(other.neighbor_offsets_)
    , neighbor_counts_(other.neighbor_counts_)
    , neighbor_indices_(other.neighbor_indices_) {
    other.particle_count_ = 0u;
    other.cell_count_ = 0u;
    other.neighbor_capacity_ = 0u;
    other.total_neighbors_ = 0u;
    other.truncated_particle_count_ = 0u;
    other.sorted_particle_idx_ = nullptr;
    other.cell_start_ = nullptr;
    other.cell_end_ = nullptr;
    other.neighbor_offsets_ = nullptr;
    other.neighbor_counts_ = nullptr;
    other.neighbor_indices_ = nullptr;
}

ParticleGridResult& ParticleGridResult::operator=(ParticleGridResult&& other) noexcept {
    if (this != &other) {
        if (sorted_particle_idx_ != nullptr) { phi::BufferFree(sorted_particle_idx_); }
        if (cell_start_          != nullptr) { phi::BufferFree(cell_start_); }
        if (cell_end_            != nullptr) { phi::BufferFree(cell_end_); }
        if (neighbor_offsets_    != nullptr) { phi::BufferFree(neighbor_offsets_); }
        if (neighbor_counts_     != nullptr) { phi::BufferFree(neighbor_counts_); }
        if (neighbor_indices_    != nullptr) { phi::BufferFree(neighbor_indices_); }
        particle_count_ = other.particle_count_;
        cell_count_ = other.cell_count_;
        neighbor_capacity_ = other.neighbor_capacity_;
        total_neighbors_ = other.total_neighbors_;
        truncated_particle_count_ = other.truncated_particle_count_;
        sorted_particle_idx_ = other.sorted_particle_idx_;
        cell_start_ = other.cell_start_;
        cell_end_ = other.cell_end_;
        neighbor_offsets_ = other.neighbor_offsets_;
        neighbor_counts_ = other.neighbor_counts_;
        neighbor_indices_ = other.neighbor_indices_;
        other.particle_count_ = 0u;
        other.cell_count_ = 0u;
        other.neighbor_capacity_ = 0u;
        other.total_neighbors_ = 0u;
        other.truncated_particle_count_ = 0u;
        other.sorted_particle_idx_ = nullptr;
        other.cell_start_ = nullptr;
        other.cell_end_ = nullptr;
        other.neighbor_offsets_ = nullptr;
        other.neighbor_counts_ = nullptr;
        other.neighbor_indices_ = nullptr;
    }
    return *this;
}

const uint32_t* ParticleGridResult::DeviceSortedParticleIndex() const {
    return static_cast<const uint32_t*>(
        sorted_particle_idx_ != nullptr ? phi::BufferBase(sorted_particle_idx_) : nullptr);
}
const uint32_t* ParticleGridResult::DeviceCellStart() const {
    return static_cast<const uint32_t*>(
        cell_start_ != nullptr ? phi::BufferBase(cell_start_) : nullptr);
}
const uint32_t* ParticleGridResult::DeviceCellEnd() const {
    return static_cast<const uint32_t*>(
        cell_end_ != nullptr ? phi::BufferBase(cell_end_) : nullptr);
}
const uint32_t* ParticleGridResult::DeviceNeighborOffsets() const {
    return static_cast<const uint32_t*>(
        neighbor_offsets_ != nullptr ? phi::BufferBase(neighbor_offsets_) : nullptr);
}
const uint32_t* ParticleGridResult::DeviceNeighborCounts() const {
    return static_cast<const uint32_t*>(
        neighbor_counts_ != nullptr ? phi::BufferBase(neighbor_counts_) : nullptr);
}
const uint32_t* ParticleGridResult::DeviceNeighborIndices() const {
    return static_cast<const uint32_t*>(
        neighbor_indices_ != nullptr ? phi::BufferBase(neighbor_indices_) : nullptr);
}

std::vector<uint32_t> ParticleGridResult::DownloadNeighborOffsets() const {
    return phi::DownloadVectorV2<uint32_t>(neighbor_offsets_, particle_count_);
}
std::vector<uint32_t> ParticleGridResult::DownloadNeighborCounts() const {
    return phi::DownloadVectorV2<uint32_t>(neighbor_counts_, particle_count_);
}
std::vector<uint32_t> ParticleGridResult::DownloadNeighborIndices() const {
    return phi::DownloadVectorV2<uint32_t>(neighbor_indices_, total_neighbors_);
}

ParticleGridConfig MakeParticleGridConfig(math::Vec3 world_min,
                                          math::Vec3 world_max,
                                          float cell_size) {
    ParticleGridConfig cfg;
    const float cs = cell_size > 0.0f ? cell_size : 1.0f;
    cfg.cell_size = {cs, cs, cs};
    // Pad one cell on the low side so a particle exactly on the boundary still
    // has a -1 neighbor cell in-range (the query clamps anyway, but the pad
    // keeps the 27-neighbourhood symmetric for interior particles).
    cfg.grid_min = {world_min.x - cs, world_min.y - cs, world_min.z - cs};
    const float ext_x = (world_max.x - world_min.x) + 2.0f * cs;
    const float ext_y = (world_max.y - world_min.y) + 2.0f * cs;
    const float ext_z = (world_max.z - world_min.z) + 2.0f * cs;
    auto dim = [cs](float ext) -> uint32_t {
        const float d = std::floor(ext / cs) + 1.0f;
        const long v = static_cast<long>(d);
        return static_cast<uint32_t>(v < 1 ? 1 : v);
    };
    cfg.grid_dims[0] = dim(ext_x);
    cfg.grid_dims[1] = dim(ext_y);
    cfg.grid_dims[2] = dim(ext_z);
    return cfg;
}

ParticleGridResult BuildParticleUniformGrid(const phi::DeviceContext& context,
                                            const math::Vec3* positions,
                                            uint32_t particle_count,
                                            const ParticleGridConfig& config,
                                            float query_radius) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    const uint32_t cell_count = config.CellCount();
    const ParticleGridConfigDevice cfg = ToDevice(config);

    if (particle_count == 0u || cell_count == 0u) {
        OwnedBuffer e0(0u);
        OwnedBuffer e1(0u);
        OwnedBuffer e2(0u);
        OwnedBuffer e3(0u);
        OwnedBuffer e4(0u);
        OwnedBuffer e5(0u);
        return ParticleGridResult(particle_count, cell_count,
                                  kParticleGridMaxNeighbors, 0u, 0u,
                                  e0.Release(), e1.Release(), e2.Release(),
                                  e3.Release(), e4.Release(), e5.Release());
    }

    const uint32_t blocks = (particle_count + kBlockSize - 1u) / kBlockSize;

    // --- 1. Cell keys -------------------------------------------------------
    OwnedBuffer d_keys(particle_count * sizeof(uint32_t));
    OwnedBuffer d_sorted_idx(particle_count * sizeof(uint32_t));
    ComputeCellKeysKernel<<<blocks, kBlockSize, 0, stream>>>(
        particle_count, positions, cfg,
        static_cast<uint32_t*>(d_keys.Base()),
        static_cast<uint32_t*>(d_sorted_idx.Base()));
    CheckCuda(cudaGetLastError(), "ComputeCellKeysKernel launch");

    // --- 2. Stable radix sort by cell key (D1) ------------------------------
    thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        static_cast<uint32_t*>(d_keys.Base()),
        static_cast<uint32_t*>(d_keys.Base()) + particle_count,
        static_cast<uint32_t*>(d_sorted_idx.Base()));
    CheckCuda(cudaGetLastError(), "stable_sort_by_key cell keys");

    // --- 3. Cell ranges -----------------------------------------------------
    OwnedBuffer d_cell_start(cell_count * sizeof(uint32_t));
    OwnedBuffer d_cell_end(cell_count * sizeof(uint32_t));
    CheckCuda(cudaMemsetAsync(d_cell_start.Base(), 0, cell_count * sizeof(uint32_t), stream),
              "memset cell_start");
    CheckCuda(cudaMemsetAsync(d_cell_end.Base(), 0, cell_count * sizeof(uint32_t), stream),
              "memset cell_end");
    BuildCellRangesKernel<<<blocks, kBlockSize, 0, stream>>>(
        particle_count,
        static_cast<const uint32_t*>(d_keys.Base()),
        static_cast<uint32_t*>(d_cell_start.Base()),
        static_cast<uint32_t*>(d_cell_end.Base()));
    CheckCuda(cudaGetLastError(), "BuildCellRangesKernel launch");

    // --- 4. Count neighbors (pass 1) ----------------------------------------
    OwnedBuffer d_counts(particle_count * sizeof(uint32_t));
    CountNeighborsKernel<<<blocks, kBlockSize, 0, stream>>>(
        particle_count, positions, query_radius, cfg,
        static_cast<const uint32_t*>(d_cell_start.Base()),
        static_cast<const uint32_t*>(d_cell_end.Base()),
        static_cast<const uint32_t*>(d_sorted_idx.Base()),
        kParticleGridMaxNeighbors,
        static_cast<uint32_t*>(d_counts.Base()));
    CheckCuda(cudaGetLastError(), "CountNeighborsKernel launch");

    // --- 5. Exclusive scan -> CSR offsets (D1) ------------------------------
    OwnedBuffer d_offsets(particle_count * sizeof(uint32_t));
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        static_cast<const uint32_t*>(d_counts.Base()),
        static_cast<const uint32_t*>(d_counts.Base()) + particle_count,
        static_cast<uint32_t*>(d_offsets.Base()));
    CheckCuda(cudaGetLastError(), "exclusive_scan counts");

    // Total neighbors = offsets[last] + counts[last].
    context.stream.Synchronize();
    uint32_t last_offset = 0u;
    uint32_t last_count = 0u;
    // Read the final offset + count from the device tails.
    {
        const auto* off_dev = static_cast<const uint32_t*>(d_offsets.Base());
        const auto* cnt_dev = static_cast<const uint32_t*>(d_counts.Base());
        CheckCuda(cudaMemcpy(&last_offset, off_dev + (particle_count - 1u),
                             sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "copy last offset");
        CheckCuda(cudaMemcpy(&last_count, cnt_dev + (particle_count - 1u),
                             sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "copy last count");
    }
    const uint32_t total_neighbors = last_offset + last_count;

    // --- 6. Fill neighbors (pass 2) -----------------------------------------
    OwnedBuffer d_neighbors(
        (total_neighbors == 0u ? 1u : total_neighbors) * sizeof(uint32_t));
    OwnedBuffer d_truncated(sizeof(uint32_t));
    CheckCuda(cudaMemsetAsync(d_truncated.Base(), 0, sizeof(uint32_t), stream),
              "memset truncated");
    FillNeighborsKernel<<<blocks, kBlockSize, 0, stream>>>(
        particle_count, positions, query_radius, cfg,
        static_cast<const uint32_t*>(d_cell_start.Base()),
        static_cast<const uint32_t*>(d_cell_end.Base()),
        static_cast<const uint32_t*>(d_sorted_idx.Base()),
        kParticleGridMaxNeighbors,
        static_cast<const uint32_t*>(d_offsets.Base()),
        static_cast<uint32_t*>(d_neighbors.Base()),
        static_cast<uint32_t*>(d_truncated.Base()));
    CheckCuda(cudaGetLastError(), "FillNeighborsKernel launch");

    context.stream.Synchronize();
    uint32_t truncated = 0u;
    phi::BufferDownload(d_truncated.Handle(), &truncated, 0, sizeof(truncated));

    return ParticleGridResult(particle_count, cell_count,
                              kParticleGridMaxNeighbors, total_neighbors,
                              truncated,
                              d_sorted_idx.Release(),
                              d_cell_start.Release(),
                              d_cell_end.Release(),
                              d_offsets.Release(),
                              d_counts.Release(),
                              d_neighbors.Release());
}

ParticleGridResult BuildParticleUniformGrid(const math::Vec3* positions,
                                            uint32_t particle_count,
                                            const ParticleGridConfig& config,
                                            float query_radius) {
    auto context = phi::MakeDefaultDeviceContext();
    return BuildParticleUniformGrid(context, positions, particle_count, config,
                                    query_radius);
}

} // namespace nuka::collision::gpu
