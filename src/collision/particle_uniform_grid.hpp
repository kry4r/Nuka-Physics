#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu::ParticleUniformGrid -- spatial-hash uniform grid for
// particle broadphase (v0.7 p05, Tasks 7.5.1 / 7.5.2).
//
// Standard PBF / XPBD broadphase: hash each particle to a grid cell, stable-sort
// by cell key (radix; deterministic), build per-cell [start,end) index ranges
// via a segmented scan, then answer per-particle neighbor queries over the 27
// surrounding cells. Feeds the soft / fluid / coupling phases (p08-p11).
//
// D1 (HARD): the cell-key sort is thrust::stable_sort_by_key (radix); the only
// atomic is a uint32 truncation counter; and every per-particle neighbor list is
// written into a PRIVATE output slice then sorted by neighbor index ascending,
// so the final neighbor lists are byte-identical across runs regardless of warp
// scheduling.
//
// 32-candidate cap (memory R-C, spec 7.5.5; bumped from 16 for p10 PBF): each
// particle keeps at most kMaxNeighbors candidates (the 32 LOWEST neighbor indices
// -- a deterministic subset). If a particle has more in-radius neighbors, the
// surplus is counted in TruncatedParticleCount() (surfaced, never silently
// dropped). PBF density estimation needs >=~25 neighbors for a stable SPH sum, so
// the cap was raised 16 -> 32; the deterministic "keep the lowest-index
// neighbors" truncation algorithm is UNCHANGED (D1-preserving), only the cap grew.
//
// HOST header: included from g++ TUs (tests). Config uses math::Vec3 + a
// uint32_t[3]; the kernels convert to float3/uint3 internally. No CUDA types.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace nuka::collision::gpu {

// Per-particle neighbor cap (32, spec 7.5.5 memory R-C; bumped from 16 for p10
// PBF density estimation). A particle with more in-radius neighbors keeps its 32
// lowest-index neighbors and increments the truncation counter.
inline constexpr uint32_t kParticleGridMaxNeighbors = 32u;

// Uniform-grid configuration. `cell_size` is typically 2 x particle radius (so
// the query radius fits within one cell ring -> a 27-cell search suffices).
// `grid_min` is the world AABB lower corner; `grid_dims` is cells per axis. A
// particle outside [grid_min, grid_min + cell_size*grid_dims) is CLAMPED into
// the boundary cells (so it still hashes to a valid cell; D1-stable).
struct ParticleGridConfig {
    math::Vec3 cell_size = {1.0f, 1.0f, 1.0f};
    math::Vec3 grid_min = {0.0f, 0.0f, 0.0f};
    uint32_t grid_dims[3] = {1u, 1u, 1u};

    // Total cell count = product of grid_dims (host helper).
    uint32_t CellCount() const {
        return grid_dims[0] * grid_dims[1] * grid_dims[2];
    }
};

// Build the grid config from a particle set's world bounds + a target cell size.
// cell_size is clamped to a positive value; grid_dims is sized to cover the AABB
// (with a 1-cell pad each side so boundary 27-neighbourhoods stay in-range). A
// host convenience -- the GPU build only consumes the resulting config.
ParticleGridConfig MakeParticleGridConfig(math::Vec3 world_min,
                                          math::Vec3 world_max,
                                          float cell_size);

// Result of a uniform-grid build + neighbor query. All buffers are device
// resident. The neighbor list is CSR-like: particle i's neighbors occupy
// [NeighborOffsets()[i], NeighborOffsets()[i] + NeighborCounts()[i]) in the flat
// NeighborIndices() buffer, sorted ascending by neighbor particle index. Indices
// refer to the ORIGINAL (pre-sort) particle ordering.
class ParticleGridResult {
public:
    ParticleGridResult() = default;
    ParticleGridResult(uint32_t particle_count,
                       uint32_t cell_count,
                       uint32_t neighbor_capacity,
                       uint32_t total_neighbors,
                       uint32_t truncated_particle_count,
                       phi::Buffer* sorted_particle_idx,
                       phi::Buffer* cell_start,
                       phi::Buffer* cell_end,
                       phi::Buffer* neighbor_offsets,
                       phi::Buffer* neighbor_counts,
                       phi::Buffer* neighbor_indices);
    ~ParticleGridResult();

    ParticleGridResult(const ParticleGridResult&) = delete;
    ParticleGridResult& operator=(const ParticleGridResult&) = delete;
    ParticleGridResult(ParticleGridResult&& other) noexcept;
    ParticleGridResult& operator=(ParticleGridResult&& other) noexcept;

    uint32_t ParticleCount() const { return particle_count_; }
    uint32_t CellCount() const { return cell_count_; }
    uint32_t NeighborCapacity() const { return neighbor_capacity_; }
    uint32_t TotalNeighbors() const { return total_neighbors_; }
    // Number of particles whose in-radius neighbor count exceeded the cap and
    // were truncated to kParticleGridMaxNeighbors (surfaced for diagnostics).
    uint32_t TruncatedParticleCount() const { return truncated_particle_count_; }

    // Device-resident sorted particle index array (length particle_count_): the
    // particle indices in cell-sorted order (output of the stable sort).
    const uint32_t* DeviceSortedParticleIndex() const;
    // Per-cell [start,end) ranges into the sorted index array (length cell_count_).
    const uint32_t* DeviceCellStart() const;
    const uint32_t* DeviceCellEnd() const;
    // CSR neighbor lists (offsets length particle_count_, indices flat).
    const uint32_t* DeviceNeighborOffsets() const;
    const uint32_t* DeviceNeighborCounts() const;
    const uint32_t* DeviceNeighborIndices() const;

    std::vector<uint32_t> DownloadNeighborOffsets() const;
    std::vector<uint32_t> DownloadNeighborCounts() const;
    std::vector<uint32_t> DownloadNeighborIndices() const;

private:
    uint32_t particle_count_ = 0;
    uint32_t cell_count_ = 0;
    uint32_t neighbor_capacity_ = 0;
    uint32_t total_neighbors_ = 0;
    uint32_t truncated_particle_count_ = 0;
    phi::Buffer* sorted_particle_idx_ = nullptr;  // phi v2 handles; freed in dtor.
    phi::Buffer* cell_start_ = nullptr;
    phi::Buffer* cell_end_ = nullptr;
    phi::Buffer* neighbor_offsets_ = nullptr;
    phi::Buffer* neighbor_counts_ = nullptr;
    phi::Buffer* neighbor_indices_ = nullptr;
};

// Build the uniform grid over `particle_count` device-resident positions and
// compute each particle's neighbor list within `query_radius`. `positions` must
// point to `particle_count` consecutive math::Vec3 in DEVICE memory.
//
// PRECONDITION (correctness): the smallest config.cell_size component MUST be
// >= query_radius. The neighbor search only visits the 27 cells around a
// particle's own cell, so a cell smaller than the query radius can leave an
// in-radius neighbor two cells away -- silently MISSED. MakeParticleGridConfig
// sets cell_size == cell_size argument, so passing that same value as the query
// radius (the common case) satisfies this. Consumers (p08-p11) that pick the two
// independently must keep cell_size >= query_radius.
//
// Self is excluded from each particle's neighbor list. Neighbors are capped at
// kParticleGridMaxNeighbors per particle (lowest-index subset; surplus counted).
ParticleGridResult BuildParticleUniformGrid(cudaStream_t stream, int device_id,
                                            const math::Vec3* positions,
                                            uint32_t particle_count,
                                            const ParticleGridConfig& config,
                                            float query_radius);

ParticleGridResult BuildParticleUniformGrid(const math::Vec3* positions,
                                            uint32_t particle_count,
                                            const ParticleGridConfig& config,
                                            float query_radius);

} // namespace nuka::collision::gpu
