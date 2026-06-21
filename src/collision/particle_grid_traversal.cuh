#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu -- particle uniform-grid device helpers. Cell hashing
// + per-particle 27-cell neighbor query.
//
// Device-only; included only from .cu TUs. The cell hash is a pure function of
// (position, config) -> identical floats give identical cell, so it is D1-safe.
// The neighbor query writes into a caller-provided private buffer; ordering /
// the cap / the final sort are handled by the kernel that calls it.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>

#include <cuda_runtime.h>

namespace nuka::collision::gpu {

// Device-side mirror of the host ParticleGridConfig (float3 / uint3 for the
// kernels). Populated on the host from the host config and passed by value.
struct ParticleGridConfigDevice {
    float3 cell_size;
    float3 inv_cell_size; // 1/cell_size (precomputed; 0 if a component is 0)
    float3 grid_min;
    uint3 grid_dims;
};

// Clamp x into [lo, hi] (integer).
__device__ __forceinline__ int ClampInt(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Compute the integer cell coordinate of a position, clamped into the grid.
__device__ __forceinline__ uint3 CellCoord(float3 p,
                                            const ParticleGridConfigDevice& cfg) {
    const int cx = static_cast<int>(floorf((p.x - cfg.grid_min.x) * cfg.inv_cell_size.x));
    const int cy = static_cast<int>(floorf((p.y - cfg.grid_min.y) * cfg.inv_cell_size.y));
    const int cz = static_cast<int>(floorf((p.z - cfg.grid_min.z) * cfg.inv_cell_size.z));
    uint3 c;
    c.x = static_cast<uint32_t>(ClampInt(cx, 0, static_cast<int>(cfg.grid_dims.x) - 1));
    c.y = static_cast<uint32_t>(ClampInt(cy, 0, static_cast<int>(cfg.grid_dims.y) - 1));
    c.z = static_cast<uint32_t>(ClampInt(cz, 0, static_cast<int>(cfg.grid_dims.z) - 1));
    return c;
}

// Flatten a cell coordinate to a linear cell key (x-fastest). Deterministic.
__device__ __forceinline__ uint32_t CellKey(uint3 c,
                                             const ParticleGridConfigDevice& cfg) {
    return (c.z * cfg.grid_dims.y + c.y) * cfg.grid_dims.x + c.x;
}

__device__ __forceinline__ uint32_t CellKeyFromPos(float3 p,
                                                   const ParticleGridConfigDevice& cfg) {
    return CellKey(CellCoord(p, cfg), cfg);
}

// Per-particle 27-cell neighbor query. Iterates the 3x3x3 block of cells around
// the query particle's cell, collecting neighbors within `radius` (squared-
// distance test), EXCLUDING self (`self_idx`). Writes up to `max_count`
// neighbor original-indices into `out_neighbors` (a private slice owned by this
// thread); returns the number written. If more than `max_count` neighbors are
// in range, keeps the `max_count` LOWEST original indices (deterministic subset)
// and sets `*out_overflow = true`. The written list is NOT yet sorted -- the
// calling kernel sorts the private slice ascending (D1).
//
// `cell_start` / `cell_end` are the per-cell [start,end) ranges into
// `particle_idx_sorted`; `positions` is the ORIGINAL (unsorted) position array.
__device__ inline uint32_t QueryParticleNeighbors(
    float3 p,
    uint32_t self_idx,
    float radius,
    const ParticleGridConfigDevice& cfg,
    const uint32_t* __restrict__ cell_start,
    const uint32_t* __restrict__ cell_end,
    const uint32_t* __restrict__ particle_idx_sorted,
    const math::Vec3* __restrict__ positions,
    uint32_t* __restrict__ out_neighbors,
    uint32_t max_count,
    bool* __restrict__ out_overflow) {
    const float r2 = radius * radius;
    const uint3 base = CellCoord(p, cfg);
    uint32_t count = 0u;
    bool overflow = false;

    const int dimx = static_cast<int>(cfg.grid_dims.x);
    const int dimy = static_cast<int>(cfg.grid_dims.y);
    const int dimz = static_cast<int>(cfg.grid_dims.z);

    for (int dz = -1; dz <= 1; ++dz) {
        const int cz = static_cast<int>(base.z) + dz;
        if (cz < 0 || cz >= dimz) {
            continue;
        }
        for (int dy = -1; dy <= 1; ++dy) {
            const int cy = static_cast<int>(base.y) + dy;
            if (cy < 0 || cy >= dimy) {
                continue;
            }
            for (int dx = -1; dx <= 1; ++dx) {
                const int cx = static_cast<int>(base.x) + dx;
                if (cx < 0 || cx >= dimx) {
                    continue;
                }
                const uint32_t key =
                    (static_cast<uint32_t>(cz) * cfg.grid_dims.y +
                     static_cast<uint32_t>(cy)) * cfg.grid_dims.x +
                    static_cast<uint32_t>(cx);
                const uint32_t s = cell_start[key];
                const uint32_t e = cell_end[key];
                for (uint32_t k = s; k < e; ++k) {
                    const uint32_t other = particle_idx_sorted[k];
                    if (other == self_idx) {
                        continue;
                    }
                    const math::Vec3 q = positions[other];
                    const float ddx = q.x - p.x;
                    const float ddy = q.y - p.y;
                    const float ddz = q.z - p.z;
                    const float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                    if (d2 > r2) {
                        continue;
                    }
                    // Insert `other` keeping the buffer sorted ascending and
                    // capped at max_count (keep the lowest indices). This is an
                    // insertion sort into a small (<=32) array -- O(max_count)
                    // per insert, but max_count is tiny so it is cheap and gives
                    // a deterministic, already-sorted output.
                    if (count < max_count) {
                        // Insert into sorted position.
                        uint32_t pos = count;
                        while (pos > 0u && out_neighbors[pos - 1u] > other) {
                            out_neighbors[pos] = out_neighbors[pos - 1u];
                            --pos;
                        }
                        out_neighbors[pos] = other;
                        ++count;
                    } else {
                        // Buffer full. Only keep `other` if it is smaller than
                        // the current largest (out_neighbors[count-1]); either
                        // way we have overflowed the cap.
                        overflow = true;
                        if (other < out_neighbors[count - 1u]) {
                            uint32_t pos = count - 1u;
                            while (pos > 0u && out_neighbors[pos - 1u] > other) {
                                out_neighbors[pos] = out_neighbors[pos - 1u];
                                --pos;
                            }
                            out_neighbors[pos] = other;
                        }
                    }
                }
            }
        }
    }

    if (out_overflow != nullptr) {
        *out_overflow = overflow;
    }
    return count;
}

} // namespace nuka::collision::gpu
