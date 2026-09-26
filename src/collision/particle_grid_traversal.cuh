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

// Unbounded integer cell of a position; far values saturate so the cast stays defined.
__device__ __forceinline__ int3 CellIndex(float3 p, const ParticleGridConfigDevice& cfg) {
    const auto axis = [](float x, float lo, float inv) {
        return static_cast<int>(fminf(fmaxf(floorf((x - lo) * inv), -1.0e9f), 1.0e9f));
    };
    return make_int3(axis(p.x, cfg.grid_min.x, cfg.inv_cell_size.x),
                     axis(p.y, cfg.grid_min.y, cfg.inv_cell_size.y),
                     axis(p.z, cfg.grid_min.z, cfg.inv_cell_size.z));
}

// The grid tiles space periodically, so any position maps to a table cell.
__device__ __forceinline__ uint32_t WrapCell(int c, uint32_t dim) {
    const int d = static_cast<int>(dim);
    const int r = c % d;
    return static_cast<uint32_t>(r < 0 ? r + d : r);
}

// Flatten a wrapped cell coordinate to a linear cell key (x-fastest). Deterministic.
__device__ __forceinline__ uint32_t CellKey(int3 c, const ParticleGridConfigDevice& cfg) {
    return (WrapCell(c.z, cfg.grid_dims.z) * cfg.grid_dims.y + WrapCell(c.y, cfg.grid_dims.y)) *
               cfg.grid_dims.x + WrapCell(c.x, cfg.grid_dims.x);
}

__device__ __forceinline__ uint32_t CellKeyFromPos(float3 p,
                                                   const ParticleGridConfigDevice& cfg) {
    return CellKey(CellIndex(p, cfg), cfg);
}

// Enumerate all in-radius neighbors; a null output returns the exact count.
// A bounded output keeps ascending particle IDs and reports any discarded neighbors.
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
    const int3 base = CellIndex(p, cfg);
    uint32_t count = 0u;
    uint32_t attempted = 0u;
    bool overflow = false;

    // Narrow axes visit each wrapped cell once: one cell for dim 1, two for dim 2.
    const auto lo = [](uint32_t dim) { return dim >= 2u ? -1 : 0; };
    const auto hi = [](uint32_t dim) { return dim >= 3u ? 1 : 0; };
    for (int dz = lo(cfg.grid_dims.z); dz <= hi(cfg.grid_dims.z); ++dz) {
        for (int dy = lo(cfg.grid_dims.y); dy <= hi(cfg.grid_dims.y); ++dy) {
            for (int dx = lo(cfg.grid_dims.x); dx <= hi(cfg.grid_dims.x); ++dx) {
                const uint32_t key = CellKey(make_int3(base.x + dx, base.y + dy, base.z + dz), cfg);
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
                    ++attempted;
                    if (out_neighbors == nullptr) continue;
                    if (max_count == 0u) {
                        overflow = true;
                        continue;
                    }
                    // Insertion by particle ID gives a deterministic output order.
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
    return out_neighbors == nullptr ? attempted : count;
}

} // namespace nuka::collision::gpu
