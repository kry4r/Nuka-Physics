#pragma once
// ---------------------------------------------------------------------------
// nuka::import::cooker – sparse narrow-band SDF cook-time data layout (v0.7 p07)
// ---------------------------------------------------------------------------
// Plain data structures produced by the SDF cooker. Like convex_piece.hpp these
// carry no scene/runtime types, so they live in the leaf `nuka_cooker` lib that
// nuka_scene links without a dependency cycle. The scene cooker concatenates
// these per-piece SparseSdfData into CookedBlob::sdfs (CookedSdfTable).
//
// COOK-TIME / HOST-CPU ONLY (master plan §5.6 sanctioned exception). No CUDA.
//
// DETERMINISM: SparseSdfData is the byte-reproducible cook output. cell_keys is
// ASCENDING-sorted; values/gradients parallel it. Two cooks of the same mesh +
// params yield byte-identical SparseSdfData.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

namespace nuka::import::cooker {

/// Parameters controlling narrow-band SDF generation. Determinism contract: the
/// SDF is a PURE function of (mesh, these params). All fields are part of the
/// dedup / cache key.
struct SparseSdfParams {
    float    voxel_size      = 0.0f;  // edge length of one voxel. 0 => auto from AABB.
    uint32_t band_voxels     = 4u;    // narrow band half-width N: store |phi| <= N*h
    float    auto_resolution = 64.0f; // when voxel_size==0: longest AABB axis / this
    float    padding_voxels  = 1.0f;  // extra voxels of AABB padding (>= 1 recommended)
    // Fill the full enclosed interior (not just the |phi|<=band shell) so a deep
    // inside point has a value+gradient. Off => narrow-band-only (byte-identical).
    bool     solid           = false;
};

/// One cooked narrow-band SDF (mesh-local frame). cell_keys[n] packs the voxel
/// index (PackSdfCellKey in runtime/sdf/sparse_sdf_query.cuh); cell_values[n] and
/// cell_gradients[n] are the signed distance + central-diff gradient at that cell.
struct SparseSdfData {
    // Header.
    float    origin[3]   = {0.0f, 0.0f, 0.0f};  // local-space corner of voxel (0,0,0)
    float    voxel_size  = 0.0f;
    uint32_t dims[3]     = {0u, 0u, 0u};         // full (dense) grid dims

    // Narrow-band cells (sorted ascending by key).
    std::vector<uint64_t> cell_keys;
    std::vector<float>    cell_values;
    std::vector<float>    cell_gradients;        // x,y,z triples, parallel to keys

    uint32_t CellCount() const { return static_cast<uint32_t>(cell_values.size()); }

    /// Dense voxel count for the full grid (used for the memory-savings report).
    uint64_t DenseVoxelCount() const {
        return static_cast<uint64_t>(dims[0]) * static_cast<uint64_t>(dims[1]) *
               static_cast<uint64_t>(dims[2]);
    }

    /// Bytes the narrow band actually stores (key + value + 3-float gradient).
    uint64_t NarrowBandBytes() const {
        return static_cast<uint64_t>(CellCount()) *
               (sizeof(uint64_t) + sizeof(float) + 3u * sizeof(float));
    }

    /// Bytes a dense grid would need (value + 3-float gradient per voxel).
    uint64_t DenseBytes() const {
        return DenseVoxelCount() * (sizeof(float) + 3u * sizeof(float));
    }
};

} // namespace nuka::import::cooker
