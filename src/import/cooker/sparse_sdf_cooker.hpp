#pragma once
// ---------------------------------------------------------------------------
// nuka::import::cooker – sparse narrow-band SDF cooker (v0.7 p07)
// ---------------------------------------------------------------------------
// Generates a narrow-band signed distance field for one triangle mesh (one
// V-HACD convex piece) at COOK TIME on the HOST CPU.
//
// Why host CPU (not the spec's CUDA): the cooked SDF is effectively a golden —
// it MUST be byte-reproducible run-to-run. Host single-threaded generation with
// a fixed cell-traversal order and order-independent reductions (min distance,
// fixed-stencil central-diff gradients) is byte-reproducible for free, with no
// GPU FP-reorder nondeterminism. Cook-time is the sanctioned GPU-only exception
// (master plan §5.6); we decline the exception here because determinism is the
// dominant requirement for a cooked golden. (Scope note: fast sweeping / FMM
// from the spec is NOT implemented — for a ±N band, EXACT triangle-point
// distance over the band candidates is both more accurate and trivially
// deterministic, and sweeping only pays off filling large far-field regions we
// do not store. See the .cpp banner.)
//
// Algorithm (per piece):
//   1. Mesh AABB, padded by (band + padding) voxels. origin = padded min corner
//      so all band indices are >= 0 (the packed key is then monotonic).
//   2. For each voxel within (band+1) voxels of the surface (AABB-overlap test
//      per triangle), compute the EXACT min triangle-point distance and the
//      winding-number sign (robust to non-watertight meshes).
//   3. Central-difference gradients on the (band+1) layer.
//   4. Extract the narrow band (|phi| <= band*h), drop the +1 guard ring, sort
//      cell keys ascending, compact into SparseSdfData.
// ---------------------------------------------------------------------------

#include "import/cooker/sparse_sdf_storage.hpp"

#include <cstdint>
#include <string>

namespace nuka::import::cooker {

/// Cook a narrow-band SDF for one triangle mesh. PURE + deterministic: a given
/// (mesh, params) always yields byte-identical SparseSdfData.
///
/// @param vertices       x,y,z triples (length == 3 * vertex_count)
/// @param vertex_count   number of vertices
/// @param indices        triangle indices (length == 3 * triangle_count)
/// @param triangle_count number of triangles
/// @param params         generation parameters (voxel size / band width / ...)
SparseSdfData CookSparseSdf(const float* vertices, uint32_t vertex_count,
                            const uint32_t* indices, uint32_t triangle_count,
                            const SparseSdfParams& params);

/// Content-hash key for (mesh bytes || serialized SDF params): SHA-256 hex.
/// Stable across runs/processes for identical inputs — the per-mesh dedup key.
std::string ComputeSdfCacheKey(const float* vertices, uint32_t vertex_count,
                               const uint32_t* indices, uint32_t triangle_count,
                               const SparseSdfParams& params);

} // namespace nuka::import::cooker
