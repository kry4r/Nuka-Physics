#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::soft -- the ONE general particle-set -> render surface mesh.
//
// Turns a live particle set + its KNOWN cook topology into a deformed triangle
// MeshGeometry the renderer draws: positions copied from the live particles,
// indices = the authored triangle list, normals RECOMPUTED per frame (shared
// render::SmoothNormals) so the surface shades smoothly as it deforms. The same
// path serves cloth now (the lattice triangles) and tet/fluid surfaces later --
// the medium is data (which triangles), never a per-demo render fork.
//
// HOST-ONLY, ZERO CUDA: a pure-host TU so nuka_render links it without pulling
// CUDA. SurfaceTopology is a CUDA-free POD (a flat index list) carrying the cook
// triangles, so the builder needs no GPU/cook-lib dependency.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "render/render_world.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::soft {

// The triangle connectivity of a particle surface, authored once at cook time.
// `triangles` is a flat index list (3 vertex indices per triangle) into the
// particle set; an optional per-vertex outward offset inflates the surface along
// its smooth normal so a contacting body visibly touches it (the d_min/2 contact
// convention) rather than intersecting the zero-thickness sheet. `smooth_iters`
// applies that many Laplacian (uniform-neighbour-average) passes to the rendered
// positions so a single proud particle relaxes into the cloth instead of becoming
// a sharp visual spike; `smooth_lambda` is the per-pass blend weight.
struct SurfaceTopology {
    std::vector<uint32_t> triangles;       // 3 indices per triangle
    float                 normal_offset = 0.0f;  // outward inflation along the smooth normal
    uint32_t              smooth_iters = 0u;      // Laplacian relaxation passes (render-only)
    float                 smooth_lambda = 0.5f;   // per-pass blend weight in [0,1]
};

// Fill `out` with a deformed surface mesh from the live particle positions and
// the cook topology: positions from `live_positions`, indices = topo.triangles,
// smooth per-vertex normals recomputed over the live geometry, then each vertex
// pushed out by topo.normal_offset along its normal. Deterministic (D1-safe):
// identical inputs always produce a byte-identical mesh.
void BuildSurfaceMesh(const std::vector<math::Vec3>& live_positions,
                      const SurfaceTopology& topo,
                      render::MeshGeometry& out);

}  // namespace nuka::runtime::soft
