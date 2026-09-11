#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::cook -- one cooked medium's render surface (a CUDA-free POD).
//
// The boundary triangle list (3 GLOBAL particle indices per triangle, base-offset
// to the medium's slot in CookSceneMedia's [soft|fluid] particle layout) plus the
// MediaRenderSkin params. A live beauty render rebuilds each as a deforming surface
// from the live particle field via runtime::soft::BuildSurfaceMesh -- the medium is
// data (which triangles + which skin), never a per-demo render fork. A tiny header
// (no model/scene/render includes) so the C-ABI WorldRecord retains it cheaply.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

namespace nuka::scene::cook {

struct MediaRenderSurface {
    std::vector<uint32_t> triangles;            // 3 global particle indices per triangle
    float                 normal_offset = 0.0f;  // outward inflation along the smooth normal
    uint32_t              smooth_iters = 0u;      // Laplacian relaxation passes (render-only)
    float                 smooth_lambda = 0.5f;   // per-pass blend weight in [0,1]

    // The medium's authored scene render material (0xFFFFFFFF = none -> the
    // renderer's default media material).
    uint32_t render_material_id = 0xFFFFFFFFu;

    // Particle media retain a range and sampling radius; zero count means the whole field.
    // Without density reconstruction, the range renders as individual grains.
    float    particle_radius = 0.0f;
    uint32_t particle_first = 0u;
    uint32_t particle_count = 0u;

    // Positive spacing selects a continuous density surface over the particle range.
    float surface_spacing = 0.0f;

    // Grain look for the instanced-sphere skin: analytic spheres vs octahedra, plus
    // deterministic per-grain radius/albedo scatter (all 0/false => uniform octahedra).
    uint32_t grain_round = 0u;
    float    grain_radius_jitter = 0.0f;
    float    grain_tint_jitter = 0.0f;
};

}  // namespace nuka::scene::cook
