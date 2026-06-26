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
};

}  // namespace nuka::scene::cook
