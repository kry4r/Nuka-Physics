// ---------------------------------------------------------------------------
// nuka::runtime::soft -- particle-set -> deformed render surface mesh (host).
// ---------------------------------------------------------------------------
// HOST cook/render-bridge code (no device): copies live particle positions into
// a MeshGeometry, recomputes smooth normals, and inflates along them. Pure host
// so nuka_render links it without CUDA.
// ---------------------------------------------------------------------------

#include "runtime/soft/particle_surface.hpp"

#include "render/mesh_normals.hpp"

#include <cstddef>

namespace nuka::runtime::soft {

void BuildSurfaceMesh(const std::vector<math::Vec3>& live_positions,
                      const SurfaceTopology& topo,
                      render::MeshGeometry& out) {
    const size_t n = live_positions.size();
    out.positions.assign(n * 3u, 0.0f);
    for (size_t v = 0; v < n; ++v) {
        const math::Vec3& p = live_positions[v];
        out.positions[v * 3 + 0] = p.x;
        out.positions[v * 3 + 1] = p.y;
        out.positions[v * 3 + 2] = p.z;
    }
    out.uvs.clear();
    out.indices = topo.triangles;
    out.normals = render::SmoothNormals(out.positions, out.indices);

    // Inflate each vertex outward along its smooth normal so a contacting body
    // touches the surface (the d_min/2 contact convention) instead of clipping.
    if (topo.normal_offset != 0.0f) {
        for (size_t v = 0; v < n; ++v) {
            out.positions[v * 3 + 0] += topo.normal_offset * out.normals[v * 3 + 0];
            out.positions[v * 3 + 1] += topo.normal_offset * out.normals[v * 3 + 1];
            out.positions[v * 3 + 2] += topo.normal_offset * out.normals[v * 3 + 2];
        }
    }
}

}  // namespace nuka::runtime::soft
