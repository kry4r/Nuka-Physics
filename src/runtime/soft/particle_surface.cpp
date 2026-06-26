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
#include <vector>

namespace nuka::runtime::soft {

namespace {
// Uniform-Laplacian relaxation of render positions: each pass moves a vertex
// toward the average of its edge-neighbours. Deterministic (fixed accumulation
// order); render-only so it never touches physics state.
void LaplacianSmooth(std::vector<float>& positions,
                     const std::vector<uint32_t>& indices,
                     uint32_t iters, float lambda) {
    const size_t n = positions.size() / 3u;
    if (n == 0u || iters == 0u || lambda <= 0.0f) return;
    std::vector<uint32_t> nbr_offset(n + 1u, 0u);
    std::vector<uint32_t> nbr;
    nbr.reserve(indices.size() * 2u);
    // Two passes: count neighbour slots (with duplicates across shared edges, which
    // simply weights frequently-shared edges more), then fill them.
    auto add_edge = [&](uint32_t a, uint32_t b, bool counting) {
        if (counting) { ++nbr_offset[a + 1u]; ++nbr_offset[b + 1u]; }
    };
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t a = indices[i], b = indices[i + 1], c = indices[i + 2];
        add_edge(a, b, true); add_edge(b, c, true); add_edge(c, a, true);
    }
    for (size_t v = 0; v < n; ++v) nbr_offset[v + 1u] += nbr_offset[v];
    nbr.assign(nbr_offset[n], 0u);
    std::vector<uint32_t> cur(nbr_offset.begin(), nbr_offset.begin() + n);
    auto put = [&](uint32_t a, uint32_t b) { nbr[cur[a]++] = b; nbr[cur[b]++] = a; };
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t a = indices[i], b = indices[i + 1], c = indices[i + 2];
        put(a, b); put(b, c); put(c, a);
    }
    std::vector<float> next(positions.size(), 0.0f);
    for (uint32_t it = 0; it < iters; ++it) {
        for (size_t v = 0; v < n; ++v) {
            const uint32_t b0 = nbr_offset[v], b1 = nbr_offset[v + 1u];
            const uint32_t deg = b1 - b0;
            float ax = 0.0f, ay = 0.0f, az = 0.0f;
            for (uint32_t k = b0; k < b1; ++k) {
                const uint32_t u = nbr[k];
                ax += positions[u * 3 + 0]; ay += positions[u * 3 + 1]; az += positions[u * 3 + 2];
            }
            if (deg > 0u) {
                const float inv = 1.0f / static_cast<float>(deg);
                ax *= inv; ay *= inv; az *= inv;
                next[v * 3 + 0] = positions[v * 3 + 0] + lambda * (ax - positions[v * 3 + 0]);
                next[v * 3 + 1] = positions[v * 3 + 1] + lambda * (ay - positions[v * 3 + 1]);
                next[v * 3 + 2] = positions[v * 3 + 2] + lambda * (az - positions[v * 3 + 2]);
            } else {
                next[v * 3 + 0] = positions[v * 3 + 0];
                next[v * 3 + 1] = positions[v * 3 + 1];
                next[v * 3 + 2] = positions[v * 3 + 2];
            }
        }
        positions.swap(next);
    }
}
}  // namespace

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
    // Relax the surface BEFORE normals/inflation so a single proud particle does
    // not survive as a sharp spike under the divergent inflated normals.
    LaplacianSmooth(out.positions, out.indices, topo.smooth_iters, topo.smooth_lambda);
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
