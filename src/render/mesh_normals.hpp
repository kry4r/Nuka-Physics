#pragma once
// ---------------------------------------------------------------------------
// nuka::render -- shared area-weighted SMOOTH per-vertex normal helper.
//
// The ONE smooth-normal implementation used by both BuildRenderWorld (when a
// cooked .nka MESH carries no/zero normals) and the live particle->surface mesh
// builder (runtime::soft::BuildSurfaceMesh). Without it the shader normalizes a
// zero normal -> NaN and the surface renders black; vertex-averaged normals also
// read smoother than flat on curved shells and deformed cloth.
//
// HOST-ONLY, ZERO CUDA (src/render/** red-line). Deterministic accumulation
// order (D1-safe): the same positions+indices always yield the same normals.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nuka::render {

// Area-weighted smooth vertex normals from positions (x,y,z per vertex) and a
// triangle index list. Degenerate (zero-area) fan -> the +Z fallback so the
// shader never normalizes a zero vector.
inline std::vector<float> SmoothNormals(const std::vector<float>& positions,
                                        const std::vector<uint32_t>& indices) {
    std::vector<float> normals(positions.size(), 0.0f);
    auto at = [&](uint32_t v, int c) { return positions[static_cast<size_t>(v) * 3 + c]; };
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t a = indices[i], b = indices[i + 1], c = indices[i + 2];
        const math::Vec3 pa{at(a, 0), at(a, 1), at(a, 2)};
        const math::Vec3 pb{at(b, 0), at(b, 1), at(b, 2)};
        const math::Vec3 pc{at(c, 0), at(c, 1), at(c, 2)};
        // Un-normalized cross => area-weighted contribution (larger faces weigh more).
        const math::Vec3 fn{
            (pb.y - pa.y) * (pc.z - pa.z) - (pb.z - pa.z) * (pc.y - pa.y),
            (pb.z - pa.z) * (pc.x - pa.x) - (pb.x - pa.x) * (pc.z - pa.z),
            (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x)};
        for (uint32_t v : {a, b, c}) {
            normals[static_cast<size_t>(v) * 3 + 0] += fn.x;
            normals[static_cast<size_t>(v) * 3 + 1] += fn.y;
            normals[static_cast<size_t>(v) * 3 + 2] += fn.z;
        }
    }
    for (size_t v = 0; v + 2 < normals.size(); v += 3) {
        const float len = std::sqrt(normals[v] * normals[v] +
                                    normals[v + 1] * normals[v + 1] +
                                    normals[v + 2] * normals[v + 2]);
        if (len > 1e-12f) {
            normals[v] /= len; normals[v + 1] /= len; normals[v + 2] /= len;
        } else {
            normals[v] = 0.0f; normals[v + 1] = 0.0f; normals[v + 2] = 1.0f;
        }
    }
    return normals;
}

}  // namespace nuka::render
