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
#include <cstring>
#include <unordered_map>
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

// Area-weighted vertex normals over every face touching the vertex's position whose normal lies within
// the crease of the vertex's own faces, so unwelded triangle soups shade smoothly but keep hard edges.
inline std::vector<float> CreaseNormals(const std::vector<float>& positions,
                                        const std::vector<uint32_t>& indices, float crease_cos) {
    struct Key {
        uint32_t x, y, z;
        bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            return (size_t{k.x} * 73856093u) ^ (size_t{k.y} * 19349663u) ^ (size_t{k.z} * 83492791u);
        }
    };
    const size_t vertex_count = positions.size() / 3, face_count = indices.size() / 3;
    std::vector<uint32_t> group(vertex_count);
    std::unordered_map<Key, uint32_t, KeyHash> ids;
    for (size_t v = 0; v < vertex_count; ++v) {
        Key key;
        std::memcpy(&key, &positions[v * 3], sizeof(key));
        group[v] = ids.emplace(key, static_cast<uint32_t>(ids.size())).first->second;
    }
    std::vector<float> face(face_count * 3), own(positions.size(), 0.0f);
    std::vector<uint32_t> start(ids.size() + 1, 0u), faces(face_count * 3);
    auto at = [&](uint32_t v, int c) { return positions[static_cast<size_t>(v) * 3 + c]; };
    for (size_t f = 0; f < face_count; ++f) {
        const uint32_t a = indices[f * 3], b = indices[f * 3 + 1], c = indices[f * 3 + 2];
        const float e1[3] = {at(b, 0) - at(a, 0), at(b, 1) - at(a, 1), at(b, 2) - at(a, 2)};
        const float e2[3] = {at(c, 0) - at(a, 0), at(c, 1) - at(a, 1), at(c, 2) - at(a, 2)};
        face[f * 3 + 0] = e1[1] * e2[2] - e1[2] * e2[1];
        face[f * 3 + 1] = e1[2] * e2[0] - e1[0] * e2[2];
        face[f * 3 + 2] = e1[0] * e2[1] - e1[1] * e2[0];
        for (uint32_t v : {a, b, c}) {
            for (int k = 0; k < 3; ++k) own[static_cast<size_t>(v) * 3 + k] += face[f * 3 + k];
            ++start[group[v] + 1];
        }
    }
    for (size_t g = 0; g + 1 < start.size(); ++g) start[g + 1] += start[g];
    std::vector<uint32_t> cursor(start.begin(), start.end() - 1);
    for (size_t i = 0; i < face_count * 3; ++i) faces[cursor[group[indices[i]]]++] = static_cast<uint32_t>(i / 3);
    std::vector<float> normals(positions.size(), 0.0f);
    for (size_t v = 0; v < vertex_count; ++v) {
        const float* r = &own[v * 3];
        const float r_len = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        float sum[3] = {0.0f, 0.0f, 0.0f};
        for (uint32_t i = start[group[v]]; i < start[group[v] + 1]; ++i) {
            const float* n = &face[static_cast<size_t>(faces[i]) * 3];
            const float n_len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (n_len > 0.0f && r[0] * n[0] + r[1] * n[1] + r[2] * n[2] >= crease_cos * r_len * n_len)
                for (int k = 0; k < 3; ++k) sum[k] += n[k];
        }
        const float len = std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
        for (int k = 0; k < 3; ++k) normals[v * 3 + k] = len > 1e-12f ? sum[k] / len : (k == 2 ? 1.0f : 0.0f);
    }
    return normals;
}

}  // namespace nuka::render
