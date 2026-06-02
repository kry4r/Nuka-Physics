#pragma once
// ---------------------------------------------------------------------------
// Programmatic test meshes for the sparse narrow-band SDF cooker tests (p07).
// ---------------------------------------------------------------------------
// Hand-built closed, outward-wound triangle meshes with known analytical SDFs:
//   - SphereMesh:  UV sphere (analytical SDF: |p - c| - r)
//   - CapsuleMesh: cylinder + two hemispherical caps along +Z
//   - ThinPanelMesh: a thin axis-aligned slab (the thin-shell test)
// Reuses tests/import/vhacd_test_meshes.hpp's TestMesh + AppendBox.
// ---------------------------------------------------------------------------

#include "tests/import/vhacd_test_meshes.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace nuka::test {

// UV sphere centered at `center`, radius `r`. `stacks` latitude bands, `slices`
// longitude segments. Outward-wound triangles. Analytical SDF: |p-center|-r.
inline TestMesh SphereMesh(float cx, float cy, float cz, float r,
                           uint32_t stacks = 24, uint32_t slices = 48) {
    TestMesh m;
    const float pi = 3.14159265358979323846f;
    // Vertices: top pole, ring grid, bottom pole.
    auto push = [&](float x, float y, float z) {
        m.vertices.push_back(x);
        m.vertices.push_back(y);
        m.vertices.push_back(z);
    };
    // Generate (stacks+1) rings of `slices` vertices (poles are degenerate rings
    // but we keep the simple grid + close it with triangles).
    for (uint32_t s = 0; s <= stacks; ++s) {
        const float phi = pi * static_cast<float>(s) / static_cast<float>(stacks);  // 0..pi
        const float sp = std::sin(phi);
        const float cp = std::cos(phi);
        for (uint32_t j = 0; j < slices; ++j) {
            const float theta = 2.0f * pi * static_cast<float>(j) / static_cast<float>(slices);
            const float x = cx + r * sp * std::cos(theta);
            const float y = cy + r * sp * std::sin(theta);
            const float z = cz + r * cp;
            push(x, y, z);
        }
    }
    auto idx = [&](uint32_t s, uint32_t j) -> uint32_t {
        return s * slices + (j % slices);
    };
    for (uint32_t s = 0; s < stacks; ++s) {
        for (uint32_t j = 0; j < slices; ++j) {
            const uint32_t a = idx(s, j);
            const uint32_t b = idx(s + 1, j);
            const uint32_t c = idx(s + 1, j + 1);
            const uint32_t d = idx(s, j + 1);
            // Outward winding (CCW seen from outside).
            m.indices.push_back(a); m.indices.push_back(b); m.indices.push_back(c);
            m.indices.push_back(a); m.indices.push_back(c); m.indices.push_back(d);
        }
    }
    return m;
}

// Capsule along the Z axis centered at origin: a cylinder of radius `r` and
// half-height `hh` (segment endpoints at z = -hh and z = +hh), capped by two
// hemispheres. Analytical SDF: distance to the segment minus r.
inline TestMesh CapsuleMesh(float r, float hh, uint32_t slices = 48,
                            uint32_t cap_stacks = 12) {
    TestMesh m;
    const float pi = 3.14159265358979323846f;
    auto push = [&](float x, float y, float z) {
        m.vertices.push_back(x);
        m.vertices.push_back(y);
        m.vertices.push_back(z);
    };
    std::vector<uint32_t> ring_starts;  // start vertex index of each latitude ring

    // Top hemisphere (phi from 0 at +pole to pi/2 at equator), centered at +hh.
    for (uint32_t s = 0; s <= cap_stacks; ++s) {
        const float phi = (pi * 0.5f) * static_cast<float>(s) / static_cast<float>(cap_stacks);
        const float sp = std::sin(phi);
        const float cp = std::cos(phi);
        ring_starts.push_back(static_cast<uint32_t>(m.vertices.size() / 3));
        for (uint32_t j = 0; j < slices; ++j) {
            const float theta = 2.0f * pi * static_cast<float>(j) / static_cast<float>(slices);
            push(r * sp * std::cos(theta), r * sp * std::sin(theta), hh + r * cp);
        }
    }
    // Bottom hemisphere (phi from pi/2 at equator to pi at -pole), centered -hh.
    for (uint32_t s = 0; s <= cap_stacks; ++s) {
        const float phi = (pi * 0.5f) + (pi * 0.5f) * static_cast<float>(s) / static_cast<float>(cap_stacks);
        const float sp = std::sin(phi);
        const float cp = std::cos(phi);
        ring_starts.push_back(static_cast<uint32_t>(m.vertices.size() / 3));
        for (uint32_t j = 0; j < slices; ++j) {
            const float theta = 2.0f * pi * static_cast<float>(j) / static_cast<float>(slices);
            push(r * sp * std::cos(theta), r * sp * std::sin(theta), -hh + r * cp);
        }
    }
    const uint32_t ring_count = static_cast<uint32_t>(ring_starts.size());
    for (uint32_t ri = 0; ri + 1 < ring_count; ++ri) {
        const uint32_t s0 = ring_starts[ri];
        const uint32_t s1 = ring_starts[ri + 1];
        for (uint32_t j = 0; j < slices; ++j) {
            const uint32_t jn = (j + 1) % slices;
            const uint32_t a = s0 + j;
            const uint32_t b = s1 + j;
            const uint32_t c = s1 + jn;
            const uint32_t d = s0 + jn;
            m.indices.push_back(a); m.indices.push_back(b); m.indices.push_back(c);
            m.indices.push_back(a); m.indices.push_back(c); m.indices.push_back(d);
        }
    }
    return m;
}

// Thin axis-aligned slab: x in [-hx,hx], y in [-hy,hy], z in [-half_thick,
// +half_thick]. Closed box (AppendBox is outward-wound). Used by the thin-shell
// test (e.g. half_thick = 0.25mm => a 0.5mm panel).
inline TestMesh ThinPanelMesh(float hx, float hy, float half_thick) {
    TestMesh m;
    AppendBox(m, -hx, -hy, -half_thick, hx, hy, half_thick);
    return m;
}

}  // namespace nuka::test
